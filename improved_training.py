"""
Improved training setup for FFT coefficient prediction network.
Predicts joint angle FFT coefficients from leg lengths and desired speed.

Includes:
- Hyperparameter tuning
- Early stopping
- Learning rate scheduling
- Testing with prediction vs ground truth plots
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
from itertools import product

# Import model architectures from gait_generator_net.py
from gait_generator_net import SimpleFCNN, GaitFFTPredictor


# ============================================================================
# LOSS FUNCTION
# ============================================================================

class CombinedFFTLoss(nn.Module):
    """
    Combined loss: MSE on FFT coefficients with emphasis on low frequencies.
    Low frequencies are more perceptually important for gait.
    """
    def __init__(self, fft_weight=1.0, low_freq_weight=2.0):
        super().__init__()
        self.fft_weight = fft_weight
        self.low_freq_weight = low_freq_weight
        self.mse = nn.MSELoss()
        
    def forward(self, pred, target):
        # Basic FFT coefficient loss
        fft_loss = self.mse(pred, target)
        
        # Weighted loss: emphasize lower frequencies (more perceptually important)
        # Reshape to [batch, 6 joints, 2 (real/imag), 17 freq bins]
        pred_reshaped = pred.view(-1, 6, 2, 17)
        target_reshaped = target.view(-1, 6, 2, 17)
        
        # Lower frequencies (first 5 bins) get higher weight
        low_freq_loss = self.mse(pred_reshaped[:, :, :, :5], target_reshaped[:, :, :, :5])
        
        total_loss = self.fft_weight * fft_loss + self.low_freq_weight * low_freq_loss
        
        return total_loss, {
            'fft_loss': fft_loss.item(),
            'low_freq_loss': low_freq_loss.item(),
        }


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

class EarlyStopping:
    """Stop training when validation loss stops improving."""
    def __init__(self, patience=50, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
        self.best_model_state = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict().copy()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def create_dataloaders(input_path, output_path, batch_size=64, val_split=0.15, test_split=0.00):
    """Create train/val/test dataloaders."""
    inputs = np.load(input_path).astype(np.float32)
    outputs = np.load(output_path).astype(np.float32)
    
    # Output shape from notebook: [N, 17, 6, 2]
    # After transpose(0,2,3,1): [N, 6, 2, 17]
    # Flatten to [N, 204] for training
    outputs = outputs.transpose(0, 2, 3, 1)  # [N, 6, 2, 17]
    outputs = outputs.reshape(outputs.shape[0], -1)  # [N, 204]
    
    # Convert to tensors
    inputs = torch.from_numpy(inputs)
    outputs = torch.from_numpy(outputs)
    
    dataset = TensorDataset(inputs, outputs)
    
    # Split dataset
    n = len(dataset)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_val - n_test
    
    train_data, val_data, test_data = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


# ============================================================================
# SINGLE MODEL TRAINING
# ============================================================================

def train_single_model(
    train_loader,
    val_loader,
    model_class,
    hidden_size,
    learning_rate,
    weight_decay,
    num_epochs,
    patience,
    device,
    verbose=True,
):
    """Train a single model with given hyperparameters."""
    # Model
    torch.manual_seed(42)
    model = model_class(input_size=3, output_size=204, hidden_size=hidden_size)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = CombinedFFTLoss(fft_weight=1.0, low_freq_weight=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=200, min_lr=1e-6
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience)
    
    # Training history
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    best_val_loss = float('inf')
    best_epoch = 0
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss, _ = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss, _ = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_loader.dataset)
        history['val_loss'].append(val_loss)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Track best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
        
        # Logging
        if verbose and ((epoch + 1) % 200 == 0 or epoch == 0):
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch+1:4d}/{num_epochs} | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"Time: {elapsed:.1f}s")
        
        # Early stopping
        if early_stopping(val_loss, model):
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break
    
    # Restore best model
    if early_stopping.best_model_state is not None:
        model.load_state_dict(early_stopping.best_model_state)
    
    history['best_val_loss'] = best_val_loss
    history['best_epoch'] = best_epoch
    history['total_epochs'] = epoch + 1
    
    return model, history


# ============================================================================
# HYPERPARAMETER TUNING
# ============================================================================

def hyperparameter_tuning(
    input_path="gait reference fft5.00/newnormalized_input_vector.npy",
    output_path="gait reference fft5.00/newnormalized_output_fft_constants.npy",
    results_dir="hyperparameter_results",
    # Hyperparameter search space
    model_classes=None,  # List of model classes to try
    batch_sizes=[32, 64],
    hidden_sizes=[256, 512],
    learning_rates=[1e-3, 3e-4],
    weight_decay=1e-5,
    num_epochs=8000,
    patience=500,
):
    """
    Perform hyperparameter tuning over given search space.
    """
    os.makedirs(results_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Default model classes
    if model_classes is None:
        model_classes = [SimpleFCNN, GaitFFTPredictor]
    
    # Results storage
    tuning_results = []
    best_overall_val_loss = float('inf')
    best_overall_params = None
    best_overall_model_state = None
    
    # Generate all hyperparameter combinations
    combinations = list(product(model_classes, batch_sizes, hidden_sizes, learning_rates))
    total_combinations = len(combinations)
    
    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER TUNING")
    print(f"{'='*80}")
    print(f"Total combinations: {total_combinations}")
    print(f"Model classes: {[m.__name__ for m in model_classes]}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Hidden sizes: {hidden_sizes}")
    print(f"Learning rates: {learning_rates}")
    print(f"{'='*80}\n")
    
    for idx, (model_class, bs, hs, lr) in enumerate(combinations):
        print(f"\n[{idx+1}/{total_combinations}] Training Configuration:")
        print("-" * 60)
        
        # Create temporary model to count parameters
        temp_model = model_class(input_size=3, output_size=204, hidden_size=hs)
        num_params = sum(p.numel() for p in temp_model.parameters())
        
        # Print current hyperparameters
        print(f"  Model:         {model_class.__name__}")
        print(f"  Hidden size:   {hs}")
        print(f"  Batch size:    {bs}")
        print(f"  Learning rate: {lr}")
        print(f"  Weight decay:  {weight_decay}")
        print(f"  Parameters:    {num_params:,}")
        print("-" * 60)
        
        del temp_model  # Free memory
        
        # Create dataloaders with current batch size
        train_loader, val_loader, test_loader = create_dataloaders(
            input_path, output_path, batch_size=bs
        )
        
        # Train model
        model, history = train_single_model(
            train_loader=train_loader,
            val_loader=val_loader,
            model_class=model_class,
            hidden_size=hs,
            learning_rate=lr,
            weight_decay=weight_decay,
            num_epochs=num_epochs,
            patience=patience,
            device=device,
            verbose=True,
        )
        
        # Record results
        result = {
            'model_class': model_class.__name__,
            'batch_size': bs,
            'hidden_size': hs,
            'learning_rate': lr,
            'best_val_loss': history['best_val_loss'],
            'best_epoch': history['best_epoch'],
            'total_epochs': history['total_epochs'],
        }
        tuning_results.append(result)
        
        print(f"  Best val loss: {history['best_val_loss']:.6f} @ epoch {history['best_epoch']}")
        
        # Track overall best
        if history['best_val_loss'] < best_overall_val_loss:
            best_overall_val_loss = history['best_val_loss']
            best_overall_params = result.copy()
            best_overall_model_state = model.state_dict().copy()
            
            # Save best model
            model_name = f"best_model_{model_class.__name__}_hs{hs}_lr{lr}_bs{bs}.pth"
            torch.save(best_overall_model_state, os.path.join(results_dir, model_name))
            print(f"  ✓ New best model saved: {model_name}")
        
        # Save loss plot for this run
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train', alpha=0.8)
        plt.plot(history['val_loss'], label='Val', alpha=0.8)
        plt.axvline(history['best_epoch']-1, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{model_class.__name__} bs={bs} hs={hs} lr={lr}')
        plt.legend()
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(history['lr'])
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('LR Schedule')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_name = f"loss_{model_class.__name__}_bs{bs}_hs{hs}_lr{lr}.png"
        plt.savefig(os.path.join(results_dir, plot_name), dpi=150)
        plt.close()
    
    # Save results to CSV
    results_df = pd.DataFrame(tuning_results)
    results_df = results_df.sort_values('best_val_loss')
    results_df.to_csv(os.path.join(results_dir, 'tuning_results.csv'), index=False)
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER TUNING COMPLETE")
    print(f"{'='*80}")
    print(f"Best parameters:")
    for key, value in best_overall_params.items():
        print(f"  {key}: {value}")
    print(f"\nResults saved to: {results_dir}/")
    print(f"Best model saved as: best_model_{best_overall_params['model_class']}_*.pth")
    print(f"{'='*80}\n")
    
    return results_df, best_overall_params, best_overall_model_state


# ============================================================================
# TESTING UTILITIES
# ============================================================================

def denormalize(pred, gt, normalization_const):
    """Denormalize predictions and ground truth using normalization constants."""
    for i in range(17):
        for k in range(2):
            pred[:, k, i] = pred[:, k, i] * normalization_const[i * 2 + k]
            gt[:, k, i] = gt[:, k, i] * normalization_const[i * 2 + k]
    return pred, gt


def fft_to_time_domain(predictions, ground_truth):
    """Convert FFT coefficients to time domain signals using IRFFT."""
    real_pred = predictions[:, 0, :]
    imag_pred = predictions[:, 1, :]
    pred_complex = real_pred + 1j * imag_pred
    
    real_gt = ground_truth[:, 0, :]
    imag_gt = ground_truth[:, 1, :]
    gt_complex = real_gt + 1j * imag_gt
    
    pred_time = np.fft.irfft(pred_complex, axis=1)
    gt_time = np.fft.irfft(gt_complex, axis=1)
    
    pred_time = pred_time.transpose(1, 0)
    gt_time = gt_time.transpose(1, 0)
    
    return pred_time, gt_time


def plot_prediction_vs_ground_truth(pred_time, gt_time, speed, save_path):
    """Plot and save prediction vs ground truth for all 6 joints."""
    joint_names = ['Right Hip', 'Right Knee', 'Right Ankle', 
                   'Left Hip', 'Left Knee', 'Left Ankle']
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    
    for i, (ax, name) in enumerate(zip(axes, joint_names)):
        ax.plot(pred_time[:, i], 'b-', linewidth=2, label='Predicted')
        ax.plot(gt_time[:, i], 'r--', linewidth=2, label='Ground Truth')
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Joint Angle (rad)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Prediction vs Ground Truth (Speed: {speed:.2f} m/s)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def test_model(
    model_path,
    model_class=GaitFFTPredictor,
    input_path="gait reference fft5.00/newnormalized_input_vector.npy",
    output_path="gait reference fft5.00/newnormalized_output_fft_constants.npy",
    norm_path="gait reference fft5.00/newnormalization_constants.npy",
    results_dir="newgaitgenresults",
    hidden_size=512,
    num_samples=None,
    test_loader=None,
):
    """Test the trained model and save prediction vs ground truth plots."""
    os.makedirs(results_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load normalization constants
    norm_consts = np.load(norm_path)
    
    # Load model
    model = model_class(input_size=3, output_size=204, hidden_size=hidden_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Model loaded from: {model_path}")
    
    test_losses = []
    mse_fn = nn.MSELoss()
    
    # Use test_loader if provided
    if test_loader is not None:
        total_samples = len(test_loader.dataset)
        print(f"\nTesting {total_samples} samples from test_loader...")
        print(f"Saving plots to: {results_dir}/")
        
        sample_idx = 0
        with torch.no_grad():
            for inputs_batch, targets_batch in test_loader:
                inputs_batch = inputs_batch.to(device)
                targets_batch = targets_batch.to(device)
                
                # Process each sample in the batch
                for j in range(inputs_batch.size(0)):
                    input_sample = inputs_batch[j:j+1]
                    target_flat = targets_batch[j:j+1]
                    
                    pred_flat = model(input_sample)
                    loss = mse_fn(pred_flat, target_flat)
                    test_losses.append(loss.item())
                    
                    pred = pred_flat.cpu().numpy().reshape(6, 2, 17)
                    gt = target_flat.cpu().numpy().reshape(6, 2, 17)
                    
                    speed = input_sample[0, 0].cpu().item() * 3
                    
                    pred_denorm, gt_denorm = denormalize(pred.copy(), gt.copy(), norm_consts)
                    pred_time, gt_time = fft_to_time_domain(pred_denorm, gt_denorm)
                    
                    save_path = os.path.join(results_dir, f"sample_{sample_idx:04d}_speed_{speed:.2f}ms.png")
                    plot_prediction_vs_ground_truth(pred_time, gt_time, speed, save_path)
                    
                    if (sample_idx + 1) % 50 == 0 or sample_idx == 0:
                        print(f"  Processed {sample_idx+1}/{total_samples} samples | Loss: {loss.item():.6f}")
                    
                    sample_idx += 1
    else:
        # Fallback: load all data from files
        inputs = np.load(input_path).astype(np.float32)
        outputs = np.load(output_path).astype(np.float32)
        outputs = outputs.transpose(0, 2, 3, 1)  # [N, 6, 2, 17]
        
        print(f"Loaded {len(inputs)} samples")
        
        inputs_tensor = torch.from_numpy(inputs).to(device)
        outputs_flat = outputs.reshape(outputs.shape[0], -1)
        
        if num_samples is None:
            num_samples = len(inputs)
        
        print(f"\nTesting {num_samples} samples...")
        print(f"Saving plots to: {results_dir}/")
        
        with torch.no_grad():
            for i in range(min(num_samples, len(inputs))):
                input_sample = inputs_tensor[i:i+1]
                target_flat = torch.from_numpy(outputs_flat[i:i+1]).to(device)
                
                pred_flat = model(input_sample)
                loss = mse_fn(pred_flat, target_flat)
                test_losses.append(loss.item())
                
                pred = pred_flat.cpu().numpy().reshape(6, 2, 17)
                gt = outputs[i].copy()
                
                speed = inputs[i, 0] * 3
                
                pred_denorm, gt_denorm = denormalize(pred.copy(), gt.copy(), norm_consts)
                pred_time, gt_time = fft_to_time_domain(pred_denorm, gt_denorm)
                
                save_path = os.path.join(results_dir, f"sample_{i:04d}_speed_{speed:.2f}ms.png")
                plot_prediction_vs_ground_truth(pred_time, gt_time, speed, save_path)
                
                if (i + 1) % 50 == 0 or i == 0:
                    print(f"  Processed {i+1}/{num_samples} samples | Loss: {loss.item():.6f}")
    
    avg_loss = np.mean(test_losses)
    std_loss = np.std(test_losses)
    
    print(f"\n{'='*50}")
    print(f"Testing Complete!")
    print(f"{'='*50}")
    print(f"Samples tested: {len(test_losses)}")
    print(f"Average MSE Loss: {avg_loss:.6f} ± {std_loss:.6f}")
    print(f"Min Loss: {np.min(test_losses):.6f}")
    print(f"Max Loss: {np.max(test_losses):.6f}")
    print(f"Results saved to: {results_dir}/")
    
    summary = {
        'num_samples': len(test_losses),
        'avg_loss': avg_loss,
        'std_loss': std_loss,
        'min_loss': np.min(test_losses),
        'max_loss': np.max(test_losses),
        'losses': test_losses
    }
    np.save(os.path.join(results_dir, 'test_summary.npy'), summary)
    
    return test_losses


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Run hyperparameter tuning
    results_df, best_params, best_model_state = hyperparameter_tuning(
        input_path="gait reference fft5.00/newnormalized_input_vector.npy",
        output_path="gait reference fft5.00/newnormalized_output_fft_constants.npy",
        results_dir="hyperparameter_results",
        # Search space
        model_classes=[GaitFFTPredictor],
        batch_sizes=[32, 64],
        hidden_sizes=[512],
        learning_rates=[1e-3, 3e-4],
        # Training settings
        weight_decay=1e-5,
        num_epochs=8000,
        patience=200,
    )
    
    # Print top 5 results
    print("\nTop 5 configurations:")
    print(results_df.head())
    
    # Test the best model
    print("\n" + "="*50)
    print("Starting Testing Phase")
    print("="*50 + "\n")
    
    # Determine the best model class
    best_model_class = SimpleFCNN if best_params['model_class'] == 'SimpleFCNN' else GaitFFTPredictor
    
    # Find the best model file
    best_model_path = os.path.join(
        "hyperparameter_results",
        f"best_model_{best_params['model_class']}_hs{best_params['hidden_size']}_"
        f"lr{best_params['learning_rate']}_bs{best_params['batch_size']}.pth"
    )
    
    test_losses = test_model(
        model_path=best_model_path,
        model_class=best_model_class,
        input_path="gait reference fft5.00/newnormalized_input_vector.npy",
        output_path="gait reference fft5.00/newnormalized_output_fft_constants.npy",
        norm_path="gait reference fft5.00/newnormalization_constants.npy",
        results_dir="newgaitgenresults",
        hidden_size=best_params['hidden_size'],
        num_samples=None,
    )
