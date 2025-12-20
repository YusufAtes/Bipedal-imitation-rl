import torch
from gait_generator_net import GaitFFTPredictor, SimpleFCNN
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample


def denormalize(pred_flat, mean, std):
    """Denormalizes flattening frequency data."""
    pred = (pred_flat * std) + mean
    return pred

def recover_shape(flat_data):
    """
    FIXED RESHAPE FUNCTION
    Reconstructs (4, 2, 17) from flat (136,) vector.
    """
    # 1. Reshape to creation shape: (Freqs=17, Joints=4, Real/Imag=2)
    recovered = flat_data.reshape(17, 4, 2)
    # 2. Transpose to IFFT shape: (Joints=4, Real/Imag=2, Freqs=17)
    structured = recovered.transpose(1, 2, 0)
    return structured

def pred_ifft(predictions):
    """
    Performs Inverse FFT to get time-domain signals.
    Input Shape: (4, 2, 17) -> [Joints, Real/Imag, Freqs]
    """
    # Combine Real and Imaginary parts
    # predictions[:, 0, :] is Real, predictions[:, 1, :] is Imag
    complex_pred = predictions[:, 0, :] + 1j * predictions[:, 1, :]
    
    # Inverse FFT (n=32 points)
    pred_time = np.fft.irfft(complex_pred, n=32, axis=1)
    
    # Transpose for plotting: (Time, Joints)
    return pred_time.T



model = SimpleFCNN(input_size=3, output_size=137, hidden_size=512)
model.load_state_dict(torch.load(r'C:\Users\yusuf\Bipedal-imitation-rl\kfold_results\FINAL_BEST_MODEL.pth'))
model.eval()  # Set model to evaluation mode to disable dropout
mean = np.load(r'C:\Users\yusuf\Bipedal-imitation-rl\gait reference phase 2\mean.npy')
std = np.load(r'C:\Users\yusuf\Bipedal-imitation-rl\gait reference phase 2\std.npy')
print(f"mean: {mean} and std: {std}")
print(f"mean shape: {mean.shape} and std shape: {std.shape}")
input_vec = torch.ones(1,3)
input_vec[0] = 0.3

input_vec[1:] = input_vec[1:] * 0.75
with torch.no_grad():  # Disable gradient computation for inference
    freq_out = model(input_vec)
    freqs_pred = freq_out[0, :136].numpy()
    per_pred = freq_out[0, 136].numpy()
    # 1. Denormalize
    f_pred_dn = denormalize(freqs_pred, mean, std)
    
    # 2. Recover Shape
    struct_pred = recover_shape(f_pred_dn)
    
    # 3. IFFT
    pred_t = pred_ifft(struct_pred)


period_len = per_pred
pred_time = pred_t

fig, axs = plt.subplots(2, 2, figsize=(10, 6))
fig.suptitle(f"Period Length: {period_len}")
# Plot columns 0 and 1 in row 0
axs[0, 0].grid(True)
axs[0, 1].grid(True)
axs[0, 0].plot(pred_time[:, 0])
axs[0, 0].set_title("pred_time[:, 0]")
axs[0, 1].plot(pred_time[:, 1])
axs[0, 1].set_title("pred_time[:, 1]")

# Plot columns 3 and 4 in row 1
axs[1, 0].grid(True)
axs[1, 1].grid(True)
axs[1, 0].plot(pred_time[:, 2])
axs[1, 0].set_title("pred_time[:, 3]")
axs[1, 1].plot(pred_time[:, 3])
axs[1, 1].set_title("pred_time[:, 4]")

plt.tight_layout()
plt.show()




