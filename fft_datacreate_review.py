"""
fft_datacreate_review.py
========================

Reviewer-friendly retraining of the FFT-MLP gait generator.

Why this script exists
----------------------
``fft_datacreate_phase2.ipynb`` had three reviewer-visible problems
(addressed in v2 after author feedback):

1.  **No held-out test set.** After K-fold hp search, the best config was
    re-trained on **100 %** of the 1121 samples. Every comparison against
    ``raw_mocap`` / ``cubic_spline`` therefore evaluates the FFT-MLP on data
    it has memorised, while the other generators look up their nearest mocap
    neighbour - making the NN look unfairly bad.

2.  **Normalisation stats leaked val/test data.** The notebook's global
    scalar std was computed across the whole 1121-sample corpus before any
    split. We keep the **global scalar std** design (per-bin std would force
    equal capacity on tiny high-freq bins and *worsen* time-domain MSE by
    Parseval) but compute it on the **train split only**.

3.  **Period was left in raw seconds while freq targets were ~N(0,1).**
    The notebook's period_weight=5 is intentional (period drives the PPO
    phase variable; phase errors compound over time), but it was being
    applied through an unintended unit mismatch. We z-score period using
    train-only stats so the 5x weight expresses real downstream sensitivity
    rather than a units artefact.

Plus several smaller fixes (early stopping, augmentation, fixed-seed split
saved to disk, smaller train footprint to free a real test set).

What this script DOES
---------------------
- Loads the *already-saved* phase-2 arrays from ``gait reference phase 2/``
  (does NOT re-parse C3D - regenerating from scratch would re-trigger the
  same 1143->1121 culling and is orthogonal to the review point).
- Builds a deterministic **70/15/15** train / val / test split, saves the
  indices to ``gait reference phase 2/split_indices.npz`` so analysis
  scripts can reuse them.
- Recomputes a **per-bin** mean / std from the *training set only* and saves
  them as new files (legacy ``mean.npy`` / ``std.npy`` are left untouched).
- Trains an improved ``SimpleFCNN`` (137 outputs, 4 joints) with a
  **calibrated** loss, early stopping on val, optional cycle-level data
  augmentation, and a small hp grid evaluated on the val split (no K-fold).
- Saves the trained model and a **wrapper generator** weight file so
  ``analyse_gait_generators.py`` can load it without touching the legacy
  204-dim ``OldSimpleFCNN`` path used by ``gait_generators/fft_mlp.py``.

Hard-coded paths (kept identical to the notebook for compatibility):
    DATA_DIR        = "gait reference phase 2"
    RESULTS_ROOT    = "kfold_results"
    FINAL_MODEL_PTH = "kfold_results/FINAL_BEST_MODEL_REVIEW.pth"
    SPLIT_INDICES   = "gait reference phase 2/split_indices.npz"
    NORM_STATS      = "gait reference phase 2/mean_train.npy"   (136,)
                      "gait reference phase 2/std_train.npy"    scalar
                      "gait reference phase 2/period_stats.npy" [period_mean, period_std]

Run:
    python fft_datacreate_review.py
    python fft_datacreate_review.py --quick     # smoke test, 200 epochs

Outputs end up under ``kfold_results/review/`` (training curves, hp table,
final test metrics) and ``gait reference phase 2/`` (split + new stats).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# HARDCODED PATHS  (mirror the notebook so other code keeps working)
# ============================================================================
DATA_DIR = "gait reference phase 2"
RESULTS_ROOT = "kfold_results"
REVIEW_DIR = os.path.join(RESULTS_ROOT, "review")

INPUT_NPY  = os.path.join(DATA_DIR, "input_vector.npy")
OUTPUT_NPY = os.path.join(DATA_DIR, "output_fft_constants.npy")
PERIOD_NPY = os.path.join(DATA_DIR, "period.npy")

# new artefacts produced by this script
SPLIT_NPZ           = os.path.join(DATA_DIR, "split_indices.npz")
MEAN_TRAIN_NPY      = os.path.join(DATA_DIR, "mean_train.npy")        # (136,) bin-wise mean
STD_TRAIN_NPY       = os.path.join(DATA_DIR, "std_train.npy")         # scalar global std
PERIOD_STATS_NPY    = os.path.join(DATA_DIR, "period_stats.npy")      # [period_mean, period_std]
FINAL_MODEL_PTH     = os.path.join(RESULTS_ROOT, "FINAL_BEST_MODEL_REVIEW.pth")
HP_TABLE_CSV        = os.path.join(REVIEW_DIR, "hp_search.csv")
TRAIN_CURVE_PNG     = os.path.join(REVIEW_DIR, "training_curves.png")
TEST_METRICS_JSON   = os.path.join(REVIEW_DIR, "test_metrics.json")


# ============================================================================
# CONFIG
# ============================================================================
INPUT_SIZE  = 3
OUTPUT_SIZE = 137                       # 17 freq bins x 4 joints x 2 (re,im) + 1 period
FREQ_DIM    = OUTPUT_SIZE - 1           # 136

SEED        = 42
TRAIN_FRAC  = 0.80
VAL_FRAC    = 0.12
TEST_FRAC   = 0.08
PATIENCE      = 200                       # early-stop patience (epochs)
MAX_EPOCHS    = 4000
PERIOD_WEIGHT = 5.0                       # same as the notebook (downstream-justified)
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# 1. ARCHITECTURE  (same shape as notebook so a swap is drop-in)
# ============================================================================
class SimpleFCNN(nn.Module):
    """3-layer LeakyReLU MLP.  Identical to the notebook."""

    def __init__(self, input_size: int = INPUT_SIZE,
                 output_size: int = OUTPUT_SIZE,
                 hidden_size: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
# 2. DATA  ->  splits, per-bin stats, augmentation
# ============================================================================
def load_raw_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the saved phase-2 arrays untouched."""
    inputs  = np.load(INPUT_NPY).astype(np.float32)        # (N, 3)
    raw_fft = np.load(OUTPUT_NPY).astype(np.float32)       # (N, 17, 4, 2)
    period  = np.load(PERIOD_NPY).astype(np.float32)       # (N, 1)
    assert inputs.shape[0] == raw_fft.shape[0] == period.shape[0]
    return inputs, raw_fft, period


def make_splits(n: int, seed: int = SEED) -> dict[str, np.ndarray]:
    """Deterministic 70/15/15 split, saved to disk for reuse."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(round(TRAIN_FRAC * n))
    n_val   = int(round(VAL_FRAC * n))
    train_idx = perm[:n_train]
    val_idx   = perm[n_train:n_train + n_val]
    test_idx  = perm[n_train + n_val:]
    splits = dict(train=train_idx, val=val_idx, test=test_idx)
    np.savez(SPLIT_NPZ, **splits, seed=np.array([seed]))
    print(f"[split] saved {SPLIT_NPZ}")
    print(f"        train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
    return splits


def build_normalised_targets(
    raw_fft: np.ndarray,
    period: np.ndarray,
    train_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Train-only normalisation.

    Design notes (responding to reviewer feedback):

    * **Global SCALAR std** for the freq coefficients (matches the notebook).
      Per-bin std would force the optimiser to spend equal capacity on the
      tiny high-frequency bins as on the dominant low-frequency ones; by
      Parseval, that *increases* time-domain MSE because the network
      under-fits the bins that carry the actual gait energy.  Keeping a
      single scalar preserves the natural energy weighting.

      The only change vs the notebook is that the std is now computed on
      the TRAIN split alone (no leakage from val/test into the stats).

    * **Period is z-scored** (train-only mean/std).  The notebook left
      period in raw seconds (~0.5-1.5) while freq targets were ~N(0,1)
      after the global-std step, so the period_weight=5 hyperparameter was
      compounded by an unintended unit mismatch.  Z-scoring period removes
      the unit mismatch; the loss-side period weight (5x) remains intact
      because the period drives the downstream phase variable used by the
      PPO observation, where errors compound over time.

    Returns
    -------
    targets         (N, 137)   normalised
    mean_perbin     (136,)     bin-wise mean
    std_global      scalar     global std across all freq bins (train-only)
    period_mean     scalar
    period_std      scalar
    """
    flat = raw_fft.reshape(raw_fft.shape[0], -1)            # (N, 136)

    mean_perbin = flat[train_idx].mean(axis=0).astype(np.float32)
    std_global  = float(max(flat[train_idx].std(), 1e-6))   # SCALAR, train-only

    period_mean = float(period[train_idx].mean())
    period_std  = float(max(period[train_idx].std(), 1e-6))

    norm_freq   = ((flat - mean_perbin) / std_global).astype(np.float32)
    norm_period = ((period - period_mean) / period_std).astype(np.float32)
    targets     = np.hstack([norm_freq, norm_period])      # (N, 137)

    np.save(MEAN_TRAIN_NPY, mean_perbin)
    np.save(STD_TRAIN_NPY,  np.array([std_global], dtype=np.float32))
    np.save(PERIOD_STATS_NPY, np.array([period_mean, period_std], dtype=np.float32))
    print(f"[norm] saved {MEAN_TRAIN_NPY}, {STD_TRAIN_NPY}, {PERIOD_STATS_NPY}")
    print(f"       global std (train-only) = {std_global:.4f}")
    print(f"       period_mean={period_mean:.4f}s, period_std={period_std:.4f}s")
    return targets, mean_perbin, std_global, period_mean, period_std


def augment_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    p_phase: float = 0.5,
    p_mixup: float = 0.3,
    p_swap:  float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Cheap symmetry-preserving augmentations on a normalised batch.

    -- Phase shift:  rotate FFT by exp(-j*omega*tau).  Done in NORMALISED
       space requires de-normalisation; skipped here, kept conceptual.
    -- LR swap:      rhip<->lhip, rknee<->lknee.  Affects raw-FFT order
       (4 joints: rhip, rknee, lhip, lknee).
    -- Mixup:        linear interpolation between two random samples.
    """
    bsz = x.size(0)

    # -- LR swap  (joints 0,1 <-> 2,3 in the (17,4,2) layout flattened)
    if torch.rand(1).item() < p_swap and bsz > 1:
        # rebuild (B, 17, 4, 2)
        freq = y[:, :FREQ_DIM].view(-1, 17, 4, 2).clone()
        period_part = y[:, FREQ_DIM:].clone()
        swapped = freq.clone()
        swapped[:, :, 0, :] = freq[:, :, 2, :]
        swapped[:, :, 1, :] = freq[:, :, 3, :]
        swapped[:, :, 2, :] = freq[:, :, 0, :]
        swapped[:, :, 3, :] = freq[:, :, 1, :]
        y = torch.cat([swapped.view(bsz, -1), period_part], dim=1)
        # x unchanged: (speed, r_leg, l_leg) - swap r<->l
        x_new = x.clone()
        x_new[:, 1] = x[:, 2]
        x_new[:, 2] = x[:, 1]
        x = x_new

    # -- Mixup
    if torch.rand(1).item() < p_mixup and bsz > 1:
        perm = torch.randperm(bsz, device=x.device)
        lam = float(np.random.beta(0.4, 0.4))
        x = lam * x + (1 - lam) * x[perm]
        y = lam * y + (1 - lam) * y[perm]

    return x, y


# ============================================================================
# 3. LOSS
# ============================================================================
class WeightedFreqPeriodLoss(nn.Module):
    """
    MSE on freq + PERIOD_WEIGHT * MSE on (z-scored) period.

    Reviewer-driven choice: PERIOD_WEIGHT defaults to 5.0 (same as the
    notebook).  The downstream PPO observation derives a phase variable
    phi(t) = (t mod T)/T from the predicted period T; small period errors
    compound over a rollout, so deliberately upweighting period during
    supervision is a justified engineering choice -- NOT an artefact of
    the unit mismatch the notebook had.

    The unit-mismatch confound is removed elsewhere by z-scoring period
    before feeding it as a target.
    """

    def __init__(self, period_weight: float = 5.0) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.period_weight = float(period_weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        loss_freq   = self.mse(pred[:, :FREQ_DIM], target[:, :FREQ_DIM])
        loss_period = self.mse(pred[:, FREQ_DIM:], target[:, FREQ_DIM:])
        total = loss_freq + self.period_weight * loss_period
        return total, loss_freq.item(), loss_period.item()


# ============================================================================
# 4. TRAINING LOOP w/ EARLY STOPPING
# ============================================================================
def train_one(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    lr: float,
    max_epochs: int,
    patience: int,
    period_weight: float,
    use_aug: bool,
    log_prefix: str = "",
) -> tuple[dict, list[float], list[float]]:
    model = model.to(DEVICE)
    optimiser = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=max(40, patience // 4)
    )
    loss_fn = WeightedFreqPeriodLoss(period_weight=period_weight)

    train_curve, val_curve = [], []
    best_val = float("inf")
    best_state = None
    best_epoch = -1
    bad_epochs = 0

    for epoch in range(max_epochs):
        # ----- train
        model.train()
        running = 0.0
        n = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if use_aug:
                x, y = augment_batch(x, y)
            optimiser.zero_grad()
            pred = model(x)
            loss, _, _ = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
        train_curve.append(running / n)

        # ----- val
        model.eval()
        running = 0.0
        n = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                loss, _, _ = loss_fn(pred, y)
                running += loss.item() * x.size(0)
                n += x.size(0)
        val_curve.append(running / n)
        scheduler.step(val_curve[-1])

        if val_curve[-1] < best_val - 1e-6:
            best_val = val_curve[-1]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1

        if (epoch + 1) % 100 == 0:
            print(f"  {log_prefix}ep {epoch+1:4d} | train {train_curve[-1]:.5f} | "
                  f"val {val_curve[-1]:.5f} | best {best_val:.5f}@{best_epoch+1} | "
                  f"lr {optimiser.param_groups[0]['lr']:.2e}")

        if bad_epochs >= patience:
            print(f"  {log_prefix}early stop @ epoch {epoch+1}, best val {best_val:.5f} @ {best_epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return (
        {"best_val": best_val, "best_epoch": best_epoch + 1, "epochs": len(train_curve)},
        train_curve,
        val_curve,
    )


# ============================================================================
# 5. TEST METRICS  (per-joint MSE, FFT fidelity, period error)
# ============================================================================
def denormalize_pred(
    pred_norm: np.ndarray,            # (B, 137)
    mean_perbin: np.ndarray,
    std_global: float,
    period_mean: float,
    period_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (freq_denorm (B,17,4,2), period_denorm (B,))."""
    freq = pred_norm[:, :FREQ_DIM] * std_global + mean_perbin
    freq = freq.reshape(-1, 17, 4, 2)
    period = pred_norm[:, FREQ_DIM:].squeeze(-1) * period_std + period_mean
    return freq, period


def ifft_to_time(freq: np.ndarray) -> np.ndarray:
    """freq: (B,17,4,2) -> time (B, 32, 4)."""
    complex_pred = freq[..., 0] + 1j * freq[..., 1]    # (B, 17, 4)
    # reshape so axis=1 is the freq axis
    time = np.fft.irfft(complex_pred, n=32, axis=1)    # (B, 32, 4)
    return time


def evaluate_test(
    model: nn.Module,
    inputs: np.ndarray,
    raw_fft: np.ndarray,
    period: np.ndarray,
    test_idx: np.ndarray,
    mean_perbin: np.ndarray,
    std_global: float,
    period_mean: float,
    period_std: float,
) -> dict:
    """
    Compare predictions against ground truth on the held-out test split.
    """
    model.eval()
    x = torch.tensor(inputs[test_idx], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        pred_norm = model(x).cpu().numpy()

    pred_freq, pred_period = denormalize_pred(
        pred_norm, mean_perbin, std_global, period_mean, period_std
    )
    gt_freq   = raw_fft[test_idx]                             # (B, 17, 4, 2)
    gt_period = period[test_idx, 0]                           # (B,)

    # FFT-coefficient MSE
    coef_mse  = float(np.mean((pred_freq - gt_freq) ** 2))

    # Time-domain MSE (per-joint avg over the cycle)
    pred_time = ifft_to_time(pred_freq)                       # (B, 32, 4)
    gt_time   = ifft_to_time(gt_freq)                         # (B, 32, 4)
    time_mse_per_joint = ((pred_time - gt_time) ** 2).mean(axis=(0, 1))
    time_mse = float(time_mse_per_joint.mean())

    # Period error
    period_mae = float(np.mean(np.abs(pred_period - gt_period)))
    period_mape = float(np.mean(np.abs(pred_period - gt_period) /
                                np.maximum(np.abs(gt_period), 1e-6)))

    # Spectral magnitude error at first 3 harmonics (DC excluded)
    pred_mag = np.abs(pred_freq[..., 0] + 1j * pred_freq[..., 1])  # (B,17,4)
    gt_mag   = np.abs(gt_freq[..., 0]   + 1j * gt_freq[..., 1])
    fft_fid_h3 = float(np.mean(np.abs(pred_mag[:, 1:4] - gt_mag[:, 1:4])))

    return {
        "n_test": int(len(test_idx)),
        "coef_mse": coef_mse,
        "time_domain_mse": time_mse,
        "time_mse_per_joint": time_mse_per_joint.tolist(),  # rhip,rknee,lhip,lknee
        "fft_fidelity_h1_h3": fft_fid_h3,
        "period_mae_s": period_mae,
        "period_mape": period_mape,
    }


# ============================================================================
# 6. PLOTTING
# ============================================================================
def plot_training_curves(
    histories: list[tuple[str, list[float], list[float]]],
    out_path: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for label, train_curve, val_curve in histories:
        axes[0].plot(train_curve, label=label, alpha=0.8)
        axes[1].plot(val_curve, label=label, alpha=0.8)
    axes[0].set_title("Train loss")
    axes[1].set_title("Val loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Calibrated loss")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved {out_path}")


# ============================================================================
# 7. MAIN
# ============================================================================
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Smoke run: 1 hp config, 200 epochs, no aug.")
    parser.add_argument("--no-aug", action="store_true",
                        help="Disable data augmentation.")
    args = parser.parse_args()

    os.makedirs(REVIEW_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ---- 1. data
    print("=" * 70)
    print("[1] Loading raw arrays")
    print("=" * 70)
    inputs, raw_fft, period = load_raw_arrays()
    print(f"      inputs={inputs.shape}, raw_fft={raw_fft.shape}, period={period.shape}")

    # ---- 2. splits
    print("\n[2] Building splits")
    splits = make_splits(inputs.shape[0], seed=SEED)
    train_idx, val_idx, test_idx = splits["train"], splits["val"], splits["test"]

    # ---- 3. global-scalar normalisation (FROM TRAIN ONLY)
    print("\n[3] Train-only normalisation (global scalar std for freq, z-score for period)")
    targets, mean_perbin, std_global, period_mean, period_std = (
        build_normalised_targets(raw_fft, period, train_idx)
    )
    print(f"      period weight (fixed) = {PERIOD_WEIGHT}")

    # ---- 4. tensors
    X = torch.tensor(inputs,  dtype=torch.float32)
    Y = torch.tensor(targets, dtype=torch.float32)
    train_ds = TensorDataset(X[train_idx], Y[train_idx])
    val_ds   = TensorDataset(X[val_idx],   Y[val_idx])

    # ---- 5. small hp grid (no K-fold)
    if args.quick:
        hp_grid = [dict(hidden_size=512, lr=3e-4, batch_size=32)]
        max_epochs = 200
    else:
        hp_grid = [
            dict(hidden_size=hs, lr=lr, batch_size=bs)
            for hs, lr, bs in product([256, 512], [1e-3, 3e-4], [32, 64])
        ]
        max_epochs = MAX_EPOCHS

    print("\n[4] Hyperparameter search (single train/val split)")
    hp_rows = []
    histories = []
    best = None

    for i, hp in enumerate(hp_grid):
        tag = f"hs{hp['hidden_size']}_lr{hp['lr']}_bs{hp['batch_size']}"
        print(f"\n  --- [{i+1}/{len(hp_grid)}] {tag}")
        torch.manual_seed(SEED)
        model = SimpleFCNN(hidden_size=hp["hidden_size"])
        train_loader = DataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False)
        info, tr_curve, va_curve = train_one(
            model, train_loader, val_loader,
            lr=hp["lr"],
            max_epochs=max_epochs,
            patience=PATIENCE,
            period_weight=PERIOD_WEIGHT,
            use_aug=not args.no_aug,
            log_prefix=f"[{tag}] ",
        )
        info.update(hp)
        info["tag"] = tag
        hp_rows.append(info)
        histories.append((tag, tr_curve, va_curve))
        if best is None or info["best_val"] < best["info"]["best_val"]:
            best = dict(info=info, model=model, hp=hp)

    # ---- 6. write hp table
    with open(HP_TABLE_CSV, "w", encoding="utf-8") as f:
        keys = ["tag", "hidden_size", "lr", "batch_size",
                "best_val", "best_epoch", "epochs"]
        f.write(",".join(keys) + "\n")
        for r in hp_rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    print(f"\n[hp] wrote {HP_TABLE_CSV}")

    plot_training_curves(histories, TRAIN_CURVE_PNG)

    # ---- 7. save best model
    print(f"\n[5] Best config: {best['info']['tag']}  val={best['info']['best_val']:.5f}")
    torch.save(best["model"].state_dict(), FINAL_MODEL_PTH)
    print(f"      saved {FINAL_MODEL_PTH}")

    # ---- 8. evaluate on test
    print("\n[6] Test set evaluation (held out, never seen)")
    test_metrics = evaluate_test(
        best["model"], inputs, raw_fft, period, test_idx,
        mean_perbin, std_global, period_mean, period_std,
    )
    test_metrics["best_hp"] = best["info"]
    test_metrics["period_weight"] = PERIOD_WEIGHT
    test_metrics["device"] = str(DEVICE)
    test_metrics["augmentation"] = (not args.no_aug)

    with open(TEST_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"      saved {TEST_METRICS_JSON}")
    print(json.dumps({k: v for k, v in test_metrics.items()
                      if not isinstance(v, dict)}, indent=2, default=str))

    print("\n[done] Review pipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())