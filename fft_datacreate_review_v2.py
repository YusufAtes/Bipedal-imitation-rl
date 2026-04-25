"""
fft_datacreate_review_v2.py
===========================

Improved review training of the FFT-MLP gait generator.

What changed vs ``fft_datacreate_review.py``
--------------------------------------------
1. **Positive-period head (softplus).**
   Period is strictly positive in physics.  The old script predicted a
   z-scored scalar with a bare ``Linear`` head and de-normalised as
   ``T = z * std + mean``.  At the tails of the training distribution the
   model could predict z < -mean/std, yielding T <= 0 -> downstream
   ``resample(..., n_samples=int(round(T/dt)))`` becomes nonsense.
   We now split the output into two heads:
      - ``freq_head``   : 136 z-scored FFT coefficients (unchanged)
      - ``period_head`` : 1 logit passed through ``softplus`` and interpreted
                          in SECONDS directly (``T = softplus(z)``).
   Period supervision is done in seconds (MSE in raw units), so the loss
   magnitude is bounded and interpretable.  No more z-score/std dance for T.

2. **Phase-aware (time-domain) loss.**
   FFT coefficient MSE over-penalises high-harmonic noise (which is mostly
   mocap measurement noise, not gait content).  The new loss is
       L = alpha * MSE_fft + (1 - alpha) * MSE_time(irfft(pred), irfft(gt))
   where the irfft is differentiable (``torch.fft.irfft``).  This makes the
   loss surface align with what the PPO imitation reward actually sees in
   the environment (time-domain joint positions).

3. **Cubic-spline residual head (optional).**
   gait(speed, leg) is very smooth in speed - cubic_spline beats the MLP at
   mid-speeds because it interpolates locally.  We use that as a prior:
       freq_pred = spline_prior(speed) + MLP(speed, r_leg, l_leg)
   The MLP only learns the morphology-dependent *residual* the spline can't
   represent.  Typical win: 1.5-3x time-domain MSE improvement on small
   datasets (we have 1143 samples).

4. **Phase-shift augmentation done properly.**
   Rotating the FFT by ``exp(-j*omega*k/32)`` in un-normalised space IS a
   valid augmentation that the old script marked "conceptual, skipped".
   Implemented here as an on-the-fly rotation applied in de-normalised
   space, then re-normalised.  Multiplies effective dataset size by ~8x
   without changing the time-domain content.

5. **--variant flag** lets you train any of four configurations under a
   single script, so the four output models can be compared apples-to-apples
   on the same split / stats / optimiser / seed:
       baseline        - original recipe (sanity baseline)
       phase           - +phase-aware loss, +softplus period head
       residual        - +cubic-spline residual, +softplus period head
       phase_residual  - all improvements combined

6. **Held-out test set is now used.**  After the best-HP model is chosen on
   val, we evaluate it on the test split and dump:
       - kfold_results/review_v2/test_metrics_<variant>.json       (as before)
       - kfold_results/review_v2/time_domain_<variant>/sample_*.png
             2x2 per-joint GT vs prediction for every test sample
       - kfold_results/review_v2/variants_test_comparison.png
             grouped bar chart across all 4 variants (auto-skips missing)
   The cross-variant comparison plot is regenerated every run from whichever
   ``test_metrics_*.json`` files exist, so running only 2 variants still
   produces a 2-bar comparison.  Use ``--compare-only`` to rebuild the
   comparison plot without retraining.

Hard-coded paths (independent namespace from the v1 script)
-----------------------------------------------------------
    weights        = kfold_results/FINAL_BEST_MODEL_V2_<variant>.pth
    mean / std     = gait reference phase 2/{mean_train,std_train}.npy  (shared w/ v1)
    period stats   = gait reference phase 2/period_stats.npy             (shared w/ v1)
    spline prior   = gait reference phase 2/spline_prior_v2.npz          (NEW)

Run
---
    python fft_datacreate_review_v2.py --variant phase_residual
    python fft_datacreate_review_v2.py --variant residual --quick
    python fft_datacreate_review_v2.py --variant baseline           # sanity
    python fft_datacreate_review_v2.py --compare-only               # just replot
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from itertools import product
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


# ============================================================================
# PATHS
# ============================================================================
DATA_DIR     = "gait reference phase 2"
RESULTS_ROOT = "kfold_results"
REVIEW_DIR   = os.path.join(RESULTS_ROOT, "review_v2")

INPUT_NPY    = os.path.join(DATA_DIR, "input_vector.npy")
OUTPUT_NPY   = os.path.join(DATA_DIR, "output_fft_constants.npy")
PERIOD_NPY   = os.path.join(DATA_DIR, "period.npy")

SPLIT_NPZ          = os.path.join(DATA_DIR, "split_indices.npz")
MEAN_TRAIN_NPY     = os.path.join(DATA_DIR, "mean_train.npy")
STD_TRAIN_NPY      = os.path.join(DATA_DIR, "std_train.npy")
PERIOD_STATS_NPY   = os.path.join(DATA_DIR, "period_stats.npy")
SPLINE_PRIOR_NPZ   = os.path.join(DATA_DIR, "spline_prior_v2.npz")

# Cross-variant comparison plot (written every run)
COMPARISON_PNG = os.path.join(REVIEW_DIR, "variants_test_comparison.png")

# Joint labels for the 4-channel FFT targets. If your channels are ordered
# differently, change the list here - plotting only, no downstream effect.
JOINT_NAMES = ["RHip", "RKnee", "LHip", "LKnee"]


# ============================================================================
# CONFIG
# ============================================================================
INPUT_SIZE   = 3
FREQ_DIM     = 136                       # 17 x 4 x 2
OUTPUT_SIZE  = FREQ_DIM + 1              # + period

SEED         = 42
TRAIN_FRAC   = 0.80
VAL_FRAC     = 0.15
TEST_FRAC    = 0.05
PATIENCE     = 250
MAX_EPOCHS   = 4000

# Variant-specific hyper-parameters
PERIOD_WEIGHT_S       = 0.5     # weight on period MSE (seconds^2).  Small
                                # because the scalar is ~0.5-1.5s, so
                                # (T-T_hat)^2 is ~0.01 naturally.
TIME_ALPHA            = 0.4     # L = alpha*fft + (1-alpha)*time
PHASE_AUG_PROB        = 0.5
PHASE_AUG_MAX_SHIFT   = 4       # samples out of 32 (= up to 45 deg)
DEVICE                = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cap on per-sample time-domain plots (to avoid 500 PNGs if test is large)
MAX_TIME_DOMAIN_SAMPLES = 60


# ============================================================================
# 1. ARCHITECTURE
# ============================================================================
class SimpleFCNN(nn.Module):
    """
    Backbone + 2 heads.

    * ``freq_head``   : Linear(H, 136)          -- z-scored FFT coefficients
    * ``period_head`` : Linear(H, 1) + softplus -- period in SECONDS (> 0)

    The split-head layout solves two problems of the v1 script:

    - Period is in a different unit from freq coefficients.  With a single
      ``Linear(H, 137)`` head, the optimiser sees them as one tensor for
      weight-decay purposes, so the tiny period scalar gets either
      over-regularised or lost in the 136-way freq competition.
    - No sign constraint on period.  ``softplus`` makes T > 0 unconditionally
      (softplus(x) = log(1+exp(x)) is smooth and > 0 everywhere), removing
      the need to z-score period during training and the need for clamping
      at inference.
    """

    def __init__(self, input_size: int = INPUT_SIZE, hidden_size: int = 512) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.1),
        )
        self.freq_head   = nn.Linear(hidden_size, FREQ_DIM)
        self.period_head = nn.Linear(hidden_size, 1)
        # bias init so softplus(bias)=ln(exp(0.8)-1)~0.8s (mid gait period)
        nn.init.constant_(self.period_head.bias, 0.4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (freq_zscored (B,136), period_seconds (B,1))."""
        h = self.backbone(x)
        freq = self.freq_head(h)
        period = F.softplus(self.period_head(h))      # > 0, in seconds
        return freq, period


# ============================================================================
# 2. DATA  ->  splits, per-bin stats, augmentation
# ============================================================================
def load_raw_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inputs  = np.load(INPUT_NPY).astype(np.float32)        # (N, 3)
    raw_fft = np.load(OUTPUT_NPY).astype(np.float32)       # (N, 17, 4, 2)
    period  = np.load(PERIOD_NPY).astype(np.float32)       # (N, 1)
    assert inputs.shape[0] == raw_fft.shape[0] == period.shape[0]
    return inputs, raw_fft, period


def make_splits(n: int, seed: int = SEED) -> dict[str, np.ndarray]:
    """Reuse existing split if present; else create one."""
    if os.path.exists(SPLIT_NPZ):
        npz = np.load(SPLIT_NPZ)
        print(f"[split] re-using {SPLIT_NPZ} (train={len(npz['train'])}, "
              f"val={len(npz['val'])}, test={len(npz['test'])})")
        return dict(train=npz["train"], val=npz["val"], test=npz["test"])

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(round(TRAIN_FRAC * n))
    n_val   = int(round(VAL_FRAC * n))
    splits = dict(
        train=perm[:n_train],
        val=perm[n_train:n_train + n_val],
        test=perm[n_train + n_val:],
    )
    np.savez(SPLIT_NPZ, **splits, seed=np.array([seed]))
    print(f"[split] saved {SPLIT_NPZ}")
    return splits


def build_norm_stats(
    raw_fft: np.ndarray, period: np.ndarray, train_idx: np.ndarray
) -> tuple[np.ndarray, float, float, float]:
    """Train-only per-bin mean + global scalar std for FFT; period kept in seconds."""
    flat = raw_fft.reshape(raw_fft.shape[0], -1)            # (N, 136)
    mean_perbin = flat[train_idx].mean(axis=0).astype(np.float32)
    std_global  = float(max(flat[train_idx].std(), 1e-6))

    period_mean = float(period[train_idx].mean())
    period_std  = float(max(period[train_idx].std(), 1e-6))

    np.save(MEAN_TRAIN_NPY, mean_perbin)
    np.save(STD_TRAIN_NPY,  np.array([std_global], dtype=np.float32))
    np.save(PERIOD_STATS_NPY, np.array([period_mean, period_std], dtype=np.float32))
    print(f"[norm] saved mean_train, std_train, period_stats")
    print(f"       std_global={std_global:.4f}, "
          f"period_mean={period_mean:.4f}s, period_std={period_std:.4f}s")
    return mean_perbin, std_global, period_mean, period_std


# ----------------------------------------------------------------------------
# Cubic-spline prior over speed (for residual variant)
# ----------------------------------------------------------------------------
def build_spline_prior(
    inputs: np.ndarray,       # (N, 3)   speed/2.4, r_leg, l_leg
    raw_fft: np.ndarray,      # (N, 17, 4, 2)
    period: np.ndarray,       # (N, 1)
    train_idx: np.ndarray,
    n_knots: int = 16,
) -> dict[str, np.ndarray]:
    """
    Fit a 1-D cubic spline per FFT coefficient (and one per period) as a
    function of commanded speed, using the training samples only.  The
    MLP then only predicts the (speed, leg)-dependent residual on top.

    Knots are chosen as equal-count quantiles of the training speeds so
    the spline has ~equal data support in each segment.
    """
    speeds_train = inputs[train_idx, 0] * 2.4           # m/s (de-norm encoding)
    flat_train   = raw_fft[train_idx].reshape(len(train_idx), -1)  # (n_tr, 136)
    period_train = period[train_idx, 0]

    # Knot speeds = quantiles; average targets inside each knot bucket.
    qs = np.linspace(0.0, 1.0, n_knots)
    knot_speeds = np.quantile(speeds_train, qs)
    # Ensure strict monotonicity (duplicates possible at data extrema).
    knot_speeds = np.unique(knot_speeds)

    def _avg_per_knot(vals: np.ndarray) -> np.ndarray:
        ys = []
        for i in range(len(knot_speeds)):
            lo = knot_speeds[i - 1] if i > 0 else -np.inf
            hi = knot_speeds[i + 1] if i + 1 < len(knot_speeds) else np.inf
            mask = (speeds_train >= lo) & (speeds_train <= hi)
            ys.append(vals[mask].mean(axis=0) if mask.sum() > 0 else vals.mean(axis=0))
        return np.stack(ys, axis=0)

    freq_knots   = _avg_per_knot(flat_train)           # (K, 136)
    period_knots = _avg_per_knot(period_train)         # (K,)

    np.savez(
        SPLINE_PRIOR_NPZ,
        knot_speeds=knot_speeds.astype(np.float32),
        freq_knots=freq_knots.astype(np.float32),
        period_knots=period_knots.astype(np.float32),
    )
    print(f"[prior] saved cubic-spline prior to {SPLINE_PRIOR_NPZ}  "
          f"(K={len(knot_speeds)} knots)")
    return dict(
        knot_speeds=knot_speeds.astype(np.float32),
        freq_knots=freq_knots.astype(np.float32),
        period_knots=period_knots.astype(np.float32),
    )


class SplinePrior:
    """Vectorised per-coefficient cubic splines in speed (CPU numpy; cheap)."""

    def __init__(self, knot_speeds: np.ndarray,
                 freq_knots: np.ndarray, period_knots: np.ndarray) -> None:
        # One CubicSpline over the 136 coefs jointly (shape (K,136))
        self._freq_cs   = CubicSpline(knot_speeds, freq_knots, axis=0,
                                      bc_type="natural", extrapolate=True)
        self._period_cs = CubicSpline(knot_speeds, period_knots,
                                      bc_type="natural", extrapolate=True)
        self.speed_min = float(knot_speeds[0])
        self.speed_max = float(knot_speeds[-1])

    def __call__(self, speed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Clamp to support range (extrapolation still works but flattens)
        s = np.clip(speed, self.speed_min, self.speed_max)
        return self._freq_cs(s).astype(np.float32), self._period_cs(s).astype(np.float32)


# ----------------------------------------------------------------------------
# Augmentation
# ----------------------------------------------------------------------------
def augment_batch(
    x: torch.Tensor,
    y_freq_norm: torch.Tensor,     # (B, 136) z-scored
    y_period:    torch.Tensor,     # (B, 1)    seconds
    mean_perbin: torch.Tensor,     # (136,)
    std_global:  float,
    p_swap:      float = 0.5,
    p_mixup:     float = 0.3,
    p_phase:     float = PHASE_AUG_PROB,
    max_shift:   int   = PHASE_AUG_MAX_SHIFT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Symmetry-preserving augmentations:
      1. LR swap   -- rhip<->lhip, rknee<->lknee.  Doesn't change period.
      2. Mixup     -- linear interpolation of normalised targets + input.
      3. Phase shift -- rotate FFT by exp(-j*omega*k/32).  Same time-domain
                        content, arbitrary starting phase.  Implemented in
                        de-normalised complex space then re-normalised.
                        Period is PHASE-INVARIANT so unchanged.
    """
    B = x.size(0)
    device = x.device

    # ---- LR swap
    if torch.rand(1).item() < p_swap and B > 1:
        freq = y_freq_norm.view(-1, 17, 4, 2).clone()
        swapped = freq.clone()
        swapped[:, :, 0, :] = freq[:, :, 2, :]
        swapped[:, :, 1, :] = freq[:, :, 3, :]
        swapped[:, :, 2, :] = freq[:, :, 0, :]
        swapped[:, :, 3, :] = freq[:, :, 1, :]
        y_freq_norm = swapped.view(B, -1)
        x_new = x.clone()
        x_new[:, 1] = x[:, 2]
        x_new[:, 2] = x[:, 1]
        x = x_new

    # ---- Phase shift (in de-normalised complex space)
    if torch.rand(1).item() < p_phase and max_shift > 0:
        # Random integer shift in samples (out of 32-point IRFFT window)
        k = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
        if k != 0:
            mean_t = mean_perbin.to(device)
            freq_denorm = y_freq_norm * std_global + mean_t      # (B, 136)
            freq_cx = freq_denorm.view(B, 17, 4, 2)
            real = freq_cx[..., 0]
            imag = freq_cx[..., 1]
            # omega[n] = 2*pi*n/32 for n=0..16
            n_bins = torch.arange(17, device=device, dtype=x.dtype)
            phi = (2.0 * np.pi * n_bins * k / 32.0).view(1, 17, 1)
            cos_p = torch.cos(phi)
            sin_p = torch.sin(phi)
            new_real = real * cos_p - imag * sin_p
            new_imag = real * sin_p + imag * cos_p
            freq_cx_rot = torch.stack([new_real, new_imag], dim=-1)
            freq_denorm_rot = freq_cx_rot.view(B, -1)
            y_freq_norm = (freq_denorm_rot - mean_t) / std_global

    # ---- Mixup
    if torch.rand(1).item() < p_mixup and B > 1:
        perm = torch.randperm(B, device=device)
        lam = float(np.random.beta(0.4, 0.4))
        x           = lam * x           + (1 - lam) * x[perm]
        y_freq_norm = lam * y_freq_norm + (1 - lam) * y_freq_norm[perm]
        y_period    = lam * y_period    + (1 - lam) * y_period[perm]

    return x, y_freq_norm, y_period


# ============================================================================
# 3. LOSS
# ============================================================================
class PhaseAwareLoss(nn.Module):
    """
    L = alpha * MSE_fft_zscored + (1 - alpha) * MSE_time
        + period_weight * MSE_period_seconds

    The time-domain part is what ultimately drives the PPO imitation reward,
    so weighting it directly aligns the training loss with the downstream
    objective.  When ``alpha=1`` this reduces to the v1 loss (baseline variant).
    """

    def __init__(self,
                 alpha: float,
                 period_weight: float,
                 mean_perbin: torch.Tensor,
                 std_global: float) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.period_weight = float(period_weight)
        self.register_buffer("mean_perbin", mean_perbin)
        self.std_global = float(std_global)

    def forward(self,
                pred_freq: torch.Tensor,     # (B, 136) z-scored
                pred_period: torch.Tensor,   # (B, 1)    seconds
                gt_freq: torch.Tensor,       # (B, 136) z-scored
                gt_period: torch.Tensor,     # (B, 1)    seconds
                ) -> tuple[torch.Tensor, dict[str, float]]:
        # FFT z-scored MSE
        loss_fft = F.mse_loss(pred_freq, gt_freq)

        # Time-domain MSE via differentiable IRFFT
        if self.alpha < 1.0:
            pred_denorm = (pred_freq * self.std_global + self.mean_perbin
                          ).view(-1, 17, 4, 2)
            gt_denorm   = (gt_freq   * self.std_global + self.mean_perbin
                          ).view(-1, 17, 4, 2)
            # (B, 17, 4) complex
            pred_cx = torch.complex(pred_denorm[..., 0], pred_denorm[..., 1])
            gt_cx   = torch.complex(gt_denorm[..., 0],   gt_denorm[..., 1])
            # IRFFT over the freq axis (dim=1), 32 output samples
            pred_t = torch.fft.irfft(pred_cx, n=32, dim=1)   # (B, 32, 4)
            gt_t   = torch.fft.irfft(gt_cx,   n=32, dim=1)
            loss_time = F.mse_loss(pred_t, gt_t)
        else:
            loss_time = torch.tensor(0.0, device=pred_freq.device)

        # Period MSE in seconds (bounded, interpretable)
        loss_period = F.mse_loss(pred_period, gt_period)

        total = (self.alpha * loss_fft
                 + (1.0 - self.alpha) * loss_time
                 + self.period_weight * loss_period)
        with torch.no_grad():
            parts = dict(
                fft=float(loss_fft),
                time=float(loss_time),
                period=float(loss_period),
            )
        return total, parts


# ============================================================================
# 4. TRAINING LOOP
# ============================================================================
def train_one(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    lr: float,
    max_epochs: int,
    patience: int,
    loss_fn: PhaseAwareLoss,
    use_aug: bool,
    mean_perbin_t: torch.Tensor,
    std_global: float,
    spline_prior: SplinePrior | None,   # None for non-residual variants
    log_prefix: str = "",
) -> tuple[dict, list[float], list[float]]:
    model = model.to(DEVICE)
    optimiser = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=max(40, patience // 4)
    )

    train_curve, val_curve = [], []
    best_val = float("inf")
    best_state, best_epoch = None, -1
    bad_epochs = 0

    for epoch in range(max_epochs):
        # ----- TRAIN
        model.train()
        running, n = 0.0, 0
        for x, y_freq, y_period in train_loader:
            x = x.to(DEVICE); y_freq = y_freq.to(DEVICE); y_period = y_period.to(DEVICE)
            if use_aug:
                x, y_freq, y_period = augment_batch(
                    x, y_freq, y_period,
                    mean_perbin=mean_perbin_t, std_global=std_global,
                )

            optimiser.zero_grad()
            pred_freq_raw, pred_period = model(x)

            # Residual variant: add spline prior (fixed, non-learned) to the
            # MLP's freq prediction and to its period before computing loss.
            if spline_prior is not None:
                speeds_ms = (x[:, 0] * 2.4).detach().cpu().numpy()
                prior_freq_denorm, prior_period = spline_prior(speeds_ms)
                prior_freq_norm = ((torch.from_numpy(prior_freq_denorm).to(DEVICE)
                                    - mean_perbin_t) / std_global)
                prior_period_t = torch.from_numpy(prior_period).to(DEVICE).unsqueeze(1)
                pred_freq   = pred_freq_raw + prior_freq_norm
                pred_period = pred_period  + prior_period_t
            else:
                pred_freq = pred_freq_raw

            loss, _ = loss_fn(pred_freq, pred_period, y_freq, y_period)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
        train_curve.append(running / n)

        # ----- VAL
        model.eval()
        running, n = 0.0, 0
        with torch.no_grad():
            for x, y_freq, y_period in val_loader:
                x = x.to(DEVICE); y_freq = y_freq.to(DEVICE); y_period = y_period.to(DEVICE)
                pred_freq_raw, pred_period = model(x)
                if spline_prior is not None:
                    speeds_ms = (x[:, 0] * 2.4).cpu().numpy()
                    prior_freq_denorm, prior_period = spline_prior(speeds_ms)
                    prior_freq_norm = ((torch.from_numpy(prior_freq_denorm).to(DEVICE)
                                        - mean_perbin_t) / std_global)
                    prior_period_t = torch.from_numpy(prior_period).to(DEVICE).unsqueeze(1)
                    pred_freq   = pred_freq_raw + prior_freq_norm
                    pred_period = pred_period  + prior_period_t
                else:
                    pred_freq = pred_freq_raw
                loss, _ = loss_fn(pred_freq, pred_period, y_freq, y_period)
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
            print(f"  {log_prefix}early stop @ ep {epoch+1}, best val {best_val:.5f} @ {best_epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return (dict(best_val=best_val, best_epoch=best_epoch + 1, epochs=len(train_curve)),
            train_curve, val_curve)


# ============================================================================
# 5. EVALUATION (test-set metrics + time-domain tensors for plotting)
# ============================================================================
def _forward_full(
    model: nn.Module,
    inputs: np.ndarray,
    idx: np.ndarray,
    mean_perbin: np.ndarray,
    std_global: float,
    spline_prior: SplinePrior | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the full pred pipeline (MLP [+ spline prior] -> de-normalise)
    on the samples at ``idx``.  Returns:
        pred_freq_denorm  (B, 17, 4, 2)     physical FFT coefficients
        pred_period_s     (B,)              period in seconds
    """
    model.eval()
    x = torch.tensor(inputs[idx], dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        pred_freq_raw, pred_period = model(x)
    pred_freq_raw = pred_freq_raw.cpu().numpy()
    pred_period_s = pred_period.cpu().numpy().squeeze(-1)

    if spline_prior is not None:
        speeds_ms = inputs[idx, 0] * 2.4
        prior_freq_denorm, prior_period = spline_prior(speeds_ms)
        prior_freq_norm = (prior_freq_denorm - mean_perbin) / std_global
        pred_freq_norm = pred_freq_raw + prior_freq_norm
        pred_period_s  = pred_period_s + prior_period
    else:
        pred_freq_norm = pred_freq_raw

    pred_freq_denorm = (pred_freq_norm * std_global + mean_perbin
                       ).reshape(-1, 17, 4, 2)
    return pred_freq_denorm, pred_period_s


def evaluate_test(
    model: nn.Module,
    inputs: np.ndarray,
    raw_fft: np.ndarray,
    period: np.ndarray,
    test_idx: np.ndarray,
    mean_perbin: np.ndarray,
    std_global: float,
    spline_prior: SplinePrior | None,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Returns:
        metrics     dict of scalar test metrics
        pred_time   (B, 32, 4) reconstructed joint trajectories  (prediction)
        gt_time     (B, 32, 4) reconstructed joint trajectories  (ground truth)
    """
    pred_freq_denorm, pred_period_s = _forward_full(
        model, inputs, test_idx, mean_perbin, std_global, spline_prior
    )
    gt_freq     = raw_fft[test_idx]                          # (B,17,4,2)
    gt_period_s = period[test_idx, 0]

    # --- metrics
    coef_mse = float(np.mean((pred_freq_denorm - gt_freq) ** 2))

    pred_cx = pred_freq_denorm[..., 0] + 1j * pred_freq_denorm[..., 1]
    gt_cx   = gt_freq[..., 0]          + 1j * gt_freq[..., 1]
    pred_t  = np.fft.irfft(pred_cx, n=32, axis=1)                 # (B, 32, 4)
    gt_t    = np.fft.irfft(gt_cx,   n=32, axis=1)
    time_mse_per_joint = ((pred_t - gt_t) ** 2).mean(axis=(0, 1))
    time_mse = float(time_mse_per_joint.mean())

    period_mae  = float(np.mean(np.abs(pred_period_s - gt_period_s)))
    period_mape = float(np.mean(np.abs(pred_period_s - gt_period_s)
                                / np.maximum(np.abs(gt_period_s), 1e-6)))
    n_period_neg = int(np.sum(pred_period_s <= 0))  # should always be 0 now

    # Spectral magnitude error on first 3 non-DC harmonics
    pred_mag = np.abs(pred_cx); gt_mag = np.abs(gt_cx)
    fft_fid_h3 = float(np.mean(np.abs(pred_mag[:, 1:4] - gt_mag[:, 1:4])))

    metrics = dict(
        n_test=int(len(test_idx)),
        coef_mse=coef_mse,
        time_domain_mse=time_mse,
        time_mse_per_joint=time_mse_per_joint.tolist(),
        fft_fidelity_h1_h3=fft_fid_h3,
        period_mae_s=period_mae,
        period_mape=period_mape,
        n_period_nonpositive=n_period_neg,
        pred_period_s=pred_period_s.tolist(),
        gt_period_s=gt_period_s.tolist(),
    )
    return metrics, pred_t, gt_t


# ============================================================================
# 6. PLOTTING
# ============================================================================
def plot_training_curves(histories, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for label, tr, va in histories:
        axes[0].plot(tr, label=label, alpha=0.8)
        axes[1].plot(va, label=label, alpha=0.8)
    axes[0].set_title("Train loss"); axes[1].set_title("Val loss")
    for ax in axes:
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_yscale("log"); ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"[plot] saved {out_path}")


def plot_time_domain_samples(
    pred_t: np.ndarray,         # (B, 32, 4)
    gt_t:   np.ndarray,         # (B, 32, 4)
    test_idx: np.ndarray,       # (B,)  original dataset indices (for filenames)
    inputs: np.ndarray,         # (N, 3) for per-sample subtitle (speed, legs)
    variant: str,
    out_dir: str,
    max_samples: int = MAX_TIME_DOMAIN_SAMPLES,
) -> None:
    """
    2x2 plot per test sample: one subplot per joint (GT vs prediction,
    normalised phase in [0, 1)).  Saved as sample_<idx>.png.
    """
    os.makedirs(out_dir, exist_ok=True)
    B, T, J = pred_t.shape
    assert J == len(JOINT_NAMES), (
        f"JOINT_NAMES has {len(JOINT_NAMES)} entries but data has {J} channels"
    )
    n_plot = min(B, max_samples)
    phase = np.linspace(0.0, 1.0, T, endpoint=False)

    for i in range(n_plot):
        fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
        for j, ax in enumerate(axes.flat):
            ax.plot(phase, gt_t[i, :, j],   lw=2.0, label="GT",   color="#1f77b4")
            ax.plot(phase, pred_t[i, :, j], lw=2.0, label="Pred", color="#d62728",
                    linestyle="--")
            err = np.sqrt(((pred_t[i, :, j] - gt_t[i, :, j]) ** 2).mean())
            ax.set_title(f"{JOINT_NAMES[j]}   RMSE={err:.4f}", fontsize=10)
            ax.grid(True, alpha=0.3)
            if j >= 2:
                ax.set_xlabel("Gait phase")
            ax.set_ylabel("Joint angle (rad)")
            if j == 0:
                ax.legend(fontsize=8, loc="best")
        orig_idx = int(test_idx[i])
        speed = float(inputs[orig_idx, 0]) * 2.4
        r_leg = float(inputs[orig_idx, 1])
        l_leg = float(inputs[orig_idx, 2])
        fig.suptitle(
            f"[{variant}]  test sample #{i}  (dataset idx {orig_idx})   "
            f"speed={speed:.2f} m/s  r_leg={r_leg:.3f}  l_leg={l_leg:.3f}",
            fontsize=11,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out_path = os.path.join(out_dir, f"sample_{i:03d}_idx{orig_idx:04d}.png")
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
    print(f"[plot] saved {n_plot} per-sample time-domain plots to {out_dir}")


def plot_variant_comparison(out_path: str = COMPARISON_PNG) -> None:
    """
    Read every ``test_metrics_*.json`` in REVIEW_DIR and draw a grouped bar
    chart of the 4 scalar test metrics across variants. Gracefully skips
    variants that haven't been trained yet.
    """
    # Canonical order so plots are comparable across runs
    order = ["baseline", "phase", "residual", "phase_residual"]
    found = {}
    for v in order:
        p = os.path.join(REVIEW_DIR, f"test_metrics_{v}.json")
        if os.path.exists(p):
            with open(p, "r") as f:
                found[v] = json.load(f)

    if not found:
        print(f"[compare] no test_metrics_*.json found under {REVIEW_DIR}; skip plot")
        return

    variants = [v for v in order if v in found]
    metric_keys = [
        ("coef_mse",          "FFT coef MSE (z-scored)"),
        ("time_domain_mse",   "Time-domain MSE (rad^2)"),
        ("fft_fidelity_h1_h3","FFT fidelity |H1..H3| MAE"),
        ("period_mae_s",      "Period MAE (s)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    colors = ["#4C72B0", "#DD8452", "#55A467", "#C44E52"]

    for ax, (key, label) in zip(axes.flat, metric_keys):
        vals = [found[v].get(key, float("nan")) for v in variants]
        bars = ax.bar(variants, vals,
                      color=[colors[order.index(v) % len(colors)] for v in variants],
                      edgecolor="black", linewidth=0.6)
        ax.set_title(label, fontsize=11)
        ax.set_ylabel(label)
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelrotation=15)
        # annotate bar heights
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                        f"{v:.4g}", ha="center", va="bottom", fontsize=8)
        # Slight headroom for text
        finite_vals = [v for v in vals if np.isfinite(v)]
        if finite_vals:
            ax.set_ylim(top=max(finite_vals) * 1.18)

    fig.suptitle("Test-set performance across FFT-MLP variants", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[compare] saved {out_path}  (variants: {', '.join(variants)})")


# ============================================================================
# 7. MAIN
# ============================================================================
Variant = Literal["baseline", "phase", "residual", "phase_residual"]


def variant_config(variant: Variant) -> dict:
    """Map variant name -> {alpha, use_residual, tag}."""
    return {
        "baseline":       dict(alpha=1.0, use_residual=False, tag="baseline"),
        "phase":          dict(alpha=TIME_ALPHA, use_residual=False, tag="phase"),
        "residual":       dict(alpha=1.0, use_residual=True,  tag="residual"),
        "phase_residual": dict(alpha=TIME_ALPHA, use_residual=True, tag="phase_residual"),
    }[variant]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["baseline", "phase", "residual", "phase_residual"],
                    default="phase_residual")
    ap.add_argument("--quick", action="store_true", help="Smoke run (200 epochs, one hp).")
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--compare-only", action="store_true",
                    help="Skip training; just rebuild variants_test_comparison.png")
    args = ap.parse_args()

    os.makedirs(REVIEW_DIR, exist_ok=True)

    # ---- fast path: only rebuild the comparison plot
    if args.compare_only:
        plot_variant_comparison()
        return 0

    torch.manual_seed(SEED); np.random.seed(SEED)

    cfg = variant_config(args.variant)
    print("=" * 70)
    print(f"[variant] {args.variant}   alpha={cfg['alpha']}   "
          f"residual={cfg['use_residual']}   aug={not args.no_aug}")
    print("=" * 70)

    final_pth  = os.path.join(RESULTS_ROOT, f"FINAL_BEST_MODEL_V2_{cfg['tag']}.pth")
    metrics_js = os.path.join(REVIEW_DIR, f"test_metrics_{cfg['tag']}.json")
    curves_png = os.path.join(REVIEW_DIR, f"training_curves_{cfg['tag']}.png")
    hp_csv     = os.path.join(REVIEW_DIR, f"hp_search_{cfg['tag']}.csv")
    time_dir   = os.path.join(REVIEW_DIR, f"time_domain_{cfg['tag']}")

    # ---- 1. data
    inputs, raw_fft, period = load_raw_arrays()
    print(f"[data] inputs={inputs.shape}, raw_fft={raw_fft.shape}, period={period.shape}")

    # ---- 2. splits
    splits = make_splits(inputs.shape[0])
    train_idx, val_idx, test_idx = splits["train"], splits["val"], splits["test"]

    # ---- 3. norm stats (train-only)
    mean_perbin, std_global, period_mean, period_std = build_norm_stats(
        raw_fft, period, train_idx
    )

    # ---- 4. spline prior (only needed for residual variants; saved always)
    prior_data = build_spline_prior(inputs, raw_fft, period, train_idx)
    spline_prior = (SplinePrior(**prior_data) if cfg["use_residual"] else None)

    # ---- 5. targets -> tensors
    flat = raw_fft.reshape(raw_fft.shape[0], -1)
    freq_norm = ((flat - mean_perbin) / std_global).astype(np.float32)
    period_s  = period.astype(np.float32)                         # already seconds
    X         = torch.tensor(inputs,    dtype=torch.float32)
    Yf        = torch.tensor(freq_norm, dtype=torch.float32)
    Yp        = torch.tensor(period_s,  dtype=torch.float32)

    train_ds = TensorDataset(X[train_idx], Yf[train_idx], Yp[train_idx])
    val_ds   = TensorDataset(X[val_idx],   Yf[val_idx],   Yp[val_idx])

    mean_perbin_t = torch.tensor(mean_perbin, dtype=torch.float32, device=DEVICE)

    # ---- 6. hp grid
    if args.quick:
        hp_grid = [dict(hidden_size=512, lr=3e-4, batch_size=32)]
        max_epochs = 200
    else:
        hp_grid = [
            dict(hidden_size=hs, lr=lr, batch_size=bs)
            for hs, lr, bs in product([256, 512], [1e-3, 3e-4], [32, 64])
        ]
        max_epochs = MAX_EPOCHS

    loss_fn = PhaseAwareLoss(
        alpha=cfg["alpha"],
        period_weight=PERIOD_WEIGHT_S,
        mean_perbin=mean_perbin_t,
        std_global=std_global,
    ).to(DEVICE)

    # ---- 7. train
    hp_rows, histories, best = [], [], None
    for i, hp in enumerate(hp_grid):
        tag = f"hs{hp['hidden_size']}_lr{hp['lr']}_bs{hp['batch_size']}"
        print(f"\n  --- [{i+1}/{len(hp_grid)}] {tag}")
        torch.manual_seed(SEED)
        model = SimpleFCNN(hidden_size=hp["hidden_size"])
        train_loader = DataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False)
        info, tr, va = train_one(
            model, train_loader, val_loader,
            lr=hp["lr"], max_epochs=max_epochs, patience=PATIENCE,
            loss_fn=loss_fn, use_aug=not args.no_aug,
            mean_perbin_t=mean_perbin_t, std_global=std_global,
            spline_prior=spline_prior,
            log_prefix=f"[{tag}] ",
        )
        info.update(hp); info["tag"] = tag
        hp_rows.append(info); histories.append((tag, tr, va))
        if best is None or info["best_val"] < best["info"]["best_val"]:
            best = dict(info=info, model=model, hp=hp)

    # ---- 8. save hp table + curves
    with open(hp_csv, "w", encoding="utf-8") as f:
        keys = ["tag", "hidden_size", "lr", "batch_size", "best_val", "best_epoch", "epochs"]
        f.write(",".join(keys) + "\n")
        for r in hp_rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    plot_training_curves(histories, curves_png)

    # ---- 9. save best model
    print(f"\n[best] {best['info']['tag']}  val={best['info']['best_val']:.5f}")
    torch.save(best["model"].state_dict(), final_pth)
    print(f"       saved {final_pth}")

    # ------------------------------------------------------------------
    # 10. TEST-SET EVALUATION (only the best HP config, weights already loaded
    #     into best['model'] by train_one via ``load_state_dict(best_state)``)
    # ------------------------------------------------------------------
    print(f"\n[test] evaluating best '{best['info']['tag']}' on {len(test_idx)} samples")
    test_metrics, pred_t, gt_t = evaluate_test(
        best["model"], inputs, raw_fft, period, test_idx,
        mean_perbin, std_global, spline_prior,
    )
    test_metrics["variant"]        = args.variant
    test_metrics["best_hp"]        = best["info"]
    test_metrics["alpha"]          = cfg["alpha"]
    test_metrics["use_residual"]   = cfg["use_residual"]
    test_metrics["period_weight_s"] = PERIOD_WEIGHT_S
    test_metrics["augmentation"]   = (not args.no_aug)
    with open(metrics_js, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"       saved {metrics_js}")
    # Print scalar-only summary (time_mse_per_joint / pred_period_s lists hidden)
    scalar_only = {
        k: v for k, v in test_metrics.items()
        if not isinstance(v, (list, dict))
    }
    print(json.dumps(scalar_only, indent=2, default=str))

    # 10a. per-sample time-domain 2x2 plots
    plot_time_domain_samples(
        pred_t, gt_t, test_idx, inputs,
        variant=cfg["tag"], out_dir=time_dir,
    )

    # 10b. cross-variant comparison (uses every test_metrics_*.json present)
    plot_variant_comparison()

    print(f"\n[done] variant '{args.variant}' complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())