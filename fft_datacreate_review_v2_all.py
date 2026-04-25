"""
fft_datacreate_review_v2.py
===========================

Improved review training of the FFT-MLP gait generator.

Key features
------------
1. **Positive-period head (softplus)** — period strictly > 0 in seconds.
2. **Phase-aware (time-domain) loss** — differentiable IRFFT MSE available
   via ``--loss phase`` and friends.
3. **Cubic-spline residual head (optional)** — controlled by ``--variant residual``.
4. **Phase-shift + LR-swap + mixup augmentation** on the fly.
5. **Loss function is a first-class hyperparameter** via ``--loss``:
       notebook          MSE_fft_z + 5 * MSE_period_raw_s          (reproduce ipynb)
       notebook_zperiod  MSE_fft_z + 5 * MSE_period_z              (remove unit mismatch)
       baseline_v2       MSE_fft_z + 0.5 * MSE_period_s            (old v2 default)
       phase             a*MSE_fft_z + (1-a)*MSE_time + 0.5*MSE_period_s
       time_heavy_period a*MSE_fft_z + (1-a)*MSE_time + 5*MSE_period_s
       time_only         MSE_time + 5 * MSE_period_s
6. **Fair-comparison val metric** — loss-invariant, used for early
   stopping, HP selection, and cross-config ranking:
       val_fair = MSE_time_domain(rad^2) + 5 * MSE_period_seconds^2
7. **Only baseline + residual variants** — ``phase`` / ``phase_residual``
   are redundant once ``--loss`` exposes the time-domain branch directly.
8. **Per-config HP grid.** Each (variant, loss) config runs the full HP
   grid internally; the best HP (by val_fair) is retained. This avoids
   assuming one HP point wins across all losses.

Run
---
    # Full sweep: 2 variants x 6 losses = 12 configs, each with 8 HP
    # points -> 96 training runs. Keeps the BEST HP per config.
    python fft_datacreate_review_v2.py --all

    # Single-config (still runs full HP grid):
    python fft_datacreate_review_v2.py --variant residual --loss phase

    # Smoke test (forces single HP + 200 epochs):
    python fft_datacreate_review_v2.py --all --quick

    # Skip HP search, use one robust point (hs=256, lr=3e-4, bs=64):
    python fft_datacreate_review_v2.py --all --hp-grid single

    # Just rebuild comparison plot:
    python fft_datacreate_review_v2.py --compare-only
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

COMPARISON_PNG = os.path.join(REVIEW_DIR, "variants_test_comparison.png")

JOINT_NAMES = ["RHip", "RKnee", "LHip", "LKnee"]


# ============================================================================
# CONFIG
# ============================================================================
INPUT_SIZE   = 3
FREQ_DIM     = 136
OUTPUT_SIZE  = FREQ_DIM + 1

SEED         = 42
TRAIN_FRAC   = 0.80
VAL_FRAC     = 0.15
TEST_FRAC    = 0.05
PATIENCE     = 250
MAX_EPOCHS   = 4000

TIME_ALPHA            = 0.4
PHASE_AUG_PROB        = 0.5
PHASE_AUG_MAX_SHIFT   = 4
DEVICE                = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_TIME_DOMAIN_SAMPLES = 60

NOTEBOOK_PERIOD_WEIGHT = 5.0
V2_PERIOD_WEIGHT_S     = 0.5
FAIR_VAL_PERIOD_WEIGHT = 5.0


# ============================================================================
# HP GRIDS
# ============================================================================
# Full grid -- cartesian product yields 8 HP points per (variant, loss) config.
# Each point is trained to convergence; best (by val_fair) is kept.
HP_GRID_FULL: list[dict] = [
    dict(hidden_size=hs, lr=lr, batch_size=bs)
    for hs, lr, bs in product([256, 512], [3e-4, 1e-3], [32, 64])
]

# Single-point grid -- for smoke tests or when the user wants to skip HP
# search. Chosen as the robust default: smaller network for a ~900-sample
# train set, larger batch for gradient stability, 3e-4 Adam LR since
# time-domain losses have smaller gradient magnitude than coef MSE.
HP_GRID_SINGLE: list[dict] = [
    dict(hidden_size=256, lr=3e-4, batch_size=64),
]


def _hp_tag(hp: dict) -> str:
    return f"hs{hp['hidden_size']}_lr{hp['lr']}_bs{hp['batch_size']}"


# ============================================================================
# 1. ARCHITECTURE
# ============================================================================
class SimpleFCNN(nn.Module):
    """Backbone + freq_head + softplus period_head (> 0 seconds)."""

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
        nn.init.constant_(self.period_head.bias, 0.4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        freq = self.freq_head(h)
        period = F.softplus(self.period_head(h))
        return freq, period


# ============================================================================
# 2. DATA
# ============================================================================
def load_raw_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inputs  = np.load(INPUT_NPY).astype(np.float32)
    raw_fft = np.load(OUTPUT_NPY).astype(np.float32)
    period  = np.load(PERIOD_NPY).astype(np.float32)
    assert inputs.shape[0] == raw_fft.shape[0] == period.shape[0]
    return inputs, raw_fft, period


def make_splits(n: int, seed: int = SEED) -> dict[str, np.ndarray]:
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
    flat = raw_fft.reshape(raw_fft.shape[0], -1)
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


def build_spline_prior(
    inputs: np.ndarray, raw_fft: np.ndarray, period: np.ndarray,
    train_idx: np.ndarray, n_knots: int = 16,
) -> dict[str, np.ndarray]:
    speeds_train = inputs[train_idx, 0] * 2.4
    flat_train   = raw_fft[train_idx].reshape(len(train_idx), -1)
    period_train = period[train_idx, 0]

    qs = np.linspace(0.0, 1.0, n_knots)
    knot_speeds = np.quantile(speeds_train, qs)
    knot_speeds = np.unique(knot_speeds)

    def _avg_per_knot(vals: np.ndarray) -> np.ndarray:
        ys = []
        for i in range(len(knot_speeds)):
            lo = knot_speeds[i - 1] if i > 0 else -np.inf
            hi = knot_speeds[i + 1] if i + 1 < len(knot_speeds) else np.inf
            mask = (speeds_train >= lo) & (speeds_train <= hi)
            ys.append(vals[mask].mean(axis=0) if mask.sum() > 0 else vals.mean(axis=0))
        return np.stack(ys, axis=0)

    freq_knots   = _avg_per_knot(flat_train)
    period_knots = _avg_per_knot(period_train)

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
    def __init__(self, knot_speeds: np.ndarray,
                 freq_knots: np.ndarray, period_knots: np.ndarray) -> None:
        self._freq_cs   = CubicSpline(knot_speeds, freq_knots, axis=0,
                                      bc_type="natural", extrapolate=True)
        self._period_cs = CubicSpline(knot_speeds, period_knots,
                                      bc_type="natural", extrapolate=True)
        self.speed_min = float(knot_speeds[0])
        self.speed_max = float(knot_speeds[-1])

    def __call__(self, speed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = np.clip(speed, self.speed_min, self.speed_max)
        return self._freq_cs(s).astype(np.float32), self._period_cs(s).astype(np.float32)


def augment_batch(
    x: torch.Tensor,
    y_freq_norm: torch.Tensor,
    y_period:    torch.Tensor,
    mean_perbin: torch.Tensor,
    std_global:  float,
    p_swap:      float = 0.5,
    p_mixup:     float = 0.3,
    p_phase:     float = PHASE_AUG_PROB,
    max_shift:   int   = PHASE_AUG_MAX_SHIFT,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B = x.size(0)
    device = x.device

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

    if torch.rand(1).item() < p_phase and max_shift > 0:
        k = int(torch.randint(-max_shift, max_shift + 1, (1,)).item())
        if k != 0:
            mean_t = mean_perbin.to(device)
            freq_denorm = y_freq_norm * std_global + mean_t
            freq_cx = freq_denorm.view(B, 17, 4, 2)
            real = freq_cx[..., 0]
            imag = freq_cx[..., 1]
            n_bins = torch.arange(17, device=device, dtype=x.dtype)
            phi = (2.0 * np.pi * n_bins * k / 32.0).view(1, 17, 1)
            cos_p = torch.cos(phi)
            sin_p = torch.sin(phi)
            new_real = real * cos_p - imag * sin_p
            new_imag = real * sin_p + imag * cos_p
            freq_cx_rot = torch.stack([new_real, new_imag], dim=-1)
            freq_denorm_rot = freq_cx_rot.view(B, -1)
            y_freq_norm = (freq_denorm_rot - mean_t) / std_global

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
LossKind = Literal[
    "notebook_zperiod",
    "baseline_v2",
    "time_heavy_period",
]

_LOSS_KINDS: tuple[str, ...] = (
    "notebook_zperiod",
    "baseline_v2",
    "time_heavy_period",
)


class ConfigurableLoss(nn.Module):
    """See module docstring and ``--loss`` help for per-kind semantics."""

    def __init__(
        self,
        loss_kind: LossKind,
        alpha: float,
        mean_perbin: torch.Tensor,
        std_global: float,
    ) -> None:
        super().__init__()
        assert loss_kind in _LOSS_KINDS, f"unknown loss kind: {loss_kind}"
        self.loss_kind  = str(loss_kind)
        self.alpha      = float(alpha)
        self.std_global = float(std_global)
        self.register_buffer("mean_perbin", mean_perbin)

    def _time_domain_mse(
        self, pred_freq_z: torch.Tensor, gt_freq_z: torch.Tensor,
    ) -> torch.Tensor:
        pred_denorm = (pred_freq_z * self.std_global + self.mean_perbin
                       ).view(-1, 17, 4, 2)
        gt_denorm   = (gt_freq_z   * self.std_global + self.mean_perbin
                       ).view(-1, 17, 4, 2)
        pred_cx = torch.complex(pred_denorm[..., 0], pred_denorm[..., 1])
        gt_cx   = torch.complex(gt_denorm[..., 0],   gt_denorm[..., 1])
        pred_t  = torch.fft.irfft(pred_cx, n=32, dim=1)
        gt_t    = torch.fft.irfft(gt_cx,   n=32, dim=1)
        return F.mse_loss(pred_t, gt_t)

    def forward(
        self,
        pred_freq:   torch.Tensor,
        pred_period: torch.Tensor,
        gt_freq:     torch.Tensor,
        gt_period:   torch.Tensor,
        period_mean: float,
        period_std:  float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        mse_fft     = F.mse_loss(pred_freq, gt_freq)
        mse_per_sec = F.mse_loss(pred_period, gt_period)

        if self.loss_kind == "notebook_zperiod":
            p_std = max(period_std, 1e-6)
            pred_pz = (pred_period - period_mean) / p_std
            gt_pz   = (gt_period   - period_mean) / p_std
            mse_per_z = F.mse_loss(pred_pz, gt_pz)
        else:
            mse_per_z = torch.tensor(0.0, device=pred_freq.device)

        needs_time = self.loss_kind in ("phase", "time_heavy_period", "time_only")
        mse_time = (self._time_domain_mse(pred_freq, gt_freq)
                    if needs_time
                    else torch.tensor(0.0, device=pred_freq.device))

        if self.loss_kind == "notebook":
            total = mse_fft + NOTEBOOK_PERIOD_WEIGHT * mse_per_sec
        elif self.loss_kind == "notebook_zperiod":
            total = mse_fft + NOTEBOOK_PERIOD_WEIGHT * mse_per_z
        elif self.loss_kind == "baseline_v2":
            total = mse_fft + V2_PERIOD_WEIGHT_S * mse_per_sec
        elif self.loss_kind == "phase":
            total = (self.alpha * mse_fft
                     + (1.0 - self.alpha) * mse_time
                     + V2_PERIOD_WEIGHT_S * mse_per_sec)
        elif self.loss_kind == "time_heavy_period":
            total = (self.alpha * mse_fft
                     + (1.0 - self.alpha) * mse_time
                     + NOTEBOOK_PERIOD_WEIGHT * mse_per_sec)
        elif self.loss_kind == "time_only":
            total = mse_time + NOTEBOOK_PERIOD_WEIGHT * mse_per_sec
        else:
            raise RuntimeError(f"unreachable loss kind: {self.loss_kind}")

        with torch.no_grad():
            parts = dict(
                fft=float(mse_fft),
                time=float(mse_time),
                period_sec=float(mse_per_sec),
                period_z=float(mse_per_z),
            )
        return total, parts


def compute_fair_val_metric(
    pred_freq_z: torch.Tensor,
    pred_period: torch.Tensor,
    gt_freq_z:   torch.Tensor,
    gt_period:   torch.Tensor,
    mean_perbin: torch.Tensor,
    std_global:  float,
) -> tuple[float, float, float]:
    """Loss-invariant val metric. See module docstring."""
    pred_denorm = (pred_freq_z * std_global + mean_perbin).view(-1, 17, 4, 2)
    gt_denorm   = (gt_freq_z   * std_global + mean_perbin).view(-1, 17, 4, 2)
    pred_cx = torch.complex(pred_denorm[..., 0], pred_denorm[..., 1])
    gt_cx   = torch.complex(gt_denorm[..., 0],   gt_denorm[..., 1])
    pred_t  = torch.fft.irfft(pred_cx, n=32, dim=1)
    gt_t    = torch.fft.irfft(gt_cx,   n=32, dim=1)
    time_mse = float(F.mse_loss(pred_t, gt_t))
    period_mse_sec = float(F.mse_loss(pred_period, gt_period))
    val_fair = time_mse + FAIR_VAL_PERIOD_WEIGHT * period_mse_sec
    return val_fair, time_mse, period_mse_sec


# ============================================================================
# 4. TRAINING LOOP
# ============================================================================
def _apply_prior_if_needed(
    pred_freq_raw: torch.Tensor,
    pred_period:   torch.Tensor,
    x: torch.Tensor,
    spline_prior: SplinePrior | None,
    mean_perbin_t: torch.Tensor,
    std_global: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if spline_prior is None:
        return pred_freq_raw, pred_period
    speeds_ms = (x[:, 0] * 2.4).detach().cpu().numpy()
    prior_freq_denorm, prior_period = spline_prior(speeds_ms)
    prior_freq_norm = ((torch.from_numpy(prior_freq_denorm).to(x.device)
                        - mean_perbin_t) / std_global)
    prior_period_t = torch.from_numpy(prior_period).to(x.device).unsqueeze(1)
    return pred_freq_raw + prior_freq_norm, pred_period + prior_period_t


def train_one(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    lr: float,
    max_epochs: int,
    patience: int,
    loss_fn: ConfigurableLoss,
    period_mean: float,
    period_std: float,
    use_aug: bool,
    mean_perbin_t: torch.Tensor,
    std_global: float,
    spline_prior: SplinePrior | None,
    log_prefix: str = "",
) -> tuple[dict, list[float], list[float], list[float]]:
    """Train one HP point to convergence. Early-stopping on val_fair."""
    model = model.to(DEVICE)
    optimiser = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=max(40, patience // 4)
    )

    train_curve, val_curve_inloss, val_curve_fair = [], [], []
    best_val_fair = float("inf")
    best_val_inloss_at_best = float("inf")
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
            pred_freq_raw, pred_period_raw = model(x)
            pred_freq, pred_period = _apply_prior_if_needed(
                pred_freq_raw, pred_period_raw, x,
                spline_prior, mean_perbin_t, std_global,
            )

            loss, _ = loss_fn(pred_freq, pred_period, y_freq, y_period,
                              period_mean, period_std)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
        train_curve.append(running / n)

        # ----- VAL
        model.eval()
        running_inloss, running_fair, n = 0.0, 0.0, 0
        with torch.no_grad():
            for x, y_freq, y_period in val_loader:
                x = x.to(DEVICE); y_freq = y_freq.to(DEVICE); y_period = y_period.to(DEVICE)
                pred_freq_raw, pred_period_raw = model(x)
                pred_freq, pred_period = _apply_prior_if_needed(
                    pred_freq_raw, pred_period_raw, x,
                    spline_prior, mean_perbin_t, std_global,
                )
                loss_inloss, _ = loss_fn(pred_freq, pred_period, y_freq, y_period,
                                         period_mean, period_std)
                val_fair, _, _ = compute_fair_val_metric(
                    pred_freq, pred_period, y_freq, y_period,
                    mean_perbin_t, std_global,
                )
                running_inloss += loss_inloss.item() * x.size(0)
                running_fair   += val_fair           * x.size(0)
                n += x.size(0)
        val_inloss = running_inloss / n
        val_fair   = running_fair   / n
        val_curve_inloss.append(val_inloss)
        val_curve_fair.append(val_fair)
        scheduler.step(val_fair)

        if val_fair < best_val_fair - 1e-7:
            best_val_fair = val_fair
            best_val_inloss_at_best = val_inloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1

        if (epoch + 1) % 100 == 0:
            print(f"  {log_prefix}ep {epoch+1:4d} | tr {train_curve[-1]:.5f} | "
                  f"val_inloss {val_inloss:.5f} | val_fair {val_fair:.5f} | "
                  f"best_fair {best_val_fair:.5f}@{best_epoch+1} | "
                  f"lr {optimiser.param_groups[0]['lr']:.2e}")

        if bad_epochs >= patience:
            print(f"  {log_prefix}early stop @ ep {epoch+1}, "
                  f"best val_fair {best_val_fair:.5f} @ {best_epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    info = dict(
        best_val_fair=best_val_fair,
        best_val_inloss=best_val_inloss_at_best,
        best_epoch=best_epoch + 1,
        epochs=len(train_curve),
    )
    return info, train_curve, val_curve_inloss, val_curve_fair


# ============================================================================
# 5. EVALUATION
# ============================================================================
def _forward_full(
    model: nn.Module,
    inputs: np.ndarray,
    idx: np.ndarray,
    mean_perbin: np.ndarray,
    std_global: float,
    spline_prior: SplinePrior | None,
) -> tuple[np.ndarray, np.ndarray]:
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
    pred_freq_denorm, pred_period_s = _forward_full(
        model, inputs, test_idx, mean_perbin, std_global, spline_prior
    )
    gt_freq     = raw_fft[test_idx]
    gt_period_s = period[test_idx, 0]

    coef_mse = float(np.mean((pred_freq_denorm - gt_freq) ** 2))

    pred_cx = pred_freq_denorm[..., 0] + 1j * pred_freq_denorm[..., 1]
    gt_cx   = gt_freq[..., 0]          + 1j * gt_freq[..., 1]
    pred_t  = np.fft.irfft(pred_cx, n=32, axis=1)
    gt_t    = np.fft.irfft(gt_cx,   n=32, axis=1)
    time_mse_per_joint = ((pred_t - gt_t) ** 2).mean(axis=(0, 1))
    time_mse = float(time_mse_per_joint.mean())

    period_mae  = float(np.mean(np.abs(pred_period_s - gt_period_s)))
    period_mape = float(np.mean(np.abs(pred_period_s - gt_period_s)
                                / np.maximum(np.abs(gt_period_s), 1e-6)))
    period_mse_sec = float(np.mean((pred_period_s - gt_period_s) ** 2))
    n_period_neg = int(np.sum(pred_period_s <= 0))

    pred_mag = np.abs(pred_cx); gt_mag = np.abs(gt_cx)
    fft_fid_h3 = float(np.mean(np.abs(pred_mag[:, 1:4] - gt_mag[:, 1:4])))

    fair_test = time_mse + FAIR_VAL_PERIOD_WEIGHT * period_mse_sec

    metrics = dict(
        n_test=int(len(test_idx)),
        coef_mse=coef_mse,
        time_domain_mse=time_mse,
        time_mse_per_joint=time_mse_per_joint.tolist(),
        fft_fidelity_h1_h3=fft_fid_h3,
        period_mae_s=period_mae,
        period_mape=period_mape,
        period_mse_sec=period_mse_sec,
        fair_test=fair_test,
        n_period_nonpositive=n_period_neg,
        pred_period_s=pred_period_s.tolist(),
        gt_period_s=gt_period_s.tolist(),
    )
    return metrics, pred_t, gt_t


# ============================================================================
# 6. PLOTTING
# ============================================================================
def plot_training_curves(histories, out_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for label, tr, va_il, va_fair in histories:
        axes[0, 0].plot(tr,      label=label, alpha=0.8)
        axes[0, 1].plot(va_il,   label=label, alpha=0.8)
        axes[1, 0].plot(va_fair, label=label, alpha=0.8)
        axes[1, 1].plot(va_fair, label=label, alpha=0.8)

    axes[0, 0].set_title("Train loss (in-loss units)")
    axes[0, 1].set_title("Val loss (in-loss units)")
    axes[1, 0].set_title("Val fair metric (rad^2 + 5*s^2)")
    axes[1, 1].set_title("Val fair metric (log y)")

    for ax in axes.flat:
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")
    axes[0, 0].set_yscale("log")
    axes[0, 1].set_yscale("log")
    axes[1, 1].set_yscale("log")

    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"[plot] saved {out_path}")


def plot_time_domain_samples(
    pred_t: np.ndarray, gt_t: np.ndarray, test_idx: np.ndarray,
    inputs: np.ndarray, variant: str, out_dir: str,
    max_samples: int = MAX_TIME_DOMAIN_SAMPLES,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    B, T, J = pred_t.shape
    assert J == len(JOINT_NAMES)
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
    matches = sorted(glob.glob(os.path.join(REVIEW_DIR, "test_metrics_*.json")))
    if not matches:
        print(f"[compare] no test_metrics_*.json found under {REVIEW_DIR}; skip plot")
        return

    found = {}
    for p in matches:
        tag = os.path.splitext(os.path.basename(p))[0].replace("test_metrics_", "")
        with open(p, "r") as f:
            found[tag] = json.load(f)

    tags = sorted(found.keys())
    metric_keys = [
        ("coef_mse",          "FFT coef MSE (z-scored)"),
        ("time_domain_mse",   "Time-domain MSE (rad^2)"),
        ("fft_fidelity_h1_h3","FFT fidelity |H1..H3| MAE"),
        ("period_mae_s",      "Period MAE (s)"),
        ("fair_test",         "Fair metric (time + 5*period^2)"),
    ]

    n_metrics = len(metric_keys)
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(max(11, 0.6 * len(tags) + 8), 4 * n_rows))
    axes = np.atleast_2d(axes)

    cmap = plt.get_cmap("tab10" if len(tags) <= 10 else "tab20")
    colors = [cmap(i % cmap.N) for i in range(len(tags))]

    for k, (key, label) in enumerate(metric_keys):
        ax = axes[k // n_cols, k % n_cols]
        vals = [found[t].get(key, float("nan")) for t in tags]
        bars = ax.bar(range(len(tags)), vals, color=colors,
                      edgecolor="black", linewidth=0.6)
        ax.set_xticks(range(len(tags)))
        ax.set_xticklabels(tags, rotation=30, ha="right", fontsize=8)
        ax.set_title(label, fontsize=11)
        ax.set_ylabel(label)
        ax.grid(True, axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                        f"{v:.4g}", ha="center", va="bottom", fontsize=7)
        finite_vals = [v for v in vals if np.isfinite(v)]
        if finite_vals:
            ax.set_ylim(top=max(finite_vals) * 1.18)

    for k in range(n_metrics, n_rows * n_cols):
        axes[k // n_cols, k % n_cols].axis("off")

    fig.suptitle("Test-set performance across FFT-MLP variants x losses",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[compare] saved {out_path}  ({len(tags)} configs)")


# ============================================================================
# 7. MAIN
# ============================================================================
# Only baseline + residual. The old `phase` / `phase_residual` variants are
# redundant once the phase-aware behaviour is selectable via --loss.
Variant = Literal["residual", "baseline"]
ALL_VARIANTS: tuple[str, ...] = ("residual", "baseline")
ALL_LOSSES:   tuple[str, ...] = _LOSS_KINDS   # 6


def variant_config(variant: str) -> dict:
    """variant flag now ONLY controls whether the cubic-spline prior is added."""
    return {
        "baseline": dict(use_residual=False),
        "residual": dict(use_residual=True),
    }[variant]


def _config_tag(variant: str, loss_kind: str) -> str:
    return f"{variant}__{loss_kind}"


def _resolve_effective_alpha(loss_kind: str) -> float:
    """Return the alpha to use with ConfigurableLoss for the given loss kind."""
    # Coefficient-only losses ignore alpha; set to 1.0 so the time-domain
    # branch is short-circuited inside ConfigurableLoss.
    if loss_kind in ("notebook", "notebook_zperiod", "baseline_v2"):
        return 1.0
    # time_only has no freq-coefficient branch; set alpha = 0.0 so even if
    # something added mse_fft it would be multiplied out.
    if loss_kind == "time_only":
        return 0.0
    # phase / time_heavy_period: use module-level TIME_ALPHA
    return TIME_ALPHA


# ============================================================================
# Shared data pipeline
# ============================================================================
def _build_data_pipeline() -> dict:
    inputs, raw_fft, period = load_raw_arrays()
    print(f"[data] inputs={inputs.shape}, raw_fft={raw_fft.shape}, period={period.shape}")

    splits = make_splits(inputs.shape[0])
    train_idx, val_idx, test_idx = splits["train"], splits["val"], splits["test"]

    mean_perbin, std_global, period_mean, period_std = build_norm_stats(
        raw_fft, period, train_idx
    )

    prior_data = build_spline_prior(inputs, raw_fft, period, train_idx)

    flat = raw_fft.reshape(raw_fft.shape[0], -1)
    freq_norm = ((flat - mean_perbin) / std_global).astype(np.float32)
    period_s  = period.astype(np.float32)
    X  = torch.tensor(inputs,    dtype=torch.float32)
    Yf = torch.tensor(freq_norm, dtype=torch.float32)
    Yp = torch.tensor(period_s,  dtype=torch.float32)

    train_ds = TensorDataset(X[train_idx], Yf[train_idx], Yp[train_idx])
    val_ds   = TensorDataset(X[val_idx],   Yf[val_idx],   Yp[val_idx])

    mean_perbin_t = torch.tensor(mean_perbin, dtype=torch.float32, device=DEVICE)
    return dict(
        inputs=inputs, raw_fft=raw_fft, period=period,
        train_idx=train_idx, val_idx=val_idx, test_idx=test_idx,
        mean_perbin=mean_perbin, std_global=std_global,
        period_mean=period_mean, period_std=period_std,
        prior_data=prior_data,
        train_ds=train_ds, val_ds=val_ds,
        mean_perbin_t=mean_perbin_t,
    )


# ============================================================================
# Single (variant, loss) training + test pipeline  (HP grid inside)
# ============================================================================
def run_single_config(
    variant: str,
    loss_kind: str,
    *,
    hp_grid: list[dict],
    data: dict,
    max_epochs: int,
    patience: int,
    use_aug: bool,
) -> dict:
    """
    Train one (variant, loss) pair across the ENTIRE HP grid, keep the best
    HP (by val_fair), evaluate it on the test split, and persist artefacts.

    Why the HP-grid loop lives INSIDE this function:
    * each (variant, loss) may peak at a different HP -- no reason to
      assume one HP wins universally;
    * model selection across HPs uses val_fair (loss-invariant), so the
      within-config comparison is apples-to-apples;
    * all HP runs for one config share a tag namespace so we write one
      consolidated hp_search_<tag>.csv and one training_curves_<tag>.png
      containing every HP point's curves.
    """
    cfg = variant_config(variant)
    tag = _config_tag(variant, loss_kind)

    final_pth  = os.path.join(RESULTS_ROOT, f"FINAL_BEST_MODEL_V2_{tag}.pth")
    metrics_js = os.path.join(REVIEW_DIR, f"test_metrics_{tag}.json")
    curves_png = os.path.join(REVIEW_DIR, f"training_curves_{tag}.png")
    hp_csv     = os.path.join(REVIEW_DIR, f"hp_search_{tag}.csv")
    time_dir   = os.path.join(REVIEW_DIR, f"time_domain_{tag}")

    effective_alpha = _resolve_effective_alpha(loss_kind)
    spline_prior = SplinePrior(**data["prior_data"]) if cfg["use_residual"] else None

    print("\n" + "=" * 70)
    print(f"[RUN] variant={variant}   loss={loss_kind}   alpha={effective_alpha}   "
          f"residual={cfg['use_residual']}   aug={use_aug}")
    print(f"[HP ] grid size = {len(hp_grid)}   max_epochs={max_epochs}")
    print(f"[tag] {tag}")
    print("=" * 70)

    loss_fn = ConfigurableLoss(
        loss_kind=loss_kind,
        alpha=effective_alpha,
        mean_perbin=data["mean_perbin_t"],
        std_global=data["std_global"],
    ).to(DEVICE)

    # ---- HP GRID LOOP --------------------------------------------------
    hp_rows: list[dict] = []
    histories: list[tuple] = []
    best: dict | None = None      # {"info": ..., "model": ..., "hp": ...}

    for i_hp, hp in enumerate(hp_grid):
        hp_tag = _hp_tag(hp)
        print(f"\n  --- HP [{i_hp+1}/{len(hp_grid)}] {hp_tag}")
        torch.manual_seed(SEED)            # reproducible init per HP
        model = SimpleFCNN(hidden_size=hp["hidden_size"])
        train_loader = DataLoader(
            data["train_ds"], batch_size=hp["batch_size"], shuffle=True
        )
        val_loader = DataLoader(data["val_ds"], batch_size=128, shuffle=False)

        info, tr, va_il, va_fair = train_one(
            model, train_loader, val_loader,
            lr=hp["lr"], max_epochs=max_epochs, patience=patience,
            loss_fn=loss_fn,
            period_mean=data["period_mean"], period_std=data["period_std"],
            use_aug=use_aug,
            mean_perbin_t=data["mean_perbin_t"], std_global=data["std_global"],
            spline_prior=spline_prior,
            log_prefix=f"[{hp_tag}] ",
        )
        info.update(hp); info["tag"] = hp_tag
        hp_rows.append(info)
        histories.append((hp_tag, tr, va_il, va_fair))

        # Keep the HP with the lowest val_fair
        if best is None or info["best_val_fair"] < best["info"]["best_val_fair"]:
            best = dict(info=info, model=model, hp=hp)

    assert best is not None, "HP grid must have at least one point"

    # ---- HP search CSV + combined training-curve plot ------------------
    with open(hp_csv, "w", encoding="utf-8") as f:
        keys = ["tag", "hidden_size", "lr", "batch_size",
                "best_val_fair", "best_val_inloss", "best_epoch", "epochs"]
        f.write(",".join(keys) + "\n")
        for r in hp_rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    print(f"[hp] wrote {hp_csv}")
    plot_training_curves(histories, curves_png)

    # ---- Save best model ----------------------------------------------
    print(f"\n[best HP] {best['info']['tag']}  "
          f"val_fair={best['info']['best_val_fair']:.5f}  "
          f"val_inloss={best['info']['best_val_inloss']:.5f}  "
          f"(ep {best['info']['best_epoch']}/{best['info']['epochs']})")
    torch.save(best["model"].state_dict(), final_pth)
    print(f"          saved {final_pth}")

    # ---- Test eval on BEST HP -----------------------------------------
    print(f"[test] evaluating best HP on {len(data['test_idx'])} samples")
    test_metrics, pred_t, gt_t = evaluate_test(
        best["model"],
        data["inputs"], data["raw_fft"], data["period"], data["test_idx"],
        data["mean_perbin"], data["std_global"], spline_prior,
    )
    test_metrics["variant"]       = variant
    test_metrics["loss_kind"]     = loss_kind
    test_metrics["tag"]           = tag
    test_metrics["best_hp"]       = best["info"]
    test_metrics["alpha"]         = effective_alpha
    test_metrics["use_residual"]  = cfg["use_residual"]
    test_metrics["augmentation"]  = use_aug
    test_metrics["hp_grid_size"]  = len(hp_grid)
    test_metrics["fair_val_period_weight"] = FAIR_VAL_PERIOD_WEIGHT
    with open(metrics_js, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"       saved {metrics_js}")
    scalar_only = {
        k: v for k, v in test_metrics.items()
        if not isinstance(v, (list, dict))
    }
    print(json.dumps(scalar_only, indent=2, default=str))

    plot_time_domain_samples(
        pred_t, gt_t, data["test_idx"], data["inputs"],
        variant=tag, out_dir=time_dir,
    )
    return test_metrics


# ============================================================================
# Sweep-wide summary
# ============================================================================
def write_sweep_summary(summary_rows: list[dict], out_csv: str) -> None:
    keys = [
        "variant", "loss_kind", "tag",
        "coef_mse", "time_domain_mse", "fft_fidelity_h1_h3",
        "period_mae_s", "period_mape", "period_mse_sec",
        "fair_test", "n_test",
    ]
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in summary_rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    print(f"[sweep] summary CSV -> {out_csv}")


def print_sweep_ranking(summary_rows: list[dict]) -> None:
    ranked = sorted(summary_rows, key=lambda r: r.get("fair_test", float("inf")))
    print("\n" + "=" * 78)
    print(f"{'rank':<5}{'tag':<40}{'fair_test':>12}{'time_mse':>12}{'period_mae':>10}")
    print("-" * 78)
    for i, r in enumerate(ranked):
        print(f"{i+1:<5}{r.get('tag',''):<40}"
              f"{r.get('fair_test', float('nan')):>12.5f}"
              f"{r.get('time_domain_mse', float('nan')):>12.5f}"
              f"{r.get('period_mae_s', float('nan')):>10.4f}")
    print("=" * 78)


# ============================================================================
# MAIN
# ============================================================================
def _resolve_hp_grid(mode: str, quick: bool) -> list[dict]:
    """
    ``full``   : 8-point cartesian product (HP_GRID_FULL)
    ``single`` : 1-point (hs=256, lr=3e-4, bs=64)
    ``--quick`` forces mode='single' so smoke tests stay fast.
    """
    if quick:
        if mode == "full":
            print("[hp] --quick forces HP grid to 'single' (hs=256, lr=3e-4, bs=64)")
        return list(HP_GRID_SINGLE)
    if mode == "single":
        return list(HP_GRID_SINGLE)
    return list(HP_GRID_FULL)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--variant",
        choices=list(ALL_VARIANTS),
        default="baseline",
        help="Single-config mode: which variant (ignored if --all).",
    )
    ap.add_argument(
        "--loss",
        choices=list(ALL_LOSSES),
        default="phase",
        help="Single-config mode: which loss (ignored if --all).",
    )
    ap.add_argument(
        "--all", action="store_true",
        help="Sweep all %d (variant x loss) configs; each runs the HP grid." %
             (len(ALL_VARIANTS) * len(ALL_LOSSES)),
    )
    ap.add_argument("--quick", action="store_true",
                    help="Smoke run: 200 epochs + single HP point.")
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--compare-only", action="store_true",
                    help="Skip training; rebuild variants_test_comparison.png only.")
    ap.add_argument(
        "--hp-grid", choices=["full", "single"], default="full",
        help="Per-config HP grid size. 'full' = 8 points; 'single' = 1 "
             "(hs=256, lr=3e-4, bs=64). --quick overrides to 'single'.",
    )
    args = ap.parse_args()

    os.makedirs(REVIEW_DIR, exist_ok=True)

    if args.compare_only:
        plot_variant_comparison()
        return 0

    torch.manual_seed(SEED); np.random.seed(SEED)

    hp_grid = _resolve_hp_grid(args.hp_grid, args.quick)
    max_epochs = 200 if args.quick else MAX_EPOCHS
    use_aug = not args.no_aug

    data = _build_data_pipeline()

    # ---- SINGLE-CONFIG MODE -------------------------------------------
    if not args.all:
        _ = run_single_config(
            args.variant, args.loss,
            hp_grid=hp_grid, data=data,
            max_epochs=max_epochs, patience=PATIENCE,
            use_aug=use_aug,
        )
        plot_variant_comparison()
        print(f"\n[done] single config '{_config_tag(args.variant, args.loss)}' complete.")
        return 0

    # ---- FULL SWEEP: 2 variants x 6 losses = 12 configs, each with HP grid
    pairs = [(v, l) for v in ALL_VARIANTS for l in ALL_LOSSES]
    n_total = len(pairs)
    total_runs = n_total * len(hp_grid)
    print("\n" + "#" * 70)
    print(f"# SWEEP MODE: {n_total} configs x {len(hp_grid)} HPs = {total_runs} training runs")
    print(f"# variants={ALL_VARIANTS}")
    print(f"# losses  ={ALL_LOSSES}")
    print(f"# hp_grid ={[_hp_tag(h) for h in hp_grid]}")
    print(f"# max_epochs={max_epochs}, patience={PATIENCE}, aug={use_aug}")
    print("#" * 70)

    summary_rows: list[dict] = []
    sweep_failures: list[tuple[str, str, str]] = []

    for i, (variant, loss_kind) in enumerate(pairs):
        tag = _config_tag(variant, loss_kind)
        print(f"\n\n{'*' * 70}")
        print(f"* [{i+1:2d}/{n_total}]  {tag}")
        print(f"{'*' * 70}")
        try:
            metrics = run_single_config(
                variant, loss_kind,
                hp_grid=hp_grid, data=data,
                max_epochs=max_epochs, patience=PATIENCE,
                use_aug=use_aug,
            )
            summary_rows.append({
                k: v for k, v in metrics.items()
                if not isinstance(v, (list, dict))
            })
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] config {tag} FAILED: {type(e).__name__}: {e}")
            sweep_failures.append((variant, loss_kind, f"{type(e).__name__}: {e}"))

    # ---- sweep-level outputs
    sweep_csv  = os.path.join(REVIEW_DIR, "sweep_summary.csv")
    sweep_json = os.path.join(REVIEW_DIR, "sweep_summary.json")
    write_sweep_summary(summary_rows, sweep_csv)
    with open(sweep_json, "w") as f:
        json.dump(
            dict(
                hp_grid=hp_grid,
                hp_grid_mode=args.hp_grid,
                max_epochs=max_epochs, augmentation=use_aug,
                n_configs=n_total, n_succeeded=len(summary_rows),
                results=summary_rows,
                failures=[
                    dict(variant=v, loss_kind=l, error=e)
                    for (v, l, e) in sweep_failures
                ],
            ),
            f, indent=2, default=str,
        )
    print(f"[sweep] summary JSON -> {sweep_json}")

    plot_variant_comparison()

    if summary_rows:
        print_sweep_ranking(summary_rows)
    if sweep_failures:
        print(f"\n[warn] {len(sweep_failures)} configs failed:")
        for v, l, e in sweep_failures:
            print(f"  - {v} x {l}: {e}")

    print(f"\n[done] sweep complete: "
          f"{len(summary_rows)}/{n_total} succeeded.")
    return 0 if not sweep_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())