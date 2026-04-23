"""
gait_generator_v2_residual.py
==============================

Drop-in replacement for the 3D study's findgait() using the v2-residual
FFT-MLP from the 2D biped project.

Why this file exists
--------------------
The v2 model has two structural differences from the legacy v1 model:

1. **Two heads instead of one.**
   v1: Linear(H, 137) - 136 freqs + 1 period, single tensor.
   v2: Linear(H, 136) + (Linear(H, 1) -> softplus) - freq + period-in-seconds.
   The softplus guarantees T > 0 without any z-score dance, so we drop the
   "multiply by episode_len then clamp" step (the 3D code's clamping to >=12
   was a safety net for negative raw outputs; no longer needed).

2. **Cubic-spline residual prior.**
   The MLP only learns residuals on top of a fixed cubic-spline prior over
   commanded speed.  The prior is stored in spline_prior_v2.npz with three
   arrays:
       knot_speeds     (K,)         -- m/s, strictly increasing
       freq_knots      (K, 136)     -- de-normalised FFT coefs at each knot
       period_knots    (K,)         -- period in seconds at each knot
   At inference we evaluate a cubic spline in speed to get
       (prior_freq_denorm, prior_period)
   and add them to the MLP's outputs BEFORE the IRFFT.

Paths required on the 3D machine
--------------------------------
Copy these four files from the 2D repo to the 3D project:
    FINAL_BEST_MODEL_V2_residual.pth    (from kfold_results/)
    mean_train.npy                       (from gait reference phase 2/)
    std_train.npy                        (from gait reference phase 2/)
    spline_prior_v2.npz                  (from gait reference phase 2/)

Usage pattern (inside the 3D env class, matching your existing code)
--------------------------------------------------------------------
    from gait_generator_v2_residual import GaitGeneratorV2Residual

    class Locomotion(...):
        def __init__(self, ...):
            ...
            self.gait = GaitGeneratorV2Residual(
                weights_path="path/to/FINAL_BEST_MODEL_V2_residual.pth",
                mean_path="path/to/mean_train.npy",
                std_path="path/to/std_train.npy",
                spline_prior_path="path/to/spline_prior_v2.npz",
                device=self.sim.device,
                dt=self.cfg.dt,
                episode_len=self.cfg.episode_len,
            )

        def findgait(self, input_vec):
            return self.gait.findgait(input_vec)

Signature match: returns (pred_torch: [N, T, 4], periods: [N] int32),
just like your old findgait().
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Model architecture (MUST match fft_datacreate_review_v2.SimpleFCNN exactly)
# ============================================================================
class _SimpleFCNNv2(nn.Module):
    """Backbone + freq_head (136, linear) + period_head (1, softplus)."""

    def __init__(self, input_size: int = 3, hidden_size: int = 512,
                 freq_dim: int = 136) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(0.1),
        )
        self.freq_head   = nn.Linear(hidden_size, freq_dim)
        self.period_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        freq = self.freq_head(h)                    # [N, 136]  z-scored
        period = F.softplus(self.period_head(h))    # [N, 1]    seconds, > 0
        return freq, period


# ============================================================================
# Vectorised cubic-spline prior in torch (keeps everything on GPU)
# ============================================================================
class _TorchCubicSpline:
    """Natural cubic spline for a batch of 1D queries, fully on-device.

    Implements the standard tri-diagonal solve once at __init__; query is
    O(log K) via bucketize + polynomial evaluation.
    """

    def __init__(self, knots_x: torch.Tensor, knots_y: torch.Tensor) -> None:
        # knots_x: (K,) strictly increasing; knots_y: (K, D) or (K,)
        self.device = knots_x.device
        self.dtype  = knots_x.dtype
        if knots_y.dim() == 1:
            knots_y = knots_y.unsqueeze(1)          # (K, 1)
        self.x = knots_x                            # (K,)
        self.y = knots_y                            # (K, D)
        K = knots_x.shape[0]
        h = knots_x[1:] - knots_x[:-1]              # (K-1,)

        # Natural cubic spline second-derivative solve (Thomas algorithm)
        # Build tridiagonal system for interior points.
        a = torch.zeros(K, device=self.device, dtype=self.dtype)
        b = torch.ones(K,  device=self.device, dtype=self.dtype) * 2.0
        c = torch.zeros(K, device=self.device, dtype=self.dtype)
        d = torch.zeros((K, knots_y.shape[1]), device=self.device, dtype=self.dtype)

        # Interior rows
        a[1:-1] = h[:-1]
        b[1:-1] = 2.0 * (h[:-1] + h[1:])
        c[1:-1] = h[1:]
        d[1:-1] = 6.0 * ((knots_y[2:] - knots_y[1:-1]) / h[1:].unsqueeze(1)
                       - (knots_y[1:-1] - knots_y[:-2]) / h[:-1].unsqueeze(1))
        # Natural boundary: M[0] = M[K-1] = 0
        b[0] = 1.0;  c[0] = 0.0
        a[-1] = 0.0; b[-1] = 1.0

        # Thomas algorithm
        cp = torch.zeros(K, device=self.device, dtype=self.dtype)
        dp = torch.zeros_like(d)
        cp[0] = c[0] / b[0]
        dp[0] = d[0] / b[0]
        for i in range(1, K):
            m = b[i] - a[i] * cp[i - 1]
            cp[i] = c[i] / m if i < K - 1 else 0.0
            dp[i] = (d[i] - a[i] * dp[i - 1]) / m
        M = torch.zeros_like(d)
        M[-1] = dp[-1]
        for i in range(K - 2, -1, -1):
            M[i] = dp[i] - cp[i] * M[i + 1]

        self.h = h                                  # (K-1,)
        self.M = M                                  # (K, D)  second derivs

    def __call__(self, xq: torch.Tensor) -> torch.Tensor:
        """Evaluate spline at xq (N,).  Returns (N, D).  Clamps out-of-range."""
        x = self.x;  y = self.y;  M = self.M;  h = self.h
        xq = torch.clamp(xq, min=float(x[0]), max=float(x[-1]))
        # idx such that x[idx] <= xq <= x[idx+1]
        idx = torch.bucketize(xq, x) - 1
        idx = torch.clamp(idx, 0, x.shape[0] - 2)
        x0 = x[idx];         x1 = x[idx + 1]
        h_i = h[idx]                                # (N,)
        y0 = y[idx];         y1 = y[idx + 1]        # (N, D)
        M0 = M[idx];         M1 = M[idx + 1]        # (N, D)
        a = (x1 - xq) / h_i                         # (N,)
        b = (xq - x0) / h_i
        a = a.unsqueeze(1); b = b.unsqueeze(1); h_i = h_i.unsqueeze(1)
        out = (a * y0 + b * y1
               + ((a.pow(3) - a) * M0 + (b.pow(3) - b) * M1) * h_i.pow(2) / 6.0)
        return out                                  # (N, D)


# ============================================================================
# The generator class (drop-in for your 3D findgait)
# ============================================================================
class GaitGeneratorV2Residual:
    """Drop-in v2-residual replacement for the legacy findgait() pipeline.

    Public API
    ----------
    findgait(input_vec: Tensor[N, 3] or [3]) -> (pred_torch[N, T, 4], periods[N])
    """

    def __init__(
        self,
        weights_path: str,
        mean_path: str,
        std_path: str,
        spline_prior_path: str,
        device: torch.device | str = "cuda",
        dt: float = 1.0 / 200,                     # 3D Isaac step rate
        episode_len: int = 1000,                   # for periods scaling
        hidden_size: int = 512,
        min_period_steps: int = 12,
    ) -> None:
        self.device = torch.device(device)
        self.dt = float(dt)
        self.episode_len = int(episode_len)
        self.min_period_steps = int(min_period_steps)

        # --- model
        self.net = _SimpleFCNNv2(input_size=3, hidden_size=hidden_size, freq_dim=136)
        self.net.load_state_dict(torch.load(weights_path, map_location=self.device,
                                            weights_only=True))
        self.net.to(self.device).eval()

        # --- normalisation stats (train-only, from v2 script)
        mean_np = np.load(mean_path).astype(np.float32)                 # (136,)
        std_np  = np.load(std_path).astype(np.float32).reshape(-1)[0]   # scalar
        self.mean = torch.tensor(mean_np, device=self.device, dtype=torch.float32)
        self.std  = float(std_np)

        # --- cubic-spline prior
        d = np.load(spline_prior_path)
        knot_speeds  = torch.tensor(d["knot_speeds"],  device=self.device,
                                    dtype=torch.float32)                # (K,)
        freq_knots   = torch.tensor(d["freq_knots"],   device=self.device,
                                    dtype=torch.float32)                # (K, 136)
        period_knots = torch.tensor(d["period_knots"], device=self.device,
                                    dtype=torch.float32)                # (K,)
        self._freq_spline   = _TorchCubicSpline(knot_speeds, freq_knots)
        self._period_spline = _TorchCubicSpline(knot_speeds, period_knots)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def findgait(self, input_vec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        input_vec : Tensor of shape [3] or [N, 3]
            (speed / 2.4, r_leg_length, l_leg_length)
            -- same encoding as fft_datacreate_review_v2.py uses.

        Returns
        -------
        pred_torch : Tensor [N, T, 4]   time-domain trajectories for the
                                        four mocap joints (rhip, rknee,
                                        lhip, lknee)
        periods    : Tensor [N] int32   period length in integer env steps
                                        (>=min_period_steps)
        """
        if input_vec.ndim == 1:
            input_vec = input_vec.unsqueeze(0)
        input_vec = input_vec.to(self.device, dtype=torch.float32)

        # 1. MLP forward -- residual part
        freq_resid_norm, period_resid = self.net(input_vec)    # [N,136], [N,1]

        # 2. Cubic-spline prior evaluated at speed in m/s (= input[:, 0] * 2.4)
        speed_ms = input_vec[:, 0] * 2.4                       # [N]
        prior_freq_denorm = self._freq_spline(speed_ms)        # [N, 136]
        prior_period      = self._period_spline(speed_ms)      # [N, 1]  (D=1)

        # 3. Combine:
        #    * freq: add residual (z-scored) to prior (de-normalised then re-normalised).
        #      Equivalently: prior + (resid * std + 0) in de-norm space.
        prior_freq_norm = (prior_freq_denorm - self.mean) / self.std
        pred_freq_norm  = freq_resid_norm + prior_freq_norm     # [N, 136]
        pred_period_s   = period_resid + prior_period            # [N, 1] seconds, >0

        # 4. De-normalise freq -> complex -> time domain
        pred_freq_denorm = pred_freq_norm * self.std + self.mean         # [N, 136]
        pred_time = self._irfft_and_resample(pred_freq_denorm)           # [N, T, 4]

        # 5. Periods in env steps. NOTE: v2 period is ALREADY in seconds
        #    (softplus output), NOT a dimensionless ratio as in the v1 head.
        #    So convert seconds -> integer steps using 1/dt, not episode_len.
        #    We keep the min_period_steps clamp as a belt-and-braces guard.
        periods_steps = (pred_period_s.squeeze(-1) / self.dt).to(torch.int32)
        periods_steps = torch.clamp(periods_steps, min=self.min_period_steps)

        return pred_time, periods_steps

    # ------------------------------------------------------------------
    # Internals: recover shape + IRFFT + resample to env control rate
    # ------------------------------------------------------------------
    def _irfft_and_resample(self, flat_freq_denorm: torch.Tensor) -> torch.Tensor:
        """[N, 136] de-normalised -> [N, T, 4] time-domain trajectory.

        Mirrors the legacy 3D pipeline's shape conventions exactly:
        * 136 = 17 bins x 4 joints x 2 (re,im)
        * output 4 joints ordered (rhip, rknee, lhip, lknee) -- ankles NOT
          produced here, matching the old findgait()
        """
        N = flat_freq_denorm.shape[0]

        # (N, 17, 4, 2)
        recovered = flat_freq_denorm.view(N, 17, 4, 2)

        # Permute so freq axis is last for irfft: (N, 4, 2, 17)
        structured = recovered.permute(0, 2, 3, 1)

        # Complex combine: (N, 4, 17) complex
        complex_pred = torch.complex(structured[:, :, 0, :],
                                     structured[:, :, 1, :])

        # IRFFT over last dim -> (N, 4, 32)
        pred_time = torch.fft.irfft(complex_pred, n=32, dim=2)

        # Resample to env rate.  The mocap rate was 10 Hz (see 2D training
        # script: fft window = 32 samples over 3.2 s -> org_rate=10).
        org_rate = 10
        if self.dt < 0.1:
            num_samples = int(pred_time.shape[2] * (1.0 / self.dt) / org_rate)
            # F.interpolate wants [B, C, L]; (N, 4, 32) is already that.
            pred_time = F.interpolate(pred_time, size=num_samples,
                                      mode="linear", align_corners=True)

        # -> [N, T, 4] for downstream consumers
        return pred_time.permute(0, 2, 1)


# ============================================================================
# Quick self-test
# ============================================================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights",  default="FINAL_BEST_MODEL_V2_residual.pth")
    ap.add_argument("--mean",     default="mean_train.npy")
    ap.add_argument("--std",      default="std_train.npy")
    ap.add_argument("--prior",    default="spline_prior_v2.npz")
    ap.add_argument("--dt",       type=float, default=1.0 / 200)
    ap.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    gen = GaitGeneratorV2Residual(
        weights_path=args.weights,
        mean_path=args.mean,
        std_path=args.std,
        spline_prior_path=args.prior,
        device=args.device,
        dt=args.dt,
    )

    # Batched test: 3 different (speed, leg_r, leg_l) queries
    speeds_ms = torch.tensor([0.4, 1.0, 2.0])
    legs_r    = torch.tensor([0.85, 0.94, 0.80])
    legs_l    = torch.tensor([0.85, 0.94, 0.80])
    inp = torch.stack([speeds_ms / 2.4, legs_r, legs_l], dim=1)   # [3, 3]

    pred, periods = gen.findgait(inp)
    print(f"inputs      : {inp.shape}  (speed/2.4, r_leg, l_leg)")
    print(f"pred_torch  : {tuple(pred.shape)}  [N, T, 4]")
    print(f"periods     : {periods.tolist()}  env-steps   (dt={args.dt})")
    print(f"period (s)  : {[p * args.dt for p in periods.tolist()]}")
    print(f"pred range  : min={pred.min().item():.4f}  max={pred.max().item():.4f}")
    print(f"finite?     : {torch.isfinite(pred).all().item()}")
    assert torch.isfinite(pred).all(), "NaN/Inf in output!"
    assert (periods > 0).all(),        "non-positive period!"
    print("OK")