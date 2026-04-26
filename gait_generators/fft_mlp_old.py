"""
gait_generators/fft_mlp_review.py
==================================

Drop-in generator that loads the *review-trained* 137-dim, 4-joint
``SimpleFCNN`` produced by ``fft_datacreate_review.py``.

Why a separate class?
---------------------
The legacy ``FFTMLPGenerator`` loads a 204-dim ``OldSimpleFCNN`` plus
``newnormalization_constants.npy`` from a different (older) training round.
Pointing it at the 137-dim review model would silently mis-load weights.
Keeping a parallel class with **its own paths** preserves the original B1
results while letting the review eval use the new artefacts.

Hard-coded paths (must match ``fft_datacreate_review.py``):
    weights        = ``kfold_results/FINAL_BEST_MODEL_REVIEW.pth``
    mean / std     = ``gait reference phase 2/{mean_train,std_train}.npy``
                     (mean is per-bin (136,); std is a single scalar)
    period stats   = ``gait reference phase 2/period_stats.npy``
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from scipy.signal import resample

from .base import BaseGaitGenerator


_REPO = Path(__file__).resolve().parents[1]


class FFTMLPOldGenerator(BaseGaitGenerator):
    """The review-trained 137-dim, 4-joint FFT-MLP."""

    name = "fft_mlp_old"

    # ankles are not in mocap -> stay zero in both legacy and review paths
    _ANKLE_INDICES = (2, 5)

    def __init__(
        self,
        weights_path: str | Path = _REPO / "gait reference phase 2/old_original_gaitgen" / "FINAL_BEST_MODEL.pth",
        mean_path: str | Path = _REPO / "gait reference phase 2/old_original_gaitgen"  / "mean.npy",
        std_path:  str | Path = _REPO / "gait reference phase 2/old_original_gaitgen"  / "std.npy",
        period_stats_path: str | Path = _REPO / "gait reference phase 2" / "period_stats.npy",
        dt: float = 1e-3,
        tile_repeats: int = 50,
        hidden_size: int = 512,
    ) -> None:
        super().__init__(dt=dt, tile_repeats=tile_repeats)
        # local import - avoid circular dependency with the training script
        from gait_generator_net import SimpleFCNN as _LegacySimpleFCNN

        # The review script saves a 137-output SimpleFCNN; the in-tree
        # ``gait_generator_net.SimpleFCNN`` defaults to 204 outputs, so we
        # construct it explicitly with the right output_size.
        self.net = _LegacySimpleFCNN(
            input_size=3, output_size=137, hidden_size=hidden_size
        )
        self.net.load_state_dict(torch.load(weights_path, weights_only=True))
        self.net.eval()

        self.mean_perbin = np.load(mean_path).astype(np.float32)        # (136,)
        std_arr = np.load(std_path).astype(np.float32)                  # (1,)
        self.std_global = float(std_arr.reshape(-1)[0])                 # scalar
        ps = np.load(period_stats_path).astype(np.float32)              # (2,)
        self.period_mean = float(ps[0])
        self.period_std  = float(ps[1])

    # ------------------------------------------------------------------
    def _input_vec(self, speed: float, leg_lengths: Sequence[float]) -> torch.Tensor:
        # Same encoding used during training (notebook): speed/2.4, leg/1.0
        v = np.empty(3, dtype=np.float32)
        v[0] = float(speed) / 2.4
        v[1] = float(leg_lengths[0]) / 1.0
        v[2] = float(leg_lengths[1]) / 1.0
        return torch.tensor(v, dtype=torch.float32).unsqueeze(0)

    def _denormalize(self, pred: np.ndarray) -> tuple[np.ndarray, float]:
        freq_norm   = pred[:136]
        period_norm = float(pred[136])
        freq = freq_norm * self.std_global + self.mean_perbin           # (136,)
        period = period_norm * self.period_std + self.period_mean
        return freq.reshape(17, 4, 2), period

    def _ifft(self, freq_174x2: np.ndarray, period: float) -> np.ndarray:
        complex_pred = freq_174x2[..., 0] + 1j * freq_174x2[..., 1]     # (17, 4)
        time4 = np.fft.irfft(complex_pred, n=32, axis=0)                # (32, 4)

        # 4 -> 6 joints (ankles stay zero, no mocap data exists for them)
        time6 = np.zeros((time4.shape[0], 6), dtype=np.float64)
        time6[:, 0] = time4[:, 0]   # rhip
        time6[:, 1] = time4[:, 1]   # rknee
        time6[:, 3] = time4[:, 2]   # lhip
        time6[:, 4] = time4[:, 3]   # lknee
        # rankle (idx 2) and lankle (idx 5) remain zero

        # Resample to environment control rate.
        # The model emits a 32-sample cycle covering one period (seconds).
        n_samples = max(2, int(round(period / self.dt)))
        if n_samples != time6.shape[0]:
            time6 = resample(time6, n_samples, axis=0)
        return np.tile(time6, (self.tile_repeats, 1))

    # ------------------------------------------------------------------
    def predict(self, speed: float, leg_lengths: Sequence[float]) -> np.ndarray:
        x = self._input_vec(speed, leg_lengths)
        with torch.no_grad():
            pred = self.net(x).cpu().numpy()[0]                         # (137,)
        freq, period = self._denormalize(pred)
        return self._ifft(freq, period)


__all__ = ["FFTMLPGenerator"]