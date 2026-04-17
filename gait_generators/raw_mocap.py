"""Raw mocap-playback generator: pick the nearest-speed sample and loop it."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.signal import resample

from .base import BaseGaitGenerator


_REPO = Path(__file__).resolve().parents[1]


class RawMocapGenerator(BaseGaitGenerator):
    """Nearest-neighbour motion-capture baseline.

    Loads the paper's phase-2 dataset (``gait reference phase 2/`` — 1121
    samples of 6 joints x 32 Fourier coefficients together with the measured
    gait period) and, for a query ``speed``, returns the time-domain
    trajectory reconstructed from the nearest-speed sample's FFT.

    This is the simplest possible "no learning at all" baseline: if PPO can
    match its locomotion quality, the FFT-MLP generator contributes little.
    """

    name = "raw_mocap"

    def __init__(
        self,
        dataset_dir: str | Path = _REPO / "gait reference phase 2",
        dt: float = 1e-3,
        tile_repeats: int = 50,
    ) -> None:
        super().__init__(dt=dt, tile_repeats=tile_repeats)
        dataset_dir = Path(dataset_dir)
        self.input_vector = np.load(dataset_dir / "input_vector.npy")  # (N, 3)
        self.targets = np.load(dataset_dir / "targets.npy")            # (N, 137)
        self.period = np.load(dataset_dir / "period.npy")              # (N, 1)
        self.mean = np.load(dataset_dir / "mean.npy")
        self.std = np.load(dataset_dir / "std.npy")
        self.speeds = self.input_vector[:, 0] * 3.0

    def _reconstruct_from_freqs(self, flat_freqs: np.ndarray) -> np.ndarray:
        denorm = flat_freqs * self.std + self.mean  # shape (136,)
        recovered = denorm.reshape(17, 4, 2).transpose(1, 2, 0)  # (4, 2, 17)
        complex_pred = recovered[:, 0, :] + 1j * recovered[:, 1, :]
        time = np.fft.irfft(complex_pred, n=32, axis=1).T  # (32, 4)
        return time

    def predict(self, speed: float, leg_lengths: Sequence[float]) -> np.ndarray:
        idx = int(np.argmin(np.abs(self.speeds - float(speed))))
        flat = self.targets[idx, :136]
        period = float(self.period[idx, 0])
        traj4 = self._reconstruct_from_freqs(flat)  # (32, 4) rhip, rknee, lhip, lknee

        traj6 = np.zeros((traj4.shape[0], 6), dtype=np.float64)
        traj6[:, 0] = traj4[:, 0]  # rhip
        traj6[:, 1] = traj4[:, 1]  # rknee
        traj6[:, 2] = 0.0          # rankle (unavailable in mocap)
        traj6[:, 3] = traj4[:, 2]  # lhip
        traj6[:, 4] = traj4[:, 3]  # lknee
        traj6[:, 5] = 0.0          # lankle

        n_samples = int(round(period / self.dt))
        if n_samples > 1:
            traj6 = resample(traj6, n_samples, axis=0)
        traj6 = np.tile(traj6, (self.tile_repeats, 1))
        return traj6


__all__ = ["RawMocapGenerator"]
