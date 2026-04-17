"""Per-speed cubic-spline interpolation baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import resample

from .base import BaseGaitGenerator
from .raw_mocap import RawMocapGenerator


_REPO = Path(__file__).resolve().parents[1]


class CubicSplineGenerator(BaseGaitGenerator):
    """Bucket the mocap corpus by speed, interpolate with a cubic spline.

    For a query speed we take the :attr:`k_neighbours` closest-speed mocap
    samples, average their time-domain trajectories, and fit a natural cubic
    spline in phase so the returned trajectory is smooth and periodic. Period
    and amplitude therefore vary continuously with speed, unlike the
    nearest-neighbour :class:`RawMocapGenerator`.
    """

    name = "cubic_spline"

    def __init__(
        self,
        dataset_dir: str | Path = _REPO / "gait reference phase 2",
        k_neighbours: int = 8,
        dt: float = 1e-3,
        tile_repeats: int = 50,
    ) -> None:
        super().__init__(dt=dt, tile_repeats=tile_repeats)
        self._mocap = RawMocapGenerator(
            dataset_dir=dataset_dir, dt=dt, tile_repeats=1
        )
        self.k_neighbours = int(k_neighbours)

    def predict(self, speed: float, leg_lengths: Sequence[float]) -> np.ndarray:
        neighbours = np.argsort(np.abs(self._mocap.speeds - float(speed)))[
            : self.k_neighbours
        ]
        periods = np.array([float(self._mocap.period[i, 0]) for i in neighbours])
        mean_period = float(periods.mean())

        traj_list = []
        for idx in neighbours:
            traj = self._mocap._reconstruct_from_freqs(self._mocap.targets[idx, :136])
            if traj.shape[0] != 32:
                traj = resample(traj, 32, axis=0)
            traj_list.append(traj)
        mean_traj4 = np.mean(np.stack(traj_list, axis=0), axis=0)  # (32, 4)

        phase = np.linspace(0.0, 1.0, mean_traj4.shape[0] + 1)
        closed = np.vstack([mean_traj4, mean_traj4[:1]])
        cs = CubicSpline(phase, closed, bc_type="periodic", axis=0)

        n_samples = max(2, int(round(mean_period / self.dt)))
        phase_grid = np.linspace(0.0, 1.0, n_samples, endpoint=False)
        sampled4 = cs(phase_grid)

        traj6 = np.zeros((sampled4.shape[0], 6), dtype=np.float64)
        traj6[:, 0] = sampled4[:, 0]
        traj6[:, 1] = sampled4[:, 1]
        traj6[:, 2] = 0.0
        traj6[:, 3] = sampled4[:, 2]
        traj6[:, 4] = sampled4[:, 3]
        traj6[:, 5] = 0.0

        return np.tile(traj6, (self.tile_repeats, 1))


__all__ = ["CubicSplineGenerator"]
