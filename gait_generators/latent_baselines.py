"""PCA-score MLP and residual-GRU gait generators (Reviewer 2, point 4).

Weights come from ``train_latent_baselines.py``. Both return a (T, 6)
trajectory at the control rate with zero ankle columns, exactly like
:class:`CubicSplineGenerator`, so they drop into the fair protocol of
``analyse_gait_generators.py`` unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from scipy.signal import resample

from .base import BaseGaitGenerator

_REPO = Path(__file__).resolve().parents[1]
_W = _REPO / "kfold_results" / "latent_baselines"


def _to_six(w32: np.ndarray, dt: float, reps: int) -> np.ndarray:
    traj = resample(w32, int(32 * (1 / dt) / 10), axis=0) if dt < 0.1 else w32
    six = np.zeros((traj.shape[0], 6)); six[:, [0, 1, 3, 4]] = traj
    return np.tile(six, (reps, 1))


class PCAMLPGenerator(BaseGaitGenerator):
    name = "pca_mlp"

    def __init__(self, dt: float = 1e-3, tile_repeats: int = 50) -> None:
        super().__init__(dt=dt, tile_repeats=tile_repeats)
        from gait_generator_net import SimpleFCNN
        self.net = SimpleFCNN(3, 128, 512); self.net.load_state_dict(torch.load(_W / "pca_mlp.pth")); self.net.eval()
        b = np.load(_W / "pca_mlp_basis.npz")
        self.comp, self.mean, self.mu, self.sd = b["components"], b["mean"], b["mu"], b["sd"]

    def predict(self, speed: float, leg_lengths: Sequence[float]) -> np.ndarray:
        x = torch.tensor([speed / 2.4, leg_lengths[0], leg_lengths[1]], dtype=torch.float32)
        with torch.no_grad():
            s = self.net(x).numpy() * self.sd + self.mu
        return _to_six((s @ self.comp + self.mean).reshape(32, 4), self.dt, self.tile_repeats)


class GRUSeqGenerator(BaseGaitGenerator):
    name = "gru_seq"

    def __init__(self, dt: float = 1e-3, tile_repeats: int = 50) -> None:
        super().__init__(dt=dt, tile_repeats=tile_repeats)
        from train_latent_baselines import GRUSeq
        self.net = GRUSeq(); self.net.load_state_dict(torch.load(_W / "gru_seq.pth")); self.net.eval()

    def predict(self, speed: float, leg_lengths: Sequence[float]) -> np.ndarray:
        x = torch.tensor([[speed / 2.4, leg_lengths[0], leg_lengths[1]]], dtype=torch.float32)
        with torch.no_grad():
            w = self.net(x)[0].numpy()
        return _to_six(w, self.dt, self.tile_repeats)


__all__ = ["PCAMLPGenerator", "GRUSeqGenerator"]
