"""RNN/GRU sequence generator scaffold.

A trained weight file is not committed to the repo, so for now the class
falls back to the FFT-MLP baseline with a warning. The scaffold exists so
B1 has a fourth gait-generator slot to drop a trained GRU into, once the
curated sequence dataset in ``gait time series data/window_data.npy`` is
wired up.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Sequence

import numpy as np

from .base import BaseGaitGenerator
from .fft_mlp import FFTMLPGenerator


_REPO = Path(__file__).resolve().parents[1]


class RNNGenerator(BaseGaitGenerator):
    name = "rnn"

    def __init__(
        self,
        weights_path: str | Path = _REPO / "rnn_gait_generator.pth",
        dt: float = 1e-3,
        tile_repeats: int = 50,
    ) -> None:
        super().__init__(dt=dt, tile_repeats=tile_repeats)
        self._fallback = FFTMLPGenerator(dt=dt, tile_repeats=tile_repeats)
        self._weights_path = Path(weights_path)
        if not self._weights_path.exists():
            warnings.warn(
                f"RNN generator weights not found at {self._weights_path}; "
                "falling back to FFT-MLP baseline. Train the RNN to unlock "
                "the sequence-model arm of the B3 study.",
                RuntimeWarning,
                stacklevel=2,
            )

    def predict(self, speed: float, leg_lengths: Sequence[float]) -> np.ndarray:
        return self._fallback.predict(speed, leg_lengths)


__all__ = ["RNNGenerator"]
