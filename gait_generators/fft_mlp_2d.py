"""FFT-MLP generator (the paper's current gait network)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from scipy.signal import resample

from .base import BaseGaitGenerator


_REPO = Path(__file__).resolve().parents[1]


class FFTMLP2DGenerator(BaseGaitGenerator):
    """Wraps :class:`gait_generator_net.OldSimpleFCNN`.

    Mirrors the path currently coded inside :class:`biped_env.BipedEnv`
    (previously :mod:`ppoenv_guide`): the MLP predicts 6 x 2 x 17 = 204
    complex Fourier coefficients (real + imaginary), which are de-normalised
    with ``newnormalization_constants.npy`` and inverse-FFT'd to the time
    domain.
    """

    name = "fft_mlp_2d"

    def __init__(
        self,
        weights_path: str | Path = _REPO / "final_model.pth",
        normalization_path: str | Path = _REPO / "newnormalization_constants.npy",
        dt: float = 1e-3,
        tile_repeats: int = 50,
    ) -> None:
        super().__init__(dt=dt, tile_repeats=tile_repeats)
        from gait_generator_net import OldSimpleFCNN

        self.net = OldSimpleFCNN()
        self.net.load_state_dict(torch.load(weights_path, weights_only=True))
        self.net.eval()
        self.normalization = np.load(normalization_path)

    def _denormalize(self, pred: np.ndarray) -> np.ndarray:
        for i in range(17):
            for k in range(2):
                pred[:, k, i] = pred[:, k, i] * self.normalization[i * 2 + k]
        return pred

    def _ifft(self, predictions: np.ndarray) -> np.ndarray:
        real_pred = predictions[:, 0, :]
        imag_pred = predictions[:, 1, :]
        complex_pred = real_pred + 1j * imag_pred
        pred_time = np.fft.irfft(complex_pred, axis=1).T  # (T0, 6)
        org_rate = 10
        if self.dt < 0.1:
            num_samples = int(pred_time.shape[0] * (1 / self.dt) / org_rate)
            pred_time = resample(pred_time, num_samples, axis=0)
            pred_time = np.tile(pred_time, (self.tile_repeats, 1))
        return pred_time

    def predict(self, speed: float, leg_lengths: Sequence[float]) -> np.ndarray:
        encoder_vec = np.empty(3, dtype=np.float32)
        encoder_vec[0] = speed / 3
        encoder_vec[1] = float(leg_lengths[0]) / 1.5
        encoder_vec[2] = float(leg_lengths[1]) / 1.5
        with torch.no_grad():
            freqs = self.net(torch.tensor(encoder_vec, dtype=torch.float32))
        predictions = freqs.reshape(-1, 6, 2, 17).detach().cpu().numpy()[0]
        predictions = self._denormalize(predictions)
        return self._ifft(predictions)


__all__ = ["FFTMLPGenerator"]
