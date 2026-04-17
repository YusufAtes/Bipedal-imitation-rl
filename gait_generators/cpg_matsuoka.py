"""Matsuoka/Taga coupled-oscillator CPG baseline.

This is a classical Central Pattern Generator that requires no training
data. It gives the B3 study a "no imitation at all" generator — if PPO +
CPG already walks well, it reinforces the reviewers' observation that the
imitation reward adds little.

The model follows Matsuoka (1985) with bilateral coupling, exactly the
formulation used by Taga et al. for bipedal locomotion. The commanded
``speed`` modulates the driving tonic input, which in turn sets the
oscillation frequency linearly.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import BaseGaitGenerator


class CPGMatsuokaGenerator(BaseGaitGenerator):
    """Two-neuron reciprocal-inhibition oscillator per joint pair.

    Attributes chosen to roughly match the 0.8-1.2 s gait periods observed
    in the mocap dataset across the 0.2-2.2 m/s commanded speed range.
    """

    name = "cpg_matsuoka"

    def __init__(
        self,
        dt: float = 1e-3,
        tile_repeats: int = 50,
        tau1: float = 0.12,
        tau2: float = 0.24,
        beta: float = 2.5,
        w_self: float = 2.0,
        amplitude: float = 0.35,
        integration_dt: float = 1e-3,
    ) -> None:
        super().__init__(dt=dt, tile_repeats=tile_repeats)
        self.tau1 = tau1
        self.tau2 = tau2
        self.beta = beta
        self.w_self = w_self
        self.amplitude = amplitude
        self.integration_dt = integration_dt

    def _simulate(self, drive: float, n_steps: int) -> np.ndarray:
        x = np.zeros(4)
        v = np.zeros(4)
        y = np.zeros(4)
        out = np.zeros((n_steps, 4))
        for t in range(n_steps):
            y = np.maximum(x, 0.0)
            dx = (
                -x
                - self.w_self * y[[1, 0, 3, 2]]
                - self.beta * v
                + drive
            ) / self.tau1
            dv = (-v + y) / self.tau2
            x = x + dx * self.integration_dt
            v = v + dv * self.integration_dt
            out[t, 0] = y[0] - y[1]
            out[t, 1] = y[2] - y[3]
            out[t, 2] = -out[t, 0]
            out[t, 3] = -out[t, 1]
        return out

    def predict(self, speed: float, leg_lengths: Sequence[float]) -> np.ndarray:
        drive = 0.5 + 0.25 * float(speed)
        period_s = max(0.35, 1.0 - 0.25 * float(speed))
        n_period = int(round(period_s / self.integration_dt))

        raw = self._simulate(drive, n_steps=n_period * 2)
        raw = raw[n_period:]  # drop transient
        raw = raw / (np.max(np.abs(raw)) + 1e-6) * self.amplitude

        n_samples = max(2, int(round(period_s / self.dt)))
        if n_samples != raw.shape[0]:
            idx_src = np.linspace(0, raw.shape[0] - 1, n_samples)
            raw = np.stack(
                [np.interp(idx_src, np.arange(raw.shape[0]), raw[:, j]) for j in range(4)],
                axis=1,
            )

        traj6 = np.zeros((raw.shape[0], 6), dtype=np.float64)
        traj6[:, 0] = raw[:, 0]           # rhip
        traj6[:, 1] = np.maximum(raw[:, 1] - 0.1, -0.9)  # rknee (flexes negative)
        traj6[:, 2] = 0.0
        traj6[:, 3] = raw[:, 2]           # lhip
        traj6[:, 4] = np.maximum(raw[:, 3] - 0.1, -0.9)  # lknee
        traj6[:, 5] = 0.0

        return np.tile(traj6, (self.tile_repeats, 1))


__all__ = ["CPGMatsuokaGenerator"]
