"""AMP baseline placeholder.

Unlike the other generators, AMP is adversarial and does not produce an
explicit reference trajectory; it provides a style reward over mocap
clips. The paper compares against AMP trained in
:mod:`amp_implementation`, so for the B3 study we treat AMP as a *zero
reference* generator that disables the imitation reward but keeps a mocap
*clip database* for the AMP training pipeline to consume. This mirrors how
the AMP paper (Peng et al. 2021) handles things.

In the PPO + BipedEnv training loop, selecting ``amp_placeholder`` means:
the policy receives zero reference targets (so the imitation reward is 0)
and the discriminator reward should be provided by the training script
(see :mod:`amp_implementation.train_amp`).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import BaseGaitGenerator


class AMPPlaceholderGenerator(BaseGaitGenerator):
    name = "amp_placeholder"

    def predict(self, speed: float, leg_lengths: Sequence[float]) -> np.ndarray:
        period_s = max(0.35, 1.0 - 0.25 * float(speed))
        n_samples = max(2, int(round(period_s / self.dt)))
        zeros = np.zeros((n_samples, 6), dtype=np.float64)
        return np.tile(zeros, (self.tile_repeats, 1))


__all__ = ["AMPPlaceholderGenerator"]
