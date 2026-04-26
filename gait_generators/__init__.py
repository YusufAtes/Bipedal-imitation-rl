"""Swappable gait generators for the PPO bipedal policy (B2).

Every generator implements :class:`BaseGaitGenerator.predict` returning a
joint-position trajectory of shape ``(T, 6)`` for the six controlled joints
``[rhip, rknee, rankle, lhip, lknee, lankle]``, resampled to the environment
control rate ``1/dt``. See :mod:`biped_env` which consumes this via the
``gait_generator`` constructor argument.

The baseline suite used by the B3 head-to-head study is:

- ``fft_mlp``  -- the paper's OldSimpleFCNN predicting FFT coefficients
- ``raw_mocap`` -- nearest-speed motion-capture playback
- ``cubic_spline`` -- per-speed bucket cubic-spline interpolation
- ``cpg_matsuoka`` -- Matsuoka/Taga coupled oscillator CPG
- ``rnn`` -- sequence generator (scaffold falls back to FFT baseline)
- ``amp`` -- Adversarial Motion Priors placeholder dispatched via
  :mod:`amp_implementation`
"""

from __future__ import annotations

from .base import BaseGaitGenerator
from .fft_mlp import FFTMLPGenerator
from .raw_mocap import RawMocapGenerator
from .cubic_spline import CubicSplineGenerator
from .cpg_matsuoka import CPGMatsuokaGenerator
from .rnn_generator import RNNGenerator
from .amp_generator import AMPPlaceholderGenerator
from .fft_mlp_old import FFTMLPOldGenerator
from .registry import GENERATORS, build_generator

__all__ = [
    "BaseGaitGenerator",
    "FFTMLPGenerator",
    "RawMocapGenerator",
    "CubicSplineGenerator",
    "CPGMatsuokaGenerator",
    "RNNGenerator",
    "AMPPlaceholderGenerator",
    "FFTMLPOldGenerator",
    "GENERATORS",
    "build_generator",
]
