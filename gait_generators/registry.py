"""Registry mapping string names to gait-generator classes."""

from __future__ import annotations

from typing import Any, Callable, Dict

from .amp_generator import AMPPlaceholderGenerator
from .base import BaseGaitGenerator
from .cpg_matsuoka import CPGMatsuokaGenerator
from .cubic_spline import CubicSplineGenerator
from .fft_mlp import FFTMLPGenerator
from .raw_mocap import RawMocapGenerator
from .rnn_generator import RNNGenerator
from .fft_mlp_review import FFTMLPReviewGenerator

GENERATORS: Dict[str, Callable[..., BaseGaitGenerator]] = {
    FFTMLPGenerator.name: FFTMLPGenerator,
    RawMocapGenerator.name: RawMocapGenerator,
    CubicSplineGenerator.name: CubicSplineGenerator,
    CPGMatsuokaGenerator.name: CPGMatsuokaGenerator,
    RNNGenerator.name: RNNGenerator,
    AMPPlaceholderGenerator.name: AMPPlaceholderGenerator,
    FFTMLPReviewGenerator.name: FFTMLPReviewGenerator,
}


def build_generator(name: str, **kwargs: Any) -> BaseGaitGenerator:
    """Instantiate a gait generator by its registry name."""
    if name not in GENERATORS:
        raise KeyError(
            f"Unknown gait generator {name!r}. Registered: {sorted(GENERATORS)}"
        )
    return GENERATORS[name](**kwargs)


__all__ = ["GENERATORS", "build_generator"]
