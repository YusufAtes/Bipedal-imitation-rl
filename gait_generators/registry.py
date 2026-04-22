"""Registry mapping string names to gait-generator classes."""

from __future__ import annotations

from typing import Any, Callable, Dict

from .amp_generator import AMPPlaceholderGenerator
from .base import BaseGaitGenerator
from .cpg_matsuoka import CPGMatsuokaGenerator
from .cubic_spline import CubicSplineGenerator
from .fft_mlp_2d import FFTMLP2DGenerator
from .raw_mocap import RawMocapGenerator
from .rnn_generator import RNNGenerator
from .fft_mlp_review import FFTMLPReviewGenerator
from .fft_mlp import FFTMLPGenerator
from .fft_mlp_v2 import FFTMLPv2Generator

GENERATORS: Dict[str, Callable[..., BaseGaitGenerator]] = {
    FFTMLP2DGenerator.name: FFTMLP2DGenerator,
    RawMocapGenerator.name: RawMocapGenerator,
    CubicSplineGenerator.name: CubicSplineGenerator,
    CPGMatsuokaGenerator.name: CPGMatsuokaGenerator,
    RNNGenerator.name: RNNGenerator,
    AMPPlaceholderGenerator.name: AMPPlaceholderGenerator,
    FFTMLPReviewGenerator.name: FFTMLPReviewGenerator,
    FFTMLPGenerator.name: FFTMLPGenerator,
    FFTMLPv2Generator.name: FFTMLPv2Generator,
}


def build_generator(name: str, **kwargs: Any) -> BaseGaitGenerator:
    """Instantiate a gait generator by its registry name."""
    if name not in GENERATORS:
        raise KeyError(
            f"Unknown gait generator {name!r}. Registered: {sorted(GENERATORS)}"
        )
    return GENERATORS[name](**kwargs)


__all__ = ["GENERATORS", "build_generator"]