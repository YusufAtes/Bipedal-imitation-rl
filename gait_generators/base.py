"""Abstract base class for gait generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class BaseGaitGenerator(ABC):
    """All generators return joint trajectories of shape ``(T, 6)``.

    The six columns are ``[rhip, rknee, rankle, lhip, lknee, lankle]`` in
    radians. ``T`` spans roughly the number of control steps required by
    :mod:`biped_env` (which tiles the output to cover a full episode).
    """

    name: str = "base"

    def __init__(self, dt: float = 1e-3, tile_repeats: int = 50) -> None:
        self.dt = dt
        self.tile_repeats = tile_repeats

    @abstractmethod
    def predict(
        self,
        speed: float,
        leg_lengths: Sequence[float],
    ) -> np.ndarray:
        """Return the reference trajectory for the given command.

        Parameters
        ----------
        speed:
            Commanded forward speed in m/s.
        leg_lengths:
            ``(r_leg_length, l_leg_length)`` in metres. Most baselines are
            morphology-agnostic but receive this tuple so morphology
            generalisation experiments (B1) can include them as a control.

        Returns
        -------
        np.ndarray
            Shape ``(T, 6)`` in radians, tiled ``self.tile_repeats`` times so
            that an episode running at the control rate never runs out of
            reference samples.
        """


__all__ = ["BaseGaitGenerator"]
