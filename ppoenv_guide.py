"""Backward-compatible entry point that re-exports the refactored :class:`BipedEnv`.

Historical scripts (``demo.py``, ``train_mlp.py``, ``train_lstm.py``,
``amp_implementation/amp_demo.py``, etc.) import :class:`BipedEnv` from this
module. Before the A1 refactor this file was a drifted working copy. It is now
a thin shim that instantiates the canonical environment from
:mod:`biped_env` with a paper-accurate default configuration matching
Configuration 1 (``nodecay_mlp_rsi/PPO_39``).

Callers that want a non-default configuration should import the env from
:mod:`biped_env` directly and pass a :class:`biped_config.BipedEnvConfig` or a
YAML file from ``configs/``.
"""

from __future__ import annotations

from biped_config import BipedEnvConfig
from biped_env import BipedEnv as _BipedEnv


class BipedEnv(_BipedEnv):
    """Shim preserving the historical ``__init__`` signature.

    Parameters
    ----------
    render:
        Legacy flag kept for backward compatibility with
        ``train_mlp_vecenv.BipedEnv(render=False, demo_mode=False)``.
    render_mode:
        Either ``None`` (DIRECT backend) or ``'human'`` (GUI backend).
    demo_mode / demo_type:
        Forwarded verbatim; see :class:`biped_env.BipedEnv`.
    config:
        Optional :class:`BipedEnvConfig`. When omitted the paper-accurate
        Configuration 1 defaults are used (full observation, full reward, RSI,
        no decay). The legacy zero-pad dimensions at ``state[2:5]`` are
        dropped (observation is 55-D) — override with
        ``include_pad_dims=True`` on the config to restore 58-D for historical
        checkpoints.
    """

    def __init__(
        self,
        render: bool = False,
        render_mode: str | None = None,
        demo_mode: bool = False,
        demo_type: str | None = None,
        config: BipedEnvConfig | None = None,
    ) -> None:
        _ = render  # unused; kept for backward compatibility
        super().__init__(
            config=config,
            render_mode=render_mode,
            demo_mode=demo_mode,
            demo_type=demo_type,
        )


__all__ = ["BipedEnv"]
