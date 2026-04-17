"""Declarative configuration schema for BipedEnv and training.

Every historical run under ``configurations/<tag>/<run>/params.txt`` is a frozen
copy of the training script plus the environment. This module collapses those
snapshots into a single dataclass so the same code path can reproduce every row
of Table 3 in the paper by pointing at a different YAML file.

The top-level :class:`RunConfig` is serialised to YAML by
:func:`save_run_config` and loaded by :func:`load_run_config`. See
``configs/`` for one YAML per historical configuration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml


ObservationMode = Literal["full", "no_im_state"]
RewardMode = Literal["full", "no_im", "gait_only"]
DecaySchedule = Literal["none", "linear_step"]
PolicyArch = Literal["mlp_256_256", "lstm_64_256", "lstm_256_256"]
GaitGeneratorName = Literal[
    "fft_mlp",
    "raw_mocap",
    "cubic_spline",
    "cpg_matsuoka",
    "rnn",
    "amp_placeholder",
]


@dataclass
class DomainRandomizationConfig:
    """Per-episode domain randomization knobs used in Track C5.

    All ranges are inclusive. Setting ``enabled=False`` bypasses every term so
    historical runs can load identical physics to the frozen snapshots.
    """

    enabled: bool = False
    friction_range: tuple[float, float] = (1.0, 1.0)
    foot_mass_scale_range: tuple[float, float] = (1.0, 1.0)
    motor_delay_ms_range: tuple[float, float] = (0.0, 0.0)
    encoder_noise_std: float = 0.0


@dataclass
class BipedEnvConfig:
    """Environment-level switches.

    Attributes
    ----------
    observation:
        ``"full"`` yields the 55-D observation described in Table 1 of the
        paper (with the legacy 3-D zero pad removed, see A3).  ``"no_im_state"``
        zeroes out the 24 reference-preview slots at inference time so the
        policy must rely on proprioception only — used by the ``nostate_mlp``
        ablation.
    reward:
        Selects which terms of equation (5) in the paper are active.
    rsi:
        Random State Initialization; when ``True`` each episode starts at a
        random phase of the reference motion as in Peng et al. 2018.
    decay_schedule / decay_alpha:
        Paper equation (3): ``ω_imitation = 1 - α * (step / total_step)`` and
        ``ω_gait = 1 + α * (step / total_step)``. ``decay_schedule="none"``
        disables the schedule entirely.
    include_pad_dims:
        Legacy compatibility. The frozen snapshots pad three unused zeros at
        positions 2..5 of the observation. Set to ``True`` to reproduce their
        58-D state exactly. The refactored default is ``False`` (55-D).
    """

    observation: ObservationMode = "full"
    reward: RewardMode = "full"
    rsi: bool = True
    decay_schedule: DecaySchedule = "none"
    decay_alpha: float = 0.0
    dt: float = 1e-3
    speed_limit: float = 2.2
    ramp_limit_deg: float = 5.0
    max_episode_seconds: float = 3.0
    include_pad_dims: bool = False
    dr: DomainRandomizationConfig = field(default_factory=DomainRandomizationConfig)

    # Reward sub-component weights. Keeping these explicit (rather than hard
    # coding) lets Track C3 zero them out individually without touching the
    # environment source.
    alive_weight: float = 0.5
    contact_weight: float = 0.6
    speed_weight: float = 0.6
    torque_weight: float = 1e-3
    imitation_weight_hip_pos: float = 0.75
    imitation_weight_knee_pos: float = 0.75
    imitation_weight_ankle_pos: float = 0.25
    imitation_weight_hip_vel: float = 0.15
    imitation_weight_knee_vel: float = 0.15
    imitation_weight_ankle_vel: float = 0.1

    # B2 — swappable gait generator. When ``fft_mlp`` (default) the paper's
    # OldSimpleFCNN is loaded via :class:`gait_generators.FFTMLPGenerator`.
    # Other values point at the baselines registered in
    # :mod:`gait_generators.registry`.
    gait_generator: GaitGeneratorName = "fft_mlp"

    def obs_dim(self) -> int:
        """Return the dimensionality of the observation produced by this config."""
        base = 58 if self.include_pad_dims else 55
        return base


@dataclass
class PolicyConfig:
    """Architecture selector for the PPO policy."""

    arch: PolicyArch = "mlp_256_256"
    activation: Literal["relu", "tanh"] = "relu"


@dataclass
class TrainConfig:
    """Training hyper-parameters used by :mod:`train_mlp` / :mod:`train_lstm`."""

    total_timesteps: int = 15_000_000
    n_steps: int = 8192
    batch_size: int = 256
    n_epochs: int = 5
    clip_range: float = 0.15
    target_kl: float = 0.2
    learning_rate: float = 3e-4
    learning_rate_final: float = 1e-4
    entropy_coef_start: float = 1e-3
    entropy_coef_end: float = 1e-4
    save_freq: int = 500_000
    seed: int = 42


@dataclass
class RunConfig:
    """Top-level container: one YAML file -> one :class:`RunConfig` -> one run."""

    name: str
    description: str = ""
    env: BipedEnvConfig = field(default_factory=BipedEnvConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _to_primitive(value: Any) -> Any:
    if is_dataclass(value):
        return {f.name: _to_primitive(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Mapping):
        return {k: _to_primitive(v) for k, v in value.items()}
    return value


def _from_primitive(dc_cls: type, data: Mapping[str, Any]) -> Any:
    kwargs: dict[str, Any] = {}
    for f in fields(dc_cls):
        if f.name not in data:
            continue
        raw = data[f.name]
        if is_dataclass(f.type) and isinstance(raw, Mapping):
            kwargs[f.name] = _from_primitive(f.type, raw)
        elif f.name == "dr" and isinstance(raw, Mapping):
            kwargs[f.name] = _from_primitive(DomainRandomizationConfig, raw)
        elif f.name == "env" and isinstance(raw, Mapping):
            kwargs[f.name] = _from_primitive(BipedEnvConfig, raw)
        elif f.name == "policy" and isinstance(raw, Mapping):
            kwargs[f.name] = _from_primitive(PolicyConfig, raw)
        elif f.name == "train" and isinstance(raw, Mapping):
            kwargs[f.name] = _from_primitive(TrainConfig, raw)
        elif f.name.endswith("_range") and isinstance(raw, list):
            kwargs[f.name] = tuple(raw)
        else:
            kwargs[f.name] = raw
    return dc_cls(**kwargs)


def save_run_config(cfg: RunConfig, path: str | Path) -> None:
    """Write ``cfg`` to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(_to_primitive(cfg), f, sort_keys=False)


def load_run_config(path: str | Path) -> RunConfig:
    """Load a :class:`RunConfig` from ``path``."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return _from_primitive(RunConfig, data)


__all__ = [
    "BipedEnvConfig",
    "DomainRandomizationConfig",
    "PolicyConfig",
    "RunConfig",
    "TrainConfig",
    "load_run_config",
    "save_run_config",
]
