"""Guard against regressions to the legacy 58-D observation space (A3).

Table 1 in the paper claims 55 observation dimensions. The frozen snapshots
under ``configurations/<tag>/params.txt`` used 58 because they carried three
unused zero-pad slots at ``state[2:5]``. After A3 the canonical code path is
55-D; the pad dims are only restorable via ``include_pad_dims=True`` for
loading historical checkpoints.

This test fails if anyone flips the default back to 58.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biped_config import BipedEnvConfig  # noqa: E402
from biped_env import BipedEnv  # noqa: E402


def test_default_observation_is_55d() -> None:
    cfg = BipedEnvConfig()
    assert cfg.include_pad_dims is False, "A3 regression: default should drop pad dims"
    assert cfg.obs_dim() == 55, f"Expected 55, got {cfg.obs_dim()}"

    env = BipedEnv(config=cfg, demo_mode=True)
    obs, _ = env.reset(seed=1, test_speed=1.0, test_angle=0.0)
    env.close()
    assert obs.shape == (55,), f"obs shape mismatch: {obs.shape}"
    assert np.all(np.isfinite(obs))


def test_legacy_58d_available_for_backward_compat() -> None:
    cfg = BipedEnvConfig(include_pad_dims=True)
    assert cfg.obs_dim() == 58, f"Expected 58 legacy, got {cfg.obs_dim()}"

    env = BipedEnv(config=cfg, demo_mode=True)
    obs, _ = env.reset(seed=1, test_speed=1.0, test_angle=0.0)
    env.close()
    assert obs.shape == (58,), f"legacy obs shape mismatch: {obs.shape}"
    assert obs[2] == 0.0 and obs[3] == 0.0 and obs[4] == 0.0, (
        "pad dims must remain zero for checkpoint compatibility"
    )


if __name__ == "__main__":
    test_default_observation_is_55d()
    print("[ok] default observation is 55-D")
    test_legacy_58d_available_for_backward_compat()
    print("[ok] legacy 58-D available via include_pad_dims=True")
