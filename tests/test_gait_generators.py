"""Smoke tests for the gait-generator subpackage (B2)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biped_config import load_run_config  # noqa: E402
from biped_env import BipedEnv  # noqa: E402
from gait_generators import (  # noqa: E402
    GENERATORS,
    CPGMatsuokaGenerator,
    CubicSplineGenerator,
    FFTMLPGenerator,
    RawMocapGenerator,
    RNNGenerator,
    build_generator,
)


EXPECTED_NAMES = {
    "fft_mlp",
    "raw_mocap",
    "cubic_spline",
    "cpg_matsuoka",
    "rnn",
    "amp_placeholder",
}


def test_registry_covers_all_generators() -> None:
    assert set(GENERATORS.keys()) == EXPECTED_NAMES


def test_every_generator_returns_valid_trajectory() -> None:
    speeds = [0.3, 1.0, 2.0]
    leg_lengths = (0.94, 0.94)
    for name in sorted(EXPECTED_NAMES):
        gen = build_generator(name, dt=1e-3, tile_repeats=2)
        for spd in speeds:
            traj = gen.predict(spd, leg_lengths)
            assert traj.ndim == 2 and traj.shape[1] == 6, (name, spd, traj.shape)
            assert traj.shape[0] > 10, (name, spd, traj.shape)
            assert np.all(np.isfinite(traj)), (name, spd)
            assert np.max(np.abs(traj)) < np.pi, (
                name,
                spd,
                float(np.max(np.abs(traj))),
            )


def test_env_accepts_gen_configs() -> None:
    repo = Path(__file__).resolve().parents[1]
    for yaml in sorted((repo / "configs").glob("gen_*.yaml")):
        cfg = load_run_config(yaml)
        env = BipedEnv(config=cfg.env, demo_mode=True)
        obs, _ = env.reset(seed=1, test_speed=1.2, test_angle=0.0)
        for _ in range(5):
            obs, reward, done, truncated, info = env.step(
                np.zeros(7, dtype=np.float32)
            )
            if done or truncated:
                break
        env.close()
        assert obs.shape == (cfg.env.obs_dim(),), (yaml.name, obs.shape)


if __name__ == "__main__":
    test_registry_covers_all_generators()
    print("[ok] registry has all 6 generators")
    test_every_generator_returns_valid_trajectory()
    print("[ok] every generator returns a valid (T,6) trajectory")
    test_env_accepts_gen_configs()
    print("[ok] BipedEnv instantiates each gen_*.yaml cleanly")
