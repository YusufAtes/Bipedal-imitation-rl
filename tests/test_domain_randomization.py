"""Regression tests for the Track C5 domain randomization plumbing.

Covers the four DR knobs wired through :class:`BipedEnv`:

* friction range         -> ``_friction`` is sampled inside the configured range
* foot mass scale        -> foot masses actually change after ``reset()``
* motor delay range      -> ``_motor_delay_steps`` lives in the right range
* encoder noise std      -> policy observations differ across resets

Also asserts that ``enabled=False`` is a no-op.
"""

from __future__ import annotations

import math

import numpy as np

from biped_config import BipedEnvConfig, DomainRandomizationConfig
from biped_env import BipedEnv, _LEFT_FOOT_LINK, _RIGHT_FOOT_LINK


class _Approx:
    """Tiny stand-in for pytest.approx so these tests run without pytest."""

    def __init__(self, expected: float, rel: float = 1e-6, abs: float = 1e-6):
        self.expected = float(expected)
        self.rel = rel
        self.abs = abs

    def __eq__(self, other: object) -> bool:
        o = float(other)
        return math.isclose(o, self.expected, rel_tol=self.rel, abs_tol=self.abs)

    def __repr__(self) -> str:
        return f"approx({self.expected})"


def _approx(expected: float, **kw) -> _Approx:
    return _Approx(expected, **kw)


def _dr_env(**dr_kwargs) -> BipedEnv:
    cfg = BipedEnvConfig()
    cfg.dr = DomainRandomizationConfig(enabled=True, **dr_kwargs)
    return BipedEnv(config=cfg, demo_mode=True, demo_type="track")


def test_dr_disabled_is_identity():
    cfg = BipedEnvConfig()
    env = BipedEnv(config=cfg, demo_mode=True, demo_type="track")
    env.reset(test_speed=1.0, test_angle=0.0)
    assert env._friction == _approx(1.0)
    assert env._motor_delay_steps == 0
    assert env._encoder_noise_std == _approx(0.0)
    assert env._foot_mass_scale == _approx(1.0)
    env.close()


def test_friction_sampled_in_range():
    env = _dr_env(friction_range=(0.6, 1.4))
    samples = []
    for _ in range(6):
        env.reset(test_speed=1.0, test_angle=0.0)
        samples.append(env._friction)
    env.close()
    assert all(0.6 - 1e-6 <= s <= 1.4 + 1e-6 for s in samples)
    assert len(set(samples)) > 1


def test_motor_delay_sampled_in_range():
    env = _dr_env(motor_delay_ms_range=(0.0, 20.0))
    samples = []
    for _ in range(6):
        env.reset(test_speed=1.0, test_angle=0.0)
        samples.append(env._motor_delay_steps)
    env.close()
    assert all(0 <= s <= 20 for s in samples)
    assert len(set(samples)) > 1


def test_foot_mass_scaled():
    env = _dr_env(foot_mass_scale_range=(0.5, 0.5))
    env.reset(test_speed=1.0, test_angle=0.0)
    right = env.p.getDynamicsInfo(env.robot, _RIGHT_FOOT_LINK,
                                  physicsClientId=env.physics_client)[0]
    left = env.p.getDynamicsInfo(env.robot, _LEFT_FOOT_LINK,
                                 physicsClientId=env.physics_client)[0]
    env.close()

    env2 = BipedEnv(config=BipedEnvConfig(), demo_mode=True, demo_type="track")
    env2.reset(test_speed=1.0, test_angle=0.0)
    right_nom = env2.p.getDynamicsInfo(env2.robot, _RIGHT_FOOT_LINK,
                                       physicsClientId=env2.physics_client)[0]
    left_nom = env2.p.getDynamicsInfo(env2.robot, _LEFT_FOOT_LINK,
                                      physicsClientId=env2.physics_client)[0]
    env2.close()

    assert right == _approx(0.5 * right_nom, rel=1e-6)
    assert left == _approx(0.5 * left_nom, rel=1e-6)


def test_encoder_noise_perturbs_observation():
    env = _dr_env(encoder_noise_std=0.05)
    obs_a, _ = env.reset(test_speed=1.0, test_angle=0.0)
    obs_b, _ = env.reset(test_speed=1.0, test_angle=0.0)
    env.close()
    assert np.linalg.norm(obs_a - obs_b) > 1e-6


def test_apply_external_impulse_does_not_crash():
    cfg = BipedEnvConfig()
    env = BipedEnv(config=cfg, demo_mode=True, demo_type="track")
    env.reset(test_speed=1.0, test_angle=0.0)
    env.apply_external_impulse(magnitude=100.0, direction=(1.0, 0.0, 0.0))
    obs, _r, _done, _t, _info = env.step(np.zeros(env.action_space.shape))
    assert obs.shape == env.observation_space.shape
    env.close()
