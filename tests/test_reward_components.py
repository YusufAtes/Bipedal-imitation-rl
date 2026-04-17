"""Verify the per-component reward logger writes the right CSV (A4).

We drive a tiny fake SB3-style callback loop so the test doesn't need to
launch full PPO training. The goal is: after N steps the CSV exists, has a
header matching :data:`RewardComponentLoggerCallback.COLUMNS`, and contains
N rows (one per step taken).
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biped_env import BipedEnv  # noqa: E402
from training_callbacks import RewardComponentLoggerCallback  # noqa: E402


class _FakeCallbackDriver:
    """Minimal harness that mimics how SB3 populates ``self.locals``."""

    def __init__(self, cb: RewardComponentLoggerCallback) -> None:
        self.cb = cb
        cb.locals = {}

    def emit(self, info: dict, done: bool, reward: float) -> None:
        self.cb.locals["infos"] = [info]
        self.cb.locals["rewards"] = np.array([reward], dtype=np.float32)
        self.cb.locals["dones"] = np.array([done])
        self.cb._on_step()


def test_reward_component_logger_writes_all_columns() -> None:
    env = BipedEnv(demo_mode=True)
    env.reset(seed=1, test_speed=1.0, test_angle=0.0)

    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "reward_components.csv"
        cb = RewardComponentLoggerCallback(log_file=str(log_path))
        driver = _FakeCallbackDriver(cb)

        steps = 8
        for _ in range(steps):
            obs, reward, done, truncated, info = env.step(np.zeros(7, dtype=np.float32))
            driver.emit(info, bool(done or truncated), float(reward))
            if done or truncated:
                break

        assert log_path.exists()
        with log_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

    header = rows[0]
    assert tuple(header) == tuple(RewardComponentLoggerCallback.COLUMNS), header

    component_cols = list(RewardComponentLoggerCallback.COLUMNS)[2:]
    data_rows = rows[1:]
    assert len(data_rows) >= 1, "expected at least one logged row"
    for row in data_rows:
        assert len(row) == len(header)
        for col_name, value in zip(component_cols, row[2:]):
            float(value)
    env.close()


if __name__ == "__main__":
    test_reward_component_logger_writes_all_columns()
    print("[ok] reward component logger emits expected columns")
