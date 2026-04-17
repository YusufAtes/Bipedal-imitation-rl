"""Shared Stable-Baselines3 callbacks used by train_mlp.py and train_lstm.py.

Extracted during the A1 refactor so that the per-component reward logger
(A4) and the other callbacks have a single source of truth.
"""

from __future__ import annotations

import csv
import os
import time
from typing import Callable, Iterable

from stable_baselines3.common.callbacks import BaseCallback


_t0 = time.time()


class RewardLoggerCallback(BaseCallback):
    """Per-episode total-reward logger.

    Writes ``episode, total_reward, termination_step`` to ``log_file``.
    """

    def __init__(self, log_file: str, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_file = log_file
        self.episode_rewards: list[float] = []
        self.current_episode_reward = 0.0
        self.current_step = 0
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", encoding="utf-8") as f:
                f.write("Episode,Total Reward,Termination Step\n")

    def _on_step(self) -> bool:
        dones = self.locals["dones"]
        rewards = self.locals["rewards"]
        self.current_episode_reward += float(rewards[0])
        self.current_step += 1
        if dones[0]:
            self.episode_rewards.append(self.current_episode_reward)
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(
                    f"{len(self.episode_rewards)},{self.current_episode_reward},{self.current_step}\n"
                )
            self.current_episode_reward = 0.0
            self.current_step = 0
        return True

    def _on_training_end(self) -> None:
        print("Training finished. Total episodes:", len(self.episode_rewards))


class RewardComponentLoggerCallback(BaseCallback):
    """Per-step reward-component logger (A4).

    Reads ``info['reward_components']`` populated by
    :class:`biped_env.BipedEnv` and writes a CSV with one column per component
    so later ablations (C3) can attribute gains.
    """

    COLUMNS: Iterable[str] = (
        "step",
        "episode",
        "alive",
        "contact",
        "speed",
        "torque",
        "im_hip_pos",
        "im_knee_pos",
        "im_ankle_pos",
        "im_hip_vel",
        "im_knee_vel",
        "im_ankle_vel",
    )

    def __init__(self, log_file: str, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_file = log_file
        self._episode = 0
        self._step = 0
        if not os.path.exists(self.log_file):
            with open(self.log_file, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.COLUMNS)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        info = infos[0] if infos else {}
        comps = info.get("reward_components")
        if comps is not None:
            self._step += 1
            row = [self._step, self._episode] + [
                float(comps.get(col, 0.0)) for col in list(self.COLUMNS)[2:]
            ]
            with open(self.log_file, "a", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)
        if self.locals.get("dones") is not None and self.locals["dones"][0]:
            self._episode += 1
        return True


class CustomCheckpointCallback(BaseCallback):
    """Save ``model_checkpoint_{i}{name}.zip`` every ``save_freq`` steps."""

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        checkpoint_name: str = "ppo_256_256.zip",
        init_no: int = 0,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.init_no = init_no
        self.checkpoint_name = checkpoint_name
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.init_no += 1
            model_path = os.path.join(
                self.save_path,
                f"model_checkpoint_{self.init_no}{self.checkpoint_name}",
            )
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Model saved at step {self.n_calls} to {model_path}")
                print(f"Time taken for this checkpoint: {time.time() - _t0:.2f} seconds")
        return True


class EntropyDecayCallback(BaseCallback):
    """Linearly decays ``model.ent_coef`` from ``start`` to ``end``."""

    def __init__(self, start: float, end: float, total_timesteps: int, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.start = start
        self.end = end
        self.total = float(total_timesteps)

    def _on_step(self) -> bool:
        progress_remaining = 1.0 - self.model.num_timesteps / self.total
        self.model.ent_coef = self.end + (self.start - self.end) * progress_remaining
        return True


def linear_schedule(initial_value: float, final_value: float = 1e-4) -> Callable[[float], float]:
    """Return a schedule function that linearly decays ``initial_value`` -> ``final_value``."""

    def func(progress_remaining: float) -> float:
        return max(
            final_value + (initial_value - final_value) * progress_remaining, final_value
        )

    return func


__all__ = [
    "RewardLoggerCallback",
    "RewardComponentLoggerCallback",
    "CustomCheckpointCallback",
    "EntropyDecayCallback",
    "linear_schedule",
]
