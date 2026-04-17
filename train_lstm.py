"""Recurrent-PPO training driven by a YAML :mod:`biped_config.RunConfig`."""

from __future__ import annotations

import argparse
import os
import time

import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CallbackList

from biped_config import (
    BipedEnvConfig,
    PolicyConfig,
    RunConfig,
    TrainConfig,
    load_run_config,
    save_run_config,
)
from biped_env import BipedEnv
from training_callbacks import (
    CustomCheckpointCallback,
    EntropyDecayCallback,
    RewardComponentLoggerCallback,
    RewardLoggerCallback,
    linear_schedule,
)
from utils import set_global_seed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Recurrent PPO on BipedEnv")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def _lstm_policy_kwargs(policy_cfg: PolicyConfig) -> dict:
    activation = torch.nn.ReLU if policy_cfg.activation == "relu" else torch.nn.Tanh
    if policy_cfg.arch == "lstm_64_256":
        hidden = 64
        head = dict(pi=[256], vf=[256, 256])
    elif policy_cfg.arch == "lstm_256_256":
        hidden = 256
        head = dict(pi=[256], vf=[256, 256])
    else:
        raise ValueError(
            f"train_lstm.py expected an LSTM arch, got {policy_cfg.arch!r}."
        )
    return dict(
        lstm_hidden_size=hidden,
        n_lstm_layers=1,
        shared_lstm=False,
        enable_critic_lstm=False,
        activation_fn=activation,
        net_arch=head,
    )


def main() -> None:
    t0 = time.time()
    args = _parse_args()

    if args.config is None:
        cfg = RunConfig(
            name="config2_lstm_64_256",
            description="Configuration 2 defaults (LSTM 64-256).",
            env=BipedEnvConfig(),
            policy=PolicyConfig(arch="lstm_64_256"),
            train=TrainConfig(),
        )
    else:
        cfg = load_run_config(args.config)

    if args.seed is not None:
        cfg.train.seed = int(args.seed)

    save_dir = args.save_dir or cfg.name
    os.makedirs(save_dir, exist_ok=True)
    save_run_config(cfg, os.path.join(save_dir, "config.yaml"))

    set_global_seed(cfg.train.seed, deterministic=True)

    train_env = BipedEnv(config=cfg.env)
    train_env.total_train_steps = cfg.train.total_timesteps

    callback_list = CallbackList(
        [
            CustomCheckpointCallback(
                save_freq=cfg.train.save_freq,
                save_path=save_dir,
                checkpoint_name=f"{cfg.name}.zip",
                verbose=1,
            ),
            RewardLoggerCallback(log_file=os.path.join(save_dir, "rewards.csv")),
            RewardComponentLoggerCallback(
                log_file=os.path.join(save_dir, "reward_components.csv")
            ),
            EntropyDecayCallback(
                cfg.train.entropy_coef_start,
                cfg.train.entropy_coef_end,
                cfg.train.total_timesteps,
            ),
        ]
    )

    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=train_env,
        n_steps=cfg.train.n_steps,
        batch_size=cfg.train.batch_size,
        n_epochs=cfg.train.n_epochs,
        clip_range=cfg.train.clip_range,
        target_kl=cfg.train.target_kl,
        learning_rate=linear_schedule(
            cfg.train.learning_rate, cfg.train.learning_rate_final
        ),
        ent_coef=cfg.train.entropy_coef_start,
        policy_kwargs=_lstm_policy_kwargs(cfg.policy),
        tensorboard_log=save_dir,
        device="cpu",
        seed=cfg.train.seed,
    )
    print(model.policy)

    model.learn(
        total_timesteps=cfg.train.total_timesteps,
        callback=callback_list,
    )

    model.save(os.path.join(save_dir, "final_model"))
    print(f"Training complete. Models and logs are in: {save_dir}")
    print(f"Total training time: {time.time() - t0:.2f} seconds")


if __name__ == "__main__":
    main()
