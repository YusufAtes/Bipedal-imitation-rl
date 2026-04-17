"""PPO training entry point driven by a YAML :mod:`biped_config.RunConfig`.

Usage
-----
    python train_mlp.py                            # Configuration 1 defaults
    python train_mlp.py --config configs/config_05decay_mlp_rsi.yaml
    python train_mlp.py --config configs/rgait_no_torque.yaml --seed 7 \
                        --save-dir runs/rgait_no_torque_s7

The old behaviour (calling ``python train_mlp.py`` with no args) is preserved:
it trains Configuration 1 and writes to ``new_decay_0.25_1.2/`` as before.
"""

from __future__ import annotations

import argparse
import os
import time

import torch
from stable_baselines3 import PPO

from biped_config import RunConfig, load_run_config, save_run_config
from biped_config import BipedEnvConfig, PolicyConfig, TrainConfig
from biped_env import BipedEnv
from stable_baselines3.common.callbacks import CallbackList
from training_callbacks import (
    CustomCheckpointCallback,
    EntropyDecayCallback,
    RewardComponentLoggerCallback,
    RewardLoggerCallback,
    linear_schedule,
)
from utils import set_global_seed


os.environ["PYTHONHASHSEED"] = "23"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on BipedEnv")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a RunConfig YAML. If omitted, Configuration 1 defaults are used.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Output directory. Defaults to ./<config name>.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the seed in the YAML (useful for C6's 5-seed study).",
    )
    return parser.parse_args()


def _build_policy_kwargs(policy_cfg: PolicyConfig) -> dict:
    activation = torch.nn.ReLU if policy_cfg.activation == "relu" else torch.nn.Tanh
    if policy_cfg.arch == "mlp_256_256":
        return dict(activation_fn=activation, net_arch=dict(pi=[256, 256], vf=[256, 256]))
    raise ValueError(
        f"train_mlp.py does not support arch={policy_cfg.arch!r}. Use train_lstm.py for LSTM archs."
    )


def main() -> None:
    t0 = time.time()
    args = _parse_args()

    if args.config is None:
        cfg = RunConfig(
            name="config1_nodecay_mlp_rsi",
            description="Configuration 1 defaults (paper-accurate).",
            env=BipedEnvConfig(),
            policy=PolicyConfig(),
            train=TrainConfig(),
        )
    else:
        cfg = load_run_config(args.config)

    if args.seed is not None:
        cfg.train.seed = int(args.seed)

    save_dir = args.save_dir or cfg.name
    os.makedirs(save_dir, exist_ok=True)
    save_run_config(cfg, os.path.join(save_dir, "config.yaml"))

    torch.set_num_threads(1)
    set_global_seed(cfg.train.seed, deterministic=True)

    train_env = BipedEnv(config=cfg.env)
    train_env.total_train_steps = cfg.train.total_timesteps

    checkpoint_name = f"{cfg.name}.zip"
    checkpoint_cb = CustomCheckpointCallback(
        save_freq=cfg.train.save_freq,
        save_path=save_dir,
        checkpoint_name=checkpoint_name,
        verbose=1,
    )
    reward_logger = RewardLoggerCallback(
        log_file=os.path.join(save_dir, "rewards.csv")
    )
    reward_component_logger = RewardComponentLoggerCallback(
        log_file=os.path.join(save_dir, "reward_components.csv")
    )
    entropy_decay_cb = EntropyDecayCallback(
        cfg.train.entropy_coef_start,
        cfg.train.entropy_coef_end,
        cfg.train.total_timesteps,
    )
    callback_list = CallbackList(
        [checkpoint_cb, reward_logger, reward_component_logger, entropy_decay_cb]
    )

    policy_kwargs = _build_policy_kwargs(cfg.policy)

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        device="cpu",
        tensorboard_log=save_dir,
        n_steps=cfg.train.n_steps,
        batch_size=cfg.train.batch_size,
        n_epochs=cfg.train.n_epochs,
        clip_range=cfg.train.clip_range,
        target_kl=cfg.train.target_kl,
        learning_rate=linear_schedule(
            cfg.train.learning_rate, cfg.train.learning_rate_final
        ),
        ent_coef=cfg.train.entropy_coef_start,
        policy_kwargs=policy_kwargs,
        seed=cfg.train.seed,
    )

    model.learn(
        total_timesteps=cfg.train.total_timesteps,
        callback=callback_list,
    )

    model.save(os.path.join(save_dir, "final_model"))
    print(f"Training complete. Models and logs are in: {save_dir}")
    print(f"Total training time: {time.time() - t0:.2f} seconds")


if __name__ == "__main__":
    main()
