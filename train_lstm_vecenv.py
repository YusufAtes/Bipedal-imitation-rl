from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
    BaseCallback,
)
import torch
from ppoenv_guide import BipedEnv
import os
import datetime
from math import cos, pi
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3 import PPO
import time
from typing import Callable
from utils import set_global_seed
from sb3_contrib import RecurrentPPO
import gymnasium as gym
set_global_seed(23)

t0 = time.time()
class RewardLoggerCallback(BaseCallback):
    def __init__(self, log_file: str, verbose: int = 0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.log_file = log_file
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.current_step = 0  # Track the number of steps in the current episode

        # Create the log file if it doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write("Episode,Total Reward,Termination Step\n")

    def _on_step(self) -> bool:
        # Check if the episode has ended
        dones = self.locals["dones"]
        rewards = self.locals["rewards"]

        # Accumulate rewards for the current episode
        self.current_episode_reward += rewards[0]
        self.current_step += 1  # Increment step count for the current episode

        # If the episode is done, log the reward and termination step
        if dones[0]:
            self.episode_rewards.append(self.current_episode_reward)
            with open(self.log_file, 'a') as f:
                f.write(f"{len(self.episode_rewards)},{self.current_episode_reward},{self.current_step}\n")
            
            # Reset counters for the next episode
            self.current_episode_reward = 0
            self.current_step = 0

        return True
    def _on_training_end(self) -> None:
        # Optionally summarize results at the end of training
        print("Training finished. Total episodes:", len(self.episode_rewards))

class CustomCheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path,init_no = 0, verbose=0):
        super(CustomCheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.init_no = init_no

    def _on_step(self) -> bool:
        done = False
        # Save the model every `save_freq` steps
        if self.n_calls % self.save_freq == 0:
            self.init_no +=1
            model_path = f"{self.save_path}/model_checkpoint_{self.init_no}"+checkpoint_name
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Model saved at step {self.n_calls} to {model_path}")
                print(f"Time taken for this checkpoint: {time.time() - t0:.2f} seconds")
                time.sleep(5)

        return True
    
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Returns a function that computes
    `progress_remaining * initial_value`.
    """
    def func(progress_remaining: float) -> float:
        return max(progress_remaining * initial_value, 1e-4)  # Ensure a minimum value
    return func

namelist = ["ppo_256_256"]
checkpoint_name = namelist[0]+".zip"
reward_logger_name = namelist[0]+".csv"

# ---------- Entropy‑decay callback ---------------------------------------------

class EntropyDecayCallback(BaseCallback):
    """Linearly decays `model.ent_coef` from *start* → *end*.
    SB3 (≤2.0) stores `ent_coef` as a plain float, so scheduling must be manual.
    """
    def __init__(self, start: float, end: float, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.start = start
        self.end = end
        self.total = float(total_timesteps)

    def _on_step(self) -> bool:
        # progress_remaining: 1 → 0 over training
        progress_remaining = 1.0 - self.model.num_timesteps / self.total
        new_coef = self.end + (self.start - self.end) * progress_remaining
        self.model.ent_coef = new_coef
        return True

# ---------- MAIN TRAINING LOOP -------------------------------------------------
import os, gym
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

def make_env(idx, render=False, demo_mode=False, max_episode_steps=301, base_seed=0):
    def _init():
        # IMPORTANT: start PyBullet in DIRECT mode in your env constructor
        # e.g., in BipedEnv.__init__: p.connect(p.DIRECT) when render=False
        env = BipedEnv(render=render, demo_mode=demo_mode)
        env = Monitor(env)
        env = gym.wrappers.TimeLimit(env, max_episode_steps)
        env.seed(base_seed + idx)
        return env
    return _init

N_ENVS = 8  # try 4, 6, 8 and compare throughput (steps/sec)
vec_env = SubprocVecEnv([make_env(i, render=False, demo_mode=False, base_seed=1234)
                         for i in range(N_ENVS)],
                        start_method="spawn")  # robust with PyTorch

vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

# ------------------- TRAINING CONFIG ------------------------------------------
if __name__ == "__main__":
    t0 = time.time()
    TOTAL_TIMESTEPS = 15_000_000
    SAVE_DIR = "ppo_lstm"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ------------------- CALLBACKS --------------------------------------------
    checkpoint_cb = CustomCheckpointCallback(
        save_freq=500_000,
        save_path=SAVE_DIR,
        verbose=1,
    )
    reward_logger = RewardLoggerCallback(
        log_file=os.path.join(SAVE_DIR, "rewards.csv")
    )

    ENT_START = 1e-3
    ENT_END   = 1e-4
    entropy_decay_cb = EntropyDecayCallback(ENT_START, ENT_END, TOTAL_TIMESTEPS)

    callback_list = CallbackList([checkpoint_cb, reward_logger, entropy_decay_cb])

    # ------------------- PPO (Recurrent) --------------------------------------
    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=vec_env,
        n_steps=512,             # shorter rollouts help BPTT
        batch_size=128,          # must divide (n_envs * n_steps)
        n_epochs=10,
        learning_rate=3e-4,      # can decay linearly to 1e-4
        clip_range=0.2,
        clip_range_vf=0.2,
        gae_lambda=0.9,
        gamma=0.99,
        ent_coef=ENT_START,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.02,
        policy_kwargs=dict(
            lstm_hidden_size=128,
            n_lstm_layers=1,
            shared_lstm=True,          # actor+critic share recurrent core
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[256], vf=[256])  # simpler heads (LSTM handles time)
        ),
        tensorboard_log=SAVE_DIR,
        device="cuda",
    )

    print(model.policy)

    # ------------------- TRAIN ------------------------------------------------
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback_list,
        progress_bar=True,
    )

    # ------------------- SAVE FINAL ------------------------------------------
    model.save(os.path.join(SAVE_DIR, "final_model"))
    vec_env.save(os.path.join(SAVE_DIR, "vecnormalize.pkl"))

    print(f"Training complete. Models/logs: {SAVE_DIR}")
    print(f"Total training time: {time.time() - t0:.2f} seconds")
