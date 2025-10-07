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

import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList
from sb3_contrib import RecurrentPPO
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

## train_lstm_vecenv.py


def make_env(rank: int, base_seed: int = 0, max_episode_steps: int = 301):
    def _init():
        env = BipedEnv(render=False, demo_mode=False)   # ensure p.DIRECT inside BipedEnv
        env = Monitor(env)
        env = gym.wrappers.TimeLimit(env, max_episode_steps)

        # Gymnasium-style seeding:
        seed = base_seed + rank
        # 1) seed action/obs spaces (optional but nice)
        if hasattr(env, "action_space") and hasattr(env.action_space, "seed"):
            env.action_space.seed(seed)
        if hasattr(env, "observation_space") and hasattr(env.observation_space, "seed"):
            env.observation_space.seed(seed)
        # 2) reset with seed (this is the canonical way in Gymnasium)
        env.reset(seed=seed)

        return env
    return _init



if __name__ == "__main__":
    # ---- 1) Multiprocessing start method
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)  # crucial on Linux + PyTorch

    # ---- 2) Build vectorized env in subprocesses
    N_ENVS = 4   # try 4/6/8 and benchmark
    env_fns = [make_env(i, base_seed=1234) for i in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns, start_method="spawn")

    # ---- 3) Normalization AFTER vec env
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                           clip_obs=10.0, clip_reward=10.0)

    # ---- 4) Callbacks (define these classes elsewhere; they must be importable)
    checkpoint_cb = CustomCheckpointCallback(save_freq=500_000, save_path="ppo_lstm", verbose=1)
    reward_logger = RewardLoggerCallback(log_file=os.path.join("ppo_lstm", "rewards.csv"))
    ENT_START, ENT_END = 1e-3, 1e-4
    entropy_decay_cb = EntropyDecayCallback(ENT_START, ENT_END, total_timesteps=15_000_000)
    callback_list = CallbackList([checkpoint_cb, reward_logger, entropy_decay_cb])

    # ---- 5) Model
    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=vec_env,
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        learning_rate=3e-4,
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
            shared_lstm=True,
            enable_critic_lstm=False,      
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[256], vf=[256]),
        ),
        tensorboard_log="ppo_lstm",
        device="cuda",
    )

    # ---- 6) Train
    model.learn(total_timesteps=15_000_000, callback=callback_list)

    # ---- 7) Save artifacts (including normalization stats)
    os.makedirs("ppo_lstm", exist_ok=True)
    model.save(os.path.join("ppo_lstm", "final_model"))
    vec_env.save(os.path.join("ppo_lstm", "vecnormalize.pkl"))
    print("Training complete.")