from pybullet_bipedenv_torquecontrolled import BipedEnv
import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3 import PPO
import time
t0 = time.time()

total_timesteps = 10000000
rewar_Logger_name ="512_256_normalplant_rampchanged.csv"
checkpoint_name = "_512_256_normalplant_rampchanged.zip"
weight_file_name = "ppo_10M_torquecontrol_512_256_normalplant_rampchanged"
use_past_weights = False
past_weight_path = "weights/model_checkpoint_3200000_512_256_rampchanged.zip"

class RewardLoggerCallback(BaseCallback):
    def __init__(self, log_file: str, verbose: int = 0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.log_file = log_file
        self.episode_rewards = []
        self.current_episode_reward = 0

        # Create the log file if it doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write("Episode,Total Reward\n")

    def _on_step(self) -> bool:
        # Check if the episode has ended by using `done`
        dones = self.locals["dones"]
        rewards = self.locals["rewards"]

        # Accumulate rewards for the current episode
        self.current_episode_reward += rewards[0]

        # If the episode is done, log the reward
        if dones[0]:
            self.episode_rewards.append(self.current_episode_reward)
            with open(self.log_file, 'a') as f:
                
                f.write(f"{len(self.episode_rewards)},{self.current_episode_reward}\n")
            # Reset the reward counter for the next episode
            self.current_episode_reward = 0

        return True

    def _on_training_end(self) -> None:
        # Optionally summarize results at the end of training
        print("Training finished. Total episodes:", len(self.episode_rewards))
        print("Episode rewards:", self.episode_rewards)

class CustomCheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=0):
        super(CustomCheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        done = False
        # Save the model every `save_freq` steps
        if self.n_calls % self.save_freq == 0:
            model_path = f"weights/model_checkpoint_{self.n_calls}"+checkpoint_name
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Model saved at step {self.n_calls} to {model_path}")
                print(f"Time taken for this checkpoint: {time.time() - t0:.2f} seconds")

        return True

# Usage
checkpoint_callback = CustomCheckpointCallback(
    save_freq=200000, save_path='./checkpoints/', verbose=1
)
reward_logger = RewardLoggerCallback(log_file=rewar_Logger_name)

callbacks = CallbackList([checkpoint_callback, reward_logger])


env = BipedEnv(render_mode=None)

policy_kwargs = dict(net_arch=dict(pi=[512, 256], vf=[256, 256]))

model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    # learning_rate=1e-4,
    # gamma=0.999,
    # clip_range=0.1,
    # ent_coef=0.01,
    # vf_coef=0.75,
    # batch_size=16,
    # max_grad_norm=0.3,
    # gae_lambda=0.99,
    # verbose=0,
    device="cpu"
)
if use_past_weights:
    model.load(past_weight_path)
model.learn(total_timesteps=total_timesteps, callback=callbacks)

model.save(weight_file_name)