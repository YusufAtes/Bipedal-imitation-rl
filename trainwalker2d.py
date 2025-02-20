import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from walker2d_env import CustomWalker2dEnv
class CustomTrainingCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose: int = 1):
        """
        Callback for saving model checkpoints every `save_freq` timesteps and logging
        episode rewards and lengths.
        """
        super(CustomTrainingCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.last_save_step = 0

    def _on_step(self) -> bool:
        # Save model checkpoint every save_freq timesteps.
        if self.num_timesteps - self.last_save_step >= self.save_freq:
            self.last_save_step = self.num_timesteps
            save_file = os.path.join(self.save_path, f"model_{self.num_timesteps}.zip")
            self.model.save(save_file)
            if self.verbose > 0:
                print(f"[Checkpoint] Saved model to {save_file} at {self.num_timesteps} timesteps")
        
        # Log episode rewards if available (Monitor wrapper inserts "episode" info).
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                ep_length = info["episode"]["l"]
                print(f"[Episode] Reward: {ep_reward:.2f} | Length: {ep_length} timesteps")
        return True

# Create the base Walker2d environment (ensure the version matches your installation).
env = gym.make("Walker2d-v4")
# Wrap the environment with your custom reward and observation modifications.
env = CustomWalker2dEnv(env)
# Wrap with Monitor to log episode statistics.
log_dir = "./walker2d_logs"
os.makedirs(log_dir, exist_ok=True)
env = Monitor(env, log_dir)

# Create the PPO model with an MLP policy.
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./walker2d_tensorboard/")

# Setup directory for saving checkpoints.
save_path = "./walker2d_checkpoints"
os.makedirs(save_path, exist_ok=True)
# Instantiate the custom callback: save every 1M steps.
callback = CustomTrainingCallback(save_freq=1_000_000, save_path=save_path, verbose=1)

# Train for 20 million timesteps.
total_timesteps = 20_000_000
model.learn(total_timesteps=total_timesteps, callback=callback)

# Save the final model.
model.save(os.path.join(save_path, "final_model.zip"))
env.close()