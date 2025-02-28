# from pybullet_bipedenv_torquecontrolled import BipedEnv
from pybullet_bipedenv_poscontrolled import POS_Biped
from pybullett_bipedenv_trcontrol_ankle import BipedEnv
import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3 import PPO, SAC
import time
from typing import Callable
from stable_baselines3.common.env_util import make_vec_env
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
        print("Episode rewards:", self.episode_rewards)

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
                time.sleep(45)

        return True

# Usage

total_timesteps = 10000000
namelist = ["ppo_128_64_ent01"]

for i in range(len(namelist)):
    rewar_Logger_name = namelist[i]+".csv"
    checkpoint_name = namelist[i]+".zip"
    weight_file_name = "final_"+namelist[i]
    use_past_weights = False

    if os.path.exists(namelist[i]):
        pass
    else:
        os.makedirs(namelist[i])
    
    past_weight_path = "weights_upt/model_checkpoint_2ppo_upt_128_64_softupdate_ramp.zip"
    init_no = 3

    if use_past_weights:
        checkpoint_callback = CustomCheckpointCallback(
            save_freq=1000000, save_path=namelist[i],init_no=init_no, verbose=1
        )
    else:
        checkpoint_callback = CustomCheckpointCallback(
            save_freq=1000000, save_path=namelist[i], verbose=1
        )
    reward_logger = RewardLoggerCallback(log_file=rewar_Logger_name)

    callbacks = CallbackList([checkpoint_callback, reward_logger])

    env = BipedEnv(render_mode=None)
    env.reset()

    policy_kwargs = dict(net_arch=dict(pi=[128, 64], vf=[128, 64]))
    print("Starting training")

    model = PPO(
        "MlpPolicy",
        policy_kwargs=policy_kwargs,
        device="cpu",
        env=env,
        tensorboard_log="./"+namelist[i] +"/",
        ent_coef=0.01,
        learning_rate=1e-4,
        clip_range=0.15,
    )

    if use_past_weights:
        model = PPO.load(past_weight_path,device="cpu",ent_coef=0.01)
        model.set_env(env)
        print("Loaded past weights")

    model.learn(total_timesteps=total_timesteps, callback=callbacks)