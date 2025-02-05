from pybullet_bipedenv_torquecontrolled import BipedEnv
import os
import numpy as np
from stable_baselines3 import PPO
import time


model = PPO.load("weights/model_checkpoint_3000000_512_256_gptplant_rampchanged.zip")
env = BipedEnv(render_mode='human')
obs, info = env.reset()  # Gym API
for i in range(0, 300):
    action, _states = model.predict(obs)
    obs, rewards, dones, truncated, info = env.step(action)
    time.sleep(1/240)  