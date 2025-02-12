from pybullet_bipedenv_torquecontrolled import BipedEnv
import os
import numpy as np
from stable_baselines3 import PPO, SAC
import time

dt = 1e-3 #default of pybullet
model = PPO.load("weights/model_checkpoint_45ppo_acd6_initref_10khz.zip",device='cpu')
env = BipedEnv(render_mode='human',control='torque',action_dim=6)

for k in range(10):
    obs, info = env.reset()  # Gym API
    t0 = time.time()
    for i in range(0, int(3 * (1/dt))):
        action, _states = model.predict(obs)
        obs, rewards, dones, truncated, info = env.step(action)
        time.sleep(dt)
        if i %1000 == 0:
            print(obs)
            print(f"Time taken for this checkpoint: {time.time() - t0:.2f} seconds")