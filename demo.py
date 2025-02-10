from pybullet_bipedenv_torquecontrolled import BipedEnv
import os
import numpy as np
from stable_baselines3 import PPO, SAC
import time

dt = 1/240 #default of pybullet
model = SAC.load("weights/model_checkpoint_1sac_bpd2d_256_poscontrolnoref.zip",device='cpu')
env = BipedEnv(render_mode='human',control='position',action_dim=6)

for k in range(10):
    obs, info = env.reset()  # Gym API
    t0 = time.time()
    for i in range(0, int(3 * (1/dt))):
        action, _states = model.predict(obs)
        obs, rewards, dones, truncated, info = env.step(action)
        time.sleep(dt)  
    t1 = time.time()
    print(f"Time taken: {t1-t0}")
    time.sleep(2)