from pybullet_bipedenv_trcontrol_updated import BipedEnv
from pybullet_bipedenv_poscontrolled import POS_Biped
import os
import numpy as np
from stable_baselines3 import PPO, SAC
import time


dt = 1e-3 #default of pybullet
env = BipedEnv(render_mode='human')
model = PPO.load("weights_upt/model_checkpoint_2ppo_upt_64_64_softupdate.zip",device='cpu',deterministic=True)
model.set_env(env)   

# model = PPO(
#     "MlpPolicy",
#     env,
#     device="cpu"
# )


for k in range(10):

    obs, info = env.reset()  # Gym API
    t0 = time.time()
    first_frame = True
    for i in range(0, int(3* (1/ (10* dt)))):
        action, _states = model.predict(obs)
        obs, rewards, dones, truncated, info = env.step(action)
        time.sleep(dt)
        if first_frame:
            time.sleep(1)
            first_frame = False
        # if dones:
        #     break
