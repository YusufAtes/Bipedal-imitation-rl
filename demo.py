from pybullett_bipedenv_trcontrol_ankle import BipedEnv
from pybullet_bipedenv_poscontrolled import POS_Biped
import os
import numpy as np
from stable_baselines3 import PPO, SAC
import time


dt = 1e-3 #default of pybullet
env = BipedEnv(render_mode='human')
model = PPO.load("ppo_128_64_ent01/PPO_8/model_checkpoint_2ppo_128_64_ent01.zip",device='cpu',deterministic=True)
model.set_env(env)   

# model = PPO(
#     "MlpPolicy",
#     env,
#     device="cpu"
# )
rewards_list = []
for k in range(10):
    total_rew = 0
    obs, info = env.reset()  # Gym API
    t0 = time.time()
    first_frame = True
    past_pos = 0
    for i in range(0, int(2* (1/ (10* dt)))):
        if first_frame:
            time.sleep(0.5)
            first_frame = False
        action, _states = model.predict(obs)
        obs, rewards, dones, truncated, info = env.step(action)
        total_rew += rewards
        time.sleep(0.01)
        if dones:
            print("DONE")
            break
    print(f'Reward: {total_rew}')
    rewards_list.append(total_rew)
print(f"Average reward: {np.mean(rewards_list)}")
