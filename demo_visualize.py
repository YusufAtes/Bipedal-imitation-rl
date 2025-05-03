# from pybullett_bipedenv_trcontrol_ankle import BipedEnv
# from pybullet_bipedenv_poscontrolled import POS_Biped
from pybullet_biped_7d_ppo13 import BipedEnv
import os
import numpy as np
from stable_baselines3 import PPO, SAC
import time
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting


ppo_path = "ppo_256_256/PPO_36"
env = BipedEnv(demo_mode=True,render_mode='human')
ppo_file = "model_checkpoint_2ppo_256_256.zip"

def count_sign_changes(values):
    if not values or len(values) < 2:
        return 0

    sign_changes = 0
    prev_sign = 0

    for val in values:
        current_sign = 1 if val > 0 else -1 if val < 0 else 0

        if current_sign != 0:
            if prev_sign != 0 and current_sign != prev_sign:
                sign_changes += 1
            prev_sign = current_sign

    return sign_changes


model = PPO.load(os.path.join(ppo_path,ppo_file),device='cpu',deterministic=True)
model.set_env(env) 

case_no = 10


total_attempts = 0
failed_attempts = 0
start_pos = 0
max_speed = 0
episode_len = 5
past_rhip = []
past_lhip = []

for current_no in range(case_no):
    total_rew = 0
    total_attempts += 1
    succes = True
    test_speed = np.random.uniform(0.2, 3)
    test_angle = np.random.uniform(-15,15)*3.14159/180

    dt = 1e-3 #default of pybullet
    total_rew = 0
    episode_len = 10
    max_steps = int(episode_len*(1/dt))
    obs, info = env.reset(test_speed=test_speed, test_angle= test_angle,demo_max_steps = max_steps)
                            #,ground_noise=ground_noise,ground_resolution=0.1)  # Gym API
    t0 = time.time()
    start_pos = 0
    ending_pos = 0
    terminated = False
    no_contact = 0
    first_init = True
    required_fall_time = 10
    j = 0
    print(f'Current case: {current_no} Speed: {test_speed} Angle: {test_angle*180/3.14159}')
    for i in range(0, int(max_steps/10)):
        action, _states = model.predict(obs)

        obs, rewards, dones, truncated, info = env.step(action)
        time.sleep(0.01)
        past_rhip.append(obs[6])
        past_lhip.append(obs[9])
        total_rew += rewards            
        ext_state = env.return_external_state()
        if dones:
            succes = False
            mean_speed = 0
            failed_attempts += 1
            terminated = True
            break

    total_travel_dist = ext_state[1]
    mean_speed = total_travel_dist / episode_len
    #find me the sign change in the past_rhip and past_lhip
    rhip_sign_changes = count_sign_changes(past_rhip)
    lhip_sign_changes = count_sign_changes(past_lhip)
    print("Episode was successful: ", succes)
    if terminated == False:
        if mean_speed < 0.1:
            mean_speed = 0
            succes = False
            failed_attempts += 1
        elif (mean_speed > 3):
            mean_speed = 0
            failed_attempts += 1
            succes = False
        elif rhip_sign_changes < episode_len/2 or lhip_sign_changes < episode_len/2:
            mean_speed = 0
            succes = False
            failed_attempts += 1
        if mean_speed > max_speed:
            max_speed = mean_speed
print(f'maximum speed: {max_speed}')
print(f'failed attempts: {failed_attempts}')