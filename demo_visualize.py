from ppo_demoenv import BipedEnv
import os
import numpy as np
from stable_baselines3 import PPO
import time
import pandas as pd
import matplotlib.pyplot as plt
from utils import create_noisy_plane

ppo_path = "ppo_newreward/PPO_16"
env = BipedEnv(demo_mode=True,render_mode="human")
ppo_file = "final_model.zip"

# ppo_path = "ppo_256_256/PPO_22"
# env = BipedEnv(demo_mode=True,render_mode="human")
# ppo_file = "model_checkpoint_20ppo_256_256.zip"

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

total_rew = 0
ground_noise = 0.05
gamma = 0.5
for current_no in range(case_no):
    
    past_rhip = []
    past_lhip = []
    total_attempts += 1
    succes = True
    test_speed = 1.5
    test_angle = 0.0

    dt = 1e-3 #default of pybullet
    episode_len = 5
    max_steps = int(episode_len*(1/dt))
    if ground_noise > 0:
        heightfield_data = create_noisy_plane(gamma=gamma, omega=ground_noise)

        obs, info = env.reset(test_speed=test_speed, test_angle= test_angle,demo_max_steps = max_steps
                                    ,ground_noise=ground_noise,ground_resolution=0.05, heightfield_data=heightfield_data)  # Gym API
    else:
        obs, info = env.reset(test_speed=test_speed, test_angle= test_angle,demo_max_steps = max_steps)
    t0 = time.time()
    start_pos = 0
    ending_pos = 0
    terminated = False
    no_contact = 0
    first_init = True
    required_fall_time = 10
    j = 0
    contact_list = []
    
    mean_reward = 0
    episode_steps = 0
    for i in range(0, int(max_steps/10)):
        action, _states = model.predict(obs)

        obs, rewards, dones, truncated, info = env.step(action)
        # time.sleep(0.01)
        past_rhip.append(obs[7])
        past_lhip.append(obs[10])


        ext_state = env.return_external_state()
        # contact_list.append(len(contact_points))
        if len(contact_list) == 50:
            #remove the first element
            if np.mean(contact_list) == 0:
                print("Contact points: ",np.mean(contact_list))
            contact_list.pop(0)
            
        if dones:
            succes = False
            mean_speed = 0
            failed_attempts += 1
            terminated = True
            break
        mean_reward += rewards
        episode_steps += 1
    mean_reward /= episode_steps
    total_rew += mean_reward
    # print(f"Mean Reward: {mean_reward}")       
    total_travel_dist = ext_state[1]
    mean_speed = total_travel_dist / episode_len
    # print(f'Current case: {current_no} Speed: {test_speed} Angle: {test_angle}, Actual speed: {mean_speed}')
    if terminated == False:
        if mean_speed < 0.1:
            mean_speed = 0
            succes = False
            failed_attempts += 1
        elif (mean_speed > 3):
            mean_speed = 0
            failed_attempts += 1
            succes = False
total_rew /= case_no
print(f"Total Reward: {total_rew}")