# from pybullett_bipedenv_trcontrol_ankle import BipedEnv
# from pybullet_bipedenv_poscontrolled import POS_Biped
# from pybullet_biped_7d_ppo13 import BipedEnv
from ppoenv_guide import BipedEnv
import time
import os
import numpy as np
from stable_baselines3 import PPO, SAC
import time
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
ppo_path = "ppo_256_256/PPO_27"
env = BipedEnv(demo_mode=True)
ppo_file = "final_model.zip"

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

speed_len = 6
angle_len = 41

speeds = np.linspace(0.4, 2.8, speed_len)
angles = np.linspace(-20, 20, angle_len)

ground_noises = [0.0, 0.02,0.04 ,0.06,0.08 ,0.1]
ground_resolutions = [0.1]
covered_distance = np.zeros(len(speeds))
start_pos = 0
max_speed = 0
past_rhip = []
past_lhip = []
avg_trial_no= 3
scenario_count = 10
failed_experiences = 0
total_experiences = 0
total_exp_no = speed_len * angle_len * avg_trial_no * scenario_count 

for ground_noise in ground_noises:

    for ground_resolution in ground_resolutions:
        t0 = time.time()
        exp_speeds = np.zeros((speed_len, angle_len))
        exp_success = np.zeros((speed_len, angle_len))

        for scenario in range(scenario_count):
            # Generate heightfield data with noise in the range [-ground_noise, ground_noise]
            heightfield_data = np.random.uniform(low=-ground_noise, high=ground_noise, size=128 * 1024).tolist()

            for angle_no in range(len(angles)):

                for speed_no in range(len(speeds)):
                    desired_speed = speeds[speed_no]
                    angle = angles[angle_no]
                    avg_mean_speeds = 0
                    failed_attempts = 0

                    for j in range(avg_trial_no):
                        total_experiences += 1
                        total_rew = 0
                        success = True

                        dt = 1e-3 #default of pybullet
                        total_rew = 0
                        episode_len = 4 # seconds
                        max_steps = int(episode_len*(1/dt))
                        obs, img, info = env.reset(test_speed=desired_speed, test_angle=angle, demo_max_steps=max_steps
                                                , ground_noise=ground_noise, ground_resolution=ground_resolution, heightfield_data=heightfield_data)  # Gym API
                        if (angle == 0) and (scenario == 0):
                            img.save(os.path.join(ppo_path, f"demo_render_{ground_noise}_{ground_resolution}.jpg"))
                        start_pos = 0
                        ending_pos = 0
                        terminated = False
                        no_contact = 0
                        first_init = True
                        j = 0
                        contact_list = []

                        for i in range(0, int(max_steps/10)):
                            action, _states = model.predict(obs)
                            obs, rewards, dones, truncated, info = env.step(action)
                            past_rhip.append(obs[7])
                            past_lhip.append(obs[10])
                            total_rew += rewards            
                            ext_state = env.return_external_state()

                            if dones:
                                success = False
                                mean_speed = 0
                                failed_attempts += 1
                                terminated = True
                                break
                        total_travel_dist = ext_state[1]
                        mean_speed = total_travel_dist / episode_len

                        if terminated == False:
                            if mean_speed < 0.0:
                                mean_speed = 0
                                success = False
                                failed_attempts += 1
                        if success != False:
                            avg_mean_speeds += mean_speed

                    if total_experiences % 250 == 0:
                        print(f'{total_experiences / total_exp_no:.2%}% of {total_exp_no} experiences done for Resolution: {ground_resolution}, Amplitude: {ground_noise}')
                        print(f"Time taken for this scenario: {time.time() - t0:.2f} seconds")
                        t0 = time.time()
                    if failed_attempts >= 2:
                        avg_mean_speeds = 0
                        avg_success = 0
                    else:
                        avg_mean_speeds /= (avg_trial_no - failed_attempts)
                        avg_success = 1

                    exp_speeds[speed_no, angle_no] += avg_mean_speeds
                    exp_success[speed_no, angle_no] += avg_success

        exp_speeds /= scenario_count
        exp_success /= scenario_count
        S, A = np.meshgrid(speeds, angles)
        exp_speeds = exp_speeds.flatten()
        exp_success = exp_success.flatten()
        # plot the results of the demo make the x axis the speed and the y axis the angle and the color the distance and make the failures red
        plt.figure(figsize=(8, 6))
        mean_speed_plot = plt.scatter(S, A, c=exp_speeds, cmap='Blues', edgecolors='k')
        # Add colorbar
        speed_cbar = plt.colorbar(mean_speed_plot)
        speed_cbar.set_label('Mean Speed')
        plt.xlabel('Speed Value')
        plt.ylabel('Ramp Angle')
        plt.title(f'{ground_noise} noise level {ground_resolution} resolution')
        plt.savefig(os.path.join(ppo_path, f"demo_avg_speeds_{ground_noise}_{ground_resolution}.png"))
        plt.close()

        plt.figure(figsize=(8, 6))
        avg_success_plot = plt.scatter(S, A, c=exp_success, cmap='Greens', edgecolors='k')
        plt.xlabel('Speed Value')
        plt.ylabel('Ramp Angle')
        plt.title(f'Success Rate for {ground_noise} noise level {ground_resolution} resolution')
        success_cbar = plt.colorbar(avg_success_plot)
        success_cbar.set_label('Success Rate')
        plt.savefig(os.path.join(ppo_path, f"demo_avg_success_{ground_noise}_{ground_resolution}.png"))
        plt.close()