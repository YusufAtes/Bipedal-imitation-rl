from ppo_demoenv import BipedEnv
import os
import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
import time
import pandas as pd
import matplotlib.pyplot as plt
import imageio


ppo_paths = [
                r"C:\Users\bates\Desktop\Bipedal-imitation-rl\configurations\nodecay_lstm_256_rsi\RecurrentPPO_1",
                r"C:\Users\bates\Desktop\Bipedal-imitation-rl\configurations\nodecay_lstm_256_256_rsi\RecurrentPPO_3"

            ]


env = BipedEnv(demo_mode=True,render_mode="None")
ppo_file = "final_model.zip"
# ppo_file = "final_model.zip"


demo_type = "lstm"


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

# Constant speed of 1.25 m/s on ramps ranging from -15 to 15 degrees in 5 degree increments
test_degrees = list(range(-15, 16, 3))

for ppo_path in ppo_paths:
    if demo_type == "lstm":
        model = RecurrentPPO.load(os.path.join(ppo_path,ppo_file),device='cpu')
    else:
        model = PPO.load(os.path.join(ppo_path,ppo_file),device='cpu')
    model.set_env(env)


    total_attempts = 0
    failed_attempts = 0
    start_pos = 0
    max_speed = 0
    episode_len = 5

    total_rew = 0
    case_no = 1

    for test_degree in test_degrees:

        video_filename = f"demo_lstm_ramp_{test_degree}.mp4"
        if os.path.exists(os.path.join(ppo_path, "videos")):
            pass
        else:
            os.makedirs(os.path.join(ppo_path, "videos"))

        video_path = os.path.join(ppo_path, "videos", video_filename)
        video_fps = 100
        video_frames = []

        past_rhip = []
        past_lhip = []
        total_attempts += 1
        succes = True
        test_speed = 1.25
        test_angle = float(test_degree)

        episode_start = True
        lstm_states = None

        dt = 1e-3 #default of pybullet
        max_steps = int(episode_len*(1/dt))
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
        prev_pos = 0
        speed_change_interval = 500
        avg_duration = 200
        desired_speeds = []
        actual_speeds = []
        current_time = []
        reverse = False
        for i in range(0, int(max_steps/10)):
            print(f"Step {i} of {max_steps/10}")
            if demo_type == "lstm":
                action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_start, deterministic=True)
                episode_start = False
            else:
                action, _states = model.predict(obs)

            obs, rewards, dones, truncated, info = env.step(action)
            past_rhip.append(obs[7])
            past_lhip.append(obs[10])

            frame = env.get_follow_camera_image(
                overlay_text=f"ramp angle {test_degree} deg  speed {test_speed} m/s"
            )
            video_frames.append(np.asarray(frame))

            if i % avg_duration == 0:
                if i != 0:
                    desired_speeds.append(test_speed)
                    actual_speeds.append((env.return_external_state()[1]-prev_pos)/(avg_duration*0.01))
                    current_time.append(i*0.01)
                prev_pos = env.return_external_state()[1]
            ext_state = env.return_external_state()
            time.sleep(0.01)
            if dones:
                succes = False
                mean_speed = 0
                failed_attempts += 1
                terminated = True
                break
            mean_reward += rewards
            episode_steps += 1
        print(f"Desired Speed: {test_speed}, Actual Speed: {(ext_state[1]-prev_pos)/avg_duration*0.01}")
        print(f"Total Time taken: {time.time() - t0} seconds for {episode_steps} steps")
        mean_reward /= episode_steps
        total_rew += mean_reward
        print(f"Mean Reward: {mean_reward}")
        total_travel_dist = ext_state[1]
        mean_speed = total_travel_dist / episode_len
        print(f'Current ramp angle: {test_degree} Speed: {test_speed} Angle: {test_angle}, Actual speed: {mean_speed}')
        if terminated == False:
            if mean_speed < 0.1:
                mean_speed = 0
                succes = False
                failed_attempts += 1
            elif (mean_speed > 3):
                mean_speed = 0
                failed_attempts += 1
                succes = False

        if len(video_frames) > 0:
            imageio.mimsave(video_path, video_frames, fps=video_fps, macro_block_size=None)
            print(f"Saved video with {len(video_frames)} frames to {video_path}")
        else:
            print("No frames captured; video not saved.")
