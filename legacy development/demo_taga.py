from bipedplant_roughness import BipedEnv
import os
import numpy as np
from stable_baselines3 import PPO, SAC
import time
import animatebiped
import gait_generator

print('All packages are installed and working.')
desired_speed = 2.0
env = BipedEnv(render_mode='None')
# model = PPO.load("weights/model_checkpoint_15ppo_taga_.zip",device='cpu',env= env)
# policy_kwargs = dict(net_arch=dict(pi=[512, 256], vf=[512, 256]))

model = PPO(
    "MlpPolicy",
    env,
    device="cpu"
    # learning_rate=1e-4,
    # gamma=0.999,
    # clip_range=0.1,
    # ent_coef=0.01,
    # vf_coef=0.75,
    # batch_size=16,
    # max_grad_norm=0.3,
    # gae_lambda=0.99,
    # verbose=0,
)
import matplotlib.pyplot as plt
obs, info = env.reset()  # Gym API
angle_array = np.empty((0,4))
plt.figure()
plt.ion()  # Interactive mode ON
for i in range(0, 50000):
        action, _states = model.predict(obs)
        action = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
        obs, rewards, dones, truncated, info = env.step(action)

        if i % 100 == 0:
            plt.clf()
            states = obs[2:16]
            angles = states[[4,10,7,13]]
            print(angles)
            angle_array = np.vstack((angle_array, angles))
            l_1 = 0.5
            l_2 = 0.6
            hip_x = states[0]
            hip_y = states[1]
            r_knee_x = states[0]    + l_1 * np.cos(states[4]) 
            r_knee_y = states[1]    - l_1 * np.sin(states[4]) 
            l_knee_x = states[0]    + l_1 * np.cos(states[7]) 
            l_knee_y = states[1]    - l_1 * np.sin(states[7]) 
            r_ankle_x = states[0]   + l_2 * np.cos(states[10]) +r_knee_x
            r_ankle_y = states[1]   - l_2 * np.sin(states[10]) -r_knee_y
            l_ankle_x = states[0]   + l_2 * np.cos(states[13]) +l_knee_x
            l_ankle_y = states[1]   - l_2 * np.sin(states[13]) -l_knee_y
            
            plt.plot([hip_x, r_knee_x], [hip_y, r_knee_y], 'bo-')
            plt.plot([hip_x, l_knee_x], [hip_y, l_knee_y], 'ro-')
            plt.plot([r_knee_x, r_ankle_x], [r_knee_y, r_ankle_y], 'bo-')
            plt.plot([l_knee_x, l_ankle_x], [l_knee_y, l_ankle_y], 'ro-')
            plt.plot(hip_x, hip_y, 'go')
            plt.title(f"Time: {i/10000} seconds")
            plt.draw()
            plt.pause(0.01)
plt.ioff()  # Turn OFF interactive mode
plt.show()
