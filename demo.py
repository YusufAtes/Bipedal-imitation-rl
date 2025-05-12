from pybullett_bipedenv_trcontrol_ankle import BipedEnv
from pybullet_bipedenv_poscontrolled import POS_Biped
import os
import numpy as np
from stable_baselines3 import PPO, SAC
import time




# model = PPO(
#     "MlpPolicy",
#     env,
#     device="cpu"
# )

env = BipedEnv(render_mode="human")
model = PPO.load("ppo_512_256\model_checkpoint_3ppo_512_256.zip",device='cpu',deterministic=True)
model.set_env(env) 

speeds = np.linspace(0.5, 3, 6)
covered_distance = np.zeros(len(speeds))
total_attempts = len(speeds)
failed_attempts = 0
start_pos = 0

for k in range(len(speeds)):
    test_speed = speeds[k]
    test_angle =   0*3.14159/180
    dt = 1e-3 #default of pybullet
    total_rew = 0
    obs, info = env.reset(test_speed=test_speed, test_angle= test_angle)  # Gym API
    t0 = time.time()
    first_frame = True
    past_pos = 0
    for i in range(0, int(3* (1/ (10* dt)))):
        action, _states = model.predict(obs)
        obs, rewards, dones, truncated, info = env.step(action)
        ext_state = env.return_external_state()
        if ext_state[2] < 0.7:
            covered_distance[k] = ext_state[1]
            failed_attempts += 1
            break
    covered_distance[k] = ext_state[1]
    print("Speed: ", test_speed, "Distance: ", covered_distance[k])
    # del env
    # del model
print("Failed attempts: ", failed_attempts)
print("Total attempts: ", total_attempts)
print("Success rate: ", (total_attempts - failed_attempts)/total_attempts)

