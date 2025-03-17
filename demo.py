from pybullett_bipedenv_trcontrol_ankle import BipedEnv
from pybullet_bipedenv_poscontrolled import POS_Biped
import os
import numpy as np
from stable_baselines3 import PPO, SAC
import time
import pandas as pd
import matplotlib.pyplot as plt

ppo_path = "ppo_256_128/PPO_25"
env = BipedEnv()
ppo_file = "model_checkpoint_2ppo_256_128.zip"


model = PPO.load(os.path.join(ppo_path,ppo_file),device='cpu',deterministic=True)
model.set_env(env) 

speeds = np.linspace(0.3, 3, 20) 
angles = np.linspace(-15, 15, 90)

covered_distance = np.zeros(len(speeds))
total_attempts = 0
failed_attempts = 0
start_pos = 0
demo_cases = pd.DataFrame(columns=['Speed', 'Angle', 'Distance', 'Success','Reward'])
max_speed = 0
for angle in angles:
    for k in range(len(speeds)):
        
        total_attempts += 1
        succes = True
        test_speed = speeds[k]
        test_angle = angle*3.14159/180

        dt = 1e-3 #default of pybullet
        total_rew = 0
        max_steps = int(6*(1/dt))
        obs, info = env.reset(test_speed=test_speed, test_angle= test_angle,demo_max_steps = max_steps)  # Gym API
        t0 = time.time()
        start_pos = 0
        ending_pos = 0

        for i in range(0, int(max_steps/10)):
            action, _states = model.predict(obs)
            obs, rewards, dones, truncated, info = env.step(action)
            total_rew += rewards
            ext_state = env.return_external_state()

            if dones:
                failed_attempts += 1
                succes = False
                break

            if i == int(4* (1/ (10* dt))):
                start_pos = ext_state[1]
            if i == int(5* (1/ (10* dt))):
                ending_pos = ext_state[1]


        distance = ending_pos-start_pos
        if distance > max_speed:
            max_speed = distance
            print(f'New max speed: {max_speed}')
        if (distance < 0.2) or (distance > 2.5):
            distance = 0
            failed_attempts += 1
            succes = False
        test_angle_deg = 180*test_angle/3.14159
        #print the results only two decimal points
        
        # print(f"Speed: {test_speed:.2f}, Angle: {test_angle_deg:.2f}, Distance: {distance:.2f}, Success: {succes}, Reward: {total_rew}")
        
        #add to the dataframe
        demo_cases = pd.concat([demo_cases, pd.DataFrame([[test_speed, test_angle_deg, distance, 
                                                        succes, total_rew]], columns=['Speed', 'Angle', 'Distance', 
                                                                                        'Success','Reward'])], ignore_index=True)
        
demo_cases.to_csv(os.path.join(ppo_path, "demo_results.csv"), index=False)

#plot the results of the demo make the x axis the speed and the y axis the angle and the color the distance and make the failures red
plt.figure()
plt.figure(figsize=(8, 6))
sc = plt.scatter(demo_cases['Speed'], demo_cases['Angle'], c=demo_cases['Distance'], cmap='Blues', edgecolors='k')
failed_cases = demo_cases[demo_cases['Success'] == False]
plt.scatter(failed_cases['Speed'], failed_cases['Angle'], c='red')

# Add colorbar
cbar = plt.colorbar(sc)
cbar.set_label('Distance')

plt.legend(['Success', 'Failed'])

plt.xlabel('Speed Value')
plt.ylabel('Ramp Angle')
plt.title('Demo results')
plt.savefig(os.path.join(ppo_path, "demo_results.png"))
plt.close()
#
print("Failed attempts: ", failed_attempts)
print("Total attempts: ", total_attempts)
print("Success rate: ", (total_attempts - failed_attempts)/total_attempts)
print(f'path: {ppo_path} done')