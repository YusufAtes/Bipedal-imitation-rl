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
ppo_path = "ppo_256_256/PPO_35"
env = BipedEnv(demo_mode=True)
ppo_file = "model_checkpoint_12ppo_256_256.zip"


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

speeds = np.linspace(0.1, 3, 9)
angles = np.linspace(-20,20,41)
noises = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
ground_noise = 0.0
covered_distance = np.zeros(len(speeds))
total_attempts = 0
failed_attempts = 0
start_pos = 0
demo_cases = pd.DataFrame(columns=['Speed Value', 'Angle', 'Ground Noise','Mean Speed','Success','Reward'])
max_speed = 0
past_rhip = []
past_lhip = []
for angle in angles:
    for k in range(len(speeds)):
        total_rew = 0
        total_attempts += 1
        succes = True
        test_speed = speeds[k]
        test_angle = angle*3.14159/180

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
        for i in range(0, int(max_steps/10)):
            action, _states = model.predict(obs)

            obs, rewards, dones, truncated, info = env.step(action)
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
        
        #print('total reward: ', total_rew)
        total_travel_dist = ext_state[1]
        mean_speed = total_travel_dist / episode_len
        #find me the sign change in the past_rhip and past_lhip
        rhip_sign_changes = count_sign_changes(past_rhip)
        lhip_sign_changes = count_sign_changes(past_lhip)
        # print('rhip sign changes: ', rhip_sign_changes)
        # print('lhip sign changes: ', lhip_sign_changes)

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
                print(f'New max speed: {max_speed} Speed Value: {test_speed} Angle: {test_angle*180/3.14159} Ground Noise: {ground_noise}')

        test_angle_deg = 180*test_angle/3.14159
        #print the results only two decimal points
        
        # print(f"Speed: {test_speed:.2f}, Angle: {test_angle_deg:.2f}, Distance: {distance:.2f}, Success: {succes}, Reward: {total_rew}")
        
        #add to the dataframe
        demo_cases = pd.concat([demo_cases, pd.DataFrame([[test_speed, test_angle_deg, ground_noise,mean_speed, 
                                                        succes, total_rew]], columns=['Speed Value', 'Angle', 'Ground Noise','Mean Speed', 
                                                                                        'Success','Reward'])], ignore_index=True)


    # plot the results of the demo make the x axis the speed and the y axis the angle and the color the distance and make the failures red
    plt.figure(figsize=(8, 6))
    curent_demo_cases = demo_cases[demo_cases['Ground Noise'] == ground_noise]
    sc = plt.scatter(curent_demo_cases['Speed Value'], curent_demo_cases['Angle'], c=curent_demo_cases['Mean Speed'], cmap='Blues', edgecolors='k')
    failed_cases = curent_demo_cases[curent_demo_cases['Success'] == False]
    plt.scatter(failed_cases['Speed Value'], failed_cases['Angle'], c='red')

    # Add colorbar
    cbar = plt.colorbar(sc)
    cbar.set_label('Mean Speed')

    plt.legend(['Success', 'Failed'])

    plt.xlabel('Speed Value')
    plt.ylabel('Ramp Angle')
    plt.title(f'Demo results for {ground_noise} ground noise')
    plt.savefig(os.path.join(ppo_path, f"demo_results{ground_noise}.png"))
    plt.close()
demo_cases.to_csv(os.path.join(ppo_path, "demo_results.csv"), index=False)



# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Plot successful cases
# sc = ax.scatter(
#     demo_cases['Speed'],
#     demo_cases['Angle'],
#     demo_cases['Ground Noise'],
#     c=demo_cases['Distance'],
#     cmap='Blues',
#     edgecolors='k',
#     label='Success'
# )

# # Plot failed cases in red
# failed_cases = demo_cases[demo_cases['Success'] == False]
# ax.scatter(
#     failed_cases['Speed'],
#     failed_cases['Angle'],
#     failed_cases['Ground Noise'],
#     c='red',
#     label='Failed'
# )

# # Add colorbar for 'Distance'
# cbar = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10)
# cbar.set_label('Distance')

# # Labels and title
# ax.set_xlabel('Speed Value')
# ax.set_ylabel('Ramp Angle')
# ax.set_zlabel('Ground Noise')
# ax.set_title('Demo Results in 3D')
# ax.legend()

# # Save and close
# plt.savefig(os.path.join(ppo_path, "demo_results_3d.png"))
# plt.close()

print("Failed attempts: ", failed_attempts)
print("Total attempts: ", total_attempts)
print("Success rate: ", (total_attempts - failed_attempts)/total_attempts)
with open(os.path.join(ppo_path, "success_rate.txt"), 'w') as f:
    f.write(f"Failed attempts: {failed_attempts}\n")
    f.write(f"Total attempts: {total_attempts}\n")
    f.write(f"Success rate: {(total_attempts - failed_attempts)/total_attempts}\n")
    f.write(f"Max speed: {max_speed}\n")
    f.write(f"Demo cases: \n")
    f.write(demo_cases.to_string())
print(f'path: {ppo_path} done')