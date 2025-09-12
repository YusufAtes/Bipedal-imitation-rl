from ppoenv_guide import BipedEnv
import time
import os
import numpy as np
from stable_baselines3 import PPO, SAC
import time
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting


#INSTEAD OF SUCCES RATE ADD HOW MANY METERS IT TRAVELED
def made_demo(scenario_mode=0,speed_len=10,angle_len = 45,episode_len=4,
              avg_trial_no=3,scenario_count=1,fail_threshold=1,floor_length=9.99,
              ppo_path=None,ppo_file="final_model.zip",demo_type="rotation",
              ppo_type="mlp"):
    
    speeds = np.linspace(0.2, 2.1, speed_len)
    angles = np.linspace(-12, 12, angle_len)

    noise_levels = np.arange(1,20,1)
    gammas = [0.5, 1.0, 1.5, 2.0]  # Different resolutions for the ground

    if demo_type == "noisy":
        env = BipedEnv(demo_mode=True, demo_type=demo_type, render_mode= None)
        model = PPO.load(os.path.join(ppo_path,ppo_file),device='cpu',deterministic=True)
        model.set_env(env) 
        total_exp_no = speed_len * avg_trial_no * scenario_count * len(noise_levels)
        for gamma in gammas:
            t0 = time.time()
            exp_speeds = np.zeros((speed_len,len(noise_levels)))
            exp_ranges = np.zeros((speed_len,len(noise_levels)))
            exp_success = np.zeros((speed_len,len(noise_levels)))
            total_experiences = 0

            for noise_level in noise_levels:
                # Scenario count is always 1 (this for loop is just for future use)
                for scenario in range(scenario_count):
                    # Generate heightfield data with noise in the range [-ground_noise, ground_noise]
                    heightfield_data = np.load(f"noise_planes/plane_{gamma}_{scenario_mode}.npy")
                    heightfield_data = heightfield_data * noise_level
                    
                    for speed_no in range(len(speeds)):
                        desired_speed = speeds[speed_no]
                        angle = 0.0
                        avg_mean_speeds = 0
                        failed_attempts = 0
                        max_range = 0
                        success_counter = 0
                        for trial in range(avg_trial_no):
                            total_experiences += 1
                            total_rew = 0
                            success = True

                            dt = 1e-3 #default of pybullet
                            total_rew = 0
                            
                            max_steps = int(20 *(1/dt)) # max 20 seconds to cross the floor
                            obs, info = env.reset(test_speed=desired_speed, test_angle=angle, demo_max_steps=max_steps,
                                                    ground_noise=noise_level, ground_resolution=gamma, 
                                                    heightfield_data=heightfield_data)  # Gym API

                            if (angle == 0) and (scenario == 0) and (noise_level == 1) :
                                img = env.get_image()
                                img.save(os.path.join(ppo_path, f"demo_render_{noise_level}_{gamma}.jpg"))

                            terminated = False
                            episode_start = True
                            lstm_states = None

                            for i in range(0, int(max_steps/10)):
                                if ppo_type == "lstm":
                                    action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_start)
                                    episode_start = False
                                else:
                                    action, _states = model.predict(obs)
                                obs, rewards, dones, truncated, info = env.step(action)
                                total_rew += rewards            
                                ext_state = env.return_external_state()

                                if dones:   #If done end the episode
                                    success = False
                                    mean_speed = 0.0
                                    failed_attempts += 1
                                    terminated = True
                                    break
        
                                if ext_state[1] > floor_length:   #If it has crossed the floor end the episode
                                    success = True
                                    break

                            total_travel_dist = ext_state[1]

                            if total_travel_dist > max_range:       # Keep the max range it has traveled
                                max_range = total_travel_dist

                            mean_speed = total_travel_dist / ((i+1)*dt*10)    # Calculate mean speed
                            avg_mean_speeds += mean_speed                   # Add to the average mean speed

                            if terminated == False:                 # If robot is not fallen but has not crossed the floor
                                if mean_speed < 0.0:
                                    mean_speed = 0
                                    success = False
                                    failed_attempts += 1

                            if success == True:                            # Keep track of success count
                                success_counter += 1

                            if total_experiences % int(total_exp_no/5) == 0:
                                print(f'{total_experiences / total_exp_no:.2%}% of {total_exp_no} experiences done for Resolution: {gamma}, Amplitude: {noise_level}')
                                print(f"Time taken for this scenario: {time.time() - t0:.2f} seconds")
                                t0 = time.time()
                                

                        avg_mean_speeds /= avg_trial_no
                        avg_success = success_counter / avg_trial_no

                        exp_speeds[speed_no, noise_level-1] += avg_mean_speeds
                        exp_success[speed_no, noise_level-1] += avg_success
                        exp_ranges[speed_no, noise_level-1] += max_range

            exp_speeds = exp_speeds.T      # swap axes so it matches meshgrid
            exp_success = exp_success.T
            exp_ranges = exp_ranges.T

            exp_speeds /= scenario_count
            exp_success /= scenario_count

            S, A = np.meshgrid(speeds, noise_levels)

            exp_speeds = exp_speeds.flatten()
            exp_success = exp_success.flatten()
            exp_ranges = exp_ranges.flatten()

            # Mean Speed Plot
            plt.figure(figsize=(8, 6))
            mean_speed_plot = plt.scatter(S.ravel(), A.ravel(),
                                        c=exp_speeds.ravel(), cmap='Blues', edgecolors='k',marker='o',s=50)
            # Add colorbar
            speed_cbar = plt.colorbar(mean_speed_plot)
            speed_cbar.set_label('Mean Speed')
            plt.xlabel('Speed Value')
            plt.ylabel('Noise Level')
            plt.title(f'Speed versus Noise Level')
            plt.savefig(os.path.join(ppo_path, f"demo_avg_speeds_{scenario_mode}_{gamma}.png"))
            plt.close()

            # Success Plot
            plt.figure(figsize=(8, 6))
            avg_success_plot = plt.scatter(S.ravel(), A.ravel(), 
                                        c=exp_success.ravel(), cmap='Greens', edgecolors='k',marker='o',s=50)
            plt.xlabel('Speed Value')
            plt.ylabel('Noise Level')
            plt.title(f'Experiment Success without Fall')
            # Add colorbar
            success_cbar = plt.colorbar(avg_success_plot)
            success_cbar.set_label('Success Rate')
            plt.savefig(os.path.join(ppo_path, f"demo_avg_success_{scenario_mode}_{gamma}.png"))
            plt.close()

            # Range Plot
            plt.figure(figsize=(8, 6))
            avg_range_plot = plt.scatter(S.ravel(), A.ravel(), 
                                        c=exp_ranges.ravel(), cmap='Oranges', edgecolors='k',marker='o',s=50)
            plt.xlabel('Speed Value')
            plt.ylabel('Noise Level')
            plt.title(f'Total Distance Covered')
            # Add colorbar
            range_cbar = plt.colorbar(avg_range_plot)
            range_cbar.set_label('Max Range (m)')
            plt.savefig(os.path.join(ppo_path, f"demo_avg_range_{scenario_mode}_{gamma}.png"))
            plt.close()

    elif demo_type == "vel_diff":
        t0 = time.time()
        env = BipedEnv(demo_mode=True, demo_type=demo_type, render_mode= None)
        model = PPO.load(os.path.join(ppo_path,ppo_file),device='cpu',deterministic=True)
        model.set_env(env) 
        speed_range = np.linspace(0.1, 2.1, 41)
        actual_speeds = np.zeros((len(speed_range)))

        trial_no = 5
        for speed_no in range(len(speed_range)):
            failed_attempts = 0
            avg_mean_speeds = 0

            for trials in range(trial_no):
                desired_speed = speed_range[speed_no]
                angle = 0.0

                total_rew = 0
                success = True

                dt = 1e-3 #default of pybullet
                total_rew = 0
                
                max_steps = int(episode_len*(1/dt))
                obs, info = env.reset(test_speed=desired_speed, test_angle=angle, demo_max_steps=max_steps)  # Gym API
                terminated = False

                episode_start = True
                lstm_states = None

                for i in range(0, int(max_steps/10)):
                    if ppo_type == "lstm":
                        action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_start)
                        episode_start = False
                    else:
                        action, _states = model.predict(obs)
                    obs, rewards, dones, truncated, info = env.step(action)
                    total_rew += rewards            
                    ext_state = env.return_external_state()

                    if dones:
                        success = False
                        failed_attempts += 1
                        terminated = True
                        break

                if success == True:
                    total_travel_dist = ext_state[1]
                    mean_speed = total_travel_dist / episode_len
                    avg_mean_speeds += mean_speed

            actual_speeds[speed_no] = avg_mean_speeds / (trial_no - failed_attempts)

        # plot the results of the demo make the x axis the speed and the y axis the angle and the color the distance and make the failures red
        plt.figure(figsize=(8, 6))
        plt.plot(speed_range, actual_speeds)
        plt.xlabel("Commanded Speed (m/s)")
        plt.ylabel('Actual Speed (m/s)')
        plt.savefig(os.path.join(ppo_path, f"demo_vel_diff.png"))
        plt.close()

    elif demo_type == "rotation":
        t0 = time.time()

        env = BipedEnv(demo_mode=True, demo_type=demo_type, render_mode= None)
        model = PPO.load(os.path.join(ppo_path,ppo_file),device='cpu',deterministic=True)
        model.set_env(env) 

        exp_speeds = np.zeros((speed_len, angle_len))
        exp_success = np.zeros((speed_len, angle_len))
        total_experiences = 0
        total_exp_no = speed_len * angle_len * avg_trial_no * scenario_count

        for scenario in range(scenario_count):

            for angle_no in range(len(angles)):

                for speed_no in range(len(speeds)):
                    desired_speed = speeds[speed_no]
                    angle = angles[angle_no]
                    avg_mean_speeds = 0
                    failed_attempts = 0

                    for trial in range(avg_trial_no):
                        total_experiences += 1
                        total_rew = 0
                        success = True

                        dt = 1e-3 #default of pybullet
                        total_rew = 0
                        
                        max_steps = int(episode_len*(1/dt))
                        obs, info = env.reset(test_speed=desired_speed, test_angle=angle, demo_max_steps=max_steps)  # Gym API
                        terminated = False

                        episode_start = True
                        lstm_states = None

                        for i in range(0, int(max_steps/10)):
                            if ppo_type == "lstm":
                                action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_start)
                                episode_start = False
                            else:   
                                action, _states = model.predict(obs)
                            obs, rewards, dones, truncated, info = env.step(action)
                            total_rew += rewards            
                            ext_state = env.return_external_state()

                            if dones:
                                success = False
                                mean_speed = 0.0
                                failed_attempts += 1
                                terminated = True
                                break
                        total_travel_dist = ext_state[1]
                        mean_speed = total_travel_dist / episode_len

                        if total_experiences % int(total_exp_no/5) == 0:
                            print(f'{total_experiences / total_exp_no:.2%}% of {total_exp_no} experiences done')
                            print(f"Time taken for this scenario: {time.time() - t0:.2f} seconds")
                            t0 = time.time()
                            
                        if terminated == False:
                            if mean_speed < 0.0:
                                mean_speed = 0
                                success = False
                                failed_attempts += 1

                        if success != False:
                            avg_mean_speeds += mean_speed

                        if failed_attempts > fail_threshold:
                            break

                    if failed_attempts > fail_threshold:
                        avg_mean_speeds = 0
                        avg_success = 0
                    else:
                        avg_mean_speeds /= (avg_trial_no - failed_attempts)
                        avg_success = 1

                    exp_speeds[speed_no, angle_no] += avg_mean_speeds
                    exp_success[speed_no, angle_no] += avg_success

        exp_speeds = exp_speeds.T      # swap axes so it matches meshgrid
        exp_success = exp_success.T

        exp_speeds /= scenario_count
        exp_success /= scenario_count
        
        S, A = np.meshgrid(speeds, angles)
        exp_speeds = exp_speeds.flatten()
        exp_success = exp_success.flatten()

        # plot the results of the demo make the x axis the speed and the y axis the angle and the color the distance and make the failures red
        plt.figure(figsize=(8, 6))
        mean_speed_plot = plt.scatter(S.ravel(), A.ravel(),
                            c=exp_speeds.ravel(), cmap='Blues', edgecolors='k',marker='o',s=50)
        # Add colorbar
        speed_cbar = plt.colorbar(mean_speed_plot)
        speed_cbar.set_label('Mean Speed')
        plt.xlabel('Speed Value')
        plt.ylabel('Ramp Angle')
        plt.title(f'Avg Speed Plot')
        plt.savefig(os.path.join(ppo_path, f"demo_avg_speeds.png"))
        plt.close()


        plt.figure(figsize=(8, 6))
        avg_success_plot = plt.scatter(S.ravel(), A.ravel(), c=exp_success.ravel(), cmap='Greens', 
                                       edgecolors='k',marker='o',s=50)
        plt.xlabel('Speed Value')
        plt.ylabel('Ramp Angle')
        plt.title(f'Avg Success Plot')
        # Add colorbar
        success_cbar = plt.colorbar(avg_success_plot)
        success_cbar.set_label('Success Rate')
        plt.savefig(os.path.join(ppo_path, f"demo_avg_success.png"))
        plt.close()

    env.close()
if __name__ == "__main__":

    # # First Vel Diff demo (Fastest to be done)
    # # Rotation demo       
    # # Noisy demo
    # # ----------------------------     DEFINE PARAMS     ----------------------------

    ppo_type = "lstm"                # "lstm" or "mlp"
    demo_type = "vel_diff"          # "rotation" or "noisy"

    ppo_path = "ppo_lstm/RecurrentPPO_3"
    ppo_file = "final_model.zip"

    scenario_mode = 0
    speed_len = 18
    angle_len = 51

    episode_len = 4 # seconds
    fail_threshold = 1
    avg_trial_no= 3
    scenario_count = 1
    floor_length = 9.999  # meters
    t0 = time.time()

    # # ----------------------------     END PARAMS     ----------------------------

    # made_demo(scenario_mode=scenario_mode,speed_len=speed_len,angle_len=angle_len,episode_len=episode_len,
    #             avg_trial_no=avg_trial_no,scenario_count=scenario_count,fail_threshold=fail_threshold,
    #             floor_length=floor_length,ppo_path=ppo_path,ppo_file=ppo_file,
    #             demo_type=demo_type,ppo_type=ppo_type)
    # print("Vel diff demo done, starting rotation demo now...")
    # print("Time taken for vel diff demo: {:.2f} seconds".format(time.time() - t0))

    t0 = time.time()
    demo_type = "rotation"
    made_demo(scenario_mode=scenario_mode,speed_len=speed_len,angle_len=angle_len,episode_len=episode_len,
                avg_trial_no=avg_trial_no,scenario_count=scenario_count,fail_threshold=fail_threshold,
                floor_length=floor_length,ppo_path=ppo_path,ppo_file=ppo_file,
                demo_type=demo_type,ppo_type=ppo_type)
    
    print("Rotation demo done, starting noisy demo now...")
    print("Time taken for rotation demo: {:.2f} seconds".format(time.time() - t0))

    t0 = time.time()
    demo_type = "noisy"

    made_demo(scenario_mode=scenario_mode,speed_len=speed_len,angle_len=angle_len,episode_len=episode_len,
                avg_trial_no=avg_trial_no,scenario_count=scenario_count,fail_threshold=fail_threshold,
                floor_length=floor_length,ppo_path=ppo_path,ppo_file=ppo_file,
                demo_type=demo_type,ppo_type=ppo_type)
    print(f"Noisy demo {scenario_mode} done!")
    print("Time taken for noisy demo: {:.2f} seconds".format(time.time() - t0))


    t0 = time.time()
    scenario_mode = 1
    demo_type = "noisy"

    made_demo(scenario_mode=scenario_mode,speed_len=speed_len,angle_len=angle_len,episode_len=episode_len,
                avg_trial_no=avg_trial_no,scenario_count=scenario_count,fail_threshold=fail_threshold,
                floor_length=floor_length,ppo_path=ppo_path,ppo_file=ppo_file,
                demo_type=demo_type,ppo_type=ppo_type)
    print(f"Noisy demo {scenario_mode} done!")
    print("Time taken for noisy demo: {:.2f} seconds".format(time.time() - t0))