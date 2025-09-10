from ppoenv_guide import BipedEnv
import time
import os
import numpy as np
from stable_baselines3 import PPO, SAC
import time
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
import plot_utils

#INSTEAD OF SUCCES RATE ADD HOW MANY METERS IT TRAVELED
def made_demo(scenario_mode=0,speed_len=10,angle_len = 45,episode_len=4,
              avg_trial_no=3,scenario_count=1,fail_threshold=1,floor_length=9.99,
              ppo_path=None,ppo_file="final_model.zip",demo_type="rotation",
              ppo_type="mlp"):
    
    speeds = np.linspace(0.2, 2.0, speed_len)
    angles = np.linspace(-15, 15, angle_len)

    noise_levels = np.arange(1,20,1)
    gammas = [0.5, 1.0, 1.5, 2.0]  # Different resolutions for the ground
    record_data = pd.DataFrame(columns=["demo type", "cmd speed", "angle", "mean speed","noise level",
                                        "resolution","success","max range","trial_no"])
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
                        # Record data
                        record_data = pd.concat([record_data, pd.DataFrame([{"demo type": demo_type, "cmd speed": desired_speed, "angle": angle,
                                            "mean speed": avg_mean_speeds,"noise level": noise_level,
                                            "resolution": gamma,"success": avg_success,"max range": max_range,
                                            "trial_no": None}])], ignore_index=True)
            exp_speeds = exp_speeds.T
            exp_success = exp_success.T
            exp_ranges = exp_ranges.T

            exp_speeds /= scenario_count
            exp_success /= scenario_count

            ### Speed Plot      --------------------------------------------------------
            S, A = np.meshgrid(speeds, noise_levels)
            x = S.ravel()
            y = A.ravel()
            cvals = exp_speeds.ravel()
            fig, ax = plt.subplots(figsize=(8, 6))

            # --- key part: compute a marker size that fills each (speed, angle) cell ---
            uniq_x = np.unique(x)
            uniq_y = np.unique(y)
            dx = np.diff(uniq_x).min()
            dy = np.diff(uniq_y).min()

            # convert cell size in data units -> points^2 for scatter 's'
            (x0, y0) = ax.transData.transform((0, 0))
            (x1, y1) = ax.transData.transform((dx, dy))
            cell_px = min(abs(x1 - x0), abs(y1 - y0))           # pixel size of the smaller step
            cell_pt = cell_px * 72.0 / fig.dpi                  # pixels -> points
            marker_area = (cell_pt * 0.98) ** 2                 # 98% of cell, fill with slight overlap

            sc = ax.scatter(
                x, y,
                c=cvals,
                cmap='Blues',
                marker='s',             # squares tile the grid
                s=marker_area,          # fills each cell
                linewidths=0,
                edgecolors='none'
            )

            # colorbar + labels
            speed_cbar = plt.colorbar(sc, ax=ax)
            speed_cbar.set_label('Mean Speed')
            ax.set_xlabel('Speed Value')
            ax.set_ylabel('Noise Level')
            ax.set_title('Avg Speed Plot')

            plt.savefig(os.path.join(ppo_path, f"demo_avg_speed_{scenario_mode}_{gamma}.png"), bbox_inches="tight")
            plt.close()

            ### Success Plot      --------------------------------------------------------
            S, A = np.meshgrid(speeds, noise_levels)
            x = S.ravel()
            y = A.ravel()
            cvals = exp_success.ravel()
            fig, ax = plt.subplots(figsize=(8, 6))

            # --- key part: compute a marker size that fills each (speed, angle) cell ---
            uniq_x = np.unique(x)
            uniq_y = np.unique(y)
            dx = np.diff(uniq_x).min()
            dy = np.diff(uniq_y).min()

            # convert cell size in data units -> points^2 for scatter 's'
            (x0, y0) = ax.transData.transform((0, 0))
            (x1, y1) = ax.transData.transform((dx, dy))
            cell_px = min(abs(x1 - x0), abs(y1 - y0))           # pixel size of the smaller step
            cell_pt = cell_px * 72.0 / fig.dpi                  # pixels -> points
            marker_area = (cell_pt * 0.98) ** 2                 # 98% of cell, fill with slight overlap

            sc = ax.scatter(
                x, y,
                c=cvals,
                cmap='Greens',
                marker='s',             # squares tile the grid
                s=marker_area,          # fills each cell
                linewidths=0,
                edgecolors='none'
            )

            # colorbar + labels
            speed_cbar = plt.colorbar(sc, ax=ax)
            speed_cbar.set_label('Success Rate')
            ax.set_xlabel('Speed Value')
            ax.set_ylabel('Noise Level')
            ax.set_title('Success Rate Plot')

            plt.savefig(os.path.join(ppo_path, f"demo_avg_success_{scenario_mode}_{gamma}.png"), bbox_inches="tight")
            plt.close()

            ### Range Plot      --------------------------------------------------------
            S, A = np.meshgrid(speeds, noise_levels)
            x = S.ravel()
            y = A.ravel()
            cvals = exp_ranges.ravel()
            fig, ax = plt.subplots(figsize=(8, 6))

            # --- key part: compute a marker size that fills each (speed, angle) cell ---
            uniq_x = np.unique(x)
            uniq_y = np.unique(y)
            dx = np.diff(uniq_x).min()
            dy = np.diff(uniq_y).min()

            # convert cell size in data units -> points^2 for scatter 's'
            (x0, y0) = ax.transData.transform((0, 0))
            (x1, y1) = ax.transData.transform((dx, dy))
            cell_px = min(abs(x1 - x0), abs(y1 - y0))           # pixel size of the smaller step
            cell_pt = cell_px * 72.0 / fig.dpi                  # pixels -> points
            marker_area = (cell_pt * 0.98) ** 2                 # 98% of cell, fill with slight overlap

            sc = ax.scatter(
                x, y,
                c=cvals,
                cmap='Oranges',
                marker='s',             # squares tile the grid
                s=marker_area,          # fills each cell
                linewidths=0,
                edgecolors='none'
            )

            # colorbar + labels
            speed_cbar = plt.colorbar(sc, ax=ax)
            speed_cbar.set_label(' Max Range Travelled (m)')
            ax.set_xlabel('Speed Value')
            ax.set_ylabel('Noise Level')
            ax.set_title('Max Range Plot')

            plt.savefig(os.path.join(ppo_path, f"demo_avg_range_{scenario_mode}_{gamma}.png"), bbox_inches="tight")
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

            for trial in range(trial_no):
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

            record_data = pd.concat([record_data, pd.DataFrame([{"demo type": demo_type, "cmd speed": desired_speed, "angle": None,
                "mean speed": actual_speeds[speed_no],"noise level": None,
                "resolution": None,"success": success,"max range": None,
                "trial_no": None}])], ignore_index=True)
            
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

                    record_data = pd.concat([record_data, pd.DataFrame([{"demo type": demo_type, "cmd speed": desired_speed, "angle": angle,
                        "mean speed": avg_mean_speeds,"noise level": None,
                        "resolution": None,"success": avg_success,"max range": None,
                        "trial_no": None}])], ignore_index=True)

        exp_speeds = exp_speeds.T
        exp_success = exp_success.T
        exp_speeds /= scenario_count
        exp_success /= scenario_count

        S, A = np.meshgrid(speeds, angles)
        x = S.ravel()
        y = A.ravel()
        cvals = exp_speeds.ravel()

        # figure/axes
        fig, ax = plt.subplots(figsize=(8, 6))

        # --- key part: compute a marker size that fills each (speed, angle) cell ---
        uniq_x = np.unique(x)
        uniq_y = np.unique(y)
        dx = np.diff(uniq_x).min()
        dy = np.diff(uniq_y).min()

        # convert cell size in data units -> points^2 for scatter 's'
        (x0, y0) = ax.transData.transform((0, 0))
        (x1, y1) = ax.transData.transform((dx, dy))
        cell_px = min(abs(x1 - x0), abs(y1 - y0))           # pixel size of the smaller step
        cell_pt = cell_px * 72.0 / fig.dpi                  # pixels -> points
        marker_area = (cell_pt * 0.98) ** 2                 # 98% of cell, fill with slight overlap

        sc = ax.scatter(
            x, y,
            c=cvals,
            cmap='Blues',
            marker='s',             # squares tile the grid
            s=marker_area,          # fills each cell
            linewidths=0,
            edgecolors='none'
        )

        # colorbar + labels
        speed_cbar = plt.colorbar(sc, ax=ax)
        speed_cbar.set_label('Mean Speed')
        ax.set_xlabel('Speed Value')
        ax.set_ylabel('Ramp Angle')
        ax.set_title('Avg Speed Plot')
        
        plt.savefig(os.path.join(ppo_path, "demo_avg_speeds.png"), bbox_inches="tight")
        plt.close()


        # # Success Plot
        S, A = np.meshgrid(speeds, angles)
        x = S.ravel()
        y = A.ravel()
        cvals = exp_success.ravel()

        # figure/axes
        fig, ax = plt.subplots(figsize=(8, 6))

        # --- key part: compute a marker size that fills each (speed, angle) cell ---
        uniq_x = np.unique(x)
        uniq_y = np.unique(y)
        dx = np.diff(uniq_x).min()
        dy = np.diff(uniq_y).min()

        # convert cell size in data units -> points^2 for scatter 's'
        (x0, y0) = ax.transData.transform((0, 0))
        (x1, y1) = ax.transData.transform((dx, dy))
        cell_px = min(abs(x1 - x0), abs(y1 - y0))           # pixel size of the smaller step
        cell_pt = cell_px * 72.0 / fig.dpi                  # pixels -> points
        marker_area = (cell_pt * 0.98) ** 2                 # 98% of cell, fill with slight overlap

        sc = ax.scatter(
            x, y,
            c=cvals,
            cmap='Greens',
            marker='s',             # squares tile the grid
            s=marker_area,          # fills each cell
            linewidths=0,
            edgecolors='none'
        )

        # colorbar + labels
        speed_cbar = plt.colorbar(sc, ax=ax)
        speed_cbar.set_label('Success Rate')
        ax.set_xlabel('Speed Value')
        ax.set_ylabel('Ramp Angle')
        ax.set_title('Avg Success Plot')

        plt.savefig(os.path.join(ppo_path, "demo_avg_success.png"), bbox_inches="tight")
        plt.close()

    record_data.to_csv(os.path.join(ppo_path, f"demo_data_{demo_type}_{ppo_type}_{scenario_mode}.csv"), index=False)

    env.close()
if __name__ == "__main__":

    # # First Vel Diff demo (Fastest to be done)
    # # Rotation demo       
    # # Noisy demo
    # # ----------------------------     DEFINE PARAMS     ----------------------------

    ppo_type = "mlp"                # "lstm" or "mlp"
    demo_type = "vel_diff"          # "rotation" or "noisy"

    ppo_path = "ppo_newreward/PPO_40"
    ppo_file = "final_model.zip"

    scenario_mode = 0
    speed_len = 21
    angle_len = 55

    episode_len = 4 # seconds
    fail_threshold = 1
    avg_trial_no= 3
    scenario_count = 1
    floor_length = 9.999  # meters
    t0 = time.time()

    # # ----------------------------     END PARAMS     ----------------------------

    made_demo(scenario_mode=scenario_mode,speed_len=speed_len,angle_len=angle_len,episode_len=episode_len,
                avg_trial_no=avg_trial_no,scenario_count=scenario_count,fail_threshold=fail_threshold,
                floor_length=floor_length,ppo_path=ppo_path,ppo_file=ppo_file,
                demo_type=demo_type,ppo_type=ppo_type)
    print("Vel diff demo done, starting rotation demo now...")
    print("Time taken for vel diff demo: {:.2f} seconds".format(time.time() - t0))

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