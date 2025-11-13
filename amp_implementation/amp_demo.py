import imageio
from amp_biped import BipedEnv
import time
import os
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from skrl.agents.torch.amp import AMP, AMP_DEFAULT_CONFIG
from amp_models import PolicyMLP, ValueMLP, DiscriminatorMLP
import torch
#INSTEAD OF SUCCES RATE ADD HOW MANY METERS IT TRAVELED

def made_demo(scenario_mode=0,speed_len=10,angle_len = 45,episode_len=4,
              avg_trial_no=3,scenario_count=1,fail_threshold=1,floor_length=9.99,
              ppo_path=None,ppo_file="final_model.zip",demo_type="rotation",
              ppo_type="mlp"):
    
    speeds = np.linspace(0.2, 2.0, speed_len)
    angles = np.linspace(-15, 15, angle_len)

    noise_levels = np.arange(1,20,1)
    gammas = [0.25, 0.5, 1.0, 2.0]  # Different resolutions for the ground
    record_data = pd.DataFrame(columns=["demo type", "cmd speed", "angle", "mean speed","noise level",
                                        "resolution","success","max range","trial_no"])

    if demo_type == "noisy":
        env = BipedEnv(demo_mode=True, demo_type=demo_type, render_mode= None)

        device = env.device
        models = {
            "policy":        PolicyMLP(observation_space=56, action_space=7, device=device),    
            "value":         ValueMLP(observation_space=56, action_space=None,device=device),
            "discriminator": DiscriminatorMLP(observation_space=300,action_space=None, device=device),  # <- was 60
        }
        agent = AMP(models=models,  # models dict
            memory=None,  # memory instance, or None if not required
            cfg=AMP_DEFAULT_CONFIG,  # configuration dict (preprocessors, learning rate schedulers, etc.)
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device)
        agent.load(os.path.join(ppo_path, ppo_file))
        agent.init()
        agent.set_running_mode("eval") 
        total_exp_no = speed_len * avg_trial_no * scenario_count * len(noise_levels)
        for gamma in gammas:
            t0 = time.time()
            exp_speeds = np.zeros((speed_len,len(noise_levels)))
            exp_ranges = np.zeros((speed_len,len(noise_levels)))
            exp_success = np.zeros((speed_len,len(noise_levels)))
            exp_steps = np.zeros((speed_len,len(noise_levels)))
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
                            episode_start = True
                            lstm_states = None
                            obs, info = env.reset(test_speed=desired_speed, test_angle=angle, demo_max_steps=max_steps,
                                                    ground_noise=noise_level, ground_resolution=gamma, 
                                                    heightfield_data=heightfield_data)  # Gym API
                            obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)
                            if (angle == 0) and (scenario == 0) and (noise_level == 1) :
                                img = env.get_image()
                                img.save(os.path.join(ppo_path, f"demo_render_{noise_level}_{gamma}.jpg"))

                            terminated = False


                            for i in range(0, int(max_steps/10)):
                                if ppo_type == "lstm":
                                    action, lstm_states = agent.act(obs, state=lstm_states, episode_start=episode_start)
                                    episode_start = False
                                else:
                                    action, log_prob, extra = agent.act(obs_t, timestep=i, timesteps=max_steps)

                                # 3) env step
                                act_np = action.squeeze(0).detach().cpu().numpy()
                                obs, reward, dones, truncated, info = env.step(act_np)

                                # 4) feed next obs back
                                obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)

                                # 5) let the agent update its internal step counters
                                #agent.post_interaction(timestep=i, timesteps=max_steps)

                                total_rew += reward            
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
                        exp_steps[speed_no, noise_level-1] += env.return_step_taken()
                        # Record data
                        record_data = pd.concat([record_data, pd.DataFrame([{"demo type": demo_type, "cmd speed": desired_speed, "angle": angle,
                                            "mean speed": avg_mean_speeds,"noise level": noise_level,
                                            "resolution": gamma,"success": avg_success,"max range": max_range,
                                            "trial_no": None,"steps taken": env.return_step_taken()}])], ignore_index=True)
            exp_speeds = exp_speeds.T
            exp_success = exp_success.T
            exp_ranges = exp_ranges.T
            exp_steps = exp_steps.T

            exp_speeds /= scenario_count
            exp_success /= scenario_count
            exp_ranges /= scenario_count
            exp_steps /= scenario_count

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

            ### Steps Taken Plot      --------------------------------------------------------
            S, A = np.meshgrid(speeds, noise_levels)
            x = S.ravel()
            y = A.ravel()
            cvals = exp_steps.ravel()
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
            speed_cbar.set_label('Steps Taken')
            ax.set_xlabel('Speed Value')
            ax.set_ylabel('Noise Level')
            ax.set_title('Steps Taken Plot')

            plt.savefig(os.path.join(ppo_path, f"demo_avg_steps_{scenario_mode}_{gamma}.png"), bbox_inches="tight")
            plt.close()

    elif demo_type == "vel_diff":
        t0 = time.time()
        env = BipedEnv(demo_mode=True, demo_type=demo_type, render_mode= "human")

        device = env.device
        models = {
            "policy":        PolicyMLP(observation_space=56, action_space=7, device=device),    
            "value":         ValueMLP(observation_space=56, action_space=None,device=device),
            "discriminator": DiscriminatorMLP(observation_space=300,action_space=None, device=device),  # <- was 60
        }
        agent = AMP(models=models,  # models dict
            memory=None,  # memory instance, or None if not required
            cfg=AMP_DEFAULT_CONFIG,  # configuration dict (preprocessors, learning rate schedulers, etc.)
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device)
        agent.load(os.path.join(ppo_path, ppo_file))
        agent.init()
        agent.set_running_mode("eval") 
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
                episode_start = True
                lstm_states = None
                obs, info = env.reset(test_speed=desired_speed, test_angle=angle, demo_max_steps=max_steps)  # Gym API
                obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)
                terminated = False

                for i in range(0, int(max_steps/10)):
                    with torch.no_grad():
                        if ppo_type == "lstm":
                            action, lstm_states = agent.act(obs, state=lstm_states, episode_start=episode_start)
                            episode_start = False
                        else:
                            action, log_prob, extra = agent.act(obs_t, timestep=i, timesteps=max_steps)
                            obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)

                    # 3) env step
                    act_np = action.squeeze(0).detach().cpu().numpy()
                    obs, reward, dones, truncated, info = env.step(act_np)
                    
                    # # 4) feed next obs back
                    obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)
                    # print(f"action: {action}, dones {dones}, obs {obs}")
                    # 5) let the agent update its internal step counters
                    #agent.post_interaction(timestep=i, timesteps=max_steps)
                    
                    total_rew += reward            
                    ext_state = env.return_external_state()

                    if dones:
                        success = False
                        failed_attempts += 1
                        terminated = True
                        break
                print("end happened")
                if success == True:
                    total_travel_dist = ext_state[1]
                    mean_speed = total_travel_dist / episode_len
                    avg_mean_speeds += mean_speed
            print("passed here")
            if failed_attempts < trial_no:
                actual_speeds[speed_no] = avg_mean_speeds / (trial_no - failed_attempts)
            else:
                speed_range = speed_range[0:speed_no]
                actual_speeds = actual_speeds[0:speed_no]
                break
            record_data = pd.concat([record_data, pd.DataFrame([{"demo type": demo_type, "cmd speed": desired_speed, "angle": None,
                "mean speed": actual_speeds[speed_no],"noise level": None,
                "resolution": None,"success": success,"max range": None,
                "trial_no": None,"steps taken": env.return_step_taken()}])], ignore_index=True)
            
        # plot the results of the demo make the x axis the speed and the y axis the angle and the color the distance and make the failures red
        plt.figure(figsize=(8, 6))
        plt.plot(speed_range, actual_speeds)
        plt.plot(speed_range, speed_range, '--', color='gray')  # Reference line y=x
        plt.title('Commanded Speed vs Actual Speed')
        plt.legend(['Actual Speed', 'Commanded Speed (y=x)'], loc='upper left')
        plt.xlabel("Commanded Speed (m/s)")
        plt.ylabel('Actual Speed (m/s)')
        plt.savefig(os.path.join(ppo_path, f"demo_vel_diff.png"))
        plt.close()

    elif demo_type == "rotation":
        t0 = time.time()

        env = BipedEnv(demo_mode=True, demo_type=demo_type, render_mode= None)

        device = env.device
        models = {
            "policy":        PolicyMLP(observation_space=56, action_space=7, device=device),    
            "value":         ValueMLP(observation_space=56, action_space=None,device=device),
            "discriminator": DiscriminatorMLP(observation_space=300,action_space=None, device=device),  # <- was 60
        }
        agent = AMP(models=models,  # models dict
            memory=None,  # memory instance, or None if not required
            cfg=AMP_DEFAULT_CONFIG,  # configuration dict (preprocessors, learning rate schedulers, etc.)
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device)
        agent.load(os.path.join(ppo_path, ppo_file))    
        agent.init()
        agent.set_running_mode("eval") 
        exp_speeds = np.zeros((speed_len, angle_len))
        exp_success = np.zeros((speed_len, angle_len))
        exp_steps = np.zeros((speed_len, angle_len))
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
                        episode_start = True
                        lstm_states = None
                        obs, info = env.reset(test_speed=desired_speed, test_angle=angle, demo_max_steps=max_steps)  # Gym API
                        obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)

                        terminated = False



                        for i in range(0, int(max_steps/10)):
                            if ppo_type == "lstm":
                                action, lstm_states = agent.act(obs, state=lstm_states, episode_start=episode_start)
                                episode_start = False
                            else:   
                                action, log_prob, extra = agent.act(obs_t, timestep=i, timesteps=max_steps)
                                obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)
                            
                            # 3) env step
                            act_np = action.squeeze(0).detach().cpu().numpy()
                            obs, reward, dones, truncated, info = env.step(act_np)

                            # 4) feed next obs back
                            obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)

                            # 5) let the agent update its internal step counters
                            #agent.post_interaction(timestep=i, timesteps=max_steps)
                            total_rew += reward            
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
                    exp_steps[speed_no, angle_no] += env.return_step_taken()

                    record_data = pd.concat([record_data, pd.DataFrame([{"demo type": demo_type, "cmd speed": desired_speed, "angle": angle,
                        "mean speed": avg_mean_speeds,"noise level": None,
                        "resolution": None,"success": avg_success,"max range": None,
                        "trial_no": None,"steps taken": env.return_step_taken()}])], ignore_index=True)

        exp_speeds = exp_speeds.T
        exp_success = exp_success.T
        exp_steps = exp_steps.T

        exp_speeds /= scenario_count
        exp_success /= scenario_count
        exp_steps /= scenario_count

        ## Speed Plot
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

        ## Steps Taken Plot
        S, A = np.meshgrid(speeds, angles)
        x = S.ravel()
        y = A.ravel()
        cvals = exp_steps.ravel()

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
        speed_cbar.set_label('Steps Taken')
        ax.set_xlabel('Speed Value')
        ax.set_ylabel('Ramp Angle')
        ax.set_title('Avg Steps Taken Plot')

        plt.savefig(os.path.join(ppo_path, "demo_avg_steps_rotation.png"), bbox_inches="tight")
        plt.close()

    elif demo_type == "track":

        episode_len = 80 # seconds total of 8000 steps

        t0 = time.time()
        env = BipedEnv(demo_mode=True, demo_type=demo_type, render_mode= None)

        device = env.device
        models = {
            "policy":        PolicyMLP(observation_space=56, action_space=7, device=device),    
            "value":         ValueMLP(observation_space=56, action_space=None,device=device),
            "discriminator": DiscriminatorMLP(observation_space=300,action_space=None, device=device),  # <- was 60
        }
        agent = AMP(models=models,  # models dict
            memory=None,  # memory instance, or None if not required
            cfg=AMP_DEFAULT_CONFIG,  # configuration dict (preprocessors, learning rate schedulers, etc.)
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device)
        agent.load(os.path.join(ppo_path, ppo_file))    
        agent.init()
        agent.set_running_mode("eval") 
        speed_range_forward = np.linspace(0.1, 2.1, episode_len)
        speed_range_backward = speed_range_forward[::-1]
        speed_range = np.concatenate((speed_range_forward, speed_range_backward), axis=0)
        actual_speeds = np.zeros((len(speed_range)))

        desired_speed = speed_range[0]
        angle = 0.0

        total_rew = 0
        success = True

        dt = 1e-3 #default of pybullet
        total_rew = 0
        desired_speeds = []
        actual_speeds = []
        current_time = []
        frames = []
        change_interval = episode_len * 100 / len(speed_range)  # Change speed every 5 seconds
        max_steps = int(episode_len*(1/dt))
        episode_start = True
        lstm_states = None
        obs, info = env.reset(test_speed=desired_speed, test_angle=angle, demo_max_steps=max_steps)  # Gym API
        obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)
        terminated = False

        previous_place = 0.0

        for i in range(0, int(max_steps/10)):
            if ppo_type == "lstm":
                action, lstm_states = agent.act(obs, state=lstm_states, episode_start=episode_start)
                episode_start = False
            else:
                action, log_prob, extra = agent.act(obs_t, timestep=i, timesteps=max_steps)
                obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)

            # 3) env step
            act_np = action.squeeze(0).detach().cpu().numpy()
            obs, reward, dones, truncated, info = env.step(act_np)
            # 4) feed next obs back
            obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)

            # 5) let the agent update its internal step counters
            #agent.post_interaction(timestep=i, timesteps=max_steps)

            if i % change_interval == 0:
                print(i/change_interval)
                if i != 0:
                    desired_speeds.append(desired_speed)
                    actual_speeds.append((env.return_external_state()[1]-previous_place)/(change_interval*0.01))
                    current_time.append(i*0.01)

                if i == 0:
                    desired_speeds.append(0.0)
                    actual_speeds.append(0.0)
                    current_time.append(0.0)

                previous_place = env.return_external_state()[1]
                desired_speed = speed_range[int(i/change_interval)]
                env.change_ref_speed(desired_speed)

            if i % 10 == 0:
                speed_text = f"Desired Speed: {desired_speed:.2f} m/s"
                img = env.get_follow_camera_image(overlay_text=speed_text)
                frames.append(np.array(img))
            total_rew += reward            
            ext_state = env.return_external_state()

            if dones:
                print("Fallen!")
                break

        plt.figure(figsize=(8, 6))
        plt.plot(current_time, desired_speeds, label='Desired Speed')
        plt.plot(current_time, actual_speeds, label='Actual Speed')
        plt.title('Desired vs Actual Speed over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Speed (m/s)')
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(ppo_path, f"demo_velocity_track.png"))
        plt.close()
        imageio.mimsave("biped_follow_mlp.mp4", frames, fps=30)


    elif demo_type == "joint_comparison":
        t0 = time.time()
        env = BipedEnv(demo_mode=True, demo_type=demo_type, render_mode= None)
        device = env.device
        models = {
            "policy":        PolicyMLP(observation_space=56, action_space=7, device=device),    
            "value":         ValueMLP(observation_space=56, action_space=None,device=device),
            "discriminator": DiscriminatorMLP(observation_space=300,action_space=None, device=device),  # <- was 60
        }
        agent = AMP(models=models,  # models dict
            memory=None,  # memory instance, or None if not required
            cfg=AMP_DEFAULT_CONFIG,  # configuration dict (preprocessors, learning rate schedulers, etc.)
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device)
        agent.load(os.path.join(ppo_path, ppo_file))    
        agent.init()
        agent.set_running_mode("eval") 

        desired_speed = 1.5
        angle = 0.0

        total_rew = 0
        success = True

        dt = 1e-3 #default of pybullet
        total_rew = 0
        
        max_steps = int(3*(1/dt))
        episode_start = True
        lstm_states = None
        obs, info = env.reset(test_speed=desired_speed, test_angle=angle, demo_max_steps=max_steps)  # Gym API
        obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)
        terminated = False

        rhip_posses = []
        rknee_posses = []
        rankle_posses = []
        lhip_posses = []
        lknee_posses = []
        lankle_posses = []
        times = []

        for i in range(0, int(max_steps/10)):
            if ppo_type == "lstm":
                action, lstm_states = agent.act(obs, state=lstm_states, episode_start=episode_start)
                episode_start = False
            else:
                action, log_prob, extra = agent.act(obs_t, timestep=i, timesteps=max_steps)
                obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)

            # 3) env step
            act_np = action.squeeze(0).detach().cpu().numpy()
            obs, reward, dones, truncated, info = env.step(act_np)

            # 4) feed next obs back
            obs_t = torch.as_tensor(obs, device=device).unsqueeze(0)

            # 5) let the agent update its internal step counters
            #agent.post_interaction(timestep=i, timesteps=max_steps)
            total_rew += reward       
            ext_state = env.return_external_state()

            if dones:
                success = False
                failed_attempts += 1
                terminated = True
                break
            rhip_posses.append(obs[7])
            rknee_posses.append(obs[8])
            rankle_posses.append(obs[9])
            lhip_posses.append(obs[10])
            lknee_posses.append(obs[11])
            lankle_posses.append(obs[12])
            times.append(i*0.01)

        record_data = pd.concat([record_data, pd.DataFrame([{"demo type": demo_type, "cmd speed": 1.5, "angle": None,
            "mean speed": None,"noise level": None,
            "resolution": None,"success": success,"max range": None,
            "trial_no": None,"steps taken": env.return_step_taken()}])], ignore_index=True)
            
        # plot the results of the demo make the x axis the speed and the y axis the angle and the color the distance and make the failures red
        plt.figure(figsize=(8, 6))
        plt.plot(times, rhip_posses, label='Right Hip')
        plt.plot(times, rknee_posses, label='Right Knee')
        plt.plot(times, rankle_posses, label='Right Ankle')
        plt.plot(times, lhip_posses, label='Left Hip')
        plt.plot(times, lknee_posses, label='Left Knee')
        plt.plot(times, lankle_posses, label='Left Ankle')
        plt.title('Joint Positions over Time')
        plt.legend(loc='upper left')
        plt.xlabel("Time (s)")
        plt.ylabel('Joint Position (rad)')
        plt.savefig(os.path.join(ppo_path, f"demo_joint_positions.png"))
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
    ppo_path = "/home/baran/Bipedal-imitation-rl"
    ppo_file = "amp_runs_15m/amp_trial3/checkpoints/agent_3000000.pt"

    scenario_mode = 0
    speed_len = 21
    angle_len = 31

    episode_len = 4 # seconds
    fail_threshold = 1
    avg_trial_no= 3
    scenario_count = 1
    floor_length = 9.9  # meters
    t0 = time.time()

    # # # ----------------------------     END PARAMS     ----------------------------

    made_demo(scenario_mode=scenario_mode,speed_len=speed_len,angle_len=angle_len,episode_len=episode_len,
                avg_trial_no=avg_trial_no,scenario_count=scenario_count,fail_threshold=fail_threshold,
                floor_length=floor_length,ppo_path=ppo_path,ppo_file=ppo_file,
                demo_type=demo_type,ppo_type=ppo_type)
    print("Vel diff demo done, starting rotation demo now...")
    print("Time taken for vel diff demo: {:.2f} seconds".format(time.time() - t0))

    # t0 = time.time()
    # demo_type = "rotation"
    # made_demo(scenario_mode=scenario_mode,speed_len=speed_len,angle_len=angle_len,episode_len=episode_len,
    #             avg_trial_no=avg_trial_no,scenario_count=scenario_count,fail_threshold=fail_threshold,
    #             floor_length=floor_length,ppo_path=ppo_path,ppo_file=ppo_file,
    #             demo_type=demo_type,ppo_type=ppo_type)
    
    # print("Rotation demo done, starting noisy demo now...")
    # print("Time taken for rotation demo: {:.2f} seconds".format(time.time() - t0))

    # t0 = time.time()
    # demo_type = "noisy"

    # made_demo(scenario_mode=scenario_mode,speed_len=speed_len,angle_len=angle_len,episode_len=episode_len,
    #             avg_trial_no=avg_trial_no,scenario_count=scenario_count,fail_threshold=fail_threshold,
    #             floor_length=floor_length,ppo_path=ppo_path,ppo_file=ppo_file,
    #             demo_type=demo_type,ppo_type=ppo_type)
    # print(f"Noisy demo {scenario_mode} done!")
    # print("Time taken for noisy demo: {:.2f} seconds".format(time.time() - t0))


    # t0 = time.time()
    # scenario_mode = 1
    # demo_type = "noisy"

    # made_demo(scenario_mode=scenario_mode,speed_len=speed_len,angle_len=angle_len,episode_len=episode_len,
    #             avg_trial_no=avg_trial_no,scenario_count=scenario_count,fail_threshold=fail_threshold,
    #             floor_length=floor_length,ppo_path=ppo_path,ppo_file=ppo_file,
    #             demo_type=demo_type,ppo_type=ppo_type)
    # print(f"Noisy demo {scenario_mode} done!")
    # print("Time taken for noisy demo: {:.2f} seconds".format(time.time() - t0))

    # t0 = time.time()
    # demo_type = "track"
    # made_demo(scenario_mode=scenario_mode,speed_len=speed_len,angle_len=angle_len,episode_len=episode_len,
    #             avg_trial_no=avg_trial_no,scenario_count=scenario_count,fail_threshold=fail_threshold,
    #             floor_length=floor_length,ppo_path=ppo_path,ppo_file=ppo_file,
    #             demo_type=demo_type,ppo_type=ppo_type)
    # print("Track demo done!")
    # print("Time taken for track demo: {:.2f} seconds".format(time.time() - t0))