# from pybullet_bipedenv_torquecontrolled import BipedEnv
from pybullet_bipedenv_poscontrolled import POS_Biped
from pybullett_bipedenv_trcontrol_ankle import BipedEnv
import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3 import PPO, SAC
import time
from typing import Callable
from stable_baselines3.common.env_util import make_vec_env
import pandas as pd
import matplotlib.pyplot as plt
t0 = time.time()
class RewardLoggerCallback(BaseCallback):
    def __init__(self, log_file: str, verbose: int = 0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.log_file = log_file
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.current_step = 0  # Track the number of steps in the current episode

        # Create the log file if it doesn't exist
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write("Episode,Total Reward,Termination Step\n")

    def _on_step(self) -> bool:
        # Check if the episode has ended
        dones = self.locals["dones"]
        rewards = self.locals["rewards"]

        # Accumulate rewards for the current episode
        self.current_episode_reward += rewards[0]
        self.current_step += 1  # Increment step count for the current episode

        # If the episode is done, log the reward and termination step
        if dones[0]:
            self.episode_rewards.append(self.current_episode_reward)
            with open(self.log_file, 'a') as f:
                f.write(f"{len(self.episode_rewards)},{self.current_episode_reward},{self.current_step}\n")
            
            # Reset counters for the next episode
            self.current_episode_reward = 0
            self.current_step = 0

        return True
    def _on_training_end(self) -> None:
        pass
        # Optionally summarize results at the end of training
        # print("Training finished. Total episodes:", len(self.episode_rewards))
        # print("Episode rewards:", self.episode_rewards)

class CustomCheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path,init_no = 0, verbose=0):
        super(CustomCheckpointCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.init_no = init_no

    def _on_step(self) -> bool:
        done = False
        # Save the model every `save_freq` steps
        if self.n_calls % self.save_freq == 0:
            self.init_no +=1
            model_path = f"{self.save_path}/model_checkpoint_{self.init_no}.zip"
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Model saved at step {self.n_calls} to {model_path}")
                print(f"Time taken for this checkpoint: {time.time() - t0:.2f} seconds")
                time.sleep(15)

        return True

# Usage

ent_coefs = [0.0005,0.001,0.01]
n_steps = [1024, 2048, 4096]
clip_ranges = [0.15,0.18,0.2]
batch_sizes = [64,32,128]
lrs = [3e-4,1e-4,3e-5]
#make a dictionary that contains all combinations of the above
hyperparameters = {
    "ent_coef": ent_coefs,
    "n_steps": n_steps,
    "clip_range": clip_ranges,
    "batch_size": batch_sizes,
    "learning_rate": lrs
}
#make a list of all combinations of the above
from itertools import product
combinations = list(product(*hyperparameters.values()))
namelist = []
for i in range(len(combinations)):
    namelist.append("hpt_trials/ppo_256_256" + "_".join([str(x) for x in combinations[i]]))

total_timesteps = 5000000

for k in range(len(namelist)):
    ppo_folder = namelist[k].split("/")[1]
    reward_Logger_name = namelist[k]+"/"+ppo_folder+".csv"
    checkpoint_name = namelist[k]+".zip"
    weight_file_name = "final_"+namelist[k]
    use_past_weights = False

    if os.path.exists(namelist[k]):
        pass
    else:
        os.makedirs(namelist[k])


    checkpoint_callback = CustomCheckpointCallback(
            save_freq =500000, save_path=namelist[k], verbose=1
        )
    reward_logger = RewardLoggerCallback(log_file=reward_Logger_name)

    callbacks = CallbackList([checkpoint_callback, reward_logger])

    env = BipedEnv(render_mode=None)
    env.reset()

    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
    print("Starting training")

    model = PPO(
        "MlpPolicy",
        policy_kwargs=policy_kwargs,
        device="cpu",
        env=env,
        tensorboard_log="./"+namelist[k] +"/",
        ent_coef= combinations[k][0],
        n_steps = combinations[k][1],
        batch_size=combinations[k][3],
        learning_rate=combinations[k][4],
        clip_range=combinations[k][2])

    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    ppo_path = namelist[k]
    speeds = np.linspace(0.2, 3, 7)
    angles = np.linspace(-20, 20, 20)

    covered_distance = np.zeros(len(speeds))
    total_attempts = 0
    failed_attempts = 0
    start_pos = 0
    demo_cases = pd.DataFrame(columns=['Speed', 'Angle', 'Distance', 'Success','Reward'])
    max_speed = 0
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
                if mean_speed > max_speed:
                    max_speed = mean_speed
                    print(f'New max speed: {max_speed} Speed Value: {test_speed} Angle: {test_angle*180/3.14159}')

            test_angle_deg = 180*test_angle/3.14159
            #print the results only two decimal points
            
            # print(f"Speed: {test_speed:.2f}, Angle: {test_angle_deg:.2f}, Distance: {distance:.2f}, Success: {succes}, Reward: {total_rew}")
            
            #add to the dataframe
            demo_cases = pd.concat([demo_cases, pd.DataFrame([[test_speed, test_angle_deg, mean_speed, 
                                                            succes, total_rew]], columns=['Speed', 'Angle', 'Distance', 
                                                                                            'Success','Reward'])], ignore_index=True)
            
    demo_cases.to_csv(os.path.join(ppo_path, "demo_results.csv"), index=False)

    # plot the results of the demo make the x axis the speed and the y axis the angle and the color the distance and make the failures red
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
    print("Done Combination: ", namelist[k])
    with open(os.path.join(ppo_path, "success_rate.txt"), 'w') as f:
        f.write(f"Failed attempts: {failed_attempts}\n")
        f.write(f"Total attempts: {total_attempts}\n")
        f.write(f"Success rate: {(total_attempts - failed_attempts)/total_attempts}\n")
        f.write(f"Max speed: {max_speed}\n")
        f.write(f"Demo cases: \n")
        f.write(demo_cases.to_string())
    env.close()
    del model
    time.sleep(1)
