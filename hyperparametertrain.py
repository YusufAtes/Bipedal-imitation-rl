# from pybullet_bipedenv_torquecontrolled import BipedEnv
# from pybullet_bipedenv_poscontrolled import POS_Biped
# from pybullett_bipedenv_trcontrol_ankle import BipedEnv

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
    BaseCallback,
)
from pybullett_bipedenv_trcontrol_ankle import BipedEnv
import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3 import PPO
import time
from typing import Callable
import re


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
        # Optionally summarize results at the end of training
        print("Training finished. Total episodes:", len(self.episode_rewards))
        print("Episode rewards:", self.episode_rewards)

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
            model_path = f"{self.save_path}/model_checkpoint_{self.init_no}"+checkpoint_name
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Model saved at step {self.n_calls} to {model_path}")
                print(f"Time taken for this checkpoint: {time.time() - t0:.2f} seconds")
                time.sleep(15)

        return True
    
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Returns a function that computes
    `progress_remaining * initial_value`.
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

namelist = []

target_kl_list = [0.1, 0.2,None]
clip_range_list = [0.15, linear_schedule(0.2)]
learning_rate_list = [3e-4, linear_schedule(3e-4),1e-4]
batch_control_list = [[4096, 512, 6], [2048, 256, 8], [2048, 64, 10]]
ent_coef_list = [1e-3, 5e-4]

hyperparameters = {
    "target_kl": target_kl_list,
    "clip_range": clip_range_list,
    "learning_rate": learning_rate_list,
    "batch_control": batch_control_list,
    "ent_coef": ent_coef_list,
}

from itertools import product
combinations = list(product(*hyperparameters.values()))
namelist = []
for i in range(len(combinations)):
    namelist.append("hpt_trials/ppo_256_256" + "_".join([str(x) for x in combinations[i]]))


# ========== MAIN TRAINING LOOP ==========
for k in range(len(namelist)):
    SAVE_DIR = namelist[k]
    checkpoint_name = SAVE_DIR+".zip"
    rewar_Logger_name = SAVE_DIR+".csv"
    TOTAL_TIMESTEPS = 10_000
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 1) ENVIRONMENT
    train_env = BipedEnv(render_mode=None)

    # 2) CALLBACKS
    # a) checkpoint every 500k steps
    checkpoint_cb = CustomCheckpointCallback(
        save_freq=200_000,
        save_path=SAVE_DIR,
        verbose=1,
    )
    # b) logging raw episode rewards to CSV
    reward_logger = RewardLoggerCallback(log_file=os.path.join(SAVE_DIR, "rewards.csv"))


    callback_list = [checkpoint_cb, reward_logger]

    # 3) MODEL CONFIGURATION
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        # orthogonal init & layer norm could go here if desired
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        device="cpu",
        tensorboard_log=SAVE_DIR,

        # --- General HYPERPARAMS ---        
        # --- KL & clipping ---

        target_kl=combinations[k][0],                           # hard KL ceiling
        clip_range=combinations[k][1],        # decay from 0.15 → 0
        clip_range_vf=None,                      # keep value clipping default

        # --- Learning rate schedule ---
        learning_rate=combinations[k][2],

        # --- Exploration & entropy ---
        ent_coef=combinations[k][4],          # high early entropy → 0

        # --- Batch / epoch control ---
        n_steps=combinations[k][3][0],                             # longer rollout for smoother adv
        batch_size=combinations[k][3][1],                           # big minibatches
        n_epochs=combinations[k][3][2],                               # fewer passes per rollout

        # # --- Gradient clipping ---
        # max_grad_norm=0.3,

        policy_kwargs=policy_kwargs
    )

    # 4) TRAIN
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback_list,
    )

    # 5) SAVE final model & stats
    model.save(os.path.join(SAVE_DIR, "final_model"))
    print("Training complete. Models and logs are in:", SAVE_DIR)

    import pandas as pd
    import matplotlib.pyplot as plt
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
            obs, info = train_env.reset(test_speed=test_speed, test_angle= test_angle,demo_max_steps = max_steps)
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

                obs, rewards, dones, truncated, info = train_env.step(action)

                total_rew += rewards            
                ext_state = train_env.return_external_state()
                if dones:
                    succes = False
                    mean_speed = 0
                    failed_attempts += 1
                    terminated = True
                    break
            
            #print('total reward: ', total_rew)
            total_travel_dist = ext_state[1]
            mean_speed = total_travel_dist / episode_len

            if terminated == False:
                if mean_speed < 0.1:
                    mean_speed = 0
                elif (mean_speed > 3):
                    mean_speed = 0
                    failed_attempts += 1
                    succes = False

            test_angle_deg = 180*test_angle/3.14159
            
            #add to the dataframe
            demo_cases = pd.concat([demo_cases, pd.DataFrame([[test_speed, test_angle_deg, mean_speed, 
                                                            succes, total_rew]], columns=['Speed', 'Angle', 'Distance', 
                                                                                            'Success','Reward'])], ignore_index=True)
            
    demo_cases.to_csv(os.path.join(SAVE_DIR, "demo_results.csv"), index=False)

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
    plt.savefig(os.path.join(SAVE_DIR, "demo_results.png"))
    plt.close()
    print("Done Combination: ", namelist[k])
    with open(os.path.join(SAVE_DIR, "success_rate.txt"), 'w') as f:
        f.write(f"Failed attempts: {failed_attempts}\n")
        f.write(f"Total attempts: {total_attempts}\n")
        f.write(f"Success rate: {(total_attempts - failed_attempts)/total_attempts}\n")
        f.write(f"Max speed: {max_speed}\n")
        f.write(f"Demo cases: \n")
        f.write(demo_cases.to_string())
    train_env.close()
    del model
    time.sleep(10)
