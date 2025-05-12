import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym
import os
from bipedplant_roughness import BipedPlant
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

env = BipedPlant()
policy_kwargs = dict(activation_fn=nn.ReLU,
                     net_arch=dict(pi=[256, 256]))

model = PPO("MlpPolicy", env, device="cuda", n_steps=400,policy_kwargs=policy_kwargs) 

model.learn(total_timesteps=20_000)

model.save("bipedplant_roughness_torqueout")

del model
