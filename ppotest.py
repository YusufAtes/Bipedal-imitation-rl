import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3 import PPO, SAC



demo = False
if demo:
    env = gym.make("BipedalWalker-v3", hardcore=False, render_mode="human")
    model = SAC("MlpPolicy", env, verbose=0,device='cuda')
    model.load("ppo_bipedalwalker",device='cpu')
    obs, info = env.reset()
    for i in range(1600):
        action, _states = model.predict(obs)
        obs, rewards, terminated, truncated, info = env.step(action)
        env.render()
else:
    env = gym.make("BipedalWalker-v3", hardcore=False, render_mode="rgb_array")
    model = SAC("MlpPolicy", env, verbose=0,device='cuda')
    model.learn(total_timesteps=500000)
    model.save("ppo_bipedalwalker")
