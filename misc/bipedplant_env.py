import gym
from gym import spaces
import numpy as np

class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, config):
        super(CustomEnv, self).__init__()
        self.config = config

        self.action_space = gym.spaces.Box(low=np.array([-1, -1, -1, -1, -1,  -1]), 
                                       high=np.array([1, 1, 1, 1, 1, 1]), 
                                       dtype=np.float32)
        
        # Observation space: The agent's position in the 2D plane [x, y]
        self.observation_space = gym.spaces.Box(low=np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973,  0.5   , -2.8973,-2.8973, -1.7628, -2.8973, -3.0718, -2.8973,  0.5   , -2.8973, -9.70, -4.18, -8.32, -5.35, -5.96 ,-13.71, -6.12, -9.70, -4.18, -8.32, -5.35, -5.96 ,-13.71, -6.12]), 
                                       high=np.array([2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.    ,  2.8973, 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.    ,  2.8973, 9.34, 13.23, 10.33, 8.79, 6.37, 5.69, 5.11,9.34, 13.23, 10.33, 8.79, 6.37, 5.69, 5.11]), 
                                       dtype=np.float32)

    def step(self, action):
        # Execute one time step within the environment
        state = self._take_action(action)
        reward = self._get_reward()
        done = self._is_done()
        info = {}
        
        return state, reward, done, info

    def reset(self):
        # Reset the state of the environment to an initial state
        self.state = self._get_initial_state()
        return self.state

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass

    def _take_action(self, action):
        # Define how the action affects the state
        pass

    def _get_reward(self):
        # Define the reward function
        pass

    def _is_done(self):
        # Define the condition to end the episode
        pass

    def _get_initial_state(self):
        # Define the initial state of the environment
        pass