import gymnasium as gym
import numpy as np
import torch
from scipy.signal import resample

class CustomWalker2dEnv(gym.Wrapper):
    def __init__(self, env):
        """
        Wraps the Walker2d environment to include a custom reward and 
        augmented observation (original observation plus desired speed and current ramp angle).

        Args:
            env: The original Walker2d environment.
            desired_speed (float): The desired speed value to include in the observation.
        """
        super(CustomWalker2dEnv, self).__init__(env)
        

        # Update the observation space to account for two extra dimensions:
        # one for desired speed and one for current ramp angle.
        orig_space = self.env.observation_space
        extra_low = np.array([-np.inf, -np.inf], dtype=np.float32)
        extra_high = np.array([np.inf, np.inf], dtype=np.float32)
        new_low = np.concatenate([orig_space.low, extra_low])
        new_high = np.concatenate([orig_space.high, extra_high])
        self.observation_space = gym.spaces.Box(low=new_low, high=new_high, dtype=np.float32)

    def step(self, action):
        # Take a step in the underlying environment.
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Calculate your custom reward (which can use the original reward).
        custom_reward = self.custom_reward_function(observation, reward, terminated, truncated, info, action)
        
        # Augment the observation with desired speed and current ramp angle.
        custom_observation = self.customize_observation(observation, info)
        return custom_observation, custom_reward, terminated, truncated, info

    def custom_reward_function(self, observation, reward, terminated, truncated, info, action):
        self.reference_weight = 0.2
        
        custom_reward = reward  # Replace or augment this value as needed.
        rhip = observation[2]
        rknee = observation[3]
        rankle = observation[4]
        lhip = observation[5]
        lknee = observation[6]
        lankle = observation[7]

        vel = observation[8]
        vel_rankle = observation[13]
        vel_lankle = observation[16]

        if self.init:
            rhip_diff = np.abs(self.reference[0,0] - rhip)
            rknee_diff = np.abs(self.reference[0,1] - rknee)
            lhip_diff = np.abs(self.reference[0,2] - lhip)
            lknee_diff = np.abs(self.reference[0,3] - lknee)
            mean_diff = np.mean(np.abs([rhip_diff, rknee_diff, lhip_diff, lknee_diff]))
            custom_reward += (0.3 - mean_diff) * self.reference_weight
            if mean_diff < 0.2:
                self.init = False
                self.reference_idx = 1
        else:
            rhip_diff = np.abs(self.reference[self.reference_idx,0] - rhip)
            rknee_diff = np.abs(self.reference[self.reference_idx,1] - rknee)
            lhip_diff = np.abs(self.reference[self.reference_idx,2] - lhip)
            lknee_diff = np.abs(self.reference[self.reference_idx,3] - lknee)
            custom_reward += self.ref_reward(self,rhip_diff, rknee_diff, lhip_diff, lknee_diff)
            self.reference_idx += 1
    
        return custom_reward

    def customize_observation(self, observation, info):

        # Try to extract current ramp angle from info; default to 0.0 if not provided.
        current_ramp_angle = info.get("ramp_angle", 0.0)
        extra_features = np.array([self.desired_speed, current_ramp_angle], dtype=np.float32)
        custom_obs = np.concatenate([observation, extra_features])
        return custom_obs

    def reset(self, **kwargs):
        # Reset the underlying environment.
        self.init = True
        self.dt = 1/500
        self.desired_speed = 0.3 + np.random.rand() *2.7  # Randomly set the desired speed on reset.
        self.leg_len = 1.05
        observation, info = self.env.reset(**kwargs)
        # Augment the observation on reset as well.
        custom_observation = self.customize_observation(observation, info)
        encoder_vec = np.empty((3))   # init_pos + speed + r_leglength + l_leglength + ramp_angle = 0
        encoder_vec[0] = self.desired_speed/3
        encoder_vec[1] = self.leg_len /1.5
        encoder_vec[2] = self.leg_len /1.5
        encoder_vec = torch.tensor(encoder_vec, dtype=torch.float32)    
        self.reference = self.findgait(encoder_vec)                     #Find the gait
        self.reference = np.clip(self.reference, -np.pi/2, np.pi/2)     #Clip the gait

        return custom_observation, info

    def findgait(self,input_vec):

        freqs = self.gaitgen_net(input_vec)
        predictions = freqs.reshape(-1,4,2,17)
        predictions = predictions.detach().numpy()
        predictions = predictions[0]
        predictions = self.denormalize(predictions)
        pred_time = self.pred_ifft(predictions)

        return pred_time

    def denormalize(self,pred):
        #form is [5,2,17]
        for i in range(17):
            for k in range(2):
                pred[:,k,i] = pred[:,k,i] * self.normalizationconst[i*2+k]
        return pred
    
    def pred_ifft(self,predictions):
        #form is [5,2,17]
        real_pred = predictions[:,0,:]
        imag_pred = predictions[:,1,:]
        predictions = real_pred + 1j*imag_pred

        pred_time = np.fft.irfft(predictions, axis=1)
        pred_time = pred_time.transpose(1,0)
        org_rate = 10

        if self.dt < 0.1:
            num_samples = int((pred_time.shape[0]) * (1/self.dt)/(org_rate))  # resample with self.dt
            # Upsample using Fourier method
            pred_time = resample(pred_time, num_samples, axis=0)
        # pred_time = np.tile(pred_time, (int(self.max_steps*self.dt),1))    # Create loop for reference movement
        return pred_time
    

    def ref_reward(self,rhip_diff, rknee_diff, lhip_diff, lknee_diff):
        reward = 0
        if rhip_diff < 0.15:
            reward += 1 * self.reference_weight
        else:
            reward -= 1 * self.reference_weight

        if rknee_diff < 0.15:
            reward += 1 * self.reference_weight
        else:
            reward-=1* self.reference_weight

        if lhip_diff < 0.15:
            reward += 1* self.reference_weight
        else:
            reward -= 1* self.reference_weight
        
        if lknee_diff < 0.15:
            reward += 1* self.reference_weight
        else:
            reward -= 1* self.reference_weight

        return reward
    