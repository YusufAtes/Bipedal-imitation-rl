import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
from gait_generator_net import SimpleFCNN
from scipy.signal import resample
class BipedEnv(gym.Env):
    def __init__(self,render=False, render_mode= None):
        super(BipedEnv, self).__init__()
        self.initialized = 0
        self.M = 48
        self.m_1 = 7
        self.m_2 = 4
        self.l_1 = 0.5
        self.l_2 = 0.6
        self.g = 9.8
        self.b_1 = 10.0
        self.b_2 = 10.0
        self.b_k = 1000.0
        self.b_g = 1000.0
        self.k_k = 10000.0
        self.k_g = 10000.0
        self.l_1_2 = self.l_1 / 2
        self.l_2_2 = self.l_2 / 2
        self.ramp_angle = 0
        self.roughness_multiplier = 0

        self.I_1 =self.m_1 * self.l_1**2 / 12
        self.I_2 = self.m_2 * self.l_2**2 / 12
        self.MInv = 1 / self.M
        self.m_1Inv = 1 / self.m_1
        self.m_2Inv = 1 / self.m_2
        self.max_steps = 50000
        self.action_space = spaces.Box(low=-10, high=10, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-200, high=200, shape=(30,), dtype=np.float32)
        self.t = 0
        self.dt = 0.0001
        self.gaitgen_net = SimpleFCNN()
        self.gaitgen_net.load_state_dict(torch.load('model_hs512_lpmse_bs64_epoch1000_fft.pth',weights_only=True))
        self.normalizationconst = np.load(rf"normalization_constants.npy")
        #x,dx are the states
        #torques are the actions

    def reset(self,seed=None):
        # Reset the state of the environment to an initial state
        self.initialized +=1
        self.t = 0
        selected_speed = np.random.rand()*3
        self.reference_speed = selected_speed
        self.ramp_angle = 0
        self.rough_terrain_array = np.random.rand(200)*self.roughness_multiplier
        right_len = self.l_1 + self.l_2
        left_len = self.l_1 + self.l_2
        encoder_vec = np.empty((3))   # init_pos + speed + r_leglength + l_leglength + ramp_angle = 0
        encoder_vec[0] = selected_speed/3
        encoder_vec[1] = right_len
        encoder_vec[2] = left_len
        encoder_vec = torch.tensor(encoder_vec, dtype=torch.float32)    
        self.reference = self.findgait(encoder_vec)                     #Find the gait
        self.reference = np.clip(self.reference, -np.pi/2, np.pi/2)     #Clip the gait
        self.reference[:,1] = self.reference[:,1] + self.reference[:,0] #adjust for angle arrangment
        self.reference[:,3] = self.reference[:,3] + self.reference[:,2]
        distances = np.linalg.norm(self.reference[:int(1/self.dt)], axis=1)
        closest_idx = np.argmin(distances)                              #Starting point of the gait
        self.reference = self.reference[closest_idx:,:]
        self.reference = np.pi/2 - self.reference
        x = np.zeros((14))
        x[0] = 0
        x[1] = 1.09
        x[4] = 0.45 * np.pi
        x[10] = 0.45 * np.pi
        x[7] = 0.57 * np.pi
        x[13] = 0.57 * np.pi

        # Calculate other values based on the equations
        x[2] = x[0] + self.l_1 * np.cos(x[4]) / 2
        x[3] = x[1] - self.l_1 * np.sin(x[4]) / 2
        x[5] = x[0] + self.l_1 * np.cos(x[7]) / 2
        x[6] = x[1] - self.l_1 * np.sin(x[7]) / 2
        x[8] = x[0] + self.l_1 * np.cos(x[4]) + self.l_2 * np.cos(x[10]) / 2
        x[9] = x[1] - self.l_1 * np.sin(x[4]) - self.l_2 * np.sin(x[10]) / 2
        x[11] = x[0] + self.l_1 * np.cos(x[7]) + self.l_2 * np.cos(x[13]) / 2
        x[12] = x[1] - self.l_1 * np.sin(x[7]) - self.l_2 * np.sin(x[13]) / 2

        env_info = np.array([self.ramp_angle,selected_speed])
        self.state_x = np.concatenate([env_info,x])
        self.state_dx = np.zeros(14)
        self.state = np.concatenate([self.state_x, self.state_dx])
        self.reset_info = {}
        return self.state, self.reset_info

    def step(self,torques):

        x = self.state[2:16]
        dx = self.state[16:]
        self.t += 1

        x[4] = x[4] % (2*np.pi)
        x[7] = x[7] % (2*np.pi)
        x[10] = x[10] % (2*np.pi)
        x[13] = x[13] % (2*np.pi)

        sin_x5  =np.sin(x[4])
        cos_x5  =np.cos(x[4])

        sin_x8  =np.sin(x[7])
        cos_x8  =np.cos(x[7])

        sin_x11 =np.sin(x[10])
        cos_x11 =np.cos(x[10])

        sin_x14 =np.sin(x[13])
        cos_x14 =np.cos(x[13])

        l_1_2_sin_x5=self.l_1_2*sin_x5
        l_1_2_cos_x5=self.l_1_2*cos_x5
        l_1_2_sin_x8=self.l_1_2*sin_x8
        l_1_2_cos_x8=self.l_1_2*cos_x8
        l_2_2_sin_x11=self.l_2_2*sin_x11
        l_2_2_cos_x11=self.l_2_2*cos_x11
        l_2_2_sin_x14=self.l_2_2*sin_x14
        l_2_2_cos_x14=self.l_2_2*cos_x14
        l_1_2_sin_x5_I_1=-l_1_2_sin_x5/self.I_1
        l_1_2_cos_x5_I_1=-l_1_2_cos_x5/self.I_1
        l_1_2_sin_x8_I_1=-l_1_2_sin_x8/self.I_1
        l_1_2_cos_x8_I_1=-l_1_2_cos_x8/self.I_1
        l_2_2_sin_x11_I_2=-l_2_2_sin_x11/self.I_2
        l_2_2_cos_x11_I_2=-l_2_2_cos_x11/self.I_2
        l_2_2_sin_x14_I_2=-l_2_2_sin_x14/self.I_2
        l_2_2_cos_x14_I_2=-l_2_2_cos_x14/self.I_2

        P_x = np.array([
            [self.MInv, 0, self.MInv, 0, 0, 0, 0, 0],
            [0, self.MInv, 0, self.MInv, 0, 0, 0, 0],
            [-self.m_1Inv, 0, 0, 0, self.m_1Inv, 0, 0, 0],
            [0, -self.m_1Inv, 0, 0, 0, self.m_1Inv, 0, 0],
            [l_1_2_sin_x5_I_1, l_1_2_cos_x5_I_1, 0, 0, l_1_2_sin_x5_I_1, l_1_2_cos_x5_I_1, 0, 0],
            [0, 0, -self.m_1Inv, 0, 0, 0, self.m_1Inv, 0],
            [0, 0, 0, -self.m_1Inv, 0, 0, 0, self.m_1Inv],
            [0, 0, l_1_2_sin_x8_I_1, l_1_2_cos_x8_I_1, 0, 0, l_1_2_sin_x8_I_1, l_1_2_cos_x8_I_1],
            [0, 0, 0, 0, -self.m_2Inv, 0, 0, 0],
            [0, 0, 0, 0, 0, -self.m_2Inv, 0, 0],
            [0, 0, 0, 0, l_2_2_sin_x11_I_2, l_2_2_cos_x11_I_2, 0, 0],
            [0, 0, 0, 0, 0, 0, -self.m_2Inv, 0],
            [0, 0, 0, 0, 0, 0, 0, -self.m_2Inv],
            [0, 0, 0, 0, 0, 0, l_2_2_sin_x14_I_2, l_2_2_cos_x14_I_2]
        ])
        C_x = np.array([
            [ 1.0,   0.0,  -1.0,   0.0,  -l_1_2_sin_x5,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0 ],
            [ 0.0,   1.0,   0.0,  -1.0,  -l_1_2_cos_x5,  0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0,   0.0 ],
            [ 1.0,   0.0,   0.0,   0.0,   0.0,          -1.0,   0.0,  -l_1_2_sin_x8, 0.0,  0.0,   0.0,   0.0,   0.0,   0.0 ],
            [ 0.0,   1.0,   0.0,   0.0,   0.0,           0.0,  -1.0,  -l_1_2_cos_x8, 0.0,  0.0,   0.0,   0.0,   0.0,   0.0 ],
            [ 0.0,   0.0,   1.0,   0.0,  -l_1_2_sin_x5,   0.0,   0.0,   0.0,  -1.0,   0.0,  -l_2_2_sin_x11, 0.0,  0.0,   0.0 ],
            [ 0.0,   0.0,   0.0,   1.0,  -l_1_2_cos_x5,   0.0,   0.0,   0.0,   0.0,  -1.0,  -l_2_2_cos_x11, 0.0,  0.0,   0.0 ],
            [ 0.0,   0.0,   0.0,   0.0,   0.0,           1.0,   0.0,  -l_1_2_sin_x8, 0.0,   0.0,   0.0,  -1.0,  0.0,  -l_2_2_sin_x14 ],
            [ 0.0,   0.0,   0.0,   0.0,   0.0,           0.0,   1.0,  -l_1_2_cos_x8, 0.0,   0.0,   0.0,   0.0, -1.0,  -l_2_2_cos_x14 ]
        ])

        x_r = x[8] + l_2_2_cos_x11
        y_r = x[9] - l_2_2_sin_x11
        y_g_x_r = self.y_g(x_r, self.ramp_angle) #+ self.rough_terrain_array[int(x_r)*10]

        if (y_r - y_g_x_r) < 0:
            y_d = y_g_x_r - y_r
            hh = y_d / np.sin(x[10])
            x_d = min(self.l_2, hh) * np.cos(x[10])
            F_g_1 = -self.k_g * x_d - self.b_g * (dx[9] - l_2_2_sin_x11 * dx[10])
            F_g_2 = -self.k_g * (y_r - y_g_x_r) + self.b_g * np.max(-(dx[9] - l_2_2_cos_x11 * dx[10]),0)
        else:
            F_g_1, F_g_2 = 0, 0

        x_l = x[11] + l_2_2_cos_x14
        y_l = x[12] - l_2_2_sin_x14
        y_g_x_l = self.y_g(x_l, self.ramp_angle) #+ self.rough_terrain_array[int(x_l)*10]

        if (y_l - y_g_x_l) < 0:
            y_d = y_g_x_l - y_l
            hh = y_d / np.sin(x[13])
            x_d = min(self.l_2, hh) * np.cos(x[13])
            F_g_3 = -self.k_g * x_d - self.b_g * (dx[11] - l_2_2_sin_x14 * dx[13])
            F_g_4 = -self.k_g * (y_l - y_g_x_l) + self.b_g * np.max(-(dx[13] - l_2_2_cos_x14 * dx[13]),0)
        else:
            F_g_3, F_g_4 = 0, 0

        f_x5_x11 = max(0, x[4] - x[10])
        f_x8_x14 = max(0, x[7] - x[13])

        h_x5_x11 = self.h(x[4] - x[10])
        h_x8_x14 = self.h(x[7] - x[13])

        T_r1_y = torques[0]
        T_r2_y = torques[1]
        T_r3_y = torques[2]
        T_r4_y = torques[3]
        h_F_g_2 = self.h(F_g_2)
        h_F_g_4 = self.h(F_g_4)

        T_r5_x_dx_y = torques[4] * h_F_g_2
        T_r6_x_dx_y = torques[5] * h_F_g_4

        Q_x_dx_y_F_g = np.array([
            0,
            -self.g,
            0,
            -self.g,
            (-self.b_1 * abs(x[4] - np.pi / 2) * dx[4] - (self.b_2 + self.b_k * f_x5_x11) * (dx[4] - dx[10]) - self.k_k * h_x5_x11 + T_r1_y + T_r3_y) / self.I_1,
            0,
            -self.g,
            (-self.b_1 * abs(x[7] - np.pi / 2) * dx[7] - (self.b_2 + self.b_k * f_x8_x14) * (dx[7] - dx[13]) - self.k_k * h_x8_x14 + T_r2_y + T_r4_y) / self.I_1,
            F_g_1 / self.m_2,
            F_g_2 / self.m_2 - self.g,
            (-F_g_1 * l_2_2_sin_x11 - F_g_2 * l_2_2_cos_x11 - (self.b_2 + self.b_k * f_x5_x11) * (dx[10] - dx[4]) + self.k_k * h_x5_x11 - T_r3_y - T_r5_x_dx_y) / self.I_2,
            F_g_3 / self.m_2,
            F_g_4 / self.m_2 - self.g,
            (-F_g_3 * l_2_2_sin_x14 - F_g_4 * l_2_2_cos_x14 - (self.b_2 + self.b_k * f_x8_x14) * (dx[13] - dx[7]) + self.k_k * h_x8_x14 - T_r4_y - T_r6_x_dx_y) / self.I_2
        ])

        dx5_2 = dx[4]**2
        dx8_2 = dx[7]**2
        dx11_2 = dx[10]**2
        dx14_2 = dx[13]**2

        l_1_2_cos_x5_dx5_2 = l_1_2_cos_x5 * dx5_2
        l_1_2_sin_x5_dx5_2 = l_1_2_sin_x5 * dx5_2
        l_1_2_cos_x8_dx8_2 = l_1_2_cos_x8 * dx8_2
        l_1_2_sin_x8_dx8_2 = l_1_2_sin_x8 * dx8_2

        D_x_dx = np.array([
            l_1_2_cos_x5_dx5_2,
            -l_1_2_sin_x5_dx5_2,
            l_1_2_cos_x8_dx8_2,
            -l_1_2_sin_x8_dx8_2,
            l_1_2_cos_x5_dx5_2 + l_2_2_cos_x11 * dx11_2,
            -l_1_2_sin_x5_dx5_2 - l_2_2_sin_x11 * dx11_2,
            l_1_2_cos_x8_dx8_2 + l_2_2_cos_x14 * dx14_2,
            -l_1_2_sin_x8_dx8_2 - l_2_2_sin_x14 * dx14_2
        ])
        temp_8x1 = D_x_dx - (C_x @ Q_x_dx_y_F_g)  # (8x1)
        d2x = P_x @ np.linalg.solve((C_x @ P_x), temp_8x1) + Q_x_dx_y_F_g  # (14x1)
        dx = dx + self.dt * d2x
        x = x + self.dt * dx
        x[[4,10,7,13]] = x[[4,10,7,13]] % (2*np.pi)
        self.state[2:16] = x
        self.state[16:] = dx
        #force = np.array([F_g_1, F_g_2, F_g_3, F_g_4])

        reward, done = self.biped_reward(self.state[2:16], self.state[16:],torques)
        info = {}
        truncated = False
        if self.t == self.max_steps:
            truncated = True
        return self.state, reward, done, truncated, info

    # def biped_reward(self,x, dx,torques):
    #     done = False
    #     reward = 0
    #     current_step = (self.t) * self.dt
    #     if self.t  < self.max_steps:
    #         reward -=  0.5* np.mean(np.linalg.norm(self.reference[self.t:self.t+5,:] - x[[4,10,7,13]]))

    #     if x[1] < 0.9:
    #         reward -= 100
    #         done = True
    #     else:
    #         reward += 0.1
    
    #     # include speed reward something like
    #     reward -= (np.abs(x[0] - self.reference_speed * current_step))
    #     return reward, done
    def biped_reward(self,x,dx,torques):

        reward = 0
        current_time = self.t*self.dt
        #contacts = p.getContactPoints(bodyA=self.robot, bodyB=self.planeId)
        done = False

        # # Reward for staying close to the reference trajectory
        # reference_diff = np.linalg.norm(np.min(np.abs(self.reference[self.t:self.t+10,:] - x[[4,10,7,13]]),axis=0))
        # if reference_diff < 0.3:
        #     reward += 0.3-reference_diff
        # else:
        #     reward -= 0.1*reference_diff
        # Reward to penalize falling

        if x[1] < 0.9:
            reward -= 1000
            done = True

        else:
            reward += 0.01
        reward += x[0]
        # # Reward for staying close to the reference speed
        # if np.abs(x[0] - self.reference_speed*current_time) < 0.20:
        #     reward += x[0]
        # else:
        #     reward -= np.abs(x[0] - self.reference_speed*current_time)
        return reward, done
    
    def close(self):
        pass

    def y_g(self,xrl, rampAngle):
        """
        Ground pattern function.
        
        Parameters:
            xrl: Input value (e.g., distance or position).
            rampAngle: Angle of the ramp in degrees.
        
        Returns:
            output: The calculated ground pattern value.
        """
        # Calculate the ground pattern value
        output = np.sin(np.pi * rampAngle / 180) * xrl
        return output

    def h(self,input):
        if input > 0:
            return 1
        else:
            return 0    

    def findgait(self,input_vec):

        freqs = self.gaitgen_net(input_vec)
        predictions = freqs.reshape(-1,2,25,4)
        predictions = predictions.detach().numpy()
        predictions = predictions[0]
        predictions = self.denormalize(predictions)
        pred_time = self.pred_ifft(predictions)

        return pred_time

    def denormalize(self,pred):

        pred[0,:,0] = pred[0,:,0] * self.normalizationconst[0]
        pred[0,:,1] = pred[0,:,1] * self.normalizationconst[1]
        pred[0,:,2] = pred[0,:,2] * self.normalizationconst[2]
        pred[0,:,3] = pred[0,:,3] * self.normalizationconst[3]
        pred[1,:,0] = pred[1,:,0] * self.normalizationconst[4]
        pred[1,:,1] = pred[1,:,1] * self.normalizationconst[5]
        pred[1,:,2] = pred[1,:,2] * self.normalizationconst[6]
        pred[1,:,3] = pred[1,:,3] * self.normalizationconst[7]
        
        return pred
        

    def pred_ifft(self,predictions):

        real_pred = predictions[0,:,:]
        imag_pred = predictions[0,:,:]
        predictions = real_pred + 1j*imag_pred

        padded_pred = np.zeros((257,4),dtype=complex)
        padded_pred[:25,:] = predictions

        padded_time = np.fft.irfft(padded_pred, axis=0)
        pred_time = padded_time[106:-106,:]
        #upsample from 100 hz to 1/dt hz
        if self.dt < 0.01:
            num_samples = int(300 * (int(1/self.dt) / 100))  # 960
            # Upsample using Fourier method
            pred_time = resample(pred_time, num_samples, axis=0)

        pred_time = np.repeat(pred_time, 10, axis=0)

        return pred_time

#     # Define action and observation space
#     # They must be gym.spaces objects
#     # Example when using discrete actions:
#     self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
#     # Example for using image as input:
#     self.observation_space = spaces.Box(low=0, high=255,
#                                         shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)

#   def step(self, action):
#     ...
#     return observation, reward, done, info
#   def reset(self):
#     ...
#     return observation  # reward, done, info can't be included
#   def render(self, mode='human'):
#     ...
#   def close (self):
#     ...

# def biped_plant_force_out_touch_sensitive_rough_terrain_env(dt, x, dx, torques, ramp_angle, rough_terrain_array):
#     M = 48
#     m_1 = 7
#     m_2 = 4
#     l_1 = 0.5
#     l_2 = 0.6
#     g = 9.8
#     b_1 = 10
#     b_2 = 10
#     b_k = 1000
#     b_g = 1000
#     k_k = 10000
#     k_g = 10000
#     l_1_2 = l_1 / 2
#     l_2_2 = l_2 / 2

#     I_1 = m_1 * l_1**2 / 12
#     I_2 = m_2 * l_2**2 / 12
#     MInv = 1 / M
#     m_1Inv = 1 / m_1
#     m_2Inv = 1 / m_2

#     sin_x5  =np.sin(x[4])
#     cos_x5  =np.cos(x[4])
#     sin_x8  =np.sin(x[7])
#     cos_x8  =np.cos(x[7])
#     sin_x11 =np.sin(x[10])
#     cos_x11 =np.cos(x[10])
#     sin_x14 =np.sin(x[13])
#     cos_x14 =np.cos(x[13])

#     l_1_2_sin_x5=l_1_2*sin_x5
#     l_1_2_cos_x5=l_1_2*cos_x5
#     l_1_2_sin_x8=l_1_2*sin_x8
#     l_1_2_cos_x8=l_1_2*cos_x8
#     l_2_2_sin_x11=l_2_2*sin_x11
#     l_2_2_cos_x11=l_2_2*cos_x11
#     l_2_2_sin_x14=l_2_2*sin_x14
#     l_2_2_cos_x14=l_2_2*cos_x14
#     l_1_2_sin_x5_I_1=-l_1_2_sin_x5/I_1
#     l_1_2_cos_x5_I_1=-l_1_2_cos_x5/I_1
#     l_1_2_sin_x8_I_1=-l_1_2_sin_x8/I_1
#     l_1_2_cos_x8_I_1=-l_1_2_cos_x8/I_1
#     l_2_2_sin_x11_I_2=-l_2_2_sin_x11/I_2
#     l_2_2_cos_x11_I_2=-l_2_2_cos_x11/I_2
#     l_2_2_sin_x14_I_2=-l_2_2_sin_x14/I_2
#     l_2_2_cos_x14_I_2=-l_2_2_cos_x14/I_2


#     P_x = np.array([
#         [MInv, 0, MInv, 0, 0, 0, 0, 0],
#         [0, MInv, 0, MInv, 0, 0, 0, 0],
#         [-m_1Inv, 0, 0, 0, m_1Inv, 0, 0, 0],
#         [0, -m_1Inv, 0, 0, 0, m_1Inv, 0, 0],
#         [l_1_2_sin_x5_I_1, l_1_2_cos_x5_I_1, 0, 0, l_1_2_sin_x5_I_1, l_1_2_cos_x5_I_1, 0, 0],
#         [0, 0, -m_1Inv, 0, 0, 0, m_1Inv, 0],
#         [0, 0, 0, -m_1Inv, 0, 0, 0, m_1Inv],
#         [0, 0, l_1_2_sin_x8_I_1, l_1_2_cos_x8_I_1, 0, 0, l_1_2_sin_x8_I_1, l_1_2_cos_x8_I_1],
#         [0, 0, 0, 0, -m_2Inv, 0, 0, 0],
#         [0, 0, 0, 0, 0, -m_2Inv, 0, 0],
#         [0, 0, 0, 0, l_2_2_sin_x11_I_2, l_2_2_cos_x11_I_2, 0, 0],
#         [0, 0, 0, 0, 0, 0, -m_2Inv, 0],
#         [0, 0, 0, 0, 0, 0, 0, -m_2Inv],
#         [0, 0, 0, 0, 0, 0, l_2_2_sin_x14_I_2, l_2_2_cos_x14_I_2]
#     ])

#     C_x = np.array([
#         [1, 0, -1, 0, -l_1_2_sin_x5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 1, 0, -1, -l_1_2_cos_x5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [1, 0, 0, 0, 0, -1, 0, -l_1_2_sin_x8, 0, 0, 0, 0, 0, 0],
#         [0, 1, 0, 0, 0, 0, -1, -l_1_2_cos_x8, 0, 0, 0, 0, 0, 0],
#         [0, 0, 1, 0, -l_1_2_sin_x5, 0, 0, 0, -1, 0, -l_2_2_sin_x11, 0, 0, 0],
#         [0, 0, 0, 1, -l_1_2_cos_x5, 0, 0, 0, 0, -1, -l_2_2_cos_x11, 0, 0, 0],
#         [0, 0, 0, 0, 0, 1, 0, -l_1_2_sin_x8, 0, 0, 0, -1, 0, -l_2_2_sin_x14],
#         [0, 0, 0, 0, 0, 0, 1, -l_1_2_cos_x8, 0, 0, 0, 0, -1, -l_2_2_cos_x14]
#     ])

#     x_r = x[8] + l_2_2_cos_x11
#     y_r = x[9] - l_2_2_sin_x11
#     y_g_x_r = y_g(x_r, ramp_angle) + rough_terrain_array(int(x_r)*10)

#     if (y_r - y_g_x_r) < 0:
#         y_d = y_g_x_r - y_r
#         hh = y_d / np.sin(x[10])
#         x_d = min(l_2, hh) * np.cos(x[10])
#         F_g_1 = -k_g * x_d - b_g * (dx[9] - l_2_2_sin_x11 * dx[10])
#         F_g_2 = -k_g * (y_r - y_g_x_r) + b_g * np.max(-(dx[9] - l_2_2_cos_x11 * dx[10]),0)
#     else:
#         F_g_1, F_g_2 = 0, 0

#     x_l = x[11] + l_2_2_cos_x14
#     y_l = x[12] - l_2_2_sin_x14
#     y_g_x_l = y_g(x_l, ramp_angle) + rough_terrain_array(int(x_l)*10)

#     if (y_l - y_g_x_l) < 0:
#         y_d = y_g_x_l - y_l
#         hh = y_d / np.sin(x[13])
#         x_d = min(l_2, hh) * np.cos(x[13])
#         F_g_3 = -k_g * x_d - b_g * (dx[11] - l_2_2_sin_x14 * dx[13])
#         F_g_4 = -k_g * (y_l - y_g_x_l) + b_g * np.max(-(dx[13] - l_2_2_cos_x14 * dx[13]),0)
#     else:
#         F_g_3, F_g_4 = 0, 0

#     f_x5_x11 = max(0, x[4] - x[10])
#     f_x8_x14 = max(0, x[7] - x[13])

#     h_x5_x11 = self.h(x[4] - x[10])
#     h_x8_x14 = h(x[7] - x[13])

#     T_r1_y = torques[0]
#     T_r2_y = torques[1]
#     T_r3_y = torques[2]
#     T_r4_y = torques[3]
#     h_F_g_2 = h(F_g_2)
#     h_F_g_4 = h(F_g_4)

#     T_r5_x_dx_y = torques[4] * h_F_g_2
#     T_r6_x_dx_y = torques[5] * h_F_g_4

#     Q_x_dx_y_F_g = np.array([
#         0,
#         -g,
#         0,
#         -g,
#         (-b_1 * abs(x[4] - np.pi / 2) * dx[4] - (b_2 + b_k * f_x5_x11) * (dx[4] - dx[10]) - k_k * h_x5_x11 + T_r1_y + T_r3_y) / I_1,
#         0,
#         -g,
#         (-b_1 * abs(x[7] - np.pi / 2) * dx[7] - (b_2 + b_k * f_x8_x14) * (dx[7] - dx[13]) - k_k * h_x8_x14 + T_r2_y + T_r4_y) / I_1,
#         F_g_1 / m_2,
#         F_g_2 / m_2 - g,
#         (-F_g_1 * l_2_2_sin_x11 - F_g_2 * l_2_2_cos_x11 - (b_2 + b_k * f_x5_x11) * (dx[10] - dx[4]) + k_k * h_x5_x11 - T_r3_y - T_r5_x_dx_y) / I_2,
#         F_g_3 / m_2,
#         F_g_4 / m_2 - g,
#         (-F_g_3 * l_2_2_sin_x14 - F_g_4 * l_2_2_cos_x14 - (b_2 + b_k * f_x8_x14) * (dx[13] - dx[7]) + k_k * h_x8_x14 - T_r4_y - T_r6_x_dx_y) / I_2
#     ])

#     dx5_2 = dx[4]**2
#     dx8_2 = dx[7]**2
#     dx11_2 = dx[10]**2
#     dx14_2 = dx[13]**2

#     l_1_2_cos_x5_dx5_2 = l_1_2_cos_x5 * dx5_2
#     l_1_2_sin_x5_dx5_2 = l_1_2_sin_x5 * dx5_2
#     l_1_2_cos_x8_dx8_2 = l_1_2_cos_x8 * dx8_2
#     l_1_2_sin_x8_dx8_2 = l_1_2_sin_x8 * dx8_2

#     D_x_dx = np.array([
#         l_1_2_cos_x5_dx5_2,
#         -l_1_2_sin_x5_dx5_2,
#         l_1_2_cos_x8_dx8_2,
#         -l_1_2_sin_x8_dx8_2,
#         l_1_2_cos_x5_dx5_2 + l_2_2_cos_x11 * dx11_2,
#         -l_1_2_sin_x5_dx5_2 - l_2_2_sin_x11 * dx11_2,
#         l_1_2_cos_x8_dx8_2 + l_2_2_cos_x14 * dx14_2,
#         -l_1_2_sin_x8_dx8_2 - l_2_2_sin_x14 * dx14_2
#     ])

#     d2x = P_x *((C_x * P_x ) /(D_x_dx - C_x * Q_x_dx_y_F_g)) + Q_x_dx_y_F_g
#     dx = dx + dt * d2x
#     x = x + dt * dx
#     #observation space is x, dx 
#     force = np.array([F_g_1, F_g_2, F_g_3, F_g_4])

#     return x, dx, force
