import numpy as np
from stable_baselines.common.env_checker import check_env
import gym
from gym import spaces

x_init = [0,1.1,]
def y_g(xrl, rampAngle):
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

def h(input):
    if input > 0:
        return 1
    else:
        return 0    

import gym
from gym import spaces

def biped_reward(x, dx, t,torques, ramp_angle, reference=None):
    pos_weight = 0.5
    fall_weight = 10
    torque_weight = 0.1
    # Define the reward function
    refrence_move = reference[t]
    pos_diff = np.sum(np.linalg.norm(x[4,11,7,13] - refrence_move))
    pos_reward = np.exp(-pos_weight*pos_diff)
    if x[2] < 0.5:
        fall_reward = np.exp(-fall_weight*(1.1-x[2]))
    else:
        fall_reward = 0
    torque_reward = np.exp(-torque_weight*np.sum(torques**2))
    reward = pos_reward + fall_reward + torque_reward
    return reward



    return None
class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CustomEnv, self).__init__()
        self.M = 48
        self.m_1 = 7
        self.m_2 = 4
        self.l_1 = 0.5
        self.l_2 = 0.6
        self.g = 9.8
        self.b_1 = 10
        self.b_2 = 10
        self.b_k = 1000
        self.b_g = 1000
        self.k_k = 10000
        self.k_g = 10000
        self.l_1_2 = self.l_1 / 2
        self.l_2_2 = self.l_2 / 2

        self.I_1 =self.m_1 * self.l_1**2 / 12
        self.I_2 = self.m_2 * self.l_2**2 / 12
        self.MInv = 1 / self.M
        self.m_1Inv = 1 / self.m_1
        self.m_2Inv = 1 / self.m_2

        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32)
        self.t = 0

    def step(self,torques,x,dx,dt,ramp_angle,rough_terrain_array,reference=None):
        self.t += 1
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
            [1, 0, -1, 0, -l_1_2_sin_x5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, -1, -l_1_2_cos_x5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, -1, 0, -l_1_2_sin_x8, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, -1, -l_1_2_cos_x8, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, -l_1_2_sin_x5, 0, 0, 0, -1, 0, -l_2_2_sin_x11, 0, 0, 0],
            [0, 0, 0, 1, -l_1_2_cos_x5, 0, 0, 0, 0, -1, -l_2_2_cos_x11, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, -l_1_2_sin_x8, 0, 0, 0, -1, 0, -l_2_2_sin_x14],
            [0, 0, 0, 0, 0, 0, 1, -l_1_2_cos_x8, 0, 0, 0, 0, -1, -l_2_2_cos_x14]
        ])

        x_r = x[8] + l_2_2_cos_x11
        y_r = x[9] - l_2_2_sin_x11
        y_g_x_r = y_g(x_r, ramp_angle) + rough_terrain_array(int(x_r)*10)

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
        y_g_x_l = y_g(x_l, ramp_angle) + rough_terrain_array(int(x_l)*10)

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

        h_x5_x11 = h(x[4] - x[10])
        h_x8_x14 = h(x[7] - x[13])

        T_r1_y = torques[0]
        T_r2_y = torques[1]
        T_r3_y = torques[2]
        T_r4_y = torques[3]
        h_F_g_2 = h(F_g_2)
        h_F_g_4 = h(F_g_4)

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

        d2x = P_x *((C_x * P_x ) /(D_x_dx - C_x * Q_x_dx_y_F_g)) + Q_x_dx_y_F_g
        dx = dx + dt * d2x
        x = x + dt * dx

        force = np.array([F_g_1, F_g_2, F_g_3, F_g_4])

        obs = np.concatenate([x,dx])
        reward = biped_reward(x, dx, self.t-1,torques, ramp_angle, reference)
        done = False
        info = {}
        return obs, reward, done, info

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

def biped_plant_force_out_touch_sensitive_rough_terrain_env(dt, x, dx, torques, ramp_angle, rough_terrain_array):
    M = 48
    m_1 = 7
    m_2 = 4
    l_1 = 0.5
    l_2 = 0.6
    g = 9.8
    b_1 = 10
    b_2 = 10
    b_k = 1000
    b_g = 1000
    k_k = 10000
    k_g = 10000
    l_1_2 = l_1 / 2
    l_2_2 = l_2 / 2

    I_1 = m_1 * l_1**2 / 12
    I_2 = m_2 * l_2**2 / 12
    MInv = 1 / M
    m_1Inv = 1 / m_1
    m_2Inv = 1 / m_2

    sin_x5  =np.sin(x[4])
    cos_x5  =np.cos(x[4])
    sin_x8  =np.sin(x[7])
    cos_x8  =np.cos(x[7])
    sin_x11 =np.sin(x[10])
    cos_x11 =np.cos(x[10])
    sin_x14 =np.sin(x[13])
    cos_x14 =np.cos(x[13])

    l_1_2_sin_x5=l_1_2*sin_x5
    l_1_2_cos_x5=l_1_2*cos_x5
    l_1_2_sin_x8=l_1_2*sin_x8
    l_1_2_cos_x8=l_1_2*cos_x8
    l_2_2_sin_x11=l_2_2*sin_x11
    l_2_2_cos_x11=l_2_2*cos_x11
    l_2_2_sin_x14=l_2_2*sin_x14
    l_2_2_cos_x14=l_2_2*cos_x14
    l_1_2_sin_x5_I_1=-l_1_2_sin_x5/I_1
    l_1_2_cos_x5_I_1=-l_1_2_cos_x5/I_1
    l_1_2_sin_x8_I_1=-l_1_2_sin_x8/I_1
    l_1_2_cos_x8_I_1=-l_1_2_cos_x8/I_1
    l_2_2_sin_x11_I_2=-l_2_2_sin_x11/I_2
    l_2_2_cos_x11_I_2=-l_2_2_cos_x11/I_2
    l_2_2_sin_x14_I_2=-l_2_2_sin_x14/I_2
    l_2_2_cos_x14_I_2=-l_2_2_cos_x14/I_2


    P_x = np.array([
        [MInv, 0, MInv, 0, 0, 0, 0, 0],
        [0, MInv, 0, MInv, 0, 0, 0, 0],
        [-m_1Inv, 0, 0, 0, m_1Inv, 0, 0, 0],
        [0, -m_1Inv, 0, 0, 0, m_1Inv, 0, 0],
        [l_1_2_sin_x5_I_1, l_1_2_cos_x5_I_1, 0, 0, l_1_2_sin_x5_I_1, l_1_2_cos_x5_I_1, 0, 0],
        [0, 0, -m_1Inv, 0, 0, 0, m_1Inv, 0],
        [0, 0, 0, -m_1Inv, 0, 0, 0, m_1Inv],
        [0, 0, l_1_2_sin_x8_I_1, l_1_2_cos_x8_I_1, 0, 0, l_1_2_sin_x8_I_1, l_1_2_cos_x8_I_1],
        [0, 0, 0, 0, -m_2Inv, 0, 0, 0],
        [0, 0, 0, 0, 0, -m_2Inv, 0, 0],
        [0, 0, 0, 0, l_2_2_sin_x11_I_2, l_2_2_cos_x11_I_2, 0, 0],
        [0, 0, 0, 0, 0, 0, -m_2Inv, 0],
        [0, 0, 0, 0, 0, 0, 0, -m_2Inv],
        [0, 0, 0, 0, 0, 0, l_2_2_sin_x14_I_2, l_2_2_cos_x14_I_2]
    ])

    C_x = np.array([
        [1, 0, -1, 0, -l_1_2_sin_x5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, -1, -l_1_2_cos_x5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, -1, 0, -l_1_2_sin_x8, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, -1, -l_1_2_cos_x8, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, -l_1_2_sin_x5, 0, 0, 0, -1, 0, -l_2_2_sin_x11, 0, 0, 0],
        [0, 0, 0, 1, -l_1_2_cos_x5, 0, 0, 0, 0, -1, -l_2_2_cos_x11, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, -l_1_2_sin_x8, 0, 0, 0, -1, 0, -l_2_2_sin_x14],
        [0, 0, 0, 0, 0, 0, 1, -l_1_2_cos_x8, 0, 0, 0, 0, -1, -l_2_2_cos_x14]
    ])

    x_r = x[8] + l_2_2_cos_x11
    y_r = x[9] - l_2_2_sin_x11
    y_g_x_r = y_g(x_r, ramp_angle) + rough_terrain_array(int(x_r)*10)

    if (y_r - y_g_x_r) < 0:
        y_d = y_g_x_r - y_r
        hh = y_d / np.sin(x[10])
        x_d = min(l_2, hh) * np.cos(x[10])
        F_g_1 = -k_g * x_d - b_g * (dx[9] - l_2_2_sin_x11 * dx[10])
        F_g_2 = -k_g * (y_r - y_g_x_r) + b_g * np.max(-(dx[9] - l_2_2_cos_x11 * dx[10]),0)
    else:
        F_g_1, F_g_2 = 0, 0

    x_l = x[11] + l_2_2_cos_x14
    y_l = x[12] - l_2_2_sin_x14
    y_g_x_l = y_g(x_l, ramp_angle) + rough_terrain_array(int(x_l)*10)

    if (y_l - y_g_x_l) < 0:
        y_d = y_g_x_l - y_l
        hh = y_d / np.sin(x[13])
        x_d = min(l_2, hh) * np.cos(x[13])
        F_g_3 = -k_g * x_d - b_g * (dx[11] - l_2_2_sin_x14 * dx[13])
        F_g_4 = -k_g * (y_l - y_g_x_l) + b_g * np.max(-(dx[13] - l_2_2_cos_x14 * dx[13]),0)
    else:
        F_g_3, F_g_4 = 0, 0

    f_x5_x11 = max(0, x[4] - x[10])
    f_x8_x14 = max(0, x[7] - x[13])

    h_x5_x11 = h(x[4] - x[10])
    h_x8_x14 = h(x[7] - x[13])

    T_r1_y = torques[0]
    T_r2_y = torques[1]
    T_r3_y = torques[2]
    T_r4_y = torques[3]
    h_F_g_2 = h(F_g_2)
    h_F_g_4 = h(F_g_4)

    T_r5_x_dx_y = torques[4] * h_F_g_2
    T_r6_x_dx_y = torques[5] * h_F_g_4

    Q_x_dx_y_F_g = np.array([
        0,
        -g,
        0,
        -g,
        (-b_1 * abs(x[4] - np.pi / 2) * dx[4] - (b_2 + b_k * f_x5_x11) * (dx[4] - dx[10]) - k_k * h_x5_x11 + T_r1_y + T_r3_y) / I_1,
        0,
        -g,
        (-b_1 * abs(x[7] - np.pi / 2) * dx[7] - (b_2 + b_k * f_x8_x14) * (dx[7] - dx[13]) - k_k * h_x8_x14 + T_r2_y + T_r4_y) / I_1,
        F_g_1 / m_2,
        F_g_2 / m_2 - g,
        (-F_g_1 * l_2_2_sin_x11 - F_g_2 * l_2_2_cos_x11 - (b_2 + b_k * f_x5_x11) * (dx[10] - dx[4]) + k_k * h_x5_x11 - T_r3_y - T_r5_x_dx_y) / I_2,
        F_g_3 / m_2,
        F_g_4 / m_2 - g,
        (-F_g_3 * l_2_2_sin_x14 - F_g_4 * l_2_2_cos_x14 - (b_2 + b_k * f_x8_x14) * (dx[13] - dx[7]) + k_k * h_x8_x14 - T_r4_y - T_r6_x_dx_y) / I_2
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

    d2x = P_x *((C_x * P_x ) /(D_x_dx - C_x * Q_x_dx_y_F_g)) + Q_x_dx_y_F_g
    dx = dx + dt * d2x
    x = x + dt * dx

    force = np.array([F_g_1, F_g_2, F_g_3, F_g_4])

    return x, dx, force
