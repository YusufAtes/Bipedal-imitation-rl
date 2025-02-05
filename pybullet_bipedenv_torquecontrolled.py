import os
import numpy as np
from scipy.signal import resample
import torch
from gait_generator_net import SimpleFCNN
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import pybullet as p
import pybullet_data
import time

dt=1/240
GRAVITY=-9.8
mu = 0.65
kp = np.array([ 0. ,  0.3,  0.2,  0.1,  0.3,  0.2,  0.1])
kd = 0.1*kp

class BipedEnv(gym.Env):
    def __init__(self,render=False, render_mode= None):

        if render_mode == 'human':
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        self.num_control_steps=1
        self.leg_len = 0.9 
        self.render_mode = render_mode
        # self.kp = kp
        # self.kd = kd
        self.joint_idx = [2,3,4,5,6,7,8]
        self.scale = 1.
        self.dt = dt
        self.mu = mu
        self.ground_kp=1e6
        self.ground_kd=6e3

        self.max_steps = int(3*1/dt)
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-500, high=500, shape=(20,), dtype=np.float32)
        self.t = 0
        self.dt = 0.01

        self.gaitgen_net = SimpleFCNN()
        self.gaitgen_net.load_state_dict(torch.load('model_hs512_lpmse_bs64_epoch1000_fft.pth',weights_only=True))
        
        self.normalizationconst = np.load(rf"normalization_constants.npy")
        self.robot = p.loadURDF("assets/biped2d.urdf", [0,0,1.185], p.getQuaternionFromEuler([0.,0.,0.]),physicsClientId=self.physics_client)
        self.planeId = p.loadURDF("assets/plane.urdf",physicsClientId=self.physics_client)
        p.changeDynamics(self.planeId, -1, lateralFriction=self.mu, contactStiffness=self.ground_kp, contactDamping=self.ground_kd)
        p.setGravity(0,0,-9.8)
        p.setTimeStep(self.dt)
        p.changeDynamics(self.planeId, -1, lateralFriction=1.0)
        self.joint_no = p.getNumJoints(self.robot)
        self.joint_indices = np.arange(2,9)
        self.max_torque = 300

    def reset(self,seed=None):

        p.resetSimulation(physicsClientId=self.physics_client)
        p.setTimeStep(self.dt)
        self.ramp_angle = np.random.uniform(-2, 2) * np.pi / 180 # Random angle between -3 and 3
        plane_orientation = p.getQuaternionFromEuler([self.ramp_angle, 0 , 0])
        self.robot = p.loadURDF("assets/biped2d.urdf", [0,0,1.185], p.getQuaternionFromEuler([0.,0.,0.]),physicsClientId=self.physics_client)
        self.planeId = p.loadURDF("assets/plane.urdf",physicsClientId=self.physics_client, baseOrientation=plane_orientation)
        # self.planeId = p.loadSDF("assets/plane_stadium.sdf",physicsClientId=self.physics_client)
        p.setGravity(0,0,-9.8)
        p.changeDynamics(self.planeId, -1, lateralFriction=self.mu, contactStiffness=self.ground_kp, contactDamping=self.ground_kd)
        p.changeDynamics(self.planeId, -1, lateralFriction=1.0)
        
        desired_speed = np.random.rand()*3
        self.reference_speed = desired_speed
        self.t = 0
        encoder_vec = np.empty((3))   # init_pos + speed + r_leglength + l_leglength + ramp_angle = 0
        encoder_vec[0] = self.reference_speed/3
        encoder_vec[1] = self.leg_len
        encoder_vec[2] = self.leg_len
        encoder_vec = torch.tensor(encoder_vec, dtype=torch.float32)    
        self.reference = self.findgait(encoder_vec)                     #Find the gait
        self.reference = np.clip(self.reference, -np.pi/2, np.pi/2)     #Clip the gait
        distances = np.linalg.norm(self.reference[:200], axis=1)
        closest_idx = np.argmin(distances)                              #Starting point of the gait
        self.reference = self.reference[closest_idx:closest_idx+self.max_steps+10,:]
        self.state, self.state_info = self.return_state()
        self.reset_info = {'current state':self.state, "state info":self.state_info}
        if self.render_mode == 'human':
            print(self.reference_speed, self.ramp_angle)
            self.reference_speed = 1.5
            self.ramp_angle = 0
        return self.state, self.reset_info

    def step(self,torques):

        torques = np.clip(torques, -1, 1)
        torques = torques * self.max_torque

        self.t+=1
        p.setJointMotorControlArray(
            bodyIndex=self.robot,
            jointIndices=self.joint_indices,
            controlMode=p.TORQUE_CONTROL,
            forces=torques,
            physicsClientId=self.physics_client
        )

        # Step simulation
        p.stepSimulation()
        #time.sleep(self.dt)
        self.state, state_info = self.return_state()
        reward, done = self.biped_reward(self.state,torques)
        truncated = False

        if self.t > self.max_steps:
            truncated = True

        return self.state, reward, done, truncated, state_info

    def biped_reward(self,x,torques):

        current_time = self.t*self.dt
        done = False
        reward = 0
        # Reward for staying close to the reference trajectory
        reward -= np.mean(np.linalg.norm(self.reference[self.t:self.t+10,:] - x[[7,8,10,11]]))
        # Reward to penalize falling
        forward_reward =  0.1 * x[3]  # Encourage moving forward
        reward += forward_reward
        if x[4] > 1.3:
            reward -= 100
            done = True
        elif x[4] < 0.9:
            reward -= 100
            done = True
        else:
            reward += 1
    
        # Reward for staying close to the reference speed
        if np.abs(x[3] - self.reference_speed*current_time) < 0.25:
            reward += 1
        return reward, done

    def close(self):
        self.physics_client.disconnect()
        print("Environment closed")

    # def y_g(self,xrl, rampAngle):
    #     """
    #     Ground pattern function.
        
    #     Parameters:
    #         xrl: Input value (e.g., distance or position).
    #         rampAngle: Angle of the ramp in degrees.
        
    #     Returns:
    #         output: The calculated ground pattern value.
    #     """
    #     # Calculate the ground pattern value
    #     output = np.sin(np.pi * rampAngle / 180) * xrl
    #     return output

    # def h(self,input):
    #     if input > 0:
    #         return 1
    #     else:
    #         return 0    

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
        pred_time = padded_time[56:-56,:]
        #upsample from 100 hz to 240 hz
        if self.dt < 0.01:
            num_samples = int(400 * 2.4)  # 960
            # Upsample using Fourier method
            pred_time = resample(pred_time, num_samples, axis=0)

        pred_time = np.repeat(pred_time, 10, axis=0)

        return pred_time

    def return_state(self):
        link_state = p.getLinkState(self.robot, 2,computeLinkVelocity=True)          #link index 2 is for torso
        (pos_x,pos_y,pos_z) = link_state[0]                #3D position of the link
        y_vel = link_state[6][1]                           #y velocity of the link

        init_states = p.getJointStates(self.robot, np.arange(0,self.joint_no))
        init_states = init_states[2:]                       #First two joints are external joints (3rd is torso and so on) 
        self.state = np.zeros(len(init_states)*2+6)

        self.state[0] = self.reference_speed
        self.state[1] = self.ramp_angle
        self.state[2] = pos_x
        self.state[3] = pos_y
        self.state[4] = pos_z
        self.state[5] = y_vel
        state_info = {0:"reference_speed",
                      1:"ramp_angle",
                      2:"pos_x",
                      3:"pos_y",
                      4:"pos_z",
                      5:"y_vel",
                      6:"torso_pos",
                      7:"rhip_pos",
                      8:"rknee_pos",
                      9:"rankle_pos",
                      10:"lhip_pos",
                      11:"lknee_pos",
                      12:"lankle_pos",
                      13:"torso_vel",
                      14:"rhip_vel",
                      15:"rknee_vel",
                      16:"rangle_vel",
                      17:"lhip_vel",
                      18:"lknee_vel",
                      19:"lankle_vel"}
        for i in range(len(init_states)):
            self.state[i+6] = init_states[i][0]
            self.state[i+6+len(init_states)] = init_states[i][1]
        return self.state, state_info
