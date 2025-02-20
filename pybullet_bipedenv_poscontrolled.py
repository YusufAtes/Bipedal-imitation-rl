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


class POS_Biped(gym.Env):
    def __init__(self,render=False, render_mode= None):
        self.init_no = 0
        if render_mode == 'human':
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        self.scale = 1.
        self.dt = 1e-3
        self.mu = 0.8
        self.ground_kp=1e5
        self.ground_kd=6e3
        
        self.robot = p.loadURDF("assets/biped2d_revolute.urdf", [0,0,1.18], p.getQuaternionFromEuler([0.,0.,0.]),physicsClientId=self.physics_client)
        self.planeId = p.loadURDF("assets/plane.urdf",physicsClientId=self.physics_client)
        p.changeDynamics(self.planeId, -1, lateralFriction=self.mu, contactStiffness=self.ground_kp, contactDamping=self.ground_kd)
        p.setGravity(0,0,-9.81)
        p.setTimeStep(self.dt)
        self.leg_len = 0.9 
        self.render_mode = render_mode
        self.joint_idx = [2,3,4,5,6,7,8]
        self.max_steps = int(3*(1/self.dt))

        self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(7,), dtype=np.float32)
    
        self.observation_space = spaces.Box(low=-50, high=50, shape=(20,), dtype=np.float32)
        self.t = 0
        self.gaitgen_net = SimpleFCNN()
        self.gaitgen_net.load_state_dict(torch.load('ref_gait_results/tagamodel_hs512_lr0.0001_bs32_epochs10000_val0.0139.pth',weights_only=True))
        
        self.normalizationconst = np.load(rf"gait reference fft_5.00/newnormalization_constants.npy")
        self.joint_no = p.getNumJoints(self.robot)
        self.max_torque = 1000

    def reset(self,seed=None):
        self.t = 0
        self.init_no += 1
        self.reference_idx = 1

        self.alive_weight = 0.3
        self.forward_weight = 0.2
        self.reference_weight = 0.4
        self.hip_weight = 0.1

        self.reference_speed = np.random.rand()*3
        self.ramp_angle = np.random.uniform(-1, 1) * np.pi / 180 # Random angle between -3 and 3

        encoder_vec = np.empty((3))   # init_pos + speed + r_leglength + l_leglength + ramp_angle = 0
        encoder_vec[0] = self.reference_speed/3
        encoder_vec[1] = self.leg_len /1.5
        encoder_vec[2] = self.leg_len /1.5
        encoder_vec = torch.tensor(encoder_vec, dtype=torch.float32)    
        self.reference = self.findgait(encoder_vec)                     #Find the gait
        self.reference = np.clip(self.reference, -np.pi/2, np.pi/2)     #Clip the gait

        init_z = self.starting_height()
        p.resetSimulation(physicsClientId=self.physics_client)
        plane_orientation = p.getQuaternionFromEuler([self.ramp_angle, 0 , 0])

        self.robot = p.loadURDF("assets/biped2d.urdf", [0,0,init_z+0.03], p.getQuaternionFromEuler([0.,0.,0.]),physicsClientId=self.physics_client)
        self.planeId = p.loadURDF("assets/plane.urdf",physicsClientId=self.physics_client, baseOrientation=plane_orientation)

        p.setGravity(0,0,-9.81)
        p.setTimeStep(self.dt)
        p.changeDynamics(self.planeId, -1, lateralFriction=self.mu, contactStiffness=self.ground_kp, contactDamping=self.ground_kd)
        
        p.resetJointState(self.robot, 3, targetValue = self.reference[0,0])
        p.resetJointState(self.robot, 4, targetValue = self.reference[0,1])
        p.resetJointState(self.robot, 6, targetValue = self.reference[0,2])
        p.resetJointState(self.robot, 7, targetValue = self.reference[0,3])

        self.state, self.state_info = self.return_state()
        self.reset_info = {'current state':self.state, "state info":self.state_info}
        
        if self.render_mode == 'human':
            print(self.reference_speed, self.ramp_angle)
        return self.state, self.reset_info

    def step(self,x):
        self.t+=1
        
        p.setJointMotorControlArray(
            bodyIndex=self.robot,
            jointIndices=self.joint_idx,
            controlMode=p.POSITION_CONTROL,
            targetPositions=x,
            physicsClientId=self.physics_client
        )
        p.stepSimulation()
        self.state, state_info = self.return_state()
        reward, done = self.biped_reward(self.state)
        truncated = False

        if self.t > self.max_steps:
            truncated = True

        return self.state, reward, done, truncated, state_info


    def biped_reward(self,x):
        done = False
        reward = 0

        # Conditions for early termination regarding stability
        if x[4] > 1.05:
            reward += self.alive_weight
        else:
            reward -= self.alive_weight
            done = True
        if x[3] < -0.5:
            reward -= self.alive_weight
            done = True
        if (x[7] > 0.2 and x[10] > 0.2):
            reward -= self.alive_weight
            done = True
        if (x[7] < -0.2 and x[10] < -0.2):
            reward -= self.alive_weight
            done = True
        #Condition for forward movement
        if x[5] > 1e-3:
            reward +=self.forward_weight
        else:
            reward -=self.forward_weight
        #Reference tracking indexing
        if self.t%10 == 0:
            self.reference_idx +=1

        reference_diff = np.linalg.norm(self.reference[self.reference_idx,:] - x[[7,8,10,11]])
        #Condition for reference tracking
        if reference_diff < self.reference_weight:
            reward += (self.reference_weight - reference_diff)
        elif reference_diff < 0.7:
            reward -= (reference_diff - self.reference_weight)
        else:
            reward -= self.reference_weight

        if np.abs(x[7]) > 0.785:
            reward -= self.hip_weight
        if np.abs(x[10]) > 0.785:
            reward -= self.hip_weight
        else:
            reward += self.hip_weight
        return reward, done
    
    def close(self):
        self.physics_client.disconnect()
        print("Environment closed")

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
    
    # def denormalize(self,pred):

    #     pred[0,:,0] = pred[0,:,0] * self.normalizationconst[0]
    #     pred[0,:,1] = pred[0,:,1] * self.normalizationconst[1]
    #     pred[0,:,2] = pred[0,:,2] * self.normalizationconst[2]
    #     pred[0,:,3] = pred[0,:,3] * self.normalizationconst[3]
    #     pred[1,:,0] = pred[1,:,0] * self.normalizationconst[4]
    #     pred[1,:,1] = pred[1,:,1] * self.normalizationconst[5]
    #     pred[1,:,2] = pred[1,:,2] * self.normalizationconst[6]
    #     pred[1,:,3] = pred[1,:,3] * self.normalizationconst[7]
        
    #     return pred
        
    def pred_ifft(self,predictions):
        #form is [5,2,17]
        real_pred = predictions[:,0,:]
        imag_pred = predictions[:,1,:]
        predictions = real_pred + 1j*imag_pred

        pred_time = np.fft.irfft(predictions, axis=1)
        pred_time = pred_time.transpose(1,0)
        org_rate = 10

        if self.dt < 0.1:
            num_samples = int((pred_time.shape[0]) * (1/self.dt)/(org_rate*10))  # resample with self.dt
            # Upsample using Fourier method
            pred_time = resample(pred_time, num_samples, axis=0)
        # pred_time = np.tile(pred_time, (int(self.max_steps*self.dt),1))    # Create loop for reference movement
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
                      16:"rankle_vel",
                      17:"lhip_vel",
                      18:"lknee_vel",
                      19:"lankle_vel"}
        for i in range(len(init_states)):
            self.state[i+6] = init_states[i][0]
            self.state[i+6+len(init_states)] = init_states[i][1]
        return self.state, state_info
    def starting_height(self):
        upper_len = 0.45
        lower_len = 0.45
        foot_len = 0.185

        hip_short = upper_len - (upper_len * np.cos(self.reference[0,0]))
        knee_short = lower_len - (lower_len * np.cos(self.reference[0,1] + self.reference[0,0]))
        foot_exten = (foot_len/2) * np.sin(self.reference[0,1] + self.reference[0,0])

        init_pos = 1.185 - hip_short - knee_short + foot_exten
        return init_pos