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
from animatebiped import animate_biped
from scipy.interpolate import interp1d

class BipedEnv(gym.Env):
    def __init__(self,render=False, render_mode= None):
        self.p = p
        self.init_no = 0
        if render_mode == 'human':
            self.physics_client = self.p.connect(self.p.GUI)
        else:
            self.physics_client = self.p.connect(self.p.DIRECT)
        self.observe_mode = False
        self.scale = 1.
        self.dt = 1e-3

        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.robot = self.p.loadURDF("assets/biped2d.urdf", [0,0,1.18], self.p.getQuaternionFromEuler([0.,0.,0.]),physicsClientId=self.physics_client)
        self.planeId = self.p.loadURDF("plane.urdf",physicsClientId=self.physics_client)
        self.leg_len = 0.94
        self.render_mode = render_mode
        self.joint_idx = [2,3,4,5,6,7,8]

        self.max_steps = int(3*(1/self.dt))
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-50, high=50, shape=(61,), dtype=np.float32)

        self.t = 0
        self.gaitgen_net = SimpleFCNN()
        self.gaitgen_net.load_state_dict(torch.load('newnorm_final_hs512_lr0.0001_bs32_epochs10000.pth',weights_only=True))
        
        self.normalizationconst = np.load(rf"gait reference fft5.00/newnormalization_constants.npy")
        self.joint_no = self.p.getNumJoints(self.robot)
        self.max_torque = 500

        self.kp = np.array([ 0.1 ,  0.3,  0.2,  0.1,  0.3,  0.2,  0.1])
        self.kd = 0.1*self.kp
        self.state = np.zeros(61)
        self.update_const = 0.7
    def reset(self,seed=None):
        self.max_steps = int(3*(1/self.dt))
        self.t = 0
        self.init_no += 1
        self.p.resetSimulation(physicsClientId=self.physics_client)
        
        self.reference_speed = 0.4 + np.random.rand()*2.6
        self.ramp_angle = 0.0

        encoder_vec = np.empty((3))   # init_pos + speed + r_leglength + l_leglength + ramp_angle = 0
        encoder_vec[0] = self.reference_speed/3
        encoder_vec[1] = self.leg_len /1.5
        encoder_vec[2] = self.leg_len /1.5
        encoder_vec = torch.tensor(encoder_vec, dtype=torch.float32)    
        self.reference = self.findgait(encoder_vec)                     #Find the gait
        self.reference = np.clip(self.reference, -np.pi/2, np.pi/2)     #Clip the gait

        plane_orientation = self.p.getQuaternionFromEuler([self.ramp_angle, 0 , 0])
        self.planeId = self.p.loadURDF("plane.urdf",physicsClientId=self.physics_client, baseOrientation=plane_orientation)


        self.reset_info = {'current state':self.state}
        self.past_action_error = np.zeros(7)
        self.current_action = np.zeros(7)
        self.target_action = np.zeros(7)
        self.past_target_action = np.zeros(7)
        self.past2_target_action = np.zeros(7)

        self.control_freq = 10
        if self.render_mode == 'human':
            print(self.reference_speed, self.ramp_angle)
        self.init_state()
        self.state, self.state_info = self.return_state()
        return self.state, self.reset_info

    def step(self,torques):
        # Set torques
        self.target_action = torques * self.max_torque
        for i in range(10):
            self.current_action = self.update_const*self.target_action + (1-self.update_const)*self.current_action
            self.t+=1
            self.p.setJointMotorControlArray(
                bodyIndex=self.robot,
                jointIndices=self.joint_idx,
                controlMode=self.p.TORQUE_CONTROL,
                forces=self.current_action,
                physicsClientId=self.physics_client
            )
            # Step simulation
            
            self.p.stepSimulation()
            
            reward, done = self.biped_reward(self.state,torques=self.target_action)
            truncated = False

            if self.t > self.max_steps:
                truncated = True
            if self.render_mode == 'human':
                time.sleep(self.dt)
            
        self.past_target_action = self.target_action
        self.past2_target_action = self.past_target_action
        self.state, state_info = self.return_state()
        return self.state, reward, done, truncated, state_info

    def biped_reward(self,x,torques):

        self.alive_weight = 0.05
        self.forward_weight = 0.05
        self.ankle_weight = 0.1
        done = False
        reward = 0
        contact_points = self.p.getContactPoints(self.robot, self.planeId)
        # Conditions for early termination regarding stability

        if not contact_points:
            reward -=1  * self.alive_weight
        if len(contact_points) > 0:
            reward +=1  * self.alive_weight * len(contact_points)
        
        if x[3] < 0:
            reward -= 1 * self.forward_weight

        if (x[7] > 0.15) and (x[10] > 0.15):
            reward -=1  * self.alive_weight
        elif (x[7]) < -0.15 and (x[10] < -0.15):
            reward -=1  * self.alive_weight
        else:
            reward += 1 * self.alive_weight

        if x[4] > 1.4:
            reward -=1
            done = True
        elif x[4] < 0.9:
            reward -=1
            done = True
        elif x[4] < 1.085:
            reward -= 1 * self.alive_weight
        else:
            reward += 1 * self.alive_weight

        joint_pos = x[[7,8,10,11]]
        ref_pos = x[[41,42,43,44]]
        reward += np.exp(-np.linalg.norm(joint_pos - ref_pos)) 

        joint_vel = x[[35,36,38,39]]
        ref_vel = x[[57,58,59,60]]
        reward += 0.1* np.exp(-0.1*np.linalg.norm(joint_vel - ref_vel))

        reward -= np.abs(x[6] +0.2)  * self.alive_weight
        reward -= 1e-3 * np.abs(x[[37]])
        reward -= 1e-3 * np.abs(x[[40]])
        reward -= 1e-3 * np.mean(np.abs(torques))

        reward += x[3]/10
        
        reward += 1/(1-np.exp(- (self.t/400)))
        if (np.abs(x[7]) < 0.2) and (np.abs(x[10]) < 0.2):
            reward -= (np.abs(x[9] + x[8] + x[7]) + np.abs(x[10] + x[11] + x[12])) * self.ankle_weight
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

    # def starting_height(self,rhip_pos,rknee_pos,lhip_pos,lknee_pos,r_flat):
    #     upper_len = 0.45
    #     lower_len = 0.45
    #     foot_len = 0.185
    #     if r_flat:
    #         hip_short = upper_len - (upper_len * np.cos(rhip_pos))
    #         knee_short = lower_len - (lower_len * np.cos(rhip_pos + rknee_pos))
    #         init_pos = 1.185 - hip_short - knee_short 
    #     else:
    #         hip_short = upper_len - (upper_len * np.cos(lhip_pos))
    #         knee_short = lower_len - (lower_len * np.cos(lhip_pos + lknee_pos))
    #         init_pos = 1.185 - hip_short - knee_short  
    #     return init_pos
    def starting_height(self,hip_init,knee_init):
        upper_len = 0.45
        lower_len = 0.45
        foot_len = 0.185

        hip_short = upper_len - (upper_len * np.cos(hip_init))
        knee_short = lower_len - (lower_len * np.cos(hip_init + knee_init))
        init_pos = 1.185 - hip_short - knee_short 
        return init_pos
    def ref_reward(self,rhip_diff, rknee_diff, lhip_diff, lknee_diff, rankle, lankle,rhip,lhip):
        reward = 0
        if rhip_diff < 0.26:
            reward += 0.3- rhip_diff
        else:
            reward -= 1 * self.reference_weight

        if rknee_diff < 0.26:
            reward += 0.3- rknee_diff
        else:
            reward -= 1* self.reference_weight

        if lhip_diff < 0.26:
            reward += 0.3- lhip_diff
        else:
            reward -= 1* self.reference_weight
        
        if lknee_diff < 0.26:
            reward += 0.3- lknee_diff
        else:
            reward -= 1* self.reference_weight


        return reward
    def custom_reward(self,x,torques):
        pass
    def init_state(self):

        start_idx = 0
        self.max_steps = self.max_steps - start_idx
        self.reference_idx = start_idx

        rhip_pos = self.reference[start_idx,0]
        rknee_pos = self.reference[start_idx,1]
        lhip_pos = self.reference[start_idx,2]
        lknee_pos = self.reference[start_idx,3]
        hip_init = (rhip_pos - lhip_pos)/2
        knee_init = (rknee_pos - lknee_pos)/2

        # if np.abs(rknee_pos) < np.abs(lknee_pos):
        #     right_flat = True
        #     left_flat = False
        # else:
        #     left_flat = True
        #     right_flat = False

        init_z = self.starting_height(hip_init,knee_init)

        self.robot = self.p.loadURDF("assets/biped2d.urdf", [0,0,init_z+0.01], self.p.getQuaternionFromEuler([0.,0.,0.]))
        self.p.setJointMotorControlArray(self.robot,[0,1,2,3,4,5,6,7,8], self.p.VELOCITY_CONTROL, forces=[0,0,0,0,0,0,0,0,0])
        self.p.resetJointState(self.robot, 3, targetValue = hip_init)
        self.p.resetJointState(self.robot, 4, targetValue = knee_init)
        self.p.resetJointState(self.robot, 5, targetValue = -(hip_init + knee_init))
        # if left_flat:
        #     self.p.resetJointState(self.robot, 5, targetValue = 0)
        # else:
        #     self.p.resetJointState(self.robot, 5, targetValue = -(rhip_pos+ rknee_pos))

        self.p.resetJointState(self.robot, 6, targetValue = -hip_init)
        self.p.resetJointState(self.robot, 7, targetValue = knee_init)
        self.p.resetJointState(self.robot, 8, targetValue = -(-hip_init + knee_init))

        # if right_flat:
        #     self.p.resetJointState(self.robot, 8, targetValue = 0)
        # else:
        #     self.p.resetJointState(self.robot, 8, targetValue = -(lhip_pos+lknee_pos))

        self.p.setGravity(0,0,-9.81)
        self.p.setTimeStep(self.dt)

        self.t1_torso_pos = self.p.getJointState(self.robot, 2)[0]
        self.t1_rhip_pos = self.p.getJointState(self.robot, 3)[0]
        self.t1_rknee_pos = self.p.getJointState(self.robot, 4)[0]
        self.t1_rankle_pos = self.p.getJointState(self.robot, 5)[0]
        self.t1_lhip_pos = self.p.getJointState(self.robot, 6)[0]
        self.t1_lknee_pos = self.p.getJointState(self.robot, 7)[0]
        self.t1_lankle_pos = self.p.getJointState(self.robot, 8)[0]

    # def pd_controller(self,i):
    #     action_error = self.target_action - self.current_action
    #     if i == 0:
    #         derivative = 0
    #     else:
    #         derivative = (action_error - self.past_action_error) / self.dt
    #     self.current_action = self.kp*action_error + self.kd* derivative
    #     self.past_action_error = action_error
    #     return self.current_action

    def return_state(self):

        link_state = self.p.getLinkState(self.robot, 2,computeLinkVelocity=True)          #link index 2 is for torso
        (pos_x,pos_y,pos_z) = link_state[0]                #3D position of the link
        y_vel = link_state[6][1]                           #y velocity of the link

        self.torso_pos = self.p.getJointState(self.robot, 2)[0]
        self.rhip_pos = self.p.getJointState(self.robot, 3)[0]
        self.rknee_pos = self.p.getJointState(self.robot, 4)[0]
        self.rankle_pos = self.p.getJointState(self.robot, 5)[0]
        self.lhip_pos = self.p.getJointState(self.robot, 6)[0]
        self.lknee_pos = self.p.getJointState(self.robot, 7)[0]
        self.lankle_pos = self.p.getJointState(self.robot, 8)[0]

        self.torso_vel = self.p.getJointState(self.robot, 2)[1]
        self.rhip_vel = self.p.getJointState(self.robot, 3)[1]
        self.rknee_vel = self.p.getJointState(self.robot, 4)[1]
        self.rankle_vel = self.p.getJointState(self.robot, 5)[1]
        self.lhip_vel = self.p.getJointState(self.robot, 6)[1]
        self.lknee_vel = self.p.getJointState(self.robot, 7)[1]
        self.lankle_vel = self.p.getJointState(self.robot, 8)[1]

        ref_rhip_vel = (self.reference[self.reference_idx+self.t,0] - self.reference[self.reference_idx+self.t-1,0])/self.dt
        ref_rknee_vel = (self.reference[self.reference_idx+self.t,1] - self.reference[self.reference_idx+self.t-1,1])/self.dt
        ref_lhip_vel = (self.reference[self.reference_idx+self.t,2] - self.reference[self.reference_idx+self.t-1,2])/self.dt
        ref_lknee_vel = (self.reference[self.reference_idx+self.t,3] - self.reference[self.reference_idx+self.t-1,3])/self.dt

        self.state[0] = self.reference_speed
        self.state[1] = self.ramp_angle
        self.state[2] = pos_x
        self.state[3] = pos_y
        self.state[4] = pos_z
        self.state[5] = y_vel

        self.state[6:13] = [self.torso_pos, self.rhip_pos, self.rknee_pos, self.rankle_pos, self.lhip_pos, self.lknee_pos, self.lankle_pos]
        self.state[13:20] = [self.past_target_action[0]/self.max_torque, self.past_target_action[1]/self.max_torque, self.past_target_action[2]/self.max_torque, 
                             self.past_target_action[3]/self.max_torque, self.past_target_action[4]/self.max_torque, self.past_target_action[5]/self.max_torque, 
                             self.past_target_action[6]/self.max_torque]
        self.state[20:27] = [self.t1_torso_pos, self.t1_rhip_pos, self.t1_rknee_pos, self.t1_rankle_pos, self.t1_lhip_pos, self.t1_lknee_pos, self.t1_lankle_pos]
        self.state[27:34] = [self.past2_target_action[0]/self.max_torque, self.past2_target_action[1]/self.max_torque, self.past2_target_action[2]/self.max_torque,
                             self.past2_target_action[3]/self.max_torque, self.past2_target_action[4]/self.max_torque, self.past2_target_action[5]/self.max_torque,
                             self.past2_target_action[6]/self.max_torque]
        self.state[34:41] = [self.torso_vel, self.rhip_vel, self.rknee_vel, self.rankle_vel, self.lhip_vel, self.lknee_vel, self.lankle_vel]

        self.state[41:45] = [self.reference[self.reference_idx+self.t,0], self.reference[self.reference_idx+self.t,1], 
                             self.reference[self.reference_idx+self.t,2], self.reference[self.reference_idx+self.t,3]]
        
        self.state[45:49] = [self.reference[self.reference_idx+self.t+10,0], self.reference[self.reference_idx+self.t+10,1],
                                self.reference[self.reference_idx+self.t+10,2], self.reference[self.reference_idx+self.t+10,3]]
        
        self.state[49:53] = [self.reference[self.reference_idx+self.t+50,0], self.reference[self.reference_idx+self.t+50,1],
                                self.reference[self.reference_idx+self.t+50,2], self.reference[self.reference_idx+self.t+50,3]]
        
        self.state[53:57] = [self.reference[self.reference_idx+self.t+100,0], self.reference[self.reference_idx+self.t+100,1],
                                self.reference[self.reference_idx+self.t+100,2], self.reference[self.reference_idx+self.t+100,3]]
        
        self.state[57:61] = [ref_rhip_vel, ref_rknee_vel, ref_lhip_vel, ref_lknee_vel]

        self.t1_torso_pos = self.torso_pos
        self.t1_rhip_pos = self.rhip_pos
        self.t1_rknee_pos = self.rknee_pos
        self.t1_rankle_pos = self.rankle_pos
        self.t1_lhip_pos = self.lhip_pos
        self.t1_lknee_pos = self.lknee_pos
        self.t1_lankle_pos = self.lankle_pos

        state_info = {0:"reference_speed",
                    1:"ramp_angle", 2:"pos_x", 3:"pos_y", 4:"pos_z", 5:"y_vel",

                    6:"torso_pos", 7:"rhip_pos", 8:"rknee_pos", 9:"rankle_pos", 10:"lhip_pos", 11:"lknee_pos", 12:"lankle_pos",
                    #self.past_target_action
                    13:"t1_torso_action", 14:"t1_rhip_action", 15:"t1_rknee_action", 16:"t1_rankle_action", 17:"t1_lhip_action", 18:"t1_lknee_action", 19:"t1_lankle_action",

                    20:"t1torso_pos", 21:"t1rhip_pos", 22:"t1rknee_pos", 23:"t1rankle_pos", 24:"t1lhip_pos", 25:"t1lknee_pos", 26:"t1lankle_pos",
                    #self.past2_target_action
                    27:"t2_torso_action", 28:"t2_rhip_action", 29:"t2_rknee_action", 30:"t2_rankle_action", 31:"t2_lhip_action", 32:"t2_lknee_action", 33:"t2_lankle_action",

                    34:"torso_vel", 35:"rhip_vel", 36:"rknee_vel", 37:"rankle_vel", 38:"lhip_vel", 39:"lknee_vel", 40:"lankle_vel",
                    
                    41:"t1_ref_rhip", 42:"t1_ref_rknee", 43:"t1_ref_lhip", 44:"t1_ref_lknee",

                    45:"t2_ref_rhip", 46:"t2_ref_rknee", 47:"t2_ref_lhip", 48:"t2_ref_lknee",

                    49:"t50_ref_rhip", 50:"t50_ref_rknee", 51:"t50_ref_lhip", 52:"t50_ref_lknee",

                    53:"t100_ref_rhip", 54:"t100_ref_rknee", 55:"t100_ref_lhip", 56:"t100_ref_lknee",

                    57:'ref_rhip_vel', 58:'ref_rknee_vel', 59:'ref_lhip_vel', 60:'ref_lknee_vel'}

        return self.state, state_info