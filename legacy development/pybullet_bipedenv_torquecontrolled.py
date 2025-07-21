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
        self.init_no = 0
        if render_mode == 'human':
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        self.observe_mode = False
        self.scale = 1.
        self.dt = 1e-3
        # self.mu = 1.0
        # self.ground_kp=1e6
        # self.ground_kd=6e3
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.robot = p.loadURDF("assets/biped2d.urdf", [0,0,1.18], p.getQuaternionFromEuler([0.,0.,0.]),physicsClientId=self.physics_client)
        self.planeId = p.loadURDF("plane.urdf",physicsClientId=self.physics_client)
        self.leg_len = 0.94
        self.render_mode = render_mode
        self.joint_idx = [2,3,4,5,6,7,8]

        self.max_steps = int(3*(1/self.dt))
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)

    
        self.observation_space = spaces.Box(low=-50, high=50, shape=(20,), dtype=np.float32)
        self.t = 0
        self.gaitgen_net = SimpleFCNN()
        self.gaitgen_net.load_state_dict(torch.load('newnorm_final_hs512_lr0.0001_bs32_epochs10000.pth',weights_only=True))
        
        self.normalizationconst = np.load(rf"gait reference fft5.00/newnormalization_constants.npy")
        self.joint_no = p.getNumJoints(self.robot)
        self.max_torque = 500


    def reset(self,seed=None):
        self.t = 0
        self.init_no += 1
        self.reference_idx = 0
        p.resetSimulation(physicsClientId=self.physics_client)
        

        self.reference_speed = 0.2 + np.random.rand()*2.8
        self.ramp_angle = 0.0

        encoder_vec = np.empty((3))   # init_pos + speed + r_leglength + l_leglength + ramp_angle = 0
        encoder_vec[0] = self.reference_speed/3
        encoder_vec[1] = self.leg_len /1.5
        encoder_vec[2] = self.leg_len /1.5
        encoder_vec = torch.tensor(encoder_vec, dtype=torch.float32)    
        self.reference = self.findgait(encoder_vec)                     #Find the gait
        self.reference = np.clip(self.reference, -np.pi/2, np.pi/2)     #Clip the gait

        hip_init = (self.reference[0,0] - self.reference[0,2])/2
        knee_init = (self.reference[0,1] - self.reference[0,3])/2
        init_z = self.starting_height(hip_init,knee_init)
        p.resetSimulation(physicsClientId=self.physics_client)
        plane_orientation = p.getQuaternionFromEuler([self.ramp_angle, 0 , 0])
        self.planeId = p.loadURDF("plane.urdf",physicsClientId=self.physics_client, baseOrientation=plane_orientation)
        self.robot = p.loadURDF("assets/biped2d.urdf", [0,0,init_z+0.01], p.getQuaternionFromEuler([0.,0.,0.]))
        p.setJointMotorControlArray(self.robot,[0,1,2,3,4,5,6,7,8], p.VELOCITY_CONTROL, forces=[0,0,0,0,0,0,0,0,0])

        self.alive_weight = 0.2
        self.forward_weight = 0.2
        self.reference_weight = 0.2
        self.ankle_weight = 0.2
        
        # self.planeId = p.loadSDF("assets/plane_stadium.sdf",physicsClientId=self.physics_client)
        p.setGravity(0,0,-9.81)
        p.setTimeStep(self.dt)
        # p.changeDynamics(self.planeId, -1, lateralFriction=self.mu, contactStiffness=self.ground_kp, contactDamping=self.ground_kd)

        p.resetJointState(self.robot, 3, targetValue = hip_init)
        p.resetJointState(self.robot, 4, targetValue = knee_init)
        p.resetJointState(self.robot, 5, targetValue = - (hip_init + knee_init))
        p.resetJointState(self.robot, 6, targetValue = -hip_init)
        p.resetJointState(self.robot, 7, targetValue = knee_init)
        p.resetJointState(self.robot, 8, targetValue = - (-hip_init + knee_init))

        self.state, self.state_info = self.return_state()
        self.reset_info = {'current state':self.state, "state info":self.state_info}
        
        if self.render_mode == 'human':
            print(self.reference_speed, self.ramp_angle)
        return self.state, self.reset_info

    def step(self,torques):
        # Set torques
        self.t+=1
        torques = torques * self.max_torque
        p.setJointMotorControlArray(
            bodyIndex=self.robot,
            jointIndices=self.joint_idx,
            controlMode=p.TORQUE_CONTROL,
            forces=torques,
            physicsClientId=self.physics_client
        )
        # Step simulation
           
        p.stepSimulation()
        self.state, state_info = self.return_state()
        reward, done = self.biped_reward(self.state,torques=torques)
        truncated = False

        if self.t > self.max_steps:
            truncated = True
        if self.observe_mode:
            time.sleep(self.dt)
        return self.state, reward, done, truncated, state_info


    def biped_reward(self,x,torques):

        done = False
        reward = 0
        contact_points = p.getContactPoints(self.robot, self.planeId)
        # Conditions for early termination regarding stability

        if not contact_points:
            reward -=1  * self.alive_weight
        if len(contact_points) > 3:
            reward +=1  * self.ankle_weight
        
        if x[3] < 0:
            reward -= 1 * self.forward_weight

        if x[6] < -0.1:
            reward += (0.4 + x[6]) *self.forward_weight
        else:
            reward -= x[6]  * self.forward_weight

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

        self.reference_idx +=1
        if self.reference_idx > 20:
            rhip_diff = np.min(np.abs(x[7] - self.reference[self.reference_idx-10:self.reference_idx+10,0]))
            rknee_diff = np.min(np.abs(x[10] - self.reference[self.reference_idx-10:self.reference_idx+10,1]))
            lhip_diff = np.min(np.abs(x[13] - self.reference[self.reference_idx-10:self.reference_idx+10,2]))
            lknee_diff = np.min(np.abs(x[16] - self.reference[self.reference_idx-10:self.reference_idx+10,3]))
            reference_reward = self.ref_reward(rhip_diff, rknee_diff, lhip_diff, lknee_diff, x[9], x[12],x[7],x[10])
            reward += reference_reward

        reward -= np.mean(np.abs(torques)) * 1e-3
        reward -= np.abs(x[13]) * 1e-2
        reward -= np.abs(x[16]) * 1e-2
        reward -= np.abs(x[19]) * 1e-2

        reward += x[3]/10
        
        reward += 1/(1-np.exp(- (self.t/400)))

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
    
        # if rhip > 0.2:
        #     reward -= np.abs(rankle)* self.ankle_weight
        # if rhip < -0.4:
        #     reward -= rankle* self.ankle_weight
        # # if lhip > 0.2:
        # #     reward -= np.abs(lankle)* self.ankle_weight
        # if lhip < -0.4:
        #     reward -= lankle* self.ankle_weight

        return reward
    # def custom_reward(self,x,torques):
    #     pass
    # def init_state(self):
    #     start_idx = np.random.randint(0,1000)
    #     rhip_pos = self.reference[start_idx,0]
    #     rknee_pos = self.reference[start_idx,1]
    #     lhip_pos = self.reference[start_idx,2]
    #     lknee_pos = self.reference[start_idx,3]
    #     if np.abs(rknee_pos) < np.abs(lknee_pos):
    #         right_flat = True
    #         left_flat = False
    #     else:
    #         left_flat = True
    #         right_flat = False
        
    #     self.robot = p.loadURDF("assets/biped2d.urdf", [0,0,init_z+0.01], p.getQuaternionFromEuler([0.,0.,0.]))
    #     p.setJointMotorControlArray(self.robot,[0,1,2,3,4,5,6,7,8], p.VELOCITY_CONTROL, forces=[0,0,0,0,0,0,0,0,0])
    #     p.resetJointState(self.robot, 3, targetValue = rhip_pos)
    #     p.resetJointState(self.robot, 4, targetValue = rknee_pos)
    #     if left_flat:
    #         p.resetJointState(self.robot, 5, targetValue = 0)
    #     else:
    #         p.resetJointState(self.robot, 5, targetValue = -(rhip_pos+ rknee_pos))
    #     p.resetJointState(self.robot, 6, targetValue = lhip_pos)
    #     p.resetJointState(self.robot, 7, targetValue = lknee_pos)
    #     if right_flat:
    #         p.resetJointState(self.robot, 8, targetValue = 0)
    #     else:
    #         p.resetJointState(self.robot, 8, targetValue = -(lhip_pos+lknee_pos))

    #     self.alive_weight = 0.2
    #     self.forward_weight = 0.2
    #     self.reference_weight = 0.2
    #     self.ankle_weight = 0.2
        
    #     # self.planeId = p.loadSDF("assets/plane_stadium.sdf",physicsClientId=self.physics_client)
    #     p.setGravity(0,0,-9.81)
    #     p.setTimeStep(self.dt)
    #     # p.changeDynamics(self.planeId, -1, lateralFriction=self.mu, contactStiffness=self.ground_kp, contactDamping=self.ground_kd)

