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

    def __init__(self,render=False, render_mode= None, demo_mode=False):
        super().__init__()
        self.p = p
        self.init_no = 0
        if render_mode == 'human':
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        self.observe_mode = False
        self.scale = 1.
        self.dt = 1e-3
        self.demo_mode = demo_mode
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.robot = p.loadURDF("assets/biped2d.urdf", [0,0,1.195], p.getQuaternionFromEuler([0.,0.,0.]),physicsClientId=self.physics_client)
        self.planeId = p.loadURDF("plane.urdf",physicsClientId=self.physics_client)
        self.leg_len = 0.94
        self.render_mode = render_mode
        self.joint_idx = [3,4,5,6,7,8]

        self.max_steps = int(3*(1/self.dt))
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-50, high=50, shape=(54,), dtype=np.float32)

        self.t = 0
        self.gaitgen_net = SimpleFCNN()
        self.gaitgen_net.load_state_dict(torch.load('newnorm_final_hs512_lr0.0001_bs32_epochs10000.pth',weights_only=True))
        
        self.normalizationconst = np.load(rf"gait reference fft5.00/newnormalization_constants.npy")
        self.joint_no = p.getNumJoints(self.robot)
        self.max_torque = np.array([1000,500,250,1000,500,250])
        # self.max_torque = 500

        self.state = np.zeros(54)
        self.update_const = 0.8
        self.velocity_norrmcoeff = 10.0
        self.pos_normcoeff = np.pi
        self.torque_normcoeff = 1000
        self.pos_noise = 0.003
        self.vel_noise = 0.003

    def reset(self,seed=None,test_speed = None, test_angle = None,demo_max_steps = None, ground_noise = None, ground_resolution = None):
        
        self.test_speed = test_speed
        self.test_angle = test_angle
        self.max_steps = int(3*(1/self.dt))
        self.t = 0
        self.init_no += 1
        p.resetSimulation(physicsClientId=self.physics_client)
        
        self.reference_speed = 0.1 + np.random.rand()*2.9
        self.ramp_angle = np.random.uniform(-5,5) *np.pi / 180

        if demo_max_steps:
            self.max_steps = demo_max_steps
        if self.test_speed is not None:
            self.reference_speed = self.test_speed

        if self.test_angle is not None:
            self.ramp_angle = self.test_angle

        encoder_vec = np.empty((3))   # init_pos + speed + r_leglength + l_leglength + ramp_angle = 0
        encoder_vec[0] = self.reference_speed/3
        encoder_vec[1] = self.leg_len /1.5
        encoder_vec[2] = self.leg_len /1.5
        encoder_vec = torch.tensor(encoder_vec, dtype=torch.float32)    
        self.reference = self.findgait(encoder_vec)                     #Find the gait
        self.reference = np.clip(self.reference, -np.pi/2, np.pi/2)     #Clip the gait

        plane_orientation = p.getQuaternionFromEuler([self.ramp_angle, 0 , 0])
        if self.demo_mode:
            if ground_noise:
                self.ground_noise = ground_noise
                self.init_noisy_plane(noise_level=ground_noise, num_rows=128, num_columns=1024, mesh_scale=[ground_resolution, ground_resolution, 1.0],
                                                    baseOrientation=plane_orientation)
            else:
                self.planeId = p.loadURDF("plane.urdf",physicsClientId=self.physics_client, baseOrientation=plane_orientation)
        else:
            self.planeId = p.loadURDF("plane.urdf",physicsClientId=self.physics_client, baseOrientation=plane_orientation)


        self.reset_info = {'current state':self.state}
        self.past_action_error = np.zeros(6)
        self.current_action = np.zeros(6)
        self.target_action = np.zeros(6)
        self.past_target_action = np.zeros(6)
        self.past2_target_action = np.zeros(6)
        p.setGravity(0,0,-9.81)
        p.setTimeStep(self.dt)
        self.control_freq = 10
        self.init_state()
        self.return_state()

        return self.state, self.reset_info

    def step(self,torques):
        # Set torques
        self.target_action = torques * self.max_torque
        for i in range(10):
            self.current_action = self.update_const*self.target_action + (1-self.update_const)*self.current_action 
            self.t+=1
            p.setJointMotorControlArray(
                bodyIndex=self.robot,
                jointIndices=self.joint_idx,
                controlMode=p.TORQUE_CONTROL,
                forces=self.current_action,
                physicsClientId=self.physics_client
            )
            # Step simulation
            p.stepSimulation()

        self.past_target_action = self.target_action
        self.past2_target_action = self.past_target_action
        self.return_state()

        truncated = False
        
        if self.t > self.max_steps:
            truncated = True

        if self.demo_mode:
            reward, done, contact_points = self.biped_reward(self.state,torques=self.current_action)
            
            return self.state, reward, done, truncated, self.state_info, contact_points
        else:
            reward, done = self.biped_reward(self.state,torques=self.current_action)
            return self.state, reward, done, truncated, self.state_info

    def biped_reward(self,x,torques):

        self.alive_weight = 0.5
        self.contact_weight = 0.15
        done = False
        reward = 0
        contact_points = p.getContactPoints(self.robot, self.planeId)
        # Conditions for early termination regarding stability

        if not contact_points:
            reward -=1  * self.contact_weight
        if x[8] < -1:
            reward -=1  * self.contact_weight
        if x[11] < -1:
            reward -=1  * self.contact_weight
        else:
            reward +=len(contact_points)  * self.contact_weight
        
        # if x[3] < 0:
        #     reward -= 1 * self.forward_weight

        # if (x[7] > 0.15) and (x[10] > 0.15):
        #     reward -=1  * self.alive_weight
        # elif (x[7] < -0.15) and (x[10] < -0.15):
        #     reward -=1  * self.alive_weight
        # else:
        #     reward += 1 * self.alive_weight

        if self.external_states[2] > 1.4 + np.tan(self.ramp_angle) * self.external_states[1]:
            reward -=10
            done = True
        elif self.external_states[2] > 1.2 + np.tan(self.ramp_angle) * self.external_states[1]:
            reward -= self.alive_weight
        elif self.external_states[2] < 0.8 + np.tan(self.ramp_angle) * self.external_states[1]:
            reward -=10
            done = True
        elif self.external_states[2] < 1.0+ np.tan(self.ramp_angle) * self.external_states[1]:
            reward -= 1 * self.alive_weight
        else:
            reward += 1 * self.alive_weight

        # if x[[7]] > 0.35 and x[[10]] > 0.35:
        #     reward -= 10
        #     done = True
        # elif x[[7]] < -0.35 and x[[10]] < -0.35:
        #     reward -= 10
        #     done = True

        hip_joint_pos = x[[6,9]] *self.pos_normcoeff
        hip_ref_pos = x[[30,33]] *self.pos_normcoeff
        reward += np.exp(-4*np.linalg.norm(hip_joint_pos - hip_ref_pos))

        knee_joint_pos = x[[7,10]] *self.pos_normcoeff
        knee_ref_pos = x[[31,34]] *self.pos_normcoeff
        reward += np.exp(-4*np.linalg.norm(knee_joint_pos - knee_ref_pos))

        ankle_joint_pos = x[[8,11]] *self.pos_normcoeff
        ankle_ref_pos = x[[32,35]] *self.pos_normcoeff
        reward += np.exp(-4*np.linalg.norm(ankle_joint_pos - ankle_ref_pos))

        hip_joint_vel = x[[24,27]] * self.velocity_norrmcoeff
        hip_ref_vel = x[[48,51]] * self.velocity_norrmcoeff
        reward += 0.4*np.exp(-0.1*np.linalg.norm(hip_joint_vel - hip_ref_vel))

        knee_joint_vel = x[[25,28]] * self.velocity_norrmcoeff
        knee_ref_vel = x[[49,52]] * self.velocity_norrmcoeff
        reward += 0.4*np.exp(-0.1*np.linalg.norm(knee_joint_vel - knee_ref_vel))

        ankle_joint_vel = x[[26,29]] * self.velocity_norrmcoeff
        ankle_ref_vel = x[[50,53]] * self.velocity_norrmcoeff
        reward += 0.4 * np.exp(-0.1*np.linalg.norm(ankle_joint_vel - ankle_ref_vel))

        reward -= 3e-3 * np.mean(np.abs(torques))
        reward += self.external_states[1] / 5
        if self.demo_mode:
            return reward, done, contact_points
        else:
            return reward, done

    def findgait(self,input_vec):

        freqs = self.gaitgen_net(input_vec)
        predictions = freqs.reshape(-1,6,2,17)
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
            pred_time = np.tile(pred_time, (5,1))    # Create loop for reference movement
        return pred_time

    # def starting_height(self,hip_init,knee_init,ankle_init):


    #     upper_len = 0.45
    #     lower_len = 0.45
    #     foot_len = 0.09

    #     hip_short = upper_len - (upper_len * np.cos(hip_init) )
    #     knee_short = lower_len - (lower_len * np.cos(knee_init))
    #     foot_exten = foot_len * np.sin(np.abs(ankle_init))
    #     init_pos = 1.19 - hip_short - knee_short 

    #     return init_pos
    
    def init_height(self):
        # --- Set Initial Joint Pose ---
        num_joints = p.getNumJoints(self.robot)

        # Force PyBullet to update kinematics
        p.stepSimulation() # Step once to ensure internal transforms are updated

        # --- Find Robot's Current Lowest Point (Z) and its (X, Y) Location ---
        min_z = float('inf')

        # Get current base position (needed later for offset calculation)
        current_base_pos, current_base_orn = p.getBasePositionAndOrientation(self.robot)

        # Iterate through all links to find the lowest point
        for link_index in range(-1, num_joints): # Include base (-1)
            try:
                aabb = p.getAABB(self.robot, link_index)
                link_min_z = aabb[0][2]
                if link_min_z < min_z:
                    min_z = link_min_z
                    min_z_link_index = link_index
                    # Use the center of the bottom face of the AABB as the X,Y probe point
                    min_z_x = (aabb[0][0] + aabb[1][0]) / 2.0
                    min_z_y = (aabb[0][1] + aabb[1][1]) / 2.0
                    # print(f"Link {link_index}: New min_z = {min_z} at approx ({min_z_x:.2f}, {min_z_y:.2f})") # Debug
            except Exception as e:
                # print(f"Could not get AABB for link {link_index}: {e}") # Debug
                continue
        target_x = 0
        target_y = 0

        final_base_pos = [target_x, target_y, 0] # Fallback
        # --- Find Terrain Height Below the Lowest Point using Ray Casting ---
        ray_start = [0, min_z_y, 1] # Start ray high up
        ray_end = [0, min_z_y, -1] # End ray well below expected terrain

        terrain_z_target = 0.0 # Default if ray doesn't hit
        try:
            ray_results = p.rayTest(ray_start, ray_end)
            if ray_results:
                for hit in ray_results:
                    hit_object_id, hit_link_index, hit_fraction, hit_position, hit_normal = ray_results[0]

                    if hit_position[2] > terrain_z_target:
                        terrain_z_target = hit_position[2]

        except Exception as e:
            print(f"Error during ray casting: {e}. Assuming Z=0.")

        vertical_shift = terrain_z_target - min_z

        # Apply this shift to the robot's *current base Z* coordinate
        new_base_z = current_base_pos[2] + vertical_shift

        # Add a tiny safety margin
        safety_margin = 0.02 # Slightly larger margin might be safer on uneven terrain
        final_base_z = new_base_z + safety_margin
        if final_base_z > 1.25:
            final_base_z = 1.22

        # Final position uses the target X, Y and the calculated Z
        final_base_pos = [target_x, target_y, final_base_z] # Add a bit more height for safety
        
        # --- Reset Robot to Final Position ---
        p.resetBasePositionAndOrientation(self.robot, final_base_pos, p.getQuaternionFromEuler([0, 0, 0]))

    def init_state(self):
        if self.demo_mode == False:

            start_idx = np.random.randint(0,200)

            # self.max_steps = self.max_steps - start_idx
            self.reference_idx = start_idx

            rhip_pos = self.reference[start_idx,0]
            rknee_pos = self.reference[start_idx,1]
            rankle_pos = self.reference[start_idx,2]
            lhip_pos = self.reference[start_idx,3]
            lknee_pos = self.reference[start_idx,4]
            lankle_pos = self.reference[start_idx,5]
            
            if np.abs(rhip_pos) > np.abs(lhip_pos):
                hip_init = lhip_pos
            else:
                hip_init = rhip_pos

            if np.abs(rknee_pos) > np.abs(lknee_pos):
                knee_init = lknee_pos
            else:
                knee_init = rknee_pos

            if np.abs(rankle_pos) < np.abs(lankle_pos):
                ankle_init = lankle_pos
            else:
                ankle_init = rankle_pos

            # init_z = self.starting_height(hip_init,knee_init,ankle_init)
            del self.robot
            self.robot = p.loadURDF("assets/biped2d.urdf", [0,0,1.195], p.getQuaternionFromEuler([0.,0.,0.]))
            # p.resetBasePositionAndOrientation(self.robot, [0,0,1.195], p.getQuaternionFromEuler([0, 0, 0]))
            p.resetJointState(self.robot, 3, targetValue = rhip_pos) 
            p.resetJointState(self.robot, 4, targetValue = rknee_pos)
            p.resetJointState(self.robot, 5, targetValue = rankle_pos)
            p.resetJointState(self.robot, 6, targetValue = lhip_pos)
            p.resetJointState(self.robot, 7, targetValue = lknee_pos)
            p.resetJointState(self.robot, 8, targetValue = lankle_pos)
            
            p.setJointMotorControlArray(self.robot,[0,1,2,3,4,5,6,7,8], p.VELOCITY_CONTROL, forces=[0,0,0,0,0,0,0,0,0])
            # self.init_height()
            

        else:
            start_idx = 0
            self.reference_idx = start_idx

            del self.robot
            self.robot = p.loadURDF("assets/biped2d.urdf", [0,0,1.195], p.getQuaternionFromEuler([0.,0.,0.]))

            p.setJointMotorControlArray(self.robot,[0,1,2,3,4,5,6,7,8], p.VELOCITY_CONTROL, forces=[0,0,0,0,0,0,0,0,0])
            # self.init_height()

        # self.t1_torso_pos = p.getJointState(self.robot, 2)[0]
        self.t1_rhip_pos = p.getJointState(self.robot, 3)[0]
        self.t1_rknee_pos = p.getJointState(self.robot, 4)[0]
        self.t1_rankle_pos = p.getJointState(self.robot, 5)[0]
        self.t1_lhip_pos = p.getJointState(self.robot, 6)[0]
        self.t1_lknee_pos = p.getJointState(self.robot, 7)[0]
        self.t1_lankle_pos = p.getJointState(self.robot, 8)[0]

    def return_state(self):
        
        link_state = p.getLinkState(self.robot, 2,computeLinkVelocity=True)          #link index 2 is for torso
        (pos_x,pos_y,pos_z) = link_state[0]                #3D position of the link
        y_vel = link_state[6][1]                           #y velocity of the link

        # self.torso_pos = p.getJointState(self.robot, 2)[0]
        self.rhip_pos = p.getJointState(self.robot, 3)[0]
        self.rknee_pos = p.getJointState(self.robot, 4)[0]
        self.rankle_pos = p.getJointState(self.robot, 5)[0]
        self.lhip_pos = p.getJointState(self.robot, 6)[0]
        self.lknee_pos = p.getJointState(self.robot, 7)[0]
        self.lankle_pos = p.getJointState(self.robot, 8)[0]

        # self.torso_vel = p.getJointState(self.robot, 2)[1]
        self.rhip_vel = p.getJointState(self.robot, 3)[1]
        self.rknee_vel = p.getJointState(self.robot, 4)[1]
        self.rankle_vel = p.getJointState(self.robot, 5)[1]
        self.lhip_vel = p.getJointState(self.robot, 6)[1]
        self.lknee_vel = p.getJointState(self.robot, 7)[1]
        self.lankle_vel = p.getJointState(self.robot, 8)[1]

        ref_rhip_vel = (self.reference[self.reference_idx+self.t,0] - self.reference[self.reference_idx+self.t-1,0])/self.dt
        ref_rknee_vel = (self.reference[self.reference_idx+self.t,1] - self.reference[self.reference_idx+self.t-1,1])/self.dt
        ref_rankle_vel = (self.reference[self.reference_idx+self.t,2] - self.reference[self.reference_idx+self.t-1,2])/self.dt
        ref_lhip_vel = (self.reference[self.reference_idx+self.t,3] - self.reference[self.reference_idx+self.t-1,3])/self.dt
        ref_lknee_vel = (self.reference[self.reference_idx+self.t,4] - self.reference[self.reference_idx+self.t-1,4])/self.dt
        ref_lankle_vel = (self.reference[self.reference_idx+self.t,5] - self.reference[self.reference_idx+self.t-1,5])/self.dt

        self.state[0] = self.reference_speed /3 
        self.state[1] = self.ramp_angle /0.3491 #Equivalent to 20 degrees
        self.state[2] = 0
        self.state[3] = 0
        self.state[4] = 0
        self.external_states = [pos_x,pos_y,pos_z]
        self.state[5] = y_vel   / 3

        self.state[6:12] = np.array([self.rhip_pos, self.rknee_pos, self.rankle_pos, self.lhip_pos, self.lknee_pos, self.lankle_pos]) /self.pos_normcoeff
        #
        if self.demo_mode == False:
            self.state[6:12] += np.random.normal(0,1,6)*self.pos_noise
        self.state[12:18] = np.array([self.past_target_action[0]/self.max_torque[0], self.past_target_action[1]/self.max_torque[1], self.past_target_action[2]/self.max_torque[2], 
                             self.past_target_action[3]/self.max_torque[3], self.past_target_action[4]/self.max_torque[4], 
                             self.past_target_action[5]/self.max_torque[5]])
        
        self.state[18:24] = np.array([self.t1_rhip_pos, self.t1_rknee_pos, self.t1_rankle_pos, self.t1_lhip_pos, 
                             self.t1_lknee_pos, self.t1_lankle_pos]) /self.pos_normcoeff
        # self.state[27:34] = [self.past2_target_action[0]/self.max_torque, self.past2_target_action[1]/self.max_torque, self.past2_target_action[2]/self.max_torque,
        #                      self.past2_target_action[3]/self.max_torque, self.past2_target_action[4]/self.max_torque, self.past2_target_action[5]/self.max_torque,
        #                      self.past2_target_action[6]/self.max_torque]
        self.state[24:30] = np.array([self.rhip_vel, self.rknee_vel, self.rankle_vel, self.lhip_vel, 
                             self.lknee_vel, self.lankle_vel]) /self.velocity_norrmcoeff
        if self.demo_mode == False:
            self.state[24:30] += np.random.normal(0,1,6)*self.vel_noise
        self.state[30:36] = np.array([self.reference[self.reference_idx+self.t,0], self.reference[self.reference_idx+self.t,1], 
                             self.reference[self.reference_idx+self.t,2], self.reference[self.reference_idx+self.t,3],
                             self.reference[self.reference_idx+self.t,4], self.reference[self.reference_idx+self.t,5]]) / self.pos_normcoeff
        
        self.state[36:42] = np.array([self.reference[self.reference_idx+self.t+10,0], self.reference[self.reference_idx+self.t+10,1],
                                self.reference[self.reference_idx+self.t+10,2], self.reference[self.reference_idx+self.t+10,3],
                                self.reference[self.reference_idx+self.t+10,4], self.reference[self.reference_idx+self.t+10,5]]) / self.pos_normcoeff
        
        # self.state[53:59] = [self.reference[self.reference_idx+self.t+50,0], self.reference[self.reference_idx+self.t+50,1],
        #                         self.reference[self.reference_idx+self.t+50,2], self.reference[self.reference_idx+self.t+50,3],
        #                         self.reference[self.reference_idx+self.t+50,4], self.reference[self.reference_idx+self.t+50,5]]
        
        self.state[42:48] = np.array([self.reference[self.reference_idx+self.t+100,0], self.reference[self.reference_idx+self.t+100,1],
                                self.reference[self.reference_idx+self.t+100,2], self.reference[self.reference_idx+self.t+100,3],
                                self.reference[self.reference_idx+self.t+100,4], self.reference[self.reference_idx+self.t+100,5]]) / self.pos_normcoeff
        
        self.state[48:54] = np.array([ref_rhip_vel, ref_rknee_vel, ref_rankle_vel,ref_lhip_vel, ref_lknee_vel, ref_lankle_vel]) /self.velocity_norrmcoeff
        
        self.t1_rhip_pos = self.rhip_pos
        self.t1_rknee_pos = self.rknee_pos
        self.t1_rankle_pos = self.rankle_pos
        self.t1_lhip_pos = self.lhip_pos
        self.t1_lknee_pos = self.lknee_pos
        self.t1_lankle_pos = self.lankle_pos

        self.state_info = {
            0: "reference_speed (normalized)",
            1: "ramp_angle (normalized)",
            2: "reserved / unused",  # remains 0
            3: "reserved / unused",  # remains 0
            4: "reserved / unused",  # remains 0
            5: "y_velocity (normalized)",

            6: "rhip_pos (normalized)",
            7: "rknee_pos (normalized)",
            8: "rankle_pos (normalized)",
            9: "lhip_pos (normalized)",
            10: "lknee_pos (normalized)",
            11: "lankle_pos (normalized)",

            12: "past_action_rhip (normalized)",
            13: "past_action_rknee (normalized)",
            14: "past_action_rankle (normalized)",
            15: "past_action_lhip (normalized)",
            16: "past_action_lknee (normalized)",
            17: "past_action_lankle (normalized)",

            18: "t1_rhip_pos (normalized)",
            19: "t1_rknee_pos (normalized)",
            20: "t1_rankle_pos (normalized)",
            21: "t1_lhip_pos (normalized)",
            22: "t1_lknee_pos (normalized)",
            23: "t1_lankle_pos (normalized)",

            24: "rhip_vel (normalized)",
            25: "rknee_vel (normalized)",
            26: "rankle_vel (normalized)",
            27: "lhip_vel (normalized)",
            28: "lknee_vel (normalized)",
            29: "lankle_vel (normalized)",

            30: "t0_ref_rhip_pos (normalized)",
            31: "t0_ref_rknee_pos (normalized)",
            32: "t0_ref_rankle_pos (normalized)",
            33: "t0_ref_lhip_pos (normalized)",
            34: "t0_ref_lknee_pos (normalized)",
            35: "t0_ref_lankle_pos (normalized)",

            36: "t10_ref_rhip_pos (normalized)",
            37: "t10_ref_rknee_pos (normalized)",
            38: "t10_ref_rankle_pos (normalized)",
            39: "t10_ref_lhip_pos (normalized)",
            40: "t10_ref_lknee_pos (normalized)",
            41: "t10_ref_lankle_pos (normalized)",

            42: "t100_ref_rhip_pos (normalized)",
            43: "t100_ref_rknee_pos (normalized)",
            44: "t100_ref_rankle_pos (normalized)",
            45: "t100_ref_lhip_pos (normalized)",
            46: "t100_ref_lknee_pos (normalized)",
            47: "t100_ref_lankle_pos (normalized)",

            48: "ref_rhip_vel (normalized)",
            49: "ref_rknee_vel (normalized)",
            50: "ref_rankle_vel (normalized)",
            51: "ref_lhip_vel (normalized)",
            52: "ref_lknee_vel (normalized)",
            53: "ref_lankle_vel (normalized)"
        }



    def init_noisy_plane(self, noise_level=0.1, num_rows=128, num_columns=1024, mesh_scale=[0.1, 0.1, 1],
                        baseOrientation=None):

        # Use identity quaternion if none provided.
        if baseOrientation is None:
            baseOrientation = p.getQuaternionFromEuler([0, 0, 0])
        
        # Generate heightfield data with noise in the range [-noise_level, noise_level]
        heightfield_data = np.random.uniform(low=-noise_level, high=noise_level, size=num_rows * num_columns).tolist()

        # Create a collision shape for the heightfield
        terrain_shape = p.createCollisionShape(
            shapeType=p.GEOM_HEIGHTFIELD,
            meshScale=mesh_scale,
            heightfieldData=heightfield_data,
            numHeightfieldRows=num_rows,
            numHeightfieldColumns=num_columns,
            physicsClientId=self.physics_client
        )

        # Create a static (mass=0) multi-body using the heightfield shape
        self.planeId = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=terrain_shape,
            basePosition=[0, 0, 0],
            baseOrientation=baseOrientation,
            physicsClientId=self.physics_client
        )
        p.changeDynamics(self.planeId, -1, lateralFriction=1.0)
    def return_external_state(self):
        # return self.external_states
        return self.external_states
    def close(self):
        p.disconnect()  # VERY important for PyBullet cleanup
        # cleanup logic (e.g., close files, windows, etc.)
        print("Environment closed.")

