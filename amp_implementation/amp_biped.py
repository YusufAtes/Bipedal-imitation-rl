import numpy as np
from scipy.signal import resample
from skrl.agents.torch.amp.amp import F
import torch
from gait_generator_net import SimpleFCNN
import gymnasium as gym
from gymnasium import spaces
import pybullet_data
import time
from PIL import Image
import pybullet as pyb
from PIL import Image, ImageDraw, ImageFont

class BipedEnv(gym.Env):
    def __init__(self,render=False, render_mode= None, demo_mode=False, demo_type=None):
        self.step_counter = 0
        self.p = pyb
        if render_mode == 'human':
            self.physics_client = self.p.connect(self.p.GUI)
        else:
            self.physics_client = self.p.connect(self.p.DIRECT)
        self.observe_mode = False
        self.scale = 1.
        self.dt = 1e-3
        self.demo_mode = demo_mode
        self.demo_type = demo_type
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        if self.demo_mode == False:
            # Optimize physics for speed
            self.p.setPhysicsEngineParameter(
                fixedTimeStep=self.dt,
                numSolverIterations=10,  # Reduce from default 20
                enableConeFriction=0,    # Disable cone friction for speed
                deterministicOverlappingPairs=0  # Disable for speed
            )

        self.robot = self.p.loadURDF("assets/biped2d.urdf", [0,0,1.185], self.p.getQuaternionFromEuler([0.,0.,0.])
        ,physicsClientId=self.physics_client)
        self.planeId = self.p.loadURDF("plane.urdf",physicsClientId=self.physics_client)
        self.leg_len = 0.94
        self.render_mode = render_mode
        self.joint_idx = [2,3,4,5,6,7,8] # torso, rhip, rknee, rankle, lhip, lknee, lankle

        # === MODIFICATION 1: Device for internal model ===
        # This device is for the gaitgen_net, which runs inside the env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.max_steps = int(3*(1/self.dt))
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-100, high=100, shape=(56,), dtype=np.float32)
        self.amp_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(300,), dtype=np.float32)

        # === MODIFICATION 2: Use NumPy for amp_q_hist (for multiprocessing) ===
        self.amp_q_hist = np.zeros((1, 50, 6), dtype=np.float32)
        # self.num_envs = 1  <-- REMOVED
        # self.num_agents = 1 <-- REMOVED

        self.t = 0
        self.gaitgen_net = SimpleFCNN()
        self.gaitgen_net.load_state_dict(torch.load('final_model.pth',weights_only=True))
        self.gaitgen_net.to(self.device)
        
        self.normalizationconst = np.load(rf"newnormalization_constants.npy")
        self.joint_no = self.p.getNumJoints(self.robot,physicsClientId=self.physics_client)
        self.max_torque = np.array([500,500,300,150,500,300,150])  # max torque for each joint defined in urdf file
        self.state = np.zeros(56)
        self.update_const = 0.75
        self.velocity_normcoeff = 10.0
        self.pos_normcoeff = np.pi
        self.torque_normcoeff = 500

        self.double_support = True
        self.right_swing = False
        self.left_swing = False

    def reset(self,seed=None,test_speed = None, test_angle = None,demo_max_steps = None, 
              ground_noise = None, ground_resolution = None,heightfield_data=[None],options= None):
        if seed is not None:
            np.random.seed(seed)
            
        self.test_speed = test_speed
        self.test_angle = test_angle
        self.max_steps = int(3*(1/self.dt))
        self.t = 0
        self.p.resetSimulation(physicsClientId=self.physics_client)
        self.p.setGravity(0,0,-9.81)
        self.p.setTimeStep(self.dt)
        self.taken_step_counter = 0
        if self.demo_mode == True:
            self.p.setPhysicsEngineParameter(
                fixedTimeStep       = 1.0/1000.0,
                numSolverIterations = 100,
                deterministicOverlappingPairs = 1,
                enableConeFriction  = 0,
                physicsClientId=self.physics_client
            )
            self.p.setPhysicsEngineParameter(numSubSteps=0) # OR a fixed, high value
            self.p.setPhysicsEngineParameter(enableFileCaching=0) # Prevents reading from disk, which can have timing issues


        speed_limit = 2.2 #np.clip(self.step_counter/3_500_000,0,1.9) + 0.7
        ramp_Limit = 5 #np.clip(self.step_counter/1_200_000,0,5)

        self.reference_speed = np.random.uniform(0.2,speed_limit)
        self.ramp_angle = np.random.uniform(-ramp_Limit,ramp_Limit) *np.pi / 180

        if self.demo_mode == True:
            if demo_max_steps:
                self.max_steps = demo_max_steps
            if self.test_speed is not None:
                self.reference_speed = self.test_speed

            if self.test_angle is not None:
                self.ramp_angle = self.test_angle *np.pi / 180

        encoder_vec = np.empty((3))   # init_pos + speed + r_leglength + l_leglength + ramp_angle = 0
        encoder_vec[0] = self.reference_speed/3
        encoder_vec[1] = self.leg_len /1.5
        encoder_vec[2] = self.leg_len /1.5
        encoder_vec = torch.tensor(encoder_vec, dtype=torch.float32, device=self.device)    
        self.reference = self.findgait(encoder_vec)                     #Find the gait
        self.reference = np.clip(self.reference, -np.pi/2, np.pi/2)     #Clip the gait

        plane_orientation = self.p.getQuaternionFromEuler([self.ramp_angle, 0 , 0],physicsClientId=self.physics_client)

        if self.demo_mode == False:
            self.planeId = self.p.loadURDF("plane.urdf",physicsClientId=self.physics_client, baseOrientation=plane_orientation)
            self.p.changeDynamics(self.planeId, -1, lateralFriction=1.0,physicsClientId=self.physics_client)
        else:
            if (ground_noise != None):
                self.init_noisy_plane(ground_resolution=ground_resolution, noise_level=ground_noise,baseOrientation=plane_orientation,
                                      heightfield_data=heightfield_data)
                self.heightfield_data = heightfield_data
            else:
                self.planeId = self.p.loadURDF("plane.urdf",physicsClientId=self.physics_client, baseOrientation=plane_orientation)
                self.p.changeDynamics(self.planeId, -1, lateralFriction=1.0,
                frictionAnchor=1,physicsClientId=self.physics_client)


        self.reset_info = {'current state':self.state}
        self.past_action_error = np.zeros(7)
        self.current_action = np.zeros(7)
        self.target_action = np.zeros(7)
        self.past_target_action = np.zeros(7)
        self.past2_target_action = np.zeros(7)
        self.past_forward_place = 0
        self.control_freq = 10
        self.external_states = np.zeros(4)
        self.ground_noise = ground_noise if ground_noise is not None else 0.0
        self.init_state()
        self.return_state()
        
        # === MODIFICATION 3: Return NumPy array, not tensor ===
        return self.state, self.reset_info

    def step(self,torques):
        self.step_counter += 1
        
        # === MODIFICATION 4: Input `torques` is NumPy (from wrapper) ===
        if torch.is_tensor(torques):
             # This might still be needed if not using a wrapper, but
             # with gym.vector wrappers, `torques` will be NumPy.
            torques = torques.cpu().numpy()
       
        if torques.ndim > 1:
            torques = torques.flatten()
            
        # Set torques
        if self.demo_mode == False:
            self.target_action = (torques) * self.max_torque
        else:
            self.target_action = torques * self.max_torque
        
        for i in range(10):
            self.current_action = self.update_const*self.target_action + (1-self.update_const)*self.current_action 
            self.t +=1
            self.p.setJointMotorControlArray(
                bodyIndex=self.robot,
                jointIndices=self.joint_idx,
                controlMode=self.p.TORQUE_CONTROL,
                forces=self.current_action,
                physicsClientId=self.physics_client
            )
            # Step simulation
            self.p.stepSimulation(physicsClientId=self.physics_client)
            
        self.past_target_action = self.target_action
        self.past2_target_action = self.past_target_action
        self.return_state()

        # === MODIFICATION 5: Update NumPy amp_q_hist ===
        self.amp_q_hist = np.roll(self.amp_q_hist, shift=-1, axis=1)
        self.amp_q_hist[0, -1, :] = self.state[4:10] * self.pos_normcoeff

        if self.render_mode == 'human':
            time.sleep(self.dt)
            
        reward, done = self.biped_reward(self.state,torques=self.target_action)
        truncated = False

        if self.t > self.max_steps:
            truncated = True

        # === MODIFICATION 6: Return NumPy/standard types, not tensors ===
        amp_obs = self.collect_observation() # This is now NumPy
        info = {"amp_obs": amp_obs}
        
        # Return standard gym types
        return self.state, float(reward), bool(done), bool(truncated), info

    def biped_reward(self,x,torques):
        if self.step_counter % 100_000 == 0:
            print(self.step_counter)
        # alpha_coeff = 0.25*(self.step_counter / 15_000_000)
        # self.imitation_weight = 1.0 -  alpha_coeff
        self.gait_weight = 1.0

        self.imitation_weight_hip_pos = 0.75
        self.imitation_weight_knee_pos = 0.75
        self.imitation_weight_ankle_pos = 0.25

        self.imitation_weight_hip_vel = 0.15
        self.imitation_weight_knee_vel = 0.15
        self.imitation_weight_ankle_vel = 0.1

        # 10 M steps is usually 35k episodes
        self.alive_weight = 0.5 * self.gait_weight
        self.contact_weight = 0.6 * self.gait_weight
        done = False
        reward = 0

        #Contact Reward
        contact_points = self.p.getContactPoints(self.robot, self.planeId,physicsClientId=self.physics_client)

        # Left Contact Points
        left_contact_forces = [i[9] for i in contact_points if i[3] == 8]
        left_contact = len(left_contact_forces)
        left_contact_forces = np.mean(left_contact_forces) if len(left_contact_forces) > 0 else 0
        lfoot_state = self.p.getLinkState(self.robot, 8,computeLinkVelocity=True,physicsClientId=self.physics_client)        #link index 8 is for left foot
        lfoot_pos = lfoot_state[0]

        # Right Contact Points
        right_contact_forces = [i[9] for i in contact_points if i[3] == 5]
        right_contact = len(right_contact_forces)
        right_contact_forces = np.mean(right_contact_forces) if len(right_contact_forces) > 0 else 0
        rfoot_state = self.p.getLinkState(self.robot, 5,computeLinkVelocity=True,physicsClientId=self.physics_client)        #link index 5 is for right foot
        rfoot_pos = rfoot_state[0]

        reward  += self.contact_weight * self.calculate_contact_reward(left_contact, right_contact, left_contact_forces, 
                                                                       right_contact_forces, lfoot_pos, rfoot_pos)

        # #Imitation Reward
        # ... (omitted for brevity, no changes) ...

        #Torque Reward
        reward -= 1e-3 * np.mean(np.abs(self.target_action)) * self.gait_weight
        
        # Forward Speed Reward
        current_speed = (self.external_states[1] - self.past_forward_place) / (self.dt * 10)  # forward speed
        reward += 0.6* self.gait_weight * np.exp(-2*np.abs(current_speed - self.reference_speed))  # reward for maintaining speed newly adjusted

        #Angle Reward
        if np.abs(self.external_states[3]) > 0.98:  # Robot outside healthy angle range
            reward =- 100
            done = True

        if self.demo_type != "noisy":
            #Height Reward
            if self.external_states[2] > 1.45 + np.tan(self.ramp_angle) * self.external_states[1]:
                reward =- 100
                done = True
            elif self.external_states[2] > 1.3 + np.tan(self.ramp_angle) * self.external_states[1]:
                reward -= self.alive_weight
            elif self.external_states[2] < 0.8 + np.tan(self.ramp_angle) * self.external_states[1]:
                reward =- 100
                done = True
            elif self.external_states[2] < 0.98 + np.tan(self.ramp_angle) * self.external_states[1]:
                reward -= 1 * self.alive_weight
            else:
                reward += 1 * self.alive_weight
        
        else:
            x_pos = self.external_states[1]
            plane_z_location = self.heightfield_data[int((x_pos / 0.05+512)*32)]
            #Height Reward
            if self.external_states[2] > 1.45 + np.tan(self.ramp_angle) * self.external_states[1] + plane_z_location:
                reward =- 100
                done = True
            elif self.external_states[2] > 1.3+ np.tan(self.ramp_angle) * self.external_states[1] + plane_z_location:
                reward -= self.alive_weight
            elif self.external_states[2] < 0.8+ np.tan(self.ramp_angle) * self.external_states[1] + plane_z_location:
                reward =- 100
                done = True
            elif self.external_states[2] < 0.98+ np.tan(self.ramp_angle) * self.external_states[1] + plane_z_location:
                reward -= 1 * self.alive_weight
            else:
                reward += 1 * self.alive_weight
        return reward, done
    
    def calculate_contact_reward(self, left_contact, right_contact, left_contact_forces,
                                  right_contact_forces, lfoot_pos, rfoot_pos, force_eps=10):
        # ... (omitted for brevity, no changes) ...
        if left_contact_forces > force_eps and right_contact_forces > force_eps:
            if self.double_support == False:
                self.taken_step_counter += 1
            self.double_support = True
            self.right_swing = False
            self.left_swing = False

        elif left_contact_forces > force_eps and right_contact_forces <= force_eps:
            if self.right_swing == False:
                self.taken_step_counter += 1
            self.double_support = False
            self.right_swing = True
            self.left_swing = False

        elif right_contact_forces > force_eps and left_contact_forces <= force_eps:
            if self.left_swing == False:
                self.taken_step_counter += 1
            self.double_support = False
            self.right_swing = False
            self.left_swing = True
        elif right_contact_forces <= force_eps and left_contact_forces <= force_eps:
            self.double_support = False
            self.right_swing = False
            self.left_swing = False

        if self.double_support:
            contact_no = left_contact + right_contact
            return 1 / (1 + np.exp(-2 * (contact_no - 4.0)))  # Sigmoid function centered at 4.0

        elif self.right_swing:
            plane_height = np.tan(self.ramp_angle) * rfoot_pos[1]
            clearence_reward = 1 / (1 + np.exp(20 * np.abs(rfoot_pos[2] - plane_height - 0.15)))  # Sigmoid function centered at 0.15m above ground
            contact_reward = 1 / (1 + np.exp(-2 * (left_contact - 2)))  # Sigmoid function centered at 2
            return 0.5 * clearence_reward + 0.5 * contact_reward
        
        elif self.left_swing:   
            plane_height = np.tan(self.ramp_angle) * lfoot_pos[1]
            clearence_reward = 1 / (1 + np.exp(20 * np.abs(lfoot_pos[2] - plane_height - 0.15)))  # Sigmoid function centered at 0.15m above ground
            contact_reward = 1 / (1 + np.exp(-2 * (right_contact - 2)))  # Sigmoid function centered at 2
            return 0.5 * clearence_reward + 0.5 * contact_reward
        else:
            return 0
        
    def render(self):
        """Render the environment. For PyBullet environments, this is handled automatically."""
        pass
    
    def close(self):
        self.p.disconnect(physicsClientId=self.physics_client)
        # print("Environment closed") # Quieten down for parallel envs


    def findgait(self,input_vec):
        # ... (omitted for brevity, no changes) ...
        freqs = self.gaitgen_net(input_vec)
        predictions = freqs.reshape(-1,6,2,17)
        predictions = predictions.detach().cpu().numpy()
        predictions = predictions[0]
        predictions = self.denormalize(predictions)
        pred_time = self.pred_ifft(predictions)

        return pred_time

    def denormalize(self,pred):
        # ... (omitted for brevity, no changes) ...
        #form is [5,2,17]
        for i in range(17):
            for k in range(2):
                pred[:,k,i] = pred[:,k,i] * self.normalizationconst[i*2+k]
        return pred
    
        
    def pred_ifft(self,predictions):
        # ... (omitted for brevity, no changes) ...
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
            pred_time = np.tile(pred_time, (50,1))    # Create loop for reference movement
        return pred_time


    def starting_height(self,hip_init,knee_init,ankle_init):
        # ... (omitted for brevity, no changes) ...
        upper_len = 0.45
        lower_len = 0.45
        foot_len = 0.09

        hip_short = upper_len - (upper_len * np.cos(hip_init) )
        knee_short = lower_len - (lower_len * np.cos(knee_init))
        foot_exten = foot_len * np.sin(np.abs(ankle_init))
        init_pos = 1.195 - hip_short - knee_short 

        return init_pos
    

    def init_state(self):
        if self.demo_mode == False:
            start_idx = np.random.randint(0,500)
            # ... (omitted for brevity, no changes) ...
            self.reference_idx = start_idx
            # ... (omitted for brevity, no changes) ...
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

            init_z = self.starting_height(hip_init,knee_init,ankle_init)
            del self.robot
            self.robot = self.p.loadURDF("assets/biped2d.urdf", [0,0,init_z + 0.02], self.p.getQuaternionFromEuler([0.,0.,0.]),physicsClientId=self.physics_client)
            self.p.setJointMotorControlArray(self.robot,[0,1,2,3,4,5,6,7,8], self.p.VELOCITY_CONTROL, forces=[0,0,0,0,0,0,0,0,0],physicsClientId=self.physics_client)

            self.p.resetJointState(self.robot, 3, targetValue = rhip_pos,physicsClientId=self.physics_client) 
            self.p.resetJointState(self.robot, 4, targetValue = rknee_pos,physicsClientId=self.physics_client)
            self.p.resetJointState(self.robot, 5, targetValue = rankle_pos,physicsClientId=self.physics_client)
            self.p.resetJointState(self.robot, 6, targetValue = lhip_pos,physicsClientId=self.physics_client)
            self.p.resetJointState(self.robot, 7, targetValue = lknee_pos,physicsClientId=self.physics_client)
            self.p.resetJointState(self.robot, 8, targetValue = lankle_pos,physicsClientId=self.physics_client)
        else:
            # ... (omitted for brevity, no changes) ...
            start_idx = 0
            # ... (omitted for brevity, no changes) ...
            init_z = self.starting_height(rhip_pos,lhip_pos,rankle_pos)
            del self.robot
            self.robot = self.p.loadURDF("assets/biped2d.urdf", [0,0,1.185], self.p.getQuaternionFromEuler([0.,0.,0.]))
            # ... (omitted for brevity, no changes) ...

        self.p.setGravity(0,0,-9.81,physicsClientId=self.physics_client)
        self.p.setTimeStep(self.dt,physicsClientId=self.physics_client)

        # === MODIFICATION 7: PyBullet Batch Call ===
        # Use p.getJointStates (plural) to batch calls
        joint_states = self.p.getJointStates(self.robot, self.joint_idx, physicsClientId=self.physics_client)
        
        (self.t1_torso_pos, self.t1_rhip_pos, self.t1_rknee_pos, self.t1_rankle_pos, 
         self.t1_lhip_pos, self.t1_lknee_pos, self.t1_lankle_pos) = [state[0] for state in joint_states]


    def return_state(self):
        
        link_state = self.p.getLinkState(self.robot, 2,computeLinkVelocity=True,physicsClientId=self.physics_client)          #link index 2 is for torso
        torso_g_quat = link_state[1]
        roll, _, _ = self.p.getEulerFromQuaternion(torso_g_quat,physicsClientId=self.physics_client)
        (pos_x,pos_y,pos_z) = link_state[0]                #3D position of the link
        y_vel = link_state[6][1]                           #y velocity of the link

        # === MODIFICATION 8: PyBullet Batch Call ===
        # Replace 14 p.getJointState calls with 1 p.getJointStates call
        joint_states = self.p.getJointStates(self.robot, self.joint_idx, physicsClientId=self.physics_client)
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]

        (self.torso_pos, self.rhip_pos, self.rknee_pos, self.rankle_pos,
         self.lhip_pos, self.lknee_pos, self.lankle_pos) = joint_positions

        (self.torso_vel, self.rhip_vel, self.rknee_vel, self.rankle_vel,
         self.lhip_vel, self.lknee_vel, self.lankle_vel) = joint_velocities
        # === End Modification ===

        ref_rhip_vel = (self.reference[self.reference_idx+self.t,0] - self.reference[self.reference_idx+self.t-1,0])/self.dt
        ref_rknee_vel = (self.reference[self.reference_idx+self.t,1] - self.reference[self.reference_idx+self.t-1,1])/self.dt
        ref_rankle_vel = (self.reference[self.reference_idx+self.t,2] - self.reference[self.reference_idx+self.t-1,2])/self.dt
        ref_lhip_vel = (self.reference[self.reference_idx+self.t,3] - self.reference[self.reference_idx+self.t-1,3])/self.dt
        ref_lknee_vel = (self.reference[self.reference_idx+self.t,4] - self.reference[self.reference_idx+self.t-1,4])/self.dt
        ref_lankle_vel = (self.reference[self.reference_idx+self.t,5] - self.reference[self.reference_idx+self.t-1,5])/self.dt

        self.state[0] = self.reference_speed /3 
        self.state[1] = self.ramp_angle 
        
        self.past_forward_place = self.external_states[1]
        self.external_states = [pos_x,pos_y,pos_z,roll]

        self.state[2] = y_vel   / 3

        self.state[3:10] = np.array([self.torso_pos, self.rhip_pos, self.rknee_pos, self.rankle_pos, self.lhip_pos, self.lknee_pos, self.lankle_pos]) /self.pos_normcoeff

        self.state[10:17] = np.array([self.past_target_action[0]/self.max_torque[0], self.past_target_action[1]/self.max_torque[1], self.past_target_action[2]/self.max_torque[2], 
                             self.past_target_action[3]/self.max_torque[3], self.past_target_action[4]/self.max_torque[4], self.past_target_action[5]/self.max_torque[5], 
                             self.past_target_action[6]/self.max_torque[6]])
        
        self.state[17:24] = np.array([self.t1_torso_pos, self.t1_rhip_pos, self.t1_rknee_pos, self.t1_rankle_pos, self.t1_lhip_pos, 
                             self.t1_lknee_pos, self.t1_lankle_pos]) /self.pos_normcoeff

        self.state[24:31] = np.array([self.torso_vel, self.rhip_vel, self.rknee_vel, self.rankle_vel, self.lhip_vel, 
                             self.lknee_vel, self.lankle_vel]) /self.velocity_normcoeff

        self.state[31:37] = np.array([self.reference[self.reference_idx+self.t,0], self.reference[self.reference_idx+self.t,1], 
                             self.reference[self.reference_idx+self.t,2], self.reference[self.reference_idx+self.t,3],
                             self.reference[self.reference_idx+self.t,4], self.reference[self.reference_idx+self.t,5]]) / self.pos_normcoeff
        
        self.state[37:43] = np.array([self.reference[self.reference_idx+self.t+10,0], self.reference[self.reference_idx+self.t+10,1],
                                self.reference[self.reference_idx+self.t+10,2], self.reference[self.reference_idx+self.t+10,3],
                                self.reference[self.reference_idx+self.t+10,4], self.reference[self.reference_idx+self.t+10,5]]) / self.pos_normcoeff

        self.state[43:49] = np.array([self.reference[self.reference_idx+self.t+100,0], self.reference[self.reference_idx+self.t+100,1],
                                self.reference[self.reference_idx+self.t+100,2], self.reference[self.reference_idx+self.t+100,3],
                                self.reference[self.reference_idx+self.t+100,4], self.reference[self.reference_idx+self.t+100,5]]) / self.pos_normcoeff
        
        self.state[49:55] = np.array([ref_rhip_vel, ref_rknee_vel, ref_rankle_vel,ref_lhip_vel, ref_lknee_vel, ref_lankle_vel]) /self.velocity_normcoeff
        
        self.t1_torso_pos = self.torso_pos
        self.t1_rhip_pos = self.rhip_pos
        self.t1_rknee_pos = self.rknee_pos
        self.t1_rankle_pos = self.rankle_pos
        self.t1_lhip_pos = self.lhip_pos
        self.t1_lknee_pos = self.lknee_pos
        self.t1_lankle_pos = self.lankle_pos

        # ... (state_info dict omitted for brevity) ...
    

    def return_external_state(self):
        # ... (omitted for brevity, no changes) ...
        return self.external_states

    def init_noisy_plane(self, noise_level=0.1,ground_resolution= 0.05 ,num_rows=32, num_columns=1024,
                        baseOrientation=None,heightfield_data=None):
        # ... (omitted for brevity, no changes) ...
        # Use identity quaternion if none provided.
        mesh_scale=[ground_resolution, ground_resolution, 1]
        if baseOrientation is None:
            baseOrientation = self.p.getQuaternionFromEuler([0, 0, 0],physicsClientId=self.physics_client)

        # Create a collision shape for the heightfield
        terrain_shape = self.p.createCollisionShape(
            shapeType=self.p.GEOM_HEIGHTFIELD,
            meshScale=mesh_scale,
            heightfieldData=heightfield_data,
            numHeightfieldRows=num_rows,
            numHeightfieldColumns=num_columns,
            physicsClientId=self.physics_client
        )
        min_height = np.min(heightfield_data)
        max_height = np.max(heightfield_data)
        z_center = 0.5 * (min_height + max_height) * mesh_scale[2]
        # Create a static (mass=0) multi-body using the heightfield shape
        self.planeId = self.p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=terrain_shape,
            basePosition=[0, 0, z_center],
            baseOrientation=baseOrientation,
            physicsClientId=self.physics_client
        )

        self.p.changeDynamics(self.planeId, -1, lateralFriction=1.0,
                              frictionAnchor=1, physicsClientId=self.physics_client)

    def get_image(self):
        # ... (omitted for brevity, no changes) ...
        view_matrix = self.p.computeViewMatrix(
            cameraEyePosition=[3, 0, 1.5],  # farther and higher
            cameraTargetPosition=[0, 0, 1.0],
            cameraUpVector=[0, 0, 1]
            
        )
        # ... (omitted for brevity, no changes) ...
        return image

    def change_ref_speed(self,new_speed):
        # ... (omitted for brevity, no changes) ...
        encoder_vec = np.empty((3))   # init_pos + speed + r_leglength + l_leglength + ramp_angle = 0
        encoder_vec[0] = new_speed/3
        # ... (omitted for brevity, no changes) ...
        self.reference = newly_reference
        self.reference_speed = new_speed


    def get_follow_camera_image(self, follow_distance=3.0, height=1.5,overlay_text=None):
        # ... (omitted for brevity, no changes) ...
        return image

    def return_step_taken(self):
        return self.taken_step_counter

    def collect_observation(self):
        # === MODIFICATION 9: Reshape for a single env (N=1) ===
        # The wrapper will stack these
        x = self.amp_q_hist.reshape(1, -1) 
        
        # optional normalization (handled by wrapper if needed, but AMP agent does it)
        if hasattr(self, "amp_mean") and hasattr(self, "amp_std"):
            x = (x - self.amp_mean) / self.amp_std
            
        # === MODIFICATION 10: Return NumPy array ===
        return x