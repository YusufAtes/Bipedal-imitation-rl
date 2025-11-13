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
import os

# === FIX 1: GET THE ABSOLUTE PATH FOR plane.urdf ===
# This is the correct way to load the plane and solves any path issues.
PLANE_PATH = os.path.join(pybullet_data.getDataPath(), "plane.urdf")


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
            self.p.setPhysicsEngineParameter(
                fixedTimeStep=self.dt,
                numSolverIterations=10,
                enableConeFriction=0,
                deterministicOverlappingPairs=0
            )

        self.robot = self.p.loadURDF("/home/baran/Bipedal-imitation-rl/assets/biped2d.urdf", [0,0,1.185], self.p.getQuaternionFromEuler([0.,0.,0.])
        ,physicsClientId=self.physics_client)
        self.planeId = self.p.loadURDF(PLANE_PATH, physicsClientId=self.physics_client)

        self.leg_len = 0.94
        self.render_mode = render_mode
        self.joint_idx = [2,3,4,5,6,7,8]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.max_steps = int(3*(1/self.dt))
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-100, high=100, shape=(56,), dtype=np.float32)
        self.amp_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(300,), dtype=np.float32)

        self.amp_q_hist = np.zeros((1, 50, 6), dtype=np.float32)
        self.t = 0
        
        # === FIX 2: SOLVE PICKLINGERROR ===
        # Do NOT load the model here. Set to None.
        self.gaitgen_net = None
        # === END FIX 2 ===
        
        self.normalizationconst = np.load(rf"/home/baran/Bipedal-imitation-rl/newnormalization_constants.npy")
        self.joint_no = self.p.getNumJoints(self.robot,physicsClientId=self.physics_client)
        self.max_torque = np.array([500,500,300,150,500,300,150])
        self.state = np.zeros(56)
        self.update_const = 0.75
        self.velocity_normcoeff = 10.0
        self.pos_normcoeff = np.pi
        self.torque_normcoeff = 500

        self.double_support = True
        self.right_swing = False
        self.left_swing = False

        np_data = np.load(rf"/home/baran/Bipedal-imitation-rl/gait time series data/window_data.npy").astype(np.float32)
        np_data = np.nan_to_num(np_data, nan=0.0, posinf=0.0, neginf=0.0)   # ← add this

        N, T, J = np_data.shape
        assert (T, J) == (50, 6), f"Expected (50,6), got {(T, J)}"
        K = T * J # 300

        REF_X = torch.from_numpy(np_data.reshape(N, K)).to(self.device)
        REF_MEAN = REF_X.mean(dim=0, keepdim=True)
        REF_STD = REF_X.std(dim=0, keepdim=True).clamp_min(1e-6)
        self.amp_mean = REF_MEAN
        self.amp_std = REF_STD


    def reset(self,seed=None,test_speed = None, test_angle = None,demo_max_steps = None, 
              ground_noise = None, ground_resolution = None,heightfield_data=[None],options= None):
        
        # === FIX 2 (CONTINUED): LAZY LOADING OF MODEL ===
        # Load the model only if it hasn't been loaded yet.
        # This code will run *inside* each child process once.
        if self.gaitgen_net is None:
            print(f"PID {os.getpid()}: Initializing gaitgen_net on device {self.device}")
            self.gaitgen_net = SimpleFCNN()
            self.gaitgen_net.load_state_dict(torch.load('/home/baran/Bipedal-imitation-rl/final_model.pth',weights_only=True))
            self.gaitgen_net.to(self.device)
            print(f"PID {os.getpid()}: gaitgen_net loaded.")
        # === END FIX 2 ===

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
            self.p.setPhysicsEngineParameter(numSubSteps=0)
            self.p.setPhysicsEngineParameter(enableFileCaching=0)


        speed_limit = 2.2
        ramp_Limit = 5

        self.reference_speed = np.random.uniform(0.2,speed_limit)
        self.ramp_angle = np.random.uniform(-ramp_Limit,ramp_Limit) *np.pi / 180

        if self.demo_mode == True:
            if demo_max_steps:
                self.max_steps = demo_max_steps
            if self.test_speed is not None:
                self.reference_speed = self.test_speed

            if self.test_angle is not None:
                self.ramp_angle = self.test_angle *np.pi / 180

        encoder_vec = np.empty((3))
        encoder_vec[0] = self.reference_speed/3
        encoder_vec[1] = self.leg_len /1.5
        encoder_vec[2] = self.leg_len /1.5
        encoder_vec = torch.tensor(encoder_vec, dtype=torch.float32, device=self.device)    
        self.reference = self.findgait(encoder_vec)
        self.reference = np.clip(self.reference, -np.pi/2, np.pi/2)

        plane_orientation = self.p.getQuaternionFromEuler([self.ramp_angle, 0 , 0],physicsClientId=self.physics_client)

        if self.demo_mode == False:
            self.planeId = self.p.loadURDF(PLANE_PATH, physicsClientId=self.physics_client, baseOrientation=plane_orientation)
            self.p.changeDynamics(self.planeId, -1, lateralFriction=1.0,physicsClientId=self.physics_client)
        else:
            if (ground_noise != None):
                self.init_noisy_plane(ground_resolution=ground_resolution, noise_level=ground_noise,baseOrientation=plane_orientation,
                                      heightfield_data=heightfield_data)
                self.heightfield_data = heightfield_data
            else:
                self.planeId = self.p.loadURDF(PLANE_PATH, physicsClientId=self.physics_client, baseOrientation=plane_orientation)
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
        
        # self.state = torch.from_numpy(self.state).float().to(self.device)
        # self.reset_info = {k: torch.from_numpy(v).float().to(self.device) if isinstance(v, np.ndarray) else v for k, v in self.reset_info.items()}
        return self.state, self.reset_info

    def step(self,torques):
        self.step_counter += 1
        
        if torch.is_tensor(torques):
            torques = torques.cpu().numpy()
       
        if torques.ndim > 1:
            torques = torques.flatten()
            
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
            self.p.stepSimulation(physicsClientId=self.physics_client)
            
        self.past_target_action = self.target_action
        self.past2_target_action = self.past_target_action
        self.return_state()

        self.amp_q_hist = np.roll(self.amp_q_hist, shift=-1, axis=1)
        self.amp_q_hist[0, -1, :] = self.state[4:10] * self.pos_normcoeff

        if self.render_mode == 'human':
            time.sleep(self.dt)
            
        reward, done = self.biped_reward(self.state,torques=self.target_action)
        truncated = False

        if self.t > self.max_steps:
            truncated = True

        amp_obs = self.collect_observation()
        info = {"amb_obs": amp_obs}
        
        # Return numpy arrays for AsyncVectorEnv compatibility
        # self.state = torch.from_numpy(self.state).float().to(self.device)
        # reward = torch.tensor(float(reward), dtype=torch.float32, device=self.device)
        # done = torch.tensor(bool(done), dtype=torch.bool, device=self.device)
        # truncated = torch.tensor(bool(truncated), dtype=torch.bool, device=self.device)
        # info = {k: torch.from_numpy(v).float().to(self.device) if isinstance(v, np.ndarray) else v for k, v in info.items()}
        
        return self.state, float(reward), bool(done), bool(truncated), info

    def biped_reward(self,x,torques):
        if self.step_counter % 100_000 == 0 and self.step_counter > 0:
            print(f"PID {os.getpid()}: Reached step {self.step_counter}")

        self.gait_weight = 1.0

        self.imitation_weight_hip_pos = 0.75
        self.imitation_weight_knee_pos = 0.75
        self.imitation_weight_ankle_pos = 0.25

        self.imitation_weight_hip_vel = 0.15
        self.imitation_weight_knee_vel = 0.15
        self.imitation_weight_ankle_vel = 0.1

        self.alive_weight = 0.5 * self.gait_weight
        self.contact_weight = 0.6 * self.gait_weight
        done = False
        reward = 0

        contact_points = self.p.getContactPoints(self.robot, self.planeId,physicsClientId=self.physics_client)

        left_contact_forces = [i[9] for i in contact_points if i[3] == 8]
        left_contact = len(left_contact_forces)
        left_contact_forces = np.mean(left_contact_forces) if len(left_contact_forces) > 0 else 0
        lfoot_state = self.p.getLinkState(self.robot, 8,computeLinkVelocity=True,physicsClientId=self.physics_client)
        lfoot_pos = lfoot_state[0]

        right_contact_forces = [i[9] for i in contact_points if i[3] == 5]
        right_contact = len(right_contact_forces)
        right_contact_forces = np.mean(right_contact_forces) if len(right_contact_forces) > 0 else 0
        rfoot_state = self.p.getLinkState(self.robot, 5,computeLinkVelocity=True,physicsClientId=self.physics_client)
        rfoot_pos = rfoot_state[0]

        reward  += self.contact_weight * self.calculate_contact_reward(left_contact, right_contact, left_contact_forces, 
                                                                       right_contact_forces, lfoot_pos, rfoot_pos)
        
        reward -= 1e-3 * np.mean(np.abs(self.target_action)) * self.gait_weight
        
        current_speed = (self.external_states[1] - self.past_forward_place) / (self.dt * 10)
        reward += 0.6* self.gait_weight * np.exp(-2*np.abs(current_speed - self.reference_speed))

        if np.abs(self.external_states[3]) > 0.98:
            reward =- 100
            done = True

        if self.demo_type != "noisy":
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

            # === FIX 3: FIX DEPRECATION WARNING/CRASH ===
            plane_z_location = 0.0 # Default value
            if self.heightfield_data is not None and self.heightfield_data[0] is not None:
                try:
                    # Calculate index
                    z_index = int((x_pos / 0.05 + 512) * 32)
                    # Clip index to be within bounds of the array
                    z_index = np.clip(z_index, 0, len(self.heightfield_data) - 1)
                    # Safely get the location
                    plane_z_location = self.heightfield_data[z_index]
                except (IndexError, TypeError):
                    pass # Keep default 0.0 if something goes wrong
            # === END FIX 3 ===

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
            return 1 / (1 + np.exp(-2 * (contact_no - 4.0)))

        elif self.right_swing:
            plane_height = np.tan(self.ramp_angle) * rfoot_pos[1]
            clearence_reward = 1 / (1 + np.exp(20 * np.abs(rfoot_pos[2] - plane_height - 0.15)))
            contact_reward = 1 / (1 + np.exp(-2 * (left_contact - 2)))
            return 0.5 * clearence_reward + 0.5 * contact_reward
        
        elif self.left_swing:   
            plane_height = np.tan(self.ramp_angle) * lfoot_pos[1]
            clearence_reward = 1 / (1 + np.exp(20 * np.abs(lfoot_pos[2] - plane_height - 0.15)))
            contact_reward = 1 / (1 + np.exp(-2 * (right_contact - 2)))
            return 0.5 * clearence_reward + 0.5 * contact_reward
        else:
            return 0
        
    def render(self):
        pass
    
    def close(self):
        self.p.disconnect(physicsClientId=self.physics_client)


    def findgait(self,input_vec):
        freqs = self.gaitgen_net(input_vec)
        predictions = freqs.reshape(-1,6,2,17)
        predictions = predictions.detach().cpu().numpy()
        predictions = predictions[0]
        predictions = self.denormalize(predictions)
        pred_time = self.pred_ifft(predictions)

        return pred_time

    def denormalize(self,pred):
        for i in range(17):
            for k in range(2):
                pred[:,k,i] = pred[:,k,i] * self.normalizationconst[i*2+k]
        return pred
    
        
    def pred_ifft(self,predictions):
        real_pred = predictions[:,0,:]
        imag_pred = predictions[:,1,:]
        predictions = real_pred + 1j*imag_pred

        pred_time = np.fft.irfft(predictions, axis=1)
        pred_time = pred_time.transpose(1,0)
        org_rate = 10

        if self.dt < 0.1:
            num_samples = int((pred_time.shape[0]) * (1/self.dt)/(org_rate))
            pred_time = resample(pred_time, num_samples, axis=0)
            pred_time = np.tile(pred_time, (50,1))
        return pred_time


    def starting_height(self,hip_init,knee_init,ankle_init):
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

            init_z = self.starting_height(hip_init,knee_init,ankle_init)
            del self.robot
            self.robot = self.p.loadURDF("/home/baran/Bipedal-imitation-rl/assets/biped2d.urdf", [0,0,init_z + 0.02], self.p.getQuaternionFromEuler([0.,0.,0.]),physicsClientId=self.physics_client)
            self.p.setJointMotorControlArray(self.robot,[0,1,2,3,4,5,6,7,8], self.p.VELOCITY_CONTROL, forces=[0,0,0,0,0,0,0,0,0],physicsClientId=self.physics_client)

            self.p.resetJointState(self.robot, 3, targetValue = rhip_pos,physicsClientId=self.physics_client) 
            self.p.resetJointState(self.robot, 4, targetValue = rknee_pos,physicsClientId=self.physics_client)
            self.p.resetJointState(self.robot, 5, targetValue = rankle_pos,physicsClientId=self.physics_client)
            self.p.resetJointState(self.robot, 6, targetValue = lhip_pos,physicsClientId=self.physics_client)
            self.p.resetJointState(self.robot, 7, targetValue = lknee_pos,physicsClientId=self.physics_client)
            self.p.resetJointState(self.robot, 8, targetValue = lankle_pos,physicsClientId=self.physics_client)
        else:
            start_idx = 0
            self.reference_idx = start_idx

            rhip_pos = 0.0
            rknee_pos = 0.0
            rankle_pos = 0.0
            lhip_pos = 0.0
            lknee_pos = 0.0
            lankle_pos = 0.0
            
            init_z = self.starting_height(rhip_pos,lhip_pos,rankle_pos)
            del self.robot
            self.robot = self.p.loadURDF("/home/baran/Bipedal-imitation-rl/assets/biped2d.urdf", [0,0,1.185], self.p.getQuaternionFromEuler([0.,0.,0.]))
            self.p.setJointMotorControlArray(self.robot,[0,1,2,3,4,5,6,7,8], self.p.VELOCITY_CONTROL, forces=[0,0,0,0,0,0,0,0,0],physicsClientId=self.physics_client)

            self.p.resetJointState(self.robot, 3, targetValue = rhip_pos,physicsClientId=self.physics_client) 
            self.p.resetJointState(self.robot, 4, targetValue = rknee_pos,physicsClientId=self.physics_client)
            self.p.resetJointState(self.robot, 5, targetValue = rankle_pos,physicsClientId=self.physics_client)
            self.p.resetJointState(self.robot, 6, targetValue = lhip_pos,physicsClientId=self.physics_client)
            self.p.resetJointState(self.robot, 7, targetValue = lknee_pos,physicsClientId=self.physics_client)
            self.p.resetJointState(self.robot, 8, targetValue = lankle_pos,physicsClientId=self.physics_client)

        self.p.setGravity(0,0,-9.81,physicsClientId=self.physics_client)
        self.p.setTimeStep(self.dt,physicsClientId=self.physics_client)

        joint_states = self.p.getJointStates(self.robot, self.joint_idx, physicsClientId=self.physics_client)
        
        (self.t1_torso_pos, self.t1_rhip_pos, self.t1_rknee_pos, self.t1_rankle_pos, 
         self.t1_lhip_pos, self.t1_lknee_pos, self.t1_lankle_pos) = [state[0] for state in joint_states]


    def return_state(self):
        
        link_state = self.p.getLinkState(self.robot, 2,computeLinkVelocity=True,physicsClientId=self.physics_client)
        torso_g_quat = link_state[1]
        roll, _, _ = self.p.getEulerFromQuaternion(torso_g_quat,physicsClientId=self.physics_client)
        (pos_x,pos_y,pos_z) = link_state[0]
        y_vel = link_state[6][1]

        joint_states = self.p.getJointStates(self.robot, self.joint_idx, physicsClientId=self.physics_client)
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]

        (self.torso_pos, self.rhip_pos, self.rknee_pos, self.rankle_pos,
         self.lhip_pos, self.lknee_pos, self.lankle_pos) = joint_positions

        (self.torso_vel, self.rhip_vel, self.rknee_vel, self.rankle_vel,
         self.lhip_vel, self.lknee_vel, self.lankle_vel) = joint_velocities

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
        
        # === FIX 5: CORRECTED SLICING BUG ===
        # Your original code [44:55] was an 11-element slice for 6 values
        self.state[49:55] = np.array([ref_rhip_vel, ref_rknee_vel, ref_rankle_vel,ref_lhip_vel, ref_lknee_vel, ref_lankle_vel]) /self.velocity_normcoeff
        # === END FIX 5 ===
        
        self.t1_torso_pos = self.torso_pos
        self.t1_rhip_pos = self.rhip_pos
        self.t1_rknee_pos = self.rknee_pos
        self.t1_rankle_pos = self.rankle_pos
        self.t1_lhip_pos = self.lhip_pos
        self.t1_lknee_pos = self.lknee_pos
        self.t1_lankle_pos = self.lankle_pos
    
    def return_external_state(self):
        return self.external_states

    def init_noisy_plane(self, noise_level=0.1,ground_resolution= 0.05 ,num_rows=32, num_columns=1024,
                        baseOrientation=None,heightfield_data=None):
        mesh_scale=[ground_resolution, ground_resolution, 1]
        if baseOrientation is None:
            baseOrientation = self.p.getQuaternionFromEuler([0, 0, 0],physicsClientId=self.physics_client)

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
        view_matrix = self.p.computeViewMatrix(
            cameraEyePosition=[3, 0, 1.5],
            cameraTargetPosition=[0, 0, 1.0],
            cameraUpVector=[0, 0, 1]
        )
        projection_matrix = self.p.computeProjectionMatrixFOV(
            fov=75,
            aspect=1.0,
            nearVal=0.1,
            farVal=100.0
        )
        res = 640
        _, _, rgbImg, _, _ = self.p.getCameraImage(
            width=res,
            height=res,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix
        )
        rgb_array = np.reshape(rgbImg, (res, res, 4))
        image = Image.fromarray(rgb_array[:, :, :3], 'RGB')
        return image

    def change_ref_speed(self,new_speed):
        encoder_vec = np.empty((3))
        encoder_vec[0] = new_speed/3
        encoder_vec[1] = self.leg_len /1.5
        encoder_vec[2] = self.leg_len /1.5
        encoder_vec = torch.tensor(encoder_vec, dtype=torch.float32, device=self.device) # Corrected device
        newly_reference = self.findgait(encoder_vec)
        newly_reference = np.clip(newly_reference, -np.pi/2, np.pi/2)
        current_ref_pos = np.array([self.reference[self.reference_idx+self.t,0], self.reference[self.reference_idx+self.t,1], 
                             self.reference[self.reference_idx+self.t,3],self.reference[self.reference_idx+self.t,4]])
        
        ref_pos = np.array([newly_reference[:,0], newly_reference[:,1], 
                             newly_reference[:,3],newly_reference[:,4]]).T
        distances = np.linalg.norm(ref_pos - current_ref_pos, axis=1)
        closest_index = np.argmin(distances) 
        self.t = closest_index
        self.reference = newly_reference
        self.reference_speed = new_speed


    def get_follow_camera_image(self, follow_distance=3.0, height=1.5,overlay_text=None):
        torso_state = self.p.getLinkState(self.robot, 2, physicsClientId=self.physics_client)
        torso_pos = torso_state[0]
        
        camera_eye = [torso_pos[0] - follow_distance, torso_pos[1], height]
        camera_target = [torso_pos[0], torso_pos[1], height]

        view_matrix = self.p.computeViewMatrix(
            cameraEyePosition=camera_eye,
            cameraTargetPosition=camera_target,
            cameraUpVector=[0, 0, 1]
        )
        projection_matrix = self.p.computeProjectionMatrixFOV(
            fov=75,
            aspect=1.0,
            nearVal=0.1,
            farVal=100.0
        )
        res = 640
        _, _, rgbImg, _, _ = self.p.getCameraImage(
            width=res,
            height=res,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix
        )
        rgb_array = np.reshape(rgbImg, (res, res, 4))
        image = Image.fromarray(rgb_array[:, :, :3], 'RGB')

        if overlay_text:
            draw = ImageDraw.Draw(image)
            try:
                font = ImageFont.truetype("DejaVuSansMono.ttf", 22)
            except:
                font = ImageFont.load_default()
            text = overlay_text
            pad = 6
            tw, th = draw.textbbox((0,0), text, font=font)[2:]
            box = [(10, 10), (10 + tw + 2*pad, 10 + th + 2*pad)]
            draw.rectangle(box, fill=(0,0,0,160))
            draw.text((10+pad, 10+pad), text, fill=(255,255,255), font=font)
        return image

    def return_step_taken(self):
        return self.taken_step_counter

    def collect_observation(self):
        # hist: (50, 6) -> (300,)
        x = self.amp_q_hist.reshape(-1).astype(np.float32)
        if hasattr(self, "amp_mean") and hasattr(self, "amp_std"):
            # if your stats are torch, move to cpu numpy for consistency
            mean = self.amp_mean.detach().cpu().numpy().reshape(-1).astype(np.float32)
            std  = self.amp_std.detach().cpu().numpy().reshape(-1).astype(np.float32)
            std  = np.clip(std, 1e-6, None)
            x = (x - mean) / std
        return x  # np.float32, (300,)
