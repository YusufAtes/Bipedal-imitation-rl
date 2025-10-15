import numpy as np
import time
import yaml
import argparse
from copy import deepcopy
from env import Biped2dBullet; 
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode",type=str, default='walk', help="specify mode of locomotion to run from config file")
parser.add_argument("-f", "--file_path", type=str, default="/home/baran/Bipedal-imitation-rl/locomotion-master(1)/locomotion-master/settings/cma_config_1.20.yml", help="path to config file (yml format)")
parser.add_argument("-t", "--sim_dt", type=float, default=0.0004, help="simulation timestep")
parser.add_argument("-i", "--init_index", type=int, default=0, help="starting index of the FSM")
parser.add_argument("-mi", "--max_iters", type=int, default=10e4, help="iterations of episode")
parser.add_argument("-s", "--save_trajectory", action='store_true', default=False, help="whether to save the motion trajectory (needed for imitation learning)")
parser.add_argument("-sp", "--save_path", type=str, default="", help="save_path of motion trajectory")
args = parser.parse_args()
mode = args.mode

def run_simbicon(mode,file_path, sim_dt, init_index, max_iters, save_trajectory, ramp_angle,ground_resolution=None,heightfield_data=None):

    filename = file_path
    with open(filename, 'r') as f:
        f_data = yaml.load(f, Loader=yaml.FullLoader)
    settings = f_data[mode]
    curr_index = init_index
    foot_iters = int(settings['dt']/sim_dt)

    max_iters = max_iters

    env = Biped2dBullet(); 
    env.dt = sim_dt
    env.mu = settings['mu']
    env.max_torque = settings['max_torque']
    env.max_iters = max_iters
    env.g = settings['g']
    env.kp = settings['kp']
    env.kd = settings['kd']


    angles = settings['targets']
    #torso control and swing placement params 
    kp_torso = settings['kp_torso']
    kd_torso = settings['kd_torso']
    c_d = settings['c_d']
    c_v = settings['c_v']

    rhip_posses = []  # IGNORE
    rknee_posses = []  # IGNORE
    rankle_posses = []  # IGNORE
    lhip_posses = []  # IGNORE
    lknee_posses = []  # IGNORE
    lankle_posses = []  # IGNORE
    if save_trajectory:
        traj = []
    def get_state_index(state, curr_index, iters_this_state, foot_iters):
        """Returns the next state in the FSM model
        """
        if curr_index==0 or curr_index==2:
            if iters_this_state>foot_iters:
                curr_index+=1
        elif curr_index==1:
            #check if right foot made ground contact
            if abs(state[-2]-1.)<1e-5:
                curr_index+=1
        else:
            if abs(state[-1]-1.)<1e-5:
                curr_index=0
        return curr_index    

    iters_this_state=0
    state = env.reset(ramp_angle=ramp_angle,ground_resolution=ground_resolution,heightfield_data=heightfield_data)  #state has 39 dimensions

    for i in range(int(max_iters)):
        targ_ang = deepcopy(angles[curr_index])
        action = np.concatenate([np.array(targ_ang[1:]), np.zeros(6)])  
        
        #BALANCE FEEDBACK
        if (abs(state[-2]-1.)<1e-5):
            #right foot on ground
            swing_ind = 3
            stance_ind = 0
            if curr_index==3: 
                d = state[14+0*(6)+1] - state[14+3*(6)+1] 
                v =  state[14+0*6+4]
                action[3] = action[3] + c_v*v +c_d*d
            
        elif (abs(state[-1]-1.)<1e-5 or i==0):
            #left foot on ground
            swing_ind= 0
            stance_ind = 3
            if curr_index==1: 
                d = state[14+0*(6)+1] - state[14+6*(6)+1] 
                v = state[14+0*6+4]
                action[0] = action[0] + c_v*v +c_d*d 
        #no pd target for stance hip     
        action[stance_ind] = np.nan
        
        #TORSO CONTROL    
        theta_torso = state[0]
        omega_torso = state[1]
        rhip_posses.append(state[2])  # IGNORE
        rknee_posses.append(state[4])  # IGNORE
        rankle_posses.append(state[6])  # IGNORE
        lhip_posses.append(state[8])  # IGNORE
        lknee_posses.append(state[10])  # IGNORE
        lankle_posses.append(state[12])  # IGNORE

        torque_stance = -kp_torso*(theta_torso-targ_ang[0]) - kd_torso*omega_torso #-0.065
        torque_stance += -env.kp[swing_ind]*(state[2*(swing_ind+1)]-targ_ang[swing_ind+1]) - env.kd[swing_ind]*state[2*(swing_ind+1)+1]
        action[6+stance_ind] = -torque_stance  
        
        state, reward, done, _ = env.step(action)       #16 is the Z position of the torso, 15 is the Y position of the torso
        # env.render()
        # Code for Done Condition

        if np.abs(theta_torso)>0.9:
            env.close()
            return 0,ramp_angle,False, state[15]
        
        elif state[16] > 1.45 + np.tan(ramp_angle*np.pi/180) * state[15]:
            env.close()
            return 0,ramp_angle,False, state[15]
        
        elif state[16] < 0.8 + np.tan(ramp_angle*np.pi/180) * state[15]:
            env.close()
            return 0,ramp_angle,False, state[15]
        
        iters_this_state+=1
        next_ind = get_state_index(state,curr_index, iters_this_state, foot_iters = foot_iters)
        if next_ind!=curr_index:
            iters_this_state=0
        curr_index = next_ind

        if state[15] > 9.99:
            return 0, ramp_angle, True, state[15]

    avg_speed = state[15]/(max_iters*env.dt)
    env.close()
    return avg_speed, ramp_angle, True, state[15]

# # ROTATION DEMO DATA COLLECTION
# record_data = pd.DataFrame(columns=["demo type", "cmd speed", "angle", "mean speed","noise level",
#                                         "resolution","success","max range","trial_no"])
# angles = np.linspace(-15,15,31)
# speeds = np.linspace(0.1,2.0,21)

# total_runs = len(angles)*len(speeds)
# i = 0
# for angle in angles:
#     for speed in speeds:
#         i+=1
#         if i%30==0:
#             print(f"Completed {i} out of {total_runs} runs")
#         file_path = f"/home/baran/Bipedal-imitation-rl/locomotion-master(1)/locomotion-master/settings/cma_config_{speed:.2f}.yml"

#         avg_speed, ramp_angle, success = run_simbicon(mode, file_path=file_path, sim_dt=args.sim_dt, init_index=args.init_index,
#                      max_iters=args.max_iters, save_trajectory=args.save_trajectory, ramp_angle=angle)

#         record_data = pd.concat([record_data, pd.DataFrame([{"demo type": 'rotation', "cmd speed": speed, "angle": angle,
#                         "mean speed": avg_speed,"noise level": None,
#                         "resolution": None,"success": success,"max range": None,
#                         "trial_no": None}])], ignore_index=True)

# record_data.to_csv("simbicon_rotation_data.csv", index=False)


# # VELOCITY DIFFERENCE DEMO DATA COLLECTION
# angle = 0.0
# record_data = pd.DataFrame(columns=["demo type", "cmd speed", "angle", "mean speed","noise level",
#                                         "resolution","success","max range","trial_no"])
# speeds = np.linspace(0.1,2.0,41)
# trial_no = 1
# total_runs = len(speeds)
# i = 0

# for speed in speeds:

#     for trial in range(trial_no):
#         file_path = f"/home/baran/Bipedal-imitation-rl/locomotion-master(1)/locomotion-master/settings/cma_config_{speed:.2f}.yml"

#         avg_speed, ramp_angle, success, y_pos = run_simbicon(mode, file_path=file_path, sim_dt=args.sim_dt, init_index=args.init_index,
#                         max_iters=args.max_iters, save_trajectory=args.save_trajectory, ramp_angle=angle)
#         if success:
#             record_data = pd.concat([record_data, pd.DataFrame([{"demo type": 'veldiff', "cmd speed": speed, "angle": angle,
#                             "mean speed": avg_speed,"noise level": None,
#                             "resolution": None,"success": success,"max range": None,
#                             "trial_no": trial}])], ignore_index=True)
#             break
#         elif trial==trial_no-1:
#             record_data = pd.concat([record_data, pd.DataFrame([{"demo type": 'veldiff', "cmd speed": speed, "angle": angle,
#                             "mean speed": avg_speed,"noise level": None,
#                             "resolution": None,"success": success,"max range": None,
#                             "trial_no": trial}])], ignore_index=True)

#     i+=1
#     if i%10==0:
#         print(f"Completed {i} out of {total_runs} runs")
# record_data.to_csv("simbicon_velocity_diff.csv", index=False)


# NOISY PLANE 0 DEMO DATA COLLECTION

angle = 0.0
scenario_mode = 0
record_data = pd.DataFrame(columns=["demo type", "cmd speed", "angle", "mean speed","noise level",
                                        "resolution","success","max range","trial_no"])
speeds = np.linspace(0.1,2.0,21)
trial_no = 1
total_runs = len(speeds)
i = 0
noise_levels = np.arange(1,20)
total_runs = len(speeds)*len(noise_levels)*4

for gamma in [0.25,0.5,1.0,2.0]:

    for noise_level in noise_levels:
        # Scenario count is always 1 (this for loop is just for future use)
        # Generate heightfield data with noise in the range [-ground_noise, ground_noise]
        heightfield_data = np.load(f"/home/baran/Bipedal-imitation-rl/noise_planes/plane_{gamma}_0.npy")
        heightfield_data = heightfield_data * noise_level 

        for speed in speeds:

            for trial in range(trial_no):
                file_path = f"/home/baran/Bipedal-imitation-rl/locomotion-master(1)/locomotion-master/settings/cma_config_{speed:.2f}.yml"

                avg_speed, ramp_angle, success, y_pos = run_simbicon(mode, file_path=file_path, sim_dt=args.sim_dt, init_index=args.init_index,
                                max_iters=args.max_iters, save_trajectory=args.save_trajectory, ramp_angle=angle, 
                                ground_resolution=gamma,heightfield_data=heightfield_data)
            
                if success:
                    record_data = pd.concat([record_data, pd.DataFrame([{"demo type": 'rotation', "cmd speed": speed, "angle": angle,
                                    "mean speed": avg_speed,"noise level": noise_level,
                                    "resolution": gamma,"success": success,"max range": y_pos,
                                    "trial_no": trial}])], ignore_index=True)

                elif trial==trial_no-1:
                    record_data = pd.concat([record_data, pd.DataFrame([{"demo type": 'rotation', "cmd speed": speed, "angle": angle,
                                    "mean speed": avg_speed,"noise level": noise_level,
                                    "resolution": gamma,"success": success,"max range": y_pos,
                                    "trial_no": trial}])], ignore_index=True)


            i+=1
            if i%100==0:
                print(f"Completed {i} out of {total_runs} runs")
print("Dome Noise 0 is done")
record_data.to_csv("simbicon_noisyplane_0.csv", index=False)


# NOISY PLANE 1 DEMO DATA COLLECTION

angle = 0.0
scenario_mode = 0
record_data = pd.DataFrame(columns=["demo type", "cmd speed", "angle", "mean speed","noise level",
                                        "resolution","success","max range","trial_no"])
speeds = np.linspace(0.1,2.0,21)
trial_no = 1
total_runs = len(speeds)
i = 0
noise_levels = np.arange(1,20)
total_runs = len(speeds)*len(noise_levels)*4

for gamma in [0.25,0.5,1.0,2.0]:

    for noise_level in noise_levels:
        # Scenario count is always 1 (this for loop is just for future use)
        # Generate heightfield data with noise in the range [-ground_noise, ground_noise]
        heightfield_data = np.load(f"/home/baran/Bipedal-imitation-rl/noise_planes/plane_{gamma}_0.npy")
        heightfield_data = heightfield_data * noise_level 

        for speed in speeds:

            for trial in range(trial_no):
                file_path = f"/home/baran/Bipedal-imitation-rl/locomotion-master(1)/locomotion-master/settings/cma_config_{speed:.2f}.yml"

                avg_speed, ramp_angle, success, y_pos = run_simbicon(mode, file_path=file_path, sim_dt=args.sim_dt, init_index=args.init_index,
                                max_iters=args.max_iters, save_trajectory=args.save_trajectory, ramp_angle=angle, 
                                ground_resolution=gamma,heightfield_data=heightfield_data)
                
                if success:
                    record_data = pd.concat([record_data, pd.DataFrame([{"demo type": 'rotation', "cmd speed": speed, "angle": angle,
                                    "mean speed": avg_speed,"noise level": noise_level,
                                    "resolution": gamma,"success": success,"max range": y_pos,
                                    "trial_no": trial}])], ignore_index=True)

                elif trial==trial_no-1:
                    record_data = pd.concat([record_data, pd.DataFrame([{"demo type": 'rotation', "cmd speed": speed, "angle": angle,
                                    "mean speed": avg_speed,"noise level": noise_level,
                                    "resolution": gamma,"success": success,"max range": y_pos,
                                    "trial_no": trial}])], ignore_index=True)

            i+=1
            if i%100==0:
                print(f"Completed {i} out of {total_runs} runs")
print(f"Demo Noise 1 is done")
record_data.to_csv("simbicon_noisyplane_1.csv", index=False)