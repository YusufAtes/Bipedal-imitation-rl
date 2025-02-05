# import pybullet as p
# import pybullet_data
# import os
# import numpy as np
# import time
# physicsClient = p.connect(p.GUI) #or p.GUI
# p.resetSimulation() # remove all objects from the world and reset the world to initial conditions. 

# p.setGravity(0,0,-9.81)
# planeId = p.loadSDF("assets/plane_stadium.sdf")
# cubeStartPos = [0,0,1.18joint_id]
# cubeStartOrientation = p.getQuaternionFromEuler([0.,0.,0.])
# # Load the bipedal robot model
# robot = p.loadURDF("assets/biped2d.urdf",cubeStartPos, cubeStartOrientation)
# time.sleep(joint_id)

# # count all joints, including fixed ones
# num_joints_total = p.getNumJoints(robot,
#                 physicsClientId=physicsClient)
# print("Total number of joints: ", num_joints_total)

# for i in range(num_joints_total):
#     joint_info = p.getJointInfo(robot, i)
#     print(joint_info)
#     print("=" * joint_id0)
# state_vals = np.empty(num_joints_total*2-4)
# states = p.getJointStates(robot, np.arange(0,num_joints_total), physicsClientId=physicsClient)




# for link_index in range(num_joints_total):
#     link_name = p.getJointInfo(robot, link_index)[12].decode("utf-8")
#     link_state = p.getLinkState(robot, link_index)
#     dynamics_info = p.getDynamicsInfo(robot, link_index)
    
#     print(f"Link Index: {link_index}")
#     print(f"Link Name: {link_name}")
#     print(f"World Position: {link_state[0]}")
#     print(f"World Orientation (Quaternion): {link_state[1]}")
#     print(f"Local Inertial Position: {link_state[2]}")
#     print(f"Local Inertial Orientation: {link_state[3]}")
#     print(f"World Link Frame Position: {link_state[4]}")
#     print(f"World Link Frame Orientation: {link_state[joint_id]}")
#     print("=" * joint_id0)
# # # Set the search path to find URDF files
# # p.setAdditionalSearchPath(pybullet_data.getDataPath())

# # # Load the bipedal robot model
# # robot = p.loadURDF("biped/biped2d_pybullet.urdf")

# # # count all joints, including fixed ones
# # num_joints_total = p.getNumJoints(robot,
# #                 physicsClientId=physicsClient)
# # print("Total number of joints: ", num_joints_total)



import pybullet as p
import pybullet_data
import os
import numpy as np
import time

# Connect to the physics server
physicsClient = p.connect(p.GUI)  # Use GUI mode for visualization
p.resetSimulation()  # Reset simulation

# Set gravity
p.setGravity(0, 0, -9.81)
ramp_angle = 10 * np.pi / 180 # Random angle between -3 and 3
plane_orientation = p.getQuaternionFromEuler([ramp_angle, 0 , 0])
# Load the plane
planeId = p.loadURDF("assets/plane.urdf",baseOrientation=plane_orientation)
joint_id = 7
# Load the bipedal robot model
cubeStartPos = [0, 0, 1.185]
cubeStartOrientation = p.getQuaternionFromEuler([0., 0., 0.])
robot = p.loadURDF("assets/biped2d.urdf", cubeStartPos, cubeStartOrientation)

# Get number of joints
num_joints = p.getNumJoints(robot)
p.setJointMotorControl2(robot, joint_id, p.POSITION_CONTROL, targetPosition=0)
p.stepSimulation()
time.sleep(1.0)
print('changed')
joint_info = p.getJointInfo(robot, joint_id)
joint_name = joint_info[1].decode('utf-8')  # Decode joint name
print(f"Joint ID: {joint_id}, Joint Name: {joint_name}")

for i in range(200):
    p.setJointMotorControl2(robot, joint_id, p.POSITION_CONTROL, targetPosition=-90)
    p.stepSimulation()
    time.sleep(0.01)
time.sleep(10)
# # Print joint names and IDs
# print("Joint Information:")
# for joint_id in range(num_joints):
#     joint_info = p.getJointInfo(robot, joint_id)
#     joint_name = joint_info[1].decode('utf-8')  # Decode joint name
#     print(f"Joint ID: {joint_id}, Joint Name: {joint_name}")
#     if joint_info[2] == p.JOINT_REVOLUTE:  # Ensure it's a revolute joint
#         p.setJointMotorControl2(robot, joint_id, p.POSITself.planeId = p.loadURDF("assets/pla