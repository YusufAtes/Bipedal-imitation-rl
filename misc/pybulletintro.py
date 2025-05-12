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
ramp_angle = 2 * np.pi / 180 # Random angle between -3 and 3
plane_orientation = p.getQuaternionFromEuler([ramp_angle, 0 , 0])
# Load the plane
planeId = p.loadURDF("assets/plane.urdf",baseOrientation=plane_orientation)
p.changeDynamics(planeId, -1, lateralFriction=0.65, contactStiffness=1e5, contactDamping=6e3)
# p.setTimeStep(0.001)
joint_id = 2
joint_idx = [2]

# # Load the bipedal robot model
cubeStartPos = [0, 0, 1.185]
cubeStartOrientation = p.getQuaternionFromEuler([0., 0., 0.])
robot = p.loadURDF("assets/biped2d.urdf", cubeStartPos, cubeStartOrientation)
p.setJointMotorControlArray(robot,[0,1,2,3,4,5,6,7,8], p.VELOCITY_CONTROL, forces=[0,0,0,0,0,0,0,0,0])
hip_init = 0.4
knee_init = -0.25
p.resetJointState(robot, 3, targetValue = hip_init)
p.resetJointState(robot, 4, targetValue = knee_init)
p.resetJointState(robot, 5, targetValue = 0)
# if left_flat:
#     self.p.resetJointState(self.robot, 5, targetValue = 0)
# else:
#     self.p.resetJointState(self.robot, 5, targetValue = -(rhip_pos+ rknee_pos))

p.resetJointState(robot, 6, targetValue = -hip_init)
p.resetJointState(robot, 7, targetValue = knee_init)
p.resetJointState(robot, 8, targetValue = 0)
# cubeStartPos = [0, 0, -0.3]
# cubeStartOrientation = p.getQuaternionFromEuler([0., 0., 0.])
# robot = p.loadURDF("assets/biped2d_pybullet.urdf")
# Get number of joints
num_joints = p.getNumJoints(robot)
for i in range(num_joints):
    joint_info = p.getJointInfo(robot, i)
    joint_name = joint_info[1].decode('utf-8')  # Decode joint name
    print(f"Joint ID: {i}, Joint Name: {joint_name}")
time.sleep(1)
# p.setJointMotorControl2(robot,joint_id, p.VELOCITY_CONTROL, force=0)
# p.enableJointForceTorqueSensor(robot, joint_id)
# # p.setTimeStep(1/1000)
# print('==========================')
for i in range(1000):
    # p.setJointMotorControl2(robot, joint_id, p.TORQUE_CONTROL, force=-200)
    # p.setJointMotorControlArray(robot, joint_idx, controlMode=p.TORQUE_CONTROL, forces=[-200])
    # p.setJointMotorControl2(robot, 2, controlMode=p.POSITION_CONTROL,targetPosition=0.3)
    p.stepSimulation()
    a = p.getContactPoints(robot, planeId)
    time.sleep(0.01)
# print('Simulation done')
# time.sleep(3)
# # # Print joint names and IDs
# # print("Joint Information:")
# # for joint_id in range(num_joints):
# #     joint_info = p.getJointInfo(robot, joint_id)
# #     joint_name = joint_info[1].decode('utf-8')  # Decode joint name
# #     print(f"Joint ID: {joint_id}, Joint Name: {joint_name}")
# #     if joint_info[2] == p.JOINT_REVOLUTE:  # Ensure it's a revolute joint
# #         p.setJointMotorControl2(robot, joint_id, p.POSITself.planeId = p.loadURDF("assets/pla