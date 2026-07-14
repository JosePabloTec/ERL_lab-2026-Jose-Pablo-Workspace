import pybullet as p
import pybullet_data
import time
import numpy as np
import random
import math
from RRT import Planner

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")
robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)


########### WORLD ############

def create_sphere(x, y, z, radius=0.2, mass=0, color=[1, 0, 0, 1]):
    collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_SPHERE,
        radius=radius
    )

    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=color
    )

    sphere_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=[x, y, z]
    )

    return sphere_id

end_effector = 11

num_joints = p.getNumJoints(robot)

for i in range(num_joints):
    info = p.getJointInfo(robot, i)
    #print(i, info[1].decode("utf-8"))


def display_positions():
    for i in range(7):
        name = p.getJointInfo(robot, i)[1].decode("utf-8")
        angle = p.getJointState(robot, i)[0]
        print(f"{name}: {angle:.3f} rad")


def set_joint_angles(arm_angles,force=200):
    for i in range(7):
        p.setJointMotorControl2(
            bodyUniqueId=robot,
            jointIndex=i,
            controlMode=p.POSITION_CONTROL,
            targetPosition=arm_angles[i],
            force=500
        )

sim_frequency = 300 # Hz

def InvereseKinematics(point):
    joint_angles = p.calculateInverseKinematics(robot,end_effector,point)
    return joint_angles


######## ALGORITHM #########

class Node:
    def __init__(self, x, y, z, g, parent=None):
        self.x = x 
        self.y = y 
        self.z = z
        self.g = g
        self.parent = parent # Pointer to the parent node
        self.children = []



create_sphere(0.5376981943753544, -0.30919789130409775, 0.682937251122326,radius=0.01)

while True:
    point = np.array([0.5376981943753544, -0.30919789130409775, 0.682937251122326])
    joint_angles = InvereseKinematics(point)
    set_joint_angles(joint_angles,force=1)
    p.stepSimulation()
    time.sleep(1/sim_frequency)
