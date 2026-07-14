import pybullet as p
import pybullet_data
import time
import numpy as np
import random
import math
from RRT import Planner, Node


p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.loadURDF("plane.urdf")
robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)


########### WORLD ############

class Sphere:

    def __init__(self, radius=0.01, mass=0, color=[0, 0, 1, 1]):

        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=radius
        )

        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=color
        )

        self.id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[10, 10, 10]   # Posición inicial
        )

    def update_position(self, x, y, z):

        p.resetBasePositionAndOrientation(
            self.id,
            [x, y, z],
            [0, 0, 0, 1]
        )

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


##### Algorithm ######

Planner = Planner()



##### UI ######

sphere = Sphere()
target_x = p.addUserDebugParameter("target x", 0.2, 0.8, 0.5)
target_y = p.addUserDebugParameter("target y", -0.5, 0.5, 0)
target_z = p.addUserDebugParameter("target z", 0.2, 0.8, 0.5)

while True:
    x = p.readUserDebugParameter(target_x)
    y = p.readUserDebugParameter(target_y)
    z = p.readUserDebugParameter(target_z)
    sphere.update_position(x,y,z)
    point = np.array([x, y, z])
    joint_angles = InvereseKinematics(point)
    set_joint_angles(joint_angles,force=1)
    p.stepSimulation()
    time.sleep(1/sim_frequency)
