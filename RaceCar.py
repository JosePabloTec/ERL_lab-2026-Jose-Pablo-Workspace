import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

import pybullet as p
import pybullet_data
import time


# Connect
cid = p.connect(p.SHARED_MEMORY)

if cid < 0:
    p.connect(p.GUI)

p.resetDebugVisualizerCamera(
    cameraDistance=15,
    cameraYaw=-0,
    cameraPitch=-55,
    cameraTargetPosition=[0, 0, 0]
)

p.resetSimulation()
p.setGravity(0,0,-10)

p.setRealTimeSimulation(1)


# Load environment
p.loadSDF(
    os.path.join(pybullet_data.getDataPath(), "stadium.sdf")
)


# Load car
car = p.loadURDF(
    os.path.join(pybullet_data.getDataPath(),
                 "racecar/racecar.urdf")
)


# --------------------------
# CONSTANT VELOCITIES
# --------------------------

linear_speed = 2      # m/s forward
angular_speed = 1   # rad/s turning


# --------------------------
# LOOP
# --------------------------

while True:

    # Get car orientation
    pos, orn = p.getBasePositionAndOrientation(car)


    # Convert local forward direction to world direction
    rotation = p.getMatrixFromQuaternion(orn)

    forward = [
        rotation[0],
        rotation[3],
        rotation[6]
    ]


    # Linear velocity vector
    linear_velocity = [
        forward[0] * linear_speed,
        forward[1] * linear_speed,
        0
    ]


    # Apply linear + angular velocity
    p.resetBaseVelocity(
        car,
        linearVelocity=linear_velocity,
        angularVelocity=[0,0,angular_speed]
    )


    time.sleep(0.01)