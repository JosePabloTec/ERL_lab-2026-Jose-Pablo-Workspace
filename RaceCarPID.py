import os
import pybullet as p
import pybullet_data
import time
import numpy as np
import math


p.connect(p.GUI)

p.resetSimulation()

p.setGravity(0,0,-10)

p.setRealTimeSimulation(1)


p.resetDebugVisualizerCamera(
    cameraDistance=15,
    cameraYaw=0,
    cameraPitch=-55,
    cameraTargetPosition=[0,0,0]
)


p.loadSDF(
    os.path.join(
        pybullet_data.getDataPath(),
        "stadium.sdf"
    )
)


car = p.loadURDF(
    os.path.join(
        pybullet_data.getDataPath(),
        "racecar/racecar.urdf"
    )
)



x = 0
y = -5

target_location = np.array([x,y])


k1 = 1
k2 = 1

k11 = 0.1
k22 = 0.1

previous_d = 0
previous_dtheta = 0

k111 = 0.001
k222 = 0.001

integral_d = 0
integral_theta = 0

dt = 0.01


while True:

    pos, orien = p.getBasePositionAndOrientation(car)
    rotation = p.getMatrixFromQuaternion(orien)

    x_current = pos[0]
    y_current = pos[1]

    dx = x - x_current
    dy = y - y_current


    d = math.sqrt(
        dx**2 + dy**2
    )


    angle = math.atan2(
        dy,
        dx
    )

    angle_current = p.getEulerFromQuaternion(orien)[2]

    dtheta = angle - angle_current

    dtheta = math.atan2(
        math.sin(dtheta),
        math.cos(dtheta)
    )


    # Proportional

    linear_speed = k1*d
    angular_speed = k2*dtheta

    # Integral

    integral_d = integral_d + d*dt
    integral_theta = integral_theta + dtheta*dt

    linear_speed = (

        linear_speed +
        k111*integral_d

    )

    angular_speed = (

        angular_speed +
        k222*integral_theta

    )



    # Derivative

    derivative_d = (
        d - previous_d
    ) / dt


    derivative_theta = (
        dtheta - previous_dtheta
    ) / dt



    linear_speed = (
        linear_speed +
        k11*derivative_d

    )


    angular_speed = (
        angular_speed +
        k22*derivative_theta

    )



    previous_d = d
    previous_dtheta = dtheta



    print("Position:")
    print(f"x: {pos[0]:.2f}, y: {pos[1]:.2f}, z: {pos[2]:.2f}")

    print("Orientation:")
    print(f"roll: {p.getEulerFromQuaternion(orien)[0]:.2f}, pitch: {p.getEulerFromQuaternion(orien)[1]:.2f}, yaw: {p.getEulerFromQuaternion(orien)[2]:.2f}")

    print("Rotation matrix:")
    print(np.array(rotation).reshape(3,3))

    print("----------------------")


    forward = [

        rotation[0],
        rotation[3],
        rotation[6]

    ]



    velocity = [

        forward[0] * linear_speed,
        forward[1] * linear_speed,
        0

    ]



    p.resetBaseVelocity(

        car,
        linearVelocity=velocity,
        angularVelocity=[
            0,
            0,
            angular_speed

        ]

    )

    time.sleep(0.01)