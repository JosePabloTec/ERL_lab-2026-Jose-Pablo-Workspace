import os
import pybullet as p
import pybullet_data
import time
import numpy as np
import math


class PIDController:

    def __init__(self):

        p.connect(p.GUI)

        p.resetSimulation()

        p.setGravity(0, 0, -10)

        p.setRealTimeSimulation(1)

        p.resetDebugVisualizerCamera(
            cameraDistance=15,
            cameraYaw=0,
            cameraPitch=-55,
            cameraTargetPosition=[0, 0, 0]
        )

        p.loadSDF(
            os.path.join(
                pybullet_data.getDataPath(),
                "stadium.sdf"
            )
        )

        self.car = p.loadURDF(
            os.path.join(
                pybullet_data.getDataPath(),
                "racecar/racecar.urdf"
            )
        )

        self.tol = 3e-3

        self.x = float(input("Target X: "))
        self.y = float(input("Target Y: "))

        self.target_location = np.array([self.x, self.y])


        # 1.029052003	4.507840493	0.550338902	0.57989302	0.009513766	0.046015469

        self.k1 = 1.029052003
        self.k2 = 4.507840493

        self.k11 = 0.550338902
        self.k22 = 0.57989302

        self.previous_d = 0
        self.previous_dtheta = 0

        self.k111 = 0.009513766
        self.k222 = 0.046015469

        self.integral_d = 0
        self.integral_theta = 0

        self.dt = 0.01

    def PID(self):

        pos, orien = p.getBasePositionAndOrientation(self.car)
        rotation = p.getMatrixFromQuaternion(orien)

        x_current = pos[0]
        y_current = pos[1]

        dx = self.x - x_current
        dy = self.y - y_current

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

        linear_speed = self.k1 * d
        angular_speed = self.k2 * dtheta

        self.integral_d = self.integral_d + d * self.dt
        self.integral_theta = self.integral_theta + dtheta * self.dt

        linear_speed = (
            linear_speed +
            self.k111 * self.integral_d
        )

        angular_speed = (
            angular_speed +
            self.k222 * self.integral_theta
        )

        derivative_d = (
            d - self.previous_d
        ) / self.dt

        derivative_theta = (
            dtheta - self.previous_dtheta
        ) / self.dt

        linear_speed = (
            linear_speed +
            self.k11 * derivative_d
        )

        angular_speed = (
            angular_speed +
            self.k22 * derivative_theta
        )

        self.previous_d = d
        self.previous_dtheta = dtheta

        if d < self.tol:
            self.x = float(input("Target X: "))
            self.y = float(input("Target Y: "))
            self.target_location = np.array([self.x, self.y])

        print("Position:")
        print(f"x: {pos[0]:.2f}, y: {pos[1]:.2f}, z: {pos[2]:.2f}")

        print("Orientation:")
        print(f"roll: {p.getEulerFromQuaternion(orien)[0]:.2f}, pitch: {p.getEulerFromQuaternion(orien)[1]:.2f}, yaw: {p.getEulerFromQuaternion(orien)[2]:.2f}")

        print("Rotation matrix:")
        print(np.array(rotation).reshape(3, 3))

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
            self.car,
            linearVelocity=velocity,
            angularVelocity=[
                0,
                0,
                angular_speed
            ]
        )

    def run(self):

        while True:
            self.PID()
            time.sleep(self.dt)


controller = PIDController()
controller.run()