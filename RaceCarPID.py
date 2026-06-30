import os
import pybullet as p
import pybullet_data
import time
import numpy as np
import math





p.connect(p.GUI)

p.resetSimulation()

p.setGravity(0, 0, -10)

p.setRealTimeSimulation(1)

p.resetDebugVisualizerCamera(
    cameraDistance=15,
    cameraYaw=0,
    cameraPitch=-35,
    cameraTargetPosition=[0, 0, 0]
)

p.loadSDF(
    os.path.join(
        pybullet_data.getDataPath(),
        "stadium.sdf"
    )
)


def create_obstacle(
    x,
    y,
    size=None,
    width=None,
    length=None,
    height=None,
    color=[0, 0, 1, 1]
):
    """
    Creates either:
      - a cube: specify 'size'
      - a rectangular box: specify width, length, and height
    """

    if size is not None:
        half_extents = [size, size, size]
        z = size
    else:
        if width is None or length is None or height is None:
            raise ValueError(
                "Specify either 'size' or all of 'width', 'length', and 'height'."
            )

        half_extents = [
            width / 2,
            length / 2,
            height / 2
        ]
        z = height / 2

    collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=half_extents
    )

    visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=color
    )

    obstacle = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=[x, y, z]
    )

    return obstacle


"""
create_obstacle(
    x=5,
    y=5,
    width=0.5,
    length=0.5,
    height=1
)
"""

class PIDController:

    def __init__(self):

        self.car = p.loadURDF(
            os.path.join(
                pybullet_data.getDataPath(),
                "racecar/racecar.urdf"
            )
        )

        self.tol = 5e-3

        self.x = float(input("Target X: "))
        self.y = float(input("Target Y: "))

        self.target_location = np.array([self.x, self.y])

        # PID gains
        self.k1 = 1.029052003
        self.k2 = 4.507840493

        self.k11 = 0.4
        self.k22 = 0.57989302

        self.k111 = 0.03
        self.k222 = 0.001

        self.previous_d = 0
        self.previous_dtheta = 0

        self.integral_d = 0
        self.integral_theta = 0

        self.dt = 0.01

        # -----------------------
        # State machine
        # -----------------------
        self.state = "TURN"

        self.turn_threshold = math.radians(5)
        self.return_to_turn_threshold = math.radians(10)

    def PID(self):

        pos, orien = p.getBasePositionAndOrientation(self.car)

        rotation = p.getMatrixFromQuaternion(orien)

        x_current = pos[0]
        y_current = pos[1]

        dx = self.x - x_current
        dy = self.y - y_current

        d = math.sqrt(dx ** 2 + dy ** 2)

        desired_heading = math.atan2(dy, dx)

        current_heading = p.getEulerFromQuaternion(orien)[2]

        dtheta = desired_heading - current_heading

        dtheta = math.atan2(
            math.sin(dtheta),
            math.cos(dtheta)
        )

        # -----------------------
        # PID
        # -----------------------

        self.integral_d += d * self.dt
        self.integral_theta += dtheta * self.dt

        derivative_d = (
            d - self.previous_d
        ) / self.dt

        derivative_theta = (
            dtheta - self.previous_dtheta
        ) / self.dt

        pid_linear = (
            self.k1 * d +
            self.k111 * self.integral_d +
            self.k11 * derivative_d
        )

        pid_angular = (
            self.k2 * dtheta +
            self.k222 * self.integral_theta +
            self.k22 * derivative_theta
        )

        self.previous_d = d
        self.previous_dtheta = dtheta

        # -----------------------
        # State machine
        # -----------------------

        if self.state == "TURN":

            linear_speed = 0.0
            angular_speed = pid_angular

            if abs(dtheta) < self.turn_threshold:
                self.state = "GO"

        elif self.state == "GO":

            linear_speed = pid_linear
            angular_speed = pid_angular

            if abs(dtheta) > self.return_to_turn_threshold:
                self.state = "TURN"

        # -----------------------
        # Goal reached
        # -----------------------

        if d < self.tol:

            print("Goal reached!")

            self.x = float(input("Target X: "))
            self.y = float(input("Target Y: "))

            self.target_location = np.array([self.x, self.y])

            self.state = "TURN"

            self.integral_d = 0
            self.integral_theta = 0

            self.previous_d = 0
            self.previous_dtheta = 0

            return

        print(f"State: {self.state}")

        print("Position:")
        print(f"x: {pos[0]:.2f}, y: {pos[1]:.2f}, z: {pos[2]:.2f}")

        print("Orientation:")
        print(
            f"roll: {p.getEulerFromQuaternion(orien)[0]:.2f}, "
            f"pitch: {p.getEulerFromQuaternion(orien)[1]:.2f}, "
            f"yaw: {p.getEulerFromQuaternion(orien)[2]:.2f}"
        )

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