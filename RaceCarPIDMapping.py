import os
import pybullet as p
import pybullet_data
import time
import numpy as np
import math
import matplotlib.pyplot as plt


def point_to_cell_coordinate(cell_size, x, y):

    x_cell = int(x // cell_size)
    y_cell = int(y // cell_size)

    return x_cell, y_cell



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
        "plane_stadium.sdf"
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


create_obstacle(
    x=5,
    y=5,
    width=3,
    length=2,
    height=1
)

create_obstacle(
    x=-7,
    y= 7,
    width=1,
    length=1,
    height=1
)

create_obstacle(
    x=-2,
    y=-4,
    width=1,
    length=1,
    height=1
)


class PIDController:

    def __init__(self):
        self.point_cloud = []
        self.lidar_debug_ids = []
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

        self.k11 = 0.2
        self.k22 = 0.2

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

        # Global point cloud
        self.point_cloud = []

        # Update every 0.5 s
        self.map_update_period = 0.5
        self.last_map_update = time.time()

        # Interactive plotting
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(7,7))
        # Occupancy grid

        self.cell_size = 0.5

        self.grid_size = 80

        self.occupancy_grid = np.zeros(
            (self.grid_size, self.grid_size)
        )

        # center map around robot world origin
        self.origin_x = -20
        self.origin_y = -20

        self.p_free = 0.35
        self.p_occ = 0.65

        self.l_occ = 0.25
        self.l_free = -0.12

        self.l_min = -5
        self.l_max = 5
        self.max_distance_rays = 10

    def get_lidar(self, num_rays=70, max_distance= 10, visualize=False):

        pos, orn = p.getBasePositionAndOrientation(self.car)

        x, y, z = pos

        yaw = p.getEulerFromQuaternion(orn)[2]

        ray_from = []
        ray_to = []

        for i in range(num_rays):

            angle = yaw + 2 * math.pi * i / num_rays

            ray_from.append([
                x,
                y,
                z + 0.2
            ])

            ray_to.append([
                x + max_distance * math.cos(angle),
                y + max_distance * math.sin(angle),
                z + 0.2
            ])


        results = p.rayTestBatch(
            ray_from,
            ray_to
        )

        lidar = []

        # remove old lines
        if visualize:
            for line in self.lidar_debug_ids:
                p.removeUserDebugItem(line)

            self.lidar_debug_ids = []


        for i, result in enumerate(results):

            hit = result[0]
            fraction = result[2]

            distance = (
                fraction * max_distance
                if hit != -1
                else max_distance
            )

            lidar.append(distance)


            if visualize:

                end = [
                    ray_from[i][0] +
                    (ray_to[i][0]-ray_from[i][0]) *
                    distance/max_distance,

                    ray_from[i][1] +
                    (ray_to[i][1]-ray_from[i][1]) *
                    distance/max_distance,

                    ray_from[i][2]
                ]

                line = p.addUserDebugLine(
                    ray_from[i],
                    end,
                    [1,0,0] if hit != -1 else [0,1,0],
                    lineWidth=1,
                    lifeTime=0
                )

                self.lidar_debug_ids.append(line)


        return np.array(lidar)

    def update_pointcloud_map(self, num_rays=360, max_distance=20):

        pos, orn = p.getBasePositionAndOrientation(self.car)

        x, y, z = pos
        yaw = p.getEulerFromQuaternion(orn)[2]

        lidar = self.get_lidar(
            num_rays=num_rays,
            max_distance=max_distance,
            visualize=False
        )

        for i, distance in enumerate(lidar):

            # Ignore rays with no obstacle
            if distance >= max_distance:
                continue

            angle = yaw + 2 * math.pi * i / num_rays

            px = x + distance * math.cos(angle)
            py = y + distance * math.sin(angle)

            self.point_cloud.append((px, py))

    def plot_pointcloud(self):

        self.ax.clear()

        if len(self.point_cloud):

            pts = np.array(self.point_cloud)

            self.ax.scatter(
                pts[:,0],
                pts[:,1],
                s=3,
                c='blue'
            )

        car_pos, _ = p.getBasePositionAndOrientation(self.car)

        self.ax.scatter(
            car_pos[0],
            car_pos[1],
            c='red',
            s=80,
            label='Car'
        )

        self.ax.set_xlim(-20,20)
        self.ax.set_ylim(-20,20)
        self.ax.set_aspect('equal')

        self.ax.set_title("Accumulated LiDAR Point Cloud")

        plt.draw()
        plt.pause(0.001)

    def world_to_grid(self,x,y):

        gx = int((x - self.origin_x) / self.cell_size)
        gy = int((y - self.origin_y) / self.cell_size)

        return gx, gy



    def update_occupancy_grid(self):

        pos, orn = p.getBasePositionAndOrientation(self.car)

        x,y,z = pos

        yaw = p.getEulerFromQuaternion(orn)[2]


        lidar = self.get_lidar(
            num_rays=360,
            max_distance=20,
            visualize=False
        )


        for i,distance in enumerate(lidar):


            angle = yaw + 2*np.pi*i/360


            # free cells along ray

            steps = int(distance/self.cell_size)


            for s in range(steps):

                rx = x + s*self.cell_size*np.cos(angle)
                ry = y + s*self.cell_size*np.sin(angle)


                gx,gy = self.world_to_grid(rx,ry)


                if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:


                    self.occupancy_grid[gx,gy] += self.l_free



            # occupied endpoint

            if distance < 10:


                end_x = x + distance*np.cos(angle)
                end_y = y + distance*np.sin(angle)


                gx,gy = self.world_to_grid(end_x,end_y)


                if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:


                    self.occupancy_grid[gx,gy] += self.l_occ



            # limit values

            self.occupancy_grid = np.clip(
                self.occupancy_grid,
                self.l_min,
                self.l_max
            )
    
    def plot_occupancy_grid(self):

        self.ax.clear()


        # Convert log-odds to probability
        probability_map = 1 / (
            1 + np.exp(-self.occupancy_grid)
        )


        img = self.ax.imshow(
            probability_map.T,
            origin="lower",
            cmap="gray",
            vmin=0,
            vmax=1,
            extent=[
                self.origin_x,
                self.origin_x + self.grid_size*self.cell_size,
                self.origin_y,
                self.origin_y + self.grid_size*self.cell_size
            ]
        )


        # draw robot
        pos,_ = p.getBasePositionAndOrientation(self.car)

        self.ax.scatter(
            pos[0],
            pos[1],
            c="red",
            s=50
        )


        self.ax.set_title(
            "Probabilistic OGM"
        )

        self.ax.set_aspect(
            "equal"
        )


        plt.draw()
        plt.pause(0.001)

    def PID(self):

        lidar = self.get_lidar(
            num_rays=360,
            max_distance=20,
            visualize=False
        )
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

        if time.time() - self.last_map_update >= self.map_update_period:

            self.update_occupancy_grid()
            self.plot_occupancy_grid()

            self.last_map_update = time.time()

    def run(self):

        while True:

            self.PID()

            time.sleep(self.dt)

print("All goodie in the hoodie")
controller = PIDController()
controller.run()

