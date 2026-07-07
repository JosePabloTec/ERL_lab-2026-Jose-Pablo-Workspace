import os
import pybullet as p
import pybullet_data
import time
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import heapq
from itertools import count

width = 30 
height = 30
cell_size = 1
p_obstacle = 0.04
p_free = 1 - p_obstacle
grid = np.random.choice([0, 1], size=(height, width), p=[p_free, p_obstacle])

Start_x = 0    #initial cell x coordinate
Start_y = 0    #initial cell y coordinate
Goal_x = width - 1
Goal_y = height - 1
grid[Start_y, Start_x] = 0
grid[Goal_y, Goal_x] = 0

def calculate_origin():
    x_0 = -width/2 * cell_size + cell_size/2 + Start_x*cell_size
    y_0 = height/2 * cell_size - cell_size/2 - Start_y*cell_size
    return x_0,y_0

def calculate_cell_center(x,y):
    x = -width/2 * cell_size + cell_size/2 + x*cell_size
    y = height/2 * cell_size - cell_size/2 - y*cell_size
    return x,y 

# Inflate obstacles
Inflate = True

def inflate_obstacles(grid, inflation_radius=1):
    structure = np.ones(
        (2 * inflation_radius + 1,
         2 * inflation_radius + 1),
        dtype=bool
    )
    inflated = binary_dilation(grid, structure=structure)
    return inflated.astype(int)

plt.figure(figsize=(8, 8))
plt.imshow(grid, cmap="binary", origin="upper")  # grid
plt.scatter(Start_x, Start_y, color="blue", s=10, label="Start")  # start
plt.scatter(Goal_x, Goal_y, color="red", s=10, label="Goal")       # end
plt.title("Occupancy Grid")  
plt.show()


def point_to_cell_coordinate(cell_size, x, y):

    x_cell = int(x // cell_size)
    y_cell = int(y // cell_size)

    return x_cell, y_cell



p.connect(p.GUI)
p.resetSimulation()
p.setGravity(0, 0, -10)
p.setRealTimeSimulation(0)
p.resetDebugVisualizerCamera(
    cameraDistance=25,
    cameraYaw=0,
    cameraPitch=-85,
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


for row in range(grid.shape[0]):
    for col in range(grid.shape[1]):
        if grid[row, col] == 1:
            x, y = calculate_cell_center(col, row)

            create_obstacle(
                x=x,
                y=y,
                width=cell_size,
                length=cell_size,
                height=cell_size
            )


if Inflate:
    grid = inflate_obstacles(grid)


##############################################################
# A* 

# Heuristic Function (Octile Distance)
def heuristic(x, y, goal_x=Goal_x, goal_y=Goal_y):
    dx = abs(x - goal_x)
    dy = abs(y - goal_y)
    return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)


# Node Class
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.g = float("inf")
        self.h = 0
        self.f = float("inf")

        self.parent = None

counter = count()

# NODE HELPERS
nodes = {}

def get_node(x, y):
    if (x, y) not in nodes:
        nodes[(x, y)] = Node(x, y)
    return nodes[(x, y)]

DIAG_COST = math.sqrt(2)

directions = [
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (1, 1), (1, -1), (-1, 1), (-1, -1)
]

# Initialize Start Node
start = Node(Start_x, Start_y)
goal = Node(Goal_x, Goal_y)

start.g = 0
start.h = heuristic(start.x , start.y)
start.f = start.g + start.h

# Open and closed sets
open_list = [start]
open_set = set()
closed_set = set()

def run_astar_v2():

    open_heap = []
    closed_set = set()

    start = Node(Start_x, Start_y)
    goal = Node(Goal_x, Goal_y)

    start.g = 0
    start.h = heuristic(start.x, start.y)
    start.f = start.g + start.h

    nodes[(start.x, start.y)] = start

    heapq.heappush(open_heap, (start.f, start.h, next(counter), start))

    while open_heap:

        _, _, _, current = heapq.heappop(open_heap)

        if (current.x, current.y) in closed_set:
            continue

        if current.x == Goal_x and current.y == Goal_y:
            print("Goal reached!")
            break

        closed_set.add((current.x, current.y))

        for dx, dy in directions:

            nx, ny = current.x + dx, current.y + dy

            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue

            if grid[ny, nx] == 1:
                continue

            if (nx, ny) in closed_set:
                continue

            step_cost = DIAG_COST if dx != 0 and dy != 0 else 1
            tentative_g = current.g + step_cost

            neighbor = get_node(nx, ny)

            if tentative_g < neighbor.g:

                neighbor.parent = current
                neighbor.g = tentative_g
                neighbor.h = heuristic(nx, ny)
                neighbor.f = neighbor.g + neighbor.h

                heapq.heappush(
                    open_heap,
                    (neighbor.f, neighbor.h, next(counter), neighbor)
                )

    # reconstruct path
    path = []
    if (Goal_x, Goal_y) in nodes:
        node = nodes[(Goal_x, Goal_y)]
        while node:
            path.append((node.x, node.y))
            node = node.parent
        path.reverse()

    # Convert grid cells to world coordinates
    goal_list = []

    for grid_x, grid_y in path:
        world_x, world_y = calculate_cell_center(grid_x, grid_y)
        goal_list.append([world_x, world_y])

    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap="binary", origin="upper")

    if path:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        plt.plot(xs, ys, color="blue")

    plt.scatter(Start_x, Start_y, c="blue")
    plt.scatter(Goal_x, Goal_y, c="red")

    plt.title("A* Path FAST")
    plt.show()

    return goal_list

##############################################################









class PIDController:

    def __init__(self):
        x_initial,y_initial = calculate_origin()
        self.car = p.loadURDF(
            os.path.join(
                pybullet_data.getDataPath(),
                "racecar/racecar.urdf"
            ),
            basePosition=[x_initial, y_initial, 0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
        )

        self.tol = 5e-3

        # -----------------------
        # Goal list
        # -----------------------
        self.goal_list = run_astar_v2()
        self.goal_index = 0

        self.x = self.goal_list[self.goal_index][0]
        self.y = self.goal_list[self.goal_index][1]

        self.target_location = np.array([self.x, self.y])

        # PID gains
        # P
        self.k1 = 20
        self.k2 = 25
        # D
        self.k11 = 0.22
        self.k22 = 0.2
        #I
        self.k111 = 0.08
        self.k222 = 0.03

        self.previous_d = 0
        self.previous_dtheta = 0

        self.integral_d = 0
        self.integral_theta = 0

        self.dt = 0.001

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

            self.goal_index += 1

            if self.goal_index >= len(self.goal_list):

                print("All goals completed!")

                p.resetBaseVelocity(
                    self.car,
                    linearVelocity=[0,0,0],
                    angularVelocity=[0,0,0]
                )

                return


            self.x = self.goal_list[self.goal_index][0]
            self.y = self.goal_list[self.goal_index][1]

            self.target_location = np.array(
                [self.x, self.y]
            )

            self.state = "TURN"

            self.integral_d = 0
            self.integral_theta = 0

            self.previous_d = 0
            self.previous_dtheta = 0

            return



        print(f"State: {self.state}")
        print(f"Current Goal: {self.goal_list[self.goal_index]}")

        print("Position:")
        print(
            f"x: {pos[0]:.2f}, "
            f"y: {pos[1]:.2f}, "
            f"z: {pos[2]:.2f}"
        )

        print("Orientation:")

        euler = p.getEulerFromQuaternion(orien)

        print(
            f"roll: {euler[0]:.2f}, "
            f"pitch: {euler[1]:.2f}, "
            f"yaw: {euler[2]:.2f}"
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

            start = time.time()

            p.stepSimulation()

            self.PID()

            elapsed = time.time() - start

            time.sleep(
                max(0.0, self.dt - elapsed)
            )



print("All goodie in the hoodie")
controller = PIDController()
controller.run()
