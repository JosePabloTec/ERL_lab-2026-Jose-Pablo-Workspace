import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import time
import math

np.set_printoptions(threshold=np.inf)

# Random occupancy grid

width = 20
height = 20

p_obstacle = 0.2
p_free = 1 - p_obstacle

# 0 = free
# 1 = obstacle
grid = np.random.choice([0, 1], size=(height, width), p=[p_free, p_obstacle])

# Start and goal are free
Start_x = 0
Start_y = 0
Goal_x = width - 1
Goal_y = height - 1

grid[Start_y, Start_x] = 0
grid[Goal_y, Goal_x] = 0


# Inflate obstacles
Inflate = False

def inflate_obstacles(grid, inflation_radius=1):
    structure = np.ones(
        (2 * inflation_radius + 1,
         2 * inflation_radius + 1),
        dtype=bool
    )
    inflated = binary_dilation(grid, structure=structure)
    return inflated.astype(int)


# Plot the map
plt.figure(figsize=(8, 8))
plt.imshow(grid, cmap="binary", origin="upper")  # grid
plt.scatter(Start_x, Start_y, color="blue", s=10, label="Start")  # start
plt.scatter(Goal_x, Goal_y, color="red", s=10, label="Goal")       # end
plt.title("Occupancy Grid")  
plt.show()


# A* 

# Heuristic Function (Manhattan Distance)
def heuristic(x, y, goal_x = Goal_x, goal_y = Goal_y):
    return abs(x - goal_x) + abs(y - goal_y)


# Node Class
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.g = float("inf")
        self.h = 0
        self.f = float("inf")

        self.parent = None

# Initialize Start Node
start = Node(Start_x, Start_y)
goal = Node(Goal_x, Goal_y)

start.g = 0
start.h = heuristic(start.x , start.y)
start.f = start.g + start.h

# Open and closed sets
open_list = [start]
closed_set = set()

# Available directions
directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

# nodes dictionary
nodes = {}  

# Function to acces old nodes or create new ones
def get_node(x, y):
    if (x, y) not in nodes:
        nodes[(x, y)] = Node(x, y)
    return nodes[(x, y)]

# Insert the start node
nodes[(start.x, start.y)] = start

while open_list:

    # 1. pick node with lowest f
    current = min(open_list, key=lambda n: (n.f, n.h))

    # 2. goal check
    if current.x == Goal_x and current.y == Goal_y:
        print("Goal reached!")
        break

    # 3. move current from open → closed
    open_list.remove(current)
    closed_set.add((current.x, current.y))

    # 4. explore neighbors
    for dx, dy in directions:

        nx = current.x + dx
        ny = current.y + dy

        # bounds
        if nx < 0 or nx >= width or ny < 0 or ny >= height:
            continue

        # obstacle
        if grid[ny, nx] == 1:
            continue

        # already processed
        if (nx, ny) in closed_set:
            continue

        # cost from start
        tentative_g = current.g + 1

        # get or create node
        neighbor = get_node(nx, ny)

        # only update if better path found
        if tentative_g < neighbor.g:

            neighbor.parent = current
            neighbor.g = tentative_g
            neighbor.h = heuristic(nx, ny)
            neighbor.f = neighbor.g + neighbor.h

            if neighbor not in open_list:
                open_list.append(neighbor)


path = []

if (Goal_x, Goal_y) in nodes:
    current = nodes[(Goal_x, Goal_y)]

    while current is not None:
        path.append((current.x, current.y))
        current = current.parent

    path.reverse()


plt.figure(figsize=(8, 8))
plt.imshow(grid, cmap="binary", origin="upper")

if path:
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    plt.plot(xs, ys, color="blue", linewidth=2)

plt.scatter(Start_x, Start_y, color="blue", s=50)
plt.scatter(Goal_x, Goal_y, color="red", s=50)

plt.title("A* Path")
plt.show()

nodes = {}

def run_astar_visual():
    fig, ax = plt.subplots(figsize=(6, 6))

    directions = [(1,0), (-1,0), (0,1), (0,-1)]

    open_list = [start]
    closed_set = set()

    start.g = 0
    start.h = heuristic(start.x, start.y)
    start.f = start.g + start.h

    # color map image (RGB)
    img = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=float)

    def update_img():
        # reset
        img[:, :] = [1, 1, 1]  # white background

        # obstacles
        img[grid == 1] = [0, 0, 0]  # black

        # closed set (red)
        for x, y in closed_set:
            img[y, x] = [1, 0, 0]

        # open set (yellow)
        for n in open_list:
            img[n.y, n.x] = [1, 1, 0]

        # current node (blue)
        img[current.y, current.x] = [0, 0, 1]

        # start / goal override colors
        img[start.y, start.x] = [0, 1, 0]
        img[goal.y, goal.x] = [0.5, 0, 0.5]

    current = None

    while open_list:

        # pick best node
        current = min(open_list, key=lambda n: n.f)

        open_list.remove(current)
        closed_set.add((current.x, current.y))

        # goal check
        if current.x == goal.x and current.y == goal.y:
            update_img()
            ax.imshow(img, origin="upper")
            plt.pause(0.01)
            break

        # expand neighbors
        for dx, dy in directions:
            nx, ny = current.x + dx, current.y + dy

            if nx < 0 or nx >= grid.shape[1] or ny < 0 or ny >= grid.shape[0]:
                continue

            if grid[ny, nx] == 1:
                continue

            if (nx, ny) in closed_set:
                continue

            neighbor = get_node(nx, ny)

            tentative_g = current.g + 1

            if tentative_g < neighbor.g:
                neighbor.parent = current
                neighbor.g = tentative_g
                neighbor.h = heuristic(nx, ny)
                neighbor.f = neighbor.g + neighbor.h

                if neighbor not in open_list:
                    open_list.append(neighbor)

        # redraw frame
        update_img()
        ax.clear()
        ax.imshow(img, origin="upper")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.pause(0.01)

    plt.show()


run_astar_visual()