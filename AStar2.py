import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import time
import math
import heapq
from itertools import count

np.set_printoptions(threshold=np.inf)

# Random occupancy grid

width = 300
height = 300

p_obstacle = 0.03
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
Inflate = True

def inflate_obstacles(grid, inflation_radius=1):
    structure = np.ones(
        (2 * inflation_radius + 1,
         2 * inflation_radius + 1),
        dtype=bool
    )
    inflated = binary_dilation(grid, structure=structure)
    return inflated.astype(int)


# Plot the map


if Inflate:
    grid = inflate_obstacles(grid)

plt.figure(figsize=(8, 8))
plt.imshow(grid, cmap="binary", origin="upper")  # grid
plt.scatter(Start_x, Start_y, color="blue", s=10, label="Start")  # start
plt.scatter(Goal_x, Goal_y, color="red", s=10, label="Goal")       # end
plt.title("Occupancy Grid")  
plt.show()


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

# ----------------------------
# NODE HELPERS
# ----------------------------
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

# ============================================================
# STANDARD A*
# ============================================================

def run_astar():

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

    # ----------------------------
    # reconstruct path
    # ----------------------------
    path = []
    if (Goal_x, Goal_y) in nodes:
        node = nodes[(Goal_x, Goal_y)]
        while node:
            path.append((node.x, node.y))
            node = node.parent
        path.reverse()

    plt.figure(figsize=(8, 8))
    plt.imshow(grid, cmap="binary", origin="upper")

    if path:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        plt.plot(xs, ys, color="blue")

    plt.scatter(Start_x, Start_y, c="blue")
    plt.scatter(Goal_x, Goal_y, c="red")

    plt.title("A* Path")
    plt.show()


# ============================================================
# VISUAL A*
# ============================================================

def run_astar_visual():

    fig, ax = plt.subplots(figsize=(6, 6))

    open_heap = []
    closed_set = set()
    nodes = {}

    def get_node(x, y):
        if (x, y) not in nodes:
            nodes[(x, y)] = Node(x, y)
        return nodes[(x, y)]

    start.g = 0
    start.h = heuristic(start.x, start.y)
    start.f = start.g + start.h

    nodes[(start.x, start.y)] = start

    heapq.heappush(open_heap, (start.f, start.h, next(counter), start))

    img = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=float)

    def update_img(current=None):

        img[:, :] = [1, 1, 1]
        img[grid == 1] = [0, 0, 0]

        for x, y in closed_set:
            img[y, x] = [1, 0, 0]

        for _, _, _, n in open_heap:
            img[n.y, n.x] = [1, 1, 0]

        if current:
            img[current.y, current.x] = [0, 0, 1]

        img[start.y, start.x] = [0, 0, 1]
        img[goal.y, goal.x] = [0, 1, 0]

    while open_heap:

        _, _, _, current = heapq.heappop(open_heap)

        if (current.x, current.y) in closed_set:
            continue

        closed_set.add((current.x, current.y))

        if current.x == goal.x and current.y == goal.y:
            update_img(current)
            ax.imshow(img, origin="upper")
            plt.pause(0.01)
            break

        for dx, dy in directions:

            nx, ny = current.x + dx, current.y + dy

            if nx < 0 or nx >= grid.shape[1] or ny < 0 or ny >= grid.shape[0]:
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

        update_img(current)

        ax.clear()
        ax.imshow(img, origin="upper")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.pause(0.0001)

    plt.show()


# ----------------------------
# run
# ----------------------------
run_astar()
run_astar_visual()