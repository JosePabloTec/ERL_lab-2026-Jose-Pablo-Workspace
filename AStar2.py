import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import time
import math
import heapq
from itertools import count
from pqdict import pqdict
from PIL import Image
import io

np.set_printoptions(threshold=np.inf)

# Random occupancy grid

width = 60
height = 60

p_obstacle = 0.04
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

# NODE HELPERS
nodes = {}


# this function helps A* access existing nodes or creates a new one
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

# STANDARD A* List - based (slow)
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
        if dx != 0 and dy != 0:
            step_cost = math.sqrt(2)
        else:
            step_cost = 1

        tentative_g = current.g + step_cost

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

plt.title("A* Path SLOW")
plt.show()

nodes = {}





# VISUAL A* List - based (slow)
def run_astar_visual_v1():
    fig, ax = plt.subplots(figsize=(6, 6))

    directions = [(1,0),(-1,0),(0,1),(0,-1),
    (1,1),(-1,-1),(1,-1),(-1,1)]

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
        img[start.y, start.x] = [0, 0, 1]
        img[goal.y, goal.x] = [0, 0, 1]

    current = None

    while open_list:

        # pick best node
        current = min(open_list, key=lambda n: (n.f, n.h))

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

            if dx != 0 and dy != 0:
                step_cost = math.sqrt(2)
            else:
                step_cost = 1
            tentative_g = current.g + step_cost

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
        plt.pause(0.0001)

    plt.show()





# STANDARD A* Priority Queue Dictionary - Based

def run_astar_v2():

    nodes.clear()

    open_set = pqdict()
    closed_set = set()

    start = Node(Start_x, Start_y)

    start.g = 0
    start.h = heuristic(Start_x, Start_y)
    start.f = start.g + start.h

    nodes[(Start_x, Start_y)] = start
    open_set[(Start_x, Start_y)] = (start.f, start.h)

    while open_set:

        (cx, cy), (cf, ch) = open_set.popitem()  # cx = current x, cy = current y, cf = current f, ch = current h
        current = nodes[(cx, cy)]

        if (cx, cy) == (Goal_x, Goal_y):   # check if goal was reached
            break

        closed_set.add((cx, cy))           # add to the closed set to prevent processing it again

        for dx, dy in directions:

            nx = cx + dx
            ny = cy + dy

            if nx < 0 or nx >= width or ny < 0 or ny >= height: # invalid, cause it's out of bounds
                continue

            if grid[ny, nx] == 1:         # invalid, cause there's an obstacle    
                continue

            if (nx, ny) in closed_set:    # invalid, cause it was already processed
                continue

            if dx == 1 and dy == 1:       # calculate cost to reach neighbor
                cost = DIAG_COST 
            else:
                cost = 1

            tentative_g = current.g + cost  # calculate a potentially better g cost

            neighbor = get_node(nx, ny)     

            if tentative_g < neighbor.g:   

                neighbor.parent = current
                neighbor.g = tentative_g
                neighbor.h = heuristic(nx, ny)
                neighbor.f = neighbor.g + neighbor.h

                open_set[(nx, ny)] = (neighbor.f, neighbor.h)

    # reconstruct path
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

    plt.title("A* Search")
    plt.show()


# VISUAL A* Priority List - Based

def run_astar_visual_v2():

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)

    open_heap = []
    closed_set = set()
    nodes = {}

    frames = []

    def get_node(x, y):
        if (x, y) not in nodes:
            nodes[(x, y)] = Node(x, y)
        return nodes[(x, y)]

    start.g = 0
    start.h = heuristic(start.x, start.y)
    start.f = start.g + start.h

    nodes[(start.x, start.y)] = start

    heapq.heappush(
        open_heap,
        (start.f, start.h, next(counter), start)
    )

    img = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=float)

    def update_img(current=None):

        img[:, :] = [1, 1, 1]

        # obstacles
        img[grid == 1] = [0, 0, 0]

        # explored nodes
        for x, y in closed_set:
            img[y, x] = [1, 0, 0]

        # frontier
        for _, _, _, n in open_heap:
            img[n.y, n.x] = [1, 1, 0]

        # current node
        if current:
            img[current.y, current.x] = [0, 0, 1]

        # start and goal
        img[start.y, start.x] = [0, 0, 1]
        img[goal.y, goal.x] = [0, 1, 0]


    def save_frame():

        ax.clear()
        ax.imshow(
            img,
            origin="upper",
            interpolation="nearest"
        )

        ax.set_xticks([])
        ax.set_yticks([])

        plt.tight_layout(pad=0)

        buffer = io.BytesIO()
        fig.savefig(
            buffer,
            format="png",
            bbox_inches="tight",
            pad_inches=0,
            dpi=200
        )

        buffer.seek(0)
        frame = Image.open(buffer).convert("RGB")

        frames.append(frame)


    while open_heap:

        _, _, _, current = heapq.heappop(open_heap)

        if (current.x, current.y) in closed_set:
            continue

        closed_set.add((current.x, current.y))


        if current.x == goal.x and current.y == goal.y:

            update_img(current)
            save_frame()

            break


        for dx, dy in directions:

            nx = current.x + dx
            ny = current.y + dy

            if nx < 0 or nx >= grid.shape[1]:
                continue

            if ny < 0 or ny >= grid.shape[0]:
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
                    (
                        neighbor.f,
                        neighbor.h,
                        next(counter),
                        neighbor
                    )
                )


        update_img(current)
        save_frame()


    # Save GIF
    save_path = r"C:\ERL\Figs\astar_visualization.gif"

    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=40,     # milliseconds per frame (~25 FPS)
        loop=0,
        optimize=False
    )

    print(f"GIF saved to: {save_path}")

    plt.close(fig)


# ----------------------------
# run
# ----------------------------
run_astar_visual_v1()
run_astar_v2()
run_astar_visual_v2()