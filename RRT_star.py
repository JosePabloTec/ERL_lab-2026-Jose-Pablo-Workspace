import numpy as np # Numerical matrix handling
import matplotlib.pyplot as plt # Plotting and animation
import random # Stochastic value generation
import math # Trigonometric operations (sin, cos, atan2)


# NODE CLASS: Data structure for the tree vertices

class Node:
    def __init__(self, x, y, g, parent=None):
        self.x = x # Horizontal coordinate (Column)
        self.y = y # Vertical coordinate (Row)
        self.g = g
        self.parent = parent # Pointer to the parent node
        self.children = []



# Neighborhood check

neighborhood_radius = 2


def check_for_obstacles(node1, node2, grid):
    x1 = node1.x
    y1 = node1.y
    x2 = node2.x
    y2 = node2.y

    v = np.array([x2 - x1, y2 - y1])

    norm = np.linalg.norm(v)

    if norm == 0:
        return False

    unit_v = v / norm
    step = 0.1
    safety_box_size = 0.6

    for i in range(int(round(norm / step))):

        new_point = unit_v * step * i + np.array([x1, y1])

        xmin = int(round(new_point[0] - safety_box_size / 2))
        xmax = int(round(new_point[0] + safety_box_size / 2))
        ymin = int(round(new_point[1] - safety_box_size / 2))
        ymax = int(round(new_point[1] + safety_box_size / 2))

        if not (
            0 <= xmin < grid.shape[1] and
            0 <= xmax < grid.shape[1] and
            0 <= ymin < grid.shape[0] and
            0 <= ymax < grid.shape[0]
        ):
            return True

        collisions = (
            grid[ymin, xmin] +
            grid[ymin, xmax] +
            grid[ymax, xmin] +
            grid[ymax, xmax]
        )

        if collisions != 0:
            return True

    return False


def euclidean_distance(x_node,y_node,x,y):
    return np.sqrt((x_node-x)**2+(y_node-y)**2)

def find_neighbors(tree, x, y, neighborhood_radius = neighborhood_radius):
    neighbors = []

    for node in tree:
        d = euclidean_distance(node.x, node.y, x, y)
        if d < neighborhood_radius:
            neighbors.append((node, d))

    return neighbors


def find_best_node(tree, current_node, grid, neighborhood_radius):
    neighbors = find_neighbors(
        tree,
        current_node.x,
        current_node.y,
        neighborhood_radius
    )

    for node, d in neighbors:
        tentative_g = node.g + d

        if tentative_g < current_node.g:
            if not check_for_obstacles(node, current_node, grid):
                current_node.g = tentative_g
                current_node.parent = node

    return current_node


def update_child_nodes(node):
    for child in node.children:
        child.g = node.g + euclidean_distance(
            node.x,
            node.y,
            child.x,
            child.y
        )
        update_child_nodes(child)


def update_neighborhood(neighborhood, new_node, grid):
    for node, d in neighborhood:
        tentative_g = new_node.g + d

        if tentative_g < node.g:
            if not check_for_obstacles(new_node, node, grid):

                old_parent = node.parent

                if old_parent is not None:
                    old_parent.children.remove(node)

                node.parent = new_node
                new_node.children.append(node)

                node.g = tentative_g

                update_child_nodes(node)







# =====================================================================
# MAIN RRT FUNCTION
# =====================================================================
def rrt(grid,
        start,
        goal,
        max_iter=5000,
        step_size=0.5,
        bias=0.05,
        neighborhood_radius=2.0):

    start_y, start_x = start
    goal_y, goal_x = goal

    tree = [Node(start_x, start_y, g=0)]

    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(grid, cmap='Greys', origin='upper')
    ax.scatter(start_x, start_y, color='blue', s=30, label='Start', zorder=5)
    ax.scatter(goal_x, goal_y, color='green', s=30, label='Goal', zorder=5)
    ax.set_title("RRT* Optimal Motion Planning")

    for i in range(max_iter):

        if random.random() < bias:
            x_rand, y_rand = goal_x, goal_y
        else:
            y_rand = random.uniform(0, grid.shape[0] - 1)
            x_rand = random.uniform(0, grid.shape[1] - 1)

        nearest_node = tree[0]
        min_dist = float("inf")

        for node in tree:
            dist = euclidean_distance(node.x, node.y, x_rand, y_rand)
            if dist < min_dist:
                min_dist = dist
                nearest_node = node

        theta = math.atan2(
            y_rand - nearest_node.y,
            x_rand - nearest_node.x
        )

        x_new = nearest_node.x + step_size * math.cos(theta)
        y_new = nearest_node.y + step_size * math.sin(theta)

        in_bounds = (
            0 <= y_new < grid.shape[0] and
            0 <= x_new < grid.shape[1]
        )

        if not in_bounds:
            continue

        tol = 0.15

        y_min = int(round(y_new - tol))
        y_max = int(round(y_new + tol))
        x_min = int(round(x_new - tol))
        x_max = int(round(x_new + tol))

        safe_bounds = (
            0 <= y_min and
            y_max < grid.shape[0] and
            0 <= x_min and
            x_max < grid.shape[1]
        )

        if not safe_bounds:
            continue

        collisions = (
            grid[y_min, x_min] +
            grid[y_min, x_max] +
            grid[y_max, x_min] +
            grid[y_max, x_max]
        )

        if collisions != 0:
            continue

        new_node = Node(
            x_new,
            y_new,
            g=nearest_node.g + euclidean_distance(
                nearest_node.x,
                nearest_node.y,
                x_new,
                y_new
            ),
            parent=nearest_node
        )

        neighborhood = find_neighbors(
            tree,
            x_new,
            y_new,
            neighborhood_radius
        )

        new_node = find_best_node(
            tree,
            new_node,
            grid,
            neighborhood_radius
        )

        new_node.parent.children.append(new_node)

        tree.append(new_node)

        update_neighborhood(
            neighborhood,
            new_node,
            grid
        )

        ax.plot(
            [new_node.parent.x, new_node.x],
            [new_node.parent.y, new_node.y],
            color="red"
        )

        if i % 3 == 0:
            plt.pause(0.001)

        dist_to_goal = euclidean_distance(
            x_new,
            y_new,
            goal_x,
            goal_y
        )

        if dist_to_goal <= step_size:

            final_node = Node(
                goal_x,
                goal_y,
                g=new_node.g + step_size,
                parent=new_node
            )

            new_node.children.append(final_node)
            tree.append(final_node)

            ax.plot(
                [new_node.x, goal_x],
                [new_node.y, goal_y],
                color="red"
            )

            path = []
            current = final_node

            while current is not None:
                path.append((current.y, current.x))
                current = current.parent

            path.reverse()

            x_path = [p[1] for p in path]
            y_path = [p[0] for p in path]

            ax.plot(
                x_path,
                y_path,
                color="blue",
                linewidth=1.5,
                label="Path"
            )

            plt.ioff()
            plt.show()

            return tree, path

    plt.ioff()
    plt.show()

    return tree, []


### Main Execution Block
if __name__ == "__main__":
    dim = 20 # Grid dimension
    grid_map = np.zeros((dim, dim)) # Free Space initialization
    start = (0, 0) # Start point (y, x)
    goal = (19, 19) # Goal point (y, x)
    num_obs = 50 # Obstacle density

    free_cells = [] 
    for y in range(dim): 
        for x in range(dim): 
            if (y, x) != start and (y, x) != goal:
                free_cells.append((y, x)) 
    
    # Obstacle sampling
    obs_random = random.sample(free_cells, num_obs)
    
    # Obstacle assignment in the grid
    for obs_y, obs_x in obs_random:
        grid_map[obs_y, obs_x] = 1

    print("Initializing path planning (RRT)...")
    result_tree, final_path = rrt(grid_map, start, goal)
    
    if len(final_path) > 0:
        print("Path Planning successful.")
    else: 
        print("Resolution failed: Configuration space obstructed or iteration limit exceeded.")