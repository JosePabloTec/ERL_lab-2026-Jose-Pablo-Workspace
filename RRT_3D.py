import numpy as np
import matplotlib.pyplot as plt
import random 
import math

# Cage dimensions 
cage_x = 10
cage_y = 10
cage_z = 10

# Obstacle probability
p = 0.02

# Seed for reproducibility
np.random.seed()

grid = np.random.rand(cage_x, cage_y, cage_z) < p

def euclidean_distance(x2,y2,z2,x1,y1,z1):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)


def check_grid(x, y, z):
    x = math.floor(x)
    y = math.floor(y)
    z = math.floor(z)
    return grid[x, y, z]


def select_start():
    while True:
        x = random.random() * cage_x
        y = random.random() * cage_y
        z = random.random() * cage_z

        if not check_grid(x, y, z):
            return x, y, z
        

def select_end(x_start , y_start , z_start, min_dist = 7):
    while True:
        x = random.random() * cage_x
        y = random.random() * cage_y
        z = random.random() * cage_z

        if not check_grid(x, y, z):
            if euclidean_distance(x,y,z,x_start,y_start,z_start) > min_dist:
                return x, y, z




##### Algorithm ######
class Node:
    def __init__(self, x, y, z, g, parent=None):
        self.x = x 
        self.y = y 
        self.z = z
        self.g = g
        self.parent = parent # Pointer to the parent node
        self.children = []

neighborhood_radius = 1.5


def find_neighbors(tree, x, y, z, neighborhood_radius):
    neighbors = []

    for node in tree:
        d = euclidean_distance(
            node.x,
            node.y,
            node.z,
            x,
            y,
            z
        )

        if d < neighborhood_radius:
            neighbors.append((node, d))

    return neighbors


def find_best_node(tree, current_node, grid, neighborhood_radius):
    neighbors = find_neighbors(
        tree,
        current_node.x,
        current_node.y,
        current_node.z,
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
            node.z,
            child.x,
            child.y,
            child.z
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



def check_for_obstacles(node1, node2, grid):
    x1 = node1.x
    y1 = node1.y
    z1 = node1.z

    x2 = node2.x
    y2 = node2.y
    z2 = node2.z

    # Direction vector in 3D
    v = np.array([x2 - x1, y2 - y1, z2 - z1])

    norm = np.linalg.norm(v)

    if norm == 0:
        return False

    unit_v = v / norm

    step = 0.03
    safety_box_size = 0.6

    for i in range(int(round(norm / step))):

        # Current point along the path
        new_point = unit_v * step * i + np.array([x1, y1, z1])

        # Safety cube limits
        xmin = int(round(new_point[0] - safety_box_size / 2))
        xmax = int(round(new_point[0] + safety_box_size / 2))

        ymin = int(round(new_point[1] - safety_box_size / 2))
        ymax = int(round(new_point[1] + safety_box_size / 2))

        zmin = int(round(new_point[2] - safety_box_size / 2))
        zmax = int(round(new_point[2] + safety_box_size / 2))

        # Check boundaries
        if not (
            0 <= xmin < grid.shape[0] and
            0 <= xmax < grid.shape[0] and
            0 <= ymin < grid.shape[1] and
            0 <= ymax < grid.shape[1] and
            0 <= zmin < grid.shape[2] and
            0 <= zmax < grid.shape[2]
        ):
            return True

        # Check all cells inside the safety cube
        for x in range(xmin, xmax + 1):
            for y in range(ymin, ymax + 1):
                for z in range(zmin, zmax + 1):
                    if grid[x, y, z]:
                        return True

    return False


def rrt(grid,
        start,
        goal,
        max_iter=2000,
        step_size=0.5,
        bias=0.1,
        neighborhood_radius=2.0):

    start_x, start_y, start_z = start
    goal_x, goal_y, goal_z = goal

    tree = [Node(start_x, start_y, start_z, g=0)]

    # 3D plot
    plt.ion()
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.voxels(grid, edgecolor='k', alpha=0.1)

    ax.scatter(
        start_x,
        start_y,
        start_z,
        color='blue',
        s=20,
        label="Start"
    )

    ax.scatter(
        goal_x,
        goal_y,
        goal_z,
        color='red',
        s=20,
        label="Goal"
    )

    ax.set_title("3D RRT* Motion Planning")


    for i in range(max_iter):

        # Random sampling
        if random.random() < bias:
            x_rand, y_rand, z_rand = goal

        else:
            x_rand = random.uniform(0, grid.shape[0]-1)
            y_rand = random.uniform(0, grid.shape[1]-1)
            z_rand = random.uniform(0, grid.shape[2]-1)


        # Find nearest node
        nearest_node = tree[0]
        min_dist = float("inf")

        for node in tree:

            dist = euclidean_distance(
                node.x,
                node.y,
                node.z,
                x_rand,
                y_rand,
                z_rand
            )

            if dist < min_dist:
                min_dist = dist
                nearest_node = node


        # Steer toward random point
        direction = np.array([
            x_rand - nearest_node.x,
            y_rand - nearest_node.y,
            z_rand - nearest_node.z
        ])

        distance = np.linalg.norm(direction)

        if distance == 0:
            continue

        direction = direction / distance


        x_new = nearest_node.x + step_size * direction[0]
        y_new = nearest_node.y + step_size * direction[1]
        z_new = nearest_node.z + step_size * direction[2]


        # Bounds check
        in_bounds = (
            0 <= x_new < grid.shape[0] and
            0 <= y_new < grid.shape[1] and
            0 <= z_new < grid.shape[2]
        )

        if not in_bounds:
            continue


        # Collision check
        temp_node = Node(x_new, y_new, z_new, g = np.inf)

        if check_for_obstacles(
            nearest_node,
            temp_node,
            grid
        ):
            continue


        # Create new node
        new_node = Node(
            x_new,
            y_new,
            z_new,
            g=nearest_node.g + euclidean_distance(
                nearest_node.x,
                nearest_node.y,
                nearest_node.z,
                x_new,
                y_new,
                z_new
            ),
            parent=nearest_node
        )


        # RRT* optimization
        neighborhood = find_neighbors(
            tree,
            x_new,
            y_new,
            z_new,
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


        # Plot edge
        ax.plot3D(
            [new_node.parent.x, new_node.x],
            [new_node.parent.y, new_node.y],
            [new_node.parent.z, new_node.z],
            color="red",
            alpha=0.2
        )


        if i % 10 == 0:
            plt.pause(0.001)


        # Goal check
        dist_to_goal = euclidean_distance(
            x_new,
            y_new,
            z_new,
            goal_x,
            goal_y,
            goal_z
        )


        if dist_to_goal <= step_size:

            final_node = Node(
                goal_x,
                goal_y,
                goal_z,
                g=new_node.g + dist_to_goal,
                parent=new_node
            )

            new_node.children.append(final_node)
            tree.append(final_node)


            ax.plot3D(
                [new_node.x, goal_x],
                [new_node.y, goal_y],
                [new_node.z, goal_z],
                color="red",
                alpha = 0.2
            )


            # Extract path
            path = []
            current = final_node

            while current is not None:
                path.append(
                    (
                        current.x,
                        current.y,
                        current.z
                    )
                )
                current = current.parent


            path.reverse()



            # Extract coordinates from path
            x_path = [p[0] for p in path]
            y_path = [p[1] for p in path]
            z_path = [p[2] for p in path]

            # Plot final path in blue
            ax.plot3D(
                x_path,
                y_path,
                z_path,
                color="blue",
                linewidth=2,
                label="Final Path"
            )

            plt.ioff()
            plt.show()

            return path


    plt.ioff()
    plt.show()

    return None



# Run
x,y,z = select_start()
start_tuple = (x,y,z)
x2,y2,z2 = select_end(x,y,z)
goal_tuple = (x2,y2,z2)


path = rrt(grid,start_tuple,goal_tuple)
