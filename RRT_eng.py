import numpy as np # Numerical matrix handling
import matplotlib.pyplot as plt # Plotting and animation
import random # Stochastic value generation
import math # Trigonometric operations (sin, cos, atan2)
from matplotlib.animation import PillowWriter



# Node class: Data structure 

class Node:
    def __init__(self, x, y, parent=None):
        self.x = x # x coord
        self.y = y # y coord
        self.parent = parent # Pointer to the parent node


# MAIN RRT FUNCTION

def rrt(grid,
        start,
        goal,
        max_iter=5000,
        step_size=0.5,
        goal_bias=0.05,
        gif_path=r"C:\ERL\Figs\rrt_animation.gif"):

    start_y, start_x = start
    goal_y, goal_x = goal

    tree = [Node(start_x, start_y)]

    # Figure 
    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)

    ax.imshow(grid,
              cmap="Greys",
              origin="upper",
              interpolation="nearest")

    ax.scatter(start_x, start_y,   # goal
               color="blue",
               s=60,
               label="Start",
               zorder=5)

    ax.scatter(goal_x, goal_y,   #end
               color="green",
               s=60,
               label="Goal",
               zorder=5)

    ax.set_title(f"Rapidly-Exploring Random Tree (Goal Bias = {goal_bias:.2f})")

    ax.set_xlim(-0.5, grid.shape[1]-0.5)
    ax.set_ylim(grid.shape[0]-0.5, -0.5)
    ax.set_aspect("equal")

    #Iteration counter 
    iter_text = ax.text(
        0.55,
        0.98,
        "Iteration: 0",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(facecolor="white",
                  edgecolor="black",
                  alpha=0.85)
    )

    writer = PillowWriter(fps=15)

    with writer.saving(fig, gif_path, dpi=180):

        writer.grab_frame()

        for i in range(max_iter):

            iter_text.set_text(f"Iteration: {i}")

            # Goal-biased sampling 
            if random.random() < goal_bias:
                x_rand, y_rand = goal_x, goal_y
            else:
                y_rand = random.uniform(0, grid.shape[0]-1)
                x_rand = random.uniform(0, grid.shape[1]-1)

            #  Nearest neighbor 
            nearest_node = min(tree, key=lambda n: (n.x-x_rand)**2 + (n.y-y_rand)**2)  # Search through the list

            theta = math.atan2(
                y_rand-nearest_node.y,
                x_rand-nearest_node.x
            )

            x_new = nearest_node.x + step_size*math.cos(theta)
            y_new = nearest_node.y + step_size*math.sin(theta)

            # Bounds check
            if not (0 <= x_new < grid.shape[1] and
                    0 <= y_new < grid.shape[0]):
                continue

            tol = 0.15

            y_min = int(round(y_new-tol))  
            y_max = int(round(y_new+tol))
            x_min = int(round(x_new-tol))
            x_max = int(round(x_new+tol))

            if not (0 <= y_min and                    # make sure the point exists within the grid
                    y_max < grid.shape[0] and
                    0 <= x_min and
                    x_max < grid.shape[1]):
                continue

            collisions = (
                grid[y_min, x_min] +
                grid[y_min, x_max] +
                grid[y_max, x_min] +
                grid[y_max, x_max]
            )

            if collisions != 0:         #only 0 when the bound box is safe
                continue

            new_node = Node(x_new, y_new, parent=nearest_node)
            tree.append(new_node)

            ax.plot(
                [nearest_node.x, x_new],
                [nearest_node.y, y_new],
                color="deeppink",
                linewidth=0.8
            )

            # Save a frame every 5 iterations
            if i % 5 == 0:
                writer.grab_frame()

            # Goal reached 
            if math.hypot(x_new-goal_x, y_new-goal_y) <= step_size:

                final_node = Node(goal_x, goal_y, parent=new_node)
                tree.append(final_node)

                ax.plot(
                    [x_new, goal_x],
                    [y_new, goal_y],
                    color="red",
                    linewidth=2
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
                    color="red",
                    linewidth=3,
                    label="Path"
                )

                iter_text.set_text(
                    f"Goal reached in {i} iterations"
                )

                # Hold the final frame
                for _ in range(30):
                    writer.grab_frame()

                plt.close(fig)

                print(f"GIF saved to:\n{gif_path}")

                return tree, path

        iter_text.set_text(
            f"No path found after {max_iter} iterations"
        )

        for _ in range(30):
            writer.grab_frame()

    plt.close(fig)

    print(f"GIF saved to:\n{gif_path}")

    return tree, []

# Main 
if __name__ == "__main__":
    dim = 40 # Grid dimension
    grid_map = np.zeros((dim, dim)) # Free Space initialization
    start = (0, 0) # Start point (y, x)
    goal = (39, 39) # Goal point (y, x)
    num_obs = 50 # Number of obstacles

    free_cells = [] 
    for y in range(dim): 
        for x in range(dim): 
            if (y, x) != start and (y, x) != goal:
                free_cells.append((y, x)) 
    
    # Obstacle sampling, random
    obs_random = random.sample(free_cells, num_obs)
    
    # Obstacle assignment in the grid
    for obs_y, obs_x in obs_random:
        grid_map[obs_y, obs_x] = 1

    print("Initializing path planning (RRT)...")
    result_tree, final_path = rrt(grid_map,start,goal,max_iter=5000,step_size=0.5,goal_bias=0.05,gif_path=r"C:\ERL\Figs\rrt_animation_0.05.gif")
    result_tree, final_path = rrt(grid_map,start,goal,max_iter=5000,step_size=0.5,goal_bias=0.1,gif_path=r"C:\ERL\Figs\rrt_animation_0.1.gif")
    
    if len(final_path) > 0:
        print("Path Planning successful.")
    else: 
        print("Resolution failed: Configuration space obstructed or iteration limit exceeded.")