import numpy as np # Numerical matrix handling
import matplotlib.pyplot as plt # Plotting and animation
import random # Stochastic value generation
import math # Trigonometric operations (sin, cos, atan2)

# =====================================================================
# NODE CLASS: Data structure for the tree vertices
# =====================================================================
class Node:
    def __init__(self, x, y, parent=None):
        self.x = x # Horizontal coordinate (Column)
        self.y = y # Vertical coordinate (Row)
        self.parent = parent # Pointer to the parent node


# =====================================================================
# MAIN RRT FUNCTION
# =====================================================================
def rrt(grid, start, goal, max_iter=5000, step_size=0.5):
    
    # Unpack tuples (row=y, col=x) to match the Cartesian plane
    start_y, start_x = start
    goal_y, goal_x = goal
    
    # Initialize the tree with the root node
    tree = [Node(start_x, start_y)]
    
    ### Visualization setup (Animation)
    plt.ion() 
    fig, ax = plt.subplots(figsize=(7, 7)) 
    ax.imshow(grid, cmap='Greys', origin='upper') 
    ax.scatter(start_x, start_y, color='blue', s=100, label='Start', zorder=5) 
    ax.scatter(goal_x, goal_y, color='green', s=100, label='Goal', zorder=5) 
    ax.set_title("RRT Path Planning") 
    
    ### Main exploration loop
    for i in range(max_iter):
        
        # p = Stochastic variable for Goal Bias (0.0 to 1.0)
        p = random.random() 
        
        # 10% probability: Direct sampling towards the goal
        if p < 0.1:
            x_rand, y_rand = goal_x, goal_y 
        # 90% probability: Uniform sampling in the configuration space
        else: 
            y_rand = random.uniform(0, grid.shape[0] - 1) 
            x_rand = random.uniform(0, grid.shape[1] - 1) 
            
        # Nearest Neighbor search
        nearest_node = tree[0] 
        min_dist = float('inf') 
        
        for node in tree: 
            # Euclidean distance
            dist = math.sqrt((node.x - x_rand)**2 + (node.y - y_rand)**2)
            
            if dist < min_dist:
                min_dist = dist 
                nearest_node = node 

        # Kinematics calculation: Direction and steering       
        theta = math.atan2(y_rand - nearest_node.y, x_rand - nearest_node.x)
        x_new = nearest_node.x + step_size * math.cos(theta)
        y_new = nearest_node.y + step_size * math.sin(theta)
        
        # Boundary check
        in_bounds = (0 <= y_new < grid.shape[0]) and (0 <= x_new < grid.shape[1])
        
        if in_bounds: 
            # tol = Physical tolerance of the robot (Bounding Box)
            tol = 0.15 
            
            # Spatial discretization of the robot's 4 corners
            y_min = int(round(y_new - tol))
            y_max = int(round(y_new + tol))
            x_min = int(round(x_new - tol))
            x_max = int(round(x_new + tol))
            
            # Verification that the collision box does not exceed the matrix limits
            safe_bounds = (0 <= y_min) and (y_max < grid.shape[0]) and \
                          (0 <= x_min) and (x_max < grid.shape[1])
            
            if safe_bounds: 
                # Collision detection: Sum of the occupancy grid corners
                collisions = (
                    grid[y_min, x_min] + 
                    grid[y_min, x_max] + 
                    grid[y_max, x_min] + 
                    grid[y_max, x_max]
                )
                
                # Zero collisions = Free Space
                if collisions == 0: 
                    # Instantiation and saving of the new vertex
                    new_node = Node(x_new, y_new, parent=nearest_node)
                    tree.append(new_node) 
                    
                    ### Edge rendering
                    ax.plot([nearest_node.x, x_new], 
                            [nearest_node.y, y_new], 
                            color='pink', alpha=0.5)
                    
                    # GUI refresh rate (every 5 iterations to prevent freezing)
                    if i % 5 == 0: 
                        plt.pause(0.001)
                    
                    # Goal Check
                    dist_to_goal = math.sqrt((x_new - goal_x)**2 + (y_new - goal_y)**2)
                    
                    if dist_to_goal <= step_size: 
                        # Insertion of the terminal node
                        final_node = Node(goal_x, goal_y, parent=new_node)
                        tree.append(final_node) 
                        
                        ax.plot([x_new, goal_x], 
                                [y_new, goal_y], 
                                color='pink', alpha=0.5)
                        
                        # Path Backtracking
                        path = [] 
                        current_node = final_node 
                        
                        while current_node is not None: 
                            # Revert to matrix format (row, col) for external consistency
                            path.append((current_node.y, current_node.x)) 
                            current_node = current_node.parent 
                            
                        path.reverse() 
                        
                        # Rendering the resulting Path
                        x_path = [coord[1] for coord in path]
                        y_path = [coord[0] for coord in path]
                        ax.plot(x_path, y_path, color='red', linewidth=3, label='Path')
                        ax.legend() 
                        
                        plt.ioff()
                        plt.show()
                        
                        return tree, path
    
    # Time-out closure if no path is found
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