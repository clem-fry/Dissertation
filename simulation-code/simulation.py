#%% boundary condition
import numpy as np
import matplotlib.pyplot as plt
import shapes # grid_check, circle_outline_check, heart_outline_check
import models
import random

#threshold = 1.5 # what counts as a neighbour?
dt = .001 # time step

class Node:
    threshold = 1.5
    def __init__(self, id, x, y, u, v):
        self.id = id
        self.x = x
        self.y = y

        self.u = u
        self.v = v

        self.neighbours = []

    def update_u(self, laplacian): # NOT USED
        dudt = f(self.u, self.v) + laplacian[int(self.id)]
        self.u = dudt * dt + self.u
    
    def update_v(self, laplacian): # NOT USED
        dvdt = g(self.u, self.v) + laplacian[int(self.id)]
        self.v  = dvdt * dt + self.v
        
    def update_coordinates(self, new_x, new_y): # NOT USED
        self.x = new_x
        self.y = new_y

    def distance_to(self, other_node):
        return np.sqrt((self.x - other_node.x) ** 2 + (self.y - other_node.y) ** 2)
    
    def add_neighbour(self, neighbor_node):
        if neighbor_node not in self.neighbours:
            self.neighbours.append(neighbor_node)

    def update_all_neighbours(self, all_nodes):
        for node in all_nodes:

            if (self.distance_to(node) < Node.threshold) and (node.id != self.id): ## STUPID ADDITION and (node.x >= )
                self.add_neighbour(node)

    def __repr__(self):
        return f"Node(id={self.id}, coordinates=({self.x}, {self.y}, u={self.u}, v={self.v}, neighbours={len(self.neighbours)})"
#%% computation

def compute_adjacency_matrix(nodes, threshold=None):
    if threshold != None: # for calling 
        Node.threshold = threshold

    n = len(nodes)
    adjacency_matrix = np.zeros((n, n))

    for i, node in enumerate(nodes): # for each node in length
        node.update_all_neighbours(nodes)
        for j, neighbour in enumerate(nodes): # for each node in width
            if neighbour in node.neighbours:
                adjacency_matrix[i, j] = 1  # There is an edge/connection between these two nodes

    return adjacency_matrix

def compute_degree_matrix(adjacency_matrix):
    degree_vector = adjacency_matrix.sum(axis = 0)
    degree_matrix = np.diag(degree_vector)
    return degree_matrix

def laplacian(degree_matrix, adjacency_matrix, c):
    L = adjacency_matrix - degree_matrix
    Lc = np.dot(L, c)
    return(Lc)

#%%
# partial differential equations on the domain E=[−1,1]^2

# du/dt = Du d^2u/dx^2 + f(u, v)
# dv/dt = Dv d^2v/dx^2 + g(u, v)

# define the linear term:

Du = 0.1; Dv = 50

a, b, c= 1, 0.5, 1
f, g = models.geirer_meinhardt(a, b, c)

#%%
grid_size = 20 # size of the 2D grid or square root of total number of robots
T = 10.0 # total time
n = int(T/dt) # number of iterations

# Create evenly spaced coordinates for x and y across the grid
# x_coords = np.linspace(-1, 1, grid_size) # threshold 1.5 gives circles
# y_coords = np.linspace(-1, 1, grid_size)

def plot_nodes_dots(valid_nodes, ax, i, save=False):
    x_coords = [node.x if node is not None  else None for node in valid_nodes]
    y_coords = [node.y if node is not None  else None for node in valid_nodes]
    u_values = [node.u if node is not None  else None for node in valid_nodes]

    if save:
        s = 600
    else:
        s = 100

    scatter = ax.scatter(x_coords, y_coords, c=u_values, cmap='viridis', s=s, edgecolors='k')
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_title(f't={i * dt:.2f}')
    #ax.invert_xaxis()
    #ax.invert_yaxis()
    ax.set_axis_off()
    return scatter


def plot_nodes(valid_nodes, ax, i, save=False):
    valid_nodes = [node for node in valid_nodes if node is not None]

    x_coords = np.array([node.x for node in valid_nodes])
    y_coords = np.array([node.y for node in valid_nodes])
    u_values = np.array([node.u for node in valid_nodes])

    unique_x = np.unique(x_coords)
    unique_y = np.unique(y_coords)
    cols = len(unique_x)
    rows = len(unique_y)

    u_grid = np.full((rows, cols), np.nan)

    for node in valid_nodes:
        x_idx = np.where(unique_x == node.x)[0][0]
        y_idx = np.where(unique_y == node.y)[0][0]
        u_grid[y_idx, x_idx] = node.u

    im = ax.imshow(u_grid, cmap='bwr', origin='lower',
                   interpolation='nearest', aspect='equal')

    ax.set_title(f't = {i:.2f}')
    ax.set_axis_off()

    return im


#%% Create the nodes with random u and v values

def shape_nodes(shape_check, grid_size, initial_conditions):
    x_coords = np.arange(1, grid_size+1, 1)
    y_coords = np.arange(1, grid_size+1, 1)
    r = grid_size/2
    all_nodes = np.full((grid_size, grid_size), None)
    border_nodes = []
    node_id = 0
    initial_index = 0
    for y in y_coords:
        for x in x_coords:
            u = initial_conditions[initial_index, 0] # np.random.rand()  # Random u value between 0 and 1
            v = initial_conditions[initial_index, 1] #np.random.rand()  # Random v value between 0 and 1
            
            if shape_check(r, (x, y)): # if in shape
                node = Node(id=node_id, x=x, y=y, u=u, v=v)
                all_nodes[node.x -1, node.y -1] = node
                neighbours = np.array([(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)])
                # if any of the neighbours are outside the circle
                # this may send a coordinate outside of the domain - throwing errors sometimes if the function can't handle that
                if np.any(~shape_check(r, neighbours)):
                        border_nodes.append(node) 
                initial_index += 1
            node_id += 1

    return(all_nodes, border_nodes)

# %%
# def get_inward_node_value(x, y, coord_to_index, new_values): # NEUMAN
#     # neuman boundary conditions. check all around for perpendicular
#     directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
#                   (-1, -1), (-1, 1), (1, -1), (1, 1)]

#     for dx, dy in directions:
#         nx, ny = x + dx, y + dy # find new coordinates
#         if (nx, ny) not in coord_to_index:
#             # not a node on the outside so perpendicular edge
#             rx, ry = x - dx, y - dy
#             if (rx, ry) in coord_to_index:
#                 # returns index of perpendicular node
#                 return new_values[coord_to_index[(rx, ry)]]
#     return None  # no valid result

def generate_nodes(threshold=None, shape = shapes.grid_check, grid_size=20, initial_conditions = 'rand'):
    if threshold != None:
        Node.threshold = threshold
    # creates all nodes
    if isinstance(initial_conditions, str) and initial_conditions == 'rand':
        initial_conditions = np.random.rand(grid_size*grid_size, 2)
    
    all_nodes, border_nodes = shape_nodes(shape, grid_size, initial_conditions)

    valid_nodes = [node for node in all_nodes.flatten() if node is not None]
    
    adjacency_matrix = compute_adjacency_matrix(valid_nodes)
    degree_matrix = compute_degree_matrix(adjacency_matrix)

    return adjacency_matrix, degree_matrix, valid_nodes

def simulation(f, g, Du, Dv, grid_size=20, n = 10000, dt = 0.001, initial_conditions = 'rand', shape = shapes.grid_check, show=True, save=False, save_path="simulation_final.png", threshold = None):
    step_plot = n // 9
    adjacency_matrix, degree_matrix, valid_nodes = generate_nodes(threshold, shape, grid_size, initial_conditions)
    coord_to_index = {(int(node.x), int(node.y)): i for i, node in enumerate(valid_nodes)}

    if show:
        fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    
    u_log = np.zeros((len(valid_nodes), n))
    v_log = np.zeros((len(valid_nodes), n))

    for i in range(n):
        if i % 1000 == 0 and show:
            print(i)

        u_vector = np.array([float(node.u) for node in valid_nodes])
        laplacian_u = laplacian(degree_matrix, adjacency_matrix, u_vector)

        v_vector = np.array([float(node.v) for node in valid_nodes])
        laplacian_v = laplacian(degree_matrix, adjacency_matrix, v_vector)

        if np.any(np.isnan(u_vector)) or np.any(np.isnan(v_vector)):
            print('blow up detected')
            return u_vector, v_vector
        
        # i have moved du dv coefficients into dudt and dvdt instead of new_u and new_v
        dudt = f(u_vector, v_vector) + Du * laplacian_u
        dvdt = g(u_vector, v_vector) + Dv * laplacian_v

        # new_u = np.round(dudt * dt, 3) + u_vector # simulate discretisation of kilobots 
        # new_v = np.round(dvdt* dt, 3) + v_vector

        new_u = dudt * dt + u_vector # simulate discretisation of kilobots 
        new_v = dvdt * dt + v_vector

        new_u = np.clip(new_u, -65504, 65504)
        new_v = np.clip(new_v, -65504, 65504)

        for node in valid_nodes:
            if node == None:
                continue
            node_index = coord_to_index[(int(node.x), int(node.y))]
            node.u = new_u[node_index]
            node.v = new_v[node_index]

        if i % step_plot == 0 and i < 9 * step_plot and show:
            ax = axes.flat[i // step_plot]
            scatter = plot_nodes(valid_nodes, ax=ax, i=i)
            #easy_plot(valid_nodes)

        u_log[:, i] = u_vector
        v_log[:, i] = v_vector

    if save:
        # Create a single figure and axis for final plot
        fig, ax = plt.subplots(figsize=(6, 6))
        scatter = plot_nodes(valid_nodes, ax=ax, i=n, save=True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Simulation final plot saved to {save_path}")

    elif show:
        plt.tight_layout()
        plt.show()

    u_vector = np.array([float(node.u) for node in valid_nodes])
    v_vector = np.array([float(node.v) for node in valid_nodes])

    return u_vector, v_vector, (u_log, v_log)

#simulation(n, f, g)
# %%

