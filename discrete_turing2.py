#%% no boundary conditions and differing distance
import numpy as np
import matplotlib.pyplot as plt

threshold = 1.5 # what counts as a neighbour?
dt = .001 # time step

class Node:
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
            if self.distance_to(node) < threshold:
                self.add_neighbour(node)

    def __repr__(self):
        return f"Node(id={self.id}, coordinates=({self.x}, {self.y}, u={self.u}, v={self.v}, neighbours={len(self.neighbours)})"
#%% computation

def compute_adjacency_matrix(all_nodes):
    n = len(all_nodes)
    adjacency_matrix = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        node = all_nodes[i]
        node.update_all_neighbours(all_nodes)
        for j in range(n):
            neighbour = all_nodes[j]
            if neighbour in node.neighbours:
                adjacency_matrix[i, j] = 1  # There is an edge/connection between these two nodes
    
    return adjacency_matrix

def compute_degree_matrix(adjacency_matrix):
    degree_vector = adjacency_matrix.sum(axis = 0)
    degree_matrix = np.diag(degree_vector)
    return degree_matrix

def laplacian(L, c):
    Lc = np.dot(L, c)
    return(Lc)

#%%
# partial differential equations on the domain E=[−1,1]^2

# du/dt = Du d^2u/dx^2 + f(u, v)
# dv/dt = Dv d^2v/dx^2 + g(u, v)

# define the linear term:
k = -.005
tau = .1

def f(u, v):
    return(u - u**3 -v + k)

def g(u, v):
    return((u-v)/tau)

Du = 2.8e-4
Dv = 5e-3 / tau


#%%
grid_size = 20 # size of the 2D grid
T = 9.0 # total time
n = int(T/dt) # number of iterations

# Create evenly spaced coordinates for x and y across the grid
# x_coords = np.linspace(-1, 1, grid_size) # threshold 1.5 gives circles
# y_coords = np.linspace(-1, 1, grid_size)

x_coords = np.linspace(0, grid_size, grid_size)
y_coords = np.linspace(0, grid_size, grid_size)

# Create the nodes with random u and v values
all_nodes = []
update_u_nodes = []
node_id = 0
for i in range(grid_size):
    for j in range(grid_size):
        x = x_coords[i]
        y = y_coords[j]

        v = np.random.rand()  # Random v value between 0 and 1
        u = np.random.rand()  # Random u value between 0 and 1

        node = Node(id=node_id, x=x, y=y, u=u, v=v)
        
        update_u_nodes.append(node)
        all_nodes.append(node)
        node_id += 1


# %%

def plot_nodes(all_nodes, ax, i):
    x_coords = [node.x for node in all_nodes]
    y_coords = [node.y for node in all_nodes]
    u_values = [node.u for node in all_nodes]

    scatter = ax.scatter(x_coords, y_coords, c=u_values, cmap='viridis', s=100, edgecolors='k')
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_title(f't={i * dt:.2f}')
    ax.set_axis_off()
    return scatter

fig, axes = plt.subplots(3, 3, figsize=(8, 8))
step_plot = n // 9

adjacency_matrix = compute_adjacency_matrix(all_nodes)
degree_matrix = compute_degree_matrix(adjacency_matrix)
L = adjacency_matrix - degree_matrix

for i in range(n):
    if i % 100 == 0:
        print(i)

    u_vector = np.array([float(node.u) for node in all_nodes])
    laplacian_u = laplacian(L, u_vector)

    v_vector = np.array([float(node.v) for node in all_nodes])
    laplacian_v = laplacian(L, v_vector)

    dudt = f(u_vector, v_vector) + laplacian_u
    dvdt = g(u_vector, v_vector) + laplacian_v

    new_u = dudt * dt + u_vector
    new_v = dvdt * dt + v_vector

    for j, node in enumerate(all_nodes):
        node.v = new_v[j]
        node.u = new_u[j]

    if i % step_plot == 0 and i < 9 * step_plot:
        ax = axes.flat[i // step_plot]
        scatter = plot_nodes(all_nodes, ax=ax, i=i)

plt.tight_layout()  # Adjust spacing between subplots
plt.show()
# %%
