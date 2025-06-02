#%% boundary condition
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

Du = 1; Dv = 20

# linear:
def f(u, v, a, b):
    return(a*u + b*v)

def g(u, v, c, d):
    return(c*u + d*v)

# gray-scott
def f(u, v, a, b):
    return(u**2 * v - (a+b)*u)

def g(u, v, a):
    return(-u**2*v+a*(1+v))

#%%
grid_size = 20 # size of the 2D grid
T = 5.0 # total time
n = int(T/dt) # number of iterations

# Create evenly spaced coordinates for x and y across the grid
# x_coords = np.linspace(-1, 1, grid_size) # threshold 1.5 gives circles
# y_coords = np.linspace(-1, 1, grid_size)

#%% Create the nodes with random u and v values
x_coords = np.linspace(1, grid_size, grid_size)
y_coords = np.linspace(1, grid_size, grid_size)

def initiate_nodes():
    all_nodes = []
    border_nodes = []
    fixed_nodes = []
    node_id = 0
    for y in y_coords:
        for x in x_coords:
            u = np.random.rand()  # Random u value between 0 and 1
            v = np.random.rand()  # Random v value between 0 and 1
            node = Node(id=node_id, x=x, y=y, u=u, v=v)
                
            node_id += 1
            all_nodes.append(node)

            if (x == 1 or x == grid_size or y == 0 or y == grid_size):
                    border_nodes.append(node)
            #     u = 0.0
            #     v = 0.0  # Random v value between 0 and 1
            #     node = Node(id=node_id, x=x, y=y, u=u, v=v)
            #     fixed_nodes.append(node)
            # else:
        
    adjacency_matrix = compute_adjacency_matrix(all_nodes)
    degree_matrix = compute_degree_matrix(adjacency_matrix)
    L = adjacency_matrix - degree_matrix
    return(all_nodes, L)


# %%

def plot_nodes(all_nodes, ax, i):
    x_coords = [node.x for node in all_nodes]
    y_coords = [node.y for node in all_nodes]
    u_values = [node.u for node in all_nodes]

    scatter = ax.scatter(x_coords, y_coords, c=u_values, cmap='viridis', s=100, edgecolors='k')
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_title(f't={i * dt:.2f}')
    #ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_axis_off()
    return scatter

def run_simulation(all_nodes, a, b, c, d, Du=1, Dv=20):
    for i in range(n):
        if i % 1000 == 0:
            print(i)

        u_vector = np.array([float(node.u) for node in all_nodes])
        laplacian_u = laplacian(L, u_vector)

        v_vector = np.array([float(node.v) for node in all_nodes])
        laplacian_v = laplacian(L, v_vector)

        # dudt = f(u_vector, v_vector, a, b) + laplacian_u
        # dvdt = g(u_vector, v_vector, c, d) + laplacian_v

        dudt = f(u_vector, v_vector, a, b) + laplacian_u
        dvdt = g(u_vector, v_vector, a) + laplacian_v

        new_u = Du * dudt * dt + u_vector
        new_v = Dv * dvdt * dt + v_vector

        new_u_square = new_u.reshape(20, 20) # match image
        new_v_square = new_v.reshape(20, 20)

        for j, node in enumerate(all_nodes): # neuman boundary conditions               
                if node.x == 1: # left hand column
                    # one x to the right same y
                    node.u = new_u_square[int(node.x), int(node.y) - 1]
                    node.v = new_v_square[int(node.x), int(node.y) - 1]
                elif node.x == grid_size:
                    node.u = new_u_square[int(node.x) - 2, int(node.y) - 1]
                    node.v = new_v_square[int(node.x) - 2, int(node.y) - 1]
                elif node.y == 1:
                    # same x
                    node.u = new_u_square[int(node.x) - 1, int(node.y)]
                    node.v = new_v_square[int(node.x) - 1, int(node.y)]
                elif node.y == grid_size:
                    node.u = new_u_square[int(node.x) - 1, int(node.y) - 2]
                    node.v = new_v_square[int(node.x) - 1, int(node.y) - 2]
                else: # not border
                    node.u = new_u_square[int(node.x) -1 , int(node.y) - 1]
                    node.v = new_v_square[int(node.x) -1, int(node.y) - 1]

    return all_nodes
# %%

a_vals = [0.05, 0.1, 1]
b_vals = [0.05, 0.1, 1]
c_vals = [-1.0, -1.5, -2]
d_vals = [-1.0, -2.0, -3]

Du_val = [0.07, 0.1, 0.15]
Dv_val = [20, 50, 100]
# Store plots
fig, axes = plt.subplots(len(a_vals), len(c_vals), figsize=(8, 8))

for i, a in enumerate(a_vals):
    for j, b in enumerate(b_vals):
        all_nodes, L = initiate_nodes()
        all_nodes = run_simulation(all_nodes, a=a, b=b, c=-2, d=-2, Du=0.1, Dv = 50)
        ax = axes[i, j]
        scatter = plot_nodes(all_nodes, ax=ax, i=i)
        ax.set_title(f"a={a}, b={b}")
        ax.axis('off')

plt.tight_layout()
plt.show()
# %%
