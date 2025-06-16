#%% https://ipython-books.github.io/124-simulating-a-partial-differential-equation-reaction-diffusion-systems-and-turing-patterns/

import numpy as np
import matplotlib.pyplot as plt
#%%
# partial differential equations on the domain E=[−1,1]^2

# du/dt = Du d^2u/dx^2 + f(u, v)
# dv/dt = Dv d^2v/dx^2 + g(u, v)

k = -.005
tau = .1

def f(u, v):
    return(u - u**3 -v + k)

def g(u, v):
    return((u-v)/tau)

Du = 2.8e-4
Dv = 5e-3 / tau



# %% We discretize time and space. The time step dt must be small enough to ensure the stability of the numerical simulation

size = 100 # size of the 2D grid
dx = 2. / size # space step

T = 20.0 # total time
dt = .001 # time step
n = int(T/dt) # number of iterations
# %% We initialize the variables u and v. 
# The matrices U and V contain the values of these variables on the vertices of the 2D grid. 
# These variables are initialized with a uniform noise between 0 and 1

U = np.random.rand(size, size)
V = np.random.rand(size, size)


#%% we define a function that computes the discrete Laplace operator of a 2D variable on the grid, using a five-point stencil finite difference method
# change in u given change in space
# change u(x, y) = u(x+h, y) + u(x-h, y) + u(x, y+h) + u(x, y-h) - 4u(x, y) / dx^2

# remove the boarder for grid operation

def laplacian(Z):
    Ztop = Z[0:-2, 1:-1]
    Zleft = Z[1:-1, 0:-2]
    Zbottom = Z[2:, 1:-1]
    Zright = Z[1:-1, 2:]
    Zcenter = Z[1:-1, 1:-1]
    return (Ztop + Zleft + Zbottom + Zright -
            4 * Zcenter) / dx**2

def show_patterns(U, ax=None):
    ax.imshow(U, cmap=plt.cm.copper,
              interpolation='bilinear',
              extent=[-1, 1, -1, 1])
    ax.set_axis_off()

#%%

fig, axes = plt.subplots(3, 3, figsize=(8, 8))
step_plot = n // 9
# We simulate the PDE with the finite difference
# method.

for i in range(n):
    # We compute the Laplacian of u and v.
    deltaU = laplacian(U)
    deltaV = laplacian(V)
    # We take the values of u and v inside the grid.
    Uc = U[1:-1, 1:-1]
    Vc = V[1:-1, 1:-1]
    # We update the variables.
    U[1:-1, 1:-1], V[1:-1, 1:-1] = \
        Uc + dt * (Du * deltaU +  f(Uc, Vc)),\
        Vc + dt * (Dv * deltaV +  g(Uc, Vc))
    # Neumann conditions: derivatives at the edges
    # are null.
    for Z in (U, V):
        Z[0, :] = Z[1, :]
        Z[-1, :] = Z[-2, :]
        Z[:, 0] = Z[:, 1]
        Z[:, -1] = Z[:, -2]

    # We plot the state of the system at
    # 9 different times.
    if i % step_plot == 0 and i < 9 * step_plot:
        ax = axes.flat[i // step_plot]
        show_patterns(U, ax=ax)
        ax.set_title(f'$t={i * dt:.2f}$')

# %%
