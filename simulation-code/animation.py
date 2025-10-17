#%%
import simulation
import models
import plots
import numpy as np
import matplotlib.pyplot as plt
import importlib
import parameters
import analysis
import shapes
#%%
importlib.reload(simulation)
importlib.reload(models)
importlib.reload(plots)
importlib.reload(shapes)
importlib.reload(parameters)
importlib.reload(analysis)
#%%

a, b, c, d = 49.373, -82.998, 61.178, -93.445
J = parameters.non_spacial_jacobian(a, b, c, d)
f, g = models.linear(a, b, 0, c, d, 0)
grid_size = 10

eig = -2
desired_growth = 30 # exact
#desired_width = 0.1 # kind of
initial = models.initial_fill(grid_size, fill = 0, pert=True)
#initial[1::2] = [1., 1.]
sigma = parameters.find_sigma_for_growth(J, desired_growth)
Du = parameters.epsilon_for_eigenvalue(J, eig, sigma)
Dv = sigma * Du

shape = shapes.grid_check
n = 1600
show = True

u, v, (u_log, v_log) = simulation.simulation(f, g, Du, Dv, n = n, initial_conditions=initial, shape=shape, show=show, threshold=1.9, grid_size=grid_size)
_ = plt.plot(u_log.T)

#%%
import matplotlib.animation as animation

y_data = u_log
n_lines, n_time = y_data.shape
time = np.arange(n_time)

# Set up figure and axis
fig, ax = plt.subplots()
ax.set_xlim(0, n_time)
#ax.set_ylim(np.min(y_data), np.max(y_data))
ax.set_xlabel("time")
ax.set_ylabel("u value")

# Create 100 empty line objects
lines = [ax.plot([], [], lw=1)[0] for _ in range(n_lines)]

# Initialization
def init():
    for line in lines:
        line.set_data([], [])
    return lines

# Update function
# def update(frame):
#     if frame == 0:
#         return lines
#     x = time[:frame]
#     for i, line in enumerate(lines):
#         y = y_data[i, :frame]
#         line.set_data(x, y)
#     return lines

def update(frame):
    if frame == 0:
        return lines
    x = time[:frame]
    current_y = []

    for i, line in enumerate(lines):
        y = y_data[i, :frame]
        line.set_data(x, y)
        current_y.append(y)

    # Flatten current_y to a 1D array of all visible points
    current_y = np.hstack(current_y)

    # Update y-axis limits with some padding
    y_min, y_max = np.min(current_y), np.max(current_y)
    padding = (y_max - y_min) * 0.1  # 10% padding
    ax.set_ylim(y_min - padding, y_max + padding)

    return lines

# Create animation
ani = animation.FuncAnimation(
    fig, update, frames=n_time,
    init_func=init, blit=True, interval=10, repeat=False
)

plt.tight_layout()
ani.save('u_log_line.mp4', writer='ffmpeg', fps=30)

# %%

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

grid_size = 10

# Set up the figure
fig, ax = plt.subplots()
data = u_log[:, 0].reshape((grid_size, grid_size))

# Initialize heatmap
im = ax.imshow(data, cmap='bwr', origin='lower', interpolation='nearest', aspect='equal')

# Tidy up
ax.set_axis_off()
plt.tight_layout()

# Update function for animation
def update(frame):
    new_data = u_log[:, frame].reshape((grid_size, grid_size))
    im.set_data(new_data)
    return [im]

# Create animation
ani = animation.FuncAnimation(
    fig, update, frames=n_time, interval=20, blit=True, repeat=False
)

plt.tight_layout()
ani.save('u_log_heat.mp4', writer='ffmpeg', fps=30)


# %%
