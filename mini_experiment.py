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

import ast
import re

valid_params = []
with open('valid_params_23june.txt', 'r') as file:
    for line in file:
        line = line.strip()
        # Remove trailing dot if present (optional)
        if line.endswith('.'):
            line = line[:-1]

        # Replace np.float64(xxx) with xxx using regex
        clean_line = re.sub(r'np\.(?:float64|complex128)\(([^)]+)\)', r'\1', line)
        
        # Safely parse the tuple string to a tuple
        tup = ast.literal_eval(clean_line)
        valid_params.append(tup)

#%%
# 612 - oscillation
param_number = 612

a, b, c, d, e, w, Du, Dv = valid_params[param_number][:8]
params = a, b, c, d, e, w, Du, Dv

lambda_alpha_one, eigvals_one = plots.dispersion_relation_plot(params, grid_size=1, plot=False,  shape = shapes.line_check, model = models.linear)
lambda_alpha_two, eigvals_two = plots.dispersion_relation_plot(params, grid_size=2, plot=False,  shape = shapes.line_check, model = models.linear)
lambda_alpha_three, eigvals_three = plots.dispersion_relation_plot(params, grid_size=3, plot=False,  shape = shapes.line_check, model = models.linear)
lambda_alpha_four, eigvals_four = plots.dispersion_relation_plot(params, grid_size=4, plot=False,  shape = shapes.line_check, model = models.linear)
lambda_alpha_five, eigvals_five = plots.dispersion_relation_plot(params, grid_size=5, plot=False,  shape = shapes.line_check, model = models.linear)
lambda_alpha_six, eigvals_six = plots.dispersion_relation_plot(params, grid_size=6, plot=False,  shape = shapes.line_check, model = models.linear)

#%% PLOT OF GROWTH RATE FOR SYSTEMS
x_axis = np.arange(np.floor(np.min(np.real(eigvals_six))), 0.1, 0.1)
y_axis = analysis.growth_rate(params, x_axis)

plt.plot(eigvals_one, lambda_alpha_one, 'x')
plt.plot(eigvals_two, lambda_alpha_two, '*')
plt.plot(eigvals_three, lambda_alpha_three, 'o')
plt.plot(eigvals_four, lambda_alpha_four, 'x')
plt.plot(eigvals_five, lambda_alpha_five, '*')
plt.plot(eigvals_six, lambda_alpha_six, 'o')

plt.axhline(y=0, color='grey', linestyle='--', linewidth=1)
plt.plot(x_axis, y_axis, color='green', linestyle='--')
plt.grid(True)
plt.xlabel('Eigenvalues')
plt.ylabel('Lambda Alpha: Growth Rate')
plt.legend(['one', 'two', 'three', 'four', 'five', 'six'], title='Number of fully connected nodes')

plt.show()

# %% PLOT OF SPACIAL INSTABILITY -> unstable when negative

x_axis = np.arange(-10, 0, 1)
y_axis = Du * Dv * x_axis ** 2 + (Du * e + Dv * a) * x_axis + (a*e - b*d)

plt.plot(x_axis, y_axis, 'x')
plt.axhline(y=0, color='grey', linestyle='--', linewidth=1)
plt.xlabel('Theoretical Eigenvalues')
plt.ylabel('C(growth rate)')
plt.grid(True)
plt.show()

# %%
a, b, c, d, e, w, Du, Dv = valid_params[param_number][:8]
params = a, b, c, d, e, w, Du, Dv
f, g = models.linear(a, b, c, d, e, w)
grid_size = 5
#initial = np.random.rand(grid_size**2, 2)
initial = np.ones((grid_size**2, 2)) /2

u0, v0 = parameters.find_steady_state(f, g, initial_guess=(1.0, 0.0))
initial = np.tile([u0, v0], (grid_size ** 2, 1)) 
initial += np.random.normal(loc=0.0, scale=1e-6, size=initial.shape)

u, v, log = simulation.simulation(f, g, Du, Dv, grid_size=grid_size, n = n, dt = 0.0001, initial_conditions = initial, shape = shapes.line_check, show=False)
# log size = 2 * nodes * n
u_log = np.array(log[0])
v_log = np.array(log[1])

# node_n = 0
plt.plot(v_log.T)
plt.xlabel('time')
plt.title(f"{grid_size} nodes fully connected with parameter set {param_number}")
plt.ylabel('u values')
plt.grid(True)
plt.show()


# %%
