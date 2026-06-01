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
import matplotlib.image as mpimg

#%%

importlib.reload(simulation)
importlib.reload(models)
importlib.reload(plots)
importlib.reload(shapes)
importlib.reload(parameters)
importlib.reload(analysis)

#%%

valid_params = parameters.load_params('valid_params.txt')
valid_params = [p[:5] for p in valid_params if p[-1] == True]


desired_eig = -1.5 # exact
desired_growth = 30 # exact
desired_width = 0.1 # kind of

optimal = parameters.find_suitable_params(desired_eig, desired_growth, desired_width, valid_params)
option = 0
a, b, c, d = optimal[option][0]
sigma = optimal[option][1]
Du = optimal[option][2]
root_width = optimal[option][3]
Dv = sigma * Du

#a, b, c, d = 49.373, -82.998, 61.178, -93.445
#Du, Dv = 2.0394, 82.322
J = parameters.non_spacial_jacobian(a, b, c, d)
f, g = models.linear(a, b, 0, c, d, 0)

params = a, b, 0, c, d, 0, Du, Dv
grid_size = 20

threshold = 1.2

eigvals, eigvecs, lambda_alpha = analysis.linear_stability(params, grid_size = grid_size, threshold=threshold)

#%% initial mode inputs (how much of each mode to start with)

np.random.seed(42)
initial_conditions = np.random.rand(grid_size*grid_size, 2)

t = 1.5

initial_mode_inputs = np.dot(eigvecs.T, initial_conditions)

solution = eigvecs @ (np.exp(lambda_alpha * t)[:, None] * initial_mode_inputs)

u_array_analytic, v_array_analytic = solution[:, 0].real, solution[:, 1].real

u_array_scaled = u_array_analytic * 1e50 

adjacency_matrix, degree_matrix, valid_nodes = simulation.generate_nodes(threshold=threshold, grid_size=grid_size)

L = adjacency_matrix - degree_matrix

#_, power_flat = analysis.plot_wavelength_spectrum(u_array_scaled.reshape((20,20)), plot_spectrum = True, log_scale=False, normalise=True)
_, power_flat = analysis.graph_spectral_analysis(u_array_analytic, plot = True, L=L, k_eigs=300)

# %% PLOTTING

#data = np.clip(u_array_analytic, -1e2, 1e2).reshape((grid_size, grid_size))
data = u_array_analytic.reshape((grid_size, grid_size))
fig, ax = plt.subplots()

# Plot with pixelated look
im = ax.imshow(data, cmap='bwr', origin='lower', interpolation='nearest', aspect='equal')

# Tidy plot
ax.set_axis_off()
plt.show()
# %% using the dominant growth rate only

lambda_alpha_dominant = np.zeros(len(lambda_alpha))
lambda_alpha_dominant[lambda_alpha.argmax()] = lambda_alpha[lambda_alpha.argmax()]

solution = eigvecs @ (np.exp(lambda_alpha_dominant * t)[:, None] * initial_mode_inputs)

u_array_dominant, v_array_dominant = solution[:, 0].real, solution[:, 1].real

# %% numerically solved

time = int(t * 1000)

u_vector_numeric, v_vector_numeric, (u_log, v_log) = simulation.simulation(f, g, Du, Dv, grid_size = grid_size, n = time, initial_conditions = initial_conditions, threshold=1.1, show=False)

data = u_vector_numeric.reshape((grid_size, grid_size))
fig, ax = plt.subplots()

# Plot with pixelated look
im = ax.imshow(data, cmap='bwr', origin='lower', interpolation='nearest', aspect='equal')

# Tidy plot
ax.set_axis_off()
plt.show()

adjacency_matrix, degree_matrix, valid_nodes = simulation.generate_nodes(threshold=1.1, shape = shapes.grid_check, grid_size=grid_size)
L = adjacency_matrix - degree_matrix

_, power_flat = analysis.graph_spectral_analysis(u_vector_numeric, plot = True, L=L, k_eigs=300)


# %% 

fig, ax = plt.subplots()

ax.plot(eigvals, lambda_alpha.real, label='Continuous')
ax.plot(eigvals, lambda_alpha.real, 'o', label='Discrete')

ax.axhline(0, color='gray', linestyle='--')

# point to annotate
idx = np.argmax(lambda_alpha.real)

x_point = eigvals[idx]
y_point = lambda_alpha.real[idx]

# annotation
ax.annotate(
    'Dominant Turing Mode ($\\Lambda_m$, $\\lambda_m$)',
    xy=(x_point, y_point),          # point location
    xytext=(x_point - 6.5, y_point - 17.0),  # text location
    arrowprops=dict(arrowstyle='->', lw=2),
    fontsize=18
)

ax.set_xlabel('Laplacian Eigenvalue $\\Lambda$')
ax.set_ylabel('Growth Rate Re($\\lambda$)')

ax.grid(True)
ax.legend(fontsize=18)

plt.show()