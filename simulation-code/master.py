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
# morphogenesis: a = A - gamma
# f(u, v) = (Au + Bv + C )- gamma_u u
# g(u, v) = (Eu - F) - gamma_v v -> ( F = w)
#  A = 0.08, B = −0.08, C = 0.03, gu = 0.03, E = 0.1, F = 0.12, and gv = 0.06
# DU = 0.5 and DV = 10

a, b, c, d, e, w= 0.05, -0.08, 0.03, 0.1, -0.06, -0.12
# morhpogensis - scaled by 160 - scaled by du and dv
#a, b, c, d, e, w = 40, -6.4, 2.4, 160, -96, -192 <- some spots
a, b, c, d, e, w = 8, -12.8, 4.8, 16, -9.6, -19.2 #<- some stripes

f, g = models.linear(a, b, c, d, e, w)

grid_size = 10 # size of the 2D grid or square root of total number of robots
T = 30.0 # total time
dt = 0.001 # time step
n = int(T/dt) # number of iterations
Du, Dv = 0.5, 10

u, v, _ = simulation.simulation(f, g, Du, Dv, grid_size=grid_size, n=n, dt=dt, show=True, threshold=1.1)
params = a, b, c, d, e, w, Du, Dv

# if Re(lamda(0)) > 0 the system is unstable without diffusion (bad)
# if Re(lamda(k)) > 0 for some nonzero k, the system is unstable and can form a pattern
# ideally there should be a peak

#%%
# (a, b, c, d, e, w, Du, Dv, peak/width, peak_lamda, peak_k, unstable_band_width, area, peak/area)

# import parameter information from txt file
import ast
import re

valid_params = []
with open('valid_params.txt', 'r') as file:
    for line in file:
        line = line.strip()

        # Remove trailing dot (if mistakenly added during printing)
        if line.endswith('.'):
            line = line[:-1]

        # Replace NumPy wrappers with raw values
        line = re.sub(r'np\.(?:float64|complex128)\(([^)]+)\)', r'\1', line)
        line = line.replace('np.False_', 'False').replace('np.True_', 'True')

        # Now it's a clean literal tuple
        try:
            tup = ast.literal_eval(line)
            valid_params.append(tup)
        except Exception as e:
            print(f"Skipping line due to parse error: {line} ({e})")

valid_params = [p[:5] for p in valid_params if p[-1] == True]

#%% PARAM CHOOSING

goal_eig = -4
goal_growth = 30
goal_root_width = 0.5

param_list = parameters.find_suitable_params(goal_eig, goal_growth, goal_root_width, valid_params)
a, b, c, d = param_list[0][0]
sigma = param_list[0][1]
epsilon = param_list[0][2]
_ = plots.dispersion_relation_plot((a, b, 0, c, d, 0, epsilon, sigma*epsilon), grid_size = 10, continuous=False)

#%%

# %%
desired_eig = -4 # exact
desired_growth = 5 # exact
desired_width = 0.1 # kind of

grid_size = 20
initial = models.initial_fill(grid_size, 0.5)

optimal = parameters.find_suitable_params(desired_eig, desired_growth, desired_width, valid_params)

plots.plot_dispersion_turing(optimal, [0, 1, 2, 3, 5], grid_size = grid_size, initial_conditions = initial)
#%%

a, b, c, d = optimal[2][0]
a, b, c, d = 49.373, -82.998, 61.178, -93.445
sigma = optimal[2][1]
Du = optimal[2][2]
root_width = optimal[2][3]
Dv = sigma * Du
Du, Dv = 2.0394, 82.322
f, g = models.linear(a, b, 0, c, d, 0)
grid_size = 10
u_vector, v_vector, (u_log, v_log) = simulation.simulation(f, g, Du, Dv, grid_size=grid_size, n = 2500, dt = 0.001, initial_conditions = initial, shape = shapes.grid_check, show=True)

#%%

u_grid = u_vector.reshape((grid_size, grid_size))
fig, ax = plt.subplots(figsize=(6,6))
im = ax.imshow(u_grid, cmap='viridis', interpolation='none')

# Set ticks at pixel edges
ax.set_xticks(np.arange(u_grid.shape[1] + 1) - 0.5)
ax.set_yticks(np.arange(u_grid.shape[0] + 1) - 0.5)

# Draw white grid lines
ax.grid(which='major', color='white', linestyle='-', linewidth=1)

# Hide ticks (optional)
ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

# Remove axes spines
for spine in ax.spines.values():
    spine.set_visible(False)

plt.show()
#%% CHOOSE INDICIES
# some k = 2.4 start becoming overly pixelated and more difficult
# could be worked on?
# k = 0.5 gives max two spots
# k = 0.8 gives approx 3 (1 across)
# k = 1 -> 5 (1.5 across)
# k = 1.5 -> 9 (3 across)
# k = 2 -> 13 (3.5 across?)
grid_size = 20
kx = np.fft.fftfreq(grid_size, d=1)  # no 2π
ky = np.fft.fftfreq(grid_size, d=1)

kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
k_squared = ((2 * np.sin(np.pi * kx_grid))**2 + (2 * np.sin(np.pi * ky_grid))**2).flatten()
k_sqrt = np.sqrt(k_squared)

k_unique = np.unique(np.round(np.sort(-k_sqrt), decimals=6))  # to remove duplicate magnitudes

# k_unique = [2.828427, 2.81107 , 2.793604, 2.760079, 2.742289, 2.689994, 2.678726, 2.660392, 2.606454, 2.572554, 2.553458, 2.520147, 2.497212, 2.44949 , 2.429426, 2.406991, 2.370239, 2.319906, 2.298712, 2.288246, 2.274988, 2.236068, 2.196458, 2.174061, 2.148961, 2.134839, 2.107715, 2.093315, 2.069802, 2.024324, 2.      , 1.975377, 1.927672, 1.902113, 1.886143, 1.855388, 1.839012, 1.80927 , 1.782013, 1.732051, 1.680604, 1.662508, 1.648005, 1.618034, 1.543362, 1.485394, 1.448408, 1.414214, 1.328131, 1.284079, 1.216492, 1.175571, 1.09836 , 0.960373, 0.907981, 0.874032, 0.692714, 0.618034, 0.442463, 0.312869, 0.]
#%%
# peak/area with 1.404963 is good
# k = 2.66  and 0.314159 rubbish regardless of scoring - wavelength too big
 # k between 0.0888577 ->
 

eigevalue_wavesize = -12
num_selected = 20

#indices = [i for i, param in enumerate(valid_params)]

# filter where the peak is positive (param[10] is lambda)
#filtered_growth = [(i, param) for i, param in enumerate(valid_params) if np.real(param[9]) > 0]

# find the k value at peak and the peak/width score
# must be positive lambda growth
peak_growth = np.array([param[10] for _, param in filtered_growth])

# param[8] = peak/width (prioritise crispness); param[-1] = peak/area (robust?) prefer this?
scores = np.array([param[-1] for _, param in filtered_growth])

indices = [i for i, _ in filtered_growth]

# this might be smaller if limiting by positive growth
differences = np.abs(np.abs(peak_growth) - np.abs(eigevalue_wavesize))

ranked_indices = np.argsort(differences)[:num_selected]
closest_indices = [indices[i] for i in ranked_indices]


# # combine these and sort - then pick out these indicies
# composite_scores = (1 / (1 + differences) ** 2) * scores
# sorted_indices = np.argsort(-composite_scores)
# closest_indices = np.array(indices)[sorted_indices[:num_selected]]

# closest_indices = np.array(indices)[np.argsort(differences)[:num_selected]]
#%% PLOT PATTERN AND DISPERSION FOR DIFFERENT PARAMETERS

# n = 10000

# flicking behaviour when dominant root changes sign??

initial_conditions = np.ones((400, 2)) /2
#initial_conditions = np.random.rand(400, 2)
#grid_size = 6  
parameters.plot_dispersion_turing(valid_params, closest_indices[:5], initial_conditions=initial_conditions) #, initial_conditions=initial_conditions

#%% SPECIFIC REPEAT?
a, b, c, d = 49.373, -82.998, 61.178, -93.445
J = parameters.non_spacial_jacobian(a, b, c, d)
f, g = models.linear(a, b, 0, c, d, 0)
grid_size = 10
dt = 0.001
eig = -2
desired_growth = 30 # exact
#desired_width = 0.1 # kind of
initial = models.initial_fill(grid_size, fill = -10, pert=True)
#initial[1::2] = [1., 1.]
sigma = parameters.find_sigma_for_growth(J, desired_growth)
Du = parameters.epsilon_for_eigenvalue(J, eig, sigma)
Dv = sigma * Du

# params = valid_params[300][:8]
# a, b, c, d, e, w, Du, Dv = params
# model = models.linear
# f, g = model(a, b, c, d, e, w)

# initial not same u and v
#initial_conditions = np.column_stack((np.full(400, 0.2), np.full(400, 0.8)))

initial_conditions = np.ones((400, 2)) /2
initial = models.initial_fill(grid_size=grid_size, pert=True, fill = 0)

shape = shapes.grid_check
n = 5000
show = True

u, v, (u_log, v_log) = simulation.simulation(f, g, Du, Dv, dt = 0.001, n = n, initial_conditions=initial, shape=shape, show=show, grid_size=grid_size)
#%%
_ = plt.plot(u_log.T)
_ = plots.dispersion_relation_plot((a, b, 0, c, d, 0, Du, Dv), continuous=True, grid_size=10, threshold=1.9)
wavelengths, powers = analysis.plot_wavelength_spectrum(u.reshape((10,10)), log_scale = False)
#plots.turing_check(f, g, Du, Dv)
#%%
plots.dispersion_plot(params, model, grid_size)

plots.fourier_spectrum_plot(u)

#%% scoring for robustness and sharpness with fourier spectrum

#%%

def reshape_output(u, v, grid_size):
    u_reshape = u.reshape(grid_size, grid_size)
    v_reshape = v.reshape(grid_size, grid_size)
    u_reshape = u_reshape.T 
    v_reshape = v_reshape.T 
    u_reshape = u_reshape.flatten() 
    v_reshape = v_reshape.flatten() 
    return u_reshape, v_reshape

#%% CHANGING SHAPE?

# a, b, c, d, e, w, Du, Dv = valid_params[478][:8]
# f, g = models.linear(a, b, c, d, e, w)

u_reshape, v_reshape = reshape_output(u, v, grid_size=10)
initial_conditions = np.column_stack((u_reshape, v_reshape))
#initial_conditions = np.column_stack((np.full(400, 0.2), np.full(400, 0.8)))

shape = shapes.circle_outline_check
# n = 30000
n = 1000
u, v, (u_log, v_log) = simulation.simulation(f, g, Du, Dv, n = n, initial_conditions=initial_conditions, shape=shape, show=True, threshold=1.5, grid_size=grid_size)
_ = plt.plot(u_log.T)

#%%
# adds the small grid into the big grid:
def add_grid_in(u, v, r):
    inner_min = r // 4 # small grid conditions
    inner_max = 3 * r // 4

    x, y = np.meshgrid(np.arange(r), np.arange(r))
    coords = np.stack([x.ravel(), y.ravel()], axis=1)  # shape (400, 2)

    mask = (inner_min <= coords[:, 0]) & (coords[:, 0] < inner_max) & \
        (inner_min <= coords[:, 1]) & (coords[:, 1] < inner_max)

    u_array = np.ones(r * r)
    v_array = np.ones(r * r)

    u_array[mask] = u
    v_array[mask] = v
    return u_array, v_array

#%%
def fix_u_nodes(u, perc=70):
    def fixed_shape(r, coords):
        grid_size = int(r*2)
        coord_to_index = {
        (i, j): (i - 1) * grid_size + (j - 1)
        for i in range(1, grid_size + 1)
        for j in range(1, grid_size + 1)
        }
        threshold = np.percentile(u, abs(perc))
        # print('coords: ', coords)
        if np.ndim(coords) == 1:
            x, y = coords
            key = (int(x), int(y))
            if perc < 0:
                mask = np.array([
                    u[coord_to_index[key]] > threshold if key in coord_to_index else False
                ])
            else:
                mask = np.array([
                    u[coord_to_index[key]] < threshold if key in coord_to_index else False
                ])
        else:
            x = coords[:, 0]
            y = coords[:, 1]
            # print('x: ', x)
            # print('y: ', y)
            if perc < 0:
                mask = np.array([
                    u[coord_to_index[(int(x), int(y))]] > threshold if (int(x), int(y)) in coord_to_index else False
                    for x, y in coords
                ])
            else:
                mask = np.array([
                    u[coord_to_index[(int(x), int(y))]] < threshold if (int(x), int(y)) in coord_to_index else False
                    for x, y in coords
                ])
            
        return mask
    
    return fixed_shape
    


#%%
u_reshape, v_reshape = reshape_output(u, v, grid_size=10)
initial_conditions = np.column_stack((u_reshape, v_reshape))
shape = fix_u_nodes(u, perc=95)
n = 20000
u_new, v_new, _ = simulation.simulation(f, g, Du, Dv, n = n, initial_conditions=initial_conditions, shape=shape, show=True)

#%%
def apply_mask_with_fill(u, grid_size, mask, fill_value=np.nan): # refill to grid size before removal
    result = np.full(grid_size**2, fill_value, dtype=float)  # create full
    print(len(u))
    result[mask] = u  # fill values where mask True
    return result

x_coords = np.arange(1, grid_size+1, 1)
y_coords = np.arange(1, grid_size+1, 1)
xx, yy = np.meshgrid(x_coords, y_coords) 
coords = np.stack([xx.ravel(), yy.ravel()], axis=1) 
r = grid_size/2
# recreate the shape restrictions of the last iteration and create the mask
reshape_u, _ = reshape_output(u, u)
shape = fix_u_nodes(reshape_u)
mask = shape(r, coords)

# fill in this mask so its back to a 20x20 grid, but the extra ones will be excluded
u_full = apply_mask_with_fill(u_new, grid_size, mask, fill_value=(max(u_new)))
v_full = apply_mask_with_fill(v_new, grid_size, mask, fill_value=(max(u_new)))
# reshape ready to go
u_reshape, v_reshape = reshape_output(u_full, v_full)

# resend with the new info
initial_conditions = np.column_stack((u_reshape, v_reshape))
# new shape reapplies condition
shape = fix_u_nodes(u_full, perc=60)
n = 20000
u_newest, v_newest, _ = simulation.simulation(f, g, Du, Dv, n = n, initial_conditions=initial_conditions, shape=shape, show=True)


# plt.scatter(np.arange(len(u)), u)

#%% GENERAL TURING CHECK
a, b, c, d, e, w, Du, Dv = valid_params[278][:8]
f, g = models.linear(a, b, c, d, e, w)

parameters.check_for_instability(f, g, Du, Dv)
# plots.dispersion_relation_plot(f, g, Du, Dv)
# %%

# %% PLOT RELATION BETWEEN PARAMETERS
import pandas as pd
import seaborn as sns

# Create DataFrame from valid_params
cols = ['a', 'b', 'c', 'd', 'e', 'w', 'Du', 'Dv', 'peak', 'peak_lambda', 'peak_k', 'unstable_band', 'area']
df = pd.DataFrame(valid_params, columns=cols)

sns.pairplot(df, vars=['a', 'b', 'c', 'd', 'e', 'w', 'Du', 'Dv'],
             hue='area', palette='viridis', plot_kws={'alpha':0.7, 's':30}, corner=True)
plt.suptitle('Parameter Pairwise Comparison (colored by area)', y=1.02)
plt.show()

