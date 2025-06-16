#%%
import discrete_turning
import models
import evaluation
import numpy as np
import matplotlib.pyplot as plt
import importlib
import parameters
import shapes
#%%
importlib.reload(discrete_turning)
importlib.reload(models)
importlib.reload(evaluation)
importlib.reload(shapes)
importlib.reload(parameters)

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

grid_size = 20 # size of the 2D grid or square root of total number of robots
T = 30.0 # total time
dt = 0.001 # time step
n = int(T/dt) # number of iterations
Du, Dv = 0.5, 10

u, v = discrete_turning.simulation(f, g, Du, Dv, n=n, dt=dt, show=True)
evaluation.turing_check(f, g, Du, Dv)

# if Re(lamda(0)) > 0 the system is unstable without diffusion (bad)
# if Re(lamda(k)) > 0 for some nonzero k, the system is unstable and can form a pattern
# ideally there should be a peak

#%%
# (a, b, c, d, e, w, Du, Dv, peak/width, peak_lamda, peak_k, unstable_band_width, area, peak/area)

# import parameter information from txt file
import ast
import re

valid_params = []
with open('valid_params_scored.txt', 'r') as file:
    for line in file:
        line = line.strip()
        # Remove trailing dot if present (optional)
        if line.endswith('.'):
            line = line[:-1]

        # Replace np.float64(xxx) with xxx using regex
        clean_line = re.sub(r'np\.float64\(([^)]+)\)', r'\1', line)
        
        # Safely parse the tuple string to a tuple
        tup = ast.literal_eval(clean_line)
        valid_params.append(tup)

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

# k_unique = [2.828427, 2.81107 , 2.793604, 2.760079, 2.742289, 2.689994,
#    2.678726, 2.660392, 2.606454, 2.572554, 2.553458, 2.520147,
#    2.497212, 2.44949 , 2.429426, 2.406991, 2.370239, 2.319906,
#    2.298712, 2.288246, 2.274988, 2.236068, 2.196458, 2.174061,
#    2.148961, 2.134839, 2.107715, 2.093315, 2.069802, 2.024324,
#    2.      , 1.975377, 1.927672, 1.902113, 1.886143, 1.855388,
#    1.839012, 1.80927 , 1.782013, 1.732051, 1.680604, 1.662508,
#    1.648005, 1.618034, 1.543362, 1.485394, 1.448408, 1.414214,
#    1.328131, 1.284079, 1.216492, 1.175571, 1.09836 , 0.960373,
#    0.907981, 0.874032, 0.692714, 0.618034, 0.442463, 0.312869,
#    0.]
#%%
# peak/area with 1.404963 is good
# k = 2.66  and 0.314159 rubbish regardless of scoring - wavelength too big
 # k between 0.0888577 ->
k_val = 1.414214
num_selected = 20

indices = [i for i, param in enumerate(valid_params) if param[10] > 0]

# find the k value at peak and the peak/width score
peak_ks_positive = np.array([valid_params[i][10] for i in indices])
# param[8] = peak/width (prioritise crispness); param[-1] = peak/area (robust?) prefer this?
scores = np.array([valid_params[i][-1] for i in indices]) 

differences = np.abs(peak_ks_positive - k_val)

# combine these and sort - then pick out these indicies
composite_scores = (1 / (1 + differences) ** 2) * scores
sorted_indices = np.argsort(-composite_scores)
closest_indices = np.array(indices)[sorted_indices[:num_selected]]

#%% PLOT PATTERN AND DISPERSION FOR DIFFERENT PARAMETERS

# n = 10000

initial_conditions = np.ones((400, 2)) /2
#initial_conditions = np.random.rand(400, 2)
#grid_size = 6  
parameters.plot_dispersion_turing(valid_params, closest_indices[:5], initial_conditions=initial_conditions) #, initial_conditions=initial_conditions

#%% SPECIFIC REPEAT?
grid_size = 20
params = valid_params[1181][:8]
a, b, c, d, e, w, Du, Dv = params
model = models.linear
f, g = model(a, b, c, d, e, w)

# initial not same u and v
#initial_conditions = np.column_stack((np.full(400, 0.2), np.full(400, 0.8)))

initial_conditions = np.ones((400, 2)) /2
#initial_conditions = np.random.rand(400, 2)

shape = shapes.grid_check
n = 5000
show = True
u, v = discrete_turning.simulation(f, g, Du, Dv, n = n, initial_conditions=initial_conditions, shape=shape, show=show)
#evaluation.turing_check(f, g, Du, Dv)
#%%
evaluation.dispersion_plot(params, model, grid_size)

evaluation.fourier_spectrum_plot(u)

#%% scoring for robustness and sharpness with fourier spectrum



#%% CHANGING SHAPE?

a, b, c, d, e, w, Du, Dv = valid_params[478][:8]
f, g = models.linear(a, b, c, d, e, w)

initial_conditions = np.ones((400, 2)) /2
#initial_conditions = np.column_stack((np.full(400, 0.2), np.full(400, 0.8)))

shape = shapes.grid_check
n = 30000
u, v = discrete_turning.simulation(f, g, Du, Dv, n = n, initial_conditions=initial_conditions, shape=shape, show=True)

#%%
# adds the small grid into the big grid:
def add_grid_in(u, v):
    r = 20 
    inner_min = r // 4 # small grid conditions
    inner_max = 3 * r // 4

    x, y = np.meshgrid(np.arange(r), np.arange(r))
    coords = np.stack([x.ravel(), y.ravel()], axis=1)  # shape (400, 2)

    mask = (inner_min <= coords[:, 0]) & (coords[:, 0] < inner_max) & \
        (inner_min <= coords[:, 1]) & (coords[:, 1] < inner_max)

    u_array = np.ones(400)
    v_array = np.ones(400)

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

def reshape_output(u, v):
    u_reshape = u.reshape(20, 20)
    v_reshape = v.reshape(20, 20)
    u_reshape = u_reshape.T 
    v_reshape = v_reshape.T 
    u_reshape = u_reshape.flatten() 
    v_reshape = v_reshape.flatten() 
    return u_reshape, v_reshape

#%%
u_reshape, v_reshape = reshape_output(u, v)
initial_conditions = np.column_stack((u_reshape, v_reshape))
shape = fix_u_nodes(u, perc=95)
n = 20000
u_new, v_new = discrete_turning.simulation(f, g, Du, Dv, n = n, initial_conditions=initial_conditions, shape=shape, show=True)

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
u_newest, v_newest = discrete_turning.simulation(f, g, Du, Dv, n = n, initial_conditions=initial_conditions, shape=shape, show=True)


# plt.scatter(np.arange(len(u)), u)

#%% GENERAL TURING CHECK
a, b, c, d, e, w, Du, Dv = valid_params[278][:8]
f, g = models.linear(a, b, c, d, e, w)

parameters.check_for_instability(f, g, Du, Dv)
# evaluation.dispersion_relation_plot(f, g, Du, Dv)
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

