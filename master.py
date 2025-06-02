#%%
import discrete_turning
import models
import evaluation
import numpy as np
import matplotlib.pyplot as plt
import importlib
import parameters
#%%
importlib.reload(discrete_turning)
importlib.reload(models)
importlib.reload(evaluation)
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
T = 15.0 # total time
dt = 0.001 # time step
n = int(T/dt) # number of iterations
Du, Dv = 0.5, 10

u, v = discrete_turning.simulation(f, g, Du, Dv, n=n, dt=dt, show=True)
evaluation.turing_check(f, g, Du, Dv)

# if Re(lamda(0)) > 0 the system is unstable without diffusion (bad)
# if Re(lamda(k)) > 0 for some nonzero k, the system is unstable and can form a pattern
# ideally there should be a peak

#%%
# (a, b, c, d, e, w, Du, Dv, peak, peak_lamda, peak_k, unstable_band_width, area)

# import parameter information from txt file
import ast
import re

valid_params = []
with open('valid_params.txt', 'r') as file:
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

k_val = 1
num_selected = 10

indices = [i for i, param in enumerate(valid_params) if param[10] > 0]
peak_ks_positive = np.array([valid_params[i][10] for i in indices])
differences = np.abs(peak_ks_positive - k_val)
closest_indices = np.array(indices)[np.argsort(differences)[:num_selected]]

#%% PLOT PATTERN AND DISPERSION FOR DIFFERENT PARAMETERS

parameters.plot_dispersion_turing(valid_params, closest_indices[:5])

#%% SCORING SYSTEM?
selected_params = [valid_params[i] for i in closest_indices]
scored_params = parameters.score_params(selected_params)
top_indices = [p for _, p, _ in scored_params[0:5]]
top_indices = np.array(closest_indices[top_indices])

parameters.plot_dispersion_turing(valid_params, top_indices)

#%% TRY TO MINIMISE AROUND FUNCTION?
from scipy.signal import minimize

result = minimize(parameters.loss_function, (19.319, 5.589, -7.764, -4.133, 10.384, -11.089, 0.787, 8.588))

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

# %% BIFURCATION - NOT YET RELEVANT

# the measure of pattern strength isnt very sophisticated
a_vals = np.linspace(0, 1, 11)
b_vals = np.linspace(0, 1, 11)

model = models.gray_scott
evaluation.bifurcation(a_vals, b_vals, model, Du, Dv)