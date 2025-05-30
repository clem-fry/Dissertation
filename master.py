#%%
import discrete_turning
import models
import evaluation
import numpy as np
import matplotlib.pyplot as plt
import importlib

#%%
importlib.reload(discrete_turning)
importlib.reload(models)
importlib.reload(evaluation)
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

# %%
from scipy.signal import find_peaks
from scipy.optimize import minimize

# if Re(lamda(0)) > 0 the system is unstable without diffusion (bad)
# if Re(lamda(k)) > 0 for some nonzero k, the system is unstable and can form a pattern
# ideally there should be a peak

#%%


# (a, b, c, d, e, w, Du, Dv, peak, peak_lamda, peak_k, unstable_band_width, area)



#%%
# some k = 2.4 start becoming overly pixelated and more difficult
# could be worked on?
# k = 0.5 gives max two spots
# k = 0.8 gives approx 3 (1 across)
# k = 1 -> 5 (1.5 across)
# k = 1.5 -> 9 (3 across)
# k = 2 -> 13 (3.5 across?)

k_val = 2
indices = [i for i, param in enumerate(valid_params) if param[10] > 0]
peak_ks_positive = np.array([valid_params[i][10] for i in indices])
differences = np.abs(peak_ks_positive - k_val)
closest_indices = np.array(indices)[np.argsort(differences)[:5]]

#%%
top_indices = [p for _, p, _ in scored_params[0:5]]
top_indices = np.array(top_indices)

for i in closest_indices:
    a, b, c, d, e, w, Du, Dv = valid_params[i][0:8]
    f, g = models.linear(a, b, c, d, e, w)
    u, v = discrete_turning.simulation(f, g, Du, Dv, n=n, dt=dt, show=False, save=True, save_path=f"plots/simulation_plot_{i}.png")
    print(evaluation.turing_check(f, g, Du, Dv))
    _ = evaluation.dispersion_relation_plot(f, g, Du, Dv, plot=False, save=True, save_path=f"plots/dispersion_plot_{i}.png")

import matplotlib.image as mpimg

fig, axes = plt.subplots(5, 2, figsize=(12, 20))

for idx, i in enumerate(closest_indices):
    sim_img = mpimg.imread(f"plots/simulation_plot_{i}.png")
    disp_img = mpimg.imread(f"plots/dispersion_plot_{i}.png")
    
    axes[idx, 0].imshow(sim_img)
    axes[idx, 0].axis('off')
    axes[idx, 0].set_title(f'Simulation {i}')
    
    axes[idx, 1].imshow(disp_img)
    axes[idx, 1].axis('off')
    axes[idx, 1].set_title(f'Dispersion Relation {i}')

plt.tight_layout()
plt.show()

#%%
def loss_function(params):
    a, b, c, d, e, w, Du, Dv = params
    f, g = models.linear(a, b, c, d, e, w)
    if not np.any(evaluation.turing_check(f, g, Du, Dv)):
        return 10e10
    
    real_parts, k_vals = evaluation.dispersion_relation_plot(f, g, Du, Dv, plot=False)
    peak1, _ = find_peaks(real_parts[:, 0])
    peak2, _ = find_peaks(real_parts[:, 1])
    peaks = np.append(peak1, peak2)
    if len(peaks) == 0:
        return 10e10
    
    else:
        return -np.max(peaks)

result = minimize(loss_function, (19.319, 5.589, -7.764, -4.133, 10.384, -11.089, 0.787, 8.588))

#%%
a, b, c, d, e, w = 89.095, -19.115, 19.419, 17.184, 4.539, -10.605

Du, Dv =  0.798, 6.887

# e and w cause shift not change in pattern itself

#for d in params:
    # check for local stability

f, g = models.linear(a, b, c, d, e, w)

def check_for_instability(f, g, Du, Dv):
    if np.all(evaluation.turing_check(f, g, Du, Dv)):
        real_parts, k_vals = evaluation.dispersion_relation_plot(f, g, Du, Dv, plot=False)
        peak1, _ = find_peaks(real_parts[:, 0])
        peak2, _ = find_peaks(real_parts[:, 1])

        turing_flag = (
            (len(peak1) > 0 and np.any(real_parts[peak1, 0] > 0)) or
            (len(peak2) > 0 and np.any(real_parts[peak2, 1] > 0))
        )

        if turing_flag:
            print("Turing-like instability detected")
            print(a, b, c, d, e, w)
            _ = evaluation.dispersion_relation_plot(f, g, Du, Dv, plot=True)
            u, v = discrete_turning.simulation(f, g, Du, Dv, n=n, show=True)
                #break

        else:
            _ = evaluation.dispersion_relation_plot(f, g, Du, Dv, plot=True)
            #u, v = discrete_turning.simulation(f, g, Du, Dv, n=n, show=True)
            print('no peak')

    else:
        print('Not locally unstable')

check_for_instability(f, g, Du, Dv)
# evaluation.dispersion_relation_plot(f, g, Du, Dv)
# %%

# %%
from itertools import product
import seaborn as sns
import pandas as pd

a_vals = np.linspace(-5, 50, 100)
b_vals = np.linspace(-5, 50, 100)
d_vals = np.linspace(-5, 50, 100)
e_vals = np.linspace(-5, 50, 100)

valid_points = []

for a, b, d, e in product(a_vals, b_vals, d_vals, e_vals):
    f, g = models.linear(a, b, e, d)

    if np.all(evaluation.turing_check(f, g, Du, Dv)):
        valid_points.append((a, b, d, e))

print(f"Found {len(valid_points)} valid Turing parameter sets.")

points = np.array(valid_points)
plt.scatter(points[:, 0], points[:, 2], s=2, alpha=0.3)
plt.xlabel("a")
plt.ylabel("d")
plt.title("Turing Valid Region (a vs d)")
plt.grid(True)
plt.show()

df = pd.DataFrame(valid_points, columns=['a', 'b', 'd', 'e'])

# Plot pairwise scatter plots
# the diagonal is a histogram of that variable's distribution
sns.pairplot(df, plot_kws={'alpha': 0.3, 's': 10})
plt.suptitle("Turing Valid Region - Parameter Pairwise Scatter", y=1.02)
plt.show()

# %%

# the measure of pattern strength isnt very sophisticated
a_vals = np.linspace(0, 1, 11)
b_vals = np.linspace(0, 1, 11)

model = models.gray_scott
evaluation.bifurcation(a_vals, b_vals, model, Du, Dv)