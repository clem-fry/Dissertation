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
from scipy.optimize import curve_fit
import pandas as pd
import seaborn as sns

#%%
importlib.reload(simulation)
importlib.reload(models)
importlib.reload(plots)
importlib.reload(shapes)
importlib.reload(parameters)
importlib.reload(analysis)

#%%

kilobot_wavelengths = []
labels_kilo = []
all_powers = []
highest_power = []

file_names = ["ref21.npz", "ref20.npz", "ref19.npz"]
eigenvalues = [-7, -12, -3]
diffusion_rates = [(0.679787, 27.44068), (1.16535, 47.04227), (2.719149,	109.7627)]

for idx, name in enumerate(file_names):
    data = np.load(name)
    eigen = eigenvalues[idx]
    for i, (name, array) in enumerate(data.items()):
        masked_grid = np.array(array, dtype=float)
        zero_mask = masked_grid == 0
        one_mask = masked_grid == 1
        two_mask = masked_grid == 2
        mean_value = masked_grid[~zero_mask].mean()
        masked_grid[zero_mask] =  mean_value
        masked_grid[one_mask] =  -65504
        masked_grid[two_mask] =  65504

        wavelength = analysis.strongest_wavelength(masked_grid)
        wavelengths, power_flat = analysis.plot_wavelength_spectrum(masked_grid, plot_spectrum = False, log_scale=False)
        all_powers.append(power_flat)
        highest_power.append(power_flat.max())

        kilobot_wavelengths.append(wavelength)
        labels_kilo.append((eigen, "Kilobot"))

all_powers_flat = np.concatenate(all_powers)

cv = np.std(all_powers_flat, ddof=1) / np.mean(all_powers_flat)
highest_power_mean = np.array(highest_power).mean()

print("organic")
print(f"cv: {cv}")
print(f"highest_power_mean: {highest_power_mean}")

# cv = np.float64(2.9793901578765523) of powers
# mean highest power = np.float64(2.304701763873399)

#%% PLOT
data = []

for wave, (eigen, source) in zip(kilobot_wavelengths, labels_kilo):
    data.append({"Eigenvalue": eigen, "Wavelength": wave, "Source": source})

# for wave, (eigen, source) in zip(simulation_wavelengths, labels_sim):
#     data.append({"Eigenvalue": eigen, "Wavelength": wave, "Source": source})

def model(x, A, B):
    return A / np.sqrt(-x) + B

# def model(x, A, B):
#     return A *x  + B

df = pd.DataFrame(data)
unique_eigen = sorted(df['Eigenvalue'].unique())
groups = [df[df['Eigenvalue'] == ev]['Wavelength'] for ev in unique_eigen]

y_data = np.array([np.median(group) for group in groups])

params, _ = curve_fit(model, unique_eigen, y_data)
A_fit, B_fit = params
print(f"Fitted parameters: A = {A_fit:.3f}, B = {B_fit:.3f}")

x_smooth = np.linspace(-13, -2, 200)
y_smooth = model(x_smooth, A_fit, B_fit)

plt.figure(figsize=(8, 6))
plt.boxplot(groups, positions=unique_eigen, widths=0.6, patch_artist=True,
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red'))

# 6. Plot the fitted curve
plt.plot(x_smooth, y_smooth, 'b--', linewidth=2, label=r'$y = \frac{A}{\sqrt{x}} + B$')

# 7. Axis and formatting
plt.xticks(unique_eigen, unique_eigen)
plt.xlabel("Dominant Turing Mode $\Lambda_m$")
plt.ylabel("Strongest Wavelength $\ell$")
plt.ylim(0, 10)
plt.grid()
plt.rcParams.update({'font.size': 22})
#plt.legend()
plt.tight_layout()
plt.show()
#%% CALCULATE VARIANCE

cv_per_box = [np.std(group, ddof=1) / np.mean(group) for group in groups]
print('average cv: ', np.mean(cv_per_box))

variances = [np.var(group, ddof=1) for group in groups]  # ddof=1 for sample variance

# Compute average variance
cv = np.std(variances, ddof=1) / np.mean(variances) 

print("Variances:", variances)
print("Average variance:", cv)

# Average variance: 0.9676778387041268
#%%
label_array =np.array([
    [1, 2, 0, 1, 2, 1, 1, 2, 1, 2, 2],
    [2, 0, 1, 1, 1, 2, 1, 1, 2, 2, 0],
    [0, 2, 1, 2, 2, 2, 2, 1, 1, 2, 0],
    [0, 1, 1, 1, 1, 2, 2, 1, 2, 2, 0],
    [2, 0, 1, 1, 1, 2, 2, 1, 2, 1, 0],
    [0, 1, 2, 1, 2, 2, 2, 1, 1, 0, 2],
    [1, 2, 1, 2, 2, 2, 0, 2, 2, 2, 1],
    [1, 1, 2, 1, 2, 0, 1, 1, 0, 0, 0],
    [2, 0, 2, 2, 2, 0, 1, 1, 2, 1, 0],
    [0, 0, 0, 0, 2, 1, 1, 2, 2, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
])

masked_grid = np.array(label_array, dtype=float)
zero_mask = masked_grid == 0
mean_value = masked_grid[~zero_mask].mean()
masked_grid[zero_mask] = mean_value

_ = analysis.plot_wavelength_spectrum(masked_grid, log_scale=False)
wavelength = analysis.strongest_wavelength(masked_grid)
print(wavelength)

masked_grid = np.array(label_array, dtype=float)
zero_mask = masked_grid == 0
masked_grid[zero_mask] = 1.5

_ = analysis.plot_wavelength_spectrum(label_array, log_scale=False)
wavelength = analysis.strongest_wavelength(label_array)
print(wavelength)
# %%
np.array([
    [0, 0, 2, 1, 0, 2, 2, 1, 0, 2, 0, 2],
    [2, 0, 2, 1, 2, 1, 2, 0, 1, 1, 1, 0],
    [0, 2, 0, 0, 2, 1, 1, 2, 0, 0, 2, 0],
    [1, 1, 0, 2, 1, 1, 2, 1, 0, 2, 1, 0],
    [2, 2, 0, 2, 2, 0, 0, 2, 2, 1, 0, 0],
    [1, 1, 2, 2, 1, 2, 1, 1, 2, 0, 2, 0],
    [0, 2, 1, 1, 0, 1, 2, 0, 0, 1, 1, 0],
    [1, 0, 1, 2, 2, 2, 0, 0, 1, 2, 2, 0],
    [2, 2, 1, 1, 1, 2, 0, 2, 1, 1, 1, 0],
    [1, 0, 2, 1, 1, 1, 0, 1, 2, 2, 2, 0],
    [2, 0, 2, 2, 0, 2, 0, 1, 1, 2, 0, 0],
    [1, 1, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0]
])