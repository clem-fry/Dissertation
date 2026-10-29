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

data = np.load('ref24.npz')
print(data.files)
# %%

kilobot_wavelengths = []
labels_kilo = []
all_powers = []
highest_power = []

file_names = ["ref10.npz", "ref11.npz", "ref13.npz", "ref14.npz", "ref16.npz", "ref17.npz"]
eigenvalues = [-7, -3, -9, -5, -12, -2]
diffusion_rates = [(1.16535, 47.04227), (2.719149, 109.7627), (0.9064,	36.587), (1.63148,	65.85763), (0.679787, 27.44068), (4.07872, 131.715267)]

for idx, name in enumerate(file_names):
    data = np.load(name)
    eigen = eigenvalues[idx]
    for i, (name, array) in enumerate(data.items()):
        if i > 9:  # Only use 10 samples
            break
        array_grid = array.reshape((10, 10))
        # plt.imshow(array_grid, cmap='bwr')
        # plt.axis('off')
        # plt.show()
        #_ = analysis.plot_wavelength_spectrum(array_grid, 
        
        wavelength = analysis.strongest_wavelength(array_grid, plot=False)
        wavelengths, power_flat = analysis.plot_wavelength_spectrum(array_grid, plot_spectrum = False, log_scale=False)
        all_powers.append(power_flat)
        kilobot_wavelengths.append(wavelength)
        highest_power.append(power_flat.max())
        labels_kilo.append((eigen, "Kilobot"))

cv = np.std(all_powers, ddof=1) / np.mean(all_powers)
highest_power_mean = np.array(highest_power).mean()

print("grid")
print(f"cv: {cv}")
print(f"highest_power_mean: {highest_power_mean}")

#%% GRAPH FOURIER TRANSFORM

connection = 2.1

kilobot_wavelengths = []
labels_kilo = []
all_powers = []
highest_power = []

file_names = ["ref10.npz", "ref11.npz", "ref13.npz", "ref14.npz", "ref16.npz", "ref17.npz"]
eigenvalues = [-7, -3, -9, -5, -12, -2]
diffusion_rates = [(1.16535, 47.04227), (2.719149, 109.7627), (0.9064,	36.587), (1.63148,	65.85763), (0.679787, 27.44068), (4.07872, 131.715267)]

for idx, name in enumerate(file_names):
    data = np.load(name)
    eigen = eigenvalues[idx]
    for i, (name, array) in enumerate(data.items()):
        if i > 9:  # Only use 10 samples
            break
        array_grid = array.reshape((10, 10))
        # plt.imshow(array_grid, cmap='bwr')
        # plt.axis('off')
        # plt.show()
        #_ = analysis.plot_wavelength_spectrum(array_grid, 
        
        adjacency_matrix, degree_matrix, valid_nodes = simulation.generate_nodes(threshold=connection, grid_size=10)

        L = adjacency_matrix - degree_matrix
        evals, power, dominant_lambda = analysis.graph_spectral_analysis(np.array(array), L, k_eigs=100, plot=False)

        all_powers.append(power)
        kilobot_wavelengths.append(dominant_lambda)
        highest_power.append(power.max())
        labels_kilo.append((eigen, "Kilobot"))

cv = np.std(all_powers, ddof=1) / np.mean(all_powers)
highest_power_mean = np.array(highest_power).mean()

print("grid")
print(f"cv: {cv}")
print(f"highest_power_mean: {highest_power_mean}")


# cv = np.float64(2.235908347557892) for powers
# mean = highest power np.float64(62878497577.86704)
#%% WAVELENGTH PLOT
data = []

for wave, (eigen, source) in zip(kilobot_wavelengths, labels_kilo):
    data.append({"Eigenvalue": eigen, "Wavelength": wave, "Source": source})

# for wave, (eigen, source) in zip(simulation_wavelengths, labels_sim):
#     data.append({"Eigenvalue": eigen, "Wavelength": wave, "Source": source})

df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="Eigenvalue", y="Wavelength", hue="Source", palette="Set2")
plt.xlabel("Dominant Turing mode $\Lambda_m$")
plt.ylabel("Strongest Wavelength $\ell$")
#plt.legend(title="Source")
plt.ylim(bottom=0)
plt.tight_layout()
plt.show()

#%% EIGENMODE PLOT
data = []

for wave, (eigen, source) in zip(kilobot_wavelengths, labels_kilo):
    data.append({"Eigenvalue": eigen, "Wavelength": wave, "Source": source})

# for wave, (eigen, source) in zip(simulation_wavelengths, labels_sim):
#     data.append({"Eigenvalue": eigen, "Wavelength": wave, "Source": source})

df = pd.DataFrame(data)

unique_eigen = sorted(df['Eigenvalue'].unique())
groups = [df[df['Eigenvalue']==ev]['Wavelength'] for ev in unique_eigen]

plt.figure(figsize=(8, 6))
plt.boxplot(groups, positions=unique_eigen)
plt.plot(unique_eigen, unique_eigen, '--k')
plt.title('12 connectivity')
plt.xlabel("Dominant Turing mode $\Lambda_m$")
plt.ylabel("GFT Eigenmode $\ell$")
#plt.legend(title="Source")
#plt.ylim(bottom=0)
plt.tight_layout()
plt.show()

# %%

df = []

for wave, (eigen, source) in zip(kilobot_wavelengths, labels_kilo):
    df.append({"Eigenvalue": eigen, "Wavelength": wave, "Source": source})

df = pd.DataFrame(df)

def model(x, A, B):
    return A / np.sqrt(-x) + B

print(df)
unique_eigen = sorted(df['Eigenvalue'].unique())

# Group wavelengths by eigenvalue
groups = [df[df['Eigenvalue'] == ev]['Wavelength'] for ev in unique_eigen]
y_data = np.array([np.median(group) for group in groups])

# 🔧 Scale factor to compress spacing
scale_factor = 0.5
scaled_positions = [ev * scale_factor for ev in unique_eigen]

# Prepare fit data
x_fit = np.linspace(min(unique_eigen), max(unique_eigen), 200)

params, _ = curve_fit(model, unique_eigen, y_data)
A_fit, B_fit = params
print(f"Fitted parameters: A = {A_fit:.3f}, B = {B_fit:.3f}")


#y_fit = np.where(x_fit < 0, 14.07 / np.sqrt(-x_fit) - 1.52, np.nan)
x_fit_scaled = x_fit * scale_factor
y_fit = model(x_fit, A_fit, B_fit)
plt.rcParams.update({'font.size': 20})

# Plot
plt.figure(figsize=(10, 6))

# Compressed boxplots
plt.boxplot(groups, positions=scaled_positions, widths=0.4, patch_artist=True,
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red'))

# Fit curve
plt.plot(x_fit_scaled, y_fit, 'b--', linewidth=2)

# Axis labeling
plt.xlabel("Dominant Turing Mode $\Lambda_m$")
plt.ylabel("Strongest Wavelength $\ell$")
plt.ylim(bottom=0)
plt.grid(True)
plt.tight_layout()

# Custom x-axis ticks with original eigenvalue labels
plt.xticks(scaled_positions, [f"{int(ev)}" for ev in unique_eigen], fontsize=20)
plt.yticks(fontsize=20)

# x_min, x_max = min(unique_eigen), max(unique_eigen)
# plt.xticks(np.arange(np.floor(x_min), np.ceil(x_max) + 1, 1))  # step=1 for all integers

plt.show()

#%% CALCULATE R SQUARED
from sklearn.metrics import r2_score  # For R²

predicted_wavelength = model(df['Eigenvalue'], A_fit, B_fit)


r_squared = r2_score(df['Wavelength'], predicted_wavelength)
print(f"R² = {r_squared:.4f}")

#%% CALCULATE VARIANCE

cv_per_box = [np.std(group, ddof=1) / np.mean(group) for group in groups]
print('average cv: ', np.mean(cv_per_box))

variances = [np.var(group, ddof=1) for group in groups]  # ddof=1 for sample variance

# Compute average variance
cv = np.std(variances, ddof=1) / np.mean(variances) 

print("Variances:", variances)
print("Average variance:", cv)
# cv Average variance: 0.7393254231435022 for boxes
# %% LOG SCALE
unique_eigen = sorted(df['Eigenvalue'].unique())
assert all(ev < 0 for ev in unique_eigen), "All eigenvalues must be negative for log transformation."

# Log-scaled positions for plotting
log_scaled_positions = [-0.5*np.log(-ev) for ev in unique_eigen]

# Group wavelengths by eigenvalue
groups = [df[df['Eigenvalue'] == ev]['Wavelength'] for ev in unique_eigen]

# Prepare theoretical fit curve
x_fit = np.linspace(min(unique_eigen), max(unique_eigen), 200)
y_fit = np.where(x_fit < 0, 14.07 / np.sqrt(-x_fit) - 1.52, np.nan)
x_fit_log_scaled = [-0.5*np.log(-x) if x < 0 else np.nan for x in x_fit]

# Plotting
plt.figure(figsize=(10, 6))

# Compressed boxplots
plt.boxplot(groups, positions=log_scaled_positions, widths=0.1, patch_artist=True,
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red'))

# Theoretical fit curve
plt.plot(x_fit_log_scaled, y_fit, 'b--', linewidth=2, label="Theoretical Fit")
padding = 0.1  # Adjust if needed
plt.xlim(min(log_scaled_positions) - padding, max(log_scaled_positions) + padding)
# Axis labels and grid
plt.xlabel("Dominant Eigenvalue", fontsize=16)
plt.ylabel("Strongest Wavelength", fontsize=16)
plt.ylim(bottom=0)
plt.grid(True)
plt.tight_layout()

# Custom x-axis ticks using original eigenvalue labels
plt.xticks(log_scaled_positions, [f"{ev:.2f}" for ev in unique_eigen], fontsize=14)
plt.yticks(fontsize=14)

# Show plot
plt.show()
#%% LOGLOG (NOT LINEAR)

from scipy.stats import linregress

x_fit = np.linspace(min(unique_eigen), max(unique_eigen), 200)
y_fit = np.where(x_fit < 0, 14.07 / np.sqrt(-x_fit) - 1.52, np.nan)

# Apply transforms
x_trans = -0.5 * np.log(x_fit)
y_trans = np.log(y_fit)


# Prepare data for boxplots (group by original eigenvalues)
scaled_positions = -0.5 * np.log(-np.array(unique_eigen))

plt.figure(figsize=(10, 6))

# Boxplots with transformed y
plt.boxplot(groups, positions=scaled_positions, widths=0.1, patch_artist=True,
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red'))

# Plot the linear fit line over the range of scaled_positions
x_fit = np.linspace(min(scaled_positions), max(scaled_positions), 200)
plt.plot(x_fit, y_fit, 'r--', linewidth=2, label='Linear Fit')
padding = 0.1  # Adjust if needed
plt.xlim(min(scaled_positions) - padding, max(scaled_positions) + padding)
# Axis labels with math notation
plt.xlabel(r"$-0.5 \times \log(-\mathrm{Eigenvalue})$")
plt.ylabel(r"$\log(\mathrm{Strongest\ Wavelength})$")
plt.grid(True)
plt.tight_layout()

# Custom x-axis ticks with original eigenvalues labels (rounded)
plt.xticks(scaled_positions, [f"{int(ev)}" for ev in unique_eigen], fontsize=20)
plt.yticks(fontsize=20)
plt.show()

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# -- Assume df and groups already defined as in your snippet --

# 1. Define model: y = A / sqrt(x) + B
def model_func(x, A, B):
    return A / np.sqrt(-x) + B

def model_func(x, A, B):
    return A * x + B

df = []

for wave, (eigen, source) in zip(kilobot_wavelengths, labels_kilo):
    df.append({"Eigenvalue": eigen, "Wavelength": wave, "Source": source})

df = pd.DataFrame(df)

unique_eigen = sorted(df['Eigenvalue'].unique())

# Group wavelengths by eigenvalue
groups = [df[df['Eigenvalue'] == ev]['Wavelength'] for ev in unique_eigen]

# 2. Prepare data for fitting
x_data = np.array([-12, -9, -7, -5, -3, -2])  # x = index of group (0, 1, 2, ...)
y_data = np.array([np.median(group) for group in groups])  # median of each group

# Avoid x = 0 since sqrt(0) is problematic
x_fit_data = x_data
y_fit_data = y_data

# 3. Fit the model
params, _ = curve_fit(model_func, x_fit_data, y_fit_data, p0=[10, 1])  # initial guess

A_fit, B_fit = params
print(f"Fitted parameters: A = {A_fit:.3f}, B = {B_fit:.3f}")

# 4. Create smooth fit curve over x range
x_smooth = np.linspace(-12.5, -1.5, 200)
y_smooth = model_func(x_smooth, A_fit, B_fit)

#y_smooth = 14.07 / np.sqrt(-x_smooth) - 1.5

# 5. Plot boxplot
plt.figure(figsize=(8, 6))
plt.boxplot(groups, positions=x_data, widths=0.6, patch_artist=True,
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red'))

# 6. Plot the fitted curve
plt.plot(x_smooth, y_smooth, 'b--', linewidth=2, label=r'$y = \frac{A}{\sqrt{x}} + B$')

# 7. Axis and formatting
plt.xticks(x_data, unique_eigen)
plt.xlabel("Dominant Eigenvalue")
plt.ylabel("Strongest Frequency")
plt.ylim(bottom=0)
plt.grid()
plt.rcParams.update({'font.size': 22})
#plt.legend()
plt.tight_layout()
plt.show()


# %%
# Convert to array if needed
strongest_wavelengths = np.array(strongest_wavelengths)

# Set corresponding x positions (same length, all -4)
x_strong = np.full_like(strongest_wavelengths, fill_value=-4, dtype=float)

# Plot original content
plt.errorbar(x, y, yerr=std_wavelengths, fmt='o', label='Measured from DCT', ecolor='red', capsize=5)
plt.plot(x_fit[:-10], y_fit[:-10], 'g--', label=f'Fit: {a_fit:.2f}/sqrt(-x) + {b_fit:.2f}')

# 🔹 Add strongest wavelength markers at x = -4
plt.plot(
    x_strong, 
    strongest_wavelengths, 
    'x', 
    color='purple', 
    markersize=8, 
    label='Strongest Wavelengths (x=-4)'
)

# Final styling
plt.xlabel('Dominant Eigenvalue')
plt.ylabel('Wavelength (nodes)')
plt.grid(True)
plt.legend()
plt.show()

#%% connectivity processing:

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Create a color list in order from 4 to 10 (index corresponds to color number)
colors = [
    'none',       # index 0 placeholder (not used)
    'none',       # index 1 placeholder (not used)
    'none',       # index 2 placeholder (not used)
    'none',       # index 3 placeholder (not used)
    'blue',       # 4
    'purple',     # 5
    'white',      # 6
    'cyan',       # 7
    'green',      # 8
    'yellow',     # 9
    'red'         # 10
]

# Create a colormap from the above list (only from 4 to 10)
cmap = mcolors.ListedColormap(colors[4:11])

# Normalize to match the values 4-10 to colormap indices 0-6
norm = mcolors.BoundaryNorm(boundaries=np.arange(3.5, 11.5), ncolors=cmap.N)

arr_num = np.array([
    [4, 4, 5, 5, 6, 5, 6, 6, 4, 0, 4],
    [5, 5, 10, 6, 5, 7, 0, 0, 0, 5, 0],
    [0, 7, 10, 8, 7, 6, 6, 7, 5, 6, 0],
    [5, 5, 0, 5, 6, 5, 8, 8, 7, 0, 4],
    [6, 7, 0, 4, 0, 0, 8, 8, 0, 4, 0],
    [0, 0, 0, 5, 6, 0, 10, 7, 7, 4, 0],
    [10, 4, 5, 6, 8, 5, 5, 0, 0, 4, 0],
    [0, 6, 8, 7, 5, 0, 10, 6, 5, 0, 4],
    [5, 5, 5, 6, 0, 4, 5, 0, 6, 7, 0],
    [0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0],
])

plt.figure(figsize=(6,6))
plt.imshow(arr_num, cmap=cmap, norm=norm)
plt.colorbar(ticks=np.arange(4, 11), label='Color Code')
plt.title('Color-coded 10x10 Array Visualization')
plt.axis('off')
plt.show()

plt.figure(figsize=(6,6))
plt.imshow(arr_num, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Value')
plt.axis('off')  # Optional: hide axis ticks and labels
plt.show()
# %%
import numpy as np
import matplotlib.pyplot as plt

arr_num = np.array([
    [4, 4, 0, 5, 5, 6, 5, 6, 6, 4, 0, 4],
    [5, 5, 10, 6, 5, 7, 6, 0, 0, 4, 0, 0],
    [0, 7, 7, 8, 5, 7, 6, 7, 5, 5, 5, 0],
    [5, 5, 0, 5, 6, 10, 0, 6, 8, 7, 6, 4],
    [4, 7, 0, 4, 0, 6, 5, 8, 6, 0, 4, 0],
    [6, 4, 0, 5, 5, 8, 5, 8, 8, 0, 0, 0],
    [0, 0, 0, 6, 6, 8, 5, 10, 7, 7, 4, 0],
    [10, 4, 5, 8, 5, 8, 0, 5, 0, 4, 4, 0],
    [0, 6, 8, 0, 7, 4, 5, 10, 6, 5, 0, 4],
    [5, 5, 5, 6, 0, 0, 4, 5, 6, 5, 7, 0],
    [0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0],
])

# Mask values less than 3
masked_arr = np.ma.masked_less(arr_num, 2)

# Use viridis and set masked values (bad) to black
cmap = plt.cm.viridis.copy()
cmap.set_bad(color='white')

boundaries = np.arange(2.5, 11.5, 1)  # Bin edges: 2.5–3.5 for "3", etc.
norm = mcolors.BoundaryNorm(boundaries, ncolors=plt.cm.viridis.N)

plt.figure(figsize=(6, 6))
img = plt.imshow(masked_arr, cmap=cmap, norm=norm, interpolation='nearest')
cbar = plt.colorbar(img, ticks=np.arange(3, 11))
cbar.set_label('Number of Neighbours')

plt.axis('off')
plt.show()

# %%
