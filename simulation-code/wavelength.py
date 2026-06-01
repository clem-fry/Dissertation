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
# remove any oscillations!!
valid_params = [p[:5] for p in valid_params if p[-1] == True]

# -3, 10, 0.1 is good
# smaller eig bigger wavelength
desired_eig = -2 # exact
desired_growth = 30 # exact
desired_width = 0.1 # kind of

print('Wavelength: ', (2*np.pi/np.sqrt(-desired_eig)))

optimal = parameters.find_suitable_params(desired_eig, desired_growth, desired_width, valid_params)
option = 0
a, b, c, d = optimal[option][0]
sigma = optimal[option][1]
Du = optimal[option][2]
root_width = optimal[option][3]
Dv = sigma * Du

#%%
a, b, c, d = 49.373, -82.998, 61.178, -93.445
Du, Dv = 2.0394, 82.322
J = parameters.non_spacial_jacobian(a, b, c, d)
f, g = models.linear(a, b, 0, c, d, 0)

params = a, b, 0, c, d, 0, Du, Dv
grid_size = 10

#%%
lambda_alpha, eigvals = plots.dispersion_relation_plot(params, grid_size=grid_size, plot=False,  shape = shapes.grid_check, model = models.linear)

f, g = models.linear(a, b, 0, c, d, 0)

u0, v0 = parameters.find_steady_state(f, g, initial_guess=(0.0, 0.0))
n = 2000
grid_size = 10

initial = models.initial_fill(grid_size, fill = 0, pert=True)

u_vector, v_vector, (u_log, v_log) = simulation.simulation(f, g, Du, Dv, grid_size = grid_size, n = n, initial_conditions = initial, shape = shapes.grid_check)

x_axis = np.arange(0, n, 1)
plt.plot(x_axis, u_log.T)
plt.xlabel('time (s)')
plt.title(f"{grid_size} nodes fully connected with params ")
plt.ylabel('u values')
plt.grid(True)
plt.show()

#%%
lambda_alpha, eigvals = plots.dispersion_relation_plot(params, grid_size=grid_size, plot=True,  shape = shapes.grid_check, model = models.linear)
u_grid = u_vector.reshape((grid_size, grid_size))

params = a, b, 0, c, d, 0, Du, Dv
eigvals, eigvecs, lambda_alpha = analysis.linear_stability(params, grid_size = grid_size)

#%% PLOT LINE GRAPHS

plt.plot(np.arange(0, len(v_log.T[:1200]), 1) * 0.85, v_log.T[:1200])
plt.rcParams.update({'axes.labelsize': 22})

# Axis labels with larger font
plt.xlabel('time (s)', fontsize=22)
plt.ylabel('v values', fontsize=22)

# Format tick labels in scientific notation
ax = plt.gca()
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax.xaxis.get_offset_text().set_fontsize(22)
ax.yaxis.get_offset_text().set_fontsize(22)
# # Optional: fine-tune tick font size
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

plt.grid(True)
plt.tight_layout()
plt.show()

#%% plot array?
data = u_log[:, -1].reshape((grid_size, grid_size))
fig, ax = plt.subplots()

# Plot with pixelated look
im = ax.imshow(data, cmap='bwr', origin='lower', interpolation='nearest', aspect='equal')

# Tidy plot
ax.set_axis_off()
plt.show()
#%%

wavelengths = np.array([2 * np.pi / np.sqrt(-eig) for eig in eigvals if eig < -0.001])

#%%


grid_size = 20
kx = np.fft.fftfreq(grid_size, d=1)  # cycles per grid unit
ky = np.fft.fftfreq(grid_size, d=1)
kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')

# True FFT wavenumber magnitudes
k_magnitude = np.sqrt(kx_grid**2 + ky_grid**2)

# Avoid divide-by-zero
k_nonzero = k_magnitude[k_magnitude > 0]

# Corresponding wavelengths
wavelengths_grid = 1 / k_nonzero

# Round and remove duplicates
wavelengths_unique = np.unique(np.round(wavelengths_grid, 2))
print("Feasible wavelengths (rounded):", wavelengths_unique)

eig_wavelengths = (2*np.pi/wavelengths_unique)**2
#%%
#%%
def possible_wavelengths(grid_size=10):
    freqs = np.fft.fftshift(np.fft.fftfreq(grid_size, d=1))
    fx, fy = np.meshgrid(freqs, freqs, indexing='ij')
    k_magnitude = np.sqrt(fx**2 + fy**2)
    
    # Remove zero frequency (DC)
    valid = k_magnitude > 0
    wavelengths = 1 / k_magnitude[valid]
    
    # Unique and sorted
    unique_wavelengths = np.unique(np.round(wavelengths, 4))
    return unique_wavelengths
#%%
from scipy.fftpack import dctn, fftshift

def strongest_wavelength(u_grid, plot=False):
    grid_size = u_grid.shape[0]  # assume square grid (N x N)

    # 1. Compute 2D DCT (Type-II) instead of FFT
    pattern_dct = dctn(u_grid, norm='ortho')
    power_spectrum = np.abs(pattern_dct) ** 2

    # 2. Remove DC component (zero frequency)
    power_spectrum[0:2, 0:2] = 0

    # 3. Generate DCT frequency coordinates
    # DCT frequencies are like FFT but only non-negative
    freqs = np.arange(grid_size) / (2 * grid_size)
    fx, fy = np.meshgrid(freqs, freqs, indexing='ij')

    # 4. Find the frequency with maximum power
    max_idx = np.unravel_index(np.argmax(power_spectrum), power_spectrum.shape)
    kx = fx[max_idx]
    ky = fy[max_idx]
    k_mag = np.sqrt(kx**2 + ky**2)

    # 5. Convert frequency to wavelength
    if k_mag > 0:
        wavelength = 1 / k_mag
        wavelength = round(wavelength, 4)
        print(f"Strongest detected wavelength (Neumann/DCT): {wavelength} grid units")
    else:
        print("No significant frequency components found.")
        wavelength = None

    # 6. Optional plot
    if plot:
        plt.figure(figsize=(6, 6))
        plt.imshow(np.log1p(power_spectrum), cmap='inferno', origin='lower')
        plt.title('2D DCT Power Spectrum (log scale)')
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    return wavelength


#%%
grid_size = 10

eig_wavelengths = possible_wavelengths(grid_size)
eigenvalues = np.linspace(-12, -0, 30) #-(2*np.pi/eig_wavelengths) ** 2

#%%
#eigenvalues, eigvecs, lambda_alpha = analysis.linear_stability(params, grid_size=10, shape = shapes.grid_check, model=models.linear)
# log_vals = np.logspace(-2, 0, 20)
# eigenvalues = -0.1 - (log_vals - log_vals.min()) * (grid_size - 0.1) / (log_vals.max() - log_vals.min())
eigenvalues = np.arange(-12, -0, 0.5)

wavelengths = []
std_array = []

a, b, c, d = 49.373, -82.998, 61.178, -93.445
J = parameters.non_spacial_jacobian(a, b, c, d)
f, g = models.linear(a, b, 0, c, d, 0)

# changing node number
#num_nodes = np.arange(5, 20, 3)
#threshold_range = np.array([1.1, 1.5, 2.1, 2.9, 3.1, 4.3, 5])
grid_size = 10
# eig = -4
n = 1800
desired_growth = 30 # exact
desired_width = 0.1 # kind of
initial = models.initial_fill(grid_size, fill = 0.5, pert=True)
sigma = parameters.find_sigma_for_growth(J, desired_growth)
# Du = parameters.epsilon_for_eigenvalue(J, eig, sigma)
# Dv = sigma * Du
#lambda_alpha, eigvals = plots.dispersion_relation_plot(params, continuous = True, grid_size=20, shape = shapes.grid_check, model = models.linear)
# optimal = parameters.find_suitable_params(eig, desired_growth, desired_width, valid_params)
# option = 0
# a, b, c, d = optimal[option][0]
# sigma = optimal[option][1]
# Du = optimal[option][2]
# root_width = optimal[option][3]
# Dv = sigma * Du
params = a, b, 0, c, d, 0, Du, Dv
connectivities = [1.1, 1.9, 2.1]

all_wavelengths = []
all_powers = []

#%%

for (c, connection) in enumerate(connectivities):
    wavelengths = []
    powers = []
    for eig in eigenvalues:
        if eig > -3:
            dt = 0.0001
        elif eig > -2:
            dt = 0.00001
        else:
            dt = 0.001
    #for thresh in threshold_range:
        repeat_array = []
        repeat_std_array = []
        powers_eig = []
        print(eig)
        for repeat in range(50):
            print(repeat)
            #grid_size = num
            Du = parameters.epsilon_for_eigenvalue(J, eig, sigma)
            Dv = sigma * Du # choose diffusion parameters based off ideal peak - continuous
            initial = models.initial_fill(grid_size, fill = 0, pert=True)
            u_vector, v_vector, (u_log, v_log) = simulation.simulation(f, g, Du, Dv, dt = dt, n = 5000, grid_size = grid_size, \
                                                                    initial_conditions = initial, \
                                                                    shape = shapes.grid_check, \
                                                                    show=False, threshold=connection)
            u_grid = u_vector.reshape((grid_size, grid_size))
            #wavelength = analysis.strongest_wavelength(u_grid, plot=False)

            adjacency_matrix, degree_matrix, valid_nodes = simulation.generate_nodes(threshold=connection, shape = shapes.grid_check, grid_size=grid_size)

            L = adjacency_matrix - degree_matrix
            evals, power, dominant_lambda = analysis.graph_spectral_analysis(u_vector, L, k_eigs=100, plot=False)

            _, power_flat = analysis.plot_wavelength_spectrum(u_grid, plot_spectrum = False, log_scale=False)
            #powers_eig.append(power_flat) # len 50
            repeat_array.append(dominant_lambda)

        # powers.append(powers_eig) # len 24 eigenvalues
        # in this case only len 3
        wavelengths.append(repeat_array)

    # once per connection
    # all_powers[c][21] = powers
    all_wavelengths.append(wavelengths)
    #all_wavelengths[c] = wavelengths

    
#%% PLOT ALL


# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the fitting function
def fit_func(x, A, B):
    return A / np.sqrt(-x) + B

mean_vals = np.mean(all_wavelengths, axis=2)  # shape: (3, 24)
std_vals = np.std(all_wavelengths, axis=2)    # shape: (3, 24)
x = eigenvalues

legend_labels = ['4', '8', '12']
colors = ['tab:blue', 'tab:orange', 'tab:green']

plt.figure()

for i in range(3):  # Assuming 3 curves
    # Plot error bars
    plt.errorbar(x, mean_vals[i], yerr=std_vals[i], label=legend_labels[i],
                 capsize=3, fmt='o', color=colors[i])

    if i == 1:
        x_line = x[:-1]
        y_line = mean_vals[i][:-1]
    elif i == 2:
        x_line = x[:-4]
        y_line = mean_vals[i][:-4]
    else:
        x_line = x
        y_line = mean_vals[i]

    popt, _ = curve_fit(fit_func, x_line, y_line, p0=(1, 1))
    A, B = popt

    # Plot the fitted curve
    x_fit = np.linspace(min(x_line), max(x_line), 300)
    y_fit = fit_func(x_fit, A, B)
    plt.plot(x_fit, y_fit, '--', color=colors[i])

# Styling
plt.rcParams.update({'font.size': 15})
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Dominant Turing Mode $\Lambda_m$', fontsize=15)
plt.ylabel('Strongest Wavelength $\ell$', fontsize=15)
plt.grid(True)
plt.legend(title="Connections", fontsize=15, title_fontsize=15)
plt.tight_layout()
plt.show()

# A: 6.537063078583903, B: -0.48404362285756874
# A: 11.35644947957741, B: -1.2673784054802593
# A: 22.176479363032175, B: -4.018636515087657

#%% DISPERSION DIAGRAM PLOT
a, b, c, d = 49.373, -82.998, 61.178, -93.445
J = parameters.non_spacial_jacobian(a, b, c, d)
f, g = models.linear(a, b, 0, c, d, 0)
grid_size = 10

eig = -3
desired_growth = 30 # exact
sigma = parameters.find_sigma_for_growth(J, desired_growth)
Du = parameters.epsilon_for_eigenvalue(J, eig, sigma)
Dv = sigma * Du
params = (a, b, 0, c, d, 0, Du, Dv)
eigvals, eigvecs, lambda_alpha = analysis.linear_stability(params, grid_size=10, shape = shapes.grid_check, model=models.linear, threshold=1.9)
plt.rcParams.update({'font.size': 15})

fig, ax = plt.subplots()
ax.plot(eigvals, lambda_alpha.real, 'o')
ax.axhline(0, color='gray', linestyle='--')
ax.set_xlabel('Laplacian Eigenvalue $\Lambda$', fontsize = 15)
ax.set_ylabel('Growth rate Re(λ)', fontsize = 15)
ax.grid(True)

# Add annotation
ax.annotate(
    'Turing mode $\lambda_m$, $\Lambda_m$',
    xy=(eig, desired_growth),               # Location of the annotation
    xytext=(eig-5, desired_growth - 15),           # Location of the text
    arrowprops = dict(facecolor ='red')
)
#%%

combined = np.column_stack((eigenvalues, wavelengths))

# Save to text file
np.savetxt("eigen_wavelengths_10_oneparam_twopointone.txt", combined, delimiter="\t", fmt="%.6f",
           header="Eigenvalue\tW1\tW2\tW3\tW4\tW5\tW6\tW7\tW8\tW9\tW10", comments='')
#%% from import
data = np.loadtxt("eigen_wavelengths_10_connectivity.txt", delimiter="\t", skiprows=1)

# Separate eigenvalues and wavelengths
eigenvalues = data[:, 0]       # First column
wavelengths = data[:, 1:]      # Remaining columns

print("Eigenvalues shape:", eigenvalues.shape)  # (N,)
print("Wavelengths shape:", wavelengths.shape)

#%%
mean_wavelengths = np.mean(wavelengths, axis=1)
std_wavelengths = np.std(wavelengths, axis=1)

from scipy.optimize import curve_fit

# Your data
x = np.array(eigenvalues)
y = np.array(mean_wavelengths)

# Define the model: inverse sqrt with offset
def model(x, a, b):
    return a / np.sqrt(-x) + b

# Fit the model
popt, pcov = curve_fit(model, x[0:-8], y[0:-8])
a_fit, b_fit = popt

# Generate smooth curve for plotting
x_fit = np.linspace(min(x), max(x), 300)
y_fit = model(x_fit, a_fit, b_fit)

#%%Plot
plt.rcParams.update({'font.size': 15})

plt.errorbar(x, y, yerr=std_wavelengths, fmt='o', label='Measured from FFT', ecolor='red', capsize=5)
plt.plot(x_fit[:-75], y_fit[:-75], 'g--', label=f'Fit: {a_fit:.2f}/sqrt(-x) + {b_fit:.2f}')
plt.xlabel('Dominant Eigenvalue')
plt.ylabel('Strongest Wavelength')
plt.grid(True)
plt.legend()
plt.show()
# %%
from scipy.ndimage import distance_transform_edt
from scipy.signal import correlate2d
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

def autocorrelation(u_grid, show=False):
    def radial_profile(data, center):
        y, x = np.indices(data.shape)
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        r = r.astype(int)
        tbin = np.bincount(r.ravel(), data.ravel())
        nr = np.bincount(r.ravel())
        radial_prof = tbin / np.maximum(nr, 1)
        return radial_prof
    autocorr = correlate2d(u_grid, u_grid, mode='full', boundary='wrap')
    autocorr = autocorr / np.max(autocorr)
    center = np.array(autocorr.shape) // 2
    profile = radial_profile(autocorr, center)
    mid = np.array(autocorr.shape) // 2

    peaks, _ = find_peaks(profile)

    # Optionally skip the first peak if too close to center
    valid_peaks = peaks[peaks > 3]
    if show:
        plt.figure(figsize=(6, 6))
        plt.imshow(autocorr, cmap='inferno', origin='lower')
        plt.scatter(*mid[::-1], color='white', marker='x', label='Origin')
        plt.title('2D Autocorrelation')
        plt.colorbar()
        plt.legend()
        plt.show()

        plt.plot(profile)
        plt.plot(valid_peaks, profile[valid_peaks], "rx", label='Detected Peaks')
        plt.title('Radial Autocorrelation Profile')
        plt.xlabel('Radius (grid units)')
        plt.ylabel('Autocorrelation')
        plt.grid(True)
        plt.show()

    # Calculate differences between successive peaks
    if len(valid_peaks) >= 2:
        peak_spacings = np.diff(valid_peaks)
        avg_wavelength = np.mean(peak_spacings)
        std_dev = np.std(peak_spacings)
        print(f"Average wavelength: {avg_wavelength:.2f} ± {std_dev:.2f}")
        return(avg_wavelength, std_dev)
    else:
        print("Not enough peaks to compute average wavelength.")
        return(None, None)

#%%

params = a, b, 0, c, d, 0, Du, Dv
eigvals, eigvecs, lambda_alpha = analysis.linear_stability(params, grid_size = grid_size)

all_nodes, border_nodes = simulation.shape_nodes(shapes.grid_check, grid_size, initial)
valid_nodes = [node for node in all_nodes.flatten() if node is not None]
adjacency = simulation.compute_adjacency_matrix(valid_nodes)

degrees = [len(node.neighbours) for node in valid_nodes]
sorted_indices = np.argsort(degrees)

sorted_degrees = np.array(degrees)[sorted_indices]
u_sorted = u_vector[sorted_indices]

plt.plot(sorted_degrees, u_vector, 'x')

#%%
from scipy.fftpack import dctn

def plot_wavelength_spectrum(u_grid, plot_spectrum=True, log_scale=True):
    grid_size = u_grid.shape[0]
    
    # 1. Compute 2D DCT (Type-II), respects Neumann BCs
    pattern_dct = dctn(u_grid, norm='ortho')
    power_spectrum = np.abs(pattern_dct) ** 2

    # 2. Remove DC component (small region near [0, 0])
    power_spectrum[0:3, 0:3] = 0

    # 3. Generate DCT frequency coordinates
    freqs = np.arange(grid_size) / (2 * grid_size)  # normalized frequency [0, 0.5)
    fx, fy = np.meshgrid(freqs, freqs, indexing='ij')

    # 4. Flatten to 1D arrays
    fx_flat = fx.ravel()
    fy_flat = fy.ravel()
    power_flat = power_spectrum.ravel()

    # 5. Compute frequency magnitude and wavelength
    k_mags = np.sqrt(fx_flat**2 + fy_flat**2)
    with np.errstate(divide='ignore'):
        wavelengths = 1 / k_mags

    # 6. Filter out infinite and zero frequency components
    valid = np.isfinite(wavelengths) & (k_mags > 0)
    wavelengths = wavelengths[valid]
    power_flat = power_flat[valid]

    # 7. Plot
    if plot_spectrum:
        plt.figure(figsize=(8, 4))
        plt.scatter(wavelengths, power_flat, s=10, alpha=0.7)
        if log_scale:
            plt.xscale('log')
            plt.yscale('log')
        plt.xlabel('Wavelength (grid units)')
        plt.ylabel('Power')
        plt.title('DCT-Based Wavelength Spectrum (Neumann BCs)')
        plt.grid(True, which='both', ls='--', lw=0.5)
        plt.tight_layout()
        plt.show()

    return wavelengths, power_flat


#%%


print(possible_wavelengths(10))
# %%

# %%
# %%

import networkx as nx
from scipy.sparse import csgraph
from scipy.linalg import eigh  # dense eigen-decomposition

def gft_power_spectrum(u_grid, eigvals, eigvecs, plot=True):
    N, M = u_grid.shape
    num_nodes = N * M

    u_flat = u_grid.flatten()

    # Step 5: GFT = projection onto Laplacian eigenvectors
    u_hat = eigvecs.T @ u_flat  # Shape: (num_nodes,)

    # Step 6: Power spectrum
    power_spectrum = np.abs(u_hat)**2

    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(eigvals, power_spectrum, marker='o', linestyle='-')
        plt.xlabel('Graph Frequency (Laplacian Eigenvalue)')
        plt.ylabel('Power')
        plt.title('Graph Fourier Power Spectrum')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return eigvals, power_spectrum


# %% EIGENVALUES FOR DIFFERENT LAPLACIAN SETUPS

lambda_alpha_four, eigvals_four = plots.dispersion_relation_plot(params, grid_size=grid_size, plot=False,  shape = shapes.grid_check, model = models.linear, threshold = 2.1)

lambda_alpha_eight, eigvals_eight = plots.dispersion_relation_plot(params, grid_size=grid_size, plot=False,  shape = shapes.grid_check, model = models.linear, threshold = 2.3)

lambda_alpha_twelve, eigvals_twelve = plots.dispersion_relation_plot(params, grid_size=grid_size, plot=False,  shape = shapes.grid_check, model = models.linear, threshold = 3.1)

plt.figure(figsize=(8, 5))

plt.plot(eigvals_four, lambda_alpha_four,'X',
         label='4-neighbour')

plt.plot(eigvals_eight, lambda_alpha_eight, 'o',
         label='8-neighbour')

plt.plot(eigvals_twelve, lambda_alpha_twelve, 'x',
         label='12-neighbour')

plt.axhline(0, color='k', ls='--', alpha=0.5)

plt.xlabel('Laplacian eigenvalue')
plt.ylabel('Growth rate Re(λ)')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.show()
