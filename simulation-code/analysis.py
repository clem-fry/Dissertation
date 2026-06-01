#%%
import ast
import re
import simulation
import models
import plots
import numpy as np
import matplotlib.pyplot as plt
import importlib
import parameters
import shapes
from sklearn.preprocessing import MinMaxScaler
import scipy.sparse.linalg as spla
from scipy.fftpack import dctn, fftshift

#%%

importlib.reload(models)
importlib.reload(plots)
importlib.reload(shapes)
importlib.reload(parameters)

# functions for calculating quadratic of growth rate from laplacian eigenvalues

def linear_stability(params, grid_size = 20, shape = shapes.grid_check, model = models.linear, threshold=None, initial_conditions='rand'):
    a, b, c, d, e, w, Du, Dv = params
    #c, w = 0, 0
    f, g = model(a, b, c, d, e, w)

    # initial_conditions = np.ones((grid_size ** 2, 2)) /2

    # all_nodes, border_nodes = simulation.shape_nodes(shape, grid_size, initial_conditions)
    # valid_nodes = [node for node in all_nodes.flatten() if node is not None]

    # adjacency_matrix = simulation.compute_adjacency_matrix(valid_nodes, threshold=threshold)
    # degree_matrix = simulation.compute_degree_matrix(adjacency_matrix)

    adjacency_matrix, degree_matrix, valid_nodes = simulation.generate_nodes(threshold=threshold, shape = shape, grid_size=grid_size, initial_conditions = initial_conditions)

    L = adjacency_matrix - degree_matrix

    eigvals, eigvecs = np.linalg.eigh(L)

    lambda_alpha = growth_rate(params, eigvals) 
    return eigvals, eigvecs, lambda_alpha

def growth_rate(params, eigvals):
    lambda_alpha = np.zeros_like(eigvals, dtype = complex)
    a, b, c, d, e, w, Du, Dv = params
    for i, eig in enumerate(eigvals): # calculating growth rate lambda
        # growth rate and eigenvector are directly proportional
        trace_J = a + e
        det_J = a*e - b*d
        B = -(trace_J + (Du+Dv)*eig)
        C = Dv * Du * eig **2 + Du * (e + Dv/Du * a) * eig + det_J
        #lambda_alpha[i] = 0.5 * (a + e + (1+Dv/Du)*Du*eig + np.sqrt(4*b*d + (a-e + (1-Dv/Du) * Du * eig)**2 + 0j))
        discriminant = B**2 - 4*C
        discriminant = np.real_if_close(discriminant, tol=1e-10)

        lambda_alpha[i] = 0.5 * (-B + np.sqrt(discriminant + 0j))
    return lambda_alpha

def eigvalue_roots(params):
    a, b, c, d, e, w, Du, Dv = params
    det_J = a * e - b * d
    A = Dv
    B = e + a * Dv/Du
    C = det_J / Du

    root_pos = (-B + np.sqrt(B**2 - 4 * A * C))/(2*A)
    root_neg = (-B - np.sqrt(B**2 - 4 * A * C))/(2*A)
    return [root_pos, root_neg]


def calc_laplacian(shape, grid_size):
    initial_conditions = np.random.rand(grid_size*grid_size, 2)
    all_nodes, border_nodes = simulation.shape_nodes(shape, grid_size, initial_conditions)
    valid_nodes = [node for node in all_nodes.flatten() if node is not None]
    adjacency_matrix = simulation.compute_adjacency_matrix(valid_nodes)
    degree_matrix = simulation.compute_degree_matrix(adjacency_matrix)
    L = adjacency_matrix - degree_matrix
    return L

def plot_wavelength_spectrum(u_grid, plot_spectrum=True, log_scale=True, normalise=False):
    ny, nx = u_grid.shape  # Allow rectangular grids
    
    # 1. Compute 2D DCT (Type-II), respects Neumann BCs
    pattern_dct = dctn(u_grid, norm='ortho')
    power_spectrum = np.abs(pattern_dct) ** 2

    # 2. Remove DC component (small region near [0, 0])
    power_spectrum[0:2, 0:2] = 0

    # 3. Generate DCT frequency coordinates
    fx = np.arange(nx) / (2 * nx)  # x-frequencies
    fy = np.arange(ny) / (2 * ny)  # y-frequencies
    fx_grid, fy_grid = np.meshgrid(fx, fy, indexing='xy')

    # 4. Flatten to 1D arrays
    fx_flat = fx_grid.ravel()
    fy_flat = fy_grid.ravel()
    power_flat = power_spectrum.ravel()

    # 5. Compute frequency magnitude and wavelength
    k_mags = np.sqrt(fx_flat**2 + fy_flat**2)
    with np.errstate(divide='ignore'):
        wavelengths = 1 / k_mags

    # 6. Filter out infinite and zero frequency components
    valid = np.isfinite(wavelengths) & (k_mags > 0)
    wavelengths = wavelengths[valid]
    power_flat = power_flat[valid]

    if normalise:
        scaler = MinMaxScaler()
        power_flat = scaler.fit_transform(power_flat.flatten().reshape(-1, 1))

    # 7. Plot
    if plot_spectrum:
        plt.figure(figsize=(8, 4))
        plt.scatter(wavelengths, power_flat, s=10, alpha=0.7)
        if log_scale:
            plt.xscale('log')
            plt.yscale('log')
        plt.xlabel('Wavelength $\ell$ (grid units)')
        plt.ylabel('Power')
        plt.grid(True, which='both', ls='--', lw=0.5)
        plt.tight_layout()
        plt.show()

    return wavelengths, power_flat

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift, fftfreq
from sklearn.preprocessing import MinMaxScaler


def plot_wavelength_spectrum_fft(
    u_grid,
    plot_spectrum=True,
    log_scale=True,
    normalise=False,
):

    ny, nx = u_grid.shape

    # 1. Compute 2D FFT
    pattern_fft = fft2(u_grid)

    # 2. Power spectrum
    power_spectrum = np.abs(pattern_fft)**2

    # 3. Shift zero-frequency to center
    power_spectrum = fftshift(power_spectrum)

    # 4. Frequency coordinates
    fx = fftshift(fftfreq(nx))
    fy = fftshift(fftfreq(ny))

    fx_grid, fy_grid = np.meshgrid(fx, fy, indexing='xy')

    # 5. Flatten
    fx_flat = fx_grid.ravel()
    fy_flat = fy_grid.ravel()
    power_flat = power_spectrum.ravel()

    # 6. Frequency magnitude
    k_mags = np.sqrt(fx_flat**2 + fy_flat**2)

    # 7. Convert to wavelength
    with np.errstate(divide='ignore'):
        wavelengths = 1 / k_mags

    # 8. Remove DC mode
    valid = np.isfinite(wavelengths) & (k_mags > 0)

    wavelengths = wavelengths[valid]
    power_flat = power_flat[valid]

    # 9. Optional normalization
    if normalise:
        scaler = MinMaxScaler()
        power_flat = scaler.fit_transform(
            power_flat.reshape(-1, 1)
        ).flatten()

    # 10. Plot
    if plot_spectrum:

        plt.figure(figsize=(8,4))

        plt.scatter(
            wavelengths,
            power_flat,
            s=10,
            alpha=0.7
        )

        if log_scale:
            plt.xscale('log')
            plt.yscale('log')

        plt.xlabel(r'Wavelength $\ell$ (grid units)')
        plt.ylabel('Power')

        plt.grid(True, which='both', ls='--', lw=0.5)

        plt.tight_layout()
        plt.show()

    return wavelengths, power_flat

def graph_spectral_analysis(u, L, k_eigs=100, plot=False):
    """
    Graph spectral power analysis.
    Parameters
    ----------
    u : (N,) array
        Graph signal
    L : sparse matrix
        Graph Laplacian
    k_eigs : int
        Number of eigenmodes to calculate
    Returns
    -------
    results : dict
    """

    N = L.shape[0]

    # eigenmodes
    evals, evecs = spla.eigsh( L, k=min(k_eigs, N - 1), 
        which='SM' # smallest magnitude eigenvalues
    )

    # sort
    idx = np.argsort(evals)

    evals = evals[idx]
    evecs = evecs[:, idx]

    # graph Fourier transform
    coeffs = evecs.T @ u

    # power spectrum
    power = np.abs(coeffs)**2

    # remove DC mode
    power[0] = 0

    # dominant frequency
    peak_idx = np.argmax(power)

    dominant_lambda = evals[peak_idx]

    # spectral centroid
    centroid = np.sum(evals * power) / np.sum(power)

    # spectral bandwidth
    bandwidth = np.sqrt(
        np.sum(power * (evals - centroid)**2)
        / np.sum(power)
    )

    if plot:

        plt.figure(figsize=(6,4))
        plt.plot(evals, power, 'o')
        plt.xlabel('Graph Laplacian eigenvalue')
        plt.ylabel('Power')
        plt.title('Graph Spectral Power')

        plt.tight_layout()
        plt.show()

    #return {
    #     "evals": evals,
    #     "power": power,
    #     "dominant_lambda": dominant_lambda,
    #     "spectral_centroid": centroid,
    #     "spectral_bandwidth": bandwidth,
    # }

    return evals, power, dominant_lambda

def strongest_frequency(u_grid, plot=False): #2x2 heatmap ctn
    ny, nx = u_grid.shape  # support rectangular
    
    # Compute 2D DCT
    pattern_dct = dctn(u_grid, norm='ortho')
    power_spectrum = np.abs(pattern_dct) ** 2

    # Remove DC component
    power_spectrum[0:2, 0:2] = 0

    # 3. Generate DCT frequency coordinates
    fx = np.arange(nx) / (2 * nx)  # x-frequencies
    fy = np.arange(ny) / (2 * ny)  # y-frequencies
    fx_grid, fy_grid = np.meshgrid(fx, fy, indexing='xy')

    # Find max power frequency index
    max_idx = np.unravel_index(np.argmax(power_spectrum), power_spectrum.shape)
    kx = fx_grid[max_idx]
    ky = fy_grid[max_idx]
    k_mag = np.sqrt(kx**2 + ky**2)

    if plot:
        plt.figure(figsize=(6, 6))
        plt.imshow(np.log1p(power_spectrum), cmap='inferno', origin='lower')
        plt.title('2D DCT Power Spectrum (log scale)')
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    return k_mag


def strongest_wavelength_fft(u_grid, plot=False): # 2x2 heatmap fourier
    ny, nx = u_grid.shape  # assume square grid (N x N)

    # 1. Compute 2D FFT and power spectrum
    pattern_fft = np.fft.fft2(u_grid)
    power_spectrum = np.abs(pattern_fft) ** 2

    # 2. Shift zero frequency to center
    power_spectrum = np.fft.fftshift(power_spectrum)

    # 3. Remove DC component (center of the spectrum)
    center_y, center_x = ny // 2, nx // 2
    power_spectrum[center_y:center_y+1, center_x:center_x+1] = 0

    # 4. Generate FFT frequency coordinates
     # range: [-0.5, 0.5)
    fx = np.fft.fftshift(np.fft.fftfreq(nx))
    fy = np.fft.fftshift(np.fft.fftfreq(ny))
    fx_grid, fy_grid = np.meshgrid(fx, fy, indexing='xy')

    # 5. Find the frequency with maximum power
    max_idx = np.unravel_index(np.argmax(power_spectrum), power_spectrum.shape)
    kx = fx_grid[max_idx]
    ky = fy_grid[max_idx]
    k_mag = np.sqrt(kx**2 + ky**2)

    # 6. Convert frequency magnitude to wavelength
    if k_mag > 0:
        wavelength = 1 / k_mag
        wavelength = round(wavelength, 4)
    else:
        wavelength = None

    # 7. Optional plot
    if plot:
        plt.figure(figsize=(6, 6))
        plt.imshow(np.log1p(power_spectrum), cmap='inferno', origin='lower')
        plt.title('2D FFT Power Spectrum (log scale)')
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    return wavelength

def strongest_wavelength(u_grid, plot=False):
    ny, nx = u_grid.shape  # support rectangular
    
    # Compute 2D DCT
    pattern_dct = dctn(u_grid, norm='ortho')
    power_spectrum = np.abs(pattern_dct) ** 2

    # Remove DC component
    power_spectrum[0:2, 0:2] = 0

    # 3. Generate DCT frequency coordinates
    fx = np.arange(nx) / (2 * nx)  # x-frequencies
    fy = np.arange(ny) / (2 * ny)  # y-frequencies
    fx_grid, fy_grid = np.meshgrid(fx, fy, indexing='xy')

    # Find max power frequency index
    max_idx = np.unravel_index(np.argmax(power_spectrum), power_spectrum.shape)
    kx = fx_grid[max_idx]
    ky = fy_grid[max_idx]
    k_mag = np.sqrt(kx**2 + ky**2)

    if k_mag > 0:
        wavelength = 1 / k_mag
        wavelength = round(wavelength, 4)
    else:
        wavelength = None

    if plot:
        plt.figure(figsize=(6, 6))
        plt.imshow(np.log1p(power_spectrum), cmap='inferno', origin='lower')
        plt.title('2D DCT Power Spectrum (log scale)')
        plt.colorbar()
        plt.tight_layout()
        plt.show()

    return wavelength

# 752 , 404, 847

# #c, w = 0, 0
# critical_ratio = (a*e - 2 * b * d + 2 * np.sqrt( b * d * ( b * d - a*e)))/a**2
# Dv = Du * critical_ratio # sigma = Dv/Du

# critical_eigenvalue = ((a - e)*critical_ratio - np.sqrt(abs(b)*d*critical_ratio+0j)*(critical_ratio+1))/(Du * critical_ratio*(critical_ratio-1))

# eigvals, eigvecs, lambda_alpha = linear_stability(params)

# valid = eigvals < -0.000001

# log_eigval = np.log(-eigvals[valid])

# # where would it touch the x axis
# eigenvalue_roots = eigvalue_roots(params)

# # expansion of line to see trend
# x_axis = np.arange(int(eigenvalue_roots[1]), int(eigenvalue_roots[0]), 1)
# lambda_alpha_all = growth_rate(params, x_axis)



# # sort_idx = log_eigval.argsort()[::-1] 
# # log_eigval = log_eigval[sort_idx] 
# branch_1, branch_2, k_unique = plots.dispersion_relation_plot(f, g, Du, Dv, plot=False)


# plt.plot(eigenvalue_roots, [0, 0], 'r*')
# #plt.plot(-(k_unique**2), branch_1, 'o', label='From k (dispersion)', alpha=0.7)
# plt.plot(eigvals[valid], lambda_alpha.real[valid], 'o')
# plt.plot(x_axis, lambda_alpha_all.real, 'k--')
# plt.axhline(0, color='gray', linestyle='--')
# plt.xlabel("Laplacian eigenvalue Λ")
# plt.ylabel("Growth rate Re(λ)")
# plt.title("Dispersion Relation")

# params = valid_params[752][:8]
# eigvals, eigvecs, lambda_alpha = linear_stability(params)
# #%%
# # Dv array changes the height (bigger -> higher)
# # Du array shifts the height (bigger -> left)
# for i in range(3):
#     Dv_array = [1, 5, 10]
#     Du_array = [0.1, 0.2, 0.3]
#     params = valid_params[752][:8]
#     params = params[:6] + (0.165, Dv_array[i])
#     #params = params[:6] + (Du_array[i], Du_array[i]*20)
#     eigvals, eigvecs, lambda_alpha = linear_stability(params)
#     valid = eigvals < -0.000001
#     plt.plot(np.log(-eigvals[valid]), lambda_alpha.real[valid], 'o')

# #plt.axhline(0, color='gray', linestyle='--')
# plt.xlabel("Laplacian eigenvalue Λ")
# plt.ylabel("Growth rate Re(λ)")
# plt.title("Dispersion Relation")

# np.any(lambda_alpha.real > 0)
# np.sum(lambda_alpha.real > 0)
# # %%
# params = valid_params[404][:8]

# model = models.linear
# grid_size = 20
# a, b, c, d, e, w, Du, Dv = params
# f, g = model(a, b, c, d, e, w)

# %%
# a, b, c, d, e, w, Du, Dv = valid_params[0][:8]
# #c, w = 0, 0
# model = models.linear
# f, g = model(a, b, c, d, e, w)

# shape = shapes.grid_check
# initial_conditions = np.ones((grid_size ** 2, 2)) /2

# u, v = simulation.simulation(f, g, Du, Dv, n = 50000, initial_conditions=initial_conditions)
# # %%

# critical_ratio = a*e - 2 * b * d + 2 * np.sqrt( b * d * ( b * d - a*e))/a**2
# print(critical_ratio)

# # d needs to be negative or this is imaginary?
# critical_eigenvalue = ((a - e)*critical_ratio - np.sqrt(abs(b)*d*critical_ratio+0j)*(critical_ratio+1))/(Du * critical_ratio*(critical_ratio-1))
# B_critical = (-a + e + (critical_ratio -1)*Du * critical_eigenvalue + np.sqrt(4*b*d+(a-e-(critical_ratio -1)*(Du * critical_eigenvalue))**2))/(2*b)

# # the roots of c(eigval_alpha) where a growth rate**2 + b growth rate + c = 0

# # turing instability -> some of the eigenvalues are in this range
# sigma = Dv/Du
# root1 = -(e + a * sigma) + np.sqrt((e + sigma * a)**2 - 4 * sigma * (a*e-b*d))/2*sigma*Du
# root2 = -(e + a * sigma) - np.sqrt((e + sigma * a)**2 - 4 * sigma * (a*e-b*d))/2*sigma*Du



# #%% Comparison of growth rate found from laplacian and found from fourier

# branch_1, branch_2, k_unique = plots.dispersion_relation_plot(f, g, Du, Dv)

# plt.figure(figsize=(7, 5))
# plt.plot(-(k_unique**2), branch_1, 'o', label='From k (dispersion)', alpha=0.7)
# plt.plot(eigvals, lambda_alpha, '.', label='From Λ (linear_stability)', alpha=0.7)

# plt.axhline(0, linestyle='--', color='gray')
# plt.xlabel("Laplacian eigenvalue Λ or wave number -k**2")
# plt.ylabel("Growth rate Re(λ)")
# plt.title("Overlay of Dispersion Relations")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()


# %%

# # b±√(b²-4ac))/(2a) .
#         # 1/2 -B +- sqrt (4 AC + B^2)

#         A = b # fv
#         # sigma = Dv/Du
#         # fu - gv + epsilon * (1 - sigma) * eigenvalue_alpha
#         B = a - e + Du * (1 - Dv/Du) * lam
#         C = -d # gu

#         discriminant = B**2 - 4*A*C
#         sqrt_disc = np.sqrt(discriminant + 0j)
#         B1 = (-B + sqrt_disc) / (2*A)
#         B2 = (-B - sqrt_disc) / (2*A)

#         # choose the solution with the larger real part
#         B_alpha[i] = B1 if B1.real > B2.real else B2

#         # compute growth rate
#         # lambda_alpha = fu + epsilon eigenvalue_alpha + fv B_alpha
#         lambda_alpha[i] = a + Du * lam + b * B_alpha[i] # matches paper equation
#         #

    # eigenvalue problem: J_alpha [1, B_alpha] =  lamda_alpha [1, B_alpha]

    # equations:
    # lamda_alpha = f_u + epsilon eigenvalue_alpha + f_v B_alpha
    # lamda_alpha B_alpha = g_u + (g_v) + delta epsilon eigenvalue_alpha) B_alpha

    # substitution and rearrangement -> quadratic
    # AB_alpha^2 + BB_alpha + C = 0
    #A = fv
    #B = (fu - gv) + epsilon ( 1 - delta) eigenvalue_alpha
    #C = - gu

    # eigenvalues