#%%
import discrete_turning
import models
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

# %%

def find_steady_state(f, g, initial_guess):
    def equations(vars):
        u, v = vars
        return [f(u, v), g(u, v)]

    sol = root(equations, initial_guess, method='hybr')
    if sol.success:
        return sol.x
    else:
        raise RuntimeError("Failed to find steady state")
    
# %%

def dispersion_relation_plot(f, g, Du, Dv, grid_size=20, plot=True, save=False, save_path = "simulation_final.png"):
    u0, v0 = find_steady_state(f, g, initial_guess=(1.0, 0.0))
    #print("u0 =", u0, "v0 =", v0)

    #eigenvalues, J = models.local_stability(f, g, u0, v0)
    # k is the spatial frequency / wavenumber (how many cycles per unit length)
    #k_vals = np.linspace(0.0, 5.0, 400)
    kx = np.fft.fftfreq(grid_size, d=1)  # no 2π
    ky = np.fft.fftfreq(grid_size, d=1)

    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
    k_squared = ((2 * np.sin(np.pi * kx_grid))**2 + (2 * np.sin(np.pi * ky_grid))**2).flatten()
    k_sqrt = np.sqrt(k_squared)

    k_unique = np.unique(np.round(np.sort(k_sqrt), decimals=6))[::-1]  # to remove duplicate magnitudes
    # to remove duplicate magnitudes
    real_parts = []
    prev_eigs = None
    branch_1 = []
    branch_2 = []

    for k_val in k_unique:
        real_parts, _ = models.dispersion(f, g, u0, v0, Du, Dv, k_val)
        if prev_eigs is None:
            branch_1.append(real_parts[0])
            branch_2.append(real_parts[1])
            prev_eigs = real_parts

        else:
            # Match current eigenvalues to previous based on continuity
            if abs(real_parts[0] - prev_eigs[0]) < abs(real_parts[1] - prev_eigs[0]):
                branch_1.append(real_parts[0])
                branch_2.append(real_parts[1])
                prev_eigs = real_parts
            else:
                branch_1.append(real_parts[1])
                branch_2.append(real_parts[0])
                prev_eigs = real_parts[::-1]

    branch_1 = np.array(branch_1)
    branch_2 = np.array(branch_2)

    if plot or save:
        fig, ax = plt.subplots()
        ax.plot(k_unique, branch_1, '*', markersize=4, label="λ₁ (tracked)")
        ax.plot(k_unique, branch_2, '*', markersize=4, label="λ₂ (tracked)")
        ax.axhline(0, color='k', linestyle='--')
        ax.set_title('Dispersion Relation (Real part of λ vs k)')
        ax.set_xlabel('Wave number k')
        ax.set_ylabel('Re(λ)')
        ax.grid(True)
        if save:
            fig.savefig(save_path)
            if not plot:
                plt.close(fig) 
        if plot:
            plt.show()

    return(branch_1, branch_2, k_unique)
# %%
from scipy.signal import find_peaks

def suggest_parameters(param_ranges, model, n_trials=1000):
    # Set ranges for parameters

    valid_params = []

    for n in range(n_trials):
        if n % 100 == 0:
            print(n)
        # peak = False
        a = round(np.random.uniform(*param_ranges['a']), 3)
        b = round(np.random.uniform(*param_ranges['b']), 3)
        e = round(np.random.uniform(*param_ranges['e']), 3)
        d = round(np.random.uniform(*param_ranges['d']), 3)
        c = round(np.random.uniform(*param_ranges['c']), 3)
        w = round(np.random.uniform(*param_ranges['w']), 3)
        Du = round(np.random.uniform(*param_ranges['Du']), 3)
        Dv = round(np.random.uniform(*param_ranges['Dv']), 3)

        f, g = model(a, b, c, d, e, w)
        parmas = a, b, c, d, e, w, Du, Dv
        if np.all(turing_check(parmas)):
            branch1, branch2, k_vals = dispersion_relation_plot(f, g, Du, Dv, plot=False)
            peak1, _ = find_peaks(branch1)
            peak2, _ = find_peaks(branch2)
            peaks = np.append(peak1, peak2)

            real_parts_branch = np.maximum(branch1, branch2)
            peak_lamda = np.max(real_parts_branch)
            peak_k = k_vals[np.argmax(real_parts_branch)]
            unstable_indices = np.where(real_parts_branch > 0)[0]
            if len(unstable_indices) > 0:
                k_min = k_vals[unstable_indices[0]]
                k_max = k_vals[unstable_indices[-1]]
                unstable_band_width = k_max - k_min
                area = np.trapz(real_parts_branch[unstable_indices], k_vals[unstable_indices])
            else:
                unstable_band_width = 0
                area = 0

            peak_width = peak_lamda / unstable_band_width if unstable_band_width > 0 else 0
            peak_area = peak_lamda / area if area > 0 else 0
            # if len(peaks) != 0:
            #     peak = True

            valid_params.append((a, b, c, d, e, w, Du, Dv, peak_width, peak_lamda, peak_k, unstable_band_width, area, peak_area))

    print(f"Found {len(valid_params)} parameter sets with positive steady states out of {n_trials} trials.")
    for params in valid_params[:10]:  # print first 10 valid sets
        print(f"{params[0]:.3f}, {params[1]:.3f}, {params[2]:.3f}, {params[3]:.3f}, {params[4]:.3f}, {params[5]:.3f}, {params[6]:.3f}, {params[7]:.3f}")

    return valid_params

# Run to get suggestions
#suggest_parameters()

#%%

def turing_check(params, verbose=False):
    a, b, c, d, e, w, Du, Dv = params
    nonspacial_stability = False # we want both of these to be true
    spacial_instability = False
    # try:
    #     u0, v0 = find_steady_state(f, g, initial_guess=(1.0, 0.0))
    # except:
    #     if verbose:
    #         print('non-spacial system is unstable - not turing instability')
    #     return False, False # undefined steady state

    # eigenvalues of nonspacial stability - should be positive and not satisfy
    # must satisfy a + e > 0 and ae - bd > 0
    # !! just changed to negative
    # this ensure that the tr(A) = a + e < 0
    # det(A) = a*e - b*d > 0
    # if np.all(models.nonspacial_eigenvalues(f, g) < 0): # nonspacial stability
    #     nonspacial_stability = True

    det_J = a * e - b * d > 0
    tr_J = a + d < 0

    if det_J and tr_J:
        nonspacial_stability = True

    # spacial instability - real part of leading eigenvalue is positive
    # Du * e + Du * a > 2 * sqrt(Du Dv ( a * e - b * d)) https://genecompute.medium.com/deriving-the-conditions-for-turing-patterns-3496b7fa1e1a
    # we want some growth rate > 0

    #lamda0 = max(models.dispersion(f, g, u0, v0, Du, Dv, 0)[0])

    spacial_instability = -(Dv * a + Du * e)**2 + 4 * Du * Dv * det_J < 0
    # if lamda0 > 0:
    #     spacial_instability = True
    #     if verbose:
    #         print('spacial system is unstable - turing instability')
    # else:
    #     if verbose:
    #         print('spacial system is stable - not turing instability')

    return nonspacial_stability, spacial_instability

#%% not currently used really:

def measure_pattern_strength(u):
    # change this to a more sophisticated mesure?
    return np.std(u)

def bifurcation(a_vals, b_vals, model, Du, Dv):
    pattern_strengths = np.zeros((len(a_vals), len(b_vals)))

    for i, A in enumerate(a_vals):
        print(i)
        for j, B in enumerate(b_vals):
            f, g = model(A, B)
            u, v = discrete_turning.simulation(f, g, Du, Dv, n=3000, show=False)
            pattern_strengths[j, i] = measure_pattern_strength(v)

    # Plot bifurcation heatmap
    plt.imshow(pattern_strengths, extent=[a_vals[0], a_vals[-1], b_vals[-1], b_vals[0]],
            cmap='magma', aspect='auto')
    plt.colorbar(label='Pattern Strength (std of v)')
    plt.xlabel('a')
    plt.ylabel('b')
    plt.title('Bifurcation Map of Model')
    plt.show()

def dispersion_plot(params, model, grid_size):
    # The brightest areas (where Re(λ) is largest) show the wavevectors most likely to dominate.
    # A circular ring means that all directions of a certain wave number (magnitude) are unstable
    #  — leads to isotropic patterns like spots.
    # A striped unstable region would favor certain directions
    #  — leads to anisotropic patterns like stripes or ridges.

    a, b, c, d, e, w, Du, Dv = params
    f, g = model(a, b, c, d, e, w)
    u0, v0 = find_steady_state(f, g, initial_guess=(1.0, 0.0))

    def dispersion(k_mag):
        real_parts, _ = models.dispersion(f, g, u0, v0, Du, Dv, k_mag)
        return max(real_parts)

    dx = 1 # use the relevant k values
    kx = 2 * np.pi * np.fft.fftfreq(grid_size, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(grid_size, d=dx) # INCLUDES DIAGONAL WAVES
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')

    k_mags = np.sqrt(kx_grid**2 + ky_grid**2)

    # find the dispersion value for each k
    dispersion_grid = np.vectorize(dispersion)(k_mags)

    dispersion_shifted = np.fft.fftshift(dispersion_grid)

    # Plot
    plt.imshow(dispersion_shifted, extent=[kx.min(), kx.max(), ky.min(), ky.max()],
            cmap='coolwarm', origin='lower')
    plt.title("Dispersion: Re(λ) over (kₓ, k_y)")
    plt.xlabel("kₓ")
    plt.ylabel("k_y")
    plt.colorbar(label="Re(λ)")
    plt.show()

def fourier_spectrum_plot(u):
    # stripe has two strong peaks
    # dot grid (hexagonal spots) has peaks forming a hexagonal symmetry.
    # Noise has a broad, featureless spectrum.
    # Symmetry in the Fourier plot reflects symmetry in the spatial pattern.
    # Peaks = dominant spatial frequencies (wavelengths or periodicities).

    u_reshaped = np.reshape(u, (20, 20)) 

    U_fft = np.fft.fft2(u_reshaped)
    U_fft_shifted = np.fft.fftshift(U_fft)

    power_spectrum = np.abs(U_fft_shifted)

    # Plot
    plt.imshow(power_spectrum, cmap='inferno', origin='lower')
    plt.title("Fourier Spectrum of u(x, y)")
    plt.xlabel("k_x index")
    plt.ylabel("k_y index")
    plt.colorbar(label="|FFT|")
    plt.show()