#%%
import simulation
import models
import analysis
import numpy as np
import matplotlib.pyplot as plt
import shapes
import matplotlib.image as mpimg

plt.rcParams.update({'font.size': 20})


def dispersion_relation_plot(params, continuous = False, grid_size=20, plot=True, save=False, save_path = "simulation_final.png", shape = shapes.grid_check, model = models.linear, threshold=None):
    if continuous:
        eigvals = np.arange(-30, 0, 0.1)
        lambda_alpha = analysis.growth_rate(params, eigvals)
    else:
        eigvals, eigvecs, lambda_alpha = analysis.linear_stability(params, grid_size=grid_size, shape = shape, model=model, threshold=threshold)
    #valid = eigvals < -0.000001

    if plot or save:
        fig, ax = plt.subplots()
        if continuous:
            ax.plot(eigvals, lambda_alpha.real)
        else:
            ax.plot(eigvals, lambda_alpha.real, 'o')
        ax.axhline(0, color='gray', linestyle='--')
        ax.set_xlabel('Laplacian Eigenvalue $\Lambda$')
        ax.set_ylabel('Growth rate Re(λ)')
        ax.grid(True)
        if save:
            fig.savefig(save_path)
            if not plot:
                plt.close(fig) 
        if plot:
            plt.show()

    return lambda_alpha, eigvals

def plot_dispersion_turing(parameters, indicies, initial_conditions=None, grid_size = 20):
    for i in indicies:
        a, b, c, d = parameters[i][0]
        sigma = parameters[i][1]
        Du = parameters[i][2]
        root_width = parameters[i][3]
        Dv = sigma * Du
        f, g = models.linear(a, b, 0, c, d, 0)

        if np.any(initial_conditions == None):
            u, v, _ = simulation.simulation(f, g, Du, Dv, n = 30000, show=False, save=True, grid_size=grid_size, save_path=f"plots/simulation_plot_{parameters[i][4]}.png")
        else:
            u, v, _ = simulation.simulation(f, g, Du, Dv, n = 30000, initial_conditions = initial_conditions, grid_size=grid_size, show=False, save=True, save_path=f"plots/simulation_plot_{parameters[i][4]}.png")
        
        params = (a, b, 0, c, d, 0, Du, Dv)
        _ = dispersion_relation_plot(params, plot=False, save=True, save_path=f"plots/dispersion_plot_{parameters[i][4]}.png")

    fig, axes = plt.subplots(5, 2, figsize=(12, 20))

    for idx, i in enumerate(indicies):

        sim_img = mpimg.imread(f"plots/simulation_plot_{parameters[i][4]}.png")
        disp_img = mpimg.imread(f"plots/dispersion_plot_{parameters[i][4]}.png")
        axes[idx, 0].imshow(sim_img)
        axes[idx, 0].axis('off')
        axes[idx, 0].set_title(f'Simulation {parameters[i][4]}')
        
        axes[idx, 1].imshow(disp_img)
        axes[idx, 1].axis('off')
        axes[idx, 1].set_title(f'Dispersion Relation {parameters[i][4]}')

    plt.tight_layout()
    plt.show()

from scipy.signal import find_peaks


def measure_pattern_strength(u):
    # change this to a more sophisticated mesure?
    return np.std(u)

def bifurcation(a_vals, b_vals, model, Du, Dv):
    pattern_strengths = np.zeros((len(a_vals), len(b_vals)))

    for i, A in enumerate(a_vals):
        print(i)
        for j, B in enumerate(b_vals):
            f, g = model(A, B)
            u, v, _ = simulation.simulation(f, g, Du, Dv, n=3000, show=False)
            pattern_strengths[j, i] = measure_pattern_strength(v)

    # Plot bifurcation heatmap
    plt.imshow(pattern_strengths, extent=[a_vals[0], a_vals[-1], b_vals[-1], b_vals[0]],
            cmap='magma', aspect='auto')
    plt.colorbar(label='Pattern Strength (std of v)')
    plt.xlabel('a')
    plt.ylabel('b')
    plt.title('Bifurcation Map of Model')
    plt.show()

def dispersion_2d(params, model, grid_size): # OLD LINEAR TECHNIQUE
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