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

def dispersion_relation_plot(f, g, Du, Dv, plot=True, save=False, save_path = "simulation_final.png"):
    u0, v0 = find_steady_state(f, g, initial_guess=(1.0, 0.0))
    #print("u0 =", u0, "v0 =", v0)

    #eigenvalues, J = models.local_stability(f, g, u0, v0)
    # k is the spatial frequency / wavenumber (how many cycles per unit length)
    k_vals = np.linspace(0.0, 5.0, 400)
    real_parts = []
    prev_eigs = None
    branch_1 = []
    branch_2 = []

    for k_val in k_vals:
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
        ax.plot(k_vals, branch_1, label="λ₁ (tracked)")
        ax.plot(k_vals, branch_2, label="λ₂ (tracked)")
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

    return(branch_1, branch_2, k_vals)
# %%
from scipy.signal import find_peaks

def suggest_parameters(param_ranges, model, n_trials=1000):
    # Set ranges for parameters

    valid_params = []

    for n in range(n_trials):
        if n % 100 == 0:
            print(n)
        peak = False
        a = round(np.random.uniform(*param_ranges['a']), 3)
        b = round(np.random.uniform(*param_ranges['b']), 3)
        e = round(np.random.uniform(*param_ranges['e']), 3)
        d = round(np.random.uniform(*param_ranges['d']), 3)
        c = round(np.random.uniform(*param_ranges['c']), 3)
        w = round(np.random.uniform(*param_ranges['w']), 3)
        Du = round(np.random.uniform(*param_ranges['Du']), 3)
        Dv = round(np.random.uniform(*param_ranges['Dv']), 3)

        f, g = model(a, b, c, d, e, w)
        if np.all(turing_check(f, g, Du, Dv)):
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

            if len(peaks) != 0:
                peak = True

            valid_params.append((a, b, c, d, e, w, Du, Dv, peak, peak_lamda, peak_k, unstable_band_width, area))

    print(f"Found {len(valid_params)} parameter sets with positive steady states out of {n_trials} trials.")
    for params in valid_params[:10]:  # print first 10 valid sets
        print(f"{params[0]:.3f}, {params[1]:.3f}, {params[2]:.3f}, {params[3]:.3f}, {params[4]:.3f}, {params[5]:.3f}, {params[6]:.3f}, {params[7]:.3f}")

    return valid_params

# Run to get suggestions
#suggest_parameters()

#%%

def turing_check(f, g, Du, Dv, verbose=False):
    nonspacial_stability = False # we want both of these to be true
    spacial_instability = False
    try:
        u0, v0 = find_steady_state(f, g, initial_guess=(1.0, 0.0))
    except:
        if verbose:
            print('non-spacial system is unstable - not turing instability')
        return False, False # undefined steady state

    if np.all(models.nonspacial_eigenvalues(f, g) > 0): # nonspacial stability
        nonspacial_stability = True

    # spacial instability - real part of leading eigenvalue is positive
    lamda0 = max(models.dispersion(f, g, u0, v0, Du, Dv, 0)[0])
    if lamda0 > 0:
        spacial_instability = True
        if verbose:
            print('spacial system is unstable - turing instability')
    else:
        if verbose:
            print('spacial system is stable - not turing instability')

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