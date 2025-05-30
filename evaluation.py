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

def dispersion_relation_plot(f, g, Du, Dv, plot=True):
    u0, v0 = find_steady_state(f, g, initial_guess=(1.0, 0.0))
    #print("u0 =", u0, "v0 =", v0)

    #eigenvalues, J = models.local_stability(f, g, u0, v0)
    # k is the spatial frequency / wavenumber (how many cycles per unit length)
    k_vals = np.linspace(0.0, 5.0, 400)
    real_parts = []
    for k_val in k_vals:
        real, _ = models.dispersion(f, g, u0, v0, Du, Dv, k_val)
        real_parts.append(real)

    if plot:
        plt.plot(k_vals, real_parts)
        plt.axhline(0, color='k', linestyle='--')
        plt.title('Dispersion Relation (Real part of λ vs k)')
        plt.xlabel('Wave number k')
        plt.ylabel('Re(λ)')
        plt.grid(True)
        plt.show()

    return(np.array(real_parts), k_vals)
# %%

def suggest_parameters(param_ranges, model, n_trials=1000):
    # Set ranges for parameters

    valid_params = []

    for _ in range(n_trials):
        a = np.random.uniform(*param_ranges['a'])
        b = np.random.uniform(*param_ranges['b'])
        e = np.random.uniform(*param_ranges['e'])
        d = np.random.uniform(*param_ranges['d'])
        c = np.random.uniform(*param_ranges['c'])
        w = np.random.uniform(*param_ranges['w'])
        Du = np.random.uniform(*param_ranges['Du'])
        Dv = np.random.uniform(*param_ranges['Dv'])

        f, g = model(a, b, e, d, c=c, w=w)
        if np.all(turing_check(f, g, Du, Dv)):
            valid_params.append((a, b, c, d, e, w, Du, Dv))

    print(f"Found {len(valid_params)} parameter sets with positive steady states out of {n_trials} trials.")
    for params in valid_params[:10]:  # print first 10 valid sets
        print(f"{params[0]:.3f}, {params[1]:.3f}, {params[2]:.3f}, {params[3]:.3f}, {params[4]:.3f}, {params[5]:.3f}, {params[6]:.3f}, {params[7]:.3f}")

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

    if (u0 > 0) and (v0 > 0): # nonspacial stability
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