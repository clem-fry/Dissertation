import models
import plots
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import discrete_turning
import numpy as np
from scipy.signal import find_peaks

def find_params():
    param_ranges = {
        'a': (-20, 20),
        'b': (-20, 20),
        'c': (-20, 20),  # c and d positive to keep reaction meaningful
        'd': (-20, 20),
        'e': (-20, 20),
        'w':(-20, 20),
        'Du': (0.1, 1), # POSITIVE
        'Dv': (5, 15) # POSTIVE
    }

    model = models.linear
    valid_params = suggest_parameters(param_ranges, model, n_trials = 5000)

    with open('valid_params_19june.txt', 'w') as f:
        for item in valid_params:
            f.write(str(item) + '\n')

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
        params = a, b, c, d, e, w, Du, Dv
        if np.all(turing_check(params)):
            eigvals, eigvecs, lambda_alpha = analysis.linear_stability(params)

            smoothed = gaussian_filter1d(lambda_alpha, sigma=2)
            all_peaks, _ = find_peaks(smoothed)

            if len(all_peaks) == 0:
                peak_k = 0
                peak_lamda = 0
                
            else:
                peak = all_peaks[0]
                peak_lamda = lambda_alpha[peak]
                peak_k = eigvals[peak]

            unstable_indices = np.where(lambda_alpha > 0)[0]
            if len(unstable_indices) > 0:
                k_min = eigvals[unstable_indices[0]]
                k_max = eigvals[unstable_indices[-1]]
                unstable_band_width = k_max - k_min
                area = np.trapz(lambda_alpha[unstable_indices], eigvals[unstable_indices])
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

    #spacial_instability = -(Dv * a + Du * e)**2 + 4 * Du * Dv * det_J < 0
    eigvals, eigvecs, lambda_alpha = analysis.linear_stability(params)
    if np.any(lambda_alpha > 0):
        spacial_instability = True
    # if lamda0 > 0:
    #     spacial_instability = True
    #     if verbose:
    #         print('spacial system is unstable - turing instability')
    # else:
    #     if verbose:
    #         print('spacial system is stable - not turing instability')

    return nonspacial_stability, spacial_instability


def find_steady_state(f, g, initial_guess):
    def equations(vars):
        u, v = vars
        return [f(u, v), g(u, v)]

    sol = root(equations, initial_guess, method='hybr')
    if sol.success:
        return sol.x
    else:
        raise RuntimeError("Failed to find steady state")
    