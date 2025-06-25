import models
import plots
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import simulation
import analysis
import numpy as np
from scipy.optimize import root
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import sympy as sp


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

    with open('valid_params_23june.txt', 'w') as f:
        for item in valid_params:
            f.write(str(item) + '\n')

def critical_ratio(J):
    fu, fv, gu, gv = J[0, 0], J[0, 1], J[1, 0], J[1, 1]
    ratio = (fu*gv - 2*fv* gu + 2*np.sqrt(fv*gu * (fv*gu - fu*gv)+ 0j))/fu**2
    return ratio

def critical_eigenvalue(J, sigma, Du):
    fu, fv, gu, gv = J[0, 0], J[0, 1], J[1, 0], J[1, 1]
    eigenvalue = ((fu - gv)*sigma - np.sqrt(abs(fv)*gu*sigma +0j) * (sigma + 1))/(Du * sigma * (sigma-1))
    return eigenvalue

def epsilon_for_eigenvalue(J, eig, sigma):
    fu, fv, gu, gv = J[0, 0], J[0, 1], J[1, 0], J[1, 1]
    Du = ((fu - gv)*sigma - np.sqrt(abs(fv)*gu*sigma +0j) * (sigma + 1))/(eig * sigma * (sigma-1))
    return Du

def eigenvalue_roots(J, ratio, Du):
    fu, fv, gu, gv = J[0, 0], J[0, 1], J[1, 0], J[1, 1]
    root1 = -(gv + fu * ratio) + np.sqrt((gv + ratio * fu)**2 - 4 * ratio * (fu*gv-fv*gu))/2*ratio*Du
    root2 = -(gv + fu * ratio) - np.sqrt((gv + ratio * fu)**2 - 4 * ratio * (fu*gv-fv*gu))/2*ratio*Du
    return root1, root2

def suggest_parameters(param_ranges, model, n_trials=1000):
    # Set ranges for parameters

    valid_params = []

    for n in range(n_trials):
        turing = False
        if n % 100 == 0:
            print(n)
        # peak = False
        a = round(np.random.uniform(*param_ranges['a']), 3)
        b = round(np.random.uniform(*param_ranges['b']), 3)
        e = round(np.random.uniform(*param_ranges['e']), 3)
        d = round(np.random.uniform(*param_ranges['d']), 3)
        #c = round(np.random.uniform(*param_ranges['c']), 3)
        #w = round(np.random.uniform(*param_ranges['w']), 3)
        # Du = round(np.random.uniform(*param_ranges['Du']), 3)
        # Dv = round(np.random.uniform(*param_ranges['Dv']), 3)
        c, w = 0, 0
        f, g = model(a, b, c, d, e, w)
        try:
            u0, v0 = find_steady_state(f, g, initial_guess=(1.0, 0.0))
        except:
            turing = False

        J = non_spacial_stability(f, g)
        if np.all(J != False):
            ratio = critical_ratio(J) # sigma
            turing = turing_check(f, g, ratio)
        if turing:
            valid_params.append((a, b, d, e, ratio))

    print(f"Found {len(valid_params)} parameter sets with positive steady states out of {n_trials} trials.")
    for params in valid_params[:10]:  # print first 10 valid sets
        print(f"{params[0]:.3f}, {params[1]:.3f}, {params[2]:.3f}, {params[3]:.3f}, {params[4]:.3f}, {params[5]:.3f}, {params[6]:.3f}, {params[7]:.3f}")

    return valid_params

# Run to get suggestions
#suggest_parameters()

def find_steady_state(f, g, initial_guess):
    def equations(vars):
        u, v = vars
        return [f(u, v), g(u, v)]

    sol = root(equations, initial_guess, method='hybr')
    if sol.success:
        return sol.x
    else:
        raise RuntimeError("Failed to find steady state")
    

def spacial_instabilitiy(J, sigma):
        determinant = np.linalg.det(J)
        fu, fv, gu, gv = J[0, 0], J[0, 1], J[1, 0], J[1, 1]
        condition1 = (gv + sigma * fu)
        condition2 =((gv+sigma * fu ) ** 2 - 4* sigma  * determinant)
        return condition1, condition2

def non_spacial_stability(f, g):
        try:
            u0, v0 = find_steady_state(f, g, initial_guess=(1.0, 0.0))
        except:
            return False # undefined steady state

        u = sp.symbols('u')
        v = sp.symbols('v')

        F = f(u, v)
        G = g(u, v)

        J = sp.Matrix([
        [sp.diff(F, u), sp.diff(F, v)],
        [sp.diff(G, u), sp.diff(G, v)]
        ])

        J_sub = J.subs({u: u0, v: v0}).evalf()  # Evaluate to floats
        J_sub = np.array(J_sub).astype(np.float64)  # Convert to NumPy array

        return J_sub

#%%

def turing_check(f, g, sigma, verbose=False):
    #a, b, c, d, e, w, Du, Dv = params
    nonspacial_condition = False # we want both of these to be true
    spacial_condition = False
    try:
        u0, v0 = find_steady_state(f, g, initial_guess=(1.0, 0.0))
    except:
        return False, False

    # eigenvalues of nonspacial stability - should be positive and not satisfy
    # must satisfy a + e > 0 and ae - bd > 0
    # !! just changed to negative
    # this ensure that the tr(A) = a + e < 0
    # det(A) = a*e - b*d > 0
    # if np.all(models.nonspacial_eigenvalues(f, g) < 0): # nonspacial stability
    #     nonspacial_stability = True
    
    J_sub = non_spacial_stability(f, g)
    trace = np.trace(J_sub) < 0
    determinant = np.linalg.det(J_sub) > 0
    nonspacial_condition = np.all((trace, determinant))

    # spacial instability - real part of leading eigenvalue is positive
    # Du * e + Du * a > 2 * sqrt(Du Dv ( a * e - b * d)) https://genecompute.medium.com/deriving-the-conditions-for-turing-patterns-3496b7fa1e1a
    # we want some growth rate > 0

    #lamda0 = max(models.dispersion(f, g, u0, v0, Du, Dv, 0)[0])

    #spacial_instability = -(Dv * a + Du * e)**2 + 4 * Du * Dv * det_J < 0
    
    condition = spacial_instabilitiy(J_sub, sigma)
    spacial_condition = np.all([np.all(r > 0) for r in condition])
    # if lamda0 > 0:
    #     spacial_instability = True
    #     if verbose:
    #         print('spacial system is unstable - turing instability')
    # else:
    #     if verbose:
    #         print('spacial system is stable - not turing instability')

    return nonspacial_condition, spacial_condition

            # eigvals, eigvecs, lambda_alpha = analysis.linear_stability(params)

            # smoothed = gaussian_filter1d(lambda_alpha, sigma=2)
            # all_peaks, _ = find_peaks(smoothed)

            # if len(all_peaks) == 0:
            #     peak_k = 0
            #     peak_lamda = 0
                
            # else:
            #     peak = all_peaks[0]
            #     peak_lamda = lambda_alpha[peak]
            #     peak_k = eigvals[peak]

            # unstable_indices = np.where(lambda_alpha > 0)[0]
            # if len(unstable_indices) > 0:
            #     k_min = eigvals[unstable_indices[0]]
            #     k_max = eigvals[unstable_indices[-1]]
            #     unstable_band_width = k_max - k_min
            #     area = np.trapz(lambda_alpha[unstable_indices], eigvals[unstable_indices])
            # else:
            #     unstable_band_width = 0
            #     area = 0

            # peak_width = peak_lamda / unstable_band_width if unstable_band_width > 0 else 0
            # peak_area = peak_lamda / area if area > 0 else 0
            # # if len(peaks) != 0:
            # #     peak = True
