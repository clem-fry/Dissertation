#%%
import models
import numpy as np
from scipy.optimize import root
import sympy as sp
from scipy.optimize import root_scalar
import ast
import re

def find_params():
    param_ranges = {
        'a': (0, 100), # a and e must be different signs
        'b': (-100, 0), # b and d must be different signs
        'd': (0, 100),
        'e': (-100,0),
    }

    model = models.linear
    valid_params = suggest_parameters(param_ranges, model, n_trials = 100000)

    with open('valid_params_25june.txt', 'w') as f:
        for item in valid_params:
            f.write(str(item) + '\n')

def load_params(file_name):
    valid_params = []
    with open(file_name, 'r') as file:
        for line in file:
            line = line.strip()

            # Remove trailing dot (if mistakenly added during printing)
            if line.endswith('.'):
                line = line[:-1]

            # Replace NumPy wrappers with raw values
            line = re.sub(r'np\.(?:float64|complex128)\(([^)]+)\)', r'\1', line)
            line = line.replace('np.False_', 'False').replace('np.True_', 'True')

            # Now it's a clean literal tuple
            try:
                tup = ast.literal_eval(line)
                valid_params.append(tup)
            except Exception as e:
                print(f"Skipping line due to parse error: {line} ({e})")
    return(valid_params)

def critical_ratio(J):
    fu, fv, gu, gv = J[0, 0], J[0, 1], J[1, 0], J[1, 1]
    ratio = (fu*gv - 2*fv* gu + 2*np.sqrt(fv*gu * (fv*gu - fu*gv)+ 0j))/fu**2
    return ratio

def critical_eigenvalue(J, sigma, Du):
    fu, fv, gu, gv = J[0, 0], J[0, 1], J[1, 0], J[1, 1]
    eigenvalue = ((fu - gv)*sigma - np.sqrt(abs(fv*gu)*sigma +0j) * (sigma + 1))/(Du * sigma * (sigma-1))
    return eigenvalue

def epsilon_for_eigenvalue(J, eig, sigma):
    fu, fv, gu, gv = J[0, 0], J[0, 1], J[1, 0], J[1, 1]
    Du = ((fu - gv)*sigma - np.sqrt(abs(fv*gu)*sigma +0j) * (sigma + 1))/(eig * sigma * (sigma-1))
    return Du

def eigenvalue_roots(J, ratio, Du):
    fu, fv, gu, gv = J[0, 0], J[0, 1], J[1, 0], J[1, 1]
    root1 = -(gv + fu * ratio) + np.sqrt((gv + ratio * fu)**2 - 4 * ratio * (fu*gv-fv*gu))/2*ratio*Du
    root2 = -(gv + fu * ratio) - np.sqrt((gv + ratio * fu)**2 - 4 * ratio * (fu*gv-fv*gu))/2*ratio*Du
    return root1, root2

def peak_lambda(J, sigma): # peak growth based off sigma
    fu, fv, gu, gv = J[0, 0], J[0, 1], J[1, 0], J[1, 1]
    # peak_eig/epsilon
    peak_eig = ((fu - gv)*sigma - np.sqrt(abs(fv*gu)*sigma +0j) * (sigma + 1))/(sigma * (sigma-1))
    trace_J = fu + gv
    det_J = fu*gv - fv*gu

    B = -(trace_J + (1+sigma)*peak_eig)
    C = sigma * peak_eig **2 + (gv + sigma * fu) * peak_eig + det_J
    discriminant = B**2 - 4*C
    discriminant = np.real_if_close(discriminant, tol=1e-10)

    peak_growth = 0.5 * (-B + np.sqrt(discriminant + 0j))
    return peak_growth

def find_sigma_for_growth(J, target_growth):
    def error(sigma):
        return peak_lambda(J, sigma) - target_growth

    critical_sigma = critical_ratio(J)
    bracket = (critical_sigma, 300)
    sol = root_scalar(error, bracket=bracket, method='brentq')
    if sol.converged:
        return sol.root
    else:
        raise RuntimeError("No suitable sigma found")
    
def find_suitable_params(eig, growth, desired_root_spread, params):
    # ideal peak eig and growth, all valid params and the ideal spread of the roots
    param_sigma_root_list = []

    for idx, param in enumerate(params):
        a, b, c, d = param[:4]
        J = non_spacial_jacobian(a, b, c, d)
        try:
            sigma = find_sigma_for_growth(J, growth)
        except:
            sigma = None
        if sigma is not None:
            epsilon = epsilon_for_eigenvalue(J, eig, sigma)
            root_width = np.sqrt((d + sigma * a)**2 - 4 * sigma * (a*d - b*c))/(sigma * epsilon)
            param_sigma_root_list.append((param[:4], sigma, epsilon, root_width, idx))
            
    sorted_params = sorted(
        param_sigma_root_list,
        key=lambda x: abs(np.real(x[3]) - desired_root_spread)
    )
    
    return sorted_params

def suggest_parameters(param_ranges, model, n_trials=5000):
    # Set ranges for parameters

    valid_params = []

    for n in range(n_trials):
        turing = False
        if n % 100 == 0:
            print(n)
        # peak = False
        a = np.round(np.random.uniform(*param_ranges['a']), 3)
        b = np.round(np.random.uniform(*param_ranges['b']), 3)
        d = np.round(np.random.uniform(*param_ranges['d']), 3)
        e = np.round(np.random.uniform(*param_ranges['e']), 3)

        c, w = 0, 0
        f, g = model(a, b, c, d, e, w)
        try:
            u0, v0 = find_steady_state(f, g, initial_guess=(1.0, 0.0))
        except:
            turing = False

        J = non_spacial_jacobian(a, b, d, e)
        if np.all(J != False):
            ratio = critical_ratio(J) # sigma
            turing = turing_check(J, ratio)
            if np.all(turing):
                oscillations = J.trace() ** 2 - 4 * np.linalg.det(J) > 0
                valid_params.append((a, b, d, e, ratio, oscillations))

    print(f"Found {len(valid_params)} parameter sets with positive steady states out of {n_trials} trials.")
    # for params in valid_params[:10]:  # print first 10 valid sets
    #     print(f"{params[0]:.3f}, {params[1]:.3f}, {params[2]:.3f}, {params[3]:.3f}, {params[4]:.3f}, {params[5]:.3f}, {params[6]:.3f}, {params[7]:.3f}")

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

# def non_spacial_jacobian(f, g):
#         try:
#             u0, v0 = find_steady_state(f, g, initial_guess=(1.0, 0.0))
#         except:
#             return False # undefined steady state

#         u = sp.symbols('u')
#         v = sp.symbols('v')

#         F = f(u, v)
#         G = g(u, v)

#         J = sp.Matrix([
#         [sp.diff(F, u), sp.diff(F, v)],
#         [sp.diff(G, u), sp.diff(G, v)]
#         ])

#         J_sub = J.subs({u: u0, v: v0}).evalf()  # Evaluate to floats
#         J_sub = np.array(J_sub).astype(np.float64)  # Convert to NumPy array

#         return J_sub

def non_spacial_jacobian(a, b, c, d):
    J = np.array([[a, b], [c, d]])
    return J

def turing_check(J, sigma, verbose=False):
    #a, b, c, d, e, w, Du, Dv = params
    nonspacial_condition = False # we want both of these to be true
    spacial_condition = False
    # try:
    #     u0, v0 = find_steady_state(f, g, initial_guess=(1.0, 0.0))
    # except:
    #     return False, False

    # eigenvalues of nonspacial stability - should be positive and not satisfy
    # must satisfy a + e > 0 and ae - bd > 0
    # !! just changed to negative
    # this ensure that the tr(A) = a + e < 0
    # det(A) = a*e - b*d > 0
    # if np.all(models.nonspacial_eigenvalues(f, g) < 0): # nonspacial stability
    #     nonspacial_stability = True
    
    trace = np.trace(J) < 0
    determinant = np.linalg.det(J) > 0
    nonspacial_condition = np.all((trace, determinant))

    # spacial instability - real part of leading eigenvalue is positive
    # Du * e + Du * a > 2 * sqrt(Du Dv ( a * e - b * d)) https://genecompute.medium.com/deriving-the-conditions-for-turing-patterns-3496b7fa1e1a
    # we want some growth rate > 0

    #lamda0 = max(models.dispersion(f, g, u0, v0, Du, Dv, 0)[0])

    #spacial_instability = -(Dv * a + Du * e)**2 + 4 * Du * Dv * det_J < 0
    
    condition = spacial_instabilitiy(J, sigma)
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

# %%
