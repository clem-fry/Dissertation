#%%
import ast
import re
import discrete_turning
import models
import plots
import numpy as np
import matplotlib.pyplot as plt
import importlib
import parameters
import shapes

#%%

importlib.reload(discrete_turning)
importlib.reload(models)
importlib.reload(plots)
importlib.reload(shapes)
importlib.reload(parameters)

#%%

valid_params = []
with open('valid_params_scored.txt', 'r') as file:
    for line in file:
        line = line.strip()
        # Remove trailing dot if present (optional)
        if line.endswith('.'):
            line = line[:-1]

        # Replace np.float64(xxx) with xxx using regex
        clean_line = re.sub(r'np\.float64\(([^)]+)\)', r'\1', line)
        
        # Safely parse the tuple string to a tuple
        tup = ast.literal_eval(clean_line)
        valid_params.append(tup)


# functions for calculating quadratic of growth rate from laplacian eigenvalues

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

def linear_stability(params, grid_size = 20, shape = shapes.grid_check, model = models.linear):
    a, b, c, d, e, w, Du, Dv = params
    #c, w = 0, 0
    f, g = model(a, b, c, d, e, w)

    initial_conditions = np.ones((grid_size ** 2, 2)) /2

    all_nodes, border_nodes = discrete_turning.shape_nodes(shape, grid_size, initial_conditions)
    valid_nodes = [node for node in all_nodes.flatten() if node is not None]

    adjacency_matrix = discrete_turning.compute_adjacency_matrix(valid_nodes)
    degree_matrix = discrete_turning.compute_degree_matrix(adjacency_matrix)

    L = adjacency_matrix - degree_matrix

    eigvals, eigvecs = np.linalg.eigh(L)

    lambda_alpha = growth_rate(params, eigvals) 
    return eigvals, eigvecs, lambda_alpha

params = valid_params[752][:8]
eigvals, eigvecs, lambda_alpha = linear_stability(params)
#%%
# Dv array changes the height (bigger -> higher)
# Du array shifts the height (bigger -> left)
for i in range(3):
    Dv_array = [1, 5, 10]
    Du_array = [0.1, 0.2, 0.3]
    params = valid_params[752][:8]
    params = params[:6] + (0.165, Dv_array[i])
    #params = params[:6] + (Du_array[i], Du_array[i]*20)
    eigvals, eigvecs, lambda_alpha = linear_stability(params)
    valid = eigvals < -0.000001
    plt.plot(np.log(-eigvals[valid]), lambda_alpha.real[valid], 'o')

#plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("Laplacian eigenvalue Λ")
plt.ylabel("Growth rate Re(λ)")
plt.title("Dispersion Relation")

np.any(lambda_alpha.real > 0)
np.sum(lambda_alpha.real > 0)
# %%
params = valid_params[404][:8]

model = models.linear
grid_size = 20
a, b, c, d, e, w, Du, Dv = params
f, g = model(a, b, c, d, e, w)

def eigvalue_roots(params):
    a, b, c, d, e, w, Du, Dv = params
    det_J = a * e - b * d
    A = Dv
    B = e + a * Dv/Du
    C = det_J / Du

    root_pos = (-B + np.sqrt(B**2 - 4 * A * C))/(2*A)
    root_neg = (-B - np.sqrt(B**2 - 4 * A * C))/(2*A)
    return [root_pos, root_neg]

# 752 , 404, 847

#c, w = 0, 0
critical_ratio = (a*e - 2 * b * d + 2 * np.sqrt( b * d * ( b * d - a*e)))/a**2
Dv = Du * critical_ratio # sigma = Dv/Du

critical_eigenvalue = ((a - e)*critical_ratio - np.sqrt(abs(b)*d*critical_ratio+0j)*(critical_ratio+1))/(Du * critical_ratio*(critical_ratio-1))

eigvals, eigvecs, lambda_alpha = linear_stability(params)

valid = eigvals < -0.000001

log_eigval = np.log(-eigvals[valid])

# where would it touch the x axis
eigenvalue_roots = eigvalue_roots(params)

# expansion of line to see trend
x_axis = np.arange(int(eigenvalue_roots[1]), int(eigenvalue_roots[0]), 1)
lambda_alpha_all = growth_rate(params, x_axis)



# sort_idx = log_eigval.argsort()[::-1] 
# log_eigval = log_eigval[sort_idx] 
branch_1, branch_2, k_unique = plots.dispersion_relation_plot(f, g, Du, Dv, plot=False)


plt.plot(eigenvalue_roots, [0, 0], 'r*')
#plt.plot(-(k_unique**2), branch_1, 'o', label='From k (dispersion)', alpha=0.7)
plt.plot(eigvals[valid], lambda_alpha.real[valid], 'o')
plt.plot(x_axis, lambda_alpha_all.real, 'k--')
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("Laplacian eigenvalue Λ")
plt.ylabel("Growth rate Re(λ)")
plt.title("Dispersion Relation")


# %%
a, b, c, d, e, w, Du, Dv = valid_params[0][:8]
#c, w = 0, 0
model = models.linear
f, g = model(a, b, c, d, e, w)

shape = shapes.grid_check
initial_conditions = np.ones((grid_size ** 2, 2)) /2

u, v = discrete_turning.simulation(f, g, Du, Dv, n = 50000, initial_conditions=initial_conditions)
# %%

critical_ratio = a*e - 2 * b * d + 2 * np.sqrt( b * d * ( b * d - a*e))/a**2
print(critical_ratio)

# d needs to be negative or this is imaginary?
critical_eigenvalue = ((a - e)*critical_ratio - np.sqrt(abs(b)*d*critical_ratio+0j)*(critical_ratio+1))/(Du * critical_ratio*(critical_ratio-1))
B_critical = (-a + e + (critical_ratio -1)*Du * critical_eigenvalue + np.sqrt(4*b*d+(a-e-(critical_ratio -1)*(Du * critical_eigenvalue))**2))/(2*b)

# the roots of c(eigval_alpha) where a growth rate**2 + b growth rate + c = 0

# turing instability -> some of the eigenvalues are in this range
sigma = Dv/Du
root1 = -(e + a * sigma) + np.sqrt((e + sigma * a)**2 - 4 * sigma * (a*e-b*d))/2*sigma*Du
root2 = -(e + a * sigma) - np.sqrt((e + sigma * a)**2 - 4 * sigma * (a*e-b*d))/2*sigma*Du



#%% Comparison of growth rate found from laplacian and found from fourier

branch_1, branch_2, k_unique = plots.dispersion_relation_plot(f, g, Du, Dv)

plt.figure(figsize=(7, 5))
plt.plot(-(k_unique**2), branch_1, 'o', label='From k (dispersion)', alpha=0.7)
plt.plot(eigvals, lambda_alpha, '.', label='From Λ (linear_stability)', alpha=0.7)

plt.axhline(0, linestyle='--', color='gray')
plt.xlabel("Laplacian eigenvalue Λ or wave number -k**2")
plt.ylabel("Growth rate Re(λ)")
plt.title("Overlay of Dispersion Relations")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


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