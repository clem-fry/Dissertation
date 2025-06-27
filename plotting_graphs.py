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
#%%
importlib.reload(simulation)
importlib.reload(models)
importlib.reload(plots)
importlib.reload(shapes)
importlib.reload(parameters)
importlib.reload(analysis)

#%%

valid_params = parameters.load_params('valid_params_25june.txt')
# remove any oscillations!!
valid_params = [p[:5] for p in valid_params if p[-1] == True]

#%% PLOT WITH CHOSEN CRITICAL EIGENVALUE
param_nums = (12, 13, 14, 15)
y_axis = []
x_axis = np.arange(-5, 0, 0.1)
eig = -2

for param in param_nums:
    a, b, d, e, sigma = valid_params[param]
    J = parameters.non_spacial_jacobian(a, b, d, e)
    Du = parameters.epsilon_for_eigenvalue(J, eig, sigma)
    Dv = sigma * Du

    c, w = 0, 0
    params = a, b, c, d, e, w, Du, Dv
    
    y = analysis.growth_rate(params, x_axis)
    y_axis.append(y)

for i in range(len(param_nums)):
    plt.plot(x_axis, y_axis[i], label=f'Params={param_nums[i]:.2f}')

plt.axhline(y=0, color='grey', linestyle='--', linewidth=1)
plt.grid(True)
plt.xlabel('Eigenvalues')
plt.ylabel('Lambda Alpha: Growth Rate')
param_labels = [str(p) for p in param_nums]
plt.legend(param_labels, title='params')
plt.show()

#%% COMPARING GRADIENTS FOR FIXED PEAK

params = valid_params[10]
a, b, d, e, sigma = params
J = parameters.non_spacial_jacobian(a, b, d, e)
peak = 1
sigma = parameters.find_sigma_for_growth(J, peak)
eigenvalue = -4
Du = parameters.epsilon_for_eigenvalue(J, eigenvalue, sigma)
Dv = sigma * Du

fu_array = np.array([0.1, 0.5, 1, 1.5, 2]) * 10
lambda_alpha_list = []

for i in range(len(fu_array)):
    a = fu_array[i]
    J = parameters.non_spacial_jacobian(a, b, d, e)
    peak = 2
    sigma = parameters.find_sigma_for_growth(J, peak)
    eigenvalue = -4
    Du = parameters.epsilon_for_eigenvalue(J, eigenvalue, sigma)
    Dv = sigma * Du
    eigvals = np.arange(-10, 0, 0.1)
    parameter = a, b, 0, d, e, 0, Du, Dv
    lambda_alpha = analysis.growth_rate(parameter, eigvals)
    lambda_alpha_list.append(lambda_alpha.real)

for i in range(len(fu_array)):
    plt.plot(eigvals, lambda_alpha_list[i], label=f'gv={gv_array[i]:.2f}')

plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("Eigenvalues")
plt.ylabel("Lambda Alpha")
plt.legend()
plt.title("Stability curves for varying gv , Dv = 13, Du = 0.1, params = 20")
plt.grid(True)
plt.show()


#%% COMPARING CHANGING DU

params = valid_params[20]
a, b, d, e, sigma = params
# Du_array = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
# sigma = 130
# Dv_array = np.array([0.1, 0.2, 0.3, 0.4, 0.5]) * sigma
Dv = 13
# Dv_array = np.array([30, 40, 50, 60, 70])
Du = 0.1

gu_array = np.array([0.1, 0.5, 1, 1.5, 2]) * 10
eigvals_list = []
lambda_alpha_list = []

for i in range(len(gu_array)):
    #Du = Du_array[i]
    #Dv = Dv_array[i]
    params = a, b, 0, d, gv_array[i], 0,Du, Dv
    eigvals, eigvecs, lambda_alpha = analysis.linear_stability(params)
    eigvals_list.append(eigvals)
    lambda_alpha_list.append(lambda_alpha.real)

for i in range(len(gv_array)):
    plt.plot(eigvals_list[i], lambda_alpha_list[i], label=f'gv={gv_array[i]:.2f}')

plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("Eigenvalues")
plt.ylabel("Lambda Alpha")
plt.legend()
plt.title("Stability curves for varying gv , Dv = 13, Du = 0.1, params = 20")
plt.grid(True)
plt.show()

#%% COMPARING EIGENVALUES OF DIFFERENT NUMBER OF NODES
# 612 - oscillation
optimal = parameters.find_suitable_params(-2, 0.1, valid_params, 20)
a, b, d, e = optimal["params"]
sigma = optimal["sigma"]
Du = optimal["epsilon"]
Dv = sigma * Du
c, w = 0, 0

J = parameters.non_spacial_jacobian(a, b, d, e)
Du = parameters.epsilon_for_eigenvalue(J, -2, sigma)
Dv = sigma * Du

c, w = 0, 0
params = a, b, c, d, e, w, Du, Dv

lambda_alpha_one, eigvals_one = plots.dispersion_relation_plot(params, grid_size=1, plot=False,  shape = shapes.line_check, model = models.linear)
lambda_alpha_two, eigvals_two = plots.dispersion_relation_plot(params, grid_size=2, plot=False,  shape = shapes.line_check, model = models.linear)
lambda_alpha_three, eigvals_three = plots.dispersion_relation_plot(params, grid_size=3, plot=False,  shape = shapes.line_check, model = models.linear)
lambda_alpha_four, eigvals_four = plots.dispersion_relation_plot(params, grid_size=4, plot=False,  shape = shapes.line_check, model = models.linear)
lambda_alpha_five, eigvals_five = plots.dispersion_relation_plot(params, grid_size=5, plot=False,  shape = shapes.line_check, model = models.linear)
lambda_alpha_six, eigvals_six = plots.dispersion_relation_plot(params, grid_size=6, plot=False,  shape = shapes.line_check, model = models.linear)

#%% PLOT OF GROWTH RATE FOR SYSTEMS
x_axis = np.arange(np.floor(np.min(np.real(eigvals_six))), 0.1, 0.1)
y_axis = analysis.growth_rate(params, x_axis)

plt.plot(eigvals_one, lambda_alpha_one, 'x')
plt.plot(eigvals_two, lambda_alpha_two, '*')
plt.plot(eigvals_three, lambda_alpha_three, 'o')
plt.plot(eigvals_four, lambda_alpha_four, 'x')
plt.plot(eigvals_five, lambda_alpha_five, '*')
plt.plot(eigvals_six, lambda_alpha_six, 'o')

plt.axhline(y=0, color='grey', linestyle='--', linewidth=1)
plt.plot(x_axis, y_axis, color='green', linestyle='--')
plt.grid(True)
plt.xlabel('Eigenvalues')
plt.ylabel('Lambda Alpha: Growth Rate')
plt.legend(['one', 'two', 'three', 'four', 'five', 'six'], title='Number of fully connected nodes')

plt.show()

# %% PLOT OF SPACIAL INSTABILITY -> unstable when negative

x_axis = np.arange(-10, 0, 1)
y_axis = Du * Dv * x_axis ** 2 + (Du * e + Dv * a) * x_axis + (a*e - b*d)

plt.plot(x_axis, y_axis, 'x')
plt.axhline(y=0, color='grey', linestyle='--', linewidth=1)
plt.xlabel('Theoretical Eigenvalues')
plt.ylabel('C(growth rate)')
plt.grid(True)
plt.show()

# %% PLOT OF DU AT CRITICAL RATIO
params = valid_params[3]
a, b, d, e, sigma = params
c, w = 0, 0
Du_array = np.array([0.3]) #, 0.2, 0.3, 0.4, 0.5
#critical_ratio = (a*e - 2*b*d + 2*np.sqrt(b*d*(b*d - a*e))) / a**2
#Dv_array = Du_array * critical_ratio
sigma = 40000
eigvals_list = []
lambda_alpha_list = []

for i in range(len(Du_array)):
    Du = Du_array[i]
    Dv =sigma * Du
    params_send = a, b, c, d, e, w, Du, Dv
    eigvals, eigvecs, lambda_alpha = analysis.linear_stability(params_send)
    eigvals_list.append(eigvals)
    lambda_alpha_list.append(lambda_alpha.real)
for i in range(len(Du_array)):
    plt.plot(eigvals_list[i], lambda_alpha_list[i], label=f'Du={Du_array[i]:.2f}')

plt.axhline(0, color='gray', linestyle='--')
plt.xlabel("Eigenvalues")
plt.ylabel("Lambda Alpha")
plt.legend()
plt.title("Stability curves for varying Du")
plt.grid(True)
plt.show()

# %%
optimal = parameters.find_suitable_params(-2, 1, valid_params, 10)
a, b, d, e = optimal["params"]
sigma = optimal["sigma"]
Du = optimal["epsilon"]
Dv = sigma * Du
c, w = 0, 0
f, g = models.linear(a, b, c, d, e, w)
grid_size = 5
#initial = np.random.rand(grid_size**2, 2)
initial = np.ones((grid_size**2, 2)) /2

u0, v0 = parameters.find_steady_state(f, g, initial_guess=(1.0, 0.0))
initial = np.tile([u0, v0], (grid_size ** 2, 1)) 
initial += np.random.normal(loc=0.0, scale=1e-6, size=initial.shape)
n = 200
u, v, log = simulation.simulation(f, g, Du, Dv, grid_size=grid_size, n = n, dt = 0.0001, initial_conditions = initial, shape = shapes.line_check, show=False)
# log size = 2 * nodes * n
u_log = np.array(log[0])
v_log = np.array(log[1])

# node_n = 0
plt.plot(v_log.T)
plt.xlabel('time')
plt.title(f"{grid_size} nodes fully connected with parameter set {param_number}")
plt.ylabel('u values')
plt.grid(True)
plt.show()


# %%
