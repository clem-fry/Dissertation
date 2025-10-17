#%%
import models
import parameters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import simulation
import shapes
import importlib
import plots
import analysis
#%%

importlib.reload(simulation)
importlib.reload(models)
importlib.reload(plots)
importlib.reload(shapes)
importlib.reload(parameters)
importlib.reload(analysis)
#%%

video_data_import = pd.ExcelFile('node-results/video-data.xlsx')
ref_data_import = pd.ExcelFile('node-results/references.xlsx')

print(video_data_import.sheet_names)

video_data_sheets = video_data_import.sheet_names
references = np.array(sorted(set(ref.split('_')[0] for ref in video_data_sheets)))


video_data = pd.read_excel(video_data_import, sheet_name=None)
ref_data = pd.read_excel(ref_data_import, sheet_name='Sheet1')
ref_data.set_index('Unnamed: 0', inplace = True)

ref = references[2]

kilobot_repeats = [key for key in video_data.keys() if key.startswith(ref)]


# %%

variables = ref_data.loc[ref]
a, b, c, d, Du, Dv = variables[['a', 'b', 'c', 'd', 'Du', 'Dv']].astype(float)
grid_size = variables[['size']].astype(int).item()
connected = variables[['Full Connected']].astype(bool).item()
dt = variables[['dt']].astype(float).item()

flash_time_scale = 0.85

dt = dt * flash_time_scale

kilobots = video_data[ref]

time_cols = [col for col in kilobots.columns if col.startswith('t')]
u_cols = [col for col in kilobots.columns if col.startswith('u')]

n = int(kilobots[time_cols].max().max()) + 5 # max time

col1 = list(map(float, variables.loc['initial conditions u'].split(',')))
col2 = list(map(float, variables.loc['Unnamed: 8'].split(',')))
initial_conditions_array = np.column_stack((col1, col2))

if connected:
    threshold = 100
else:
    threshold = 1

f, g = models.linear(a, b, 0, c, d, 0)

u0, v0 = parameters.find_steady_state(f, g, initial_guess=(0.0, 0.0))

print(variables[['initial conditions u']])

initial = np.zeros((grid_size**2, 2))
initial[0:grid_size, :] = initial_conditions_array

# iters = 
# init_u = -0.1
# init_v = -0.1
# for i in range(iters):
#     init_u = (f(init_u, init_v) * dt)+init_u
#     init_v = (g(init_u, init_v) * dt) +init_v
#     initial[1][0] = init_u
#     initial[1][1] = init_v

# iters = 0
# init_u = 0.1
# init_v = 0.1
# for i in range(iters):
#     init_u = (f(init_u, init_v) * dt)+init_u
#     init_v = (g(init_u, init_v) * dt) +init_v
#     initial[0][0] = init_u
#     initial[0][1] = init_v

# initial[0][0] = 
# initial[0][1] = 0.11
# initial[1][0] = (f(init_u, init_v) * dt)+init_u
# initial[1][1] = (g(init_u, init_v) * dt) +init_v

u_vector, v_vector, (u_log, v_log) = simulation.simulation(f, g, Du, Dv, grid_size = grid_size, n = n, initial_conditions = initial, dt = dt, shape = shapes.point_number, threshold = threshold, show=False)
# node_n = 0
x_axis = np.arange(0, n, 1)

for repeat in kilobot_repeats:
    time_cols = [col for col in video_data[repeat].columns if col.startswith('t')]

    for i in range(len(time_cols)):
        plt.plot(video_data[repeat][time_cols[i]].values, video_data[repeat][u_cols[i]].values, 'x')

plt.plot(x_axis, u_log.T)
plt.xlabel('time (s)')
plt.title(f"{grid_size} nodes fully connected with params ")
plt.ylabel('u values')
plt.grid(True)
plt.show()


# %%

for i in range(len(time_cols)):
    all_repeat_u = []
    all_repeat_t = []

    for repeat in kilobot_repeats:
        df = video_data[repeat]
        all_repeat_t.append(df[time_cols[i]].values)
        all_repeat_u.append(df[u_cols[i]].values)

    # Convert to numpy arrays for easier handling
    all_repeat_t = np.array(all_repeat_t)
    all_repeat_u = np.array(all_repeat_u)

    # Compute the mean and std dev
    mean_t = np.nanmean(all_repeat_t, axis=0)
    mean_u = np.nanmean(all_repeat_u, axis=0)
    std_t = np.nanstd(all_repeat_t, axis=0)

    # Plot with error bars
    plt.errorbar(
        mean_t, mean_u,
        xerr=std_t,
        fmt='o',
        capsize=3,
        elinewidth=1,
        label=f'u{i}'
    )



# %%
