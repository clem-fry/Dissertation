import models
import evaluation
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
    valid_params = evaluation.suggest_parameters(param_ranges, model, n_trials = 5000)

    with open('valid_params_19june.txt', 'w') as f:
        for item in valid_params:
            f.write(str(item) + '\n')

def score_params(valid_params):
    scored_params = []

    for p, param in enumerate(valid_params):
        a, b, c, d, e, w, Du, Dv, peak, peak_lambda, peak_k, band_width, area = param

        if not peak:
            continue  # skip if no dispersion peak

        # Compute score
        score = peak_lambda * band_width * area
        #score = criteria
        # Append as (score, full_param_tuple)
        scored_params.append((score, p, param))

    # Sort by score descending (best patterns first)
    scored_params.sort(reverse=True, key=lambda x: x[0])

    # Print top 5
    for score, p, param in scored_params[:5]:
        print(f"Score: {score:.4f}, Index: {p} Parameters: {param}")

    return scored_params

def plot_dispersion_turing(valid_params, indicies, initial_conditions=None, grid_size = 20):
    for i in indicies:
        a, b, c, d, e, w, Du, Dv = valid_params[i][0:8]
        f, g = models.linear(a, b, c, d, e, w)
        if np.any(initial_conditions == None):
            u, v = discrete_turning.simulation(f, g, Du, Dv, n = 30000, show=False, save=True, grid_size=grid_size, save_path=f"plots/simulation_plot_{i}.png")
        else:
            u, v = discrete_turning.simulation(f, g, Du, Dv, n = 30000, initial_conditions = initial_conditions, grid_size=grid_size, show=False, save=True, save_path=f"plots/simulation_plot_{i}.png")
        
        print(evaluation.turing_check(valid_params[i][0:8]))
        _ = evaluation.dispersion_relation_plot(f, g, Du, Dv, plot=False, save=True, save_path=f"plots/dispersion_plot_{i}.png")

    fig, axes = plt.subplots(5, 2, figsize=(12, 20))

    for idx, i in enumerate(indicies):
        sim_img = mpimg.imread(f"plots/simulation_plot_{i}.png")
        disp_img = mpimg.imread(f"plots/dispersion_plot_{i}.png")
        
        axes[idx, 0].imshow(sim_img)
        axes[idx, 0].axis('off')
        axes[idx, 0].set_title(f'Simulation {i}')
        
        axes[idx, 1].imshow(disp_img)
        axes[idx, 1].axis('off')
        axes[idx, 1].set_title(f'Dispersion Relation {i}')

    plt.tight_layout()
    plt.show()

def loss_function(params):
    a, b, c, d, e, w, Du, Dv = params
    f, g = models.linear(a, b, c, d, e, w)
    if not np.any(evaluation.turing_check(f, g, Du, Dv)):
        return 10e10
    
    real_parts, k_vals = evaluation.dispersion_relation_plot(f, g, Du, Dv, plot=False)
    peak1, _ = find_peaks(real_parts[:, 0])
    peak2, _ = find_peaks(real_parts[:, 1])
    peaks = np.append(peak1, peak2)
    if len(peaks) == 0:
        return 10e10
    
    else:
        return -np.max(peaks)
    
def check_for_instability(f, g, Du, Dv):
    if np.all(evaluation.turing_check(f, g, Du, Dv)):
        real_parts, k_vals = evaluation.dispersion_relation_plot(f, g, Du, Dv, plot=False)
        peak1, _ = find_peaks(real_parts[:, 0])
        peak2, _ = find_peaks(real_parts[:, 1])

        turing_flag = (
            (len(peak1) > 0 and np.any(real_parts[peak1, 0] > 0)) or
            (len(peak2) > 0 and np.any(real_parts[peak2, 1] > 0))
        )

        if turing_flag:
            print("Turing-like instability detected")
            print(a, b, c, d, e, w)
            _ = evaluation.dispersion_relation_plot(f, g, Du, Dv, plot=True)
            u, v = discrete_turning.simulation(f, g, Du, Dv, n=n, show=True)
                #break

        else:
            _ = evaluation.dispersion_relation_plot(f, g, Du, Dv, plot=True)
            #u, v = discrete_turning.simulation(f, g, Du, Dv, n=n, show=True)
            print('no peak')

    else:
        print('Not locally unstable')