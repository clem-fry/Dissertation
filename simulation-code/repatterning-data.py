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
from scipy.optimize import curve_fit
import pandas as pd
import seaborn as sns

#%%
importlib.reload(simulation)
importlib.reload(models)
importlib.reload(plots)
importlib.reload(shapes)
importlib.reload(parameters)
importlib.reload(analysis)

#%%

data = np.load('ref23.npz')
print(data.files)

#%%

array_dict = {}

for i, (name, array) in enumerate(data.items()):
    array_grid = array.reshape((10, 10))
    array_dict[name] = array_grid
    plt.imshow(array_grid, cmap='bwr')
    plt.axis('off')
    plt.show()
    #_ = analysis.plot_wavelength_spectrum(array_grid, 
    
    # wavelength = analysis.strongest_wavelength(array_grid, plot=False)
    # kilobot_wavelengths.append(wavelength)
    # labels_kilo.append((eigen, "Kilobot"))

# %%

_ = analysis.plot_wavelength_spectrum(data['repeat4_phase1'].reshape((10, 10)), plot_spectrum = True, log_scale=False)
wavelength1 = analysis.strongest_wavelength(data['repeat4_phase1'].reshape((10, 10)), plot=False)
print(wavelength1)

_ = analysis.plot_wavelength_spectrum(data['repeat4_phase2'].reshape((10, 10)), plot_spectrum = True, log_scale=False)
wavelength2 = analysis.strongest_wavelength(data['repeat4_phase2'].reshape((10, 10)), plot=False)
print(wavelength2)

_ = analysis.plot_wavelength_spectrum(data['repeat4_phase3'].reshape((10, 10)), plot_spectrum = True, log_scale=False)
wavelength3 = analysis.strongest_wavelength(data['repeat4_phase3'].reshape((10, 10)), plot=False)
print(wavelength3)

# %%
