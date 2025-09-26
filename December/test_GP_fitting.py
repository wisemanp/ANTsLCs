

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.optimize as opt
from matplotlib import cm
from matplotlib.colors import Normalize
sys.path.append("C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS") # this allows us to import plotting preferences and functions
from plotting_preferences import band_colour_dict, band_ZP_dict, band_obs_centwl_dict, ANT_redshift_dict, ANT_luminosity_dist_cm_dict, MJDs_for_fit
from functions import load_ANT_data, ANT_data_L_rf, bin_lc, polyfit_lc, fit_BB_across_lc, chisq

# load in the data
lc_df_list, transient_names, list_of_bands = load_ANT_data()

# calculate the rest frame luminosity + emitted central wavelength
add_lc_df_list = ANT_data_L_rf(lc_df_list, transient_names, ANT_redshift_dict, ANT_luminosity_dist_cm_dict, band_ZP_dict, band_obs_centwl_dict)

# bin up the light curve into 1 day MJD bins
MJD_binsize = 1
binned_df_list = bin_lc(add_lc_df_list, MJD_binsize)

# fitting ONE light curve
idx = 0
ANT_name = transient_names[idx]
ANT_df = binned_df_list[idx]
ANT_bands = list_of_bands[idx]

plt.figure(figsize = (16, 7.5))
for b in ANT_bands:
    if (b == 'WISE_W1') or (b == 'WISE_W2'):
        continue

    band_df = ANT_df[ANT_df['band'] == b].copy() # selecting one band to Gausian fit
    # Example data: MJD, flux, flux_error
    MJD = np.array(band_df['wm_MJD'].copy())
    L_scaledown = 1e-41
    L = np.array(band_df['wm_L_rf'].copy()) * L_scaledown
    L_error = np.array(band_df['wm_L_rf_err'].copy()) * L_scaledown


    # ENTIRELY FROM CHAT GPT
    import numpy as np
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF



    # Define kernel without a fixed noise component
    kernel = 1*RBF(length_scale=50)

    # Fit the GP model with individual error variances for each data point
    gp = GaussianProcessRegressor(kernel=kernel, alpha=L_error**2)
    gp.fit(MJD[:, np.newaxis], L)

    # Make predictions
    MJD_pred = np.linspace(min(MJD), max(MJD), 5000)
    L_pred, L_pred_std = gp.predict(MJD_pred[:, np.newaxis], return_std=True)
    L_pred = L_pred/L_scaledown
    L_pred_std = L_pred_std/L_scaledown
    # Plotting
    L = L / L_scaledown
    L_error = L_error/L_scaledown

    
    plt.errorbar(MJD, L, yerr=L_error, fmt='o', label = b, markeredgecolor = 'k', markeredgewidth = '0.5', c = band_colour_dict[b])
    plt.fill_between(MJD_pred, L_pred - 2*L_pred_std, L_pred + 2*L_pred_std, alpha=0.2, color = band_colour_dict[b])
    plt.plot(MJD_pred, L_pred, c =  band_colour_dict[b])#, label='GP fit')
plt.legend()
plt.xlabel('MJD')
plt.ylabel('L_rf')
plt.show()
