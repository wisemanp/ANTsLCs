import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS") # this allows us to access the plotting_preferences.py file 
from plotting_preferences import band_colour_dict, band_marker_dict, band_ZP_dict, band_obs_centwl_dict, ANT_redshift_dict, ANT_luminosity_dist_cm_dict
from functions import load_ANT_data, ANT_data_L_rf, bin_lc



# load in the data
lc_df_list, transient_names, list_of_bands = load_ANT_data() 

# produce a list of ANT light curve dataframes with the addition of rest frame luminosity, its error and emitted wavelength columns
mod_lc_df_list = ANT_data_L_rf(lc_df_list, transient_names, ANT_redshift_dict, ANT_luminosity_dist_cm_dict, band_ZP_dict, band_obs_centwl_dict) 

# bin the light curves
MJD_binsize = 1
binned_ANT_lc_list = bin_lc(mod_lc_df_list, MJD_binsize, drop_na_bins = True)
"""  this commented out snippet just shows the problematic ANT datafiles, since some of them have magerr = 0.0 which gives the runtimewarning for the weighted mean function
print()
print()
TEST_PROBLEM_IDX = [14, 15, 16]
for k in TEST_PROBLEM_IDX:
    print(f'{k}: {transient_names[k]}')
    problem_df = lc_df_list[k].copy()
    print(problem_df[problem_df['L_rf_err'] == 0.0])
    print()

 """
# choose an ANT to test the binning funciton on
#ANT = 'ZTF20acvfraq'
#ANT = 'ZTF20abodaps'
ANT = 'ZTF22aadesap'
idx = transient_names.index(ANT)
ANT_lc = mod_lc_df_list[idx].copy()
ANT_name = transient_names[idx]
ANT_bands = list_of_bands[idx]
binned_ANT_lc = binned_ANT_lc_list[idx].copy()


# plot the binned light curve on top of the original light curve to check 
plt.figure(figsize = (16, 7.5))
for band in ANT_bands:
    band_df = binned_ANT_lc[binned_ANT_lc['band'] == band].copy()
    unbinned_band_df = ANT_lc[ANT_lc['band'] == band].copy()
    band_color = band_colour_dict[band]
    band_marker = band_marker_dict[band]
    plt.errorbar(unbinned_band_df['MJD'], unbinned_band_df['L_rf'], yerr = unbinned_band_df['L_rf_err'], fmt = band_marker, c = band_color, alpha = 0.5, linestyle = 'None', 
                 markeredgecolor = 'k', markeredgewidth = '0.5')
    plt.errorbar(band_df['wm_MJD'], band_df['wm_L_rf'], yerr = band_df['wm_L_rf_err'], xerr = (band_df['MJD_lower_err'], band_df['MJD_upper_err']), fmt = band_marker, c = band_color,
                 label = band, linestyle = 'None', markeredgecolor = 'k', markeredgewidth = '1.5')
    


plt.xlabel('MJD')
plt.ylabel('rest frame luminosity')
plt.grid()
plt.legend()
plt.title(f'{ANT_name} binned. MJD binsize = {MJD_binsize}')
plt.show()




