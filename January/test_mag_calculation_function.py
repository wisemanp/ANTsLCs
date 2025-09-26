import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from astropy import constants as const
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
import scipy.optimize as opt
from tqdm import tqdm
import sys

sys.path.append("C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS") # this allows us to import plotting preferences and functions
from plotting_preferences import band_colour_dict, band_ZP_dict, band_obs_centwl_dict, ANT_redshift_dict, ANT_luminosity_dist_cm_dict, MJDs_for_fit, band_marker_dict, band_offset_dict, band_offset_label_dict, MJD_xlims
from functions import load_ANT_data, ANT_data_L_rf, bin_lc, fit_BB_across_lc, chisq, polyfit_lc, L_rf_to_mag, ANT_data_mags





# load in the data
lc_df_list, transient_names, list_of_bands = load_ANT_data()

# calculate the rest frame luminosity + emitted central wavelength
add_lc_df_list = ANT_data_L_rf(lc_df_list, transient_names, ANT_redshift_dict, ANT_luminosity_dist_cm_dict, band_ZP_dict, band_obs_centwl_dict)

# bin up the light curve into 1 day MJD bins
MJD_binsize = 1
binned_df_list = bin_lc(add_lc_df_list, MJD_binsize)


# after binning the light curve's rest frame luminosity, re-calculate the apparent magnitude of the binned lightcurve
binned_df_list = ANT_data_mags(binned_df_list, transient_names, ANT_redshift_dict, ANT_luminosity_dist_cm_dict, band_ZP_dict)



# choose ONE light curve
idx = 10
ANT_name = transient_names[idx]
original_ANT_df = lc_df_list[idx]
binned_ANT_df = binned_df_list[idx]
ANT_bands = list_of_bands[idx]

ANT_xlim = MJD_xlims[ANT_name]
fig = plt.figure(figsize = (16, 7.5))
for b in ANT_bands:
    og_b_df = original_ANT_df[original_ANT_df['band'] == b].copy()
    bin_b_df = binned_ANT_df[binned_ANT_df['band'] == b].copy()   

    b_colour = band_colour_dict[b]
    b_marker = band_marker_dict[b]
    b_offset = band_offset_dict[b]
    b_label = band_offset_label_dict[b]

    plt.errorbar(og_b_df['MJD'], (og_b_df['mag'] + b_offset), yerr = og_b_df['magerr'], fmt = b_marker, mfc = b_colour, c = 'k', alpha = 0.5, markeredgecolor = 'k', markeredgewidth = '0.5', capsize = 5, capthick = 15)
    plt.errorbar(bin_b_df['wm_MJD'], (bin_b_df['mag'] + b_offset), yerr = bin_b_df['magerr'], xerr = (bin_b_df['MJD_lower_err'], bin_b_df['MJD_upper_err']), fmt = b_marker, c = b_colour,
                  markeredgecolor = 'k', markeredgewidth = '0.5', label = b_label, markersize = 4)
    
fig.gca().invert_yaxis()
plt.grid()
plt.legend()
plt.xlim(ANT_xlim)
plt.xlabel('MJD')
plt.ylabel('apparent mag')
plt.title(f'{ANT_name}, I put the band original data in the large feint markers, and their errorbars are black with \n caps on the top, so you can see that the errors match, too')
plt.show()


