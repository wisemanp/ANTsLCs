import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import sys
import scipy.optimize as opt
from tqdm import tqdm
sys.path.append("C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS") # this allows us to import plotting preferences and functions
from plotting_preferences import band_colour_dict, band_ZP_dict, band_obs_centwl_dict, ANT_redshift_dict, ANT_luminosity_dist_cm_dict, MJDs_for_fit
from functions import load_ANT_data, ANT_data_L_rf, bin_lc, polyfit_lc, fit_BB_across_lc


# load in the data
lc_df_list, transient_names, list_of_bands = load_ANT_data()

# calculate the rest frame luminosity + emitted central wavelength
add_lc_df_list = ANT_data_L_rf(lc_df_list, transient_names, ANT_redshift_dict, ANT_luminosity_dist_cm_dict, band_ZP_dict, band_obs_centwl_dict)

# bin up the light curve into 1 day MJD bins
MJD_binsize = 1
binned_df_list = bin_lc(add_lc_df_list, MJD_binsize)

# polyfitting ONE light curve
ANT_name = transient_names[0]
ANT_df = binned_df_list[0]
ANT_bands = list_of_bands[0]
bands_for_BB = [b for b in ANT_bands if (b != 'WISE_W1') and (b != 'WISE_W2')] # remove the WISE bands from the interpolation since we don't want to use this data for the BB fit anyway
interp_lc, plot_poltfit_df = polyfit_lc(ANT_name, ANT_df, fit_order = 5, df_bands = bands_for_BB, trusted_band = 'ZTF_g', fit_MJD_range = MJDs_for_fit[ANT_name],
                        extrapolate = False, b_colour_dict = band_colour_dict, plot_polyfit = True)



BB_fit_results = fit_BB_across_lc(interp_lc, brute = False, curvefit = True)
print()
print()
print()
print()
print(interp_lc['MJD'].unique()[:50])
print()
print(BB_fit_results.head(50))
print()
print()
print()
print()