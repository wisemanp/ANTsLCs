import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS") # this allows us to import plotting preferences and functions
from plotting_preferences import band_colour_dict, band_ZP_dict, band_obs_centwl_dict, ANT_redshift_dict, ANT_luminosity_dist_cm_dict, MJDs_for_fit
from functions import load_ANT_data, ANT_data_L_rf, bin_lc, polyfit_lc

##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################


# load in the ANT data
lc_df_list, transient_names, list_of_bands = load_ANT_data()

# calculate the rest frame luminosities and emitted wavelength equivalent of the observed bands
mod_lc_df_list = ANT_data_L_rf(lc_df_list, transient_names, ANT_redshift_dict, ANT_luminosity_dist_cm_dict, band_ZP_dict, band_obs_centwl_dict)

# bin the lightcurve data for the rest frame luminosities
MJD_binsize = 1
bin_lc_df_list = bin_lc(mod_lc_df_list, MJD_binsize)
polyfit_order = 6
want_plot = True
#ANT = 'ZTF22aadesap'
#ANT = 'ZTF19aailpwl'
# now, we want to fit polynomials to each band of the binned up light curves
for i, ANT in enumerate(transient_names):
    if i in [14, 15, 16]: # these ANT light curves have some magerr = 0.0, which causes issues with taking the weighted mean of L_rf so idk if the binning funciton would actually work
        continue

    if i == 1:
        reference_band = 'ZTF_g'
        print(f'{i}, {ANT}')
        print()
        ANT_bands = list_of_bands[i]
        binned_ANT_df = bin_lc_df_list[i].copy()
        poly_interp_df, plot_polyfit_df = polyfit_lc(ANT, binned_ANT_df, fit_order = 5, df_bands = ANT_bands, trusted_band = reference_band, fit_MJD_range = MJDs_for_fit[ANT], extrapolate = False, b_colour_dict = band_colour_dict, plot_polyfit = True)








