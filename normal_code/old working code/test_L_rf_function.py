import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from astropy import constants as const
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM

import sys 
sys.path.append("C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS") # this allows us to access the plotting_preferences.py file 
from plotting_preferences import band_colour_dict, band_marker_dict, band_offset_dict, band_ZP_dict, band_obs_centwl_dict, band_offset_label_dict, ANT_redshift_dict, ANT_luminosity_dist_cm_dict
from functions import load_ANT_data, ANT_data_L_rf







##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################


#""" 
# test the L_rf function ----------------------------------------------------------------------------------------------------------------------------------------
lc_df_list, transient_names, list_of_bands = load_ANT_data()

# modified ANT dataframes to add rest frame luminosity, its error and the observed band's central wavelength converted into the rest frame
mod_lc_df_list = ANT_data_L_rf(lc_df_list, transient_names, ANT_redshift_dict, ANT_luminosity_dist_cm_dict, band_ZP_dict, band_obs_centwl_dict) 

print(mod_lc_df_list[0])

"""
ANT = 'ZTF20acvfraq'
ANT = 'ZTF20abodaps'
ANT = 'ZTF22aadesap'
idx = transient_names.index(ANT)
ANT_lc = mod_lc_df_list[idx]
ANT_name = transient_names[idx]
ANT_bands = list_of_bands[idx]


plt.figure(figsize = (16, 7.5))

for i, band in enumerate(ANT_bands):
    band_df = ANT_lc[ANT_lc['band'] == band].copy()
    band_marker = band_marker_dict[band]
    band_color = band_colour_dict[band]
    band_offset = band_offset_dict[band]
    band_label = band_offset_label_dict[band]

    plt.errorbar(band_df['MJD'], band_df['L_rest'], yerr = band_df['L_rest_err'], label = f'{band} in rest-frame', fmt = band_marker, c = band_color, linestyle = 'None', 
                 markeredgecolor = 'k', markeredgewidth = '0.5')
    
plt.xlabel('MJD')
plt.ylabel('L_rf / ergs/s/Angstrom')
plt.title(f'{ANT_name} testing the rest frame luminosity function')
plt.grid()
plt.legend()
plt.show()
 """






