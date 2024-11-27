import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import sys
import scipy.interpolate as interp
sys.path.append("C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS") # this allows us to access the plotting_preferences.py file 
from plotting_preferences import band_colour_dict, band_marker_dict, band_offset_dict, band_ZP_dict, band_obs_centwl_dict, ANT_redshift_dict, band_offset_label_dict
from functions import bin_lc, chisq



##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################

# loading in the files
folder_path = "C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS Data/modified Phil's lightcurves" # folder path containing Phil's light curve data files
lc_df_list = []
transient_names = []
list_of_bands = []
for file in os.listdir(folder_path): # file is a string of the file name such as 'file_name.dat'
    file_path = os.path.join(folder_path, file) 
    file_df = pd.read_csv(file_path, delimiter = ',') # the lightcurve data in a dataframe
    lc_df_list.append(file_df)

    trans_name = file[:-7] # the name of the transient
    transient_names.append(trans_name)

    trans_bands = file_df['band'].unique() # the bands in which this transient has data for
    list_of_bands.append(trans_bands)





for i, lc_df in enumerate(lc_df_list):
    bands_present = list_of_bands[i]
    ANT_name = transient_names[i]

    if ANT_name == 'ZTF22aadesap':# or ANT_name == 'ZTF18aczpgwm':
        #binned_lc = bin_lc(lc_df, MJD_binsize = 1.0, drop_na_bins = True)

        fig = plt.figure(figsize = (16, 7.5))
        interp_MJD_min = 59690
        interp_MJD_max = 60400
        for idx, band in enumerate(bands_present):
            band_df = lc_df[lc_df['band'] == band].copy()
            # the reference band data, from which we can interpolate. We can't take the whole light curve because sometimes we have a few straggler datapoints like 500 days away
            # and we don't want to interpolate across that 500 day gap
            ref_band_df = band_df[(band_df['MJD'] > interp_MJD_min) & (band_df['MJD'] < interp_MJD_max)].copy() 

            # polynomial fit on the light curve
            poly_order = 12
            poly_coefficents = np.polyfit(ref_band_df['MJD'], ref_band_df['mag'], deg = poly_order)
            poly_coefficients = list(poly_coefficents)
            MJD_for_poly = np.arange(ref_band_df['MJD'].min(), ref_band_df['MJD'].max(), 1) # interpolate at these MJD values
            polynomial_fit = np.poly1d(poly_coefficents, r = False)
            poly_mag_for_chi = polynomial_fit(ref_band_df['MJD']) # using the poly fit to predict the real values of mag which we have measured for the ANT to use for chi squared
            poly_red_chi, poly_chi_err = chisq(poly_mag_for_chi, ref_band_df['mag'], ref_band_df['magerr'], M = (poly_order + 1), reduced_chi = True) #reduced chi squared of the polyfit
            poly_mag = polynomial_fit(MJD_for_poly)
            
            band_color = band_colour_dict[band]
            band_offset = band_offset_dict[band]
            band_label = band_offset_label_dict[band]

            plt.errorbar(ref_band_df['MJD'], (ref_band_df['mag'] + band_offset), yerr = ref_band_df['magerr'], color = band_color, fmt = 'o', label = band_label, 
                         markeredgecolor = 'k', markeredgewidth = '1.0', markersize = 6)
            plt.plot(MJD_for_poly, (poly_mag + band_offset), c = band_color, label = f'{band_label}, \nred chi = {poly_red_chi:.3f} +/- {poly_chi_err:.3f}')
            
        plt.xlabel('MJD')
        plt.ylabel('apparent mag')
        fig.gca().invert_yaxis()
        plt.legend(loc = 'lower right', bbox_to_anchor = (1.2, 0.0))
        plt.title(f'{ANT_name}  fitting the light curve, polynomial order = {poly_order}')
        plt.xlim((59500, 60500))
        fig.subplots_adjust(top=0.885,
                            bottom=0.09,
                            left=0.035,
                            right=0.84,
                            hspace=0.2,
                            wspace=0.2)
        plt.grid()
        plt.show()







