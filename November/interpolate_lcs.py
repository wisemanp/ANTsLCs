import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import sys
import scipy.interpolate as interp
sys.path.append("C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS") # this allows us to access the plotting_preferences.py file 
from November.plotting_preferences import band_colour_dict, band_marker_dict, band_offset_dict, band_ZP_dict, band_obs_centwl_dict, ANT_redshift_dict, band_offset_label_dict


##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################



def weighted_mean(data, errors):
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    if not isinstance(errors, np.ndarray):
        errors = np.array(errors)
    
    if len(errors) == 0: # if the bin has no data within it, then the weighted mean and its error = NaN
        wm = pd.NA
        wm_err = pd.NA

    else: # if the bin has data within it, then take the weighted mean
        weights = 1/(errors**2)
        wm = np.sum(data * weights) / (np.sum(weights))
        wm_err = np.sqrt( 1/(np.sum(weights)) )

    return wm, wm_err


##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################


""" 
def wm_same_band_MJD(df, df_bands):
    
    for b in df_bands:
        b_df = df[df['band'] == b].copy()
        duplicates = df.duplicated(subset = ['MJD'], keep = 'False') # a boolean list of whether this row contains an MJD duplicate
        duplicate_df = df[duplicates].copy()

        df_no_duplicates = df[~duplicates].copy() # the '~' inverts boolean values, so this is a dataframe with ALL duplicate data removed

        if len(duplicate_df['MJD']) > 0:
            unique_MJDs = duplicate_df['MJD'].unique()

            for mjd in unique_MJDs:
                mjd_df = duplicate_df[duplicate_df['MJD'] == mjd].copy()
                wm_mag




    return
 """


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

    if ANT_name == 'ZTF22aadesap':
        fig = plt.figure(figsize = (16, 7.5))
        interp_MJD_min = 59600
        interp_MJD_max = 60400
        for band in bands_present:
            band_df = lc_df[lc_df['band'] == band].copy()
            # the reference band data, from which we can interpolate. We can't take the whole light curve because sometimes we have a few straggler datapoints like 500 days away
            # and we don't want to interpolate across that 500 day gap
            ref_band_df = band_df[(band_df['MJD'] > interp_MJD_min) & (band_df['MJD'] < interp_MJD_max)].copy() 
            MJD_for_interp = np.arange(ref_band_df['MJD'].min(), ref_band_df['MJD'].max(), 5) # interpolate at these MJD values
            
            interp_function = interp.interp1d(band_df['MJD'], band_df['mag'], kind = 'linear') # this function is like mag as a function of MJD, so we will input the MJD values to interpolate at and it will give us the mag
            interp_mag = interp_function(MJD_for_interp) # interpolated mag values at MJD_for_interp

            band_color = band_colour_dict[band]
            band_offset = band_offset_dict[band]
            band_label = band_offset_label_dict[band]

            plt.scatter(MJD_for_interp, (interp_mag + band_offset), marker = 'P', color = band_color, label = f'{band_label} interp', edgecolors = 'k', linewidths = 0.5, s = 5)
            plt.errorbar(band_df['MJD'], (band_df['mag'] + band_offset), yerr = band_df['magerr'], color = band_color, fmt = 'o', label = band_label, 
                         markeredgecolor = 'k', markeredgewidth = '1.0', markersize = 6)
            
        plt.xlabel('MJD')
        plt.ylabel('apparent mag')
        fig.gca().invert_yaxis()
        plt.legend(loc = 'lower right', bbox_to_anchor = (1.2, 0.0))
        plt.title(f'{ANT_name}  interpolating the light curve')
        plt.xlim((59500, 60500))
        fig.subplots_adjust(top=0.885,
                            bottom=0.09,
                            left=0.035,
                            right=0.84,
                            hspace=0.2,
                            wspace=0.2)
        plt.grid()
        plt.show()







