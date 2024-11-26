import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import sys
import scipy.interpolate as interp
sys.path.append("C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS") # this allows us to access the plotting_preferences.py file 
from plotting_preferences import band_colour_dict, band_marker_dict, band_offset_dict, band_ZP_dict, band_obs_centwl_dict, ANT_redshift_dict, band_offset_label_dict


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





def bin_lc(lc_df, MJD_binsize, drop_na_bins = True):
    """
    Takes each band within the light curve data provided and puts it into MJD bins, tkaing the weighted mean of the flux, its errro and the weighted mean 
    MJD to match this, with the upper and lower errors on the MJD indicating the range of MJD values within the bin
    """
    lc_df = lc_df.copy() # this creates a copy rather than a view of the dataframe so that we don't modify the original dataframe
    
    bands_present = lc_df['band'].unique()
    for i, b in enumerate(bands_present):
        b_df = lc_df[lc_df['band'] == b].copy()

        # this rounds the min_MJD present in V_band data for the transient to the nearest 10, then goes to the bin below this to ensure that 
        # we capture the first data point, in case this rounded the first bin up. Since we need data for both V_band and B_band within each
        # MJD bin, it's fine to base the min MJD bin to start at on just the V_band data, since even if we had B_band data before this date, 
        # we wouldn't be able to pair it with data from V_band to calculcte the color anyways
        MJD_bin_min = int( round(b_df['MJD'].min(), -1) - 10 )
        MJD_bin_max = int( round(b_df['MJD'].max(), -1) + 10 )
        MJD_bins = range(MJD_bin_min, MJD_bin_max + MJD_binsize, MJD_binsize) # create the bins

        # data frame for the binned band data  - just adds a column of MJD_bin to the data, then we can group by all datapoints in the same MJD bin
        b_df['MJD_bin'] = pd.cut(b_df['MJD'], MJD_bins)
        
        # binning the data by MJD_bin
        b_binned_df = b_df.groupby('MJD_bin', observed = drop_na_bins).apply(lambda g: pd.Series({
                                                            'wm_flux': weighted_mean(g['flux'], g['flux_err'])[0], 
                                                            'wm_flux_err': weighted_mean(g['flux'], g['flux_err'])[1], 
                                                            'wm_MJD': weighted_mean(g['MJD'], g['flux_err'])[0], 
                                                            'min_MJD': g['MJD'].min(), 
                                                            'max_MJD': g['MJD'].max()
                                                            })).reset_index()

        b_binned_df['MJD_lower_err'] = b_binned_df['wm_MJD'] - b_binned_df['min_MJD'] # lower errorbar value in MJD
        b_binned_df['MJD_upper_err'] = b_binned_df['max_MJD'] - b_binned_df['wm_MJD'] # upper errorbar value in MJD
        b_binned_df['band'] = [b]*len(b_df) # band column
        b_binned_df = b_binned_df.drop(columns = ['min_MJD', 'max_MJD']) # drop these intermediate step columns
        # we should now have a light curve which is binned up with weighted mean flux + its error, the weighed mean MJD, with its error bar showing the range of MJD values 
        # within the bin, and the band

        if i == 0:
            whole_lc_binned_df = b_binned_df

        else:
            whole_lc_binned_df = pd.concat([whole_lc_binned_df, b_binned_df], ignore_index = True)



    return whole_lc_binned_df



##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################





def chisq(y_m, y, yerr, M, reduced_chi = False):
    """
    Calculates the chi squared, as well as the reduced chi squared and its 1 sigma uncertainty allowance if wanted

    INPUTS:
    --------------
    y_m: model y data

    y: observed y data

    yerr: observed y errors

    M: number of model paramaters

    reduced_chi: if True, this function will return chi, reduced_chi and red_chi_1sig, if False (default) it will just return chi

    OUTPUTS 
    ----------------
    chi: the chi squared of the model

    red_chi: the reduced chi squared

    red_chi_1sig: the 1 sigma error tolerance on the reduced chi squared. If the reduced chi squared falls within 1 +/- red_chi_1sig, it's considered a good model
    """
    if not isinstance(y_m, np.ndarray):
        y_m = np.array(y_m)

    if not isinstance(y, np.ndarray):
        y = np.array(y)

    if not isinstance(yerr, np.ndarray):
        yerr = np.array(yerr)

    chi = np.sum( ((y - y_m )**2) / (yerr**2))
    
    if reduced_chi == True:
        N = len(y) # the number of datapoints
        N_M = N-M # (N - M) the degrees of freedom
        red_chi = chi / (N_M)
        red_chi_1sig = np.sqrt(2/N_M) # red_chi is a good chisq if it falls within (1 +/- red_chi_1sig)
        
        return red_chi, red_chi_1sig
    
    else:
        return chi



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







