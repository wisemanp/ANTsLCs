import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS") # this allows us to import plotting preferences and functions
from plotting_preferences import band_colour_dict, band_marker_dict, band_offset_dict, band_ZP_dict, band_obs_centwl_dict, ANT_redshift_dict, ANT_luminosity_dist_cm_dict
from functions import load_ANT_data, ANT_data_L_rf, bin_lc, chisq

##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################

MJDs_for_fit = {'ZTF18aczpgwm': (58350, 60300), 
                'ZTF19aailpwl': (58420, 59500), 
                'ZTF19aamrjar': (58350, None), 
                'ZTF19aatubsj': (58360, None), 
                'ZTF20aanxcpf': (59140, None), 
                'ZTF20abgxlut': (58900, None), 
                'ZTF20abodaps': (58920, None), 
                'ZTF20abrbeie': (None, None), 
                'ZTF20acvfraq': (58700, None), 
                'ZTF21abxowzx': (59400, 60200), # this gets rid of the little UVOT burst
                'ZTF22aadesap': (59500, None), 
                'ASASSN-17jz': (None, None), 
                'ASASSN-18jd': (None, None), 
                'CSS100217': (None, None), 
                'Gaia16aaw': (None, None), 
                'Gaia18cdj': (None, None), 
                'PS1-10adi': (None, None), 
                'PS1-13jw': (None, None)} 


def polyfit_lcs(ant_name, df, fit_order, df_bands, fit_MJD_range, b_color_dict, plot_polyfit = False):
    
    if plot_polyfit == True: 
        fig = plt.figure(figsize = (16, 7.5))

    fit_min_MJD, fit_max_MJD = fit_MJD_range # unpack the tuple that goes as (MJD min, MJD max)
    for i, b in enumerate(df_bands):
        b_df = df[df['band'] == b].copy() # the dataframe for the band
        b_colour = b_color_dict[b]

        # limiting the MJD over which we are polyfitting, because for some ANTs, we have some straggling datapoints far away from the rest of the light curve and we don't wnat to fit these
        if fit_min_MJD != None:
            b_lim_df = b_df[b_df['wm_MJD'] > fit_min_MJD].copy() 

        if fit_max_MJD != None:
            b_lim_df = b_lim_df[b_lim_df['wm_MJD'] < fit_max_MJD].copy()

        elif (fit_min_MJD == None) & (fit_max_MJD == None): # in the event that we don't need to limit the MJD values over which we are fitting:
            b_lim_df = b_df.copy()


        # when doing the polyfit, it keeps giving 'RankWarning: Polyfit may be poorly conditioned', so chatGPT suggested to try scaling down the x values input -------------------------
        #MJD_scalefactor = b_lim_df['MJD'].mean()
        #MJD_scaled = b_lim_df['MJD'] - MJD_scalefactor
        

        # do the polynomial fit ---------------------------------------------------------------------------------------------------------------------------------------------------------
        poly_coeffs = np.polyfit(b_lim_df['wm_MJD'], b_lim_df['wm_L_rf'], deg = fit_order)
        poly_coeffs = list(poly_coeffs)
        polynomial_fit = np.poly1d(poly_coeffs, r = False) # r = False just measn that I am providing the poly coefficients, not the roots
        poly_plot_MJD = np.arange(b_lim_df['wm_MJD'].min(), b_lim_df['wm_MJD'].max(), 1) # the MJD values at which the polyfit will be evaluated
        poly_plot_L_rf = polynomial_fit(poly_plot_MJD)


        # calculate chi squared of the polyfit
        poly_L_rf_for_chi = polynomial_fit(b_lim_df['wm_MJD']) 
        red_chi, red_chi_1sig = chisq(y_m = poly_L_rf_for_chi, y = b_lim_df['wm_L_rf'], yerr = b_lim_df['wm_L_rf_err'], M = (fit_order + 1))


        # plotting -------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if plot_polyfit == True:
            plt.errorbar(b_df['wm_MJD'], b_df['wm_L_rf'], yerr = b_df['wm_L_rf_err'], fmt = 'o', c = b_colour, linestyle = None, markeredgecolor = 'k', markeredgewidth = '0.5', markersize = 5, 
                         label = b)
            plt.plot(poly_plot_MJD, poly_plot_L_rf, c = b_colour, label = f'red chi = {red_chi:.3f} +/- {red_chi_1sig:.3f}')

    if plot_polyfit == True:
        plt.xlabel('MJD')
        plt.ylabel('rest frame luminosity')
        plt.title(f'{ant_name} polyfit order = {fit_order}')
        plt.legend(loc = 'lower right', bbox_to_anchor = (1.2, 0.0), fontsize = 10)
        fig.subplots_adjust(right = 0.845, left = 0.07)
        plt.grid()
        plt.show()

    return






##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
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

# now, we want to fit polynomials to each band of the binned up light curves
ANT = 'ZTF22aadesap'
ANT = 'ZTF19aailpwl'
idx = transient_names.index(ANT)
ANT_bands = list_of_bands[idx]
binned_ANT_df = bin_lc_df_list[idx].copy()
polyfit_lcs(ANT, binned_ANT_df, fit_order = 12, df_bands = ANT_bands, fit_MJD_range = MJDs_for_fit[ANT], b_color_dict = band_colour_dict, plot_polyfit = True)







