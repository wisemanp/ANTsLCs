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
from November.load_data_function import load_ANT_data 





##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################




def restframe_luminosity(d_l_cm, bandZP, z, m, m_err):
    """
    Calculates the rest frame luminosity 

    INPUTS:
    -----------------------
    d_l_cm: luminosity distance in cm ( can be calculated using astropy.cosmology.luminosity_distance(z) )

    bandZP: observed band's AB mag zeropoint in ergs/s/cm^2/Angstrom

    z: object's redshift

    m: magnitude (either abs or apparent, just account for this with d_l)

    m_err: magnitude error


    OUTPUTS
    -----------------------
    L_rf: rest frame luminosity in ergs/s/Angstrom

    L_rf_err: the rest frame luminosity's error, propagated from the error on the magnitude only - in ergs/s/Angstrom

    """
    L_rf = 4 * np.pi * (d_l_cm**2) * bandZP * (1 + z) * (10**(-0.4 * m)) # in ergs/s/Angstrom

    L_rf_err = 0.4 * np.log(10) * L_rf * m_err # in ergs/s/Angstrom

    return L_rf, L_rf_err





##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################




def obs_wl_to_em_wl(obs_wavelength, z):
    return obs_wavelength / (1+z)




##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################




def ANT_data_L_rf(ANT_df_list, ANT_names, dict_ANT_z, dict_ANT_D_lum, dict_band_ZP, dict_band_obs_cent_wl):
    """

    INPUTS
    ---------------
    ANT_df_list: a list of dataframes for each ANT

    ANT_names: a list of names of each ANT, MUST be in the same order as the dataframes, so ANT_df_list[i] and ANT_names[i] MUST correspond to the same ANT

    dict_ANT_z: dictionary of ANT redshift values

    dist_ANT_D_lum: dictioanry of ANT luminosity distances (calculated under a few assumptions, which can be checked in the plotting_preferences file)

    dict_band_ZP: dictioanry of the zeropoints of each of the bands present for any ANT

    dict_band_obs_cent_wl: dictionary of the observed central wavelengths of the bends present for each of the ANTs
    


    OUTPUTS
    ---------------
    new_ANT_df_list: a list of dataframes of the ANTs, in the same order as ANT_df_list and ANT_names
    
    """

    band_names = list(dict_band_obs_cent_wl.keys()) # a list of the band names for all ANTs
    observed_wl_list = list(dict_band_obs_cent_wl.values())

    new_ANT_df_list = [] # the list of ANT_dfs with the new data added such as L_rf, L_rf_err and em_cent_wl
    for i, ANT_df in enumerate(ANT_df_list):
        name = ANT_names[i] # ANT name
        z = dict_ANT_z[name] # redshift
        d_lum = dict_ANT_D_lum[name] # luminosity distance in cm


        # we need to create a dictionary of the observed band's ecntral wavelength which has been converted into the rest-frame by correcting for the redshift
        band_em_cent_wl = [obs_wl_to_em_wl(obs_wl, z) for obs_wl in observed_wl_list] # emitted central wavelength for each observed band
        band_em_cent_wl_dict = dict(zip(band_names, band_em_cent_wl)) # a dictionary of the observed band, and its central wavelength converted into the rest-frame wavelength

        ANT_df['em_cent_wl'] = ANT_df['band'].map(band_em_cent_wl_dict) # producing a column in ANT_df which gives the band's central wavelength converted into the rest-frame


        # rest frame luminosity
        rf_L = []
        rf_L_err = []
        for i in range(len(ANT_df['MJD'])):
            band = ANT_df['band'].iloc[i] # the datapoint's band
            band_ZP = dict_band_ZP[band] # the band's zeropoint
            mag = ANT_df['mag'].iloc[i] # the magnitude
            magerr = ANT_df['magerr'].iloc[i] # the mag error

            L_rest, L_rest_err = restframe_luminosity(d_lum, band_ZP, z, mag, magerr) # the rest frame luminosity and its error
            rf_L.append(L_rest)
            rf_L_err.append(L_rest_err)

        # add these columns to the ANT's dataframe
        ANT_df['L_rf'] = rf_L
        ANT_df['L_rf_err'] = rf_L_err

        new_ANT_df_list.append(ANT_df)


    return new_ANT_df_list




##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################


""" 
# test the L_rf function ----------------------------------------------------------------------------------------------------------------------------------------
lc_df_list, transient_names, list_of_bands = load_ANT_data()

# modified ANT dataframes to add rest frame luminosity, its error and the observed band's central wavelength converted into the rest frame
mod_lc_df_list = ANT_data_L_rf(lc_df_list, transient_names, ANT_redshift_dict, ANT_luminosity_dist_cm_dict, band_ZP_dict, band_obs_centwl_dict) 


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






