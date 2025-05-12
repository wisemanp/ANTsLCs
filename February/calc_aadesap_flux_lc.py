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
from plotting_preferences import band_colour_dict, band_marker_dict, band_ZP_dict, band_obs_centwl_dict, ANT_redshift_dict, ANT_proper_redshift_dict, ANT_luminosity_dist_cm_dict, peak_MJD_dict
from functions import load_ANT_data, ANT_data_L_rf, bin_lc, chisq, polyfit_lightcurve




# load in the data
lc_df_list, transient_names, list_of_bands = load_ANT_data()


# calculate the rest frame luminosity + emitted central wavelength
L_rf_lc_df_list = ANT_data_L_rf(lc_df_list, transient_names, ANT_redshift_dict, ANT_luminosity_dist_cm_dict, band_ZP_dict, band_obs_centwl_dict)


idx = transient_names.index('ZTF22aadesap')
lc_df = L_rf_lc_df_list[idx]
ANT_name = 'ZTF22aadesap'

def L_rf_to_observed_flux_density(L_rf, z, d_cm):
    """
    Converts rest frame spectral luminosity density to observer frame spectral flux density
    """
    denom = 4 * np.pi * (d_cm**2) * (1 + z)
    F_lam = L_rf / denom
    return F_lam


def convert_MJD_to_restframe_DSP(peak_MJD, MJD, z):
    return (MJD - peak_MJD) / (1 + z)


lc_df['obs_flux_density'] = L_rf_to_observed_flux_density(L_rf = lc_df['L_rf'], z = ANT_proper_redshift_dict[ANT_name], d_cm = ANT_luminosity_dist_cm_dict[ANT_name])
lc_df['obs_flux_density_err'] = L_rf_to_observed_flux_density(L_rf = lc_df['L_rf_err'], z = ANT_proper_redshift_dict[ANT_name], d_cm = ANT_luminosity_dist_cm_dict[ANT_name])
lc_df['d_since_peak'] = convert_MJD_to_restframe_DSP(peak_MJD = peak_MJD_dict[ANT_name], MJD = lc_df['MJD'], z = ANT_proper_redshift_dict[ANT_name])

print(lc_df)


lc_df.to_csv("C:\\Users\\laure\\OneDrive\\Desktop\\YoRiS desktop\\YoRiS\\February\\ZTF22aadesap_flux_density.csv", index = True)