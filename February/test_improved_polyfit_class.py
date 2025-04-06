
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# THIS CODE IS USED TO GET THE INTERPOLATED DFS FROM THE MODIFIED PHIL'S LIGHTCURVE FILES 
#   - LOADS MODIFIED PHIL'S LIGHTCURVES
#   - CALCULATES THE REST FRAME LUMINOSITIES AND CENTRAL WAVELENGTHS
#   - BINS THE RESTFRAME LUMINOSITIES INTO 1 DAY BINS
#   - FITS A POLYNOMIAL TO EACH BAND OF THE LIGHT CURVE
#   - INTERPOLATES THE LIGHT CURVE (ACCORDING TO SOME SETTINGS YOU INPUT)
#   - SAVES THE INTERPOLATED DATAFRAMES ALONG WITH A README FILE OF THE INTERPOLATION AND POLYFIT INPUTS
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


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
from plotting_preferences import band_colour_dict, band_marker_dict, band_ZP_dict, band_obs_centwl_dict, ANT_redshift_dict, ANT_luminosity_dist_cm_dict, MJDs_for_fit, override_ref_band_dict
from functions import load_ANT_data, ANT_data_L_rf, bin_lc, chisq, polyfit_lightcurve





# load in the data
lc_df_list, transient_names, list_of_bands = load_ANT_data()


# calculate the rest frame luminosity + emitted central wavelength
add_lc_df_list = ANT_data_L_rf(lc_df_list, transient_names, ANT_redshift_dict, ANT_luminosity_dist_cm_dict, band_ZP_dict, band_obs_centwl_dict)


# bin up the light curve into 1 day MJD bins
MJD_binsize = 1
binned_df_list = bin_lc(add_lc_df_list, MJD_binsize)

ANTs_for_polyfit_dict = {'ZTF18aczpgwm': True,  # ONLY BOTHER POLYFITTING AND INTERPOLATING LIGHT CURVES WITH MORE THAN ONE DECENT BAND
                        'ZTF19aailpwl': True, 
                        'ZTF19aamrjar': True, 
                        'ZTF19aatubsj': True, 
                        'ZTF20aanxcpf': True, 
                        'ZTF20abgxlut': True, 
                        'ZTF20abodaps': True, 
                        'ZTF20abrbeie': True, 
                        'ZTF20acvfraq': True, 
                        'ZTF21abxowzx': True, 
                        'ZTF22aadesap': True, 
                        'ASASSN-17jz': True, 
                        'ASASSN-18jd': True, 
                        'CSS100217': False, 
                        'Gaia16aaw': False, 
                        'Gaia18cdj': False, 
                        'PS1-10adi': True, 
                        'PS1-13jw': False }


max_poly_order = 14
min_band_dps = 4
straggler_dist = 70
max_interp_distance = 20
gapsize = 70 # the size of a gap whete we allow NO interpolation at all over it to occur
interp_at_ref_band = True
max_interp_dist = 20
plot_polyfit = True
save_interp_df = True
save_README = True # this doesn't go into the class







# polyfitting light curves
for idx in range(len(transient_names)):
#for idx in [12]:
    ANT_name = transient_names[idx] # THERE ARE SOME LIGHT CURVES THAT AREN'T WORTH POLYFITTING
    if ANTs_for_polyfit_dict[ANT_name] == False:
        continue
        
    
    ANT_df = binned_df_list[idx]
    if ANT_name == 'ASASSN-18jd':
        ANT_df = ANT_df.dropna(subset = ['wm_L_rf_err']) # this light curve has NaN values in magerr when it's quoting an upper limit, so drop these rows for polyfitting

    ANT_bands = list_of_bands[idx]
    polyfit_MJD_range = MJDs_for_fit[ANT_name]
    bands_for_BB = [b for b in ANT_bands if (b != 'WISE_W1') and (b != 'WISE_W2')] # remove the WISE bands from the interpolation since we don't want to use this data for the BB fit anyway

    print(ANT_name)

    

    lightcurve = polyfit_lightcurve(ant_name = ANT_name, 
                                    ant_z = ANT_redshift_dict[ANT_name],
                                    df = ANT_df, 
                                    bands = bands_for_BB, 
                                    override_ref_band_dict = override_ref_band_dict,   
                                    interp_at_ref_band = interp_at_ref_band, 
                                    min_band_dps = min_band_dps, 
                                    straggler_dist = straggler_dist,
                                    gapsize = gapsize,
                                    fit_MJD_range = polyfit_MJD_range, 
                                    max_interp_distance = max_interp_distance, 
                                    max_poly_order = max_poly_order, 
                                    b_colour_dict = band_colour_dict, 
                                    b_marker_dict = band_marker_dict,
                                    plot_polyfit = plot_polyfit, 
                                    save_interp_df = save_interp_df)
    
    lightcurve.run_fitting_pipeline()

    if (save_README == True) & (idx == 0):
        save_interp_data_folder = f"C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS/data/interpolated_lcs/"
        readme_content = f"Interpolated light curves using the following paramaters: \n max_poly_order = {max_poly_order} \n min_band_dps = {min_band_dps} \n straggler_dist = {straggler_dist} \n gapsize = {gapsize} \n max_interp_distance = {max_interp_distance} \n interp_at_ref_band = {interp_at_ref_band} \n max_interp_dist = {max_interp_dist}"
        with open(save_interp_data_folder+"README.txt", "w") as f:
            f.write(readme_content)
   
    
    print()


