import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from astropy import constants as const
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM





##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
# LOADING IN THE ANT DATA




def load_ANT_data():
    """
    loads in the ANT data files

    INPUTS:
    -----------
    None


    OUTPUTS:
    ----------
    dataframes: a list of the ANT data in dataframes
    names: a list of the ANT names
    ANT_bands: a list of lists, each inner list is a list of all of the bands which are present in the light curve

    """
    phils_ANTs = "C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS Data/modified Phil's lightcurves"
    updated_Phils_ANTs = "C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS Data/Phil's lcs updated modified" # this contains forced photometry files for abodaps and mrjar, so we'll use these instead of the data files for these ants in Phils_ANTs
    other_ANT_data = "C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS Data/other ANT data/ALL_FULL_LCs" 
    directories = [phils_ANTs, updated_Phils_ANTs, other_ANT_data]

    dataframes = []
    names = []
    ANT_bands = []
    for dir in directories:
        for file in os.listdir(dir): # file is a string of the file name such as 'file_name.dat'
            # getting the ANT name from the file name
            if dir == other_ANT_data:
                ANT_name = file[:-12]
            else:
                ANT_name = file[:-7] # the name of the transient

            # skip over the old verisons of these files from Phil, since we have better ones in updated_phils_ANTs
            if (ANT_name == 'ZTF19aamrjar' or ANT_name == 'ZTF20abodaps') and dir == phils_ANTs: 
                continue

            # load in the files
            file_path = os.path.join(dir, file)
            file_df = pd.read_csv(file_path, delimiter = ',') # the lightcurve data in a dataframe
            bands = file_df['band'].unique() # the bands in which this transient has data for

            dataframes.append(file_df)
            names.append(ANT_name)            
            ANT_bands.append(bands)

    return dataframes, names, ANT_bands





##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
# REST FRAME LUMINOSITY CALCULATION



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
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
# BINNING FUNCTION




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





def bin_lc(list_lc_df, MJD_binsize, drop_na_bins = True):
    """
    Takes each band within the light curve data provided and puts it into MJD bins, taking the weighted mean of the rest frame luminosity, its error and the weighted mean 
    MJD to match this, with the upper and lower errors on the MJD indicating the range of MJD values within the bin.

    INPUTS
    -----------
    list_lc_df: a list of dataframes for the ANT's light curve data across all bands. Must contain the columns L_rf and L_rf_err 

    MJD_binsize: the size of the MJD bins that you want

    
    OUTPUTS
    -----------
    list_binned_lc_dfs: a list of dataframes of the whole ANT light curve binned into bin size = MJD_binsize. Each band within the light curve is binned separately.

    """
    list_binned_lc_dfs = []
    for idx, lc_df in enumerate(list_lc_df):
        #print(f'index = {idx}')
        lc_df = lc_df.copy() # this creates a copy rather than a view of the dataframe so that we don't modify the original dataframe
        
        bands_present = lc_df['band'].unique()
        for i, b in enumerate(bands_present):
            b_df = lc_df[lc_df['band'] == b].copy()

            # cretaing the bins -------------------------------------------------------------------------------------------------------------------------------------------------------------
            # this rounds the min_MJD present in V_band data for the transient to the nearest 10, then goes to the bin below this to ensure that 
            # we capture the first data point, in case this rounded the first bin up. Since we need data for both V_band and B_band within each
            # MJD bin, it's fine to base the min MJD bin to start at on just the V_band data, since even if we had B_band data before this date, 
            # we wouldn't be able to pair it with data from V_band to calculcte the color anyways
            MJD_bin_min = int( round(b_df['MJD'].min(), -1) - 10 )
            MJD_bin_max = int( round(b_df['MJD'].max(), -1) + 10 )
            MJD_bins = range(MJD_bin_min, MJD_bin_max + MJD_binsize, MJD_binsize) # create the bins

            # binning the data --------------------------------------------------------------------------------------------------------------------------------------------------------------
            # data frame for the binned band data  - just adds a column of MJD_bin to the data, then we can group by all datapoints in the same MJD bin
            b_df['MJD_bin'] = pd.cut(b_df['MJD'], MJD_bins)
            
            # binning the data by MJD_bin
            b_binned_df = b_df.groupby('MJD_bin', observed = drop_na_bins).apply(lambda g: pd.Series({
                                                                'wm_L_rf': weighted_mean(g['L_rf'], g['L_rf_err'])[0], 
                                                                'wm_L_rf_err': weighted_mean(g['L_rf'], g['L_rf_err'])[1], 
                                                                'wm_MJD': weighted_mean(g['MJD'], g['L_rf_err'])[0], 
                                                                'min_MJD': g['MJD'].min(), 
                                                                'max_MJD': g['MJD'].max(), 
                                                                'count': g['MJD'].count()
                                                                })).reset_index()

            # creating the upper and lower MJD errorbars-------------------------------------------------------------------------------------------------------------------------------------
            MJD_lower_err = []
            MJD_upper_err = []
            for j in b_binned_df.index: 
                if b_binned_df['count'].loc[j] == 1.0: # accounting for the casae if we have one datapoint in the bin, hence the MJD range within the bin = 0.0, but if we don't set it to 0.0 explicitly, it'll be calculated as like -1e-12
                    mjd_lerr = 0.0
                    mjd_uerr = 0.0

                else: # if there are > 1 datapoints within the bin
                    mjd_lerr = b_binned_df['wm_MJD'].iloc[j] - b_binned_df['min_MJD'].iloc[j] # lower errorbar value in MJD
                    mjd_uerr = b_binned_df['max_MJD'].iloc[j] - b_binned_df['wm_MJD'].iloc[j] # upper errorbar value in MJD

                    # getting rid of any negative mjd upper or lower errors, since the only way that it's possible for these numbers to be negative is that the MJD of the datapoints 
                    # in the bin are so close together, you're essentially doing mjd upper/lower = a - (~a), which, due to floating point errors, could give a value like -1e-12
                    if mjd_lerr < 0.0:
                        mjd_lerr = 0.0
                    elif mjd_uerr < 0.0:
                        mjd_uerr = 0.0

                MJD_lower_err.append(mjd_lerr)
                MJD_upper_err.append(mjd_uerr)

            b_binned_df['MJD_lower_err'] = MJD_lower_err
            b_binned_df['MJD_upper_err'] = MJD_upper_err
            b_binned_df['band'] = [b]*len(b_binned_df['wm_L_rf']) # band column
            b_binned_df = b_binned_df.drop(columns = ['min_MJD', 'max_MJD', 'count']) # drop these intermediate step columns

            # we should now have a light curve which is binned up with weighted mean flux + its error, the weighed mean MJD, with its error bar showing the range of MJD values 
            # within the bin, and the band
            # concatenate together the band dataframes to produce a dataframe for the whole light curve across all bands again -------------------------------------------------------------
            if i == 0:
                whole_lc_binned_df = b_binned_df
            else:
                whole_lc_binned_df = pd.concat([whole_lc_binned_df, b_binned_df], ignore_index = True)

        list_binned_lc_dfs.append(whole_lc_binned_df) # a list of the binned ANT dataframes


    return list_binned_lc_dfs






##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
# CHI SQUARED FUNCTION



def chisq(y_m, y, yerr, M, reduced_chi = True):
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
    chi: the chi squared of the model (returned if reduced_chi == False)

    red_chi: the reduced chi squared (returned if reduced_chi == True)

    red_chi_1sig: the 1 sigma error tolerance on the reduced chi squared. If the reduced chi squared falls within 1 +/- red_chi_1sig, it's considered a good model 
                (returned if reduced_chi == True)
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
    




