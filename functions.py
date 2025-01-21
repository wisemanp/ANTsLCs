import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from astropy import constants as const
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
import scipy.optimize as opt
from tqdm import tqdm




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
    other_ANT_data = "C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS Data/other ANT data/ALL_FULL_LCs" 
    directories = [phils_ANTs, other_ANT_data]

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
    ANT_df_list: a list of dataframes for each ANT, with the columns: MJD, mag, magerr, band

    ANT_names: a list of names of each ANT, MUST be in the same order as the dataframes, so ANT_df_list[i] and ANT_names[i] MUST correspond to the same ANT

    dict_ANT_z: dictionary of ANT redshift values

    dist_ANT_D_lum: dictioanry of ANT luminosity distances (calculated under a few assumptions, which can be checked in the plotting_preferences file)

    dict_band_ZP: dictioanry of the zeropoints of each of the bands present for any ANT

    dict_band_obs_cent_wl: dictionary of the observed central wavelengths of the bends present for each of the ANTs
    


    OUTPUTS
    ---------------
    new_ANT_df_list: a list of dataframes of the ANTs, in the same order as ANT_df_list and ANT_names. Each dataframe will have the columns:
                    MJD, mag, magerr, band, em_cent_wl (in Angstrom), L_rf (in ergs/(s * cm^2 * Angstrom)), L_rf_err (in ergs/(s * cm^2 * Angstrom)).
    
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

    if (sum(errors)) == 0.0: # if the errors in the bin all == 0.0 (like with Gaia_G) then just take the regular mean and set its error to NA
        wm = np.mean(data)
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
    list_lc_df: a list of dataframes for the ANT's light curve data across all bands. Must contain the columns:  MJD, L_rf, L_rf_err, band, em_cent_wl

    MJD_binsize: the size of the MJD bins that you want

    
    OUTPUTS
    -----------
    list_binned_lc_dfs: a list of dataframes of the whole ANT light curve binned into bin size = MJD_binsize. Each band within the light curve is binned separately.
                        Each df contains the columns:  MJD_bin, wm_L_rf (ergs/s/cm^2/Angstrom), wm_L_rf_err (args/s/cm^2/Angstrom), wm_MJD, band, em_cent_wl, MJD_lower_err, MJD_upper_err

    """
    list_binned_lc_dfs = []
    for idx, lc_df in enumerate(list_lc_df):
        #print(f'index = {idx}')
        lc_df = lc_df.copy() # this creates a copy rather than a view of the dataframe so that we don't modify the original dataframe
        
        bands_present = lc_df['band'].unique()
        for i, b in enumerate(bands_present):
            b_df = lc_df[lc_df['band'] == b].copy()

            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # cretaing the bins 
            # this rounds the min_MJD present in V_band data for the transient to the nearest 10, then goes to the bin below this to ensure that 
            # we capture the first data point, in case this rounded the first bin up. Since we need data for both V_band and B_band within each
            # MJD bin, it's fine to base the min MJD bin to start at on just the V_band data, since even if we had B_band data before this date, 
            # we wouldn't be able to pair it with data from V_band to calculcte the color anyways
            MJD_bin_min = int( round(b_df['MJD'].min(), -1) - 10 )
            MJD_bin_max = int( round(b_df['MJD'].max(), -1) + 10 )
            MJD_bins = range(MJD_bin_min, MJD_bin_max + MJD_binsize, MJD_binsize) # create the bins

            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # binning the data
            # data frame for the binned band data  - just adds a column of MJD_bin to the data, then we can group by all datapoints in the same MJD bin
            b_df['MJD_bin'] = pd.cut(b_df['MJD'], MJD_bins)
            
            # binning the data by MJD_bin
            b_binned_df = b_df.groupby('MJD_bin', observed = drop_na_bins).apply(lambda g: pd.Series({
                                                                'wm_L_rf': weighted_mean(g['L_rf'], g['L_rf_err'])[0], 
                                                                'wm_L_rf_err': weighted_mean(g['L_rf'], g['L_rf_err'])[1], 
                                                                'wm_MJD': weighted_mean(g['MJD'], g['L_rf_err'])[0], 
                                                                'min_MJD': g['MJD'].min(), 
                                                                'max_MJD': g['MJD'].max(), 
                                                                'count': g['MJD'].count(), 
                                                                'band': g['band'].iloc[0],
                                                                'em_cent_wl': g['em_cent_wl'].iloc[0]
                                                                })).reset_index()

            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # creating the upper and lower MJD errorbars
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
            b_binned_df = b_binned_df.drop(columns = ['min_MJD', 'max_MJD', 'count']) # drop these intermediate step columns

            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # we should now have a light curve which is binned up with weighted mean flux + its error, the weighed mean MJD, with its error bar showing the range of MJD values 
            # within the bin, and the band
            # concatenate together the band dataframes to produce a dataframe for the whole light curve across all bands again 
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
# CONVERTING FROM REST FRAME LUMINOSITY TO APPARENT MAGNITUDE








def L_rf_to_mag(d_l_cm, bandZP, z, L_rf, L_rf_err):
    """
    Calculates the magnitude from the rest frame luminosity 

    INPUTS:
    -----------------------
    d_l_cm: luminosity distance in cm ( can be calculated using astropy.cosmology.luminosity_distance(z) )

    bandZP: observed band's AB mag zeropoint in ergs/s/cm^2/Angstrom

    z: object's redshift

    L_rf: rest frame luminosity in ergs/s/Angstrom

    L_rf_err: rest frame luminosity error in ergs/s/Angstrom


    OUTPUTS
    -----------------------
    m: magnitude (either abs or apparent, depending on d_l_cm)

    m_err: magnitude error

    """
    denom = 4 * np.pi * (d_l_cm**2) * bandZP * (1 + z)
    m = -2.5 * np.log10(L_rf / denom)

    m_err = (2.5 / (np.log(10) * L_rf)) * L_rf_err # THIS HAD A NEGATIVE SIGN IN IT BUT I THINK ITS FINE TO TAKE THE POSITIVE, CHECK GOODNOTES NOTEs

    return m, m_err







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
# AFTER BINNING THE LIGHT CURVE USING THE REST FRAME LUMINOSITY, WE CAN RE-CALCULATE THE APPARENT MAGNITUDE OF THE BIN









def ANT_data_mags(ANT_df_list, ANT_names, dict_ANT_z, dict_ANT_D_lum, dict_band_ZP):
    """
    Once we've binned the light curves (which we can only do using rest frame luminosity or flux), we can calculate the apparent ot absolute magnitude associated with this averaged rest frame luminosity in the bin. 
    This funciton calculates the apparent of absolute magnitude + its error given the rest frame luminosity and its error and adds these columns to each ANT light curve

    INPUTS
    ---------------
    ANT_df_list: a list of dataframes for each ANT, with the columns: MJD, mag, magerr, band

    ANT_names: a list of names of each ANT, MUST be in the same order as the dataframes, so ANT_df_list[i] and ANT_names[i] MUST correspond to the same ANT

    dict_ANT_z: dictionary of ANT redshift values

    dist_ANT_D_lum: dictioanry of ANT luminosity distances (calculated under a few assumptions, which can be checked in the plotting_preferences file)

    dict_band_ZP: dictioanry of the zeropoints of each of the bands present for any ANT
    


    OUTPUTS
    ---------------
    new_ANT_df_list: a list of dataframes of the ANTs, in the same order as ANT_df_list and ANT_names. Each dataframe will have the columns mag amd magerr added to it
    
    """

    new_ANT_df_list = [] # the list of ANT_dfs with the new data added such as L_rf, L_rf_err and em_cent_wl
    for i, ANT_df in enumerate(ANT_df_list):
        name = ANT_names[i] # ANT name
        z = dict_ANT_z[name] # redshift
        d_lum = dict_ANT_D_lum[name] # luminosity distance in cm

        # rest frame luminosity
        mag_list = []
        magerr_list = []
        for i in range(len(ANT_df['wm_MJD'])):
            band = ANT_df['band'].iloc[i] # the datapoint's band
            band_ZP = dict_band_ZP[band] # the band's zeropoint
            L_rf = ANT_df['wm_L_rf'].iloc[i] # the magnitude
            L_rf_err = ANT_df['wm_L_rf_err'].iloc[i] # the mag error

            mag, magerr = L_rf_to_mag(d_lum, band_ZP, z, L_rf, L_rf_err) # the rest frame luminosity and its error
            mag_list.append(mag)
            magerr_list.append(magerr)

        # add these columns to the ANT's dataframe
        ANT_df['mag'] = mag_list
        ANT_df['magerr'] = magerr_list

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

        if N_M <= 0.0: # if the number of parameters >= number of datapoints to fit
            red_chi = pd.NA
            red_chi_1sig = pd.NA
        else:
            red_chi = chi / (N_M)
            red_chi_1sig = np.sqrt(2/N_M) # red_chi is a good chisq if it falls within (1 +/- red_chi_1sig)
        
        return red_chi, red_chi_1sig
    
    else:
        return chi
    






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
# POLYFITTING A LIGHT CURVE







def polyfit_lc(ant_name, df, fit_order, df_bands, trusted_band, fit_MJD_range, extrapolate, b_colour_dict, plot_polyfit = False):
    """
    Peforms a polynomial fit for each band within df_bands for the light curve. 

    INPUTS
    ---------------
    ant_name: the ANT's name (str)

    df: a dataframe containing the ANT's light curve, ideally this would be binned into 1 day bins or something so that we don't get a flare of scattered L_rf values at ~= MJD
         which confuses the polyfit funciton. It is assumed that there are columns within df named: 'wm_L_rf' - the weighted mean rest frame luminosity within the MJD bin, 
         'wm_L_rf_err' the error on this weighted mean rest frame luminosity and 'wm_MJD', the weighted mean MJD value within the bin. (dataframe). Dataframe must contain the columns:
         wm_MJD, wm_L_rf, wm_L_rf_err, band, em_cent_wl

    df_bands: a list of the bands within the ANT's lightcurve that you want to have a polyfit for. If df contains WISE_W1 data, if you don't want this to be polyfitted, just don't include
               'WISE_W1' in df_bands. (list)

    trusted_band: the name of the band which is used as the reference band for the light curve. This band would ideally have a high cadence and good coverage of the light curve. 
                  at the MJD values for which df contains data on this band, this function will use the polynomial fit for each other band and evaluate L_rf for each band at this MJD. 
                  This band is also the one from which we will calcuate the peak of the light curve. (str)

    fit_MJD_range: a tuple like (min_polyfit_MJD, max_polyfit_MJD), giving the MJD limits between which you want the bands in df_bands to be polyfitted. (tuple)

    extrapolate: True if you want to extrapolate with the polyfits, False if not. (bool)

    b_colour_dict: a dictionary indicating the marker colours for each photometric band such that b_colour_dict['ZTF_g'] = ZTF_g_colour. (dict)

    plot_polyfit: True if you want a plot of the polfit over the top of the actual light curve's data, as well as the interpolated data taken from the polyfit. Plot also displays the
                    reduced chi squareds for each band's polyfit. Default is False (bool)

    
    OUTPUTS
    ---------------
    polyfit_ref_lc_df: a dataframe containing MJD, L_rf, L_rf_err, band for: the trusted_band's (also referred to as the reference band) actual MJD, L_rf, L_rf_err data, and then for 
                        the other bands, their data was 'interpolated' using a polynomial fit to the band's data which was evaluated at the MJD values of the trusted band. This funciton 
                        both interpolates and extrapolates using the polyfit, so at any MJD value present within this dataframe, there will be a L_rf value for every band within df_bands. 
                        This is good if you wanted to do something like blackbody fitting, since you then have datapoints for lots of bands at a given MJD, allowing for a better blackbody
                        fit. This contains the columns: MJD, L_rf, L_rf_err, band, em_cent_wl

                        NOTE: L_rf_err in polyfit_ref_lc_df for all bands except the trusted_band is calculated using a fudge formula

    plot_polyline_df: a dataframe containing the polyfit data so that it can be plotted outside of this function. Contains the columns: band, poly_MJD, poly_L_rf. Each band has just one
                        row in the dataframe, and poly_MJD and poly_L_rf will be lists/arrays of the MJD/L_rf data for this band's polyfit

    """
    
    # =========================================================================================================================================================================================================
    # =========================================================================================================================================================================================================
    def fudge_polyfit_L_rf_err(real_b_df, scaled_polyfit_L_rf, scaled_reference_MJDs, MJD_scaledown, L_rf_scaledown):
        """
        Fudges the uncertainties on the rest frame luminosity values calculated using the polynomial fit of the band,
        evaluated at the trusted_band's MJD values. 

        Fudged_L_rf_err = mean(L_rf_err values of the 20 closest datapoints in MJD) + 0.05*polyfit_L_rf*|mjd - mjd of 
        closest datapoint in the band's real data|/10

        The first term is the mean of the closest 10 datapoints in MJD's rest frame luminosity errors. The second term 
        adds a 5% error for every 10 dats interpolated. For this fudge errors formula, I took inspo from
        Superbol who took inspo from someone else.
        
        INPUTS
        ---------------
        real_b_df: the actual band's dataframe (assuming the light curve has been through the binning function to be 
                    binned into like 1 day MJD bins). This data should not be limited by
                    the MJD values which we selected (fit_MJD_range in the outer funciton), 
                    just input the whole band's dataframe. 

        polyfit_L_rf: a list/array of the polyfitted rest frame luminosity values which were evaluated at the 
                      trusted band's MJD values (the reference_MJDs)

        reference_MJDs: a list/array of MJD values which are the values at which the trusted band has data. This should
                        be restricted to the region chosen to fit the polynomials to, not the
                        entire light curve with any little stragglers far before the start/'end', i.e. should be limited 
                        by fit_MJD_range from the outer function.

                        
        OUTPUTS
        ---------------
        poly_L_rf_err_list: a list of our fudged L_rf_err values for our polyfit interpolated data. 

        """
        def fudge_error_formula(mean_err, mjd_dif, L_scaled):
            x = abs( L_scaled * (mjd_dif / 10) )
            er = mean_err + 0.05 * x #+ 0.0001*x**2  # this adds some % error for every 10 days interpolated - inspired by Superbol who were inspired by someone else

            if er > abs(L_scaled):
                er = abs(L_scaled)

            return er
        
        rb_df = real_b_df.sort_values(by = 'wm_MJD', ascending = True) # making sure that the dataframe is sorted in ascending order of MJD, just in case. The real band's dataframe
        rb_MJDs = np.array(rb_df['wm_MJD'].copy())
        scaled_rb_MJDs = rb_MJDs - MJD_scaledown # scaled down band's real MJD values
        rb_L_rf_err = np.array(real_b_df['wm_L_rf_err'].copy()) # an array of the band's real L_rf errors
        scaled_rb_L_rf_err = rb_L_rf_err * L_rf_scaledown # scaled down band's real weighted mean rest frame luminosity error values

        fudge_err_list = [] # a list of fudged rest frame luminosity uncertainties on the L_rf values calculated by polyfit, at the reference MJDs
        for i, sc_L in enumerate(scaled_polyfit_L_rf): # iterate through each L_rf value calculated by evaluating the polyfit at the reference band's MJDs
            sc_ref_mjd = scaled_reference_MJDs[i]
            MJD_diff = [abs(sc_ref_mjd - sc_mjd) for sc_mjd in scaled_rb_MJDs] # take the abs value of the difference between the mjd of the datapoint and the mjd values in the band's actual data
            sort_by_MJD_closeness = np.argsort(MJD_diff) # argsort returns a list of the index arrangement of the elements within the list/array its given that would sort the list/array in ascending order
            closest_20_idx = sort_by_MJD_closeness[:20] 
            sc_closest_20_err = [scaled_rb_L_rf_err[j] for j in closest_20_idx] #  a list of the wm_L_rf_err values for the 20 closest datapoints to our interpolated datapoint
            sc_mean_L_rf_err = np.mean(np.array(sc_closest_20_err)) # part of the error formula = mean L_rf_err of the 20 closest datapoints in MJD
            closest_MJD_diff = MJD_diff[sort_by_MJD_closeness[0]]

            sc_poly_L_rf_er = fudge_error_formula(sc_mean_L_rf_err, closest_MJD_diff, sc_L)
            poly_L_rf_er =  sc_poly_L_rf_er / L_rf_scaledown # to scale L_rf (fudged) error
            
            if poly_L_rf_er < 0.0:
                print(sc_L, poly_L_rf_er)
            fudge_err_list.append(poly_L_rf_er)

        return fudge_err_list
    # =========================================================================================================================================================================================================
    # =========================================================================================================================================================================================================
    # setting up the plot
    if plot_polyfit == True:
        fig = plt.figure(figsize = (16, 7.5))
    
    # limiting the MJD over which we are polyfitting, because for some ANTs, we have some straggling datapoints far away from the rest of the light curve and we don't wnat to fit these
    fit_min_MJD, fit_max_MJD = fit_MJD_range # unpack the tuple that goes as (MJD min, MJD max)

    lim_df = df.copy()
    if fit_min_MJD != None:
        lim_df = lim_df[lim_df['wm_MJD'] > fit_min_MJD].copy() 

    if fit_max_MJD != None:
        lim_df = lim_df[lim_df['wm_MJD'] < fit_max_MJD].copy()


    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # getting the MJD values for the reference band 
    ref_band_df = df[df['band'] == trusted_band].copy()
    ref_band_MJD = np.array(ref_band_df['wm_MJD'].copy()) # the reference MJD values at which we should evaluate all other bands' polyfits
    MJD_scaleconst = np.mean(ref_band_MJD) # scale the MJD down to make it easier for polyfit
    ref_band_MJD_scaled = ref_band_MJD - MJD_scaleconst
    L_rf_scalefactor = 1e-41 # define the scale factor for the rest frame luminosity, too

    # within this dataframe we will store the values of each band's polyfit at the MJD values of the chosen band to generate a light curve which has datapoints for all bands
    # at the MJD values of the chosen band. I want a dataframe containing wm_MJD, wm_L_rf, wm_L_rf_err, band, but probs best to rename the columns to remove the 'wm' since
    # most of the data within it will be calculated from the polyfit, not the weighted mean. 
    polyfit_ref_lc_df = pd.DataFrame({'MJD': ref_band_MJD, 
                                      'L_rf': np.array(ref_band_df['wm_L_rf'].copy()), 
                                      'L_rf_err': np.array(ref_band_df['wm_L_rf_err'].copy()),
                                      'band': list(ref_band_df['band'].copy()), 
                                      'em_cent_wl': list(ref_band_df['em_cent_wl'].copy())
                                      })
    
    columns = ['band', 'poly_MJD', 'poly_L_rf', 'red_chi', 'red_chi_1sig', 'chi_sigma_dist'] # allows us to plot the polyfit as a line and the interpolated values.                                                                 
    plot_polyline_df = pd.DataFrame(columns = columns)
    # iterate through the bands and polyfit them ===========================================================================================================================================
    for i, b in enumerate(df_bands):
        b_df = df[df['band'] == b].copy()
        b_lim_df = lim_df[lim_df['band'] == b].copy() # the dataframe for the band, with MJD values limited to the main light curve
        b_em_cent_wl = b_df['em_cent_wl'].iloc[0] # take the first value here because this dataframe only contains data from 1 band anyways so all em_cent_wl values will be the same
        plot_polyline_rowno = len(plot_polyline_df['band']) # the row number for this band's data
        plot_polyline_row = list(np.zeros(len(columns))) # fill the plot data with zeros then overwrite these values
        if b == trusted_band: # we don't need a polyfit of the trusted band because we're just evaluating the polyfits of all other bands at the trusted_band's MJDs
            if plot_polyfit == True:
                plot_polyline_row[:] = [trusted_band, np.nan, np.nan, np.nan, np.nan, np.nan]
                plot_polyline_df.loc[plot_polyline_rowno] = plot_polyline_row # append with this band's data

                b_colour = b_colour_dict[b]
                plt.errorbar(b_df['wm_MJD'], b_df['wm_L_rf'], yerr = b_df['wm_L_rf_err'], c = b_colour, fmt = 'o', linestyle = 'None', markeredgecolor = 'k', 
                            markeredgewidth = '1.0', label = f'{b} - REF BAND')
            continue

        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # when doing the polyfit, it keeps giving 'RankWarning: Polyfit may be poorly conditioned', so chatGPT suggested to try scaling down the x values input 
        # to correct for x scaling after the polyfit has been taken, we just need to generate the MJDs at which we want the polyfit evaluated, then input (MJD - MJD_scaleconst)
        # into the polynomial fit instead of just MJD. To correct for the y scaling, just need to multiply the L_rf calculated by polyfit by 1/L_rf_scalefactor 
        MJD_scaled = b_lim_df['wm_MJD'] - MJD_scaleconst
        L_rf_scaled = b_lim_df['wm_L_rf']*L_rf_scalefactor # also scaling down the y values

        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # do the polynomial fit 
        poly_coeffs = np.polyfit(MJD_scaled, L_rf_scaled, deg = fit_order)
        poly_coeffs = list(poly_coeffs)
        polynomial_fit_scaled = np.poly1d(poly_coeffs, r = False) # r = False just measn that I am providing the poly coefficients, not the roots
        poly_plot_MJD = np.arange(b_lim_df['wm_MJD'].min(), b_lim_df['wm_MJD'].max(), 1) # the MJD values at which the polyfit will be evaluated
        poly_plot_MJD_scaled = poly_plot_MJD - MJD_scaleconst # the scaled MJDs to input into the polynomial fit
        poly_plot_L_rf_scaled = polynomial_fit_scaled(poly_plot_MJD_scaled)
        poly_plot_L_rf = poly_plot_L_rf_scaled / L_rf_scalefactor

        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # calculate chi squared of the polyfit
        poly_MJD_for_chi_scaled = b_lim_df['wm_MJD'] - MJD_scaleconst
        poly_L_rf_for_chi_scaled = polynomial_fit_scaled(poly_MJD_for_chi_scaled) 
        poly_L_rf_for_chi = poly_L_rf_for_chi_scaled / L_rf_scalefactor
        red_chi, red_chi_1sig = chisq(y_m = poly_L_rf_for_chi, y = b_lim_df['wm_L_rf'], yerr = b_lim_df['wm_L_rf_err'], M = (fit_order + 1))
        chi_sigma_dist = abs(1 - red_chi) / red_chi_1sig


        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # using our chosen 'trusted' band, use the polyfit to calculate the L_rf at the MJD values present in our trusted band's data 
        if extrapolate == True: # if we're extrapolating, then evaluate the polynomial at all of the MJDs of the trusted band, even if it's way outside the region over which we have data for this band
            interp_MJD = ref_band_MJD
            interp_MJD_scaled = ref_band_MJD_scaled

        else: # if we're not extrapolating, only evaluate the band's polyfit at the trusted band's mjd values which are within the mjd region covered by the band's actual data
            interp_MJD = [mjd for mjd in ref_band_MJD if (mjd >= b_lim_df['wm_MJD'].min()) and (mjd <= b_lim_df['wm_MJD'].max() )] 
            interp_MJD_scaled = np.array(interp_MJD) - MJD_scaleconst
        
        interp_L_rf_scaled = polynomial_fit_scaled(interp_MJD_scaled)
        interp_L_rf = interp_L_rf_scaled / L_rf_scalefactor


        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # calculate the fudged errors on the polyfit L_rf values
        interp_column_len = len(interp_MJD)
        interp_L_rf_err = fudge_polyfit_L_rf_err(b_df, interp_L_rf_scaled, interp_MJD_scaled, MJD_scaleconst, L_rf_scalefactor)
        interp_b_df = pd.DataFrame({'MJD': interp_MJD, 
                                    'L_rf': interp_L_rf, 
                                    'L_rf_err': interp_L_rf_err, 
                                    'band': [b] * interp_column_len, 
                                    'em_cent_wl': [b_em_cent_wl] * interp_column_len 
                                    })

        polyfit_ref_lc_df = pd.concat([polyfit_ref_lc_df, interp_b_df], ignore_index = True)

        plot_polyline_row[:] = [b, poly_plot_MJD, poly_plot_L_rf, red_chi, red_chi_1sig, chi_sigma_dist]
        plot_polyline_df.loc[plot_polyline_rowno] = plot_polyline_row
        # PLOTTING ====================================================================================================================================================================
        if plot_polyfit == True:
            b_colour = b_colour_dict[b]

            plt.errorbar(b_df['wm_MJD'], b_df['wm_L_rf'], yerr = b_df['wm_L_rf_err'], fmt = 'o', markeredgecolor = 'k', markeredgewidth = '1.0', linestyle = 'None', 
                         label = b, c = b_colour)
            plt.plot(poly_plot_MJD, poly_plot_L_rf, c = b_colour, label = f'red chi = {red_chi:.3f}  \n +/- {red_chi_1sig:.3f}')
            plt.errorbar(interp_b_df['MJD'], interp_b_df['L_rf'], yerr = interp_b_df['L_rf_err'], fmt = '^', c = b_colour, markeredgecolor = 'k', markeredgewidth = '0.5', 
                         linestyle = 'None', alpha = 0.5, capsize = 5, capthick = 5)
            
    if plot_polyfit == True:
        plt.xlabel('MJD')
        plt.ylabel('rest frame luminosity')
        plt.title(f'{ant_name} polyfit order = {fit_order}, reference band = {trusted_band}')
        plt.legend(loc = 'lower right', bbox_to_anchor = (1.2, 0.0), fontsize = 9)
        fig.subplots_adjust(right = 0.845, left = 0.07)
        plt.grid()
        plt.show()

    return polyfit_ref_lc_df, plot_polyline_df







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
# BLACLBODY MODEL FOR REST FRAME LUMINOSITY




def blackbody(lam_cm, R_cm, T_K):
    """
    Planck's blackbody formula modified to give luminosity per unit wavelength in units ergs/s/Angstrom

    INPUTS
    --------------
    lam: the wavelength in cm

    R_cm: Blackbody radius in cm - a parameter to fit for

    T_K: Blackbody temperature in Kelvin - a parameter to fit for

    RETURNS
    --------------
    L: blackbody luminosity per unit wavelength for the wavelength input. Units: ergs/s/Angstrom
    """

    c_cgs = const.c.cgs.value
    h_cgs = const.h.cgs.value
    k_cgs = const.k_B.cgs.value

    C = 8 * (np.pi**2) * h_cgs * (c_cgs**2) * 1e-8 # the constant coefficient of the equation 
    denom = np.exp((h_cgs * c_cgs) / (lam_cm * k_cgs * T_K)) - 1

    L = C * ((R_cm**2) / (lam_cm**5)) * (1 / (denom)) # ergs/s/Angstrom

    return L






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
# BLACK BODY FITTING ACROSS THE WHOLE (BINNED) LIGHT CURVE




def fit_BB_across_lc(interp_df, brute, curvefit):
    """


    INPUTS
    ---------------
    interp_df: the ANT's dataframe containing a light curve which has been interpolated using a polynomial fit to each band. 
        Each ANT had a chosen reference band. At the MJD values present in the reference band's real data, the polyfit for all other bands
        were evaluated (provided that we aren't extrapolating). This means that if there was a band which had data at the min and max 
        MJDs of the flare, there will be interpolated data for this band across the whole flare light curve, whereas if there is a band
        which only has data on the plateau of the light curve, this band will only have interpolated data within this region, at
        the MJD values of the reference band's data. This means that we don't need to bin the light curve in order to take the data
        for the blackbody fit, we can take each MJD present within this dataframe and take whatever band is present at this MJD as
        the data for the BB fit. So, we can fit a BB curve for each MJD within this dataframe, as long as it has >2 bands present. Prior
        to being interpolated, the ANT data should (ideally) be binned into small bins like 1 day, meaning that we will only have 0 or 1 datapoint 
        per per band per MJD value (this is only really relevant for the reference band, though, since the interpolation function ensures either 0 or 1
        value per band per MJD value for the interpolated data, since the polynomials are single-valued for any given MJD value)

    brute: if True, the BB fit will be tried using the brute force method (manually creating a grid of trial parameter values and minimising the chi squared). If 
        False, no brute force calculation will be tried

    curvefit: if True, the BB fit will be tried using scipy's curve_fit. If False, no curve_fit calculation will be tried


    RETURNS
    ---------------

    """
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # establish the bounds of the parameter space which we will investigate, as well as establishing a scaled radius (I don't scale the temp because I don't really k)
    BB_R_min = 1e13 
    BB_R_max = 1e19 
    BB_T_min = 1e3
    BB_T_max = 1e7


    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # scale down the radius values to explore - this requires scaling down the rest frame luminosity by (R_scalefactor)**2 because L ~ R^2 
    R_scalefactor = 1e-16
    L_scalefactor = (R_scalefactor)**2
    interp_df['L_rf_scaled'] = interp_df['L_rf'] * L_scalefactor # scale down the rest frame luminosity and its error 
    interp_df['L_rf_err_scaled'] = interp_df['L_rf_err'] * L_scalefactor
    BB_R_min_sc = BB_R_min * R_scalefactor # scaling down the bounds for the radius parameter space
    BB_R_max_sc = BB_R_max * R_scalefactor
    interp_df['em_cent_wl_cm'] = interp_df['em_cent_wl'] * 1e-8 # the blackbody function takes wavelength in centimeters. 1A = 1e-10 m.     1A = 1e-8 cm


    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # iterate through each value of MJD within the dataframe and see if we have enough bands to take a BB fit to it 

    mjd_values = interp_df['MJD'].unique() 
    columns = ['MJD', 'no_bands', 'cf_T_K', 'cf_T_err_K', 'cf_R_cm', 'cf_R_err_cm', 'cf_red_chi', 'cf_chi_sigma_dist', 'red_chi_1sig', 'brute_T_K', 'brute_R_cm', 'brute_red_chi', 'brute_chi_sigma_dist']
    BB_fit_results = pd.DataFrame(columns = columns)
    for MJD in tqdm(mjd_values, desc = 'Progress BB fitting each MJD value', total = len(mjd_values), leave = True):
        MJD_df = interp_df[interp_df['MJD'] == MJD].copy() # THERE COULD BE FLOATING POINT ERRORS HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        MJD_no_bands = len( MJD_df['band'].unique() ) # the number of bands (and therefore datapoints) we have available at this MJD for the BB fit
        df_row_index = len(BB_fit_results['MJD'])
        BB_result_row = np.zeros(len(columns))
        BB_result_row[:] = np.nan # set all values to nan for now, then overwrite them if we have data for thsi column, so that if (e.g.) brute = False, then the brute columns would contain nan values
        BB_result_row[0:2] = [MJD, MJD_no_bands] # the first column in the dataframe is MJD, so set the first value in the row as the MJD
       
        if MJD_no_bands <= 2: # if there's <= 2 bands present for a particular MJD, don't bother fitting a BB 
            BB_result_row[2: -1] = np.nan
            BB_fit_results.loc[df_row_index] = BB_result_row # adding the array of results from this MJD to the BB results dataframe
            continue

        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # because of the way that interp_df is set up, we don't need to bin by MJD for BB data and we only have 0 or 1 datapoint per band per MJD value, so no need to average the data for ecah band. 
        # so, MJD_df actually contains all of the necessary data for the BB fit. 
        # curve_fit fitting 
        if curvefit == True:
            popt, pcov = opt.curve_fit(blackbody, xdata = MJD_df['em_cent_wl_cm'], ydata = MJD_df['L_rf_scaled'], sigma = MJD_df['L_rf_err_scaled'], absolute_sigma = True, 
                                       bounds = (np.array([BB_R_min_sc, BB_T_min]), np.array([BB_R_max_sc, BB_T_max])))
            sc_cf_R, cf_T = popt
            sc_cf_R_err = np.sqrt(pcov[0, 0])
            cf_T_err = np.sqrt(pcov[1, 1])
            cf_R = sc_cf_R / R_scalefactor
            cf_R_err = sc_cf_R_err / R_scalefactor


            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # calculate the reduced chi squared of the curve_fit result
            BB_sc_L_chi = [blackbody(wl_cm, sc_cf_R, cf_T) for wl_cm in MJD_df['em_cent_wl_cm']] # evaluating the BB model from curve_fit at the emitted central wavelengths present in our data to use for chi squared calculation
            cf_red_chi, red_chi_1sig = chisq(y_m = BB_sc_L_chi, y = MJD_df['L_rf_scaled'], yerr = MJD_df['L_rf_err_scaled'], M = 2, reduced_chi = True)
            cf_chi_sigma_dist = abs(1 - cf_red_chi)/red_chi_1sig

            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # add the result to the results row which will be appended to the results dataframe
            BB_result_row[2:9] = [cf_T, cf_T_err, cf_R, cf_R_err, cf_red_chi, cf_chi_sigma_dist, red_chi_1sig]


        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # brute force grid curve fitting
        if brute == True:
            # creating the values of R and T that we will try
            grid_length = 100 # the number of R and T values to trial in the grid. The combinations of R and T form a 2D grid, so the number of R and T values that we try give the side lengths of the grid
            sc_R_values = np.logspace(np.log10(BB_R_min_sc), np.log10(BB_R_max_sc), grid_length)
            T_values = np.logspace(np.log10(BB_T_min), np.log10(BB_T_max), grid_length)
            chi_grid = np.zeros((grid_length, grid_length)) # this 2D array will fill with the chi squared values of each blackbody fit tried with different combinations of BB T and scaled R
            for i, T_K in enumerate(T_values):
                for j, sc_R_cm in enumerate(sc_R_values):
                    BB_L_sc = [blackbody(wl_cm, sc_R_cm, T_K) for wl_cm in MJD_df['em_cent_wl_cm']] # the calculated value of scaled rest frame luminosity using this value of T and scaled R
                    # calculate the NON reduced chi squared here because for each MJD_df that we try to fit a BB to, the value of N-M will be a const regardless of the value of R and T tried, 
                    # therefore we can calculate the reduced chi squareds afterwards by taking chi_grid/( N-M ) and the 1 sigma uncertainty using sqrt(2/ (N-M)), instead of re-calculating these
                    # with every attempted combo of R and T
                    chi = chisq(y_m = BB_L_sc, y = MJD_df['L_rf_scaled'], yerr = MJD_df['L_rf_err_scaled'], M = 2, reduced_chi = False) # calculate the chi squared for this BB fit (NOT the reduced chi)
                    chi_grid[i, j] = chi

            min_chi = np.min(chi_grid)
            row, col = np.where(chi_grid == min_chi)

            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # make sure that there i sonly one combination of parameters which gives this minimum chi squared value
            if (len(row) == 1) & (len(col) == 1): 
                r = row[0]
                c = col[0]
                brute_T = T_values[r] # the parameters which give the minimum chi squared
                brute_R = sc_R_values[c] / R_scalefactor
                N_M = len(MJD_df['band']) - 2
                brute_red_chi = min_chi / N_M 
                red_chi_1sig = np.sqrt(2/N_M)
                brute_chi_sigma_dist = abs(1 - brute_red_chi) / red_chi_1sig
            else:
                print()
                print(f"WARNING - MULTIPLE R AND T PARAMETER PAIRS GIVE THIS MIN CHI VALUE. MJD = {MJD_df['MJD'].iloc[0]} \n Ts = {[T_values[r] for r in row]}, Rs = {[sc_R_values[c]/R_scalefactor for c in col]}")
                print()

            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # add the result to the results row which will be appended to the results dataframe
            BB_result_row[8:13] = [red_chi_1sig, brute_T, brute_R, brute_red_chi, brute_chi_sigma_dist]

        BB_fit_results.loc[df_row_index] = BB_result_row # adding the array of results from this MJD to the BB results dataframe


    return BB_fit_results



















