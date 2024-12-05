import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from astropy import constants as const
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
import scipy.optimize as opt





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
    
    columns = ['band', 'poly_MJD', 'poly_L_rf'] # allows us to plot the polyfit as a line and the interpolated values.                                                                 
    plot_polyline_df = pd.DataFrame(columns = columns)
    # iterate through the bands and polyfit them ===========================================================================================================================================
    for i, b in enumerate(df_bands):
        b_df = df[df['band'] == b].copy()
        b_lim_df = lim_df[lim_df['band'] == b].copy() # the dataframe for the band, with MJD values limited to the main light curve
        b_em_cent_wl = b_df['em_cent_wl'].iloc[0] # take the first value here because this dataframe only contains data from 1 band anyways so all em_cent_wl values will be the same
        plot_polyline_rowno = len(plot_polyline_df['band']) # the row number for this band's data
        plot_polyline_row = np.zeros(len(columns)) # fill the plot data with zeros then overwrite these values
        if b == trusted_band: # we don't need a polyfit of the trusted band because we're just evaluating the polyfits of all other bands at the trusted_band's MJDs
            if plot_polyfit == True:
                plot_polyline_row[:] = [trusted_band, np.nan, np.nan]
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

        plot_polyline_row[:] = [b, poly_plot_MJD, poly_plot_L_rf]
        plot_polyline_df.loc[plot_polyline_rowno] = plot_polyline_row
        # PLOTTING ====================================================================================================================================================================
        if plot_polyfit == True:
            b_colour = b_colour_dict[b]

            plt.errorbar(b_df['wm_MJD'], b_df['wm_L_rf'], yerr = b_df['wm_L_rf_err'], fmt = 'o', markeredgecolor = 'k', markeredgewidth = '1.0', linestyle = 'None', 
                         label = b, c = b_colour)
            plt.plot(poly_plot_MJD, poly_plot_L_rf, c = b_colour, label = f'red chi = {red_chi:.3f}  \n +/- {red_chi_1sig:.3f}')
            plt.errorbar(interp_b_df['MJD'], interp_b_df['L_rf'], yerr = interp_b_df['L_rf_err'], fmt = '^', c = b_colour, markeredgecolor = 'k', markeredgewidth = '0.5', 
                         linestyle = 'None', alpha = 0.5)
            
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
# 