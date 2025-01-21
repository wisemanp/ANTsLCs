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
from plotting_preferences import band_colour_dict, band_ZP_dict, band_obs_centwl_dict, ANT_redshift_dict, ANT_luminosity_dist_cm_dict, MJDs_for_fit
from functions import load_ANT_data, ANT_data_L_rf, bin_lc, fit_BB_across_lc, chisq, polyfit_lc







def polyfit_lc2(ant_name, df, fit_order, df_bands, trusted_band, min_band_dps, fit_MJD_range, extrapolate, b_colour_dict, plot_polyfit = False):
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

    extrapolate: True if you want to extrapolate with the polyfits to the MJDs specified in fit_MJD_range, False if not. (bool)

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
    # =========================================================================================================================================================================================================
    # =========================================================================================================================================================================================================
    def fudge_polyfit_L_rf_err(real_b_df, scaled_polyfit_L_rf, scaled_reference_MJDs, MJD_scaledown, L_rf_scaledown):
        """
        Fudges the uncertainties on the rest frame luminosity values calculated using the polynomial fit of the band,
        evaluated at the trusted_band's MJD values. 

        Fudged_L_rf_err = mean(L_rf_err values of the 20 closest datapoints in MJD) + 0.05*polyfit_L_rf*|mjd - mjd of 
        closest datapoint in the band's real data|/10

        The first term is the mean of the closest 20 datapoints in MJD's rest frame luminosity errors. The second term 
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
    # =========================================================================================================================================================================================================
    # =========================================================================================================================================================================================================

    # setting up the plot
    if plot_polyfit == True:
        fig = plt.figure(figsize = (16, 7.5))
    
    # limiting the MJD over which we are polyfitting, because for some ANTs, we have some straggling datapoints far away from the rest of the light curve and we don't want to fit these
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
    
    columns = ['band', 'poly_MJD', 'poly_L_rf', 'red_chi', 'red_chi_1sig', 'chi_sigma_dist'] # allows us to plot the polyfit as a line using plt.plot() and the interpolated values.                                                                 
    plot_polyline_df = pd.DataFrame(columns = columns)
    
    # iterate through the bands and polyfit them ===========================================================================================================================================
    for i, b in enumerate(df_bands):
        b_df = df[df['band'] == b].copy()
        b_lim_df = lim_df[lim_df['band'] == b].copy() # the dataframe for the band, with MJD values limited to the main light curve
        b_em_cent_wl = b_df['em_cent_wl'].iloc[0] # take the first value here because this dataframe only contains data from 1 band anyways so all em_cent_wl values will be the same
        plot_polyline_rowno = len(plot_polyline_df['band']) # the row number for this band's data (because we're appending each row to the last row of the dataframe essentially)
        plot_polyline_row = list(np.zeros(len(columns))) # fill the plot data with zeros then overwrite these values
        if b == trusted_band: # we don't need a polyfit of the trusted band because we're just evaluating the polyfits of all other bands at the trusted_band's MJDs
            if plot_polyfit == True:
                plot_polyline_row[:] = [trusted_band, np.nan, np.nan, np.nan, np.nan, np.nan]
                plot_polyline_df.loc[plot_polyline_rowno] = plot_polyline_row # append with this band's data

                b_colour = b_colour_dict[b]
                plt.errorbar(b_df['wm_MJD'], b_df['wm_L_rf'], yerr = b_df['wm_L_rf_err'], c = b_colour, fmt = 'o', linestyle = 'None', markeredgecolor = 'k', 
                            markeredgewidth = '1.0', label = f'{b} - REF BAND')
            continue

        #print(b_lim_df['wm_MJD'])
        elif len(b_lim_df['wm_MJD']) < min_band_dps: # GETTING RID OF BANDS WITH TOO LITTLE DATA ********************************************************************************************************************
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
        poly_plot_MJD = np.arange(b_lim_df['wm_MJD'].min(), b_lim_df['wm_MJD'].max(), 1) # the MJD values at which the polyfit will be evaluated for displaying it on the plot, not for interpolation
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
            interp_MJD = [mjd for mjd in ref_band_MJD if (mjd >= b_lim_df['wm_MJD'].min()) and (mjd <= b_lim_df['wm_MJD'].max() )] ########################################################### HERE IS WHERE WE CHANGE THE INTERPOLATION
            interp_MJD_scaled = np.array(interp_MJD) - MJD_scaleconst
        
        #band_int_lim_C = len(b_lim_df['MJD'].copy()) / (b_lim_df['MJD'].max() - b_lim_df['MJD'].min()) # the constant factor in my interpolation limit equation for the band.  = no of datapoints in band's real data / its MJD span
        #for j, int_mjd in enumerate(interp_MJD_scaled): # loop through the MJD values at which we're trying to interpolate at and don't allow interpolation beyond a certain limit of MJD
        #    mjd_dif = [abs(int_mjd - MJD_scaled[k]) for k in range(len(MJD_scaled))] # the MJD difference between this MJD that wer're trying to interpolate at (int_mjd) and the MJDs present in the band's real data 
            ############################################################ SHOULD I TECHNICALLY USE THE BAND'S FULL DATASET FOR THIS (above) INSTEAD OF THE MJD LIMITED ONE ############################################################
        #     count_within_50 = len([el for el in mjd_dif if el <=50.0]) # a count of the number of the band's real datapoints 50 days either side of the int_mjd value
            #interp_limit = 


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
                         linestyle = 'None', alpha = 0.5,  capsize = 5, capthick = 5)
            
    if plot_polyfit == True:
        plt.xlabel('MJD')
        plt.ylabel('rest frame luminosity')
        plt.title(f'{ant_name} polyfit order = {fit_order}, reference band = {trusted_band}')
        plt.legend(loc = 'lower right', bbox_to_anchor = (1.2, 0.0), fontsize = 9)
        fig.subplots_adjust(right = 0.845, left = 0.07)
        plt.grid()
        plt.show()

    return polyfit_ref_lc_df, plot_polyline_df





















# load in the data
lc_df_list, transient_names, list_of_bands = load_ANT_data()

# calculate the rest frame luminosity + emitted central wavelength
add_lc_df_list = ANT_data_L_rf(lc_df_list, transient_names, ANT_redshift_dict, ANT_luminosity_dist_cm_dict, band_ZP_dict, band_obs_centwl_dict)

# bin up the light curve into 1 day MJD bins
MJD_binsize = 1
binned_df_list = bin_lc(add_lc_df_list, MJD_binsize)


# polyfitting ONE light curve
idx = 10
ANT_name = transient_names[idx]
ANT_df = binned_df_list[idx]
ANT_bands = list_of_bands[idx]
reference_band = 'ZTF_g'
bands_for_BB = [b for b in ANT_bands if (b != 'WISE_W1') and (b != 'WISE_W2')] # remove the WISE bands from the interpolation since we don't want to use this data for the BB fit anyway




polyfit_MJD_range = MJDs_for_fit[ANT_name]
#polyfit_MJD_range = (58500, 58770)
interp_lc, plot_polyfit_df = polyfit_lc2(ANT_name, ANT_df, fit_order = 9, df_bands = bands_for_BB, min_band_dps = 10, trusted_band = reference_band, fit_MJD_range = polyfit_MJD_range,
                        extrapolate = False, b_colour_dict = band_colour_dict, plot_polyfit = True)

#interp_lc, plot_polyfit_df = polyfit_lc2(ANT_name, ANT_df, fit_order = 5, df_bands = bands_for_BB, trusted_band = reference_band, fit_MJD_range = MJDs_for_fit[ANT_name],
#                        extrapolate = False, b_colour_dict = band_colour_dict, plot_polyfit = True)









