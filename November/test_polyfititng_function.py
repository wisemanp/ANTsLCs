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


def polyfit_lcs(ant_name, df, fit_order, df_bands, trusted_band, fit_MJD_range, extrapolate, b_colour_dict, plot_polyfit = False):
    """
    Peforms a polynomial fit for each band within df_bands for the light curve. 

    INPUTS
    ---------------
    ant_name: the ANT's name (str)

    df: a dataframe containing the ANT's light curve, ideally this would be binned into 1 day bins or something so that we don't get a flare of scattered L_rf values at ~= MJD
         which confuses the polyfit funciton. It is assumed that there are columns within df named: 'wm_L_rf' - the weighted mean rest frame luminosity within the MJD bin, 
         'wm_L_rf_err' the error on this weighted mean rest frame luminosity and 'wm_MJD', the weighted mean MJD value within the bin. (dataframe)

    df_bands: a list of the bands within the ANT's lightcurve that you want to have a polyfit for. If df contains WISE_W1 data, if you don't want this to be polyfitted, just don't include
               'WISE_W1' in df_bands. (list)

    trusted_band: the name of the band which is used as the reference band for the light curve. This band would ideally have a high cadence and good coverage of the light curve. 
                  at the MJD values for which df contains data on this band, this function will use the polynomial fit for each other band and evaluate L_rf for each band at this MJD. 
                  This band is also the one from which we will calcuate the peak of the light curve. (str)

    fit_MJD_range: a tuple like (min_polyfit_MJD, max_polyfit_MJD), giving the MJD limits between which you want the bands in df_bands to be polyfitted. (tuple)

    extrapolate: True if you want to extrapolate with the polyfits, False if not. (bool)

    b_colour_dict: a dictionary indicating the marker colours for each photometric band such that b_colour_dict['ZTF_g'] = ZTF_g_colour (dict)

    plot_polyfit: True if you want a plot of the polfit over the top of the actual light curve's data, as well as the interpolated data taken from the polyfit. Plot also displays the
                    reduced chi squareds for each band's polyfit. Default is False (bool)

    
    OUTPUTS
    ---------------
    polyfit_ref_lc_df: a dataframe containing MJD, L_rf, L_rf_err, band for: the trusted_band's (also referred to as the reference band) actual MJD, L_rf, L_rf_err data, and then for 
                        the other bands, their data was 'interpolated' using a polynomial fit to the band's data which was evaluated at the MJD values of the trusted band. This funciton 
                        both interpolates and extrapolates using the polyfit, so at any MJD value present within this dataframe, there will be a L_rf value for every band within df_bands. 
                        This is good if you wanted to do something like blackbody fitting, since you then have datapoints for lots of bands at a given MJD, allowing for a better blackbody
                        fit. 

                        NOTE: L_rf_err in polyfit_ref_lc_df for all bands except the trusted_band is calculated using a fudge formula

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
            er = mean_err + 0.05 * x #+ 0.0001*x**2

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
            #sc_interp_err_term = abs( 0.05 * sc_L * (closest_MJD_diff/10) ) # this adds some % error for every 10 days interpolated - inspired by Superbol who were inspired by someone else

            # our fudged error = mean wm_L_rf_err of the closest 10 datapoints in MJD + some % error for every 10 days interpolated
            #sc_poly_L_rf_er = sc_mean_L_rf_err + sc_interp_err_term # the fudged error on L_rf, scaled down
            sc_poly_L_rf_er = fudge_error_formula(sc_mean_L_rf_err, closest_MJD_diff, sc_L)
            #if sc_poly_L_rf_er > abs(sc_L): # don't let the error bars get larger than the data point itself
            #    sc_poly_L_rf_er = abs(sc_L)
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


    # getting the MJD values for the reference band --------------------------------------------------------------------------------------------------------------------------------------------
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
                                      'band': list(ref_band_df['band'].copy())
                                      })
    
                                                                      

    # iterate through the bands and polyfit them ===========================================================================================================================================
    for i, b in enumerate(df_bands):
        b_df = df[df['band'] == b].copy()
        b_lim_df = lim_df[lim_df['band'] == b].copy() # the dataframe for the band, with MJD values limited to the main light curve

        # plot the 
        if b == trusted_band: # we don't need a polyfit of the trusted band because we're just evaluating the polyfits of all other bands at the trusted_band's MJDs
            if plot_polyfit == True:
                b_colour = b_colour_dict[b]
                plt.errorbar(b_df['wm_MJD'], b_df['wm_L_rf'], yerr = b_df['wm_L_rf_err'], c = b_colour, fmt = 'o', linestyle = 'None', markeredgecolor = 'k', 
                            markeredgewidth = '1.0', label = f'{b} - REF BAND')
            continue

        
        # when doing the polyfit, it keeps giving 'RankWarning: Polyfit may be poorly conditioned', so chatGPT suggested to try scaling down the x values input -------------------------
        # to correct for x scaling after the polyfit has been taken, we just need to generate the MJDs at which we want the polyfit evaluated, then input (MJD - MJD_scaleconst)
        # into the polynomial fit instead of just MJD. To correct for the y scaling, just need to multiply the L_rf calculated by polyfit by 1/L_rf_scalefactor 
        MJD_scaled = b_lim_df['wm_MJD'] - MJD_scaleconst
        L_rf_scaled = b_lim_df['wm_L_rf']*L_rf_scalefactor # also scaling down the y values


        # do the polynomial fit ---------------------------------------------------------------------------------------------------------------------------------------------------------
        poly_coeffs = np.polyfit(MJD_scaled, L_rf_scaled, deg = fit_order)
        poly_coeffs = list(poly_coeffs)
        polynomial_fit_scaled = np.poly1d(poly_coeffs, r = False) # r = False just measn that I am providing the poly coefficients, not the roots
        poly_plot_MJD = np.arange(b_lim_df['wm_MJD'].min(), b_lim_df['wm_MJD'].max(), 1) # the MJD values at which the polyfit will be evaluated
        poly_plot_MJD_scaled = poly_plot_MJD - MJD_scaleconst # the scaled MJDs to input into the polynomial fit
        poly_plot_L_rf_scaled = polynomial_fit_scaled(poly_plot_MJD_scaled)
        poly_plot_L_rf = poly_plot_L_rf_scaled / L_rf_scalefactor


        # calculate chi squared of the polyfit
        poly_MJD_for_chi_scaled = b_lim_df['wm_MJD'] - MJD_scaleconst
        poly_L_rf_for_chi_scaled = polynomial_fit_scaled(poly_MJD_for_chi_scaled) 
        poly_L_rf_for_chi = poly_L_rf_for_chi_scaled / L_rf_scalefactor
        red_chi, red_chi_1sig = chisq(y_m = poly_L_rf_for_chi, y = b_lim_df['wm_L_rf'], yerr = b_lim_df['wm_L_rf_err'], M = (fit_order + 1))


        # using our chosen 'trusted' band, use the polyfit to calculate the L_rf at the MJD values present in our trusted band's data ----------------------------------------------------
        if extrapolate == True: # if we're extrapolating, then evaluate the polynomial at all of the MJDs of the trusted band, even if it's way outside the region over which we have data for this band
            interp_MJD = ref_band_MJD
            interp_MJD_scaled = ref_band_MJD_scaled

        else: # if we're not extrapolating, only evaluate the band's polyfit at the trusted band's mjd values which are within the mjd region covered by the band's actual data
            interp_MJD = [mjd for mjd in ref_band_MJD if (mjd >= b_lim_df['wm_MJD'].min()) and (mjd <= b_lim_df['wm_MJD'].max() )] 
            interp_MJD_scaled = np.array(interp_MJD) - MJD_scaleconst
        
        interp_L_rf_scaled = polynomial_fit_scaled(interp_MJD_scaled)
        interp_L_rf = interp_L_rf_scaled / L_rf_scalefactor


        # calculate the fudged errors on the polyfit L_rf values
        interp_L_rf_err = fudge_polyfit_L_rf_err(b_df, interp_L_rf_scaled, interp_MJD_scaled, MJD_scaleconst, L_rf_scalefactor)
        interp_b_df = pd.DataFrame({'MJD': interp_MJD, 
                                    'L_rf': interp_L_rf, 
                                    'L_rf_err': interp_L_rf_err, 
                                    'band': [b]*(len(interp_MJD))
                                    })

        polyfit_ref_lc_df = pd.concat([polyfit_ref_lc_df, interp_b_df], ignore_index = True)

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




    return polyfit_ref_lc_df
    






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
polyfit_order = 6
want_plot = True
#ANT = 'ZTF22aadesap'
#ANT = 'ZTF19aailpwl'
# now, we want to fit polynomials to each band of the binned up light curves
for i, ANT in enumerate(transient_names):
    if i in [14, 15, 16]: # these ANT light curves have some magerr = 0.0, which causes issues with taking the weighted mean of L_rf so idk if the binning funciton would actually work
        continue

    if i == 1:
        reference_band = 'ZTF_g'
        print(f'{i}, {ANT}')
        print()
        ANT_bands = list_of_bands[i]
        binned_ANT_df = bin_lc_df_list[i].copy()
        poly_interp_df= polyfit_lcs(ANT, binned_ANT_df, fit_order = 5, df_bands = ANT_bands, trusted_band = reference_band, fit_MJD_range = MJDs_for_fit[ANT], extrapolate = False, b_colour_dict = band_colour_dict, plot_polyfit = True)








