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






def allow_interpolation(interp_x, all_data_x, band_lc_density, local_density_region = 50, interp_cap = 150, factor = 100, simple_cutoff = False, simple_cut = 50):
    """
    Here I have written a formula to determine whether or not to interpolate a band at a given mjd value based on
    how well-sampled the band is, so that poorly sampled bands cannot interpolate very far, and well-sampled bands can interpolate much further

    NOTE: there are edge cases where the band may be very well sampled over a small MJD range, at which point the formula may allow this to interpolate
    further than it should. However, if you bin the light curve into somehting like 1 day bins before polyfitting, at least you won't get weird little spikes of loads of 
    data making this worse. 

    INPUTS
    ------------
    interp_x: float, the x value at which we're trying to interpolate

    all_data_x: array of the x values of teh actual data that we're interpolating

    band_lc_density: float. band_lc_density = no. datapoints in the band / MJD span of the band. So, this is a const across the whole band

    local_density_region: The region over which we count the lcoal datapoint density. This would be: (interp_x - local_density_region) <= x <= (interp_x + local_density_region)

    interp_cap: the maximum value out to which we can interpolate. Only the well-sampled bands would reach this cap

    factor: a factor in ther equation. 

    RETURNS
    ------------
    interp_alowed: Bool. If True, then interpolation is allowed at interp_x, if False, interpolation si not allowed at interp_x

    MJD_diff: an array ocntaining the magnitude of the difference between interp_x and each value of x in all_data_x, which is used in the fudge error formula for interpolation
                (provided that we're allowed to interpolate anyway)
    """

    MJD_diff = abs(all_data_x - interp_x) # takes the MJD difference between interp_x and the MJDs in the band's real data
    local_density = (MJD_diff <= local_density_region).sum() # counts the number of datapoints within local_density_region days' of interp_x
    closest_MJD_diff = min(MJD_diff)

    if simple_cutoff == False:
        interp_lim = (band_lc_density * local_density * factor) 
        interp_lim = min(interp_cap, interp_lim) # returns the smaller of the two, so this caps the interpolation limit at interp_cap

        # THE MIDDLE SECTIONS GET PREFERENTIAL TREATMENT COMPARED TO THE CENTRAL ONES
        #if ((interp_x - local_density_region) <= min(all_data_x)) or ((interp_x + local_density_region) >= max(all_data_x)): # if interp_x is near thestart/end of the band's lightcurve
        #    interp_lim = interp_lim*2
        

        if closest_MJD_diff < interp_lim: # if interp_x lies within our calculated interpolation limit, then allow interpolation here
            interp_allowed = True
        else: 
            interp_allowed = False

    else:
        if closest_MJD_diff <= simple_cut: # if interp_x lies within our calculated interpolation limit, then allow interpolation here
            interp_allowed = True
        else: 
            interp_allowed = False

    return interp_allowed, MJD_diff
    










def polyfitting(b_df, mjd_scale_C, L_rf_scalefactor, max_poly_order):
    """
    This function uses chi squred minimisation to optimise the choice of the polynomial order to fit to a band in a light curve, and also uses curve_fit to find
    the optimal parameters for each polynomial fit. Bands with little data are not allowed to use higher order polynomials to fit them

    INPUTS:
    ---------------
    b_df: a dataframe of the (single-band) lightcurve which has been MJD-limited to the region that you want to be polyfitted. 

    mjd_scale_C: float. best as the mean MJD in the band or overall lightcurve. Used to scale-down the MJD values that we're fitting

    L_rf_scalefactor: float. Used to scale-down the rest frame luminosity. Usually = 1e-41.   

    max_poly_order: int between 3 <= max_poly_order <= 9. The maximum order of polynomial that you want to be allowed to fit. 


    OUTPUTS
    --------------------
    optimal_params: a list of the coefficients of from the polyfit in descenidng order, e.g. if we have ax^2 + bx + c, optimal params = [a, b, c] so the coefficient of the highest
                    order term goes first

    plot_polt_MJD: a list of MJD values at which we have evaluated the polyfit, for plotting purposes

    plot_polt_L: a list of rest frame luminosity values calculated using the polyfit, for plotting purposes

    best_redchi: the reduced chi squared of the optimal polynomial fit

    best_redchi_1sig: the reduced chi squared's one sigma tolerance of the optimal polynomial fit

    poly_sigma_dist: the reduced chi squared's sigma distance for the optimal polynomial fit (should ideally be 1)

    """
    def poly1(x, a, b):
        return b*x + a

    def poly2(x, a, b, c):
        return c*x**2 + b*x + a

    def poly3(x, a, b, c, d):
        return d*x**3 + c*x**2 + b*x + a

    def poly4(x, a, b, c, d, e):
        return e*x**4 + d*x**3 + c*x**2 + b*x + a

    def poly5(x, a, b, c, d, e, f):
        return f*x**5 + e*x**4 + d*x**3 + c*x**2 + b*x + a

    def poly6(x, a, b, c, d, e, f, g):
        return g*x**6 + f*x**5 + e*x**4 + d*x**3 + c*x**2 + b*x + a

    def poly7(x, a, b, c, d, e, f, g, h):
        return h*x**7 + g*x**6 + f*x**5 + e*x**4 + d*x**3 + c*x**2 + b*x + a

    def poly8(x, a, b, c, d, e, f, g, h, i):
        return i*x**8 + h*x**7 + g*x**6 + f*x**5 + e*x**4 + d*x**3 + c*x**2 + b*x + a

    def poly9(x, a, b, c, d, e, f, g, h, i, j):
        return j*x**9 + i*x**8 + h*x**7 + g*x**6 + f*x**5 + e*x**4 + d*x**3 + c*x**2 + b*x + a
    
    def poly10(x, a, b, c, d, e, f, g, h, i, j, k):
        return k*x**10 + j*x**9 + i*x**8 + h*x**7 + g*x**6 + f*x**5 + e*x**4 + d*x**3 + c*x**2 + b*x + a
    
    def poly11(x, a, b, c, d, e, f, g, h, i, j, k, l):
        return l*x**11 + k*x**10 + j*x**9 + i*x**8 + h*x**7 + g*x**6 + f*x**5 + e*x**4 + d*x**3 + c*x**2 + b*x + a
    
    def poly12(x, a, b, c, d, e, f, g, h, i, j, k, l, m):
        return m*x**12 + l*x**11 + k*x**10 + j*x**9 + i*x**8 + h*x**7 + g*x**6 + f*x**5 + e*x**4 + d*x**3 + c*x**2 + b*x + a
    
    def poly13(x, a, b, c, d, e, f, g, h, i, j, k, l, m, n):
        return n*x**13 + m*x**12 + l*x**11 + k*x**10 + j*x**9 + i*x**8 + h*x**7 + g*x**6 + f*x**5 + e*x**4 + d*x**3 + c*x**2 + b*x + a
    
    def poly14(x, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o):
        return o*x**14 + n*x**13 + m*x**12 + l*x**11 + k*x**10 + j*x**9 + i*x**8 + h*x**7 + g*x**6 + f*x**5 + e*x**4 + d*x**3 + c*x**2 + b*x + a


    poly_order_dict = {1: poly1,  # a dictionary of polynomial orders and their associated functions, used for curve_fit in the polyfitting function
                        2: poly2, # feels like there should be a much more efficient way to write this I am slightly ashamed lol
                        3: poly3, 
                        4: poly4, 
                        5: poly5, 
                        6: poly6, 
                        7: poly7, 
                        8: poly8, 
                        9: poly9, 
                        10: poly10, 
                        11: poly11, 
                        12: poly12, 
                        13: poly13, 
                        14: poly14}

    # prepping the data for fitting
    b_L_scaled = b_df['wm_L_rf'].copy() * L_rf_scalefactor # scaled L_rf
    b_L_err_scaled = b_df['wm_L_rf_err'].copy() * L_rf_scalefactor # scaled_L_rf_err
    b_MJD_scaled = b_df['wm_MJD'].copy() - mjd_scale_C # scaled MJD

    # calculate how well-sampled the band is 
    b_MJD_span = b_df['wm_MJD'].max() - b_df['wm_MJD'].min() # the span of MJDs that the band makes
    b_count = len(b_df['wm_MJD']) # the number of datapoints in the band's data

    # restrict the order of polynomial available to poorly sampled band lightcurves
    if b_MJD_span < 30:
        poly_orders_available = [1]

    elif (b_MJD_span < 100) or (b_count/b_MJD_span) < 0.01 or (b_count < 20):
        poly_orders_available = [1, 2, 3]
    
    elif (b_MJD_span < 500) and (b_count < 10):
        poly_orders_available = [1, 2]

    else:
        poly_orders_available = np.arange(1, (max_poly_order + 1), 1)

    if b_count in poly_orders_available: # if the number of datapoints = polynomial order
        idx = poly_orders_available.index(b_count)
        print(f'FLAG: {b_count}, {poly_orders_available}')
        poly_orders_available = poly_orders_available[:(idx - 1)] # this should be fine since we have a lower limit of 2 datapoints right now
        print(f'NEW polyorders = {poly_orders_available}')

    # iterate thriugh different polynomial orders
    best_redchi = 1e10 # start off very high so it's immediately overwritten by the first fit's results
    for order in poly_orders_available: 
        poly_function = poly_order_dict[order]
        popt, pcov = opt.curve_fit(poly_function, xdata = b_MJD_scaled, ydata = b_L_scaled, sigma = b_L_err_scaled)
        
        # now calculate the reduced chi squared of the polynomial fit
        polyval_coeffs = popt[::-1] # using popt[::-1] just reverses the array because my polynomial functions inputs go in ascending order coefficients, whereeas polyval does the opposite
        chi_sc_poly_L = np.polyval(polyval_coeffs, b_MJD_scaled) 
        redchi, redchi_1sig = chisq(chi_sc_poly_L, b_L_scaled, b_L_err_scaled, M = order + 1, reduced_chi = True)
        
        # if we get a better reduced chi squared than before, overwrite the optimal parameters
        if pd.notna(redchi):
            if (redchi < best_redchi): 
                best_redchi = redchi
                best_redchi_1sig = redchi_1sig
                optimal_params = polyval_coeffs
        
    #print(len(optimal_params), redchi, '+/-', redchi_1sig)
    
    poly_sigma_dist = abs(1 - best_redchi)/(best_redchi_1sig)
    plot_poly_sc_MJD = np.arange(min(b_MJD_scaled), max(b_MJD_scaled), 1.0) # for plotting the polynomial fit
    #plot_poly_sc_L = np.polyval(polyval_coeffs, plot_poly_sc_MJD) # for plotting the polynomial fit
    plot_poly_sc_L = np.polyval(optimal_params, plot_poly_sc_MJD)

    plot_poly_MJD = plot_poly_sc_MJD + mjd_scale_C
    plot_poly_L = plot_poly_sc_L/L_rf_scalefactor

    return optimal_params, plot_poly_MJD, plot_poly_L, best_redchi, best_redchi_1sig, poly_sigma_dist







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

    scaled_polyfit_L_rf: a list/array of the polyfitted scaled rest frame luminosity values which were evaluated at the 
                    trusted band's MJD values (the reference_MJDs)

    scaled_reference_MJDs: a list/array of scaled MJD values which are the values at which the trusted band has data. This should
                    be restricted to the region chosen to fit the polynomials to, not the
                    entire light curve with any little stragglers far before the start/'end', i.e. should be limited 
                    by fit_MJD_range from the outer function.

    MJD_scaledown: float. A scale constant to scale-down the MJD values. 

    L_rf_scaledown: float. A scale factor to scale-down the rest frame luminosity

                    
    OUTPUTS
    ---------------
    poly_L_rf_err_list: a list of our fudged L_rf_err values for our polyfit interpolated data. 

    """
    def fudge_error_formula(mean_err, mjd_dif, L_scaled):
        x = abs( L_scaled * (mjd_dif / 10) )
        er = mean_err + 0.05 * x  # this adds some % error for every 10 days interpolated - inspired by Superbol who were inspired by someone else

        if er > abs(L_scaled):
            er = abs(L_scaled)

        return er
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

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










def polyfit_lc4(ant_name, df, df_bands, trusted_band, max_poly_order, min_band_dps, fit_MJD_range, extrapolate, b_colour_dict, plot_polyfit = False):
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

    max_poly_order: int between 3 <= max_poly_order <= 9. The maximum order of polynomial that you want to be allowed to fit. 

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
    

    columns = ['band', 'poly_coeffs', 'poly_plot_MJD', 'poly_plot_L_rf', 'red_chi', 'red_chi_1sig', 'chi_sigma_dist'] # allows us to plot the polyfit as a line using plt.plot() and the interpolated values.                                                                 
    plot_polyline_df = pd.DataFrame(columns = columns)
    
    # iterate through the bands and polyfit them ===========================================================================================================================================
    for i, b in enumerate(df_bands):
        print(b)
        b_df = df[df['band'] == b].copy()
        b_lim_df = lim_df[lim_df['band'] == b].copy() # the dataframe for the band, with MJD values limited to the main light curve
        b_em_cent_wl = b_df['em_cent_wl'].iloc[0] # take the first value here because this dataframe only contains data from 1 band anyways so all em_cent_wl values will be the same
        plot_polyline_rowno = len(plot_polyline_df['band']) # the row number for this band's data (because we're appending each row to the last row of the dataframe essentially)
        plot_polyline_row = list(np.zeros(len(columns))) # fill the plot data with zeros then overwrite these values

        

        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if len(b_lim_df['wm_MJD']) <= min_band_dps: # getting rid of bands with too little data
            plt.errorbar(b_df['wm_MJD'], b_df['wm_L_rf'], yerr = b_df['wm_L_rf_err'], fmt = 'o', markeredgecolor = 'k', markeredgewidth = '1.0', linestyle = 'None', 
                         label = b, c = b_colour_dict[b])
            continue

        b_lc_density = (len(b_lim_df['wm_MJD'])) / (b_lim_df['wm_MJD'].max() - b_lim_df['wm_MJD'].min()) # the density of datapoints across the band
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # when doing the polyfit, it keeps giving 'RankWarning: Polyfit may be poorly conditioned', so chatGPT suggested to try scaling down the x values input 
        # to correct for x scaling after the polyfit has been taken, we just need to generate the MJDs at which we want the polyfit evaluated, then input (MJD - MJD_scaleconst)
        # into the polynomial fit instead of just MJD. To correct for the y scaling, just need to multiply the L_rf calculated by polyfit by 1/L_rf_scalefactor 
        MJD_scaled = b_lim_df['wm_MJD'] - MJD_scaleconst
        L_rf_scaled = b_lim_df['wm_L_rf']*L_rf_scalefactor # also scaling down the y values

        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # using our chosen 'trusted' band, use the polyfit to calculate the L_rf at the MJD values present in our trusted band's data 
        if extrapolate == True: # if we're extrapolating, then evaluate the polynomial at all of the MJDs of the trusted band, even if it's way outside the region over which we have data for this band
            interp_MJD = ref_band_MJD
            interp_MJD_scaled = ref_band_MJD_scaled

        else: # if we're not extrapolating, only evaluate the band's polyfit at the trusted band's mjd values which are within the mjd region covered by the band's actual data
            interp_MJD = [mjd for mjd in ref_band_MJD if (mjd >= b_lim_df['wm_MJD'].min()) and (mjd <= b_lim_df['wm_MJD'].max() )] ########################################################### HERE IS WHERE WE CHANGE THE INTERPOLATION
            interp_MJD_scaled = np.array(interp_MJD) - MJD_scaleconst

        
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # do the polynomial fit + calculate the reduced chi squared
        poly_coeffs, plot_poly_MJD, plot_poly_L_rf, redchi, redchi_1sig, chi_sig_dist = polyfitting(b_df = b_lim_df, mjd_scale_C = MJD_scaleconst, L_rf_scalefactor = L_rf_scalefactor, max_poly_order = max_poly_order)

        plot_polyline_row[:] = [b, poly_coeffs, plot_poly_MJD, plot_poly_L_rf, redchi, redchi_1sig, chi_sig_dist]
        plot_polyline_df.loc[plot_polyline_rowno] = plot_polyline_row # appending our polyfit info to a dataframe containing info on all of the bands' polynomial fits

        if b == trusted_band: # we don't need ot interpolate the trusted band
            b_colour = b_colour_dict[b]

            plt.errorbar(b_df['wm_MJD'], b_df['wm_L_rf'], yerr = b_df['wm_L_rf_err'], fmt = 'o', markeredgecolor = 'k', markeredgewidth = '1.0', linestyle = 'None', 
                         label = b, c = b_colour)
            plt.plot(plot_poly_MJD, plot_poly_L_rf, c = b_colour, label = f'fit order = {(len(poly_coeffs)-1)} \nred chi = {redchi:.3f}  \n +/- {redchi_1sig:.3f}')
            continue

        # evaluate whether each MJD is worth interpolating, e.g. if it's like 500 days away from all other datapoints, don't interpolate there because the polyfit isn't 
        # well constrained there. We allow better sampled bands to interpolate further out than poorly sampled bands since their fits are better constrained. 
        allow_interp = []
        for int_sc_mjd in interp_MJD_scaled:
            allow_int, MJD_dist = allow_interpolation(interp_x = int_sc_mjd, all_data_x = MJD_scaled, band_lc_density = b_lc_density, local_density_region = 50, interp_cap = 50, factor = 100, simple_cutoff = False, simple_cut = 50)
            allow_interp.append(allow_int)

        #allow_interp = [True]*len(allow_interp) # TESTING SOMETHING GET RID OF THIS IT OVERWRITES THE ALLOW_INTERPOLATION FUNCTION'S RESULT
        # interpolate at the MJDs which we have calculated are good enough
        interp_MJD_scaled = np.array(interp_MJD_scaled)
        interp_MJD = np.array(interp_MJD)
        interp_MJD_scaled = interp_MJD_scaled[allow_interp] # the MJD values which we have deemed good enough to interpolate at
        interp_MJD = interp_MJD[allow_interp]
        interp_sc_L_rf = np.polyval(poly_coeffs, interp_MJD_scaled)
        interp_L_rf = interp_sc_L_rf / L_rf_scalefactor
   
        # calculate the fudged errors
        interp_L_rf_err = fudge_polyfit_L_rf_err(real_b_df = b_df, scaled_polyfit_L_rf = interp_sc_L_rf, scaled_reference_MJDs = interp_MJD_scaled, MJD_scaledown = MJD_scaleconst, L_rf_scaledown = L_rf_scalefactor)


        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # calculate the fudged errors on the polyfit L_rf values
        interp_column_len = len(interp_MJD)
        interp_b_df = pd.DataFrame({'MJD': interp_MJD, 
                                    'L_rf': interp_L_rf, 
                                    'L_rf_err': interp_L_rf_err, 
                                    'band': [b] * interp_column_len, 
                                    'em_cent_wl': [b_em_cent_wl] * interp_column_len 
                                    })

        polyfit_ref_lc_df = pd.concat([polyfit_ref_lc_df, interp_b_df], ignore_index = True)

        
        # PLOTTING ====================================================================================================================================================================
        if plot_polyfit == True:
            b_colour = b_colour_dict[b]

            plt.errorbar(b_df['wm_MJD'], b_df['wm_L_rf'], yerr = b_df['wm_L_rf_err'], fmt = 'o', markeredgecolor = 'k', markeredgewidth = '1.0', linestyle = 'None', 
                         label = b, c = b_colour)
            plt.plot(plot_poly_MJD, plot_poly_L_rf, c = b_colour, label = f'fit order = {(len(poly_coeffs)-1)} \nred chi = {redchi:.3f}  \n +/- {redchi_1sig:.3f}')
            plt.errorbar(interp_b_df['MJD'], interp_b_df['L_rf'], yerr = interp_b_df['L_rf_err'], fmt = '^', c = b_colour, markeredgecolor = 'k', markeredgewidth = '1.0', 
                         linestyle = 'None', alpha = 0.5,  capsize = 5, capthick = 5)
            #plt.fill_between(x = interp_b_df['MJD'], y1 = (interp_b_df['L_rf'] - interp_b_df['L_rf_err']), y2 = (interp_b_df['L_rf'] + interp_b_df['L_rf_err']), color = b_colour, alpha = 0.5)
            
    if plot_polyfit == True:
        savepath = f"C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS/plots/light curves/polyfits/{ant_name}_polyfit.png" #C:\Users\laure\OneDrive\Desktop\YoRiS desktop\YoRiS\plots\light curves\polyfits
        plt.xlabel('MJD')
        plt.ylabel('rest frame luminosity')
        #plt.ylim((-1e41, 5e42))
        plt.title(f'{ant_name} polyfit, reference band = {trusted_band}')
        plt.legend(loc = 'lower right', bbox_to_anchor = (1.15, 0.0), fontsize = 7.5)
        fig.subplots_adjust(right = 0.845, left = 0.07)
        plt.grid()
        plt.savefig(savepath)
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
for idx in range(11):
    ANT_name = transient_names[idx]
    ANT_df = binned_df_list[idx]
    ANT_bands = list_of_bands[idx]
    polyfit_MJD_range = MJDs_for_fit[ANT_name]
    reference_band = 'ZTF_g'
    bands_for_BB = [b for b in ANT_bands if (b != 'WISE_W1') and (b != 'WISE_W2')] # remove the WISE bands from the interpolation since we don't want to use this data for the BB fit anyway

    print(ANT_name, bands_for_BB)
    if ANT_name == 'ZTF20abrbeie':
        print(ANT_df[ANT_df['band'] == 'PS_i'])

    interp_lc, plot_polyfit_df = polyfit_lc4(ANT_name, ANT_df, df_bands = bands_for_BB, trusted_band = reference_band, max_poly_order = 9, min_band_dps = 4, 
                                            fit_MJD_range = polyfit_MJD_range, extrapolate = False, b_colour_dict = band_colour_dict, plot_polyfit = True)
    
    print()










