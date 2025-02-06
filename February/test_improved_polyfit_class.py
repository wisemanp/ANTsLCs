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



def identify_straggler_datapoints(b_df, min_band_datapoints = 5, min_non_straggler_count = 4, straggler_dist = 200):
    """
    We want to target bands which have a few straggling datapoints near the start/end of the light curve and we don't want to include these in our data for polyfitting. Instead, we want to put these datapoints in a bin
    and put them with the closest interpolated datapoint position, provided that it's not too far away. For example, if ZTF_g has one datapoint 400 days after the rest of the light curve, at MJD = 58100 and L_rf = L_Rf_dp, we don't want this
    influencing our polynomial fit. Lets say that our reference band has nearby datapoints at [..., 58079, 58090, 58125, ...] we will assume that the L_rf is unchanged over a period of a maximum of 20 days, so we
    will set an interpolated interp_L_rf point at 58090 = L_rf_dp. Basically, we're assuming that there will be no significant evolution of L_rf over a maximum fo a 20 day period. We should caluclate the error on this value as
    we would with the usual interpolated datapoints, removing the term which calculates the error required to set the local reduced chi squared to 1, so just the mean error of the vlosest 20 datapoints, then a term proportional 
    to the number of days interpolated. 
    
    INPUTS:
    --------------
    b_df: dataframe. A single band's dataframe

    min_band_datapoints: (int) the minimum number of datapoints of a band for us to even bother polyfitting it. 

    straggler_dist: int. The number of days out which the datapoint has to be to be considered a potential straggler. i.e. if it's > straggler_dist away from either side of the data, an
    

    OUTPUTS
    -------------
    stragglers: a dataframe with all the same columns as b_df, where we have taken the rows in b_df which are considered to have 'straggler' data and we have put them into this dataframe. This is the part
    of the light curve that we don't want to fit to, but we will put into bins with the nearest interpolated datapoints (as long as this isn't too far away)

    non_stragglers: a dataframe, similar to stragglers which contains all of the 'non-straggler' rows from b_df. This is the part of the light curve that we want to fit to
    """
    b_count = b_df['wm_MJD'].count()
    colnames = b_df.columns.tolist()

    if b_count < min_band_datapoints: # if there are less than 5 datapoints, consider each point in the band a straggler
        stragglers = b_df.copy()
        non_stragglers = pd.DataFrame(columns = colnames) # an empty dataframe

    
    else: # if we have a decent number of datapoints in the band, then we're just looking for general straggling datapoints at the start or end of the band's light curve that we want to cut out
        if len(b_df['wm_MJD']) > 20:
            check_for_stragglers1 = b_df.iloc[:10].copy()
            check_for_stragglers2 = b_df.iloc[-10:].copy()
            check_for_stragglers_df = pd.concat([check_for_stragglers1, check_for_stragglers2], ignore_index=False) # the first and last 10 datapoints, keep original indicies from b_df

        else:
            check_for_stragglers_df = b_df.copy()

        straggler_indicies = []
        check_for_stragglers_idx = check_for_stragglers_df.index
        for j in range(len(check_for_stragglers_idx)):
            i = check_for_stragglers_idx[j] # the row index in check_for_stragglers
            dp_row = check_for_stragglers_df.loc[i].copy() # the row associated with this index
            mjd = dp_row['wm_MJD']
            directional_MJD_diff = b_df['wm_MJD'] - mjd
            mjd_diff_before = directional_MJD_diff[directional_MJD_diff < 0.0] # don't put <= here or below since it will then take the MJD diff with itself into account. 
            mjd_diff_after = directional_MJD_diff[directional_MJD_diff > 0.0]
            no_dps_50_days = (abs(directional_MJD_diff) <= 50.0 ).sum()

            #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # counting the number of datapoints before and after this mjd
            no_dps_before = len(mjd_diff_before)
            no_dps_after = len(mjd_diff_after)

            #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # calculating the closest mjd difference before and after the datapoint
            closest_MJD_diff_before = abs(max(mjd_diff_before)) if no_dps_before > 0 else None # if this datapoint isn't the very first one. This part is more used to catch out stragglers at the end of the light curve. It will catch the last datapoint
            closest_MJD_diff_after = abs(min(mjd_diff_after)) if no_dps_after > 0 else None # if this datapoint isn't the very last one. This part is more used to catch out stragglers at the start of the light curve. It will catch the first datapoint

            #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # given the straggler criteria, put this datapoint's row in either the stragglers or non-stragglers dataframe
            if (no_dps_before > 0) and (no_dps_after > 0): # if the datapoint is not the first or last datapoint
                if ((closest_MJD_diff_before >= straggler_dist) and (closest_MJD_diff_after >= straggler_dist)) and (no_dps_50_days < 4): 
                    
                    if (no_dps_after < 3): # looking to the end of the light curve
                        if i not in straggler_indicies:
                            straggler_indicies.append(i)
                        b = j
                        for k in range(no_dps_after): # # iterate through the datapoints after the straggler and add them to the stragglers list
                            b = b + 1
                            idx = check_for_stragglers_idx[b]

                            if idx not in straggler_indicies:
                                straggler_indicies.append(idx)


                    elif (no_dps_before < 3): # looking to the start of the light curve
                        if i not in straggler_indicies:
                            straggler_indicies.append(i)

                        b = j
                        for k in range(no_dps_before): # iterate through the datapoints before the straggler and add them to the stragglers list
                            b = b - 1
                            idx = check_for_stragglers_idx[b]

                            if idx not in straggler_indicies:
                                straggler_indicies.append(idx)


            #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            elif no_dps_before == 0: # if it's the very first datapoint
                if closest_MJD_diff_after >= straggler_dist:
                    if i not in straggler_indicies:
                        straggler_indicies.append(i)

            #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            elif no_dps_after == 0: # if it's the very last datapoint
                if closest_MJD_diff_before >= straggler_dist:
                    if i not in straggler_indicies:
                        straggler_indicies.append(i)

        stragglers = b_df.loc[straggler_indicies].copy().reset_index(drop = True)
        non_stragglers = b_df.drop(index = straggler_indicies).reset_index(drop = True)

    #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # if, after all of those checks, we are now left with a non-straggler dataframe of very few datapoints, we may as well not fit the light curve at all
    if len(non_stragglers.index) < min_non_straggler_count:
        stragglers = b_df
        non_stragglers = pd.DataFrame(columns = colnames)      

    return stragglers, non_stragglers














def allow_interpolation(interp_x, all_data_x, b_coverage_quality, local_density_region = 50, interp_cap = 150, gapsize = 100, factor = 100, simple_cutoff = False, simple_cut = 50):
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

    b_coverage_density: float. output in the function check_lightcurve_coverage. A score based on how many datapoints the band has and how well distrbuted this data is. The higher, the better the lightcurve

    local_density_region: The region over which we count the lcoal datapoint density. This would be: (interp_x - local_density_region) <= x <= (interp_x + local_density_region)

    interp_cap: the maximum value out to which we can interpolate. Only the well-sampled bands would reach this cap

    gapsize: float. The distance between 2 consecutive datapoints which is considered to be a 'gap'. Since polyfits vary wildly over gaps, we will not interpolate whatsoever across a gap, so if there is a 
            gap in the data > gapsize, we will not interpolate at all here

    factor: a factor in ther equation. 

    RETURNS
    ------------
    interp_alowed: Bool. If True, then interpolation is allowed at interp_x, if False, interpolation si not allowed at interp_x

    """

    directional_MJD_diff = all_data_x - interp_x # takes the MJD difference between interp_x and the MJDs in the band's real data. Directional because there will be -ve values for datapoints before interp_x and +ve ones for datapoints after
   
    # calculating the closest MJD difference between the closest datapoints before and after interp_x, if there is one (there only wouldn't be one if interp_x was at the very edge of )
    # in the equations below I have allowed directional_MJD_diff >=/<= 0.0 because if directional_MJD_diff == 0.0, then we'd be interpolating at an MJD at which we have a real datapoint, so we're allowed to interpolate at interp_x here, regardlesss 
    # if there's a gap afterwrads because this wouldn't be considered interpolating over a gap, we're interpolating at a point which already had a true datapoint there
    closest_MJD_diff_before = abs(max(directional_MJD_diff[directional_MJD_diff <= 0.0])) # closest MJD diff of datapoints before 

    closest_MJD_diff_after = min(directional_MJD_diff[directional_MJD_diff >= 0.0])


    MJD_diff = abs(directional_MJD_diff) # takes the MJD difference between interp_x and the MJDs in the band's real data
    local_density = (MJD_diff <= local_density_region).sum() # counts the number of datapoints within local_density_region days' of interp_x
    closest_MJD_diff = min(MJD_diff)

    if simple_cutoff == False:
        interp_lim = (b_coverage_quality * local_density * factor) 
        interp_lim = min(interp_cap, interp_lim) # returns the smaller of the two, so this caps the interpolation limit at interp_cap

        # THE MIDDLE SECTIONS GET PREFERENTIAL TREATMENT COMPARED TO THE CENTRAL ONES
        #if ((interp_x - local_density_region) <= min(all_data_x)) or ((interp_x + local_density_region) >= max(all_data_x)): # if interp_x is near thestart/end of the band's lightcurve
        #    interp_lim = interp_lim*2
        

        if (closest_MJD_diff < interp_lim) and (closest_MJD_diff_before < gapsize) and (closest_MJD_diff_after < gapsize): # if interp_x lies within our calculated interpolation limit, then allow interpolation here. Also don't allow interpolation over gaps
            interp_allowed = True
        else: 
            interp_allowed = False
        
    else:
        if (closest_MJD_diff <= simple_cut) and (closest_MJD_diff_before < gapsize) and (closest_MJD_diff_after < gapsize): # if interp_x lies within our calculated interpolation limit, then allow interpolation here. Also don't allow interpolation over gaps
            interp_allowed = True
        else: 
            interp_allowed = False


    return interp_allowed
    




def check_lightcurve_coverage(b_df, mjd_binsize = 50):
    """
    Bins the light curve into mjd bins and counts the number of dapoints in each bin. Can be used as a measure of the light curve data coverage for the band. It's an improvement on just doing 
    (number of datapoints in the band)/(MJD span of the band) because a lot of the data could be densely packed in a particular region and sparsely distributed across the rest, and this would give the 
    impression that the light curve was quite well sampled when it's not really. 

    coverage_term = mean(count of datapoints in mjd_binsize bin) * (no datapoints across the lightcurve) / (1 + std dev(count of datapoints in mjd_binsize bin)) as a calculation of the light curve coverage. 

    INPUTS
    -------------
    b_df: a dataframe containing the band's data. We only actually look at the column 'wm_MJD'

    mjd_binsize: int. The size of the bins within which we would like to count the number of datapoints in. This is to test how well-distributed the light curve's data is


    RETURNS
    ----------
    
    coverage_term = A score based on light curve coverage. The higher, the better. Higher scores will come from lightcurves which have well-distributed data across the light curve and lots of data in general. 

    """

    MJD_bin_min = int( round(b_df['wm_MJD'].min(), -1) - mjd_binsize )
    MJD_bin_max = int( round(b_df['wm_MJD'].max(), -1) + mjd_binsize )
    MJD_bins = range(MJD_bin_min, (MJD_bin_max + mjd_binsize), mjd_binsize) # create the bins

    # binning the data
    # data frame for the binned band data  - just adds a column of MJD_bin to the data, then we can group by all datapoints in the same MJD bin
    b_df['MJD_bin'] = pd.cut(b_df['wm_MJD'], MJD_bins)

    # binning the data by MJD_bin
    b_binned_df = b_df.groupby('MJD_bin', observed = False).apply(lambda g: pd.Series({'count': g['wm_MJD'].count()})).reset_index()
    mean_count = b_binned_df['count'].mean()
    std_count = b_binned_df['count'].std()
    coverage_term = mean_count * len(b_df['wm_MJD']) / (1 + std_count) # = mean * no datapoints / (1 + std)

    return coverage_term





def polyfitting(b_df, band_coverage_quality, mjd_scale_C, L_rf_scalefactor, max_poly_order):
    """
    This function uses chi squred minimisation to optimise the choice of the polynomial order to fit to a band in a light curve, and also uses curve_fit to find
    the optimal parameters for each polynomial fit. Bands with little data are not allowed to use higher order polynomials to fit them

    INPUTS:
    ---------------
    b_df: a dataframe of the (single-band) lightcurve which has been MJD-limited to the region that you want to be polyfitted. 

    band_coverage_quality: (float). A score calculated within teh check_lightcurve_coverage function which si large for light curves with lots fo datapoints
    which are well-distributed across the light curve's mjd span

    mjd_scale_C: float. best as the mean MJD in the band or overall lightcurve. Used to scale-down the MJD values that we're fitting

    L_rf_scalefactor: float. Used to scale-down the rest frame luminosity. Usually = 1e-41.   

    max_poly_order: int between 3 <= max_poly_order <= 14. The maximum order of polynomial that you want to be allowed to fit. 


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
    if band_coverage_quality >= 80.0: # the best covered light curves are allowed to try the max order of polynomial
        poly_orders_available = np.arange(1, (max_poly_order + 1), 1)

    elif (band_coverage_quality >= 50) and (band_coverage_quality < 80):
        poly_orders_available = np.arange(1, (12 + 1), 1)

    elif  (band_coverage_quality >= 30) and (band_coverage_quality < 50):
        poly_orders_available = np.arange(1, (10 + 1), 1)

    elif  (band_coverage_quality >= 10) and (band_coverage_quality < 30):
        poly_orders_available = np.arange(1, (8 + 1), 1)

    #elif  (band_coverage_quality >= 10) and (band_coverage_quality < 20):
    #    poly_orders_available = np.arange(1, (7 + 1), 1)

    elif  (band_coverage_quality >= 6) and (band_coverage_quality < 10):
        poly_orders_available = np.arange(1, (6 + 1), 1)

    elif  (band_coverage_quality >= 2) and (band_coverage_quality < 6):
        poly_orders_available = np.arange(1, (3 + 1), 1)

    elif  (band_coverage_quality >= 1) and (band_coverage_quality < 2):
        poly_orders_available = [1, 2]

    elif  (band_coverage_quality >= 0) and (band_coverage_quality < 1): # the worst covered lightcurves have very restricted access to polynomial orders
        poly_orders_available = [1]

    if b_MJD_span < 50.0:
        poly_orders_available = [1]

    elif b_MJD_span < 100:
        poly_orders_available = [1, 2]
    
    

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
        
    
    poly_sigma_dist = abs(1 - best_redchi)/(best_redchi_1sig)
    plot_poly_sc_MJD = np.arange(min(b_MJD_scaled), max(b_MJD_scaled), 1.0) # for plotting the polynomial fit
    plot_poly_sc_L = np.polyval(optimal_params, plot_poly_sc_MJD)

    plot_poly_MJD = plot_poly_sc_MJD + mjd_scale_C
    plot_poly_L = plot_poly_sc_L/L_rf_scalefactor

    return optimal_params, plot_poly_MJD, plot_poly_L, best_redchi, best_redchi_1sig, poly_sigma_dist





def fudge_interpolation_error_formula(mean_err, mjd_dif, L):
        """
        A fudge formula inspired by superbol which calculates an error for interpolated datapoints. 

        INPUTS
        -----------
        mean_err: (float) the mean error on the nearest X datapoints. If you're utilising straggler data, just input the error on your straggler datapoint

        mjd_dif: (float) the mjd span over which we are interpolating

        L: (float) the rest frame luminosity of the interpolated datapoint. Ideally, this would be scaled down


        RETURNS
        -------------
        er: (float). The fudged error calculated for the interpolated datapoint. If you input L as a scaled value, er will be scaled by the same value as well. 

        """

        fraction = 0.05

        x = abs( L * (mjd_dif / 10) ) 
        #er = np.sqrt(mean_err**2 + (fraction * x)**2)  # this adds some % error for every 10 days interpolated - inspired by Superbol who were inspired by someone else
        er = mean_err + fraction*x
        if er > abs(L):
            er = abs(L)

        return er








def fudge_polyfit_L_rf_err(real_b_df, scaled_polyfit_L_rf, scaled_reference_MJDs, MJD_scaledown, L_rf_scaledown, optimal_params):
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

    optimal_params: a list of the coefficients of from the polyfit in descenidng order, e.g. if we have ax^2 + bx + c, optimal params = [a, b, c] so the coefficient of the highest
                    order term goes first

                    
    OUTPUTS
    ---------------
    poly_L_rf_err_list: a list of our fudged L_rf_err values for our polyfit interpolated data. 

    """
    def invert_redchi(y, y_m, y_m_err, M):
        """
        Calculates an error term based on 
        """
        return


    rb_df = real_b_df.sort_values(by = 'wm_MJD', ascending = True) # making sure that the dataframe is sorted in ascending order of MJD, just in case. The real band's dataframe
    rb_MJDs = np.array(rb_df['wm_MJD'].copy())
    scaled_rb_MJDs = rb_MJDs - MJD_scaledown # scaled down band's real MJD values
    sc_rb_L_rf = np.array(real_b_df['wm_L_rf'].copy()) * L_rf_scaledown # scaled down band's real weighted mean rest frame luminosity values
    rb_L_rf_err = np.array(real_b_df['wm_L_rf_err'].copy()) # an array of the band's real L_rf errors
    scaled_rb_L_rf_err = rb_L_rf_err * L_rf_scaledown # scaled down band's real weighted mean rest frame luminosity error values


    fudge_err_list = [] # a list of fudged rest frame luminosity uncertainties on the L_rf values calculated by polyfit, at the reference MJDs
    if isinstance(scaled_polyfit_L_rf, float): # accounting for the possibilit that scaled_polyfit_L_rf isn't a list of values but a list of one value or a float. This would only really happen when we evaluate the reference band at the straggler datapoint MJDs
        sc_ref_mjd = scaled_reference_MJDs
        sc_L = scaled_polyfit_L_rf
        MJD_diff = [abs(sc_ref_mjd - sc_mjd) for sc_mjd in scaled_rb_MJDs] # take the abs value of the difference between the mjd of the datapoint and the mjd values in the band's actual data
        sort_by_MJD_closeness = np.argsort(MJD_diff) # argsort returns a list of the index arrangement of the elements within the list/array its given that would sort the list/array in ascending order
            
        # calculate the mean of the closest 20 datapoint's errors
        closest_20_idx = sort_by_MJD_closeness[:20] 
        sc_closest_20_err = [scaled_rb_L_rf_err[j] for j in closest_20_idx] #  a list of the wm_L_rf_err values for the 20 closest datapoints to our interpolated datapoint
        sc_mean_L_rf_err = np.mean(np.array(sc_closest_20_err)) # part of the error formula = mean L_rf_err of the 20 closest datapoints in MJD
        closest_MJD_diff = MJD_diff[sort_by_MJD_closeness[0]]

        # use the fudged error formula
        sc_poly_L_rf_er = fudge_interpolation_error_formula(sc_mean_L_rf_err, closest_MJD_diff, sc_L)
        poly_L_rf_er =  sc_poly_L_rf_er / L_rf_scaledown # to scale L_rf (fudged) error
        
        if poly_L_rf_er < 0.0:
            print(sc_L, poly_L_rf_er)
        fudge_err_list.append(poly_L_rf_er)



    else:
        for i, sc_L in enumerate(scaled_polyfit_L_rf): # iterate through each interpolated L_rf value calculated by evaluating the polyfit at the reference band's MJDs
            sc_ref_mjd = scaled_reference_MJDs[i] # scaled MJD value of the reference band
            MJD_diff = [abs(sc_ref_mjd - sc_mjd) for sc_mjd in scaled_rb_MJDs] # take the abs value of the difference between the mjd of the datapoint and the mjd values in the band's actual data
            sort_by_MJD_closeness = np.argsort(MJD_diff) # argsort returns a list of the index arrangement of the elements within the list/array its given that would sort the list/array in ascending order
            
            # calculate the mean of the closest 20 datapoint's errors
            closest_20_idx = sort_by_MJD_closeness[:20] 
            sc_closest_20_err = [scaled_rb_L_rf_err[j] for j in closest_20_idx] #  a list of the wm_L_rf_err values for the 20 closest datapoints to our interpolated datapoint
            sc_mean_L_rf_err = np.mean(np.array(sc_closest_20_err)) # part of the error formula = mean L_rf_err of the 20 closest datapoints in MJD
            closest_MJD_diff = MJD_diff[sort_by_MJD_closeness[0]]

            # calculate the error on the datapoints which would force the reduced chi squared of the nearest 5 datapoints (using the errors for the interpolated datapoints) to 1
            #closest_5_idx = sort_by_MJD_closeness[:5]
            #sc_closest_rb_L = [sc_rb_L_rf[j] for j in closest_5_idx] # the scaled luminosity values of the closest 5 datapoints in the real band's data
            #sc_closest_rb_MJD = np.array([scaled_rb_MJDs[j] for j in closest_5_idx]) # the scaled MJD values of the closest 5 datapoints in the real band's data
            #sc_closest_interp_L = np.polyval(optimal_params, sc_closest_rb_MJD) * L_rf_scaledown
            
            # use the fudged error formula
            sc_poly_L_rf_er = fudge_interpolation_error_formula(sc_mean_L_rf_err, closest_MJD_diff, sc_L)
            poly_L_rf_er =  sc_poly_L_rf_er / L_rf_scaledown # to scale L_rf (fudged) error
            
            if poly_L_rf_er < 0.0:
                print(sc_L, poly_L_rf_er)
            fudge_err_list.append(poly_L_rf_er)

    return fudge_err_list







def choose_reference_band(ANT_name, df_bands, df_band_coverage_qualities, override_choice_dict):
    """
    Chooses the reference band for each band's interpolation. The polynomial fits to each band in the light curve (besides the reference band) will be evaluated at
    the reference band data's MJD values to produce an interpolated light curve which has lots of data at exact MJD values to be used for BB fitting. This function
    chooses the band with the highest band coverage quality score given by check_lightcurve_coverage(), but can override this decision if there is a preferred band
    specified in override_choice_dict. 

    INPUTS
    --------------
    ANT_name: (str) the ANT's name

    df_bands: (list) of the band names present in the light curve which we are fitting and interpolating

    df_band_coverage_qualities: (list) of the band coverage quality scores given by check_lightcurve_coverage()

    override_choice_dict: (distionary). The keys are the ANT names and the values are your choice to override the decision made on the reference band based on maximising the light curve
    coverage quality score. 

    OUTPUTS
    ---------------
    ref_band: (str) the name of the reference band for the ANT
    """

    override = override_choice_dict[ANT_name] 
    if override is not None:
        ref_band = override

    else:
        best_band_score = 0 # start off with a score which the first band is guaranteed to beat
        for i, b in enumerate(df_bands):
            band_score = df_band_coverage_qualities[i]

            if band_score > best_band_score: # every time the band's coverage quality score gets larger, we overwrite ' best_band_score' and 'ref_band'
                best_band_score = band_score
                ref_band = b


    return ref_band








def restrict_dataframe(df, min_value, max_value, column = 'wm_MJD'):
    """
    Applying a limit to a dataframe based on the values in a particular column.

    INPUTS
    ----------------
    df: (dataframe) the ANT's entire dataframe containing all bands

    min_value: (float) the minimum column value that you want to limit the dataframe to. Can be None if you don't want to limit the dataframe to a minimum value

    max_value: (float) the maximum column value  that you want to limit the dataframe to. Can be None if you don't want to limit the dataframe to a maximum value

    column: (str) the column in the dataframe which you want to limit the dataframe by. Default is 'wm_MJD'

    OUTPUT
    ---------------
    lim_df: (DataFrame). The ANT's dataframe which has been limited to the bounds specified in fit_MJD_range
    """

    lim_df = df.copy()
    if min_value != None:
        lim_df = lim_df[lim_df[column] > min_value].copy() 

    if max_value != None:
        lim_df = lim_df[lim_df[column] < max_value].copy()
    
    return lim_df














class polyfit_lightcurve:
    def __init__(self, ant_name, df, bands, override_ref_band, interp_at_ref_band, min_band_dps, straggler_dist, fit_MJD_range, max_interp_distance, max_poly_order, b_colour_dict, plot_polyfit = False, save_interp_df = False):
        self.ant_name = ant_name
        self.df = df
        self.bands = bands
        self.override_ref_band = override_ref_band
        self.interp_at_ref_band = interp_at_ref_band
        self.min_band_dps = min_band_dps
        self.straggler_dist = straggler_dist
        self.fit_MJD_range = fit_MJD_range
        self.max_interp_distance = max_interp_distance
        self.max_poly_order = max_poly_order
        self.b_colour_dict = b_colour_dict
        self.plot_polyfit = plot_polyfit
        self.save_interp_df = save_interp_df

        self.b_df_dict = {b: df[df['band'] == b].copy() for b in bands} # a dictionary of band dataframes, so self.b_df_dict[band] gives the band's dataframe
        self.lim_df = None # will be set later
        self.b_lim_df_dict = None # will be set later
        self.prepping_data = pd.DataFrame(columns = ['b_coverage_score', 'straggler_df', 'non_straggler_df', 'sc_interp_MJD'], index = self.bands)
        self.plot_results = pd.DataFrame(columns = ['poly_coeffs', 'poly_plot_MJD', 'poly_plot_L_rf', 'red_chi', 'red_chi_1sig', 'chi_sigma_dist'], index = self.bands)
        self.interp_df = pd.DataFrame(columns = ['MJD', 'L_rf', 'L_rf_err', 'band', 'em_cent_wl'])


    def get_scalefactors(self):
        self.L_scalefactor = 1e-41
        self.MJD_scaleconst = self.df['wm_MJD'].mean()
        

    
    def initialise_plot(self):
        if self.plot_polyfit == True:
            self.fig = plt.figure(figsize = (16, 7.5))




    # limiting the MJD over which we are polyfitting, because for some ANTs, we have some straggling datapoints far away from the rest of the light curve and we don't want to fit these
    def MJD_limit_df(self):
        if self.lim_df is None:
            self.lim_df = restrict_dataframe(df = self.df, min_value = self.fit_MJD_range[0], max_value = self.fit_MJD_range[1], column = 'wm_MJD')
            self.lim_df['sc_MJD'] = self.lim_df['wm_MJD'] - self.MJD_scaleconst
            self.b_lim_df_dict = {b: self.lim_df[self.lim_df['band'] == b].copy() for b in self.bands} # make a dictionary for the limited dataframes for each band
            self.b_em_cent_wl_dict = {b: self.lim_df.loc[self.lim_df['band'] == b, 'em_cent_wl'].iloc[0]for b in self.bands}



    def identify_stragglers_and_score_band(self):
        self.straggler_MJDs = []
        for b in self.bands:
            straggler_df, non_straggler_df = identify_straggler_datapoints(self.b_lim_df_dict[b], min_band_datapoints = self.min_band_dps, straggler_dist = self.straggler_dist)
            self.prepping_data.at[b, 'straggler_df'] = straggler_df
            self.prepping_data.at[b, 'non_straggler_df'] = non_straggler_df
            
            self.straggler_MJDs.extend(straggler_df['wm_MJD'].values) # extends straggler_MJDs by the values within the straggler_df['wm_MJD'].values array

            if non_straggler_df.empty: # if the band has too few datapoints to bother polyfitting, we know that this won't be the reference band so set it's coverage score to 0
                self.prepping_data.at[b, 'b_coverage_score'] = 0

                if self.plot_polyfit == True: # plot the bands which were too poorly sampled to bother polyfitting (100% stragglers)
                    b_colour = self.b_colour_dict[b]
                    plt.errorbar(self.b_df_dict[b]['wm_MJD'], self.b_df_dict[b]['wm_L_rf'], yerr = self.b_df_dict[b]['wm_L_rf_err'], fmt = 'o', markeredgecolor = 'k', markeredgewidth = '1.0', linestyle = 'None', 
                                label = b, c = b_colour)
                    plt.scatter(straggler_df['wm_MJD'], straggler_df['wm_L_rf'], c = 'k', marker = 'o', s = 70, zorder = 2)
                    plt.scatter(straggler_df['wm_MJD'], straggler_df['wm_L_rf'], c = b_colour, marker = 'x', s = 20, zorder = 3)
                continue
            
            else: # calculate the coverage score for the bands with enough non-straggler datapoints
                self.prepping_data.at[b, 'b_coverage_score'] = check_lightcurve_coverage(non_straggler_df, mjd_binsize = 50)




    def choose_reference_band(self):
        """ called within choose_interp_MJD()
        Chooses the reference band for each band's interpolation. The polynomial fits to each band in the light curve (besides the reference band) will be evaluated at
        the reference band data's MJD values to produce an interpolated light curve which has lots of data at exact MJD values to be used for BB fitting. This function
        chooses the band with the highest band coverage quality score given by check_lightcurve_coverage(), but can override this decision if there is a preferred band
        specified in override_choice_dict. 
        """
        if self.override_ref_band is not None:
            self.ref_band = self.override_ref_band

        else:
            self.ref_band = self.prepping_data['b_coverage_score'].idxmax() # the reference band is the band with the highest band coverage quality score


        
    @staticmethod
    def generate_result_df(MJD, L_rf, L_rf_err, band, em_cent_wl):
        df = pd.DataFrame({'MJD': MJD, 
                           'L_rf': L_rf, 
                           'L_rf_err': L_rf_err, 
                           'band': band, 
                           'em_cent_wl': em_cent_wl})
        return df

    

    def choose_interp_MJD(self):
        if self.interp_at_ref_band == True:
            self.choose_reference_band()
            
            # append the reference band data to the results dataframe, since we aren't interpolating at these values
            ref_lim_df = self.b_lim_df_dict[self.ref_band]
            result_df = self.generate_result_df(ref_lim_df['wm_MJD'], ref_lim_df['wm_L_rf'], ref_lim_df['wm_L_rf_err'], self.ref_band, self.b_em_cent_wl_dict[self.ref_band])
            self.interp_df = pd.concat([self.interp_df, result_df], ignore_index = True)

            interp_MJDs = list(ref_lim_df['wm_MJD']) # the reference MJD values at which we should evaluate all other bands' polyfits
            interp_MJDs.extend(self.straggler_MJDs) # adding the straggler MJD values to interp_MJDs because we want to evaluate the polynomials at the straggler points
            interp_MJDs = np.sort(np.array(interp_MJDs)) # sorting the MJDs from lowest to highest value


        else:
            self.ref_band = None
            interp_MJDs = np.arange(self.lim_df['wm_MJD'].min(), (self.lim_df['wm_MJD'].max() + 2.5) , 5.0) # if we don't want to interpolate at the reference band (+ straggler) MJDs, then interpolate every 5 days


        for b in self.prepping_data.index: # iterate through the indicies of self.prepping_data, which are the band names
            if self.interp_at_ref_band == True:
                if b == self.ref_band:
                    b_interp_MJDs = self.straggler_MJDs # making sure we still evaluate the reference band at the straggler MJD values

                else:
                    b_interp_MJDs = interp_MJDs # if it's not the reference band, then continue as normal

                b_straggler_df = self.prepping_data.at[b, 'straggler_df']
                if b_straggler_df.empty == False:
                    b_straggler_MJDs = b_straggler_df['wm_MJD'].copy().tolist()
                    b_interp_MJDs = [mjd for mjd in b_interp_MJDs if mjd not in b_straggler_MJDs] # remove the band's own straggler MJDs from the interp_MJDs list, since we aren't interpolating the band's straggler datapoints, 
                                                                                                  # we're just going to insert the straggler data into the final interp_df
            else:
                b_interp_MJDs = interp_MJDs
                    

            b_lim_df = self.b_lim_df_dict[b]
            b_non_straggler_df = self.prepping_data.at[b, 'non_straggler_df']

            filtered_interp_MJDs = [mjd for mjd in b_interp_MJDs if (mjd >= b_non_straggler_df['wm_MJD'].min()) and (mjd <= b_non_straggler_df['wm_MJD'].max())]  # make sure interp_MJD doesn't go beyond the bounds of the band's data
            
            # evaluate whether each MJD is worth interpolating, e.g. if it's like 500 days away from all other datapoints, don't interpolate there because the polyfit isn't 
            # well constrained there. We allow better sampled bands to interpolate further out than poorly sampled bands since their fits are better constrained. 
            sc_filtered_interp_MJDs = filtered_interp_MJDs - self.MJD_scaleconst
            allow_interp = []
            for sc_int_mjd in sc_filtered_interp_MJDs:
                allow_int = allow_interpolation(interp_x = sc_int_mjd, all_data_x = b_lim_df['sc_MJD'], b_coverage_quality = self.prepping_data.at[b, 'b_coverage_score'], 
                                                          local_density_region = 50, interp_cap = self.max_interp_distance, gapsize = 100, factor = 1.0, 
                                                          simple_cutoff = False, simple_cut = None)
                allow_interp.append(allow_int)

            sc_filtered_interp_MJDs = np.array(sc_filtered_interp_MJDs)
            sc_filtered_interp_MJDs = sc_filtered_interp_MJDs[allow_interp]

            self.prepping_data.at[b, 'sc_interp_MJD'] = sc_filtered_interp_MJDs
            


    

    def polynomial_fit_and_interp(self):
        for b in self.bands:
            # do the polynomial fit + calculate the reduced chi squared
            non_straggler_df = self.prepping_data.at[b, 'non_straggler_df']

            if non_straggler_df.empty == True: # if our band has no non-straggler data, don't bother polyfitting
                self.plot_results.loc[b] = [None, None, None, None, None, None]
                continue

            poly_coeffs, plot_poly_MJD, plot_poly_L_rf, redchi, redchi_1sig, chi_sig_dist = polyfitting(b_df = non_straggler_df, band_coverage_quality = self.prepping_data.at[b, 'b_coverage_score'], mjd_scale_C = self.MJD_scaleconst, L_rf_scalefactor = self.L_scalefactor, max_poly_order = self.max_poly_order)
            self.plot_results.loc[b] = [poly_coeffs, plot_poly_MJD, plot_poly_L_rf, redchi, redchi_1sig, chi_sig_dist]

            # interpolate using the polynomial fit at the MJD values determined by choose_interp_MJD
            sc_interp_MJD = self.prepping_data.at[b, 'sc_interp_MJD']
            sc_interp_L = np.polyval(poly_coeffs, sc_interp_MJD)
            final_interp_L = sc_interp_L / self.L_scalefactor
            final_interp_MJD = sc_interp_MJD + self.MJD_scaleconst

            # also, add the real (not interpolated) straggler datapoints into the final result interp_df, since we're evaluating the polyfits of each band at the straggler MJDs
            straggler_df = self.prepping_data.at[b, 'straggler_df']
            if straggler_df.empty == False:
                straggler_result_df = self.generate_result_df(MJD = straggler_df['wm_MJD'], L_rf = straggler_df['wm_L_rf'], L_rf_err = straggler_df['wm_L_rf_err'], band = [b]*len(straggler_df), em_cent_wl = [self.b_em_cent_wl_dict[b]]*len(straggler_df))
                self.interp_df = pd.concat([self.interp_df, straggler_result_df], ignore_index = True)

            # calculate the fudged errors 
            interp_L_err = fudge_polyfit_L_rf_err(real_b_df = non_straggler_df, scaled_polyfit_L_rf = sc_interp_L, scaled_reference_MJDs = sc_interp_MJD, MJD_scaledown = self.MJD_scaleconst, L_rf_scaledown = self.L_scalefactor, optimal_params = poly_coeffs)
            if isinstance(final_interp_MJD, np.ndarray): 
                len_result_df = len(final_interp_MJD)
                result_df = self.generate_result_df(MJD = final_interp_MJD, L_rf = final_interp_L, L_rf_err = interp_L_err, band = [b]*len_result_df, em_cent_wl = [self.b_em_cent_wl_dict[b]]*len_result_df)

            elif isinstance(final_interp_MJD, float): # this will only happen when b is the reference band and we have one straggler datapoint that we'd like to evaluate our polyfits at
                result_df = self.generate_result_df(MJD = final_interp_MJD, L_rf = final_interp_L, L_rf_err = interp_L_err, band = [b], em_cent_wl = [self.b_em_cent_wl_dict[b]])
            
            self.interp_df = pd.concat([self.interp_df, result_df], ignore_index = True)




    def plot_polyfit_funciton(self):
        if self.plot_polyfit == True:
            for b in self.bands:
                b_colour = self.b_colour_dict[b]
                b_df = self.b_df_dict[b]
                b_plot_polyfit = self.plot_results.loc[b]
                b_coverage_score = self.prepping_data.at[b, 'b_coverage_score']
                straggler_df = self.prepping_data.at[b, 'straggler_df']
                b_non_straggler_df = self.prepping_data.at[b, 'non_straggler_df']
                b_interp_df = self.interp_df[self.interp_df['band'] == b].copy()
                

                plt.errorbar(b_df['wm_MJD'], b_df['wm_L_rf'], yerr = b_df['wm_L_rf_err'], fmt = 'o', markeredgecolor = 'k', markeredgewidth = '1.0', linestyle = 'None', 
                                label = b, c = b_colour)
                plt.scatter(straggler_df['wm_MJD'], straggler_df['wm_L_rf'], c = 'k', marker = 'o', s = 70, zorder = 3)
                plt.scatter(straggler_df['wm_MJD'], straggler_df['wm_L_rf'], c = b_colour, marker = 'x', s = 20, zorder = 4)
                plt.errorbar(b_interp_df['MJD'], b_interp_df['L_rf'], yerr = b_interp_df['L_rf_err'], fmt = '^', c = b_colour, markeredgecolor = 'k', markeredgewidth = '1.0', 
                                linestyle = 'None', alpha = 0.5,  capsize = 5, capthick = 5, label = f'interp {b}')
                if b_non_straggler_df.empty == False: # plot the polynomial fit if we had enough non-straggler datapoints to fit it
                    plt.plot(b_plot_polyfit['poly_plot_MJD'], b_plot_polyfit['poly_plot_L_rf'], c = b_colour, label = f"b cov quality = {b_coverage_score:.3f} \nfit order = {(len(b_plot_polyfit['poly_coeffs'])-1)} \nred chi = {b_plot_polyfit['red_chi']:.3f}  \n +/- {b_plot_polyfit['red_chi_1sig']:.3f}")

        savepath = f"C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS/plots/light curves/polyfits/{self.ant_name}_polyfit" 
        plt.xlabel('MJD')
        plt.ylabel('rest frame luminosity')
        #plt.ylim((-1e41, 5e42))
        plt.title(f'{self.ant_name} polyfit, reference band = {self.ref_band}. Black circle = "straggler". Black star = "interpolated datraggler datapoint"')
        plt.legend(loc = 'lower right', bbox_to_anchor = (1.275, 0.0), fontsize = 7.5, ncols = 2)
        self.fig.subplots_adjust(top=0.92,
                                bottom=0.11,
                                left=0.055,
                                right=0.785,
                                hspace=0.2,
                                wspace=0.2)
        plt.grid()
        plt.savefig(savepath)
        plt.show()


    def save_interpolated_df(self):
        if self.save_interp_df == True:
            savepath = f"C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS/data/interpolated_lcs/{self.ant_name}_interp_lc.csv"
            self.interp_df.to_csv(savepath, index = False)



    def run_fitting_pipeline(self):
        self.get_scalefactors()
        self.initialise_plot()
        self.MJD_limit_df()
        self.identify_stragglers_and_score_band()
        self.choose_interp_MJD()
        self.polynomial_fit_and_interp()
        self.plot_polyfit_funciton()
        self.save_interpolated_df()

        
        




    
    










# load in the data
lc_df_list, transient_names, list_of_bands = load_ANT_data()


# calculate the rest frame luminosity + emitted central wavelength
add_lc_df_list = ANT_data_L_rf(lc_df_list, transient_names, ANT_redshift_dict, ANT_luminosity_dist_cm_dict, band_ZP_dict, band_obs_centwl_dict)


# bin up the light curve into 1 day MJD bins
MJD_binsize = 1
binned_df_list = bin_lc(add_lc_df_list, MJD_binsize)



max_poly_order = 14
min_band_dps = 4
straggler_dist = 80
max_interp_distance = 20
interp_at_ref_band = True
max_interp_dist = 20
plot_polyfit = True
save_interp_df = True
save_README = True # this doesn't go into the class



# polyfitting light curves
for idx in range(11):
#for idx in [10]:
    ANT_name = transient_names[idx]
    ANT_df = binned_df_list[idx]
    ANT_bands = list_of_bands[idx]
    polyfit_MJD_range = MJDs_for_fit[ANT_name]
    reference_band = 'ZTF_g'
    bands_for_BB = [b for b in ANT_bands if (b != 'WISE_W1') and (b != 'WISE_W2')] # remove the WISE bands from the interpolation since we don't want to use this data for the BB fit anyway

    print(ANT_name)
    

    lightcurve = polyfit_lightcurve(ant_name = ANT_name, 
                                    df = ANT_df, 
                                    bands = bands_for_BB, 
                                    override_ref_band=None, # CHANGE ===============================================================    
                                    interp_at_ref_band = interp_at_ref_band, 
                                    min_band_dps = min_band_dps, 
                                    straggler_dist = straggler_dist,
                                    fit_MJD_range = polyfit_MJD_range, 
                                    max_interp_distance = max_interp_distance, 
                                    max_poly_order = max_poly_order, 
                                    b_colour_dict = band_colour_dict, 
                                    plot_polyfit = plot_polyfit, 
                                    save_interp_df = save_interp_df)
    
    lightcurve.run_fitting_pipeline()

    if (save_README == True) & (idx == 0):
        save_interp_data_folder = f"C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS/data/interpolated_lcs/"
        readme_content = f"Interpolated light curves using the following paramaters: \n max_poly_order = {max_poly_order} \n min_band_dps = {min_band_dps} \n straggler_dist = {straggler_dist} \n max_interp_distance = {max_interp_distance} \n interp_at_ref_band = {interp_at_ref_band} \n max_interp_dist = {max_interp_dist}"
        with open(save_interp_data_folder+"README.txt", "w") as f:
            f.write(readme_content)
   
    
    print()







