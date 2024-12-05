import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import sys
import scipy.optimize as opt
sys.path.append("C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS") # this allows us to import plotting preferences and functions
from plotting_preferences import band_colour_dict, band_ZP_dict, band_obs_centwl_dict, ANT_redshift_dict, ANT_luminosity_dist_cm_dict, MJDs_for_fit
from functions import load_ANT_data, ANT_data_L_rf, bin_lc, polyfit_lc, blackbody, chisq


def fit_BB(interp_df, brute, curvefit):
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
    BB_T_max = 1e6


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
    columns = ['MJD', 'no_bands', 'cf_T_K', 'cf_T_err_K', 'cf_R_cm', 'cf_R_err_cm', 'cf_red_chi', 'red_chi_1sig', 'brute_T_K', 'brute_R_cm', 'brute_red_chi']
    BB_fit_results = pd.DataFrame(columns = columns)
    for MJD in mjd_values:
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


            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # add the result to the results row which will be appended to the results dataframe
            BB_result_row[2:8] = [cf_T, cf_T_err, cf_R, cf_R_err, cf_red_chi, red_chi_1sig]


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
            else:
                print()
                print(f"WARNING - MULTIPLE R AND T PARAMETER PAIRS GIVE THIS MIN CHI VALUE. MJD = {MJD_df['MJD'].iloc[0]} \n Ts = {[T_values[r] for r in row]}, Rs = {[sc_R_values[c]/R_scalefactor for c in col]}")
                print()

            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # add the result to the results row which will be appended to the results dataframe
            BB_result_row[7:11] = [red_chi_1sig, brute_T, brute_R, brute_red_chi]

        BB_fit_results.loc[df_row_index] = BB_result_row # adding the array of results from this MJD to the BB results dataframe


    return BB_fit_results








# load in the data
lc_df_list, transient_names, list_of_bands = load_ANT_data()

# calculate the rest frame luminosity + emitted central wavelength
add_lc_df_list = ANT_data_L_rf(lc_df_list, transient_names, ANT_redshift_dict, ANT_luminosity_dist_cm_dict, band_ZP_dict, band_obs_centwl_dict)

# bin up the light curve into 1 day MJD bins
MJD_binsize = 1
binned_df_list = bin_lc(add_lc_df_list, MJD_binsize)

# polyfitting ONE light curve
ANT_name = transient_names[0]
ANT_df = binned_df_list[0]
ANT_bands = list_of_bands[0]
bands_for_BB = [b for b in ANT_bands if (b != 'WISE_W1') and (b != 'WISE_W2')] # remove the WISE bands from the interpolation since we don't want to use this data for the BB fit anyway
interp_lc = polyfit_lc(ANT_name, ANT_df, fit_order = 5, df_bands = bands_for_BB, trusted_band = 'ZTF_g', fit_MJD_range = MJDs_for_fit[ANT_name],
                        extrapolate = False, b_colour_dict = band_colour_dict, plot_polyfit = True)



BB_fit_results = fit_BB(interp_lc, brute = True, curvefit = True)
print()
print()
print()
print()
print(interp_lc['MJD'].unique())
print()
print(BB_fit_results.head(50))
print()
print()
print()
print()