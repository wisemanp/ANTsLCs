import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import scipy.optimize as opt
from tqdm import tqdm
from matplotlib import cm
from matplotlib.colors import Normalize
from colorama import Fore, Style
sys.path.append("C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS") # this allows us to import plotting preferences and functions
from plotting_preferences import band_colour_dict, band_ZP_dict, band_obs_centwl_dict, ANT_redshift_dict, ANT_luminosity_dist_cm_dict, MJDs_for_fit
from functions import load_interp_ANT_data, blackbody, chisq






class fit_BB_across_lightcurve:
    def __init__(self, interp_df, curvefit, brute, brute_gridsize, brute_param_sigma = 1, BB_R_min = 1e13, BB_R_max = 1e19, BB_T_min = 1e3, BB_T_max = 1e7):
        """
        INPUTS
        ---------------
        interp_df: (dataframe) the ANT's dataframe containing a light curve which has been interpolated using a polynomial fit to each band. 
            Each ANT had a chosen reference band. At the MJD values present in the reference band's real data, the polyfit for all other bands
            were evaluated (provided that we aren't extrapolating). This means that if there was a band which had data at the min and max 
            MJDs of the flare, there will be interpolated data for this band across the whole flare light curve, whereas if there is a band
            which only has data on the plateau of the light curve, this band will only have interpolated data within this region, at
            the MJD values of the reference band's data. This means that we don't need to bin the light curve in order to take the data
            for the blackbody fit, we can take each MJD present within this dataframe and take whatever band is present at this MJD as
            the data for the BB fit. So, we can fit a BB curve for each MJD within this dataframe, as long as it has >2 bands present. Prior
            to being interpolated, the ANT data should (ideally) be binned into small bins like 1 day, meaning that we will only have 0 or 1 datapoint 
            per per band per MJD value (this is only really relevant for the reference band, though, since the interpolation function ensures either 0 or 1
            value per band per MJD value for the interpolated data, since the polynomials are single-valued for any given MJD value). 
            columns: MJD, L_rf, L_rf_err, band, em_cent_wl

        curvefit: (bool) if True, the BB fit will be tried using scipy's curve_fit. If False, no curve_fit calculation will be tried

        brute: (bool) if True, the BB fit will be tried using the brute force method (manually creating a grid of trial parameter values and minimising the chi squared). If 
            False, no brute force calculation will be tried

        brute_gridsize: (int) the number of trial values of R and T that will be tried in the brute force method. The number of trial values of R and T form a 2D grid of parameters and
        each combination of R and T will be tried in the BB fit.

        brute_param_sigma: (float or int) the number of sigma that the brute force method will use to calculate the error on the BB fit parameters. 

        BB_R_min: (float) the minimum value of the radius parameter that will be tried in the BB fit. 

        BB_R_max: (float) the maximum value of the radius parameter that will be tried in the BB fit.

        BB_T_min: (float) the minimum value of the temperature parameter that will be tried in the BB fit.

        BB_T_max: (float) the maximum value of the temperature parameter that will be tried in the BB fit.
        """
        self.interp_df = interp_df
        self.curvefit = curvefit
        self.brute = brute
        self.brute_gridsize = brute_gridsize
        self.brute_param_sigma = brute_param_sigma
        self.BB_R_min = BB_R_min
        self.BB_R_max = BB_R_max
        self.BB_T_min = BB_T_min
        self.BB_T_max = BB_T_max

        self.brute_dchi = (brute_param_sigma)**2 # the number of sigma that the brute force method will use to calculate the error on the BB fit parameters. look to Christians stats module if confused about this


        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # scale down the radius values to explore - this requires scaling down the rest frame luminosity by (R_scalefactor)**2 because L ~ R^2 
        self.R_scalefactor = 1e-16
        self.L_scalefactor = (self.R_scalefactor)**2
        self.interp_df['L_rf_scaled'] = self.interp_df['L_rf'] * self.L_scalefactor
        self.interp_df['L_rf_err_scaled'] = self.interp_df['L_rf_err'] * self.L_scalefactor
        self.BB_R_min_sc = BB_R_min * self.R_scalefactor # scaling down the bounds for the radius parameter space
        self.BB_R_max_sc = BB_R_max * self.R_scalefactor
        interp_df['em_cent_wl_cm'] = interp_df['em_cent_wl'] * 1e-8 # the blackbody function takes wavelength in centimeters. 1A = 1e-10 m.     1A = 1e-8 cm

        self.mjd_values = self.interp_df['MJD'].unique() 
        self.columns = ['MJD', 'no_bands', 'cf_T_K', 'cf_T_err_K', 'cf_R_cm', 'cf_R_err_cm', 'cf_covariance', 'cf_red_chi', 'cf_chi_sigma_dist', 'red_chi_1sig', 'brute_T_K', 'brute_R_cm', 'brute_red_chi', 'brute_chi_sigma_dist']
        self.BB_fit_results = pd.DataFrame(columns = self.columns, index = self.mjd_values)

        


    
    def BB_curvefit(self, MJD, MJD_df):
        try:
            popt, pcov = opt.curve_fit(blackbody, xdata = MJD_df['em_cent_wl_cm'], ydata = MJD_df['L_rf_scaled'], sigma = MJD_df['L_rf_err_scaled'], absolute_sigma = True, 
                                    bounds = (np.array([self.BB_R_min_sc, self.BB_T_min]), np.array([self.BB_R_max_sc, self.BB_T_max])))
            sc_cf_R, cf_T = popt
            sc_cf_R_err = np.sqrt(pcov[0, 0])
            cf_T_err = np.sqrt(pcov[1, 1])
            cf_R = sc_cf_R / self.R_scalefactor
            cf_R_err = sc_cf_R_err / self.R_scalefactor
            cf_covariance = pcov[1,0]

            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # calculate the reduced chi squared of the curve_fit result
            BB_sc_L_chi = [blackbody(wl_cm, sc_cf_R, cf_T) for wl_cm in MJD_df['em_cent_wl_cm']] # evaluating the BB model from curve_fit at the emitted central wavelengths present in our data to use for chi squared calculation
            cf_red_chi, red_chi_1sig = chisq(y_m = BB_sc_L_chi, y = MJD_df['L_rf_scaled'], yerr = MJD_df['L_rf_err_scaled'], M = 2, reduced_chi = True)
            cf_chi_sigma_dist = abs(1 - cf_red_chi)/red_chi_1sig

            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # add the result to the results row which will be appended to the results dataframe
            self.BB_fit_results.loc[MJD, self.columns[2:10]] = [cf_T, cf_T_err, cf_R, cf_R_err, cf_covariance, cf_red_chi, cf_chi_sigma_dist, red_chi_1sig]  


        except RuntimeError:
            print(f'{Fore.RED} WARNING - Curve fit failed for MJD = {MJD} {Style.RESET_ALL}')
            self.BB_fit_results.loc[MJD, self.columns[2:10]] = np.nan

        




    def BB_brute(self, MJD, MJD_df):
        # creating the values of R and T that we will try
        # the number of R and T values to trial in the grid. The combinations of R and T form a 2D grid, so the number of R and T values that we try give the side lengths of the grid
        sc_R_values = np.logspace(np.log10(self.BB_R_min_sc), np.log10(self.BB_R_max_sc), self.brute_gridsize)
        T_values = np.logspace(np.log10(self.BB_T_min), np.log10(self.BB_T_max), self.brute_gridsize)
        R_values = sc_R_values / self.R_scalefactor # use this to return the grid of parameter values tried

        wavelengths = MJD_df['em_cent_wl_cm'].to_numpy() # the emitted central wavelengths of the bands present at this MJD value
        L_rfs = MJD_df['L_rf_scaled'].to_numpy() # the scaled rest frame luminosities of the bands present at this MJD value
        L_rf_errs = MJD_df['L_rf_err_scaled'].to_numpy() # the scaled rest frame luminosity errors of the bands present at this MJD value

        # create a 3D array of the blackbody luminosities for each combination of R and T. This is done by broadcasting the 1D arrays of wavelengths, R values and T values
        # the 3D array will have dimensions (len(wavelengths), len(R_values), len(T_values)) and will contain the blackbody luminosities for each combination of R and T for each wavelength value
        BB_L_sc = blackbody(wavelengths[:, np.newaxis, np.newaxis], sc_R_values[np.newaxis, :, np.newaxis], T_values[np.newaxis, np.newaxis, :]) # the calculated value of scaled rest frame luminosity using this value of T and scaled R

        # calculate the chi squared of the fit
        chi = np.sum((L_rfs[:, np.newaxis, np.newaxis] - BB_L_sc)**2 / L_rf_errs[:, np.newaxis, np.newaxis]**2, axis = 0) # the chi squared values for each combination of R and T
        min_chi = np.min(chi) # the minimum chi squared value
        row, col = np.where(chi == min_chi) # the row and column indices of the minimum chi squared value

        if (len(row) == 1) & (len(col) == 1): 
            r = row[0]
            c = col[0]
            brute_T = T_values[c] # the parameters which give the minimum chi squared
            brute_R = sc_R_values[r] / self.R_scalefactor
            N_M = len(MJD_df['band']) - 2

            if N_M > 0: # this is for when we try to 'fit' a BB to 2 datapoints, since we have 2 parameters, we can't calculate a reduced chi squared value......
                brute_red_chi = min_chi / N_M
                red_chi_1sig = np.sqrt(2/N_M)
                brute_chi_sigma_dist = abs(1 - brute_red_chi) / red_chi_1sig
            else:
                brute_red_chi = np.nan
                red_chi_1sig = np.nan
                brute_chi_sigma_dist = np.nan

        else:
            print()
            print(f"{Fore.RED} WARNING - MULTIPLE R AND T PARAMETER PAIRS GIVE THIS MIN CHI VALUE. MJD = {MJD_df['MJD'].iloc[0]} \n Ts = {[T_values[r] for r in row]}, Rs = {[sc_R_values[c]/self.R_scalefactor for c in col]}")
            print(f"Chi values = {chi[row, col]} {Style.RESET_ALL}")
            print()

        # calculate the error on the parameters using the brute force method
        row_idx, col_idx = np.nonzero(chi <= (min_chi + self.brute_dchi)) # the row and column indices of the chi squared values which are within the error of the minimum chi squared value
        delchi_T = T_values[col_idx] # the values of T which are within the error of the minimum chi squared value
        delchi_sc_R = sc_R_values[row_idx]  # the values of R which are within the error of the minimum chi squared value

        brute_T_err_upper = np.max(delchi_T) - brute_T # the upper error on the temperature parameter
        brute_T_err_lower = brute_T - np.min(delchi_T) # the lower error on the temperature parameter
        brute_R_err_upper = np.max(delchi_sc_R) / self.R_scalefactor - brute_R # the upper error on the radius parameter
        brute_R_err_lower = brute_R - np.min(delchi_sc_R) / self.R_scalefactor # the lower error on the radius parameter

        #print(f'brute T err upper {brute_T_err_upper:.3e} brute T err lower {brute_T_err_lower:.3e} cf T err {cf_T_err:.3e}')
        #print(f'brute R err upper {brute_R_err_upper:.3e} brute R err lower {brute_R_err_lower:.3e} cf R err {cf_R_err:.3e}')
        #print()
        
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # add the result to the results row which will be appended to the results dataframe
        self.BB_fit_results.loc[MJD, self.columns[9:14]] = [red_chi_1sig, brute_T, brute_R, brute_red_chi, brute_chi_sigma_dist]#, param_grid, chi] # add parameter grid and chi grid

        


    def run_BB_fit(self):
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # iterate through each value of MJD within the dataframe and see if we have enough bands to take a BB fit to it 
        for MJD in tqdm(self.mjd_values, desc = 'Progress BB fitting each MJD value', total = len(self.mjd_values), leave = False):
            MJD_df = self.interp_df[self.interp_df['MJD'] == MJD].copy() # THERE COULD BE FLOATING POINT ERRORS HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            MJD_no_bands = len( MJD_df['band'].unique() ) # the number of bands (and therefore datapoints) we have available at this MJD for the BB fit
            self.BB_fit_results.loc[MJD, :] = np.nan # set all values to nan for now, then overwrite them if we have data for thsi column, so that if (e.g.) brute = False, then the brute columns would contain nan values
            self.BB_fit_results.loc[MJD, self.columns[0:2]] = [MJD, MJD_no_bands] # the first column in the dataframe is MJD, so set the first value in the row as the MJD

            if self.curvefit == True:
                self.BB_curvefit(MJD, MJD_df)

            if self.brute == True:
                self.BB_brute(MJD, MJD_df)

        return self.BB_fit_results
    


    #def plot_individual_BB_fits(self):





    

            





















# load in the interpolated data
interp_df_list, transient_names, list_of_bands = load_interp_ANT_data()




#for idx in range(11):
for idx in [10]:

    ANT_name = transient_names[idx]
    interp_lc= interp_df_list[idx]
    ANT_bands = list_of_bands[idx]
    print()
    print(ANT_name)
    
    #if idx == 10:
    #    interp_lc = interp_lc[~interp_lc['band'].isin(['UVOT_B', 'UVOT_U', 'UVOT_UVM2', 'UVOT_UVW1', 'UVOT_UVW2', 'UVOT_V'])]
    

    BB_curvefit = True
    BB_brute = True
    #BB_fit_results = fit_BB_across_lc(interp_lc, brute = BB_brute, curvefit = BB_curvefit, brute_gridsize = 1000)
    BB_fitting = fit_BB_across_lightcurve(interp_lc, curvefit = BB_curvefit, brute = BB_brute, brute_gridsize = 1000)
    BB_fit_results = BB_fitting.run_BB_fit()
    BB_2dp = BB_fit_results[BB_fit_results['no_bands'] == 2] # the BB fits for the MJDs which only had 2 bands, so we aren't really fitting, more solving for the BB R and T which perfectly pass through the data points
    #print(BB_fit_results)
    print()
    

    # what we want: 
    # L_rf vs MJD: on this plot there will be the actual (binned) data, the polyfit and the interpolated data from the polyfit
    # BB R vs MJD: on this plot we'll have curve_fit R and brute force grid R, perhaps with a colour scale to indicate the sigma distance of the reduced chi squared for the 
    #               BB fit to indicate how much we should trust the value
    # BB T vs MJD: basically the same as BB R vs MJD
    # sigma distance vs MJD?: sigma distance for the curve_fit and brute force results. I feel like this would be good to have alongside the polyfit, because if the polyfit was bad at this MJD,
    #               then the interpolated L_rf of the band might be bad, too, which might mean that the BB fit will struggle to fit to this poorly interpolated datapoint.


    #fig = plt.figure(figsize = (16, 7.3))
    #ax1, ax2, ax3, ax4 = [plt.subplot(2, 2, i) for i in np.arange(1, 5, 1)]
    fig, axs = plt.subplots(2, 2, sharex=True, figsize = (16, 7.2))
    ax1, ax2 = axs[0]
    ax3, ax4 = axs[1]

    # getting the colour scale for plotting the BB T and R vs MJD coloured by chi sigma distance
    colour_cutoff = 5.0
    norm = Normalize(vmin = 0.0, vmax = colour_cutoff)

    


    # top left: the L_rf vs MJD light curve
    for b in ANT_bands: # iterate through all of the bands present in the ANT's light curve
        b_df = interp_lc[interp_lc['band'] == b].copy()
        b_colour = band_colour_dict[b]
        ax1.errorbar(b_df['MJD'], b_df['L_rf'], yerr = b_df['L_rf_err'], fmt = 'o', c = b_colour, 
                    linestyle = 'None', markeredgecolor = 'k', markeredgewidth = '0.5', label = b)
        ax1.set_ylabel('Rest frame luminosity')
        


    BB_fit_results = BB_fit_results.dropna(subset = ['red_chi_1sig'])

    if BB_curvefit == True:
        # separating the BB fit results into high and low chi sigma distance so we can plot the ones wiht low chi sigma distance in a colour map, and the high sigma distance in one colour
        BB_low_chi_dist = BB_fit_results[BB_fit_results['cf_chi_sigma_dist'] <= colour_cutoff]
        BB_high_chi_dist = BB_fit_results[BB_fit_results['cf_chi_sigma_dist'] > colour_cutoff]

        #norm = Normalize(vmin = BB_fit_results['cf_chi_sigma_dist'].min(), vmax = BB_fit_results['cf_chi_sigma_dist'].max())
        # ax2 top right: blackbody radius vs MJD
        ax2.errorbar(BB_fit_results['MJD'], BB_fit_results['cf_R_cm'], yerr = BB_fit_results['cf_R_err_cm'], linestyle = 'None', c = 'k', 
                    fmt = 'o', zorder = 1, label = f'BB fit chi sig dist >{colour_cutoff}')
        ax2.errorbar(BB_2dp['MJD'], BB_2dp['cf_R_cm'], yerr = BB_2dp['cf_R_err_cm'], linestyle = 'None', c = 'k', mfc = 'white',
                    fmt = 'o', label = f'cf no bands = 2..', mec = 'k', mew = 0.5)
        sc = ax2.scatter(BB_low_chi_dist['MJD'], BB_low_chi_dist['cf_R_cm'], cmap = 'jet', c = np.ravel(BB_low_chi_dist['cf_chi_sigma_dist']), 
                    label = 'Curve fit results', marker = 'o', zorder = 2, edgecolors = 'k', linewidths = 0.5)

        cbar_label = r'CF Goodness of BB fit ($\chi_{\nu}$ sig dist)'
        cbar = plt.colorbar(sc, ax = ax2)
        cbar.set_label(label = cbar_label)


        # ax3 bottom left: reduced chi squared sigma distance vs MJD
        ax3.scatter(BB_fit_results['MJD'], BB_fit_results['cf_chi_sigma_dist'], marker = 'o', label = 'Curve fit results', edgecolors = 'k', linewidths = 0.5)

        # ax4 bottom right: blackbody temperature vs MJD
        ax4.errorbar(BB_fit_results['MJD'], BB_fit_results['cf_T_K'], yerr = BB_fit_results['cf_T_err_K'], linestyle = 'None', c = 'k', 
                    fmt = 'o', zorder = 1, label = f'BB fit chi sig dist >{colour_cutoff}')
        ax4.errorbar(BB_2dp['MJD'], BB_2dp['cf_T_K'], yerr = BB_2dp['cf_T_err_K'], linestyle = 'None', c = 'k', mfc = 'white',
                    fmt = 'o', label = f'cf no bands = 2..', mec = 'k', mew = 0.5)
        sc = ax4.scatter(BB_low_chi_dist['MJD'], BB_low_chi_dist['cf_T_K'], cmap = 'jet', c = BB_low_chi_dist['cf_chi_sigma_dist'], 
                    label = 'Curve fit results', marker = 'o', edgecolors = 'k', linewidths = 0.5, zorder = 2)
        
        #plt.colorbar(sc, ax = ax4, label = 'Chi sigma distance')
        cbar_label = r'CF Goodness of BB fit ($\chi_{\nu}$ sig dist)'
        cbar = plt.colorbar(sc, ax = ax4)
        cbar.set_label(label = cbar_label)



        
    if (BB_brute == True):
        # separating the BB fit results into high and low chi sigma distance so we can plot the ones wiht low chi sigma distance in a colour map, and the high sigma distance in one colour
        BB_low_chi_dist = BB_fit_results[BB_fit_results['brute_chi_sigma_dist'] <= colour_cutoff]
        BB_high_chi_dist = BB_fit_results[BB_fit_results['brute_chi_sigma_dist'] > colour_cutoff]

        # ax2 top right: blackbody radius vs MJD
        ax2.scatter(BB_fit_results['MJD'], BB_fit_results['brute_R_cm'], linestyle = 'None', c = 'k', 
                    label = 'brute force gridding results', marker = '^')
        
        ax2.scatter(BB_2dp['MJD'], BB_2dp['brute_R_cm'], linestyle = 'None', c = 'white', 
                    marker = '^', label = f'brute no bands = 2..', edgecolors = 'k', linewidths = 0.5)
        
        sc = ax2.scatter(BB_low_chi_dist['MJD'], BB_low_chi_dist['brute_R_cm'], cmap = 'jet', c = np.ravel(BB_low_chi_dist['brute_chi_sigma_dist']), 
                    label = 'Brute force gridding results', marker = '^', zorder = 3, edgecolors = 'k', linewidths = 0.5)

        cbar_label = r'Brute goodness of BB fit ($\chi_{\nu}$ sig dist)'
        cbar = plt.colorbar(sc, ax = ax2)
        cbar.set_label(label = cbar_label)
        
        # ax3 bottom left: reduced chi squared sigma distance vs MJD
        ax3.scatter(BB_fit_results['MJD'], BB_fit_results['brute_chi_sigma_dist'], marker = '^', label = 'Brute force gridding results', edgecolors = 'k', linewidths = 0.5)

        # ax4 bottom right: blackbody temperature vs MJD
        ax4.scatter(BB_fit_results['MJD'], BB_fit_results['brute_T_K'], linestyle = 'None', c = 'k', 
                    label = 'Brute force gridding results', marker = '^')
        
        ax4.scatter(BB_2dp['MJD'], BB_2dp['brute_T_K'], linestyle = 'None', c = 'white', 
                    marker = '^', label = f'brute no bands = 2..', edgecolors = 'k', linewidths = 0.5)
        
        sc = ax4.scatter(BB_low_chi_dist['MJD'], BB_low_chi_dist['brute_T_K'], cmap = 'jet', c = BB_low_chi_dist['brute_chi_sigma_dist'], 
                    label = 'Brute fit results', marker = '^', edgecolors = 'k', linewidths = 0.5, zorder = 3)
        
        #plt.colorbar(sc, ax = ax4, label = 'Chi sigma distance')
        cbar_label = r'Brute goodness of BB fit ($\chi_{\nu}$ sig dist)'
        cbar = plt.colorbar(sc, ax = ax4)
        cbar.set_label(label = cbar_label)


    for ax in [ax1, ax2, ax3, ax4]:
        ax.grid(True)
        ax.set_xlim(MJDs_for_fit[ANT_name])
        ax.legend(fontsize = 8)

    ax2.set_ylabel('Blackbody radius / cm')
    ax3.set_ylabel('Reduced chi squared sigma distance \n (<=1 = Good fit)')
    ax4.set_ylabel('Blackbody temperature / K')
    fig.suptitle(f"Blackbody fit results across {ANT_name}'s light curve")
    fig.supxlabel('MJD')
    fig.subplots_adjust(top=0.92,
                        bottom=0.085,
                        left=0.055,
                        right=0.97,
                        hspace=0.15,
                        wspace=0.19)
    plt.show()

