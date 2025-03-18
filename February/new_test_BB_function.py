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

pd.options.display.float_format = '{:.4e}'.format # from chatGPT - formats all floats in the dataframe in standard form to 4 decimal places




def double_blackbody(lam, R1, T1, R2, T2):
    """
    A function which returns the sum of two blackbody functions.

    INPUTS
    --------------
    lam: the wavelength in cm

    R1: Blackbody radius in cm - a parameter to fit for

    T1: Blackbody temperature in Kelvin - a parameter to fit for

    R2: Blackbody radius in cm - a parameter to fit for

    T2: Blackbody temperature in Kelvin - a parameter to fit for

    RETURNS
    --------------
    BB1 + BB2: the sum of two blackbody functions. Units: ergs/s/Angstrom
    """
    BB1 = blackbody(lam, R1, T1)
    BB2 = blackbody(lam, R2, T2)

    return BB1 + BB2



def power_law_SED(lam, A, gamma):
    """
    A function which models a power law SED like rest frame luminosity density = A*(wavelength)**gamma

    INPUTS
    --------------
    lam: (float) wavelength. Units of the wavelength should reflect the units of the luminosity density you're trying to model. If it's ergs/s/Angstrom, then input
    lam in Angstrom, if its ergs/s/cm, then input lam in cm.

    A: (float) Amplitude factor of the power law

    gamma: (float) the power of the power law


    OUTPUTS
    --------------
    L_rf: (float) the value of the rest-frame luminosity (density) given by the power law. (I say luminosity(density) because I often just refer to this as rest frame luminosity, 
            but since it's per unit wavelength, its actually a luminoisty density). Units determined by the luminosity you're trying to model. This model doesn't 
            have much physical meaning, so physical units are not explicitly defined for this model, instead they reflect the luminosity density that we're modelling

    """
    L_rf = A*(lam**gamma)

    return L_rf




class fit_SED_across_lightcurve:
    def __init__(self, interp_df, SED_type, curvefit, brute, brute_gridsize, ant_name, brute_delchi = 1, individual_BB_plot = 'None', no_indiv_SED_plots = 12, save_indiv_BB_plot = False,
                 plot_chi_contour = False, no_chi_contours = 3,
                BB_R_min = 1e13, BB_R_max = 1e19, BB_T_min = 1e3, BB_T_max = 1e7,
                DBB_T1_min = 1e2, DBB_T1_max = 1e4, DBB_T2_min = 1e4, DBB_T2_max = 1e7, DBB_R_min = 1e13, DBB_R_max = 1e19, 
                PL_A_min = 1e39, PL_A_max = 1e48, PL_gamma_min = -5.0, PL_gamma_max = 0.0):
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

        SED_type: (str) options: 'single_BB', 'double_BB' or 'power_law. If 'single_BB', the blackbody fit will be a single blackbody fit. If 'double_BB', the blackbody fit will be a double blackbody fit. 
        If 'power_law', the SED fit will be a power law fit like A*(wavelength)**gamma. 

        curvefit: (bool) if True, the BB fit will be tried using scipy's curve_fit. If False, no curve_fit calculation will be tried

        brute: (bool) if True, the BB fit will be tried using the brute force method (manually creating a grid of trial parameter values and minimising the chi squared). If 
            False, no brute force calculation will be tried

        brute_gridsize: (int) the number of trial values of R and T that will be tried in the brute force method. The number of trial values of R and T form a 2D grid of parameters and
        each combination of R and T will be tried in the BB fit.

        ant_name: (str) the ANT's name

        brute_delchi: (float). When calculating the errors on the model parameters when fitting with the brute force method, you can get 'one-at a time' 1 sigma errors on your parameters 
        by searching the 'ellipse' in parameter space which produces (min_red_chi +/- 1), so brute_delchi=1 here. For a 2 parameter model with 1 sigma errors taken 'jointly' as a joint confidence region, 
        then choose brute_delchi = 2.3. 
        
        the number of sigma that the brute force method will use to calculate the error on the BB fit parameters. 

        individual_BB_plot: (str) opitons: 'UVOT', 'whole_lc' or 'None'. If 'UVOT', the BB fits taken at MJDs which have UVOT data will be shown. If 'whole_lc', the BB fits taken
        at MJDs across the light curve will be shown. If 'None', no individual BB fits will be shown. The plot will display a grid of 12 BB fits and their chi squared parameter space. 

        no_indiv_SED_plots: (int) how many individual SEDs you want plotted, if individual_BB_plot != 'None' (if you want the individual SED plot). Options are: 24, 20, 12
        
        save_indiv_BB_plot: (bool) if True, the plot of the individual BB fits will be saved.

        plot_chi_contour: (bool). If True and brute == True, the code will plot some chi squared contour plots in the parameter space. This is helpful to sanity check errors. 

        no_chi_contours: (int). Number of individual plots of chi squared contours for a particular fit to a randomly selected SEDs. This has only been implemented in the brute force power law fitting so far
        
        BB_R_min, BB_R_max, BB_T_min, BB_T_max: (each are floats). The parameter space limits for the single BB SED fits

        DBB_T1_min, DBB_T1_max, DBB_T2_min, DBB_T2_ma, DBB_R_min, DBB_R_max: (each are floats). The parameter space limits for the double BB SED fits

        PL_A_min, PL_A_max, PL_gamma_min, PL_gamma_max: (each are floats). The parameter space limits for the power law SED fitting
        

        """
        self.interp_df = interp_df
        self.SED_type = SED_type
        self.curvefit = curvefit
        self.brute = brute
        self.brute_gridsize = brute_gridsize
        self.ant_name = ant_name
        self.brute_delchi = brute_delchi
        
        self.individual_BB_plot = individual_BB_plot
        self.no_indiv_SED_plots = no_indiv_SED_plots
        self.save_indiv_BB_plot = save_indiv_BB_plot
        self.no_chi_contours = no_chi_contours
        self.plot_chi_contour = plot_chi_contour


        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # if we scale down the radius values to explore - this requires scaling down the rest frame luminosity by (R_scalefactor)**2 because L ~ R^2 
        self.R_scalefactor = 1e-16
        self.L_scalefactor = (self.R_scalefactor)**2
        self.interp_df['L_rf_scaled'] = self.interp_df['L_rf'] * self.L_scalefactor
        self.interp_df['L_rf_err_scaled'] = self.interp_df['L_rf_err'] * self.L_scalefactor
        interp_df['em_cent_wl_cm'] = interp_df['em_cent_wl'] * 1e-8 # the blackbody function takes wavelength in centimeters. 1A = 1e-10 m.     1A = 1e-8 cm

        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # define limits on our SED model parameters and the results dataframe (whole columns are dependent on the SED type)
        self.mjd_values = self.interp_df['MJD'].unique()

        if self.SED_type == 'single_BB':
            self.columns = ['MJD', 'd_since_peak', 'no_bands', 'cf_T_K', 'cf_T_err_K', 'cf_R_cm', 'cf_R_err_cm', 'cf_covariance', 'cf_red_chi', 'cf_chi_sigma_dist', 'red_chi_1sig', 'brute_T_K', 'brute_T_err_K', 'brute_R_cm', 'brute_R_err_cm', 'brute_red_chi', 'brute_chi_sigma_dist']
            self.BB_R_min = BB_R_min
            self.BB_R_max = BB_R_max
            self.BB_T_min = BB_T_min
            self.BB_T_max = BB_T_max

            self.BB_R_min_sc = BB_R_min * self.R_scalefactor # scaling down the bounds for the radius parameter space
            self.BB_R_max_sc = BB_R_max * self.R_scalefactor


        elif self.SED_type == 'double_BB':
            self.columns = ['MJD', 'd_since_peak', 'no_bands', 'cf_T1_K', 'cf_T1_err_K', 'cf_R1_cm', 'cf_R1_err_cm', 'cf_T2_K', 'cf_T2_err_K', 'cf_R2_cm', 'cf_R2_err_cm', 'cf_red_chi', 'cf_chi_sigma_dist', 'red_chi_1sig']
            self.DBB_T1_min = DBB_T1_min
            self.DBB_T1_max = DBB_T1_max
            self.DBB_T2_min = DBB_T2_min
            self.DBB_T2_max = DBB_T2_max
            self.DBB_R_min = DBB_R_min
            self.DBB_R_max = DBB_R_max

            self.DBB_R_min_sc = self.DBB_R_min * self.R_scalefactor # scaling down the bounds for the radius parameter space (we input this into curve_fit as the bound rather than the unscaled one)
            self.DBB_R_max_sc = self.DBB_R_max * self.R_scalefactor


        elif self.SED_type == 'power_law':
            # for the brute force parameter error on A, we'll input the lower and upper errors as a tuple like (lower_err, upper_err), but single-valued for gamma's error
            #                  0          1              2        3         4           5             6             7                8                  9             10          11               12              13              14                    15
            self.columns = ['MJD', 'd_since_peak', 'no_bands', 'cf_A', 'cf_A_err', 'cf_gamma', 'cf_gamma_err', 'cf_red_chi', 'cf_chi_sigma_dist', 'red_chi_1sig', 'brute_A', 'brute_A_err', 'brute_gamma', 'brute_gamma_err', 'brute_red_chi', 'brute_chi_sigma_dist']
            self.PL_A_max = PL_A_max
            self.PL_A_min = PL_A_min
            self.PL_gamma_max = PL_gamma_max
            self.PL_gamma_min = PL_gamma_min

            self.A_scalefactor = self.L_scalefactor # scaling down the bounds for the amplitude parameter space (we input this into curve_fit as the bound rather than the unscaled one)
            self.PL_sc_A_max = self.PL_A_max * self.A_scalefactor
            self.PL_sc_A_min = self.PL_A_min * self.A_scalefactor

            if self.plot_chi_contour == True:
                self.contour_MJDs = np.random.choice(self.mjd_values, self.no_chi_contours)

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
            cf_correlation = pcov[1,0] / (cf_T_err * sc_cf_R_err) # using the scaled R error here since curve_fit is not told about the fact that R has been scaled down

            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # calculate the reduced chi squared of the curve_fit result
            BB_sc_L_chi = [blackbody(wl_cm, sc_cf_R, cf_T) for wl_cm in MJD_df['em_cent_wl_cm']] # evaluating the BB model from curve_fit at the emitted central wavelengths present in our data to use for chi squared calculation
            cf_red_chi, red_chi_1sig = chisq(y_m = BB_sc_L_chi, y = MJD_df['L_rf_scaled'], yerr = MJD_df['L_rf_err_scaled'], M = 2, reduced_chi = True)
            cf_chi_sigma_dist = (cf_red_chi - 1)/red_chi_1sig

            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # add the result to the results row which will be appended to the results dataframe
            self.BB_fit_results.loc[MJD, self.columns[3:11]] = [cf_T, cf_T_err, cf_R, cf_R_err, cf_correlation, cf_red_chi, cf_chi_sigma_dist, red_chi_1sig]  


        except RuntimeError:
            print(f'{Fore.RED} WARNING - Curve fit failed for MJD = {MJD} {Style.RESET_ALL}')
            self.no_failed_curvefits += 1 # counting the number of failed curve fits
            self.BB_fit_results.loc[MJD, self.columns[3:11]] = np.nan

        




    def double_BB_curvefit(self, MJD, MJD_df):
        try:
            popt, pcov = opt.curve_fit(double_blackbody, xdata = MJD_df['em_cent_wl_cm'], ydata = MJD_df['L_rf_scaled'], sigma = MJD_df['L_rf_err_scaled'], absolute_sigma = True, 
                                    bounds = (np.array([self.DBB_R_min_sc, self.DBB_T1_min, self.DBB_R_min_sc, self.DBB_T2_min]), np.array([self.DBB_R_max_sc, self.DBB_T1_max, self.DBB_R_max_sc, self.DBB_T2_max])))
                                    #                     (R1_min,           T1_min,             R2_min,        T2_min)                   (     R1_max,          T1_max,              R2_max,           T2_max)
            
            sc_cf_R1, cf_T1, sc_cf_R2, cf_T2 = popt
            sc_cf_R1_err = np.sqrt(pcov[0, 0])
            cf_T1_err = np.sqrt(pcov[1, 1])
            sc_cf_R2_err = np.sqrt(pcov[2, 2])
            cf_T2_err = np.sqrt(pcov[3, 3])
            cf_R1 = sc_cf_R1 / self.R_scalefactor
            cf_R1_err = sc_cf_R1_err / self.R_scalefactor
            cf_R2 = sc_cf_R2 / self.R_scalefactor
            cf_R2_err = sc_cf_R2_err / self.R_scalefactor
            cf_cov = pcov[1,0]  # BUT THIS IS PROBABLY THE COVARIANCE BETWEEN JUST THE T1 AND R1 PARAMETERS OR SOMETHING, NOT THE COVARIANCE BETWEEN ALL 4 PARAMETERS. 

            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # calculate the reduced chi squared of the curve_fit result
            BB_sc_L_chi = [double_blackbody(wl_cm, sc_cf_R1, cf_T1, sc_cf_R2, cf_T2) for wl_cm in MJD_df['em_cent_wl_cm']] # evaluating the BB model from curve_fit at the emitted central wavelengths present in our data to use for chi squared calculation
            cf_red_chi, red_chi_1sig = chisq(y_m = BB_sc_L_chi, y = MJD_df['L_rf_scaled'], yerr = MJD_df['L_rf_err_scaled'], M = 4, reduced_chi = True)
            cf_chi_sigma_dist = (cf_red_chi - 1)/red_chi_1sig

            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # add the result to the results row which will be appended to the results dataframe
            
            self.BB_fit_results.loc[MJD, self.columns[3:14]] = [cf_T1, cf_T1_err, cf_R1, cf_R1_err, cf_T2, cf_T2_err, cf_R2, cf_R2_err, cf_red_chi, cf_chi_sigma_dist, red_chi_1sig]


        except RuntimeError:
            print(f'{Fore.RED} WARNING - Curve fit failed for MJD = {MJD} {Style.RESET_ALL}')
            self.no_failed_curvefits += 1 # counting the number of failed curve fits
            self.BB_fit_results.loc[MJD, self.columns[3:14]] = np.nan





    def power_law_curvefit(self, MJD, MJD_df):
        try:
            A_scalefactor = self.L_scalefactor # L = A(wavelength)^gamma . If we scale L down by 5, it would scale A down by 5
            popt, pcov = opt.curve_fit(power_law_SED, xdata = MJD_df['em_cent_wl'], ydata = MJD_df['L_rf_scaled'], sigma = MJD_df['L_rf_err_scaled'], absolute_sigma = True, 
                                       bounds = (np.array([self.PL_sc_A_min, self.PL_gamma_min]), np.array([self.PL_sc_A_max, self.PL_gamma_max])))
            cf_A_sc = popt[0]
            cf_A = cf_A_sc/A_scalefactor
            cf_gamma = popt[1]
            cf_A_err_sc = np.sqrt(pcov[0, 0])
            cf_A_err = cf_A_err_sc/A_scalefactor
            cf_gamma_err = np.sqrt(pcov[1, 1])
            cf_correlation = pcov[1, 0] / (cf_A_err_sc * cf_gamma_err) # I feel like we should use A's scaled error here, since curve_fit is not told about the fact that A has been scaled down at all.  

            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # calculate the reduced chi squared of the curve_fit result
            PL_sc_L_chi = [power_law_SED(wl_A, cf_A_sc, cf_gamma) for wl_A in MJD_df['em_cent_wl']] # evaluating the power law SED model from curve_fit at the emitted central wavelengths present in our data to use for chi squared calculation
            cf_red_chi, red_chi_1sig = chisq(y_m = PL_sc_L_chi, y = MJD_df['L_rf_scaled'], yerr = MJD_df['L_rf_err_scaled'], M = 2, reduced_chi = True)
            cf_chi_sigma_dist = (cf_red_chi - 1)/red_chi_1sig

            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # add the result to the results row which will be appended to the results dataframe
            self.BB_fit_results.loc[MJD, self.columns[3:10]] = [cf_A, cf_A_err, cf_gamma, cf_gamma_err, cf_red_chi, cf_chi_sigma_dist, red_chi_1sig]


        except RuntimeError:
            print(f'{Fore.RED} WARNING - Curve fit failed for MJD = {MJD} {Style.RESET_ALL}')
            self.no_failed_curvefits += 1 # counting the number of failed curve fits
            self.BB_fit_results.loc[MJD, self.columns[3:10]] = np.nan




    
    def power_law_brute(self, MJD, MJD_df):
        """
        fits a power law SED to the data for a given MJD. For this brute force gridding, since gamma only really needs to explore values between -5 to 0, 
        I decide to balance computational time and accuracy by increasing the number of A values that we try, and decreasing the number of gamma values
        so that it takes just as long as if we had a square parameter grid but I think we need better resolution in A so we're prioritising it a bit. 
        """
        # HERE I AM EXPLORING (grid_scalefactor)^2 TIMES AS MANY VALUES OF A AS OPPOSED TO GAMMA SO THIS IS A RECTANGULAR PARAMETER GRID NOW, BECAUSE I THINK THE A IS STRUGGING TO BE CONSTRAINED MORE THAN GAMMA
        grid_scalefactor = 2
        A_values = np.logspace(np.log10(self.PL_A_min), np.log10(self.PL_A_max), int(np.round(self.brute_gridsize*grid_scalefactor, -1)))
        sc_A_values = A_values*self.A_scalefactor
        gamma_values = np.linspace(self.PL_gamma_min, self.PL_gamma_max, int(np.round(self.brute_gridsize/grid_scalefactor)))

        wavelengths = MJD_df['em_cent_wl'].to_numpy() # the emitted central wavelengths of the bands present at this MJD value
        L_rfs = MJD_df['L_rf_scaled'].to_numpy() # the scaled rest frame luminosities of the bands present at this MJD value
        L_rf_errs = MJD_df['L_rf_err_scaled'].to_numpy() # the scaled rest frame luminosity errors of the bands present at this MJD value

        PL_L_sc = power_law_SED(wavelengths[:, np.newaxis, np.newaxis], sc_A_values[np.newaxis, :, np.newaxis], gamma_values[np.newaxis, np.newaxis, :]) # the calculated value of scaled rest frame luminosity using this value of T and scaled R
        
        # calculate the chi squared of the fit
        chi = np.sum((L_rfs[:, np.newaxis, np.newaxis] - PL_L_sc)**2 / L_rf_errs[:, np.newaxis, np.newaxis]**2, axis = 0) # the chi squared values for each combination of R and T
        min_chi = np.min(chi) # the minimum chi squared value
        row, col = np.where(chi == min_chi) # the row and column indices of the minimum chi squared value

        if (len(row) == 1) & (len(col) == 1): 
            r = row[0]
            c = col[0]
            brute_gamma = gamma_values[c] # the parameters which give the minimum chi squared
            brute_A = A_values[r] 
            N_M = len(MJD_df['band']) - 2

            # getting the errors on the model parameters
            row_idx, col_idx = np.nonzero(chi <= (min_chi + self.brute_delchi)) # the row and column indices of the chi squared values which are within the error of the minimum chi squared value
            delchi_A = A_values[row_idx] # the values of A and gamma which fall within this delchi 'ellipse'
            delchi_gamma = gamma_values[col_idx]

            A_err_upper = max(delchi_A) - brute_A # getting assymetric errors since our trialed parameter grids were logarithmically spaced, so you wouldn't expect a symmetric error about the model paremeter
            A_err_lower = brute_A - min(delchi_A)
            brute_A_err = (A_err_lower, A_err_upper)

            gamma_err_upper = max(delchi_gamma) - brute_gamma
            gamma_err_lower = brute_gamma - min(delchi_gamma)
            brute_gamma_err = (gamma_err_lower + gamma_err_upper)/2 # take the mean (this assumes that gamma's lower and upper error are quite close in value, if they aren't we should decrease the grid spacing)
            

            if N_M > 0: # this is for when we try to 'fit' a BB to 2 datapoints, since we have 2 parameters, we can't calculate a reduced chi squared value......
                brute_red_chi = min_chi / N_M
                red_chi_1sig = np.sqrt(2/N_M)
                brute_chi_sigma_dist = (brute_red_chi - 1) / red_chi_1sig
            else:
                brute_red_chi = np.nan
                red_chi_1sig = np.nan
                brute_chi_sigma_dist = np.nan
            
            # -------------------------------------------------------------------------------------------------------------------------------------------------
            # plotting a contour plot of chi squareds for randomly chosen when desired
            if self.plot_chi_contour:
                if MJD in self.contour_MJDs:
                    fig, ax = plt.subplots(figsize = (16, 7.2))
                    A_grid, gamma_grid = np.meshgrid(A_values, gamma_values)
                    chi_cutoff = 2.3
                    masked_chi = np.ma.masked_where((chi > (min_chi + chi_cutoff)), chi)
                    sc = ax.pcolormesh(A_grid, gamma_grid, masked_chi.T, cmap = 'jet', zorder = 2)
                    high_chi_mask = np.where((chi > (min_chi + chi_cutoff)), 100, np.nan)
                    ax.pcolormesh(A_grid, gamma_grid, high_chi_mask.T, color = 'k', zorder = 1)

                    fig.colorbar(sc, ax = ax, label = r'$\mathbf{\chi^2}$')
                    ax.errorbar(brute_A, brute_gamma, yerr = brute_gamma_err, xerr = ([A_err_lower], [A_err_upper]), markersize = 15, fmt = '*', mec = 'k', mew = '0.5', color = 'white', zorder = 3)
                    ax.set_xlabel('A', fontweight = 'bold')
                    ax.set_ylabel(f'$\mathbf{{\gamma}}$', fontweight = 'bold')
                    ax.set_xlim(((brute_A/50), (brute_A*50)))
                    ax.set_ylim(((brute_gamma - brute_gamma_err*2), (brute_gamma + brute_gamma_err*2)))
                    ax.set_xscale('log')
                    ax.set_title(f'{self.ant_name},    MJD = {MJD:.1f}, \n'+ fr"$ \mathbf{{ A = {brute_A:.1e}^{{+{A_err_upper:.1e}}}_{{-{A_err_lower:.1e}}} }}$    "+ fr'$\mathbf{{  \gamma = {brute_gamma:.1f} \pm {brute_gamma_err:.1f}  }}$' + r'    $\mathbf{\chi}$ sig dist'+ f' = {brute_chi_sigma_dist:.2e}', fontweight = 'bold') 
                    plt.show() #                                            fr"$ \mathbf{{ A = {brute_A:.1e}^{{+{A_err_upper:.1e}}}_{{-{A_err_lower:.1e}}} }}$"

  
        else:
            print()
            print(f"{Fore.RED} WARNING - MULTIPLE R AND T PARAMETER PAIRS GIVE THIS MIN CHI VALUE. MJD = {MJD_df['MJD'].iloc[0]} \n Ts = {[A_values[r] for r in row]}, Rs = {[gamma_values[c] for c in col]}")
            print(f"Chi values = {chi[row, col]} {Style.RESET_ALL}")
            print()


        self.BB_fit_results.loc[MJD, self.columns[10:16]] = [brute_A, brute_A_err, brute_gamma, brute_gamma_err, brute_red_chi, brute_chi_sigma_dist] 





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
                brute_chi_sigma_dist = (brute_red_chi - 1) / red_chi_1sig
            else:
                brute_red_chi = np.nan
                red_chi_1sig = np.nan
                brute_chi_sigma_dist = np.nan

        else:
            print()
            print(f"{Fore.RED} WARNING - MULTIPLE R AND T PARAMETER PAIRS GIVE THIS MIN CHI VALUE. MJD = {MJD_df['MJD'].iloc[0]} \n Ts = {[T_values[c] for c in col]}, Rs = {[sc_R_values[r]/self.R_scalefactor for r in row]}")
            print(f"Chi values = {chi[row, col]} {Style.RESET_ALL}")
            print()

        # calculate the error on the parameters using the brute force method
        row_idx, col_idx = np.nonzero(chi <= (min_chi + self.brute_delchi)) # the row and column indices of the chi squared values which are within the error of the minimum chi squared value
        delchi_T = T_values[col_idx] # the values of T which are within the error of the minimum chi squared value
        delchi_sc_R = sc_R_values[row_idx]  # the values of R which are within the error of the minimum chi squared value

        brute_T_err_upper = np.max(delchi_T) - brute_T # the upper error on the temperature parameter
        brute_T_err_lower = brute_T - np.min(delchi_T) # the lower error on the temperature parameter
        brute_R_err_upper = np.max(delchi_sc_R) / self.R_scalefactor - brute_R # the upper error on the radius parameter
        brute_R_err_lower = brute_R - np.min(delchi_sc_R) / self.R_scalefactor # the lower error on the radius parameter
        brute_T_err = (brute_T_err_lower, brute_T_err_upper)
        brute_R_err = (brute_R_err_lower, brute_R_err_upper)

        #print(f'brute T err upper {brute_T_err_upper:.3e} brute T err lower {brute_T_err_lower:.3e} cf T err {cf_T_err:.3e}')
        #print(f'brute R err upper {brute_R_err_upper:.3e} brute R err lower {brute_R_err_lower:.3e} cf R err {cf_R_err:.3e}')
        #print()
        
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # add the result to the results row which will be appended to the results dataframe
        self.BB_fit_results.loc[MJD, self.columns[10:17]] = [red_chi_1sig,    brute_T,     brute_T_err,     brute_R,     brute_R_err,     brute_red_chi,    brute_chi_sigma_dist]
        #                                                   'red_chi_1sig', 'brute_T_K', 'brute_T_err_K', 'brute_R_cm', 'brute_R_err_cm', 'brute_red_chi', 'brute_chi_sigma_dist']

    







    def run_BB_fit(self):
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # iterate through each value of MJD within the dataframe and see if we have enough bands to take a BB fit to it 
        single_BB = self.SED_type == 'single_BB'
        double_BB = self.SED_type == 'double_BB'
        power_law = self.SED_type == 'power_law'

        if self.curvefit: # count the number of failed curve_fits
            self.no_failed_curvefits = 0

        for MJD in tqdm(self.mjd_values, desc = 'Progress SED fitting each MJD value', total = len(self.mjd_values), leave = False):
            MJD_df = self.interp_df[self.interp_df['MJD'] == MJD].copy() # THERE COULD BE FLOATING POINT ERRORS HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            MJD_d_since_peak = MJD_df['d_since_peak'].iloc[0]
            MJD_no_bands = len( MJD_df['band'].unique() ) # the number of bands (and therefore datapoints) we have available at this MJD for the BB fit
            self.BB_fit_results.loc[MJD, :] = np.nan # set all values to nan for now, then overwrite them if we have data for thsi column, so that if (e.g.) brute = False, then the brute columns would contain nan values
            self.BB_fit_results.loc[MJD, self.columns[0:3]] = [MJD, MJD_d_since_peak, MJD_no_bands] # the first column in the dataframe is MJD, so set the first value in the row as the MJD
            
            if MJD_no_bands <= 1: # don't try fitting a BB spectrum to a single datapoint, so the BB results in this row will all be nan
                continue
            
            #for a single BB fit
            if single_BB:
                if self.curvefit:
                    self.BB_curvefit(MJD, MJD_df)

                if self.brute:
                    self.BB_brute(MJD, MJD_df)

            # for a double BB fit
            elif double_BB:
                if self.curvefit:
                    self.double_BB_curvefit(MJD, MJD_df)

            elif power_law:
                if self.curvefit:
                    self.power_law_curvefit(MJD, MJD_df)

                if self.brute:
                    self.power_law_brute(MJD, MJD_df)


        # print a message to indicate that the fitting was successful
        if self.curvefit:
            print(f'{Fore.GREEN}SED fitting complete for {self.ant_name} :)  (# curve_fits failed = {self.no_failed_curvefits}) ============================================================================================= {Style.RESET_ALL}')

        return self.BB_fit_results
    





    def get_individual_BB_fit_MJDs(self):
        """
        This function will find the MJD values at which we will plot the individual BB fits. We will plot the BB fits at these MJD values in a grid of (no_MJD) number of plots, 
        with the BB fit and the chi squared parameter space grid

        if you choose self.individual_BB_plot == 'UVOT', and, for example self.no_indiv_plots == 12 and there are < 12 MJDS with UVOT data, the code will plot some non-UVOT SEDs too

        """
        BB_MJDs = self.BB_fit_results[self.BB_fit_results['no_bands'] > 1].index # the MJD values at which we have more than 1 band present, so we can fit a BB to the data
        if self.individual_BB_plot == 'UVOT':
            UVOT_df = self.interp_df[self.interp_df['band'].isin(['UVOT_U', 'UVOT_B', 'UVOT_V', 'UVOT_UVM2', 'UVOT_UVW1', 'UVOT_UVW2'])].copy()
            UVOT_MJDs = UVOT_df['MJD'].unique()
            if UVOT_df.empty == True: # if we have no UVOT data, just get MJDs from the whole light curve
                self.indiv_plot_MJDs = BB_MJDs[::int(len(BB_MJDs)/self.no_indiv_SED_plots)] # plot every 12th MJD value
                self.indiv_plot_MJDs = self.indiv_plot_MJDs[:self.no_indiv_SED_plots] # only plot the first 12 values, in case the line above finds 13
                return
            
            self.indiv_plot_MJDs = UVOT_MJDs[::int(len(UVOT_MJDs)/self.no_indiv_SED_plots)] # plot every 12th MJD value
            self.indiv_plot_MJDs = self.indiv_plot_MJDs[:self.no_indiv_SED_plots] # only plot the first 12 values, in case the line above finds 13
            len_UVOT_mjd_choice = len(self.indiv_plot_MJDs)

            if len_UVOT_mjd_choice < self.no_indiv_SED_plots: # if we want to plot the UVOT data but there are less than 12 instances of UVOT data, add some regular MJDs to the plot too
                no_missing = self.no_indiv_SED_plots - len_UVOT_mjd_choice
                add_MJDs = BB_MJDs[::int(len(BB_MJDs)/no_missing)]
                add_MJDs = add_MJDs[:no_missing]
                self.indiv_plot_MJDs = np.concatenate((self.indiv_plot_MJDs, add_MJDs))


        elif self.individual_BB_plot == 'whole_lc':
            self.indiv_plot_MJDs = BB_MJDs[::int(len(BB_MJDs)/self.no_indiv_SED_plots)] # plot every 12th MJD value
            self.indiv_plot_MJDs = self.indiv_plot_MJDs[:self.no_indiv_SED_plots] # only plot the first 12 values, in case the line above finds 13
            

        else:
            self.indiv_plot_MJDs = None





    @staticmethod
    def get_indiv_SED_plot_rows_cols(no_SEDs):
        """
        (Gets called within plot_individual_BB_fits() and plot_individual_double_BB_fits() )
        This function takes in the number of individual SEDs you want to plot in a single plot and gives the number of rows and columns needed to allow for this. 
        """
        if no_SEDs == 24:
            nrows, ncols = (4, 6)

        elif no_SEDs == 20:
            nrows, ncols = (4, 5)

        elif no_SEDs == 12:
            nrows, ncols = (3, 4)

        return nrows, ncols





    def plot_individual_BB_fits(self, band_colour_dict):
        """
        Make a subplot of many of the individual single BB SEDs fit at particular MJDs.
        """
        nrows, ncols = self.get_indiv_SED_plot_rows_cols(no_SEDs = self.no_indiv_SED_plots) # calculate the number of rows and columns needed given the number of individual SEDs we want to plot

        if self.indiv_plot_MJDs is not None:
            fig, axs = plt.subplots(nrows, ncols, figsize = (16, 7.5), sharex = True)
            axs = axs.flatten()
            legend_dict = {}
            for i, MJD in enumerate(self.indiv_plot_MJDs):
                ax = axs[i]
                MJD_df = self.interp_df[self.interp_df['MJD'] == MJD].copy()
                d_since_peak = MJD_df['d_since_peak'].iloc[0]

                title1 = f'DSP = {d_since_peak:.0f}'+ r'  $\chi_{\nu}$ sig dist = '+f'{self.BB_fit_results.loc[MJD, "brute_chi_sigma_dist"]:.2f}\n'
                title2 = r'$T_{cf} =$'+f"{self.BB_fit_results.loc[MJD, 'cf_T_K']:.1e} +/- {self.BB_fit_results.loc[MJD, 'cf_T_err_K']:.1e} K \n"
                title3 = r'$R_{cf} =$'+f"{self.BB_fit_results.loc[MJD, 'cf_R_cm']:.1e} +/- {self.BB_fit_results.loc[MJD, 'cf_R_err_cm']:.1e} cm"

                plot_wl = np.linspace(1000, 8000, 300)*1e-8 # wavelength range to plot out BB at in cm
                plot_BB_L = blackbody(plot_wl, self.BB_fit_results.loc[MJD, 'brute_R_cm'], self.BB_fit_results.loc[MJD, 'brute_T_K'])
                h_BB, = ax.plot(plot_wl*1e8, plot_BB_L, c = 'k', label = title2 + title3)
                ax.grid(True)
                
                for b in MJD_df['band'].unique():
                    b_df = MJD_df[MJD_df['band'] == b].copy()
                    b_colour = band_colour_dict[b]
                    h = ax.errorbar(b_df['em_cent_wl'], b_df['L_rf'], yerr = b_df['L_rf_err'], fmt = 'o', c = b_colour, label = b)
                    legend_dict[b] = h[0]
                
                ax.legend(handles = [h_BB], labels = [title2 + title3], prop = {'weight': 'bold', 'size': '4.5'})
                ax.set_title(title1, fontsize = 7.5, fontweight = 'bold')
            
            titlefontsize = 18
            fig.supxlabel('Emitted wavelength / $\AA$', fontweight = 'bold', fontsize = (titlefontsize - 5))
            fig.supylabel('Rest frame luminosity / erg s$^{-1}$ $\AA^{-1}$', fontweight = 'bold', fontsize = (titlefontsize - 5))
            titleline1 = f'Brute force blackbody fits at MJD values across {self.ant_name} lightcurve \n'
            titleline2 = f'Parameter limits: (R: {self.BB_R_min:.1e} - {self.BB_R_max:.1e}), (T: {self.BB_T_min:.1e} - {self.BB_T_max:.1e})'
            fig.suptitle(titleline1 + titleline2, fontweight = 'bold', fontsize = titlefontsize)
            fig.legend(legend_dict.values(), legend_dict.keys(), loc = 'upper right', fontsize = 8, bbox_to_anchor = (1.0, 0.95))
            fig.subplots_adjust(top=0.82,
                                bottom=0.094,
                                left=0.06,
                                right=0.92,
                                hspace=0.7,
                                wspace=0.2)
            
            if self.save_indiv_BB_plot == True:
                savepath = f"C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS/plots/BB fits/proper_BB_fits/{self.ant_name}/{self.ant_name}_subplot_indiv_BB_fits.png"
                plt.savefig(savepath, dpi = 300) 





    def plot_individual_double_BB_fits(self, band_colour_dict): 
        """
        Make a subplot of many of the individual double BB SEDs fit at particular MJDs.
        """
        nrows, ncols = self.get_indiv_SED_plot_rows_cols(self.no_indiv_SED_plots) # calculate the number of rows and columns needed given the number of individual SEDs we want to plot

        if self.indiv_plot_MJDs is not None:
            fig, axs = plt.subplots(nrows, ncols, figsize = (16, 7.5), sharex = True)
            axs = axs.flatten()
            legend_dict = {}
            for i, MJD in enumerate(self.indiv_plot_MJDs):
                ax = axs[i]
                MJD_df = self.interp_df[self.interp_df['MJD'] == MJD].copy()
                d_since_peak = MJD_df['d_since_peak'].iloc[0]


                # sort out the titles to present all of the model parameters
                title1 = f'DSP = {d_since_peak:.0f}'+ r'  $\chi_{\nu}$ sig dist = '+f'{self.BB_fit_results.loc[MJD, "cf_chi_sigma_dist"]:.2f}'
                title2 = r'T1 = '+f"{self.BB_fit_results.loc[MJD, 'cf_T1_K']:.1e} +/- {self.BB_fit_results.loc[MJD, 'cf_T1_err_K']:.1e} K"
                title3 = r'R1 = '+f"{self.BB_fit_results.loc[MJD, 'cf_R1_cm']:.1e} +/- {self.BB_fit_results.loc[MJD, 'cf_R1_err_cm']:.1e} cm"
                title4 = r'T2 = '+f"{self.BB_fit_results.loc[MJD, 'cf_T2_K']:.1e} +/- {self.BB_fit_results.loc[MJD, 'cf_T2_err_K']:.1e} K"
                title5 = r'R1 = '+f"{self.BB_fit_results.loc[MJD, 'cf_R2_cm']:.1e} +/- {self.BB_fit_results.loc[MJD, 'cf_R2_err_cm']:.1e} cm"

                plot_wl = np.linspace(1000, 8000, 300)*1e-8 # wavelength range to plot out BB at in cm
                plot_BB_L = double_blackbody(lam = plot_wl, R1 = self.BB_fit_results.loc[MJD, 'cf_R1_cm'], T1 = self.BB_fit_results.loc[MJD, 'cf_T1_K'], R2 = self.BB_fit_results.loc[MJD, 'cf_R2_cm'], T2 = self.BB_fit_results.loc[MJD, 'cf_T2_K'])
                plot_BB1_L = blackbody(lam_cm = plot_wl, R_cm = self.BB_fit_results.loc[MJD, 'cf_R1_cm'], T_K = self.BB_fit_results.loc[MJD, 'cf_T1_K'])
                plot_BB2_L = blackbody(lam_cm = plot_wl, R_cm = self.BB_fit_results.loc[MJD, 'cf_R2_cm'], T_K = self.BB_fit_results.loc[MJD, 'cf_T2_K'])
                plot_wl_A = plot_wl*1e8 # the wavelengths for the plot in Angstrom
                ax.plot(plot_wl_A, plot_BB_L, c = 'k')
                h1, = ax.plot(plot_wl_A, plot_BB1_L, c = 'red', linestyle = '--', alpha = 0.5) # 'h1, =' upakcs the 2D array given by ax.plot(), although there is only one element in this since we're only plotting one line
                h2, = ax.plot(plot_wl_A, plot_BB2_L, c = 'blue', linestyle = '--', alpha = 0.5)
                ax.grid(True)
                
                for b in MJD_df['band'].unique():
                    b_df = MJD_df[MJD_df['band'] == b].copy()
                    b_colour = band_colour_dict[b]
                    h = ax.errorbar(b_df['em_cent_wl'], b_df['L_rf'], yerr = b_df['L_rf_err'], fmt = 'o', c = b_colour, label = b)
                    legend_dict[b] = h[0]

                title = title1 #+ title2 + title3 + title4 + title5
                ax.set_title(title, fontsize = 7.5, fontweight = 'bold')
                ax.legend(handles = [h1, h2], labels = [title2 + '\n'+ title3, title4 + '\n'+ title5], fontsize = 4.5, prop = {'weight': 'bold', 'size': 4.5})
            
            titlefontsize = 18
            fig.supxlabel('Emitted wavelength / $\AA$', fontweight = 'bold', fontsize = (titlefontsize - 5))
            fig.supylabel(r'Rest frame luminosity / erg s$^{-1}$ $\AA^{-1}$', fontweight = 'bold', fontsize = (titlefontsize - 5))
            titleline1 = f'Curve fit double blackbody fits at MJD values across {self.ant_name} lightcurve'
            titleline2 = f'\nParameter limits: (R: {self.DBB_R_min:.1e} - {self.DBB_R_max:.1e}), (T1: {self.DBB_T1_min:.1e} - {self.DBB_T1_max:.1e}), (T2: {self.DBB_T2_min:.1e} - {self.DBB_T2_max:.1e})'
            fig.suptitle(titleline1 + titleline2, fontweight = 'bold', fontsize = titlefontsize)
            fig.legend(legend_dict.values(), legend_dict.keys(), loc = 'upper right', fontsize = 8, bbox_to_anchor = (1.0, 0.95))
            fig.subplots_adjust(top=0.82,
                                bottom=0.094,
                                left=0.065,
                                right=0.92,
                                hspace=0.355,
                                wspace=0.2)
            

            if self.save_indiv_BB_plot == True:
                savepath = f"C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS/plots/BB fits/proper_BB_fits/{self.ant_name}/{self.ant_name}_subplot_indiv_double_BB_fits.png"
                plt.savefig(savepath, dpi = 300) 

            plt.show()





    def plot_individual_power_law_SED_fits(self, band_colour_dict):
        """
        Make a subplot of many of the individual power law SEDs fit at particular MJDs.
        """
        nrows, ncols = self.get_indiv_SED_plot_rows_cols(no_SEDs = self.no_indiv_SED_plots) # calculate the number of rows and columns needed given the number of individual SEDs we want to plot

        if self.indiv_plot_MJDs is not None:
            fig, axs = plt.subplots(nrows, ncols, figsize = (16, 7.5), sharex = True)
            axs = axs.flatten()
            legend_dict = {}
            for i, MJD in enumerate(self.indiv_plot_MJDs):
                ax = axs[i]
                MJD_df = self.interp_df[self.interp_df['MJD'] == MJD].copy()
                d_since_peak = MJD_df['d_since_peak'].iloc[0]

                title1 = f'DSP = {d_since_peak:.0f}'+ r'  $\chi_{\nu}$ sig dist = '+f'{self.BB_fit_results.loc[MJD, "brute_chi_sigma_dist"]:.2f}'
                #title2 = r'$A_{cf} =$'+f"{self.BB_fit_results.loc[MJD, 'cf_A']:.1e} +/- {self.BB_fit_results.loc[MJD, 'cf_A_err']:.1e} \n"
                title2 = fr"$ \mathbf{{ A = {self.BB_fit_results.loc[MJD, 'brute_A']:.1e}^{{+{self.BB_fit_results.loc[MJD, 'brute_A_err'][1]:.1e}}}_{{-{self.BB_fit_results.loc[MJD, 'brute_A_err'][0]:.1e}}} }}$"+'\n'
                title3 = r'$\gamma = $'+f"{self.BB_fit_results.loc[MJD, 'brute_gamma']:.1e} +/- {self.BB_fit_results.loc[MJD, 'brute_gamma_err']:.1e}"

                plot_wl = np.linspace(1000, 8000, 300)*1e-8 # wavelength range to plot out BB at in cm
                plot_wl_A = plot_wl*1e8
                plot_PL_L = power_law_SED(plot_wl_A, self.BB_fit_results.loc[MJD, 'brute_A'], self.BB_fit_results.loc[MJD, 'brute_gamma'])
                h_BB, = ax.plot(plot_wl_A, plot_PL_L, c = 'k', label = title2 + title3)
                ax.grid(True)
                
                for b in MJD_df['band'].unique():
                    b_df = MJD_df[MJD_df['band'] == b].copy()
                    b_colour = band_colour_dict[b]
                    h = ax.errorbar(b_df['em_cent_wl'], b_df['L_rf'], yerr = b_df['L_rf_err'], fmt = 'o', c = b_colour, label = b)
                    legend_dict[b] = h[0]
                
                ax.legend(handles = [h_BB], labels = [title2 + title3], prop = {'weight': 'bold', 'size': '7.5'})
                ax.set_title(title1, fontsize = 7.5, fontweight = 'bold')
            
            titlefontsize = 18
            fig.supxlabel('Emitted wavelength / $\mathbf{\AA}$', fontweight = 'bold', fontsize = (titlefontsize - 5))
            fig.supylabel('Rest frame luminosity / erg s$ ^\mathbf{-1}$ $\mathbf{\AA^{-1}}$', fontweight = 'bold', fontsize = (titlefontsize - 5))
            titleline1 = f"Brute force power law SED fits at MJD values across {self.ant_name}'s lightcurve \n"
            titleline2 = fr'Parameter limits: ($\mathbf{{\Delta \chi = }}${self.brute_delchi}), (A: {self.PL_A_min:.1e} - {self.PL_A_max:.1e}), ($\gamma$: {self.PL_gamma_min:.1e} - {self.PL_gamma_max:.1e})'
            fig.suptitle(titleline1+titleline2, fontweight = 'bold', fontsize = titlefontsize)
            fig.legend(legend_dict.values(), legend_dict.keys(), loc = 'upper right', fontsize = 8, bbox_to_anchor = (1.0, 0.95))
            fig.subplots_adjust(top=0.82,
                                bottom=0.094,
                                left=0.065,
                                right=0.92,
                                hspace=0.355,
                                wspace=0.2)
            
            if self.save_indiv_BB_plot == True:
                savepath = f"C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS/plots/BB fits/proper_BB_fits/{self.ant_name}/{self.ant_name}_subplot_indiv_power_law_fits.png"
                plt.savefig(savepath, dpi = 300) 

            plt.show()


    
        

    def run_SED_fitting_process(self, band_colour_dict):
        SED_fit_results = self.run_BB_fit() # iterate through the MJDs and fit our chosen SED to them
        self.get_individual_BB_fit_MJDs() # if we want to plot the individual SEDs, get the MJDs at which we will plot their SEDs
        
        # plot the individual SEDs:
        if self.SED_type == 'single_BB': # only plots if the individual_fit_MJDs is not None, so this should be find even if you dont want the individual BB fits plot
            self.plot_individual_BB_fits(band_colour_dict)

        elif self.SED_type == 'double_BB':
            self.plot_individual_double_BB_fits(band_colour_dict)

        elif self.SED_type == 'power_law':
            self.plot_individual_power_law_SED_fits(band_colour_dict)

        return SED_fit_results





    

            
















#===============================================================================================================================================
#===============================================================================================================================================
#===============================================================================================================================================
#===============================================================================================================================================
#===============================================================================================================================================
#===============================================================================================================================================
#===============================================================================================================================================
#===============================================================================================================================================
#===============================================================================================================================================





# load in the interpolated data
interp_df_list, transient_names, list_of_bands = load_interp_ANT_data()

SED_plots = 'compare_SEDs' #'usual'




#===============================================================================================================================================
#===============================================================================================================================================
#===============================================================================================================================================
#===============================================================================================================================================
#===============================================================================================================================================
#===============================================================================================================================================
#===============================================================================================================================================
#===============================================================================================================================================
#===============================================================================================================================================


if SED_plots == 'usual':
    for idx in range(11):
    #for idx in [10]:

        ANT_name = transient_names[idx]
        interp_lc= interp_df_list[idx]
        ANT_bands = list_of_bands[idx]
        print()
        print(ANT_name)
        
        #if idx == 10:
        #    interp_lc = interp_lc[~interp_lc['band'].isin(['UVOT_B', 'UVOT_U', 'UVOT_UVM2', 'UVOT_UVW1', 'UVOT_UVW2', 'UVOT_V'])]
        #interp_lc = interp_lc[~interp_lc['band'].isin(['UVOT_B', 'UVOT_U', 'UVOT_UVM2', 'UVOT_UVW1', 'UVOT_UVW2', 'UVOT_V'])]

        # FITTING METHOD
        BB_curvefit = True
        BB_brute = True
        #SED_type = 'single_BB'
        #SED_type = 'double_BB'
        SED_type = 'power_law'
        brute_gridsize = 2000
        brute_delchi = 2.3 # = 2.3 to consider parameters jointly for a 1 sigma error. good if you want to quote their value but if you're going to propagate the errors I think u need to use = 1, which considers them 'one at a time'

        no_indiv_SED_plots = 24 # current options are 24, 20, 12

        plot_chi_contour = False
        no_chi_contours = 3 

        # SAVING PLOTS
        save_BB_plot = False
        save_indiv_BB_plot = True


        BB_fitting = fit_SED_across_lightcurve(interp_lc, SED_type = SED_type, curvefit = BB_curvefit, brute = BB_brute, ant_name = ANT_name, 
                                            brute_delchi = brute_delchi,  brute_gridsize = brute_gridsize, individual_BB_plot = 'whole_lc', 
                                            no_indiv_SED_plots = no_indiv_SED_plots, save_indiv_BB_plot = save_indiv_BB_plot, 
                                            plot_chi_contour = plot_chi_contour, no_chi_contours = no_chi_contours)
        
        #===============================================================================================================================================
        #===============================================================================================================================================
        #===============================================================================================================================================
        #===============================================================================================================================================
        #===============================================================================================================================================
        #===============================================================================================================================================
        #===============================================================================================================================================
        #===============================================================================================================================================
        #===============================================================================================================================================
        
        
        
        if SED_type == 'double_BB':
            BB_fit_results = BB_fitting.run_SED_fitting_process(band_colour_dict=band_colour_dict)

        if SED_type == 'power_law':
            BB_fit_results = BB_fitting.run_SED_fitting_process(band_colour_dict=band_colour_dict)
            pd.options.display.float_format = '{:.4e}'.format # from chatGPT - formats all floats in the dataframe in standard form to 4 decimal places
            #print(BB_fit_results.head(20).iloc[:, 10:16])
            #print(BB_fit_results.head(20).loc[:, ['MJD', 'cf_A', 'cf_A_err', 'cf_chi_sigma_dist', 'brute_A', 'brute_A_err', 'brute_chi_sigma_dist']])
            #print()
            #print(BB_fit_results.head(20).loc[:, ['MJD', 'cf_gamma', 'cf_gamma_err', 'cf_chi_sigma_dist', 'brute_gamma', 'brute_gamma_err', 'brute_chi_sigma_dist']])
            #print()
            #print('max A = ', BB_fit_results['cf_A'].max())
            #print('min A = ',  BB_fit_results['cf_A'].min())
            #print()
            #print(BB_fit_results.tail(20).loc[:, ['MJD', 'cf_A', 'cf_A_err', 'cf_chi_sigma_dist', 'brute_A', 'brute_A_err', 'brute_chi_sigma_dist']])
            #print()
            #print(BB_fit_results.tail(20).loc[:, ['MJD', 'cf_gamma', 'cf_gamma_err', 'cf_chi_sigma_dist', 'brute_gamma', 'brute_gamma_err', 'brute_chi_sigma_dist']])
            #print(BB_fit_results.tail(20).iloc[:, 10:16])
            #print()
        
        
        if SED_type == 'single_BB':
            BB_fit_results = BB_fitting.run_SED_fitting_process(band_colour_dict=band_colour_dict)
            BB_2dp = BB_fit_results[BB_fit_results['no_bands'] == 2] # the BB fits for the MJDs which only had 2 bands, so we aren't really fitting, more solving for the BB R and T which perfectly pass through the data points
            
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
                ax1.errorbar(b_df['d_since_peak'], b_df['L_rf'], yerr = b_df['L_rf_err'], fmt = 'o', c = b_colour, 
                            linestyle = 'None', markeredgecolor = 'k', markeredgewidth = '0.5', label = b)
                ax1.set_ylabel(r'Rest frame luminosity erg s$^{-1}$ $\AA^{-1}$', fontweight = 'bold')
                


            BB_fit_results = BB_fit_results.dropna(subset = ['red_chi_1sig'])

            if BB_curvefit == True:
                # separating the BB fit results into high and low chi sigma distance so we can plot the ones wiht low chi sigma distance in a colour map, and the high sigma distance in one colour
                BB_low_chi_dist = BB_fit_results[BB_fit_results['cf_chi_sigma_dist'] <= colour_cutoff]
                BB_high_chi_dist = BB_fit_results[BB_fit_results['cf_chi_sigma_dist'] > colour_cutoff]

                #norm = Normalize(vmin = BB_fit_results['cf_chi_sigma_dist'].min(), vmax = BB_fit_results['cf_chi_sigma_dist'].max())
                # ax2 top right: blackbody radius vs MJD
                ax2.errorbar(BB_fit_results['d_since_peak'], BB_fit_results['cf_R_cm'], yerr = BB_fit_results['cf_R_err_cm'], linestyle = 'None', c = 'k', 
                            fmt = 'o', zorder = 1, label = f'BB fit chi sig dist >{colour_cutoff}')
                ax2.errorbar(BB_2dp['d_since_peak'], BB_2dp['cf_R_cm'], yerr = BB_2dp['cf_R_err_cm'], linestyle = 'None', c = 'k', mfc = 'white',
                            fmt = 'o', label = f'cf no bands = 2', mec = 'k', mew = 0.5)
                sc = ax2.scatter(BB_low_chi_dist['d_since_peak'], BB_low_chi_dist['cf_R_cm'], cmap = 'jet', c = np.ravel(BB_low_chi_dist['cf_chi_sigma_dist']), 
                            label = 'Curve fit results', marker = 'o', zorder = 2, edgecolors = 'k', linewidths = 0.5)

                cbar_label = r'CF Goodness of BB fit ($\chi_{\nu}$ sig dist)'
                cbar = plt.colorbar(sc, ax = ax2)
                cbar.set_label(label = cbar_label)


                # ax3 bottom left: reduced chi squared sigma distance vs MJD
                ax3.scatter(BB_fit_results['d_since_peak'], BB_fit_results['cf_chi_sigma_dist'], marker = 'o', label = 'Curve fit results', edgecolors = 'k', linewidths = 0.5)

                # ax4 bottom right: blackbody temperature vs MJD
                ax4.errorbar(BB_fit_results['d_since_peak'], BB_fit_results['cf_T_K'], yerr = BB_fit_results['cf_T_err_K'], linestyle = 'None', c = 'k', 
                            fmt = 'o', zorder = 1, label = f'BB fit chi sig dist >{colour_cutoff}')
                ax4.errorbar(BB_2dp['d_since_peak'], BB_2dp['cf_T_K'], yerr = BB_2dp['cf_T_err_K'], linestyle = 'None', c = 'k', mfc = 'white',
                            fmt = 'o', label = f'cf no bands = 2', mec = 'k', mew = 0.5)
                sc = ax4.scatter(BB_low_chi_dist['d_since_peak'], BB_low_chi_dist['cf_T_K'], cmap = 'jet', c = BB_low_chi_dist['cf_chi_sigma_dist'], 
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
                ax2.scatter(BB_fit_results['d_since_peak'], BB_fit_results['brute_R_cm'], linestyle = 'None', c = 'k', 
                            label = 'brute force gridding results', marker = '^')
                
                ax2.scatter(BB_2dp['d_since_peak'], BB_2dp['brute_R_cm'], linestyle = 'None', c = 'white', 
                            marker = '^', label = f'brute no bands = 2', edgecolors = 'k', linewidths = 0.5)
                
                sc = ax2.scatter(BB_low_chi_dist['d_since_peak'], BB_low_chi_dist['brute_R_cm'], cmap = 'jet', c = np.ravel(BB_low_chi_dist['brute_chi_sigma_dist']), 
                            label = 'Brute force gridding results', marker = '^', zorder = 3, edgecolors = 'k', linewidths = 0.5)

                cbar_label = r'Brute goodness of BB fit ($\chi_{\nu}$ sig dist)'
                cbar = plt.colorbar(sc, ax = ax2)
                cbar.set_label(label = cbar_label)
                
                # ax3 bottom left: reduced chi squared sigma distance vs MJD
                ax3.scatter(BB_fit_results['d_since_peak'], BB_fit_results['brute_chi_sigma_dist'], marker = '^', label = 'Brute force gridding results', edgecolors = 'k', linewidths = 0.5)

                # ax4 bottom right: blackbody temperature vs MJD
                ax4.scatter(BB_fit_results['d_since_peak'], BB_fit_results['brute_T_K'], linestyle = 'None', c = 'k', 
                            label = 'Brute force gridding results', marker = '^')
                
                ax4.scatter(BB_2dp['d_since_peak'], BB_2dp['brute_T_K'], linestyle = 'None', c = 'white', 
                            marker = '^', label = f'brute no bands = 2', edgecolors = 'k', linewidths = 0.5)
                
                sc = ax4.scatter(BB_low_chi_dist['d_since_peak'], BB_low_chi_dist['brute_T_K'], cmap = 'jet', c = BB_low_chi_dist['brute_chi_sigma_dist'], 
                            label = 'Brute fit results', marker = '^', edgecolors = 'k', linewidths = 0.5, zorder = 3)
                
                #plt.colorbar(sc, ax = ax4, label = 'Chi sigma distance')
                cbar_label = r'Brute goodness of BB fit ($\chi_{\nu}$ sig dist)'
                cbar = plt.colorbar(sc, ax = ax4)
                cbar.set_label(label = cbar_label)


            for ax in [ax1, ax2, ax3, ax4]:
                ax.grid(True)
                #ax.set_xlim(MJDs_for_fit[ANT_name])
                ax.legend(fontsize = 8)

            if ANT_name == 'ZTF22aadesap':
                ax4.set_ylim(0.0, 4e4)

            elif ANT_name == 'ZTF19aailpwl':
                ax4.set_ylim(0.0, 2e4)

            elif ANT_name == 'ZTF19aamrjar':
                ax4.set_ylim(0.0, 2.5e4)

            elif ANT_name == 'ZTF19aatubsj':
                ax4.set_ylim(0.0, 2.5e4)

            elif ANT_name == 'ZTF20abgxlut':
                ax2.set_ylim(0.0, 8e15)
                ax4.set_ylim(0.0, 2.3e4)

            elif ANT_name == 'ZTF20abodaps':
                ax2.set_ylim(0.0, 1.2e16)
                ax4.set_ylim(0.0, 4e4)

            elif ANT_name == 'ZTF20abrbeie':
                ax4.set_ylim(0.0, 2e4)

            elif ANT_name == 'ZTF21abxowzx':
                ax4.set_ylim(0.0, 2.5e4)

            elif ANT_name == 'ZTF22aadesap':
                ax2.set_ylim(0.0, 5e15)
                ax4.set_ylim(0.0, 2.5e4)

            ax2.set_ylabel('Blackbody radius / cm', fontweight = 'bold')
            ax3.set_ylabel('Reduced chi squared sigma distance \n (<=2-3 = Good fit)', fontweight = 'bold')
            ax4.set_ylabel('Blackbody temperature / K', fontweight = 'bold')
            fig.suptitle(f"Blackbody fit results across {ANT_name}'s light curve", fontweight = 'bold')
            fig.supxlabel('days since peak (rest frame time)', fontweight = 'bold')
            fig.subplots_adjust(top=0.92,
                                bottom=0.085,
                                left=0.055,
                                right=0.97,
                                hspace=0.15,
                                wspace=0.19)
            
            if save_BB_plot == True:
                savepath = f"C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS/plots/BB fits/proper_BB_fits/{ANT_name}/{ANT_name}_lc_BB_fit.png"
                plt.savefig(savepath, dpi = 300) 
            plt.show()






#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
# TESTING WHICH SED BEST FITS THE ANT

if SED_plots == 'compare_SEDs':
    for idx in range(11):
    #for idx in [10]:
        ANT_name = transient_names[idx]
        interp_lc= interp_df_list[idx]
        ANT_bands = list_of_bands[idx]
        print()
        print(ANT_name)

        # FITTING METHOD
        brute_gridsize = 2000
        brute_delchi = 2.3 # = 2.3 to consider parameters jointly for a 1 sigma error. good if you want to quote their value but if you're going to propagate the errors I think u need to use = 1, which considers them 'one at a time'
        individual_BB_plot = 'None'
        no_indiv_SED_plots = 24 # current options are 24, 20, 12
        plot_chi_contour = False
        no_chi_contours = 3 
        # SAVING PLOTS
        save_BB_plot = False
        save_indiv_BB_plot = False

        fig, axs = plt.subplots(2, 2, figsize = (16.5, 7.5))
        ax1, ax2, ax3, ax4 = axs.ravel()

        for i, SED_type in enumerate(['single_BB', 'double_BB', 'power_law']):
            if SED_type == 'single_BB':
                SED_label = 'Single-BB'
                BB_curvefit = False
                BB_brute = True
                sig_dist_colname = 'brute_chi_sigma_dist'
                SED_colour = '#02f5dd'
                SED_ecolour = '#0f3d7d'
                SED_linestyle = '-'
                SED_hatch = '\\'
                M = 2

                
            elif SED_type == 'double_BB':
                SED_label = 'Double-BB'
                BB_curvefit = True
                BB_brute = False
                sig_dist_colname = 'cf_chi_sigma_dist'
                SED_colour = 'k'
                SED_ecolour = 'k'
                SED_linestyle = '-'
                SED_hatch = '/'
                M = 4


            elif SED_type == 'power_law':
                SED_label = 'Power-law'
                BB_curvefit = False
                BB_brute = True
                sig_dist_colname = 'brute_chi_sigma_dist'
                SED_colour = '#f502dd'
                SED_ecolour = '#780d61'
                SED_linestyle = '-'
                M = 2

        
            BB_fitting = fit_SED_across_lightcurve(interp_lc, SED_type = SED_type, curvefit = BB_curvefit, brute = BB_brute, ant_name = ANT_name, 
                                                brute_delchi = brute_delchi,  brute_gridsize = brute_gridsize, individual_BB_plot = individual_BB_plot, 
                                                no_indiv_SED_plots = no_indiv_SED_plots, save_indiv_BB_plot = save_indiv_BB_plot, 
                                                plot_chi_contour = plot_chi_contour, no_chi_contours = no_chi_contours)
            

            def median_absolute_deviation(median, data): # need to maybe handle the possibility that data is empty and median is None?
                if len(data) == 0:
                    return np.nan
                MAD = np.sum(abs(median - data)) / len(data)
                return MAD

            # the whole lc SED fits
            BB_fit_results = BB_fitting.run_SED_fitting_process(band_colour_dict=band_colour_dict)
            BB_fit_results = BB_fit_results[BB_fit_results[sig_dist_colname].notna()]
            all_lc_median = BB_fit_results[sig_dist_colname].median(skipna = True)
            all_lc_MAD = median_absolute_deviation(all_lc_median, BB_fit_results[sig_dist_colname].to_numpy())

            # SED fits where N > M
            fit_results_N_greater_M = BB_fit_results[BB_fit_results['no_bands'] > M].copy()
            N_greater_M_median = fit_results_N_greater_M[sig_dist_colname].median(skipna = True)
            N_greater_M_MAD = median_absolute_deviation(N_greater_M_median, fit_results_N_greater_M[sig_dist_colname].to_numpy())

            # SED fits with UVOT data
            MJDs_with_UVOT = interp_lc[interp_lc['band'].isin(['UVOT_B', 'UVOT_U', 'UVOT_UVM2', 'UVOT_UVW1', 'UVOT_UVW2', 'UVOT_V'])]['MJD']
            SEDs_with_UVOT = BB_fit_results[BB_fit_results['MJD'].isin(MJDs_with_UVOT)].copy()
            UVOT_median = SEDs_with_UVOT[sig_dist_colname].median()
            UVOT_MAD = median_absolute_deviation(UVOT_median, SEDs_with_UVOT[sig_dist_colname].to_numpy())

            # SED fits on rise/near peak
            SEDs_rise_peak = BB_fit_results[BB_fit_results['d_since_peak'] <= 100].copy()
            rise_peak_median = SEDs_rise_peak[sig_dist_colname].median(skipna = True)
            rise_peak_MAD = median_absolute_deviation(rise_peak_median, SEDs_rise_peak[sig_dist_colname].to_numpy())


            # PLOTTING IT UP IN HISTOGRAMS
            label1 = fr'$\mathbf{{{SED_label}}}$ '+'\n'+fr'Median $D_{{\sigma}}$ = {all_lc_median:.3g}'+'\n'+fr'MAD $D_{{\sigma}}$ = {all_lc_MAD:.3g}'
            ax1.hist(BB_fit_results[sig_dist_colname], bins = 50, color = SED_colour, label = label1, alpha = 0.5)
            ax1.hist(BB_fit_results[sig_dist_colname], bins = 50, edgecolor = SED_ecolour, color = SED_colour, alpha = 0.75, histtype='step', linewidth = 1.5, linestyle = SED_linestyle)#, hatch = SED_hatch)

            label2 = fr'$\mathbf{{{SED_label}}}$ '+'\n'+fr'Median $D_{{\sigma}}$ = {N_greater_M_median:.3g} '+'\n'+fr'MAD $D_{{\sigma}}$ = {N_greater_M_MAD:.3g}'
            ax2.hist(fit_results_N_greater_M[sig_dist_colname], bins = 50, color = SED_colour, label = label2, alpha = 0.5)
            ax2.hist(fit_results_N_greater_M[sig_dist_colname], bins = 50, edgecolor = SED_ecolour, color = SED_colour, alpha = 0.75, histtype='step', linewidth = 1.5, linestyle = SED_linestyle)#, hatch = SED_hatch)

            label3 = fr'$\mathbf{{{SED_label}}}$ '+'\n'+fr'Median $D_{{\sigma}}$ = {rise_peak_median:.3g} '+'\n'+fr'MAD $D_{{\sigma}}$ = {rise_peak_MAD:.3g}'
            ax3.hist(SEDs_rise_peak[sig_dist_colname], bins = 50, color = SED_colour, label = label3, alpha = 0.5)
            ax3.hist(SEDs_rise_peak[sig_dist_colname], bins = 50, edgecolor = SED_ecolour, color = SED_colour,  alpha = 0.75, histtype='step', linewidth = 1.5, linestyle = SED_linestyle)#, hatch = SED_hatch)

            label4 = fr'$\mathbf{{{SED_label}}}$ '+'\n'+fr'Median $D_{{\sigma}}$ = {UVOT_median:.3g} '+'\n'+fr'MAD $D_{{\sigma}}$ = {UVOT_MAD:.3g}'
            ax4.hist(SEDs_with_UVOT[sig_dist_colname], bins = 50, color = SED_colour, label = label4, alpha = 0.5)
            ax4.hist(SEDs_with_UVOT[sig_dist_colname], bins = 50, edgecolor = SED_ecolour, color = SED_colour, alpha = 0.75, histtype='step', linewidth = 1.5, linestyle = SED_linestyle)#, hatch = SED_hatch)

        ax1.set_title('SED fit results entire lightcurve', fontweight = 'bold')
        ax2.set_title('SED fit results where N > M', fontweight = 'bold')
        ax3.set_title('SED fit results for days since peak < 100', fontweight = 'bold')
        ax4.set_title('SED fit results for MJDs with UVOT data', fontweight = 'bold')

        for ax in [ax1, ax2, ax3, ax4]:
            ax.legend()
            ax.grid(True)

        titlefontsize = 20
        fig.supxlabel(r'(sigma distance) = $\mathbf{ D_{\sigma =} \frac{\chi_{\nu} - 1} {\sigma_{\chi_{\nu}}}  }$', fontsize = (titlefontsize - 5), fontweight = 'bold')
        fig.supylabel('Frequency density', fontweight = 'bold', fontsize = (titlefontsize - 5))
        fig.suptitle(f"{ANT_name}'s SED fit results comparing a single blackbody, double blackbody and \n power law SED (MAD = median absolute deviation)", fontweight = 'bold', fontsize = titlefontsize)
        fig.subplots_adjust(top=0.82,
                            bottom=0.11,
                            left=0.08,
                            right=0.955,
                            hspace=0.2,
                            wspace=0.2)
        savepath = f"C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS/plots/BB fits/proper_BB_fits/{ANT_name}/{ANT_name}_compare_SED_fits.png"
        plt.savefig(savepath, dpi = 300)
        #plt.show()











        

        

            




