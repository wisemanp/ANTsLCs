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
    lam: (float) wavelength in cm

    A: (float) Amplitude factor of the power law

    gamma: (float) the power of the power law


    OUTPUTS
    --------------
    L_rf: (float) the value of the rest-frame luminosity (density) given by the power law. (I say luminosity(density) because I often just refer to this as rest frame luminosity, 
            but since it's per unit Angstrom, its actually a luminoisty density) I think the units are kind of arbitrary?

    """
    L_rf = A*(lam**gamma)

    return L_rf




class fit_BB_across_lightcurve:
    def __init__(self, interp_df, SED_type, curvefit, brute, brute_gridsize, ant_name, brute_param_sigma = 1, individual_BB_plot = 'None', no_indiv_SED_plots = 12, save_indiv_BB_plot = False,
                BB_R_min = 1e13, BB_R_max = 1e19, BB_T_min = 1e3, BB_T_max = 1e7,
                DBB_T1_min = 1e2, DBB_T1_max = 1e4, DBB_T2_min = 1e4, DBB_T2_max = 1e7, DBB_R_min = 1e13, DBB_R_max = 1e19):
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

        brute_param_sigma: (float or int) the number of sigma that the brute force method will use to calculate the error on the BB fit parameters. 

        individual_BB_plot: (str) opitons: 'UVOT', 'whole_lc' or 'None'. If 'UVOT', the BB fits taken at MJDs which have UVOT data will be shown. If 'whole_lc', the BB fits taken
        at MJDs across the light curve will be shown. If 'None', no individual BB fits will be shown. The plot will display a grid of 12 BB fits and their chi squared parameter space. 

        no_indiv_SED_plots: (int) how many individual SEDs you want plotted, if individual_BB_plot != 'None' (if you want the individual SED plot). Options are: 24, 20, 12
        
        save_indiv_BB_plot: (bool) if True, the plot of the individual BB fits will be saved.

        BB_R_min, BB_R_max, BB_T_min, BB_T_max: (each are floats). The parameter space limits for the single BB SED fits

        DBB_T1_min, DBB_T1_max, DBB_T2_min, DBB_T2_ma, DBB_R_min, DBB_R_max: (each are floats). The parameter space limits for the double BB SED fits
        

        """
        self.interp_df = interp_df
        self.SED_type = SED_type
        self.curvefit = curvefit
        self.brute = brute
        self.brute_gridsize = brute_gridsize
        self.ant_name = ant_name
        self.brute_param_sigma = brute_param_sigma
        
        self.individual_BB_plot = individual_BB_plot
        self.no_indiv_SED_plots = no_indiv_SED_plots
        self.save_indiv_BB_plot = save_indiv_BB_plot

        self.brute_dchi = (brute_param_sigma)**2 # the number of sigma that the brute force method will use to calculate the error on the BB fit parameters. look to Christians stats module if confused about this


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
            self.columns = ['MJD', 'd_since_peak', 'no_bands', 'cf_T_K', 'cf_T_err_K', 'cf_R_cm', 'cf_R_err_cm', 'cf_covariance', 'cf_red_chi', 'cf_chi_sigma_dist', 'red_chi_1sig', 'brute_T_K', 'brute_R_cm', 'brute_red_chi', 'brute_chi_sigma_dist']
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
            self.columns = ['MJD', 'd_since_peak', 'no_bands', 'cf_A', 'cf_A_err', 'cf_gamma', 'cf_gamma_err', 'cf_red_chi', 'cf_chi_sigma_dist', 'red_chi_1sig']#, 'brute_A', 'brute_A_err', 'brute_gamma', 'brute_gamma_err', 'brute_chi', 'brute_red_chi']

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
            self.BB_fit_results.loc[MJD, self.columns[3:11]] = [cf_T, cf_T_err, cf_R, cf_R_err, cf_covariance, cf_red_chi, cf_chi_sigma_dist, red_chi_1sig]  


        except RuntimeError:
            print(f'{Fore.RED} WARNING - Curve fit failed for MJD = {MJD} {Style.RESET_ALL}')
            self.no_failed_curvefits += 1 # counting the number of failed curve fits
            self.BB_fit_results.loc[MJD, self.columns[3:11]] = np.nan

        




    def double_BB_curvefit(self, MJD, MJD_df, T1_min = 1e2, T1_max = 1e4, T2_min = 1e4, T2_max = 1e7):
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
            cf_covariance = pcov[1,0] # BUT THIS IS PROBABLY THE COVARIANCE BETWEEN JUST THE T1 AND R1 PARAMETERS, NOT THE COVARIANCE BETWEEN ALL 4 PARAMETERS

            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # calculate the reduced chi squared of the curve_fit result
            BB_sc_L_chi = [double_blackbody(wl_cm, sc_cf_R1, cf_T1, sc_cf_R2, cf_T2) for wl_cm in MJD_df['em_cent_wl_cm']] # evaluating the BB model from curve_fit at the emitted central wavelengths present in our data to use for chi squared calculation
            cf_red_chi, red_chi_1sig = chisq(y_m = BB_sc_L_chi, y = MJD_df['L_rf_scaled'], yerr = MJD_df['L_rf_err_scaled'], M = 4, reduced_chi = True)
            cf_chi_sigma_dist = abs(1 - cf_red_chi)/red_chi_1sig

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
            popt, pcov = opt.curve_fit(power_law_SED, xdata = MJD_df['em_cent_wl'], ydata = MJD_df['L_rf_scaled'], sigma = MJD_df['L_rf_err_scaled'], absolute_sigma = True)
            cf_A_sc = popt[0]
            cf_A = cf_A_sc/A_scalefactor
            cf_gamma = popt[1]
            cf_A_err_sc = np.sqrt(pcov[0, 0])
            cf_A_err = cf_A_err_sc/A_scalefactor
            cf_gamma_err = np.sqrt(pcov[1, 1])
            cf_cov = pcov[1, 0] # IDK IF THIS IS COVARIANCE OR CORRELATION - I WANT THE CORRELATION?

            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # calculate the reduced chi squared of the curve_fit result
            PL_sc_L_chi = [power_law_SED(wl_A, cf_A_sc, cf_gamma) for wl_A in MJD_df['em_cent_wl']] # evaluating the power law SED model from curve_fit at the emitted central wavelengths present in our data to use for chi squared calculation
            cf_red_chi, red_chi_1sig = chisq(y_m = PL_sc_L_chi, y = MJD_df['L_rf_scaled'], yerr = MJD_df['L_rf_err_scaled'], M = 2, reduced_chi = True)
            cf_chi_sigma_dist = abs(1 - cf_red_chi)/red_chi_1sig

            # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # add the result to the results row which will be appended to the results dataframe
            self.BB_fit_results.loc[MJD, self.columns[3:11]] = [cf_A, cf_A_err, cf_gamma, cf_gamma_err, cf_red_chi, cf_chi_sigma_dist, red_chi_1sig]


        except RuntimeError:
            print(f'{Fore.RED} WARNING - Curve fit failed for MJD = {MJD} {Style.RESET_ALL}')
            self.no_failed_curvefits += 1 # counting the number of failed curve fits
            self.BB_fit_results.loc[MJD, self.columns[3:11]] = np.nan




    
    def power_law_brute(self, MJD, MJD_df):
        A_scalefactor = self.L_scalefactor
        A_values = np.logspace(35, 45, 1000, self.brute_gridsize)
        sc_A_values = A_values*A_scalefactor
        gamma_values = np.linspace(0.0, 10.0, self.brute_gridsize)

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
            brute_A = sc_A_values[r] / self.A_scalefactor
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
            print(f"{Fore.RED} WARNING - MULTIPLE R AND T PARAMETER PAIRS GIVE THIS MIN CHI VALUE. MJD = {MJD_df['MJD'].iloc[0]} \n Ts = {[A_values[r] for r in row]}, Rs = {[gamma_values[c] for c in col]}")
            print(f"Chi values = {chi[row, col]} {Style.RESET_ALL}")
            print()


        self.BB_fit_results[3:11] = [] # UNFINISHED



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
            print(f"{Fore.RED} WARNING - MULTIPLE R AND T PARAMETER PAIRS GIVE THIS MIN CHI VALUE. MJD = {MJD_df['MJD'].iloc[0]} \n Ts = {[T_values[c] for c in col]}, Rs = {[sc_R_values[r]/self.R_scalefactor for r in row]}")
            print(f"Chi values = {chi[row, col]} {Style.RESET_ALL}")
            print()

        # calculate the error on the parameters using the brute force method
        #row_idx, col_idx = np.nonzero(chi <= (min_chi + self.brute_dchi)) # the row and column indices of the chi squared values which are within the error of the minimum chi squared value
        #delchi_T = T_values[col_idx] # the values of T which are within the error of the minimum chi squared value
        #delchi_sc_R = sc_R_values[row_idx]  # the values of R which are within the error of the minimum chi squared value

        #brute_T_err_upper = np.max(delchi_T) - brute_T # the upper error on the temperature parameter
        #brute_T_err_lower = brute_T - np.min(delchi_T) # the lower error on the temperature parameter
        #brute_R_err_upper = np.max(delchi_sc_R) / self.R_scalefactor - brute_R # the upper error on the radius parameter
        #brute_R_err_lower = brute_R - np.min(delchi_sc_R) / self.R_scalefactor # the lower error on the radius parameter

        #print(f'brute T err upper {brute_T_err_upper:.3e} brute T err lower {brute_T_err_lower:.3e} cf T err {cf_T_err:.3e}')
        #print(f'brute R err upper {brute_R_err_upper:.3e} brute R err lower {brute_R_err_lower:.3e} cf R err {cf_R_err:.3e}')
        #print()
        
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # add the result to the results row which will be appended to the results dataframe
        self.BB_fit_results.loc[MJD, self.columns[10:15]] = [red_chi_1sig, brute_T, brute_R, brute_red_chi, brute_chi_sigma_dist]#, param_grid, chi] # add parameter grid and chi grid

    







    def run_BB_fit(self):
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # iterate through each value of MJD within the dataframe and see if we have enough bands to take a BB fit to it 
        single_BB = self.SED_type == 'single_BB'
        double_BB = self.SED_type == 'double_BB'
        power_law = self.SED_type == 'power_law'

        if self.curvefit: # count the number of failed curve_fits
            self.no_failed_curvefits = 0

        for MJD in tqdm(self.mjd_values, desc = 'Progress BB fitting each MJD value', total = len(self.mjd_values), leave = False):
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
                savepath = f"C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS/plots/BB fits/proper_BB_fits/{self.ant_name}_subplot_indiv_BB_fits.png"
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
                savepath = f"C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS/plots/BB fits/proper_BB_fits/{self.ant_name}_subplot_indiv_double_BB_fits.png"
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

                title1 = f'DSP = {d_since_peak:.0f}'+ r'  $\chi_{\nu}$ sig dist = '+f'{self.BB_fit_results.loc[MJD, "cf_chi_sigma_dist"]:.2f}'
                title2 = r'$A_{cf} =$'+f"{self.BB_fit_results.loc[MJD, 'cf_A']:.1e} +/- {self.BB_fit_results.loc[MJD, 'cf_A_err']:.1e} \n"
                title3 = r'$\gamma_{cf} =$'+f"{self.BB_fit_results.loc[MJD, 'cf_gamma']:.1e} +/- {self.BB_fit_results.loc[MJD, 'cf_gamma_err']:.1e}"

                plot_wl = np.linspace(1000, 8000, 300)*1e-8 # wavelength range to plot out BB at in cm
                plot_wl_A = plot_wl*1e8
                plot_PL_L = power_law_SED(plot_wl_A, self.BB_fit_results.loc[MJD, 'cf_A'], self.BB_fit_results.loc[MJD, 'cf_gamma'])
                h_BB, = ax.plot(plot_wl_A, plot_PL_L, c = 'k', label = title2 + title3)
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
            titleline1 = f"Curve_fit power law SED fits at MJD values across {self.ant_name}'s lightcurve \n"
            #titleline2 = f'Parameter limits: (R: {self.BB_R_min:.1e} - {self.BB_R_max:.1e}), (T: {self.BB_T_min:.1e} - {self.BB_T_max:.1e})'
            fig.suptitle(titleline1, fontweight = 'bold', fontsize = titlefontsize)
            fig.legend(legend_dict.values(), legend_dict.keys(), loc = 'upper right', fontsize = 8, bbox_to_anchor = (1.0, 0.95))
            fig.subplots_adjust(top=0.88,
                                bottom=0.094,
                                left=0.065,
                                right=0.92,
                                hspace=0.355,
                                wspace=0.2)
            
            if self.save_indiv_BB_plot == True:
                savepath = f"C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS/plots/BB fits/proper_BB_fits/{self.ant_name}_subplot_indiv_power_law_fits.png"
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





    

            





















# load in the interpolated data
interp_df_list, transient_names, list_of_bands = load_interp_ANT_data()




#
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

    BB_curvefit = True
    BB_brute = True
    #SED_type = 'single_BB'
    #SED_type = 'double_BB'
    SED_type = 'power_law'
    save_BB_plot = False
    save_indiv_BB_plot = True
    no_indiv_SED_plots = 24 # current options are 24, 20, 12

    BB_fitting = fit_BB_across_lightcurve(interp_lc, SED_type = SED_type, curvefit = BB_curvefit, brute = BB_brute, ant_name = ANT_name, brute_gridsize = 1000, individual_BB_plot = 'whole_lc', no_indiv_SED_plots = no_indiv_SED_plots, save_indiv_BB_plot = save_indiv_BB_plot)
    
    
    
    
    #print(BB_fit_results)
    print()
    if SED_type == 'double_BB':
        BB_fit_results = BB_fitting.run_SED_fitting_process(band_colour_dict=band_colour_dict)

    if SED_type == 'power_law':
        BB_fit_results = BB_fitting.run_SED_fitting_process(band_colour_dict=band_colour_dict)
    
    
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
            savepath = f"C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS/plots/BB fits/proper_BB_fits/{ANT_name}_lc_BB_fit.png"
            plt.savefig(savepath, dpi = 300) 
        plt.show()

