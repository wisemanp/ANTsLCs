import skysurvey
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from skysurvey.tools import blackbody
from skysurvey.tools import power_law
from skysurvey.tools import double_blackbody
from skysurvey.tools.utils import random_radec
import pandas as pd
from astropy.cosmology import FlatLambdaCDM # THIS IS FOR THE LUMINOSITY DISTANCE DICTIONARY
import astropy.units as u
from colorama import Fore, Style
from astropy import constants as const
from datetime import datetime
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as ticker
from normal_code.plotting_preferences import ANT_sim_redshift_upper_lim_dict, ANT_proper_redshift_dict, band_colour_dict, band_marker_dict, band_obs_centwl_dict








##################################################################################################################################################################

# ASSUMPTIONS FOR LUMINOSITY DISTANCE ARE BELOW ----------------------------------------------
H0 = 70 #km/s/Mpc
om_M = 0.3 # non relativistic matter density fraction
fcdm = FlatLambdaCDM(H0 = H0, Om0 = om_M)
# ASSUMPTIONS FOR LUMINOSITY DISTANCE ARE ABOVE ----------------------------------------------

ANT_d_cm_list = []
for z in ANT_proper_redshift_dict.values():
    d = fcdm.luminosity_distance(z).to(u.cm).value # this gives the luminosity distance in cm
    ANT_d_cm_list.append(d)

ANT_names = list(ANT_proper_redshift_dict.keys())

ANT_luminosity_dist_cm_dict = dict(zip(ANT_names, ANT_d_cm_list))




##################################################################################################################################################################
##################################################################################################################################################################







def blackbody_L_lam_rf(lam_cm, R_cm, T_K):
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




def blackbody_F_lam(lam_cm, T_K, R_cm, D_cm):
    """
    Planck's blackbody formula modified to give flux density in ergs/ (s cm^2 Angstrom)

    INPUTS
    --------------
    lam: the wavelength in cm

    T_K: Blackbody temperature in Kelvin - a parameter to fit for

    R_cm: Blackbody radius in cm - a parameter to fit for

    D_cm: The distance to the object in cm (calculated from the redshift)

    RETURNS
    --------------
    B_lam: blackbody flux density for the wavelength input. Units:  ergs/ (s cm^2 Angstrom)
    """

    c_cgs = const.c.cgs.value
    h_cgs = const.h.cgs.value
    k_cgs = const.k_B.cgs.value

    C = 2 * np.pi * h_cgs * (c_cgs**2) * 1e-8 # the constant coefficient of the equation - the 10^-8 is what converts this from ergs/ (s cm^2 cm) to ergs/ (s cm^2 Angstrom)

    denom = np.exp((h_cgs * c_cgs) / (lam_cm * k_cgs * T_K)) - 1

    F_lam = C * ((R_cm / D_cm)**2) * (1/ (lam_cm**5)) * (1 / (denom)) # ergs/s/Angstrom

    return F_lam





def get_wein_lbdamax(temperature):
    """ 
    TOOK DIRECTLY FROM SKYSURVEY'S BLACKBODY SECTION - I need to use it before initialising the class so I just copied it here so I can use it whenever

    lambda max temperature
    {\displaystyle \lambda _{m}={\frac {hc}{4.96511423174\,\mathrm {kT} }}}
    """
    if not hasattr(temperature, 'unit'): # assumed Kelvin
        temperature = u.Quantity(temperature, u.Kelvin)

    lbda = const.h*const.c/(4.96511423174*const.k_B * temperature)
    return lbda.to(u.Angstrom)











def sim_ANY_LSST_lc(aadesap_lc_df, SED_filename, SED_filepath, max_sig_dist, max_chi_N_equal_M, size, model, mysurvey, tstart, tstop, no_plots_to_save, ANT_max_sim_redshift_dict, 
                    ANT_luminosity_dist_cm_dict, band_colour_dict, band_marker_dict, band_obs_cent_wl_dict, zmin, zmax, 
                    time_spline_degree = 1, wavelength_spline_degree = 3, 
                    plot_skysurvey_inputs = True, plot_SED_results = False):
    """
    This function takes a file containing the SED evolution of a transient and simulates more light curves based on this SED evolution

    INPUTS
    --------------
    aadesap_lc_df: ZTF22aadesap's real light curve data frame to plot up to compare
    SED_filename: (str) the filename of the csv file which contains the SED evolution. It is expected to start with the transient name and contain an abbreviated
                    name for the type of SED fit which was taken. These abbreviations include 'SBB' for single blackbody, 'DBB' for double blackbody and 'PL'
                    for power-law. 
    
    SED_filepath: (str) the file path to the csv file containing the SED fit evolution data

    max_sig_dist: (float) the maximum value of the reduced chi squared sigma distance of an SED which is still considered to be a good fit. SEDs with sigma distances
                    above this value will be considered to be bad fits and will not be used for simulation

    max_chi_N_equal_M: (float) We include fits where N=M in our simulations. Since they have no defined value of red chi or red chi dig dist, set a limit for the chi to 
                        consider the fit 'good'

    size: (int) the number of transient objects to generate (NOT the same as the number of plots that will be generated or the number of transients observed, since
            if you are randomly selecting ra and dec in your transient model, then there's a good chance you will simulate a transient which cannot be observed by LSST.)

    model: (dict) the Skysurvey-style model which contains your transient model parameters and how to generate them

    mysurvey: (type = Skysurvey survey?). This should be LSST loaded in from the opsim file. 

    tstart: (date) the start time of the window over which we will simulate transients

    tstop: (date) the end time of the window over which we will simulate transients

    no_plots_to_save: (float) the number of simulated lightcurves you would like to save

    ANT_max_sim_redshift_dict: (dict) contains the maximum redshift to simulate each ANT to

    ANT_luminosity_dist_cm_dict: (dict) contains the luminosity distance of each ANT calculated in cm ASSUMING A COSMOLOGY

    plot_skysurvey_inputs: (bool) if you want to plot and save the SED parameters than you are inputting into skysurvey to make sure they're doing what you want

    plot_SED_results: (bool) if you want to plot and save what the individual SEDs look like. This will be a single plot with many SEDs plotted onto it. I would like to see the point at which they become negative..?


    RETURNS
    --------------
    None                

    """
    ANT_name = SED_filename.split('_')[0] # this splits up the string that is the file name by the underscores, then we take all of the file name up until the first underscore
    #ANT_name = SED_filename[:12]
    ANT_D_cm = ANT_luminosity_dist_cm_dict[ANT_name]
    ANT_SED_df = pd.read_csv(SED_filepath, delimiter = ',', index_col = 'MJD')

    lbda_A = np.linspace(1000, 12000, 1000) # wavelength values to compute the flux at, in Angstrom
    
    


    # ================================================================================================================================================================================================================================================================================================
    # POWER LAW TRANSIENT SIMULATION
    if 'PL' in SED_filename:
        SED_label = 'PL'
        print(f'{ANT_name}: {SED_label}')

        # filter for the good SED fits only 
        ANT_good_SEDs = ANT_SED_df[ANT_SED_df['brute_chi_sigma_dist'].abs() <= max_sig_dist].copy()
        ANT_SED_N_equals_M = ANT_SED_df[ANT_SED_df['brute_chi_sigma_dist'].isna()].copy() # take all fits where N=M
        ANT_SED_N_equals_M = ANT_SED_N_equals_M[ANT_SED_N_equals_M['brute_chi'] <= max_chi_N_equal_M].copy() # take the fits where N=M and chi^2 <= 0.1

        ANT_good_SEDs = pd.concat([ANT_good_SEDs, ANT_SED_N_equals_M], ignore_index = True)
        ANT_good_SEDs = ANT_good_SEDs.sort_values(by = 'd_since_peak', ascending = True)
        
        
        # get the model params to input into the PL transient model
        phase = ANT_good_SEDs['d_since_peak'].to_numpy()
        PL_A_L_rf = ANT_good_SEDs['brute_A'].to_numpy() # the REST FRAME LUMINOSITY power law amplitude
        PL_A_F_density = PL_A_L_rf / (4 * np.pi * (ANT_luminosity_dist_cm_dict[ANT_name]**2)) # the FLUX DENSITY power law amplitude 
        PL_gamma = ANT_good_SEDs['brute_gamma'].to_numpy()

        # input the SED evolution into Skysurvey
        pl_source = power_law.get_power_law_transient_source(phase = phase, 
                                                                amplitude = PL_A_F_density, 
                                                                gamma = PL_gamma, 
                                                                lbda = lbda_A, 
                                                                time_spline_degree = time_spline_degree, 
                                                                wavelength_spline_degree = wavelength_spline_degree) # generates a TimeSeriesTransient 

        pl_transient = skysurvey.TSTransient.from_draw(size = size, model = model, template = pl_source, tstart=tstart, tstop=tstop, zmax = zmax, zmin = zmin)
        sources = pl_transient # generically label this 'sources' to be plotted later









    # ================================================================================================================================================
    # GENERATE LIGHT CURVES

    dset = skysurvey.DataSet.from_targets_and_survey(sources, mysurvey, trim_observations = True) # trim_observations just means that when it simulates the light curve, it ensures that the whole simulated light 
    #                                                                                                  curve fits within the MJD range that you specify rather than potentially just the start/end of the light curve, resulting in it getting cut off

    dset.get_ndetection()

    b_obs_cent_wl_dict = {'lsstu': 3671, # in Angstrom
                        'lsstg': 4827, 
                        'lsstr': 6223, 
                        'lssti': 7546, 
                        'lsstz': 8691, 
                        'lssty': 9712}
    
    # get the conversion from photons/s/cm^2 to ergs/s/cm^2/Angstrom
    c_cgs = const.c.cgs.value
    h_cgs = const.h.cgs.value
    F_density_conversion = c_cgs * h_cgs

    band_conversion = [(F_density_conversion / wl) for wl in b_obs_cent_wl_dict.values()]
    band_F_density_conversion_dict = dict(zip(b_obs_cent_wl_dict.keys(), band_conversion)) # a dictionary to convert from photons/s/cm^2 to a more familiar ergs/s/cm^2/Angstrom

    b_colour_dict = {'lsstu': 'purple', 
                        'lsstg': 'blue', 
                        'lsstr': 'green', 
                        'lssti': 'yellow', 
                        'lsstz': 'orange', 
                        'lssty': 'red'}
    
    b_name_dict = {'lsstu': 'LSST_u', 
                'lsstg': 'LSST_g', 
                'lsstr': 'LSST_r', 
                'lssti': 'LSST_i', 
                'lsstz': 'LSST_z', 
                'lssty': 'LSST_y'}
    
    b_cent_wl_dict = {'lsstu': 3694, 
                    'lsstg': 4841, 
                    'lsstr': 6258, 
                    'lssti': 7560, 
                    'lsstz': 8701, 
                    'lssty': 9749}

    def standard_form_tex(x, pos):
        if x == 0:
            return "0"
        exponent = int(np.floor(np.log10(abs(x))))
        coeff = x / (10 ** exponent)
        return rf"${coeff:.1f} \times 10^{{{exponent}}}$"
    formatter = FuncFormatter(standard_form_tex)



    aadesap_bands = aadesap_lc_df['band'].unique()
    i_before_run = None 
    for j, index in enumerate(dset.obs_index):

        observation = dset.get_target_lightcurve(index = index).copy() # observtaion is a dataframe of the light curve

        # get the individual transient's info
        sim_transient_params = sources.data.iloc[index, :]
        sim_z = sim_transient_params['z']
        sim_magabs = sim_transient_params['magabs']
        sim_ra = sim_transient_params['ra']
        sim_dec = sim_transient_params['dec']
        sim_magobs = sim_transient_params['magobs']


        # PLOT THE LIGHT CURVE
        bands = observation['band'].unique()

        fig, axs = plt.subplots(1, 2, figsize = (16, 7.5))
        ax1, ax2 = axs

        sorted_bands = sorted([b for b in aadesap_bands if b not in ['WISE_W1', 'WISE_W2']], key=lambda b: band_obs_centwl_dict[b])
                            
        # plot the aadesap light curve 
        for b in sorted_bands: # aadesap_bands
            if b in ['WISE_W1', 'WISE_W2']:
                continue
            b_lc_df = aadesap_lc_df[aadesap_lc_df['band'] == b].copy()
            b_colour = band_colour_dict[b]
            b_marker = band_marker_dict[b]
            label = fr'{band_obs_centwl_dict[b]:.0f}$\,\AA$ ({b})'
            ax1.errorbar(b_lc_df['MJD'], b_lc_df['obs_flux_density'], yerr = b_lc_df['obs_flux_density_err'], c = b_colour, mec = 'k', mew = '0.5', fmt = b_marker, label = label)

        # plot the simulated light curve
        sorted_LSST_bands = sorted(bands, key=lambda b: b_cent_wl_dict[b])
        for b in sorted_LSST_bands:
            b_observation = observation[observation['band'] == b].copy()
            label = fr'{b_cent_wl_dict[b]:.0f}$\,\AA$ ({b_name_dict[b]})'
            ax2.errorbar(b_observation['mjd'], b_observation['flux'] * band_F_density_conversion_dict[b], yerr = b_observation['fluxerr']* band_F_density_conversion_dict[b], c = b_colour_dict[b], mec = 'k', mew = '0.5', fmt = 'o', label =  label)
        #ax2.set_ylabel(r'Flux density (scaled by AB mag system) \n'+r'/ ergs $\mathbf{s^{-1} cm^{-2} \AA^{-1}} $', fontweight = 'bold', fontsize = (titlefontsize - 5))

        for ax in [ax1, ax2]:
            ax.yaxis.set_major_formatter(formatter)  
            ax.get_yaxis().get_offset_text().set_visible(False) # Hide the offset that matplotlib adds 
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))
            ax.tick_params(axis='both', labelsize = 15)
            ax.legend(fontsize = 14)
            ax.grid(True)

        titlefontsize = 21
        ax1.set_xlim(59600, 60600)
        ax1.set_title("ZTF22aadesap's observed light curve", fontsize = 17, fontweight = 'bold')
        ax2.set_title("Simulated light curve using ZTF22aadesap's \npower-law spectral evolution", fontsize = 17, fontweight = 'bold')
        fig.supylabel(r'Spectral flux density [ergs $\mathbf{s^{-1} \, cm^{-2} \, \AA^{-1}}$]', fontweight = 'bold', fontsize = (titlefontsize - 4))
        fig.supxlabel('MJD [days]', fontweight = 'bold', fontsize = (titlefontsize - 4))
        fig.subplots_adjust(left = 0.15, wspace = 0.25, right = 0.95)

        # save the plot - since the result of each run of the code is different, make sure to save each plot with a different name (adding  anumber at the end) os we don't overwrite the previously saved plots
        base_savepath = f"/media/data3/lauren/YoRiS/final_simulated_lcs/{ANT_name}/LSST_{SED_label}/COMPARE_REAL_VS_SIM_LC"
        i = 1
        while os.path.exists(f"{base_savepath}_{i}.png"):
            i += 1
            
        if j == 0: # this takes the plot number before we ran this code over. 
            i_before_run = i - 1

        if i > (i_before_run + no_plots_to_save): # if we have already saved (no_plots_to_save) plots, then stop. 
            break

        new_savepath = f"{base_savepath}_{i}.png" # if we've saved less than (no_plots_to_save) plots so far, then we can save this one
        plt.savefig(new_savepath, dpi = 300)
        plt.close(fig)





















##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################

# ================================================================================================================================================================================================================================================================================================
# MAKING THE SURVEY 

opsim_path = 'baseline_v4.3.1_10yrs.db'
LSST_survey = skysurvey.LSST.from_opsim(opsim_path)
print()
print()
print(f'{Fore.GREEN} =========================================={Style.RESET_ALL}')
print()
print(f'{Fore.GREEN} LSST survey loaded in {Style.RESET_ALL}')
print()
print(f'{Fore.GREEN} =========================================={Style.RESET_ALL}')
print()
print()




# ================================================================================================================================================================================================================================================================================================
# THE TRANSIENT MODEL PARAM GENERATOR MODEL



_RATE = 0.001
_MODEL = dict( redshift = {"kwargs": {"zmax": 0.05}, 
                            "as": "z"},
                t0 = {"func": np.random.uniform,
                        "kwargs": {"low": 56_000, "high": 56_200}
                    },
                        
                magabs = {"func": np.random.normal,
                            "kwargs": {"loc": -18.5, "scale": 0.5}
                        },
                            
                magobs = {"func": "magabs_to_magobs",
                            "kwargs": {"z":"@z", "magabs": "@magabs"}
                        },
                            
                amplitude = {"func": "magobs_to_amplitude",
                            "kwargs": {"magobs": "@magobs"}
                        },
                # This you need to match with the survey
                radec = {"func": random_radec,
                        "as": ["ra","dec"]
                        }
                )



tstart = "2026-04-10"
tstop = "2030-04-01"
size = 500
no_plots_to_save = 100
time_spline_degree = 1 # traces back to sncosmo
wavelength_spline_degree = 3 # traces back to sncosmo

max_sig_dist = 3.0
max_chi_N_equal_M = 0.1


aadesap_z = ANT_proper_redshift_dict['ZTF22aadesap']
zmin = aadesap_z - 0.001
zmax = aadesap_z + 0.001

path = "/media/data3/lauren/YoRiS/normal_code/data/SED_fits/ZTF22aadesap/ZTF22aadesap_UVOT_guided_PL_SED_fit_across_lc.csv"

aadesap_lc_path = "/media/data3/lauren/YoRiS/normal_code/ZTF22aadesap_flux_density.csv"
aadesap_lc_df = pd.read_csv(aadesap_lc_path, delimiter = ',')
# I RECON PUT ALL ANT'S SED EVOLUTION FILES THAT WE WANT TO USE INTO A SINGLE FOLDER, THEN ITERATE THROUGH ALL FILES IN THIS FOLDER

# ================================================================================================================================================================================================================================================================================================

SED_filename = path.split('/')[-1] # this takes the last part of the path which is the file name
sim_ANY_LSST_lc(aadesap_lc_df = aadesap_lc_df, SED_filename = SED_filename, SED_filepath = path, max_sig_dist = max_sig_dist, max_chi_N_equal_M = max_chi_N_equal_M, size = size, model = _MODEL, mysurvey = LSST_survey, 
                        tstart = tstart, tstop = tstop, no_plots_to_save = no_plots_to_save, ANT_max_sim_redshift_dict = ANT_sim_redshift_upper_lim_dict, band_colour_dict = band_colour_dict, band_marker_dict = band_marker_dict, band_obs_cent_wl_dict = band_obs_centwl_dict,
                        zmin = zmin, zmax = zmax,
                        ANT_luminosity_dist_cm_dict = ANT_luminosity_dist_cm_dict, time_spline_degree = time_spline_degree, wavelength_spline_degree = wavelength_spline_degree,
                        plot_skysurvey_inputs = True, plot_SED_results = True)
