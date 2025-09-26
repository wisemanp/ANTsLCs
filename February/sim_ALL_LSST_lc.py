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

max_sig_dist = 3.0


ANT_redshift_dict = {'ZTF18aczpgwm': 0.4279, 
                     'ZTF19aailpwl': 0.3736, 
                     'ZTF19aamrjar': 0.697, 
                     'ZTF19aatubsj': 0.2666, 
                     'ZTF20aanxcpf': 0.083, 
                     'ZTF20abgxlut': 0.257, # this one was also hard to find, also in Wiseman(2024) but in footnote 24's link
                     'ZTF20abodaps': 0.607, 
                     'ZTF20abrbeie': 0.9945, 
                     'ZTF20acvfraq': 0.26, 
                     'ZTF21abxowzx': 0.419, 
                     'ZTF22aadesap': 0.073, # this one was harder to find, its still in Phil 2024 paper of 10 ANTs
                     'ASASSN-17jz': 0.1641, # Holoien, T. W. S(2022) measured by them
                     'ASASSN-18jd': 0.1192, # Neustadt, J. M. M(2020) using H alpha lines in the spectra
                     'CSS100217': 0.147, # Drake, A. J(2011) from spectroscopic observations of which galaxy it is within
                     'Gaia16aaw': 1.03, # Hinkle, T. J(2024) a broad feature interpreted as Mg II  
                     'Gaia18cdj': 0.93747, # Hinkle, T. J(2024) clear Mg II absorption doublet
                     'PS1-10adi': 0.203, # +/- 0.001 from Kankare, E(2017) from the Balmer lines of PS1-10adi, which is in agreement with the redshift of its host galaxy, z = 0.219 +/- 0.025
                     'PS1-13jw': 0.345 # from Kankare, E(2017) who got it from spectroscopic redshifts from SDSS
                     } 



##################################################################################################################################################################

# ASSUMPTIONS FOR LUMINOSITY DISTANCE ARE BELOW ----------------------------------------------
H0 = 70 #km/s/Mpc
om_M = 0.3 # non relativistic matter density fraction
fcdm = FlatLambdaCDM(H0 = H0, Om0 = om_M)
# ASSUMPTIONS FOR LUMINOSITY DISTANCE ARE ABOVE ----------------------------------------------

ANT_d_cm_list = []
for z in ANT_redshift_dict.values():
    d = fcdm.luminosity_distance(z).to(u.cm).value # this gives the luminosity distance in cm
    ANT_d_cm_list.append(d)

ANT_names = list(ANT_redshift_dict.keys())

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











def sim_ANY_LSST_lc(SED_filename, SED_filepath, max_sig_dist, size, model, mysurvey, tstart, tstop, no_plots_to_save, ANT_redshift_dict, 
                    ANT_luminosity_dist_cm_dict, time_spline_degree = 1, wavelength_spline_degree = 3, 
                    plot_skysurvey_inputs = True, plot_SED_results = False):
    """
    This function takes a file containing the SED evolution of a transient and simulates more light curves based on this SED evolution

    INPUTS
    --------------
    SED_filename: (str) the filename of the csv file which contains the SED evolution. It is expected to start with the transient name and contain an abbreviated
                    name for the type of SED fit which was taken. These abbreviations include 'SBB' for single blackbody, 'DBB' for double blackbody and 'PL'
                    for power-law. 
    
    SED_filepath: (str) the file path to the csv file containing the SED fit evolution data

    max_sig_dist: (float) the maximum value of the reduced chi squared sigma distance of an SED which is still considered to be a good fit. SEDs with sigma distances
                    above this value will be considered to be bad fits and will not be used for simulation

    size: (int) the number of transient objects to generate (NOT the same as the number of plots that will be generated or the number of transients observed, since
            if you are randomly selecting ra and dec in your transient model, then there's a good chance you will simulate a transient which cannot be observed by LSST.)

    model: (dict) the Skysurvey-style model which contains your transient model parameters and how to generate them

    mysurvey: (type = Skysurvey survey?). This should be LSST loaded in from the opsim file. 

    tstart: (date) the start time of the window over which we will simulate transients

    tstop: (date) the end time of the window over which we will simulate transients

    no_plots_to_save: (float) the number of simulated lightcurves you would like to save

    ANT_redshift_dict: (dict) contains the redshift of each ANT

    ANT_luminosity_dist_cm_dict: (dict) contains the luminosity distance of each ANT calculated in cm ASSUMING A COSMOLOGY

    plot_skysurvey_inputs: (bool) if you want to plot and save the SED parameters than you are inputting into skysurvey to make sure they're doing what you want

    plot_SED_results: (bool) if you want to plot and save what the individual SEDs look like. This will be a single plot with many SEDs plotted onto it. I would like to see the point at which they become negative..?


    RETURNS
    --------------
    None                

    """
    ANT_name = SED_filename.split('_')[0] # this splits up the string that is the file name by the underscores, then we take all of the file name up until the first underscore
    #ANT_name = SED_filename[:12]
    ANT_z = ANT_redshift_dict[ANT_name]
    ANT_max_simulation_z = ANT_z # SHOULD CALCULATE THE ACTUAL ZMAX AND ZMIN
    ANT_D_cm = ANT_luminosity_dist_cm_dict[ANT_name]
    ANT_SED_df = pd.read_csv(SED_filepath, delimiter = ',', index_col = 'MJD')

    lbda_A = np.linspace(1000, 12000, 1000) # wavelength values to compute the flux at, in Angstrom
    

    # check what kind of SED was fit to this data
    # ================================================================================================================================================================================================================================================================================================
    # SINGLE BLACKBODY TRANSIENT SIMULATION
    if 'SBB' in SED_filename:
        SED_label = 'SBB'

        # filter for the good SED fits only 
        ANT_good_SEDs = ANT_SED_df[ANT_SED_df['brute_chi_sigma_dist'].abs() <= max_sig_dist].copy()
        ANT_good_SEDs = ANT_good_SEDs.sort_values(by = 'd_since_peak', ascending = True)

        # get the model params to input into the SBB transient model
        phase = ANT_good_SEDs['d_since_peak'].to_numpy()
        BB_T = ANT_good_SEDs['brute_T_K'].to_numpy()
        BB_R = ANT_good_SEDs['brute_R_cm'].to_numpy()

        # caluclate the amplitude input that it wants (the flux density at the BB's peak in flux density units)
        BB_Wein_lambda_max_Angstrom = [get_wein_lbdamax(T) for T in BB_T] # the peak BB wavelength calculated by Wein's law in astropy Angstrom units
        BB_Wein_lambda_max_Angstrom_values = [i.value for i in BB_Wein_lambda_max_Angstrom] # the above but with the units removed
        BB_Wein_lambda_max_cm_values = [i*1e-8 for i in BB_Wein_lambda_max_Angstrom_values] # the above but with the values being unitless but in cm (if that makes sense, it was converted into cm then had the units removed)
        BB_amplitude = np.array([blackbody_F_lam(lam_cm = lam_cm, T_K = BB_T[i], R_cm = BB_R[i], D_cm = ANT_D_cm) for i, lam_cm in enumerate(BB_Wein_lambda_max_cm_values)]) # the amplitude to input into get_blackbody_transient_source(), it is equal to the BB flux density at the peak wavelength, in units of flux

        # input the SED evolution into Skysurvey
        bb_source = blackbody.get_blackbody_transient_source(phase = phase, # generates a TimeSeriesTransient 
                                                amplitude = BB_amplitude, 
                                                temperature = BB_T, 
                                                lbda = lbda_A, 
                                                time_spline_degree = time_spline_degree, 
                                                wavelength_spline_degree = wavelength_spline_degree) # lbda has a default
        
        bb_transient = skysurvey.TSTransient.from_draw(size = size, model = model, template = bb_source, tstart=tstart, tstop=tstop, zmax = ANT_max_simulation_z)
        sources = bb_transient # generically label this 'sources' to be plotted later
        
        # PLOTTING THE SKYSURVEY INPUT PARAMS
        if plot_skysurvey_inputs:
            fig, axs = plt.subplots(3, 1, figsize = (10, 12), sharex = True)
            ax1, ax2, ax3 = axs.flatten()
            ax1.set_title('INPUT: BB T', fontweight = 'bold')
            ax1.set_ylabel('BB Temperature / K', fontweight = 'bold')
            ax1.plot(phase, BB_T, marker = 'o', linestyle = 'None', mec = 'k', mew = 0.5)
            
            ax2.set_title('INPUT: BB amplitude (= BB flux density at the BB peak wavelength) \n'+r'amplitude $\mathbf{ = 2\pi h c^2 \frac{R_{BB}}{D_{L}}^2 \frac{1}{exp(\frac{hc}{\lambda k T_{BB}}) -1} }$', fontweight = 'bold')
            ax2.set_ylabel('BB Amplitude / cm', fontweight = 'bold')
            ax2.plot(phase, BB_amplitude, marker = 'o', linestyle = 'None', mec = 'k', mew = 0.5)

            ax3.set_title('Not input: BB R, but used to calculate the amplitude (alongside BB T and Wein peak wavelength)', fontweight = 'bold')
            ax3.set_ylabel('BB Radius / cm', fontweight = 'bold')
            ax3.plot(phase, BB_R, marker = 'o', linestyle = 'None', mec = 'k', mew = 0.5)

            for ax in axs:
                ax.grid(True)
            fig.supxlabel('Phase (days since peak) / rest frame days', fontweight = 'bold')
            fig.suptitle(f"Input params for {ANT_name}'s SBB SED fit", fontweight = 'bold')
            plt.savefig(f"/media/data3/lauren/YoRiS/test_simulated_lcs/{ANT_name}/LSST_SBB/{ANT_name}_INPUT_PARAMS_SBB.png", dpi = 300)



    # ================================================================================================================================================================================================================================================================================================
    # POWER LAW TRANSIENT SIMULATION
    elif 'PL' in SED_filename:
        SED_label = 'PL'

        # filter for the good SED fits only 
        ANT_good_SEDs = ANT_SED_df[ANT_SED_df['brute_chi_sigma_dist'].abs() <= max_sig_dist].copy()
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

        pl_transient = skysurvey.TSTransient.from_draw(size = size, model = _MODEL, template = pl_source, tstart=tstart, tstop=tstop, zmax = ANT_max_simulation_z)
        sources = pl_transient # generically label this 'sources' to be plotted later

        # PLOTTING THE SKYSURVEY INPUT PARAMS
        if plot_skysurvey_inputs:
            fig, axs = plt.subplots(3, 1, figsize = (10, 12), sharex = True)
            ax1, ax2, ax3 = axs.flatten()
            ax1.set_title('INPUT: PL gamma', fontweight = 'bold')
            ax1.plot(phase, PL_gamma, marker = 'o', linestyle = 'None', mec = 'k', mew = 0.5)
            ax1.set_ylabel('gamma / no units', fontweight = 'bold')

            ax2.set_title(r'INPUT: PL Amplitude for flux density ($\mathbf{ = \frac{ A_{L_{rf}} }{4\pi D_{L}^{2}} }$)', fontweight = 'bold')
            ax2.set_ylabel(r'amplitude (flux density) / erg $\mathbf{s^{-1} cm^{-2} \AA^{-1}}$', fontweight = 'bold')
            ax2.plot(phase, PL_A_F_density, marker = 'o', linestyle = 'None', mec = 'k', mew = 0.5)
            ax2.set_yscale('log')

            ax3.set_title(r'Not input: PL Amplitude for luminosity ($\mathbf{ = A_{L_{rf}} }$)', fontweight = 'bold')            
            ax3.set_ylabel(r'amplitude (L_rf) / erg $\mathbf{s^{-1} \AA^{-1}}$', fontweight = 'bold')
            ax3.plot(phase, PL_A_L_rf, marker = 'o', linestyle = 'None', mec = 'k', mew = 0.5)
            ax3.set_yscale('log')
            

            for ax in axs:
                ax.grid(True)
            fig.supxlabel('Phase (days since peak) / rest frame days', fontweight = 'bold')
            fig.suptitle(f"Input params for {ANT_name}'s SBB SED fit", fontweight = 'bold')
            fig.tight_layout()
            plt.savefig(f"/media/data3/lauren/YoRiS/test_simulated_lcs/{ANT_name}/LSST_PL/{ANT_name}_INPUT_PARAMS_PL.png", dpi = 300)



    elif 'DBB' in SED_filename:
        SED_label = 'DBB'

        # filter for the good SED fits only 
        ANT_good_SEDs = ANT_SED_df[ANT_SED_df['cf_chi_sigma_dist'].abs() <= max_sig_dist].copy()
        ANT_good_SEDs = ANT_good_SEDs.sort_values(by = 'd_since_peak', ascending = True)

        # get the model params to input into the DBB transient model
        phase = ANT_good_SEDs['d_since_peak'].to_numpy()
        BB_T1 = ANT_good_SEDs['cf_T1_K'].to_numpy()
        BB_R1 = ANT_good_SEDs['cf_R1_cm'].to_numpy()
        BB_T2 = ANT_good_SEDs['cf_T2_K'].to_numpy()
        BB_R2 = ANT_good_SEDs['cf_R2_cm'].to_numpy()
        
        # get the 'amplitude' input that Skysurvey wants - this is each BB's peak flux density (found using Wein's law)
        BB1_Wein_lambda_max_Angstrom = [get_wein_lbdamax(T) for T in BB_T1] # the peak BB wavelength at each phase, calculated by Wein's law in astropy Angstrom units
        BB1_Wein_lambda_max_Angstrom_values = [i.value for i in BB1_Wein_lambda_max_Angstrom] # the above but with the units removed
        BB1_Wein_lambda_max_cm_values = [i*1e-8 for i in BB1_Wein_lambda_max_Angstrom_values] # the above but with the values being unitless but in cm (if that makes sense, it was converted into cm then had the units removed)
        BB1_amplitude = np.array([blackbody_F_lam(lam_cm = lam_cm, T_K = BB_T1[i], R_cm = BB_R1[i], D_cm = ANT_D_cm) for i, lam_cm in enumerate(BB1_Wein_lambda_max_cm_values)]) # the amplitude to input into get_blackbody_transient_source(), it is equal to the BB flux density at the peak wavelength, in units of flux

        BB2_Wein_lambda_max_Angstrom = [get_wein_lbdamax(T) for T in BB_T2]# the peak BB wavelength at each phase, calculated by Wein's law in astropy Angstrom units
        BB2_Wein_lambda_max_Angstrom_values = [i.value for i in BB2_Wein_lambda_max_Angstrom] # the above but with the units removed
        BB2_Wein_lambda_max_cm_values = [i*1e-8 for i in BB2_Wein_lambda_max_Angstrom_values] # the above but with the values being unitless but in cm (if that makes sense, it was converted into cm then had the units removed)
        BB2_amplitude = np.array([blackbody_F_lam(lam_cm = lam_cm, T_K = BB_T2[i], R_cm = BB_R2[i], D_cm = ANT_D_cm) for i, lam_cm in enumerate(BB2_Wein_lambda_max_cm_values)]) # the amplitude to input into get_blackbody_transient_source(), it is equal to the BB flux density at the peak wavelength, in units of flux

        # input SED evolution into Skysurvey
        dbb_source = double_blackbody.get_double_blackbody_transient_source(phase = phase, # generates a TimeSeriesTransient 
                                                                amplitude_1 = BB1_amplitude, 
                                                                temperature_1 = BB_T1, 
                                                                amplitude_2 = BB2_amplitude, 
                                                                temperature_2 = BB_T2,
                                                                lbda = lbda_A, 
                                                                time_spline_degree = time_spline_degree, 
                                                                wavelength_spline_degree = wavelength_spline_degree) # lbda has a default

        dbb_transient = skysurvey.TSTransient.from_draw(size = 200, model = _MODEL, template = dbb_source, tstart="2026-04-10", tstop="2031-04-01", zmax = ANT_max_simulation_z)
        sources = dbb_transient # generically label this 'sources' to be plotted later

        if plot_skysurvey_inputs:
            fig, axs = plt.subplots(3, 2, figsize = (16, 12), sharex = True)
            ax1, ax2, ax3, ax4, ax5, ax6 = axs.flatten()
            ax1.set_title('INPUT: BB T1', fontweight = 'bold')
            ax1.set_ylabel('BB Temperature / K', fontweight = 'bold')
            ax1.plot(phase, BB_T1, marker = 'o', linestyle = 'None', mec = 'k', mew = 0.5)
            
            ax3.set_title('INPUT: BB amplitude1 (= BB1 flux density at the BB1 peak wavelength) \n'+r'amplitude $\mathbf{ = 2\pi h c^2 \left( \frac{R_{1, BB}}{D_{L}} \right) ^2 \frac{1}{exp(\frac{hc}{\lambda k T_{1, BB}}) -1} }$', fontweight = 'bold')
            ax3.set_ylabel('BB Amplitude / cm', fontweight = 'bold')
            ax3.plot(phase, BB1_amplitude, marker = 'o', linestyle = 'None', mec = 'k', mew = 0.5)

            ax5.set_title('Not input: BB R1, but used to calculate the amplitude (alongside BB T1 \nand Wein peak wavelength)', fontweight = 'bold')
            ax5.set_ylabel('BB Radius / cm', fontweight = 'bold')
            ax5.plot(phase, BB_R1, marker = 'o', linestyle = 'None', mec = 'k', mew = 0.5)

            ax2.set_title('INPUT: BB T2', fontweight = 'bold')
            ax2.set_ylabel('BB Temperature / K', fontweight = 'bold')
            ax2.plot(phase, BB_T2, marker = 'o', linestyle = 'None', mec = 'k', mew = 0.5, c = 'red')
            
            ax4.set_title('INPUT: BB amplitude2 (= BB2 flux density at the BB2 peak wavelength) \n'+r'amplitude $\mathbf{ = 2\pi h c^2 \left( \frac{R_{2, BB}}{D_{L}} \right) ^2 \frac{1}{exp(\frac{hc}{\lambda k T_{2, BB}}) -1} }$', fontweight = 'bold')
            ax4.set_ylabel('BB Amplitude / cm', fontweight = 'bold')
            ax4.plot(phase, BB2_amplitude, marker = 'o', linestyle = 'None', mec = 'k', mew = 0.5, c = 'red')

            ax6.set_title('Not input: BB R2, but used to calculate the amplitude (alongside BB T2 \nand Wein peak wavelength)', fontweight = 'bold')
            ax6.set_ylabel('BB Radius / cm', fontweight = 'bold')
            ax6.plot(phase, BB_R2, marker = 'o', linestyle = 'None', mec = 'k', mew = 0.5, c = 'red')



            for ax in axs.ravel():
                ax.grid(True)
            fig.supxlabel('Phase (days since peak) / rest frame days', fontweight = 'bold')
            fig.suptitle(f"Input params for {ANT_name}'s DBB SED fit", fontweight = 'bold', fontsize = 20)
            fig.tight_layout()
            plt.savefig(f"/media/data3/lauren/YoRiS/test_simulated_lcs/{ANT_name}/LSST_DBB/{ANT_name}_INPUT_PARAMS_DBB.png", dpi = 300)


    # ================================================================================================================================================
    # PLOT THE SEDS ONTO A SINGLE PLOT TO SEE IF THEY GO NEGATIVE AT ANY POINT...
    if plot_SED_results:
        titlefontsize = 17
        if SED_label == 'SBB':
            fluxes = blackbody.get_blackbody_transient_flux(lbda = lbda_A, temperature = BB_T, amplitude = BB_amplitude, normed = True) # it normalises it in the function 'blackbody_lambda' but in get_blackbody_transient_flux it does flux = normed_BB * amplitude
            fig = plt.figure(figsize = (16, 7.5))
            ax = fig.add_subplot(111)
            cmap = plt.get_cmap("jet") 
            #colors = cmap((BB_T - BB_T.min())/(BB_T.max()-BB_T.min()))
            colors = cmap((phase - phase.min())/(phase.max()-phase.min()))
            _ = [ax.plot(lbda_A, flux_, color=c) for flux_,c in zip(fluxes, colors)]
            
            norm = Normalize(vmin = phase.min(), vmax = phase.max())
            sm = ScalarMappable(norm, cmap)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax = ax)
            cbar.set_label('phase (days since peak) / rest frame days', fontweight = 'bold')

            ax.set_xlabel(r'Wavelength / $\mathbf{\AA}$', fontweight = 'bold', fontsize = (titlefontsize - 2))
            ax.set_ylabel(r'Flux density  / ergs$ \mathbf{ s^{-1} cm^{-2} \AA^{-1} } $', fontweight = 'bold', fontsize = (titlefontsize - 2))
            ax.set_title(f"Using blackbody.get_blackbody_transient_flux() and inputting our BB T and amplitudes \n for {ANT_name}'s SBB SED fits. \nWe use this to generate the transient model to simulate lightcurves from", fontsize = titlefontsize, fontweight = 'bold')
            fig.tight_layout()
            plt.savefig(f"/media/data3/lauren/YoRiS/test_simulated_lcs/{ANT_name}/LSST_SBB/{ANT_name}_GENERATE_INPUT_SBB_SEDs.png", dpi = 300)

        elif SED_label == 'PL':
            fluxes = power_law.get_power_law_transient_flux(lbda = lbda_A, A = PL_A_F_density, gamma = PL_gamma) # it normalises it in the function 'blackbody_lambda' but in get_blackbody_transient_flux it does flux = normed_BB * amplitude
            fig = plt.figure(figsize = (16, 7.5))
            ax = fig.add_subplot(111)
            cmap = plt.get_cmap("jet") 
            #colors = cmap((PL_gamma - PL_gamma.min())/(PL_gamma.max()-PL_gamma.min()))
            colors = cmap((phase - phase.min())/(phase.max()-phase.min()))

            norm = Normalize(vmin = phase.min(), vmax = phase.max())
            sm = ScalarMappable(norm, cmap)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax = ax)
            cbar.set_label('phase (days since peak) / rest frame days', fontweight = 'bold')

            _ = [ax.plot(lbda_A, flux_, color=c) for flux_,c in zip(fluxes, colors)]
            ax.set_xlabel(r'Wavelength / $\mathbf{\AA}$', fontweight = 'bold', fontsize = (titlefontsize - 2))
            ax.set_ylabel(r'Flux density  / ergs$ \mathbf{ s^{-1} cm^{-2} \AA^{-1} } $', fontweight = 'bold', fontsize = (titlefontsize - 2))
            ax.set_title(f"Using power_law.get_power_law_transient_flux() and inputting our "+r"$\mathbf{ A_{F_{\lambda}}}$ and $\mathbf{\gamma}$s"+f"  \n for {ANT_name}'s PL SED fits. \nWe use this to generate the transient model to simulate lightcurves from", fontsize = titlefontsize, fontweight = 'bold')
            fig.tight_layout()
            plt.savefig(f"/media/data3/lauren/YoRiS/test_simulated_lcs/{ANT_name}/LSST_PL/{ANT_name}_GENERATE_INPUT_PL_SEDs.png", dpi = 300)

        elif SED_label == 'DBB':
            fluxes1 = double_blackbody.get_blackbody_transient_flux(lbda = lbda_A, temperature = BB_T1, amplitude = BB1_amplitude, normed = True) # it normalises it in the function 'blackbody_lambda' but in get_blackbody_transient_flux it does flux = normed_BB * amplitude
            fluxes2 = double_blackbody.get_blackbody_transient_flux(lbda = lbda_A, temperature = BB_T2, amplitude = BB2_amplitude, normed = True) # it normalises it in the function 'blackbody_lambda' but in get_blackbody_transient_flux it does flux = normed_BB * amplitude
            fluxes = fluxes1 + fluxes2
            
            fig = plt.figure(figsize = (16, 7.5))
            ax = fig.add_subplot(111)
            cmap = plt.get_cmap("jet") 
            colors = cmap((phase - phase.min())/(phase.max()-phase.min()))
            _ = [ax.plot(lbda_A, flux_, color=c) for flux_,c in zip(fluxes, colors)]
            
            norm = Normalize(vmin = phase.min(), vmax = phase.max())
            sm = ScalarMappable(norm, cmap)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax = ax)
            cbar.set_label('phase (days since peak) / rest frame days', fontweight = 'bold')

            ax.set_xlabel(r'Wavelength / $\mathbf{\AA}$', fontweight = 'bold', fontsize = (titlefontsize - 2))
            ax.set_ylabel(r'Flux density  / ergs$ \mathbf{ s^{-1} cm^{-2} \AA^{-1} } $', fontweight = 'bold', fontsize = (titlefontsize - 2))
            ax.set_title(f"Using blackbody.get_blackbody_transient_flux() and inputting our BB T1, amplitude1, \nBB T2 and amplitude2s for {ANT_name}'s DBB SED fits. \nWe use this to generate the transient model to simulate lightcurves from", fontsize = titlefontsize, fontweight = 'bold')
            fig.tight_layout()
            plt.savefig(f"/media/data3/lauren/YoRiS/test_simulated_lcs/{ANT_name}/LSST_DBB/{ANT_name}_GENERATE_INPUT_DBB_SEDs.png", dpi = 300)


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
        b_colour_dict = {'lsstu': 'purple', 
                        'lsstg': 'blue', 
                        'lsstr': 'green', 
                        'lssti': 'yellow', 
                        'lsstz': 'orange', 
                        'lssty': 'red'}

        fig, axs = plt.subplots(1, 2, figsize = (16, 7.5))
        ax1, ax2 = axs

        for b in bands:

            b_observation = observation[observation['band'] == b].copy()
            ax1.errorbar(b_observation['mjd'], b_observation['flux'], yerr = b_observation['fluxerr'], c = b_colour_dict[b], mec = 'k', mew = '0.5', fmt = 'o', label = b)
            ax2.errorbar(b_observation['mjd'], b_observation['flux'] * band_F_density_conversion_dict[b], yerr = b_observation['fluxerr']* band_F_density_conversion_dict[b], c = b_colour_dict[b], mec = 'k', mew = '0.5', fmt = 'o', label = b)
        ax1.set_ylabel(r'Flux (scaled by AB mag system)\n'+r'/ photons $\mathbf{s^{-1} cm^{-2}} $', fontweight = 'bold', fontsize = (titlefontsize - 5))
        ax2.set_ylabel(r'Flux density (scaled by AB mag system) \n'+r'/ ergs $\mathbf{s^{-1} cm^{-2} \AA^{-1}} $', fontweight = 'bold', fontsize = (titlefontsize - 5))
        ax1.legend()
        ax2.grid(True)
        ax1.grid(True)

        titlefontsize = 20
        fig.supxlabel('MJD', fontweight = 'bold', fontsize = (titlefontsize - 5))
        title = f"Simulated light curve using {ANT_name}'s {SED_label} SED fit results\n" + \
                f"z = {sim_z:.4f}, magabs = {sim_magabs:.4f}, magobs = {sim_magobs:.4f}, ra, dec = ({sim_ra:.4f}, {sim_dec:.4f})"
        fig.suptitle(title, fontweight = 'bold', fontsize = titlefontsize)


        # save the plot - since the result of each run of the code is different, make sure to save each plot with a different name (adding  anumber at the end) os we don't overwrite the previously saved plots
        base_savepath = f"/media/data3/lauren/YoRiS/test_simulated_lcs/{ANT_name}/LSST_{SED_label}/test_{ANT_name}_SED_{SED_label}_lc_sim"
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
                            "kwargs": {"loc": -24, "scale": 1}
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
size = 200
no_plots_to_save = 10
time_spline_degree = 1 # traces back to sncosmo
wavelength_spline_degree = 3 # traces back to sncosmo

folder = False
#path = "/media/data3/lauren/YoRiS/ZTF18aczpgwm_SBB_SED_fit_across_lc.csv"
#path = "/media/data3/lauren/YoRiS/ZTF18aczpgwm_PL_SED_fit_across_lc.csv"
path = "/media/data3/lauren/YoRiS/ZTF22aadesap_UVOT_guided_DBB_SED_fit_across_lc.csv"




# ================================================================================================================================================================================================================================================================================================
if not folder:
    SED_filename = path.split('/')[-1] # this takes the last part of the path which is the file name
    sim_ANY_LSST_lc(SED_filename = SED_filename, SED_filepath = path, max_sig_dist = max_sig_dist, size = size, model = _MODEL, mysurvey = LSST_survey, 
                            tstart = tstart, tstop = tstop, no_plots_to_save = no_plots_to_save, ANT_redshift_dict = ANT_redshift_dict, 
                            ANT_luminosity_dist_cm_dict = ANT_luminosity_dist_cm_dict, time_spline_degree = time_spline_degree, wavelength_spline_degree = wavelength_spline_degree,
                            plot_skysurvey_inputs = True, plot_SED_results = True)


elif folder:
    for subfolder in os.listdir(path):
        full_subfolder_path = os.path.join(path, subfolder)

        for SED_filename in os.listdir(subfolder):
            full_filepath = os.path.join(full_subfolder_path, SED_filename)

            sim_ANY_LSST_lc(SED_filename = SED_filename, SED_filepath = full_filepath, max_sig_dist = max_sig_dist, size = size, model = _MODEL, mysurvey = LSST_survey, 
                            tstart = tstart, tstop = tstop, no_plots_to_save = no_plots_to_save, ANT_redshift_dict = ANT_redshift_dict, 
                            ANT_luminosity_dist_cm_dict = ANT_luminosity_dist_cm_dict, time_spline_degree = time_spline_degree, wavelength_spline_degree = wavelength_spline_degree,
                            plot_skysurvey_inputs = True, plot_SED_results = True)