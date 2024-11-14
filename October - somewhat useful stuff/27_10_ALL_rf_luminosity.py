import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import scipy.optimize as opt
from astropy import constants as const
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM 

print()
st = time.time()
H0 = 70 #km/s/Mpc
om_M = 0.3 # non relativistic matter density fraction
fcdm = FlatLambdaCDM(H0 = H0, Om0 = om_M)


##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################


def restframe_luminosity(d_l_cm, bandZP, z, m, m_err):
    """
    Calculates the rest frame luminosity 

    INPUTS:
    -----------------------
    d_l_cm: luminosity distance in cm ( can be calculated using astropy.cosmology.luminosity_distance(z) )

    bandZP: observed band's AB mag zeropoint in ergs/s/cm^2/Angstrom

    z: object's redshift

    m: magnitude (either abs or apparent, just account for this with d_l)

    m_err: magnitude error


    OUTPUTS
    -----------------------
    L_rf: rest frame luminosity in ergs/s/Angstrom

    L_rf_err: the rest frame luminosity's error, propagated from the error on the magnitude only - in ergs/s/Angstrom

    """
    L_rf = 4 * np.pi * (d_l_cm**2) * bandZP * (1 + z) * (10**(-0.4 * m)) # in ergs/s/Angstrom

    L_rf_err = 0.4 * np.log(10) * L_rf * m_err # in ergs/s/Angstrom

    return L_rf, L_rf_err

##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################


""" def obs_wl_to_em_wl_dict(obs_band_cent_wl_dict, z):
    alldata_obs_cent_wl = list(obs_band_cent_wl_dict.values()) # a list of all of the observed band central luminosites (INCLUDES THE VALUES WHICH ARE SET TO 0.0 BECASUE WE DONT HAVE ANY DATA ON THE BAND)
    obs_cent_wl = [wl for wl in alldata_obs_cent_wl if wl > 0.0] # ASSUMES THAT IF YOU DON'T HAVE DATA ON A BAND, YOU SET ITS CENTRAL WALEVENGTH TO 0.0 + FILTERS THESE OUT
    obs
    return obs_wavelength / (1+z) """


def obs_wl_to_em_wl(obs_wavelength, z):
    return obs_wavelength / (1+z)


##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################











##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################



def weighted_mean(data, errors):
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    if not isinstance(errors, np.ndarray):
        errors = np.array(errors)
    
    if len(errors) == 0: # if the bin has no data within it, then the weighted mean and its error = NaN
        wm = pd.NA
        wm_err = pd.NA

    else: # if the bin has data within it, then take the weighted mean
        weights = 1/(errors**2)
        wm = np.sum(data * weights) / (np.sum(weights))
        wm_err = np.sqrt( 1/(np.sum(weights)) )

    return wm, wm_err



##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################



def blackbody(lam_cm, R_cm, T_K):
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



##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################


def chisq(y_m, y, yerr, M, reduced_chi = False):
    """
    Calculates the chi squared, as well as the reduced chi squared and its 1 sigma uncertainty allowance if wanted

    INPUTS:
    --------------
    y_m: model y data

    y: observed y data

    yerr: observed y errors

    M: number of model paramaters

    reduced_chi: if True, this function will return chi, reduced_chi and red_chi_1sig, if False (default) it will just return chi

    OUTPUTS 
    ----------------
    chi: the chi squared of the model

    red_chi: the reduced chi squared

    red_chi_1sig: the 1 sigma error tolerance on the reduced chi squared. If the reduced chi squared falls within 1 +/- red_chi_1sig, it's considered a good model
    """
    if not isinstance(y_m, np.ndarray):
        y_m = np.array(y_m)

    if not isinstance(y, np.ndarray):
        y = np.array(y)

    if not isinstance(yerr, np.ndarray):
        yerr = np.array(yerr)

    chi = np.sum( ((y - y_m )**2) / (yerr**2))
    
    if reduced_chi == True:
        N = len(y) # the number of datapoints
        N_M = N-M # (N - M) the degrees of freedom
        red_chi = chi / (N_M)
        red_chi_1sig = np.sqrt(2/N_M) # red_chi is a good chisq if it falls within (1 +/- red_chi_1sig)
        
        return chi, red_chi, red_chi_1sig
    
    else:
        return chi



##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################


def L_rf_subplot(df_list, ANT_bands_list, band_c_dict, band_marker_dict, trans_names, ANT_xlims):
    fig, axs = plt.subplots(3, 4, figsize = (18, 7.5))
    axs = axs.ravel() # this converts axs from a 2D array into a 1D one to easily iterate over

    leg_handles = []
    leg_labels = []
    for i, ax in enumerate(axs):
        band_count = 0
        #print()
        if i==11: # we only have 11 ANTs from Phil, so turn off the bottom right axes + use this for the legend
            ax.axis('Off')
            break

        lc_df = df_list[i]
        ANT_name = transient_names[i]
        ANT_xlim = ANT_xlims[ANT_name]

        
        for band in ANT_bands_list[i]: # loop through the bands which are present for the ANT (provided that we have the ZP and central wl for this band....)
            band_color = band_c_dict[band]
            band_marker = band_marker_dict[band]
            band_data = lc_df[lc_df['band'] == band].copy() # band dataframe (for the bands which i have ZP and centrla wavelengths for)
            if len(band_data['MJD']) == 0:
                continue
            else:
                band_count+=1

            #if ANT_name == 'ZTF20abrbeie':
            #    print('BARBIE BAND', band)

            # add x errors to the plot
            if 'MJD_lower_err' in band_data.columns and 'MJD_upper_err' in band_data.columns:
                xerr = [band_data['MJD_lower_err'], band_data['MJD_upper_err']]

            elif 'MJD_err' in band_data.columns:
                xerr = band_data['MJD_err']

            else:
                xerr = [0]*len(band_data['MJD'])
                
            # plotting
            h = ax.errorbar(band_data['MJD'], band_data['L_rf'], yerr = band_data['L_rf_err'], xerr = xerr, 
                            fmt = band_marker, c = band_color, linestyle = 'None', markeredgecolor = 'k', 
                            markeredgewidth = '0.5', label = band)
            
            # making the handles and labels for the legend
            handle = h[0]
            label = band
            if label not in leg_labels:
                leg_labels.append(label)
                leg_handles.append(handle)
            
            sorted_handles_labels = sorted(zip(leg_handles, leg_labels), key = lambda tp: tp[0].get_marker())
            sorted_handles, sorted_labels = zip(*sorted_handles_labels)

        ax.grid(True)
        ax.set_title(f'{trans_names[i]}, B = {band_count}', fontweight = 'bold', fontsize = 9)
        ax.set_xlim(ANT_xlim)

        if ANT_name =='ZTF20abodaps':
            ax.set_ylim(-0.2e41, 1e42)
        
    fig.legend(sorted_handles, sorted_labels, loc = 'lower right', 
               bbox_to_anchor = (0.97, 0.0), ncol = 2, fontsize = 8)
    plt.suptitle('Rest frame luminosity for Phils ANTS, only for the bands for which I have ZP and cent wl for \n THESE BANDS IN THE FIGURE ARENT COMMPARABLE BECAUSE THEYVE BEEN CONVERTED TO REST FRAME CENTRAL WAVELENGTH WHICH IS REDSHIFT DEPENDENT SO VARIES BETWEEN ANTS', fontweight = 'bold', fontsize = 9, y = 0.99)
    fig.supxlabel('MJD', fontweight = 'bold', fontsize = 9)
    fig.supylabel('Rest frame luminosity / ergs/s/Angstrom', fontweight = 'bold',  fontsize = 9, x = 0.01)
    fig.subplots_adjust(top=0.925,
                        bottom=0.07,
                        left=0.04,
                        right=0.985,
                        hspace=0.23,
                        wspace=0.16)
    #plt.show()
    
    return



##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
"""

NEED TO CHECK THAT THE MEAN WAVELENGTHS ARE THE SPECIFIED ONES INSTEAD OF CALCULATED ONES WHEN SPECIFIED IS GIVEN

"""
# I USED THE SLOAN FULL TRANSIMSSION FILERS RATHER THAN THE u', g', r', i', z', PRIMED FILTERS
SLOAN_u_ZP  = 8.60588e-9
SLOAN_g_ZP = 4.92255e-9 
SLOAN_r_ZP = 2.85425e-9	
SLOAN_i_ZP = 1.94038e-9
SLOAN_z_ZP = 1.35994e-9

SLOAN_u_cw  = 3572.18	
SLOAN_g_cw = 4750.82
SLOAN_r_cw = 6204.29
SLOAN_i_cw = 7519.27
SLOAN_z_cw = 8992.26

band_ZP_key =  {'ATLAS_c': 3.89323e-9, 
                'ATLAS_o': 2.38902e-9, 
                'PS_i': 1.91728e-9, 
                'PS_w': 2.75493e-9, 
                'PS_y': 1.17434e-9, 
                'PS_z': 1.44673e-9, 
                'UVOT_B': 5.75381e-9, # FILTER + FILTER EFFECTIVE AREA - THE FILTER DEGRADES OVER TIME 
                'UVOT_U': 9.05581e-9, # FOR ALL UVOT FILTERS. THERES A WEBSITE THAT TELLS YOU ABOUT THIS 
                'UVOT_UVM1': 0.0, # DEGRADATION
                'UVOT_UVM2': 2.15706e-8, 
                'UVOT_UVW2': 2.57862e-8, 
                'UVOT_V': 3.69824e-9, 
                'WISE_W1': 9.59502e-11, 
                'WISE_W2': 5.10454e-11, 
                'ZTF_g': 4.75724e-9, 
                'ZTF_r': 2.64344e-9, 
                'ASAS-SN_V': 0.0, 
                'ASAS-SN_g': 0.0, 
                'B': 0.0, 
                'CSS_V': 0.0, 
                'Gaia_G': 2.78534e-9, # GAIA_G PRE RELEASE - THERE'S 2 OTHER OPTIONS
                'H': 0.0, 
                'I': 0.0, 
                'J': 0.0, 
                'LCOGT_B': 0.0, # ASSUME THESE ARE THE SAME VALUES AS THE STANDARD - USE BESSEL UBVRI FRMOM WHAT THE FLUX
                'LCOGT_V': 0.0, 
                'LCOGT_g': SLOAN_g_ZP, 
                'LCOGT_i': SLOAN_i_ZP, 
                'LCOGT_r': SLOAN_r_ZP, 
                'R': 0.0, 
                'SMARTS_B': 0.0,
                'SMARTS_V': 0.0, 
                'Swift_1': 0.0, 
                'Swift_2': 0.0, 
                'Swift_B': 0.0, # SAME AS UVOT_B? YES 
                'Swift_U': 0.0, # SAME AS UVOT_U?
                'Swift_V': 0.0, # SAME AS UVOT_V?
                'Swope_B': 0.0, 
                'Swope_V': 0.0, 
                'Swope_g': SLOAN_g_ZP, #  A PAPER SHOULD EXIST TO SAY THE EXACT VALUES, BUT FOR NOW FOR ugrizy ASSUME SDSS - FOR CAPITAL LETTERS ASSUME BESSEL
                'Swope_i': SLOAN_i_ZP, 
                'Swope_r': SLOAN_r_ZP, 
                'Swope_u': SLOAN_u_ZP, 
                'V': 0.0, 
                'g': SLOAN_g_ZP, # ASSUME THESE ARE THE SAME AS THE STANDARD VALUES - STANDARD WOULD EB CONSIDERED
                'i': SLOAN_i_ZP, 
                'r': SLOAN_r_ZP, 
                'U': 0.0, 
                'UVM2': 0.0, 
                'UVOT_UVW1': 1.6344e-8}







band_obs_centwl_dict = {'ATLAS_c': 5408.66, 
                        'ATLAS_o': 6866.26, 
                        'PS_i': 7563.76, 
                        'PS_w': 6579.22, 
                        'PS_y': 9644.63, 
                        'PS_z': 8690.10, 
                        'UVOT_B': 4377.97, 
                        'UVOT_U': 3492.67,
                        'UVOT_UVM1': 0.0, #---
                        'UVOT_UVM2': 2272.71, 
                        'UVOT_UVW2': 2140.26, 
                        'UVOT_V': 5439.64, 
                        'WISE_W1': 33526, #specified
                        'WISE_W2': 46028, # specified
                        'ZTF_g': 4829.50, 
                        'ZTF_r': 6463.75	, 
                        'ASAS-SN_V': 0.0, #---
                        'ASAS-SN_g': 0.0, #---
                        'B': 0.0, #---
                        'CSS_V': 0.0, #---
                        'Gaia_G': 6735.41, # GAIA_G PRE RELEASE - THERE'S 2 OTHER OPTIONS
                        'H': 0.0, #---
                        'I': 0.0, #---
                        'J': 0.0, #---
                        'LCOGT_B': 0.0, #---
                        'LCOGT_V': 0.0, #---
                        'LCOGT_g': SLOAN_g_cw, #---
                        'LCOGT_i': SLOAN_i_cw, #---
                        'LCOGT_r': SLOAN_r_cw, #---
                        'R': 0.0, #---
                        'SMARTS_B': 0.0,#---
                        'SMARTS_V': 0.0, #---
                        'Swift_1': 0.0, #---
                        'Swift_2': 0.0, #---
                        'Swift_B': 0.0, # same as UVOT_B?
                        'Swift_U': 0.0, # same as UVOT_U?
                        'Swift_V': 0.0, # same as UVOT_V?
                        'Swope_B': 0.0, #---
                        'Swope_V': 0.0, #---
                        'Swope_g': SLOAN_g_cw, #---
                        'Swope_i': SLOAN_i_cw, #---
                        'Swope_r': SLOAN_r_cw, #---
                        'Swope_u': SLOAN_u_cw, #---
                        'V': 0.0, #---
                        'g': SLOAN_g_cw, #---
                        'i': SLOAN_i_cw, #---
                        'r': SLOAN_r_cw, #---
                        'U': 0.0, #---
                        'UVM2': 0.0, #---
                        'UVOT_UVW1': 2688.46}


list_all_obs_cent_wl = list(band_obs_centwl_dict.values()) # all values in band_obs_centwl_dict
list_obs_cent_wl = [wl for wl in list_all_obs_cent_wl if wl > 0.0] # only the observed central wavelengths for teh bands for which I actually have data (if I don't have data on them I've set their values to 0.0)
list_bands_em_cent_wl = [b for idx, b in enumerate(list(band_obs_centwl_dict.keys())) if list_all_obs_cent_wl[idx] > 0.0] # the keys for the emitted band central wavelength dictionary - this is the same for all
#                                                                                                                           ANTs as it's just determined by whether of not I have data on this band or not, whereas 
#                                                                                                                           the value of the band's central wavelength is ANT dependent as it depends on the redshift

red = 'r'
orange = '#fc6b03'
yellow = '#fcc603'
bright_green = '#73fc03'
dark_green = '#238511'
bright_light_blue = '#02f5dd'
mid_blue = '#0298f5'
deep_blue = '#020ef5'
purple = '#8402f5'
pink = '#f502dd'
light_pink = '#e37bd9'
light_green = '#84e385'
light_purple = '#ad84e3'
grey = '#8c8c8c'
brown = '#5c3b2e'

band_colour_key =  {'ATLAS_c': red, 
                    'ATLAS_o': orange, 
                    'PS_i': yellow, 
                    'PS_w': bright_green, 
                    'PS_y': dark_green, 
                    'PS_z': bright_light_blue, 
                    'UVOT_B': mid_blue, # FILTER + FILTER EFFECTIVE AREA - THE FILTER DEGRADES OVER TIME 
                    'UVOT_U': deep_blue, # FOR ALL UVOT FILTERS. THERES A WEBSITE THAT TELLS YOU ABOUT THIS 
                    'UVOT_UVM1': purple, # DEGRADATION
                    'UVOT_UVM2': pink, 
                    'UVOT_UVW2': light_pink, 
                    'UVOT_V': light_green, 
                    'WISE_W1': light_purple, 
                    'WISE_W2': grey, 
                    'ZTF_g': brown, 
                    'ZTF_r': red, 
                    'ASAS-SN_V': orange, 
                    'ASAS-SN_g': yellow, 
                    'B': bright_green, 
                    'CSS_V': dark_green, 
                    'Gaia_G': bright_light_blue, # GAIA_G PRE RELEASE - THERE'S 2 OTHER OPTIONS
                    'H': mid_blue, 
                    'I': deep_blue, 
                    'J': purple, 
                    'LCOGT_B': pink, 
                    'LCOGT_V': light_pink, 
                    'LCOGT_g': light_green, 
                    'LCOGT_i': light_purple, 
                    'LCOGT_r': grey, 
                    'R': brown, 
                    'SMARTS_B': red,
                    'SMARTS_V': orange, 
                    'Swift_1': yellow, 
                    'Swift_2': bright_green, 
                    'Swift_B': dark_green, # SAME AS UVOT_B?
                    'Swift_U': bright_light_blue, # SAME AS UVOT_U?
                    'Swift_V': mid_blue, # SAME AS UVOT_V?
                    'Swope_B': deep_blue, 
                    'Swope_V': purple, 
                    'Swope_g': pink, 
                    'Swope_i': light_pink, 
                    'Swope_r': grey, 
                    'Swope_u': brown, 
                    'V': red, 
                    'g': orange, 
                    'i': yellow, 
                    'r': bright_green, 
                    'U': dark_green, 
                    'UVM2': bright_light_blue, 
                    'UVOT_UVW1': mid_blue}




band_marker_key =  {'ATLAS_c': 'o', 
                    'ATLAS_o': 'o', 
                    'PS_i': 'o',  
                    'PS_w': 'o', 
                    'PS_y': 'o',  
                    'PS_z': 'o',  
                    'UVOT_B': 'o',  # FILTER + FILTER EFFECTIVE AREA - THE FILTER DEGRADES OVER TIME 
                    'UVOT_U': 'o',  # FOR ALL UVOT FILTERS. THERES A WEBSITE THAT TELLS YOU ABOUT THIS 
                    'UVOT_UVM1': 'o',  # DEGRADATION
                    'UVOT_UVM2': 'o',  
                    'UVOT_UVW2': 'o', 
                    'UVOT_V': 'o',  
                    'WISE_W1': 'o', 
                    'WISE_W2': 'o', 
                    'ZTF_g': 'o', 
                    'ZTF_r': 's',
                    'ASAS-SN_V': 's',
                    'ASAS-SN_g': 's',
                    'B': 's',
                    'CSS_V': 's',
                    'Gaia_G': 's', # GAIA_G PRE RELEASE - THERE'S 2 OTHER OPTIONS
                    'H': 's', 
                    'I': 's',
                    'J': 's',
                    'LCOGT_B': 's',
                    'LCOGT_V': 's',
                    'LCOGT_g': 's',
                    'LCOGT_i': 's',
                    'LCOGT_r': 's',
                    'R': 's',
                    'SMARTS_B': '^',
                    'SMARTS_V': '^',
                    'Swift_1': '^', 
                    'Swift_2': '^', 
                    'Swift_B': '^', # SAME AS UVOT_B?
                    'Swift_U': '^',  # SAME AS UVOT_U?
                    'Swift_V': '^', # SAME AS UVOT_V?
                    'Swope_B': '^',
                    'Swope_V': '^',
                    'Swope_g': '^', 
                    'Swope_i': '^',
                    'Swope_r': '^',
                    'Swope_u': '^',
                    'V': '*', 
                    'g': '*', 
                    'i': '*', 
                    'r': '*', 
                    'U': '*',  
                    'UVM2': '*', 
                    'UVOT_UVW1': '*'}





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
                     'ZTF22aadesap': 0.073} # this one was harder to find, its still in Phil 2024 paper of 10 ANTs



ANT_xlim_dict = {'ZTF18aczpgwm': (58350, 60250),
                'ZTF19aailpwl': (58450, 60050), 
                'ZTF19aamrjar': (58340, 60250),
                'ZTF19aatubsj': (58550, 60250),
                'ZTF20aanxcpf': (59160, 60320),
                'ZTF20abgxlut': (58990, 59500), # this one was also hard to find, also in Wiseman(2024) but in footnote 24's link
                'ZTF20abodaps': (58860, 60300), 
                'ZTF20abrbeie': (59000, 59950),
                'ZTF20acvfraq': (59060, 60200),
                'ZTF21abxowzx': (59400, 60250),
                'ZTF22aadesap': (59650, 60400)} # this one was harder to find, its still in Phil 2024 paper of 10 ANTs



##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################

# loading in the files
folder_path = "C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS Data/modified Phil's lightcurves" # folder path containing Phil's light curve data files
lc_df_list = []
transient_names = []
list_of_bands = []
for file in os.listdir(folder_path): # file is a string of the file name such as 'file_name.dat'
    file_path = os.path.join(folder_path, file) 
    file_df = pd.read_csv(file_path, delimiter = ',') # the lightcurve data in a dataframe
    lc_df_list.append(file_df)

    trans_name = file[:-7] # the name of the transient
    transient_names.append(trans_name)

    trans_bands = file_df['band'].unique() # the bands in which this transient has data for
    list_of_bands.append(trans_bands)



#  I DONT HAVE ANY DATA ON THE UVOT_UVM1 BAND SO I CAN'T USE THAT HERE - THIS MIGHT BE UV ANYWAYS SO NOT THE END OF THE WORLD


# optical light between 3800 to 7500 Angstroms
MIR_cutoff = 15000


restframe_L_plot_dfs = []
rf_no_MIR_dfs = [] # a list of the ANT lcs which have been filtered, and only contains their optical data
for i, lc_df in enumerate(lc_df_list): # iterate though ANTs
    ANT_name = transient_names[i]
    ANT_z = ANT_redshift_dict[ANT_name] #redshift
    ANT_d = fcdm.luminosity_distance(ANT_z).to(u.cm).value # luminosity distance in cm


    # make a dictionary for the band's central wavelength converted into the emitted frame
    list_em_cent_wl = [obs_wl_to_em_wl(wl, ANT_z) for wl in list_obs_cent_wl] # the emitted central wavelengths of the bands for which I have data
    band_em_centwl_dict = dict(zip(list_bands_em_cent_wl, list_em_cent_wl))
    print(ANT_name)
    print(f'z = {ANT_z}')
    print()
    #for key, value in band_em_centwl_dict.items():
    #    if value > MIR_cutoff:
    #        MIR = '------MIR------'
    #
    #    else:
    #        MIR = ''
    #    print(f'{key}   {value} {MIR}')
    #    
    #print()
    

    # ONCE MY BAND DATA IS COMPLETE THIS SHOULDNT BE NECESSARY -----------------
    # filtering the ANT light curve dataframe based on whether or not I have the ZP and central wavelength for this band
    lc_filtered = lc_df[lc_df['band'].isin(list_bands_em_cent_wl)].copy()
    ANT_filtered_bands = lc_filtered['band'].unique()
    lc_filtered['em_cent_wl'] = lc_filtered['band'].map(band_em_centwl_dict) # this maps the band to its corresponding value within the emitted central wavelength dictionary
    lc_filtered['band_ZP'] = lc_filtered['band'].map(band_ZP_key)
    
    #for band, group in lc_filtered.groupby('band'):
    #    c_wl = group['em_cent_wl'].iloc[0]
    #
    #    if c_wl < MIR_cutoff:
    #        MIR = ''
    #
    #    else:
    #        MIR = '-------MIR-------'
    #    print(f"{band}    {group['em_cent_wl'].iloc[0]}    {MIR}")
    #
    #print('no bands = ', len(lc_filtered['band'].unique()))
    #print()

    # calculate the rest frame luminosity of the bands for which I have data
    restframe_L = []
    restframe_L_err = []
    for j in range(len(lc_filtered['MJD'])):
        dp_ZP = lc_filtered['band_ZP'].iloc[j] # the datapoint's zeropoint
        dp_mag = lc_filtered['mag'].iloc[j] # the datapoint's mag
        dp_magerr = lc_filtered['magerr'].iloc[j] # the datapoints magerr

        L_rest, L_rest_err = restframe_luminosity(ANT_d, dp_ZP, ANT_z, dp_mag, dp_magerr) # the datapoint's rest frame luminosity in ergs/s/Angstrom
        restframe_L.append(L_rest)
        restframe_L_err.append(L_rest_err)

    lc_filtered['L_rf'] = restframe_L
    lc_filtered['L_rf_err'] = restframe_L_err
    restframe_L_plot_dfs.append(lc_filtered) # use this list of dataframes to plot the rf luminosity subplot

    # optical light between 3800 to 7500 Angstroms, BUT pHIL SAID TO JUST GET OF TEH MID INFARRED SINCE THIS IS THE PART THAT PROBABLY COMES FROM OUTSIDE THE ANT
    lc_no_MIR = lc_filtered[lc_filtered['em_cent_wl'] < MIR_cutoff].copy()
    rf_no_MIR_dfs.append(lc_no_MIR)



"""     for band, group in lc_no_MIR.groupby('band'):
        print(f"{band}    {group['em_cent_wl'].iloc[0]}")

    print('no of bands = ', len(lc_no_MIR['band'].unique()))
    print('========================================')
    print() """


    

L_rf_subplot(restframe_L_plot_dfs, list_of_bands, band_colour_key, band_marker_key, transient_names, ANT_xlim_dict)
L_rf_subplot(rf_no_MIR_dfs, list_of_bands, band_colour_key, band_marker_key, transient_names, ANT_xlim_dict)
plt.show()

check_L_rf_errs = False
if check_L_rf_errs == True:
    # AFTER LOOKING AT THIS, I THINK MY ERRORS ON RESTFRAME LUMINOSITY ARE ACTUALLY FINE - PHIL SAID THAT THE ONE SORT OF RULE WE CAN USE IS THAT IF THERE'S A 0.1 MAGERR
    # THAT CORRESPONDS TO A ~10% ERROR IN LINEAR SPACE. THIS PLOT IS IN AGREEMENT WITH THAT 
    plt.figure(figsize = (16, 7.5))
    plt.scatter(restframe_L_plot_dfs[0]['magerr'], (restframe_L_plot_dfs[0]['L_rf_err']/restframe_L_plot_dfs[0]['L_rf']))
    plt.xlabel('magerr')
    plt.ylabel('L_rf_err/L_rf ')
    plt.grid()
    plt.title(f'Checkimg my rest frame luminosity errors - {transient_names[0]}')
    plt.show()















