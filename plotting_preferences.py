from astropy.cosmology import FlatLambdaCDM # THIS IS FOR THE LUMINOSITY DISTANCE DICTIONARY
import astropy.units as u
import pandas as pd

"""

NEED TO CHECK THAT THE MEAN WAVELENGTHS ARE THE SPECIFIED ONES INSTEAD OF CALCULATED ONES WHEN SPECIFIED IS GIVEN

"""


#####################################################################################################################################################################################
#####################################################################################################################################################################################
#####################################################################################################################################################################################
# ZEROPOINTS ########################################################################################################################################################################


# I USED THE SLOAN FULL TRANSIMSSION FILERS RATHER THAN THE u', g', r', i', z', PRIMED FILTERS
SLOAN_u_ZP  = 8.60588e-9
SLOAN_g_ZP = 4.92255e-9 
SLOAN_r_ZP = 2.85425e-9	
SLOAN_i_ZP = 1.94038e-9
SLOAN_z_ZP = 1.35994e-9


BESSEL_U_ZP = 8.47077e-9	
BESSEL_B_ZP = 5.69733e-9	
BESSEL_V_ZP = 3.62786e-9
BESSEL_R_ZP = 2.57796e-9
BESSEL_I_ZP = 1.69232e-9
BESSEL_J_ZP = 7.19139e-10
BESSEL_H_ZP = 4.04903e-10

BESSEL_U_ZP_VEGA = 3.96526e-9	
BESSEL_B_ZP_VEGA = 6.13268e-9
BESSEL_V_ZP_VEGA = 3.62708e-9	
BESSEL_R_ZP_VEGA = 2.17037e-9
BESSEL_I_ZP_VEGA = 1.12588e-9

BESSEL_J_ZP_VEGA = 3.12398e-10
BESSEL_H_ZP_VEGA = 1.13166e-10


band_ZP_dict =  {'ATLAS_c': 3.89323e-9, 
                'ATLAS_o': 2.38902e-9, 
                'PS_i': 1.91728e-9, 
                'PS_w': 2.75493e-9, 
                'PS_y': 1.17434e-9, 
                'PS_z': 1.44673e-9, 
                'UVOT_B': 5.75381e-9, # FILTER + FILTER EFFECTIVE AREA - THE FILTER DEGRADES OVER TIME 
                'UVOT_U': 9.05581e-9, # FOR ALL UVOT FILTERS. THERES A WEBSITE THAT TELLS YOU ABOUT THIS DEGRADATION
                'UVOT_UVM2': 2.15706e-8, 
                'UVOT_UVW2': 2.57862e-8, 
                'UVOT_V': 3.69824e-9, 
                'WISE_W1': 9.59502e-11, 
                'WISE_W2': 5.10454e-11, 
                'ZTF_g': 4.75724e-9, 
                'ZTF_r': 2.64344e-9, 
                'ASAS-SN_V': BESSEL_V_ZP, # https://www.zooniverse.org/projects/tharinduj/citizen-asas-sn/about/research Previously, we used Johnson V-band filters which are "green" colored, with an effective central wavelength of 551 nm, and a FWHM of 88 nm.
                'ASAS-SN_g': SLOAN_g_ZP, # https://www.zooniverse.org/projects/tharinduj/citizen-asas-sn/about/research ASAS-SN currently uses Sloan g-band filters which are "teal" colored, with an effective central wavelength of 480 nm, and a FWHM of 141 nm
                'B': BESSEL_B_ZP, 
                'CSS_V': BESSEL_V_ZP, 
                'Gaia_G': 2.78534e-9, # GAIA_G PRE RELEASE - THERE'S 2 OTHER OPTIONS
                'H': BESSEL_H_ZP, 
                'I': BESSEL_I_ZP, 
                'J': BESSEL_J_ZP, 
                'LCOGT_B': BESSEL_B_ZP, # ASSUME THESE ARE THE SAME VALUES AS THE STANDARD - USE BESSEL UBVRI FRMOM WHAT THE FLUX
                'LCOGT_V': BESSEL_V_ZP, 
                'LCOGT_g': SLOAN_g_ZP, 
                'LCOGT_i': SLOAN_i_ZP, 
                'LCOGT_r': SLOAN_r_ZP, 
                'R': BESSEL_R_ZP, 
                'SMARTS_B': BESSEL_B_ZP,
                'SMARTS_V': BESSEL_V_ZP, 
                'Swope_B': BESSEL_B_ZP, 
                'Swope_V': BESSEL_V_ZP, 
                'Swope_g': SLOAN_g_ZP, #  A PAPER SHOULD EXIST TO SAY THE EXACT VALUES, BUT FOR NOW FOR ugrizy ASSUME SDSS - FOR CAPITAL LETTERS ASSUME BESSEL
                'Swope_i': SLOAN_i_ZP, 
                'Swope_r': SLOAN_r_ZP, 
                'Swope_u': SLOAN_u_ZP, 
                'V': BESSEL_V_ZP, 
                'g': SLOAN_g_ZP, # ASSUME THESE ARE THE SAME AS THE STANDARD VALUES - STANDARD WOULD EB CONSIDERED
                'i': SLOAN_i_ZP, 
                'r': SLOAN_r_ZP, 
                'U': BESSEL_U_ZP, 
                'UVOT_UVW1': 1.6344e-8, 
                'U (Vega)': BESSEL_U_ZP_VEGA,
                'B (Vega)': BESSEL_B_ZP_VEGA,  
                'V (Vega)': BESSEL_V_ZP_VEGA, 
                'R (Vega)': BESSEL_R_ZP_VEGA, 
                'I (Vega)': BESSEL_I_ZP_VEGA, 
                'J (Vega)': BESSEL_J_ZP_VEGA,
                'H (Vega)': BESSEL_H_ZP_VEGA}



#####################################################################################################################################################################################
#####################################################################################################################################################################################
#####################################################################################################################################################################################
# OBSERVED CENTRAL WAVELENGTHS  #####################################################################################################################################################


SLOAN_u_cw  = 3572.18	
SLOAN_g_cw = 4750.82
SLOAN_r_cw = 6204.29
SLOAN_i_cw = 7519.27
SLOAN_z_cw = 8992.26


BESSEL_U_cw = 3605.07
BESSEL_B_cw = 4413.08
BESSEL_V_cw = 5512.10
BESSEL_R_cw = 6585.92
BESSEL_I_cw = 8059.88

BESSEL_J_cw = 12369.87	
BESSEL_H_cw = 16464.36	

# for the Vega bands, I think we need to use the effective wavelength from SVO2, whereas for AB bands, we use the mean wavelength from SVO2
BESSEL_U_VEGA_cw = 3659.88
BESSEL_B_VEGA_cw = 4380.74	
BESSEL_V_VEGA_cw = 5445.43
BESSEL_R_VEGA_cw = 6411.47
BESSEL_I_VEGA_cw = 7982.09
BESSEL_J_VEGA_cw = 12207.54
BESSEL_H_VEGA_cw = 16303.58

band_obs_centwl_dict = {'ATLAS_c': 5408.66, 
                        'ATLAS_o': 6866.26, 
                        'PS_i': 7563.76, 
                        'PS_w': 6579.22, 
                        'PS_y': 9644.63, 
                        'PS_z': 8690.10, 
                        'UVOT_B': 4377.97, 
                        'UVOT_U': 3492.67,
                        'UVOT_UVM2': 2272.71, 
                        'UVOT_UVW2': 2140.26, 
                        'UVOT_V': 5439.64, 
                        'WISE_W1': 33526, #specified
                        'WISE_W2': 46028, # specified
                        'ZTF_g': 4829.50, 
                        'ZTF_r': 6463.75, 
                        'ASAS-SN_V': BESSEL_V_cw, #---
                        'ASAS-SN_g': SLOAN_g_cw, #---
                        'B': BESSEL_B_cw, #---
                        'CSS_V': BESSEL_V_cw, #---
                        'Gaia_G': 6735.41, # GAIA_G PRE RELEASE - THERE'S 2 OTHER OPTIONS
                        'H': 0.0, #---
                        'I': BESSEL_I_cw, #---
                        'J': 0.0, #---
                        'LCOGT_B': BESSEL_B_cw, #---
                        'LCOGT_V': BESSEL_V_cw, #---
                        'LCOGT_g': SLOAN_g_cw, #---
                        'LCOGT_i': SLOAN_i_cw, #---
                        'LCOGT_r': SLOAN_r_cw, #---
                        'R': BESSEL_R_cw, #---
                        'SMARTS_B': BESSEL_B_cw,#---
                        'SMARTS_V': BESSEL_V_cw, #---
                        'Swope_B': BESSEL_B_cw, #---
                        'Swope_V': BESSEL_V_cw, #---
                        'Swope_g': SLOAN_g_cw, #---
                        'Swope_i': SLOAN_i_cw, #---
                        'Swope_r': SLOAN_r_cw, #---
                        'Swope_u': SLOAN_u_cw, #---
                        'V': BESSEL_V_cw, #---
                        'g': SLOAN_g_cw, #---
                        'i': SLOAN_i_cw, #---
                        'r': SLOAN_r_cw, #---
                        'U': BESSEL_U_cw, #---
                        'UVOT_UVW1': 2688.46, 
                        'U (Vega)': BESSEL_U_VEGA_cw,
                        'B (Vega)': BESSEL_B_VEGA_cw,  
                        'V (Vega)': BESSEL_V_VEGA_cw, 
                        'R (Vega)': BESSEL_R_VEGA_cw, 
                        'I (Vega)': BESSEL_I_VEGA_cw, 
                        'J (Vega)': BESSEL_J_VEGA_cw,
                        'H (Vega)': BESSEL_H_VEGA_cw}



#####################################################################################################################################################################################
#####################################################################################################################################################################################
#####################################################################################################################################################################################
# BAND MARKER COLOURS ###############################################################################################################################################################



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

band_colour_dict =  {'ATLAS_c': red, 
                    'ATLAS_o': orange, 
                    'PS_i': yellow, 
                    'PS_w': bright_green, 
                    'PS_y': dark_green, 
                    'PS_z': bright_light_blue, 
                    'UVOT_B': mid_blue, # FILTER + FILTER EFFECTIVE AREA - THE FILTER DEGRADES OVER TIME 
                    'UVOT_U': deep_blue, # FOR ALL UVOT FILTERS. THERES A WEBSITE THAT TELLS YOU ABOUT THIS DEGRADATION
                    'UVOT_UVM2': pink, 
                    'UVOT_UVW2': light_pink, 
                    'UVOT_V': light_green, 
                    'WISE_W1': light_purple, 
                    'WISE_W2': grey, 
                    'ZTF_g': brown, 
                    'ZTF_r': purple, 
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
                    'UVOT_UVW1': light_purple, 
                    'U (Vega)': deep_blue,
                    'B (Vega)': purple,  
                    'V (Vega)': pink, 
                    'R (Vega)': light_pink, 
                    'I (Vega)': light_green, 
                    'J (Vega)': light_purple,
                    'H (Vega)': grey}





#####################################################################################################################################################################################
#####################################################################################################################################################################################
#####################################################################################################################################################################################
# BAND MARKER TYPES ########################################################################################################################################################################



band_marker_dict =  {'ATLAS_c': 'o', 
                    'ATLAS_o': 'o', 
                    'PS_i': 'o',  
                    'PS_w': 'o', 
                    'PS_y': 'o',  
                    'PS_z': 'o',  
                    'UVOT_B': 'o',  # FILTER + FILTER EFFECTIVE AREA - THE FILTER DEGRADES OVER TIME 
                    'UVOT_U': 'o',  # FOR ALL UVOT FILTERS. THERES A WEBSITE THAT TELLS YOU ABOUT THIS DEGREDATION
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
                    'UVOT_UVW1': '*',
                    'U (Vega)': '*',
                    'B (Vega)': '*',  
                    'V (Vega)': '*', 
                    'R (Vega)': '*', 
                    'I (Vega)': '*', 
                    'J (Vega)': '*',
                    'H (Vega)': '*'}




#####################################################################################################################################################################################
#####################################################################################################################################################################################
#####################################################################################################################################################################################
# BAND OFFSETS IN MAG ################################################################################################################################################################




band_offset_dict = {'ATLAS_c': 1.5, 
                    'ATLAS_o':-2.5, 
                    'PS_i': -1.5, 
                    'PS_w': -1.0, 
                    'PS_y': 0.0, 
                    'PS_z': 0.0, 
                    'UVOT_B': 0.0, 
                    'UVOT_U': -2.5,
                    'UVOT_UVM2': -2.0, 
                    'UVOT_UVW2': 0.0, 
                    'UVOT_V': 0.0, 
                    'WISE_W1': 1.5, 
                    'WISE_W2': 1.5, 
                    'ZTF_g': 2.5, 
                    'ZTF_r': 0.5, 
                    'ASAS-SN_V': 0.0, 
                    'ASAS-SN_g': 0.0, 
                    'B': 2.0, 
                    'CSS_V': 0.0, 
                    'Gaia_G': 0.0, 
                    'H': 0.0, 
                    'I': -1.0, 
                    'J': 0.0, 
                    'LCOGT_B': 0.0, 
                    'LCOGT_V': 0.0, 
                    'LCOGT_g': 0.0, 
                    'LCOGT_i': 0.0, 
                    'LCOGT_r': 0.0, 
                    'R': 1.5, 
                    'SMARTS_B': 0.0,
                    'SMARTS_V': 0.0, 
                    'Swope_B': 0.0, 
                    'Swope_V': 0.0, 
                    'Swope_g': 0.0, 
                    'Swope_i': 0.0, 
                    'Swope_r': 0.0, 
                    'Swope_u': 0.0, 
                    'V': 0.5, 
                    'g': 1.0, 
                    'i': -0.5, 
                    'r': 0.0, 
                    'U': 0.0, 
                    'UVOT_UVW1': -1.5, 
                    'U (Vega)': 0.0,
                    'B (Vega)': 0.0,  
                    'V (Vega)': 0.0, 
                    'R (Vega)': 0.0, 
                    'I (Vega)': 0.0, 
                    'J (Vega)': 0.0,
                    'H (Vega)': 0.0}






#####################################################################################################################################################################################
#####################################################################################################################################################################################
#####################################################################################################################################################################################
# BAND OFFSET LABELS ################################################################################################################################################################


bands = list(band_offset_dict.keys())
offsets = list(band_offset_dict.values())


label_list = []
for i, ofs in enumerate(offsets):
    if ofs >= 0:
        label = f'{bands[i]} + {ofs}'

    else:
        label = f'{bands[i]} {ofs}'

    label_list.append(label)

    
band_offset_label_dict = dict(zip(bands, label_list))





#####################################################################################################################################################################################
#####################################################################################################################################################################################
#####################################################################################################################################################################################
# ANT PLOTTING MJD XLIMS ################################################################################################################################################################



MJD_xlims = {'ZTF18aczpgwm': (58350, 60250), 
            'ZTF19aailpwl': (58450, 60050), 
            'ZTF19aamrjar': (58340, 60250), 
            'ZTF19aatubsj': (58550, 60250), 
            'ZTF20aanxcpf': (58400, 60320), 
            'ZTF20abgxlut': (58950, 59500), 
            'ZTF20abodaps': (58450, 60250), 
            'ZTF20abrbeie': (59000, 59950), 
            'ZTF20acvfraq': (58700, 60200), 
            'ZTF21abxowzx': (59450, 60250), 
            'ZTF22aadesap': (59650, 60400), 
            'ASASSN-17jz': None, 
            'ASASSN-18jd': None, 
            'CSS100217': (54500, 56500), 
            'Gaia16aaw': None, 
            'Gaia18cdj': None, 
            'PS1-10adi': None, 
            'PS1-13jw': None} 





#####################################################################################################################################################################################
#####################################################################################################################################################################################
#####################################################################################################################################################################################
# ANT REDSHIFT DATA ################################################################################################################################################################




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
                     'PS1-10adi': 0.0, # 0.203 BUT ASSUME  = 0.0 SINCE WE HAVE MEASUREMENTS FOR ABS MAG SO d=10pc,  +/- 0.001 from Kankare, E(2017) from the Balmer lines of PS1-10adi, which is in agreement with the redshift of its host galaxy, z = 0.219 +/- 0.025
                     'PS1-13jw': 0.0 #0.345 BUT ASSUME  = 0.0 SINCE WE HAVE MEASUREMENTS FOR ABS MAG SO d=10pc from Kankare, E(2017) who got it from spectroscopic redshifts from SDSS
                     } 




#####################################################################################################################################################################################
#####################################################################################################################################################################################
#####################################################################################################################################################################################
# ANT LUMINOSITY DISTANCES ##########################################################################################################################################################





# ASSUMPTIONS FOR LUMINOSITY DISTANCE ARE BELOW ----------------------------------------------
H0 = 70 #km/s/Mpc
om_M = 0.3 # non relativistic matter density fraction
fcdm = FlatLambdaCDM(H0 = H0, Om0 = om_M)
# ASSUMPTIONS FOR LUMINOSITY DISTANCE ARE ABOVE ----------------------------------------------

ANT_d_cm_list = []
for i, z in enumerate(list(ANT_redshift_dict.values())):
    if i <= 15: 
        d = fcdm.luminosity_distance(z).to(u.cm).value # this gives the luminosity distance in cm

    else: # FOR PS1-10ADI AND PS1-13JW WE ARE GIVEN (VEGA) APPARENT MAG, SO z=~ 0.0 AND d = 10pc
        d = 3.086e19 # distance in cm,  1pc = 3.086e16 m, so 1pc = 3.086e18cm, so 10pc = 3.086e19 cm

    ANT_d_cm_list.append(d)

ANT_names = list(ANT_redshift_dict.keys())

ANT_luminosity_dist_cm_dict = dict(zip(ANT_names, ANT_d_cm_list))





#####################################################################################################################################################################################
#####################################################################################################################################################################################
#####################################################################################################################################################################################
# THE MJD VALUES TO TAKE A POLYFIT OVER    ##########################################################################################################################################################
# this is necessary because some ANTs have a few rouge datapoints veyr far away from peak and we don't want these to affect our polynomial fits



MJDs_for_fit = {'ZTF18aczpgwm': (58400, 59680), 
                'ZTF19aailpwl': (58420, 59500), 
                'ZTF19aamrjar': (58350, None), 
                'ZTF19aatubsj': (58360, None), 
                'ZTF20aanxcpf': (59300, 60200), 
                'ZTF20abgxlut': (58900, 59430), 
                'ZTF20abodaps': (58920, None), 
                'ZTF20abrbeie': (None, None), 
                'ZTF20acvfraq': (58700, None), 
                'ZTF21abxowzx': (59400, 60400), 
                'ZTF22aadesap': (59500, None), 
                'ASASSN-17jz': (None, None), 
                'ASASSN-18jd': (58200, 59000), 
                'CSS100217': (None, None), 
                'Gaia16aaw': (None, None), 
                'Gaia18cdj': (None, None), 
                'PS1-10adi': (None, None), 
                'PS1-13jw': (None, None)} 




#####################################################################################################################################################################################
#####################################################################################################################################################################################
#####################################################################################################################################################################################
# ANTS TO BOTHER POLYFITTING, INTERPOLATING AND SED FITTING    ######################################################################################################################
# This is necessary because a few ANTs only haev one band's worth of data, so don't bother polyfitting these since we can't SED fit them anyway




ANTs_for_fitting_dict = {'ZTF18aczpgwm': True,  # ONLY BOTHER POLYFITTING AND INTERPOLATING LIGHT CURVES WITH MORE THAN ONE DECENT BAND
                        'ZTF19aailpwl': True, 
                        'ZTF19aamrjar': True, 
                        'ZTF19aatubsj': True, 
                        'ZTF20aanxcpf': True, 
                        'ZTF20abgxlut': True, 
                        'ZTF20abodaps': True, 
                        'ZTF20abrbeie': True, 
                        'ZTF20acvfraq': True, 
                        'ZTF21abxowzx': True, 
                        'ZTF22aadesap': True, 
                        'ASASSN-17jz': True, 
                        'ASASSN-18jd': True, 
                        'CSS100217': False, 
                        'Gaia16aaw': False, 
                        'Gaia18cdj': False, 
                        'PS1-10adi': True, 
                        'PS1-13jw': False }






#####################################################################################################################################################################################
#####################################################################################################################################################################################
#####################################################################################################################################################################################
# WHETHER OR NOT WE'D LIKE TO OVERRIDE THE CHOICE OF REFERENCE BAND FOR THE POLYNOMIAL FIT INTERPOLATION    #########################################################################



override_ref_band_dict = {'ZTF18aczpgwm': None, 
                        'ZTF19aailpwl': None, 
                        'ZTF19aamrjar': None, 
                        'ZTF19aatubsj': None, 
                        'ZTF20aanxcpf': None, 
                        'ZTF20abgxlut': None, 
                        'ZTF20abodaps': 'ZTF_g', 
                        'ZTF20abrbeie': None, 
                        'ZTF20acvfraq': 'ZTF_r', 
                        'ZTF21abxowzx': None, 
                        'ZTF22aadesap': None, 
                        'ASASSN-17jz': None, 
                        'ASASSN-18jd': None, 
                        'CSS100217': None, 
                        'Gaia16aaw': None, 
                        'Gaia18cdj': None, 
                        'PS1-10adi': None, 
                        'PS1-13jw': None} 






#####################################################################################################################################################################################
#####################################################################################################################################################################################
#####################################################################################################################################################################################
# HUMAN INPUT OF STRAGGLER DATAPOINTS      ##########################################################################################################################################

aadesap_straggler = pd.DataFrame({'MJD': [59674.309], 'band': ['ATLAS_o']})
ASASSN17jz_straggler = pd.DataFrame({'MJD': [57900.92], 'band':['V']})

manual_stragglers_dict = {'ZTF18aczpgwm': None, 
                            'ZTF19aailpwl': None, 
                            'ZTF19aamrjar': None, 
                            'ZTF19aatubsj': None, 
                            'ZTF20aanxcpf': None, 
                            'ZTF20abgxlut': None, 
                            'ZTF20abodaps': None, 
                            'ZTF20abrbeie': None, 
                            'ZTF20acvfraq': None, 
                            'ZTF21abxowzx': None, 
                            'ZTF22aadesap': aadesap_straggler, # one here
                            'ASASSN-17jz': ASASSN17jz_straggler, #  one here
                            'ASASSN-18jd': None, 
                            'CSS100217': None, 
                            'Gaia16aaw': None, 
                            'Gaia18cdj': None, 
                            'PS1-10adi': None, 
                            'PS1-13jw': None} 