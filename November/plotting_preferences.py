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
                'ASAS-SN_V': BESSEL_V_ZP, 
                'ASAS-SN_g': SLOAN_g_ZP, 
                'B': BESSEL_B_ZP, 
                'CSS_V': BESSEL_V_ZP, 
                'Gaia_G': 2.78534e-9, # GAIA_G PRE RELEASE - THERE'S 2 OTHER OPTIONS
                'H': 0.0, 
                'I': 0.0, 
                'J': 0.0, 
                'LCOGT_B': BESSEL_B_ZP, # ASSUME THESE ARE THE SAME VALUES AS THE STANDARD - USE BESSEL UBVRI FRMOM WHAT THE FLUX
                'LCOGT_V': BESSEL_V_ZP, 
                'LCOGT_g': SLOAN_g_ZP, 
                'LCOGT_i': SLOAN_i_ZP, 
                'LCOGT_r': SLOAN_r_ZP, 
                'R': BESSEL_R_ZP, 
                'SMARTS_B': BESSEL_B_ZP,
                'SMARTS_V': BESSEL_V_ZP, 
                'Swift_1': 0.0, 
                'Swift_2': 0.0, 
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
                'UVOT_UVW1': 1.6344e-8}



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
                        'I': 0.0, #---
                        'J': 0.0, #---
                        'LCOGT_B': BESSEL_B_cw, #---
                        'LCOGT_V': BESSEL_V_cw, #---
                        'LCOGT_g': SLOAN_g_cw, #---
                        'LCOGT_i': SLOAN_i_cw, #---
                        'LCOGT_r': SLOAN_r_cw, #---
                        'R': BESSEL_R_cw, #---
                        'SMARTS_B': BESSEL_B_cw,#---
                        'SMARTS_V': BESSEL_V_cw, #---
                        'Swift_1': 0.0, #---
                        'Swift_2': 0.0, #---
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
                        'UVOT_UVW1': 2688.46}



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
                    'UVOT_UVW1': mid_blue}





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
                    'Swift_1': '^', 
                    'Swift_2': '^', 
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
                    'UVOT_UVW1': '*'}




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
                    'Swift_1': 0.0, 
                    'Swift_2': 0.0, 
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
                    'UVOT_UVW1': -1.5}






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
                     'ZTF22aadesap': 0.073} # this one was harder to find, its still in Phil 2024 paper of 10 ANTs




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
            'ZTF22aadesap': (59650, 60400)} 