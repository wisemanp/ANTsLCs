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


def obs_wl_to_em_wl(obs_wavelength, z):
    return obs_wavelength / (1+z)


##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################



def MJD_bin_for_BB_fit(lc_df, MJD_binsize, drop_na_bins = True):
    """
    Bins data for the light curve and finds the bin with the most data across different bands to allow to the best blackbody fit (a better approach would be 
    to fit a polynomial to the lightcurve so we could take data from the peak, but that's for later)
    """
    lc_df = lc_df.copy() # this creates a copy rather than a view of the dataframe so that we don't modify the original dataframe
    
    # this rounds the min_MJD present in V_band data for the transient to the nearest 10, then goes to the bin below this to ensure that 
    # we capture the first data point, in case this rounded the first bin up. Since we need data for both V_band and B_band within each
    # MJD bin, it's fine to base the min MJD bin to start at on just the V_band data, since even if we had B_band data before this date, 
    # we wouldn't be able to pair it with data from V_band to calculcte the color anyways
    MJD_bin_min = int( round(lc_df['MJD'].min(), -1) - 10 )
    MJD_bin_max = int( round(lc_df['MJD'].max(), -1) + 10 )
    MJD_bins = range(MJD_bin_min, MJD_bin_max + MJD_binsize, MJD_binsize) # create the bins

    # data frames for the binned band data 
    lc_df['MJD_bin'] = pd.cut(lc_df['MJD'], MJD_bins)

    lc_binned_df = lc_df.groupby('MJD_bin', observed = drop_na_bins).apply(lambda g: pd.Series({
                                                        'no_bands': len(g['band'].unique()),
                                                        'mean_MJD': g['MJD'].mean()
                                                        })).reset_index()
    
    



    return lc_binned_df



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
##################################################################################################################################################################



""" band_ZP_dict = {'PS_i': 1.91728e-9, # in ergs/s/cm^2/Angstrom
                'PS_w': 2.75493e-9, 
                'WISE_W1': 9.59502e-11, 
                'WISE_W2': 5.10454e-11, 
                'ZTF_g': 4.75724e-9, 
                'ZTF_r': 2.64344e-9, 
                'ATLAS_c': 3.89323e-9, 
                'ATLAS_o': 2.38902e-9} """



"""

NEED TO CHECK THAT THE MEAN WAVELENGTHS ARE THE SPECIFIED ONES INSTEAD OF CALCULATED ONES WHEN SPECIFIED IS GIVEN

"""
band_ZP_dict =  {'ATLAS_c': 3.89323e-9, 
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
                'LCOGT_B': 0.0, 
                'LCOGT_V': 0.0, 
                'LCOGT_g': 0.0, 
                'LCOGT_i': 0.0, 
                'LCOGT_r': 0.0, 
                'R': 0.0, 
                'SMARTS_B': 0.0,
                'SMARTS_V': 0.0, 
                'Swift_1': 0.0, 
                'Swift_2': 0.0, 
                'Swift_B': 0.0, # SAME AS UVOT_B?
                'Swift_U': 0.0, # SAME AS UVOT_U?
                'Swift_V': 0.0, # SAME AS UVOT_V?
                'Swope_B': 0.0, 
                'Swope_V': 0.0, 
                'Swope_g': 0.0, 
                'Swope_i': 0.0, 
                'Swope_r': 0.0, 
                'Swope_u': 0.0, 
                'V': 0.0, 
                'g': 0.0, 
                'i': 0.0, 
                'r': 0.0, 
                'U': 0.0, 
                'UVM2': 0.0, 
                'UVOT_UVW1': 1.6344e-8}


""" bands_obs_central_wl_dict = {'PS_i': 7563.76, # in Angstrom THIS IS ACTUALLY THE MEAN WAVELENGTH FROM SVO2
                            'PS_w': 6579.22, 
                            'WISE_W1': 33526.00, 
                            'WISE_W2': 46028.00, 
                            'ZTF_g': 4829.50, 
                            'ZTF_r': 6463.75, 
                            'ATLAS_c': 5408.66, 
                            'ATLAS_o': 6866.26} """



bands_obs_central_wl_dict = {'ATLAS_c': 5408.66, 
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
                        'LCOGT_g': 0.0, #---
                        'LCOGT_i': 0.0, #---
                        'LCOGT_r': 0.0, #---
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
                        'Swope_g': 0.0, #---
                        'Swope_i': 0.0, #---
                        'Swope_r': 0.0, #---
                        'Swope_u': 0.0, #---
                        'V': 0.0, #---
                        'g': 0.0, #---
                        'i': 0.0, #---
                        'r': 0.0, #---
                        'U': 0.0, #---
                        'UVM2': 0.0, #---
                        'UVOT_UVW1': 2688.46}




##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################
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






##################################################################################################################################################################
# ANT and band data

ANT = 'ZTF18aczpgwm'
#ANT = 'ZTF20abrbeie'
#ANT = 'ZTF20abodaps'
idx = transient_names.index(ANT) # also named AT2019kn
gwm = lc_df_list[idx].copy()
gwm_bands = list_of_bands[idx]
print(f'all bands for which we have data on {ANT}: ', gwm_bands)

gwm_z = 0.4279 # redshift found from Phil's paper (the one with 10 objects)
gwm_d = fcdm.luminosity_distance(gwm_z) # luminosity distance in Mpc using the assumed cosmological consts at the top of this file
gwm_d = gwm_d.to(u.cm).value # luminosity distance in cm



band_names = list(bands_obs_central_wl_dict.keys())
bands_em_central_wl = [obs_wl_to_em_wl(bands_obs_central_wl_dict[b], gwm_z) for b in band_names] # obs_wl_to_em_wl(bands_obs_central_wl_dict[band], gwm_z) for band in band_names
band_em_central_wl_dict = dict(zip(band_names, bands_em_central_wl)) # dictionary giving the central wavelength converted into the rest frame for each band in Angstrom






##################################################################################################################################################################
# rest frame luminosity

restframe_L = []
restframe_L_err = []
em_cent_wl = []
for i in range(len(gwm['MJD'])):
    dp_band = gwm['band'].iloc[i] # the datapoints band
    dp_band_zp = band_ZP_dict[dp_band] # the band's AB zeropoint
    em_cent = band_em_central_wl_dict[dp_band] # the band's emitted central wavelength
    em_cent_wl.append(em_cent)
    dp_mag = gwm['mag'].iloc[i] # the datapoints mag
    dp_magerr = gwm['magerr'].iloc[i] # the datapoint's magerr

    L_rest, L_rest_err = restframe_luminosity(gwm_d, dp_band_zp, gwm_z, dp_mag, dp_magerr)
    restframe_L.append(L_rest)
    restframe_L_err.append(L_rest_err)

gwm['L_rest'] = restframe_L
gwm['L_rest_err'] = restframe_L_err
gwm['em_cent_wl'] = em_cent_wl







##################################################################################################################################################################
# find an MJD bin which has data across many different bands to allow for a better BB fit
#gwm_optical = gwm[gwm['band'] != 'WISE_W1'].copy()
#gwm_optical = gwm_optical[gwm_optical['band'] != 'WISE_W2'].copy()
print()
print('ALL BANDS EMITTED CENTRAL WAVELENGTH - BETWEEN 3800-7500 IS OPTICAL')
for band, group in gwm.groupby('band'):
    print(f"{band}      {group['em_cent_wl'].iloc[0]}")

gwm_optical = gwm[(gwm['em_cent_wl'] > 3800) & (gwm['em_cent_wl'] < 7500)].copy()
gwm_emitted_optical_bands = gwm_optical['band'].unique()
print()
print('OPTICAL BANDS: ', gwm_emitted_optical_bands)
print()
print('=========================================================================================')
print()

gwm_opt_binned = MJD_bin_for_BB_fit(gwm_optical, MJD_binsize=5)
plt.figure(figsize=(16, 7.5))
plt.scatter(gwm_opt_binned['mean_MJD'], gwm_opt_binned['no_bands'])
plt.xlabel('mean MJD in bin')
plt.ylabel('no of bands within the bin')
plt.grid()
plt.show()
print(gwm_opt_binned[gwm_opt_binned['no_bands'] > 3.0])
# the bin chosen is (58735, 58740] - this is the closest bin to the peak which has 4 diff bands within it
bin_min_MJD = 58694.3#58694.3 #58734 #58694 #58735 MANUALLY CHOSEN MJD BIN OF 6 DAYS
bin_max_MJD = 58699.3#58699.4 #58740 #58700 #58740
opt_BB_bin = gwm_optical[gwm_optical['MJD'] >= bin_min_MJD].copy()
opt_BB_bin = opt_BB_bin[opt_BB_bin['MJD'] < bin_max_MJD].copy()
opt_BB_bin_binned = opt_BB_bin.groupby('band', observed = False).apply(lambda g: pd.Series({'wm_L_rf': weighted_mean(g['L_rest'], g['L_rest_err'])[0],
                                                                                    'wm_L_rf_err': weighted_mean(g['L_rest'], g['L_rest_err'])[1],
                                                                                    'em_cent_wl': g['em_cent_wl'].iloc[0], 
                                                                                    'band': g['band'].iloc[0]
                                                                                    }))
print()
print('CHOSEN MJD BIN FOR BLACKBODY FIT')
print(opt_BB_bin)
print()
print('BINNED_BB_BIN')
print(opt_BB_bin_binned)





plt.figure(figsize = (16, 7.5))
for band in gwm_emitted_optical_bands:
    band_df = gwm[gwm['band'] == band].copy()

    plt.errorbar(band_df['t_since_peak'], band_df['L_rest'], yerr = band_df['L_rest_err'], fmt='o', label = 'observed band = '+band, linestyle = 'None', markeredgecolor = 'k',
                  markeredgewidth = '0.5')

plt.xlabel('time since peak/days')
#plt.axvline(x = bin_min_MJD, label = f'MJD bin chosen by eye. Width = {bin_max_MJD - bin_min_MJD}')
#plt.axvline(x = bin_max_MJD)
#plt.axvline(x = 58735, c='r')
#plt.axvline(x = 58740, c='r', label = 'bin picked up by my code. Width = 5.0')
plt.ylabel('Rest-frame luminosity / ergs/s/Angstrom')
plt.grid()
plt.legend()
plt.title(f'Rest frame luminosity vs time - {ANT}')
plt.show()



""" 

##################################################################################################################################################################
# FITTING THE BLACKBODY CURVE....


datapoints = 100
T_lb_index = 3
T_ub_index = 5
R_lb_index = 14
R_ub_index = 18
temp_range = np.logspace(T_lb_index, T_ub_index, datapoints) # Kelvin. blackbody temperatures to iterate though 
rad_range = np.logspace(R_lb_index, R_ub_index, datapoints) # cm. blackbody radii to iterate through

chi = np.zeros((len(temp_range), len(rad_range))) # a 2D array for the chi^2

opt_BB_bin_binned['em_cent_wl_cm'] = opt_BB_bin_binned['em_cent_wl']*1e-8
ydata = opt_BB_bin_binned['wm_L_rf']
yerr = opt_BB_bin_binned['wm_L_rf_err']

for i, T_K in enumerate(temp_range):
    for j, R_cm in enumerate(rad_range):

        y_modeldata = []
        for wl_cm in opt_BB_bin_binned['em_cent_wl_cm']:
            y_model = blackbody(wl_cm, R_cm, T_K)
            y_modeldata.append(y_model)

        chi_sq = chisq(y_modeldata, ydata, yerr, 2)
        chi[i, j] = chi_sq


best_chi = np.min(chi)
row, col = np.where(chi == best_chi)

if len(row) == 1 and len(col) == 1:
    r = row[0]
    c = col[0]
    best_T = temp_range[r]
    best_R = rad_range[c]

else:
    print('WARNING - MULTIPLE PARAMETER PAIRS GIVE THIS MIN CHI VALUE')


model_wl_cm = np.linspace(1000, 15000, 300)*1e-8 # lambda in cm
modely = [blackbody(x, best_R, best_T) for x in model_wl_cm]

print()
print()
print(f'chi = {best_chi}')
print(f'T = {best_T:.6e} K')
print(f'R = {best_R:.6e} cm')
print()

cf_lbound = np.array([(10**R_lb_index), (10**T_lb_index)])
cf_ubound = np.array([(10**R_ub_index), (10**T_ub_index)])
popt, pcov = opt.curve_fit(blackbody, xdata = opt_BB_bin_binned['em_cent_wl_cm'], ydata = ydata, sigma = yerr, absolute_sigma = False, p0 = (1e15, 5e3), bounds = (cf_lbound, cf_ubound))
cf_T = popt[1]
cf_R = popt[0]
cf_T_err= np.sqrt(pcov[1, 1])
cf_R_err = np.sqrt(pcov[0, 0])
print('CURVEFIT RESULTS')
print(f'T = {cf_T:.6e}  + /- {cf_T_err:.6e}  K    (6s.f. for easier reading)')
print(f'R = {cf_R:.6e}  +/-  {cf_R_err:.6e}  cm   (6s.f. for easier reading)')

cf_modely = [blackbody(x, cf_R, cf_T) for x in model_wl_cm]
cf_modely_for_chi = [blackbody(wl, cf_R, cf_T) for wl in opt_BB_bin_binned['em_cent_wl_cm']]
cf_chi = chisq(cf_modely_for_chi, ydata, yerr = yerr, M = 2)
##################################################################################################################################################################
# plotting

plt.figure(figsize = (16, 7.5))
for band in gwm_bands:
    band_df = gwm[gwm['band'] == band].copy()

    plt.errorbar(band_df['MJD'], band_df['L_rest'], yerr = band_df['L_rest_err'], fmt='o', label = 'observed band = '+band, linestyle = 'None', markeredgecolor = 'k',
                  markeredgewidth = '0.5')

plt.xlabel('MJD')
plt.axvline(x = bin_min_MJD, label = f'MJD bin chosen by eye. Width = {bin_max_MJD - bin_min_MJD}')
plt.axvline(x = bin_max_MJD)
plt.axvline(x = 58735, c='r')
plt.axvline(x = 58740, c='r', label = 'bin picked up by my code. Width = 5.0')
plt.ylabel('Rest-frame luminosity / ergs/s/Angstrom')
plt.grid()
plt.legend()
plt.title(f'Rest frame luminosity vs time - {ANT}')
#plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------


fig = plt.figure(figsize = (16, 7.5))
ax1 = plt.subplot(1, 2, 1)
cmap = plt.get_cmap('jet')
max_plot_chi = 5e6
#cs = ax1.contourf(temp_range, rad_range, chi, cmap = cmap, vmin = 0, vmax = max_plot_chi, levels = np.linspace(0, max_plot_chi, 1000))
cs = ax1.pcolormesh(temp_range, rad_range, chi.T, cmap = cmap, vmin = 0, vmax = max_plot_chi)
fig.colorbar(cs, ax = ax1)
ax1.set_xlabel('blackbody Temp range / K')
ax1.set_ylabel('Blackbody Radius range / cm')
ax1.set_title('contour plot of chi squared for different blackbody temperatures and radii', y = 1.03)
ax1.set_xscale('log')
ax1.set_yscale('log')
#ax1.set_ylim((7.5e14, 4.2e16))
#ax1.set_xlim((5770, 1e5))


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------

ax2 = plt.subplot(1, 2, 2)
ax2.errorbar(opt_BB_bin_binned['em_cent_wl'], opt_BB_bin_binned['wm_L_rf'], yerr = opt_BB_bin_binned['wm_L_rf_err'], fmt='o', c = 'r', markeredgecolor = 'k', 
             markeredgewidth = '0.5', linestyle = 'None') 
ax2.plot(model_wl_cm*1e8, modely, c='b', label = f'best BB fit, chi = {best_chi:.2e}, \nT = {best_T:.2e} K, \n R = {best_R:.2e} cm \n', linestyle = '-')
ax2.plot(model_wl_cm*1e8, cf_modely, c = 'g', label = f'curve fit, chi = {cf_chi:.2e}\n T = {cf_T:.2e} + /- {cf_T_err:.2e} K,  \n R = {cf_R:.2e} +/- {cf_R_err:.2e} cm')
ax2.set_xlabel("Band's central wavelength in the rest-frame / Angstrom")
ax2.set_ylabel("Luminosity density / ergs/s/Angstrom")
ax2.set_title(f'{ANT} - MJD bin for data = ({bin_min_MJD}, {bin_max_MJD}] MANUALLY CHOSEN MJD BIN OF 6 DAYS')
ax2.legend()
ax2.grid()
fig.subplots_adjust(top=0.91,
                    bottom=0.085,
                    left=0.055,
                    right=0.96,
                    hspace=0.2,
                    wspace=0.135)
plt.show()



et = time.time()
print(f'RUN TIME = {et - st} s')
print()  """