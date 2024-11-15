import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import scipy.optimize as opt
from astropy import constants as const
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM 
import sys
sys.path.append("C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS") # this allows us to access the plotting_preferences.py file 
from November.plotting_preferences import band_colour_dict, band_marker_dict, band_offset_dict, band_ZP_dict, band_obs_centwl_dict, ANT_redshift_dict, band_offset_label_dict
from November.load_data_function import load_ANT_data

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

    # data frame for the binned band data  - just adds a column of MJD_bin to the data, then we can group by all datapoints in the same MJD bin
    lc_df['MJD_bin'] = pd.cut(lc_df['MJD'], MJD_bins)
    
    # binning the data by MJD_bin
    lc_binned_df = lc_df.groupby('MJD_bin', observed = drop_na_bins).apply(lambda g: pd.Series({
                                                        'no_bands': len(g['band'].unique()),
                                                        'mean_MJD': g['MJD'].mean(), 
                                                        'wm_mag': weighted_mean(g['mag'], g['magerr'])[0]
                                                        })).reset_index()
    

    print(type(lc_binned_df['MJD_bin'].iloc[0]))
    bin_mean_MJD_list = []
    bin_lhs_list = []
    bin_rhs_list = []
    for i in range(len(lc_binned_df['MJD_bin'])):
        bin = lc_binned_df['MJD_bin'].iloc[i]
        bin_lhs = bin.left
        bin_rhs = bin.right
        bin_lhs_list.append(bin_lhs)
        bin_rhs_list.append(bin_rhs)

        bin_mean_MJD = (bin_lhs + bin_rhs)/2
        bin_mean_MJD_list.append(bin_mean_MJD)

    lc_binned_df['bin_central_MJD'] = bin_mean_MJD_list
    lc_binned_df['bin_lhs'] = bin_lhs_list
    lc_binned_df['bin_rhs'] = bin_rhs_list
    

    print('LC_DF!!')
    print(lc_binned_df)

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

    #print(f'c = {c_cgs:.6e}')
    #print(f'h = {h_cgs:.6e}')
    #print(f'k = {k_cgs:.6e}')

    C = 8 * (np.pi**2) * h_cgs * (c_cgs**2) * 1e-8 # the constant coefficient of the equation 
    denom = np.exp((h_cgs * c_cgs) / (lam_cm * k_cgs * T_K)) - 1

    #print(f'C = {C:.6e}')
    #print(f'denom = {denom:.6e}')
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
##################################################################################################################################################################
##################################################################################################################################################################
# loading in the files
lc_df_list, transient_names, list_of_bands = load_ANT_data()

# ANT and band data

#ANT = 'ZTF18aczpgwm'
#ANT = 'ZTF20abrbeie'
#ANT = 'ZTF20abodaps'
ANT = 'ZTF19aailpwl'
idx = transient_names.index(ANT) # also named AT2019kn
ANT_df = lc_df_list[idx].copy()
ANT_bands = list_of_bands[idx]
print(f'all bands for which we have data on {ANT}: ', ANT_bands)

ANT_z = ANT_redshift_dict[ANT] # redshift found from Phil's paper (the one with 10 objects) 
ANT_d = fcdm.luminosity_distance(ANT_z) # luminosity distance in Mpc using the assumed cosmological consts at the top of this file
ANT_d = ANT_d.to(u.cm).value # luminosity distance in cm



band_names = list(band_obs_centwl_dict.keys())
bands_em_central_wl = [obs_wl_to_em_wl(band_obs_centwl_dict[b], ANT_z) for b in band_names] # obs_wl_to_em_wl(band_obs_centwl_dict[band], gwm_z) for band in band_names
band_em_central_wl_dict = dict(zip(band_names, bands_em_central_wl)) # dictionary giving the central wavelength converted into the rest frame for each band in Angstrom






##################################################################################################################################################################
# rest frame luminosity

restframe_L = []
restframe_L_err = []
em_cent_wl = []
for i in range(len(ANT_df['MJD'])):
    dp_band = ANT_df['band'].iloc[i] # the datapoints band
    dp_band_zp = band_ZP_dict[dp_band] # the band's AB zeropoint
    em_cent = band_em_central_wl_dict[dp_band] # the band's emitted central wavelength
    em_cent_wl.append(em_cent)
    dp_mag = ANT_df['mag'].iloc[i] # the datapoints mag
    dp_magerr = ANT_df['magerr'].iloc[i] # the datapoint's magerr

    L_rest, L_rest_err = restframe_luminosity(ANT_d, dp_band_zp, ANT_z, dp_mag, dp_magerr)
    restframe_L.append(L_rest)
    restframe_L_err.append(L_rest_err)

ANT_df['L_rest'] = restframe_L
ANT_df['L_rest_err'] = restframe_L_err
ANT_df['em_cent_wl'] = em_cent_wl







##################################################################################################################################################################
# find an MJD bin which has data across many different bands to allow for a better BB fit
ANT_optical = ANT_df[ANT_df['band'] != 'WISE_W1'].copy()
ANT_optical = ANT_optical[ANT_optical['band'] != 'WISE_W2'].copy()
ANT_emitted_optical_bands = ANT_optical['band'].unique()

MJD_binsize = 5
ANT_opt_binned = MJD_bin_for_BB_fit(ANT_optical, MJD_binsize = MJD_binsize) # binned up the lightcurve into MJD bins


plt.figure(figsize = (16, 7.5))
for band in ANT_emitted_optical_bands:
    band_df = ANT_df[ANT_df['band'] == band].copy()
    band_colour = band_colour_dict[band]
    band_marker = band_marker_dict[band]

    plt.errorbar(band_df['t_since_peak'], band_df['L_rest'], yerr = band_df['L_rest_err'], fmt=band_marker, label = 'observed band = '+band, linestyle = 'None', markeredgecolor = 'k',
                  markeredgewidth = '0.5', color = band_colour)

plt.xlabel('time since peak/days')
plt.ylabel('Rest-frame luminosity / ergs/s/Angstrom')
plt.grid()
plt.legend()
plt.title(f'Rest frame luminosity vs time - {ANT}')
plt.show()





##################################################################################################################################################################
# FROM THIS, I CHOSE TO USE THE BIN 58595 - 58600 FOR THE BB FIT
bin_min_MJD = 58595.0
bin_max_MJD = 58600.0
ANT_BB_data = ANT_optical[ANT_optical['MJD'] >= bin_min_MJD].copy()
ANT_BB_data = ANT_BB_data[ANT_BB_data['MJD'] < bin_max_MJD].copy()


BB_data_binned = ANT_BB_data.groupby('band', observed = False).apply(lambda g: pd.Series({'wm_L_rf': weighted_mean(g['L_rest'], g['L_rest_err'])[0],
                                                                                        'wm_L_rf_err': weighted_mean(g['L_rest'], g['L_rest_err'])[1],
                                                                                        'em_cent_wl': g['em_cent_wl'].iloc[0], 
                                                                                        'band': g['band'].iloc[0]
                                                                                        }))
print('ANT BB DATA')
print(ANT_BB_data)
print()
print()
print('BB DATA BY BAND')
print(BB_data_binned)
print()


fig = plt.figure(figsize = (16, 7))
for band in ANT_emitted_optical_bands:
    band_df = ANT_df[ANT_df['band'] == band].copy()
    band_colour = band_colour_dict[band]
    band_marker = band_marker_dict[band]
    band_offset = band_offset_dict[band]
    band_label = band_offset_label_dict[band]

    plt.errorbar(band_df['MJD'], (band_df['mag'] + band_offset) , yerr = band_df['magerr'], fmt=band_marker, label = band_label, linestyle = 'None', markeredgecolor = 'k',
                  markeredgewidth = '0.5', color = band_colour)
    
fig.gca().invert_yaxis()
for i in range(len(ANT_opt_binned['mean_MJD'])):
    plt.axvline(x = ANT_opt_binned['bin_lhs'].iloc[i], c = 'k')

plt.axvline(x = bin_min_MJD, c = 'r', label = 'bin chosen')
plt.axvline(x = bin_max_MJD, c = 'r')
plt.xlabel('MJD')
plt.ylabel('mag')
plt.title(f'{MJD_binsize} day bins')
plt.grid()
plt.legend()
plt.show()



##################################################################################################################################################################
# FITTING THE BLACKBODY CURVE....

# BRUTE FORCE METHOD OF FITTING THE BB CURVE
datapoints = 100
T_lb_index = 3
T_ub_index = 5 #7
R_lb_index = 14.5 #13
R_ub_index = 16.5#19
temp_range = np.logspace(T_lb_index, T_ub_index, datapoints) # Kelvin. blackbody temperatures to iterate though 
rad_range = np.logspace(R_lb_index, R_ub_index, datapoints) # cm. blackbody radii to iterate through

# scale down the T and R values to avoid errors with small and large numbers
T_scale_down = 1e0 # I don't really get how scaling down the temperature would work here, so I'll just scale down the radius since this is easier to reverse
R_scale_down = 1e-16
L_scale_down = (R_scale_down)**2 # this is because L ~ R^2, so scaling down R by 1e-16, scales down L by 1e-32

scaled_T_range = temp_range * T_scale_down # scaled temp and radius
scaled_R_range = rad_range * R_scale_down

BB_data_binned['scaled_wm_L_rf'] = BB_data_binned['wm_L_rf'] * L_scale_down
BB_data_binned['scaled_wm_L_rf_err'] = BB_data_binned['wm_L_rf_err'] * L_scale_down
BB_data_binned['em_cent_wl_cm'] = BB_data_binned['em_cent_wl'] * 1e-8 # get the bands emitted central wavelength from Angstrom --> cm


all_BB_data = BB_data_binned.copy()
BB_data_less_3000A = BB_data_binned[BB_data_binned['em_cent_wl'] < 3000].copy()
BB_data_more_3000A = BB_data_binned[BB_data_binned['em_cent_wl'] >= 3000].copy()




####################################################################################################################################################################




# iterating through different data sets to fit the BB curve to, since my data seems to fit well to when 
BB_fit_options = [all_BB_data, BB_data_more_3000A, BB_data_less_3000A]
plot_titles = ['All BB data used', 'Only data with emitted central wl > 3000A used', 'Only data with emitted central wl < 3000A used']

fig, axs = plt.subplots(2, 3, figsize = (16, 7))
for idx, BB_data in enumerate(BB_fit_options):
    # a 2D grid for the chi^2 values
    chi = np.zeros((len(scaled_T_range), len(scaled_R_range))) 
    real_ydata = BB_data['scaled_wm_L_rf']
    real_yerr = BB_data['scaled_wm_L_rf_err']


    for i, T_scaled_K in enumerate(scaled_T_range):
        for j, R_scaled_cm in enumerate(scaled_R_range):
            
            y_modeldata = []
            for wl_cm in BB_data['em_cent_wl_cm']:
                model_L_rf = blackbody(wl_cm, R_scaled_cm, T_scaled_K) # this should be restframe L*1e-32 since we scaled-down R
                y_modeldata.append(model_L_rf)

            chi_sq = chisq(y_modeldata, real_ydata, real_yerr, M=2)
            chi[i, j] = chi_sq/2 # reduced chi


    best_chi = np.min(chi)
    row, col = np.where(chi == best_chi)



    if len(row) == 1 and len(col) == 1:
        r = row[0]
        c = col[0]
        brtueforce_best_T = scaled_T_range[r]
        bruteforce_best_R = scaled_R_range[c]

    else:
        print('WARNING - MULTIPLE PARAMETER PAIRS GIVE THIS MIN CHI VALUE')

    descaled_bruteforce_best_T = brtueforce_best_T/T_scale_down
    descaled_bruteforce_best_R = bruteforce_best_R/R_scale_down
    print()
    print()
    print()
    print()
    print('BRUTE FORCE METHOD RESULT')
    print('---------------------------------------------')
    print(f'Best chi = {best_chi:.6e}')
    print(f'Best scaled T = {brtueforce_best_T:.6e} K      T scaling = {T_scale_down:.6f}     Best T = {descaled_bruteforce_best_T:.6f}     log (best T) = {np.log10(descaled_bruteforce_best_T):.6f}')
    print(f'Best scaled R = {bruteforce_best_R:.6e} cm     R scaling = {R_scale_down:.6f}     Best_R = {descaled_bruteforce_best_R:.6f}     log (best R) = {np.log10(descaled_bruteforce_best_R):.6f}')

    



    # create model data according to the optimal parameters from the bruteforce method
    model_wl_cm = np.linspace(1000, 15000, 300)*1e-8 # lambda range to model in cm
    bruteforce_modeldata = np.array([blackbody(x, bruteforce_best_R, brtueforce_best_T) for x in model_wl_cm]) # GIVES SCALED DOWN RESTFRAME LUMINOSITY
    bruteforce_modeldata = bruteforce_modeldata/L_scale_down # the modelled rest frame luminosity with the scaling removed


    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # CURVE FIT METHOD OF FITTING THE BB CURVE


    cf_lbound = np.array([(10**R_lb_index)*R_scale_down, (10**T_lb_index)*T_scale_down])
    cf_ubound = np.array([(10**R_ub_index)*R_scale_down, (10**T_ub_index)*T_scale_down])

    popt, pcov = opt.curve_fit(blackbody, xdata = BB_data['em_cent_wl_cm'], ydata = BB_data['scaled_wm_L_rf'], sigma = BB_data['scaled_wm_L_rf_err'], 
                            absolute_sigma = False, p0 = (1e15*R_scale_down, 5e3*T_scale_down), bounds = (cf_lbound, cf_ubound))
    cf_R = popt[0] # SCALED VALUES
    cf_T = popt[1]

    cf_R_err = np.sqrt(pcov[0, 0]) # SCALED ERRORS
    cf_T_err = np.sqrt(pcov[1, 1])

    cf_modely_forchi = np.array([blackbody(BB_data['em_cent_wl_cm'], cf_R, cf_T)]) # SCALED L VALUES
    cf_chi = chisq(cf_modely_forchi, BB_data['scaled_wm_L_rf'], BB_data['scaled_wm_L_rf_err'], M=2) # USING SCALED L VALUES
    cf_chi = cf_chi/2 # reduced chi
    cf_modeldata = np.array([blackbody(x, cf_R, cf_T) for x in model_wl_cm]) # SCALED VALUES
    cf_modeldata = cf_modeldata/L_scale_down # UN-SCALED VALUES FOR PLOTTING



    # PLOTTING  ####################################################################################################################################################################


    ax1 = axs[0, idx]
    ax1.set_title(plot_titles[idx], fontweight = 'bold', fontsize = 8)
    ax1.errorbar(BB_data_binned['em_cent_wl'], BB_data_binned['wm_L_rf'], yerr = BB_data_binned['wm_L_rf_err'], fmt = 'o', c = 'k', linestyle = 'None', 
                markeredgecolor = 'k', markeredgewidth = '0.5', label = 'All BB data available', markersize = 6)
    ax1.errorbar(BB_data['em_cent_wl'], BB_data['wm_L_rf'], yerr = BB_data['wm_L_rf_err'], fmt = 'o', c = 'magenta', linestyle = 'None', 
                markeredgecolor = 'k', markeredgewidth = '0.5', label = 'BB data used for these fits', markersize = 5)
    ax1.plot(model_wl_cm*1e8, bruteforce_modeldata, c = 'red', label = f'brute force BB model, \nbest chi = {best_chi:.3e}, \nlog(best T) = {np.log10(descaled_bruteforce_best_T):.2f}, \nlog(best_R) = {np.log10(descaled_bruteforce_best_R):.2f}')
    ax1.plot(model_wl_cm*1e8, cf_modeldata, c = 'blue', label = f'curve_fit BB model, \nbest chi = {cf_chi:.3e}, \nlog(best T) = {np.log10(cf_T/T_scale_down):.2f} +/- {np.log10(cf_T_err/T_scale_down):.2f}, \nlog(best_R) = {np.log10(cf_R/R_scale_down):.2f} +/- {np.log10(cf_R_err/R_scale_down):.2f}')

    ax1.grid()
    ax1.legend(fontsize = 8)
    ax1.set_xlabel('wavelength / Angstrom')
    ax1.set_ylabel('Rest frame luminosity')


    ax2 = axs[1, idx]
    max_plot_chi = 1e5
    cs = ax2.pcolormesh(temp_range, rad_range, chi.T, cmap = 'jet', vmin = 0, vmax = max_plot_chi)
    fig.colorbar(cs, ax = ax2)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Temperature / K')
    ax2.set_ylabel('Radius / cm')
    ax2.set_title('Contour plot of reduced chi squared values', fontweight = 'bold', fontsize = 8)


fig.subplots_adjust(top=0.88,
                    bottom=0.09,
                    left=0.05,
                    right=0.975,
                    hspace=0.38,
                    wspace=0.19)
fig.suptitle(f'BB fits of {ANT}, testing how using different parts of the BB data affects the fit produced', fontweight = 'bold')
savepath = "C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS/plots/BB fits/13_11_test_BB_fit_ZTF19aailpwl.png"
plt.savefig(savepath, dpi=150)
plt.show()

