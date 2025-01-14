import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.optimize as opt
from matplotlib import cm
from matplotlib.colors import Normalize
sys.path.append("C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS") # this allows us to import plotting preferences and functions
from plotting_preferences import band_colour_dict, band_ZP_dict, band_obs_centwl_dict, ANT_redshift_dict, ANT_luminosity_dist_cm_dict, MJDs_for_fit
from functions import load_ANT_data, ANT_data_L_rf, bin_lc, polyfit_lc, fit_BB_across_lc


# load in the data
lc_df_list, transient_names, list_of_bands = load_ANT_data()

# calculate the rest frame luminosity + emitted central wavelength
add_lc_df_list = ANT_data_L_rf(lc_df_list, transient_names, ANT_redshift_dict, ANT_luminosity_dist_cm_dict, band_ZP_dict, band_obs_centwl_dict)

# bin up the light curve into 1 day MJD bins
MJD_binsize = 1
binned_df_list = bin_lc(add_lc_df_list, MJD_binsize)

# polyfitting ONE light curve
idx = 0
ANT_name = transient_names[idx]
ANT_df = binned_df_list[idx]
ANT_bands = list_of_bands[idx]
reference_band = 'ZTF_g'
bands_for_BB = [b for b in ANT_bands if (b != 'WISE_W1') and (b != 'WISE_W2')] # remove the WISE bands from the interpolation since we don't want to use this data for the BB fit anyway



interp_lc, plot_polyfit_df = polyfit_lc(ANT_name, ANT_df, fit_order = 5, df_bands = bands_for_BB, trusted_band = reference_band, fit_MJD_range = MJDs_for_fit[ANT_name],
                        extrapolate = False, b_colour_dict = band_colour_dict, plot_polyfit = False)


BB_curvefit = True
BB_brute = False
BB_fit_results = fit_BB_across_lc(interp_lc, brute = BB_brute, curvefit = BB_curvefit)
print()
print()
print()
print()
print(interp_lc['MJD'].unique()[:50])
print()
print(BB_fit_results.head(50))
print()
print()
print()
print()
print(plot_polyfit_df)

# what we want: 
# L_rf vs MJD: on this plot there will be the actual (binned) data, the polyfit and the interpolated data from the polyfit
# BB R vs MJD: on this plot we'll have curve_fit R and brute force grid R, perhaps with a colour scale to indicate the sigma distance of the reduced chi squared for the 
#               BB fit to indicate how much we should trust the value
# BB T vs MJD: basically the same as BB R vs MJD
# sigma distance vs MJD?: sigma distance for the curve_fit and brute force results. I feel like this would be good to have alongside the polyfit, because if the polyfit was bad at this MJD,
#               then the interpolated L_rf of the band might be bad, too, which might mean that the BB fit will struggle to fit to this poorly interpolated datapoint.


#fig = plt.figure(figsize = (16, 7.3))
#ax1, ax2, ax3, ax4 = [plt.subplot(2, 2, i) for i in np.arange(1, 5, 1)]
fig, axs = plt.subplots(1, 1, figsize = (16, 8))
ax4 = axs


BB_fit_results = BB_fit_results.dropna(subset = ['red_chi_1sig'])
#print('BB fit results[cf_chi_sigma_dist]: ', BB_fit_results['cf_chi_sigma_dist'][:50])
#print('BB_fit_results[cf_chi_sigma_dist].iloc[0] :', BB_fit_results['cf_chi_sigma_dist'].iloc[0])
#print('np.ravel(BB_fit_results[cf_chi_sigma_dist]): ', np.ravel(BB_fit_results['cf_chi_sigma_dist'])[:50])
#print('np.array(BB_fit_results[cf_chi_sigma_dist]): ', np.array(BB_fit_results['cf_chi_sigma_dist'])[:50])

#removing that one datapoint with large errors, just for the sake of this rpesentation plot
BB_fit_results = BB_fit_results[BB_fit_results['cf_T_K'] < 20000.0].copy()

colour_cutoff = 5.0
titlefontsize = 25

norm = Normalize(vmin = 0.0, vmax = colour_cutoff)

BB_fit_res_low_SD = BB_fit_results[BB_fit_results['cf_chi_sigma_dist'] <= colour_cutoff].copy()
BB_fit_res_high_SD = BB_fit_results[BB_fit_results['cf_chi_sigma_dist'] > colour_cutoff].copy()

# blackbody temperature vs MJD
ax4.errorbar(BB_fit_res_low_SD['MJD'], BB_fit_res_low_SD['cf_T_K'], yerr = BB_fit_res_low_SD['cf_T_err_K'], linestyle = 'None', 
            fmt = 'None', c = 'k', zorder = 1)
sc = ax4.scatter(BB_fit_res_low_SD['MJD'], BB_fit_res_low_SD['cf_T_K'], 
                marker = 'o', edgecolors = 'k', linewidths = 0.5, zorder = 2,  cmap = 'viridis', c = np.ravel(BB_fit_res_low_SD['cf_chi_sigma_dist']))

cbar_label = r"Blackobdy fit's $\chi_{\nu}$"
cbar_label = r'Goodness of BB fit ($\chi_{\nu}$)'
cbar = plt.colorbar(sc, ax = ax4)
cbar.set_label(label = cbar_label, fontsize = (titlefontsize - 4), fontweight = 'bold')


ax4.errorbar(BB_fit_res_high_SD['MJD'], BB_fit_res_high_SD['cf_T_K'], yerr = BB_fit_res_high_SD['cf_T_err_K'], linestyle = 'None', 
            fmt = 'None', c = 'k', zorder = 3)
ax4.scatter(BB_fit_res_high_SD['MJD'], BB_fit_res_high_SD['cf_T_K'], 
                marker = 'o', edgecolors = 'k', linewidths = 0.5, zorder = 4, c = 'yellow')

    
ax4.grid(True)
ax4.set_xlim(MJDs_for_fit[ANT_name])
ax4.set_xlim((58450, 59570))

plt.xlabel('Time', fontweight = 'bold', fontsize = (titlefontsize - 2), labelpad = 15)
plt.ylabel('Blackbody temperature / K', fontweight = 'bold', fontsize = (titlefontsize - 2), labelpad = 15)
plt.title(ANT_name, fontweight = 'bold', fontsize = (titlefontsize), pad = 15)
ax4.tick_params(axis = 'both', which = 'major', labelsize=(titlefontsize - 4))
plt.setp(ax4.get_xticklabels(), fontweight='bold') # from chatGPT
plt.setp(ax4.get_yticklabels(), fontweight='bold')

fig.subplots_adjust(top=0.92,
                    bottom=0.12,
                    left=0.12,
                    right=1.07,
                    hspace=0.15,
                    wspace=0.19)

savepath = "C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS/plots/BB fits/OX_FL_BB_T_vs_MJD.png"
plt.savefig(savepath, dpi=300)
plt.show()











""" fig = plt.figure(figsize = (16, 8))
ax4 = fig.gca()


BB_fit_results = BB_fit_results.dropna(subset = ['red_chi_1sig'])
#print('BB fit results[cf_chi_sigma_dist]: ', BB_fit_results['cf_chi_sigma_dist'][:50])
#print('BB_fit_results[cf_chi_sigma_dist].iloc[0] :', BB_fit_results['cf_chi_sigma_dist'].iloc[0])
#print('np.ravel(BB_fit_results[cf_chi_sigma_dist]): ', np.ravel(BB_fit_results['cf_chi_sigma_dist'])[:50])
#print('np.array(BB_fit_results[cf_chi_sigma_dist]): ', np.array(BB_fit_results['cf_chi_sigma_dist'])[:50])




norm = Normalize(vmin = BB_fit_results['cf_chi_sigma_dist'].min(), vmax = BB_fit_results['cf_chi_sigma_dist'].max())

# bottom right: blackbody temperature vs MJD
ax4.errorbar(BB_fit_results['MJD'], BB_fit_results['cf_T_K'], yerr = BB_fit_results['cf_T_err_K'], linestyle = 'None', 
            fmt = 'None', c = 'k')
sc = ax4.scatter(BB_fit_results['MJD'], BB_fit_results['cf_T_K'], 
                marker = 'o', edgecolors = 'k', linewidths = 0.5, zorder = 2,  cmap = 'jet', c = np.ravel(BB_fit_results['cf_chi_sigma_dist']))
cbar = plt.colorbar(sc, ax = ax4)
cbar_label = r"$\chi_{\nu}$ sigma distance"
cbar.set_label(label = cbar_label, fontsize = 12)

    

ax4.grid(True)
ax4.set_xlim(MJDs_for_fit[ANT_name])
ax4.set_xlim((58450, 59570))
ax4.tick_params(axis='both', which='major', labelsize=12)

labelpad_val = 0.3
plt.ylabel('Blackbody temperature / K', fontweight = 'bold', fontsize = 14, labelpad = labelpad_val)
plt.xlabel('MJD', fontweight = 'bold', fontsize = 14, labelpad = labelpad_val)
plt.title(f"{ANT_name}'s evolution of blackbody temperature with time", fontweight = 'bold', fontsize = 16, pad = labelpad_val)

fig.subplots_adjust(top=0.925,
                    bottom=0.1,
                    left=0.08,
                    right=1.0,
                    hspace=0.205,
                    wspace=0.2) 

savepath = "C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS/plots/BB fits/OX_FL_BB_T_vs_MJD_other.png"
plt.savefig(savepath, dpi=300)
#plt.show()

 """





