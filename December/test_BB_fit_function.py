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
                        extrapolate = False, b_colour_dict = band_colour_dict, plot_polyfit = True)


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
fig, axs = plt.subplots(2, 2, sharex=True, figsize = (16, 7.2))
ax1, ax2 = axs[0]
ax3, ax4 = axs[1]

# top left: the L_rf vs MJD light curve
for b in ANT_bands: # iterate through all of the bands present in the ANT's light curve
    b_df = ANT_df[ANT_df['band'] == b].copy()
    b_interp_df = interp_lc[interp_lc['band'] == b].copy()
    b_colour = band_colour_dict[b]
    ax1.errorbar(b_df['wm_MJD'], b_df['wm_L_rf'], yerr = b_df['wm_L_rf_err'], xerr = (b_df['MJD_lower_err'], b_df['MJD_upper_err']), fmt = 'o', c = b_colour, 
                 linestyle = 'None', markeredgecolor = 'k', markeredgewidth = '1.0', label = (b if b != reference_band else f'{b} - REF BAND'))
    
    if (b in bands_for_BB) and (b != reference_band): # plot the polyfit for the bands which actually have polyfits..
        band_polyfit = plot_polyfit_df[plot_polyfit_df['band'] == b].copy()
        #print()
        #print("poly_MJD:", band_polyfit['poly_MJD'])
        #print("poly_L_rf.iloc[0]:", band_polyfit['poly_L_rf'].iloc[0])
        ax1.plot(np.ravel(band_polyfit['poly_MJD'].iloc[0]), np.ravel(band_polyfit['poly_L_rf'].iloc[0]), c = b_colour, label = f"red chi sigma dist = \n {band_polyfit['chi_sigma_dist'].iloc[0]:.3f}")
        ax1.errorbar(b_interp_df['MJD'], b_interp_df['L_rf'], yerr = b_interp_df['L_rf_err'], fmt = '^', c = b_colour, alpha = 0.5, markeredgecolor = 'k', 
                     markeredgewidth = '0.5', linestyle = 'None')
        
    ax1.set_ylabel('Rest frame luminosity')
    


BB_fit_results = BB_fit_results.dropna(subset = ['red_chi_1sig'])
#print('BB fit results[cf_chi_sigma_dist]: ', BB_fit_results['cf_chi_sigma_dist'][:50])
#print('BB_fit_results[cf_chi_sigma_dist].iloc[0] :', BB_fit_results['cf_chi_sigma_dist'].iloc[0])
#print('np.ravel(BB_fit_results[cf_chi_sigma_dist]): ', np.ravel(BB_fit_results['cf_chi_sigma_dist'])[:50])
#print('np.array(BB_fit_results[cf_chi_sigma_dist]): ', np.array(BB_fit_results['cf_chi_sigma_dist'])[:50])

if BB_curvefit == True:
    norm = Normalize(vmin = BB_fit_results['cf_chi_sigma_dist'].min(), vmax = BB_fit_results['cf_chi_sigma_dist'].max())
    # top right: blackbody radius vs MJD
    ax2.errorbar(BB_fit_results['MJD'], BB_fit_results['cf_R_cm'], yerr = BB_fit_results['cf_R_err_cm'], linestyle = 'None', c = 'k', 
                 fmt = 'None', zorder = 1)
    sc = ax2.scatter(BB_fit_results['MJD'], BB_fit_results['cf_R_cm'], cmap = 'jet', c = np.ravel(BB_fit_results['cf_chi_sigma_dist']), 
                 label = 'Curve fit results', marker = 'o', zorder = 2, edgecolors = 'k', linewidths = 0.5)
    plt.colorbar(sc, ax = ax2, label = 'Chi sigma distance')

    # bottom left: reduced chi squared sigma distance vs MJD
    ax3.scatter(BB_fit_results['MJD'], BB_fit_results['cf_chi_sigma_dist'], marker = 'o', label = 'Curve fit results', edgecolors = 'k', linewidths = 0.5)

    # bottom right: blackbody temperature vs MJD
    ax4.errorbar(BB_fit_results['MJD'], BB_fit_results['cf_T_K'], yerr = BB_fit_results['cf_T_err_K'], linestyle = 'None', c = 'k', 
                fmt = 'None', zorder = 1)
    sc = ax4.scatter(BB_fit_results['MJD'], BB_fit_results['cf_T_K'], cmap = 'jet', c = BB_fit_results['cf_chi_sigma_dist'], 
                 label = 'Curve fit results', marker = 'o', edgecolors = 'k', linewidths = 0.5, zorder = 2)
    
    plt.colorbar(sc, ax = ax4, label = 'Chi sigma distance')
    
if BB_brute == True:
    # top right: blackbody radius vs MJD
    ax2.scatter(BB_fit_results['MJD'], BB_fit_results['brute_R_cm'], linestyle = 'None', cmap = 'jet', c = BB_fit_results['brute_chi_sigma_dist'], 
                 label = 'brute force gridding results', fmt = 'None')
    
    # bottom left: reduced chi squared sigma distance vs MJD
    ax3.scatter(BB_fit_results['MJD'], BB_fit_results['brute_chi_sigma_dist'], marker = '^', label = 'Brute force gridding results', ecolors = 'k', linewdiths = 0.5)

    # bottom right: blackbody temperature vs MJD
    ax4.scatter(BB_fit_results['MJD'], BB_fit_results['brute_T_K'], linestyle = 'None', cmap = 'jet', c = BB_fit_results['brute_chi_sigma_dist'], 
                 label = 'Brute force gridding results', fmt = '^')

for ax in [ax1, ax2, ax3, ax4]:
    ax.grid(True)
    ax.set_xlim(MJDs_for_fit[ANT_name])
    ax.legend(fontsize = 8)
ax2.set_ylabel('Blackbody radius / cm')
ax3.set_ylabel('Reduced chi squared sigma distance \n (<=1 = Good fit)')
ax4.set_ylabel('Blackbody temperature / K')
fig.suptitle(f"Blackbody fit results across {ANT_name}'s light curve")
fig.supxlabel('MJD')
fig.subplots_adjust(top=0.92,
                    bottom=0.085,
                    left=0.055,
                    right=0.97,
                    hspace=0.15,
                    wspace=0.19)
plt.show()







