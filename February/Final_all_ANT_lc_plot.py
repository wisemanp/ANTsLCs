import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from matplotlib.ticker import MaxNLocator

sys.path.append("C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS") # this allows us to access the plotting_preferences.py file 
from plotting_preferences import band_colour_dict, band_marker_dict, band_offset_dict, band_offset_label_dict, MJD_xlims, ANT_proper_redshift_dict, peak_MJD_dict
from functions import load_ANT_data



#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
#############################################################################################################################################################################################################


# loading in the files
lc_df_list, transient_names, list_of_bands = load_ANT_data()

ANTs_to_remove = ['PS1-13jw', 'Gaia18cdj', 'Gaia16aaw', 'CSS100217']

for ANT_to_remove in ANTs_to_remove:
    index = transient_names.index(ANT_to_remove)
    #lc_df_list.pop(index)
    del lc_df_list[index]
    del transient_names[index]
    del list_of_bands[index]


#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
# make a list of the bands present across all ANTs
free_list_of_bands = [element for sublist in list_of_bands for element in sublist]
free_list_of_bands = np.array(free_list_of_bands)
unique_bands = np.unique(free_list_of_bands)


#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
want_separate_plots = False








def convert_MJD_to_restframe_DSP(peak_MJD, MJD, z):
    return (MJD - peak_MJD) / (1 + z)



#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
# PLOT THEM ALL ONTO ONE SUBPLOT
leg_handles = [] # for the master legend
leg_labels = [] # for the master legend
fig, axs = plt.subplots(5, 3, figsize = (8.2, 11.6))
axs = axs.ravel() # this converts axs from a 2D array into a 1D one to easily iterate over
for i, ax in enumerate(axs): # loop through the light curve data frames
    if i==14:
        ax.axis('Off')
        #axs[i+1].axis('Off')
        #axs[i+2].axis('Off')
        break

    lc_df = lc_df_list[i]
    ANT_name = transient_names[i]
    MJD_xlim = MJD_xlims[ANT_name] # the MJD that the transient occur over
    ANT_z = ANT_proper_redshift_dict[ANT_name]
    ANT_peak_MJD = peak_MJD_dict[ANT_name]

    if ANT_name == 'ASASSN-17jz':
        MJD_xlim = (57800, 59200)

    if MJD_xlim is not None:
        MJD_lower_xlim, MJD_upper_xlim = MJD_xlim
        phase_lower_xlim = convert_MJD_to_restframe_DSP(peak_MJD = ANT_peak_MJD, MJD = MJD_lower_xlim, z = ANT_z)
        phase_upper_xlim = convert_MJD_to_restframe_DSP(peak_MJD = ANT_peak_MJD, MJD = MJD_upper_xlim, z = ANT_z)
        phase_xlim = (phase_lower_xlim, phase_upper_xlim)

    else:
        phase_xlim = None




    print(i, ANT_name)
    print(ANT_z)
    print(ANT_peak_MJD)
    #print(lc_df.head())
    #print(lc_df['band'].unique())
    print()


    for band in list_of_bands[i]: # looking at the list of bands that are present for this particular lightcurve
        band_color = band_colour_dict[band]
        band_marker = band_marker_dict[band]
        band_data = lc_df[lc_df['band']==band].copy() # filter for the lightcurve data in this specific band
        band_data = band_data[band_data['magerr'] < 2.0].copy() # filter out the data points with HUGE errors BINNED DATA POINTS WITH HUGE MAGERR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11
        #  adding x errors to the plot
        if 'MJD_err' in band_data.columns:
            xerr = band_data['MJD_err']

        elif 'MJD_lower_err' in band_data.columns and 'MJD_upper_err' in band_data.columns:
            xerr = [band_data['MJD_lower_err'], band_data['MJD_upper_err']]

        else:
            xerr = [0]*len(band_data['MJD'])

        # THIS IS TO PLOT MJD
        #if ANT_name == 'PS1-10adi':
        #    band_offset_data = np.array(band_data['app_mag']) + band_offset_dict[band] # offsetting the band to make it easier to see on the plot
        #    h = ax.errorbar(band_data['MJD'], band_offset_data, yerr = band_data['app_magerr'], xerr = xerr, fmt = band_marker, c = band_color, label = band_offset_label_dict[band], linestyle = 'None',
        #             markeredgecolor = 'k', markeredgewidth = '0.5', markersize = 5)

        #else:
        #    band_offset_data = np.array(band_data['mag']) + band_offset_dict[band] # offsetting the band to make it easier to see on the plot
        #    h = ax.errorbar(band_data['MJD'], band_offset_data, yerr = band_data['magerr'], xerr = xerr, fmt = band_marker, c = band_color, label = band_offset_label_dict[band], linestyle = 'None',
        #                markeredgecolor = 'k', markeredgewidth = '0.5', markersize = 5)
        #ax.set_xlim(MJD_xlim)

        if ANT_name == 'PS1-10adi':
            band_offset_data = np.array(band_data['app_mag']) + band_offset_dict[band] # offsetting the band to make it easier to see on the plot
            h = ax.errorbar(convert_MJD_to_restframe_DSP(peak_MJD = ANT_peak_MJD, MJD = band_data['MJD'].to_numpy(), z = ANT_z), band_offset_data, yerr = band_data['app_magerr'], xerr = xerr, fmt = band_marker, c = band_color, label = band_offset_label_dict[band], linestyle = 'None',
                     markeredgecolor = 'k', markeredgewidth = '0.5', markersize = 5)

        else:
            band_offset_data = np.array(band_data['mag']) + band_offset_dict[band] # offsetting the band to make it easier to see on the plot
            h = ax.errorbar(convert_MJD_to_restframe_DSP(peak_MJD = ANT_peak_MJD, MJD = band_data['MJD'].to_numpy(), z = ANT_z), band_offset_data, yerr = band_data['magerr'], xerr = xerr, fmt = band_marker, c = band_color, label = band_offset_label_dict[band], linestyle = 'None',
                        markeredgecolor = 'k', markeredgewidth = '0.5', markersize = 5)
            

        ax.set_xlim(phase_xlim)


        handle = h[0]
        label = band_offset_label_dict[band]
        if label not in leg_labels:
            leg_labels.append(label)
            leg_handles.append(handle)

        sorted_handles_and_labels = sorted(zip(leg_handles, leg_labels), key = lambda tp: tp[0].get_marker())
        sorted_handels, sorted_labels = zip(*sorted_handles_and_labels)

    

    ax.invert_yaxis()
    ax.grid(True)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 6)
    ax.set_title(transient_names[i], fontsize = 10.5, fontweight = 'bold')
    ax.tick_params(axis='both', labelsize=9.5)
    #ax.locator_params(axis='x', nbins=5)
    #ax.locator_params(axis='y', nbins=5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    if transient_names[i] == 'ZTF20abodaps': # this one lightcurve needs a y lim..
        ax.set_ylim((24, 13))
        ax.set_xlim((-100, None))
    if transient_names[i] == 'ZTF20aanxcpf':
        ax.set_xlim((-150, None))
    if transient_names[i] == 'ZTF20acvfraq':
        ax.set_xlim((-200, None))

    

#fig.legend(leg_handles, leg_labels, loc = 'lower right', bbox_to_anchor = (0.97, 0.00), ncols = 4, fontsize = 8)
fig.legend(sorted_handels, sorted_labels, loc = 'lower right', bbox_to_anchor = (1.01, 0.00), ncols = 5, fontsize = 9)
plt.suptitle('Light curves of all ANTs used in this study', fontsize=18, fontweight='bold', va = 'center', y = 0.98)
axisfontsize = 14
fig.supxlabel('Phase (rest-frame) / days', fontweight = 'bold', va = 'center', y = 0.17, fontsize = axisfontsize)
fig.supylabel('Apparent magnitude', fontweight = 'bold', va = 'center', x = 0.03, y = 0.6, fontsize = axisfontsize)
plt.subplots_adjust(top=0.93,
    bottom=0.215,
    left=0.12,
    right=0.978,
    hspace=0.47,
    wspace=0.3)
savepath = "C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS/plots/light curves/FINAL_ALL_ANT_lc_phase.png"
plt.savefig(savepath, dpi=300)
#plt.show() 




