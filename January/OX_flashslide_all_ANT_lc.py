import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS") # this allows us to access the plotting_preferences.py file 
from plotting_preferences import band_colour_dict, band_marker_dict, band_offset_dict, band_offset_label_dict, MJD_xlims
from functions import load_ANT_data


#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
#############################################################################################################################################################################################################


# loading in the files
lc_df_list, transient_names, list_of_bands = load_ANT_data()


#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
# make a colour key for each band, so that all plots use the same colour key for each band for easier comparison
free_list_of_bands = [element for sublist in list_of_bands for element in sublist]
free_list_of_bands = np.array(free_list_of_bands)
unique_bands = np.unique(free_list_of_bands)




#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
want_separate_plots = False
# PLOTTING 
if want_separate_plots == True:
    for i, lc_df in enumerate(lc_df_list): # loop through the light curve data frames
        plot_name = transient_names[i]+'_lc.png'
        ANT_name = transient_names[i]
        plt.figure(figsize = (14, 7))
        MJD_xlim = MJD_xlims[ANT_name] # the MJD that the transient occur over

        for band in list_of_bands[i]: # looking at the list of bands that are present for this particular lightcurve
            band_color = band_colour_dict[band]
            band_data = lc_df[lc_df['band']==band] # filter for the lightcurve data in this specific band
            band_data = band_data[band_data['magerr'] < 2.0] # filter out the data points with HUGE errors BINNED DATA POINTS WITH HUGE MAGERR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11
            band_offset_data = np.array(band_data['mag']) + band_offset_dict[band] # offsetting the band to make it easier to see on the plot
            plt.errorbar(band_data['MJD'], band_offset_data, yerr = band_data['magerr'], fmt = 'o', c = band_color, label = band_offset_label_dict[band], linestyle = 'None',
                        markeredgecolor = 'k', markeredgewidth = '0.5')

        if transient_names[i]== 'ZTF20abodaps': # this one lightcurve needs a y lim..
            plt.ylim((13, 24))
        plt.title(transient_names[i], fontsize=12, fontweight='bold')
        plt.xlabel('MJD')
        plt.ylabel('Apparent magnitude')
        plt.xlim(MJD_xlim)
        plt.legend()
        plt.grid()
        plt.gca().invert_yaxis()
        #plt.savefig(plot_name)
        plt.show()



#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
# PLOT THEM ALL ONTO ONE SUBPLOT
leg_handles = [] # for the master legend
leg_labels = [] # for the master legend
fig, axs = plt.subplots(3, 7, figsize = (16, 8))
axs = axs.ravel() # this converts axs from a 2D array into a 1D one to easily iterate over
for i, ax in enumerate(axs): # loop through the light curve data frames
    if i==18:
        ax.axis('Off')
        axs[i+1].axis('Off')
        axs[i+2].axis('Off')
        break

    lc_df = lc_df_list[i]
    ANT_name = transient_names[i]
    MJD_xlim = MJD_xlims[ANT_name] # the MJD that the transient occur over
    print(i, ANT_name)
    print(lc_df.head())
    print(lc_df['band'].unique())
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

        band_offset_data = np.array(band_data['mag']) + band_offset_dict[band] # offsetting the band to make it easier to see on the plot
        h = ax.errorbar(band_data['MJD'], band_offset_data, yerr = band_data['magerr'], xerr = xerr, fmt = band_marker, c = band_color, label = band_offset_label_dict[band], linestyle = 'None',
                     markeredgecolor = 'k', markeredgewidth = '0.5', markersize = 5)
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
    ax.set_xlim(MJD_xlim)

    if transient_names[i] == 'PS1-10adi' or transient_names[i] == 'PS1-13jw':
        subplot_title = transient_names[i]+' (abs mag)'
    else:
        subplot_title = transient_names[i]

    ax.set_title(subplot_title, fontsize = 10, fontweight = 'bold')

    if transient_names[i] == 'ZTF20abodaps': # this one lightcurve needs a y lim..
        ax.set_ylim((24, 13))
    



#fig.legend(leg_handles, leg_labels, loc = 'lower right', bbox_to_anchor = (0.97, 0.00), ncols = 4, fontsize = 8)
titlefontsize = 25
fig.legend(sorted_handels, sorted_labels, loc = 'lower right', bbox_to_anchor = (0.98, 0.04), ncols = 4, fontsize = 8)
plt.suptitle('All ANT light curves used in this study', fontsize=titlefontsize, fontweight='bold', va = 'center', y = 0.97)
fig.supxlabel('Time', fontweight = 'bold', va = 'center', y = 0.03, fontsize = (titlefontsize-2))
fig.supylabel('Apparent magnitude', fontweight = 'bold', va = 'center', x = 0.015, fontsize = (titlefontsize-2))
plt.subplots_adjust(top=0.912,
                    bottom=0.09,
                    left=0.061,
                    right=0.981,
                    hspace=0.241,
                    wspace=0.181)
savepath = "C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS/plots/light curves/OX_FL_ALL_ANT_lc.png"
plt.savefig(savepath, dpi=300)







