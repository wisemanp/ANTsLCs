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
with plt.rc_context({'figure.figsize': (13.33, 7.5),
                     'font.size': 16,
                     'axes.titlesize': 13,
                     'axes.labelsize': 14,
                     'xtick.labelsize': 11,
                     'ytick.labelsize': 11, 
                     'figure.titlesize': 21, 
                     'legend.fontsize': 9}):
    titlefontsize = 23
    fig, axs = plt.subplots(3, 5)
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
            if band_marker == '*':
                markersize = 7.5
            else:
                markersize = 5

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
                        markeredgecolor = 'k', markeredgewidth = '0.5', markersize = markersize)

            else:
                band_offset_data = np.array(band_data['mag']) + band_offset_dict[band] # offsetting the band to make it easier to see on the plot
                h = ax.errorbar(convert_MJD_to_restframe_DSP(peak_MJD = ANT_peak_MJD, MJD = band_data['MJD'].to_numpy(), z = ANT_z), band_offset_data, yerr = band_data['magerr'], xerr = xerr, fmt = band_marker, c = band_color, label = band_offset_label_dict[band], linestyle = 'None',
                            markeredgecolor = 'k', markeredgewidth = '0.5', markersize = markersize)
                

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
        ax.tick_params(axis = 'both', which = 'major')
        ax.set_title(transient_names[i],  fontweight = 'bold')
        ax.tick_params(axis='both')
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
            ax.set_xlim((-150, None))
        if transient_names[i] == 'ASASSN-17jz':
            ax.set_xlim((None, 800))  
        if transient_names[i] == 'ASASSN-18jd':
            ax.set_xlim((-70, 600))
        if transient_names[i] == 'ZTF18aczpgwm':
            ax.set_xlim((None, 750))
        if transient_names[i] == 'ZTF19aailpwl':
            ax.set_xlim((None, 500))
        if transient_names[i] == 'ZTF20aanxcpf':
            ax.set_xlim((None, 700))
        if transient_names[i] == 'ZTF20abgxlut':
            ax.set_xlim((None, 300))
        if transient_names[i] == 'ZTF19aamrjar':
            ax.set_xlim((-400, 650))

        

    #fig.legend(leg_handles, leg_labels, loc = 'lower right', bbox_to_anchor = (0.97, 0.00), ncols = 4, fontsize = 8)
    #fig.legend(sorted_handels, sorted_labels, loc = 'lower right', bbox_to_anchor = (0.99, 0.00), ncols = 8)
    plt.suptitle('Light Curves of All ANTs Used in This Study', fontweight='bold', va = 'center', fontsize = titlefontsize, y = 0.965)
    
    fig.supxlabel('Phase (rest-frame) [days]', fontweight = 'bold', va = 'center', y = 0.04) #, y = 0.205
    fig.supylabel('Apparent Magnitude (+ Offset)', fontweight = 'bold', va = 'center') # , x = 0.03, y = 0.6
    plt.subplots_adjust(top=0.875,
        bottom=0.12,
        left=0.09,
        right=0.978,
        hspace=0.5,
        wspace=0.3)
    savepath = "C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS/plots/light curves/FINAL_PRESENTATION_ALL_ANT_lc_phase.png"
    plt.savefig(savepath, dpi=500)
    #plt.show() 





#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################
# PLOTTING LWX

with plt.rc_context({'figure.figsize': (7, 4),
                     'font.size': 11,
                     'axes.titlesize': 13,
                     'axes.labelsize': 11,
                     'xtick.labelsize': 10.5,
                     'ytick.labelsize': 10.5, 
                     'figure.titlesize': 21, 
                     'legend.fontsize': 9}):

    titlefontsize = 15
    fig, ax = plt.subplots(1, 1)
    #axs = axs.ravel() # this converts axs from a 2D array into a 1D one to easily iterate over

    ANT_name = 'ZTF20abrbeie'
    idx = transient_names.index(ANT_name)
    lc_df = lc_df_list[idx]
    MJD_xlim = MJD_xlims[ANT_name] # the MJD that the transient occur over
    ANT_z = ANT_proper_redshift_dict[ANT_name]
    ANT_peak_MJD = peak_MJD_dict[ANT_name]


    if MJD_xlim is not None:
        MJD_lower_xlim, MJD_upper_xlim = MJD_xlim
        phase_lower_xlim = convert_MJD_to_restframe_DSP(peak_MJD = ANT_peak_MJD, MJD = MJD_lower_xlim, z = ANT_z)
        phase_upper_xlim = convert_MJD_to_restframe_DSP(peak_MJD = ANT_peak_MJD, MJD = MJD_upper_xlim, z = ANT_z)
        phase_xlim = (phase_lower_xlim, phase_upper_xlim)

    else:
        phase_xlim = None

    for band in list_of_bands[idx]: # looking at the list of bands that are present for this particular lightcurve
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

        band_offset_data = np.array(band_data['mag']) + band_offset_dict[band] # offsetting the band to make it easier to see on the plot
        ax.errorbar(convert_MJD_to_restframe_DSP(peak_MJD = ANT_peak_MJD, MJD = band_data['MJD'].to_numpy(), z = ANT_z), band_offset_data, yerr = band_data['magerr'], xerr = xerr, fmt = band_marker, c = band_color, label = band_offset_label_dict[band], linestyle = 'None',
                    markeredgecolor = 'k', markeredgewidth = '0.5', markersize = 5)

        ax.set_xlim(phase_xlim)



    

    ax.invert_yaxis()
    ax.grid(True)
    ax.tick_params(axis = 'both', which = 'major')
    ax.tick_params(axis='both')
    #ax.locator_params(axis='x', nbins=5)
    #ax.locator_params(axis='y', nbins=5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.legend(loc = 'upper right')

        

    #fig.legend(leg_handles, leg_labels, loc = 'lower right', bbox_to_anchor = (0.97, 0.00), ncols = 4, fontsize = 8)
    #fig.legend(sorted_handels, sorted_labels, loc = 'lower right', bbox_to_anchor = (0.99, 0.00), ncols = 8)
    plt.suptitle('Light Curve of AT2021lwx/ZTF20abrbeie', fontweight='bold', va = 'center', fontsize = titlefontsize, y = 0.965)

    fig.supxlabel('Phase (rest-frame) [days]', fontweight = 'bold', va = 'center', y = 0.04) #, y = 0.205
    fig.supylabel('Apparent Magnitude (+ Offset)', fontweight = 'bold', va = 'center') # , x = 0.03, y = 0.6
    plt.subplots_adjust(top=0.875,
        bottom=0.14,
        left=0.13,
        right=0.978,
        hspace=0.5,
        wspace=0.3)
    savepath = "C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS/plots/light curves/FINAL_PRESENTATION_LWX_lc_phase.png"
    plt.savefig(savepath, dpi=500)
    #plt.show() 




