import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS") # this allows us to access the plotting_preferences.py file 
from November.plotting_preferences import band_colour_dict, band_marker_dict, band_offset_dict, band_ZP_dict, band_obs_centwl_dict

# loading in the files
#folder_path = "C:/Users/laure/OneDrive/Desktop/YoRiS Data/Phil's lightcurves 17_9" # folder path containing the light curve data files
folder_path = "C:/Users/laure/OneDrive/Desktop/YoRiS Desktop/YoRiS Data/modified Phil's lightcurves" # I added stuff like days since peak + removed outliers
lc_df_list = []
transient_names = []
list_of_bands = []
for file in os.listdir(folder_path): # file is a string of the file name such as 'file_name.dat'
    file_path = os.path.join(folder_path, file) 
    file_df = pd.read_csv(file_path, delimiter = ',') # the lightcurve data in a dataframe
    lc_df_list.append(file_df)
    print(file_df)

    trans_name = file[:-7] # the name of the transient
    transient_names.append(trans_name)

    trans_bands = file_df['band'].unique() # the bands in which this transient has data for
    list_of_bands.append(trans_bands)




# make a colour key for each band, so that all plots use the same colour key for each band for easier comparison
free_list_of_bands = [element for sublist in list_of_bands for element in sublist]
free_list_of_bands = np.array(free_list_of_bands)
unique_bands = np.unique(free_list_of_bands)
band_plot_colours = ['magenta', 'y', 'm', 'c', 'r', 'g', 'b', 'darkviolet', 'orange', 'silver', 'lime', 'darkturquoise', 'mediumspringgreen', 'maroon', 'tomato', 'lightpink']
band_colour_key = dict(zip(unique_bands, band_plot_colours)) # a dictionary which maps each band to its given color




# the labels for the offset bands in the legend, e.g. 'WISE_1 + 3' if the WISE_1 band is offset by +3 mag
# 'ATLAS_c' 'ATLAS_o' 'PS_i' 'PS_w' 'PS_y' 'PS_z' 'UVOT_B' 'UVOT_U' 'UVOT_UVM1' 'UVOT_UVM2' 'UVOT_UVW2' 'UVOT_V' 'WISE_W1' 'WISE_W2' 'ZTF_g' 'ZTF_r'
band_offset = [1.5, -2.5, -1.5, -1, 0, 0, 0, -2.5, -1.5, -2, 0, 0, 1.5, 1.5, 2.5, 0.5]
#band_offset = [0]*16
band_offset_key = dict(zip(unique_bands, band_offset))
band_offset_name = []
for i in range(len(unique_bands)):
    if band_offset[i] >= 0:
        offset_name = unique_bands[i]+' + '+str(band_offset[i])

    elif band_offset[i] < 0:
        offset_name = unique_bands[i]+' '+str(band_offset[i])

    band_offset_name.append(offset_name)

band_offset_name_key = dict(zip(unique_bands, band_offset_name))




# limiting the MJD to plot the light curve over - only plot the more interesting part
# THE ORDER OF TRANSIENTS: 
#            'ZTF18aczpgwm', 'ZTF19aailpwl', 'ZTF19aamrjar', 'ZTF19aatubsj', 'ZTF20aanxcpf', 'ZTF20abgxlut', 'ZTF20abodaps', 'ZTF20abrbeie', 'ZTF20acvfraq', 'ZTF21abxowzx', 'ZTF22aadesap'
MJD_xlims = [(58350, 60250), (58450, 60050), (58340, 60250), (58550, 60250), (58400, 60320), (58950, 59500), (58450, 60250), (59000, 59950), (58700, 60200), (59450, 60250), (59650, 60400)]



want_separate_plots = True

# PLOTTING 
if want_separate_plots == True:
    for i, lc_df in enumerate(lc_df_list): # loop through the light curve data frames
        plot_name = transient_names[i]+'_lc.png'
        plt.figure(figsize = (14, 7))
        MJD_xlim = MJD_xlims[i] # the MJD that the transient occur over

        for band in list_of_bands[i]: # looking at the list of bands that are present for this particular lightcurve
            band_color = band_colour_key[band]
            band_data = lc_df[lc_df['band']==band] # filter for the lightcurve data in this specific band
            band_data = band_data[band_data['magerr'] < 2.0] # filter out the data points with HUGE errors BINNED DATA POINTS WITH HUGE MAGERR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11
            band_offset_data = np.array(band_data['mag']) + band_offset_key[band] # offsetting the band to make it easier to see on the plot
            zero_magerr_data = band_data[band_data['magerr'] < 0.000001].copy()

            #plt.scatter(zero_magerr_data['MJD'], zero_magerr_data['mag'], c = 'k', label = 'ZERO MAGERR DATA', s = 45, marker = '*')
            plt.errorbar(band_data['MJD'], band_offset_data, yerr = band_data['magerr'], fmt = 'o', c = band_color, label = band_offset_name_key[band], linestyle = 'None',
                        markeredgecolor = 'k', markeredgewidth = '0.5')
            
        if transient_names[i]== 'ZTF20abodaps': # this one lightcurve needs a y lim..
            plt.ylim((13, 24))
        plt.title(transient_names[i], fontsize=12, fontweight='bold')
        plt.axvline(x = lc_df['peak_MJD'].iloc[0], c = 'k', label = 'peak')
        plt.xlabel('MJD')
        plt.ylabel('Apparent magnitude')
        plt.xlim(MJD_xlim)
        plt.legend()
        plt.grid()
        plt.gca().invert_yaxis()
        #plt.savefig(plot_name)
        plt.show()


# PLOT THEM ALL ONTO ONE SUBPLOT
leg_handles = [] # for the master legend
leg_labels = [] # for the master legend
fig, axs = plt.subplots(3, 4, figsize = (18, 7))
axs = axs.ravel() # this converts axs from a 2D array into a 1D one to easily iterate over
for i, ax in enumerate(axs): # loop through the light curve data frames
    if i==11:
        ax.axis('Off')
        break

    lc_df = lc_df_list[i]
    MJD_xlim = MJD_xlims[i] # the MJD that the transient occur over

    for band in list_of_bands[i]: # looking at the list of bands that are present for this particular lightcurve
        band_color = band_colour_key[band]
        band_data = lc_df[lc_df['band']==band] # filter for the lightcurve data in this specific band
        band_data = band_data[band_data['magerr'] < 2.0] # filter out the data points with HUGE errors BINNED DATA POINTS WITH HUGE MAGERR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11
        band_offset_data = np.array(band_data['mag']) + band_offset_key[band] # offsetting the band to make it easier to see on the plot
        

        h = ax.errorbar(band_data['MJD'], band_offset_data, yerr = band_data['magerr'], fmt = 'o', c = band_color, label = band_offset_name_key[band], linestyle = 'None',
                     markeredgecolor = 'k', markeredgewidth = '0.5', markersize = 5)
        
        handle = h[0]
        label = band_offset_name_key[band]
        if label not in leg_labels:
            leg_labels.append(label)
            leg_handles.append(handle)

    ax.invert_yaxis()
    ax.axvline(x = lc_df['peak_MJD'].iloc[0], c = 'k', label = 'peak')
    ax.grid(True)
    ax.set_xlim(MJD_xlim)
    ax.set_title(transient_names[i])

    if transient_names[i]== 'ZTF20abodaps': # this one lightcurve needs a y lim..
        ax.set_ylim((24, 13))

fig.legend(leg_handles, leg_labels, loc = 'lower right', bbox_to_anchor = (0.98, 0.07), ncols = 2)
plt.suptitle('All transients (bands shifted for visibility) (removed data points with magerr > 2.0)', fontsize=15, fontweight='bold')
fig.supxlabel('MJD', fontweight = 'bold')
fig.supylabel('Apparent magnitude', fontweight = 'bold')
plt.subplots_adjust(top = 0.912, bottom = 0.092, left = 0.061, right = 0.976, hspace = 0.361, wspace = 0.211)
#plt.savefig('subplot_of_all_LCs_no_band_shift.png')
plt.show()