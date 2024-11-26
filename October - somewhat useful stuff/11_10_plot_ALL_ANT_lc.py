import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS") # this allows us to access the plotting_preferences.py file 
from plotting_preferences import band_colour_dict, band_marker_dict, band_offset_dict, band_offset_label_dict, MJD_xlims
from functions import load_ANT_data

def bin_data(bands_df, MJD_binsize):
    """
    Bins data for a given band 
    """
    bands_df = bands_df.copy() # this creates a copy rather than a view of the dataframe so that we don't modify the original dataframe
    
    # this rounds the min_MJD present in V_band data for the transient to the nearest 10, then goes to the bin below this to ensure that 
    # we capture the first data point, in case this rounded the first bin up. Since we need data for both V_band and B_band within each
    # MJD bin, it's fine to base the min MJD bin to start at on just the V_band data, since even if we had B_band data before this date, 
    # we wouldn't be able to pair it with data from V_band to calculcte the color anyways
    MJD_bin_min = int( round(bands_df['MJD'].min(), -1) - 10 )
    MJD_bin_max = int( round(bands_df['MJD'].max(), -1) + 10 )
    MJD_bins = range(MJD_bin_min, MJD_bin_max + MJD_binsize, MJD_binsize) # create the bins

    # data frames for the binned band data 
    bands_df.loc[:,'MJD_bin'] = pd.cut(bands_df['MJD'], MJD_bins)

    # what we want: 
    # mean MJD
    # min and max MJD within this bin
    # mean mag 
    # standard error on mean mag
    #   - std dev
    #   - count of no of data points within the bin
    #   - std err = std dev / sqrt(count)

    bands_binned_df = bands_df.groupby('MJD_bin', observed = False).agg(mean_MJD = ('MJD', 'mean'), 
                                                                        min_MJD = ('MJD', 'min'), 
                                                                        max_MJD = ('MJD', 'max'), 
                                                                        mean_mag = ('mag', 'mean'), 
                                                                        std_dev_mag = ('mag', 'std'), 
                                                                        count = ('mag', 'count'))
    
    bands_binned_df['mag_std_err'] = bands_binned_df['std_dev_mag'] / np.sqrt(bands_binned_df['count']) # std err of mean mag
    bands_binned_df.rename(columns = {'mean_MJD' : 'MJD', 'mean_mag' : 'mag', 'mag_std_err':'magerr'}) # rename columns to the usual naming system
    bands_binned_df.drop(['std_dev_mag', 'count'], axis = 1) # drop unnecessary columns
    #print(bands_binned_df)

    return bands_binned_df



#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
#############################################################################################################################################################################################################


# loading in the files
""" folder_path = "C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS Data/modified Phil's lightcurves" # folder path containing the light curve data files
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


other_folder_path = "C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS Data/other ANT data/ALL_FULL_LCs"
for file in os.listdir(other_folder_path):
    file_path = os.path.join(other_folder_path, file)
    file_df = pd.read_csv(file_path, delimiter = ',')
    lc_df_list.append(file_df)

    trans_name = file[:-len('_FULL_LC.csv')]
    transient_names.append(trans_name)
    
    trans_bands = file_df['band'].unique()
    list_of_bands.append(trans_bands)

print()
#print(transient_names) """

lc_df_list, transient_names, list_of_bands = load_ANT_data()


#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
# make a colour key for each band, so that all plots use the same colour key for each band for easier comparison
free_list_of_bands = [element for sublist in list_of_bands for element in sublist]
free_list_of_bands = np.array(free_list_of_bands)
unique_bands = np.unique(free_list_of_bands)





#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
# limiting the MJD to plot the light curve over - only plot the more interesting part
# THE ORDER OF TRANSIENTS: 
#            'ZTF18aczpgwm', 'ZTF19aailpwl', 'ZTF19aamrjar', 'ZTF19aatubsj', 'ZTF20aanxcpf', 'ZTF20abgxlut', 'ZTF20abodaps', 'ZTF20abrbeie', 'ZTF20acvfraq', 'ZTF21abxowzx', 'ZTF22aadesap'
""" MJD_xlims = [(58350, 60250), (58450, 60050), (58340, 60250), (58550, 60250), (58400, 60320), (58950, 59500), (58450, 60300), (59000, 59950), (58700, 60200), (59400, 60250), (59650, 60400)]
no_None_xlim = len(transient_names) - len(MJD_xlims)
for i in range(no_None_xlim): # for now, append None to the x lims so that I can plot them all FOR NOW LALLALALA
    MJD_xlims.append(None) 
 """

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
fig, axs = plt.subplots(3, 7, figsize = (18, 7.5))
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
    ax.set_title(transient_names[i], fontsize = 9, fontweight = 'bold')

    if transient_names[i] == 'ZTF20abodaps': # this one lightcurve needs a y lim..
        ax.set_ylim((24, 13))


#fig.legend(leg_handles, leg_labels, loc = 'lower right', bbox_to_anchor = (0.97, 0.00), ncols = 4, fontsize = 8)
fig.legend(sorted_handels, sorted_labels, loc = 'lower right', bbox_to_anchor = (0.97, 0.00), ncols = 4, fontsize = 8)
plt.suptitle('All transients (bands shifted for visibility) (removed data points with magerr > 2.0). Better to use days since peak. Bin ATLAS data? Note the 3sig upper limits?', fontsize=11, fontweight='bold', va = 'center', y = 0.98)
fig.supxlabel('MJD', fontweight = 'bold', va = 'center', y = 0.01)
fig.supylabel('Apparent magnitude', fontweight = 'bold', va = 'center', x = 0.01)
plt.subplots_adjust(top=0.937, bottom=0.060, left=0.036, right=0.991, hspace=0.231, wspace=0.161)
savepath = "C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS/plots/light curves/ALL_ANT_lc.png"
plt.savefig(savepath, dpi=150)
plt.show() 




