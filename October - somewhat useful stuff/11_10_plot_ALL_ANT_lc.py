import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
folder_path = "C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS Data/modified Phil's lightcurves" # folder path containing the light curve data files
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
#print(transient_names)


#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
# QUICKLY TEST THE BINNING FUNCTION
"""idx = transient_names.index('ASASSN-18jd')
jd_lc = lc_df_list[idx]
jd_ATLAS_o = jd_lc[jd_lc['band'] == 'ATLAS_o'].copy()
jd_ATLAS_c = jd_lc[jd_lc['band'] == 'ATLAS_c'].copy()

binsize = 10
binned_jd_o = bin_data(jd_ATLAS_o, binsize)
binned_jd_c = bin_data(jd_ATLAS_c, binsize)

print(jd_ATLAS_c)
print()
print(binned_jd_c)
print()

plt.figure(figsize = (14, 7))
plt.errorbar(binned_jd_o['MJD'], binned_jd_o['mag'], yerr = binned_jd_o['magerr'], xerr = binned_jd_o['MJD_err'], fmt = 'o', label = 'binned ATLAS_o', c='r')
plt.xlabel('MJD/days')
plt.ylabel('mag')
plt.title('Testing my binning function is working') """



#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
# make a colour key for each band, so that all plots use the same colour key for each band for easier comparison
free_list_of_bands = [element for sublist in list_of_bands for element in sublist]
free_list_of_bands = np.array(free_list_of_bands)
unique_bands = np.unique(free_list_of_bands)
""" print()
print(unique_bands)
print()
print(len(unique_bands)) """
#band_plot_colours = ['k', 'y', 'm', 'c', 'r', 'g', 'b', 'darkviolet', 'orange', 'silver', 'lime', 'darkturquoise', 'mediumspringgreen', 'maroon', 'tomato', 'lightpink']
#band_plot_colours = ['#d62728', '#8a2be2', '#1f77b4', '#e377c2', '#17becf', '#00ced1', '#2ca02c', '#228b22', '#b57edc', '#ff00ff', '#1f3c88', '#ff6347', '#ffd700', '#ff7f0e', '#7f7f7f']
band_plot_colours = ['#1f77b4', '#ff7f0e','#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#ff00ff', '#ff6347', '#00ced1', 'purple', '#1f3c88']
a=0
band_marker_colour = [] # this for loop means that I can iterate through the same set of colours and I'll just use different shapes for the markers instead

for i in range(len(unique_bands)):
    band_marker_colour.append(band_plot_colours[a])
    a += 1

    if a >= len(band_plot_colours):
        a=0


""" print(band_marker_colour[0:15])
print(band_marker_colour[15:30])
print(band_marker_colour[30:45])
print(band_marker_colour[45:])
print()

print(band_marker_colour) """
band_colour_key = dict(zip(unique_bands, band_marker_colour)) # a dictionary which maps each band to its given color





#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
# make a shape key for each band
band_plot_markers = []
for i in range(len(unique_bands)):
    if i<15:
        band_plot_markers.append('o')

    elif i>=15 and i< 30:
        band_plot_markers.append('s')

    elif i >=30 and i< 45:
        band_plot_markers.append('*')

    else:
        band_plot_markers.append('^')
band_marker_key = dict(zip(unique_bands, band_plot_markers))

#print((band_marker_key))




#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
# the labels for the offset bands in the legend, e.g. 'WISE_1 + 3' if the WISE_1 band is offset by +3 mag

band_offset_key = {'ATLAS_c': 1.5, 
                    'ATLAS_o':-2.5, 
                    'PS_i': -1.5, 
                    'PS_w': -1.0, 
                    'PS_y': 0.0, 
                    'PS_z': 0.0, 
                    'UVOT_B': 0.0, 
                    'UVOT_U': -2.5,
                    'UVOT_UVM1': -1.5, 
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
                    'Swift_B': 0.0, 
                    'Swift_U': 0.0, 
                    'Swift_V': 0.0, 
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
                    'UVM2': 0.0, 
                    'UVOT_UVW1': 0.0}




band_offset_name = []
for b in unique_bands:
    print(b, band_offset_key[b])
    if band_offset_key[b] >= 0:
        offset_name = b +' + '+str(band_offset_key[b])

    elif band_offset_key[b] < 0:
        offset_name = b +' '+str(band_offset_key[b])

    band_offset_name.append(offset_name)

band_offset_name_key = dict(zip(unique_bands, band_offset_name))



#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
# limiting the MJD to plot the light curve over - only plot the more interesting part
# THE ORDER OF TRANSIENTS: 
#            'ZTF18aczpgwm', 'ZTF19aailpwl', 'ZTF19aamrjar', 'ZTF19aatubsj', 'ZTF20aanxcpf', 'ZTF20abgxlut', 'ZTF20abodaps', 'ZTF20abrbeie', 'ZTF20acvfraq', 'ZTF21abxowzx', 'ZTF22aadesap'
MJD_xlims = [(58350, 60250), (58450, 60050), (58340, 60250), (58550, 60250), (58400, 60320), (58950, 59500), (58450, 60300), (59000, 59950), (58700, 60200), (59400, 60250), (59650, 60400)]
no_None_xlim = len(transient_names) - len(MJD_xlims)
for i in range(no_None_xlim): # for now, append None to the x lims so that I can plot them all FOR NOW LALLALALA
    MJD_xlims.append(None) 

#print(MJD_xlims)

 
#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
want_separate_plots = False
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
            plt.errorbar(band_data['MJD'], band_offset_data, yerr = band_data['magerr'], fmt = 'o', c = band_color, label = band_offset_name_key[band], linestyle = 'None',
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
    MJD_xlim = MJD_xlims[i] # the MJD that the transient occur over
    print(i, transient_names[i])
    print(lc_df.head())
    print()


    for band in list_of_bands[i]: # looking at the list of bands that are present for this particular lightcurve
        band_color = band_colour_key[band]
        band_marker = band_marker_key[band]
        band_data = lc_df[lc_df['band']==band].copy() # filter for the lightcurve data in this specific band
        band_data = band_data[band_data['magerr'] < 2.0].copy() # filter out the data points with HUGE errors BINNED DATA POINTS WITH HUGE MAGERR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11
        #  adding x errors to the plot
        if 'MJD_err' in band_data.columns:
            xerr = band_data['MJD_err']

        elif 'MJD_lower_err' in band_data.columns and 'MJD_upper_err' in band_data.columns:
            xerr = [band_data['MJD_lower_err'], band_data['MJD_upper_err']]

        else:
            xerr = [0]*len(band_data['MJD'])

        band_offset_data = np.array(band_data['mag']) + band_offset_key[band] # offsetting the band to make it easier to see on the plot
        h = ax.errorbar(band_data['MJD'], band_offset_data, yerr = band_data['magerr'], xerr = xerr, fmt = band_marker, c = band_color, label = band_offset_name_key[band], linestyle = 'None',
                     markeredgecolor = 'k', markeredgewidth = '0.5', markersize = 5)
        handle = h[0]
        label = band_offset_name_key[band]
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




