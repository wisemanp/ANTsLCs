import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from astropy.stats import bayesian_blocks
sys.path.append("C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS") # this allows us to access the plotting_preferences.py file 
from plotting_preferences import band_colour_dict, band_marker_dict, band_offset_dict
from functions import bin_lc, load_ANT_data, ANT_data_L_rf



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

for i in range(len(lc_df_list)):
    print(transient_names[i], '   ', list_of_bands[i])
print()
print()
for j, ANT_df in enumerate(lc_df_list):
    print()
    ANT_name = transient_names[j]
    ANT_df = lc_df_list[j]
    ANT_bands = list_of_bands[j]

    fig, axs = plt.subplots(3, 4, figsize = (16, 7))
    axs = axs.ravel()
    for i, ax in enumerate(axs):
        # break the loop if we've covered all of the bands present in the ANT data
        if i > len(ANT_bands) - 1:
            break
        band = ANT_bands[i]
        band_df = ANT_df[ANT_df['band'] == band].copy()
        print(ANT_name, band)
        # band plotting preferences
        band_color = band_colour_dict[band]
        band_marker = band_marker_dict[band]
        band_offset = band_offset_dict[band]
        band_offset_name = f'{band} + {band_offset}'

        # bayesian bins 
        edges = bayesian_blocks(t = band_df['MJD'], x = band_df['mag'], fitness = 'measures', sigma = band_df['magerr'])

        ax.errorbar(band_df['MJD'], band_df['mag'], yerr = band_df['magerr'], c = band_color, fmt = band_marker, linestyle = 'None', 
                    markeredgecolor = 'k', markeredgewidth = '0.5', label = band_offset_name)
        for edge in edges:
            ax.axvline(x = edge, c = 'k', label = 'Bayesian bin' if edge == edges[0] and i==0 else None)
        ax.invert_yaxis()

    fig.supxlabel('MJD')
    fig.supylabel('mag')
    fig.suptitle(ANT_name+' Bayesian bins')
    fig.legend()
    fig.subplots_adjust(top=0.945,
                        bottom=0.075,
                        left=0.065,
                        right=0.985,
                        hspace=0.15,
                        wspace=0.11)
    plt.show()
