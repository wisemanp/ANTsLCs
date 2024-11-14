
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS") # this allows us to access the plotting_preferences.py file 
from November.plotting_preferences import band_colour_dict, band_marker_dict, band_offset_dict, band_offset_label_dict


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#
# PHIL GAVE ME SOME IMPROVED DATA FILES ON ABODAPS AND AMRJAR, SO LETS COMPARE BETWEEN THE OLD AND NEW FILES TO MAKE SURE THAT I'VE LOADED THEM IN CORRECTLY.
#
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# loading in the files, this contains the old files for abodaps and amrjar
folder_path = "C:/Users/laure/OneDrive/Desktop/YoRiS Desktop/YoRiS Data/modified Phil's lightcurves" # I added stuff like days since peak + removed outliers
lc_df_list = []
transient_names = []
list_of_bands = []
for file in os.listdir(folder_path): # file is a string of the file name such as 'file_name.dat'
    file_path = os.path.join(folder_path, file) 
    file_df = pd.read_csv(file_path, delimiter = ',') # the lightcurve data in a dataframe
    

    trans_name = file[:-7] # the name of the transient
    if trans_name == 'ZTF19aamrjar' or trans_name == 'ZTF20abodaps':
        lc_df_list.append(file_df)
        print(file_df)
        name = trans_name+'_old'
        transient_names.append(name)

        trans_bands = file_df['band'].unique() # the bands in which this transient has data for
        list_of_bands.append(trans_bands)



folder_path = "C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS Data/Phil's lcs updated modified" # I added stuff like days since peak + removed outliers
for file in os.listdir(folder_path): # file is a string of the file name such as 'file_name.dat'
    file_path = os.path.join(folder_path, file) 
    file_df = pd.read_csv(file_path, delimiter = ',') # the lightcurve data in a dataframe
    

    trans_name = file[:-4] # the name of the transient
    lc_df_list.append(file_df)
    print(file_df)
    name = trans_name+'_new'
    transient_names.append(name)

    trans_bands = file_df['band'].unique() # the bands in which this transient has data for
    list_of_bands.append(trans_bands)


fig, axs = plt.subplots(2, 2, figsize = (16, 7))
axs = axs.ravel()
label_list = []
marker_list = []
for i, df in enumerate(lc_df_list):
    ANT_bands = list_of_bands[i]
    ANT_name = transient_names[i]
    ax = axs[i]

    
    for band in ANT_bands:
        band_df = df[df['band'] == band].copy()
        band_colour = band_colour_dict[band]
        band_marker = band_marker_dict[band]
        band_offset = band_offset_dict[band]
        band_label = band_offset_label_dict[band]

        h = ax.errorbar(band_df['MJD'], (band_df['mag'] + band_offset), yerr = band_df['magerr'], color = band_colour, fmt = band_marker, label = band_label, 
                    linestyle = 'None', markeredgecolor = 'k', markeredgewidth = '0.5')
        
        marker = h[0]
        label = band_label
        if label not in label_list:
            label_list.append(label)
            marker_list.append(marker)
        
    ax.set_title(ANT_name, fontweight = 'bold')
    ax.axvline(x = df['peak_MJD'].iloc[0], c = 'k', label = 'peak')
    ax.grid()

    ax.invert_yaxis()

fig.supxlabel('MJD')
fig.supylabel('mag')
fig.legend(marker_list, label_list)
fig.subplots_adjust(top=0.93,
                    bottom=0.08,
                    left=0.065,
                    right=0.875,
                    hspace=0.2,
                    wspace=0.115)
plt.show()



    