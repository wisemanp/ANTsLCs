import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#############################################################################################################################################################################################################


def weighted_mean(data, errors):
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    if not isinstance(errors, np.ndarray):
        errors = np.array(errors)
    
    if len(errors) == 0: # if the bin has no data within it, then the weighted mean and its error = NaN
        wm = pd.NA
        wm_err = pd.NA

    else: # if the bin has data within it, then take the weighted mean
        weights = 1/(errors**2)
        wm = np.sum(data * weights) / (np.sum(weights))
        wm_err = np.sqrt( 1/(np.sum(weights)) )

    return wm, wm_err



#############################################################################################################################################################################################################


def bin_data(bands_df, MJD_binsize, drop_na_bins = True):
    """
    Bins data for a given band - calculates the weighted mean of the flux, its error, the weighted mean of the MJD and the upper and lower errors for each bin
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
    bands_df['MJD_bin'] = pd.cut(bands_df['MJD'], MJD_bins)

    bands_binned_df = bands_df.groupby('MJD_bin', observed = drop_na_bins).apply(lambda g: pd.Series({
                                                        'wm_flux': weighted_mean(g['flux'], g['flux_err'])[0],
                                                        'wm_flux_err': weighted_mean(g['flux'], g['flux_err'])[1],
                                                        'wm_MJD': weighted_mean(g['MJD'], g['flux_err'])[0],
                                                        'BAD_mean_mag': g['mag'].mean(),
                                                        'BAD_std_dev_mag': g['mag'].std(),
                                                        'BAD_mean_MJD': g['MJD'].mean(),
                                                        'count': g['mag'].count(),
                                                        'min_MJD': g['MJD'].min(),
                                                        'max_MJD': g['MJD'].max()
                                                        }))
    

    bands_binned_df['BAD_mag_std_err'] = bands_binned_df['BAD_std_dev_mag'] / np.sqrt(bands_binned_df['count']) # std err of mean mag

    wm_lower_MJD_err = []
    wm_upper_MJD_err = []
    for i in range(len(bands_binned_df['wm_flux'])):
        if bands_binned_df['count'].iloc[i] == 1:
            MJD_lower = 0.0
            MJD_upper = 0.0
    
        else:
            MJD_lower = bands_binned_df['wm_MJD'].iloc[i] - bands_binned_df['min_MJD'].iloc[i]
            MJD_upper = bands_binned_df['max_MJD'].iloc[i] - bands_binned_df['wm_MJD'].iloc[i]

        wm_lower_MJD_err.append(MJD_lower)
        wm_upper_MJD_err.append(MJD_upper)

    bands_binned_df['wm_lower_MJD_err'] = wm_lower_MJD_err
    bands_binned_df['wm_upper_MJD_err'] = wm_upper_MJD_err

    #bands_binned_df = bands_binned_df.rename(columns = {'mean_MJD' : 'MJD', 'mean_mag' : 'mag', 'mag_std_err':'magerr'}) # rename columns to the usual naming system
    #bands_binned_df = bands_binned_df.drop(['std_dev_mag', 'count'], axis = 1) # drop unnecessary columns
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


#########################################################################################
#########################################################################################
ant_name = 'ASASSN-18jd'
#ant_name = 'Gaia18cdj'
band1_name = 'ATLAS_o'
band2_name = 'ATLAS_c'
MJD_binsize = 10
#########################################################################################
#########################################################################################
idx = transient_names.index(ant_name)
ant = lc_df_list[idx]

# filter the ANT light curve for each band
band1= ant[ant['band'] ==  band1_name].copy()
band2 = ant[ant['band'] == band2_name].copy()

# binning the data
band1_bin = bin_data(band1, MJD_binsize)
band2_bin = bin_data(band2, MJD_binsize)
print(band1_bin)

# plot
band2_offset = 250
plt.figure(figsize = (16, 7.5))
plt.errorbar(band1['MJD'], band1['flux'], yerr = band1['flux_err'], fmt='o', c = 'green', linestyle = 'None', markeredgecolor = 'k', markeredgewidth = 0.5, 
             markersize = 5, alpha = 0.3, label = 'un-binned ATLAS_o')
plt.errorbar(band1_bin['wm_MJD'], band1_bin['wm_flux'], yerr = band1_bin['wm_flux_err'], xerr = [band1_bin['wm_lower_MJD_err'], band1_bin['wm_upper_MJD_err']], fmt='o', 
             c = '#008ae6', linestyle = 'None', markeredgecolor = 'k', markeredgewidth = 1.0, markersize = 5, alpha = 1.0, label = 'binned ATLAS_o' )


plt.errorbar(band2['MJD'], (band2['flux'] + band2_offset), yerr = band2['flux_err'], fmt='o', c = 'red', linestyle = 'None', markeredgecolor = 'k', markeredgewidth = 0.5, 
             markersize = 5, alpha = 0.3, label = f'un-binned ATLAS_c + {band2_offset}')
plt.errorbar(band2_bin['wm_MJD'], (band2_bin['wm_flux'] + band2_offset), yerr = band2_bin['wm_flux_err'], xerr = [band2_bin['wm_lower_MJD_err'], band2_bin['wm_upper_MJD_err']], fmt='o', 
             c = '#e65c00', linestyle = 'None', markeredgecolor = 'k', markeredgewidth = 1.0, markersize = 5, alpha = 1.0, label = f'binned ATLAS_c + {band2_offset}' )

plt.xlabel(f'MJD ({MJD_binsize} day bins)')
plt.ylabel('Flux / uJy')
plt.ylim((-100, 2000))
plt.grid()
plt.legend()
plt.title(f'ANT = {ant_name}. YLIM SHOULDNT BE NECESSARY - THIS MEANS THAT OUR DATA PROBS NEEDS MORE SORTING OUT')
plt.show()


