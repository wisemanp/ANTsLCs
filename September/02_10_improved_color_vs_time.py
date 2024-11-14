import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# loading in the files
folder_path = "C:/Users/laure/OneDrive/Desktop/YoRiS/September/Phil's lightcurves 17_9" # folder path containing the light curve data files
lc_df_list = []
transient_names = []
list_of_bands = []
for file in os.listdir(folder_path): # file is a string of the file name such as 'file_name.dat'
    file_path = os.path.join(folder_path, file) 
    file_df = pd.read_csv(file_path, delimiter = ' ') # the lightcurve data in a dataframe
    lc_df_list.append(file_df)

    trans_name = file[:-7] # the name of the transient
    transient_names.append(trans_name)

    trans_bands = file_df['band'].unique() # the bands in which this transient has data for
    list_of_bands.append(trans_bands)


# ADD ANOTHER BIT OF CODE HERE TO IMPORT THE OTHER ANT DATA IN THE SAME WAY AND CONVERT THEM INTO DATAFRAMES
# COMBINE THE LIST OF DATAFRAMES/ JUST APPEND THE DATAFRAMES MADE FOR THE OTHER ANT DATA TO lc_df_list ALONG WITH THE BANDS AND TRANSIENT NAMES TO list_of_bands
# AND transient_names
# WITH list_of_bands, THIS WOULD ONLY WORK IF THE BANDS ARE NAMED THE SAME ACROSS ALL TRANSIENTS, E.G. THE NAME FOR ATLAS'S O BAND MUST BE CONSISTENT ACROSS ALL
# TRANSIENTS

#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################
want_colour_offsets = True # if you want to offset the color vs time data to make it easier to see ##############################################################


# make an offset key for each transient so that their colours are easily visible
if want_colour_offsets == True:
    offset_lims = len(transient_names)/2
    transient_offsets = np.arange(-offset_lims, offset_lims, 1.0) # offsets are in intervals of 0.5
else:
    transient_offsets = np.zeros(len(transient_names))

trans_offset_key = dict(zip(transient_names, transient_offsets))
trans_offset_name = []
for i, t_os in enumerate(transient_offsets):
    if t_os >=0:
        t_os_name = f'{transient_names[i]} + {t_os}'

    else:
        t_os_name = f'{transient_names[i]} {t_os}'

    trans_offset_name.append(t_os_name)

trans_offset_name_key = dict(zip(transient_names, trans_offset_name))

#################################################################################################################################################################
#################################################################################################################################################################
#################################################################################################################################################################


# WHEN I TALK ABOUT B_band AND V_band BELOW, I AM JUST REFERRING TO THE BLUER-BAND AND REDDER BAND, SO ZTF_g and ATLAS_c WOULD FALL UNDER B_band AND 
# ZTF_r AND ATLAS_o WOULD FALL UNDER V_band



check_binning_plot = False # if you want to sanity check the binning, but this will disrupt the plotting of the colour vs time for all transients plot
MJD_bin_size = 10
plt.figure(figsize = (14, 7))
# averaging the B_band and V_band band every 10 days in MJD, so I'll bin the data every 10 days, like between 58100-58110 and average the 
# data within this bin for each band
for i, lc_df in enumerate(lc_df_list): # iterate through the different light curves
    
    # checking first whether the transient has data in ZTF_g and ZTF_r band, if not, check for ATLAS_c and ATLAS_o, if not, then print a warning
    if 'ZTF_g' in list_of_bands[i] and 'ZTF_r' in list_of_bands[i]:
        B_band = 'ZTF_g'
        V_band = 'ZTF_r'

    elif 'ATLAS_c' in list_of_bands[i] and 'ATLAS_o' in list_of_bands[i]:
        B_band = 'ATLAS_c'
        V_band = 'ATLAS_o'

    else:
        print(f'WARNING: NONE OF "ZTF_g", "ZTF_r", "ATLAS_c", "ATLAS_o" PRESENT FOR TRANSIENT {transient_names[i]}')



    
    lc_B_band = lc_df[lc_df['band'] == B_band].copy()
    lc_V_band = lc_df[lc_df['band'] == V_band].copy()
    
    # this rounds the min_MJD present in V_band data for the transient to the nearest 10, then goes to the bin below this to ensure that 
    # we capture the first data point, in case this rounded the first bin up. Since we need data for both V_band and B_band within each
    # MJD bin, it's fine to base the min MJD bin to start at on just the V_band data, since even if we had B_band data before this date, 
    # we wouldn't be able to pair it with data from V_band to calculcte the color anyways
    MJD_bin_min = int( round(lc_V_band['MJD'].min(), -1) - 10 )
    MJD_bin_max = int( round(lc_V_band['MJD'].max(), -1) + 10 )
    MJD_bins = range(MJD_bin_min, MJD_bin_max + MJD_bin_size, MJD_bin_size) # create the bins

    # data frames for the B_band and V_band data binned in 10 day MJD bins
    lc_V_band.loc[:,'MJD_bin'] = pd.cut(lc_V_band['MJD'], MJD_bins)
    lc_B_band.loc[:,'MJD_bin'] = pd.cut(lc_B_band['MJD'], MJD_bins)

    # group by the bins and generate some aggregate data. We want the avg mag, propagated mag error, avg MJD, min MJD, max MJD
    # in order to get propagated mag error, need to sum the squares of the mag errs, then divide by the number of mag errors. Therefore, need a column 
    # which contains (mag_err)^2 so that I can sum it in the .agg() and also need to calculate the number of data points within each bin using .agg()
    lc_V_band.loc[:,'magerr_sqr'] = (lc_V_band['magerr'])**2
    lc_B_band.loc[:,'magerr_sqr'] = (lc_B_band['magerr'])**2

    binned_V_band = lc_V_band.groupby('MJD_bin', observed=False).agg(mean_mag = ('mag', 'mean'), 
                                                        sum_magerr_sqr = ('magerr_sqr', 'sum'), 
                                                        bin_count = ('magerr_sqr', 'count'),
                                                        mean_MJD = ('MJD', 'mean'), 
                                                        min_MJD = ('MJD', 'min'), 
                                                        max_MJD = ('MJD', 'max'))
    binned_B_band = lc_B_band.groupby('MJD_bin', observed=False).agg(mean_mag = ('mag', 'mean'), 
                                                        sum_magerr_sqr = ('magerr_sqr', 'sum'), 
                                                        bin_count = ('magerr_sqr', 'count'),
                                                        mean_MJD = ('MJD', 'mean'), 
                                                        min_MJD = ('MJD', 'min'), 
                                                        max_MJD = ('MJD', 'max'))
    
    # calculate the propagated magerr, I used the formula: propagated magerr = sqrt( sum(magerr^2) )/count
    binned_V_band.loc[:,'propagated_magerr'] = np.sqrt(binned_V_band['sum_magerr_sqr'])/binned_V_band['bin_count']
    binned_B_band.loc[:,'propagated_magerr'] = np.sqrt(binned_B_band['sum_magerr_sqr'])/binned_B_band['bin_count']
    binned_V_band = binned_V_band.drop(['sum_magerr_sqr', 'bin_count'], axis = 'columns')
    binned_B_band = binned_B_band.drop(['sum_magerr_sqr', 'bin_count'], axis = 'columns')

    # join the V_band dataframe to the B_band dataframe
    joined_VB_band = binned_B_band.join(binned_V_band, on = 'MJD_bin', lsuffix = '_B', rsuffix = '_V')


    max_MJD_column = []
    min_MJD_column = []
    for j in range(len(joined_VB_band['mean_mag_B'])):
        # min MJD - this finds the minimum MJD data point present within the MJD bin across both the V_band and B_band bands
        if joined_VB_band['min_MJD_B'].iloc[j] >= joined_VB_band['min_MJD_V'].iloc[j]:
            min_MJD = joined_VB_band['min_MJD_V'].iloc[j]

        else: 
            min_MJD = joined_VB_band['min_MJD_B'].iloc[j]


        # max MJD - this finds the maximum MJD data point present within the MJD bin across both the V_band and B_band bands
        if joined_VB_band['max_MJD_B'].iloc[j] >= joined_VB_band['max_MJD_V'].iloc[j]:
            max_MJD = joined_VB_band['max_MJD_B'].iloc[j]

        else: 
            max_MJD = joined_VB_band['max_MJD_V'].iloc[j]

        max_MJD_column.append(max_MJD)
        min_MJD_column.append(min_MJD)
    


    joined_VB_band.loc[:,'overall_min_MJD'] =  min_MJD_column
    joined_VB_band.loc[:,'overall_max_MJD'] =  max_MJD_column
    joined_VB_band.loc[:,'overall_mean_MJD'] = ( joined_VB_band['mean_MJD_B'] + joined_VB_band['mean_MJD_V'] )/2

    joined_VB_band.loc[:,'left_MJD_err'] = joined_VB_band['overall_mean_MJD'] - joined_VB_band['overall_min_MJD']
    joined_VB_band.loc[:,'right_MJD_err'] = joined_VB_band['overall_max_MJD'] - joined_VB_band['overall_mean_MJD']

    #sanity check plot that I have done all of this binning stuff correctly
    if check_binning_plot == True:
        plt.figure(figsize=(14, 7))
        # the actual data, unbinned
        plt.errorbar(lc_B_band['MJD'], lc_B_band['mag'], yerr = lc_B_band['magerr'], linestyle = 'None', fmt='o', label = B_band, alpha = 0.25, c='b')
        # the binned data I've created
        plt.errorbar(joined_VB_band['mean_MJD_B'], joined_VB_band['mean_mag_B'], yerr = joined_VB_band['propagated_magerr_B'], xerr = [(joined_VB_band['mean_MJD_B'] - joined_VB_band['min_MJD_B']), (joined_VB_band['max_MJD_B'] - joined_VB_band['mean_MJD_B'])],
                    linestyle = 'None', fmt='o', markeredgecolor = 'k', markeredgewidth = '1.0', label = f'{B_band} binned', c = 'b')
        offset_V_band = 2.0
        # the actual data, unbinned
        plt.errorbar(lc_V_band['MJD'], (lc_V_band['mag'] + offset_V_band), yerr = lc_V_band['magerr'], linestyle = 'None', fmt='o', label = f'{V_band} + {offset_V_band}', alpha = 0.25, c='r')
        # the binned data I've created
        plt.errorbar(joined_VB_band['mean_MJD_V'], (joined_VB_band['mean_mag_V'] + offset_V_band), yerr = joined_VB_band['propagated_magerr_V'], xerr = [(joined_VB_band['mean_MJD_V'] - joined_VB_band['min_MJD_V']), (joined_VB_band['max_MJD_V'] - joined_VB_band['mean_MJD_V'])],
                    linestyle = 'None', fmt='o', markeredgecolor = 'k', markeredgewidth = '1.0', label = f'{V_band} binned + {offset_V_band}', c = 'r')

        plt.xlabel('MJD')
        plt.ylabel('Apparent magnitude')
        plt.gca().invert_yaxis()
        plt.grid()
        plt.legend()
        plt.title(f'{transient_names[j]}, MJD bins = {MJD_bin_size} days')
        plt.show()
    

    joined_VB_band = joined_VB_band.drop(['min_MJD_B', 'max_MJD_B', 'min_MJD_V', 'max_MJD_V', 'overall_min_MJD', 'overall_max_MJD'], axis = 'columns')

    # loop through the MJD bins, if an MJD bin has both V_band and B_band data, then do B_band - V_band for B-V colour
    BV_colour = []
    BV_colour_err = []
    for j in range(len(joined_VB_band['mean_mag_B'])):
        if joined_VB_band['mean_mag_B'].iloc[j] == 'NaN' or joined_VB_band['mean_mag_V'].iloc[j] == 'NaN':
            BV_C = None
            BV_C_err = None

        else:
            BV_C = joined_VB_band['mean_mag_B'].iloc[j] - joined_VB_band['mean_mag_V'].iloc[j] # B-V color
            BV_C_err = np.sqrt((joined_VB_band['propagated_magerr_B'].iloc[j])**2 + (joined_VB_band['propagated_magerr_V'].iloc[j])**2) # propagate the error from average mag in g and r band to error in the B=V color
        
        BV_colour.append(BV_C)
        BV_colour_err.append(BV_C_err)


    transient = transient_names[i]
    # plotting
    plt.errorbar(joined_VB_band['overall_mean_MJD'], (BV_colour + trans_offset_key[transient]), yerr = BV_colour_err, xerr = [joined_VB_band['left_MJD_err'], joined_VB_band['right_MJD_err']],
                linestyle = 'None', fmt='o', markeredgecolor = 'k', markeredgewidth = '0.5', label = trans_offset_name_key[transient], alpha = 0.8)
plt.xlabel(f'Mean MJD value in the {MJD_bin_size}-day MJD bin / days')
plt.ylabel('B-V colour (ZTF_g - ZTF_r for now) / mag')
if want_colour_offsets == True:
    plt.yticks(np.arange(-(offset_lims + 1.0), (offset_lims + 1.0), 0.5))
plt.grid()
plt.legend()
plt.title('yerrs are questionable, they dont really account for the std dev of datapoints which were averaged within each MJD bin\n try using mean and std error instead? \n CHANGE X AXIS TO DAYS SINCE PEAK')
plt.show()
       


