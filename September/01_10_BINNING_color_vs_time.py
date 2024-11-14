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







# make an offset key for each transient so that their colours are easily visible
offset_lims = len(transient_names)/2
transient_offsets = np.arange(-offset_lims, offset_lims, 1.0) # offsets are in intervals of 0.5
#transient_offsets = np.zeros(len(transient_names))
trans_offset_key = dict(zip(transient_names, transient_offsets))
trans_offset_name = []
for i, t_os in enumerate(transient_offsets):
    if t_os >=0:
        t_os_name = f'{transient_names[i]} + {t_os}'

    else:
        t_os_name = f'{transient_names[i]} {t_os}'

    trans_offset_name.append(t_os_name)

trans_offset_name_key = dict(zip(transient_names, trans_offset_name))









check_binning_plot = True # if you want to sanity check the binning, but this will disrupt the plotting of the colour vs time for all transients plot
MJD_bin_size = 5
#plt.figure(figsize = (14, 7))
# averaging the ZTF_g and ZTF_r band every 10 days in MJD, so I'll bin the data every 10 days, like between 58100-58110 and average the 
# data within this bin for each band
for j, lc_df in enumerate(lc_df_list): # iterate through the different light curves
    if j==0:
        lc_ZTF_g = lc_df[lc_df['band'] == 'ZTF_g'].copy()
        lc_ZTF_r = lc_df[lc_df['band'] == 'ZTF_r'].copy()
        
        # this rounds the min_MJD present in ZTF_r data for the transient to the nearest 10, then goes to the bin below this to ensure that 
        # we capture the first data point, in case this rounded the first bin up. Since we need data for both ZTF_r and ZTF_g within each
        # MJD bin, it's fine to base the min MJD bin to start at on just the ZTF_r data, since even if we had ZTF_g data before this date, 
        # we wouldn't be able to pair it with data from ZTF_r to calculcte the color anyways
        MJD_bin_min = int( round(lc_ZTF_r['MJD'].min(), -1) - 10 )
        MJD_bin_max = int( round(lc_ZTF_r['MJD'].max(), -1) + 10 )
        MJD_bins = range(MJD_bin_min, MJD_bin_max + MJD_bin_size, MJD_bin_size) # create the bins

        # data frames for the ZTF_g and ZTF_r data binned in 10 day MJD bins
        lc_ZTF_r.loc[:,'MJD_bin'] = pd.cut(lc_ZTF_r['MJD'], MJD_bins)
        lc_ZTF_g.loc[:,'MJD_bin'] = pd.cut(lc_ZTF_g['MJD'], MJD_bins)

        # group by the bins and generate some aggregate data. We want the avg mag, propagated mag error, avg MJD, min MJD, max MJD
        # in order to get propagated mag error, need to sum the squares of the mag errs, then divide by the number of mag errors. Therefore, need a column 
        # which contains (mag_err)^2 so that I can sum it in the .agg() and also need to calculate the number of data points within each bin using .agg()
        lc_ZTF_r.loc[:,'magerr_sqr'] = (lc_ZTF_r['magerr'])**2
        lc_ZTF_g.loc[:,'magerr_sqr'] = (lc_ZTF_g['magerr'])**2

        binned_ZTF_r = lc_ZTF_r.groupby('MJD_bin', observed=False).agg(mean_mag = ('mag', 'mean'), 
                                                           sum_magerr_sqr = ('magerr_sqr', 'sum'), 
                                                           bin_count = ('magerr_sqr', 'count'),
                                                           mean_MJD = ('MJD', 'mean'), 
                                                           min_MJD = ('MJD', 'min'), 
                                                           max_MJD = ('MJD', 'max'))
        binned_ZTF_g = lc_ZTF_g.groupby('MJD_bin', observed=False).agg(mean_mag = ('mag', 'mean'), 
                                                           sum_magerr_sqr = ('magerr_sqr', 'sum'), 
                                                           bin_count = ('magerr_sqr', 'count'),
                                                           mean_MJD = ('MJD', 'mean'), 
                                                           min_MJD = ('MJD', 'min'), 
                                                           max_MJD = ('MJD', 'max'))
        
        # calculate the propagated magerr, I used the formula: propagated magerr = sqrt( sum(magerr^2) )/count
        binned_ZTF_r.loc[:,'propagated_magerr'] = np.sqrt(binned_ZTF_r['sum_magerr_sqr'])/binned_ZTF_r['bin_count']
        binned_ZTF_g.loc[:,'propagated_magerr'] = np.sqrt(binned_ZTF_g['sum_magerr_sqr'])/binned_ZTF_g['bin_count']
        binned_ZTF_r = binned_ZTF_r.drop(['sum_magerr_sqr', 'bin_count'], axis = 'columns')
        binned_ZTF_g = binned_ZTF_g.drop(['sum_magerr_sqr', 'bin_count'], axis = 'columns')

        # join the ZTF_r dataframe to the ZTF_g dataframe
        joined_ZTF_r_g = binned_ZTF_g.join(binned_ZTF_r, on = 'MJD_bin', lsuffix = '_g', rsuffix = '_r')

        max_MJD_column = []
        min_MJD_column = []
        for i in range(len(joined_ZTF_r_g['mean_mag_g'])):
            # min MJD - this finds the minimum MJD data point present within the MJD bin across both the ZTF_r and ZTF_g bands
            if joined_ZTF_r_g['min_MJD_g'].iloc[i] >= joined_ZTF_r_g['min_MJD_r'].iloc[i]:
                min_MJD = joined_ZTF_r_g['min_MJD_r'].iloc[i]

            else: 
                min_MJD = joined_ZTF_r_g['min_MJD_g'].iloc[i]


            # max MJD - this finds the maximum MJD data point present within the MJD bin across both the ZTF_r and ZTF_g bands
            if joined_ZTF_r_g['max_MJD_g'].iloc[i] >= joined_ZTF_r_g['max_MJD_r'].iloc[i]:
                max_MJD = joined_ZTF_r_g['max_MJD_g'].iloc[i]

            else: 
                max_MJD = joined_ZTF_r_g['max_MJD_r'].iloc[i]

            max_MJD_column.append(max_MJD)
            min_MJD_column.append(min_MJD)
        
        joined_ZTF_r_g.loc[:,'overall_min_MJD'] =  min_MJD_column
        joined_ZTF_r_g.loc[:,'overall_max_MJD'] =  max_MJD_column
        joined_ZTF_r_g.loc[:,'overall_mean_MJD'] = ( joined_ZTF_r_g['mean_MJD_g'] + joined_ZTF_r_g['mean_MJD_r'] )/2

        joined_ZTF_r_g.loc[:,'left_MJD_err'] = joined_ZTF_r_g['overall_mean_MJD'] - joined_ZTF_r_g['overall_min_MJD']
        joined_ZTF_r_g.loc[:,'right_MJD_err'] = joined_ZTF_r_g['overall_max_MJD'] - joined_ZTF_r_g['overall_mean_MJD']

        #sanity check plot that I have done all of this binning stuff correctly
        if check_binning_plot == True:
            plt.figure(figsize=(14, 7))
            # the actual data, unbinned
            plt.errorbar(lc_ZTF_g['MJD'], lc_ZTF_g['mag'], yerr = lc_ZTF_g['magerr'], linestyle = 'None', fmt='o', label = 'ZTF_g', alpha = 0.25, c='b')
            # the binned data I've created
            plt.errorbar(joined_ZTF_r_g['mean_MJD_g'], joined_ZTF_r_g['mean_mag_g'], yerr = joined_ZTF_r_g['propagated_magerr_g'], xerr = [(joined_ZTF_r_g['mean_MJD_g'] - joined_ZTF_r_g['min_MJD_g']), (joined_ZTF_r_g['max_MJD_g'] - joined_ZTF_r_g['mean_MJD_g'])],
                        linestyle = 'None', fmt='o', markeredgecolor = 'k', markeredgewidth = '1.0', label = 'ZTF_g binned', c = 'b')
            offset_ZTF_r = 0.0
            # the actual data, unbinned
            plt.errorbar(lc_ZTF_r['MJD'], (lc_ZTF_r['mag'] + offset_ZTF_r), yerr = lc_ZTF_r['magerr'], linestyle = 'None', fmt='o', label = f'ZTF_r + {offset_ZTF_r}', alpha = 0.25, c='r')
            # the binned data I've created
            plt.errorbar(joined_ZTF_r_g['mean_MJD_r'], (joined_ZTF_r_g['mean_mag_r'] + offset_ZTF_r), yerr = joined_ZTF_r_g['propagated_magerr_r'], xerr = [(joined_ZTF_r_g['mean_MJD_r'] - joined_ZTF_r_g['min_MJD_r']), (joined_ZTF_r_g['max_MJD_r'] - joined_ZTF_r_g['mean_MJD_r'])],
                        linestyle = 'None', fmt='o', markeredgecolor = 'k', markeredgewidth = '1.0', label = f'ZTF_r binned + {offset_ZTF_r}', c = 'r')

            plt.xlabel('MJD')
            plt.ylabel('Apparent magnitude')
            plt.gca().invert_yaxis()
            plt.grid()
            plt.legend()
            plt.title(f'{transient_names[j]}, MJD bins = {MJD_bin_size} days')
            plt.show()
        

"""         joined_ZTF_r_g = joined_ZTF_r_g.drop(['min_MJD_g', 'max_MJD_g', 'min_MJD_r', 'max_MJD_r', 'overall_min_MJD', 'overall_max_MJD'], axis = 'columns')

        # loop through the MJD bins, if an MJD bin has both ZTF_r and ZTF_g data, then do ZTF_g  - ZTF_r for B-V colour
        BV_colour = []
        BV_colour_err = []
        for i in range(len(joined_ZTF_r_g['mean_mag_g'])):
            if joined_ZTF_r_g['mean_mag_g'].iloc[i] == 'NaN' or joined_ZTF_r_g['mean_mag_r'].iloc[i] == 'NaN':
                BV_C = None
                BV_C_err = None

            else:
                BV_C = joined_ZTF_r_g['mean_mag_g'].iloc[i] - joined_ZTF_r_g['mean_mag_r'].iloc[i] # B-V color
                BV_C_err = np.sqrt((joined_ZTF_r_g['propagated_magerr_g'].iloc[i])**2 + (joined_ZTF_r_g['propagated_magerr_r'].iloc[i])**2) # propagate the error from average mag in g and r band to error in the B=V color
            
            BV_colour.append(BV_C)
            BV_colour_err.append(BV_C_err)

        transient = transient_names[j]

        # plotting
        plt.errorbar(joined_ZTF_r_g['overall_mean_MJD'], (BV_colour + trans_offset_key[transient]), yerr = BV_colour_err, xerr = [joined_ZTF_r_g['left_MJD_err'], joined_ZTF_r_g['right_MJD_err']],
                    linestyle = 'None', fmt='o', markeredgecolor = 'k', markeredgewidth = '0.5', label = trans_offset_name_key[transient], alpha = 0.8)
plt.xlabel(f'Mean MJD value in the {MJD_bin_size}-day MJD bin / days')
plt.ylabel('B-V colour / mag')
plt.yticks(transient_offsets)
plt.grid()
plt.legend()
plt.show() """
       











