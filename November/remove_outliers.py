import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from astropy.stats import bayesian_blocks
sys.path.append("C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS") # this allows us to access the plotting_preferences.py file 
from November.plotting_preferences import band_colour_dict, band_marker_dict, band_offset_dict



print()
print()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# THIS IS THE FILE WHICH I USE TO GATHER TOGETHER THE DATA FOR EACH OF THE ANTS - E.G. IF AN ANT HAS DATA FROM GAIA AND ATLAS, I WOULD 
# COMBINE THE GAIA AND ATLAS DATA INTO ONE FILE HERE, AND ALSO DO ALL DATA CLEANING IN THIS CODE. 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



MAGERR_LIM = 2.0 # if magerr is greater than this value, get rid of the datapoint MAYBE THIS ISN'T ACTUALLY NECESSARY........ ESPECIALLY SINCE WE HAVE A FLUXERR RATIO LIM

MIN_MAGERR = 0.00001 # if the magerr is less than this value, then remove the datapoint as there must be something wrong with it to have such a small/NO error bar

SIGMA_OUTLIER_LIM = 5.0 # this is being used just for Phil's data right now. The lightcurve is separated by band and binned up. If (weighted_mean_mag - mag)/magerr > 5SIGMA_OUTLIER_LIM, 
                        # then this datapoint is considered an outlier and is removed

FLUXERR_RATIO_LIM = 2.0 # if flux/flux_err < FLUXERR_RATIO_LIM then get rid of the data point

# ALSO GET RID OF ANY DATA FOR WHICH   mag >= 3sig_upper_lim or 5sig_upper_lim

# ALSO GET RID OF ATLAS DATA FOR WHICH tphot err =! 0

ATLAS_CHI_N_UPLIM = 3.0

ATLAS_CHI_N_LOWLIM = 0.2




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


def bin_data(all_bands_df, MJD_binsize, drop_na_bins = True):
    """
    Bins data for a given band - calculates the weighted mean of the flux, its error, the weighted mean of the MJD and the upper and lower errors for each bin
    """
    bands_present = all_bands_df['band'].unique() # bands present for the ANT
    binned_bands_dfs = [] # a list of the dataframes containing each individual datapoint for the band with their MJD, mag etc as well as the weighted mean mag within its group and more
    for band in bands_present:
        bands_df = all_bands_df[all_bands_df['band'] == band].copy() 

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
                                                            'wm_mag': weighted_mean(g['mag'], g['magerr'])[0],
                                                            'wm_mag_err': weighted_mean(g['mag'], g['magerr'])[1],
                                                            'count': g['mag'].count(), 
                                                            'wm_MJD': weighted_mean(g['MJD'], g['magerr'])[0],
                                                            'max_MJD': g['MJD'].max(), 
                                                            'min_MJD': g['MJD'].min(), 
                                                            'BAD_mean_mag': g['mag'].mean(),
                                                            'BAD_std_mag': g['mag'].std()
                                                            })).reset_index()
        
        # make the lower and upper MJD errors for plotting the weighted mean mag and therefore the weighted mean MJD, as well as the error on the weighted mean MJD being the range of MJD values within the bin
        MJD_lower_err_list = []
        MJD_upper_err_list = []
        MJD_bin_min_list = [] # these are errorbars which, when plotting wm_mag vs am_MJD, if we use these as the xerrs, then it should show how we cover all of the relevant MJD space with bins
        MJD_bin_max_list = []
        for i in range(len(bands_binned_df)):
            if bands_binned_df['count'].iloc[i] == 1:
                MJD_lower_err = 0.0
                MJD_upper_err = 0.0

            else:
                MJD_lower_err = np.abs(bands_binned_df['wm_MJD'].iloc[i] - bands_binned_df['min_MJD'].iloc[i]) # take the absolute value because there are instances where =-7e-14
                MJD_upper_err = np.abs(bands_binned_df['max_MJD'].iloc[i] - bands_binned_df['wm_MJD'].iloc[i]) 
            
            MJD_lower_err_list.append(MJD_lower_err)
            MJD_upper_err_list.append(MJD_upper_err)

            bin = bands_binned_df['MJD_bin'].iloc[i]
            min_MJD_bin = bin.left
            max_MJD_bin = bin.right
            wm_MJD = bands_binned_df['wm_MJD'].iloc[i]
            min_MJD_bin_errbar = abs(wm_MJD - min_MJD_bin) # take the mag in case we get something like -6e-14
            max_MJD_bin_errbar = abs(max_MJD_bin - wm_MJD)

            MJD_bin_min_list.append(min_MJD_bin_errbar)
            MJD_bin_max_list.append(max_MJD_bin_errbar)

        bands_binned_df['MJD_lower_err'] = MJD_lower_err_list 
        bands_binned_df['MJD_upper_err'] = MJD_upper_err_list
        bands_binned_df['MJD_bin_min'] = MJD_bin_min_list
        bands_binned_df['MJD_bin_max'] = MJD_bin_max_list

        
        bands_binned_df = bands_binned_df.drop(columns = ['max_MJD', 'min_MJD'])
        merge_band_df = bands_df.merge(bands_binned_df, on = 'MJD_bin', how = 'left') # merge to the whole band dataframe so that each datapoint has a bin associated with it
                                                                                    #   along with the wm mag, wm mag err and count within this bin
        merge_band_df['sigma_dist'] = np.abs( (merge_band_df['wm_mag'] - merge_band_df['mag']) / merge_band_df['magerr'] )# calculate how many sigma the data point is away from the wm mag
        binned_bands_dfs.append(merge_band_df)
        
    allband_merge_df = pd.concat(binned_bands_dfs, ignore_index = True)

    return allband_merge_df



##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
# band_colour_dict, band_marker_dict the plotting dictionaries

# ADDING TIME SINCE PEAK INTO PHIL'S ANT LIGHTCURVE DATA AND REMOVING OUTLIERS
# loading in the files
folder_path = "C:/Users/laure/OneDrive/Desktop/YoRiS Desktop/YoRiS Data/modified Phil's lightcurves" # folder path containing the light curve data files
for file in os.listdir(folder_path): # file is a string of the file name such as 'file_name.dat'
    file_path = os.path.join(folder_path, file) 
    file_df = pd.read_csv(file_path, delimiter = ',') # the lightcurve data in a dataframe
    ANT_name = file[:-7]

    #if ANT_name == 'ZTF18aczpgwm':
    bands_present = file_df['band'].unique()

    MJD_binsize = 10
    binned_df = bin_data(file_df, MJD_binsize)
    b_pres = binned_df['band'].unique()
    print()
    print('NO BAND DATAPOINTS IN THE BINNED DF')
    for b in b_pres:
        print(f'band = {b},    no datapoints in BINNED DF = {len(binned_df["MJD"][binned_df["band"] == b])},     no datapoints in file_df = {len(file_df["MJD"][file_df["band"] == b])}')


    cleaned_df = binned_df[binned_df['sigma_dist'] < SIGMA_OUTLIER_LIM].copy()
    print(cleaned_df.head(50))




    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # NORMAL LIGHT CURVE PLOT WITH BINS AND 'OUTLIERS' SHOWN
    want_plots = True
    #if ANT_name == 'ZTF18aczpgwm':
    if want_plots == True:
        fig = plt.figure(figsize = (16, 7))
        for b in b_pres:
            cleaned_band = cleaned_df[cleaned_df['band'] == b].copy()
            uncleaned_band = binned_df[binned_df['band'] == b].copy()
            #print(uncleaned_band.head(50))
            band_colour = band_colour_dict[b]
            band_marker = band_marker_dict[b]
            band_offset = (band_offset_dict[b])
            if b == 'ZTF_g':
                uncleaned_label = 'Uncleaned data'
                bin_label = 'y = wm mag, \nxerr = MJD bin'
            else:
                uncleaned_label = None
                bin_label = None
            band_offset_name = b+' + '+str(band_offset) # if band_offset is negative, this will show as 'band + -2.0' or something, but I just wanted to make this quickly so its fine for now
            plt.errorbar(uncleaned_band['MJD'], (uncleaned_band['mag'] + band_offset), yerr = uncleaned_band['magerr'], c = 'k', markersize = 5, label = uncleaned_label, 
                        linestyle = 'None', fmt = 'o')
            plt.errorbar(cleaned_band['MJD'], (cleaned_band['mag'] + band_offset), yerr = cleaned_band['magerr'], 
                            c = band_colour, fmt = band_marker, label = band_offset_name, linestyle = 'None', markeredgecolor = 'k', markeredgewidth = '0.5')
            plt.errorbar(uncleaned_band['wm_MJD'], (uncleaned_band['wm_mag'] + band_offset), xerr = (uncleaned_band['MJD_bin_min'], uncleaned_band['MJD_bin_max']), c = 'k', linestyle = 'None', label = bin_label)
            #plt.errorbar(cleaned_band['wm_MJD'], (cleaned_band['wm_mag'] + band_offset), yerr = cleaned_band['wm_mag_err'], xerr = (cleaned_band['MJD_lower_err'], cleaned_band['MJD_upper_err']), 
            #            fmt = '*', c = band_colour, markersize = 8, label = f'{band_offset_name}  WM', markeredgecolor = 'k', markeredgewidth = '1.0')
            

        plt.xlabel('MJD')
        plt.ylabel('mag')
        fig.gca().invert_yaxis()
        plt.legend()
        plt.grid()
        plt.title(str(ANT_name)+' THIS OUTLIER REMOVAL DOESNT WORK YET - ERRORS ARE TOOOOOOO SMALL FOR THE VARIABILITY...???? \n MJD binsize = '+str(MJD_binsize)+' SIGMA_OUTLIER_LIM = '+str(SIGMA_OUTLIER_LIM), fontsize = 15, fontweight = 'bold')
        #plt.show()





    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # MAKE A SEPARATE LIGHTCURVE FOR EACH BAND WITHIN EACH ANT FOR BETTER VISIBILITY
    subplot_lightcurve_separate_bands = False
    if subplot_lightcurve_separate_bands == True:
            fig, axs = plt.subplots(3, 4, figsize = (16, 7))
            axs = axs.ravel()
            for i, ax in enumerate(axs):
                if i > (len(b_pres) - 1):
                    break

                band = b_pres[i]
                cleaned_band = cleaned_df[cleaned_df['band'] == band].copy()
                uncleaned_band = binned_df[binned_df['band'] == band].copy()
                file_df_banddata = file_df[file_df['band'] == band].copy()
                band_colour = band_colour_dict[band]
                band_marker = band_marker_dict[band]
                band_offset = band_offset_dict[band]
                if b == 'PS_i':
                    uncleaned_label = 'Uncleaned data'
                else:
                    uncleaned_label = None
                band_offset_name = band+' + '+str(band_offset)
                ax.errorbar(uncleaned_band['MJD'], (uncleaned_band['mag'] + band_offset), yerr = uncleaned_band['magerr'], c = 'k', markersize = 5, label = uncleaned_label, 
                            linestyle = 'None', fmt = 'o')
                ax.errorbar(cleaned_band['MJD'], (cleaned_band['mag'] + band_offset), yerr = cleaned_band['magerr'], 
                                c = band_colour, fmt = band_marker, label = band_offset_name, linestyle = 'None', markeredgecolor = 'k', markeredgewidth = '0.5') # sigma clipped binned data
                ax.errorbar(uncleaned_band['wm_MJD'], (uncleaned_band['wm_mag'] + band_offset), xerr = (uncleaned_band['MJD_bin_min'], uncleaned_band['MJD_bin_max']), c = 'k', linestyle = 'None') # uncleaned binned data - this is the binned data before sigma clipping
                
                ax.scatter(file_df_banddata['MJD'], (file_df_banddata['mag'] + band_offset), marker = '^', c = 'pink', edgecolor = 'k', s=65, linewidth = 0.5) # the untampered band data (not been through binning function) - want to make sure these datapoints are the same as the unbinned
            #                                                                                                                                                   MJD and mag to make sure the dataframe merge worked correctly
                ax.invert_yaxis()

            fig.supxlabel('MJD')
            fig.supylabel('mag')
            fig.legend()
            fig.suptitle(f'{ANT_name}  - lightcurve split into bands ')
            #plt.show()





    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # HISTOGRAM OF MAGERRS TO SHOW HOW SMALL THEY ARE
    hist_of_magerrs = True
    problematic_bands = ['ZTF_g', 'ZTF_r', 'PS_i', 'PS_w'] # we're only really worried about these bands because the other ones have a lower cadence/larger errors usually and dont cause problems
    if hist_of_magerrs == True:
        fig, axs = plt.subplots(3, 4, figsize = (16, 7))
        #axs = axs.ravel()

        for i in range(len(problematic_bands)):
            band = problematic_bands[i]
            if band not in b_pres: # if the ANT has no data for this band, check the next
                continue
            
            uncleaned_band = binned_df[binned_df['band'] == band].copy()
            band_colour = band_colour_dict[band]

            # magerr histogram
            ax = axs[0, i]
            ax.hist(uncleaned_band['magerr'], bins = 50, color = band_colour, density = False)
            ax.set_xlabel('magerr', fontsize = 9, fontweight = 'bold')
            ax.set_title(band, fontweight = 'bold')

            # sigma distance within the histogram
            ax2 = axs[1, i]
            # making the bins
            if uncleaned_band['sigma_dist'].max() >25:
                bins = np.arange(0, np.round(uncleaned_band['sigma_dist'].max(), -1), 5)

            else: bins = 20

            ax2.hist(uncleaned_band['sigma_dist'], bins = bins, color = band_colour)
            ax2.set_xlabel('sigma distance', fontsize = 9, fontweight = 'bold')
            ax2.axvline(x = SIGMA_OUTLIER_LIM, c = 'k')
            

            # std dev of mags within the bin histogram
            uncleaned_band['BAD_std_mag'] = uncleaned_band['BAD_std_mag'].fillna(0) # get rid of NaN values and replace them with 0 - this just means that there was only 1 datapoint in the bin]
            ax3 = axs[2, i]
            ax3.hist(uncleaned_band['BAD_std_mag'], bins = 50, color = band_colour, density = False)
            ax3.set_xlabel('std dev within bin', fontsize = 9, fontweight = 'bold')
            

        fig.suptitle(f'{ANT_name}, bin size = {MJD_binsize}, sigma_outlier_lim = {SIGMA_OUTLIER_LIM}', fontweight = 'bold')
        fig.subplots_adjust(top=0.91,
                            bottom=0.07,
                            left=0.03,
                            right=0.985,
                            hspace=0.31,
                            wspace=0.2)
        #plt.show()
            





    #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # PLOT A SUBPLOT OF HISTOGRAMS OF THE SIGMA DISTANCE FOR EACH ANT - IN THEORY, IF THE ERROR BARS FOR EACH BAND ARE CORRECT, THEN WE SHOULD BE ABLE TO PUT ALL OF THE BANDS
    # SIGMS DISTANCES INTO ONE BIG HISTOGRAM, BECASUE THEY SHOULD DEVIATE FROM THEIR MEANS THE SAME AMOUNT.....
    hist_of_sigma_dist = False
    if hist_of_sigma_dist == True:
        fig, axs = plt.subplots(3, 4, figsize = (16, 7))
        axs = axs.ravel()
        for i, ax in enumerate(axs):
            if i > (len(b_pres) - 1):
                break

            band = b_pres[i]
            uncleaned_band = binned_df[binned_df['band']== band].copy()
            band_colour = band_colour_dict[band]

            # making the bins
            if uncleaned_band['sigma_dist'].max() >25:
                bins = np.arange(0, np.round(uncleaned_band['sigma_dist'].max(), -1), 5)

            else: bins = 20

            ax.hist(uncleaned_band['sigma_dist'], bins = bins, color = band_colour)
            ax.set_title(band)

        fig.supxlabel('sigma distance')
        fig.suptitle(f'{ANT_name}. MJD binsize = {MJD_binsize},  SIGMA_OUTLIER_LIM = {SIGMA_OUTLIER_LIM}', fontweight = 'bold')
        fig.subplots_adjust(hspace = 0.31)
        #plt.show()

    plt.show() # plot all plots for each ANT together 











