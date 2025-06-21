import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import extinction
#from astroquery.irsa_dust import IrsaDust
from astroquery.ipac.irsa.irsa_dust import IrsaDust
from astropy.coordinates import SkyCoord
import astropy.units as u
sys.path.append("C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS") # this allows us to access the plotting_preferences.py file 
from plotting_preferences import band_colour_dict, band_marker_dict, band_offset_dict, band_ZP_dict, band_obs_centwl_dict
#from functions import restframe_luminosity # used for PS1-10adi
from astropy.cosmology import FlatLambdaCDM # THIS IS FOR THE LUMINOSITY DISTANCE  (for PS1-10adi)
import astropy.units as u

print()
print()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# THIS IS THE FILE WHICH I USE TO GATHER TOGETHER THE DATA FOR EACH OF THE ANTS - E.G. IF AN ANT HAS DATA FROM GAIA AND ATLAS, I WOULD 
# COMBINE THE GAIA AND ATLAS DATA INTO ONE FILE HERE, AND ALSO DO ALL DATA CLEANING IN THIS CODE. 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
file_load_dir = "C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS Data/other ANT data/" #+ ANT_name/datafile_name.dat
file_save_dir = "C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS Data/other ANT data/ALL_FULL_LCs/" # +ANT_name_lc.csv
Phil_file_save_dir = "C:/Users/laure/OneDrive/Desktop/YoRiS Desktop/YoRiS Data/modified Phil's lightcurves/" # +ANT_name_lc.csv


MAGERR_LIM = 2.0 # if magerr is greater than this value, get rid of the datapoint MAYBE THIS ISN'T ACTUALLY NECESSARY........ ESPECIALLY SINCE WE HAVE A FLUXERR RATIO LIM

MIN_MAGERR = 0.0000001 # if the magerr is less than this value, then remove the datapoint as there must be something wrong with it to have such a small/NO error bar

FLUXERR_RATIO_LIM = 2.0 # if flux/flux_err < FLUXERR_RATIO_LIM then get rid of the data point

# ALSO GET RID OF ANY DATA FOR WHICH   mag >= 3sig_upper_lim or 5sig_upper_lim

# ALSO GET RID OF ATLAS DATA FOR WHICH tphot err =! 0

ATLAS_CHI_N_UPLIM = 3.0

ATLAS_CHI_N_LOWLIM = 0.2




#############################################################################################################################################################################################################



def fix_ANT_bandnames(dataframe):
    """
    assumes that dataframe has columns 'MJD' and 'band'. Corrects the band names to ensure that all ANTs' bands are named in a consistent way
    """
    band_fix_dict = {'UVM2': 'UVOT_UVM2', 
                 'UVOT_UVM1': 'UVOT_UVW1', 
                 'Swift_B': 'UVOT_B', 
                 'Swift_V': 'UVOT_V', 
                 'Swift_U': 'UVOT_U', 
                 'Swift_M2': 'UVOT_UVM2', 
                 'Swift_W1': 'UVOT_UVW1', 
                 'Swift_W2': 'UVOT_UVW2'}

    bands_to_fix = list(band_fix_dict.keys()) # a list of the band names which need changing

    for i in dataframe.index:
        band = dataframe.loc[i, 'band']
        
        if band in bands_to_fix:
            dataframe.loc[i, 'band'] = band_fix_dict[band]


    return dataframe






#############################################################################################################################################################################################################



def flux_density_to_mag(F, band_ZP, F_err = None):
    """
    calculates magnitude from the flux density and zero point for a particular band. 
        - if you input the flux density in the AB system, make sure to input the band_ZP in the AB system.
        - if you input the flux density in the Vega system, make sure to input the band_ZP in the Vegs system

        - if you input F as the observer frame flux density for the band, and the ZP for the observed frame band, then you will get out the apparent mag for that band
        - if you input F as the rest frame flux density for the band, and the ZP for the rest frame band, then you will get out the absolute mag for that band PROVIDED
          THAT THE REST FRAME FLUX DENSITY WAS TAKEN AT A DISTANCE OF 10pc - it is unlikely that you will need to work with the rest frame flux density much though. 

    MAKE SURE TO INPUT F, F_err AND band_ZP IN THE SAME UNITS WHICH ARE UNITS OF FLUX DENSITY


    INPUTS:
    ------------
    F: float, the flux density for your given band. In same units as all other inputs

    F_err: float, the error on the flux density for your given band. In same units as all other inputs

    band_ZP: float, the zeropoint of your given band in the magnitude system you want to use. If you use a Vega ZP, you'll get a Vega mag, if you input an AB ZP, you'll get an AB mag. 
            In same units as all other inputs

            
    OUTPUTS:
    ---------------
    mag: float, the magnitude. Whether it is an AB or a Vega mag depends on whether the band_ZP was in the Vega or AB system

    magerr: float, the error on the magnitude. 
    """

    m = -2.5 * np.log10(F) + 2.5 * np.log10(band_ZP)

    if F_err is not None:
        m_err = 2.5 * F_err * ( 1/ (np.log(10) * F) )
        return m, m_err
    
    else:
        return m





#############################################################################################################################################################################################################



def mag_to_flux_density(mag, band_ZP, magerr = None):
    """
    if you input the mag and magerr in the AB system, make sure to input the band_ZP in the AB system. 
    If you input the mag and magerrin the Vega system, make sure to input the band_ZP in the Vega system 

    If you input the abs mag for a band in the rest frame (and the ZP of this rest frame band), then you will get out the flux density in the rest frame for that rest frame band. 
    If you input the app mag in the observed frame (and the ZP of the observed band), then you will get out the flux density in the observed frame for that observed band. 
    This is because we make no K-corrections. 

    INPUTS:
    ----------------
    mag: float, the apparent or absolute magnitude of the source in the band of interest

    band_ZP: float, the zero point of the band in units of flux density. The units of this will be the units of the flux density output

    magerr: float, the error on the magnitude of the source in the band of interest

    OUTPUTS:
    -----------------
    F: float, the flux density of the source in the band of interest. This will be in the same units as the band_ZP input. If you input abs mag for the band and the ZP for that band, you will 
    get out the flux denisty in the rest frame. If you input the app mag for the band and the ZP for that band, you will get out the flux density in the observed frame (how we'd see on Earth)
    
    F_err: the error on F, propagated from magerr. In units of flux density, the same nunits as the band_ZP input. 
    
    """
    F = (band_ZP) * 10 **(-0.4 * mag) # this gives the flux density for the band which this mag was observed in. 

    if magerr is not None:
        F_err = 0.4 * np.log(10) * F * magerr # this is the error on the flux density
        return F, F_err
    
    else: 
        return F





#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
#############################################################################################################################################################################################################
#############################################################################################################################################################################################################



def host_subtraction_of_mags(host_mag_dict, band_ZP_dict, lc_df):
    """
    This function removes the host galaxy contribution to the mags/flux.
    MAKE SURE THE MAGNITUDE SYSTEMS FOR THE HOST MAG MATCHES THE MAGNITUDE SYSTEM FOR THE ZERO POINTS AND THE NAMES OF THESE BANDS

    INPUTS:
    -------------
    host_mag_dict: dictionary. Contains observed/simulated apparent magnitudes of the host in different bands

    band_ZP_dict: dictionary. Contains the zero points for differenr bands. MAKE SURE THESE ZERO POINTS ARE IN THE SAME MAGNITUDE SYSTEM AS HOST_MAG_DICT

    lc_df: DataFrame. contains the transient's light curve


    RETURNS
    ------------
    corrected_lc_df: DataFrame containing the host-emission corrected light curve data

    """
    
    # convert the magnitude values to flux density for the light curve data
    lc_df['band_ZP_per_A'] = lc_df['band'].map(band_ZP_dict) # make a band zeropoint column by mapping the bands to their zeropoint values
    # don't bother calculating the flux density error since we dont havae a flux density error for the host, so the corrected flux has the same error as the non-corrected flux
    # so when we eventually convert back to mags, you've basically just converted your original magerr to a flux density error, not changed anything about it and converted it back to a magerr. 
    F_per_A = mag_to_flux_density(mag = lc_df['mag'], band_ZP = lc_df['band_ZP_per_A'], magerr = None) # convert mag to flux density for the given band. 
    lc_df['rf_flux_density_per_A'] = F_per_A

    # convert the magnitude values in the host galaxy magnitudes into flux density
    host_mag_df = pd.DataFrame(list(host_mag_dict.items()), columns = ['band', 'mag'])
    host_mag_df['band_ZP_per_A'] = host_mag_df['band'].map(band_ZP_dict) # make a band zeropoint column by mapping the bands to their zeropoint values
    host_F_per_A = mag_to_flux_density(mag = host_mag_df['mag'], band_ZP = host_mag_df['band_ZP_per_A'], magerr = None)
    host_mag_df['rf_host_flux_density_per_A'] = host_F_per_A


    # then add a column to the lc_df dataframe for this host flux density emission for each band, ready to be subtracted
    lc_df = lc_df.merge(host_mag_df[['band', 'rf_host_flux_density_per_A']], on = 'band', how = 'left')
    lc_df['rf_corrected_flux_density_per_A'] = lc_df['rf_flux_density_per_A'] - lc_df['rf_host_flux_density_per_A'] # subtract the host flux from the observed flux to get the transient flux
    corrected_mag = flux_density_to_mag(F = lc_df['rf_corrected_flux_density_per_A'], F_err = None, band_ZP = lc_df['band_ZP_per_A'])
    lc_df['corrected_mag'] = corrected_mag
    lc_df = lc_df.drop(columns = ['mag', 'rf_flux_density_per_A', 'rf_host_flux_density_per_A', 'rf_corrected_flux_density_per_A', 'band_ZP_per_A'])
    lc_df = lc_df.rename(columns = {'corrected_mag': 'mag'})
    

    return lc_df





# ASASSN-18jd


host_mag_dict_18jd = {'UVOT_UVW2': 20.99,  
                    'UVOT_UVM2': 21.01, 
                    'UVOT_UVW1': 20.82, 
                    'UVOT_U': 19.77,
                    'Swope_u': 19.61, # for SDSS u'
                    'UVOT_B': 18.54, 
                    'LCOGT_B': 18.54, # for UVOT_B
                    'SMARTS_B': 18.54, # for UVOT_B 
                    'Swope_B': 18.54, # for UVOT_B
                    'Swope_g': 17.98, # SDSS g'
                    'LCOGT_g': 17.98, # SDSS g'
                    'UVOT_V': 17.33, 
                    'SMARTS_V': 17.33, # for UVOT_V
                    'LCOGT_V': 17.33, # for UVOT_V
                    'Swope_V': 17.33, # for UVOT_V 
                    'LCOGT_r': 16.96, # for SDSS r'
                    'Swope_r': 16.96, # for SDSS r' 
                    'LCOGT_i': 16.54, # for SDSS i'
                    'Swope_i': 16.54 # for SDSS i'
                    }



""" 
path1 = file_load_dir + "ASASSN-18jd_lc_files/ASASSN-18jd_table-phot_paper.txt"
colspecs = [(0, 9), (11, 13), (15, 16), (18, 23), (25, 29), (31, 38)]
colnames = ['MJD', 'band_nosurvey', '3sig_mag_uplim', 'mag', 'magerr', 'survey']
paper_lc = pd.read_fwf(path1, colspecs = colspecs, na_values = ['', ' '], skiprows = 31, header = None, names = colnames)

paper_lc['band'] = paper_lc['survey']+'_'+paper_lc['band_nosurvey']
paper_lc = fix_ANT_bandnames(paper_lc).copy() # this changes the band names so that they are consistent with the other ANTs. e.g. Swift_M2 --> UVOT_UVM2
paper_lc = paper_lc.drop(columns = ['band_nosurvey', 'survey'])
#paper_lc['flux'] = [None]*len(paper_lc['MJD'])
#paper_lc['flux_err'] = [None]*len(paper_lc['MJD'])
paper_lc_b4_host_subtraction = paper_lc.copy()


print()
print()
print('PAPER LC BEFORE HOST SUBTRACTION')
print(paper_lc[paper_lc['band'].isin(['ASAS-SN_V', 'ASAS-SN_g'])].tail(10))

paper_lc_NOT_for_host_correction = paper_lc[paper_lc['band'].isin(['ASAS-SN_V', 'ASAS-SN_g'])] # ASASSN data was taken with difference imaging so no correction needed
paper_lc_for_host_correction = paper_lc[~paper_lc['band'].isin(['ASAS-SN_V', 'ASAS-SN_g'])] # remove the ASASSN data since this has already been host subtracted (difference imaging)
paper_lc_for_host_correction = host_subtraction_of_mags(host_mag_dict = host_mag_dict_18jd, band_ZP_dict = band_ZP_dict, lc_df = paper_lc_for_host_correction) # SUBTRACT HOST EMISSION
paper_lc = pd.concat([paper_lc_NOT_for_host_correction, paper_lc_for_host_correction], ignore_index = True).copy() # append the ASASSN and the host corrected light curve data


print()
print()
print('PAPER LC AFTER HOST SUBTRACTION')
print(paper_lc[paper_lc['band'].isin(['ASAS-SN_V', 'ASAS-SN_g'])].tail(10))
print()
print()
print()
print('PAPER ALL AFTER HOST SUBTRACTION')
print(paper_lc)
print()
print()
print()


fig, axs = plt.subplots(1, 2, figsize = (16, 7.5), sharex = True, sharey = False)
ax1, ax2 = axs

for b in paper_lc_b4_host_subtraction['band'].unique():
    b_b4_df = paper_lc_b4_host_subtraction[paper_lc_b4_host_subtraction['band'] == b].copy()
    b_after_df = paper_lc[paper_lc['band'] == b].copy()
    b_color = band_colour_dict[b]
    b_marker = band_marker_dict[b]

    ax1.errorbar(b_b4_df['MJD'], b_b4_df['mag'], yerr = b_b4_df['magerr'], fmt = b_marker, c = b_color, mec = 'k', mew = 0.5, linestyle = 'None', label = b, capsize = 5, capthick = 2)
    ax2.errorbar(b_after_df['MJD'], b_after_df['mag'], yerr = b_after_df['magerr'], fmt = b_marker, c = b_color, mec = 'k', mew = 0.5, linestyle = 'None', label = b, capsize = 5, capthick = 2)
    
ax2.legend(loc = 'upper right', fontsize = 7, bbox_to_anchor = (1.2, 0.8))
ax1.set_title('Before host subtraction')
ax2.set_title('After host subtraction')
for ax in [ax1, ax2]:
    ax.grid(True)
    ax.invert_yaxis()

fig.supxlabel('MJD')
fig.supylabel('mag')
#fig.gca().invert_yaxis()
fig.suptitle('ASAS-SN-18jd checking the mags before and after host subtraction')
fig.subplots_adjust(top=0.88,
                    bottom=0.11,
                    left=0.07,
                    right=0.9,
                    hspace=0.2,
                    wspace=0.135)
plt.show()

subtracted_magerrs = paper_lc_b4_host_subtraction['magerr'] - paper_lc['magerr']
print('No of magerr values which were changed after host subtraction (should be 0) = ', (subtracted_magerrs > 0.0).sum())





path2 = file_load_dir + "ASASSN-18jd_lc_files/ASASSN-18jd_ATLAS.txt"
ATLAS_lc = pd.read_csv(path2, delim_whitespace = True)

ATLAS_lc = ATLAS_lc.rename(columns = {'###MJD' : 'MJD', 'm': 'mag', 'dm':'magerr', 'err':'tphot_err', 'uJy':'flux', 'duJy':'flux_err', 'mag5sig':'5sig_mag_uplim'})
# CLEANING UP DATA =================================================================================
ATLAS_lc = ATLAS_lc[ATLAS_lc['tphot_err']==0].copy() # get rid of ATLAS data with tphot errors
ATLAS_lc = ATLAS_lc[ATLAS_lc['mag']>0.0].copy() # get rid of ATLAS data with mag<0.0
#ATLAS_lc = ATLAS_lc[ATLAS_lc['magerr'] < MAGERR_LIM].copy() # if magerr is too large.....
print(f'NUMBER OF ATLAS DATAPOINTS WITH SIGNAL:NOISE RATIO < {FLUXERR_RATIO_LIM} = ', len(ATLAS_lc[(ATLAS_lc['flux']/ATLAS_lc['flux_err']) < FLUXERR_RATIO_LIM]))
ATLAS_lc = ATLAS_lc[(ATLAS_lc['flux']/ATLAS_lc['flux_err']) > FLUXERR_RATIO_LIM].copy() # if flux/flux_err is too small
ATLAS_lc = ATLAS_lc[ATLAS_lc['mag'] < ATLAS_lc['5sig_mag_uplim']].copy() # if the mag > 3 sig upper limit
ATLAS_bandname = []
for b in ATLAS_lc['F']:
    if b == 'o':
        atlas_band = 'ATLAS_o'
        ATLAS_bandname.append(atlas_band)

    elif b == 'c':
        atlas_band = 'ATLAS_c'
        ATLAS_bandname.append(atlas_band)

    else:
        print('ERROR: ATLAS DATA HAS BAND INPUT THAT IS NOT c OR o')


ATLAS_lc['band'] = ATLAS_bandname
ATLAS_lc = ATLAS_lc.drop(['RA', 'Dec', 'x', 'y', 'maj', 'min', 'phi', 'apfit', 'Sky', 'Obs', 'chi/N', 'F', 'flux', 'flux_err', '5sig_mag_uplim'], axis = 1) # can also drop the 5 sigma upper limit as we've already ensured that all mags are less than the upper limit so its job is done
ATLAS_lc['3sig_mag_uplim'] = pd.NA

columns_to_append = ['MJD', 'mag', 'magerr', 'band', '3sig_mag_uplim']#, 'flux', 'flux_err']
combined_18jz = pd.concat([ATLAS_lc[columns_to_append], paper_lc[columns_to_append]], ignore_index = True, sort = False)
#print(combined_18jz[combined_18jz['3sig_mag_uplim'].notna()].head(50)) # in this case, the upper limits are indicates with a > in the 3sig_mag_uplim ccolumn, with the mag of this upper limit in the mag column, so we can calculate the upper limit on the flux using the function above without doing anything separate for the upper limit values
combined_18jz = fix_ANT_bandnames(combined_18jz).copy() # this changes the band names to match the others that i have, e.g. 'Swift_U' --> 'UVOT_U'
print()
#print('COMBINED LIGHT CURVE')
#print(combined_18jz.tail(60))
print()
print()







# NOW CORRECT FOR GALACTIC EXTINCTION OF 18JD
combined_18jz['obs_cent_wl'] = combined_18jz['band'].map(band_obs_centwl_dict)
obs_cent_wls = list(combined_18jz['obs_cent_wl'].unique())



ra = 340.928629043 #ra of object in decimal degrees
dec = -16.9856920389 #dec of object in decimal degrees
extinction_table = IrsaDust.get_query_table(SkyCoord(ra*u.deg,dec*u.deg),section='ebv').to_pandas()


for i, obs_cent_wl in enumerate(obs_cent_wls):
    band_paper_lc = combined_18jz[combined_18jz['obs_cent_wl'] == obs_cent_wl].copy()
    a_lambda = extinction.fitzpatrick99(np.array([obs_cent_wl]), 3.1*extinction_table['ext SandF mean'])[0] #where lambda_mean is your filter mean wavelength
    
    band_paper_lc['gal_ext_corr_mag'] = band_paper_lc['mag'] - a_lambda #we are subtracting from the magnitude because the true magnitude is brighter than the one we observed.
 
    #print()
    #print(band_paper_lc.head(10))
    if i==0:
        new_paper_lc = band_paper_lc

    else:
        new_paper_lc = pd.concat([new_paper_lc, band_paper_lc], ignore_index = True)


print()
print()
print('GAL EXT LIGHT CURVE')
print(new_paper_lc.head(40))

print('len = ', len(new_paper_lc))
print()


combined_18jz = new_paper_lc.copy()
combined_18jz = combined_18jz.drop(columns = ['obs_cent_wl', 'mag'])
combined_18jz = combined_18jz.rename(columns = {'gal_ext_corr_mag': 'mag'})



print('final light curve')
print(combined_18jz.head(30))






save_path = file_save_dir + "ASASSN-18jd_FULL_LC.csv"
combined_18jz.to_csv(save_path, index = False)
print()
print() 
 """


""" # PLOT ATLAS DATA TO CHECK THAT IT'S ALRIGHT

ATLAS_oband = ATLAS_lc[ATLAS_lc['band']=='ATLAS_o']
ATLAS_cband = ATLAS_lc[ATLAS_lc['band']=='ATLAS_c']

plt.figure(figsize=(14, 7))
plt.errorbar(ATLAS_oband['MJD'], ATLAS_oband['mag'], yerr = ATLAS_oband['magerr'], fmt='o', linestyle = 'None',
                        markeredgecolor = 'k', markeredgewidth = '0.5', c='orange', label = 'ATLAS_o')
plt.errorbar(ATLAS_cband['MJD'], ATLAS_cband['mag'], yerr = ATLAS_cband['magerr'], fmt='o', linestyle = 'None',
                        markeredgecolor = 'k', markeredgewidth = '0.5', c='cyan', label = 'ATLAS_c')
plt.xlabel('MJD')
plt.legend()
plt.ylabel('mag')
plt.title(f'removed all data with magerr > {MAGERR_LIM}, transient = ASASSN-18jd')
plt.grid()
plt.gca().invert_yaxis()
plt.show() 
 """

##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################



#ASASSN-17jz
""" 
path1 = "C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS Data/other ANT data/ASASSN-17jz_lc_files/ASASSN-17jz_lc_paper.txt"
colspecs = [(0, 9), (9, 17), (17, 22), (23, 24), (24, 29), (31, 34)]
colnames = ['MJD_start', 'MJD_end', 'band_old','3sig_mag_uplim', 'mag', 'magerr']
paper_lc = pd.read_fwf(path1, colspecs = colspecs, na_values = ['', ' '], skiprows = 29, header = None, names = colnames)

bandnames = []
for b in paper_lc['band_old']:
    if b == 'o':
        band = 'ATLAS_o'

    elif b == 'c':
        band = 'ATLAS_c'

    elif b == 'UVW1' or b == 'UVW2' or b == 'UVM1':
        band = 'UVOT_' + b

    else:
        band = b

    bandnames.append(band)


MJD = []
MJD_err = []
for i in range(len(paper_lc['mag'])):
    if pd.isna( paper_lc['MJD_end'].iloc[i] ):
        mjd = paper_lc['MJD_start'].iloc[i]
        mjd_err = 0.0
        
    else:
        mjd = ( paper_lc['MJD_start'].iloc[i] + paper_lc['MJD_end'].iloc[i] )/2 # take the mean of the upper and lower lim
        mjd_err = mjd - paper_lc['MJD_start'].iloc[i]

    MJD.append(mjd)
    MJD_err.append(mjd_err)


paper_lc['band'] = bandnames
paper_lc['MJD'] = MJD
paper_lc['MJD_err'] = MJD_err

# CHECKING THE DATA OVER 
print()
print('checking the MJD bands werent too large, if they were close to 1 day, get rid of this MJD difference and take the mean MJD instead')
print('mean', paper_lc[paper_lc['MJD_err'] > 0.0]['MJD_err'].mean())
print('std dev',paper_lc[paper_lc['MJD_err'] > 0.0]['MJD_err'].std())
print('max', paper_lc[paper_lc['MJD_err'] > 0.0]['MJD_err'].max())
plt.hist(paper_lc['MJD_err'], bins = 15)
plt.xlabel('ASASSN-17jz bin size', fontweight = 'bold')
plt.title('Checking ASASSN-17jz bin size to see if we can just \ntake the mean MJD instead of using MJD error', fontweight = 'bold')
plt.show()
# from this, I have seen that there are 2 instances where the bin size was ~7, lets see what bins these were
print('THE 2 ROWS WITH MJD_ERR > 3.0 ===============')
print(paper_lc[paper_lc['MJD_err'] > 3.0]) # NOW THAT WE KNOW THAT THE BIN SIZES WERE SMALL, JUST USE THE MEAN OF THE MJD_START AND MJD_END AS THE MJD VALUE, WHICH IS CALCULATED ABOVE
print()
print('DATAPOINTS WITH MAGERR > 3.0 TO SEE IF THIS ACTUALLY IS DOING ANYTHING')
print(paper_lc[paper_lc['magerr'] > MAGERR_LIM])
print()

paper_lc = paper_lc[paper_lc['magerr'] < MAGERR_LIM].copy() # IN THIS CASE, THIS DOES NOTHING
paper_lc = paper_lc.drop(columns = ['band_old'], axis = 1)
paper_lc = fix_ANT_bandnames(paper_lc).copy() # fix the band names to match the names used by other ANTs, e.g. 'UVM2' --> 'UVOT_UVM2'
print(paper_lc)
paper_lc.to_csv("C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS Data/other ANT data/ALL_FULL_LCs/ASASSN-17jz_FULL_LC.csv", index = False)
 """







##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################

def distance_modulus_app_mag(M, d_pc):
    """
    calculates the apparent magnitude WITHOUT K-CORRECTION for a given absolute magnitdue and distance. 
    e.g.
        if you input M in ATLAS_o band, i.e. the absolute magnitude at emitted wavelengths corresponding to ATLAS_o, you will get out the apparent magnitude for this emitted wavelength. 
        HOWEVER!! this is NOT the apparent magnitude that would be measured by observed in the ATLAS_o filter, since emitted wavelength has redshifted while travelling towards us. So, 
        the apparent magnitude calculated would correspond to the apparent magnitude measured by a redder band than ATLAS_o. This is because we haven't K-corrected
    """

    m = 5 * np.log10(d_pc) - 5 + M
    return m






""" 
# PS1-10adi
z = 0.203
# ASSUMPTIONS FOR LUMINOSITY DISTANCE ARE BELOW ----------------------------------------------
H0 = 70 #km/s/Mpc
om_M = 0.3 # non relativistic matter density fraction
fcdm = FlatLambdaCDM(H0 = H0, Om0 = om_M)
# ASSUMPTIONS FOR LUMINOSITY DISTANCE ARE ABOVE ----------------------------------------------
d = fcdm.luminosity_distance(z).to(u.pc).value # this gives the luminosity distance in pc





path1 = file_load_dir + "PS1-10adi_lc_files/PS1-10adi_lc_rest_UBVRIJH"
colnames = ['Days_since_peak', 'U_mag', 'U_magerr', 'B_mag', 'B_magerr', 'V_mag', 'V_magerr', 'R_mag', 'R_magerr', 'I_mag', 'I_magerr', 'J_mag', 'J_magerr', 'H_mag', 'H_magerr']
paper_lc = pd.read_csv(path1, delim_whitespace = True, header = None, names = colnames)

band_names = ['U', 'B', 'V', 'R', 'I', 'J', 'H']
band_names = [name+' (Vega)'for name in band_names]
length = len(paper_lc['U_mag'])
for i, b in enumerate(band_names):
    placement = (i+1)*3
    paper_lc.insert(placement, 'band'+str(i), [b]*length)
    print(placement)
print(paper_lc)

# want to make the datafarme just MJD, mag, magerr, band so create a new dataframe to do this
D_since_peak = list(paper_lc['Days_since_peak'])*len(band_names)
print()
print(len(D_since_peak), 7*36)


mag = []
magerr = []
band = []
for i in range(1, 21, 3):
    for j in range(len(paper_lc['U_mag'])):
        mag.append(paper_lc.iloc[j, i]) # mag
        magerr.append(paper_lc.iloc[j,(i+1)]) # magerr
        band.append(paper_lc.iloc[j, (i+2)])

combined_paper_lc = {'days_since_peak':D_since_peak,'mag':mag, 'magerr':magerr, 'band':band}
combined_paper_lc = pd.DataFrame(combined_paper_lc)
combined_paper_lc = combined_paper_lc[combined_paper_lc['mag']<= 0.0].copy()
combined_paper_lc = combined_paper_lc[combined_paper_lc['magerr'] > MIN_MAGERR].copy()
combined_paper_lc['MJD'] = combined_paper_lc['days_since_peak'] + 2455443 - 2400000.5



# CALCULATE THE UN-K-CORRECTED APPARENT MAGNITUDE
combined_paper_lc['app_mag'] = distance_modulus_app_mag(combined_paper_lc['mag'], d_pc = d)
combined_paper_lc['app_magerr'] = combined_paper_lc['magerr']
print(combined_paper_lc)

savepath = file_save_dir + "PS1-10adi_FULL_LC.csv"
combined_paper_lc.to_csv(savepath, index = False)

 """




##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################

# Phil's lightcurves

""" # 
# loading in the files
folder_path = "C:/Users/laure/OneDrive/Desktop/YoRiS Desktop/YoRiS Data/Phil's lightcurves" # folder path containing the light curve data files
for file in os.listdir(folder_path): # file is a string of the file name such as 'file_name.dat'
    file_path = os.path.join(folder_path, file) 
    file_df = pd.read_csv(file_path, delimiter = ' ') # the lightcurve data in a dataframe
    ANT_name = file[:-7]

    # REMOVING THE OUTLIERS / 
    file_df = file_df[file_df['magerr'] > MIN_MAGERR].copy() # remove the datapoints which have magerr = 0.0000 because this screws with the binning for removing outliers
    #                                                         these would be considered outliers anyway because (weighted_mean_mag - datapoint_mag)/magerr = infinity because magerr = 0.00
    file_df = file_df[file_df['magerr'] < MAGERR_LIM].copy()

    if ANT_name == 'ZTF19aailpwl':
        filter_errors_df = file_df[(file_df['MJD'] > 58856) & (file_df['MJD'] < 58858) & (file_df['mag'] > 19.5)].copy()
        file_df = file_df.drop(filter_errors_df.index).copy()
        #print(ANT_name)
        #print(filter_errors_df)


    #print(file_df)
    #print(file_df['band'].unique())
    file_df = fix_ANT_bandnames(file_df).copy()
    #print(file_df['band'].unique())
    file_df = file_df.dropna(subset = ['mag'])
    #print()
    #print()
    #print(file_df.columns)
    savepath = Phil_file_save_dir + f"{ANT_name}_lc.csv"
    file_df.to_csv(savepath, index = False)

 """


##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
