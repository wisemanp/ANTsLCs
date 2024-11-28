import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append("C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS") # this allows us to access the plotting_preferences.py file 
from plotting_preferences import band_colour_dict, band_marker_dict, band_offset_dict


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
#############################################################################################################################################################################################################
#############################################################################################################################################################################################################





# ASASSN-18jd
""" #path1 = "C:/Users/laure/OneDrive/Desktop/YoRiS/September/other ANT data/ASASSN-18jd_lc_files/ASASSN-18jd_table-phot_paper.txt"
path1 = file_load_dir + "ASASSN-18jd_lc_files/ASASSN-18jd_table-phot_paper.txt"
colspecs = [(0, 9), (11, 13), (15, 16), (18, 23), (25, 29), (31, 38)]
colnames = ['MJD', 'band_nosurvey', '3sig_mag_uplim', 'mag', 'magerr', 'survey']
paper_lc = pd.read_fwf(path1, colspecs = colspecs, na_values = ['', ' '], skiprows = 31, header = None, names = colnames)

paper_lc['band'] = paper_lc['survey']+'_'+paper_lc['band_nosurvey']
paper_lc = fix_ANT_bandnames(paper_lc).copy() # this changes the band names so that they are consistent with the other ANTs. e.g. Swift_M2 --> UVOT_UVM2
paper_lc = paper_lc.drop(columns = ['band_nosurvey', 'survey'])
paper_lc['flux'] = [None]*len(paper_lc['MJD'])
paper_lc['flux_err'] = [None]*len(paper_lc['MJD'])
paper_lc['5sig_mag_uplim'] = [None]*len(paper_lc['MJD'])
#paper_lc_baddatasum = paper_lc[paper_lc['mag'] > paper_lc['3sig_mag_uplim']].sum()
#print('NO BAD DATA > 3 SIG UL', paper_lc_baddatasum)

#print(paper_lc)
#print(paper_lc['band'].unique())

path2 = file_load_dir + "ASASSN-18jd_lc_files/ASASSN-18jd_ATLAS.txt"
ATLAS_lc = pd.read_csv(path2, delim_whitespace = True)

ATLAS_lc = ATLAS_lc.rename(columns = {'###MJD' : 'MJD', 'm': 'mag', 'dm':'magerr', 'err':'tphot_err', 'uJy':'flux', 'duJy':'flux_err', 'mag5sig':'5sig_mag_uplim'})
# CLEANING UP DATA =================================================================================
ATLAS_lc = ATLAS_lc[ATLAS_lc['tphot_err']==0].copy() # get rid of ATLAS data with tphot errors
ATLAS_lc = ATLAS_lc[ATLAS_lc['mag']>0.0].copy() # get rid of ATLAS data with mag<0.0
ATLAS_lc = ATLAS_lc[ATLAS_lc['magerr'] < MAGERR_LIM].copy() # if magerr is too large.....
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
ATLAS_lc = ATLAS_lc.drop(['RA', 'Dec', 'x', 'y', 'maj', 'min', 'phi', 'apfit', 'Sky', 'Obs', 'chi/N', 'F'], axis = 1)
ATLAS_lc['3sig_mag_uplim'] = pd.NA
print(ATLAS_lc)

columns_to_append = ['MJD', 'mag', 'magerr', 'band', '3sig_mag_uplim', '5sig_mag_uplim' , 'flux', 'flux_err']
combined_18jz = pd.concat([ATLAS_lc[columns_to_append], paper_lc[columns_to_append]], ignore_index = True, sort = False)
print()
print()
print(combined_18jz)

combined_18jz = fix_ANT_bandnames(combined_18jz).copy() # this changes the band names to match the others that i have, e.g. 'Swift_U' --> 'UVOT_U'

save_path = file_save_dir + "ASASSN-18jd_FULL_LC.csv"
combined_18jz.to_csv(save_path, index = False)
print()
print()  """

""" # PLOT ATLAS DATA TO CHECK THAT IT'S ALRIGHT
ATLAS_oband = ATLAS_lc[ATLAS_lc['band']=='ATLAS_o']
ATLAS_cband = ATLAS_lc[ATLAS_lc['band']=='ATLAS_c']

plt.figure(figsize=(14, 7))
plt.errorbar(ATLAS_oband['MJD'], ATLAS_oband['mag'], yerr = ATLAS_oband['magerr'], fmt='o', linestyle = 'None',
                        markeredgecolor = 'k', markeredgewidth = '0.5', c='orange')
plt.errorbar(ATLAS_cband['MJD'], ATLAS_cband['mag'], yerr = ATLAS_cband['magerr'], fmt='o', linestyle = 'None',
                        markeredgecolor = 'k', markeredgewidth = '0.5', c='cyan')
plt.xlabel('MJD')
plt.ylabel('mag')
plt.title(f'removed all data with magerr > {MAGERR_LIM}, transient = ASASSN-18jd')
plt.grid()
plt.gca().invert_yaxis()
plt.show()  """


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
        mjd = ( paper_lc['MJD_start'].iloc[i] + paper_lc['MJD_end'].iloc[i] )/2
        mjd_err = mjd - paper_lc['MJD_start'].iloc[i]

    MJD.append(mjd)
    MJD_err.append(mjd_err)


paper_lc['band'] = bandnames
paper_lc['MJD'] = MJD
paper_lc['MJD_err'] = MJD_err
paper_lc = paper_lc[paper_lc['magerr'] < MAGERR_LIM].copy()
paper_lc = paper_lc.drop(columns = ['band_old'], axis = 1)
paper_lc = fix_ANT_bandnames(paper_lc).copy() # fix the band names to match the names used by other ANTs, e.g. 'UVM2' --> 'UVOT_UVM2'
print(paper_lc)
paper_lc.to_csv("C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS Data/other ANT data/ALL_FULL_LCs/ASASSN-17jz_FULL_LC.csv", index = False)
 """



##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################




# CSS100217
""" path1 = "C:/Users/laure/OneDrive/Desktop/YoRiS/September/other ANT data/CSS100217_lc_files/CSS100217_CSSdata.xlsx"
CSS_lc = pd.read_excel(path1, sheet_name = 0)
CSS_lc = CSS_lc.drop(columns = ['MasterID', 'RA', 'Dec', 'Blend'], axis = 1)
CSS_lc = CSS_lc.rename(columns ={'Mag':'mag', 'Magerr':'magerr'})
CSS_lc = CSS_lc[CSS_lc['magerr'] < MAGERR_LIM].copy()
CSS_lc['band'] = ['CSS_V']*len(CSS_lc['MJD'])
CSS_lc.to_csv("C:/Users/laure/OneDrive/Desktop/YoRiS/September/other ANT data/ALL_FULL_LCs/CSS100217_FULL_LC.csv", index = False)
print(CSS_lc)  """




##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################




# Gaia16aaw
""" path1 = "C:/Users/laure/OneDrive/Desktop/YoRiS/September/other ANT data/Gaia16aaw_lc_files/Gaia16aaw_lc_Gaia.xlsx"
gaia_lc = pd.read_excel(path1, sheet_name = 0, skiprows=1)
gaia_lc['MJD'] = gaia_lc['JD(TCB)'] - 2400000.5
gaia_lc = gaia_lc.drop(columns = ['#Date', 'JD(TCB)'])
gaia_lc = gaia_lc[gaia_lc['averagemag'].apply(lambda x: isinstance(x, float)) & gaia_lc['averagemag'].notna() ].copy() # get rid of 'untrusted' or 'NaN' in mag
gaia_lc = gaia_lc.rename(columns = {'averagemag' : 'mag'})
gaia_lc['band'] = ['Gaia_G']*len(gaia_lc['mag'])
gaia_lc['magerr'] = [0.0]*len(gaia_lc['MJD'])
#print(gaia_lc) # GAIA DATA IS AVERAGED BUT IT DOESN'T GIVE ANY ERROR BARS ANYWAYS SO IS IT EVEN WORTH MENTIONING?
gaia_lc.to_csv("C:/Users/laure/OneDrive/Desktop/YoRiS/September/other ANT data/ALL_FULL_LCs/Gaia16aaw_FULL_LC.csv", index = False)
 """


""" # the ATLAS lc isn't that great maybe? idk? it's not on the TNS but it's in the paper
path2 = "C:/Users/laure/OneDrive/Desktop/YoRiS/September/other ANT data/Gaia16aaw_lc_files/Gaia16aaw_ATLAS.txt"
ATLAS_lc = pd.read_csv(path2, delim_whitespace = True)

ATLAS_lc = ATLAS_lc.rename(columns = {'###MJD' : 'MJD', 'm': 'mag', 'dm':'magerr', 'err':'tphot_err'})
ATLAS_lc = ATLAS_lc[ATLAS_lc['tphot_err']==0].copy()
ATLAS_lc = ATLAS_lc[ATLAS_lc['mag']>0.0].copy()
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
ATLAS_lc = ATLAS_lc.drop(['RA', 'Dec', 'x', 'y', 'maj', 'min', 'phi', 'apfit', 'Sky', 'Obs', 'chi/N', 'F'], axis = 1)
columns_to_append = ['MJD', 'mag', 'magerr', 'band']

Gaia16aaw_lc_all = pd.concat([gaia_lc[columns_to_append], ATLAS_lc[columns_to_append]], ignore_index = True, sort = False)



#print(Gaia16aaw_lc_all)


ATLAS_oband = ATLAS_lc[ATLAS_lc['band']=='ATLAS_o']
ATLAS_oband = ATLAS_oband[ATLAS_oband['magerr']<3.0]
ATLAS_cband = ATLAS_lc[ATLAS_lc['band']=='ATLAS_c']
ATLAS_cband = ATLAS_cband[ATLAS_cband['magerr']<3.0]

plt.figure(figsize = (14, 7))
plt.errorbar(gaia_lc['MJD'], gaia_lc['mag'], fmt='o', linestyle = 'None',
                        markeredgecolor = 'k', markeredgewidth = '0.5', c='green')
plt.xlabel('MJD')
plt.ylabel('mag')
plt.gca().invert_yaxis()

plt.figure(figsize=(14, 7))
plt.errorbar(ATLAS_oband['MJD'], ATLAS_oband['mag'], yerr = ATLAS_oband['magerr'], fmt='o', linestyle = 'None',
                        markeredgecolor = 'k', markeredgewidth = '0.5', c='orange')
plt.errorbar(ATLAS_cband['MJD'], ATLAS_cband['mag'], yerr = ATLAS_cband['magerr'], fmt='o', linestyle = 'None',
                        markeredgecolor = 'k', markeredgewidth = '0.5', c='cyan')
plt.xlabel('MJD')
plt.ylabel('mag')
plt.title('removed all data with magerr > 3.0')
plt.grid()
plt.gca().invert_yaxis()
plt.show() """



##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################




# Gaia18cdj
""" path1 = "C:/Users/laure/OneDrive/Desktop/YoRiS/September/other ANT data/Gaia18cdj_lc_files/Gaia18cdj_Gaia.xlsx"
gaia_lc = pd.read_excel(path1, sheet_name = 0, skiprows=1)
gaia_lc['MJD'] = gaia_lc['JD(TCB)'] - 2400000.5
gaia_lc = gaia_lc.drop(columns = ['#Date', 'JD(TCB)'])
gaia_lc = gaia_lc[gaia_lc['averagemag'].apply(lambda x: isinstance(x, float)) & gaia_lc['averagemag'].notna() ].copy() # get rid of 'untrusted' or 'NaN' in mag
gaia_lc = gaia_lc.rename(columns = {'averagemag' : 'mag'})
gaia_lc['band'] = ['Gaia_G']*len(gaia_lc['mag'])
gaia_lc['magerr'] = [0.0]*len(gaia_lc['MJD'])
gaia_lc['flux'] = [None]*len(gaia_lc['MJD'])
gaia_lc['flux_err'] = [None]*len(gaia_lc['MJD'])
gaia_lc['5sig_mag_uplim'] = [None]*len(gaia_lc['MJD'])
#print(gaia_lc) # GAIA DATA IS AVERAGED BUT IT DOESN'T GIVE ANY ERROR BARS ANYWAYS SO IS IT EVEN WORTH MENTIONING?


path2 = "C:/Users/laure/OneDrive/Desktop/YoRiS/September/other ANT data/Gaia18cdj_lc_files/Gaia18cdj_ATLAS.txt"
ATLAS_lc = pd.read_csv(path2, delim_whitespace = True)

ATLAS_lc = ATLAS_lc.rename(columns = {'###MJD' : 'MJD', 'm': 'mag', 'dm':'magerr', 'err':'tphot_err', 'uJy':'flux', 'duJy':'flux_err', 'mag5sig':'5sig_mag_uplim'})
ATLAS_raw_data = ATLAS_lc.copy()
# CLEANING UP THE DATA =====================================================================================
ATLAS_lc = ATLAS_lc[ATLAS_lc['tphot_err']==0].copy()
ATLAS_lc = ATLAS_lc[ATLAS_lc['mag']>0.0].copy()
ATLAS_lc = ATLAS_lc[ATLAS_lc['magerr'] < MAGERR_LIM].copy()
ATLAS_lc = ATLAS_lc[(ATLAS_lc['flux'] / ATLAS_lc['flux_err']) > FLUXERR_RATIO_LIM].copy()
ATLAS_lc = ATLAS_lc[ ATLAS_lc['mag'] < ATLAS_lc['5sig_mag_uplim'] ].copy()
ATLAS_lc = ATLAS_lc[ATLAS_lc['chi/N'] > ATLAS_CHI_N_LOWLIM].copy() # make sure the PSF fit reduced chi^2 is alright - there is one datapoint 
ATLAS_lc = ATLAS_lc[ATLAS_lc['chi/N'] < ATLAS_CHI_N_UPLIM].copy() # for this ANT which passes all other filters but this amd its clearly terrible

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
# checking out a weird ATLAS datapoint

ATLAS_lc = ATLAS_lc.drop(['RA', 'Dec', 'x', 'y', 'maj', 'min', 'phi', 'apfit', 'Sky', 'Obs', 'chi/N', 'F'], axis = 1)
columns_to_append = ['MJD', 'mag', 'magerr', 'band', 'flux', 'flux_err', '5sig_mag_uplim']

Gaia18cdj_lc_all = pd.concat([gaia_lc[columns_to_append], ATLAS_lc[columns_to_append]], ignore_index = True, sort = False)
Gaia18cdj_lc_all.to_csv("C:/Users/laure/OneDrive/Desktop/YoRiS/September/other ANT data/ALL_FULL_LCs/Gaia18cdj_FULL_LC.csv", index = False)


ATLAS_o = ATLAS_lc[ATLAS_lc['band'] == 'ATLAS_o'].copy()
ATLAS_c = ATLAS_lc[ATLAS_lc['band'] == 'ATLAS_c'].copy()
plt.figure(figsize = (16, 7.5))
plt.errorbar(ATLAS_o['MJD'], ATLAS_o['mag'], yerr = ATLAS_o['magerr'], fmt = 'o', linestyle = 'None', c = 'orange', markeredgecolor = 'k', 
             markeredgewidth = '0.5', label = 'ATLAS_o')
plt.errorbar(ATLAS_c['MJD'], ATLAS_c['mag'], yerr = ATLAS_c['magerr'], fmt = 'o', linestyle = 'None', c = 'cyan', markeredgecolor = 'k', 
             markeredgewidth = '0.5', label = 'ATLAS_c')
plt.grid()
plt.legend()
plt.xlabel('MJD')
plt.ylabel('apparent mag')
plt.gca().invert_yaxis()
plt.show()

print(Gaia18cdj_lc_all)

# CHECKING THAT MY ATLAS FILTERS ARE OKAY USING PLOTS
plt.figure(figsize = (16, 7.5))
ax1 = plt.subplot(1, 2, 1)
ax1.scatter((ATLAS_raw_data['flux'] / ATLAS_raw_data['flux_err']), (ATLAS_raw_data['chi/N']), marker = 'o')
ax1.axvline(x = FLUXERR_RATIO_LIM, label = 'flux/flux_err lower lim', c = 'k')
ax1.axhline(y = ATLAS_CHI_N_LOWLIM, label = 'chi/N lower lim', c = 'r')
ax1.axhline(y = ATLAS_CHI_N_UPLIM, label = 'chi/N upper lim', c = 'g')
ax1.set_xlabel('flux/flux_err')
ax1.legend()
ax1.grid()
ax1.set_ylabel('chi/N')
ax1.set_title('ALL raw ATLAS data (Gaia18cdj). only data within the thin \nhorizontal rectangle on the right made the cut')

ATLAS_chi_fr = ATLAS_raw_data[ATLAS_raw_data['chi/N'] < ATLAS_CHI_N_UPLIM].copy()
ATLAS_chi_fr = ATLAS_chi_fr[ATLAS_chi_fr['chi/N'] > ATLAS_CHI_N_LOWLIM].copy()
ATLAS_chi_fr = ATLAS_chi_fr[(ATLAS_chi_fr['flux'] / ATLAS_chi_fr['flux_err']) > FLUXERR_RATIO_LIM].copy()
ax2 = plt.subplot(1, 2, 2)
ax2.scatter((np.arange(1, len(ATLAS_chi_fr['MJD'])+1, 1)), (ATLAS_chi_fr['5sig_mag_uplim'] - ATLAS_chi_fr['mag']))
ax2.set_ylabel('5sig mag upper lim - mag (detectable if +ve)')
ax2.set_xlabel('arbitrary counter')
ax2.set_title('This data is the raw ATLAS data after applying the chi/N \n and flux/flux_err filters (data within the thin rectange on the \nright of the other plot)')
ax2.grid()
ax2.axhline(y=0, c='k')

plt.show() """


##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################



# PS1-10adi
""" path1 = file_load_dir + "PS1-10adi_lc_files/PS1-10adi_lc_rest_UBVRIJH"
colnames = ['Days_since_peak', 'U_mag', 'U_magerr', 'B_mag', 'B_magerr', 'V_mag', 'V_magerr', 'R_mag', 'R_magerr', 'I_mag', 'I_magerr', 'J_mag', 'J_magerr', 'H_mag', 'H_magerr']
paper_lc = pd.read_csv(path1, delim_whitespace = True, header = None, names = colnames)

band_names = ['U', 'B', 'V', 'R', 'I', 'J', 'H']
band_names = [name+' Vega?'for name in band_names]
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
combined_paper_lc = combined_paper_lc[combined_paper_lc['magerr'] < MAGERR_LIM].copy()
combined_paper_lc['MJD'] = combined_paper_lc['days_since_peak'] + 2455443 - 2400000.5
print(combined_paper_lc)

savepath = file_save_dir + "PS1-10adi_FULL_LC.csv"
combined_paper_lc.to_csv(savepath, index = False)
 """



##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################




# PS1-13jw
""" path1 = file_load_dir + "PS1-13jw_lc_files/PS1-13jw_abs_V_2"
paper_lc = pd.read_csv(path1, delim_whitespace = True, header = None, names = ['days_since_peak', 'mag', 'magerr'])
paper_lc['MJD'] = paper_lc['days_since_peak'] + 2456435 - 2400000.5
paper_lc['band'] = ['V Vega?']*len(paper_lc['days_since_peak'])
paper_lc = paper_lc[paper_lc['magerr'] < MAGERR_LIM].copy()
savepath = file_save_dir + "PS1-13jw_FULL_LC.csv"
paper_lc.to_csv(savepath, index = False)
print(paper_lc)
 """


##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################

# Phil's lightcurves
""" 
# ADDING TIME SINCE PEAK INTO PHIL'S ANT LIGHTCURVE DATA AND REMOVING OUTLIERS
# loading in the files
folder_path = "C:/Users/laure/OneDrive/Desktop/YoRiS Desktop/YoRiS Data/Phil's lightcurves 17_9" # folder path containing the light curve data files
for file in os.listdir(folder_path): # file is a string of the file name such as 'file_name.dat'
    file_path = os.path.join(folder_path, file) 
    file_df = pd.read_csv(file_path, delimiter = ' ') # the lightcurve data in a dataframe
    ANT_name = file[:-7]

    # REMOVING THE OUTLIERS / 
    file_df = file_df[file_df['magerr'] > MIN_MAGERR].copy() # remove the datapoints which have magerr = 0.0000 because this screws with the binning for removing outliers
    #                                                         these would be considered outliers anyway because (weighted_mean_mag - datapoint_mag)/magerr = infinity because magerr = 0.00
    file_df = file_df[file_df['magerr'] < MAGERR_LIM].copy()


    # calculating time since peak column using max of the ZTF_g band NEED TO REMOVE OUTLIERS FOR THIS TO WORK
    ZTF_g = file_df[file_df['band'] == 'ZTF_g'].copy()
    min_ZTF_g = ZTF_g['mag'].min()
    peak_data = ZTF_g[ZTF_g['mag'] == min_ZTF_g].copy()
    peak_MJD = peak_data['MJD'].iloc[0]

    file_df['peak_MJD'] = peak_MJD
    file_df['t_since_peak'] = file_df['MJD'] - peak_MJD


    #print(file_df)
    print(file_df['band'].unique())
    file_df = fix_ANT_bandnames(file_df).copy()
    print(file_df['band'].unique())
    print()
    print()
    #print(file_df.columns)
    file_df.to_csv(f"C:/Users/laure/OneDrive/Desktop/YoRiS Desktop/YoRiS Data/modified Phil's lightcurves/{ANT_name}_lc.csv", index = False)

 """


##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################
##############################################################################################################################################################

""" 
# the files below have bands which are given in the superbol readable format, here is the translation:
column_renaming = {
    '#MJD': 'MJD',
    'W':'WISE_W1',
    'Q':'WISE_W2',
    'g':'ZTF_g',
    'r':'ZTF_r',
    'i':'PS_i',
    'z':'PS_z',
    'w':'PS_w',
    'y':'PS_y',
    'c':'ATLAS_c',
    'o':'ATLAS_o',
    'B':'UVOT_B',
    'U':'UVOT_U',
    'V':'UVOT_V',
    'S':'UVOT_UVW2',
    'D':'UVOT_UVM2',
    'A':'UVOT_UVW1'
    
}


folder_path = "C:/Users/laure/OneDrive/Desktop/YoRiS Desktop/YoRiS Data/Phil's lcs updated mrjar + daps" # folder path containing the light curve data files

for file in os.listdir(folder_path): # file is a string of the file name such as 'file_name.dat'
    file_path = os.path.join(folder_path, file)
    file_df = pd.read_csv(file_path, delimiter = ' ')
    file_df.replace('n', np.nan, inplace = True)
    ANT_name = file[:-12]

    # sort out the column names
    columns = file_df.columns
    for i, col in enumerate(columns):
        if len(col) == 1 or col == '#MJD':
            file_df = file_df.rename(columns = {col: column_renaming[col]})
        
        else: 
            band = columns[i-1]
            bandname = column_renaming[band]
            name = f'{bandname}_err'
            file_df = file_df.rename(columns = {col:name})
            
    
    print(file)
    print(file_df.columns)
    print()
    print(file_df)
    print()

    band_df_list = []
    for i, col in enumerate(file_df.columns):
        # ignore MJD column
        if i==0: 
            continue
        
        # ignore the error columns
        if col[-3:] == 'err':
            continue
        
        cols = ['MJD', col, f'{col}_err'] # the columns we want to take for this band's dataframe, i.e. the MJD, the band's mag and the band's magerr
        band_df = file_df[cols].dropna().copy()
        band_df['band'] = [col]*len(band_df['MJD'])
        band_df = band_df.rename(columns = {col:'mag', f'{col}_err': 'magerr'})
        band_df_list.append(band_df)

    df0 = band_df_list[0]
    dataframe = df0
    for i, band_df in enumerate(band_df_list):
        if i==0:
            continue
        
        dataframe = pd.concat([dataframe, band_df])

    # create time since peak column using ZTF_g peak as the peak of the lightcurve
    ZTF_g_band = dataframe[dataframe['band'] == 'ZTF_g'].copy()
    ZTF_g_peakmag = ZTF_g_band[ZTF_g_band['mag'] == ZTF_g_band['mag'].min()].copy()
    ZTF_g_peakMJD = ZTF_g_peakmag['MJD'].iloc[0] # the MJD associated with the peak of ZTF_g band
    dataframe['peak_MJD'] = [ZTF_g_peakMJD]*len(dataframe['MJD'])
    dataframe['t_since_peak'] = dataframe['MJD'] - ZTF_g_peakMJD
    dataframe['magerr'] = dataframe['magerr'].astype(float)

    
    

    dataframe = dataframe[dataframe['magerr'] > MIN_MAGERR].copy() # GET RID OF DATAPOINTS WITH VERY SMALL MAGERRS, LOOK AT MIN_MAGERR AT THE TOP TO SEE WHAT THIS LIMIT IS 
    

    # save to CSV
    savepath = f"C:/Users/laure/OneDrive/Desktop/YoRiS Desktop/YoRiS Data/Phil's lcs updated modified/{ANT_name}_lc.csv"
    dataframe.to_csv(savepath, index = False)
    print(dataframe)
    

    print()
    print()
    print('=========================================================================================================================================================================')
    print()

 """
#print()

