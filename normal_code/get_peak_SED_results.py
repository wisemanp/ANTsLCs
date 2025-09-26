import pandas as pd
import numpy as np
import sys
import os
import json


pd.options.display.float_format = '{:.4e}'.format # from chatGPT - formats all floats in the dataframe in standard form to 4 decimal places




SED_fits_path = "/media/data3/lauren/YoRiS/normal_code/data/SED_fits"



for ANT_folder in os.listdir(SED_fits_path): # we're essentially iterating through the ANTs here (we're iterating thrrough their folders containing SED fit results)
    ANT_name = ANT_folder
    ANT_folder_path = os.path.join(SED_fits_path, ANT_folder)
    print(ANT_name)

    #if ANT_name in ['ASASSN-17jz', 'ASASSN-18jd', 'PS1-10adi', 'ZTF18aczpgwm']:
    #    continue

    peak_results_dict_list = []
    for ANT_SED_fit_file in os.listdir(ANT_folder_path): # we're now iterating through the SED fit files for each ANT. 
        SED_model = ANT_SED_fit_file.split('_')[1]
        if SED_model == 'UVOT': # correct for the UVOTq-guided file names
            SED_model = ANT_SED_fit_file.split('_')[3]

        if SED_model in ['compare', 'SED']: # remove the files which are not the SED fit result files
            continue
        print()
        print()
        print(ANT_SED_fit_file, '==================================================================')
        
        SED_file_path = os.path.join(ANT_folder_path, ANT_SED_fit_file)
        SED_fit_df = pd.read_csv(SED_file_path, delimiter = ',')

        # only take the peak data from a 'good' fit # THE CURRENT ISSUES ARE HERE, SINCE SOME ANTS HAVE ZERO FITS THAT QUALIFY AS GOOD. 
        # IN THESE CIRCUMSTANCES, I THINK WHAT WE SHOULD DO IN THIS SCENARIO IS LIMIT THE SEARCH FOR 'GOOD' FITS NEAR PEAK BY EITHERRRR
        # DOING THIS EXACT SAME PROCESS BUT JUST TAKING THE FIT CLOSEST TO PEAK REGARDLESS OF FIT QUALITY (SINCE THEY'RE ALL BAD ANYWAY, BUT
        # SOME WILL BE LESS BAD THAN OTHERS, ORRR WE SAY LIKE 'FOR |DSP|<100, FIND THE LEAST BAD FIT, ALTHOUGH THIS WORKS BETTER FOR SOME ANTS
        # THAN OTHERS SINCE SOME ANTS EVOLVE SOLOWLY SO A DIFFERENCE OF 100 DAYS ISNT THAT MUCH, WHEREAS OTHERS EVOLBVE MUCH FASTER. 
        #
        # DO WE EVEN NEED TO QUOTE PEAK SED PARAMETERS IF THEY WERE FIT BADLY ANYWAYS? THEY HAVE NO PHYSICAL MEANING...? I GUESS YOU COULD MAKE 
        # AN ARGUMENT THAT THE 'CHI BY EYE' IS GOOD AND YOU JUST THINK YOUR DATAPOINT'S ERRORBARS ARE A BIT TOO SMALL TO ACCOUNT FOR THEIR SCATTER 
        # ABOUT THE SED THAT HAS BEEN FIT, BUT BY EYE YOU WOULD SAY ITS A GOOD FIT.
        if SED_model in ['PL', 'SBB']:
            SED_good_fit_df = SED_fit_df.dropna(subset = ['brute_chi_sigma_dist']).copy()
            SED_good_fit_df = SED_good_fit_df[SED_good_fit_df['brute_chi_sigma_dist'] <= 3.0].copy()

        elif SED_model == 'DBB':
            SED_good_fit_df = SED_fit_df.dropna(subset = ['cf_chi_sigma_dist']).copy()
            SED_good_fit_df = SED_good_fit_df[SED_good_fit_df['cf_chi_sigma_dist'] <= 3.0].copy()

        if SED_good_fit_df.empty: # FOR NOW, JUST DON'T DO THIS CALUCLATION FOR THE SEDS WITH NO GOOD SED FITS
            continue


        
        #print(SED_good_fit_df)

        # find the good-fitting SED fit closest to peak
        SED_good_fit_df = SED_good_fit_df.reset_index(drop = False)
        # AN ISSUE WITH THIS APPROACH IS THAT EACH SED MIGHT GET THE PEAK VALUES QUOTED AT A DIFFERENT DSP, SINCE THE MIN |DSP| MIGHT ONLY BE A GOOD FIT
        # WITH ONE OF THE SEDS, THEN THE OTHER 2 WOULD BE FORCED TO SEARCH FOR THE NEXT BEST DSP, OR THE ONE AFTER THAT AND SO ON. COMPLICATES THINGS WHEN IT 
        # COMES TO PUTTING THE RESULTS IN A TABLE
        SED_closest_to_peak_idx = SED_good_fit_df['d_since_peak'].abs().idxmin()
        
        SED_closest_to_peak =  SED_good_fit_df.iloc[SED_closest_to_peak_idx, :].copy()
        #print(SED_closest_to_peak)

        # save the peak SED result
        result = {'ANT': ANT_name, 'SED_model': SED_model}
        result.update(SED_closest_to_peak.to_dict())
        peak_results_dict_list.append(result)

    
    # save the results into a json file
    JSON_path = os.path.join(ANT_folder_path, 'peak_SED_fits.json')
    with open(JSON_path, 'w') as json_file:
        json.dump(peak_results_dict_list, json_file, indent = 4)

        



