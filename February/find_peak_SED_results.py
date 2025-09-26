import pandas as pd
import numpy as np
import sys
import os
import json


pd.options.display.float_format = '{:.4e}'.format # from chatGPT - formats all floats in the dataframe in standard form to 4 decimal places




SED_fits_path = "C:\\Users\\laure\\OneDrive\\Desktop\\YoRiS desktop\\YoRiS\\data\\SED_fits"



for ANT_folder in os.listdir(SED_fits_path): # we're essentially iterating through the ANTs here (we're iterating thrrough their folders containing SED fit results)
    ANT_name = ANT_folder
    ANT_folder_path = os.path.join(SED_fits_path, ANT_folder)
    print(ANT_name)

    if ANT_name in ['ASASSN-17jz', 'ASASSN-18jd', 'PS1-10adi', 'ZTF18aczpgwm']:
        continue

    peak_results_dict_list = []
    for ANT_SED_fit_file in os.listdir(ANT_folder_path): # we're now iterating through the SED fit files for each ANT. 
        SED_model = ANT_SED_fit_file.split('_')[1]
        if SED_model == 'compare': # this is the dataframe where we summarise ecah model's fit quality, not the data we want here
            continue
        print()
        print()
        print(ANT_SED_fit_file, '==================================================================')
        
        SED_file_path = os.path.join(ANT_folder_path, ANT_SED_fit_file)
        SED_fit_df = pd.read_csv(SED_file_path, delimiter = ',')

        # only take the peak data from a 'good' fit
        if SED_model in ['PL', 'SBB']:
            SED_good_fit_df = SED_fit_df.dropna(subset = ['brute_chi_sigma_dist']).copy()
            SED_good_fit_df = SED_good_fit_df[SED_good_fit_df['brute_chi_sigma_dist'] <= 3.0].copy()

        elif SED_model == 'DBB':
            SED_good_fit_df = SED_fit_df.dropna(subset = ['cf_chi_sigma_dist']).copy()
            SED_good_fit_df = SED_good_fit_df[SED_good_fit_df['cf_chi_sigma_dist'] <= 3.0].copy()

        
        #print(SED_good_fit_df)

        # find the good-fitting SED fit closest to peak
        SED_good_fit_df = SED_good_fit_df.reset_index(drop = False)
        SED_closest_to_peak_idx = SED_good_fit_df['d_since_peak'].abs().idxmin()
        print(SED_closest_to_peak_idx)
        print()
        print()
        
        SED_closest_to_peak =  SED_good_fit_df.iloc[SED_closest_to_peak_idx, :].copy()
        print(SED_closest_to_peak)

        # save the peak SED result
        result = {'ANT': ANT_name, 'SED_model': SED_model}
        result.update(SED_closest_to_peak.to_dict())
        peak_results_dict_list.append(result)

    print()
    print()
    print(peak_results_dict_list)
    print()
    print()
    # save the results into a json file
    JSON_path = os.path.join('peak_SED_fits.json', ANT_folder_path)
    with open(JSON_path, 'w') as json_file:
        json.dump(peak_results_dict_list, json_file, indent = 4)

        







        









