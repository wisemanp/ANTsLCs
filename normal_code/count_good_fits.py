import pandas as pd
import numpy as np
import sys
import os
import json
import ast


pd.options.display.float_format = '{:.4e}'.format # from chatGPT - formats all floats in the dataframe in standard form to 4 decimal places




SED_fits_path = "/media/data3/lauren/YoRiS/normal_code/data/SED_fits"



for ANT_folder in os.listdir(SED_fits_path): # we're essentially iterating through the ANTs here (we're iterating thrrough their folders containing SED fit results)
    ANT_name = ANT_folder
    ANT_folder_path = os.path.join(SED_fits_path, ANT_folder)
    print()
    print()
    print('================================================================')
    print(ANT_name)
    print('================================================================')
    print()



    results_df = pd.DataFrame(columns = ['SED_model', 'phase_region', 'count_N_greater_M', 'count_good_fits', 'percent_good_fit'])
    for ANT_SED_fit_file in os.listdir(ANT_folder_path): # we're now iterating through the SED fit files for each ANT. 
        print(ANT_SED_fit_file)
        SED_model = ANT_SED_fit_file.split('_')[1]
        check_sampled = ANT_SED_fit_file.split('_')[-2] # remove the files of sampled params
        check_new_file = ANT_SED_fit_file.split('_')[-1] #  we want the new SED fit results
        if SED_model == 'UVOT': # correct for the UVOTq-guided file names
            SED_model = ANT_SED_fit_file.split('_')[3]

        if SED_model in ['compare', 'SED']: # remove the files which are not the SED fit result files
            continue

        if check_sampled =='sampled':
            continue

        #if ANT_name in ['ZTF19aailpwl', 'ZTF20acvfraq', 'ZTF22aadesap', 'ASASSN-17jz', 'ASASSN-18jd']:
        if check_new_file != 'new.csv':
            continue
        print()
        print()
        print(ANT_SED_fit_file, '==================================================================')
        
        SED_file_path = os.path.join(ANT_folder_path, ANT_SED_fit_file)
        SED_fit_df = pd.read_csv(SED_file_path, delimiter = ',')
        SED_fit_df['bands'] = SED_fit_df['bands'].apply(ast.literal_eval)

        # filtering through the data frame for early light curve, optical-only and UV epochs
        early_SED_fit_df = SED_fit_df[SED_fit_df['d_since_peak'] <= 100.0].copy() # a dataframe fo early light curve SED fits

        UV_wavelength_threshold = 3000 # angstrom
        #UV_bands = ['UVOT_B', 'UVOT_U', 'UVOT_UVM2', 'UVOT_UVW1', 'UVOT_UVW2', 'UVOT_V']
        #no_UV_SED_fit_df = SED_fit_df[SED_fit_df['bands'].apply(lambda rows_bands: not any(b in UV_bands for b in rows_bands))] # remove SED fits which had UV data
        SED_fit_df['em_cent_wls'] = SED_fit_df['em_cent_wls'].apply(ast.literal_eval)
        no_UV_SED_fit_df = SED_fit_df[SED_fit_df['em_cent_wls'].apply(lambda rows_em_cent_wl: not any(em_cent_wl < UV_wavelength_threshold for em_cent_wl in rows_em_cent_wl))]
        UV_SED_fit_df = SED_fit_df[SED_fit_df['em_cent_wls'].apply(lambda rows_em_cent_wl: any(em_cent_wl < UV_wavelength_threshold for em_cent_wl in rows_em_cent_wl))]

        #UV_SED_fit_df = SED_fit_df[SED_fit_df['bands'].apply(lambda rows_bands: any(b in UV_bands for b in rows_bands))] # take only SED fits taken to UV data. 



        # count the number of fits for which sig dist is not NA (i.e. number of fits for which N>M) for the whole light curve and the early light curve
        # and also the number of fits considered 'good' in these regions.

        if SED_model in ['PL', 'SBB']:
            SED_good_fit_df = SED_fit_df.dropna(subset = ['brute_chi_sigma_dist']).copy()
            whole_lc_N_greater_M = len(SED_good_fit_df) # the number of fits where N>M
            SED_good_fit_df = SED_good_fit_df[SED_good_fit_df['brute_chi_sigma_dist'].abs() <= 3.0].copy()
            whole_lc_num_good_fits = len(SED_good_fit_df) # the numeber of fits considered 'good'

            early_lc_good_fit_df = early_SED_fit_df.dropna(subset = ['brute_chi_sigma_dist']).copy()
            early_lc_N_greater_M = len(early_lc_good_fit_df) # the number of early light curve fits where N>M
            early_lc_good_fit_df = early_lc_good_fit_df[early_lc_good_fit_df['brute_chi_sigma_dist'].abs() <= 3.0].copy()
            early_lc_num_good_fits = len(early_lc_good_fit_df) # the number of early light curve fits which are considered 'good'

            no_UV_good_fit_df = no_UV_SED_fit_df.dropna(subset = ['brute_chi_sigma_dist']).copy()
            no_UV_N_greater_M = len(no_UV_good_fit_df)
            no_UV_good_fit_df = no_UV_good_fit_df[no_UV_good_fit_df['brute_chi_sigma_dist'].abs() <= 3.0].copy()
            no_UV_num_good_fits = len(no_UV_good_fit_df)

            UV_good_fit_df = UV_SED_fit_df.dropna(subset = ['brute_chi_sigma_dist']).copy()
            UV_N_greater_M = len(UV_good_fit_df)
            UV_good_fit_df = UV_good_fit_df[UV_good_fit_df['brute_chi_sigma_dist'].abs() <= 3.0].copy()
            UV_num_good_fits = len(UV_good_fit_df)
            


        elif SED_model == 'DBB':
            SED_good_fit_df = SED_fit_df.dropna(subset = ['cf_chi_sigma_dist']).copy()
            whole_lc_N_greater_M = len(SED_good_fit_df) # the number of fits where N>M
            SED_good_fit_df = SED_good_fit_df[SED_good_fit_df['cf_chi_sigma_dist'].abs() <= 3.0].copy()
            whole_lc_num_good_fits = len(SED_good_fit_df) # the numeber of fits considered 'good'

            early_lc_good_fit_df = early_SED_fit_df.dropna(subset = ['cf_chi_sigma_dist']).copy()
            early_lc_N_greater_M = len(early_lc_good_fit_df) # the number of early light curve fits where N>M
            early_lc_good_fit_df = early_lc_good_fit_df[early_lc_good_fit_df['cf_chi_sigma_dist'].abs() <= 3.0].copy()
            early_lc_num_good_fits = len(early_lc_good_fit_df) # the number of early light curve fits which are considered 'good'

            no_UV_good_fit_df = no_UV_SED_fit_df.dropna(subset = ['cf_chi_sigma_dist']).copy()
            no_UV_N_greater_M = len(no_UV_good_fit_df)
            no_UV_good_fit_df = no_UV_good_fit_df[no_UV_good_fit_df['cf_chi_sigma_dist'].abs() <= 3.0].copy()
            no_UV_num_good_fits = len(no_UV_good_fit_df)

            UV_good_fit_df = UV_SED_fit_df.dropna(subset = ['cf_chi_sigma_dist']).copy()
            UV_N_greater_M = len(UV_good_fit_df)
            UV_good_fit_df = UV_good_fit_df[UV_good_fit_df['cf_chi_sigma_dist'].abs() <= 3.0].copy()
            UV_num_good_fits = len(UV_good_fit_df)


        # for each SED, we can now save these numbers for the whole_lc and early_lc into 2 rows of a dataframe
        def percentage(N, D):
            if N == 0.0:
                return 0.0
            elif D == 0.0:
                return 0.0
            else:
                return (N/D)*100
        
        row1 = [SED_model, 'whole_lc', whole_lc_N_greater_M, whole_lc_num_good_fits, percentage(whole_lc_num_good_fits, whole_lc_N_greater_M)]
        row2 = [SED_model, 'early_lc', early_lc_N_greater_M, early_lc_num_good_fits, percentage(early_lc_num_good_fits, early_lc_N_greater_M)]
        row3 = [SED_model, 'no_UVOT', no_UV_N_greater_M, no_UV_num_good_fits, percentage(no_UV_num_good_fits, no_UV_N_greater_M)]
        row4 = [SED_model, 'with_UVOT', UV_N_greater_M, UV_num_good_fits, percentage(UV_num_good_fits, UV_N_greater_M)]
        for row in [row1, row2, row3, row4]:
            results_df.loc[len(results_df)] = row


        # For each ANT: save a df like this:
        # SED, phase_region, count_N>M, count_good_fits
        # ----------------------------------------------------
        # SBB, whole_lc    ,   100    ,     82
        # SBB, early_lc    ,    47    ,     15
        # PL , whole_kc    ,    100   ,     56
        # PL ....... 
        # DBB
        # DBB

    
    # save the results into a file
    csv_path = os.path.join(ANT_folder_path, 'good_SED_fits_count.csv')
    results_df.to_csv(csv_path, index = False)


        



