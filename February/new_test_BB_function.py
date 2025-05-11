import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import scipy.optimize as opt
from tqdm import tqdm
from matplotlib import cm
from matplotlib.colors import Normalize
from colorama import Fore, Style
sys.path.append("C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS") # this allows us to import plotting preferences and functions
from plotting_preferences import band_colour_dict, band_ZP_dict, band_obs_centwl_dict, ANT_redshift_dict, ANT_luminosity_dist_cm_dict, MJDs_for_fit, band_marker_dict
from functions import load_interp_ANT_data, blackbody, chisq, fit_SED_across_lightcurve

pd.options.display.float_format = '{:.4e}'.format # from chatGPT - formats all floats in the dataframe in standard form to 4 decimal places



#===============================================================================================================================================
#===============================================================================================================================================
#===============================================================================================================================================
#===============================================================================================================================================
#===============================================================================================================================================
#===============================================================================================================================================
#===============================================================================================================================================
#===============================================================================================================================================
#===============================================================================================================================================




running_on_server = False 
# load in the interpolated data
interp_df_list, transient_names, list_of_bands = load_interp_ANT_data(running_on_server = running_on_server)

SED_plots = 'usual'#'compare_SEDs' 




#===============================================================================================================================================
#===============================================================================================================================================
#===============================================================================================================================================
#===============================================================================================================================================
#===============================================================================================================================================
#===============================================================================================================================================
#===============================================================================================================================================
#===============================================================================================================================================
#===============================================================================================================================================
# 0 = ASASSN-17jz
# 1 = ASASSN-18jd
# 2 = PS1-10adi
# 3 = ZTF18aczpgwm
# 4 = ZTF19aailpwl
# 5 = ZTF19aamrjar
# 6 = ZTF19aatubsj
# 7 = ZTF20aanxcpf
# 8 = ZTF20abgxlut
# 9 = ZTF20abodaps
# 10 = ZTF20abrbeie
# 11 = ZTF20acvfraq
# 12 = ZTF21abxowzx
# 13 = ZTF22aadesap

if SED_plots == 'usual':
    for idx in range(len(transient_names)):
    #for idx in [1]:

        ANT_name = transient_names[idx]
        interp_lc= interp_df_list[idx]
        ANT_bands = list_of_bands[idx]
        print()
        print(ANT_name)
        
        #if idx == 10:
        #    interp_lc = interp_lc[~interp_lc['band'].isin(['UVOT_B', 'UVOT_U', 'UVOT_UVM2', 'UVOT_UVW1', 'UVOT_UVW2', 'UVOT_V'])]
        #interp_lc = interp_lc[~interp_lc['band'].isin(['UVOT_B', 'UVOT_U', 'UVOT_UVM2', 'UVOT_UVW1', 'UVOT_UVW2', 'UVOT_V'])]

        # FITTING METHOD
        BB_curvefit = True
        BB_brute = True
        #SED_type = 'single_BB'
        SED_type = 'double_BB'
        #SED_type = 'power_law'
        #SED_type = 'best_SED'
        UVOT_guided_fitting = True # if True, will call run_UVOT_guided_SED_fitting_process() instead of run_SED_fitting process(). When an ANT has UVOT on the rise/peak, will use the UVOT SED fit results to constrain the parameter space to search for the nearby non-UVOT SED fits
        UVOT_guided_err_scalefactor = 0.1 
        UVOT_guided_sigma_dist_for_good_fit = 3.0 # the max reduced chi squared sigma distance that we will accept that the model is a good fit to the data
        
        brute_gridsize = 2000
        brute_delchi = 2.3 # = 2.3 to consider parameters jointly for a 1 sigma error. good if you want to quote their value but if you're going to propagate the errors I think u need to use = 1, which considers them 'one at a time'
        DBB_brute_gridsize = 10
        error_sampling_size = 9 # the number of times to sample the error in the data points to get a distribution of chi squared values for each fit. This is used to calculate the reduced chi squared value and the goodness of fit metric D_sigma

        no_indiv_SED_plots = 12 # current options are 24, 20, 12
        individual_BB_plot_type = 'whole_lc' # 'UVOT'

        plot_chi_contour = False
        no_chi_contours = 3 # the number of chi contour plots you want to plot per ANT. This only works for the power law brute force fits though

        # SAVING PLOTS
        save_BB_plot = True
        save_indiv_BB_plot = True
        save_param_vs_time_plot = True

        # SAVE SED FIT RESULTS TO A DATAFRAME?
        save_SED_fit_file = True
        


        BB_fitting = fit_SED_across_lightcurve(interp_lc, running_on_server = running_on_server, SED_type = SED_type, curvefit = BB_curvefit, brute = BB_brute, ant_name = ANT_name, 
                                            brute_delchi = brute_delchi,  brute_gridsize = brute_gridsize, DBB_brute_gridsize = DBB_brute_gridsize, error_sampling_size = error_sampling_size, individual_BB_plot = individual_BB_plot_type, 
                                            no_indiv_SED_plots = no_indiv_SED_plots, save_SED_fit_file = save_SED_fit_file,
                                            save_indiv_BB_plot = save_indiv_BB_plot, save_param_vs_time_plot = save_param_vs_time_plot,
                                            plot_chi_contour = plot_chi_contour, no_chi_contours = no_chi_contours)
        
        #===============================================================================================================================================
        #===============================================================================================================================================
        #===============================================================================================================================================
        #===============================================================================================================================================
        #===============================================================================================================================================
        #===============================================================================================================================================
        #===============================================================================================================================================
        #===============================================================================================================================================
        #===============================================================================================================================================
        if UVOT_guided_fitting == True:
            BB_fit_results = BB_fitting.run_UVOT_guided_SED_fitting_process(err_scalefactor = UVOT_guided_err_scalefactor, sigma_dist_for_good_fit = UVOT_guided_sigma_dist_for_good_fit, band_colour_dict = band_colour_dict, band_marker_dict = band_marker_dict)
        


        if (SED_type == 'best_SED') and (UVOT_guided_fitting == False): # I only require UVOT_guided_fitting == False because otherwise if it's True, then the if statement above will activate, 
                                                                        # but then another one of the ones below will run straight after since both UVOT_guided_fitting == True and SED_type will be 
                                                                        # one of 'best_SED', 'single_BB', 'double_BB', 'power_law'
            BB_fit_results = BB_fitting.run_SED_fitting_process(band_colour_dict=band_colour_dict, band_marker_dict = band_marker_dict)
        


        if (SED_type == 'double_BB') and (UVOT_guided_fitting == False):
            BB_fit_results = BB_fitting.run_SED_fitting_process(band_colour_dict=band_colour_dict, band_marker_dict = band_marker_dict)



        if (SED_type == 'power_law') and (UVOT_guided_fitting == False):
            BB_fit_results = BB_fitting.run_SED_fitting_process(band_colour_dict=band_colour_dict, band_marker_dict = band_marker_dict)
            pd.options.display.float_format = '{:.4e}'.format # from chatGPT - formats all floats in the dataframe in standard form to 4 decimal places

        
        
        if (SED_type == 'single_BB') and (UVOT_guided_fitting == False):
            BB_fit_results = BB_fitting.run_SED_fitting_process(band_colour_dict=band_colour_dict, band_marker_dict = band_marker_dict)
            BB_2dp = BB_fit_results[BB_fit_results['no_bands'] == 2] # the BB fits for the MJDs which only had 2 bands, so we aren't really fitting, more solving for the BB R and T which perfectly pass through the data points
            
 






#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
# TESTING WHICH SED BEST FITS THE ANT

if SED_plots == 'compare_SEDs':
    if running_on_server:
        base_path = "" # we don't need a base path for the server we use the relative path
    else:
        base_path = "C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS/"

    #for idx in range(len(transient_names)):
    for idx in [0]:
        ANT_name = transient_names[idx]
        interp_lc= interp_df_list[idx]
        ANT_bands = list_of_bands[idx]
        print('=================================================================================================')
        print(ANT_name)

        # FITTING METHOD
        brute_gridsize = 2000
        brute_delchi = 2.3 # = 2.3 to consider parameters jointly for a 1 sigma error. good if you want to quote their value but if you're going to propagate the errors I think u need to use = 1, which considers them 'one at a time'
        individual_BB_plot_type = 'whole_lc'
        no_indiv_SED_plots = 12 # current options are 24, 20, 12
        plot_chi_contour = False
        no_chi_contours = 3 

        UVOT_guided_fitting = True # if True, will call run_UVOT_guided_SED_fitting_process() instead of run_SED_fitting process(). When an ANT has UVOT on the rise/peak, will use the UVOT SED fit results to constrain the parameter space to search for the nearby non-UVOT SED fits
        UVOT_guided_err_scalefactor = 0.1 
        UVOT_guided_sigma_dist_for_good_fit = 3.0 # the max reduced chi squared sigma distance that we will accept that the model is a good fit to the data
        
        DBB_brute_gridsize = 10
        error_sampling_size = 9 # the number of times to sample the error in the data points to get a distribution of chi squared values for each fit. This is used to calculate the reduced chi squared value and the goodness of fit metric D_sigma


        # SAVING PLOTS
        save_BB_plot = True
        save_indiv_BB_plot = True
        save_param_vs_time_plot = True
        show_plots = False

        # SAVE SED FIT RESULTS TO A DATAFRAME?
        save_SED_fit_file = True

        # create a dataframe to save the comparison results to 
        comparison_df = pd.DataFrame(columns = ['SED_model', 'phase_subset', 'median_D', 'MAD_D', 'no_fits_N_greater_M'])
        #comparison_df['SED_model'] = ['Single-BB', 'Single-BB', 'Single-BB', 'Single-BB', 'Power-law', 'Power-law', 'Power-law', 'Power-law', 'Double-BB', 'Double-BB', 'Double-BB', 'Double-BB']

        fig, axs = plt.subplots(2, 2, figsize = (16.5, 7.5))
        ax1, ax2, ax3, ax4 = axs.ravel()

        for i, SED_type in enumerate(['single_BB', 'double_BB', 'power_law']):
            if SED_type == 'double_BB': # since we don't want to do DBB fits for the ANTs without much UVOT data, we will skip over them
                if ANT_name not in ['ZTF19aailpwl', 'ZTF20acvfraq', 'ZTF22aadesap', 'ASASSN-17jz', 'ASASSN-18jd', 'PS1-10adi']:
                    continue
                
            if SED_type == 'single_BB':
                SED_label = 'Single-BB'
                BB_curvefit = False
                BB_brute = True
                sig_dist_colname = 'brute_chi_sigma_dist'
                SED_colour = '#02f5dd'
                SED_ecolour = '#0f3d7d'
                SED_linestyle = '-'
                SED_hatch = '\\'
                M = 2

                
            elif SED_type == 'double_BB':
                SED_label = 'Double-BB'
                BB_curvefit = True
                BB_brute = False
                sig_dist_colname = 'cf_chi_sigma_dist'
                SED_colour = 'k'
                SED_ecolour = 'k'
                SED_linestyle = '-'
                SED_hatch = '/'
                M = 4


            elif SED_type == 'power_law':
                SED_label = 'Power-law'
                BB_curvefit = False
                BB_brute = True
                sig_dist_colname = 'brute_chi_sigma_dist'
                SED_colour = '#f502dd'
                SED_ecolour = '#780d61'
                SED_linestyle = '-'
                M = 2

    
            
            BB_fitting = fit_SED_across_lightcurve(interp_lc, running_on_server = running_on_server, SED_type = SED_type, curvefit = BB_curvefit, brute = BB_brute, ant_name = ANT_name, 
                                            brute_delchi = brute_delchi,  brute_gridsize = brute_gridsize, DBB_brute_gridsize = DBB_brute_gridsize, error_sampling_size = error_sampling_size, individual_BB_plot = individual_BB_plot_type, 
                                            no_indiv_SED_plots = no_indiv_SED_plots, show_plots = show_plots, save_SED_fit_file = save_SED_fit_file,
                                            save_indiv_BB_plot = save_indiv_BB_plot, save_param_vs_time_plot = save_param_vs_time_plot,
                                            plot_chi_contour = plot_chi_contour, no_chi_contours = no_chi_contours)
            

            def median_absolute_deviation(median, data): # need to maybe handle the possibility that data is empty and median is None?
                if len(data) == 0:
                    return np.nan
                MAD = np.sum(abs(median - data)) / len(data)
                return MAD

            # the whole lc SED fits
            if UVOT_guided_fitting:
                BB_fit_results = BB_fitting.run_UVOT_guided_SED_fitting_process(err_scalefactor = UVOT_guided_err_scalefactor, sigma_dist_for_good_fit = UVOT_guided_sigma_dist_for_good_fit, band_colour_dict = band_colour_dict, band_marker_dict = band_marker_dict)
            else:
                BB_fit_results = BB_fitting.run_SED_fitting_process(band_colour_dict=band_colour_dict, band_marker_dict = band_marker_dict)
            BB_fit_results = BB_fit_results[BB_fit_results[sig_dist_colname].notna()]
            all_lc_median = BB_fit_results[sig_dist_colname].median(skipna = True)
            all_lc_MAD = median_absolute_deviation(all_lc_median, BB_fit_results[sig_dist_colname].to_numpy())
            # save to the comparison results df
            comparison_df.loc[(i*4 + 0), 'SED_model'] = SED_label
            comparison_df.loc[(i*4 + 0), 'phase_subset'] = 'whole_lc'
            comparison_df.loc[(i*4 + 0), ['median_D', 'MAD_D', 'no_fits_N_greater_M']] = [all_lc_median, all_lc_MAD, len(BB_fit_results)]


            # SED fits with only optical data
            UV_wavelength_threshold = 2800 # angstrom
            bin_by_MJD = interp_lc.groupby('MJD', observed = True).apply(lambda g: pd.Series({'UVOT?': (g['em_cent_wl']< UV_wavelength_threshold).any() })).reset_index()
            MJDs_with_UVOT = bin_by_MJD[bin_by_MJD['UVOT?'] == True]['MJD'].to_numpy() # an array of the MJDs at which we have UVOT data
            MJDs_without_UVOT = bin_by_MJD[bin_by_MJD['UVOT?'] == False]['MJD'].to_numpy() # used in the next function where we SED fit the non-UVOT MJDs



            SEDs_without_UVOT = BB_fit_results[BB_fit_results['MJD'].isin(MJDs_without_UVOT)].copy()
            no_UVOT_median = SEDs_without_UVOT[sig_dist_colname].median()
            no_UVOT_MAD = median_absolute_deviation(no_UVOT_median, SEDs_without_UVOT[sig_dist_colname].to_numpy())
            # save to the comparison results df
            comparison_df.loc[(i*4 + 1), 'SED_model'] = SED_label
            comparison_df.loc[(i*4 + 1), 'phase_subset'] = 'no_UVOT'
            comparison_df.loc[(i*4 + 1), ['median_D', 'MAD_D', 'no_fits_N_greater_M']] = [no_UVOT_median, no_UVOT_MAD, len(SEDs_without_UVOT)]


            # SED fits with UVOT data
            SEDs_with_UVOT = BB_fit_results[BB_fit_results['MJD'].isin(MJDs_with_UVOT)].copy()
            UVOT_median = SEDs_with_UVOT[sig_dist_colname].median()
            UVOT_MAD = median_absolute_deviation(UVOT_median, SEDs_with_UVOT[sig_dist_colname].to_numpy())
            # save to the comparison results df
            comparison_df.loc[(i*4 + 2), 'SED_model'] = SED_label
            comparison_df.loc[(i*4 + 2), 'phase_subset'] = 'with_UVOT'
            comparison_df.loc[(i*4 + 2), ['median_D', 'MAD_D', 'no_fits_N_greater_M']] = [UVOT_median, UVOT_MAD, len(SEDs_with_UVOT)]
            

            # SED fits on rise/near peak
            SEDs_rise_peak = BB_fit_results[BB_fit_results['d_since_peak'] <= 100].copy()
            rise_peak_median = SEDs_rise_peak[sig_dist_colname].median(skipna = True)
            rise_peak_MAD = median_absolute_deviation(rise_peak_median, SEDs_rise_peak[sig_dist_colname].to_numpy())
            # save to the comparison results df
            comparison_df.loc[(i*4 + 3), 'SED_model'] = SED_label
            comparison_df.loc[(i*4 + 3), 'phase_subset'] = 'phase_leq_100'
            comparison_df.loc[(i*4 + 3), ['median_D', 'MAD_D', 'no_fits_N_greater_M']] = [rise_peak_median, rise_peak_MAD, len(SEDs_rise_peak)]


            # PLOTTING IT UP IN HISTOGRAMS
            label1 = fr'$\mathbf{{{SED_label}}}$ '+'\n'+fr'Median $D_{{\sigma}}$ = {all_lc_median:.3g}'+'\n'+fr'MAD $D_{{\sigma}}$ = {all_lc_MAD:.3g}'
            ax1.hist(BB_fit_results[sig_dist_colname], bins = 50, color = SED_colour, label = label1, alpha = 0.5)
            ax1.hist(BB_fit_results[sig_dist_colname], bins = 50, edgecolor = SED_ecolour, color = SED_colour, alpha = 0.75, histtype='step', linewidth = 1.5, linestyle = SED_linestyle)#, hatch = SED_hatch)

            label2 = fr'$\mathbf{{{SED_label}}}$ '+'\n'+fr'Median $D_{{\sigma}}$ = {no_UVOT_median:.3g} '+'\n'+fr'MAD $D_{{\sigma}}$ = {no_UVOT_MAD:.3g}'
            ax2.hist(SEDs_without_UVOT[sig_dist_colname], bins = 50, color = SED_colour, label = label2, alpha = 0.5)
            ax2.hist(SEDs_without_UVOT[sig_dist_colname], bins = 50, edgecolor = SED_ecolour, color = SED_colour, alpha = 0.75, histtype='step', linewidth = 1.5, linestyle = SED_linestyle)#, hatch = SED_hatch)

            label3 = fr'$\mathbf{{{SED_label}}}$ '+'\n'+fr'Median $D_{{\sigma}}$ = {rise_peak_median:.3g} '+'\n'+fr'MAD $D_{{\sigma}}$ = {rise_peak_MAD:.3g}'
            ax3.hist(SEDs_rise_peak[sig_dist_colname], bins = 50, color = SED_colour, label = label3, alpha = 0.5)
            ax3.hist(SEDs_rise_peak[sig_dist_colname], bins = 50, edgecolor = SED_ecolour, color = SED_colour,  alpha = 0.75, histtype='step', linewidth = 1.5, linestyle = SED_linestyle)#, hatch = SED_hatch)

            label4 = fr'$\mathbf{{{SED_label}}}$ '+'\n'+fr'Median $D_{{\sigma}}$ = {UVOT_median:.3g} '+'\n'+fr'MAD $D_{{\sigma}}$ = {UVOT_MAD:.3g}'
            ax4.hist(SEDs_with_UVOT[sig_dist_colname], bins = 50, color = SED_colour, label = label4, alpha = 0.5)
            ax4.hist(SEDs_with_UVOT[sig_dist_colname], bins = 50, edgecolor = SED_ecolour, color = SED_colour, alpha = 0.75, histtype='step', linewidth = 1.5, linestyle = SED_linestyle)#, hatch = SED_hatch)

        ax1.set_title('Entire lightcurve', fontweight = 'bold')
        ax2.set_title('Phases without UV data', fontweight = 'bold')
        ax3.set_title(r'Phases $\mathbf{\leq100}$ days', fontweight = 'bold')
        ax4.set_title('Phases with UV data', fontweight = 'bold')

        for ax in [ax1, ax2, ax3, ax4]:
            ax.legend()
            ax.grid(True)

        titlefontsize = 20
        fig.supxlabel(r'Goodness-of-fit metric, $\mathbf{ D_{\sigma\chi} }$', fontsize = (titlefontsize - 5), fontweight = 'bold')
        fig.supylabel('Frequency density', fontweight = 'bold', fontsize = (titlefontsize - 5))
        title = f"Comparing SED Model Fit Quality Across Different Phase Subsets in \n {ANT_name}'s Light Curve"
        fig.suptitle(title, fontweight = 'bold', fontsize = titlefontsize)
        fig.subplots_adjust(top=0.82,
                            bottom=0.11,
                            left=0.08,
                            right=0.955,
                            hspace=0.2,
                            wspace=0.2)
        savepath = base_path + f"plots/BB fits/proper_BB_fits/{ANT_name}/{ANT_name}_compare_SED_fits.png"
        plt.savefig(savepath, dpi = 300)
        
        print(comparison_df.head(12))
        savepath = base_path + f"data/SED_fits/{ANT_name}/{ANT_name}_compare_SED_models.csv"
        comparison_df.to_csv(savepath, index = False)











        

        

            




