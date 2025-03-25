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
from plotting_preferences import band_colour_dict, band_ZP_dict, band_obs_centwl_dict, ANT_redshift_dict, ANT_luminosity_dist_cm_dict, MJDs_for_fit
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





# load in the interpolated data
interp_df_list, transient_names, list_of_bands = load_interp_ANT_data()

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


if SED_plots == 'usual':
    #for idx in range(11):
    for idx in [8]:

        ANT_name = transient_names[idx]
        interp_lc= interp_df_list[idx]
        ANT_bands = list_of_bands[idx]
        print()
        print(ANT_name)
        
        #if idx == 10:
        #    interp_lc = interp_lc[~interp_lc['band'].isin(['UVOT_B', 'UVOT_U', 'UVOT_UVM2', 'UVOT_UVW1', 'UVOT_UVW2', 'UVOT_V'])]
        #interp_lc = interp_lc[~interp_lc['band'].isin(['UVOT_B', 'UVOT_U', 'UVOT_UVM2', 'UVOT_UVW1', 'UVOT_UVW2', 'UVOT_V'])]

        # FITTING METHOD
        BB_curvefit = False
        BB_brute = True
        #SED_type = 'single_BB'
        #SED_type = 'double_BB'
        #SED_type = 'power_law'
        SED_type = 'best_SED'
        UVOT_guided_fitting = True # if True, will call run_UVOT_guided_SED_fitting_process() instead of run_SED_fitting process(). When an ANT has UVOT on the rise/peak, will use the UVOT SED fit results to constrain the parameter space to search for the nearby non-UVOT SED fits
        UVOT_guided_err_scalefactor = 0.05
        brute_gridsize = 2000
        brute_delchi = 2.3 # = 2.3 to consider parameters jointly for a 1 sigma error. good if you want to quote their value but if you're going to propagate the errors I think u need to use = 1, which considers them 'one at a time'

        no_indiv_SED_plots = 24 # current options are 24, 20, 12

        plot_chi_contour = False
        no_chi_contours = 3 # the number of chi contour plots you want to plot per ANT. This only works for the power law brute force fits though

        # SAVING PLOTS
        save_BB_plot = False
        save_indiv_BB_plot = True
        save_param_vs_time_plot = True

        # SAVE SED FIT RESULTS TO A DATAFRAME?
        save_SED_fit_file = True


        BB_fitting = fit_SED_across_lightcurve(interp_lc, SED_type = SED_type, curvefit = BB_curvefit, brute = BB_brute, ant_name = ANT_name, 
                                            brute_delchi = brute_delchi,  brute_gridsize = brute_gridsize, individual_BB_plot = 'whole_lc', 
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
            BB_fit_results = BB_fitting.run_UVOT_guided_SED_fitting_process(err_scalefactor = UVOT_guided_err_scalefactor, band_colour_dict = band_colour_dict)
        


        if (SED_type == 'best_SED') and (UVOT_guided_fitting == False): # I only require UVOT_guided_fitting == False because otherwise if it's True, then the if statement above will activate, 
                                                                        # but then another one of the ones below will run straight after since both UVOT_guided_fitting == True and SED_type will be 
                                                                        # one of 'best_SED', 'single_BB', 'double_BB', 'power_law'
            BB_fit_results = BB_fitting.run_SED_fitting_process(band_colour_dict=band_colour_dict)
        


        if (SED_type == 'double_BB') and (UVOT_guided_fitting == False):
            BB_fit_results = BB_fitting.run_SED_fitting_process(band_colour_dict=band_colour_dict)



        if (SED_type == 'power_law') and (UVOT_guided_fitting == False):
            BB_fit_results = BB_fitting.run_SED_fitting_process(band_colour_dict=band_colour_dict)
            pd.options.display.float_format = '{:.4e}'.format # from chatGPT - formats all floats in the dataframe in standard form to 4 decimal places
            #print(BB_fit_results.head(20).iloc[:, 10:16])
            #print(BB_fit_results.head(20).loc[:, ['MJD', 'cf_A', 'cf_A_err', 'cf_chi_sigma_dist', 'brute_A', 'brute_A_err', 'brute_chi_sigma_dist']])
            #print()
            #print(BB_fit_results.head(20).loc[:, ['MJD', 'cf_gamma', 'cf_gamma_err', 'cf_chi_sigma_dist', 'brute_gamma', 'brute_gamma_err', 'brute_chi_sigma_dist']])
            #print()
            #print('max A = ', BB_fit_results['cf_A'].max())
            #print('min A = ',  BB_fit_results['cf_A'].min())
            #print()
            #print(BB_fit_results.tail(20).loc[:, ['MJD', 'cf_A', 'cf_A_err', 'cf_chi_sigma_dist', 'brute_A', 'brute_A_err', 'brute_chi_sigma_dist']])
            #print()
            #print(BB_fit_results.tail(20).loc[:, ['MJD', 'cf_gamma', 'cf_gamma_err', 'cf_chi_sigma_dist', 'brute_gamma', 'brute_gamma_err', 'brute_chi_sigma_dist']])
            #print(BB_fit_results.tail(20).iloc[:, 10:16])
            #print()
        
        
        if (SED_type == 'single_BB') and (UVOT_guided_fitting == False):
            BB_fit_results = BB_fitting.run_SED_fitting_process(band_colour_dict=band_colour_dict)
            BB_2dp = BB_fit_results[BB_fit_results['no_bands'] == 2] # the BB fits for the MJDs which only had 2 bands, so we aren't really fitting, more solving for the BB R and T which perfectly pass through the data points
            
            # what we want: 
            # L_rf vs MJD: on this plot there will be the actual (binned) data, the polyfit and the interpolated data from the polyfit
            # BB R vs MJD: on this plot we'll have curve_fit R and brute force grid R, perhaps with a colour scale to indicate the sigma distance of the reduced chi squared for the 
            #               BB fit to indicate how much we should trust the value
            # BB T vs MJD: basically the same as BB R vs MJD
            # sigma distance vs MJD?: sigma distance for the curve_fit and brute force results. I feel like this would be good to have alongside the polyfit, because if the polyfit was bad at this MJD,
            #               then the interpolated L_rf of the band might be bad, too, which might mean that the BB fit will struggle to fit to this poorly interpolated datapoint.


            #fig = plt.figure(figsize = (16, 7.3))
            #ax1, ax2, ax3, ax4 = [plt.subplot(2, 2, i) for i in np.arange(1, 5, 1)]
            fig, axs = plt.subplots(2, 2, sharex=True, figsize = (16, 7.2))
            ax1, ax2 = axs[0]
            ax3, ax4 = axs[1]

            # getting the colour scale for plotting the BB T and R vs MJD coloured by chi sigma distance
            colour_cutoff = 5.0
            norm = Normalize(vmin = 0.0, vmax = colour_cutoff)

        

        
            # top left: the L_rf vs MJD light curve
            for b in ANT_bands: # iterate through all of the bands present in the ANT's light curve
                b_df = interp_lc[interp_lc['band'] == b].copy()
                b_colour = band_colour_dict[b]
                ax1.errorbar(b_df['d_since_peak'], b_df['L_rf'], yerr = b_df['L_rf_err'], fmt = 'o', c = b_colour, 
                            linestyle = 'None', markeredgecolor = 'k', markeredgewidth = '0.5', label = b)
                ax1.set_ylabel(r'Rest frame luminosity erg s$^{-1}$ $\AA^{-1}$', fontweight = 'bold')
                


            BB_fit_results = BB_fit_results.dropna(subset = ['red_chi_1sig'])

            if BB_curvefit == True:
                # separating the BB fit results into high and low chi sigma distance so we can plot the ones wiht low chi sigma distance in a colour map, and the high sigma distance in one colour
                BB_low_chi_dist = BB_fit_results[BB_fit_results['cf_chi_sigma_dist'] <= colour_cutoff]
                BB_high_chi_dist = BB_fit_results[BB_fit_results['cf_chi_sigma_dist'] > colour_cutoff]

                #norm = Normalize(vmin = BB_fit_results['cf_chi_sigma_dist'].min(), vmax = BB_fit_results['cf_chi_sigma_dist'].max())
                # ax2 top right: blackbody radius vs MJD
                ax2.errorbar(BB_fit_results['d_since_peak'], BB_fit_results['cf_R_cm'], yerr = BB_fit_results['cf_R_err_cm'], linestyle = 'None', c = 'k', 
                            fmt = 'o', zorder = 1, label = f'BB fit chi sig dist >{colour_cutoff}')
                ax2.errorbar(BB_2dp['d_since_peak'], BB_2dp['cf_R_cm'], yerr = BB_2dp['cf_R_err_cm'], linestyle = 'None', c = 'k', mfc = 'white',
                            fmt = 'o', label = f'cf no bands = 2', mec = 'k', mew = 0.5)
                sc = ax2.scatter(BB_low_chi_dist['d_since_peak'], BB_low_chi_dist['cf_R_cm'], cmap = 'jet', c = np.ravel(BB_low_chi_dist['cf_chi_sigma_dist']), 
                            label = 'Curve fit results', marker = 'o', zorder = 2, edgecolors = 'k', linewidths = 0.5)

                cbar_label = r'CF Goodness of BB fit ($\chi_{\nu}$ sig dist)'
                cbar = plt.colorbar(sc, ax = ax2)
                cbar.set_label(label = cbar_label)


                # ax3 bottom left: reduced chi squared sigma distance vs MJD
                ax3.scatter(BB_fit_results['d_since_peak'], BB_fit_results['cf_chi_sigma_dist'], marker = 'o', label = 'Curve fit results', edgecolors = 'k', linewidths = 0.5)

                # ax4 bottom right: blackbody temperature vs MJD
                ax4.errorbar(BB_fit_results['d_since_peak'], BB_fit_results['cf_T_K'], yerr = BB_fit_results['cf_T_err_K'], linestyle = 'None', c = 'k', 
                            fmt = 'o', zorder = 1, label = f'BB fit chi sig dist >{colour_cutoff}')
                ax4.errorbar(BB_2dp['d_since_peak'], BB_2dp['cf_T_K'], yerr = BB_2dp['cf_T_err_K'], linestyle = 'None', c = 'k', mfc = 'white',
                            fmt = 'o', label = f'cf no bands = 2', mec = 'k', mew = 0.5)
                sc = ax4.scatter(BB_low_chi_dist['d_since_peak'], BB_low_chi_dist['cf_T_K'], cmap = 'jet', c = BB_low_chi_dist['cf_chi_sigma_dist'], 
                            label = 'Curve fit results', marker = 'o', edgecolors = 'k', linewidths = 0.5, zorder = 2)
                
                #plt.colorbar(sc, ax = ax4, label = 'Chi sigma distance')
                cbar_label = r'CF Goodness of BB fit ($\chi_{\nu}$ sig dist)'
                cbar = plt.colorbar(sc, ax = ax4)
                cbar.set_label(label = cbar_label)



                
            if (BB_brute == True):
                # separating the BB fit results into high and low chi sigma distance so we can plot the ones wiht low chi sigma distance in a colour map, and the high sigma distance in one colour
                BB_low_chi_dist = BB_fit_results[BB_fit_results['brute_chi_sigma_dist'] <= colour_cutoff]
                BB_high_chi_dist = BB_fit_results[BB_fit_results['brute_chi_sigma_dist'] > colour_cutoff]

                # ax2 top right: blackbody radius vs MJD
                ax2.scatter(BB_fit_results['d_since_peak'], BB_fit_results['brute_R_cm'], linestyle = 'None', c = 'k', 
                            label = 'brute force gridding results', marker = '^')
                
                ax2.scatter(BB_2dp['d_since_peak'], BB_2dp['brute_R_cm'], linestyle = 'None', c = 'white', 
                            marker = '^', label = f'brute no bands = 2', edgecolors = 'k', linewidths = 0.5)
                
                sc = ax2.scatter(BB_low_chi_dist['d_since_peak'], BB_low_chi_dist['brute_R_cm'], cmap = 'jet', c = np.ravel(BB_low_chi_dist['brute_chi_sigma_dist']), 
                            label = 'Brute force gridding results', marker = '^', zorder = 3, edgecolors = 'k', linewidths = 0.5)

                cbar_label = r'Brute goodness of BB fit ($\chi_{\nu}$ sig dist)'
                cbar = plt.colorbar(sc, ax = ax2)
                cbar.set_label(label = cbar_label)
                
                # ax3 bottom left: reduced chi squared sigma distance vs MJD
                ax3.scatter(BB_fit_results['d_since_peak'], BB_fit_results['brute_chi_sigma_dist'], marker = '^', label = 'Brute force gridding results', edgecolors = 'k', linewidths = 0.5)

                # ax4 bottom right: blackbody temperature vs MJD
                ax4.scatter(BB_fit_results['d_since_peak'], BB_fit_results['brute_T_K'], linestyle = 'None', c = 'k', 
                            label = 'Brute force gridding results', marker = '^')
                
                ax4.scatter(BB_2dp['d_since_peak'], BB_2dp['brute_T_K'], linestyle = 'None', c = 'white', 
                            marker = '^', label = f'brute no bands = 2', edgecolors = 'k', linewidths = 0.5)
                
                sc = ax4.scatter(BB_low_chi_dist['d_since_peak'], BB_low_chi_dist['brute_T_K'], cmap = 'jet', c = BB_low_chi_dist['brute_chi_sigma_dist'], 
                            label = 'Brute fit results', marker = '^', edgecolors = 'k', linewidths = 0.5, zorder = 3)
                
                #plt.colorbar(sc, ax = ax4, label = 'Chi sigma distance')
                cbar_label = r'Brute goodness of BB fit ($\chi_{\nu}$ sig dist)'
                cbar = plt.colorbar(sc, ax = ax4)
                cbar.set_label(label = cbar_label)


            for ax in [ax1, ax2, ax3, ax4]:
                ax.grid(True)
                #ax.set_xlim(MJDs_for_fit[ANT_name])
                ax.legend(fontsize = 8)

            if ANT_name == 'ZTF22aadesap':
                ax4.set_ylim(0.0, 4e4)

            elif ANT_name == 'ZTF19aailpwl':
                ax4.set_ylim(0.0, 2e4)

            elif ANT_name == 'ZTF19aamrjar':
                ax4.set_ylim(0.0, 2.5e4)

            elif ANT_name == 'ZTF19aatubsj':
                ax4.set_ylim(0.0, 2.5e4)

            elif ANT_name == 'ZTF20abgxlut':
                ax2.set_ylim(0.0, 8e15)
                ax4.set_ylim(0.0, 2.3e4)

            elif ANT_name == 'ZTF20abodaps':
                ax2.set_ylim(0.0, 1.2e16)
                ax4.set_ylim(0.0, 4e4)

            elif ANT_name == 'ZTF20abrbeie':
                ax4.set_ylim(0.0, 2e4)

            elif ANT_name == 'ZTF21abxowzx':
                ax4.set_ylim(0.0, 2.5e4)

            elif ANT_name == 'ZTF22aadesap':
                ax2.set_ylim(0.0, 5e15)
                ax4.set_ylim(0.0, 2.5e4)

            ax2.set_ylabel('Blackbody radius / cm', fontweight = 'bold')
            ax3.set_ylabel('Reduced chi squared sigma distance \n (<=2-3 = Good fit)', fontweight = 'bold')
            ax4.set_ylabel('Blackbody temperature / K', fontweight = 'bold')
            fig.suptitle(f"Blackbody fit results across {ANT_name}'s light curve", fontweight = 'bold')
            fig.supxlabel('days since peak (rest frame time)', fontweight = 'bold')
            fig.subplots_adjust(top=0.92,
                                bottom=0.085,
                                left=0.055,
                                right=0.97,
                                hspace=0.15,
                                wspace=0.19)
            
            if save_BB_plot == True:
                savepath = f"C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS/plots/BB fits/proper_BB_fits/{ANT_name}/{ANT_name}_lc_BB_fit.png"
                plt.savefig(savepath, dpi = 300) 
            plt.show()






#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################
# TESTING WHICH SED BEST FITS THE ANT

if SED_plots == 'compare_SEDs':
    for idx in range(11):
    #for idx in [10]:
        ANT_name = transient_names[idx]
        interp_lc= interp_df_list[idx]
        ANT_bands = list_of_bands[idx]
        print()
        print(ANT_name)

        # FITTING METHOD
        brute_gridsize = 2000
        brute_delchi = 2.3 # = 2.3 to consider parameters jointly for a 1 sigma error. good if you want to quote their value but if you're going to propagate the errors I think u need to use = 1, which considers them 'one at a time'
        individual_BB_plot = 'None'
        no_indiv_SED_plots = 24 # current options are 24, 20, 12
        plot_chi_contour = False
        no_chi_contours = 3 
        # SAVING PLOTS
        save_BB_plot = False
        save_indiv_BB_plot = False

        fig, axs = plt.subplots(2, 2, figsize = (16.5, 7.5))
        ax1, ax2, ax3, ax4 = axs.ravel()

        for i, SED_type in enumerate(['single_BB', 'double_BB', 'power_law']):
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

        
            BB_fitting = fit_SED_across_lightcurve(interp_lc, SED_type = SED_type, curvefit = BB_curvefit, brute = BB_brute, ant_name = ANT_name, 
                                                brute_delchi = brute_delchi,  brute_gridsize = brute_gridsize, individual_BB_plot = individual_BB_plot, 
                                                no_indiv_SED_plots = no_indiv_SED_plots, save_indiv_BB_plot = save_indiv_BB_plot, 
                                                plot_chi_contour = plot_chi_contour, no_chi_contours = no_chi_contours)
            

            def median_absolute_deviation(median, data): # need to maybe handle the possibility that data is empty and median is None?
                if len(data) == 0:
                    return np.nan
                MAD = np.sum(abs(median - data)) / len(data)
                return MAD

            # the whole lc SED fits
            BB_fit_results = BB_fitting.run_SED_fitting_process(band_colour_dict=band_colour_dict)
            BB_fit_results = BB_fit_results[BB_fit_results[sig_dist_colname].notna()]
            all_lc_median = BB_fit_results[sig_dist_colname].median(skipna = True)
            all_lc_MAD = median_absolute_deviation(all_lc_median, BB_fit_results[sig_dist_colname].to_numpy())

            # SED fits where N > M
            fit_results_N_greater_M = BB_fit_results[BB_fit_results['no_bands'] > M].copy()
            N_greater_M_median = fit_results_N_greater_M[sig_dist_colname].median(skipna = True)
            N_greater_M_MAD = median_absolute_deviation(N_greater_M_median, fit_results_N_greater_M[sig_dist_colname].to_numpy())

            # SED fits with UVOT data
            MJDs_with_UVOT = interp_lc[interp_lc['band'].isin(['UVOT_B', 'UVOT_U', 'UVOT_UVM2', 'UVOT_UVW1', 'UVOT_UVW2', 'UVOT_V'])]['MJD']
            SEDs_with_UVOT = BB_fit_results[BB_fit_results['MJD'].isin(MJDs_with_UVOT)].copy()
            UVOT_median = SEDs_with_UVOT[sig_dist_colname].median()
            UVOT_MAD = median_absolute_deviation(UVOT_median, SEDs_with_UVOT[sig_dist_colname].to_numpy())

            # SED fits on rise/near peak
            SEDs_rise_peak = BB_fit_results[BB_fit_results['d_since_peak'] <= 100].copy()
            rise_peak_median = SEDs_rise_peak[sig_dist_colname].median(skipna = True)
            rise_peak_MAD = median_absolute_deviation(rise_peak_median, SEDs_rise_peak[sig_dist_colname].to_numpy())


            # PLOTTING IT UP IN HISTOGRAMS
            label1 = fr'$\mathbf{{{SED_label}}}$ '+'\n'+fr'Median $D_{{\sigma}}$ = {all_lc_median:.3g}'+'\n'+fr'MAD $D_{{\sigma}}$ = {all_lc_MAD:.3g}'
            ax1.hist(BB_fit_results[sig_dist_colname], bins = 50, color = SED_colour, label = label1, alpha = 0.5)
            ax1.hist(BB_fit_results[sig_dist_colname], bins = 50, edgecolor = SED_ecolour, color = SED_colour, alpha = 0.75, histtype='step', linewidth = 1.5, linestyle = SED_linestyle)#, hatch = SED_hatch)

            label2 = fr'$\mathbf{{{SED_label}}}$ '+'\n'+fr'Median $D_{{\sigma}}$ = {N_greater_M_median:.3g} '+'\n'+fr'MAD $D_{{\sigma}}$ = {N_greater_M_MAD:.3g}'
            ax2.hist(fit_results_N_greater_M[sig_dist_colname], bins = 50, color = SED_colour, label = label2, alpha = 0.5)
            ax2.hist(fit_results_N_greater_M[sig_dist_colname], bins = 50, edgecolor = SED_ecolour, color = SED_colour, alpha = 0.75, histtype='step', linewidth = 1.5, linestyle = SED_linestyle)#, hatch = SED_hatch)

            label3 = fr'$\mathbf{{{SED_label}}}$ '+'\n'+fr'Median $D_{{\sigma}}$ = {rise_peak_median:.3g} '+'\n'+fr'MAD $D_{{\sigma}}$ = {rise_peak_MAD:.3g}'
            ax3.hist(SEDs_rise_peak[sig_dist_colname], bins = 50, color = SED_colour, label = label3, alpha = 0.5)
            ax3.hist(SEDs_rise_peak[sig_dist_colname], bins = 50, edgecolor = SED_ecolour, color = SED_colour,  alpha = 0.75, histtype='step', linewidth = 1.5, linestyle = SED_linestyle)#, hatch = SED_hatch)

            label4 = fr'$\mathbf{{{SED_label}}}$ '+'\n'+fr'Median $D_{{\sigma}}$ = {UVOT_median:.3g} '+'\n'+fr'MAD $D_{{\sigma}}$ = {UVOT_MAD:.3g}'
            ax4.hist(SEDs_with_UVOT[sig_dist_colname], bins = 50, color = SED_colour, label = label4, alpha = 0.5)
            ax4.hist(SEDs_with_UVOT[sig_dist_colname], bins = 50, edgecolor = SED_ecolour, color = SED_colour, alpha = 0.75, histtype='step', linewidth = 1.5, linestyle = SED_linestyle)#, hatch = SED_hatch)

        ax1.set_title('SED fit results entire lightcurve', fontweight = 'bold')
        ax2.set_title('SED fit results where N > M', fontweight = 'bold')
        ax3.set_title('SED fit results for days since peak < 100', fontweight = 'bold')
        ax4.set_title('SED fit results for MJDs with UVOT data', fontweight = 'bold')

        for ax in [ax1, ax2, ax3, ax4]:
            ax.legend()
            ax.grid(True)

        titlefontsize = 20
        fig.supxlabel(r'(sigma distance) = $\mathbf{ D_{\sigma =} \frac{\chi_{\nu} - 1} {\sigma_{\chi_{\nu}}}  }$', fontsize = (titlefontsize - 5), fontweight = 'bold')
        fig.supylabel('Frequency density', fontweight = 'bold', fontsize = (titlefontsize - 5))
        fig.suptitle(f"{ANT_name}'s SED fit results comparing a single blackbody, double blackbody and \n power law SED (MAD = median absolute deviation)", fontweight = 'bold', fontsize = titlefontsize)
        fig.subplots_adjust(top=0.82,
                            bottom=0.11,
                            left=0.08,
                            right=0.955,
                            hspace=0.2,
                            wspace=0.2)
        savepath = f"C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS/plots/BB fits/proper_BB_fits/{ANT_name}/{ANT_name}_compare_SED_fits.png"
        plt.savefig(savepath, dpi = 300)
        #plt.show()











        

        

            




