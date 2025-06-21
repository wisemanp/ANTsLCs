# Lauren's YoRiS code :D ================================================================================================================================
Takes a light curve file and performs polynomial interpolation, and then fits the interpolated broad band photometry at each epoch with SED models.
These SED models include:
    - Single-blackbody (SBB)
    - Double-blackbody (DBB) (aimed to fit one BB component to the UV, and one to the optical, so maybe not as useful if you don't have UV data)
    - Power-law (PL)




# Structure: ============================================================================================================================================
    - functions.py contains all of the functions used to analyse the data, from converting magnitudes to luminosity density to polynomial interpolation
    to SED fitting

    - plotting_preferences.py contains dictionaries of the individual ANT properties (such as redshift), along with the plotting preferences such as:
        - allocating a given observed photometric band a particular colour and marker shape for plotting (e.g. 'ATLAS_c' is plotted as a red circle)
        - giving each ANT's light curve plots an xlim in MJD





# Code examples: ==========================================================================================================================================
To see exactly how I run the code, you can look to:
    - February\test_improved_polyfit_class.py
        - Takes the raw light curve data files
        - Converts apparent mag (absolute mag for PS1-10adi) to spectral luminosity density in terms of the emitted frame wavelengths
        - Bins the spectral luminosity density into 1 day bins
        - Fits a polynomial to the each band of the light curve
        - Interpolates the light curve using these polynomials (given some settings that you input)
        - Saves the interpolated DataFrames along with a README file containing the interpolation and polynomial fit inputs the user provides

    - February\new_test_BB_function.py
        - Loads in the interpolated light curve files (which you can produce using February\test_improved_polyfit_class.py)
        - Fits SED models to each epoch of the interpolated light curves (see the top of this file for further description)
        - Can be used to compare which SED models fit the data best (see the top of this file for further description)



If you wanted to write your own files to run the the code, I would recommend running the functions in the following order:
1. load_ANT_data() 
    - to load in the original light curve data files

2. ANT_data_L_rf(lc_df_list, transient_names, ANT_redshift_dict, ANT_luminosity_dist_cm_dict, band_ZP_dict, band_obs_centwl_dict)
    - to convert magnitudes into luminosity density in terms of the emitted-frame wavelengths, and convert the observed photometric band'sample
      central wavelength into the emitted-frame wavelength

3. bin_lc()
    - to bin the light curve (in terms of luminosity density)

4. lightcurve = polyfit_lightcurve()
    - Initialise polyfit_lightcurve() class with your inputs. 

5. lightcurve.run_fitting_pipeline()
    - runs the fitting pipeline from the polyfit_lightcurve() class

6. BB_fitting = fit_SED_across_lightcurve()
    - initialise the fit_SED_across_lightcurve() class. 

7. 
7.1. If you want UVOT guided fitting, run: BB_fit_results = BB_fitting.run_UVOT_guided_SED_fitting_process()
    - this runs the UVOT guided SED fitting process on your chosen ANT
    NOTE: UVOT guided fitting will only take place if ANT_name is one of: '[ZTF19aailpwl', 'ZTF20acvfraq', 'ZTF22aadesap', 'ASASSN-17jz', 'ASASSN-18jd']
    since these were the only ANTs in our sample which had more than 10 epochs of interpolated data in at least 2 bands which were emitted below
    3000 Angstroms (which I referred to as the 'UV-rich ANTs' in my dissertation). if the ANT_name is not in that list, the fitting process will
    default to fitting each epoch's SED independently of one another

7.2. If you don't want UVOT guided fitting, run: 
    BB_fit_results = BB_fitting.run_SED_fitting_process()







# NOTE ======================================================================================================================================================
You may see a lot of red warnings printed in the terminal (especially when fitting the double-blackbody model) like:
    'WARNING - No chi values within the delta_chi = 2.3 region for MJD = 59841.66138890013. Min delchi = 2.66925817087425 '

Don't worry about these too much, this is a warning for the sampling part of the code which I never ended up using. Basically, what it's trying to do is:
    When using the brute force fitting method, sample parameter values from the region of parameter space which satisfies chi <= min_chi + 2.3, 
    where each parameter pair is sampled with a probability going like 1/chi (when I refer to chi I mean the chi squared of the SED fit to that epoch).
    However, there are instances where there are no parameter pairs within this chi <= min_chi + 2.3 region (this would mean the brute parameter grid 
    is too coarsely spaced). In my code, I used a large paremetr grid for the SBB and PL since they both have only 2 model parameters, but for the DBB, 
    I had to use curve_fit for the fitting since it has 4 model parameters, making a large brute-force parameter grid too computationally expensive. As a 
    result, I uses a much smaller DBB brute force grid to obtain an approximate chi <= min_chi + 2.3 region, however, this can often mean that
    there are no parameer combinations satisfying chi <= min_chi + 2.3. 
