# Explanation of my use of Skysurvey to simulate ANT light curves
    I used a modified version of the Python package Skysurvey to simulate ANT light curves based on their single-blackbody (SBB), power-law (PL)
    and double-blackbody (DBB) spectral energy distribution (SED) evolution. Skysurvey has this built in for a SBB SED, so I copied the SBB code
    and modified it to also enable simulation for PL and DBB SED evolutions. 




# Relevant code

    1. My version of Skysurvey - in particular, I made changes to allow for simulation of transients based on their PL or DBB SED evolution. I
        also slightly changed the way that SNcosmo and Skysurvey interact to allow for user input of the type of spline interpolation that
        you would like to use when interpolating the SED evolution over phase (by default SNcosmo uses 3rd order but this produced large artefacts
        in the simulated light curves so I would recommend using this as 1), and also how it interpolates each SED over wavelength (which SNcosmo 
        has set to 3rd order by default, and this is the order that I would recommend using). 

    2. My version of SNcosmo's TimeSeriesTransient (in SNcosmo's models.py)
        I changed this for the reason described in the point above (allowing the order of spline interpolation ot be a user input). The only 
        difference between the default SNcosmo TimeSeriesTransient and my one is in the __innit__(). 

    3. FINAL_DISS_WORKING_sim_ALL_LSST_subplot_zmax_lc.py
        This is the code which I used to input the csv files containing the spectral evolution (assuming a given SED model) fro ecah ANT, and used
        this spectral evolution (SE) to simulate more light curves with my modified version of Skysurvey. I would recommend looking to this file
        to see how I convert my csv files into the necessary Skysurvey inputs

    4. try_compare_aadeap_to_sim_lc.py
        This code just generates a plot for my diss, where I plotted a simulated light (generated using ZTF22aadesap's PL SE) curve next to 
        ZTF22aadesap's light curve 





# If you wanted to write your own code, here is the general order of functions that I would recommend:
    1. Choose a csv file which contains the measured SE of a particular ANT, assuming a particular SED model

    2. Convert your SED model parameters (from the csv file) into the requires Skysurvey inputs

    3. Input these into Skysurvey using:
        3.1. For a SBB SED model, use blackbody.get_blackbody_transient_source()
        3.2  For a PL SED model, use power_law.get_power_law_transient_source()
        3.3  For a DBB SED model, use double_blackbody.get_double_blackbody_transient_source()

    4. To simulate transients from the source produced in the step above, use: skysurvey.TSTransient.from_draw() 

    5. Combine these simulate transients with your survey (in our case, LSST) to generate a dataset (simulated observations of these transients using your survey): 
        dset = skysurvey.DataSet.from_targets_and_survey()

    6. dset.get_ndetection()

    7. dset contains all observations of the simulated transients, you can then select a random transeient from dset to plot its light curve




