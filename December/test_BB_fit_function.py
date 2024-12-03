import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import sys
import scipy.optimize as opt
sys.path.append("C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS") # this allows us to import plotting preferences and functions
from plotting_preferences import band_colour_dict, band_ZP_dict, band_obs_centwl_dict, ANT_redshift_dict, ANT_luminosity_dist_cm_dict, MJDs_for_fit
from functions import load_ANT_data, ANT_data_L_rf, bin_lc, polyfit_lc, blackbody


def fit_BB(interp_df, brute, curvefit):
    """


    INPUTS
    ---------------
    interp_df: the ANT's dataframe containing a light curve which has been interpolated using a polynomial fit to each band. 
                Each ANT had a reference band. At the MJD values present in the reference band's real data, the polyfit for each band
                was evaluated (provided that we aren't extrapolating). This means that if there was a band which had data at the min and max 
                MJD of the flare, there will be interpolated data for this band across the whole light curve, whereas if there is a band
                which only has data on the plateau of the light curve, this band will only have interpolated data within this region, at
                the MJD values of the reference band's data. This means that we don't need to bin the light curve in order to take the data
                for the blackbody fit, we can take each MJD present within this dataframe and take whatever band is present at this MJD as
                the data for the BB fit. So, we can fit a BB curve for each MJD within this dataframe, as long as it has >2 bands present. 

    brute: if True, the BB fit will be tried using the brute force method (manually creating a grid of trial parameter values and minimising the chi squared). If 
            False, no brute force calculation will be tried

    curvefit: if True, the BB fit will be tried using scipy's curve_fit. If False, no curve_fit calculation will be tried


    RETURNS
    ---------------

    """

    return