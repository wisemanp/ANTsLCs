import os
import pandas as pd




def load_ANT_data():
    """
    loads in the ANT data files

    INPUTS:
    -----------
    None


    OUTPUTS:
    ----------
    dataframes: a list of the ANT data in dataframes
    names: a list of the ANT names
    ANT_bands: a list of lists, each inner list is a list of all of the bands which are present in the light curve

    """
    phils_ANTs = "C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS Data/modified Phil's lightcurves"
    updated_Phils_ANTs = "C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS Data/Phil's lcs updated modified" # this contains forced photometry files for abodaps and mrjar, so we'll use these instead of the data files for these ants in Phils_ANTs
    other_ANT_data = "C:/Users/laure/OneDrive/Desktop/YoRiS desktop/YoRiS Data/other ANT data/ALL_FULL_LCs" 
    directories = [phils_ANTs, updated_Phils_ANTs, other_ANT_data]

    dataframes = []
    names = []
    ANT_bands = []
    for dir in directories:
        for file in os.listdir(dir): # file is a string of the file name such as 'file_name.dat'
            # getting the ANT name from the file name
            if dir == other_ANT_data:
                ANT_name = file[:-12]
            else:
                ANT_name = file[:-7] # the name of the transient

            # skip over the old verisons of these files from Phil, since we have better ones in updated_phils_ANTs
            if (ANT_name == 'ZTF19aamrjar' or ANT_name == 'ZTF20abodaps') and dir == phils_ANTs: 
                continue

            # load in the files
            file_path = os.path.join(dir, file)
            file_df = pd.read_csv(file_path, delimiter = ',') # the lightcurve data in a dataframe
            bands = file_df['band'].unique() # the bands in which this transient has data for

            dataframes.append(file_df)
            names.append(ANT_name)            
            ANT_bands.append(bands)

    return dataframes, names, ANT_bands

