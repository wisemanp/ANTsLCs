"""
Generalized script to compare real and simulated light curves for any input object.
"""
import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from utils.simulation_utils import LSSTSimulator

# --- Load configs and metadata ---
def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

rates = load_yaml("config/rates.yaml")
model = load_yaml("config/model.yaml")
luminosity_dist_dict = load_yaml("metadata/luminosity_distance_dict.yaml")
redshift_dict = load_yaml("metadata/redshift.yaml")
peak_mjd_dict = load_yaml("metadata/peak_mjd.yaml")

# --- Survey setup (example, replace with your actual survey loader) ---
import skysurvey
opsim_path = 'baseline_v4.3.1_10yrs.db'
LSST_survey = skysurvey.LSST.from_opsim(opsim_path)

# --- Parameters ---
tstart = "2026-04-10"
tstop = "2030-04-01"
size = 500
no_plots_to_save = 100
time_spline_degree = 1
wavelength_spline_degree = 3
max_sig_dist = 3.0
max_chi_N_equal_M = 0.1

# --- Input object selection ---
input_object = "ZTF22aadesap"  # Change this to any object name
sed_folder = "normal_code/data/SED_fits/" + input_object
sed_files = [f for f in os.listdir(sed_folder) if f.endswith(".csv")]
real_lc_path = f"normal_code/{input_object}_flux_density.csv"
real_lc_df = pd.read_csv(real_lc_path, delimiter=',')

# --- Redshift range ---
z = redshift_dict[input_object]
zmin = z - 0.001
zmax = z + 0.001

# --- Run simulation and comparison for each SED file ---
simulator = LSSTSimulator(
    survey=LSST_survey,
    model=model,
    luminosity_dist_dict=luminosity_dist_dict,
    max_sim_redshift_dict=redshift_dict
)

for sed_file in sed_files:
    SED_filename = sed_file
    SED_filepath = os.path.join(sed_folder, sed_file)
    simulator.compare_real_vs_sim_lc(
        real_lc_df=real_lc_df,
        SED_filename=SED_filename,
        SED_filepath=SED_filepath,
        max_sig_dist=max_sig_dist,
        max_chi_N_equal_M=max_chi_N_equal_M,
        size=size,
        tstart=tstart,
        tstop=tstop,
        no_plots_to_save=no_plots_to_save,
        zmin=zmin,
        zmax=zmax,
        time_spline_degree=time_spline_degree,
        wavelength_spline_degree=wavelength_spline_degree,
        plot_skysurvey_inputs=True,
        plot_SED_results=True
    )
