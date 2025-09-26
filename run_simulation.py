"""
Main pipeline script for running LSST light curve simulations.
Usage:
    python run_simulation.py
Or import and call main() from a notebook.
"""

import os
import sys
import logging
import numpy as np
import yaml
import argparse
from utils.simulation_utils import LSSTSimulator
from normal_code.plotting_preferences import ANT_sim_redshift_upper_lim_dict
from skysurvey_PW.tools.utils import random_radec

def load_yaml_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# Load configs
rates = load_yaml_config("config/rates.yaml")
model = _MODEL = dict( redshift = {"kwargs": {"zmax": 0.05}, 
                            "as": "z"},
                t0 = {"func": np.random.uniform,
                        "kwargs": {"low": 56_000, "high": 56_200}
                    },
                        
                magabs = {"func": np.random.normal,
                            "kwargs": {"loc": -24, "scale": 1}
                        },
                            
                magobs = {"func": "magabs_to_magobs",
                            "kwargs": {"z":"@z", "magabs": "@magabs"}
                        },
                            
                amplitude = {"func": "magobs_to_amplitude",
                            "kwargs": {"magobs": "@magobs"}
                        },
                # This you need to match with the survey
                radec = {"func": random_radec,
                        "as": ["ra","dec"]
                        }
                )
luminosity_dist_dict = load_yaml_config("metadata/luminosity_dist.yaml")
# You can add more config files as needed

# Example: set up survey and parameters here
survey = None  # TODO: Load your LSST survey object
max_sim_redshift_dict = ANT_sim_redshift_upper_lim_dict # TODO: Load or calculate

def main(object_name="ZTF18aczpgwm", sed_type="SBB", SED_filename=None, size=100, no_plots_to_save=10, 
         tstart="2026-04-10", tstop="2031-04-01", 
         max_sig_dist=3.0, max_chi_N_equal_M=0.1, plot_skysurvey_inputs=True, 
         plot_SED_results=False, config_file=None, output_dir=None):
    # If config_file is provided, load all parameters from it
    if config_file:
        with open(config_file, 'r') as f:
            cfg = yaml.safe_load(f)
        object_name = cfg.get('object_name', object_name)
        sed_type = cfg.get('sed_type', sed_type)
        SED_filename = cfg.get('SED_filename', SED_filename)
        size = cfg.get('size', size)
        no_plots_to_save = cfg.get('no_plots_to_save', no_plots_to_save)
        tstart = cfg.get('tstart', tstart)
        tstop = cfg.get('tstop', tstop)
        max_sig_dist = cfg.get('max_sig_dist', max_sig_dist)
        max_chi_N_equal_M = cfg.get('max_chi_N_equal_M', max_chi_N_equal_M)
        plot_skysurvey_inputs = cfg.get('plot_skysurvey_inputs', plot_skysurvey_inputs)
        plot_SED_results = cfg.get('plot_SED_results', plot_SED_results)
        output_dir = cfg.get('output_dir', output_dir)

    # SED_filename can be provided directly, or constructed from object/SED type
    if SED_filename is None:
        SED_filename = f"{object_name}_{sed_type}_SED_fit_across_lc.csv"
    SED_filepath = os.path.join("./", SED_filename)
    simulator = LSSTSimulator(survey, model, luminosity_dist_dict, max_sim_redshift_dict)
    simulator.sim_ANY_LSST_lc(object_name=object_name,
        SED_filename=SED_filename,
        SED_filepath=SED_filepath,
        max_sig_dist=max_sig_dist,
        max_chi_N_equal_M=max_chi_N_equal_M,
        size=size,
        tstart=tstart,
        tstop=tstop,
        no_plots_to_save=no_plots_to_save,
        plot_skysurvey_inputs=plot_skysurvey_inputs,
    plot_SED_results=plot_SED_results,
    output_dir=output_dir,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LSST light curve simulation.")
    parser.add_argument("--object", type=str, default="ZTF18aczpgwm", help="Object name (e.g. ZTF18aczpgwm)")
    parser.add_argument("--sed", type=str, default="SBB", help="SED type (e.g. SBB, PL, DBB)")
    parser.add_argument("--sedfile", type=str, default=None, help="SED filename (overrides object/sed type)")
    parser.add_argument("--size", type=int, default=100, help="Number of simulated objects")
    parser.add_argument("--noplots", type=int, default=10, help="Number of plots to save")
    parser.add_argument("--tstart", type=str, default="2026-04-10", help="Simulation start date")
    parser.add_argument("--tstop", type=str, default="2031-04-01", help="Simulation stop date")
    parser.add_argument("--maxsig", type=float, default=3.0, help="Max sigma distance")
    parser.add_argument("--maxchi", type=float, default=0.1, help="Max chi N=M")
    parser.add_argument("--plotinputs", action="store_true", help="Plot skysurvey inputs")
    parser.add_argument("--plotresults", action="store_true", help="Plot SED results")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML file")
    parser.add_argument("--outdir", type=str, default=None, help="Base directory for output plots")
    parser.add_argument("--log", dest="log_level", type=str, default="INFO", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"], help="Log level for console output")
    args = parser.parse_args()
    # Configure logging to print to terminal
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,  # override any pre-existing handlers so logs always show
    )
    main(
        object_name=args.object,
        sed_type=args.sed,
        SED_filename=args.sedfile,
        size=args.size,
        no_plots_to_save=args.noplots,
        tstart=args.tstart,
        tstop=args.tstop,
        max_sig_dist=args.maxsig,
        max_chi_N_equal_M=args.maxchi,
        plot_skysurvey_inputs=args.plotinputs,
        plot_SED_results=args.plotresults,
        config_file=args.config,
        output_dir=args.outdir,
    )
