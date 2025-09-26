# ANTsLCs — Simulate ANT light curves with LSST

Authors: Lauren Eastman, Phil Wiseman

This project simulates light curves for Ambiguous Nuclear Transients (ANTs) observed by LSST using measured spectral energy distribution (SED) evolution. It uses a single entry point (`run_simulation.py`) and modular utils for saving and plotting.

## Supported SED models
- SBB — single blackbody
- PL — power law
- DBB — double blackbody

## Requirements
- Python 3.9+
- Packages: `numpy`, `pandas`, `scipy`, `astropy`, `matplotlib`, `sncosmo`, `pyyaml`, `h5py`
- Local tools: `skysurvey_PW` (dataset, transient sources, filters)  
  Fork: https://github.com/wisemanp/skysurvey_PW
  Note: this will still be installed as `skysurvey`

## Installation
- ```bash
    python3 -m venv venv
    source venv/bin/activate
    git clone git@github.com:wisemanp/ANTsLCs.git
    cd ANTsLCs
    pip install -r requirements.txt
    cd ..
    git clone git@github.com:wisemanp/skysurvey_PW.git
    cd skysurvey_PW
    pip install .
  ```

## Project layout (key files)
- `run_simulation.py` — CLI entry point
- `utils/`
  - `simulation_utils.py` — `LSSTSimulator`; main simulation flow and guards
  - `io_utils.py` — HDF5 writer (single file, multiple keys)
  - `plot_utils.py` — plotting helpers (inputs, results, outputs)
- `filters/`
  - `zeropoints.py` — band zero-points and plotting settings
  - `central_wavelengths.py` — filter central wavelengths
- `metadata/`
  - `redshift.yaml` — ANT_proper_redshift_dict (moved from code)
  - `luminosity_distance_dict.yaml` — ANT_luminosity_dist_cm_dict
  - `peak_mjd.yaml` — peak_MJD_dict
- `SED_fits_for_simulation/` — input SED CSVs

## Inputs
- SED CSV for the selected object and SED model (e.g., `ZTF18aczpgwm_SBB_SED_fit_across_lc_new.csv`).
- Metadata YAMLs for redshift and luminosity distance (see `metadata/`).

## Run from the command line (macOS)
### Basic usage
```bash
python run_simulation.py \
  --object ZTF18aczpgwm \
  --sed SBB \
  --sedfile SED_fits_for_simulation/ZTF18aczpgwm_SBB_SED_fit_across_lc_new.csv \
  --size 100 \
  --outdir outputs \
  --plotinputs \
  --plotresults \
  --log INFO
```

### Other SEDs
```bash
python run_simulation.py \
  --object ZTF22aadesap \
  --sed PL \
  --sedfile SED_fits_for_simulation/ZTF22aadesap_UVOT_guided_PL_SED_fit_across_lc_new.csv \
  --size 200 \
  --outdir outputs

python run_simulation.py \
  --object ASASSN-18jd \
  --sed DBB \
  --sedfile SED_fits_for_simulation/ASASSN-18jd_UVOT_guided_DBB_Sed_fit_across_lc_new.csv \
  --size 150 \
  --outdir outputs
```

## Main CLI options
- `--object`: Target name, used to read metadata (e.g., `ZTF18aczpgwm`).
- `--sed`: SED model (`SBB`, `PL`, `DBB`).
- `--sedfile`: Path to SED CSV file.
- `--size`: Number of simulated targets to draw.
- `--tstart` / `--tstop`: MJD window (optional; defaults come from survey).
- `--noplots`: Max number of per-object output plots to save.
- `--outdir`: Base output directory for HDF5 and PNGs.
- `--time-spline-degree` / `--wavelength-spline-degree`: Interpolator orders (phase, wavelength).
- `--plotinputs` / `--plotresults`: Toggle input/SED sanity plots.
- `--log`: Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`).

## Use in a notebook
Minimal pattern:
```python
from utils.simulation_utils import LSSTSimulator
sim = LSSTSimulator()
sim.sim_ANY_LSST_lc(
    object_name="ZTF18aczpgwm",
    SED_filename="ZTF18aczpgwm_SBB_SED_fit_across_lc_new.csv",
    SED_filepath="SED_fits_for_simulation",
    size=100,
    tstart=None,
    tstop=None,
    no_plots_to_save=10,
    time_spline_degree=1,
    wavelength_spline_degree=3,
    plot_skysurvey_inputs=True,
    plot_SED_results=True,
    output_dir="outputs"
)
```

## Outputs
- HDF5 per run at:
  ```
  outputs/<Object>/LSST_<SED>/<Object>_SED_<SED>_simulated_lightcurves.h5
  ```
  Keys per simulated object `i`:
  - `obj_i/flux` — DataFrame (rows=MJD, cols=band)
  - `obj_i/fluxerr`
  - `obj_i/flux_density`
  - `obj_i/flux_density_err`
  - `obj_i/meta` — one-row DataFrame of draw parameters (z, ra, dec, etc.)

- Plots saved under the same folder:
  - Input parameter plots (e.g., SBB: T, amplitude, R vs phase)
  - SED sanity plots across wavelength and phase
  - Panel plots of multiple simulated light curves
  - Per-object two-panel plots (flux and flux density)

## Configuration and metadata
- Filters and zero-points: `filters/zeropoints.py`
- Central wavelengths: `filters/central_wavelengths.py`
- Redshifts: `metadata/redshift.yaml`
- Luminosity distances: `metadata/luminosity_distance_dict.yaml`
- Peak MJDs: `metadata/peak_mjd.yaml`

## Notes and guards
- Interpolation stability:
  - Phase/wavelength grids are sanitized (sorted, deduplicated, finite-only).
  - Spline degrees are clamped (1–5) and must be less than the number of unique samples in each axis.
  - This avoids native FITPACK “malloc: Double free” errors.
- Logging:
  - CLI configures logging (use `--log INFO`).
  - In notebooks, a fallback stream handler is added if none is configured.

## Troubleshooting
- No logs in terminal: pass `--log INFO` (or `DEBUG`); CLI resets handlers.
- HDF5 write errors about string/Arrow dtypes: I/O layer coerces dtypes; ensure `pandas` and `pytables` are up to date.
- Pandas FutureWarning for `get_group`: handled internally (tuple key).
- Spline errors: lower `--time-spline-degree` or clean the SED CSV for NaNs/duplicate phases.

## Extending
- New SED types: add a branch in `LSSTSimulator` to build the source and re-use the I/O and plotting utilities.
- Custom interpolators: provide desired spline degrees via CLI or subclass your `Source`.





