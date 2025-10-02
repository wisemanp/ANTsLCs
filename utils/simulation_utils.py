import numpy as np
from astropy import constants as const
import astropy.units as u
from skysurvey import TSTransient, DataSet

# --- Blackbody and related functions ---
def blackbody_L_lam_rf(lam_cm, R_cm, T_K):
    c_cgs = const.c.cgs.value
    h_cgs = const.h.cgs.value
    k_cgs = const.k_B.cgs.value
    C = 8 * (np.pi**2) * h_cgs * (c_cgs**2) * 1e-8
    denom = np.exp((h_cgs * c_cgs) / (lam_cm * k_cgs * T_K)) - 1
    L = C * ((R_cm**2) / (lam_cm**5)) * (1 / (denom))
    return L

def blackbody_F_lam(lam_cm, T_K, R_cm, D_cm):
    c_cgs = const.c.cgs.value
    h_cgs = const.h.cgs.value
    k_cgs = const.k_B.cgs.value
    C = 2 * np.pi * h_cgs * (c_cgs**2) * 1e-8
    denom = np.exp((h_cgs * c_cgs) / (lam_cm * k_cgs * T_K)) - 1
    F_lam = C * ((R_cm / D_cm)**2) * (1/ (lam_cm**5)) * (1 / (denom))
    return F_lam

def get_wein_lbdamax(temperature):
    if not hasattr(temperature, 'unit'):
        temperature = u.Quantity(temperature, u.Kelvin)
    lbda = const.h*const.c/(4.96511423174*const.k_B * temperature)
    return lbda.to(u.Angstrom)

# --- Helper function for formatting ---
def standard_form_tex(x, pos):
    if x == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(x))))
    coeff = x / (10 ** exponent)
    return rf"${{coeff:.1f}} \times 10^{{{exponent}}}$"

# --- Main simulation class ---
class LSSTSimulator:
    def __init__(self, survey, model, luminosity_dist_dict, max_sim_redshift_dict):
        from skysurvey import LSST
        opsim_path = 'baseline_v4.3.1_10yrs.db'
        print("Loading LSST survey from opsim database...")
        LSST_survey = LSST.from_opsim(opsim_path,backend="pandas")
        print("Finished loading LSST survey.")
        self.survey = LSST_survey
        self.model = model
        self.luminosity_dist_dict = luminosity_dist_dict
        self.max_sim_redshift_dict = max_sim_redshift_dict

    def sim_ANY_LSST_lc(self, object_name, SED_filename, SED_filepath, max_sig_dist, max_chi_N_equal_M, size, tstart, tstop, no_plots_to_save, time_spline_degree=1, wavelength_spline_degree=3, plot_skysurvey_inputs=True, plot_SED_results=False, output_dir=None):
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        from datetime import datetime
        from matplotlib.ticker import FuncFormatter
        import matplotlib.ticker as ticker
        from astropy import constants as const
        from skysurvey.tools import blackbody, power_law, double_blackbody
        import os
        import logging
        logger = logging.getLogger(__name__)
        if not logging.getLogger().handlers:
            # Fallback console handler if the app didn't configure logging
            _h = logging.StreamHandler()
            _h.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
            logging.getLogger().addHandler(_h)
            logging.getLogger().setLevel(logging.INFO)

        # --- Guards to avoid FITPACK malloc errors in spline construction ---
        def _sanitize_time_series(df: 'pd.DataFrame', time_col: str, required_cols: list):
            """Return a cleaned DataFrame sorted by time with no NaN/inf and no duplicate times."""
            # Keep only required columns
            cols = [time_col] + required_cols
            _df = df[cols].copy()
            # Replace inf with NaN then drop
            _df = _df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols, how='any')
            # Sort and drop duplicate times (keep last)
            _df = _df.sort_values(time_col).drop_duplicates(subset=time_col, keep='last')
            return _df

        def _safe_spline_degrees(n_time: int, n_wave: int, kx_req: int, ky_req: int):
            """Clamp spline degrees to valid FITPACK ranges given sample sizes."""
            # FITPACK bounds: 1 <= k <= 5 and k < n_samples
            safe_kx = max(1, min(5, n_time - 1, int(kx_req)))
            safe_ky = max(1, min(5, n_wave - 1, int(ky_req)))
            return int(safe_kx), int(safe_ky)

        SED_label = None
        ANT_name = object_name
        ANT_z = self.max_sim_redshift_dict[ANT_name]
        ANT_max_simulation_z = ANT_z
        ANT_D_cm = self.luminosity_dist_dict[ANT_name]
        logger.info(f"Starting simulation for {ANT_name} at z={ANT_z} with D={ANT_D_cm:.2e} cm")
        ANT_SED_df = pd.read_csv(SED_filepath, delimiter=',', index_col='MJD')
        lbda_A = np.linspace(1000, 12000, 1000)

        if ANT_name == 'PS1-10adi':
            max_sig_dist = 19.736 + 13.89
        elif ANT_name == 'ASASSN-17jz':
            max_sig_dist = 29.545 + 23.17
        elif ANT_name == 'ASASSN-18jd':
            max_sig_dist = 24.24 + 20.79

        # Base output directory
        if output_dir is None:
            output_dir = os.path.abspath(os.path.join(os.getcwd(), 'final_simulated_lcs'))
        else:
            output_dir = os.path.abspath(output_dir)

        # Choose SED model (single ladder of if/elif)
        if 'SBB' in SED_filename:
            SED_label = 'SBB'
            sed_dir = os.path.join(output_dir, ANT_name, f"LSST_{SED_label}")
            ANT_good_SEDs = ANT_SED_df[ANT_SED_df['brute_chi_sigma_dist'].abs() <= max_sig_dist].copy()
            ANT_SED_N_equals_M = ANT_SED_df[ANT_SED_df['brute_chi_sigma_dist'].isna()].copy()
            ANT_SED_N_equals_M = ANT_SED_N_equals_M[ANT_SED_N_equals_M['brute_chi'] <= max_chi_N_equal_M].copy()
            ANT_good_SEDs = pd.concat([ANT_good_SEDs, ANT_SED_N_equals_M], ignore_index=True)
            # Clean and guard time series
            raw_count = len(ANT_good_SEDs)
            clean_df = _sanitize_time_series(ANT_good_SEDs, 'd_since_peak', ['brute_T_K', 'brute_R_cm'])
            if len(clean_df) < raw_count:
                logger.info(f"Sanitized SBB inputs: {raw_count} -> {len(clean_df)} rows after NaN/inf removal and duplicate-time drop")
            phase = clean_df['d_since_peak'].to_numpy()
            BB_T = clean_df['brute_T_K'].to_numpy()
            BB_R = clean_df['brute_R_cm'].to_numpy()
            n_time = len(phase)
            if n_time < 2:
                raise ValueError(f"Not enough SBB time samples after cleaning (n={n_time}); need at least 2 to build splines.")

            # Clamp spline degrees to avoid FITPACK errors
            safe_kx, safe_ky = _safe_spline_degrees(n_time, len(lbda_A), time_spline_degree, wavelength_spline_degree)
            if (safe_kx != time_spline_degree) or (safe_ky != wavelength_spline_degree):
                logger.warning(f"Adjusted spline degrees to safe values: time k={safe_kx}, wave k={safe_ky} (requested {time_spline_degree}, {wavelength_spline_degree})")

            BB_Wein_lambda_max_Angstrom = [get_wein_lbdamax(T) for T in BB_T]
            BB_Wein_lambda_max_Angstrom_values = [i.value for i in BB_Wein_lambda_max_Angstrom]
            BB_Wein_lambda_max_cm_values = [i * 1e-8 for i in BB_Wein_lambda_max_Angstrom_values]
            BB_amplitude = np.array([
                blackbody_F_lam(lam_cm=lam_cm, T_K=BB_T[i], R_cm=BB_R[i], D_cm=ANT_D_cm)
                for i, lam_cm in enumerate(BB_Wein_lambda_max_cm_values)
            ])
            # Ensure amplitudes are finite
            finite_mask = np.isfinite(BB_amplitude)
            if not finite_mask.all():
                logger.warning(f"Dropping {np.size(finite_mask) - finite_mask.sum()} SBB rows with non-finite amplitudes")
                phase = phase[finite_mask]
                BB_T = BB_T[finite_mask]
                BB_R = BB_R[finite_mask]
                BB_amplitude = BB_amplitude[finite_mask]
                n_time = len(phase)
                if n_time < 2:
                    raise ValueError("Not enough SBB samples after dropping non-finite amplitudes.")
                safe_kx, safe_ky = _safe_spline_degrees(n_time, len(lbda_A), safe_kx, safe_ky)
            logger.info(f"Generated blackbody amplitudes for {ANT_name}")
            logger.info(f"Setting up skysurvey blackbody source for {ANT_name}")
            bb_source = blackbody.get_blackbody_transient_source(
                phase=phase,
                amplitude=BB_amplitude,
                temperature=BB_T,
                lbda=lbda_A,
                time_spline_degree=safe_kx,
                wavelength_spline_degree=safe_ky,
            )
            logger.info(f"Drawing sources for {ANT_name}")
            logger.info(rf"Here are the inputs: \
                        size: {size} \n \
                        model: {self.model} \n \
                        tstart: {tstart} \n \
                        tstop: {tstop} \n \
                        zmax: {ANT_max_simulation_z}"
            )
            sources = TSTransient.from_draw(
                size=size, model=self.model, template=bb_source, tstart=tstart, tstop=tstop, zmax=ANT_max_simulation_z
            )

            if plot_skysurvey_inputs:
                fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
                ax1, ax2, ax3 = axs.flatten()
                ax1.set_title('INPUT: BB T', fontweight='bold')
                ax1.set_ylabel('BB Temperature / K', fontweight='bold')
                ax1.plot(phase, BB_T, marker='o', linestyle='None', mec='k', mew=0.5)
                ax2.set_title('INPUT: BB amplitude (= BB flux density at the BB peak wavelength)', fontweight='bold')
                ax2.set_ylabel('BB Amplitude / cm', fontweight='bold')
                ax2.plot(phase, BB_amplitude, marker='o', linestyle='None', mec='k', mew=0.5)
                ax3.set_title('Not input: BB R, but used to calculate the amplitude', fontweight='bold')
                ax3.set_ylabel('BB Radius / cm', fontweight='bold')
                ax3.plot(phase, BB_R, marker='o', linestyle='None', mec='k', mew=0.5)
                for ax in axs.ravel():
                    ax.grid(True)
                fig.supxlabel('Phase (days since peak) / rest frame days', fontweight='bold')
                fig.suptitle(f"Input params for {ANT_name}'s SBB SED fit", fontweight='bold', fontsize=20)
                fig.tight_layout()
                os.makedirs(sed_dir, exist_ok=True)
                plt.savefig(os.path.join(sed_dir, f"{ANT_name}_INPUT_PARAMS_SBB.png"), dpi=300)

            if plot_SED_results:
                logger.info(f"Plotting SBB SED sanity check for {ANT_name}")
                titlefontsize = 17
                fluxes = blackbody.get_blackbody_transient_flux(
                    lbda=lbda_A, temperature=BB_T, amplitude=BB_amplitude, normed=True
                )
                fig = plt.figure(figsize=(16, 7.5))
                ax = fig.add_subplot(111)
                cmap = plt.get_cmap('jet')
                colors = cmap((phase - phase.min()) / (phase.max() - phase.min()))
                _ = [ax.plot(lbda_A, flux_, color=c) for flux_, c in zip(fluxes, colors)]
                norm = Normalize(vmin=phase.min(), vmax=phase.max())
                sm = ScalarMappable(norm, cmap)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax)
                cbar.set_label('phase (days since peak) / rest frame days', fontweight='bold')
                ax.set_xlabel(r'Wavelength / $\mathbf{\AA}$', fontweight='bold', fontsize=(titlefontsize - 2))
                ax.set_ylabel(
                    r'Flux density  / ergs$ \mathbf{ s^{-1} cm^{-2} \AA^{-1} } $',
                    fontweight='bold',
                    fontsize=(titlefontsize - 2),
                )
                ax.set_title(
                    f"Using blackbody.get_blackbody_transient_flux() and inputting our BB T and amplitudes \n for {ANT_name}'s SBB SED fits. \nWe use this to generate the transient model to simulate lightcurves from",
                    fontsize=titlefontsize,
                    fontweight='bold',
                )
                fig.tight_layout()
                os.makedirs(sed_dir, exist_ok=True)
                plt.savefig(os.path.join(sed_dir, f"{ANT_name}_GENERATE_INPUT_SBB_SEDs.png"), dpi=300)

        elif 'PL' in SED_filename:
            SED_label = 'PL'
            sed_dir = os.path.join(output_dir, ANT_name, f"LSST_{SED_label}")
            ANT_good_SEDs = ANT_SED_df.copy()
            # Clean and guard time series
            raw_count = len(ANT_good_SEDs)
            clean_df = _sanitize_time_series(ANT_good_SEDs, 'd_since_peak', ['A_F_density', 'gamma'])
            if len(clean_df) < raw_count:
                logger.info(f"Sanitized PL inputs: {raw_count} -> {len(clean_df)} rows after NaN/inf removal and duplicate-time drop")
            phase = clean_df['d_since_peak'].to_numpy()
            PL_A_F_density = clean_df['A_F_density'].to_numpy()
            PL_gamma = clean_df['gamma'].to_numpy()
            n_time = len(phase)
            if n_time < 2:
                raise ValueError(f"Not enough PL time samples after cleaning (n={n_time}); need at least 2 to build splines.")
            safe_kx, safe_ky = _safe_spline_degrees(n_time, len(lbda_A), time_spline_degree, wavelength_spline_degree)
            if (safe_kx != time_spline_degree) or (safe_ky != wavelength_spline_degree):
                logger.warning(f"Adjusted spline degrees to safe values: time k={safe_kx}, wave k={safe_ky} (requested {time_spline_degree}, {wavelength_spline_degree})")

            pl_source = power_law.get_power_law_transient_source(
                phase=phase,
                amplitude=PL_A_F_density,
                gamma=PL_gamma,
                lbda=lbda_A,
                time_spline_degree=safe_kx,
                wavelength_spline_degree=safe_ky,
            )
            sources = TSTransient.from_draw(
                size=size, model=self.model, template=pl_source, tstart=tstart, tstop=tstop, zmax=ANT_max_simulation_z
            )

            if plot_skysurvey_inputs:
                # Optional: add PL input plots if desired
                pass

            if plot_SED_results:
                logger.info(f"Plotting PL SED sanity check for {ANT_name}")
                titlefontsize = 17
                fluxes = power_law.get_power_law_transient_flux(
                    lbda=lbda_A, A=PL_A_F_density, gamma=PL_gamma
                )
                fig = plt.figure(figsize=(16, 7.5))
                ax = fig.add_subplot(111)
                cmap = plt.get_cmap('jet')
                colors = cmap((phase - phase.min()) / (phase.max() - phase.min()))
                norm = Normalize(vmin=phase.min(), vmax=phase.max())
                sm = ScalarMappable(norm, cmap)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax)
                cbar.set_label('phase (days since peak) / rest frame days', fontweight='bold')
                _ = [ax.plot(lbda_A, flux_, color=c) for flux_, c in zip(fluxes, colors)]
                ax.set_xlabel(r'Wavelength / $\\mathbf{\\AA}$', fontweight='bold', fontsize=(titlefontsize - 2))
                ax.set_ylabel(
                    r'Flux density  / ergs$ \\mathbf{ s^{-1} cm^{-2} \\AA^{-1} } $',
                    fontweight='bold',
                    fontsize=(titlefontsize - 2),
                )
                ax.set_title(
                    f"Using power_law.get_power_law_transient_flux() and inputting our $A_{{F_{{\\lambda}}}}$ and $\\gamma$s  \n for {ANT_name}'s PL SED fits. \nWe use this to generate the transient model to simulate lightcurves from",
                    fontsize=titlefontsize,
                    fontweight='bold',
                )
                fig.tight_layout()
                os.makedirs(sed_dir, exist_ok=True)
                plt.savefig(os.path.join(sed_dir, f"{ANT_name}_GENERATE_INPUT_PL_SEDs.png"), dpi=300)

        elif 'DBB' in SED_filename:
            SED_label = 'DBB'
            sed_dir = os.path.join(output_dir, ANT_name, f"LSST_{SED_label}")
            ANT_good_SEDs = ANT_SED_df.copy()
            # Clean and guard time series
            raw_count = len(ANT_good_SEDs)
            clean_df = _sanitize_time_series(ANT_good_SEDs, 'd_since_peak', ['brute_T1_K', 'brute_R1_cm', 'brute_T2_K', 'brute_R2_cm'])
            if len(clean_df) < raw_count:
                logger.info(f"Sanitized DBB inputs: {raw_count} -> {len(clean_df)} rows after NaN/inf removal and duplicate-time drop")
            phase = clean_df['d_since_peak'].to_numpy()
            BB_T1 = clean_df['brute_T1_K'].to_numpy()
            BB_R1 = clean_df['brute_R1_cm'].to_numpy()
            BB_T2 = clean_df['brute_T2_K'].to_numpy()
            BB_R2 = clean_df['brute_R2_cm'].to_numpy()
            n_time = len(phase)
            if n_time < 2:
                raise ValueError(f"Not enough DBB time samples after cleaning (n={n_time}); need at least 2 to build splines.")
            safe_kx, safe_ky = _safe_spline_degrees(n_time, len(lbda_A), time_spline_degree, wavelength_spline_degree)
            if (safe_kx != time_spline_degree) or (safe_ky != wavelength_spline_degree):
                logger.warning(f"Adjusted spline degrees to safe values: time k={safe_kx}, wave k={safe_ky} (requested {time_spline_degree}, {wavelength_spline_degree})")

            BB1_Wein_lambda_max_Angstrom = [get_wein_lbdamax(T) for T in BB_T1]
            BB1_Wein_lambda_max_Angstrom_values = [i.value for i in BB1_Wein_lambda_max_Angstrom]
            BB1_Wein_lambda_max_cm_values = [i * 1e-8 for i in BB1_Wein_lambda_max_Angstrom_values]
            BB1_amplitude = np.array([
                blackbody_F_lam(lam_cm=lam_cm, T_K=BB_T1[i], R_cm=BB_R1[i], D_cm=ANT_D_cm)
                for i, lam_cm in enumerate(BB1_Wein_lambda_max_cm_values)
            ])
            BB2_Wein_lambda_max_Angstrom = [get_wein_lbdamax(T) for T in BB_T2]
            BB2_Wein_lambda_max_Angstrom_values = [i.value for i in BB2_Wein_lambda_max_Angstrom]
            BB2_Wein_lambda_max_cm_values = [i * 1e-8 for i in BB2_Wein_lambda_max_Angstrom_values]
            BB2_amplitude = np.array([
                blackbody_F_lam(lam_cm=lam_cm, T_K=BB_T2[i], R_cm=BB_R2[i], D_cm=ANT_D_cm)
                for i, lam_cm in enumerate(BB2_Wein_lambda_max_cm_values)
            ])
            # Ensure amplitudes are finite and aligned
            finite_mask = np.isfinite(BB1_amplitude) & np.isfinite(BB2_amplitude)
            if not finite_mask.all():
                logger.warning(f"Dropping {np.size(finite_mask) - finite_mask.sum()} DBB rows with non-finite amplitudes")
                phase = phase[finite_mask]
                BB_T1 = BB_T1[finite_mask]
                BB_R1 = BB_R1[finite_mask]
                BB_T2 = BB_T2[finite_mask]
                BB_R2 = BB_R2[finite_mask]
                BB1_amplitude = BB1_amplitude[finite_mask]
                BB2_amplitude = BB2_amplitude[finite_mask]
                n_time = len(phase)
                if n_time < 2:
                    raise ValueError("Not enough DBB samples after dropping non-finite amplitudes.")
                safe_kx, safe_ky = _safe_spline_degrees(n_time, len(lbda_A), safe_kx, safe_ky)

            dbb_source = double_blackbody.get_double_blackbody_transient_source(
                phase=phase,
                amplitude_1=BB1_amplitude,
                temperature_1=BB_T1,
                amplitude_2=BB2_amplitude,
                temperature_2=BB_T2,
                lbda=lbda_A,
                time_spline_degree=safe_kx,
                wavelength_spline_degree=safe_ky,
            )
            sources = TSTransient.from_draw(
                size=size, model=self.model, template=dbb_source, tstart=tstart, tstop=tstop, zmax=ANT_max_simulation_z
            )

            if plot_skysurvey_inputs:
                fig, axs = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
                ax1, ax2, ax3, ax4, ax5, ax6 = axs.flatten()
                ax1.set_title('INPUT: BB T1', fontweight='bold')
                ax1.set_ylabel('BB Temperature / K', fontweight='bold')
                ax1.plot(phase, BB_T1, marker='o', linestyle='None', mec='k', mew=0.5)
                ax3.set_title('INPUT: BB amplitude1', fontweight='bold')
                ax3.set_ylabel('BB Amplitude / cm', fontweight='bold')
                ax3.plot(phase, BB1_amplitude, marker='o', linestyle='None', mec='k', mew=0.5)
                ax5.set_title('Not input: BB R1', fontweight='bold')
                ax5.set_ylabel('BB Radius / cm', fontweight='bold')
                ax5.plot(phase, BB_R1, marker='o', linestyle='None', mec='k', mew=0.5)
                ax2.set_title('INPUT: BB T2', fontweight='bold')
                ax2.set_ylabel('BB Temperature / K', fontweight='bold')
                ax2.plot(phase, BB_T2, marker='o', linestyle='None', mec='k', mew=0.5, c='red')
                ax4.set_title('INPUT: BB amplitude2', fontweight='bold')
                ax4.set_ylabel('BB Amplitude / cm', fontweight='bold')
                ax4.plot(phase, BB2_amplitude, marker='o', linestyle='None', mec='k', mew=0.5, c='red')
                ax6.set_title('Not input: BB R2', fontweight='bold')
                ax6.set_ylabel('BB Radius / cm', fontweight='bold')
                ax6.plot(phase, BB_R2, marker='o', linestyle='None', mec='k', mew=0.5, c='red')
                for ax in axs.ravel():
                    ax.grid(True)
                fig.supxlabel('Phase (days since peak) / rest frame days', fontweight='bold')
                fig.suptitle(f"Input params for {ANT_name}'s DBB SED fit", fontweight='bold', fontsize=20)
                fig.tight_layout()
                os.makedirs(sed_dir, exist_ok=True)
                plt.savefig(os.path.join(sed_dir, f"{ANT_name}_INPUT_PARAMS_DBB.png"), dpi=300)

            if plot_SED_results:
                logger.info(f"Plotting DBB SED sanity check for {ANT_name}")
                titlefontsize = 17
                fluxes1 = double_blackbody.get_blackbody_transient_flux(
                    lbda=lbda_A, temperature=BB_T1, amplitude=BB1_amplitude, normed=True
                )
                fluxes2 = double_blackbody.get_blackbody_transient_flux(
                    lbda=lbda_A, temperature=BB_T2, amplitude=BB2_amplitude, normed=True
                )
                fluxes = fluxes1 + fluxes2
                fig = plt.figure(figsize=(16, 7.5))
                ax = fig.add_subplot(111)
                cmap = plt.get_cmap('jet')
                colors = cmap((phase - phase.min()) / (phase.max() - phase.min()))
                _ = [ax.plot(lbda_A, flux_, color=c) for flux_, c in zip(fluxes, colors)]
                norm = Normalize(vmin=phase.min(), vmax=phase.max())
                sm = ScalarMappable(norm, cmap)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax)
                cbar.set_label('phase (days since peak) / rest frame days', fontweight='bold')
                ax.set_xlabel(r'Wavelength / $\\mathbf{\\AA}$', fontweight='bold', fontsize=(titlefontsize - 2))
                ax.set_ylabel(
                    r'Flux density  / ergs$ \\mathbf{ s^{-1} cm^{-2} \\AA^{-1} } $',
                    fontweight='bold',
                    fontsize=(titlefontsize - 2),
                )
                ax.set_title(
                    f"Using blackbody.get_blackbody_transient_flux() and inputting our BB T1, amplitude1, \nBB T2 and amplitude2s for {ANT_name}'s DBB SED fits. \nWe use this to generate the transient model to simulate lightcurves from",
                    fontsize=titlefontsize,
                    fontweight='bold',
                )
                fig.tight_layout()
                os.makedirs(sed_dir, exist_ok=True)
                plt.savefig(os.path.join(sed_dir, f"{ANT_name}_GENERATE_INPUT_DBB_SEDs.png"), dpi=300)

        else:
            raise ValueError(f"Unrecognized SED type for filename: {SED_filename}")

        # Generate light curves
        logger.info(f"Generating LSST light curves for {ANT_name} using {SED_label} SED fits")
        dset = DataSet.from_targets_and_survey(sources, self.survey, trim_observations=True)
        dset.get_ndetection()
        b_obs_cent_wl_dict = {'lsstu': 3671, 'lsstg': 4827, 'lsstr': 6223, 'lssti': 7546, 'lsstz': 8691, 'lssty': 9712}
        c_cgs = const.c.cgs.value
        h_cgs = const.h.cgs.value
        F_density_conversion = c_cgs * h_cgs
        band_conversion = [(F_density_conversion / wl) for wl in b_obs_cent_wl_dict.values()]
        band_F_density_conversion_dict = dict(zip(b_obs_cent_wl_dict.keys(), band_conversion))
        b_colour_dict = {'lsstu': 'purple', 'lsstg': 'blue', 'lsstr': 'green', 'lssti': 'yellow', 'lsstz': 'orange', 'lssty': 'red'}
        b_name_dict = {'lsstu': 'LSST_u', 'lsstg': 'LSST_g', 'lsstr': 'LSST_r', 'lssti': 'LSST_i', 'lsstz': 'LSST_z', 'lssty': 'LSST_y'}
        def standard_form_tex(x, pos):
            if x == 0:
                return "0"
            exponent = int(np.floor(np.log10(abs(x))))
            coeff = x / (10 ** exponent)
            return rf"${coeff:.1f} \times 10^{{{exponent}}}$"
        formatter = FuncFormatter(standard_form_tex)
        if SED_label == 'PL':
            title_label = 'power-law'
        elif SED_label == 'SBB':
            title_label = 'single-blackbody'
        elif SED_label == 'DBB':
            title_label = 'double-blackbody'

        # Save HDF5 of all simulated LCs
        from utils.io_utils import save_lightcurves_hdf5
        h5_dir = os.path.join(output_dir, ANT_name, f"LSST_{SED_label}")
        os.makedirs(h5_dir, exist_ok=True)
        h5_path = os.path.join(h5_dir, f"{ANT_name}_SED_{SED_label}_simulated_lightcurves.h5")
        written = save_lightcurves_hdf5(dset, sources, h5_path, band_F_density_conversion_dict)
        logger.info(f"Saved simulated light curves to {written}")

        for j, index in enumerate(dset.obs_index):
            if j in [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128]:
                a = 0
                fig, axs = plt.subplots(4, 2, figsize = (8.2, 11.6))
                handles = []
                labels = []
                axs = axs.ravel()
                titlefontsize = 18
                fig.supylabel('Spectral flux density '+r' [ergs $\mathbf{s^{-1} \, cm^{-2} \, \AA^{-1}} $]', fontweight = 'bold', fontsize = (titlefontsize - 4))
                fig.supxlabel('MJD [days]', fontweight = 'bold', fontsize = (titlefontsize - 4))
                fig.suptitle(f"Simulated light curves using {ANT_name}'s \n{title_label}-inferred spectral evolution", fontsize = titlefontsize, fontweight = 'bold')
                fig.subplots_adjust(left = 0.18, right = 0.85, hspace = 0.45, wspace = 0.45, bottom = 0.1)
            if j > 135:
                break
            if j not in [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128]:
                a += 1
            ax = axs[a]
            ax.grid(True)
            ax.yaxis.set_major_formatter(formatter)
            ax.get_yaxis().get_offset_text().set_visible(False)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
            ax.tick_params(axis='both', labelsize = 10)
            observation = dset.get_target_lightcurve(index = index).copy()
            sim_transient_params = sources.data.iloc[index, :]
            sim_z = sim_transient_params['z']
            sim_magabs = sim_transient_params['magabs']
            sim_ra = sim_transient_params['ra']
            sim_dec = sim_transient_params['dec']
            sim_magobs = sim_transient_params['magobs']
            bands = observation['band'].unique()
            subplot_title = f"z = {sim_z:.4f}"
            ax.set_title(subplot_title, fontweight = 'bold', fontsize = 11)
            for b in bands:
                b_observation = observation[observation['band'] == b].copy()
                h = ax.errorbar(b_observation['mjd'], b_observation['flux'] * band_F_density_conversion_dict[b], yerr = b_observation['fluxerr']* band_F_density_conversion_dict[b], c = b_colour_dict[b], mec = 'k', mew = '0.5', fmt = 'o', label = b_name_dict[b])
                handle = h[0]
                label = b_name_dict[b]
                if label not in labels:
                    handles.append(handle)
                    labels.append(label)
            if j in [7, 15, 23, 31, 39, 47, 55, 63, 71, 79, 87, 95, 103, 111, 119, 127, 135]:
                sed_dir = os.path.join(output_dir, ANT_name, f"LSST_{SED_label}")
                os.makedirs(sed_dir, exist_ok=True)
                base_savepath = os.path.join(sed_dir, f"FINAL_SUBPLOT_{ANT_name}_SED_{SED_label}_lc_sim")
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                base_savepath = f"{base_savepath}_{timestamp}"
                i = 1
                while os.path.exists(f"{base_savepath}__{i}.png"):
                    i += 1
                new_savepath = f"{base_savepath}__{i}.png"
                plt.legend(handles, labels, loc = 'upper right', fontsize = 10, bbox_to_anchor = (1.5, 5.5))
                plt.savefig(new_savepath, dpi = 300)
                plt.close(fig)
        i_before_run = None
        for j, index in enumerate(dset.obs_index):
            observation = dset.get_target_lightcurve(index = index).copy()
            sim_transient_params = sources.data.iloc[index, :]
            sim_z = sim_transient_params['z']
            sim_magabs = sim_transient_params['magabs']
            sim_ra = sim_transient_params['ra']
            sim_dec = sim_transient_params['dec']
            sim_magobs = sim_transient_params['magobs']
            bands = observation['band'].unique()
            b_colour_dict = {'lsstu': 'purple', 'lsstg': 'blue', 'lsstr': 'green', 'lssti': 'yellow', 'lsstz': 'orange', 'lssty': 'red'}
            fig, axs = plt.subplots(1, 2, figsize = (16, 7.5))
            ax1, ax2 = axs
            titlefontsize = 20
            for b in bands:
                b_observation = observation[observation['band'] == b].copy()
                ax1.errorbar(b_observation['mjd'], b_observation['flux'], yerr = b_observation['fluxerr'], c = b_colour_dict[b], mec = 'k', mew = '0.5', fmt = 'o', label = b)
                ax2.errorbar(b_observation['mjd'], b_observation['flux'] * band_F_density_conversion_dict[b], yerr = b_observation['fluxerr']* band_F_density_conversion_dict[b], c = b_colour_dict[b], mec = 'k', mew = '0.5', fmt = 'o', label = b)
            ax1.set_ylabel(r'Flux (scaled by AB mag system)\n'+r'/ photons $\mathbf{s^{-1} cm^{-2}} $', fontweight = 'bold', fontsize = (titlefontsize - 5))
            ax2.set_ylabel(r'Flux density (scaled by AB mag system) \n'+r'/ ergs $\mathbf{s^{-1} cm^{-2} \AA^{-1}} $', fontweight = 'bold', fontsize = (titlefontsize - 5))
            ax1.legend()
            ax2.grid(True)
            ax1.grid(True)
            fig.supxlabel('MJD', fontweight = 'bold', fontsize = (titlefontsize - 5))
            title = f"Simulated light curve using {ANT_name}'s {SED_label} SED fit results\n" + \
                    f"z = {sim_z:.4f}, magabs = {sim_magabs:.4f}, magobs = {sim_magobs:.4f}, ra, dec = ({sim_ra:.4f}, {sim_dec:.4f})"
            fig.suptitle(title, fontweight = 'bold', fontsize = titlefontsize)
            sed_dir = os.path.join(output_dir, ANT_name, f"LSST_{SED_label}")
            os.makedirs(sed_dir, exist_ok=True)
            base_savepath = os.path.join(sed_dir, f"{ANT_name}_SED_{SED_label}_lc_sim")
            i = 1
            while os.path.exists(f"{base_savepath}_{i}.png"):
                i += 1
            if j == 0:
                i_before_run = i - 1
            if i > (i_before_run + no_plots_to_save):
                break
            new_savepath = f"{base_savepath}_{i}.png"
            plt.savefig(new_savepath, dpi = 300)
            plt.close(fig)
        logger.info(f"Finished simulation for SED file: {SED_filename}")
