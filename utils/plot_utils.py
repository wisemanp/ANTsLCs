import os
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


def plot_sed_inputs_sbb(phase, BB_T, BB_R, BB_amplitude, outpath: str):
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
    fig.suptitle("Input params for SBB SED fit", fontweight='bold', fontsize=20)
    fig.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_sed_results_grid(lbda_A, fluxes, phase, outpath: str, title: str):
    titlefontsize = 17
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
    ax.set_ylabel(r'Flux density  / ergs$ \\mathbf{ s^{-1} cm^{-2} \\AA^{-1} } $', fontweight='bold', fontsize=(titlefontsize - 2))
    ax.set_title(title, fontsize=titlefontsize, fontweight='bold')
    fig.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_outputs_panel(fig, axs, handles, labels, legend_anchor=(1.5, 5.5), outpath=None):
    import matplotlib.pyplot as plt
    axs = axs.ravel()
    plt.legend(handles, labels, loc='upper right', fontsize=10, bbox_to_anchor=legend_anchor)
    if outpath:
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        plt.savefig(outpath, dpi=300)
    plt.close(fig)
