"""
Publication-quality Raman spectrum plots.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from .peak_fitter import PeakResult
from .analyzer    import RamanAnalysis

PEAK_COLORS = {
    "D":       "#e74c3c",
    "G":       "#2ecc71",
    "D_prime": "#f39c12",
    "2D":      "#3498db",
    "DG":      "#9b59b6",
}
PEAK_LABELS = {
    "D": "D", "G": "G", "D_prime": "D'", "2D": "2D", "DG": "D+G"
}


def _style():
    plt.rcParams.update({
        "font.family":     "DejaVu Sans",
        "font.size":       11,
        "axes.linewidth":  1.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "figure.dpi":      150,
    })


def plot_full_spectrum(wavenumber, raw_intensity, baseline, corrected,
                       filename, output_dir):
    """Plot 1: Raw spectrum with baseline overlay."""
    _style()
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(f"Baseline Correction — {filename}", fontsize=13, fontweight="bold")

    ax1, ax2 = axes
    ax1.plot(wavenumber, raw_intensity, color="#2c3e50", lw=1.2, label="Raw spectrum")
    ax1.plot(wavenumber, baseline,      color="#e74c3c", lw=1.5, ls="--", label="ALS Baseline")
    ax1.fill_between(wavenumber, baseline, raw_intensity, alpha=0.12, color="#3498db")
    ax1.set_ylabel("Intensity (a.u.)")
    ax1.legend(frameon=False)

    ax2.plot(wavenumber, corrected, color="#27ae60", lw=1.3, label="Baseline corrected")
    ax2.axhline(0, color="gray", lw=0.8, ls="--")
    ax2.set_xlabel("Raman Shift (cm⁻¹)")
    ax2.set_ylabel("Intensity (a.u.)")
    ax2.legend(frameon=False)

    plt.tight_layout()
    out = Path(output_dir) / f"{Path(filename).stem}_baseline.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    return str(out)


def plot_fitted_peaks(wavenumber, corrected, peaks: dict,
                      filename, output_dir):
    """Plot 2: Baseline-corrected spectrum with all fitted peaks labeled."""
    _style()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(wavenumber, corrected, color="#2c3e50", lw=1.3, label="Corrected spectrum", zorder=2)

    for key, p in peaks.items():
        if p.found and len(p.model_x) > 0:
            color = PEAK_COLORS.get(key, "gray")
            label = PEAK_LABELS.get(key, key)
            ax.fill_between(p.model_x, p.model_y, alpha=0.35, color=color)
            ax.plot(p.model_x, p.model_y, color=color, lw=2.0, label=f"{label} fit")
            ax.axvline(p.center, color=color, lw=0.8, ls=":", alpha=0.7)
            ax.text(p.center, p.amplitude * 1.08, label,
                    ha="center", va="bottom", fontsize=11,
                    fontweight="bold", color=color)

    ax.set_xlabel("Raman Shift (cm⁻¹)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_title(f"Peak Fitting — {filename}", fontweight="bold")
    ax.legend(frameon=False, ncol=3, fontsize=9)
    plt.tight_layout()
    out = Path(output_dir) / f"{Path(filename).stem}_peaks.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    return str(out)


def plot_individual_peaks(wavenumber, corrected, peaks: dict,
                          filename, output_dir):
    """Plot 3: Individual peak fits with residuals."""
    found = {k: p for k, p in peaks.items() if p.found and len(p.model_x) > 0}
    if not found:
        return None
    _style()
    n   = len(found)
    fig = plt.figure(figsize=(5 * n, 7))
    gs  = gridspec.GridSpec(2, n, height_ratios=[4, 1], hspace=0.05)

    for i, (key, p) in enumerate(found.items()):
        ax_main = fig.add_subplot(gs[0, i])
        ax_res  = fig.add_subplot(gs[1, i], sharex=ax_main)
        color   = PEAK_COLORS.get(key, "gray")
        label   = PEAK_LABELS.get(key, key)

        # Actual data in window
        mask = (wavenumber >= p.model_x[0]) & (wavenumber <= p.model_x[-1])
        xd = wavenumber[mask]
        yd = corrected[mask]
        yfit = np.interp(xd, p.model_x, p.model_y)

        ax_main.scatter(xd, yd, s=8, color="#2c3e50", alpha=0.6, zorder=3)
        ax_main.plot(p.model_x, p.model_y, color=color, lw=2.0)
        ax_main.fill_between(p.model_x, p.model_y, alpha=0.25, color=color)
        ax_main.set_title(f"{label} peak\n"
                          f"pos={p.center:.1f} cm⁻¹\n"
                          f"FWHM={p.fwhm:.1f} cm⁻¹\n"
                          f"R²={p.r_squared:.3f}", fontsize=9)
        ax_main.set_ylabel("Intensity" if i == 0 else "")
        plt.setp(ax_main.get_xticklabels(), visible=False)

        residuals = yd - yfit
        ax_res.axhline(0, color="gray", lw=0.8, ls="--")
        ax_res.scatter(xd, residuals, s=5, color=color, alpha=0.7)
        ax_res.set_xlabel("Raman Shift (cm⁻¹)", fontsize=8)
        ax_res.set_ylabel("Res." if i == 0 else "")

    fig.suptitle(f"Individual Peak Analysis — {filename}", fontweight="bold")
    out = Path(output_dir) / f"{Path(filename).stem}_individual.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    return str(out)


def plot_ratios(analysis: RamanAnalysis, filename, output_dir):
    """Plot 4: Summary bar chart of intensity ratios."""
    _style()
    ratios = {}
    if not np.isnan(analysis.ID_IG_height):   ratios["ID/IG"]   = analysis.ID_IG_height
    if not np.isnan(analysis.I2D_IG_height):  ratios["I2D/IG"]  = analysis.I2D_IG_height
    if not np.isnan(analysis.IDp_IG_height):  ratios["ID'/IG"]  = analysis.IDp_IG_height
    if not np.isnan(analysis.ID_IDp_height):  ratios["ID/ID'"]  = analysis.ID_IDp_height

    if not ratios:
        return None

    fig, ax = plt.subplots(figsize=(7, 4))
    colors  = ["#e74c3c", "#3498db", "#f39c12", "#9b59b6"]
    bars    = ax.bar(list(ratios.keys()), list(ratios.values()),
                     color=colors[:len(ratios)], edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, ratios.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=10)

    ax.set_ylabel("Intensity Ratio")
    ax.set_title(f"Raman Intensity Ratios — {filename}", fontweight="bold")
    ax.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.5, label="Ratio = 1")
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    out = Path(output_dir) / f"{Path(filename).stem}_ratios.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    return str(out)
