"""
Publication-quality Raman spectrum plots.

v2.4 additions:
  - D* band colour and label registered in PEAK_COLORS / PEAK_LABELS.
  - plot_batch_heatmap(): ratio heatmap + L_D bar chart for batch results
    (Roadmap feature #10 — Batch statistics + ratio heatmap).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Optional
from .peak_fitter import PeakResult
from .analyzer    import RamanAnalysis

PEAK_COLORS = {
    "D_star":  "#1abc9c",   # v2.4 — teal
    "D":       "#e74c3c",
    "G":       "#2ecc71",
    "D_prime": "#f39c12",
    "2D":      "#3498db",
    "DG":      "#9b59b6",
}
PEAK_LABELS = {
    "D_star": "D*",
    "D": "D", "G": "G", "D_prime": "D\u2032", "2D": "2D", "DG": "D+G"
}

# ── Ratio columns used in the heatmap (Roadmap #10) ───────
_HEATMAP_RATIOS = [
    ("ID_IG_height",    "I$_{\\mathrm{D}}$/I$_{\\mathrm{G}}$"),
    ("I2D_IG_height",   "I$_{\\mathrm{2D}}$/I$_{\\mathrm{G}}$"),
    ("IDp_IG_height",   "I$_{\\mathrm{D\\prime}}$/I$_{\\mathrm{G}}$"),
    ("ID_IDp_height",   "I$_{\\mathrm{D}}$/I$_{\\mathrm{D\\prime}}$"),
    ("IDstar_IG_height","I$_{\\mathrm{D*}}$/I$_{\\mathrm{G}}$"),
]


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
    fig.suptitle(f"Baseline Correction \u2014 {filename}", fontsize=13, fontweight="bold")

    ax1, ax2 = axes
    ax1.plot(wavenumber, raw_intensity, color="#2c3e50", lw=1.2, label="Raw spectrum")
    ax1.plot(wavenumber, baseline,      color="#e74c3c", lw=1.5, ls="--", label="ALS Baseline")
    ax1.fill_between(wavenumber, baseline, raw_intensity, alpha=0.12, color="#3498db")
    ax1.set_ylabel("Intensity (a.u.)")
    ax1.legend(frameon=False)

    ax2.plot(wavenumber, corrected, color="#27ae60", lw=1.3, label="Baseline corrected")
    ax2.axhline(0, color="gray", lw=0.8, ls="--")
    ax2.set_xlabel("Raman Shift (cm\u207b\u00b9)")
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

    ax.set_xlabel("Raman Shift (cm\u207b\u00b9)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_title(f"Peak Fitting \u2014 {filename}", fontweight="bold")
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

        mask = (wavenumber >= p.model_x[0]) & (wavenumber <= p.model_x[-1])
        xd = wavenumber[mask]
        yd = corrected[mask]
        yfit = np.interp(xd, p.model_x, p.model_y)

        ax_main.scatter(xd, yd, s=8, color="#2c3e50", alpha=0.6, zorder=3)
        ax_main.plot(p.model_x, p.model_y, color=color, lw=2.0)
        ax_main.fill_between(p.model_x, p.model_y, alpha=0.25, color=color)
        ax_main.set_title(
            f"{label} peak\n"
            f"pos={p.center:.1f} cm\u207b\u00b9\n"
            f"FWHM={p.fwhm:.1f} cm\u207b\u00b9\n"
            f"R\u00b2={p.r_squared:.3f}",
            fontsize=9,
        )
        ax_main.set_ylabel("Intensity" if i == 0 else "")
        plt.setp(ax_main.get_xticklabels(), visible=False)

        residuals = yd - yfit
        ax_res.axhline(0, color="gray", lw=0.8, ls="--")
        ax_res.scatter(xd, residuals, s=5, color=color, alpha=0.7)
        ax_res.set_xlabel("Raman Shift (cm\u207b\u00b9)", fontsize=8)
        ax_res.set_ylabel("Res." if i == 0 else "")

    fig.suptitle(f"Individual Peak Analysis \u2014 {filename}", fontweight="bold")
    out = Path(output_dir) / f"{Path(filename).stem}_individual.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    return str(out)


def plot_ratios(analysis: RamanAnalysis, filename, output_dir):
    """Plot 4: Summary bar chart of intensity ratios."""
    _style()
    ratios = {}
    if not np.isnan(analysis.ID_IG_height):    ratios["ID/IG"]   = analysis.ID_IG_height
    if not np.isnan(analysis.I2D_IG_height):   ratios["I2D/IG"]  = analysis.I2D_IG_height
    if not np.isnan(analysis.IDp_IG_height):   ratios["ID\u2032/IG"]  = analysis.IDp_IG_height
    if not np.isnan(analysis.ID_IDp_height):   ratios["ID/ID\u2032"]  = analysis.ID_IDp_height
    if not np.isnan(analysis.IDstar_IG_height):
        ratios["ID*/IG"] = analysis.IDstar_IG_height   # v2.4

    if not ratios:
        return None

    fig, ax = plt.subplots(figsize=(7, 4))
    colors  = ["#e74c3c", "#3498db", "#f39c12", "#9b59b6", "#1abc9c"]
    bars    = ax.bar(list(ratios.keys()), list(ratios.values()),
                     color=colors[:len(ratios)], edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, ratios.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=10)

    ax.set_ylabel("Intensity Ratio")
    ax.set_title(f"Raman Intensity Ratios \u2014 {filename}", fontweight="bold")
    ax.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.5, label="Ratio\u00a0=\u00a01")
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    out = Path(output_dir) / f"{Path(filename).stem}_ratios.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    return str(out)


def plot_batch_heatmap(
    batch_results: list[dict],
    output_dir: str,
    filename: str = "batch_heatmap.png",
) -> Optional[str]:
    """
    Batch statistics + ratio heatmap (Roadmap feature #10).

    Parameters
    ----------
    batch_results : list of dicts, each with keys:
        'filename'  (str)          — sample label
        'analysis'  (RamanAnalysis)
        'peaks'     (dict[str, PeakResult])   — optional, used for L_D
    output_dir : str
    filename   : str  — output PNG filename

    Returns
    -------
    str path to saved PNG, or None if < 2 valid samples.

    Layout
    ------
    Row 0 (top): heatmap of 5 intensity ratios  ×  N samples
                 Cells annotated with numeric value; NaN shown as hatched.
    Row 1 (bottom): horizontal bar chart of L_D [nm] per sample.
                    Bars coloured by magnitude (RdYlGn_r colormap).
    """
    if len(batch_results) < 2:
        return None

    _style()

    # ── Collect data ──────────────────────────────────────
    labels   = [r["filename"] for r in batch_results]
    n_samples = len(labels)
    n_ratios  = len(_HEATMAP_RATIOS)

    # ratio_matrix[ratio_idx, sample_idx]
    ratio_matrix = np.full((n_ratios, n_samples), np.nan)
    ld_values    = np.full(n_samples, np.nan)

    for j, rec in enumerate(batch_results):
        ana = rec.get("analysis")
        if ana is None:
            continue
        for i, (attr, _) in enumerate(_HEATMAP_RATIOS):
            val = getattr(ana, attr, np.nan)
            if not np.isnan(val):
                ratio_matrix[i, j] = val
        if not np.isnan(ana.L_D_nm):
            ld_values[j] = ana.L_D_nm

    # Drop columns where ALL ratios are NaN
    valid_cols = np.any(~np.isnan(ratio_matrix), axis=0)
    if not np.any(valid_cols):
        return None

    ratio_matrix = ratio_matrix[:, valid_cols]
    ld_values    = ld_values[valid_cols]
    labels       = [lb for lb, v in zip(labels, valid_cols) if v]
    n_samples    = len(labels)

    # Shorten labels for readability
    short_labels = [Path(lb).stem[:20] for lb in labels]

    # ── Figure layout ─────────────────────────────────────
    has_ld = np.any(~np.isnan(ld_values))
    nrows  = 2 if has_ld else 1
    fig_h  = 4.5 + n_samples * 0.35 + (2.5 if has_ld else 0)
    fig_w  = max(9, n_ratios * 1.8)

    fig = plt.figure(figsize=(fig_w, fig_h))
    if has_ld:
        gs = gridspec.GridSpec(
            2, 1,
            height_ratios=[n_ratios * 0.8, 2.2],
            hspace=0.45,
            figure=fig,
        )
        ax_heat = fig.add_subplot(gs[0])
        ax_ld   = fig.add_subplot(gs[1])
    else:
        ax_heat = fig.add_subplot(1, 1, 1)
        ax_ld   = None

    # ── Heatmap ───────────────────────────────────────────
    # Row-normalise so each ratio spans [0, 1] — highlights
    # relative variation across samples without ratio-scale
    # artefacts (e.g. ID/ID' = 7 dominating over ID/IG = 0.5).
    norm_matrix = np.full_like(ratio_matrix, np.nan)
    for i in range(n_ratios):
        row = ratio_matrix[i, :]
        valid = ~np.isnan(row)
        if valid.sum() >= 2:
            rmin, rmax = row[valid].min(), row[valid].max()
            if rmax > rmin:
                norm_matrix[i, valid] = (row[valid] - rmin) / (rmax - rmin)
            else:
                norm_matrix[i, valid] = 0.5
        elif valid.sum() == 1:
            norm_matrix[i, valid] = 0.5

    cmap = plt.cm.get_cmap("RdYlGn_r").copy()
    cmap.set_bad("#d0d0d0")

    im = ax_heat.imshow(
        norm_matrix,
        aspect="auto",
        cmap=cmap,
        vmin=0, vmax=1,
        interpolation="nearest",
    )

    # Hatch NaN cells
    for i in range(n_ratios):
        for j in range(n_samples):
            if np.isnan(ratio_matrix[i, j]):
                ax_heat.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, hatch="////",
                    edgecolor="#aaaaaa", linewidth=0.5,
                ))
            else:
                # Annotate with actual value
                val = ratio_matrix[i, j]
                txt_color = "white" if (norm_matrix[i, j] < 0.2 or norm_matrix[i, j] > 0.8) else "#1a1a1a"
                ax_heat.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=8.5, color=txt_color, fontweight="bold",
                )

    ratio_tick_labels = [lab for _, lab in _HEATMAP_RATIOS]
    ax_heat.set_yticks(range(n_ratios))
    ax_heat.set_yticklabels(ratio_tick_labels, fontsize=10)
    ax_heat.set_xticks(range(n_samples))
    ax_heat.set_xticklabels(short_labels, rotation=40, ha="right", fontsize=9)
    ax_heat.set_title("Batch Ratio Heatmap (row-normalised 0\u2013>1)",
                       fontweight="bold", fontsize=11)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.025, pad=0.02)
    cbar.set_label("Relative intensity (row-norm.)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # ── L_D bar chart ─────────────────────────────────────
    if ax_ld is not None:
        ld_plot   = np.where(np.isnan(ld_values), 0, ld_values)
        ld_valid  = ~np.isnan(ld_values)

        # Colour bars by L_D magnitude
        ld_max = ld_values[ld_valid].max() if ld_valid.any() else 1.0
        ld_min = ld_values[ld_valid].min() if ld_valid.any() else 0.0
        norm_ld = plt.Normalize(vmin=ld_min, vmax=ld_max)
        cmap_ld = plt.cm.get_cmap("viridis")

        bars = ax_ld.barh(
            range(n_samples),
            ld_plot,
            color=[cmap_ld(norm_ld(v)) if ok else "#cccccc"
                   for v, ok in zip(ld_plot, ld_valid)],
            edgecolor="white", linewidth=0.8,
        )

        # Annotate bars
        for j, (bar, ok, val) in enumerate(zip(bars, ld_valid, ld_values)):
            if ok:
                ax_ld.text(
                    bar.get_width() + ld_max * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f} nm",
                    va="center", fontsize=8.5,
                )
            else:
                ax_ld.text(
                    ld_max * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    "Stage 2 / N/A",
                    va="center", fontsize=7.5, color="#888888",
                )

        ax_ld.set_yticks(range(n_samples))
        ax_ld.set_yticklabels(short_labels, fontsize=9)
        ax_ld.set_xlabel("L$_{\\mathrm{D}}$ (nm)  [Canc\u0327ado 2011]", fontsize=10)
        ax_ld.set_title("Defect Spacing L$_{\\mathrm{D}}$ per Sample",
                         fontweight="bold", fontsize=11)
        ax_ld.invert_yaxis()   # match heatmap top-to-bottom order
        ax_ld.spines["top"].set_visible(False)
        ax_ld.spines["right"].set_visible(False)
        ax_ld.axvline(0, color="gray", lw=0.6)

    fig.suptitle(
        f"Batch Summary \u2014 {n_samples} samples",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out = Path(output_dir) / filename
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    return str(out)
