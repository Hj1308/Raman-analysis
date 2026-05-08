"""CSV and text report export."""

import csv
import numpy as np
from pathlib import Path
from .peak_fitter import PeakResult
from .analyzer    import RamanAnalysis


COLUMNS = [
    "filename", "laser_nm",
    "D_center", "D_FWHM", "D_height", "D_area", "D_R2",
    "G_center", "G_FWHM", "G_height", "G_area", "G_R2",
    "Dprime_center", "Dprime_FWHM", "Dprime_height",
    "twoD_center", "twoD_FWHM", "twoD_height", "twoD_area", "twoD_R2",
    "DG_center", "DG_FWHM", "DG_height",
    "ID_IG_height", "ID_IG_area",
    "I2D_IG_height", "I2D_IG_area",
    "IDp_IG_height", "ID_IDp_height",
    "L_D_nm", "disorder_stage", "defect_type", "estimated_layers",
]


def _peak_val(p: PeakResult, attr: str):
    if p is None or not p.found:
        return ""
    val = getattr(p, attr, np.nan)
    return f"{val:.4f}" if not np.isnan(val) else ""


def _ratio(val):
    return f"{val:.4f}" if not np.isnan(val) else ""


def append_csv(filepath: str, filename: str, laser_nm: float,
               peaks: dict, analysis: RamanAnalysis):
    """Append one row to the results CSV."""
    path    = Path(filepath)
    is_new  = not path.exists()

    D   = peaks.get("D")
    G   = peaks.get("G")
    Dp  = peaks.get("D_prime")
    twoD = peaks.get("2D")
    DG  = peaks.get("DG")

    row = {
        "filename":      filename,
        "laser_nm":      laser_nm,
        "D_center":      _peak_val(D,    "center"),
        "D_FWHM":        _peak_val(D,    "fwhm"),
        "D_height":      _peak_val(D,    "amplitude"),
        "D_area":        _peak_val(D,    "area"),
        "D_R2":          _peak_val(D,    "r_squared"),
        "G_center":      _peak_val(G,    "center"),
        "G_FWHM":        _peak_val(G,    "fwhm"),
        "G_height":      _peak_val(G,    "amplitude"),
        "G_area":        _peak_val(G,    "area"),
        "G_R2":          _peak_val(G,    "r_squared"),
        "Dprime_center": _peak_val(Dp,   "center"),
        "Dprime_FWHM":   _peak_val(Dp,   "fwhm"),
        "Dprime_height": _peak_val(Dp,   "amplitude"),
        "twoD_center":   _peak_val(twoD, "center"),
        "twoD_FWHM":     _peak_val(twoD, "fwhm"),
        "twoD_height":   _peak_val(twoD, "amplitude"),
        "twoD_area":     _peak_val(twoD, "area"),
        "twoD_R2":       _peak_val(twoD, "r_squared"),
        "DG_center":     _peak_val(DG,   "center"),
        "DG_FWHM":       _peak_val(DG,   "fwhm"),
        "DG_height":     _peak_val(DG,   "amplitude"),
        "ID_IG_height":  _ratio(analysis.ID_IG_height),
        "ID_IG_area":    _ratio(analysis.ID_IG_area),
        "I2D_IG_height": _ratio(analysis.I2D_IG_height),
        "I2D_IG_area":   _ratio(analysis.I2D_IG_area),
        "IDp_IG_height": _ratio(analysis.IDp_IG_height),
        "ID_IDp_height": _ratio(analysis.ID_IDp_height),
        "L_D_nm":        _ratio(analysis.L_D_nm),
        "disorder_stage": analysis.disorder_stage,
        "defect_type":    analysis.defect_type,
        "estimated_layers": analysis.estimated_layers,
    }

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def save_text_report(filepath: str, report_text: str):
    """Save text report to file."""
    with open(filepath, "w") as f:
        f.write(report_text)
