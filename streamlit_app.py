"""
Raman Spectrum Analyzer — Multi-Material Streamlit App
Author: Hoda Jaafari
Run:    streamlit run streamlit_app.py

Supported materials:
  - Graphene / sp2 carbon (D, G, D', 2D, D+G)
  - MoS2, WS2, MoSe2, WSe2, MoTe2  (TMDs)
  - h-BN
  - Black Phosphorus
"""

import io
import os
import sys
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openpyxl import Workbook, load_workbook
from openpyxl.styles import (
    Font, PatternFill, Alignment, Border, Side
)
from openpyxl.utils import get_column_letter
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from src.loader      import load_spectrum
from src.baseline    import correct_baseline
from src.peak_fitter import fit_all_peaks
from src.analyzer    import analyze, format_report

# ══════════════════════════════════════════════════════════
#  MATERIAL DEFINITIONS
# ══════════════════════════════════════════════════════════
MATERIAL_GROUPS = {
    "Graphene / sp² Carbon": [
        "Graphene", "Reduced Graphene Oxide (rGO)",
        "Graphene Oxide (GO)", "Carbon Nanotubes (CNT)",
        "Amorphous Carbon", "Graphite", "N-doped Graphene",
        "Other sp² Carbon"
    ],
    "TMD — Molybdenum": [
        "MoS₂", "MoSe₂", "MoTe₂", "MoO₂"
    ],
    "TMD — Tungsten": [
        "WS₂", "WSe₂", "WTe₂"
    ],
    "TMD — Other": [
        "NbSe₂", "TaS₂", "TiSe₂", "ReS₂", "ReSe₂"
    ],
    "Hexagonal Boron Nitride": [
        "h-BN", "BN nanosheet"
    ],
    "Black Phosphorus / Phosphorene": [
        "Black Phosphorus", "Phosphorene"
    ],
    "MXene": [
        "Ti₃C₂Tₓ", "Ti₂CTₓ", "V₂CTₓ", "Nb₂CTₓ"
    ],
}

# Peak windows per material at 532 nm (lo, hi) cm⁻¹
MATERIAL_PEAK_WINDOWS = {
    "graphene": {
        "D":  (1270, 1450), "G":  (1500, 1600),
        "D'": (1610, 1680), "2D": (2580, 2780), "D+G": (2850, 2960),
    },
    "mos2": {
        "E2g": (370, 395), "A1g": (398, 420),
    },
    "ws2": {
        "E2g": (345, 365), "A1g": (410, 430), "2LA": (340, 365),
    },
    "mose2": {
        "E2g": (280, 295), "A1g": (235, 250),
    },
    "wse2": {
        "A1g_E2g": (243, 258), "B2g": (302, 316),
    },
    "mote2": {
        "A1g": (168, 178), "E2g": (230, 242),
    },
    "hbn": {
        "E2g": (1355, 1385),
    },
    "bp": {
        "Ag1": (355, 370), "B2g": (430, 445), "Ag2": (458, 475),
    },
    "mxene": {
        "D":  (1270, 1450), "G":  (1500, 1600),
    },
}

PEAK_COLORS_GRAPHENE = {
    "D": "#ff6b6b", "G": "#69db7c", "D'": "#ffa94d",
    "2D": "#4fc3f7", "D+G": "#cc99ff",
}
PEAK_COLORS_TMD = {
    "E2g": "#4fc3f7", "A1g": "#ff6b6b",
    "2LA": "#ffa94d", "B2g": "#cc99ff",
    "Ag1": "#69db7c", "Ag2": "#4fc3f7", "A1g_E2g": "#ff6b6b",
    "E2g_hbn": "#ffd43b",
}


def _material_key(group: str, material: str) -> str:
    """Map UI selection to internal key."""
    s = (group + " " + material).lower()
    if "graphene" in s or "sp²" in s or "carbon" in s or \
       "rgo" in s or "go" in s or "cnt" in s or "graphite" in s or "mxene" in s:
        if "mxene" in s:
            return "mxene"
        return "graphene"
    if "mos" in s:  return "mos2"
    if "ws₂" in s or "ws2" in s: return "ws2"
    if "mose" in s: return "mose2"
    if "wse" in s:  return "wse2"
    if "mote" in s: return "mote2"
    if "bn" in s:   return "hbn"
    if "phosphor" in s: return "bp"
    return "graphene"


# ══════════════════════════════════════════════════════════
#  PEAK FITTING — GENERIC (for non-graphene materials)
# ══════════════════════════════════════════════════════════
from scipy.signal import find_peaks as sp_find_peaks
from scipy.optimize import curve_fit


def _lorentzian(x, center, amplitude, gamma):
    """Lorentzian: amplitude / (pi*gamma*(1+((x-center)/gamma)^2))"""
    return amplitude / (np.pi * gamma * (1.0 + ((x - center) / gamma) ** 2))


def fit_single_peak(wn: np.ndarray, intensity: np.ndarray,
                    lo: float, hi: float, name: str) -> dict:
    """Fit a single Lorentzian peak in window [lo, hi]."""
    mask = (wn >= lo) & (wn <= hi)
    xd, yd = wn[mask], intensity[mask]
    result = {"name": name, "found": False, "center": np.nan,
              "amplitude": np.nan, "fwhm": np.nan, "area": np.nan,
              "r2": np.nan, "x": xd, "y_fit": np.zeros_like(xd)}
    if len(xd) < 5 or yd.max() < 1:
        return result
    try:
        c0     = xd[np.argmax(yd)]
        g0     = (hi - lo) / 8.0
        a0     = yd.max() * np.pi * g0
        popt, _ = curve_fit(
            _lorentzian, xd, yd,
            p0=[c0, a0, g0],
            bounds=([lo, 0, 0.5], [hi, np.inf, (hi - lo) / 2])
        )
        y_fit   = _lorentzian(xd, *popt)
        ss_res  = np.sum((yd - y_fit) ** 2)
        ss_tot  = np.sum((yd - yd.mean()) ** 2)
        r2      = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        result.update({
            "found":     r2 > 0.65,
            "center":    popt[0],
            "amplitude": popt[1] / (np.pi * popt[2]),   # peak height
            "fwhm":      2.0 * popt[2],
            "area":      popt[1],
            "r2":        r2,
            "y_fit":     y_fit,
        })
    except Exception:
        pass
    return result


def fit_material_peaks(wn: np.ndarray, intensity: np.ndarray,
                       mat_key: str) -> dict:
    """Fit all peaks for a given material."""
    windows = MATERIAL_PEAK_WINDOWS.get(mat_key, MATERIAL_PEAK_WINDOWS["graphene"])
    return {name: fit_single_peak(wn, intensity, lo, hi, name)
            for name, (lo, hi) in windows.items()}


# ══════════════════════════════════════════════════════════
#  MATERIAL-SPECIFIC ANALYSIS
# ══════════════════════════════════════════════════════════

def analyze_graphene(peaks: dict, laser_nm: float) -> dict:
    """Full graphene/sp² carbon analysis."""
    from src.analyzer import analyze as _analyze
    from src.peak_fitter import PeakResult
    # Wrap generic peaks into PeakResult if needed
    result = _analyze(peaks, laser_nm=laser_nm)
    res = {
        "ID/IG (height)":   _fv(result.ID_IG_height),
        "ID/IG (area)":     _fv(result.ID_IG_area),
        "I2D/IG (height)":  _fv(result.I2D_IG_height),
        "I2D/IG (area)":    _fv(result.I2D_IG_area),
        "ID'/IG (height)":  _fv(result.IDp_IG_height),
        "ID/ID' (height)":  _fv(result.ID_IDp_height),
        "L_D (nm)":         _fv(result.L_D_nm, ".2f"),
        "Disorder Stage":   result.disorder_stage,
        "Defect Type":      result.defect_type,
        "Estimated Layers": result.estimated_layers,
        "Graphitization %": _graphitization(result.ID_IG_height),
    }
    return res


def _graphitization(id_ig: float) -> str:
    """
    Degree of graphitization from ID/IG.
    Dg(%) = (1 - ID/IG) * 100  [Tuinstra-Koenig approximation]
    Valid for Stage 1 (ID/IG <= 1). Above 1, result is set to 0%.
    Note: This is an approximation. For rigorous Dg use XRD d002.
    """
    if math.isnan(id_ig):
        return "N/A"
    dg = max(0.0, (1.0 - id_ig) * 100.0)
    return f"{dg:.1f} %"


def analyze_tmd(peaks: dict, mat_key: str, laser_nm: float) -> dict:
    """
    TMD-specific analysis.
    MoS2/MoSe2/WS2/WSe2: Δω = A1g - E2g → layer count
    WS2: 2LA(M)/A1g → defect density
    """
    res = {}
    if mat_key == "mos2":
        e2g = peaks.get("E2g"); a1g = peaks.get("A1g")
        if e2g and a1g and e2g["found"] and a1g["found"]:
            dw = a1g["center"] - e2g["center"]
            res["E²g center (cm⁻¹)"]  = f"{e2g['center']:.1f}"
            res["A1g center (cm⁻¹)"]  = f"{a1g['center']:.1f}"
            res["Δω (A1g − E²g)"]     = f"{dw:.1f} cm⁻¹"
            # Layer count from Δω (Li et al. 2012; Lee et al. 2010)
            if dw < 19:
                layers = "Monolayer (~18.5 cm⁻¹)"
            elif dw < 21.5:
                layers = "Bilayer (~20–21 cm⁻¹)"
            elif dw < 23:
                layers = "Trilayer (~22 cm⁻¹)"
            else:
                layers = "Bulk / thick film (Δω > 23 cm⁻¹)"
            res["Estimated Layers"] = layers
            # A1g FWHM → defect indicator (Chakraborty et al. 2012)
            res["A1g FWHM (cm⁻¹)"]  = f"{a1g['fwhm']:.1f}"
            res["A1g/E²g height"]    = f"{a1g['amplitude']/e2g['amplitude']:.3f}" \
                                        if e2g["amplitude"] > 0 else "N/A"

    elif mat_key == "ws2":
        e2g = peaks.get("E2g"); a1g = peaks.get("A1g")
        la2 = peaks.get("2LA")
        if e2g and a1g and e2g["found"] and a1g["found"]:
            dw = a1g["center"] - e2g["center"]
            res["E²g center (cm⁻¹)"]  = f"{e2g['center']:.1f}"
            res["A1g center (cm⁻¹)"]  = f"{a1g['center']:.1f}"
            res["Δω (A1g − E²g)"]     = f"{dw:.1f} cm⁻¹"
            # WS2 layer from Δω (Zhao et al. 2013)
            if dw < 63:
                layers = "Monolayer (~62 cm⁻¹)"
            elif dw < 66:
                layers = "Bilayer (~64–65 cm⁻¹)"
            else:
                layers = "Few-layer / bulk"
            res["Estimated Layers"] = layers
        if la2 and la2["found"] and a1g and a1g["found"] and a1g["amplitude"] > 0:
            res["2LA/A1g (defect)"] = f"{la2['amplitude']/a1g['amplitude']:.3f}"
            res["Defect note"] = "2LA/A1g > 1 → low defect; < 1 → defective"

    elif mat_key in ("mose2", "wse2"):
        e2g = peaks.get("E2g") or peaks.get("A1g_E2g")
        a1g = peaks.get("A1g") or peaks.get("A1g_E2g")
        if e2g and e2g["found"]:
            res["Main peak center (cm⁻¹)"] = f"{e2g['center']:.1f}"
            res["Main peak FWHM (cm⁻¹)"]  = f"{e2g['fwhm']:.1f}"
        b2g = peaks.get("B2g")
        if b2g and b2g["found"]:
            res["B²g center (cm⁻¹)"] = f"{b2g['center']:.1f}"
            res["B²g note"] = "B²g present → inversion asymmetry / substrate effect"

    elif mat_key == "mote2":
        a1g = peaks.get("A1g"); e2g = peaks.get("E2g")
        if a1g and a1g["found"]:
            res["A1g center (cm⁻¹)"] = f"{a1g['center']:.1f}"
        if e2g and e2g["found"]:
            res["E²g center (cm⁻¹)"] = f"{e2g['center']:.1f}"
        if a1g and e2g and a1g["found"] and e2g["found"]:
            res["Phase note"] = (
                "2H (semiconducting)" if a1g["center"] > 170 else "1T' (metallic)"
            )
    return res


def analyze_hbn(peaks: dict) -> dict:
    e2g = peaks.get("E2g")
    if e2g and e2g["found"]:
        return {
            "E²g center (cm⁻¹)": f"{e2g['center']:.1f}",
            "E²g FWHM (cm⁻¹)":   f"{e2g['fwhm']:.1f}",
            "Note": "FWHM < 10 cm⁻¹ → high crystallinity; > 20 cm⁻¹ → defective",
        }
    return {}


def analyze_bp(peaks: dict) -> dict:
    res = {}
    for pk in ["Ag1", "B2g", "Ag2"]:
        p = peaks.get(pk)
        if p and p["found"]:
            res[f"{pk} center (cm⁻¹)"] = f"{p['center']:.1f}"
            res[f"{pk} FWHM (cm⁻¹)"]  = f"{p['fwhm']:.1f}"
    ag1 = peaks.get("Ag1"); ag2 = peaks.get("Ag2")
    if ag1 and ag2 and ag1["found"] and ag2["found"] and ag1["amplitude"] > 0:
        res["Ag2/Ag1 ratio"] = f"{ag2['amplitude']/ag1['amplitude']:.3f}"
        res["Layer note"] = "Ag2/Ag1 increases with thickness"
    return res


def run_analysis(wn, intensity, mat_key, group, laser_nm, baseline_method, als_lam, als_p):
    """Full pipeline: baseline → fit → analyze. Returns (peaks, analysis_dict, corrected, baseline_arr)."""
    corrected, baseline_arr = correct_baseline(
        wn, intensity, method=baseline_method, lam=als_lam, p=als_p
    )
    if mat_key == "graphene" or mat_key == "mxene":
        peaks = fit_all_peaks(wn, corrected, laser_nm=laser_nm)
        # Convert PeakResult → dict for uniform handling
        peaks_dict = {}
        for k, p in peaks.items():
            peaks_dict[k] = {
                "name": p.name, "found": p.found,
                "center": p.center, "amplitude": p.amplitude,
                "fwhm": p.fwhm, "area": p.area, "r2": p.r_squared,
                "x": p.model_x, "y_fit": p.model_y,
                "is_split_2D": getattr(p, "is_split_2D", False),
            }
        analysis = analyze_graphene(peaks, laser_nm)
    else:
        peaks_dict = fit_material_peaks(wn, corrected, mat_key)
        if mat_key in ("mos2", "ws2", "mose2", "wse2", "mote2"):
            analysis = analyze_tmd(peaks_dict, mat_key, laser_nm)
        elif mat_key == "hbn":
            analysis = analyze_hbn(peaks_dict)
        elif mat_key == "bp":
            analysis = analyze_bp(peaks_dict)
        else:
            analysis = {}
    return peaks_dict, analysis, corrected, baseline_arr


# ══════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════
def _fv(v, fmt=".4f"):
    return format(v, fmt) if not math.isnan(float(v)) else "N/A"


def _make_peak_colors(mat_key):
    if mat_key in ("graphene", "mxene"):
        return PEAK_COLORS_GRAPHENE
    return PEAK_COLORS_TMD


# ══════════════════════════════════════════════════════════
#  EXCEL TEMPLATE GENERATOR
# ══════════════════════════════════════════════════════════
def make_template(n_samples: int) -> bytes:
    """Generate an empty Excel template with n_samples sheets."""
    wb = Workbook()
    wb.remove(wb.active)
    H   = Font(name="Calibri", bold=True, color="FFFFFF", size=11)
    HF  = PatternFill("solid", fgColor="1F4E79")
    C   = Alignment(horizontal="center", vertical="center")
    for i in range(1, n_samples + 1):
        ws = wb.create_sheet(title=f"Sample_{i}")
        ws.sheet_view.showGridLines = False
        ws.column_dimensions["A"].width = 20
        ws.column_dimensions["B"].width = 20
        for col, header in enumerate(["Wavenumber (cm⁻¹)", "Intensity (a.u.)"], start=1):
            cell = ws.cell(row=1, column=col, value=header)
            cell.font = H; cell.fill = HF; cell.alignment = C
        ws.row_dimensions[1].height = 22
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════
#  EXCEL EXPORT
# ══════════════════════════════════════════════════════════
def build_excel_report(samples_results: list, laser_nm: float) -> bytes:
    """
    Build full Excel report.
    samples_results: list of dicts with keys:
      name, group, material, mat_key,
      wn, intensity, baseline, corrected,
      peaks, analysis
    """
    wb  = Workbook()
    wb.remove(wb.active)

    # Styles
    H   = Font(name="Calibri", bold=True, color="FFFFFF", size=11)
    HF  = PatternFill("solid", fgColor="1F4E79")
    SF  = PatternFill("solid", fgColor="2E75B6")
    N   = Font(name="Calibri", size=11)
    B   = Font(name="Calibri", bold=True, size=11, color="1F4E79")
    C   = Alignment(horizontal="center", vertical="center")
    L   = Alignment(horizontal="left",   vertical="center", indent=1)

    def brd():
        s = Side(style="thin", color="BDD7EE")
        return Border(left=s, right=s, top=s, bottom=s)

    # ── Summary sheet ─────────────────────────────────────
    ws_sum = wb.create_sheet("Summary")
    ws_sum.sheet_view.showGridLines = False
    ws_sum.sheet_properties.tabColor = "1F4E79"

    ws_sum.merge_cells("B2:J2")
    ws_sum["B2"] = "Raman Spectroscopy — Multi-Sample Analysis Report"
    ws_sum["B2"].font = Font(name="Calibri", bold=True, color="1F4E79", size=16)
    ws_sum["B2"].alignment = C
    ws_sum.row_dimensions[2].height = 32
    ws_sum.merge_cells("B3:J3")
    ws_sum["B3"] = (
        f"λ = {laser_nm:.0f} nm  |  Samples: {len(samples_results)}  |  "
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )
    ws_sum["B3"].font = Font(name="Calibri", size=10, color="7F7F7F", italic=True)
    ws_sum["B3"].alignment = C

    # Collect all unique parameter keys across all samples
    all_params = []
    seen = set()
    for sr in samples_results:
        for k in sr["analysis"].keys():
            if k not in seen:
                all_params.append(k); seen.add(k)

    # Header row
    row = 5
    headers = ["Sample Name", "Material", "Group"] + all_params
    for ci, h in enumerate(headers):
        cell = ws_sum.cell(row=row, column=2 + ci, value=h)
        cell.font = H; cell.fill = HF; cell.alignment = C; cell.border = brd()
    ws_sum.row_dimensions[row].height = 22

    # Data rows
    for ri, sr in enumerate(samples_results):
        r = row + 1 + ri
        alt = PatternFill("solid", fgColor="D6E4F0" if ri % 2 == 0 else "EBF3FB")
        for ci in range(len(headers)):
            ws_sum.cell(row=r, column=2 + ci).fill = alt
            ws_sum.cell(row=r, column=2 + ci).border = brd()

        ws_sum.cell(row=r, column=2,  value=sr["name"]).font = B
        ws_sum.cell(row=r, column=2).alignment = L
        ws_sum.cell(row=r, column=3,  value=sr["material"]).font = N
        ws_sum.cell(row=r, column=3).alignment = C
        ws_sum.cell(row=r, column=4,  value=sr["group"]).font = N
        ws_sum.cell(row=r, column=4).alignment = C
        for ci, param in enumerate(all_params):
            val = sr["analysis"].get(param, "—")
            c = ws_sum.cell(row=r, column=5 + ci, value=val)
            c.font = N; c.alignment = C
        ws_sum.row_dimensions[r].height = 20

    # Column widths
    col_ws = [22, 22, 28] + [max(14, len(p) + 2) for p in all_params]
    for ci, w in enumerate(col_ws):
        ws_sum.column_dimensions[get_column_letter(2 + ci)].width = w
    ws_sum.column_dimensions["A"].width = 3

    # ── Per-sample sheets ─────────────────────────────────
    for sr in samples_results:
        sname  = sr["name"][:28]  # sheet name limit
        ws     = wb.create_sheet(title=sname)
        ws.sheet_view.showGridLines = False
        ws.column_dimensions["A"].width = 3
        ws.sheet_properties.tabColor = "2E75B6"

        # Header
        ws.merge_cells("B2:G2")
        ws["B2"] = f"Raman Analysis — {sr['name']}"
        ws["B2"].font = Font(name="Calibri", bold=True, color="1F4E79", size=14)
        ws["B2"].alignment = C
        ws.row_dimensions[2].height = 28
        ws.merge_cells("B3:G3")
        ws["B3"] = f"Material: {sr['material']}  |  Group: {sr['group']}  |  λ = {laser_nm:.0f} nm"
        ws["B3"].font = Font(name="Calibri", size=10, color="7F7F7F", italic=True)
        ws["B3"].alignment = C

        # Analysis parameters
        ws.merge_cells("B5:G5")
        ws["B5"] = "Analysis Parameters"
        ws["B5"].font = Font(name="Calibri", bold=True, color="1F4E79", size=13)
        ws.row_dimensions[5].height = 22

        for ci, h in enumerate(["Parameter", "Value"]):
            cell = ws.cell(row=6, column=2 + ci, value=h)
            cell.font = H; cell.fill = HF; cell.alignment = C; cell.border = brd()
        ws.row_dimensions[6].height = 22

        for ri, (param, val) in enumerate(sr["analysis"].items()):
            r   = 7 + ri
            alt = PatternFill("solid", fgColor="D6E4F0" if ri % 2 == 0 else "EBF3FB")
            for ci in range(2):
                ws.cell(row=r, column=2 + ci).fill = alt
                ws.cell(row=r, column=2 + ci).border = brd()
            ws.cell(row=r, column=2, value=param).font = B
            ws.cell(row=r, column=2).alignment = L
            ws.cell(row=r, column=3, value=str(val)).font = N
            ws.cell(row=r, column=3).alignment = C
            ws.row_dimensions[r].height = 20

        # Fitted peaks table
        pk_start = 7 + len(sr["analysis"]) + 2
        ws.merge_cells(f"B{pk_start}:G{pk_start}")
        ws.cell(row=pk_start, column=2, value="Fitted Peaks").font = Font(
            name="Calibri", bold=True, color="1F4E79", size=13)
        ws.row_dimensions[pk_start].height = 22

        pk_head = ["Peak", "Center (cm⁻¹)", "FWHM (cm⁻¹)", "Height (a.u.)", "Area (a.u.)", "R²"]
        for ci, h in enumerate(pk_head):
            cell = ws.cell(row=pk_start + 1, column=2 + ci, value=h)
            cell.font = H; cell.fill = SF; cell.alignment = C; cell.border = brd()
        for ri, (pname, p) in enumerate(sr["peaks"].items()):
            r   = pk_start + 2 + ri
            alt = PatternFill("solid", fgColor="D6E4F0" if ri % 2 == 0 else "EBF3FB")
            for ci in range(6):
                ws.cell(row=r, column=2 + ci).fill = alt
                ws.cell(row=r, column=2 + ci).border = brd()
            ws.cell(row=r, column=2, value=p["name"]).font = B
            ws.cell(row=r, column=2).alignment = L
            if p["found"]:
                for ci, v in enumerate([
                    round(float(p["center"]), 2),
                    round(float(p["fwhm"]),   2),
                    round(float(p["amplitude"]), 1),
                    round(float(p["area"]),    1),
                    round(float(p["r2"]),      4),
                ]):
                    ws.cell(row=r, column=3 + ci, value=v).font = N
                    ws.cell(row=r, column=3 + ci).alignment = C
            else:
                ws.cell(row=r, column=3, value="Not detected").font = N
            ws.row_dimensions[r].height = 20

        # Spectrum data
        data_start = pk_start + 2 + len(sr["peaks"]) + 2
        ws.merge_cells(f"B{data_start}:G{data_start}")
        ws.cell(row=data_start, column=2,
                value="Spectrum Data").font = Font(
            name="Calibri", bold=True, color="1F4E79", size=13)
        data_head = ["Wavenumber (cm⁻¹)", "Raw Intensity",
                     "Baseline", "Corrected", "Normalised (0–1)"]
        for ci, h in enumerate(data_head):
            cell = ws.cell(row=data_start + 1, column=2 + ci, value=h)
            cell.font = H; cell.fill = HF; cell.alignment = C
        wn   = sr["wn"]
        raw  = sr["intensity"]
        base = sr["baseline"]
        corr = sr["corrected"]
        norm = corr / corr.max() if corr.max() > 0 else corr
        for i, (w, r_, b, co, n) in enumerate(zip(wn, raw, base, corr, norm)):
            row_i = data_start + 2 + i
            for ci, v in enumerate([round(float(w), 3), round(float(r_), 3),
                                     round(float(b), 3), round(float(co), 3),
                                     round(float(n), 5)]):
                ws.cell(row=row_i, column=2 + ci, value=v)

        for c, w_ in [(2, 22), (3, 18), (4, 18), (5, 22), (6, 18), (7, 16)]:
            ws.column_dimensions[get_column_letter(c)].width = w_

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════
#  STREAMLIT APP
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Raman Analyzer — 2D Materials",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 Raman Analyzer")
    st.caption("Graphene · TMDs · h-BN · BP · MXene")
    st.divider()

    st.subheader("🔴 Laser Wavelength")
    laser_nm = st.number_input("Wavelength (nm)", min_value=400.0,
                                max_value=1100.0, value=532.0, step=1.0)
    c1, c2, c3, c4 = st.columns(4)
    for col, nm in zip([c1, c2, c3, c4], [488, 532, 633, 785]):
        if col.button(str(nm), use_container_width=True):
            laser_nm = float(nm)

    st.subheader("📉 Baseline Correction")
    baseline_method = st.selectbox("Method", ["als", "linear"])
    als_lam = st.number_input("ALS λ", value=1e5, min_value=1e2,
                               max_value=1e9, format="%.0e",
                               disabled=(baseline_method != "als"))
    als_p   = st.number_input("ALS p", value=0.001, min_value=1e-4,
                               max_value=0.5, format="%.4f",
                               disabled=(baseline_method != "als"))
    st.divider()

    # Template generator
    st.subheader("📥 Step 1 — Download Template")
    n_samples = st.number_input("Number of samples", min_value=1,
                                 max_value=50, value=3, step=1)
    st.download_button(
        label="⬇️  Download Excel Template",
        data=make_template(int(n_samples)),
        file_name=f"raman_template_{int(n_samples)}samples.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    st.subheader("📤 Step 2 — Upload Filled Template")
    uploaded = st.file_uploader(
        "Upload Excel file",
        type=["xlsx"],
        help="Each sheet = one sample (Wavenumber | Intensity)"
    )

# ── Main area ─────────────────────────────────────────────
st.title("Raman Spectrum Analyzer — 2D Materials")
st.caption("Graphene / rGO / TMDs / h-BN / Black Phosphorus / MXene")

if uploaded is None:
    st.info("👈  First download the template, fill it in, then upload it.")
    with st.expander("ℹ️  How to use"):
        st.markdown("""
1. **Set number of samples** in the sidebar → download the Excel template
2. **Fill in** each sheet with your Raman data (two columns: Wavenumber, Intensity)
3. **Upload** the filled Excel file
4. **Configure** each sample below (material type + sample name)
5. Click **▶ RUN ANALYSIS** → get interactive plots + download Excel report
        """)
    st.stop()

# Read sheets
try:
    xl      = pd.ExcelFile(uploaded)
    sheets  = xl.sheet_names
except Exception as e:
    st.error(f"Cannot read Excel file: {e}")
    st.stop()

st.subheader(f"📋 Step 3 — Configure {len(sheets)} sample(s)")

group_options = list(MATERIAL_GROUPS.keys())
sample_configs = []

with st.form("config_form"):
    for i, sheet in enumerate(sheets):
        st.markdown(f"**Sheet: `{sheet}`**")
        col_name, col_group, col_mat = st.columns([2, 3, 3])

        sample_name = col_name.text_input(
            "Sample name", value=sheet, key=f"name_{i}"
        )
        group = col_group.selectbox(
            "Material group", group_options, key=f"group_{i}"
        )
        materials = MATERIAL_GROUPS[group]
        material  = col_mat.selectbox(
            "Material", materials, key=f"mat_{i}"
        )
        sample_configs.append({
            "sheet": sheet, "name": sample_name,
            "group": group, "material": material
        })
        if i < len(sheets) - 1:
            st.divider()

    submitted = st.form_submit_button(
        "▶  RUN ANALYSIS", type="primary", use_container_width=True
    )

if not submitted:
    st.stop()

# ── Run analysis ──────────────────────────────────────────
st.subheader("🔄 Processing...")
progress = st.progress(0)
samples_results = []
errors = []

for i, cfg in enumerate(sample_configs):
    try:
        df = xl.parse(cfg["sheet"], header=0)
        df.columns = ["wavenumber", "intensity"] + list(df.columns[2:])
        df = df.dropna(subset=["wavenumber", "intensity"])
        wn        = df["wavenumber"].values.astype(float)
        intensity = df["intensity"].values.astype(float)
        sort_idx  = np.argsort(wn)
        wn, intensity = wn[sort_idx], intensity[sort_idx]

        mat_key = _material_key(cfg["group"], cfg["material"])

        peaks, analysis, corrected, baseline_arr = run_analysis(
            wn, intensity, mat_key, cfg["group"],
            laser_nm, baseline_method, als_lam, als_p
        )
        samples_results.append({
            **cfg, "mat_key": mat_key,
            "wn": wn, "intensity": intensity,
            "baseline": baseline_arr, "corrected": corrected,
            "peaks": peaks, "analysis": analysis,
        })
    except Exception as e:
        errors.append(f"{cfg['sheet']}: {e}")

    progress.progress((i + 1) / len(sample_configs))

if errors:
    for err in errors:
        st.warning(f"⚠️  {err}")

if not samples_results:
    st.error("No samples could be processed. Check your data format.")
    st.stop()

st.success(f"✅  {len(samples_results)} sample(s) analysed successfully!")

# ── Results tabs ──────────────────────────────────────────
if len(samples_results) == 1:
    sample_tabs = ["All Spectra"]
else:
    sample_tabs = ["All Spectra"] + [sr["name"] for sr in samples_results]

tab_objects = st.tabs(
    ["📈 All Spectra", "🔍 Peak Fits", "📋 Results Table"] +
    [f"📄 {sr['name']}" for sr in samples_results]
)

# Tab 0: All spectra overlay
with tab_objects[0]:
    st.subheader("All Spectra — Baseline Corrected")
    fig_all = go.Figure()
    palette = ["#4fc3f7","#69db7c","#ff6b6b","#ffa94d",
               "#cc99ff","#ffd43b","#a9e34b","#f783ac"]
    for idx, sr in enumerate(samples_results):
        color = palette[idx % len(palette)]
        fig_all.add_trace(go.Scatter(
            x=sr["wn"], y=sr["corrected"],
            name=sr["name"],
            line=dict(color=color, width=1.4),
        ))
    fig_all.update_layout(
        template="plotly_dark", height=450,
        xaxis_title="Raman Shift (cm⁻¹)",
        yaxis_title="Intensity (a.u.)",
        legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(fig_all, use_container_width=True)

# Tab 1: Peak fits
with tab_objects[1]:
    st.subheader("Peak Fits")
    sel_name = st.selectbox("Select sample", [sr["name"] for sr in samples_results])
    sr_sel   = next(s for s in samples_results if s["name"] == sel_name)
    colors   = _make_peak_colors(sr_sel["mat_key"])
    found_pk = [(k, p) for k, p in sr_sel["peaks"].items() if p["found"] and len(p["x"]) > 0]
    if not found_pk:
        st.warning("No peaks detected for this sample.")
    else:
        n_cols_pk = min(len(found_pk), 3)
        pk_cols   = st.columns(n_cols_pk)
        for idx, (key, p) in enumerate(found_pk):
            color = colors.get(key, "#cccccc")
            mask  = (sr_sel["wn"] >= p["x"][0]) & (sr_sel["wn"] <= p["x"][-1])
            xd    = sr_sel["wn"][mask]
            yd    = sr_sel["corrected"][mask]
            fig_pk = go.Figure()
            fig_pk.add_trace(go.Scatter(
                x=xd, y=yd, mode="markers",
                marker=dict(size=4, color="#cdd6f4", opacity=0.6), name="Data"
            ))
            fig_pk.add_trace(go.Scatter(
                x=p["x"], y=p["y_fit"],
                line=dict(color=color, width=2.5),
                fill="tozeroy",
                fillcolor=f"rgba{tuple(int(color.lstrip('#')[j:j+2],16) for j in (0,2,4))+(0.2,)}",
                name="Fit"
            ))
            split_note = " [dual-L]" if p.get("is_split_2D") else ""
            fig_pk.update_layout(
                title=dict(
                    text=f"<b style='color:{color}'>{p['name']}{split_note}</b><br>"
                         f"{p['center']:.1f} cm⁻¹ | FWHM={p['fwhm']:.1f} | R²={p['r2']:.3f}",
                    font=dict(size=11)
                ),
                template="plotly_dark", height=280, showlegend=False,
                margin=dict(t=80, b=40, l=40, r=10),
                xaxis_title="Raman Shift (cm⁻¹)",
                yaxis_title="Intensity",
            )
            pk_cols[idx % n_cols_pk].plotly_chart(fig_pk, use_container_width=True)

# Tab 2: Results table
with tab_objects[2]:
    st.subheader("Results Summary")
    all_params_keys = []
    seen_k = set()
    for sr in samples_results:
        for k in sr["analysis"]:
            if k not in seen_k:
                all_params_keys.append(k); seen_k.add(k)

    table_rows = []
    for sr in samples_results:
        row = {"Sample": sr["name"], "Material": sr["material"]}
        for k in all_params_keys:
            row[k] = sr["analysis"].get(k, "—")
        table_rows.append(row)

    df_results = pd.DataFrame(table_rows)
    st.dataframe(df_results, hide_index=True, use_container_width=True)

# Per-sample detail tabs
for ti, sr in enumerate(samples_results):
    with tab_objects[3 + ti]:
        st.subheader(f"{sr['name']} — {sr['material']}")

        # Spectrum plot
        fig_s = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Raw + Baseline", "Corrected"),
            vertical_spacing=0.12
        )
        fig_s.add_trace(go.Scatter(x=sr["wn"], y=sr["intensity"],
                                    name="Raw", line=dict(color="#4fc3f7", width=1.2)),
                         row=1, col=1)
        fig_s.add_trace(go.Scatter(x=sr["wn"], y=sr["baseline"],
                                    name="Baseline",
                                    line=dict(color="#ff6b6b", width=1.5, dash="dash")),
                         row=1, col=1)
        fig_s.add_trace(go.Scatter(x=sr["wn"], y=sr["corrected"],
                                    name="Corrected",
                                    line=dict(color="#69db7c", width=1.4),
                                    fill="tozeroy",
                                    fillcolor="rgba(105,219,124,0.07)"),
                         row=2, col=1)
        colors_s = _make_peak_colors(sr["mat_key"])
        for key, p in sr["peaks"].items():
            if p["found"]:
                col = colors_s.get(key, "gray")
                fig_s.add_vline(x=p["center"], line_width=0.8,
                                 line_dash="dot", line_color=col, row=2, col=1)
                fig_s.add_annotation(
                    x=p["center"], y=sr["corrected"].max() * 0.9,
                    text=f"<b>{p['name']}</b>",
                    font=dict(color=col, size=10),
                    showarrow=False, row=2, col=1
                )
        fig_s.update_layout(
            height=550, template="plotly_dark",
            title=dict(text=f"{sr['name']} | λ={laser_nm:.0f} nm",
                        font=dict(size=13)),
            legend=dict(orientation="h", y=1.06),
        )
        fig_s.update_xaxes(title_text="Raman Shift (cm⁻¹)", row=2, col=1)
        fig_s.update_yaxes(title_text="Intensity (a.u.)")
        st.plotly_chart(fig_s, use_container_width=True)

        # Analysis params
        st.subheader("Analysis Parameters")
        param_rows = [{"Parameter": k, "Value": v}
                      for k, v in sr["analysis"].items()]
        st.dataframe(pd.DataFrame(param_rows), hide_index=True,
                     use_container_width=True)

# ── Export Excel ──────────────────────────────────────────
st.divider()
st.subheader("📊 Download Full Excel Report")
excel_bytes = build_excel_report(samples_results, laser_nm)
st.download_button(
    label="⬇️  Download Excel Report",
    data=excel_bytes,
    file_name=f"raman_report_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True,
    type="primary",
)
