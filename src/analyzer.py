"""
Quantitative Raman analysis for graphene/sp² carbon materials.

References:
  - Ferrari & Robertson (2001) Phys. Rev. B 64, 075414   — disorder stages
  - Lucchese et al. (2010) Carbon 48, 1592                — L_D formula
  - Ferrari & Basko (2013) Nature Nanotechnology 8, 235   — peak conventions
  - Eckmann et al. (2012) Nano Letters 12, 3925           — defect type via ID/ID'
  - Ferrari et al. (2006) Phys. Rev. Lett. 97, 187401     — layer count
  - Mak et al. (2010); Ni et al. (2008)                   — laser-dep. thresholds

Fixes applied:
  1. L_D: removed dead lambda_m variable; units clarified (λ in nm).
  2. Disorder stage: threshold raised from FWHM>30 to FWHM>80 or FWHM>50+ID/IG<1.2.
  3. Layer estimation: thresholds are now laser-wavelength dependent.
"""

import numpy as np
from dataclasses import dataclass
from .peak_fitter import PeakResult


@dataclass
class RamanAnalysis:
    ID_IG_height:     float = np.nan
    I2D_IG_height:    float = np.nan
    IDp_IG_height:    float = np.nan
    ID_IDp_height:    float = np.nan
    ID_IG_area:       float = np.nan
    I2D_IG_area:      float = np.nan
    L_D_nm:           float = np.nan
    disorder_stage:   str   = "N/A"
    defect_type:      str   = "N/A"
    estimated_layers: str   = "N/A"
    G_found:          bool  = False
    D_found:          bool  = False
    twoD_found:       bool  = False


def _monolayer_threshold(laser_nm: float) -> float:
    """
    Laser-wavelength-dependent I2D/IG threshold for monolayer identification.

    Values from literature:
      ≤514 nm : > 2.5  (488 nm, approximate)
      532 nm  : > 2.0  Ferrari et al. (2006)
      633 nm  : > 1.5  Ferrari & Basko (2013)
      785 nm+ : > 0.8  Mak et al. (2010); Ni et al. (2008)
    """
    if laser_nm <= 514:
        return 2.5
    elif laser_nm <= 568:
        return 2.0
    elif laser_nm <= 660:
        return 1.5
    else:
        return 0.8


def analyze(peaks: dict[str, PeakResult], laser_nm: float = 532.0) -> RamanAnalysis:
    result = RamanAnalysis()
    D    = peaks.get("D")
    G    = peaks.get("G")
    twoD = peaks.get("2D")
    Dp   = peaks.get("D_prime")

    result.G_found    = G    is not None and G.found
    result.D_found    = D    is not None and D.found
    result.twoD_found = twoD is not None and twoD.found

    if not result.G_found:
        return result

    # ── Intensity ratios ──────────────────────────────
    if result.D_found:
        result.ID_IG_height = D.amplitude / G.amplitude if G.amplitude > 0 else np.nan
        result.ID_IG_area   = D.area      / G.area      if G.area      > 0 else np.nan

    if result.twoD_found:
        result.I2D_IG_height = twoD.amplitude / G.amplitude if G.amplitude > 0 else np.nan
        result.I2D_IG_area   = twoD.area      / G.area      if G.area      > 0 else np.nan

    if Dp is not None and Dp.found:
        result.IDp_IG_height = Dp.amplitude / G.amplitude if G.amplitude > 0 else np.nan
        if result.D_found and Dp.amplitude > 0:
            result.ID_IDp_height = D.amplitude / Dp.amplitude

    # ── L_D (Lucchese et al. 2010) ────────────────────
    # L_D² (nm²) = (1.8×10⁻⁹) × λ_L⁴ (nm⁴) × (ID/IG)⁻¹
    # λ_L must be in nm; constant 1.8e-9 is calibrated for nm units.
    if not np.isnan(result.ID_IG_height) and result.ID_IG_height > 0:
        result.L_D_nm = np.sqrt((1.8e-9 * laser_nm**4) / result.ID_IG_height)

    # ── Disorder stage (Ferrari & Robertson 2001) ─────
    # Stage 1: ID/IG increases, FWHM(G) < 50 cm⁻¹
    # Stage 2: ID/IG decreases, FWHM(G) > 50–80 cm⁻¹
    # FIX: old threshold FWHM>30 was too low (graphene already ~15–25 cm⁻¹)
    if result.G_found and result.D_found:
        fwhm_g = G.fwhm
        id_ig  = result.ID_IG_height
        if not np.isnan(id_ig):
            if fwhm_g > 80:
                result.disorder_stage = "Stage 2 (amorphous carbon, FWHM(G) > 80 cm⁻¹)"
            elif fwhm_g > 50 and id_ig < 1.2:
                result.disorder_stage = "Stage 2 (nanocrystalline→amorphous transition)"
            else:
                result.disorder_stage = "Stage 1 (nanocrystalline graphite)"
        else:
            result.disorder_stage = "Stage 1 (nanocrystalline graphite)"

    # ── Defect type (Eckmann et al. 2012) ────────────
    if not np.isnan(result.ID_IDp_height):
        r = result.ID_IDp_height
        if r >= 10:
            result.defect_type = f"sp³-type defects (ID/ID' = {r:.1f}, expected ~13)"
        elif 5 <= r < 10:
            result.defect_type = f"Vacancy-type defects (ID/ID' = {r:.1f}, expected ~7)"
        else:
            result.defect_type = f"Grain boundary / edge defects (ID/ID' = {r:.1f}, expected ~3.5)"

    # ── Layer count — laser-wavelength corrected ──────
    # FIX: threshold was fixed at >2.0 regardless of laser
    if not np.isnan(result.I2D_IG_height):
        r   = result.I2D_IG_height
        thr = _monolayer_threshold(laser_nm)
        if r > thr:
            result.estimated_layers = f"Monolayer (I2D/IG={r:.2f} > {thr:.1f} @ {laser_nm:.0f} nm)"
        elif r > thr * 0.5:
            result.estimated_layers = f"Bilayer (I2D/IG={r:.2f}, {thr*0.5:.1f}–{thr:.1f} @ {laser_nm:.0f} nm)"
        elif r > thr * 0.25:
            result.estimated_layers = f"Few-layer 3–5 (I2D/IG={r:.2f} @ {laser_nm:.0f} nm)"
        else:
            result.estimated_layers = f"Multilayer/bulk graphite (I2D/IG={r:.2f} < {thr*0.25:.1f} @ {laser_nm:.0f} nm)"

    return result


def format_report(filename: str,
                  peaks:    dict[str, PeakResult],
                  analysis: RamanAnalysis,
                  laser_nm: float) -> str:
    sep = "═" * 64
    def _fv(v, fmt=".4f"):
        return format(v, fmt) if not np.isnan(v) else "N/A"

    lines = [
        sep, "  RAMAN ANALYSIS REPORT",
        f"  File    : {filename}",
        f"  Laser   : {laser_nm} nm", sep, "",
        "  FITTED PEAKS",
        f"  {'Peak':<8} {'Center':>10} {'FWHM':>10} {'Height':>12} {'Area':>12} {'R²':>8}",
        f"  {'-'*8} {'-'*10} {'-'*10} {'-'*12} {'-'*12} {'-'*8}",
    ]
    for key in ["D", "G", "D_prime", "2D", "DG"]:
        p = peaks.get(key)
        if p:
            if p.found:
                note   = " [dual-Lorentzian]" if getattr(p, "is_split_2D", False) else ""
                status = (f"{p.center:10.1f} {p.fwhm:10.1f} "
                          f"{p.amplitude:12.1f} {p.area:12.1f} {p.r_squared:8.3f}{note}")
            else:
                status = "       Not detected"
            lines.append(f"  {p.name:<8} {status}")

    lines += [
        "", "  INTENSITY RATIOS",
        f"  ID/IG    (height) : {_fv(analysis.ID_IG_height)}",
        f"  ID/IG    (area)   : {_fv(analysis.ID_IG_area)}",
        f"  I2D/IG   (height) : {_fv(analysis.I2D_IG_height)}",
        f"  I2D/IG   (area)   : {_fv(analysis.I2D_IG_area)}",
        f"  ID'/IG   (height) : {_fv(analysis.IDp_IG_height)}",
        f"  ID/ID'   (height) : {_fv(analysis.ID_IDp_height)}",
        "", "  STRUCTURAL ANALYSIS",
        f"  L_D (defect spacing) : {_fv(analysis.L_D_nm, '.2f')} nm",
        f"  Disorder stage       : {analysis.disorder_stage}",
        f"  Defect type          : {analysis.defect_type}",
        f"  Estimated layers     : {analysis.estimated_layers}",
        "",
        f"  NOTE: Thresholds calibrated for λ = {laser_nm:.0f} nm excitation.",
        sep,
    ]
    return "\n".join(lines)
