"""
Quantitative Raman analysis for graphene/graphene-like materials.
Calculates: ID/IG, I2D/IG, L_D, defect type, layer number estimate.

References:
  - Ferrari & Robertson (2001) Phys. Rev. B 64, 075414  — disorder stages
  - Lucchese et al. (2010) Carbon 48, 1592               — L_D formula
  - Ferrari & Basko (2013) Nature Nanotechnology 8, 235  — peak conventions
  - Eckmann et al. (2012) Nano Letters 12, 3925           — defect type via ID/ID'
"""

import numpy as np
from dataclasses import dataclass
from .peak_fitter import PeakResult


@dataclass
class RamanAnalysis:
    # Intensity ratios (height-based)
    ID_IG_height:    float = np.nan
    I2D_IG_height:   float = np.nan
    IDp_IG_height:   float = np.nan    # ID'/IG
    ID_IDp_height:   float = np.nan    # ID/ID' → defect type

    # Intensity ratios (area-based)
    ID_IG_area:      float = np.nan
    I2D_IG_area:     float = np.nan

    # Structural parameters
    L_D_nm:          float = np.nan    # defect inter-distance (nm)
    disorder_stage:  str   = "N/A"     # Stage 1 or Stage 2
    defect_type:     str   = "N/A"     # sp3 / vacancy / grain boundary
    estimated_layers:str   = "N/A"     # monolayer / bilayer / few-layer

    # Quality flags
    G_found:  bool = False
    D_found:  bool = False
    twoD_found: bool = False


def analyze(peaks: dict[str, PeakResult], laser_nm: float = 532.0) -> RamanAnalysis:
    """
    Perform full quantitative analysis from fitted peaks.
    """
    result = RamanAnalysis()

    D   = peaks.get("D")
    G   = peaks.get("G")
    twoD = peaks.get("2D")
    Dp  = peaks.get("D_prime")

    result.G_found    = G   is not None and G.found
    result.D_found    = D   is not None and D.found
    result.twoD_found = twoD is not None and twoD.found

    if not result.G_found:
        return result

    # ── Intensity ratios ──────────────────────────
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

    # ── Defect inter-distance L_D ─────────────────
    # Lucchese et al. (2010): L_D² = (1.8×10⁻⁹ × λ_L⁴) × (ID/IG)⁻¹
    if not np.isnan(result.ID_IG_height) and result.ID_IG_height > 0:
        lambda_m = laser_nm * 1e-9
        lambda_nm4 = laser_nm**4
        result.L_D_nm = np.sqrt((1.8e-9 * lambda_nm4) / result.ID_IG_height)

    # ── Disorder stage ────────────────────────────
    # Ferrari & Robertson (2001):
    # Stage 1 (graphite → nanocrystalline graphite): ID/IG increases
    # Stage 2 (nanocrystalline → amorphous): ID/IG decreases, D broadens
    # Practical threshold: FWHM(G) > 30 cm⁻¹ and ID/IG < 1 → Stage 1
    if result.G_found and result.D_found:
        if G.fwhm > 30 and not np.isnan(result.ID_IG_height):
            result.disorder_stage = "Stage 2 (amorphous-like)"
        else:
            result.disorder_stage = "Stage 1 (nanocrystalline)"

    # ── Defect type from ID/ID' ───────────────────
    # Eckmann et al. (2012):
    #   ID/ID' ~13 → sp3 defects
    #   ID/ID' ~7  → vacancy-type defects
    #   ID/ID' ~3.5 → grain boundary / edge defects
    if not np.isnan(result.ID_IDp_height):
        r = result.ID_IDp_height
        if r >= 10:
            result.defect_type = f"sp3-type defects (ID/ID'={r:.1f}, expected ~13)"
        elif 5 <= r < 10:
            result.defect_type = f"Vacancy-type defects (ID/ID'={r:.1f}, expected ~7)"
        else:
            result.defect_type = f"Grain boundary / edge defects (ID/ID'={r:.1f}, expected ~3.5)"

    # ── Layer number estimate from I2D/IG ─────────
    # Ferrari et al. (2006): monolayer I2D/IG > 2
    if not np.isnan(result.I2D_IG_height):
        r = result.I2D_IG_height
        if r > 2.0:
            result.estimated_layers = "Monolayer (I2D/IG > 2)"
        elif 1.0 <= r <= 2.0:
            result.estimated_layers = "Bilayer (1 ≤ I2D/IG ≤ 2)"
        elif 0.5 <= r < 1.0:
            result.estimated_layers = "Few-layer (I2D/IG ~ 0.5–1)"
        else:
            result.estimated_layers = "Multilayer / bulk graphite (I2D/IG < 0.5)"

    return result


def format_report(filename: str,
                  peaks: dict[str, PeakResult],
                  analysis: RamanAnalysis,
                  laser_nm: float) -> str:
    """Generate human-readable text report."""
    sep = "═" * 60
    lines = [
        sep,
        f"  RAMAN ANALYSIS REPORT",
        f"  File   : {filename}",
        f"  Laser  : {laser_nm} nm",
        sep,
        "",
        "  FITTED PEAKS",
        f"  {'Peak':<8} {'Center':>10} {'FWHM':>10} {'Height':>12} {'Area':>12} {'R²':>8}",
        f"  {'-'*8} {'-'*10} {'-'*10} {'-'*12} {'-'*12} {'-'*8}",
    ]
    for key in ["D", "G", "D_prime", "2D", "DG"]:
        p = peaks.get(key)
        if p:
            status = f"{p.center:10.1f} {p.fwhm:10.1f} {p.amplitude:12.1f} {p.area:12.1f} {p.r_squared:8.3f}" if p.found else "       Not detected"
            lines.append(f"  {p.name:<8} {status}")

    lines += [
        "",
        "  INTENSITY RATIOS",
        f"  ID/IG   (height) : {analysis.ID_IG_height:.4f}"   if not np.isnan(analysis.ID_IG_height)   else "  ID/IG   : N/A",
        f"  ID/IG   (area)   : {analysis.ID_IG_area:.4f}"     if not np.isnan(analysis.ID_IG_area)     else "  ID/IG (area): N/A",
        f"  I2D/IG  (height) : {analysis.I2D_IG_height:.4f}"  if not np.isnan(analysis.I2D_IG_height)  else "  I2D/IG : N/A",
        f"  I2D/IG  (area)   : {analysis.I2D_IG_area:.4f}"    if not np.isnan(analysis.I2D_IG_area)    else "  I2D/IG (area): N/A",
        f"  ID'/IG  (height) : {analysis.IDp_IG_height:.4f}"  if not np.isnan(analysis.IDp_IG_height)  else "  ID'/IG : N/A",
        f"  ID/ID'  (height) : {analysis.ID_IDp_height:.4f}"  if not np.isnan(analysis.ID_IDp_height)  else "  ID/ID' : N/A",
        "",
        "  STRUCTURAL ANALYSIS",
        f"  Defect inter-distance L_D : {analysis.L_D_nm:.2f} nm" if not np.isnan(analysis.L_D_nm) else "  L_D : N/A",
        f"  Disorder stage            : {analysis.disorder_stage}",
        f"  Defect type               : {analysis.defect_type}",
        f"  Estimated layers          : {analysis.estimated_layers}",
        "",
        sep,
    ]
    return "\n".join(lines)
