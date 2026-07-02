"""
Quantitative Raman analysis for graphene/sp² carbon materials.

References:
  - Ferrari & Robertson (2001) Phys. Rev. B 64, 075414   — disorder stages
  - Cançado et al. (2011) Nano Lett. 11, 3190            — L_D formula (Stage 1 only)
  - Ferrari & Basko (2013) Nature Nanotechnology 8, 235   — peak conventions
  - Eckmann et al. (2012) Nano Letters 12, 3925           — defect type via ID/ID'
  - Ferrari et al. (2006) Phys. Rev. Lett. 97, 187401     — layer count
  - Mak et al. (2010); Ni et al. (2008)                   — laser-dep. thresholds
  - Kim et al. (2012) ACS Nano 6, 8203                    — B-doping fingerprint
  - Lee et al. (2021) Carbon 183, 814–822                 — D* band / C–O proxy

Change log:
  v2.0  dispersion fix — eV-based window shifts in peak_fitter
  v2.1  adaptive G fit + G+D' deconvolution
  v2.2  L_D: correct source (Cançado 2011); suppress L_D in Stage 2
        layer count: FWHM(2D) plausibility guard
        remove Graphitization % (no literature basis)
  v2.4  Feature #1: D* band → I_D*/I_G ratio + rGO C/O proxy note
        Feature #2: B-doping fingerprint flag [Kim 2012]
        Feature #3: fitting uncertainty (center_stderr, fwhm_stderr)
                    propagated to report
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from .peak_fitter import PeakResult


@dataclass
class RamanAnalysis:
    ID_IG_height:        float = np.nan
    I2D_IG_height:       float = np.nan
    IDp_IG_height:       float = np.nan
    ID_IDp_height:       float = np.nan
    ID_IG_area:          float = np.nan
    I2D_IG_area:         float = np.nan
    # ── v2.4 Feature #1: D* band ──────────────────────────
    IDstar_IG_height:    float = np.nan   # I_D*/I_G (height)
    dstar_co_note:       str   = ""       # rGO oxidation flag
    # ── v2.4 Feature #2: B-doping fingerprint ─────────────
    boron_doping_flag:   bool  = False
    boron_doping_note:   str   = ""
    # ─────────────────────────────────────────────────────
    L_D_nm:              float = np.nan
    L_D_note:            str   = ""
    disorder_stage:      str   = "N/A"
    defect_type:         str   = "N/A"
    estimated_layers:    str   = "N/A"
    twoD_fwhm_warning:   bool  = False
    G_found:             bool  = False
    D_found:             bool  = False
    twoD_found:          bool  = False
    # NOTE: graphitization_pct removed in v2.2 — (1−ID/IG)×100 has no
    # literature basis and is non-monotonic across the Stage 1/2 boundary.


# ── Monolayer I2D/IG threshold (laser-dependent) ──────────
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


# ── B-doping fingerprint (v2.4, Feature #2) ───────────────
# Criteria from Kim et al. (2012) ACS Nano 6, 8203:
#   1. G band constant (not blue-shifted; B-doping is hole-doping but
#      G shift from 1582 cm⁻¹ is typically < 5 cm⁻¹ — unlike N-doping
#      which blue-shifts G by 15–20 cm⁻¹)
#   2. I_D/I_D' ≈ 7  (sp³ boron substitutional defects, not vacancies)
#   3. I_D/I_G > 3   (heavily disordered, consistent with B-doped graphene)
#
# All three criteria must be met simultaneously to raise the flag.
# Tolerances:
#   G constant  : 1577–1587 cm⁻¹  (±5 cm⁻¹ from 1582)
#   I_D/I_D'    : 5–9             (centred on ~7, Eckmann sp³ branch)
#   I_D/I_G     : > 3
_BORON_G_CENTER_MIN  = 1577.0
_BORON_G_CENTER_MAX  = 1587.0
_BORON_ID_IDp_MIN    = 5.0
_BORON_ID_IDp_MAX    = 9.0
_BORON_ID_IG_MIN     = 3.0


def _check_boron_doping(
    G:     Optional[PeakResult],
    D:     Optional[PeakResult],
    Dp:    Optional[PeakResult],
    id_ig: float,
    id_idp: float,
) -> tuple[bool, str]:
    """
    Return (flag, note) for B-doping fingerprint check.
    All three Kim 2012 criteria must be satisfied.
    """
    if G is None or not G.found:
        return False, ""
    if D is None or not D.found:
        return False, ""
    if Dp is None or not Dp.found:
        return False, ""
    if np.isnan(id_ig) or np.isnan(id_idp):
        return False, ""

    g_ok  = _BORON_G_CENTER_MIN <= G.center <= _BORON_G_CENTER_MAX
    idp_ok = _BORON_ID_IDp_MIN <= id_idp <= _BORON_ID_IDp_MAX
    idig_ok = id_ig >= _BORON_ID_IG_MIN

    if g_ok and idp_ok and idig_ok:
        note = (
            f"Boron doping fingerprint detected [Kim et al. 2012, ACS Nano 6, 8203]: "
            f"G constant at {G.center:.1f} cm⁻¹ (expected ~1582 cm⁻¹, no N-doping blue-shift); "
            f"I_D/I_D' = {id_idp:.1f} (expected ~7, sp³ substitutional B); "
            f"I_D/I_G = {id_ig:.2f} (> 3, heavily defective). "
            f"Criteria: G ∈ [{_BORON_G_CENTER_MIN}–{_BORON_G_CENTER_MAX}] cm⁻¹, "
            f"I_D/I_D' ∈ [{_BORON_ID_IDp_MIN}–{_BORON_ID_IDp_MAX}], "
            f"I_D/I_G > {_BORON_ID_IG_MIN}."
        )
        return True, note

    # Partial match: report which criteria failed (useful for debugging)
    failed = []
    if not g_ok:
        failed.append(
            f"G at {G.center:.1f} cm⁻¹ outside [{_BORON_G_CENTER_MIN}–{_BORON_G_CENTER_MAX}] cm⁻¹"
        )
    if not idp_ok:
        failed.append(f"I_D/I_D' = {id_idp:.1f} outside [{_BORON_ID_IDp_MIN}–{_BORON_ID_IDp_MAX}]")
    if not idig_ok:
        failed.append(f"I_D/I_G = {id_ig:.2f} < {_BORON_ID_IG_MIN}")
    return False, ""


# ── Main analysis function ────────────────────────────────
def analyze(peaks: dict[str, PeakResult], laser_nm: float = 532.0) -> RamanAnalysis:
    result = RamanAnalysis()
    D     = peaks.get("D")
    G     = peaks.get("G")
    twoD  = peaks.get("2D")
    Dp    = peaks.get("D_prime")
    Dstar = peaks.get("D_star")   # v2.4 Feature #1

    result.G_found    = G    is not None and G.found
    result.D_found    = D    is not None and D.found
    result.twoD_found = twoD is not None and twoD.found

    if not result.G_found:
        return result

    # ── Intensity ratios ──────────────────────────────────
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

    # ── v2.4 Feature #1: D* band ratio + C/O proxy ────────
    # I_D*/I_G > 0.15 indicates significant residual oxidation in rGO/GO.
    # Lee et al. (2021) Carbon 183, 814–822:
    #   D* (~1150 cm⁻¹) attributed to sp² C=C stretching at the
    #   interface between oxidised and reduced domains; proportional
    #   to the degree of oxidation / C/O ratio.
    if Dstar is not None and Dstar.found and G.amplitude > 0:
        result.IDstar_IG_height = Dstar.amplitude / G.amplitude
        if result.IDstar_IG_height > 0.15:
            result.dstar_co_note = (
                f"High D* intensity (I_D*/I_G = {result.IDstar_IG_height:.3f} > 0.15): "
                "significant residual C–O groups / partial oxidation suggested. "
                "Consistent with GO or incompletely reduced rGO "
                "[Lee et al. 2021, Carbon 183, 814–822]."
            )
        else:
            result.dstar_co_note = (
                f"I_D*/I_G = {result.IDstar_IG_height:.3f} ≤ 0.15: "
                "low residual oxidation [Lee et al. 2021]."
            )

    # ── Disorder stage (Ferrari & Robertson 2001) ─────────
    stage2 = False
    if result.G_found and result.D_found:
        fwhm_g = G.fwhm
        id_ig  = result.ID_IG_height
        if not np.isnan(id_ig):
            if fwhm_g > 80:
                result.disorder_stage = "Stage 2 (amorphous carbon, FWHM(G) > 80 cm⁻¹)"
                stage2 = True
            elif fwhm_g > 50 and id_ig < 1.2:
                result.disorder_stage = "Stage 2 (nanocrystalline→amorphous transition)"
                stage2 = True
            else:
                result.disorder_stage = "Stage 1 (nanocrystalline graphite)"
        else:
            result.disorder_stage = "Stage 1 (nanocrystalline graphite)"

    # ── L_D (Cançado et al. Nano Lett. 11, 3190, 2011) ─────
    if not np.isnan(result.ID_IG_height) and result.ID_IG_height > 0:
        if stage2:
            result.L_D_nm   = np.nan
            result.L_D_note = (
                "L_D suppressed: sample is Stage 2 (amorphous/nanocrystalline — "
                "Canc\u0327ado 2011 formula is only valid in Stage 1, L_D ≳ 10 nm)"
            )
        else:
            result.L_D_nm   = np.sqrt(
                (1.8e-9 * laser_nm**4) / result.ID_IG_height
            )
            result.L_D_note = (
                f"Canc\u0327ado et al. (2011); λ={laser_nm:.0f} nm; "
                "valid for Stage 1 (L_D ≳ 10 nm); ±14\u2009% uncertainty in L_D"
            )

    # ── Defect type (Eckmann et al. 2012) ────────────────
    if not np.isnan(result.ID_IDp_height):
        r = result.ID_IDp_height
        if r >= 10:
            result.defect_type = f"sp³-type defects (ID/ID' = {r:.1f}, expected ~13)"
        elif 5 <= r < 10:
            result.defect_type = f"Vacancy-type defects (ID/ID' = {r:.1f}, expected ~7)"
        else:
            result.defect_type = f"Grain boundary / edge defects (ID/ID' = {r:.1f}, expected ~3.5)"

    # ── v2.4 Feature #2: B-doping fingerprint ────────────
    result.boron_doping_flag, result.boron_doping_note = _check_boron_doping(
        G, D, Dp,
        id_ig=result.ID_IG_height,
        id_idp=result.ID_IDp_height,
    )

    # ── Layer count (laser-wavelength corrected) ──────────
    if result.twoD_found and not np.isnan(result.I2D_IG_height):
        r    = result.I2D_IG_height
        thr  = _monolayer_threshold(laser_nm)
        fwhm_2d = twoD.fwhm

        fwhm_ok   = (not np.isnan(fwhm_2d)) and (fwhm_2d <= 35.0)
        fwhm_tag  = ""
        if not fwhm_ok and not np.isnan(fwhm_2d):
            result.twoD_fwhm_warning = True
            fwhm_tag = (
                f" [WARNING: FWHM(2D)={fwhm_2d:.1f} cm⁻¹ > 35 — "
                "I2D/IG layer count unreliable; broadened 2D suggests "
                "multilayer stacking, doping, or substrate coupling]"
            )

        if r > thr:
            result.estimated_layers = (
                f"Monolayer (I2D/IG={r:.2f} > {thr:.1f} @ {laser_nm:.0f} nm)"
                + fwhm_tag
            )
        elif r > thr * 0.5:
            result.estimated_layers = (
                f"Bilayer (I2D/IG={r:.2f}, {thr*0.5:.1f}–{thr:.1f} @ {laser_nm:.0f} nm)"
                + fwhm_tag
            )
        elif r > thr * 0.25:
            result.estimated_layers = (
                f"Few-layer 3–5 (I2D/IG={r:.2f} @ {laser_nm:.0f} nm)"
                + fwhm_tag
            )
        else:
            result.estimated_layers = (
                f"Multilayer/bulk graphite (I2D/IG={r:.2f} < {thr*0.25:.1f} @ {laser_nm:.0f} nm)"
                + fwhm_tag
            )

    return result


# ── Report formatter ──────────────────────────────────────
def format_report(filename: str,
                  peaks:    dict[str, PeakResult],
                  analysis: RamanAnalysis,
                  laser_nm: float) -> str:
    sep = "═" * 64
    def _fv(v, fmt=".4f"):
        return format(v, fmt) if not np.isnan(v) else "N/A"

    # ── Helper: format peak center with uncertainty ────────
    def _fc(p: Optional[PeakResult]) -> str:
        """Return 'center ± stderr' string or plain center if no stderr."""
        if p is None or not p.found:
            return "N/A"
        s = f"{p.center:.1f}"
        if p.center_stderr is not None:
            s += f" ± {p.center_stderr:.1f}"
        return s

    def _ff(p: Optional[PeakResult]) -> str:
        """Return 'fwhm ± stderr' string or plain fwhm if no stderr."""
        if p is None or not p.found:
            return "N/A"
        s = f"{p.fwhm:.1f}"
        if p.fwhm_stderr is not None:
            s += f" ± {p.fwhm_stderr:.1f}"
        return s

    lines = [
        sep, "  RAMAN ANALYSIS REPORT",
        f"  File    : {filename}",
        f"  Laser   : {laser_nm} nm", sep, "",
        "  FITTED PEAKS",
        f"  {'Peak':<8} {'Center (cm⁻¹)':>18} {'FWHM (cm⁻¹)':>16} {'Height':>12} {'Area':>12} {'R²':>8}",
        f"  {'-'*8} {'-'*18} {'-'*16} {'-'*12} {'-'*12} {'-'*8}",
    ]
    for key in ["D_star", "D", "G", "D_prime", "2D", "DG"]:
        p = peaks.get(key)
        if p:
            if p.found:
                note   = " [dual-Lorentzian]" if getattr(p, "is_split_2D", False) else ""
                deconv = " [deconvolved G+D']" if getattr(p, "is_deconvolved", False) else ""
                status = (
                    f"{_fc(p):>18} {_ff(p):>16} "
                    f"{p.amplitude:12.1f} {p.area:12.1f} {p.r_squared:8.3f}"
                    f"{note}{deconv}"
                )
            else:
                status = "       Not detected"
            lines.append(f"  {p.name:<8} {status}")

    lines += [
        "", "  INTENSITY RATIOS",
        f"  ID/IG      (height) : {_fv(analysis.ID_IG_height)}",
        f"  ID/IG      (area)   : {_fv(analysis.ID_IG_area)}",
        f"  I2D/IG     (height) : {_fv(analysis.I2D_IG_height)}",
        f"  I2D/IG     (area)   : {_fv(analysis.I2D_IG_area)}",
        f"  ID'/IG     (height) : {_fv(analysis.IDp_IG_height)}",
        f"  ID/ID'     (height) : {_fv(analysis.ID_IDp_height)}",
        f"  ID*/IG     (height) : {_fv(analysis.IDstar_IG_height)}",   # v2.4 #1
    ]

    # D* note (only when peak was detected)
    if analysis.dstar_co_note:
        lines.append(f"  D* note             : {analysis.dstar_co_note}")

    lines += [
        "", "  STRUCTURAL ANALYSIS",
        f"  L_D (defect spacing) : {_fv(analysis.L_D_nm, '.2f')} nm",
    ]
    if analysis.L_D_note:
        lines.append(f"  L_D note             : {analysis.L_D_note}")
    lines += [
        f"  Disorder stage       : {analysis.disorder_stage}",
        f"  Defect type          : {analysis.defect_type}",
        f"  Estimated layers     : {analysis.estimated_layers}",
    ]

    # B-doping flag (v2.4 Feature #2)
    if analysis.boron_doping_flag:
        lines += [
            "",
            "  ⚠ BORON DOPING FLAG",
            f"  {analysis.boron_doping_note}",
        ]

    lines += [
        "",
        f"  NOTE: Thresholds calibrated for λ = {laser_nm:.0f} nm excitation.",
        "  NOTE: Center and FWHM values shown as 'value ± 1σ' where",
        "        σ is derived from the scipy curve_fit covariance matrix.",
        sep,
    ]
    return "\n".join(lines)
