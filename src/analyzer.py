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
  - Pisana et al. (2007) Nature Mater. 6, 198             — doping G-shift [v2.5]
  - Wu et al. (2018) Carbon 127, 418–428                  — stage boundary [v2.5]

Change log:
  v2.0  dispersion fix — eV-based window shifts in peak_fitter
  v2.1  adaptive G fit + G+D' deconvolution
  v2.2  L_D: correct source (Cançado 2011); suppress L_D in Stage 2
        layer count: FWHM(2D) plausibility guard
        remove Graphitization % (no literature basis)
  v2.4  Feature #1: D* band → I_D*/I_G ratio + rGO C/O proxy note
        Feature #2: B-doping fingerprint flag [Kim 2012]
        Feature #3: fitting uncertainty (center_stderr, fwhm_stderr)
  v2.5  Feature #4: doping level estimator [Pisana 2007]
        Feature #5: stage boundary refinement with FWHM(G) + A_D/A_G [Wu 2018]
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
    IDstar_IG_height:    float = np.nan
    dstar_co_note:       str   = ""
    # ── v2.4 Feature #2: B-doping fingerprint ─────────────
    boron_doping_flag:   bool  = False
    boron_doping_note:   str   = ""
    # ── v2.5 Feature #4: doping level estimator ───────────
    doping_type:         str   = "N/A"   # 'n-type' | 'p-type' | 'undoped' | 'N/A'
    carrier_density_cm2: float = np.nan  # cm⁻²
    doping_note:         str   = ""
    # ── v2.5 Feature #5: refined stage boundary ───────────
    stage_refined:       str   = "N/A"
    stage_refined_note:  str   = ""
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


# ── Monolayer I2D/IG threshold (laser-dependent) ──────────
def _monolayer_threshold(laser_nm: float) -> float:
    if laser_nm <= 514:
        return 2.5
    elif laser_nm <= 568:
        return 2.0
    elif laser_nm <= 660:
        return 1.5
    else:
        return 0.8


# ── B-doping fingerprint (v2.4, Feature #2) ───────────────
_BORON_G_CENTER_MIN  = 1577.0
_BORON_G_CENTER_MAX  = 1587.0
_BORON_ID_IDp_MIN    = 5.0
_BORON_ID_IDp_MAX    = 9.0
_BORON_ID_IG_MIN     = 3.0


def _check_boron_doping(
    G:      Optional[PeakResult],
    D:      Optional[PeakResult],
    Dp:     Optional[PeakResult],
    id_ig:  float,
    id_idp: float,
) -> tuple[bool, str]:
    if G is None or not G.found:
        return False, ""
    if D is None or not D.found:
        return False, ""
    if Dp is None or not Dp.found:
        return False, ""
    if np.isnan(id_ig) or np.isnan(id_idp):
        return False, ""

    g_ok   = _BORON_G_CENTER_MIN <= G.center <= _BORON_G_CENTER_MAX
    idp_ok = _BORON_ID_IDp_MIN   <= id_idp   <= _BORON_ID_IDp_MAX
    idig_ok = id_ig >= _BORON_ID_IG_MIN

    if g_ok and idp_ok and idig_ok:
        note = (
            f"Boron doping fingerprint detected [Kim et al. 2012, ACS Nano 6, 8203]: "
            f"G at {G.center:.1f} cm\u207b\u00b9 (constant, no N-doping blue-shift); "
            f"I_D/I_D\u2032 = {id_idp:.1f} (sp\u00b3 substitutional B); "
            f"I_D/I_G = {id_ig:.2f} > 3."
        )
        return True, note
    return False, ""


# ── Doping level estimator (v2.5, Feature #4) ─────────────
# Pisana et al. (2007) Nature Mater. 6, 198:
#   G band stiffens (blue-shifts) with both electron (n) and hole (p)
#   doping due to non-adiabatic removal of Kohn anomaly.
#   G shift from undoped value (1582 cm⁻¹) correlates with |E_F|:
#
#     |ΔωG| ≈ 42 × (E_F / ℏv_F√(π|n|))   [linearised, gate-doping regime]
#
#   Simplified linear approximation used here (valid for |n| < 5×10¹³ cm⁻²):
#     n [cm⁻²] ≈ (ΔωG / α)²   where  α ≈ 2.2 × 10⁻¹² cm⁻¹ per cm⁻²·⁰·⁵
#
#   Sign convention [Das et al. 2008, Nat. Nanotechnol. 3, 210]:
#     Δω > 0 (blue-shift) → ambiguous (both n and p stiffen G).
#     Use 2D/G ratio to break the degeneracy:
#       I2D/IG ↓ with n-type (electron doping suppresses 2D)
#       I2D/IG broadly preserved or ↑ with light p-type
#
# Implementation note:
#   We report the magnitude of carrier density from |ΔωG|.
#   n/p assignment is made via I2D/IG: < 0.5 → n-type, ≥ 0.5 → p-type.
#   For |ΔωG| < 3 cm⁻¹ (within measurement noise) → 'undoped'.

_G0_UNDOPED    = 1582.0   # cm⁻¹, undoped graphene G position
_ALPHA_PISANA  = 2.2e-12  # cm⁻¹ per (cm⁻²)^0.5  [Pisana 2007 linearised]
_DOPING_NOISE  = 3.0      # cm⁻¹ — shifts below this are within noise


def _estimate_doping(
    G:      Optional[PeakResult],
    i2d_ig: float,
) -> tuple[str, float, str]:
    """
    Return (doping_type, carrier_density_cm2, note).
    """
    if G is None or not G.found:
        return "N/A", np.nan, ""

    delta_g = G.center - _G0_UNDOPED

    if abs(delta_g) < _DOPING_NOISE:
        return (
            "undoped", 0.0,
            f"G at {G.center:.1f} cm\u207b\u00b9 — shift {delta_g:+.1f} cm\u207b\u00b9 "
            f"within noise threshold (\u00b1{_DOPING_NOISE} cm\u207b\u00b9); undoped [Pisana 2007]."
        )

    # Carrier density magnitude
    n_cm2 = (abs(delta_g) / _ALPHA_PISANA) ** 2 / 1e12   # ×10¹² cm⁻²
    n_abs  = (abs(delta_g) / _ALPHA_PISANA) ** 2          # cm⁻²

    # n/p assignment via I2D/IG
    if np.isnan(i2d_ig) or i2d_ig < 0.5:
        dtype = "n-type"
        type_note = f"I2D/IG = {i2d_ig:.2f} < 0.5 → electron doping [Das 2008]"
    else:
        dtype = "p-type"
        type_note = f"I2D/IG = {i2d_ig:.2f} \u2265 0.5 \u2192 hole doping [Das 2008]"

    note = (
        f"G blue-shift \u0394\u03c9_G = {delta_g:+.1f} cm\u207b\u00b9 from undoped ({_G0_UNDOPED} cm\u207b\u00b9); "
        f"estimated |n| \u2248 {n_cm2:.2f} \u00d7 10\u00b9\u00b2 cm\u207b\u00b2; "
        f"{type_note}. "
        f"[Pisana et al. 2007, Nature Mater. 6, 198; "
        f"Das et al. 2008, Nat. Nanotechnol. 3, 210. "
        f"Valid for |n| < 5\u00d710\u00b9\u00b3 cm\u207b\u00b2; "
        f"strain effects can mimic doping shift.]"
    )
    return dtype, n_abs, note


# ── Stage boundary refinement (v2.5, Feature #5) ──────────
# Wu et al. (2018) Carbon 127, 418–428:
#   Simple I_D/I_G alone misclassifies samples near the Stage 1/2 boundary
#   because FWHM(G) and the area ratio A_D/A_G carry complementary information:
#     Stage 1 (nanocrystalline graphite, L_D valid):
#       FWHM(G) < 50 cm⁻¹  AND  A_D/A_G < 2.5
#     Stage 1→2 transition:
#       50 ≤ FWHM(G) < 80  OR   A_D/A_G ≥ 2.5
#     Stage 2 (amorphous carbon):
#       FWHM(G) ≥ 80 cm⁻¹  OR   (FWHM(G) ≥ 50 AND A_D/A_G ≥ 3.0)

def _refine_stage(
    G:      Optional[PeakResult],
    D:      Optional[PeakResult],
) -> tuple[str, str]:
    """
    Return (stage_label, note) using FWHM(G) + A_D/A_G criteria.
    """
    if G is None or not G.found:
        return "N/A", "G band not detected"

    fwhm_g = G.fwhm
    note_parts = [f"FWHM(G) = {fwhm_g:.1f} cm\u207b\u00b9"]

    ad_ag = np.nan
    if D is not None and D.found and G.area > 0:
        ad_ag = D.area / G.area
        note_parts.append(f"A_D/A_G = {ad_ag:.3f}")
    else:
        note_parts.append("A_D/A_G = N/A (D not detected)")

    # Stage 2 check
    stage2_fwhm = fwhm_g >= 80
    stage2_area = (fwhm_g >= 50) and (not np.isnan(ad_ag)) and (ad_ag >= 3.0)

    if stage2_fwhm or stage2_area:
        label = "Stage 2 (amorphous/nanocrystalline carbon)"
        note_parts.append("→ Stage 2 [Wu 2018: FWHM(G) ≥ 80 or (≥50 + A_D/A_G ≥ 3.0)]")
        return label, "  ".join(note_parts)

    # Transition zone
    transition_fwhm = 50 <= fwhm_g < 80
    transition_area = (not np.isnan(ad_ag)) and (ad_ag >= 2.5)

    if transition_fwhm or transition_area:
        label = "Stage 1→2 transition"
        note_parts.append("→ Stage 1\u21922 transition [Wu 2018: 50\u2264FWHM(G)<80 or A_D/A_G\u22652.5]")
        return label, "  ".join(note_parts)

    # Stage 1
    label = "Stage 1 (nanocrystalline graphite — L_D formula valid)"
    note_parts.append("→ Stage 1 [Wu 2018: FWHM(G) < 50 and A_D/A_G < 2.5]")
    return label, "  ".join(note_parts)


# ── Main analysis function ────────────────────────────────
def analyze(peaks: dict[str, PeakResult], laser_nm: float = 532.0) -> RamanAnalysis:
    result = RamanAnalysis()
    D     = peaks.get("D")
    G     = peaks.get("G")
    twoD  = peaks.get("2D")
    Dp    = peaks.get("D_prime")
    Dstar = peaks.get("D_star")

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

    # ── D* band ratio + C/O proxy (v2.4 Feature #1) ───────
    if Dstar is not None and Dstar.found and G.amplitude > 0:
        result.IDstar_IG_height = Dstar.amplitude / G.amplitude
        if result.IDstar_IG_height > 0.15:
            result.dstar_co_note = (
                f"High D* (I_D*/I_G = {result.IDstar_IG_height:.3f} > 0.15): "
                "residual C\u2013O groups / partial oxidation. "
                "[Lee et al. 2021, Carbon 183, 814\u2013822]"
            )
        else:
            result.dstar_co_note = (
                f"I_D*/I_G = {result.IDstar_IG_height:.3f} \u2264 0.15: low oxidation."
            )

    # ── Disorder stage (original, Ferrari & Robertson 2001) ─
    stage2 = False
    if result.G_found and result.D_found:
        fwhm_g = G.fwhm
        id_ig  = result.ID_IG_height
        if not np.isnan(id_ig):
            if fwhm_g > 80:
                result.disorder_stage = "Stage 2 (amorphous carbon, FWHM(G) > 80 cm\u207b\u00b9)"
                stage2 = True
            elif fwhm_g > 50 and id_ig < 1.2:
                result.disorder_stage = "Stage 2 (nanocrystalline\u2192amorphous transition)"
                stage2 = True
            else:
                result.disorder_stage = "Stage 1 (nanocrystalline graphite)"
        else:
            result.disorder_stage = "Stage 1 (nanocrystalline graphite)"

    # ── v2.5 Feature #5: refined stage boundary ───────────
    result.stage_refined, result.stage_refined_note = _refine_stage(G, D)
    # If refined says Stage 2 but original did not, update stage2 flag for L_D
    if "Stage 2" in result.stage_refined:
        stage2 = True

    # ── L_D (Cançado et al. 2011) ─────────────────────────
    if not np.isnan(result.ID_IG_height) and result.ID_IG_height > 0:
        if stage2:
            result.L_D_nm   = np.nan
            result.L_D_note = (
                "L_D suppressed: Stage 2 / transition "
                "(Canc\u0327ado 2011 valid only in Stage 1)"
            )
        else:
            result.L_D_nm   = np.sqrt(
                (1.8e-9 * laser_nm**4) / result.ID_IG_height
            )
            result.L_D_note = (
                f"Canc\u0327ado et al. (2011); \u03bb={laser_nm:.0f} nm; "
                "Stage 1 valid; \u00b114\u2009% uncertainty"
            )

    # ── Defect type (Eckmann et al. 2012) ─────────────────
    if not np.isnan(result.ID_IDp_height):
        r = result.ID_IDp_height
        if r >= 10:
            result.defect_type = f"sp\u00b3-type defects (ID/ID\u2032 = {r:.1f} \u224813)"
        elif 5 <= r < 10:
            result.defect_type = f"Vacancy-type defects (ID/ID\u2032 = {r:.1f} \u22487)"
        else:
            result.defect_type = f"Grain boundary/edge defects (ID/ID\u2032 = {r:.1f} \u22483.5)"

    # ── B-doping fingerprint (v2.4 Feature #2) ────────────
    result.boron_doping_flag, result.boron_doping_note = _check_boron_doping(
        G, D, Dp,
        id_ig=result.ID_IG_height,
        id_idp=result.ID_IDp_height,
    )

    # ── v2.5 Feature #4: doping level estimator ───────────
    result.doping_type, result.carrier_density_cm2, result.doping_note = \
        _estimate_doping(G, result.I2D_IG_height)

    # ── Layer count (laser-wavelength corrected) ───────────
    if result.twoD_found and not np.isnan(result.I2D_IG_height):
        r    = result.I2D_IG_height
        thr  = _monolayer_threshold(laser_nm)
        fwhm_2d = twoD.fwhm

        fwhm_ok  = (not np.isnan(fwhm_2d)) and (fwhm_2d <= 35.0)
        fwhm_tag = ""
        if not fwhm_ok and not np.isnan(fwhm_2d):
            result.twoD_fwhm_warning = True
            fwhm_tag = (
                f" [WARNING: FWHM(2D)={fwhm_2d:.1f} cm\u207b\u00b9 > 35 — "
                "I2D/IG layer count unreliable]"
            )

        if r > thr:
            result.estimated_layers = (
                f"Monolayer (I2D/IG={r:.2f} > {thr:.1f} @ {laser_nm:.0f} nm)" + fwhm_tag
            )
        elif r > thr * 0.5:
            result.estimated_layers = (
                f"Bilayer (I2D/IG={r:.2f} @ {laser_nm:.0f} nm)" + fwhm_tag
            )
        elif r > thr * 0.25:
            result.estimated_layers = (
                f"Few-layer 3\u20135 (I2D/IG={r:.2f} @ {laser_nm:.0f} nm)" + fwhm_tag
            )
        else:
            result.estimated_layers = (
                f"Multilayer/bulk graphite (I2D/IG={r:.2f} @ {laser_nm:.0f} nm)" + fwhm_tag
            )

    return result


# ── Report formatter ──────────────────────────────────────
def format_report(filename: str,
                  peaks:    dict[str, PeakResult],
                  analysis: RamanAnalysis,
                  laser_nm: float) -> str:
    sep = "\u2550" * 64
    def _fv(v, fmt=".4f"):
        return format(v, fmt) if not np.isnan(v) else "N/A"

    def _fc(p: Optional[PeakResult]) -> str:
        if p is None or not p.found:
            return "N/A"
        s = f"{p.center:.1f}"
        if p.center_stderr is not None:
            s += f" \u00b1 {p.center_stderr:.1f}"
        return s

    def _ff(p: Optional[PeakResult]) -> str:
        if p is None or not p.found:
            return "N/A"
        s = f"{p.fwhm:.1f}"
        if p.fwhm_stderr is not None:
            s += f" \u00b1 {p.fwhm_stderr:.1f}"
        return s

    lines = [
        sep, "  RAMAN ANALYSIS REPORT",
        f"  File    : {filename}",
        f"  Laser   : {laser_nm} nm", sep, "",
        "  FITTED PEAKS",
        f"  {'Peak':<8} {'Center (cm\u207b\u00b9)':>18} {'FWHM (cm\u207b\u00b9)':>16} "
        f"{'Height':>12} {'Area':>12} {'R\u00b2':>8}",
        f"  {'-'*8} {'-'*18} {'-'*16} {'-'*12} {'-'*12} {'-'*8}",
    ]
    for key in ["D_star", "D", "G", "D_prime", "2D", "DG"]:
        p = peaks.get(key)
        if p:
            if p.found:
                note   = " [dual-Lorentzian]"  if getattr(p, "is_split_2D",    False) else ""
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
        f"  ID*/IG     (height) : {_fv(analysis.IDstar_IG_height)}",
    ]
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
        f"  Stage refined (v2.5) : {analysis.stage_refined}",
    ]
    if analysis.stage_refined_note:
        lines.append(f"  Stage note           : {analysis.stage_refined_note}")

    lines += [
        f"  Defect type          : {analysis.defect_type}",
        f"  Estimated layers     : {analysis.estimated_layers}",
    ]

    # ── v2.5 Feature #4: doping ────────────────────────────
    if analysis.doping_type != "N/A":
        lines += [
            "", "  DOPING ANALYSIS (v2.5)",
            f"  Doping type          : {analysis.doping_type}",
            f"  |n| cm\u207b\u00b2           : "
            + (_fv(analysis.carrier_density_cm2, '.3e') if not np.isnan(analysis.carrier_density_cm2) else "N/A"),
        ]
        if analysis.doping_note:
            lines.append(f"  Note                 : {analysis.doping_note}")

    # B-doping flag
    if analysis.boron_doping_flag:
        lines += [
            "",
            "  \u26a0 BORON DOPING FLAG",
            f"  {analysis.boron_doping_note}",
        ]

    lines += [
        "",
        f"  NOTE: Thresholds calibrated for \u03bb = {laser_nm:.0f} nm excitation.",
        "  NOTE: Center and FWHM shown as 'value \u00b1 1\u03c3' from scipy pcov.",
        sep,
    ]
    return "\n".join(lines)
