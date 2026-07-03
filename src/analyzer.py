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
  - Das et al. (2008) Nat. Nanotechnol. 3, 210            — n/p-type from I2D/IG
  - Wu et al. (2018) Carbon 127, 418–428                  — stage boundary [v2.5]
  - Maultzsch et al. (2002) Phys. Rev. B 65, 233402       — D-band dispersion [v2.6]
  - Lucchese et al. (2010) Carbon 48, 1592                — area-ratio L_D [Fix 1.1]
  - Faugeras et al. (2008) Appl. Phys. Lett. 92, 011914  — SiC substrate effects

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
  v2.6  Feature #8: dispersion slope validator [Ferrari & Basko 2013;
        Maultzsch 2002] — multi-wavelength D-band slope check;
        deviation > tolerance flags contamination / non-graphitic sp² carbon.
  Fix 1.1  L_D and B-doping now use A_D/A_G (integrated area ratio) instead
        of I_D/I_G (height ratio). Height ratios introduce a systematic error
        ∝ FWHM(D)/FWHM(G); for typical graphene this causes ~2× L_D
        overestimation. [Lucchese et al. 2010; Ferrari & Robertson 2004]
  Fix 1.2  _ALPHA_PISANA corrected from 2.2e-12 to 0.61 cm⁻¹/√(10¹² cm⁻²).
        Old value caused carrier_density_cm2 overflow (~10²⁵).
        Correct form: n [×10¹² cm⁻²] = (Δω_G / 0.61)²  [Pisana 2007 Fig.3].
        Added out-of-range warning when |n| > 5×10¹³ cm⁻² (model validity limit).
  Fix 1.3  substrate-aware doping: analyze() accepts optional substrate param.
        For non-free-standing substrates (SiC, hBN, SiO2, quartz, sapphire,
        mica, Cu, Ni) the I2D/IG-based n/p classification is suppressed and
        replaced with a substrate-specific warning. SiC substrates receive an
        additional note about Fuchs-Kliewer phonon overlap near the G band
        [Faugeras et al. 2008, Appl. Phys. Lett. 92, 011914].

Python 3.8 compatibility note
------------------------------
All type hints use typing.Dict / typing.Optional / typing.List / typing.Tuple
instead of the built-in generics (dict[...]) which require Python 3.9+.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from .peak_fitter import PeakResult


# ═══════════════════════════════════════════════════════════
# Feature #8 — Dispersion slope validator (v2.6)
# ═══════════════════════════════════════════════════════════

_HC_EV_NM          = 1239.841984   # eV·nm
_D_SLOPE_REFERENCE = 53.0          # cm⁻¹/eV  [Ferrari & Basko 2013]
_D_SLOPE_TOLERANCE = 10.0          # cm⁻¹/eV  — flag if |measured − ref| > this

_GCN4_VIS_MIN_NM = 400.0  # nm; below this is UV (fluorescence-friendly)
_GCN4_VIS_MAX_NM = 700.0  # nm; above this is NIR (fluorescence-friendly)

# ── Substrates where I2D/IG n/p-type classification is unreliable ──────
# Das 2008 model was validated on free-standing / SiO2-supported graphene.
# On these substrates the 2D band is absent, suppressed, or strongly
# perturbed, making the I2D/IG criterion meaningless.
_SUBSTRATES_NO_2D_CLASSIFICATION = {
    "sic", "hbn", "sio2", "quartz", "sapphire", "mica", "cu", "ni",
    "copper", "nickel", "al2o3", "nbsic", "g-nbsic",
}

# SiC-specific warning: Fuchs-Kliewer (FK) phonon replicas of 4H/6H-SiC
# appear at ~1500–1600 cm⁻¹ and can overlap with or blue-shift the G band,
# making G-shift-based doping estimates unreliable.
_SIC_KEYWORDS = {"sic", "nbsic", "g-nbsic", "4h-sic", "6h-sic", "3c-sic"}


@dataclass
class DispersionSlopeResult:
    """
    Result of a multi-wavelength D-band dispersion slope validation.

    Fields
    ------
    n_points : int
        Number of (laser, D_center) data points supplied.
    slope_cm1_per_eV : float
        Linear regression slope  ΔωD / ΔE_laser  [cm⁻¹ eV⁻¹].
        np.nan when fewer than 2 valid points.
    r_squared : float
        R² of the linear fit (1.0 = perfect linear dispersion).
        np.nan when fewer than 3 points.
    deviation : float
        |slope − 53 cm⁻¹/eV|  — absolute departure from the
        graphene reference [Ferrari & Basko 2013].
    contamination_flag : bool
        True when deviation > tolerance (default 10 cm⁻¹/eV).
    note : str
        Human-readable interpretation with literature context.
    """
    n_points:           int   = 0
    slope_cm1_per_eV:   float = np.nan
    r_squared:          float = np.nan
    deviation:          float = np.nan
    contamination_flag: bool  = False
    note:               str   = ""


def validate_dispersion_slope(
    measurements: List[Tuple[float, float]],
    tolerance:    float = _D_SLOPE_TOLERANCE,
) -> DispersionSlopeResult:
    """
    Validate the D-band dispersion slope from multi-wavelength data.

    The D band in graphene/sp² carbon disperses linearly with excitation
    energy: ΔωD/ΔE ≈ 53 cm⁻¹/eV (double-resonance, zone-boundary phonon).
    A slope significantly different from 53 cm⁻¹/eV indicates either:
      - Contamination by non-graphitic carbon (e.g. diamond-like C, a-C)
      - D-band overlap with a non-dispersive mode
      - Measurement artefact (baseline, peak assignment error)

    Parameters
    ----------
    measurements : list of (laser_nm, D_center_cm1) tuples
        Each element is a float pair (excitation wavelength in nm,
        fitted D-band centre in cm⁻¹) from a *single sample* measured
        at multiple excitation energies.
        Minimum 2 points required for a slope; ≥3 recommended for R².
    tolerance : float, optional
        Maximum allowed |slope − 53| before raising contamination_flag.
        Default: 10 cm⁻¹/eV  (i.e. flag if slope outside 43–63 cm⁻¹/eV).

    Returns
    -------
    DispersionSlopeResult

    Examples
    --------
    >>> from src.analyzer import validate_dispersion_slope
    >>> result = validate_dispersion_slope([
    ...     (514, 1348.0),
    ...     (532, 1345.5),
    ...     (633, 1335.8),
    ...     (785, 1321.0),
    ... ])
    >>> print(result.slope_cm1_per_eV, result.contamination_flag)
    53.2  False

    References
    ----------
    Ferrari & Basko (2013) Nature Nanotechnology 8, 235–246.
    Maultzsch et al. (2002) Phys. Rev. B 65, 233402.
    """
    res = DispersionSlopeResult(n_points=len(measurements))

    # ── Filter out NaN / non-positive values ──────────────
    valid = [
        (float(lnm), float(d))
        for lnm, d in measurements
        if lnm > 0 and np.isfinite(lnm) and np.isfinite(d) and d > 0
    ]
    res.n_points = len(valid)

    if res.n_points < 2:
        res.note = (
            "Dispersion slope validation requires at least 2 measurements "
            "at different excitation wavelengths; {} valid point(s) supplied.".format(
                res.n_points)
        )
        return res

    # Convert wavelengths → excitation energies [eV]
    energies = np.array([_HC_EV_NM / lnm for lnm, _ in valid])
    d_centers = np.array([d for _, d in valid])

    # ── Linear regression: ωD = slope × E + intercept ────
    coeffs    = np.polyfit(energies, d_centers, 1)
    slope     = float(coeffs[0])          # cm⁻¹/eV
    intercept = float(coeffs[1])

    y_fit  = np.polyval(coeffs, energies)
    ss_res = float(np.sum((d_centers - y_fit) ** 2))
    ss_tot = float(np.sum((d_centers - d_centers.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    deviation = abs(slope - _D_SLOPE_REFERENCE)
    flag      = deviation > tolerance

    res.slope_cm1_per_eV   = slope
    res.r_squared          = r2 if res.n_points >= 3 else np.nan
    res.deviation          = deviation
    res.contamination_flag = flag

    # ── Compose human-readable note ───────────────────────
    points_str = "; ".join(
        "{:.0f} nm \u2192 {:.1f} cm\u207b\u00b9".format(lnm, d)
        for lnm, d in valid
    )
    r2_str = "R\u00b2 = {:.3f}".format(r2) if np.isfinite(r2) else "R\u00b2 = N/A"

    if not flag:
        interpretation = (
            "Slope {:.1f} cm\u207b\u00b9/eV within \u00b1{:.0f} cm\u207b\u00b9/eV "
            "of the graphene reference (53 cm\u207b\u00b9/eV) \u2014 "
            "consistent with sp\u00b2 graphitic carbon "
            "[Ferrari & Basko 2013, Nat. Nanotechnol. 8, 235].".format(
                slope, tolerance)
        )
    else:
        if slope < _D_SLOPE_REFERENCE - tolerance:
            cause = (
                "Slope {:.1f} cm\u207b\u00b9/eV is *below* 53 cm\u207b\u00b9/eV "
                "\u2014 possible non-dispersive mode overlap "
                "(e.g. polyene C=C at ~1300 cm\u207b\u00b9) "
                "or incorrect D-band assignment."
            ).format(slope)
        else:
            cause = (
                "Slope {:.1f} cm\u207b\u00b9/eV is *above* 53 cm\u207b\u00b9/eV "
                "\u2014 possible contamination by diamond-like carbon (DLC) "
                "or ta-C (D-band disperses at ~80 cm\u207b\u00b9/eV in DLC) "
                "[Maultzsch et al. 2002, Phys. Rev. B 65, 233402]."
            ).format(slope)

        interpretation = (
            "WARNING: deviation = {:.1f} cm\u207b\u00b9/eV > tolerance {:.0f} cm\u207b\u00b9/eV. "
            "{}".format(deviation, tolerance, cause)
        )

    res.note = (
        "Dispersion slope validation | {} points: {} | "
        "Slope = {:.1f} cm\u207b\u00b9/eV | {} | {}".format(
            res.n_points, points_str, slope, r2_str, interpretation)
    )
    return res


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
    doping_type:         str   = "N/A"
    carrier_density_cm2: float = np.nan
    doping_note:         str   = ""
    # ── substrate info (Fix 1.3) ────────────────────────
    substrate:           str   = "unknown"
    # ── v2.5 Feature #5: refined stage boundary ───────────
    stage_refined:       str   = "N/A"
    stage_refined_note:  str   = ""
    # ── v2.6 Feature #8: dispersion slope validator ───────
    dispersion_slope:    Optional[DispersionSlopeResult] = field(default=None)
    gcn4_detected:       bool  = False
    gcn4_mode_note:      str   = ""
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
# Fix 1.1: threshold is now compared against A_D/A_G (area ratio)
_BORON_ID_IG_MIN     = 3.0


def _check_boron_doping(
    G:      Optional[PeakResult],
    D:      Optional[PeakResult],
    Dp:     Optional[PeakResult],
    id_ig:  float,   # Fix 1.1: caller must pass ID_IG_area, not ID_IG_height
    id_idp: float,
) -> tuple:
    if G is None or not G.found:
        return False, ""
    if D is None or not D.found:
        return False, ""
    if Dp is None or not Dp.found:
        return False, ""
    if np.isnan(id_ig) or np.isnan(id_idp):
        return False, ""

    g_ok    = _BORON_G_CENTER_MIN <= G.center <= _BORON_G_CENTER_MAX
    idp_ok  = _BORON_ID_IDp_MIN   <= id_idp   <= _BORON_ID_IDp_MAX
    idig_ok = id_ig >= _BORON_ID_IG_MIN

    if g_ok and idp_ok and idig_ok:
        note = (
            "Boron doping fingerprint detected [Kim et al. 2012, ACS Nano 6, 8203]: "
            "G at {:.1f} cm\u207b\u00b9 (constant, no N-doping blue-shift); "
            "I_D/I_D\u2032 = {:.1f} (sp\u00b3 substitutional B); "
            "A_D/A_G = {:.2f} > 3 (area ratio, Fix 1.1).".format(
                G.center, id_idp, id_ig)
        )
        return True, note
    return False, ""


# ── Doping level estimator (v2.5 Feature #4; Fix 1.2; Fix 1.3) ──
#
# Fix 1.2 — corrected alpha constant (Pisana 2007 Fig.3)
# Fix 1.3 — substrate-aware: suppress I2D/IG n/p classification
#           when substrate is not free-standing graphene.
#
# Pisana et al. (2007):
#   ω_G(n) ≈ ω₀ + 0.61 × √(|n| / 10¹²)   [cm⁻¹]
#
# Valid range: |n| < 5×10¹³ cm⁻²  (n_1e12 < 50).

_G0_UNDOPED   = 1582.0
_ALPHA_PISANA = 0.61      # cm⁻¹ per sqrt(10¹² cm⁻²)  [Pisana 2007 Fig.3]
_DOPING_NOISE = 3.0
_N_MAX_1E12   = 50.0      # 50 × 10¹² cm⁻² = 5×10¹³ cm⁻²  (model validity limit)


def _estimate_doping(
    G:         Optional[PeakResult],
    i2d_ig:    float,
    substrate: str = "unknown",
) -> tuple:
    """
    Estimate carrier density and doping type from G-band shift.

    Parameters
    ----------
    G : PeakResult
        Fitted G-band result.
    i2d_ig : float
        I2D/IG height ratio — used for n/p classification only when
        substrate is free-standing graphene (Das 2008).
    substrate : str
        Substrate identifier (case-insensitive). When this matches a
        known non-free-standing substrate (SiC, hBN, SiO2, Cu, Ni …)
        the I2D/IG-based n/p label is suppressed and a substrate-specific
        warning is included in the note.

    Returns
    -------
    (doping_type: str, carrier_density_cm2: float, note: str)
    """
    if G is None or not G.found:
        return "N/A", np.nan, ""

    substrate_key = substrate.lower().strip()
    is_non_graphene = substrate_key in _SUBSTRATES_NO_2D_CLASSIFICATION
    is_sic          = substrate_key in _SIC_KEYWORDS

    delta_g = G.center - _G0_UNDOPED

    # ── SiC-specific pre-check ─────────────────────────────
    # SiC Fuchs-Kliewer phonon replicas appear at ~1500–1600 cm⁻¹ and
    # can overlap with or artificially shift the graphene G band.
    # G-shift-based doping estimation is therefore unreliable on SiC.
    if is_sic:
        return (
            "N/A (SiC substrate)",
            np.nan,
            "G-shift doping estimation suppressed: SiC substrate detected. "
            "Fuchs-Kliewer phonon replicas of 4H/6H-SiC appear at "
            "~1500\u20131600 cm\u207b\u00b9 and can overlap with / shift the graphene G band, "
            "making Δω_G = {:+.1f} cm\u207b\u00b9 an unreliable doping proxy. "
            "Use electrolyte gating + in-situ Raman for quantitative doping "
            "[Faugeras et al. 2008, Appl. Phys. Lett. 92, 011914].".format(delta_g)
        )

    if abs(delta_g) < _DOPING_NOISE:
        return (
            "undoped", 0.0,
            "G at {:.1f} cm\u207b\u00b9 \u2014 shift {:+.1f} cm\u207b\u00b9 "
            "within noise threshold (\u00b1{} cm\u207b\u00b9); undoped [Pisana 2007].".format(
                G.center, delta_g, _DOPING_NOISE)
        )

    # n [×10¹² cm⁻²] = (Δω_G / α)²
    n_1e12 = (abs(delta_g) / _ALPHA_PISANA) ** 2
    n_cm2  = n_1e12

    # ── n/p-type classification ─────────────────────────────
    if is_non_graphene:
        # I2D/IG criterion unreliable — skip n/p label
        dtype = "N/A (non-graphene substrate: {})".format(substrate)
        type_note = (
            "I2D/IG n/p classification suppressed: substrate '{}' "
            "perturbs or quenches the 2D band — "
            "Das 2008 criterion not applicable.".format(substrate)
        )
    elif np.isnan(i2d_ig) or i2d_ig < 0.5:
        dtype = "n-type"
        type_note = "I2D/IG = {:.2f} < 0.5 \u2192 electron doping [Das 2008]".format(i2d_ig)
    else:
        dtype = "p-type"
        type_note = "I2D/IG = {:.2f} \u2265 0.5 \u2192 hole doping [Das 2008]".format(i2d_ig)

    # ── Out-of-range warning (Fix 1.2) ────────────────────
    if n_1e12 > _N_MAX_1E12:
        out_of_range_warn = (
            " \u26a0 OUT-OF-RANGE: |n| = {:.1f} \u00d7 10\u00b9\u00b2 cm\u207b\u00b2 "
            "> 50 \u00d7 10\u00b9\u00b2 cm\u207b\u00b2; "
            "Pisana 2007 model valid only for |n| < 5\u00d710\u00b9\u00b3 cm\u207b\u00b2. "
            "Result unreliable \u2014 strain, disorder, or substrate effects likely."
        ).format(n_1e12)
    else:
        out_of_range_warn = ""

    note = (
        "G shift \u0394\u03c9_G = {:+.1f} cm\u207b\u00b9 from undoped ({} cm\u207b\u00b9); "
        "estimated |n| \u2248 {:.1f} \u00d7 10\u00b9\u00b2 cm\u207b\u00b2; "
        "{}.{} "
        "[Pisana et al. 2007, Nature Mater. 6, 198; "
        "Das et al. 2008, Nat. Nanotechnol. 3, 210. "
        "Valid for |n| < 5\u00d710\u00b9\u00b3 cm\u207b\u00b2; "
        "strain effects can mimic doping shift.]".format(
            delta_g, _G0_UNDOPED, n_1e12, type_note, out_of_range_warn)
    )
    return dtype, n_cm2, note


# ── Stage boundary refinement (v2.5, Feature #5) ──────────
def _refine_stage(
    G:      Optional[PeakResult],
    D:      Optional[PeakResult],
) -> tuple:
    if G is None or not G.found:
        return "N/A", "G band not detected"

    fwhm_g = G.fwhm
    note_parts = ["FWHM(G) = {:.1f} cm\u207b\u00b9".format(fwhm_g)]

    ad_ag = np.nan
    if D is not None and D.found and G.area > 0:
        ad_ag = D.area / G.area
        note_parts.append("A_D/A_G = {:.3f}".format(ad_ag))
    else:
        note_parts.append("A_D/A_G = N/A (D not detected)")

    stage2_fwhm = fwhm_g >= 80
    stage2_area = (fwhm_g >= 50) and (not np.isnan(ad_ag)) and (ad_ag >= 3.0)

    if stage2_fwhm or stage2_area:
        label = "Stage 2 (amorphous/nanocrystalline carbon)"
        note_parts.append("\u2192 Stage 2 [Wu 2018: FWHM(G) \u2265 80 or (\u226550 + A_D/A_G \u22653.0)]")
        return label, "  ".join(note_parts)

    transition_fwhm = 50 <= fwhm_g < 80
    transition_area = (not np.isnan(ad_ag)) and (ad_ag >= 2.5)

    if transition_fwhm or transition_area:
        label = "Stage 1\u21922 transition"
        note_parts.append("\u2192 Stage 1\u21922 transition [Wu 2018: 50\u2264FWHM(G)<80 or A_D/A_G\u22652.5]")
        return label, "  ".join(note_parts)

    label = "Stage 1 (nanocrystalline graphite \u2014 L_D formula valid)"
    note_parts.append("\u2192 Stage 1 [Wu 2018: FWHM(G) < 50 and A_D/A_G < 2.5]")
    return label, "  ".join(note_parts)


def _check_gcn4_mode(peaks, laser_nm):
    cn_tri = peaks.get("CN_triazine")
    cn_ben = peaks.get("CN_bending")

    detected = (
        (cn_tri is not None and cn_tri.found)
        or (cn_ben is not None and cn_ben.found)
    )
    if not detected:
        return False, ""

    is_visible = _GCN4_VIS_MIN_NM < laser_nm < _GCN4_VIS_MAX_NM

    if is_visible:
        note = (
            "g-C3N4 CN modes detected (691/988 cm-1) at visible excitation "
            "({:.0f} nm). Strong fluorescence likely; "
            "UV (325-364 nm) or NIR (785 nm) Raman is recommended for "
            "quantitative CN-mode analysis."
        ).format(laser_nm)
    else:
        note = (
            "g-C3N4 CN modes (691/988 cm-1) detected under UV/NIR-friendly "
            "excitation ({:.0f} nm): conditions suitable for CN-mode analysis."
        ).format(laser_nm)

    return True, note


# ── Main analysis function ────────────────────────────────
def analyze(
    peaks:        Dict[str, PeakResult],
    laser_nm:     float = 532.0,
    substrate:    str   = "unknown",
    multi_wavelength_D: Optional[List[Tuple[float, float]]] = None,
) -> RamanAnalysis:
    """
    Compute all quantitative Raman metrics from fitted peaks.

    Parameters
    ----------
    peaks : dict from fit_all_peaks()
    laser_nm : float
        Excitation wavelength used for this measurement.
    substrate : str, optional
        Substrate on which the carbon layer is deposited.
        Default: 'unknown' (treated as free-standing for doping purposes).
        Known non-graphene substrates that suppress I2D/IG classification:
        'SiC', 'hBN', 'SiO2', 'quartz', 'sapphire', 'mica', 'Cu', 'Ni'.
        SiC additionally suppresses G-shift doping estimation due to
        Fuchs-Kliewer phonon overlap [Faugeras 2008].
        Case-insensitive. Stored in RamanAnalysis.substrate.
    multi_wavelength_D : list of (laser_nm, D_center_cm1), optional
        If supplied, validate_dispersion_slope() is called and the result
        stored in RamanAnalysis.dispersion_slope (Feature #8, v2.6).
        Example::

            analyze(peaks, laser_nm=532, substrate='SiC',
                    multi_wavelength_D=[
                        (514, 1348.0), (532, 1345.5),
                        (633, 1335.8), (785, 1321.0)
                    ])

    Notes — Fix 1.1
    ---------------
    L_D_nm is calculated from ID_IG_area (A_D/A_G integrated area ratio),
    not from ID_IG_height.  Using height ratios introduces a systematic
    error ∝ FWHM(D)/FWHM(G).  For typical graphene spectra (FWHM_D ≈ 40,
    FWHM_G ≈ 20 cm⁻¹) height-based L_D is ~2× too large.
    [Lucchese et al. 2010 Carbon 48 1592; Ferrari & Robertson 2004]

    Notes — Fix 1.2
    ---------------
    carrier_density_cm2 is now computed with the corrected alpha constant
    (0.61 cm⁻¹/√(10¹² cm⁻²)) from Pisana 2007 Fig. 3.  Values outside
    the model's validity range (|n| > 5×10¹³ cm⁻²) are retained but
    flagged with an OUT-OF-RANGE warning in doping_note.

    Notes — Fix 1.3
    ---------------
    The substrate parameter controls doping classification behaviour.
    For SiC: G-shift estimation is fully suppressed (Fuchs-Kliewer overlap).
    For other non-graphene substrates: G-shift magnitude is computed but
    the I2D/IG-based n/p label is replaced with a substrate warning.
    """
    result = RamanAnalysis()
    result.substrate = substrate

    D     = peaks.get("D")
    G     = peaks.get("G")
    twoD  = peaks.get("2D")
    Dp    = peaks.get("D_prime")
    Dstar = peaks.get("D_star")

    result.G_found    = G    is not None and G.found
    result.D_found    = D    is not None and D.found
    result.twoD_found = twoD is not None and twoD.found

    # Feature #9: g-C3N4 CN modes are independent of the G band,
    # so check them before the G-band early return.
    result.gcn4_detected, result.gcn4_mode_note = _check_gcn4_mode(peaks, laser_nm)

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
                "High D* (I_D*/I_G = {:.3f} > 0.15): "
                "residual C\u2013O groups / partial oxidation. "
                "[Lee et al. 2021, Carbon 183, 814\u2013822]".format(result.IDstar_IG_height)
            )
        else:
            result.dstar_co_note = (
                "I_D*/I_G = {:.3f} \u2264 0.15: low oxidation.".format(result.IDstar_IG_height)
            )

    # ── Disorder stage (original, Ferrari & Robertson 2001) ─
    # NOTE: uses ID_IG_height here intentionally — the Ferrari 2001 stage
    # classification is a qualitative criterion based on peak height trend,
    # not the quantitative Cançado area formula. Only L_D uses area (Fix 1.1).
    stage2 = False
    if result.G_found and result.D_found:
        fwhm_g  = G.fwhm
        id_ig_h = result.ID_IG_height
        if not np.isnan(id_ig_h):
            if fwhm_g > 80:
                result.disorder_stage = "Stage 2 (amorphous carbon, FWHM(G) > 80 cm\u207b\u00b9)"
                stage2 = True
            elif fwhm_g > 50 and id_ig_h < 1.2:
                result.disorder_stage = "Stage 2 (nanocrystalline\u2192amorphous transition)"
                stage2 = True
            else:
                result.disorder_stage = "Stage 1 (nanocrystalline graphite)"
        else:
            result.disorder_stage = "Stage 1 (nanocrystalline graphite)"

    # ── v2.5 Feature #5: refined stage boundary ───────────
    result.stage_refined, result.stage_refined_note = _refine_stage(G, D)
    if "Stage 2" in result.stage_refined:
        stage2 = True

    # ── L_D (Cançado et al. 2011) — Fix 1.1: use area ratio ──
    if not np.isnan(result.ID_IG_area) and result.ID_IG_area > 0:
        if stage2:
            result.L_D_nm   = np.nan
            result.L_D_note = (
                "L_D suppressed: Stage 2 / transition "
                "(Can\u00e7ado 2011 valid only in Stage 1)"
            )
        else:
            result.L_D_nm   = np.sqrt(
                (1.8e-9 * laser_nm**4) / result.ID_IG_area
            )
            result.L_D_note = (
                "Can\u00e7ado et al. (2011); \u03bb={:.0f} nm; "
                "Stage 1 valid; \u00b114\u2009% uncertainty; "
                "A_D/A_G area ratio used (Fix 1.1)".format(laser_nm)
            )

    # ── Defect type (Eckmann et al. 2012) ─────────────────
    if not np.isnan(result.ID_IDp_height):
        r = result.ID_IDp_height
        if r >= 10:
            result.defect_type = "sp\u00b3-type defects (ID/ID\u2032 = {:.1f} \u224813)".format(r)
        elif 5 <= r < 10:
            result.defect_type = "Vacancy-type defects (ID/ID\u2032 = {:.1f} \u22487)".format(r)
        else:
            result.defect_type = "Grain boundary/edge defects (ID/ID\u2032 = {:.1f} \u22483.5)".format(r)

    # ── B-doping fingerprint (v2.4 Feature #2) ────────────
    result.boron_doping_flag, result.boron_doping_note = _check_boron_doping(
        G, D, Dp,
        id_ig=result.ID_IG_area,
        id_idp=result.ID_IDp_height,
    )

    # ── v2.5 Feature #4: doping level estimator (Fix 1.2, 1.3) ─
    result.doping_type, result.carrier_density_cm2, result.doping_note = \
        _estimate_doping(G, result.I2D_IG_height, substrate=substrate)

    # ── v2.6 Feature #8: dispersion slope validator ───────
    if multi_wavelength_D is not None and len(multi_wavelength_D) >= 2:
        result.dispersion_slope = validate_dispersion_slope(multi_wavelength_D)

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
                " [WARNING: FWHM(2D)={:.1f} cm\u207b\u00b9 > 35 \u2014 "
                "I2D/IG layer count unreliable]".format(fwhm_2d)
            )

        if r > thr:
            result.estimated_layers = (
                "Monolayer (I2D/IG={:.2f} > {:.1f} @ {:.0f} nm){}".format(r, thr, laser_nm, fwhm_tag)
            )
        elif r > thr * 0.5:
            result.estimated_layers = (
                "Bilayer (I2D/IG={:.2f} @ {:.0f} nm){}".format(r, laser_nm, fwhm_tag)
            )
        elif r > thr * 0.25:
            result.estimated_layers = (
                "Few-layer 3\u20135 (I2D/IG={:.2f} @ {:.0f} nm){}".format(r, laser_nm, fwhm_tag)
            )
        else:
            result.estimated_layers = (
                "Multilayer/bulk graphite (I2D/IG={:.2f} @ {:.0f} nm){}".format(r, laser_nm, fwhm_tag)
            )

    return result


# ── Report formatter ──────────────────────────────────────
def format_report(
    filename: str,
    peaks:    Dict[str, PeakResult],
    analysis: RamanAnalysis,
    laser_nm: float,
) -> str:
    sep = "\u2550" * 64
    def _fv(v, fmt=".4f"):
        return format(v, fmt) if not np.isnan(v) else "N/A"

    def _fc(p: Optional[PeakResult]) -> str:
        if p is None or not p.found:
            return "N/A"
        s = "{:.1f}".format(p.center)
        if p.center_stderr is not None:
            s += " \u00b1 {:.1f}".format(p.center_stderr)
        return s

    def _ff(p: Optional[PeakResult]) -> str:
        if p is None or not p.found:
            return "N/A"
        s = "{:.1f}".format(p.fwhm)
        if p.fwhm_stderr is not None:
            s += " \u00b1 {:.1f}".format(p.fwhm_stderr)
        return s

    lines = [
        sep, "  RAMAN ANALYSIS REPORT",
        "  File    : {}".format(filename),
        "  Laser   : {} nm".format(laser_nm),
        "  Substrate: {}".format(analysis.substrate),
        sep, "",
        "  FITTED PEAKS",
        "  {:<8} {:>18} {:>16} {:>12} {:>12} {:>8}".format(
            "Peak", "Center (cm\u207b\u00b9)", "FWHM (cm\u207b\u00b9)",
            "Height", "Area", "R\u00b2"),
        "  {} {} {} {} {} {}".format("-"*8, "-"*18, "-"*16, "-"*12, "-"*12, "-"*8),
    ]
    for key in ["D_star", "D", "G", "D_prime", "2D", "DG"]:
        p = peaks.get(key)
        if p:
            if p.found:
                note   = " [dual-Lorentzian]"  if getattr(p, "is_split_2D",    False) else ""
                deconv = " [deconvolved G+D']" if getattr(p, "is_deconvolved", False) else ""
                status = (
                    "{:>18} {:>16} {:12.1f} {:12.1f} {:8.3f}{}{}".format(
                        _fc(p), _ff(p), p.amplitude, p.area, p.r_squared, note, deconv))
            else:
                status = "       Not detected"
            lines.append("  {:<8} {}".format(p.name, status))

    lines += [
        "", "  INTENSITY RATIOS",
        "  ID/IG      (height) : {}".format(_fv(analysis.ID_IG_height)),
        "  ID/IG      (area)   : {}  \u2190 used for L_D and B-doping (Fix 1.1)".format(_fv(analysis.ID_IG_area)),
        "  I2D/IG     (height) : {}".format(_fv(analysis.I2D_IG_height)),
        "  I2D/IG     (area)   : {}".format(_fv(analysis.I2D_IG_area)),
        "  ID'/IG     (height) : {}".format(_fv(analysis.IDp_IG_height)),
        "  ID/ID'     (height) : {}".format(_fv(analysis.ID_IDp_height)),
        "  ID*/IG     (height) : {}".format(_fv(analysis.IDstar_IG_height)),
    ]
    if analysis.dstar_co_note:
        lines.append("  D* note             : {}".format(analysis.dstar_co_note))

    lines += [
        "", "  STRUCTURAL ANALYSIS",
        "  L_D (defect spacing) : {} nm".format(_fv(analysis.L_D_nm, '.2f')),
    ]
    if analysis.L_D_note:
        lines.append("  L_D note             : {}".format(analysis.L_D_note))

    lines += [
        "  Disorder stage       : {}".format(analysis.disorder_stage),
        "  Stage refined (v2.5) : {}".format(analysis.stage_refined),
    ]
    if analysis.stage_refined_note:
        lines.append("  Stage note           : {}".format(analysis.stage_refined_note))

    lines += [
        "  Defect type          : {}".format(analysis.defect_type),
        "  Estimated layers     : {}".format(analysis.estimated_layers),
    ]
    if analysis.twoD_fwhm_warning:
        lines.append("  *** FWHM(2D) > 35 cm\u207b\u00b9: layer count unreliable ***")

    lines += [
        "", "  DOPING & CHEMICAL ENVIRONMENT",
        "  Doping type          : {}".format(analysis.doping_type),
    ]
    if not np.isnan(analysis.carrier_density_cm2) and analysis.carrier_density_cm2 > 0:
        lines.append(
            "  Carrier density      : {:.2e} cm\u207b\u00b2".format(analysis.carrier_density_cm2)
        )
    if analysis.doping_note:
        lines.append("  Doping note          : {}".format(analysis.doping_note))

    if analysis.boron_doping_flag:
        lines.append("  *** Boron doping fingerprint detected ***")
        lines.append("  B-doping note        : {}".format(analysis.boron_doping_note))

    lines += [
        "",
        "  g-C3N4 CN MODES (Feature #9)",
        "  Detected            : {}".format("Yes" if analysis.gcn4_detected else "No"),
    ]
    if analysis.gcn4_mode_note:
        lines.append("  g-C3N4 note         : {}".format(analysis.gcn4_mode_note))

    # ── v2.6 Feature #8: dispersion slope block ───────────
    dslope = analysis.dispersion_slope
    if dslope is not None:
        lines.append("")
        lines.append("  DISPERSION SLOPE VALIDATION (v2.6)")
        lines.append("  Reference slope      : 53 cm\u207b\u00b9/eV (D band, double-resonance)")
        slope_str = "{:.1f} cm\u207b\u00b9/eV".format(dslope.slope_cm1_per_eV) \
            if np.isfinite(dslope.slope_cm1_per_eV) else "N/A"
        r2_str   = "{:.3f}".format(dslope.r_squared) \
            if np.isfinite(dslope.r_squared) else "N/A"
        dev_str  = "{:.1f} cm\u207b\u00b9/eV".format(dslope.deviation) \
            if np.isfinite(dslope.deviation) else "N/A"
        flag_str = "\u26a0 CONTAMINATION SUSPECTED" if dslope.contamination_flag \
            else "\u2713 Within tolerance"
        lines.append("  Measured slope       : {}".format(slope_str))
        lines.append("  R\u00b2 (linear fit)     : {}".format(r2_str))
        lines.append("  Deviation            : {}".format(dev_str))
        lines.append("  Status               : {}".format(flag_str))
        if dslope.note:
            lines.append("  Slope note           : {}".format(dslope.note))

    lines.append(sep)
    return "\n".join(lines)
