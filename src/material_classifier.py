"""
Rule-based material fingerprint classifier for graphene/sp² carbon Raman spectra.

Outputs a list of MaterialHypothesis objects, each with:
  - material   : human-readable label
  - confidence : 'confirmed' | 'likely' | 'possible' | 'not_detected'
  - evidence   : list of supporting/failing criteria strings

Design principles
─────────────────
• No hard classification: Raman alone cannot definitively identify a
  material. Confidence levels reflect the weight of spectral evidence.
• All criteria reference primary literature; references listed inline.
• Rule priority: use already-computed RamanAnalysis fields + PeakResult
  objects — no re-fitting, no new peak windows (except g-C₃N₄ detection
  which requires a probe of the raw spectrum in 650–1050 cm⁻¹).

References
──────────
  Kim et al. (2012)  ACS Nano 6, 8203            — B-doped graphene
  Eckmann et al. (2012) Nano Lett. 12, 3925       — defect type / N-doping
  Pisana et al. (2007) Nature Mater. 6, 198       — gate-doping G-shift
  Ferrari & Basko (2013) Nat. Nanotechnol. 8, 235 — pristine graphene
  Lee et al. (2021) Carbon 183, 814–822           — rGO / D* band
  Thomas et al. (2019) J. Mater. Chem. A 7, 23898 — g-C₃N₄ Raman modes
  Ferrari & Robertson (2001) PRB 64, 075414       — disorder stages
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

from .peak_fitter import PeakResult
from .analyzer import RamanAnalysis

# ─────────────────────────────────────────────────────────
# Confidence levels  (ordered weakest → strongest)
# ─────────────────────────────────────────────────────────
CONFIDENCE_RANKS = {
    "not_detected": 0,
    "possible":     1,
    "likely":       2,
    "confirmed":    3,
}


@dataclass
class MaterialHypothesis:
    material:   str
    confidence: str          # 'confirmed' | 'likely' | 'possible' | 'not_detected'
    evidence:   List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.confidence not in CONFIDENCE_RANKS:
            raise ValueError(f"Unknown confidence level: {self.confidence}")

    @property
    def rank(self) -> int:
        return CONFIDENCE_RANKS[self.confidence]

    def __repr__(self):
        return f"MaterialHypothesis({self.material!r}, {self.confidence!r})"


# ─────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────
def _found(p: Optional[PeakResult]) -> bool:
    return p is not None and p.found


def _center(p: Optional[PeakResult]) -> float:
    return p.center if _found(p) else np.nan


def _amp(p: Optional[PeakResult]) -> float:
    return p.amplitude if _found(p) else np.nan


# ─────────────────────────────────────────────────────────
# Rule set 1 — Pristine / high-quality graphene
# ─────────────────────────────────────────────────────────
def _rule_pristine_graphene(
    peaks: dict[str, PeakResult],
    analysis: RamanAnalysis,
) -> MaterialHypothesis:
    """
    Criteria [Ferrari & Basko 2013; Ferrari et al. 2006]:
      CONFIRMED  : ID/IG < 0.05  AND  I2D/IG > 1.5  AND  FWHM(2D) ≤ 35 cm⁻¹
      LIKELY     : ID/IG < 0.15  AND  I2D/IG > 1.0
      POSSIBLE   : G detected, no/weak D, I2D/IG > 0.5
      NOT_DETECTED: otherwise
    """
    G    = peaks.get("G")
    twoD = peaks.get("2D")
    ev: List[str] = []

    if not _found(G):
        return MaterialHypothesis("Pristine / high-quality graphene",
                                   "not_detected", ["G band not detected"])

    id_ig   = analysis.ID_IG_height
    i2d_ig  = analysis.I2D_IG_height
    fwhm_2d = twoD.fwhm if _found(twoD) else np.nan

    # Check D absence
    d_weak = np.isnan(id_ig) or id_ig < 0.05
    d_low  = np.isnan(id_ig) or id_ig < 0.15

    if d_weak and not np.isnan(i2d_ig) and i2d_ig > 1.5 and not np.isnan(fwhm_2d) and fwhm_2d <= 35:
        ev.append(f"ID/IG = {id_ig:.3f} < 0.05 — negligible defects")
        ev.append(f"I2D/IG = {i2d_ig:.2f} > 1.5")
        ev.append(f"FWHM(2D) = {fwhm_2d:.1f} cm⁻¹ ≤ 35 — sharp 2D peak")
        return MaterialHypothesis("Pristine / high-quality graphene", "confirmed", ev)

    if d_low and not np.isnan(i2d_ig) and i2d_ig > 1.0:
        ev.append(f"ID/IG = {id_ig:.3f} < 0.15")
        ev.append(f"I2D/IG = {i2d_ig:.2f} > 1.0")
        return MaterialHypothesis("Pristine / high-quality graphene", "likely", ev)

    if not np.isnan(i2d_ig) and i2d_ig > 0.5:
        ev.append(f"I2D/IG = {i2d_ig:.2f} > 0.5 — graphene-like 2D present")
        if not np.isnan(id_ig):
            ev.append(f"But ID/IG = {id_ig:.3f} indicates defects")
        return MaterialHypothesis("Pristine / high-quality graphene", "possible", ev)

    ev.append(f"ID/IG = {id_ig:.3f}" if not np.isnan(id_ig) else "D band dominant")
    ev.append(f"I2D/IG = {i2d_ig:.2f}" if not np.isnan(i2d_ig) else "2D band weak/absent")
    return MaterialHypothesis("Pristine / high-quality graphene", "not_detected", ev)


# ─────────────────────────────────────────────────────────
# Rule set 2 — B-doped graphene
# ─────────────────────────────────────────────────────────
def _rule_boron_doped(
    analysis: RamanAnalysis,
) -> MaterialHypothesis:
    """
    Directly uses the v2.4 Feature #2 flag already in RamanAnalysis.
    [Kim et al. 2012, ACS Nano 6, 8203]
    """
    if analysis.boron_doping_flag:
        return MaterialHypothesis(
            "B-doped graphene", "likely",
            [analysis.boron_doping_note]
        )
    # Partial evidence: G constant + elevated ID/IG but ID/ID' outside range
    id_ig  = analysis.ID_IG_height
    id_idp = analysis.ID_IDp_height
    ev: List[str] = []
    if not np.isnan(id_ig) and id_ig > 1.5 and not np.isnan(id_idp) and 3 <= id_idp < 5:
        ev.append(f"ID/IG = {id_ig:.2f} elevated; ID/ID' = {id_idp:.1f} (borderline sp³)")
        ev.append("Does not fully meet Kim 2012 criteria — XPS/EDX recommended")
        return MaterialHypothesis("B-doped graphene", "possible", ev)
    return MaterialHypothesis("B-doped graphene", "not_detected",
                              ["Kim 2012 three-criteria test not satisfied"])


# ─────────────────────────────────────────────────────────
# Rule set 3 — N-doped graphene
# ─────────────────────────────────────────────────────────
def _rule_n_doped(
    peaks: dict[str, PeakResult],
    analysis: RamanAnalysis,
) -> MaterialHypothesis:
    """
    N-doping signature [Eckmann et al. 2012; Lucchese et al. 2010]:
      • G blue-shift > 10 cm⁻¹ from 1582 cm⁻¹  (electron doping raises G)
      • D' significantly broadened (FWHM > 20 cm⁻¹)
      • ID/IG moderate (0.3–1.5) — N-doping introduces defects but
        less aggressively than B-doping
    NOTE: G blue-shift alone is ambiguous (strain, substrate, gate doping).
          Require at least 2 of the 3 criteria.
    """
    G  = peaks.get("G")
    Dp = peaks.get("D_prime")
    ev: List[str] = []

    if not _found(G):
        return MaterialHypothesis("N-doped graphene", "not_detected", ["G not found"])

    criteria_met = 0
    g_shift = G.center - 1582.0
    if g_shift > 10:
        criteria_met += 1
        ev.append(f"G blue-shifted by {g_shift:.1f} cm⁻¹ > 10 cm⁻¹ (electron doping)")
    else:
        ev.append(f"G at {G.center:.1f} cm⁻¹ — shift {g_shift:+.1f} cm⁻¹ (< 10 cm⁻¹ threshold)")

    if _found(Dp) and Dp.fwhm > 20:
        criteria_met += 1
        ev.append(f"FWHM(D') = {Dp.fwhm:.1f} cm⁻¹ > 20 — broadened by N substitution")
    elif _found(Dp):
        ev.append(f"FWHM(D') = {Dp.fwhm:.1f} cm⁻¹ — not significantly broadened")

    id_ig = analysis.ID_IG_height
    if not np.isnan(id_ig) and 0.3 <= id_ig <= 1.5:
        criteria_met += 1
        ev.append(f"ID/IG = {id_ig:.2f} in moderate range [0.3–1.5]")
    elif not np.isnan(id_ig):
        ev.append(f"ID/IG = {id_ig:.2f} outside moderate N-doping range")

    if criteria_met >= 3:
        return MaterialHypothesis("N-doped graphene", "likely", ev)
    if criteria_met == 2:
        return MaterialHypothesis("N-doped graphene", "possible", ev)
    return MaterialHypothesis("N-doped graphene", "not_detected", ev)


# ─────────────────────────────────────────────────────────
# Rule set 4 — rGO / GO
# ─────────────────────────────────────────────────────────
def _rule_rgo(
    peaks: dict[str, PeakResult],
    analysis: RamanAnalysis,
) -> MaterialHypothesis:
    """
    rGO/GO markers [Lee et al. 2021; Tuinstra & Koenig 1970 convention]:
      CONFIRMED : I_D*/I_G > 0.15  AND  ID/IG > 0.8  AND  2D suppressed
      LIKELY    : I_D*/I_G > 0.15  AND  ID/IG > 0.5
      POSSIBLE  : ID/IG > 0.8  AND  2D absent/weak (I2D/IG < 0.3)
    """
    ev: List[str] = []
    id_ig    = analysis.ID_IG_height
    idstar   = analysis.IDstar_IG_height
    i2d_ig   = analysis.I2D_IG_height

    dstar_high = not np.isnan(idstar) and idstar > 0.15
    d_high     = not np.isnan(id_ig)  and id_ig  > 0.8
    d_mod      = not np.isnan(id_ig)  and id_ig  > 0.5
    twoD_low   = np.isnan(i2d_ig) or i2d_ig < 0.3

    if dstar_high:
        ev.append(f"I_D*/I_G = {idstar:.3f} > 0.15 — residual C–O groups [Lee 2021]")
    else:
        ev.append(f"I_D*/I_G = {idstar:.3f} ≤ 0.15" if not np.isnan(idstar)
                  else "D* band not detected")

    if not np.isnan(id_ig):
        ev.append(f"ID/IG = {id_ig:.2f}")
    if twoD_low:
        ev.append(f"I2D/IG = {i2d_ig:.2f} < 0.3 — 2D suppressed" if not np.isnan(i2d_ig)
                  else "2D band absent")

    if dstar_high and d_high and twoD_low:
        return MaterialHypothesis("rGO / GO", "confirmed", ev)
    if dstar_high and d_mod:
        return MaterialHypothesis("rGO / GO", "likely", ev)
    if d_high and twoD_low:
        return MaterialHypothesis("rGO / GO", "possible", ev)
    return MaterialHypothesis("rGO / GO", "not_detected", ev)


# ─────────────────────────────────────────────────────────
# Rule set 5 — g-C₃N₄
# ─────────────────────────────────────────────────────────
# g-C₃N₄ signature bands [Thomas et al. 2019, J. Mater. Chem. A 7, 23898]:
#   691  cm⁻¹  — in-plane bending of triazine units
#   988  cm⁻¹  — breathing mode of triazine ring
#   1230 cm⁻¹  — aromatic C–N stretching
# In pure g-C₃N₄: G and 2D bands are ABSENT (no sp² carbon network).
# In g-C₃N₄ / graphene composites: graphene bands present alongside 691/988.
#
# Detection strategy:
#   We probe the raw spectrum directly in two narrow windows
#   (680–710 cm⁻¹ and 975–1005 cm⁻¹) looking for signal > noise.
#   This avoids requiring a full peak-fit for these low-wavenumber bands.

def _probe_window(
    wavenumbers: np.ndarray,
    intensities: np.ndarray,
    lo: float,
    hi: float,
    snr_min: float = 3.0,
) -> tuple[bool, float, float]:
    """
    Return (detected, peak_position, snr) for a candidate peak in [lo, hi].
    SNR = max_signal / MAD_noise.
    """
    mask = (wavenumbers >= lo) & (wavenumbers <= hi)
    if mask.sum() < 3:
        return False, np.nan, 0.0
    seg = intensities[mask]
    wn_seg = wavenumbers[mask]
    peak_idx = int(np.argmax(seg))
    peak_val = seg[peak_idx]
    noise = np.median(np.abs(seg - np.median(seg)))
    snr = peak_val / noise if noise > 0 else 0.0
    return snr >= snr_min, wn_seg[peak_idx], snr


def _rule_g_c3n4(
    peaks:       dict[str, PeakResult],
    analysis:    RamanAnalysis,
    wavenumbers: Optional[np.ndarray] = None,
    intensities: Optional[np.ndarray] = None,
) -> MaterialHypothesis:
    """
    g-C₃N₄ detection — requires raw spectrum arrays for low-wavenumber probe.
    If wavenumbers/intensities are None, falls back to absence-of-graphene logic.

    [Thomas et al. 2019, J. Mater. Chem. A 7, 23898]
    """
    ev: List[str] = []
    G_present = _found(peaks.get("G"))
    D_present = _found(peaks.get("D"))

    det_691 = det_988 = False
    snr_691 = snr_988 = 0.0

    if wavenumbers is not None and intensities is not None:
        det_691, pos_691, snr_691 = _probe_window(wavenumbers, intensities, 680, 710)
        det_988, pos_988, snr_988 = _probe_window(wavenumbers, intensities, 975, 1005)
        if det_691:
            ev.append(f"Triazine bending ~691 cm⁻¹ detected at {pos_691:.0f} cm⁻¹ (SNR={snr_691:.1f})")
        else:
            ev.append("Triazine bending ~691 cm⁻¹ NOT detected")
        if det_988:
            ev.append(f"Triazine breathing ~988 cm⁻¹ detected at {pos_988:.0f} cm⁻¹ (SNR={snr_988:.1f})")
        else:
            ev.append("Triazine breathing ~988 cm⁻¹ NOT detected")
    else:
        ev.append("Raw spectrum not provided — g-C₃N₄ low-wavenumber probe skipped")

    # Context: pure g-C₃N₄ has NO graphene bands
    if not G_present and not D_present:
        ev.append("G and D bands absent — consistent with pure g-C₃N₄")
    elif G_present:
        ev.append("G band present — possible g-C₃N₄ / graphene composite")

    if det_691 and det_988 and not G_present:
        return MaterialHypothesis("g-C₃N₄ (pure)", "likely", ev)
    if det_691 and det_988 and G_present:
        return MaterialHypothesis("g-C₃N₄ / graphene composite", "likely", ev)
    if (det_691 or det_988) and not G_present:
        return MaterialHypothesis("g-C₃N₄ (pure)", "possible", ev)
    return MaterialHypothesis("g-C₃N₄", "not_detected", ev)


# ─────────────────────────────────────────────────────────
# Rule set 6 — Amorphous carbon / Stage 2
# ─────────────────────────────────────────────────────────
def _rule_amorphous(
    peaks:    dict[str, PeakResult],
    analysis: RamanAnalysis,
) -> MaterialHypothesis:
    """
    Stage 2 → amorphous carbon [Ferrari & Robertson 2001, PRB 64, 075414]:
      CONFIRMED : disorder_stage contains 'Stage 2' AND FWHM(G) > 80 cm⁻¹
      LIKELY    : disorder_stage contains 'Stage 2'
    """
    ev: List[str] = []
    G = peaks.get("G")

    is_stage2 = "Stage 2" in analysis.disorder_stage

    if not is_stage2:
        ev.append(f"Disorder stage: {analysis.disorder_stage}")
        return MaterialHypothesis("Amorphous carbon", "not_detected", ev)

    ev.append(f"Stage: {analysis.disorder_stage}")
    if _found(G) and G.fwhm > 80:
        ev.append(f"FWHM(G) = {G.fwhm:.1f} cm⁻¹ > 80 — broad amorphous G band")
        return MaterialHypothesis("Amorphous carbon", "confirmed", ev)

    if _found(G):
        ev.append(f"FWHM(G) = {G.fwhm:.1f} cm⁻¹")
    return MaterialHypothesis("Amorphous carbon", "likely", ev)


# ─────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────
def classify(
    peaks:       dict[str, PeakResult],
    analysis:    RamanAnalysis,
    wavenumbers: Optional[np.ndarray] = None,
    intensities: Optional[np.ndarray] = None,
) -> List[MaterialHypothesis]:
    """
    Run all rule sets and return a sorted list of MaterialHypothesis objects.

    Parameters
    ----------
    peaks       : dict of PeakResult from peak_fitter.fit_all_peaks()
    analysis    : RamanAnalysis from analyzer.analyze()
    wavenumbers : optional raw wavenumber array (enables g-C₃N₄ probe)
    intensities : optional baseline-corrected intensity array

    Returns
    -------
    List of MaterialHypothesis, sorted by confidence descending.
    Results with confidence 'not_detected' are included so the UI
    can show a complete checklist.
    """
    hypotheses = [
        _rule_pristine_graphene(peaks, analysis),
        _rule_boron_doped(analysis),
        _rule_n_doped(peaks, analysis),
        _rule_rgo(peaks, analysis),
        _rule_g_c3n4(peaks, analysis, wavenumbers, intensities),
        _rule_amorphous(peaks, analysis),
    ]
    # Sort: confirmed first, then by material name for stable ordering
    hypotheses.sort(key=lambda h: (-h.rank, h.material))
    return hypotheses


def format_fingerprint_report(hypotheses: List[MaterialHypothesis]) -> str:
    """
    Plain-text summary of the material fingerprint classification.
    Suitable for appending to the main Raman report.
    """
    sep = "═" * 64
    icons = {
        "confirmed":    "✅",
        "likely":       "⚠️ ",
        "possible":     "❓",
        "not_detected": "❌",
    }
    lines = [sep, "  MATERIAL FINGERPRINT", sep]
    for h in hypotheses:
        icon = icons.get(h.confidence, "  ")
        lines.append(f"  {icon}  {h.material:<40}  [{h.confidence}]")
    lines.append("")
    lines.append("  Evidence detail:")
    for h in hypotheses:
        if h.confidence in ("confirmed", "likely", "possible"):
            lines.append(f"  ── {h.material}")
            for e in h.evidence:
                lines.append(f"     • {e}")
    lines += [
        "",
        "  ⚠ Raman alone cannot uniquely identify a material.",
        "    'confirmed'/'likely' indicate strong spectral evidence;",
        "    independent XPS/EDX/TEM is recommended for definitive ID.",
        sep,
    ]
    return "\n".join(lines)
