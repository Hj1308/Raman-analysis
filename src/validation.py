"""
validation.py — Post-fit quality-control and reliability warnings.

Why this module exists
----------------------
A fit can converge and still be scientifically untrustworthy: a peak can
pin to the edge of its allowed center range, a band can go undetected in a
very disordered spectrum, or a derived quantity (L_D via Cançado) can be
reported outside its regime of validity (Stage 2). None of these are caught
by R² alone. This layer inspects the fit + analysis outputs and emits
structured warnings so the UI/report can tell the user *how much to trust*
each number — rather than presenting every value with false confidence.

It is deliberately read-only: it never re-fits or mutates results. It takes
the dict[str, PeakResult] from peak_fitter.fit_all_peaks() and the
RamanAnalysis from analyzer.analyze(), and returns a ValidationReport.

Depends only on the project's own dataclasses + numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import numpy as np

# Center bounds are owned by peak_fitter; import them so "stuck to edge"
# detection stays in sync with the actual fit constraints.
try:
    from .peak_fitter import (
        _G_CENTER_MIN, _G_CENTER_MAX,
        _DP_CENTER_MIN, _DP_CENTER_MAX,
    )
except Exception:  # pragma: no cover - allow standalone import in tests
    _G_CENTER_MIN, _G_CENTER_MAX = 1555.0, 1605.0
    _DP_CENTER_MIN, _DP_CENTER_MAX = 1600.0, 1640.0


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class ValidationFlag:
    code: str
    severity: Severity
    message: str
    band: Optional[str] = None

    def __str__(self) -> str:
        b = f"[{self.band}] " if self.band else ""
        return f"{self.severity.value.upper()}: {b}{self.message}"


@dataclass
class ValidationReport:
    flags: List[ValidationFlag] = field(default_factory=list)

    def add(self, code, severity, message, band=None) -> None:
        self.flags.append(ValidationFlag(code, severity, message, band))

    @property
    def ok(self) -> bool:
        """True if nothing worse than INFO was raised."""
        return not any(f.severity != Severity.INFO for f in self.flags)

    @property
    def has_critical(self) -> bool:
        return any(f.severity == Severity.CRITICAL for f in self.flags)

    def by_severity(self, severity: Severity) -> List[ValidationFlag]:
        return [f for f in self.flags if f.severity == severity]

    def summary(self) -> str:
        if not self.flags:
            return "No validation issues detected."
        return "\n".join(str(f) for f in self.flags)


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #
_EDGE_TOL = 2.0        # cm⁻¹: how close to a bound counts as "pinned"
_LOW_R2 = 0.90         # global-fit R² below this is suspect
_MIN_SNR = 3.0         # a "found" peak below this SNR is marginal


def _pinned(value, lo, hi, tol=_EDGE_TOL) -> Optional[str]:
    """Return 'lower'/'upper' if value sits within tol of a bound, else None."""
    if value is None or not np.isfinite(value):
        return None
    if abs(value - lo) <= tol:
        return "lower"
    if abs(value - hi) <= tol:
        return "upper"
    return None


def _check_band_center_pinned(report, peaks) -> None:
    """Flag G or D' pinned to the edge of their allowed center window —
    the classic symptom of band swapping / an ill-constrained fit."""
    G = peaks.get("G")
    if G is not None and getattr(G, "found", False):
        edge = _pinned(G.center, _G_CENTER_MIN, _G_CENTER_MAX)
        if edge:
            report.add(
                "G_center_pinned", Severity.WARNING,
                f"G center ({G.center:.1f} cm⁻¹) is pinned to the {edge} bound "
                f"of its allowed range [{_G_CENTER_MIN:.0f}, {_G_CENTER_MAX:.0f}]. "
                "The G band may be poorly resolved (heavy D/G/D′ overlap); "
                "manual inspection recommended.",
                band="G",
            )
    Dp = peaks.get("D_prime")
    if Dp is not None and getattr(Dp, "found", False):
        edge = _pinned(Dp.center, _DP_CENTER_MIN, _DP_CENTER_MAX)
        if edge:
            report.add(
                "Dprime_center_pinned", Severity.WARNING,
                f"D′ center ({Dp.center:.1f} cm⁻¹) is pinned to the {edge} bound; "
                "the D′ assignment may be unreliable.",
                band="D_prime",
            )


def _check_core_bands_found(report, peaks) -> None:
    """D and G are the anchors for every intensity ratio. Missing/marginal
    core bands make I_D/I_G and everything derived from it untrustworthy."""
    for name in ("D", "G"):
        pk = peaks.get(name)
        if pk is None or not getattr(pk, "found", False):
            report.add(
                f"{name}_not_found", Severity.CRITICAL,
                f"{name} band was not confidently detected. Intensity ratios "
                f"involving {name} (e.g. I_D/I_G) are unreliable for this "
                "spectrum — it may be too disordered or too noisy for a "
                "three-band model.",
                band=name,
            )
        elif getattr(pk, "snr", None) is not None and np.isfinite(pk.snr) \
                and pk.snr < _MIN_SNR:
            report.add(
                f"{name}_low_snr", Severity.WARNING,
                f"{name} band detected but low SNR ({pk.snr:.1f}); "
                "treat derived ratios with caution.",
                band=name,
            )


def _check_fit_quality(report, peaks) -> None:
    """Low R² on the D/G region means the line-shape model doesn't describe
    the data — often the cue to switch Lorentzian → pseudo-Voigt."""
    D = peaks.get("D")
    if D is not None and getattr(D, "found", False):
        r2 = getattr(D, "r_squared", None)
        if r2 is not None and np.isfinite(r2) and r2 < _LOW_R2:
            report.add(
                "low_global_r2", Severity.WARNING,
                f"D/G global-fit R² = {r2:.3f} < {_LOW_R2:.2f}. The chosen "
                "line shape may not fit well; if this is a disordered material "
                "(GO/rGO/g-C₃N₄), try adaptive_lineshape='pseudo_voigt'.",
                band="D",
            )


def _check_LD_regime(report, analysis) -> None:
    """The Cançado L_D formula is Stage-1 only. If L_D was reported while the
    spectrum sits in Stage 2 (or the transition), warn that the number is out
    of its validity regime."""
    stage = getattr(analysis, "disorder_stage", "") or ""
    L_D = getattr(analysis, "L_D_nm", np.nan)
    has_LD = L_D is not None and np.isfinite(L_D)
    if has_LD and ("Stage 2" in stage or "transition" in stage.lower()):
        report.add(
            "LD_out_of_regime", Severity.WARNING,
            f"L_D = {L_D:.1f} nm was computed, but the spectrum is classified "
            f"as '{stage}'. The Cançado L_D relation is valid only in Stage 1 "
            "(low-defect). Interpret L_D with caution or treat it as a lower "
            "bound.",
        )


def _check_ratio_measure_consistency(report, analysis) -> None:
    """Guard against comparing a height ratio where an area ratio is required.
    We surface both so the user knows which is which."""
    h = getattr(analysis, "ID_IG_height", np.nan)
    a = getattr(analysis, "ID_IG_area", np.nan)
    if (h is not None and np.isfinite(h)) and (a is not None and np.isfinite(a)):
        if a > 0 and abs(h - a) / a > 0.5:
            report.add(
                "height_area_divergence", Severity.INFO,
                f"I_D/I_G differs markedly by measure (height={h:.2f}, "
                f"area={a:.2f}). Cançado L_D uses the AREA ratio; Ferrari/"
                "Eckmann thresholds typically use HEIGHT. Make sure downstream "
                "comparisons use a consistent measure.",
            )


def _check_2D_warning(report, analysis) -> None:
    """Surface the analyzer's own 2D/layer caveat as a formal flag."""
    w = getattr(analysis, "twoD_fwhm_warning", "") or ""
    if w:
        report.add("twoD_fwhm", Severity.INFO,
                   f"2D-band caveat: {w}", band="2D")


def _check_against_literature(report, peaks, laser_nm=None) -> None:
    """Cross-check fitted band positions against literature ranges (Fix #8).

    Uses the knowledge base to see whether a fitted G/D/2D position falls far
    outside the span reported in the literature. A large deviation is flagged
    as INFO (not an error — it may be genuine strain/doping), pointing the user
    to double-check. Silently does nothing if the knowledge base is absent.
    """
    try:
        from . import knowledge as _kb
    except Exception:
        try:
            import knowledge as _kb
        except Exception:
            return
    try:
        kb = _kb.active()
    except Exception:
        return
    if len(kb) == 0:
        return

    checks = [("G", "pos_G"), ("D", "pos_D"), ("2D", "pos_2D")]
    for band, metric in checks:
        pk = peaks.get(band) if band != "2D" else peaks.get("2D")
        if pk is None or not getattr(pk, "found", False):
            continue
        center = getattr(pk, "center", None)
        if center is None:
            continue
        lo, hi, sources = kb.reference_range(metric, laser_nm=laser_nm)
        if lo is None or not sources:
            continue
        # generous tolerance: positions can shift with strain/doping/substrate
        margin = 30.0  # cm^-1
        if center < lo - margin or center > hi + margin:
            report.add(
                f"{band}_pos_off_literature", Severity.INFO,
                f"{band} position ({center:.0f} cm\u207b\u00b9) is outside the "
                f"literature span for {metric} ({lo:.0f}\u2013{hi:.0f} cm\u207b\u00b9, "
                f"{len(sources)} refs). May reflect genuine strain/doping/"
                f"substrate effects, or a fit/calibration issue worth checking.",
                band=band,
            )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def validate(peaks, analysis, laser_nm=None) -> ValidationReport:
    """Run all quality-control checks over a completed fit + analysis.

    Parameters
    ----------
    peaks    : dict[str, PeakResult] from peak_fitter.fit_all_peaks()
    analysis : RamanAnalysis from analyzer.analyze()
    laser_nm : optional excitation wavelength; enables literature
               cross-checks that are laser-dependent (Fix #8).

    Returns
    -------
    ValidationReport with a list of ValidationFlag (info/warning/critical).
    Purely diagnostic — nothing is mutated or re-fit.
    """
    report = ValidationReport()
    if peaks is None:
        report.add("no_peaks", Severity.CRITICAL, "No peak results supplied.")
        return report

    _check_core_bands_found(report, peaks)
    _check_band_center_pinned(report, peaks)
    _check_fit_quality(report, peaks)
    _check_against_literature(report, peaks, laser_nm=laser_nm)
    if analysis is not None:
        _check_LD_regime(report, analysis)
        _check_ratio_measure_consistency(report, analysis)
        _check_2D_warning(report, analysis)

    return report
