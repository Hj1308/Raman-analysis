"""
Peak fitting module using lmfit.
Line shapes per Ferrari & Basko (2013):
  D, G, D' → Lorentzian
  2D       → Lorentzian (monolayer) or dual-Lorentzian (bilayer)
  D+G      → Pseudo-Voigt

Fixes applied:
  - Dispersion coefficients corrected: D~0.174, 2D~0.348 cm⁻¹/nm @ 532 nm
  - G / D_prime window overlap removed (G: 1500–1600, D': 1610–1680)
  - 2D peak: dual-Lorentzian attempted if single R² < 0.90
"""

import numpy as np
from lmfit.models import LorentzianModel, PseudoVoigtModel, ConstantModel
from dataclasses import dataclass, field


# ── Peak search windows (cm⁻¹) at 532 nm ─────────────
# G and D_prime are now non-overlapping.
PEAK_WINDOWS_532 = {
    "D":       (1270, 1450),
    "G":       (1500, 1600),   # FIX: was 1500–1620, overlapped D'
    "D_prime": (1610, 1680),   # FIX: was 1600–1680
    "2D":      (2580, 2780),
    "DG":      (2850, 2960),
}

# Dispersion coefficients in cm⁻¹/nm (Ferrari & Basko 2013):
#   D:  53 cm⁻¹/eV  → ~0.174 cm⁻¹/nm @ 532 nm  (FIX: was 0.3)
#   2D: 106 cm⁻¹/eV → ~0.348 cm⁻¹/nm @ 532 nm  (FIX: was 0.6)
_DISP_D_PER_NM  = 0.174
_DISP_2D_PER_NM = 0.348


def get_peak_windows(laser_nm: float) -> dict:
    """
    Adjust peak search windows for laser wavelength.
    Dispersive peaks: D and 2D only. G, D', D+G are non-dispersive.
    Reference: Ferrari & Basko, Nature Nanotechnology 8, 235 (2013)
    """
    delta_nm = laser_nm - 532.0
    windows  = {}
    for peak, (lo, hi) in PEAK_WINDOWS_532.items():
        if peak == "D":
            shift = _DISP_D_PER_NM * delta_nm
        elif peak == "2D":
            shift = _DISP_2D_PER_NM * delta_nm
        else:
            shift = 0.0
        windows[peak] = (lo + shift, hi + shift)
    return windows


@dataclass
class PeakResult:
    name:        str
    center:      float      = np.nan
    amplitude:   float      = np.nan   # peak height (a.u.)
    fwhm:        float      = np.nan   # cm⁻¹
    area:        float      = np.nan   # integrated area
    r_squared:   float      = np.nan
    found:       bool       = False
    model_x:     np.ndarray = field(default_factory=lambda: np.array([]))
    model_y:     np.ndarray = field(default_factory=lambda: np.array([]))
    is_split_2D: bool       = False    # True when dual-Lorentzian used


def _r_squared(y_data: np.ndarray, y_fit: np.ndarray) -> float:
    ss_res = np.sum((y_data - y_fit) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def fit_lorentzian(wavenumber: np.ndarray,
                   intensity:  np.ndarray,
                   window:     tuple,
                   peak_name:  str) -> PeakResult:
    """
    Fit a single Lorentzian peak.
    lmfit Lorentzian: f(x) = amplitude / (pi * sigma * (1 + ((x-center)/sigma)^2))
      sigma = HWHM  →  FWHM = 2*sigma
      peak height = amplitude / (pi * sigma)
      integrated area = amplitude  (lmfit convention)
    """
    lo, hi = window
    mask   = (wavenumber >= lo) & (wavenumber <= hi)
    xd, yd = wavenumber[mask], intensity[mask]

    result = PeakResult(name=peak_name)
    if len(xd) < 5 or yd.max() < 1:
        return result

    try:
        model  = LorentzianModel() + ConstantModel()
        center_guess = xd[np.argmax(yd)]
        sigma_guess  = (hi - lo) / 8.0
        params = model.make_params(
            center    = dict(value=center_guess, min=lo, max=hi),
            amplitude = dict(value=yd.max() * np.pi * sigma_guess, min=0),
            sigma     = dict(value=sigma_guess, min=1.0, max=(hi - lo) / 2),
            c         = dict(value=0, min=0)
        )
        fit = model.fit(yd, params, x=xd)
        p   = fit.params
        sigma_val = float(p["sigma"].value)
        amp_val   = float(p["amplitude"].value)

        result.center    = float(p["center"].value)
        result.fwhm      = 2.0 * sigma_val
        result.amplitude = amp_val / (np.pi * sigma_val)   # peak height
        result.area      = amp_val                          # integrated area
        result.r_squared = _r_squared(yd, fit.best_fit)
        result.found     = result.r_squared > 0.70
        result.model_x   = xd
        result.model_y   = fit.best_fit
    except Exception:
        pass
    return result


def fit_2D_peak(wavenumber: np.ndarray,
                intensity:  np.ndarray,
                window:     tuple) -> PeakResult:
    """
    Fit the 2D band.
    Tries single Lorentzian first (monolayer/few-layer).
    If R² < 0.90, attempts dual-Lorentzian (AB-stacked bilayer).
    Reference: Ferrari et al. (2006) Phys. Rev. Lett. 97, 187401
    """
    single = fit_lorentzian(wavenumber, intensity, window, "2D")
    if single.r_squared >= 0.90:
        return single

    lo, hi = window
    mask   = (wavenumber >= lo) & (wavenumber <= hi)
    xd, yd = wavenumber[mask], intensity[mask]
    if len(xd) < 10:
        return single

    try:
        m1     = LorentzianModel(prefix="p1_")
        m2     = LorentzianModel(prefix="p2_")
        model  = m1 + m2 + ConstantModel()
        mid    = (lo + hi) / 2.0
        sg     = (hi - lo) / 10.0
        params = model.make_params(
            p1_center    = dict(value=mid - 20, min=lo,  max=mid),
            p1_amplitude = dict(value=yd.max() * np.pi * sg, min=0),
            p1_sigma     = dict(value=sg, min=1.0, max=(hi - lo) / 3),
            p2_center    = dict(value=mid + 20, min=mid, max=hi),
            p2_amplitude = dict(value=yd.max() * np.pi * sg * 0.5, min=0),
            p2_sigma     = dict(value=sg, min=1.0, max=(hi - lo) / 3),
            c            = dict(value=0, min=0)
        )
        fit     = model.fit(yd, params, x=xd)
        r2_dual = _r_squared(yd, fit.best_fit)

        if r2_dual > single.r_squared:
            p   = fit.params
            a1  = float(p["p1_amplitude"].value)
            a2  = float(p["p2_amplitude"].value)
            dom = "p1_" if a1 >= a2 else "p2_"
            s_d = float(p[f"{dom}sigma"].value)
            return PeakResult(
                name        = "2D",
                center      = float(p[f"{dom}center"].value),
                fwhm        = 2.0 * s_d,
                amplitude   = float(p[f"{dom}amplitude"].value) / (np.pi * s_d),
                area        = a1 + a2,
                r_squared   = r2_dual,
                found       = r2_dual > 0.70,
                model_x     = xd,
                model_y     = fit.best_fit,
                is_split_2D = True,
            )
    except Exception:
        pass
    return single


def fit_pseudo_voigt(wavenumber: np.ndarray,
                     intensity:  np.ndarray,
                     window:     tuple,
                     peak_name:  str) -> PeakResult:
    """Fit a Pseudo-Voigt peak (D+G combination band)."""
    lo, hi = window
    mask   = (wavenumber >= lo) & (wavenumber <= hi)
    xd, yd = wavenumber[mask], intensity[mask]

    result = PeakResult(name=peak_name)
    if len(xd) < 5 or yd.max() < 1:
        return result

    try:
        model  = PseudoVoigtModel() + ConstantModel()
        params = model.make_params(
            center    = dict(value=xd[np.argmax(yd)], min=lo, max=hi),
            amplitude = dict(value=yd.max(), min=0),
            sigma     = dict(value=(hi - lo) / 8, min=1.0, max=(hi - lo) / 2),
            fraction  = dict(value=0.5, min=0, max=1),
            c         = dict(value=0, min=0)
        )
        fit = model.fit(yd, params, x=xd)
        p   = fit.params
        result.center    = float(p["center"].value)
        result.fwhm      = 2.0 * float(p["sigma"].value)
        result.amplitude = float(p["amplitude"].value)
        result.area      = float(np.trapz(fit.best_fit, xd))
        result.r_squared = _r_squared(yd, fit.best_fit)
        result.found     = result.r_squared > 0.65
        result.model_x   = xd
        result.model_y   = fit.best_fit
    except Exception:
        pass
    return result


def fit_all_peaks(wavenumber: np.ndarray,
                  intensity:  np.ndarray,
                  laser_nm:   float = 532.0) -> dict[str, PeakResult]:
    """Fit all graphene/sp² carbon Raman peaks."""
    windows = get_peak_windows(laser_nm)
    return {
        "D":       fit_lorentzian(wavenumber, intensity, windows["D"],       "D"),
        "G":       fit_lorentzian(wavenumber, intensity, windows["G"],       "G"),
        "D_prime": fit_lorentzian(wavenumber, intensity, windows["D_prime"], "D'"),
        "2D":      fit_2D_peak(wavenumber, intensity, windows["2D"]),
        "DG":      fit_pseudo_voigt(wavenumber, intensity, windows["DG"],    "D+G"),
    }
