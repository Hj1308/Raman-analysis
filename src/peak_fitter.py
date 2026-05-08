"""
Peak fitting module using lmfit.
Line shapes per Ferrari & Basko (2013):
  D, G, D' → Lorentzian
  2D       → Lorentzian (monolayer) or multi-Lorentzian
  D+G      → Pseudo-Voigt
"""

import numpy as np
from lmfit.models import LorentzianModel, PseudoVoigtModel, ConstantModel
from dataclasses import dataclass, field
from typing import Optional


# ── Peak search windows (cm⁻¹) ────────────────────
# These shift with laser wavelength (dispersive peaks: D, 2D)
# Dispersion: D ~ 53 cm⁻¹/eV, 2D ~ 106 cm⁻¹/eV
PEAK_WINDOWS_532 = {
    "D":   (1270, 1450),
    "G":   (1500, 1620),
    "D_prime": (1600, 1680),
    "2D":  (2580, 2780),
    "DG":  (2850, 2960),
}


def get_peak_windows(laser_nm: float) -> dict:
    """
    Adjust peak search windows based on laser wavelength.
    Reference wavelength: 532 nm
    Dispersion coefficients (cm⁻¹/nm):
      D:  ~0.3 cm⁻¹/nm, 2D: ~0.6 cm⁻¹/nm
    """
    delta_nm = laser_nm - 532.0
    windows  = {}
    for peak, (lo, hi) in PEAK_WINDOWS_532.items():
        if peak == "D":
            shift = 0.3 * delta_nm
        elif peak == "2D":
            shift = 0.6 * delta_nm
        else:
            shift = 0.0
        windows[peak] = (lo + shift, hi + shift)
    return windows


@dataclass
class PeakResult:
    name:      str
    center:    float          = np.nan   # cm⁻¹
    amplitude: float          = np.nan   # peak height
    fwhm:      float          = np.nan   # cm⁻¹
    area:      float          = np.nan   # integrated area
    r_squared: float          = np.nan
    found:     bool           = False
    model_x:   np.ndarray     = field(default_factory=lambda: np.array([]))
    model_y:   np.ndarray     = field(default_factory=lambda: np.array([]))


def _r_squared(y_data: np.ndarray, y_fit: np.ndarray) -> float:
    ss_res = np.sum((y_data - y_fit) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def fit_lorentzian(wavenumber: np.ndarray,
                   intensity:  np.ndarray,
                   window:     tuple,
                   peak_name:  str) -> PeakResult:
    """Fit a single Lorentzian peak within a wavenumber window."""
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
            sigma     = dict(value=sigma_guess, min=1.0, max=(hi-lo)/2),
            c         = dict(value=0, min=0)
        )
        fit = model.fit(yd, params, x=xd)
        p   = fit.params

        result.center    = float(p["center"].value)
        result.fwhm      = float(2 * p["sigma"].value)   # FWHM = 2σ for Lorentzian
        result.amplitude = float(p["amplitude"].value / (np.pi * p["sigma"].value))
        result.area      = float(p["amplitude"].value)
        result.r_squared = _r_squared(yd, fit.best_fit)
        result.found     = result.r_squared > 0.70
        result.model_x   = xd
        result.model_y   = fit.best_fit
    except Exception:
        pass
    return result


def fit_pseudo_voigt(wavenumber: np.ndarray,
                     intensity:  np.ndarray,
                     window:     tuple,
                     peak_name:  str) -> PeakResult:
    """Fit a single Pseudo-Voigt peak (for D+G band)."""
    lo, hi = window
    mask   = (wavenumber >= lo) & (wavenumber <= hi)
    xd, yd = wavenumber[mask], intensity[mask]

    result = PeakResult(name=peak_name)
    if len(xd) < 5 or yd.max() < 1:
        return result

    try:
        model  = PseudoVoigtModel() + ConstantModel()
        center_guess = xd[np.argmax(yd)]
        params = model.make_params(
            center    = dict(value=center_guess, min=lo, max=hi),
            amplitude = dict(value=yd.max(), min=0),
            sigma     = dict(value=(hi-lo)/8, min=1.0, max=(hi-lo)/2),
            fraction  = dict(value=0.5, min=0, max=1),
            c         = dict(value=0, min=0)
        )
        fit = model.fit(yd, params, x=xd)
        p   = fit.params

        result.center    = float(p["center"].value)
        result.fwhm      = float(2 * p["sigma"].value)
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
    """
    Fit all graphene Raman peaks and return results dict.
    """
    windows = get_peak_windows(laser_nm)
    results = {}

    results["D"]        = fit_lorentzian(wavenumber, intensity, windows["D"],       "D")
    results["G"]        = fit_lorentzian(wavenumber, intensity, windows["G"],       "G")
    results["D_prime"]  = fit_lorentzian(wavenumber, intensity, windows["D_prime"], "D'")
    results["2D"]       = fit_lorentzian(wavenumber, intensity, windows["2D"],      "2D")
    results["DG"]       = fit_pseudo_voigt(wavenumber, intensity, windows["DG"],    "D+G")

    return results
