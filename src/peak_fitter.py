"""
Peak fitting module — pure scipy (no lmfit dependency).
Line shapes per Ferrari & Basko (2013):
  D, G, D', 2D  -> Lorentzian
  D+G           -> Pseudo-Voigt
  2D bilayer    -> dual-Lorentzian if single R² < 0.90

Dispersion at 532 nm:
  D:  0.174 cm⁻¹/nm
  2D: 0.348 cm⁻¹/nm
"""

import numpy as np
from scipy.optimize import curve_fit
from dataclasses import dataclass, field

# ── Peak search windows (cm⁻¹) at 532 nm ─────────────────
PEAK_WINDOWS_532 = {
    "D":       (1270, 1450),
    "G":       (1500, 1600),
    "D_prime": (1610, 1680),
    "2D":      (2580, 2780),
    "DG":      (2850, 2960),
}

_DISP_D_PER_NM  = 0.174
_DISP_2D_PER_NM = 0.348


def get_peak_windows(laser_nm: float) -> dict:
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
    is_split_2D: bool       = False
    model_x:     np.ndarray = field(default_factory=lambda: np.array([]))
    model_y:     np.ndarray = field(default_factory=lambda: np.array([]))


# ── Line shapes ───────────────────────────────────────
def _lorentzian(x, center, amplitude, gamma):
    """Lorentzian: amplitude = peak height, gamma = half-width at half-max."""
    return amplitude / (1.0 + ((x - center) / gamma) ** 2)


def _dual_lorentzian(x, c1, a1, g1, c2, a2, g2):
    return _lorentzian(x, c1, a1, g1) + _lorentzian(x, c2, a2, g2)


def _pseudo_voigt(x, center, amplitude, sigma, eta):
    """Pseudo-Voigt: linear mix of Gaussian and Lorentzian."""
    eta = np.clip(eta, 0.0, 1.0)
    gauss = amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)
    loren = amplitude / (1.0 + ((x - center) / sigma) ** 2)
    return eta * loren + (1 - eta) * gauss


def _r2(y_obs, y_fit):
    ss_res = np.sum((y_obs - y_fit) ** 2)
    ss_tot = np.sum((y_obs - y_obs.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


# ── Single-peak fitter ─────────────────────────────────
def _fit_peak(wn, intensity, lo, hi, name, use_pseudo_voigt=False):
    mask = (wn >= lo) & (wn <= hi)
    xd   = wn[mask]
    yd   = intensity[mask]

    result = PeakResult(name=name)
    result.model_x = xd
    result.model_y = np.zeros_like(xd)

    if len(xd) < 5 or yd.max() <= 0:
        return result

    c0 = float(xd[np.argmax(yd)])
    a0 = float(yd.max())
    g0 = (hi - lo) / 6.0

    try:
        if use_pseudo_voigt:
            p0     = [c0, a0, g0, 0.5]
            bounds = ([lo, 0, 1.0, 0.0], [hi, np.inf, (hi-lo)/2, 1.0])
            popt, _ = curve_fit(_pseudo_voigt, xd, yd, p0=p0,
                                bounds=bounds, maxfev=5000)
            y_fit = _pseudo_voigt(xd, *popt)
            fwhm  = 2.355 * popt[2]   # Gaussian FWHM approx
            area  = popt[1] * popt[2] * np.sqrt(2 * np.pi)
            center, amplitude = popt[0], popt[1]
        else:
            p0     = [c0, a0, g0]
            bounds = ([lo, 0, 0.5], [hi, np.inf, (hi-lo)/2])
            popt, _ = curve_fit(_lorentzian, xd, yd, p0=p0,
                                bounds=bounds, maxfev=5000)
            y_fit  = _lorentzian(xd, *popt)
            center, amplitude, gamma = popt
            fwhm   = 2.0 * gamma
            area   = np.pi * amplitude * gamma

        r2 = _r2(yd, y_fit)

        result.center    = center
        result.amplitude = amplitude
        result.fwhm      = fwhm
        result.area      = area
        result.r_squared = r2
        result.found     = r2 > 0.60
        result.model_y   = y_fit

    except Exception:
        pass

    return result


def _fit_2D(wn, intensity, lo, hi):
    """Attempt single Lorentzian; upgrade to dual if R² < 0.90."""
    single = _fit_peak(wn, intensity, lo, hi, "2D")
    if single.r_squared >= 0.90 or not single.found:
        return single

    # try dual-Lorentzian
    mask = (wn >= lo) & (wn <= hi)
    xd, yd = wn[mask], intensity[mask]
    if len(xd) < 8:
        return single

    mid = (lo + hi) / 2.0
    c0 = float(xd[np.argmax(yd)])
    a0 = float(yd.max())
    g0 = (hi - lo) / 10.0

    try:
        p0     = [c0 - 10, a0 * 0.6, g0, c0 + 10, a0 * 0.4, g0]
        bounds = (
            [lo, 0, 0.5, lo, 0, 0.5],
            [hi, np.inf, (hi-lo)/3, hi, np.inf, (hi-lo)/3]
        )
        popt, _ = curve_fit(_dual_lorentzian, xd, yd, p0=p0,
                            bounds=bounds, maxfev=8000)
        y_fit = _dual_lorentzian(xd, *popt)
        r2    = _r2(yd, y_fit)

        if r2 > single.r_squared:
            # use the dominant peak as the representative result
            if popt[1] >= popt[4]:
                center, amplitude, gamma = popt[0], popt[1], popt[2]
            else:
                center, amplitude, gamma = popt[3], popt[4], popt[5]
            dual = PeakResult(
                name="2D",
                center=center,
                amplitude=amplitude,
                fwhm=2.0 * gamma,
                area=np.pi * amplitude * gamma,
                r_squared=r2,
                found=r2 > 0.60,
                is_split_2D=True,
                model_x=xd,
                model_y=y_fit,
            )
            return dual
    except Exception:
        pass

    return single


# ── Public API ─────────────────────────────────────────
def fit_all_peaks(wn: np.ndarray,
                 intensity: np.ndarray,
                 laser_nm: float = 532.0) -> dict[str, PeakResult]:
    """
    Fit all Raman peaks for graphene/sp² carbon.
    Returns dict keyed by peak name.
    """
    windows = get_peak_windows(laser_nm)
    results = {}

    results["D"]        = _fit_peak(wn, intensity, *windows["D"],       "D")
    results["G"]        = _fit_peak(wn, intensity, *windows["G"],       "G")
    results["D_prime"]  = _fit_peak(wn, intensity, *windows["D_prime"], "D'")
    results["2D"]       = _fit_2D(wn, intensity,   *windows["2D"])
    results["DG"]       = _fit_peak(wn, intensity, *windows["DG"],      "D+G",
                                    use_pseudo_voigt=True)

    return results
