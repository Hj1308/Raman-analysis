"""
Peak fitting module — pure scipy (no lmfit dependency).
Line shapes per Ferrari & Basko (2013):
  D, G, D', 2D  -> Lorentzian
  D+G           -> Pseudo-Voigt
  2D bilayer    -> dual-Lorentzian if single R² < 0.90

Dispersion at 532 nm:
  D:  0.174 cm⁻¹/nm
  2D: 0.348 cm⁻¹/nm

G-band strategy for doped / disordered graphene:
  1. Detect true G-peak position with find_peaks in a broad search window (1540–1680 cm⁻¹).
  2. Build an adaptive ±50 cm⁻¹ window centred on the detected peak.
  3. If single-Lorentzian R² < 0.60, attempt G+D' dual-Lorentzian deconvolution.
  This handles N-doped / B-doped samples where G and D' overlap or G is near 1600 cm⁻¹.
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks as _sp_find_peaks
from dataclasses import dataclass, field

# ── Peak search windows (cm⁻¹) at 532 nm ─────────────────
PEAK_WINDOWS_532 = {
    "D":       (1270, 1450),
    "G":       (1500, 1650),   # widened — true G can sit near 1600 cm⁻¹
    "D_prime": (1610, 1680),
    "2D":      (2580, 2780),
    "DG":      (2850, 2960),
}

# Broad window used to *locate* the G peak before building adaptive window
_G_SEARCH_LO = 1540.0
_G_SEARCH_HI = 1680.0
_G_HALF_WIDTH = 50.0   # cm⁻¹ each side of detected centre

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
    name:            str
    center:          float      = np.nan
    amplitude:       float      = np.nan   # peak height (a.u.)
    fwhm:            float      = np.nan   # cm⁻¹
    area:            float      = np.nan   # integrated area
    r_squared:       float      = np.nan
    found:           bool       = False
    is_split_2D:     bool       = False
    is_deconvolved:  bool       = False    # True when G was separated from D' by dual-Lorentzian
    deconv_partner:  "PeakResult | None" = field(default=None, repr=False)  # D' component
    model_x:         np.ndarray = field(default_factory=lambda: np.array([]))
    model_y:         np.ndarray = field(default_factory=lambda: np.array([]))


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


# ── G-peak locator ────────────────────────────────────
def _find_G_peak(wn: np.ndarray, intensity: np.ndarray) -> float:
    """
    Return the wavenumber of the most prominent peak in the G-search window.
    Falls back to the argmax if no distinct peak is found.
    """
    mask = (wn >= _G_SEARCH_LO) & (wn <= _G_SEARCH_HI)
    xs, ys = wn[mask], intensity[mask]
    if len(xs) < 4:
        return float(xs[np.argmax(ys)]) if len(xs) else 1590.0

    pk_idx, props = _sp_find_peaks(
        ys,
        height=ys.max() * 0.3,
        distance=3,
        prominence=0.02,
    )
    if len(pk_idx) == 0:
        return float(xs[np.argmax(ys)])

    # pick the most prominent peak
    best = pk_idx[np.argmax(props["prominences"])]
    return float(xs[best])


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
            fwhm  = 2.355 * popt[2]
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


# ── Adaptive G fitter ─────────────────────────────────
def _fit_G_adaptive(wn: np.ndarray, intensity: np.ndarray) -> PeakResult:
    """
    Step 1 — Adaptive single-Lorentzian:
      Locate the G peak, build a ±50 cm⁻¹ window around it, fit.
    Step 2 — Deconvolution fallback:
      If R² < 0.60, try dual-Lorentzian G+D' in a wider window (1480–1700).
    """
    g_centre = _find_G_peak(wn, intensity)
    lo = max(1480.0, g_centre - _G_HALF_WIDTH)
    hi = min(1700.0, g_centre + _G_HALF_WIDTH)

    result = _fit_peak(wn, intensity, lo, hi, "G")

    if result.found:
        return result

    # ── Deconvolution: G + D' dual-Lorentzian ──
    return _fit_G_deconvolve(wn, intensity, g_centre)


def _fit_G_deconvolve(
    wn: np.ndarray,
    intensity: np.ndarray,
    g_centre_hint: float,
) -> PeakResult:
    """
    Dual-Lorentzian deconvolution of overlapping G and D' bands.
    Returns a PeakResult for G with the D' component stored in .deconv_partner.
    """
    lo, hi = 1480.0, 1700.0
    mask = (wn >= lo) & (wn <= hi)
    xd, yd = wn[mask], intensity[mask]

    result = PeakResult(name="G")
    result.model_x = xd
    result.model_y = np.zeros_like(xd)

    if len(xd) < 8:
        return result

    a0 = float(yd.max())
    # G initial: near detected hint; D' ~20 cm⁻¹ above G
    c_G  = float(np.clip(g_centre_hint, 1540.0, 1640.0))
    c_Dp = float(np.clip(c_G + 20.0,   1560.0, 1680.0))

    try:
        p0 = [c_G,  a0 * 0.8, 20.0,
              c_Dp, a0 * 0.4, 15.0]
        bounds = (
            [1480, 0, 3,  1540, 0, 3],
            [1660, np.inf, 100, 1700, np.inf, 80],
        )
        popt, _ = curve_fit(
            _dual_lorentzian, xd, yd,
            p0=p0, bounds=bounds, maxfev=10000,
        )
        y_fit = _dual_lorentzian(xd, *popt)
        r2    = _r2(yd, y_fit)

        # assign G = lower-wavenumber component
        if popt[0] <= popt[3]:
            c_g, a_g, gam_g = popt[0], popt[1], popt[2]
            c_d, a_d, gam_d = popt[3], popt[4], popt[5]
        else:
            c_g, a_g, gam_g = popt[3], popt[4], popt[5]
            c_d, a_d, gam_d = popt[0], popt[1], popt[2]

        d_prime = PeakResult(
            name="D'",
            center=c_d,
            amplitude=a_d,
            fwhm=2.0 * gam_d,
            area=np.pi * a_d * gam_d,
            r_squared=r2,
            found=r2 > 0.60,
            is_deconvolved=True,
            model_x=xd,
            model_y=_lorentzian(xd, c_d, a_d, gam_d),
        )

        result.center          = c_g
        result.amplitude       = a_g
        result.fwhm            = 2.0 * gam_g
        result.area            = np.pi * a_g * gam_g
        result.r_squared       = r2
        result.found           = r2 > 0.60
        result.is_deconvolved  = True
        result.deconv_partner  = d_prime
        result.model_y         = _lorentzian(xd, c_g, a_g, gam_g)

    except Exception:
        pass

    return result


# ── 2D fitter ─────────────────────────────────────────
def _fit_2D(wn, intensity, lo, hi):
    """Attempt single Lorentzian; upgrade to dual if R² < 0.90."""
    single = _fit_peak(wn, intensity, lo, hi, "2D")
    if single.r_squared >= 0.90 or not single.found:
        return single

    mask = (wn >= lo) & (wn <= hi)
    xd, yd = wn[mask], intensity[mask]
    if len(xd) < 8:
        return single

    c0 = float(xd[np.argmax(yd)])
    a0 = float(yd.max())
    g0 = (hi - lo) / 10.0

    try:
        p0     = [c0 - 10, a0 * 0.6, g0, c0 + 10, a0 * 0.4, g0]
        bounds = (
            [lo, 0, 0.5, lo, 0, 0.5],
            [hi, np.inf, (hi-lo)/3, hi, np.inf, (hi-lo)/3],
        )
        popt, _ = curve_fit(_dual_lorentzian, xd, yd, p0=p0,
                            bounds=bounds, maxfev=8000)
        y_fit = _dual_lorentzian(xd, *popt)
        r2    = _r2(yd, y_fit)

        if r2 > single.r_squared:
            if popt[1] >= popt[4]:
                center, amplitude, gamma = popt[0], popt[1], popt[2]
            else:
                center, amplitude, gamma = popt[3], popt[4], popt[5]
            return PeakResult(
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
    except Exception:
        pass

    return single


# ── Public API ─────────────────────────────────────────
def fit_all_peaks(
    wn: np.ndarray,
    intensity: np.ndarray,
    laser_nm: float = 532.0,
) -> dict[str, PeakResult]:
    """
    Fit all Raman peaks for graphene / sp² carbon.

    G-band strategy (doping-aware):
      1. Locate G peak via find_peaks in 1540–1680 cm⁻¹.
      2. Build adaptive ±50 cm⁻¹ window → single Lorentzian.
      3. If R² < 0.60: dual-Lorentzian G+D' deconvolution in 1480–1700 cm⁻¹.
      The deconvolved D' component is stored in results['G'].deconv_partner
      and also promoted to results['D_prime'] if that slot is empty or weaker.

    Returns dict keyed by peak name: 'D', 'G', 'D_prime', '2D', 'DG'.
    """
    windows = get_peak_windows(laser_nm)
    results: dict[str, PeakResult] = {}

    results["D"]  = _fit_peak(wn, intensity, *windows["D"], "D")

    # G — adaptive / deconvolution
    g_result = _fit_G_adaptive(wn, intensity)
    results["G"] = g_result

    # D' — use deconvolved component if available and better than standalone fit
    standalone_dp = _fit_peak(wn, intensity, *windows["D_prime"], "D'")
    if g_result.is_deconvolved and g_result.deconv_partner is not None:
        dp = g_result.deconv_partner
        if not standalone_dp.found or dp.r_squared >= standalone_dp.r_squared:
            results["D_prime"] = dp
        else:
            results["D_prime"] = standalone_dp
    else:
        results["D_prime"] = standalone_dp

    results["2D"]  = _fit_2D(wn, intensity, *windows["2D"])
    results["DG"]  = _fit_peak(wn, intensity, *windows["DG"], "D+G",
                                use_pseudo_voigt=True)

    return results
