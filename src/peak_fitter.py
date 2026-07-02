"""
Peak fitting module — scipy core, optional lmfit for advanced models.
Line shapes per Ferrari & Basko (2013):
  D, G, D', 2D  -> Lorentzian
  D+G           -> Pseudo-Voigt
  2D bilayer    -> dual-Lorentzian if single R² < 0.90

Dispersion (eV-based, Cançado 2011 / Ferrari & Basko 2013):
  D:  53 cm⁻¹/eV   (double-resonance, zone-boundary phonon)
  2D: 100 cm⁻¹/eV  (double-resonance, 2× D phonon)
  Windows are shifted relative to the 532 nm reference:
    shift = dispersion × (E_laser − E_532)
  where E = hc/λ = 1239.841984 / λ_nm  [eV]
  Because E decreases with increasing λ, D and 2D windows move to
  *lower* wavenumbers at 633 nm / 785 nm — physically correct.

G-band strategy for doped / disordered graphene:
  1. Detect true G-peak position with find_peaks in a broad search window (1540–1680 cm⁻¹).
  2. Build an adaptive ±50 cm⁻¹ window centred on the detected peak.
  3. If single-Lorentzian R² < 0.60, attempt G+D' dual-Lorentzian deconvolution.
  This handles N-doped / B-doped samples where G and D' overlap or G is near 1600 cm⁻¹.

band_config (optional):
  Dict keyed by band name ('D','G','D_prime','2D','DG') with entries:
    {
      "method":     "auto"|"adaptive"|"deconvolve"|"lmfit",  # default "auto"
      "model":      "Lorentzian"|"Gaussian"|"Voigt",          # lmfit only
      "asymmetric": False,                                     # lmfit only
      "local_bg":   False,                                     # lmfit only
    }
  If band_config is None or a band key is absent, the original scipy logic runs unchanged.
  "lmfit" method requires lmfit to be installed; falls back to scipy if not available.
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks as _sp_find_peaks
from dataclasses import dataclass, field
from typing import Optional

# ── Physical constant ─────────────────────────────────────
_HC_EV_NM = 1239.841984          # eV·nm  (h·c)

# ── Peak search windows (cm⁻¹) at 532 nm ─────────────────
PEAK_WINDOWS_532 = {
    "D":       (1270, 1450),
    "G":       (1500, 1650),   # widened — true G can sit near 1600 cm⁻¹
    "D_prime": (1610, 1680),
    "2D":      (2580, 2780),
    "DG":      (2850, 2960),
}

# Excitation-energy dispersions (cm⁻¹/eV)
# D  : ~53 cm⁻¹/eV  — Cançado et al., Nano Lett. 11, 3190 (2011)
# 2D : ~100 cm⁻¹/eV — Ferrari & Basko, Nat. Nanotechnol. 8, 235 (2013)
_DISP_D_PER_EV  = 53.0
_DISP_2D_PER_EV = 100.0

# Broad window used to *locate* the G peak before building adaptive window
_G_SEARCH_LO  = 1540.0
_G_SEARCH_HI  = 1680.0
_G_HALF_WIDTH = 50.0   # cm⁻¹ each side of detected centre


def _laser_energy_ev(laser_nm: float) -> float:
    """Convert laser wavelength (nm) to photon energy (eV)."""
    if laser_nm <= 0:
        raise ValueError(f"laser_nm must be positive, got {laser_nm}")
    return _HC_EV_NM / float(laser_nm)


def get_peak_windows(laser_nm: float) -> dict:
    """
    Return Raman peak search windows (cm⁻¹) corrected for excitation-energy
    dispersion of the D and 2D bands.

    Reference windows are defined at 532 nm.  For dispersive bands the shift is:

        shift [cm⁻¹] = dispersion [cm⁻¹/eV] × (E_laser − E_532) [eV]

    Because E_laser decreases as λ increases, D and 2D windows move to
    *lower* wavenumbers for 633 nm and 785 nm excitation — consistent with
    experimental observations (Cançado 2011, Ferrari & Basko 2013).

    G, D' and D+G are non-dispersive (intervalley/zone-centre phonons) and
    their windows are not shifted.

    Parameters
    ----------
    laser_nm : float
        Laser excitation wavelength in nm.

    Returns
    -------
    dict[str, tuple[float, float]]
        Keys: 'D', 'G', 'D_prime', '2D', 'DG'
        Values: (low_wavenumber, high_wavenumber) in cm⁻¹
    """
    e_532   = _laser_energy_ev(532.0)
    e_laser = _laser_energy_ev(laser_nm)
    delta_e = e_laser - e_532          # negative for λ > 532 nm

    windows: dict = {}
    for peak, (lo, hi) in PEAK_WINDOWS_532.items():
        if peak == "D":
            shift = _DISP_D_PER_EV * delta_e
        elif peak == "2D":
            shift = _DISP_2D_PER_EV * delta_e
        else:
            shift = 0.0
        windows[peak] = (lo + shift, hi + shift)
    return windows


# ── lmfit availability check ───────────────────────────────
def _lmfit_available() -> bool:
    try:
        import lmfit  # noqa: F401
        return True
    except ImportError:
        return False


@dataclass
class PeakResult:
    name:            str
    center:          float            = np.nan
    amplitude:       float            = np.nan   # peak height (a.u.)
    fwhm:            float            = np.nan   # cm⁻¹
    area:            float            = np.nan   # integrated area
    r_squared:       float            = np.nan
    found:           bool             = False
    is_split_2D:     bool             = False
    is_deconvolved:  bool             = False    # True when G was separated from D' by dual-Lorentzian
    deconv_partner:  "PeakResult | None" = field(default=None, repr=False)  # D' component
    model_x:         np.ndarray       = field(default_factory=lambda: np.array([]))
    model_y:         np.ndarray       = field(default_factory=lambda: np.array([]))
    # uncertainty fields — populated only by lmfit path
    center_stderr:   Optional[float]  = field(default=None)
    fwhm_stderr:     Optional[float]  = field(default=None)


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


# ── Numerical FWHM (used for asymmetric profiles) ─────────
def compute_fwhm_numerical(x: np.ndarray, y: np.ndarray) -> float:
    """
    Interpolation-based FWHM. Works for any peak shape including asymmetric.
    Returns np.nan if the profile doesn't cross the half-maximum on both sides.
    """
    if len(y) < 3:
        return np.nan
    half_max = float(np.max(y)) / 2.0
    above = y >= half_max
    if not np.any(above):
        return np.nan
    idx = np.where(above)[0]

    # left crossing
    if idx[0] > 0:
        x_left = float(np.interp(
            half_max,
            [y[idx[0] - 1], y[idx[0]]],
            [x[idx[0] - 1], x[idx[0]]],
        ))
    else:
        x_left = float(x[idx[0]])

    # right crossing
    if idx[-1] < len(x) - 1:
        x_right = float(np.interp(
            half_max,
            [y[idx[-1] + 1], y[idx[-1]]],
            [x[idx[-1] + 1], x[idx[-1]]],
        ))
    else:
        x_right = float(x[idx[-1]])

    fwhm = x_right - x_left
    return fwhm if fwhm > 0 else np.nan


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

    best = pk_idx[np.argmax(props["prominences"])]
    return float(xs[best])


# ── Single-peak fitter (scipy) ─────────────────────────────
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

    a0   = float(yd.max())
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

        result.center         = c_g
        result.amplitude      = a_g
        result.fwhm           = 2.0 * gam_g
        result.area           = np.pi * a_g * gam_g
        result.r_squared      = r2
        result.found          = r2 > 0.60
        result.is_deconvolved = True
        result.deconv_partner = d_prime
        result.model_y        = _lorentzian(xd, c_g, a_g, gam_g)

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


# ── lmfit-based single-band fitter ────────────────────
def _fit_single_band_lmfit(
    wn: np.ndarray,
    intensity: np.ndarray,
    band_name: str,
    cfg: dict,
    windows: dict,
) -> PeakResult:
    """
    Fit one band using lmfit.  Supports:
      - model: 'Lorentzian' | 'Gaussian' | 'Voigt'
      - asymmetric: True → custom asymmetric Lorentzian
      - local_bg: True → add LinearModel (only when ALS residual slope is visible)

    Returns PeakResult compatible with the existing dataclass.
    Scientific note: Gaussian for the G band is physically meaningful only for
    highly disordered / amorphous carbons. A warning is embedded in the name.
    """
    import lmfit
    from lmfit.models import LorentzianModel, GaussianModel, VoigtModel, LinearModel

    lo, hi = windows.get(band_name, (1500, 1650))
    mask   = (wn >= lo) & (wn <= hi)
    xd, yd = wn[mask], intensity[mask]

    empty = PeakResult(name=band_name, found=False, model_x=xd,
                       model_y=np.zeros_like(xd))
    if len(xd) < 5 or yd.max() <= 0:
        return empty

    c0  = float(xd[np.argmax(yd)])
    a0  = float(yd.max())
    w0  = (hi - lo) / 6.0     # initial half-width guess
    asym = cfg.get("asymmetric", False)
    model_type = cfg.get("model", "Lorentzian")
    use_bg     = cfg.get("local_bg", False)

    # ── Build model ────────────────────────────────────
    if asym:
        def _asym_lor(x, center, sigma_l, sigma_r, amplitude):
            sig = np.where(x < center, sigma_l, sigma_r)
            return amplitude * (sig ** 2 / ((x - center) ** 2 + sig ** 2))

        peak_model = lmfit.Model(_asym_lor, prefix="peak_")
        params = peak_model.make_params(
            center    = dict(value=c0, min=lo, max=hi),
            sigma_l   = dict(value=w0, min=1.0, max=(hi-lo)/2),
            sigma_r   = dict(value=w0, min=1.0, max=(hi-lo)/2),
            amplitude = dict(value=a0, min=0),
        )
    else:
        if model_type == "Lorentzian":
            peak_model = LorentzianModel(prefix="peak_")
        elif model_type == "Gaussian":
            peak_model = GaussianModel(prefix="peak_")
        elif model_type == "Voigt":
            peak_model = VoigtModel(prefix="peak_")
        else:
            peak_model = LorentzianModel(prefix="peak_")

        params = peak_model.make_params(
            center    = dict(value=c0, min=lo, max=hi),
            sigma     = dict(value=w0, min=1.0, max=(hi-lo)/2),
            amplitude = dict(value=a0 * w0, min=0),
        )
        if model_type == "Voigt":
            params["peak_gamma"].set(value=w0 * 0.5, min=0.5, max=(hi-lo)/2, vary=True)

    full_model = peak_model
    if use_bg:
        lin = LinearModel(prefix="lin_")
        params.update(lin.make_params(
            slope     = dict(value=0.0),
            intercept = dict(value=float(np.min(yd))),
        ))
        full_model = peak_model + lin

    try:
        result = full_model.fit(yd, params, x=xd)
    except Exception:
        return empty

    if not result.success and result.rsquared < 0.40:
        return empty

    model_y = result.eval(x=xd)
    r2      = float(result.rsquared)

    if asym:
        center    = float(result.params["peak_center"].value)
        amplitude = float(result.params["peak_amplitude"].value)
        fwhm      = compute_fwhm_numerical(xd, model_y)
        c_err     = result.params["peak_center"].stderr
        fwhm_err  = None
    else:
        center    = float(result.params["peak_center"].value)
        fwhm      = compute_fwhm_numerical(xd, model_y)
        c_err     = result.params["peak_center"].stderr
        sig_err   = result.params["peak_sigma"].stderr
        fwhm_err  = (2.0 * sig_err) if sig_err is not None else None
        if model_type == "Lorentzian":
            gamma_v   = float(result.params["peak_sigma"].value)
            amplitude = float(result.params["peak_amplitude"].value) / (np.pi * max(gamma_v, 1e-9))
        elif model_type == "Gaussian":
            sigma_v   = float(result.params["peak_sigma"].value)
            amplitude = float(result.params["peak_amplitude"].value) / (sigma_v * np.sqrt(2 * np.pi))
        else:
            amplitude = float(np.max(model_y))

    display_name = band_name
    if band_name == "G" and model_type == "Gaussian":
        display_name = "G[Gauss!]"

    peak = PeakResult(
        name          = display_name,
        center        = center,
        amplitude     = amplitude,
        fwhm          = fwhm,
        area          = float(np.trapz(model_y, xd)),
        r_squared     = r2,
        found         = r2 > 0.60,
        model_x       = xd,
        model_y       = model_y,
        center_stderr = float(c_err) if c_err is not None else None,
        fwhm_stderr   = float(fwhm_err) if fwhm_err is not None else None,
    )
    return peak


# ── Public API ─────────────────────────────────────────
def fit_all_peaks(
    wn:          np.ndarray,
    intensity:   np.ndarray,
    laser_nm:    float = 532.0,
    band_config: Optional[dict] = None,
) -> dict[str, PeakResult]:
    """
    Fit all Raman peaks for graphene / sp² carbon.

    Parameters
    ----------
    wn, intensity : array_like
        Wavenumber axis and baseline-corrected intensity.
    laser_nm : float
        Excitation wavelength in nm (default 532).
    band_config : dict or None
        Optional per-band fitting configuration.  When None (default) the
        original scipy logic runs unchanged — fully backward compatible.

        Example (only override what you need):
          band_config = {
              "G": {"method": "adaptive"},
              "D": {"method": "lmfit",
                    "model": "Lorentzian",
                    "asymmetric": True,
                    "local_bg": False},
          }

        method values:
          "auto"        — use original scipy logic (default when key absent)
          "adaptive"    — force _fit_G_adaptive (G band only)
          "deconvolve"  — force _fit_G_deconvolve (G band only)
          "lmfit"       — use lmfit backend; falls back to scipy if lmfit absent

    Returns
    -------
    dict keyed by peak name: 'D', 'G', 'D_prime', '2D', 'DG'

    G-band strategy (doping-aware):
      1. Locate G peak via find_peaks in 1540–1680 cm⁻¹.
      2. Build adaptive ±50 cm⁻¹ window → single Lorentzian.
      3. If R² < 0.60: dual-Lorentzian G+D' deconvolution in 1480–1700 cm⁻¹.
      The deconvolved D' component is stored in results['G'].deconv_partner
      and also promoted to results['D_prime'] if that slot is empty or weaker.
    """
    windows  = get_peak_windows(laser_nm)
    band_cfg = band_config or {}
    use_lmfit = _lmfit_available()
    results: dict[str, PeakResult] = {}

    def _method(band: str) -> str:
        cfg = band_cfg.get(band, {})
        m   = cfg.get("method", "auto")
        if m == "lmfit" and not use_lmfit:
            m = "auto"
        return m

    # ── D band ─────────────────────────────────────────
    d_method = _method("D")
    if d_method == "lmfit":
        results["D"] = _fit_single_band_lmfit(wn, intensity, "D",
                                               band_cfg["D"], windows)
    else:
        results["D"] = _fit_peak(wn, intensity, *windows["D"], "D")

    # ── G band ─────────────────────────────────────────
    g_method = _method("G")
    if g_method == "lmfit":
        results["G"] = _fit_single_band_lmfit(wn, intensity, "G",
                                               band_cfg["G"], windows)
        g_result = results["G"]
    elif g_method == "deconvolve":
        g_centre = _find_G_peak(wn, intensity)
        g_result = _fit_G_deconvolve(wn, intensity, g_centre)
        results["G"] = g_result
    else:
        g_result = _fit_G_adaptive(wn, intensity)
        results["G"] = g_result

    # ── D' band ─────────────────────────────────────────
    dp_method = _method("D_prime")
    standalone_dp = _fit_peak(wn, intensity, *windows["D_prime"], "D'")

    if dp_method == "lmfit":
        lmfit_dp = _fit_single_band_lmfit(wn, intensity, "D_prime",
                                           band_cfg["D_prime"], windows)
        if lmfit_dp.found and (not standalone_dp.found or
                                lmfit_dp.r_squared >= standalone_dp.r_squared):
            results["D_prime"] = lmfit_dp
        else:
            results["D_prime"] = standalone_dp
    elif g_result.is_deconvolved and g_result.deconv_partner is not None:
        dp = g_result.deconv_partner
        if not standalone_dp.found or dp.r_squared >= standalone_dp.r_squared:
            results["D_prime"] = dp
        else:
            results["D_prime"] = standalone_dp
    else:
        results["D_prime"] = standalone_dp

    # ── 2D band ─────────────────────────────────────────
    td_method = _method("2D")
    if td_method == "lmfit":
        results["2D"] = _fit_single_band_lmfit(wn, intensity, "2D",
                                                band_cfg["2D"], windows)
    else:
        results["2D"] = _fit_2D(wn, intensity, *windows["2D"])

    # ── D+G band ─────────────────────────────────────────
    dg_method = _method("DG")
    if dg_method == "lmfit":
        results["DG"] = _fit_single_band_lmfit(wn, intensity, "DG",
                                                band_cfg["DG"], windows)
    else:
        results["DG"] = _fit_peak(wn, intensity, *windows["DG"], "D+G",
                                   use_pseudo_voigt=True)

    return results
