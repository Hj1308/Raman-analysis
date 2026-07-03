"""
Peak fitting module — scipy core, optional lmfit for advanced models.
Line shapes per Ferrari & Basko (2013):
  D, G, D', 2D  -> Lorentzian
  D+G           -> Pseudo-Voigt
  2D bilayer    -> 4-Lorentzian (Ferrari 2006) when single R² < 0.90

Dispersion (eV-based, Cançado 2011 / Ferrari & Basko 2013):
  D:  53 cm⁻¹/eV   (double-resonance, zone-boundary phonon)
  D*: 53 cm⁻¹/eV   (same double-resonance origin as D, Lee 2021)
  2D: 100 cm⁻¹/eV  (double-resonance, 2× D phonon)
  Windows are shifted relative to the 532 nm reference:
    shift = dispersion × (E_laser − E_532)
  where E = hc/λ = 1239.841984 / λ_nm  [eV]
  Because E decreases with increasing λ, D, D*, and 2D windows move to
  *lower* wavenumbers at 633 nm / 785 nm — physically correct.

Pseudo-Voigt (D+G band, fix #4)
---------------------------------
Both Gaussian and Lorentzian components share the *same* FWHM
parameter f, so:
  Gaussian half-width  σ_G = f / (2 √(2 ln 2)) = f / 2.3548
  Lorentzian half-width γ  = f / 2
This is the standard Thompson-Cox-Hastings pseudo-Voigt definition.
Area = η · A · π·(f/2) + (1−η) · A · σ_G · √(2π)
FWHM is reported as f directly (not 2.355σ, which was wrong before).

D+G+D′ global fit (fix #3)
---------------------------
D and D′ are fitted simultaneously with G in a single window
(1250–1700 cm⁻¹) using a three-Lorentzian model.  The standalone
independent D′ fit over the 1610–1680 cm⁻¹ window is *removed*
because that window overlaps the G tail and reliably produces a
spurious peak at the G shoulder, contaminating the ID/ID′ ratio
used for Eckmann defect-type analysis.

SNR detection criterion (fix #11)
-----------------------------------
A peak is accepted only when *both* conditions are met:
  1. R² > 0.75  (increased from 0.60 to reduce false positives)
  2. SNR = amplitude / σ_noise > 3.0
where σ_noise is estimated as the MAD (median absolute deviation)
of the residual y − y_fit in the fit window.  This rejects fits
that achieve high R² by tracking noise rather than a real peak.

Fitting uncertainty (v2.4, Feature #3)
----------------------------------------
curve_fit returns pcov (covariance matrix).  We extract:
  center_stderr = √pcov[0,0]    (for Lorentzian: param index 0)
  fwhm_stderr   = √pcov[2,2]×2  (gamma std → FWHM std; ×2 because FWHM=2γ)
for ALL scipy-fitted peaks (D, G, D_prime, 2D, D*, DG).  These are
stored in PeakResult.center_stderr / fwhm_stderr.  If pcov contains
inf (fit not converged / underdetermined), stderr is set to None.

D* band (v2.4, Feature #1)
----------------------------
Window 1080–1230 cm⁻¹ at 532 nm (dispersive: 53 cm⁻¹/eV).
Fitted with a single Lorentzian.  Origin: C–O stretching / sp² C=C
between oxidised regions in rGO/GO (Lee et al. 2021, Carbon 183, 814).
I_D*/I_G is a proxy for C/O ratio in rGO; values > 0.15 suggest
significant residual oxidation.

manual_peak_fwhm (v2.5, public API)
-------------------------------------
User-picked peak centre → numerical FWHM without parametric fit.
See function docstring for full details.  Intended for Jupyter /
CLI workflows where the researcher selects peaks by cursor.

2D bilayer — 4-Lorentzian model (v2.7, Feature #6)
----------------------------------------------------
For AB-stacked bilayer graphene the 2D band splits into four
Lorentzian components due to the splitting of the π bands at K
[Ferrari et al. 2006, Phys. Rev. Lett. 97, 187401]:

  Label   Position (532 nm)   Relative amplitude
  P11     ~2680 cm⁻¹          strongest  (≈1.0)
  P22     ~2695 cm⁻¹          strong     (≈0.7)
  P12     ~2670 cm⁻¹          weak       (≈0.3)
  P21     ~2710 cm⁻¹          weak       (≈0.3)

Constraints encoded in _fit_2D_bilayer_4L():
  - All four centres constrained within ±15 cm⁻¹ of their
    Ferrari 2006 reference offsets relative to the detected
    single-peak centre c0.
  - Amplitudes: A_P12, A_P21 ≤ 0.6 × A_P11 (outer branches weaker).
  - FWHM reported as numerical FWHM of the composite model_y.
  - is_bilayer_4L = True flags this fit in PeakResult.

Fitting cascade for 2D band:
  1. Single Lorentzian (always attempted first)
     → accepted if R² ≥ 0.90
  2. 4-Lorentzian bilayer [Ferrari 2006]
     → attempted if single R² < 0.90
     → accepted if R²_4L > R²_single
  3. Dual-Lorentzian fallback
     → attempted if 4-L fit fails or does not improve R²

G-band strategy for doped / disordered graphene:
  1. Detect true G-peak position with find_peaks in 1540–1680 cm⁻¹.
  2. Build an adaptive ±50 cm⁻¹ window centred on the detected peak.
  3. If R² < 0.75, attempt G+D' dual-Lorentzian deconvolution.
  This handles N-doped / B-doped samples where G and D' overlap.

band_config (optional):
  Dict keyed by band name with entries:
    {
      "method":     "auto"|"adaptive"|"deconvolve"|"lmfit",
      "model":      "Lorentzian"|"Gaussian"|"Voigt",   # lmfit only
      "asymmetric": False,                              # lmfit only
      "local_bg":   False,                              # lmfit only
    }

Python 3.8 compatibility note
------------------------------
All type hints use typing.Dict / typing.List / typing.Tuple / typing.Optional
instead of the built-in generics (dict[...] / list[...]) which require 3.9+.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks as _sp_find_peaks
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple

# ── Physical constant ─────────────────────────────────────
_HC_EV_NM = 1239.841984          # eV·nm  (h·c)

# ── Peak search windows (cm⁻¹) at 532 nm ─────────────────
PEAK_WINDOWS_532: Dict[str, Tuple[float, float]] = {
    "D_star":      (1080, 1230),   # v2.4: C–O / sp² C=C band [Lee 2021]
    "D":           (1270, 1450),
    "G":           (1500, 1650),
    "D_prime":     (1610, 1680),   # kept for lmfit/legacy; not used in global fit
    "2D":          (2580, 2780),
    "DG":          (2850, 2960),
    # Feature #9 — g-C3N4 modes (non-dispersive)
    "CN_triazine": (670, 715),
    "CN_ring_breathing":  (960, 1010),
}

# Excitation-energy dispersions (cm⁻¹/eV)
_DISP_D_PER_EV  = 53.0   # Cançado et al., Nano Lett. 11, 3190 (2011)
_DISP_2D_PER_EV = 100.0  # Ferrari & Basko, Nat. Nanotechnol. 8, 235 (2013)
# D* shares the same double-resonance origin as D → same dispersion
_DISP_DSTAR_PER_EV = 53.0  # Lee et al. (2021) Carbon 183, 814–822

# G search / adaptive window
_G_SEARCH_LO  = 1540.0
_G_SEARCH_HI  = 1680.0
_G_HALF_WIDTH = 50.0     # cm⁻¹ each side of detected centre

# Global D+G+D′ window
_DGDp_LO = 1250.0
_DGDp_HI = 1700.0

# Detection thresholds (fix #11)
_R2_THRESHOLD  = 0.75    # raised from 0.60 to cut false positives
_SNR_THRESHOLD = 3.0     # amplitude / σ_noise

# ── 2D bilayer 4-Lorentzian offsets at 532 nm [Ferrari 2006] ──
# Offsets relative to detected single-peak centre c0
_2D_P11_OFFSET = -5.0    # cm⁻¹  strongest inner branch
_2D_P22_OFFSET = +10.0   # cm⁻¹  second inner branch
_2D_P12_OFFSET = -20.0   # cm⁻¹  outer branch (weaker)
_2D_P21_OFFSET = +25.0   # cm⁻¹  outer branch (weaker)
_2D_BILAYER_R2_THRESHOLD = 0.90  # trigger 4-L fit below this


def _laser_energy_ev(laser_nm: float) -> float:
    if laser_nm <= 0:
        raise ValueError("laser_nm must be positive, got {}".format(laser_nm))
    return _HC_EV_NM / float(laser_nm)


def get_peak_windows(laser_nm: float) -> Dict[str, Tuple[float, float]]:
    """
    Return Raman peak search windows (cm⁻¹) corrected for excitation-energy
    dispersion of the D, D*, and 2D bands.

    shift [cm⁻¹] = dispersion [cm⁻¹/eV] × (E_laser − E_532) [eV]

    G, D′ and D+G are non-dispersive; their windows are not shifted.
    D* is dispersive with the same slope as D (53 cm⁻¹/eV) [Lee 2021].
    """
    e_532   = _laser_energy_ev(532.0)
    e_laser = _laser_energy_ev(laser_nm)
    delta_e = e_laser - e_532          # negative for λ > 532 nm

    windows: Dict[str, Tuple[float, float]] = {}
    for peak, (lo, hi) in PEAK_WINDOWS_532.items():
        if peak in ("D", "D_star"):
            shift = _DISP_D_PER_EV * delta_e
        elif peak == "2D":
            shift = _DISP_2D_PER_EV * delta_e
        else:
            shift = 0.0
        windows[peak] = (lo + shift, hi + shift)
    return windows


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
    amplitude:       float            = np.nan
    fwhm:            float            = np.nan
    area:            float            = np.nan
    r_squared:       float            = np.nan
    snr:             float            = np.nan
    found:           bool             = False
    is_split_2D:     bool             = False
    is_bilayer_4L:   bool             = False
    is_deconvolved:  bool             = False
    deconv_partner:  Optional[object] = field(default=None, repr=False)
    model_x:         np.ndarray       = field(default_factory=lambda: np.array([]))
    model_y:         np.ndarray       = field(default_factory=lambda: np.array([]))
    center_stderr:   Optional[float]  = field(default=None)
    fwhm_stderr:     Optional[float]  = field(default=None)


def _noise_sigma(y_obs: np.ndarray, y_fit: np.ndarray) -> float:
    residuals = y_obs - y_fit
    return float(np.median(np.abs(residuals - np.median(residuals)))) * 1.4826


def _is_detected(
    amplitude: float,
    y_obs: np.ndarray,
    y_fit: np.ndarray,
    r2: float,
) -> Tuple[bool, float]:
    if r2 < _R2_THRESHOLD:
        return False, np.nan
    sigma = _noise_sigma(y_obs, y_fit)
    snr   = amplitude / sigma if sigma > 0 else np.nan
    detected = (not np.isnan(snr)) and (snr >= _SNR_THRESHOLD)
    return detected, snr


def _pcov_stderr(pcov: np.ndarray, idx: int) -> Optional[float]:
    try:
        var = float(pcov[idx, idx])
        return float(np.sqrt(var)) if (np.isfinite(var) and var >= 0) else None
    except (IndexError, TypeError):
        return None


def _lorentzian(x, center, amplitude, gamma):
    return amplitude / (1.0 + ((x - center) / gamma) ** 2)


def _dual_lorentzian(x, c1, a1, g1, c2, a2, g2):
    return _lorentzian(x, c1, a1, g1) + _lorentzian(x, c2, a2, g2)


def _quad_lorentzian(x, c11, a11, g11, c22, a22, g22,
                         c12, a12, g12, c21, a21, g21):
    return (_lorentzian(x, c11, a11, g11)
            + _lorentzian(x, c22, a22, g22)
            + _lorentzian(x, c12, a12, g12)
            + _lorentzian(x, c21, a21, g21))


def _triple_lorentzian(x, cD, aD, gD, cG, aG, gG, cDp, aDp, gDp):
    return (_lorentzian(x, cD, aD, gD)
            + _lorentzian(x, cG, aG, gG)
            + _lorentzian(x, cDp, aDp, gDp))


def _pseudo_voigt(x, center, amplitude, fwhm, eta):
    eta   = float(np.clip(eta, 0.0, 1.0))
    sigma = fwhm / 2.3548206
    gamma = fwhm / 2.0
    gauss = amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)
    loren = amplitude / (1.0 + ((x - center) / gamma) ** 2)
    return eta * loren + (1.0 - eta) * gauss


def _r2(y_obs, y_fit):
    ss_res = np.sum((y_obs - y_fit) ** 2)
    ss_tot = np.sum((y_obs - y_obs.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def compute_fwhm_numerical(x: np.ndarray, y: np.ndarray) -> float:
    if len(y) < 3:
        return np.nan
    half_max = float(np.max(y)) / 2.0
    above    = y >= half_max
    if not np.any(above):
        return np.nan
    idx = np.where(above)[0]
    x_left  = float(np.interp(half_max, [y[idx[0]-1], y[idx[0]]],
                               [x[idx[0]-1], x[idx[0]]])) if idx[0] > 0 else float(x[idx[0]])
    x_right = float(np.interp(half_max, [y[idx[-1]+1], y[idx[-1]]],
                               [x[idx[-1]+1], x[idx[-1]]])) if idx[-1] < len(x)-1 else float(x[idx[-1]])
    fwhm    = x_right - x_left
    return fwhm if fwhm > 0 else np.nan


def manual_peak_fwhm(
    wn:               np.ndarray,
    intensity:        np.ndarray,
    peak_center:      float,
    window_half_width: float = 40.0,
) -> float:
    if wn.shape != intensity.shape:
        raise ValueError(
            "wn and intensity must have the same shape; "
            "got {} vs {}".format(wn.shape, intensity.shape)
        )
    if window_half_width <= 0:
        raise ValueError(
            "window_half_width must be positive; got {}".format(window_half_width)
        )

    lo   = peak_center - window_half_width
    hi   = peak_center + window_half_width
    mask = (wn >= lo) & (wn <= hi)
    xd   = wn[mask]
    yd   = intensity[mask]

    if xd.size < 5 or float(yd.max()) <= 0:
        return float("nan")

    return float(compute_fwhm_numerical(xd, yd))


def _find_G_peak(wn: np.ndarray, intensity: np.ndarray) -> float:
    mask = (wn >= _G_SEARCH_LO) & (wn <= _G_SEARCH_HI)
    xs, ys = wn[mask], intensity[mask]
    if len(xs) < 4:
        return float(xs[np.argmax(ys)]) if len(xs) else 1590.0
    pk_idx, props = _sp_find_peaks(
        ys,
        height=ys.max() * 0.3,
        distance=3,
        prominence=ys.max() * 0.05,
    )
    if len(pk_idx) == 0:
        return float(xs[np.argmax(ys)])
    best = pk_idx[np.argmax(props["prominences"])]
    return float(xs[best])


def _fit_peak(wn, intensity, lo, hi, name, use_pseudo_voigt=False):
    mask = (wn >= lo) & (wn <= hi)
    xd   = wn[mask]
    yd   = intensity[mask]

    result = PeakResult(name=name, model_x=xd, model_y=np.zeros_like(xd))
    if len(xd) < 5 or yd.max() <= 0:
        return result

    c0 = float(xd[np.argmax(yd)])
    a0 = float(yd.max())
    g0 = (hi - lo) / 6.0

    try:
        if use_pseudo_voigt:
            fwhm0  = (hi - lo) / 4.0
            p0     = [c0, a0, fwhm0, 0.5]
            bounds = ([lo, 0, 1.0, 0.0], [hi, np.inf, hi - lo, 1.0])
            popt, pcov = curve_fit(_pseudo_voigt, xd, yd, p0=p0,
                                   bounds=bounds, maxfev=5000)
            y_fit     = _pseudo_voigt(xd, *popt)
            center    = popt[0]
            amplitude = popt[1]
            fwhm      = popt[2]
            eta       = popt[3]
            sigma_g   = fwhm / 2.3548206
            gamma_l   = fwhm / 2.0
            area = (eta * amplitude * np.pi * gamma_l
                    + (1.0 - eta) * amplitude * sigma_g * np.sqrt(2 * np.pi))
            c_std    = _pcov_stderr(pcov, 0)
            fwhm_std = _pcov_stderr(pcov, 2)
        else:
            p0     = [c0, a0, g0]
            bounds = ([lo, 0, 0.5], [hi, np.inf, (hi - lo) / 2])
            popt, pcov = curve_fit(_lorentzian, xd, yd, p0=p0,
                                   bounds=bounds, maxfev=5000)
            y_fit     = _lorentzian(xd, *popt)
            center, amplitude, gamma = popt
            fwhm   = 2.0 * gamma
            area   = np.pi * amplitude * gamma
            c_std    = _pcov_stderr(pcov, 0)
            g_std    = _pcov_stderr(pcov, 2)
            fwhm_std = (2.0 * g_std) if g_std is not None else None

        r2              = _r2(yd, y_fit)
        detected, snr   = _is_detected(amplitude, yd, y_fit, r2)

        result.center         = center
        result.amplitude      = amplitude
        result.fwhm           = fwhm
        result.area           = area
        result.r_squared      = r2
        result.snr            = snr
        result.found          = detected
        result.model_y        = y_fit
        result.center_stderr  = c_std
        result.fwhm_stderr    = fwhm_std

    except Exception:
        pass

    return result


def _fit_G_adaptive(wn: np.ndarray, intensity: np.ndarray) -> PeakResult:
    g_centre = _find_G_peak(wn, intensity)
    lo = max(1480.0, g_centre - _G_HALF_WIDTH)
    hi = min(1700.0, g_centre + _G_HALF_WIDTH)

    result = _fit_peak(wn, intensity, lo, hi, "G")
    if result.found:
        return result
    return _fit_G_deconvolve(wn, intensity, g_centre)


def _fit_G_deconvolve(wn, intensity, g_centre_hint):
    lo, hi = 1480.0, 1700.0
    mask   = (wn >= lo) & (wn <= hi)
    xd, yd = wn[mask], intensity[mask]

    result = PeakResult(name="G", model_x=xd, model_y=np.zeros_like(xd))
    if len(xd) < 8:
        return result

    a0   = float(yd.max())
    c_G  = float(np.clip(g_centre_hint, 1540.0, 1640.0))
    c_Dp = float(np.clip(c_G + 20.0,   1560.0, 1680.0))

    try:
        p0     = [c_G, a0*0.8, 20.0, c_Dp, a0*0.4, 15.0]
        bounds = ([1480,0,3,1540,0,3],[1660,np.inf,100,1700,np.inf,80])
        popt, pcov = curve_fit(_dual_lorentzian, xd, yd,
                                p0=p0, bounds=bounds, maxfev=10000)
        y_fit = _dual_lorentzian(xd, *popt)
        r2    = _r2(yd, y_fit)

        if popt[0] <= popt[3]:
            c_g, a_g, gam_g = popt[0], popt[1], popt[2]
            c_d, a_d, gam_d = popt[3], popt[4], popt[5]
            cg_std = _pcov_stderr(pcov, 0)
            gg_std = _pcov_stderr(pcov, 2)
            cd_std = _pcov_stderr(pcov, 3)
            gd_std = _pcov_stderr(pcov, 5)
        else:
            c_g, a_g, gam_g = popt[3], popt[4], popt[5]
            c_d, a_d, gam_d = popt[0], popt[1], popt[2]
            cg_std = _pcov_stderr(pcov, 3)
            gg_std = _pcov_stderr(pcov, 5)
            cd_std = _pcov_stderr(pcov, 0)
            gd_std = _pcov_stderr(pcov, 2)

        det_g, snr_g = _is_detected(a_g, yd, _lorentzian(xd, c_g, a_g, gam_g), r2)
        det_d, snr_d = _is_detected(a_d, yd, _lorentzian(xd, c_d, a_d, gam_d), r2)

        d_prime = PeakResult(
            name="D'", center=c_d, amplitude=a_d,
            fwhm=2.0*gam_d, area=np.pi*a_d*gam_d,
            r_squared=r2, snr=snr_d, found=det_d,
            is_deconvolved=True, model_x=xd,
            model_y=_lorentzian(xd, c_d, a_d, gam_d),
            center_stderr=cd_std,
            fwhm_stderr=(2.0*gd_std) if gd_std is not None else None,
        )
        result.center         = c_g
        result.amplitude      = a_g
        result.fwhm           = 2.0 * gam_g
        result.area           = np.pi * a_g * gam_g
        result.r_squared      = r2
        result.snr            = snr_g
        result.found          = det_g
        result.is_deconvolved = True
        result.deconv_partner = d_prime
        result.model_y        = _lorentzian(xd, c_g, a_g, gam_g)
        result.center_stderr  = cg_std
        result.fwhm_stderr    = (2.0*gg_std) if gg_std is not None else None
    except Exception:
        pass

    return result


def _fit_D_G_Dp_global(
    wn: np.ndarray,
    intensity: np.ndarray,
    d_lo: float,
    d_hi: float,
) -> Dict[str, PeakResult]:
    lo = max(_DGDp_LO, d_lo - 30)
    hi = _DGDp_HI
    mask   = (wn >= lo) & (wn <= hi)
    xd, yd = wn[mask], intensity[mask]

    empty = {
        "D":       PeakResult(name="D",  model_x=xd, model_y=np.zeros_like(xd)),
        "G":       PeakResult(name="G",  model_x=xd, model_y=np.zeros_like(xd)),
        "D_prime": PeakResult(name="D'", model_x=xd, model_y=np.zeros_like(xd)),
    }
    if len(xd) < 15 or yd.max() <= 0:
        return empty

    a0  = float(yd.max())
    d_mask = (xd >= d_lo) & (xd <= d_hi)
    c_D  = float(xd[d_mask][np.argmax(yd[d_mask])]) if np.any(d_mask) else (d_lo + d_hi) / 2
    c_G  = 1580.0
    c_Dp = 1620.0

    try:
        p0 = [c_D,  a0*0.6, 25.0,
              c_G,  a0,     20.0,
              c_Dp, a0*0.2, 12.0]
        bounds = (
            [d_lo, 0, 3,   1500, 0, 5,  1600, 0, 3],
            [d_hi, np.inf, 80,  1660, np.inf, 80, 1700, np.inf, 60],
        )
        popt, pcov = curve_fit(
            _triple_lorentzian, xd, yd,
            p0=p0, bounds=bounds, maxfev=15000,
        )
        y_fit = _triple_lorentzian(xd, *popt)
        r2    = _r2(yd, y_fit)

        cD, aD, gD    = popt[0], popt[1], popt[2]
        cG, aG, gG    = popt[3], popt[4], popt[5]
        cDp, aDp, gDp = popt[6], popt[7], popt[8]

        cD_std  = _pcov_stderr(pcov, 0)
        gD_std  = _pcov_stderr(pcov, 2)
        cG_std  = _pcov_stderr(pcov, 3)
        gG_std  = _pcov_stderr(pcov, 5)
        cDp_std = _pcov_stderr(pcov, 6)
        gDp_std = _pcov_stderr(pcov, 8)

        det_D,  snr_D  = _is_detected(aD,  yd, _lorentzian(xd, cD,  aD,  gD),  r2)
        det_G,  snr_G  = _is_detected(aG,  yd, _lorentzian(xd, cG,  aG,  gG),  r2)
        det_Dp, snr_Dp = _is_detected(aDp, yd, _lorentzian(xd, cDp, aDp, gDp), r2)

        def _mk(name, c, a, g, det, snr, c_std, g_std):
            return PeakResult(
                name=name, center=c, amplitude=a,
                fwhm=2.0*g, area=np.pi*a*g,
                r_squared=r2, snr=snr, found=det,
                model_x=xd, model_y=_lorentzian(xd, c, a, g),
                center_stderr=c_std,
                fwhm_stderr=(2.0*g_std) if g_std is not None else None,
            )

        return {
            "D":       _mk("D",  cD,  aD,  gD,  det_D,  snr_D,  cD_std,  gD_std),
            "G":       _mk("G",  cG,  aG,  gG,  det_G,  snr_G,  cG_std,  gG_std),
            "D_prime": _mk("D'", cDp, aDp, gDp, det_Dp, snr_Dp, cDp_std, gDp_std),
        }

    except Exception:
        return empty


def _fit_2D_bilayer_4L(
    xd: np.ndarray,
    yd: np.ndarray,
    c0: float,
    lo: float,
    hi: float,
    r2_single: float,
) -> Optional[PeakResult]:
    if len(xd) < 12:
        return None

    a0   = float(yd.max())
    g0   = 12.0

    c11 = c0 + _2D_P11_OFFSET
    c22 = c0 + _2D_P22_OFFSET
    c12 = c0 + _2D_P12_OFFSET
    c21 = c0 + _2D_P21_OFFSET

    _W = 15.0

    p0 = [
        c11, a0 * 0.7, g0,
        c22, a0 * 0.5, g0,
        c12, a0 * 0.25, g0,
        c21, a0 * 0.25, g0,
    ]
    bounds_lo = [
        c11 - _W, 0,        3,
        c22 - _W, 0,        3,
        c12 - _W, 0,        3,
        c21 - _W, 0,        3,
    ]
    bounds_hi = [
        c11 + _W, np.inf,       40,
        c22 + _W, np.inf,       40,
        c12 + _W, a0 * 0.65,    40,
        c21 + _W, a0 * 0.65,    40,
    ]

    bounds_lo[0]  = max(bounds_lo[0],  lo)
    bounds_hi[0]  = min(bounds_hi[0],  hi)
    bounds_lo[3]  = max(bounds_lo[3],  lo)
    bounds_hi[3]  = min(bounds_hi[3],  hi)
    bounds_lo[6]  = max(bounds_lo[6],  lo)
    bounds_hi[6]  = min(bounds_hi[6],  hi)
    bounds_lo[9]  = max(bounds_lo[9],  lo)
    bounds_hi[9]  = min(bounds_hi[9],  hi)

    try:
        popt, _ = curve_fit(
            _quad_lorentzian, xd, yd,
            p0=p0,
            bounds=(bounds_lo, bounds_hi),
            maxfev=20000,
        )
    except Exception:
        return None

    y_fit  = _quad_lorentzian(xd, *popt)
    r2_4L  = _r2(yd, y_fit)

    if r2_4L <= r2_single:
        return None

    c11f, a11f, g11f = popt[0], popt[1], popt[2]
    c22f, a22f, g22f = popt[3], popt[4], popt[5]

    w_total = a11f + a22f
    center  = (c11f * a11f + c22f * a22f) / w_total if w_total > 0 else c0
    amplitude = float(np.max(y_fit))
    total_area = float(np.trapz(y_fit, xd))
    fwhm_num   = compute_fwhm_numerical(xd, y_fit)

    det, snr = _is_detected(amplitude, yd, y_fit, r2_4L)

    return PeakResult(
        name="2D",
        center=center,
        amplitude=amplitude,
        fwhm=fwhm_num,
        area=total_area,
        r_squared=r2_4L,
        snr=snr,
        found=det,
        is_bilayer_4L=True,
        model_x=xd,
        model_y=y_fit,
    )


def _fit_2D(wn, intensity, lo, hi):
    single = _fit_peak(wn, intensity, lo, hi, "2D")

    if not single.found or single.r_squared >= _2D_BILAYER_R2_THRESHOLD:
        return single

    mask = (wn >= lo) & (wn <= hi)
    xd, yd = wn[mask], intensity[mask]
    if len(xd) < 8:
        return single

    c0 = float(xd[np.argmax(yd)])

    result_4L = _fit_2D_bilayer_4L(xd, yd, c0, lo, hi, single.r_squared)
    if result_4L is not None:
        return result_4L

    a0 = float(yd.max())
    g0 = (hi - lo) / 10.0

    try:
        p0     = [c0-10, a0*0.6, g0, c0+10, a0*0.4, g0]
        bounds = ([lo,0,0.5,lo,0,0.5],[hi,np.inf,(hi-lo)/3,hi,np.inf,(hi-lo)/3])
        popt, _ = curve_fit(_dual_lorentzian, xd, yd,
                            p0=p0, bounds=bounds, maxfev=8000)
        y_fit = _dual_lorentzian(xd, *popt)
        r2    = _r2(yd, y_fit)

        if r2 > single.r_squared:
            if popt[1] >= popt[4]:
                center, amplitude, gamma = popt[0], popt[1], popt[2]
            else:
                center, amplitude, gamma = popt[3], popt[4], popt[5]
            total_area = (np.pi * popt[1] * popt[2]
                          + np.pi * popt[4] * popt[5])
            det, snr = _is_detected(amplitude, yd, y_fit, r2)
            return PeakResult(
                name="2D", center=center, amplitude=amplitude,
                fwhm=2.0*gamma, area=total_area,
                r_squared=r2, snr=snr, found=det,
                is_split_2D=True, model_x=xd, model_y=y_fit,
            )
    except Exception:
        pass

    return single


def _fit_single_band_lmfit(wn, intensity, band_name, cfg, windows):
    import lmfit
    from lmfit.models import LorentzianModel, GaussianModel, VoigtModel, LinearModel

    lo, hi = windows.get(band_name, (1500, 1650))
    mask   = (wn >= lo) & (wn <= hi)
    xd, yd = wn[mask], intensity[mask]

    empty = PeakResult(name=band_name, found=False,
                       model_x=xd, model_y=np.zeros_like(xd))
    if len(xd) < 5 or yd.max() <= 0:
        return empty

    c0  = float(xd[np.argmax(yd)])
    a0  = float(yd.max())
    w0  = (hi - lo) / 6.0
    asym       = cfg.get("asymmetric", False)
    model_type = cfg.get("model", "Lorentzian")
    use_bg     = cfg.get("local_bg", False)

    if asym:
        def _asym_lor(x, center, sigma_l, sigma_r, amplitude):
            sig = np.where(x < center, sigma_l, sigma_r)
            return amplitude * (sig**2 / ((x-center)**2 + sig**2))
        peak_model = lmfit.Model(_asym_lor, prefix="peak_")
        params = peak_model.make_params(
            center=dict(value=c0, min=lo, max=hi),
            sigma_l=dict(value=w0, min=1.0, max=(hi-lo)/2),
            sigma_r=dict(value=w0, min=1.0, max=(hi-lo)/2),
            amplitude=dict(value=a0, min=0),
        )
    else:
        ModelCls = {"Lorentzian": LorentzianModel,
                    "Gaussian":   GaussianModel,
                    "Voigt":      VoigtModel}.get(model_type, LorentzianModel)
        peak_model = ModelCls(prefix="peak_")
        params = peak_model.make_params(
            center=dict(value=c0, min=lo, max=hi),
            sigma=dict(value=w0, min=1.0, max=(hi-lo)/2),
            amplitude=dict(value=a0*w0, min=0),
        )
        if model_type == "Voigt":
            params["peak_gamma"].set(value=w0*0.5, min=0.5, max=(hi-lo)/2, vary=True)

    full_model = peak_model
    if use_bg:
        lin = LinearModel(prefix="lin_")
        params.update(lin.make_params(slope=dict(value=0.0),
                                      intercept=dict(value=float(np.min(yd)))))
        full_model = peak_model + lin

    try:
        fit_result = full_model.fit(yd, params, x=xd)
    except Exception:
        return empty

    if not fit_result.success and fit_result.rsquared < 0.40:
        return empty

    model_y = fit_result.eval(x=xd)
    r2      = float(fit_result.rsquared)
    center  = float(fit_result.params["peak_center"].value)
    fwhm    = compute_fwhm_numerical(xd, model_y)
    c_err   = fit_result.params["peak_center"].stderr

    if asym:
        amplitude = float(fit_result.params["peak_amplitude"].value)
        fwhm_err  = None
    else:
        sig_err   = fit_result.params["peak_sigma"].stderr
        fwhm_err  = (2.0 * sig_err) if sig_err is not None else None
        if model_type == "Lorentzian":
            gamma_v   = float(fit_result.params["peak_sigma"].value)
            amplitude = (float(fit_result.params["peak_amplitude"].value)
                         / (np.pi * max(gamma_v, 1e-9)))
        elif model_type == "Gaussian":
            sigma_v   = float(fit_result.params["peak_sigma"].value)
            amplitude = (float(fit_result.params["peak_amplitude"].value)
                         / (sigma_v * np.sqrt(2*np.pi)))
        else:
            amplitude = float(np.max(model_y))

    display_name = band_name
    if band_name == "G" and model_type == "Gaussian":
        display_name = "G[Gauss!]"

    det, snr = _is_detected(amplitude, yd, model_y, r2)
    return PeakResult(
        name=display_name, center=center, amplitude=amplitude,
        fwhm=fwhm,
        area=float(np.trapz(model_y, xd)),
        r_squared=r2, snr=snr, found=det,
        model_x=xd, model_y=model_y,
        center_stderr=float(c_err) if c_err is not None else None,
        fwhm_stderr=float(fwhm_err) if fwhm_err is not None else None,
    )


def fit_all_peaks(
    wn:          np.ndarray,
    intensity:   np.ndarray,
    laser_nm:    float = 532.0,
    band_config: Optional[Dict] = None,
) -> Dict[str, PeakResult]:
    windows  = get_peak_windows(laser_nm)
    band_cfg = band_config or {}
    use_lmfit = _lmfit_available()

    def _method(band):
        m = band_cfg.get(band, {}).get("method", "auto")
        return "auto" if (m == "lmfit" and not use_lmfit) else m

    results: Dict[str, PeakResult] = {}

    dstar_method = _method("D_star")
    if dstar_method == "lmfit":
        results["D_star"] = _fit_single_band_lmfit(
            wn, intensity, "D_star", band_cfg.get("D_star", {}), windows
        )
    else:
        results["D_star"] = _fit_peak(
            wn, intensity, *windows["D_star"], "D*"
        )

    d_method  = _method("D")
    g_method  = _method("G")
    dp_method = _method("D_prime")

    if d_method == "lmfit":
        results["D"] = _fit_single_band_lmfit(
            wn, intensity, "D", band_cfg.get("D", {}), windows
        )
    if g_method == "lmfit":
        results["G"] = _fit_single_band_lmfit(
            wn, intensity, "G", band_cfg.get("G", {}), windows
        )
    if dp_method == "lmfit":
        results["D_prime"] = _fit_single_band_lmfit(
            wn, intensity, "D_prime", band_cfg.get("D_prime", {}), windows
        )

    needs_global = not (
        results.get("D") and results.get("G") and results.get("D_prime")
    )
    if needs_global:
        if g_method == "deconvolve":
            g_centre = _find_G_peak(wn, intensity)
            g_res = _fit_G_deconvolve(wn, intensity, g_centre)
            results["G"] = g_res
            if g_res.is_deconvolved and g_res.deconv_partner is not None:
                results.setdefault("D_prime", g_res.deconv_partner)
            results.setdefault("D", _fit_peak(wn, intensity, *windows["D"], "D"))
        else:
            global_fit = _fit_D_G_Dp_global(wn, intensity, *windows["D"])
            results.setdefault("D",       global_fit["D"])
            results.setdefault("D_prime", global_fit["D_prime"])

            if global_fit["G"].found:
                results.setdefault("G", global_fit["G"])
            else:
                results["G"] = _fit_G_adaptive(wn, intensity)
                g_res = results["G"]
                if g_res.is_deconvolved and g_res.deconv_partner is not None:
                    dp_global = results.get("D_prime")
                    dp_deconv = g_res.deconv_partner
                    if dp_global is None or not dp_global.found or (
                        dp_deconv.found and dp_deconv.r_squared > dp_global.r_squared
                    ):
                        results["D_prime"] = dp_deconv

    td_method = _method("2D")
    if td_method == "lmfit":
        results["2D"] = _fit_single_band_lmfit(
            wn, intensity, "2D", band_cfg.get("2D", {}), windows
        )
    else:
        results["2D"] = _fit_2D(wn, intensity, *windows["2D"])

    dg_method = _method("DG")
    if dg_method == "lmfit":
        results["DG"] = _fit_single_band_lmfit(
            wn, intensity, "DG", band_cfg.get("DG", {}), windows
        )
    else:
        results["DG"] = _fit_peak(
            wn, intensity, *windows["DG"], "D+G", use_pseudo_voigt=True
        )

    # ── g-C3N4 CN modes (Feature #9) ─────────────────────
    cntri_method = _method("CN_triazine")
    if cntri_method == "lmfit":
        results["CN_triazine"] = _fit_single_band_lmfit(
            wn, intensity, "CN_triazine", band_cfg.get("CN_triazine", {}), windows
        )
    else:
        results["CN_triazine"] = _fit_peak(
            wn, intensity, *windows["CN_triazine"], "CN_triazine"
        )

    cnbrt_method = _method("CN_ring_breathing")
    if cnbrt_method == "lmfit":
        results["CN_ring_breathing"] = _fit_single_band_lmfit(
            wn, intensity, "CN_ring_breathing", band_cfg.get("CN_ring_breathing", {}), windows
        )
    else:
        results["CN_ring_breathing"] = _fit_peak(
            wn, intensity, *windows["CN_ring_breathing"], "CN_ring_breathing"
        )

    return results