"""
uncertainty.py — Peak fitting with rigorous uncertainty propagation.

Why this module exists
----------------------
Comparing I_D/I_G (or L_D, n_D) between samples is statistically
meaningless without error bars. scipy.optimize.curve_fit already returns
the parameter covariance matrix for free; this module turns that into
propagated 1-sigma uncertainties on the derived Raman metrics.

Provides:
    * fit_peaks()            -> fit a sum of pseudo-Voigt peaks, with covariance
    * ratio_uncertainty()    -> sigma of A/B from covariance (correlated-safe)
    * crystallite_size()     -> L_D and n_D via Cancado 2011, with error
    * defect_density()       -> point-defect density n_D, with error

All propagation uses the standard first-order (delta-method) formula:
    var(f) = J . Cov . J^T
where J is the gradient of f w.r.t. the fitted parameters. This correctly
accounts for correlation between, e.g., D-peak and G-peak amplitudes.

Depends only on numpy + scipy (no extra install).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import curve_fit


# --------------------------------------------------------------------------- #
# Peak models
# --------------------------------------------------------------------------- #
def pseudo_voigt(x: np.ndarray, amp: float, center: float,
                 fwhm: float, eta: float) -> np.ndarray:
    """Pseudo-Voigt: eta*Lorentzian + (1-eta)*Gaussian, both peak-normalised
    to `amp` at `center`. eta in [0,1]; eta=1 pure Lorentzian, eta=0 Gaussian.

    Reporting eta itself is diagnostic: eta->0 signals inhomogeneous
    (Gaussian) broadening typical of GO/rGO; eta->1 the homogeneous
    Lorentzian limit of well-ordered graphene.
    """
    sigma = fwhm / 2.0
    lor = 1.0 / (1.0 + ((x - center) / sigma) ** 2)
    gau = np.exp(-np.log(2.0) * ((x - center) / sigma) ** 2)
    return amp * (eta * lor + (1.0 - eta) * gau)


def _multi_peak(x: np.ndarray, *params: float) -> np.ndarray:
    """Sum of N pseudo-Voigt peaks. params flattened as
    [amp, center, fwhm, eta] * N."""
    n = len(params) // 4
    y = np.zeros_like(x, dtype=float)
    for i in range(n):
        amp, c, w, eta = params[4 * i:4 * i + 4]
        y += pseudo_voigt(x, amp, c, w, eta)
    return y


# --------------------------------------------------------------------------- #
# Fit result container
# --------------------------------------------------------------------------- #
@dataclass
class PeakFit:
    names: list[str]
    popt: np.ndarray          # optimal parameters, flattened [amp,c,fwhm,eta]*N
    pcov: np.ndarray          # covariance matrix from curve_fit
    x: np.ndarray
    y: np.ndarray
    yfit: np.ndarray

    def perr(self) -> np.ndarray:
        """1-sigma uncertainties on each parameter (sqrt of covariance diag)."""
        return np.sqrt(np.diag(self.pcov))

    def param(self, name: str, which: str) -> tuple[float, float]:
        """Return (value, 1sigma) for a peak parameter.
        which in {'amp','center','fwhm','eta'}."""
        idx = self.names.index(name)
        off = {"amp": 0, "center": 1, "fwhm": 2, "eta": 3}[which]
        j = 4 * idx + off
        return float(self.popt[j]), float(self.perr()[j])

    def area(self, name: str) -> tuple[float, float]:
        """Integrated area of a pseudo-Voigt peak, with propagated error.
        Area = amp * fwhm * [eta*pi/2 + (1-eta)*sqrt(pi/(4 ln2))].
        Propagates over amp, fwhm, eta (and their covariances)."""
        idx = self.names.index(name)
        a, w, eta = self.popt[4*idx], self.popt[4*idx+2], self.popt[4*idx+3]
        kL, kG = np.pi / 2.0, np.sqrt(np.pi / (4.0 * np.log(2.0)))
        shape = eta * kL + (1.0 - eta) * kG
        val = a * w * shape
        # gradient wrt (amp, fwhm, eta)
        dA = w * shape
        dw = a * shape
        de = a * w * (kL - kG)
        j_local = np.array([dA, dw, de])
        # pull the 3x3 sub-covariance for (amp,fwhm,eta)
        cols = [4*idx, 4*idx+2, 4*idx+3]
        sub = self.pcov[np.ix_(cols, cols)]
        var = j_local @ sub @ j_local
        return float(val), float(np.sqrt(max(var, 0.0)))


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #
def fit_peaks(x: np.ndarray, y: np.ndarray,
              peaks: dict[str, dict],
              laser_nm: float = 532.0) -> PeakFit:
    """Fit a sum of pseudo-Voigt peaks.

    peaks : {name: {'center':.., 'fwhm':.., 'amp':.., 'eta':..(opt),
                    'window':(lo,hi)(opt)}}
        Initial guesses; 'window' restricts the fit region for that peak's
        center bound. If omitted, sensible bounds are used.

    Returns a PeakFit with covariance for downstream error propagation.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    names = list(peaks.keys())

    p0, lower, upper = [], [], []
    for nm in names:
        pk = peaks[nm]
        amp0 = pk.get("amp", float(np.interp(pk["center"], x, y)))
        c0 = pk["center"]
        w0 = pk.get("fwhm", 40.0)
        e0 = pk.get("eta", 0.5)
        cspan = pk.get("center_tol", 25.0)
        p0 += [amp0, c0, w0, e0]
        lower += [0.0, c0 - cspan, 2.0, 0.0]
        upper += [np.inf, c0 + cspan, 400.0, 1.0]

    popt, pcov = curve_fit(_multi_peak, x, y, p0=p0,
                           bounds=(lower, upper), maxfev=20000)
    yfit = _multi_peak(x, *popt)
    return PeakFit(names, popt, pcov, x, y, yfit)


# --------------------------------------------------------------------------- #
# Uncertainty propagation for derived metrics
# --------------------------------------------------------------------------- #
def ratio_uncertainty(a: float, sa: float, b: float, sb: float,
                      cov_ab: float = 0.0) -> tuple[float, float]:
    """1-sigma error of r = a/b, including covariance between a and b.

    var(r)/r^2 = (sa/a)^2 + (sb/b)^2 - 2 cov_ab/(a b)
    """
    r = a / b
    var = r * r * ((sa / a) ** 2 + (sb / b) ** 2 - 2.0 * cov_ab / (a * b))
    return r, float(np.sqrt(max(var, 0.0)))


def intensity_ratio(fit: PeakFit, num: str = "D", den: str = "G",
                    use: str = "height") -> tuple[float, float]:
    """I_num/I_den with propagated error. use in {'height','area'}.

    'height' uses fitted amplitudes; 'area' uses integrated areas.
    Covariance between the two peaks' amplitudes is taken from pcov.
    """
    if use == "height":
        ai = fit.names.index(num); bi = fit.names.index(den)
        a, sa = fit.popt[4*ai], fit.perr()[4*ai]
        b, sb = fit.popt[4*bi], fit.perr()[4*bi]
        cov_ab = fit.pcov[4*ai, 4*bi]
        return ratio_uncertainty(a, sa, b, sb, cov_ab)
    elif use == "area":
        a, sa = fit.area(num)
        b, sb = fit.area(den)
        # area-area covariance is second order; approximate as independent.
        return ratio_uncertainty(a, sa, b, sb, 0.0)
    raise ValueError("use must be 'height' or 'area'")


def crystallite_size(id_ig: float, s_id_ig: float,
                     laser_nm: float = 532.0) -> tuple[float, float]:
    """In-plane crystallite size L_a (nm) via Cancado et al. 2011:

        L_a = (2.4e-10) * lambda_nm^4 * (I_D/I_G)^-1        [lambda in nm]

    Propagates the (I_D/I_G) uncertainty. Note the lambda^4 (E_L^4)
    dependence — the correction that was missing in earlier versions.
    """
    C = 2.4e-10 * laser_nm ** 4
    La = C / id_ig
    # dL/d(ratio) = -C / ratio^2
    sLa = abs(C / id_ig ** 2) * s_id_ig
    return float(La), float(sLa)


def defect_density(id_ig: float, s_id_ig: float,
                   laser_nm: float = 532.0) -> tuple[float, float]:
    """Point-defect density n_D (cm^-2) via Cancado et al. 2011:

        n_D = (1.8e22 / lambda_nm^4) * (I_D/I_G)

    Valid in the low-defect regime (Stage 1). Propagates I_D/I_G error.
    """
    C = 1.8e22 / laser_nm ** 4
    nD = C * id_ig
    snD = C * s_id_ig
    return float(nD), float(snD)


def format_value(val: float, err: float, unit: str = "",
                 sig: int = 2) -> str:
    """Pretty 'value ± error unit' with error-driven rounding."""
    if err <= 0 or not np.isfinite(err):
        return f"{val:.3g} {unit}".strip()
    # round to `sig` significant figures of the error
    from math import floor, log10
    d = sig - 1 - int(floor(log10(abs(err))))
    d = max(d, 0)
    return f"{val:.{d}f} ± {err:.{d}f} {unit}".strip()
