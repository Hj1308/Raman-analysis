"""
Baseline correction methods for Raman spectra.

Public API
----------
  als_baseline(y, lam, p, niter)         -> (corrected, baseline)
  arPLS_baseline(y, lam, ratio, niter)   -> (corrected, baseline)
  auto_baseline(y, method, **kwargs)     -> corrected   (single array)
  correct_baseline(wn, y, method, ...)   -> (corrected, baseline)  ← used by streamlit_app

Change log
----------
  v2.0    als_baseline (original)
  v2.5    arPLS_baseline + auto_baseline added [Feature #7]
  v2.5.1  Fix: np.diff(np.eye(n), 2) replaces legacy diags() call
  v2.5.2  Fix: D2 orientation corrected so H = lam * D2.T @ D2 is (n x n)
  v2.5.3  Fix: add correct_baseline alias  (ImportError on Streamlit Cloud)
  v2.5.4  Fix: correct_baseline is now a proper wrapper, not a bare alias
          - accepts (wn, y, method, lam, p, ...)  matching streamlit_app.py call
          - always returns (corrected, baseline) tuple  (unpack crash fixed)
          - 'linear' method added (subtract straight line between endpoints)
"""

import numpy as np
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import spsolve


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _second_diff_matrix(n: int):
    """
    Sparse second-difference matrix D2, shape (n-2, n).
    np.diff(np.eye(n), 2) -> (n, n-2); we need (n-2, n) so we transpose.
    """
    return csc_matrix(np.diff(np.eye(n), 2).T)


# ------------------------------------------------------------------
# ALS baseline
# ------------------------------------------------------------------

def als_baseline(
    y: np.ndarray,
    lam: float = 1e5,
    p: float   = 0.01,
    niter: int = 10,
) -> tuple:
    """
    Asymmetric Least Squares baseline correction.

    Parameters
    ----------
    y     : 1-D intensity array
    lam   : smoothness penalty  (1e3 – 1e7 typical)
    p     : asymmetry weight    (0.001 – 0.1; smaller -> hugs minima)
    niter : IRLS iterations

    Returns
    -------
    (corrected, baseline) : two 1-D arrays of the same length as y

    Reference
    ---------
    Eilers & Boelens (2005) Baseline Correction with Asymmetric
    Least Squares Smoothing.
    """
    y  = np.asarray(y, dtype=float)
    n  = len(y)
    D2 = _second_diff_matrix(n)   # (n-2, n)
    H  = lam * D2.T.dot(D2)       # (n, n)
    w  = np.ones(n)
    z  = y.copy()
    for _ in range(niter):
        W = diags(w, 0, format="csc")
        Z = W + H
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return y - z, z


# ------------------------------------------------------------------
# arPLS baseline
# ------------------------------------------------------------------

def arPLS_baseline(
    y: np.ndarray,
    lam: float   = 1e5,
    ratio: float = 1e-6,
    niter: int   = 100,
) -> tuple:
    """
    Asymmetrically Reweighted Penalised Least Squares baseline.

    Parameters
    ----------
    y     : 1-D intensity array (raw)
    lam   : smoothness penalty  (1e4 – 1e7)
    ratio : convergence criterion
    niter : maximum iterations

    Returns
    -------
    (corrected, baseline) : two 1-D arrays of the same length as y

    Reference
    ---------
    Baek et al. (2015) Analyst 140, 250-257.  DOI: 10.1039/C4AN01061B
    """
    y  = np.asarray(y, dtype=float)
    n  = len(y)
    D2 = _second_diff_matrix(n)
    H  = lam * D2.T.dot(D2)
    w  = np.ones(n)
    z  = y.copy()
    for _ in range(niter):
        W     = diags(w, 0, format="csc")
        Z     = W + H
        z_new = spsolve(Z, w * y)
        d     = y - z_new
        d_neg = d[d < 0]
        m_neg = d_neg.mean() if len(d_neg) > 0 else 0.0
        s_neg = d_neg.std()  if len(d_neg) > 0 else 1.0
        if s_neg == 0.0:
            s_neg = 1.0
        w_new  = 1.0 / (1.0 + np.exp(2.0 * (d - (2.0 * s_neg - m_neg)) / s_neg))
        change = np.linalg.norm(z_new - z) / (np.linalg.norm(z) + 1e-12)
        z = z_new
        w = w_new
        if change < ratio:
            break
    return y - z, z


# ------------------------------------------------------------------
# Linear baseline  (endpoint subtraction)
# ------------------------------------------------------------------

def _linear_baseline(y: np.ndarray) -> tuple:
    """
    Subtract a straight line connecting the first and last intensity points.
    Fast fallback; no parameters needed.

    Returns
    -------
    (corrected, baseline)
    """
    y  = np.asarray(y, dtype=float)
    n  = len(y)
    bl = np.linspace(y[0], y[-1], n)
    return y - bl, bl


# ------------------------------------------------------------------
# auto_baseline  (single-array dispatcher — internal use)
# ------------------------------------------------------------------

def auto_baseline(
    y: np.ndarray,
    method: str = "als",
    **kwargs,
) -> np.ndarray:
    """
    Dispatcher returning only the corrected array (no baseline).
    Used internally; external callers should use correct_baseline().

    Parameters
    ----------
    y      : raw intensity array
    method : 'als' (default), 'arPLS', or 'linear'
    **kwargs : forwarded to the chosen method

    Returns
    -------
    corrected : 1-D array  (y - baseline)
    """
    corrected, _ = correct_baseline(None, y, method, **kwargs)
    return corrected


# ------------------------------------------------------------------
# correct_baseline  ← PRIMARY public function for streamlit_app.py
# ------------------------------------------------------------------

def correct_baseline(
    wn,
    y: np.ndarray,
    method: str = "als",
    lam: float  = 1e5,
    p: float    = 0.001,
    ratio: float = 1e-6,
    niter: int  = 10,
) -> tuple:
    """
    Unified baseline correction entry-point.

    Signature matches the call in streamlit_app.py:
        corrected, baseline_arr = correct_baseline(
            wn, intensity, method=baseline_method, lam=als_lam, p=als_p
        )

    Parameters
    ----------
    wn     : wavenumber array (accepted but not used; kept for API compat)
    y      : 1-D raw intensity array
    method : 'als' (default)  |  'arPLS'  |  'linear'
    lam    : ALS / arPLS smoothness penalty
    p      : ALS asymmetry weight
    ratio  : arPLS convergence criterion
    niter  : iteration count

    Returns
    -------
    (corrected, baseline) : two 1-D arrays, same length as y
    """
    y = np.asarray(y, dtype=float)
    m = method.lower().replace("-", "").replace("_", "")

    if m == "als":
        return als_baseline(y, lam=lam, p=p, niter=niter)
    elif m in ("arpls", "arplsbaseline"):
        return arPLS_baseline(y, lam=lam, ratio=ratio, niter=niter)
    elif m == "linear":
        return _linear_baseline(y)
    else:
        # Unknown method: fall back to ALS with a warning-safe default
        return als_baseline(y, lam=lam, p=p, niter=niter)
