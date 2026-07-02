"""
Baseline correction methods for Raman spectra.

Methods
───────
  als_baseline   : Asymmetric Least Squares (Eilers & Boelens 2005)
                   Good for broad, slowly varying backgrounds.
  arPLS_baseline : Asymmetrically Reweighted Penalised Least Squares
                   (Baek et al. 2015, Analyst 140, 250).
                   Superior for fluorescence-heavy spectra (GO, g-C3N4):
                   automatically down-weights positive residuals (peaks)
                   so the baseline hugs the true background more tightly.

Change log
──────────
  v2.0   als_baseline (original)
  v2.5   arPLS_baseline added [Feature #7]
         auto_baseline() dispatcher added
  v2.5.1 Fix: replaced legacy diags() call with np.diff(np.eye(n), 2)
         compatible with scipy >= 1.11 (diags_array API change).
  v2.5.2 Fix: corrected D2 orientation so H = lam * D2.T @ D2 is (n x n),
         resolving 'inconsistent shapes' error in Z = W + H.
"""

import numpy as np
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import spsolve


def _second_diff_matrix(n: int):
    """
    Return sparse second-difference matrix D2 with shape (n-2, n),
    so that D2.T.dot(D2) has shape (n, n) and can be added to
    the diagonal weight matrix W (also n x n).

    np.diff(np.eye(n), 2) produces shape (n, n-2) — that is the
    forward-difference convention.  We need D in (n-2, n) form, so
    we take the transpose.
    """
    # diff result: shape (n, n-2)  -> transpose -> (n-2, n)
    return csc_matrix(np.diff(np.eye(n), 2).T)


# -------------------------------------------------------------
# ALS baseline
# -------------------------------------------------------------
def als_baseline(
    y: np.ndarray,
    lam: float = 1e5,
    p: float   = 0.01,
    niter: int = 10,
) -> tuple:
    """
    Asymmetric Least Squares baseline.

    Parameters
    ----------
    y     : intensity array (1-D)
    lam   : smoothness penalty (1e3-1e7 typical)
    p     : asymmetry weight (0.001-0.1; smaller -> baseline hugs minima)
    niter : number of IRLS iterations

    Returns
    -------
    (corrected, baseline) : tuple of two 1-D arrays, same length as y

    Reference
    ---------
    Eilers & Boelens (2005) Baseline Correction with Asymmetric
    Least Squares Smoothing.
    """
    y  = np.asarray(y, dtype=float)
    n  = len(y)
    D2 = _second_diff_matrix(n)       # shape (n-2, n)
    H  = lam * D2.T.dot(D2)           # shape (n, n)  <-- correct
    w  = np.ones(n)
    z  = y.copy()
    for _ in range(niter):
        W = diags(w, 0, format="csc")  # shape (n, n)
        Z = W + H                      # both (n, n) -> no shape error
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y <= z)
    return y - z, z


# -------------------------------------------------------------
# arPLS baseline
# -------------------------------------------------------------
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
    y     : intensity array (1-D, raw input)
    lam   : smoothness penalty (1e4-1e7)
    ratio : convergence criterion
    niter : maximum iterations

    Returns
    -------
    (corrected, baseline) : tuple of two 1-D arrays, same length as y

    Reference
    ---------
    Baek et al. (2015) Analyst 140, 250-257.
    DOI: 10.1039/C4AN01061B
    """
    y  = np.asarray(y, dtype=float)
    n  = len(y)
    D2 = _second_diff_matrix(n)       # shape (n-2, n)
    H  = lam * D2.T.dot(D2)           # shape (n, n)
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


# -------------------------------------------------------------
# Auto-dispatcher
# -------------------------------------------------------------
def auto_baseline(
    y: np.ndarray,
    method: str = "als",
    **kwargs,
) -> np.ndarray:
    """
    Dispatcher for baseline correction methods.

    Parameters
    ----------
    y      : raw intensity array
    method : 'als' (default) or 'arPLS'
    **kwargs : forwarded to the chosen method

    Returns
    -------
    baseline-corrected intensity (y - baseline)
    """
    method = method.lower().replace("-", "").replace("_", "")
    if method == "als":
        corrected, _ = als_baseline(y, **kwargs)
    elif method in ("arpls", "arplsbaseline"):
        corrected, _ = arPLS_baseline(y, **kwargs)
    else:
        raise ValueError(
            f"Unknown baseline method: {method!r}. "
            "Choose 'als' or 'arPLS'."
        )
    return corrected
