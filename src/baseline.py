"""
Baseline correction methods for Raman spectra.

Methods
───────
  als_baseline   : Asymmetric Least Squares (Eilers & Boelens 2005)
                   Good for broad, slowly varying backgrounds.
  arPLS_baseline : Asymmetrically Reweighted Penalised Least Squares
                   (Baek et al. 2015, Analyst 140, 250).
                   Superior for fluorescence-heavy spectra (GO, g-C₃N₄):
                   automatically down-weights positive residuals (peaks)
                   so the baseline hugs the true background more tightly.

Change log
──────────
  v2.0   als_baseline (original)
  v2.5   arPLS_baseline added [Feature #7]
         auto_baseline() dispatcher added
"""

import numpy as np
from scipy.sparse import diags, eye as speye
from scipy.sparse.linalg import spsolve


# ─────────────────────────────────────────────────────────
# ALS baseline (original, unchanged)
# ─────────────────────────────────────────────────────────
def als_baseline(
    y: np.ndarray,
    lam: float = 1e5,
    p: float   = 0.01,
    niter: int = 10,
) -> np.ndarray:
    """
    Asymmetric Least Squares baseline.

    Parameters
    ----------
    y     : intensity array (1-D)
    lam   : smoothness penalty (10³–10⁷ typical)
    p     : asymmetry weight (0.001–0.1; smaller = baseline hugs minima)
    niter : number of IRLS iterations (10 is usually sufficient)

    Returns
    -------
    baseline : 1-D array, same length as y

    Reference
    ---------
    Eilers & Boelens (2005) Baseline Correction with Asymmetric
    Least Squares Smoothing. Unpublished manuscript.
    """
    n  = len(y)
    D  = diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n)).toarray()
    D  = diags(D.flatten(), np.arange(-(n - 2), n), shape=(n - 2, n))
    # Use scipy sparse correctly
    from scipy.sparse import csc_matrix
    D2 = csc_matrix(np.diff(np.eye(n), 2).T)
    H  = lam * D2.dot(D2.T)
    w  = np.ones(n)
    for _ in range(niter):
        W   = diags(w, 0)
        Z   = W + H
        z   = spsolve(Z, w * y)
        w   = p * (y > z) + (1 - p) * (y <= z)
    return z


# ─────────────────────────────────────────────────────────
# arPLS baseline  (v2.5, Feature #7)
# ─────────────────────────────────────────────────────────
def arPLS_baseline(
    y: np.ndarray,
    lam: float  = 1e5,
    ratio: float = 1e-6,
    niter: int  = 100,
) -> np.ndarray:
    """
    Asymmetrically Reweighted Penalised Least Squares baseline.

    Compared to ALS, arPLS:
      • Uses a sigmoid-like weight update instead of a hard threshold.
      • Automatically down-weights spectral regions where the signal
        is above the estimated baseline (i.e., peaks).
      • Converges more robustly for fluorescence-heavy spectra.

    Parameters
    ----------
    y     : intensity array (1-D, baseline-subtracted input)
    lam   : smoothness penalty (1e4–1e7; larger = smoother baseline)
    ratio : convergence criterion; stop when change < ratio × ||y||₂
    niter : maximum iterations

    Returns
    -------
    baseline : 1-D array, same length as y

    Recommended use cases
    ----------------------
    • Graphene oxide (GO) — broad fluorescent background
    • Reduced GO (rGO)   — partially quenched but still present
    • g-C₃N₄             — strong visible fluorescence under 532 nm
    • Functionalized graphene with organic residues

    Reference
    ---------
    Baek et al. (2015) Analyst 140, 250–257.
    DOI: 10.1039/C4AN01061B
    """
    from scipy.sparse import csc_matrix, diags as sp_diags
    from scipy.sparse.linalg import spsolve as sp_solve

    n   = len(y)
    # Second-difference penalty matrix
    D   = csc_matrix(np.diff(np.eye(n), 2))
    H   = lam * D.dot(D.T)
    w   = np.ones(n)
    z   = y.copy().astype(float)

    for _ in range(niter):
        W    = sp_diags(w, 0, format="csc")
        Z    = W + H
        z_new = sp_solve(Z, w * y)
        d    = y - z_new
        # Negative residuals (below current estimate) get weight → 1
        # Positive residuals (peaks) get down-weighted toward 0
        d_neg = d[d < 0]
        m_neg = d_neg.mean() if len(d_neg) > 0 else 0.0
        s_neg = d_neg.std()  if len(d_neg) > 0 else 1.0
        w_new = 1.0 / (1.0 + np.exp(2.0 * (d - (2.0 * s_neg - m_neg)) / s_neg))
        # Convergence check
        change = np.linalg.norm(z_new - z) / (np.linalg.norm(z) + 1e-12)
        z = z_new
        w = w_new
        if change < ratio:
            break

    return z


# ─────────────────────────────────────────────────────────
# Auto-dispatcher  (v2.5)
# ─────────────────────────────────────────────────────────
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

    Usage
    -----
    corrected = auto_baseline(raw_intensity, method='arPLS', lam=1e6)
    """
    method = method.lower().replace("-", "").replace("_", "")
    if method == "als":
        bl = als_baseline(y, **kwargs)
    elif method in ("arpls", "arplsbaseline"):
        bl = arPLS_baseline(y, **kwargs)
    else:
        raise ValueError(
            f"Unknown baseline method: {method!r}. "
            "Choose 'als' or 'arPLS'."
        )
    return y - bl
