"""
Baseline correction algorithms:
1. Asymmetric Least Squares (ALS) — recommended for Raman
2. Linear baseline between anchor points

Note on negative residuals
--------------------------
``correct_baseline`` returns the raw intensity − baseline difference
*without* clipping.  Negative values arise when the ALS baseline is
slightly over-estimated and are physically meaningful: clipping them
would (a) break the least-squares noise assumption required by peak
fitting, (b) hide ALS over-subtraction silently, and (c) bias height
and area of weak peaks upward.  Callers that need non-negative values
for display purposes should clip the returned array themselves.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def als_baseline(intensity: np.ndarray,
                 lam: float = 1e5,
                 p: float   = 0.001,
                 n_iter: int = 20) -> np.ndarray:
    """
    Asymmetric Least Squares baseline correction.
    Parameters:
        lam    : smoothness (1e4–1e7 typical for Raman)
        p      : asymmetry (0.001–0.1; smaller = baseline closer to minimum)
        n_iter : number of iterations
    Returns: baseline array (same length as intensity)
    Reference: Eilers & Boelens (2005)
    """
    L = len(intensity)
    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L))
    D = lam * D.T.dot(D)
    w = np.ones(L)
    for _ in range(n_iter):
        W        = sparse.diags(w, 0)
        Z        = W + D
        baseline = spsolve(Z, w * intensity)
        w        = p * (intensity > baseline) + (1 - p) * (intensity <= baseline)
    return baseline


def linear_baseline(wavenumber: np.ndarray,
                    intensity:  np.ndarray,
                    x1: float, x2: float) -> np.ndarray:
    """
    Linear baseline between two anchor wavenumber points.
    Parameters:
        x1, x2 : anchor wavenumber positions (cm⁻¹)
    Returns: baseline array
    """
    idx1 = np.argmin(np.abs(wavenumber - x1))
    idx2 = np.argmin(np.abs(wavenumber - x2))
    y1, y2 = intensity[idx1], intensity[idx2]
    return np.interp(wavenumber,
                     [wavenumber[idx1], wavenumber[idx2]],
                     [y1, y2])


def correct_baseline(wavenumber: np.ndarray,
                     intensity:  np.ndarray,
                     method: str = "als",
                     **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply baseline correction.

    Returns: (corrected_intensity, baseline)

    The corrected array is *not* clipped to zero.  Negative residuals
    indicate local ALS over-subtraction and should be preserved so that
    peak-fitting cost functions see unbiased noise on both sides of zero.
    """
    if method == "als":
        baseline  = als_baseline(intensity, **kwargs)
    elif method == "linear":
        x1 = kwargs.get("x1", wavenumber[0])
        x2 = kwargs.get("x2", wavenumber[-1])
        baseline  = linear_baseline(wavenumber, intensity, x1, x2)
    else:
        raise ValueError(f"Unknown baseline method: '{method}'. Use 'als' or 'linear'.")

    corrected = intensity - baseline
    # NOTE: do NOT clip here — see module docstring.
    return corrected, baseline
