"""
preprocessing.py — Preprocessing adapter layer for Raman-analysis.

Design goal
-----------
Provide a *single, stable* preprocessing API for the interpretive core
(peak fitting, defect classification, reporting) while delegating the
heavy lifting to well-tested community packages when available:

    * pybaselines  -> 50+ baseline algorithms (arPLS, asPLS, morphological...)
    * ramanspy     -> file loaders (.wdf/.txt/...) and Whitaker-Hayes despike

Everything degrades gracefully: if the optional packages are missing,
a minimal, dependency-light scipy/numpy fallback is used so the core
tool still installs and runs on a laptop with no internet.

Install extras with:
    pip install pybaselines            # strongly recommended
    pip install ramanspy               # optional: loaders + despike

Author: adapter written for the CatLab / Raman-analysis project.
License note: pybaselines (BSD-3) and ramanspy (MIT) are permissively
licensed; keep their attributions in your dependency list / paper.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from scipy.sparse import csc_matrix, diags, eye
from scipy.sparse.linalg import spsolve

# --------------------------------------------------------------------------- #
# Optional dependency detection (done once, no hard import errors)
# --------------------------------------------------------------------------- #
try:
    from pybaselines import Baseline as _PybBaseline
    _HAS_PYBASELINES = True
except Exception:  # pragma: no cover - environment dependent
    _HAS_PYBASELINES = False

try:
    import ramanspy as _rp
    _HAS_RAMANSPY = True
except Exception:  # pragma: no cover - environment dependent
    _HAS_RAMANSPY = False


def available_backends() -> dict[str, bool]:
    """Report which optional backends are installed (useful for logging/tests)."""
    return {"pybaselines": _HAS_PYBASELINES, "ramanspy": _HAS_RAMANSPY}


# --------------------------------------------------------------------------- #
# Spectrum container
# --------------------------------------------------------------------------- #
@dataclass
class Spectrum:
    """Minimal Raman spectrum: wavenumber axis + intensity, plus provenance."""

    wavenumber: np.ndarray
    intensity: np.ndarray
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.wavenumber = np.asarray(self.wavenumber, dtype=float)
        self.intensity = np.asarray(self.intensity, dtype=float)
        if self.wavenumber.shape != self.intensity.shape:
            raise ValueError("wavenumber and intensity must have the same shape")
        # Ensure ascending wavenumber so downstream slicing is predictable.
        if self.wavenumber.size > 1 and self.wavenumber[0] > self.wavenumber[-1]:
            self.wavenumber = self.wavenumber[::-1]
            self.intensity = self.intensity[::-1]

    def copy(self) -> "Spectrum":
        return Spectrum(self.wavenumber.copy(), self.intensity.copy(), dict(self.meta))


# --------------------------------------------------------------------------- #
# Fallback baseline algorithms (used only when pybaselines is absent)
# --------------------------------------------------------------------------- #
def _als_fallback(y: np.ndarray, lam: float = 1e5, p: float = 0.01,
                  n_iter: int = 10) -> np.ndarray:
    """Asymmetric Least Squares (Eilers & Boelens 2005) — the classic baseline.

    Kept as a self-contained fallback so the tool works without pybaselines.
    Prefer pybaselines.arpls/aspls when available (better for fluorescence).
    """
    L = y.size
    D = diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    W = diags(w, 0, shape=(L, L))
    z = y
    for _ in range(n_iter):
        W.setdiag(w)
        Z = W + D
        z = spsolve(csc_matrix(Z), w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def _arpls_fallback(y: np.ndarray, lam: float = 1e5, ratio: float = 1e-6,
                    n_iter: int = 50) -> np.ndarray:
    """Asymmetrically Reweighted Penalized Least Squares (Baek et al. 2015).

    Better than plain ALS for the strong, smoothly-varying fluorescence
    backgrounds seen in GO / g-C3N4. This is a compact reference
    implementation; pybaselines.arpls is preferred when installed.
    """
    L = y.size
    D = diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(L, L - 2))
    H = lam * D.dot(D.transpose())
    w = np.ones(L)
    z = y
    for _ in range(n_iter):
        W = diags(w, 0, shape=(L, L))
        z = spsolve(csc_matrix(W + H), w * y)
        d = y - z
        dn = d[d < 0]
        if dn.size == 0:
            break
        m, s = dn.mean(), dn.std()
        # logistic reweighting; clip exponent to avoid overflow on real spectra
        expo = np.clip(2.0 * (d - (2.0 * s - m)) / (s + 1e-12), -500, 500)
        w_new = 1.0 / (1.0 + np.exp(expo))
        if np.linalg.norm(w - w_new) / (np.linalg.norm(w) + 1e-12) < ratio:
            w = w_new
            break
        w = w_new
    return z


# --------------------------------------------------------------------------- #
# Core preprocessing steps (backend-aware, single stable signature each)
# --------------------------------------------------------------------------- #
def despike(spec: Spectrum, threshold: float = 6.0,
            window: int = 5) -> Spectrum:
    """Remove cosmic-ray spikes.

    Uses RamanSPy's Whitaker-Hayes when available; otherwise a modified
    z-score + median-replacement fallback. Spikes are extremely common in
    real GO/rGO spectra, so this step matters for robustness.
    """
    out = spec.copy()
    if _HAS_RAMANSPY:
        try:
            rp_spec = _rp.Spectrum(out.intensity, out.wavenumber)
            step = _rp.preprocessing.despike.WhitakerHayes(
                kernel_size=window, threshold=threshold
            )
            result = step.apply(rp_spec)
            out.intensity = np.asarray(result.spectral_data, dtype=float)
            out.meta["despike"] = "ramanspy.WhitakerHayes"
            return out
        except Exception as exc:  # fall back rather than crash the pipeline
            warnings.warn(f"RamanSPy despike failed ({exc}); using fallback.")

    # Fallback: modified z-score on the discrete difference (Whitaker-Hayes idea)
    y = out.intensity
    diff = np.diff(y, prepend=y[0])
    mad = np.median(np.abs(diff - np.median(diff))) + 1e-12
    mod_z = 0.6745 * (diff - np.median(diff)) / mad
    spikes = np.abs(mod_z) > threshold
    if spikes.any():
        med = median_filter(y, size=window)
        y = np.where(spikes, med, y)
        out.intensity = y
    out.meta["despike"] = "fallback.modified_zscore"
    return out


def denoise(spec: Spectrum, window_length: int = 9,
            polyorder: int = 3) -> Spectrum:
    """Savitzky-Golay smoothing (light-touch; safe default for Raman)."""
    out = spec.copy()
    wl = window_length
    if wl % 2 == 0:
        wl += 1
    wl = min(wl, out.intensity.size - (1 - out.intensity.size % 2))
    if wl <= polyorder:
        out.meta["denoise"] = "skipped (too few points)"
        return out
    out.intensity = savgol_filter(out.intensity, wl, polyorder)
    out.meta["denoise"] = f"savgol(w={wl},p={polyorder})"
    return out


BaselineMethod = Literal["auto", "arpls", "aspls", "asls", "mor", "als"]


def remove_baseline(spec: Spectrum, method: BaselineMethod = "auto",
                    lam: float = 1e5, material: Optional[str] = None
                    ) -> Spectrum:
    """Subtract a fitted baseline.

    method="auto" picks a sensible algorithm:
        * strongly fluorescent materials (GO, rGO, g-C3N4) -> asPLS
        * otherwise                                        -> arPLS
    Uses pybaselines when available (50+ methods, well tested); otherwise
    falls back to the built-in arPLS/ALS implementations above.
    """
    out = spec.copy()
    y = out.intensity

    fluorescent = False
    if material is not None:
        fluorescent = any(k in material.lower()
                          for k in ("go", "rgo", "graphene oxide",
                                    "g-c3n4", "gcn", "c3n4", "carbon nitride"))
    if method == "auto":
        method = "aspls" if fluorescent else "arpls"

    if _HAS_PYBASELINES:
        fitter = _PybBaseline(x_data=out.wavenumber)
        try:
            func = getattr(fitter, method if method != "als" else "asls")
            baseline, _ = func(y, lam=lam) if method != "mor" else func(y)
            out.intensity = y - baseline
            out.meta["baseline"] = f"pybaselines.{method}(lam={lam:g})"
            out.meta["baseline_curve"] = baseline
            return out
        except Exception as exc:
            warnings.warn(f"pybaselines.{method} failed ({exc}); using fallback.")

    # Fallbacks
    if method in ("arpls", "aspls", "auto"):
        baseline = _arpls_fallback(y, lam=lam)
        tag = "fallback.arpls"
    else:
        baseline = _als_fallback(y, lam=lam)
        tag = "fallback.als"
    out.intensity = y - baseline
    out.meta["baseline"] = f"{tag}(lam={lam:g})"
    out.meta["baseline_curve"] = baseline
    return out


NormMethod = Literal["minmax", "area", "max", "vector", "none"]


def normalise(spec: Spectrum, method: NormMethod = "minmax") -> Spectrum:
    """Normalise intensity. Note: intensity *ratios* (I_D/I_G) are
    normalisation-invariant, so this is mainly for plotting/overlay."""
    out = spec.copy()
    y = out.intensity
    if method == "none":
        pass
    elif method == "minmax":
        rng = np.ptp(y)
        y = (y - y.min()) / rng if rng > 0 else y
    elif method == "max":
        m = y.max()
        y = y / m if m != 0 else y
    elif method == "area":
        a = np.trapz(y, out.wavenumber)
        y = y / a if a != 0 else y
    elif method == "vector":
        n = np.linalg.norm(y)
        y = y / n if n != 0 else y
    out.intensity = y
    out.meta["normalise"] = method
    return out


# --------------------------------------------------------------------------- #
# Pipeline (RamanSPy-inspired, but self-contained and picklable)
# --------------------------------------------------------------------------- #
class Pipeline:
    """Chain of preprocessing steps applied in order.

    Each step is a callable Spectrum -> Spectrum. Use the module-level
    functions with functools.partial, or the convenience `default_pipeline`.

    Example
    -------
    >>> pipe = Pipeline([
    ...     lambda s: despike(s),
    ...     lambda s: remove_baseline(s, method="auto", material="rGO"),
    ...     lambda s: normalise(s, "minmax"),
    ... ])
    >>> clean = pipe.apply(raw)
    """

    def __init__(self, steps: list[Callable[[Spectrum], Spectrum]]):
        self.steps = steps

    def apply(self, spec: Spectrum) -> Spectrum:
        out = spec.copy()
        for step in self.steps:
            out = step(out)
        return out


def default_pipeline(material: Optional[str] = None,
                     do_denoise: bool = True) -> Pipeline:
    """A sensible default: despike -> (denoise) -> baseline(auto) -> minmax.

    Baseline algorithm auto-selects asPLS for fluorescent materials.
    """
    from functools import partial

    steps: list[Callable[[Spectrum], Spectrum]] = [partial(despike)]
    if do_denoise:
        steps.append(partial(denoise))
    steps.append(partial(remove_baseline, method="auto", material=material))
    steps.append(partial(normalise, method="minmax"))
    return Pipeline(steps)


# --------------------------------------------------------------------------- #
# Unified loader (delegates to RamanSPy for instrument formats)
# --------------------------------------------------------------------------- #
def load_spectrum(path: str) -> Spectrum:
    """Load a spectrum from disk.

    Delegates .wdf/.spc/.txt instrument formats to RamanSPy when installed;
    otherwise reads simple two-column text/CSV. This abstraction is what
    lets you retire the hard-coded 'column B/C from row 4' Excel logic.
    """
    lower = path.lower()

    if _HAS_RAMANSPY and lower.endswith((".wdf", ".spc", ".0", ".tvf")):
        loaders = {
            ".wdf": getattr(_rp.load, "witec", None) or getattr(_rp.load, "wdf", None),
        }
        # RamanSPy's loader names vary by version; try the generic entry points.
        for name in ("witec", "renishaw", "ocean_insight", "labspec"):
            fn = getattr(_rp.load, name, None)
            if fn is None:
                continue
            try:
                obj = fn(path)
                return Spectrum(np.asarray(obj.spectral_axis),
                                np.asarray(obj.spectral_data),
                                {"loader": f"ramanspy.{name}", "path": path})
            except Exception:
                continue
        warnings.warn("RamanSPy present but no loader matched; trying text parse.")

    # Plain-text / CSV fallback: detect delimiter, skip non-numeric header rows.
    data = np.genfromtxt(path, delimiter=None, comments="#")
    if data.ndim == 1 or data.shape[1] < 2:
        data = np.genfromtxt(path, delimiter=",", comments="#")
    data = data[~np.isnan(data).any(axis=1)]
    return Spectrum(data[:, 0], data[:, 1], {"loader": "text", "path": path})
