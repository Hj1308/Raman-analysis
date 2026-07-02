"""
Batch statistics module for Raman-analysis v2.4 (Roadmap feature #10).

Usage
-----
from src.batch_stats import collect_batch_stats, print_batch_summary

# Assuming `batch_results` is a list[dict] with keys
# 'filename', 'analysis' (RamanAnalysis), 'peaks' (dict[str, PeakResult])
stats = collect_batch_stats(batch_results)
print_batch_summary(stats)

Returns a nested dict:
  {
    "n_samples":   int,
    "n_valid":     int,
    "ratios":      {attr: {"mean", "std", "median", "min", "max", "n"}},
    "L_D":         {"mean", "std", "median", "min", "max", "n"},
    "flags":       {"boron_doping": int, "twoD_fwhm_warning": int,
                    "stage_1": int, "stage_2": int, "stage_transition": int},
    "doping":      {"n_type": int, "p_type": int, "undoped": int, "na": int},
    "dstar_co":    {"high": int, "low": int, "nd": int},
  }
"""

from __future__ import annotations

import numpy as np
from typing import Any
from .analyzer import RamanAnalysis


# Ratio attributes to summarise (attr on RamanAnalysis, display label)
_RATIO_ATTRS: list[tuple[str, str]] = [
    ("ID_IG_height",    "ID/IG (height)"),
    ("ID_IG_area",      "ID/IG (area)"),
    ("I2D_IG_height",   "I2D/IG (height)"),
    ("I2D_IG_area",     "I2D/IG (area)"),
    ("IDp_IG_height",   "ID\u2032/IG (height)"),
    ("ID_IDp_height",   "ID/ID\u2032 (height)"),
    ("IDstar_IG_height","ID*/IG (height)"),
    ("L_D_nm",          "L_D (nm)"),
]


def _scalar_stats(values: list[float]) -> dict[str, float]:
    """Return mean/std/median/min/max/n for a list of finite floats."""
    arr = np.array([v for v in values if np.isfinite(v)], dtype=float)
    n   = len(arr)
    if n == 0:
        return {"mean": np.nan, "std": np.nan, "median": np.nan,
                "min": np.nan, "max": np.nan, "n": 0}
    return {
        "mean":   float(arr.mean()),
        "std":    float(arr.std(ddof=1)) if n > 1 else 0.0,
        "median": float(np.median(arr)),
        "min":    float(arr.min()),
        "max":    float(arr.max()),
        "n":      n,
    }


def collect_batch_stats(batch_results: list[dict]) -> dict[str, Any]:
    """
    Compute summary statistics over a batch of RamanAnalysis objects.

    Parameters
    ----------
    batch_results : list of dicts with at minimum:
        {'filename': str, 'analysis': RamanAnalysis}

    Returns
    -------
    dict — see module docstring for structure.
    """
    n_total = len(batch_results)
    valid   = [r for r in batch_results
                if r.get("analysis") is not None]
    n_valid = len(valid)

    # ── Per-ratio statistics ──────────────────────────────
    ratio_stats: dict[str, dict] = {}
    for attr, label in _RATIO_ATTRS:
        vals = [getattr(r["analysis"], attr, np.nan) for r in valid]
        ratio_stats[label] = _scalar_stats(vals)

    # ── Flag counts ───────────────────────────────────────
    flags = {
        "boron_doping":      sum(1 for r in valid if r["analysis"].boron_doping_flag),
        "twoD_fwhm_warning": sum(1 for r in valid if r["analysis"].twoD_fwhm_warning),
        "stage_1":           0,
        "stage_2":           0,
        "stage_transition":  0,
    }
    for r in valid:
        stage = r["analysis"].stage_refined
        if "Stage 2" in stage:
            flags["stage_2"] += 1
        elif "transition" in stage:
            flags["stage_transition"] += 1
        elif "Stage 1" in stage:
            flags["stage_1"] += 1

    # ── Doping type distribution ──────────────────────────
    doping: dict[str, int] = {"n_type": 0, "p_type": 0,
                               "undoped": 0, "na": 0}
    for r in valid:
        dt = r["analysis"].doping_type
        if dt == "n-type":
            doping["n_type"] += 1
        elif dt == "p-type":
            doping["p_type"] += 1
        elif dt == "undoped":
            doping["undoped"] += 1
        else:
            doping["na"] += 1

    # ── D* C/O proxy distribution ─────────────────────────
    dstar_co: dict[str, int] = {"high": 0, "low": 0, "nd": 0}
    for r in valid:
        v = r["analysis"].IDstar_IG_height
        if np.isnan(v):
            dstar_co["nd"] += 1
        elif v > 0.15:
            dstar_co["high"] += 1
        else:
            dstar_co["low"] += 1

    return {
        "n_samples": n_total,
        "n_valid":   n_valid,
        "ratios":    ratio_stats,
        "flags":     flags,
        "doping":    doping,
        "dstar_co":  dstar_co,
    }


def print_batch_summary(stats: dict[str, Any]) -> str:
    """
    Format and return a human-readable batch summary string.
    Also prints to stdout.
    """
    sep  = "\u2550" * 60
    lines = [
        sep,
        "  BATCH RAMAN ANALYSIS SUMMARY",
        f"  Total samples  : {stats['n_samples']}",
        f"  Valid analyses : {stats['n_valid']}",
        sep,
        "",
        "  INTENSITY RATIOS  (mean \u00b1 1\u03c3  |  median  |  min \u2013 max)",
        f"  {'Ratio':<25} {'N':>4} {'Mean':>8} {'\u00b11\u03c3':>8} {'Median':>8} {'Min':>8} {'Max':>8}",
        f"  {'-'*25} {'-'*4} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}",
    ]

    def _f(v, fmt=".3f"):
        return format(v, fmt) if np.isfinite(v) else "  N/A"

    for label, s in stats["ratios"].items():
        lines.append(
            f"  {label:<25} {s['n']:>4} "
            f"{_f(s['mean']):>8} {_f(s['std']):>8} "
            f"{_f(s['median']):>8} {_f(s['min']):>8} {_f(s['max']):>8}"
        )

    f = stats["flags"]
    d = stats["doping"]
    dc = stats["dstar_co"]
    n  = stats["n_valid"] or 1

    lines += [
        "",
        "  STAGE CLASSIFICATION",
        f"  Stage 1             : {f['stage_1']:>3}  ({f['stage_1']/n*100:.0f}%)",
        f"  Stage 1\u21922 transition: {f['stage_transition']:>3}  ({f['stage_transition']/n*100:.0f}%)",
        f"  Stage 2             : {f['stage_2']:>3}  ({f['stage_2']/n*100:.0f}%)",
        "",
        "  D* BAND (C/O proxy)",
        f"  High (I_D*/I_G > 0.15) : {dc['high']:>3}  ({dc['high']/n*100:.0f}%)  \u2192 significant oxidation",
        f"  Low                    : {dc['low']:>3}  ({dc['low']/n*100:.0f}%)",
        f"  Not detected           : {dc['nd']:>3}  ({dc['nd']/n*100:.0f}%)",
        "",
        "  DOPING DISTRIBUTION  [Pisana 2007]",
        f"  n-type  : {d['n_type']:>3}   p-type : {d['p_type']:>3}",
        f"  Undoped : {d['undoped']:>3}   N/A    : {d['na']:>3}",
        "",
        "  FLAGS",
        f"  Boron doping detected  : {f['boron_doping']:>3}",
        f"  FWHM(2D) > 35 warning  : {f['twoD_fwhm_warning']:>3}",
        sep,
    ]

    text = "\n".join(lines)
    print(text)
    return text
