#!/usr/bin/env python3
"""
run_on_your_spectrum.py — end-to-end check of the full pipeline on YOUR data.

Usage:
    python validation/run_on_your_spectrum.py --file my.txt --laser 532 --material rGO

Runs despike -> baseline -> global fit -> analysis -> validation, then prints
each derived quantity next to the literature reference range pulled LIVE from
the tool's knowledge base (113 cited entries) — not a hand-typed table.
"""

import argparse
import os
import sys

import numpy as np

# make repo importable when run from anywhere
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

try:
    from src.loader import load_spectrum
    from src.baseline import correct_baseline
    from src.peak_fitter import fit_all_peaks
    from src.analyzer import analyze
    from src.validation import validate
    from src import knowledge as kb_mod
except ImportError as e:
    print(f"Error importing the tool ({e}). Run from the repo root or fix PYTHONPATH.")
    sys.exit(1)

try:
    from preprocessing import despike, Spectrum as _Spec
    _HAS_DESPIKE = True
except Exception:
    _HAS_DESPIKE = False

MATERIALS = ["graphene", "GO", "rGO", "graphite", "g-C3N4"]


def lit_range(kb, metric, material=None, laser_nm=None):
    """Literature span string from the knowledge base, or an em-dash."""
    lo, hi, srcs = kb.reference_range(metric, material=material, laser_nm=laser_nm)
    if lo is None:
        # retry without material filter (many entries are generic graphene)
        lo, hi, srcs = kb.reference_range(metric, laser_nm=laser_nm)
    if lo is None:
        return "\u2014"
    if abs(hi - lo) < 1e-12:
        return f"{lo:g} ({len(srcs)} ref)"
    return f"{lo:g}\u2013{hi:g} ({len(srcs)} refs)"


def fmt(v, spec=".3f"):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "N/A"
        return format(v, spec)
    except Exception:
        return str(v)


def main():
    ap = argparse.ArgumentParser(description="Full-pipeline check on your spectrum")
    ap.add_argument("--file", required=True, help="two-column spectrum (wn, intensity)")
    ap.add_argument("--laser", type=float, required=True, help="laser wavelength, nm")
    ap.add_argument("--material", choices=MATERIALS, required=True)
    args = ap.parse_args()

    wn, y = load_spectrum(args.file)

    if _HAS_DESPIKE:
        s = despike(_Spec(wn, y))
        wn, y = s.wavenumber, s.intensity

    corrected, _ = correct_baseline(wn, y, method="auto", material=args.material)
    peaks = fit_all_peaks(wn, corrected, laser_nm=args.laser,
                          adaptive_lineshape="auto", material=args.material)
    a = analyze(peaks, laser_nm=args.laser)
    report = validate(peaks, a, laser_nm=args.laser)
    kb = kb_mod.active()

    W = 78
    print("=" * W)
    print(f" {os.path.basename(args.file)}  |  laser {args.laser:g} nm  |  material {args.material}")
    print("=" * W)
    print(f"{'Quantity':<24}{'Your value':<22}Literature (knowledge base)")
    print("-" * W)
    m = args.material
    rows = [
        ("I_D/I_G (height)", fmt(a.ID_IG_height),
         lit_range(kb, "I_D/I_G", material=m, laser_nm=args.laser)),
        ("A_D/A_G (area)",   fmt(a.ID_IG_area),
         lit_range(kb, "A_D/A_G", laser_nm=args.laser)),
        ("I_D/I_D'",         fmt(getattr(a, "ID_IDp_height", float("nan"))),
         lit_range(kb, "I_D/I_D'")),
        ("L_D (nm)",         fmt(a.L_D_nm, ".1f"),
         lit_range(kb, "L_D", material=m)),
        ("  L_D note",       str(getattr(a, "L_D_note", "") or "\u2014"), ""),
        ("Defect type",      str(a.defect_type), ""),
        ("  range note",     str(getattr(a, "defect_type_range_note", "") or "\u2014"), ""),
        ("Disorder stage",   str(a.disorder_stage), ""),
        ("Layers (2D)",      str(getattr(a, "estimated_layers", "N/A")), ""),
    ]
    for name, val, lit in rows:
        print(f"{name:<24}{val:<22}{lit}")

    print("-" * W)
    print(" Fit-quality & validity flags:")
    if not report.flags:
        print("   (none) \u2014 no issues detected")
    else:
        for f in report.flags:
            band = f"[{f.band}] " if f.band else ""
            print(f"   {f.severity.value.upper():<9}{band}{f.message}")
    print("=" * W)
    print(" Compare 'Your value' against the literature span; investigate any")
    print(" flagged items before trusting derived quantities.")


if __name__ == "__main__":
    main()
