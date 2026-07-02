#!/usr/bin/env python
"""
tools/manual_fwhm_cli.py
========================
Interactive CLI tool for manual peak-picking and numerical FWHM.

The researcher clicks on peak positions in a matplotlib window;
the tool computes FWHM numerically (no parametric fit) for each
picked position and prints a summary table.

Usage
-----
    python tools/manual_fwhm_cli.py spectrum.txt
    python tools/manual_fwhm_cli.py spectrum.csv --laser 633 --window 35
    python tools/manual_fwhm_cli.py spectrum.xlsx --output picks.csv
    python tools/manual_fwhm_cli.py spectrum.txt --no-gui   # keyboard entry

Arguments
---------
  file            Path to Raman spectrum (.txt / .csv / .xlsx)
  --laser         Laser wavelength in nm  (default: 532)
  --window        Half-width of FWHM window in cm⁻¹ (default: 40)
  --output FILE   Save results to CSV (optional)
  --no-gui        Disable matplotlib GUI; enter peak centres via keyboard

Dependencies: numpy, matplotlib (GUI only), src.loader, src.peak_fitter
"""

import argparse
import sys
import csv
from pathlib import Path

import numpy as np

# ── project root on sys.path ────────────────────────────────
_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from src.loader      import load_spectrum          # noqa: E402
from src.baseline    import als_baseline           # noqa: E402
from src.peak_fitter import manual_peak_fwhm       # noqa: E402


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────
def _print_table(rows: list[dict]) -> None:
    """Pretty-print results table to stdout."""
    header = f"{'#':>3}  {'Center (cm⁻¹)':>16}  {'FWHM (cm⁻¹)':>14}  {'Window (cm⁻¹)':>16}"
    sep    = "-" * len(header)
    print()
    print("  MANUAL FWHM RESULTS")
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        fwhm_str = f"{r['fwhm_cm1']:>10.2f}" if not np.isnan(r['fwhm_cm1']) else "       N/A"
        win_str  = f"{r['center_cm1'] - r['half_width']:.1f} – {r['center_cm1'] + r['half_width']:.1f}"
        print(f"  {r['idx']:>1}  {r['center_cm1']:>16.1f}  {fwhm_str:>14}  {win_str:>16}")
    print(sep)
    print()


def _save_csv(rows: list[dict], path: str) -> None:
    out = Path(path)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh,
            fieldnames=["idx", "center_cm1", "fwhm_cm1", "half_width"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Results saved → {out}")


# ──────────────────────────────────────────────────────────
# GUI pick (matplotlib)
# ──────────────────────────────────────────────────────────
def _gui_pick(wn: np.ndarray, intensity: np.ndarray,
              filename: str) -> list[float]:
    """
    Open a matplotlib window, let user click on peaks.
    Returns list of x-coordinates (cm⁻¹) of clicked positions.

    Controls
    --------
    Left-click   : add a peak marker at cursor position
    Right-click  : remove the last marker
    Enter / close: finish picking
    """
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.rcParams.update({
        "figure.facecolor":  "#1c1b19",
        "axes.facecolor":    "#201f1d",
        "axes.edgecolor":    "#393836",
        "axes.labelcolor":   "#cdccca",
        "xtick.color":       "#797876",
        "ytick.color":       "#797876",
        "text.color":        "#cdccca",
        "grid.color":        "#262523",
        "grid.linestyle":    "--",
        "grid.alpha":        0.4,
    })

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(wn, intensity, color="#4f98a3", lw=1.2, label="Spectrum")
    ax.set_xlabel("Raman shift (cm⁻¹)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_title(
        f"{Path(filename).name}  —  left-click to pick peaks  |  "
        "right-click to undo last  |  close window when done",
        fontsize=9,
    )
    ax.grid(True)

    picks: list[float] = []
    markers: list = []

    def _on_click(event):
        if event.inaxes is not ax:
            return
        if event.button == 1:   # left-click → add
            x = event.xdata
            picks.append(x)
            line = ax.axvline(x, color="#fdab43", lw=1.0, ls="--", alpha=0.8)
            txt  = ax.text(
                x, ax.get_ylim()[1] * 0.92,
                f"{x:.1f}",
                color="#fdab43", fontsize=7, ha="center", va="top",
            )
            markers.append((line, txt))
            fig.canvas.draw_idle()
        elif event.button == 3 and picks:  # right-click → undo
            picks.pop()
            ln, tx = markers.pop()
            ln.remove()
            tx.remove()
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", _on_click)
    plt.tight_layout()
    plt.show(block=True)
    return picks


# ──────────────────────────────────────────────────────────
# Keyboard fallback
# ──────────────────────────────────────────────────────────
def _keyboard_pick() -> list[float]:
    """
    Keyboard entry fallback when matplotlib GUI is not available.
    User enters peak centres separated by commas or spaces.
    """
    print()
    print("  [no-gui mode]  Enter peak centres (cm⁻¹), separated by spaces or commas.")
    print("  Example:  1350.5 1582 2695")
    raw = input("  > ").strip()
    picks = []
    for tok in raw.replace(",", " ").split():
        try:
            picks.append(float(tok))
        except ValueError:
            print(f"  Warning: '{tok}' is not a number — skipped.")
    return picks


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="manual_fwhm_cli",
        description="Interactive manual peak-picking and numerical FWHM tool.",
    )
    parser.add_argument("file",
        help="Raman spectrum file (.txt / .csv / .xlsx)")
    parser.add_argument("--laser", type=float, default=532.0,
        metavar="NM",
        help="Laser wavelength in nm (default: 532)")
    parser.add_argument("--window", type=float, default=40.0,
        metavar="HW",
        help="Half-width of FWHM window in cm⁻¹ (default: 40)")
    parser.add_argument("--output", metavar="CSV",
        help="Save results to this CSV file")
    parser.add_argument("--no-gui", action="store_true",
        help="Disable matplotlib GUI; enter peaks via keyboard")
    args = parser.parse_args()

    # ── load spectrum ──────────────────────────────────────
    spec_path = Path(args.file)
    if not spec_path.exists():
        print(f"Error: file not found — {spec_path}")
        sys.exit(1)

    print(f"\n  Loading  : {spec_path.name}")
    print(f"  Laser    : {args.laser:.0f} nm")
    print(f"  HW window: ± {args.window:.0f} cm⁻¹")

    try:
        wn, raw_intensity = load_spectrum(str(spec_path))
    except Exception as exc:
        print(f"  Error loading file: {exc}")
        sys.exit(1)

    # ── baseline correction ────────────────────────────────
    baseline  = als_baseline(raw_intensity)
    intensity = raw_intensity - baseline
    intensity = np.clip(intensity, 0, None)

    print(f"  Points   : {len(wn)}")
    print(f"  Range    : {wn.min():.0f} – {wn.max():.0f} cm⁻¹")

    # ── peak picking ───────────────────────────────────────
    if args.no_gui:
        centres = _keyboard_pick()
    else:
        try:
            centres = _gui_pick(wn, intensity, args.file)
        except Exception as exc:
            print(f"  GUI unavailable ({exc}); falling back to keyboard entry.")
            centres = _keyboard_pick()

    if not centres:
        print("  No peaks picked. Exiting.")
        sys.exit(0)

    # ── compute FWHM for each pick ────────────────────────
    rows: list[dict] = []
    for i, centre in enumerate(centres, start=1):
        fwhm = manual_peak_fwhm(
            wn, intensity,
            peak_center=centre,
            window_half_width=args.window,
        )
        rows.append({
            "idx":        i,
            "center_cm1": round(centre, 2),
            "fwhm_cm1":   round(fwhm, 4) if not np.isnan(fwhm) else float("nan"),
            "half_width": args.window,
        })

    # ── output ─────────────────────────────────────────────
    _print_table(rows)

    if args.output:
        _save_csv(rows, args.output)


if __name__ == "__main__":
    main()
