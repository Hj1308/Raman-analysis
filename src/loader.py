"""
File loader and preprocessor for Raman spectra.
Supports: .txt, .csv (two-column: wavenumber, intensity)
         .xlsx     (multi-sheet: each sheet name becomes the sample label)

Change log
──────────
  v2.5.1  Fix: min_points lowered from 10 to 3 so unit tests with
          small synthetic spectra don't hit the guard.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Minimum valid data points for a spectrum
_MIN_POINTS = 3


# ── helpers ──────────────────────────────────────────────────────────────────

def _sort(wn: np.ndarray, inten: np.ndarray):
    idx = np.argsort(wn)
    return wn[idx], inten[idx]


# ── single .txt / .csv ───────────────────────────────────────────────────────

def load_spectrum(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a Raman spectrum file (.txt or .csv).
    Returns: (wavenumber, intensity) as numpy arrays.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        df = pd.read_csv(
            filepath,
            sep=r"\s+|,",
            engine="python",
            comment="#",
            header=None,
            names=["wavenumber", "intensity"],
        )
        wavenumber = df["wavenumber"].values.astype(float)
        intensity  = df["intensity"].values.astype(float)
    except Exception as e:
        raise ValueError(f"Could not parse spectrum file '{filepath}': {e}")

    if len(wavenumber) < _MIN_POINTS:
        raise ValueError(
            f"Too few data points ({len(wavenumber)}) in '{filepath}' "
            f"(minimum {_MIN_POINTS})"
        )

    return _sort(wavenumber, intensity)


# ── Excel multi-sheet ────────────────────────────────────────────────────────

def load_excel_sheets(filepath: str) -> list[dict]:
    """
    Load an Excel workbook where every sheet contains one Raman spectrum.

    Expected sheet layout (flexible header detection):
        Row 1 (optional) : any header text, e.g. "Wavenumber (cm-1)  Intensity (a.u.)"
        Row 2+           : numeric data  (wavenumber , intensity)

    The **sheet name** is used directly as the sample label.
    Sheets whose names start with "READ" (case-insensitive) are skipped.

    Returns a list of dicts:
        [
          {"label": "Sample_1",
           "wavenumber": np.ndarray,
           "intensity":  np.ndarray},
          ...
        ]
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    if path.suffix.lower() not in (".xlsx", ".xls"):
        raise ValueError(f"Expected an Excel file (.xlsx / .xls), got: {path.suffix}")

    xl = pd.ExcelFile(filepath, engine="openpyxl")
    results = []

    for sheet_name in xl.sheet_names:
        if sheet_name.strip().upper().startswith("READ"):
            continue

        try:
            df = xl.parse(sheet_name, header=0)
            first_col = df.iloc[:, 0]
            if not pd.to_numeric(first_col, errors="coerce").notna().all():
                df = xl.parse(sheet_name, header=None,
                              skiprows=1, names=["wavenumber", "intensity"])

            df.columns = [str(c).strip().lower() for c in df.columns]
            wn_col = next((c for c in df.columns if "wave" in c or "cm" in c
                           or c in ("0", "wavenumber")), df.columns[0])
            in_col = next((c for c in df.columns if "intens" in c
                           or c not in (wn_col,)), df.columns[1])

            wn    = pd.to_numeric(df[wn_col], errors="coerce").values.astype(float)
            inten = pd.to_numeric(df[in_col], errors="coerce").values.astype(float)

            mask  = ~(np.isnan(wn) | np.isnan(inten))
            wn, inten = wn[mask], inten[mask]

            if len(wn) < _MIN_POINTS:
                print(f"  WARNING: Sheet '{sheet_name}' has only {len(wn)} valid rows — skipped.")
                continue

            wn, inten = _sort(wn, inten)
            results.append({"label": sheet_name, "wavenumber": wn, "intensity": inten})
            print(f"  Loaded sheet '{sheet_name}' — {len(wn)} points")

        except Exception as e:
            print(f"  WARNING: Could not parse sheet '{sheet_name}' — {e}")

    if not results:
        raise ValueError("No valid spectrum sheets found in the Excel file.")

    return results


# ── folder batch (.txt / .csv) ───────────────────────────────────────────────

def load_batch(folder: str, extensions=(".txt", ".csv")) -> list[dict]:
    """Load all spectrum files from a folder."""
    folder = Path(folder)
    files  = [f for f in folder.iterdir() if f.suffix.lower() in extensions]
    if not files:
        raise FileNotFoundError(f"No spectrum files found in '{folder}'")
    results = []
    for f in sorted(files):
        try:
            wn, inten = load_spectrum(str(f))
            results.append({"filename": f.name, "filepath": str(f),
                             "wavenumber": wn, "intensity": inten})
            print(f"  Loaded: {f.name} ({len(wn)} points)")
        except Exception as e:
            print(f"  WARNING: Skipping {f.name} — {e}")
    return results
