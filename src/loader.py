"""
File loader and preprocessor for Raman spectra.
Supports: .txt, .csv (two-column: wavenumber, intensity)
"""

import numpy as np
import pandas as pd
from pathlib import Path


def load_spectrum(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a Raman spectrum file.
    Returns: (wavenumber, intensity) as numpy arrays.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        # Try reading with pandas — handles both comma and whitespace separators
        df = pd.read_csv(filepath, sep=r"\s+|,", engine="python",
                         comment="#", header=None, names=["wavenumber", "intensity"])
        wavenumber = df["wavenumber"].values.astype(float)
        intensity  = df["intensity"].values.astype(float)
    except Exception as e:
        raise ValueError(f"Could not parse spectrum file '{filepath}': {e}")

    if len(wavenumber) < 10:
        raise ValueError(f"Too few data points ({len(wavenumber)}) in '{filepath}'")

    # Sort by wavenumber (ascending)
    sort_idx   = np.argsort(wavenumber)
    wavenumber = wavenumber[sort_idx]
    intensity  = intensity[sort_idx]

    return wavenumber, intensity


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
