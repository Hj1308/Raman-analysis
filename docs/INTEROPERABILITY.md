# Interoperability with RamanSPy

This tool is a **domain-specific interpretation layer** for sp² carbons
(graphene / GO / rGO / g-C₃N₄), not a general-purpose Raman framework. It is
designed to sit *on top of* the open-source ecosystem rather than replace it:

```
RamanSPy / pybaselines  ->  this tool
(loading, preprocessing)    (D/G/D' global fit, defect classification,
                             L_D / n_D with uncertainty, literature-anchored
                             interpretation, validation flags)
```

Where the ecosystem is used automatically:

| Capability | Provider | When |
|---|---|---|
| Baseline (arPLS / asPLS / auto) | **pybaselines** | whenever installed (`pip install pybaselines`) |
| Cosmic-ray removal (Whitaker–Hayes) | **RamanSPy** | whenever installed; z-score fallback otherwise |
| Instrument formats (.wdf, .spc) | **RamanSPy** loaders | via `load_spectrum()` |

## Example: RamanSPy pipeline → this tool's interpretation

```python
import ramanspy as rp
import numpy as np

# 1) Load + preprocess with RamanSPy (its strength)
spectrum = rp.load.witec("sample.wdf")
pipeline = rp.preprocessing.Pipeline([
    rp.preprocessing.despike.WhitakerHayes(),
    rp.preprocessing.denoise.SavGol(window_length=9, polyorder=3),
    rp.preprocessing.baseline.ASPLS(),
    rp.preprocessing.normalise.MinMax(),
])
clean = pipeline.apply(spectrum)
wn = np.asarray(clean.spectral_axis)
intensity = np.asarray(clean.spectral_data)

# 2) Interpret with this tool (our strength)
from src.peak_fitter import fit_all_peaks
from src.analyzer import analyze
from src.validation import validate

peaks = fit_all_peaks(wn, intensity, laser_nm=532,
                      adaptive_lineshape="auto", material="rGO")
analysis = analyze(peaks, laser_nm=532)
report = validate(peaks, analysis, laser_nm=532)

print(analysis.defect_type)             # Eckmann-anchored, range-aware
print(analysis.defect_type_range_note)  # citation-backed range assessment
print(report.summary())                 # fit-quality & literature flags
```

## Or let this tool drive everything

If you don't need a custom RamanSPy pipeline, the built-in path already uses
the same ecosystem under the hood:

```python
from src.loader import load_spectrum        # RamanSPy loaders for .wdf
from src.baseline import correct_baseline   # pybaselines arPLS/asPLS
from preprocessing import despike, Spectrum # Whitaker–Hayes despike

wn, intensity = load_spectrum("sample.txt")
clean = despike(Spectrum(wn, intensity))
corrected, _ = correct_baseline(clean.wavenumber, clean.intensity,
                                method="auto", material="rGO")
# ... then fit_all_peaks / analyze / validate as above
```

Everything degrades gracefully: without `ramanspy`/`pybaselines` installed the
tool falls back to built-in numpy/scipy implementations, so a minimal install
still runs — just with fewer preprocessing niceties.
