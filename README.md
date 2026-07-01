# Raman Spectrum Analyzer 🔬

A Python application for quantitative analysis of Raman spectra of **graphene and graphene-like (sp² carbon) materials**, with robust support for **doped and disordered graphene** (N-doped, B-doped, amorphous carbon).

## Features

- **Baseline correction** — Asymmetric Least Squares (ALS) algorithm
- **Peak fitting** — Lorentzian (D, G, D′) and Pseudo-Voigt (D+G) using pure `scipy` (no `lmfit` required)
- **Adaptive G-band fitting** — automatically locates the true G peak position before fitting; handles doped samples where G sits near or above 1600 cm⁻¹
- **G+D′ deconvolution** — dual-Lorentzian separation of overlapping G and D′ bands in disordered or doped graphene
- **Quantitative analysis** — I_D/I_G, I_2D/I_G, I_D′/I_G, I_D/I_D′ ratios
- **Defect characterization** — defect inter-distance L_D, disorder stage, defect type classification
- **Layer number estimation** — from I_2D/I_G ratio
- **Publication-quality plots** — 4 plots per spectrum (300 dpi)
- **Batch processing** — analyze entire folders at once
- **CSV + text reports** — structured output for all parameters

## Peak Windows (532 nm laser)

| Peak | Search Range (cm⁻¹) | Fit Window | Line Shape | Physical Origin |
|------|---------------------|------------|------------|-----------------|
| D    | 1270–1450 | fixed | Lorentzian | Breathing mode, requires defect |
| G    | 1540–1680 (search) | adaptive ±50 cm⁻¹ around detected peak | Lorentzian | E₂g phonon, all sp² carbon |
| D′   | 1610–1680 (standalone) or deconvolved from G+D′ fit | Lorentzian | Intravalley defect-induced |
| 2D   | 2580–2780 | fixed | Lorentzian (or dual) | Second order of D, always active |
| D+G  | 2850–2960 | fixed | Pseudo-Voigt | Combination band |

> **G-band strategy for doped / disordered samples:** the fitter first locates the true G peak via `find_peaks` in 1540–1680 cm⁻¹, then builds a ±50 cm⁻¹ adaptive window. If the single-Lorentzian R² < 0.60 (e.g. overlapping G+D′ in heavily doped or disordered materials), a dual-Lorentzian deconvolution is performed automatically. The D′ component from deconvolution is stored in `PeakResult.deconv_partner` and promoted to the `D_prime` result slot if it outperforms the standalone fit.

> Peak positions shift with laser wavelength (dispersive peaks: D shifts ~53 cm⁻¹/eV, 2D shifts ~106 cm⁻¹/eV).

## Installation

```bash
git clone https://github.com/Hj1308/Raman-analysis.git
cd Raman-analysis
pip install -r requirements.txt
```

**Dependencies:** `numpy`, `scipy`, `matplotlib`, `pandas`, `openpyxl`  
(no `lmfit` required — all fitting uses pure `scipy`)

## Usage

```bash
# Single file (532 nm laser)
python main.py --file spectrum.txt --laser 532

# Single file with custom output directory
python main.py --file spectrum.txt --laser 633 --output ./my_results/

# Batch mode — analyze all .txt/.csv files in a folder
python main.py --folder ./spectra/ --laser 785 --batch
```

## Input File Format

Plain text or CSV with two columns (wavenumber, intensity):
```
# Optional comment lines
1000.0   120.5
1001.0   122.3
...
```

Excel (`.xlsx`) files are also supported: wavenumber in column B, intensity in column C, data starting from row 4.

## Output

For each spectrum, the analyzer generates:
- `*_baseline.png` — raw spectrum + ALS baseline overlay
- `*_peaks.png` — fitted peaks on corrected spectrum
- `*_individual.png` — individual peak fits with residuals
- `*_ratios.png` — intensity ratio bar chart
- `*_report.txt` — full text report
- `raman_results.csv` — all parameters in one CSV (batch-compatible)

## PeakResult Fields

Each fitted peak is returned as a `PeakResult` dataclass:

| Field | Type | Description |
|-------|------|-------------|
| `center` | float | Peak position (cm⁻¹) |
| `amplitude` | float | Peak height (a.u.) |
| `fwhm` | float | Full width at half maximum (cm⁻¹) |
| `area` | float | Integrated area |
| `r_squared` | float | Goodness of fit |
| `found` | bool | True if R² > 0.60 |
| `is_deconvolved` | bool | True when G was separated from D′ by dual-Lorentzian |
| `deconv_partner` | PeakResult | D′ component extracted during G deconvolution |
| `is_split_2D` | bool | True when 2D was fitted with dual-Lorentzian (bilayer) |

## Scientific References

- Ferrari & Robertson (2001) *Phys. Rev. B* **64**, 075414 — disorder stage classification
- Lucchese et al. (2010) *Carbon* **48**, 1592 — L_D formula from I_D/I_G
- Ferrari & Basko (2013) *Nature Nanotechnology* **8**, 235 — peak conventions & line shapes
- Eckmann et al. (2012) *Nano Letters* **12**, 3925 — defect type from I_D/I_D′

## Project Structure

```
Raman-analysis/
├── main.py                  ← CLI entry point
├── requirements.txt
├── README.md
├── examples/
│   ├── graphene_test.txt    ← synthetic test spectrum
│   └── generate_test_spectrum.py
├── results/                 ← output directory
└── src/
    ├── __init__.py
    ├── loader.py            ← file reading & preprocessing (txt, csv, xlsx)
    ├── baseline.py          ← ALS baseline correction
    ├── peak_fitter.py       ← adaptive Lorentzian/Pseudo-Voigt fitting + G+D′ deconvolution
    ├── analyzer.py          ← ratio calculations & defect analysis
    ├── plotter.py           ← matplotlib visualization (300 dpi)
    └── exporter.py          ← CSV + text report export
```

## Changelog

### Latest — Adaptive G-band & Deconvolution
- `peak_fitter.py`: G-band window is now **adaptive** — the fitter detects the true G peak location in 1540–1680 cm⁻¹ before fitting, resolving failures on N-doped / B-doped and disordered graphene where G sits near 1600 cm⁻¹
- Added `_fit_G_deconvolve()`: automatic **dual-Lorentzian G+D′ deconvolution** when single-peak R² < 0.60
- `PeakResult` extended with `is_deconvolved` and `deconv_partner` fields
- Removed `lmfit` dependency; all fitting uses pure `scipy.optimize.curve_fit`

## Author

H.J — Researcher, Chemistry
