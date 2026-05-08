# Raman Spectrum Analyzer 🔬

A Python application for quantitative analysis of Raman spectra of **graphene and graphene-like (sp² carbon) materials**.

## Features

- **Baseline correction** — Asymmetric Least Squares (ALS) algorithm
- **Peak fitting** — Lorentzian (D, G, D') and Pseudo-Voigt (D+G) using `lmfit`
- **Quantitative analysis** — ID/IG, I2D/IG, ID'/IG, ID/ID' ratios
- **Defect characterization** — defect inter-distance L_D, disorder stage, defect type classification
- **Layer number estimation** — from I2D/IG ratio
- **Publication-quality plots** — 4 plots per spectrum (300 dpi)
- **Batch processing** — analyze entire folders at once
- **CSV + text reports** — structured output for all parameters

## Peak Windows (532 nm laser)

| Peak | Range (cm⁻¹) | Line Shape    | Physical Origin              |
|------|-------------|---------------|------------------------------|
| D    | 1270–1450   | Lorentzian    | Breathing mode, requires defect |
| G    | 1500–1620   | Lorentzian    | E₂g phonon, all sp² carbon   |
| D'   | 1600–1680   | Lorentzian    | Intravalley defect-induced    |
| 2D   | 2580–2780   | Lorentzian    | Second order of D, always active |
| D+G  | 2850–2960   | Pseudo-Voigt  | Combination band              |

> Peak positions shift with laser wavelength (dispersive peaks: D shifts ~53 cm⁻¹/eV, 2D shifts ~106 cm⁻¹/eV).

## Installation

```bash
git clone https://github.com/your-username/raman-analyzer.git
cd raman-analyzer
pip install -r requirements.txt
```

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

## Output

For each spectrum, the analyzer generates:
- `*_baseline.png` — raw spectrum + ALS baseline overlay
- `*_peaks.png` — fitted peaks on corrected spectrum
- `*_individual.png` — individual peak fits with residuals
- `*_ratios.png` — intensity ratio bar chart
- `*_report.txt` — full text report
- `raman_results.csv` — all parameters in one CSV (batch-compatible)

## Scientific References

- Ferrari & Robertson (2001) *Phys. Rev. B* **64**, 075414 — disorder stage classification
- Lucchese et al. (2010) *Carbon* **48**, 1592 — L_D formula from ID/IG
- Ferrari & Basko (2013) *Nature Nanotechnology* **8**, 235 — peak conventions & line shapes
- Eckmann et al. (2012) *Nano Letters* **12**, 3925 — defect type from ID/ID'

## Project Structure

```
raman_analyzer/
├── main.py                  ← CLI entry point
├── requirements.txt
├── README.md
├── examples/
│   ├── graphene_test.txt    ← synthetic test spectrum
│   └── generate_test_spectrum.py
├── results/                 ← output directory
└── src/
    ├── __init__.py
    ├── loader.py            ← file reading & preprocessing
    ├── baseline.py          ← ALS + linear baseline correction
    ├── peak_fitter.py       ← Lorentzian/Pseudo-Voigt fitting (lmfit)
    ├── analyzer.py          ← ratio calculations & defect analysis
    ├── plotter.py           ← matplotlib visualization (300 dpi)
    └── exporter.py          ← CSV + text report export
```

## Author

H.J — Researcher, Chemistry
