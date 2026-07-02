# Raman Spectrum Analyzer 🔬

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21109057.svg)](https://doi.org/10.5281/zenodo.21109057)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A Python application for quantitative analysis of Raman spectra of **graphene and graphene-like (sp² carbon) materials**, with robust support for **doped and disordered graphene** (N-doped, B-doped, SiC-graphene composites, g-C₃N₄, rGO).

## Physical Background

Raman spectroscopy is a non-destructive, chemically specific technique widely used to probe the structural quality and electronic properties of sp² carbon materials. The main spectral features and their physical origins are:

| Band | Typical Position (532 nm) | Origin | Activity |
|------|--------------------------|--------|----------|
| D    | ~1345 cm⁻¹ | Intervalley double resonance, K-point phonon; **requires a defect** for activation | Defect-induced |
| G    | ~1580 cm⁻¹ | In-plane C–C stretching (E₂g), present in all sp² carbon | Always active |
| D′   | ~1620 cm⁻¹ | Intravalley double resonance; activated by defects | Defect-induced |
| 2D   | ~2690 cm⁻¹ | Second-order of D (two-phonon); **no defect needed** | Always active |
| D+G  | ~2940 cm⁻¹ | Combination of D and G phonons | Defect-assisted |

> The D and 2D bands are **dispersive**: their positions shift with laser excitation energy at ~53 cm⁻¹/eV and ~106 cm⁻¹/eV respectively (double the D slope, because 2D is a two-phonon process). The G band position is non-dispersive but shifts with doping — both electron and hole doping upshift the G band due to phonon stiffening (Kohn anomaly removal).

### Boron-Doped Graphene (g-BSiC)

Substitutional boron introduces **chemical disorder** into the sp² network: the B–C bond is ~0.5 Å longer than the C–C bond, breaking translational symmetry and activating the D band. Key Raman signatures of B-doping include:

- **Elevated I_D/I_G**: B-doped single-layer graphene can reach I_D/I_G ≈ 7, driven by electron–boron elastic scattering *before* phonon emission [Kim 2012].
- **G band position unchanged** (vs. point defects): in substitutionally B-doped graphene, the G-band upshift from p-type doping (hole carrier injection) is counterbalanced by tensile-strain softening from the longer B–C bond. This produces a *constant* G-band position as I_D/I_G increases — a diagnostic that **distinguishes B-doping from argon-plasma-induced point defects**, where G blueshifts monotonically [Kim 2012].
- **D′ band at ~1620 cm⁻¹**: equal intensity to G band in heavily B-doped samples. The D′ band shows weak dispersive behaviour in B-doped graphene (~36 cm⁻¹/eV vs. ~77 cm⁻¹/eV for the 2D band).
- **2D (G′) downshift**: continuous 2D downshift of ~4 cm⁻¹ per unit I_D/I_G up to I_D/I_G ≈ 3; beyond that, strain-modified double resonance conditions dominate [Kim 2012].
- **Defect spacing (L_D)**: using the empirical relation I_G/I_D ≈ 10² × L_D² (valid in Stage 1), the average B–B distance in single-layer B-graphene is ~4.76 nm [Kim 2012].

### Nitrogen-Doped Graphene (g-NSiC)

Nitrogen substitution creates n-type doping, downshifts the 2D band, and typically produces lower I_D/I_G than B-doping at comparable dopant concentrations (N–C bond length is closer to C–C). The G band upshifts due to Fermi-level elevation.

### Graphitic C₃N₄ Phase (g-C₃N₄)

Graphitic carbon nitride has a fundamentally different Raman fingerprint from graphene. With **visible excitation (514/532 nm)**, strong fluorescence background often obscures the true Raman peaks, yielding only two broad overlapping bands near 1357 cm⁻¹ (D-like) and 1560 cm⁻¹ (G-like) [Zinin 2009]. These are **not** the graphene D and G bands — they are artefacts of fluorescence and visible-light limitations.

With **UV (244 nm) or NIR (785 nm) excitation**, the true g-C₃N₄ spectrum emerges:

- **~691 cm⁻¹**: sharp peak assigned to **s-triazine ring breathing mode** (ring–C–N in-plane vibration); analogous to the 677 cm⁻¹ mode in melamine [Zinin 2009].
- **~988 cm⁻¹**: second strong mode, assigned to a **second type of triazine ring breathing** (bre mode); closely matches melamine at 983 cm⁻¹ [Zinin 2009].
- **~1596 cm⁻¹**: C–N ring stretching vibration.
- **~1728 cm⁻¹**: unassigned high-frequency mode.
- The D band is **absent in UV Raman** of g-C₃N₄ — analogous to diamond-like carbon where D-band intensity decreases with UV excitation [Zinin 2009].

> ⚠️ **Important for this analyser:** When a spectrum exhibits the 691 / 988 cm⁻¹ doublet but lacks a clear G band, it should be interpreted as **g-C₃N₄**, not disordered graphene. The visible-range bands at 1357 and 1560 cm⁻¹ in g-C₃N₄ are not reliable structural indicators. The present analyser targets graphenic (sp² C–C) materials; g-C₃N₄ characterisation requires NIR or UV excitation for accurate interpretation.

### Reduced Graphene Oxide (rGO)

Structural evolution from graphite → GO → rGO can be tracked through D/G band changes [Mohan 2017]:

- **Graphite**: sharp G at ~1582 cm⁻¹, very weak D at ~1368 cm⁻¹ (edge-only). D-band intensity depends on edge type: strong at armchair edges, weak at zigzag.
- **GO**: broad D and G bands; high oxygen content disrupts conjugation; low electrical conductivity.
- **rGO**: I_D/I_G increases after reduction as new graphitic crystallites form (restoration of π-conjugation). The highest electrical conductivity (103.3 S cm⁻¹) was achieved with HI reduction, correlating with lowest defect density and narrowest D/G FWHM [Mohan 2017].
- **FWHM(G)**: consistent FWHM(G) across reduction stages indicates uniform sp² cluster size; increasing FWHM(D) signals more disordered graphitic rings [Mohan 2017].
- **I_2D/I_G**: graphene (CVD) shows I_2D/I_G ≈ 0.179 and a single sharp 2D peak at ~2690 cm⁻¹. rGO shows a distorted, broad 2D region or none at all — the 2D peak's presence depends on preparation method [Mohan 2017].
- **Functionalised rGO (frGO-PYS)**: G band downshifts by ~5 cm⁻¹ (to 1594 cm⁻¹) due to additional charge carriers from π–π interaction with pyrene molecules, providing a Raman indicator for non-covalent functionalisation [Mohan 2017].

---

## Features

- **Baseline correction** — Asymmetric Least Squares (ALS); negative residuals preserved (no clipping)
- **Peak fitting** — Lorentzian (D, G, D′) and Pseudo-Voigt (D+G) using pure `scipy`; optional `lmfit` backend via `band_config`
- **Global D / G / D′ fit** — D, G, and D′ are fitted simultaneously to prevent D′ from capturing the G tail
- **Adaptive G-band fitting** — locates the true G peak position before fitting; handles doped samples where G sits near or above 1600 cm⁻¹ (e.g., B-doped graphene where p-type doping stiffening shifts the G band)
- **G+D′ deconvolution** — dual-Lorentzian separation of overlapping G and D′ bands in disordered or doped graphene
- **SNR-gated peak detection** — a peak is accepted only when R² > 0.75 **and** SNR > 3 (SNR = amplitude / MAD-based noise estimate)
- **Pseudo-Voigt area correction** — FWHM and integrated area follow the Thompson–Cox–Hastings definition
- **Quantitative analysis** — I_D/I_G, I_2D/I_G, I_D′/I_G, I_D/I_D′ ratios
- **Defect characterisation** — L_D (Cançado 2011, E_L⁴-corrected), disorder stage, defect type classification
- **Layer number estimation** — from I_2D/I_G with FWHM(2D) reliability guard
- **eV-based laser dispersion** — peak windows shift with laser energy per Cançado 2011
- **Publication-quality plots** — 4 plots per spectrum (300 dpi)
- **Batch processing** — analyse entire folders at once
- **CSV + text reports** — structured output for all parameters

## Peak Windows (532 nm laser)

| Peak | Search Range (cm⁻¹) | Fit Window | Line Shape | Physical Origin |
|------|---------------------|------------|------------|-----------------:|
| D    | 1270–1450 | fixed | Lorentzian | K-point phonon, intervalley double resonance; requires defect [Ferrari 2001] |
| G    | 1540–1680 (search) | adaptive ±50 cm⁻¹ around detected peak | Lorentzian | E₂g in-plane C–C stretch; all sp² carbon [Tuinstra 1970] |
| D′   | global fit with D+G or standalone 1610–1680 | Lorentzian | Intravalley defect-induced; I_D/I_D′ distinguishes defect type [Eckmann 2012] |
| 2D   | 2580–2780 | fixed | Lorentzian (or dual) | Second-order overtone of D; no defect needed; sensitive to layer number [Ferrari 2006] |
| D+G  | 2850–2960 | fixed | Pseudo-Voigt | Combination band |

> **G-band strategy for doped / disordered samples:** the fitter first locates the true G peak via `find_peaks` in 1540–1680 cm⁻¹ (prominence threshold = 5 % of spectrum maximum), then builds a ±50 cm⁻¹ adaptive window. If the single-Lorentzian R² < 0.60 (e.g. overlapping G+D′ in heavily doped or disordered materials), a dual-Lorentzian deconvolution is performed automatically. The D′ component from deconvolution is stored in `PeakResult.deconv_partner` and promoted to the `D_prime` result slot if it outperforms the standalone fit.

> **Global D / G / D′ fit:** by default D, G, and D′ are optimised in a single least-squares call (`_fit_D_G_Dp_global`). This prevents spurious D′ detections caused by fitting D′ in isolation against the G tail.

> Peak positions shift with laser wavelength. Dispersive peaks: D ~53 cm⁻¹/eV, 2D ~106 cm⁻¹/eV (Cançado 2011, eV-based).

## Quantitative Metrics

### Defect Density — L_D

The inter-defect distance L_D (nm) is computed from the laser-energy-corrected Cançado 2011 formula:

```
L_D² (nm²) = (4.3 × 10³ / E_L⁴) × (I_G / I_D)
```

where E_L is the laser energy in eV. This formula is valid in **Stage 1 only** (low-defect regime, L_D ≳ 10 nm, I_D/I_G rising). Assuming substitutional dopants behave as vacancy-like defects, the same formula gives the average dopant–dopant spacing — e.g., ~4.76 nm for 0.22 at% B-doped single-layer graphene [Kim 2012].

### Disorder Stage Classification

| Stage | I_D/I_G trend | G-band FWHM | L_D | Physical regime |
|-------|--------------|-------------|-----|-----------------|
| Stage 1 | Increases with disorder | Broadens moderately | ≳ 10 nm | Crystalline graphene with point defects; Tuinstra–Koenig regime |
| Stage 2 | Decreases with disorder | Strongly broadened | < 10 nm | Amorphous / nanocrystalline carbon; L_D formula breaks down |

> FWHM(G) consistency across samples signals uniform sp² cluster size; I_D/I_G combined with FWHM(G) gives a more robust disorder metric than I_D/I_G alone [Mohan 2017].

### Defect Type — I_D / I_D′

The ratio I_D/I_D′ allows discrimination between defect types [Eckmann 2012]:

| I_D/I_D′ | Defect type |
|-----------|-------------|
| ~13 | sp³ (homoatomic, e.g. H, O functional groups) |
| ~7  | Vacancy-like (structural) |
| ~3.5 | Grain boundaries / edge defects |

In B-doped graphene, substitutional B atoms behave as vacancy-like defects → I_D/I_D′ ≈ 7 [Kim 2012].

### Layer Number — I_2D/I_G

- **Monolayer:** I_2D/I_G > 2, single sharp Lorentzian 2D peak (FWHM ~25 cm⁻¹) [Ferrari 2006]
- **Bilayer:** I_2D/I_G ≈ 1, 2D splits into 4 Lorentzian components
- **Few-layer (> 3):** I_2D/I_G < 1; 2D broadens, approaches graphite lineshape
- **rGO / functionalised graphene:** 2D region distorted or absent; I_2D/I_G not a reliable layer indicator [Mohan 2017]

## Installation

```bash
git clone https://github.com/Hj1308/Raman-analysis.git
cd Raman-analysis
pip install -r requirements.txt
```

**Dependencies:** `numpy`, `scipy`, `matplotlib`, `pandas`, `openpyxl`  
`lmfit` is optional — install it only if you use advanced `band_config` options (Voigt, AsymLorentzian, uncertainty reporting).

## Usage

```bash
# Single file (532 nm laser)
python main.py --file spectrum.txt --laser 532

# Single file with custom output directory
python main.py --file spectrum.txt --laser 633 --output ./my_results/

# Batch mode — analyse all .txt/.csv/.xlsx files in a folder
python main.py --folder ./spectra/ --laser 785 --batch
```

### Advanced: `band_config`

Pass a `band_config` dict to `fit_all_peaks()` to override the default fit strategy per band:

```python
from src.peak_fitter import PeakFitter

config = {
    "G": {"method": "adaptive"},          # default adaptive window
    "D": {"lineshape": "Lorentzian"},      # explicit lineshape
    "2D": {"method": "deconvolve"},        # force dual-Lorentzian
    "D+G": {"lineshape": "Voigt"},         # requires lmfit
}
results = fitter.fit_all_peaks(band_config=config)
```

If `lmfit` is not installed, Voigt / AsymLorentzian options fall back silently to Lorentzian and a warning is stored in `PeakResult.name`.

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

For each spectrum, the analyser generates:
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
| `area` | float | Integrated area (Thompson–Cox–Hastings for Pseudo-Voigt) |
| `r_squared` | float | Goodness of fit |
| `found` | bool | True if R² > 0.75 and SNR > 3 |
| `is_deconvolved` | bool | True when G was separated from D′ by dual-Lorentzian |
| `deconv_partner` | PeakResult \| None | D′ component extracted during G deconvolution |
| `is_split_2D` | bool | True when 2D was fitted with dual-Lorentzian (bilayer) |
| `twoD_fwhm_warning` | bool | True when FWHM(2D) > 35 cm⁻¹ (layer count unreliable) |
| `center_stderr` | float \| None | Standard error on center (lmfit backend only) |
| `fwhm_stderr` | float \| None | Standard error on FWHM (lmfit backend only) |

## Known Limitations

### 2D Band Fitting
The current implementation uses a **single Lorentzian** for the 2D band. In few-layer graphene (bilayer, trilayer), the 2D band splits into multiple sub-components requiring 4-Lorentzian decomposition. Single-peak fitting for bilayer samples will underestimate FWHM and may misassign the layer count. For monolayer graphene, the single Lorentzian is appropriate.
> Reference: Ferrari et al. (2006) *Phys. Rev. Lett.* **97**, 187401

### Laser Wavelength Dependence of L_D
L_D is calculated using the Cançado 2011 formula with full E_L⁴ dependence. This formula is valid in **Stage 1 only**. When the sample is classified as Stage 2, L_D is set to `NaN` and a warning is stored in `L_D_note`.
> Reference: Cançado et al. *Nano Lett.* **11**, 3190–3196 (2011)

### Disorder Stage Boundary
The current Stage 1 / Stage 2 boundary is based on I_D/I_G threshold. A more accurate discrimination uses FWHM(G) combined with A_D/A_G (area ratio), and distinguishes 0D point defects from 1D line/edge defects via the I_D/I_D′ ratio. In B-doped graphene, the disorder stage must be interpreted carefully: high I_D/I_G does not imply amorphous carbon — it reflects chemical disorder from substitutional boron [Kim 2012].
> Reference: Cançado et al. *Carbon* reviews; Eckmann et al. *Nano Lett.* **12**, 3925 (2012)

### g-C₃N₄ and Visible Excitation
The analyser is designed for sp² graphenic materials. Spectra measured with visible (514–633 nm) excitation on g-C₃N₄ show strong fluorescence-masked D/G-like bands at ~1357 and ~1560 cm⁻¹ that are not reliable structural indicators [Zinin 2009]. For g-C₃N₄ characterisation, NIR (785 nm) or UV (244 nm) excitation is required to resolve the diagnostic triazine ring breathing modes at ~691 and ~988 cm⁻¹.

### Layer Number Accuracy
Layer estimation from I_2D/I_G is reliable for 1–3 layers and only when FWHM(2D) ≤ 35 cm⁻¹. For rGO and functionalised graphene, the 2D band shape is distorted and I_2D/I_G is not a reliable layer indicator [Mohan 2017].
> Reference: Ferrari & Basko *Nature Nanotechnology* **8**, 235 (2013)

### G+D′ Overlap in GO/rGO and Doped Graphene
In graphene oxide (GO), reduced GO, and heavily B-doped graphene, the G and D′ bands often overlap severely. The current deconvolution assumes two Lorentzians; more complex multi-band models may be needed for quantitative separation in extreme cases.
> Reference: Claramunt et al. *Sci. Rep.* **5**, 19491 (2015)

## Roadmap

- [ ] **Multi-Lorentzian 2D fitting** — 4-component decomposition for bilayer/trilayer graphene
- [ ] **Fitting uncertainty (scipy)** — error bars from covariance matrix for all users (no lmfit required)
- [ ] **Intensity normalisation** — option to normalise to G peak amplitude or area
- [ ] **Advanced baseline** — arPLS (asymmetrically reweighted PLS) for noisy/complex spectra
- [ ] **Doping/strain flags** — detect anomalous G and 2D positions indicating charge doping or strain (B-doping: constant G position; n-doping: 2D downshift)
- [ ] **Batch statistics** — mean ± std across all samples for batch runs
- [ ] **Defect type refinement** — FWHM(G) + A_D/A_G for 0D vs 1D defect discrimination
- [ ] **NIR/UV mode for g-C₃N₄** — dedicated peak windows at 691 / 988 cm⁻¹ for triazine ring breathing modes

## Citation

If you use this software in your research, please cite:

```bibtex
@software{raman_analyzer_2026,
  author    = {H.J},
  title     = {Raman Spectrum Analyzer: Quantitative Analysis of Graphene Raman Spectra},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21109057},
  url       = {https://doi.org/10.5281/zenodo.21109057}
}
```

Or in plain text:  
H.J (2026). *Raman Spectrum Analyzer: Quantitative Analysis of Graphene Raman Spectra*. Zenodo. https://doi.org/10.5281/zenodo.21109057

## Scientific References

### Core Raman Methods
- Tuinstra, F. & Koenig, J.L. (1970) *J. Chem. Phys.* **53**, 1126 — original D/G band assignment in graphite
- Ferrari, A.C. & Robertson, J. (2001) *Phys. Rev. B* **64**, 075414 — disorder stage classification (Stage 1/2)
- Ferrari, A.C. et al. (2006) *Phys. Rev. Lett.* **97**, 187401 — 2D band and layer number in graphene
- Lucchese, M.M. et al. (2010) *Carbon* **48**, 1592 — L_D formula from I_D/I_G (ion-bombarded graphene)
- Cançado, L.G. et al. (2011) *Nano Lett.* **11**, 3190–3196 — laser-energy-dependent L_D; eV-based peak dispersion
- Ferrari, A.C. & Basko, D.M. (2013) *Nature Nanotechnology* **8**, 235–246 — comprehensive peak conventions & line shapes

### Defect Characterisation
- Eckmann, A. et al. (2012) *Nano Lett.* **12**, 3925–3930 — defect type discrimination via I_D/I_D′
- Pimenta, M.A. et al. (2007) *Phys. Chem. Chem. Phys.* **9**, 1276–1290 — studying disorder in graphite-based systems
- Claramunt, S. et al. (2015) *Sci. Rep.* **5**, 19491 — G+D′ overlap in GO/rGO

### Doped Graphene
- **Kim, Y.A. et al. (2012)** *ACS Nano* **6**(7), 6293–6300 — Raman spectroscopy of boron-doped single-layer graphene; G-band position invariance, D-band 7× enhancement, L_D ~4.76 nm, B–C elastic scattering mechanism
- Pisana, S. et al. (2007) *Nature Materials* **6**, 198–201 — doping-induced G-band shift (Kohn anomaly)

### Graphitic C₃N₄
- **Zinin, P.V. et al. (2009)** *Chem. Phys. Lett.* **472**, 69–73 — UV (244 nm) and NIR (785 nm) Raman of g-C₃N₄; triazine ring breathing modes at 691 and 988 cm⁻¹; fluorescence limitation of visible excitation

### Reduced Graphene Oxide
- **Mohan, V.B. et al. (2017)** *Graphene Technology* **2**(4–7), 6–21 — quantification and analysis of Raman spectra of graphene materials; I_D/I_G vs. conductivity; FWHM(G) as disorder metric; rGO structural evolution
- Hummers, W.S. & Offeman, R.E. (1958) *J. Am. Chem. Soc.* **80**, 1339 — GO synthesis method

### Line Shape
- Thompson, P., Cox, D.E. & Hastings, J.B. (1987) *J. Appl. Crystallogr.* **20**, 79–83 — Pseudo-Voigt profile definition

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
    ├── baseline.py          ← ALS baseline correction (no clipping of residuals)
    ├── peak_fitter.py       ← global D/G/D′ fit, adaptive G, G+D′ deconvolution, SNR gate
    ├── analyzer.py          ← ratio calculations & defect analysis (L_D E_L⁴-corrected)
    ├── plotter.py           ← matplotlib visualisation (300 dpi)
    └── exporter.py          ← CSV + text report export
```

## Changelog

### v2.3 — 2026-07-02
- **Global D / G / D′ fit** (`_fit_D_G_Dp_global`): three bands fitted simultaneously, eliminating spurious D′ detections on the G tail
- **SNR gate**: peak detection now requires R² > 0.75 **and** SNR > 3; SNR computed from MAD-based residual noise
- **Pseudo-Voigt corrected**: FWHM and integrated area follow the Thompson–Cox–Hastings definition
- **L_D implementation**: uses full Cançado 2011 formula with E_L⁴; Stage 2 samples return `NaN` with warning
- **FWHM(2D) guard**: `twoD_fwhm_warning` flag set when FWHM(2D) > 35 cm⁻¹; layer count marked unreliable
- **Removed `graphitization_pct`**: (1−I_D/I_G)×100 has no literature basis and is non-monotonic across Stage 1/2
- **Baseline clipping removed**: `np.clip` after ALS subtraction deleted; negative residuals are preserved
- **eV-based peak dispersion**: `get_peak_windows` uses eV-based shifts per Cançado 2011

### v2.2 — 2026-07-01
- Adaptive G-band window + dual-Lorentzian G+D′ deconvolution
- `PeakResult` extended with `is_deconvolved`, `deconv_partner`, `is_split_2D`
- Removed `lmfit` as a hard dependency

## Author

H.J — Researcher, Chemistry
