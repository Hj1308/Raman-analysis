
# Raman Spectrum Analyzer 🔬

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21109057.svg)](https://doi.org/10.5281/zenodo.21109057)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A Python application for **quantitative analysis of Raman spectra** of graphene and graphene-like (sp² carbon) materials, with robust support for doped and disordered systems: N-doped, B-doped, SiC-grown graphene, GO, rGO, and dedicated CN-mode detection for g-C₃N₄ under UV/NIR-friendly Raman conditions.

---

## Physical Background

Raman spectroscopy is the primary non-destructive probe of structural quality, defect density, doping level, layer number, and strain in sp² carbon materials. The key spectral features are:

| Band | Typical Position (532 nm) | Origin | Dispersive? |
|------|--------------------------|--------|-------------|
| D*   | ~1100–1200 cm⁻¹ | Disordered sp³/sp² boundary; OH/epoxy functional groups | Yes |
| D    | ~1345 cm⁻¹ | Intervalley double resonance at K-point; **requires a defect** | Yes — 53 cm⁻¹/eV |
| G    | ~1580 cm⁻¹ | In-plane C–C stretching (E₂g); present in all sp² carbon | No (but doping-sensitive) |
| D′   | ~1620 cm⁻¹ | Intravalley double resonance; defect-activated | Yes — ~37 cm⁻¹/eV |
| 2D   | ~2690 cm⁻¹ | Two-phonon second order of D; **no defect needed** | Yes — 106 cm⁻¹/eV |
| D+G  | ~2940 cm⁻¹ | Combination of D and G phonons; defect-assisted | Yes |

 > **Dispersion:** The 2D band dispersion is approximately twice the D band dispersion, but this relation is not exact. Reported 2D dispersions range from ~85–106 cm⁻¹/eV (e.g., 99 cm⁻¹/eV for turbostratic graphite and 87 cm⁻¹/eV for HOPG [Barros 2005]; ~100 cm⁻¹/eV for graphene [Ferrari & Basko 2013]), while D band dispersions span ~37–53 cm⁻¹/eV (e.g., 37.4 cm⁻¹/eV for B-doped graphene [Kim 2012]; ~53 cm⁻¹/eV [Dresselhaus 2005]; ~50 cm⁻¹/eV [Wu 2018]). Both values vary with material, doping, and stacking order. This tool uses the approximate 2:1 ratio only to predict laser-dependent peak fitting windows — sufficient for that purpose, not for precise dispersion measurements. Peak fitting windows shift automatically with `--laser` input.

> **G-band and doping:** The G band is non-dispersive in position but shifts with carrier density. Both electron and hole doping stiffen the C–C bond via removal of the Kohn anomaly, upshifting the G band. The direction and magnitude of the shift distinguish n-type from p-type doping [Pisana 2007].

---

## Material-Specific Raman Signatures

### Boron-Doped Graphene 

Substitutional boron (0.22 at% in single-layer graphene) introduces chemical disorder: the B–C bond is ~0.5 Å longer than C–C, breaking translational symmetry and activating the D band. Key signatures [Kim 2012]:

- **I_D/I_G ≈ 7**: the extremely high D-band intensity originates from electron–boron *elastic scattering before phonon emission* — distinct from point defects where the phonon is emitted first.
- **G-band position: constant** as I_D/I_G increases. This is the primary diagnostic separating B-doping from argon-plasma point defects (where G blueshifts monotonically). The p-type stiffening effect (upshift) is exactly counterbalanced by tensile-strain softening from the longer B–C bond.
- **D′ at ~1620 cm⁻¹**: in heavily B-doped samples the D′ intensity equals the G intensity. D′ dispersion in B-doped graphene is ~37 cm⁻¹/eV (vs. ~77 cm⁻¹/eV for the 2D band) — consistent with hole-doped graphene theoretical predictions.
- **2D downshift**: ~4 cm⁻¹ per unit I_D/I_G up to I_D/I_G ≈ 3; beyond that the double-resonance q-vector is modified by strain.
- **Defect spacing**: L_D ≈ 4.76 nm corresponds to 0.22 at% B (Cançado 2011 formula applied to B atoms as vacancy-like scatterers).
- **I_D/I_D′ ≈ 7**: consistent with vacancy-type scatterers [Eckmann 2012], confirming substitutional incorporation rather than edge or sp³ defects.

### Nitrogen-Doped Graphene 

Nitrogen substitution creates n-type doping. The N–C bond length is closer to C–C than B–C, so structural disorder is lower. Key signatures:

- **G-band upshift** due to Fermi-level elevation (electron doping → Kohn anomaly removal).
- **2D downshift and broadening**: n-type doping shifts 2D to lower wavenumbers and broadens the peak — opposite trend to hole doping [Pisana 2007].
- **Lower I_D/I_G** than B-doping at comparable dopant concentrations.
- **I_D/I_D′ ≈ 7–13** depending on whether N sits substitutionally (vacancy-like) or as pyridinic/pyrrolic N (sp³-like → higher ratio).

### Graphitic C₃N₄ Phase (g-C₃N₄)

g-C₃N₄ has a fundamentally different Raman signature from graphene [Zinin 2009]:

- **Visible excitation (514/532 nm):** strong fluorescence background obscures the spectrum, yielding only two broad overlapping bands near 1357 cm⁻¹ and 1560 cm⁻¹. These are **fluorescence artefacts**, not the graphene D and G bands — they carry no structural information.
- **UV (244 nm) or NIR (785 nm) excitation** reveals the true triazine fingerprint:
  - **~691 cm⁻¹**: s-triazine ring breathing mode (ring–C–N in-plane; matches melamine at 677 cm⁻¹).
  - **~988 cm⁻¹**: second triazine breathing mode (matches melamine at 983 cm⁻¹).
  - **~1596 cm⁻¹**: C–N ring stretching.
  - **~1728 cm⁻¹**: high-frequency unassigned mode.
  - The D band is **absent** in UV Raman of g-C₃N₄ — analogous to diamond-like carbon.

> ⚠️ **Analyser note:** When a spectrum shows the 691/988 cm⁻¹ doublet, the analyser can now flag g-C₃N₄-specific CN modes through dedicated `CN_triazine` and `CN_bending` peak windows. If these modes are detected under visible excitation, the report warns that fluorescence is likely and recommends UV or NIR Raman for quantitative CN-mode interpretation.

### Reduced Graphene Oxide (rGO) and GO

Structural evolution from graphite → GO → rGO [Mohan 2017]:

- **Graphite**: sharp G at ~1582 cm⁻¹, very weak D (edge only). D-band intensity is edge-dependent: strong at armchair, weak at zigzag.
- **GO**: broad, overlapping D and G bands; high oxygen content disrupts π-conjugation; low electrical conductivity.
- **rGO**: I_D/I_G *increases* after reduction as new graphitic nanocrystallites nucleate (π-conjugation is partially restored). The highest conductivity (103.3 S cm⁻¹) was achieved with HI reduction, correlating with the lowest defect density and narrowest D/G FWHM.
- **D* band (~1100–1200 cm⁻¹)**: present in GO/rGO; assigned to the disordered sp³/sp² boundary and OH/epoxy functional groups. I_D*/I_G correlates directly with the C/O ratio and tracks the degree of oxidation [Lee 2021].
- **FWHM(G)**: consistent FWHM(G) across samples signals uniform sp² cluster size; increasing FWHM(D) signals more disordered graphitic rings.
- **I_2D/I_G**: CVD graphene shows I_2D/I_G ≈ 0.179 with a single sharp 2D at ~2690 cm⁻¹. rGO shows a distorted or absent 2D region — not a reliable layer indicator.
- **Functionalised rGO**: G downshifts by ~5 cm⁻¹ due to π–π charge transfer from adsorbed molecules (e.g., pyrene) — a Raman indicator of non-covalent functionalisation.

### Functionalized Graphene — Substrate Effects

Covalent aryl functionalization of graphene is substrate-dependent [Dierke 2022]:

- Graphene on **SiO₂**: I_D/I_G after functionalization ≈ 0.9; L_D ≈ 12.4 nm.
- Graphene on **hBN**: I_D/I_G after functionalization ≈ 0.3; L_D ≈ 20.9 nm — hBN's flat, dangling-bond-free surface suppresses radical attack.
- The substrate effect allows spatial patterning of functionalization density by patterning the underlying substrate.

---
## How This Tool Compares

| Capability | This tool | RamanSPy | rampy | Fityk | OriginPro |
|---|---|---|---|---|---|
| Global D/G/D′ fit with automatic deconvolution | ✅ | ❌ | ❌ | manual | manual |
| Laser-energy-dependent fit windows | ✅ | ❌ | ❌ | ❌ | ❌ |
| L_D and n_D via Cançado with E_L⁴ correction + uncertainty | ✅ | ❌ | ❌ | ❌ | ❌ |
| Range-based Eckmann defect-type classification with citations | ✅ | ❌ | ❌ | ❌ | ❌ |
| Stage 1/2 disorder regime detection | ✅ | ❌ | ❌ | ❌ | ❌ |
| Adaptive lineshape (pseudo-Voigt for disordered materials) | ✅ | partial | ❌ | manual | manual |
| Literature knowledge base (113 cited reference values) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Fit-quality validation flags | ✅ | ❌ | ❌ | ❌ | ❌ |
| B/N doping fingerprint | ✅ | ❌ | ❌ | ❌ | ❌ |
| g-C₃N₄ dedicated support | ✅ | ❌ | ❌ | ❌ | ❌ |
| General preprocessing / ML / hyperspectral | ❌ (delegates to RamanSPy) | ✅ | partial | ❌ | ✅ |
| Raman mapping/imaging | ❌ | ✅ | ❌ | ❌ | ✅ |
| Open source | ✅ | ✅ | ✅ | ✅ | ❌ |

This tool sits on top of the open-source spectroscopy ecosystem: it uses `pybaselines` for baseline subtraction and optionally leverages RamanSPy's despiking and data loaders when available.
Its unique value is a literature-anchored interpretation layer purpose-built for sp² carbons (graphene, GO, rGO, g-C₃N₄), translating raw peak fits into physically meaningful parameters with uncertainty.
Users who need general preprocessing, machine learning, or hyperspectral imaging should use RamanSPy as the primary workflow and then feed pre-processed spectra into this tool for domain-specific analysis (see `docs/INTEROPERABILITY.md`).

## Features

- **Baseline correction** — Asymmetric Least Squares (ALS); negative residuals preserved (no clipping)
- **Peak fitting** — Lorentzian (D, G, D′) and Pseudo-Voigt (D+G) using pure `scipy`
- **Global D / G / D′ fit** — three bands fitted simultaneously to prevent D′ from capturing the G tail
- **Adaptive G-band fitting** — locates true G peak position before fitting; handles doped samples where G sits near or above 1600 cm⁻¹
- **G+D′ deconvolution** — dual-Lorentzian separation of overlapping G and D′ in heavily doped or disordered graphene
- **SNR-gated peak detection** — peak accepted only when R² > 0.75 **and** SNR > 3 (SNR = amplitude / MAD-based noise estimate)
- **Pseudo-Voigt area correction** — FWHM and integrated area follow the Thompson–Cox–Hastings definition
- **Quantitative ratios** — I_D/I_G, I_2D/I_G, I_D′/I_G, I_D/I_D′
- **Defect density** — L_D (Cançado 2011, full E_L⁴ correction)
- **g-C₃N₄ CN-mode detection** — dedicated `CN_triazine` (~691 cm⁻¹) and `CN_bending` (~988 cm⁻¹) windows for UV/NIR-friendly Raman analysis
- **g-C₃N₄ report logic** — `gcn4_detected` flag plus `gcn4_mode_note` warning for visible excitation
- **Disorder stage** — Stage 1 / Stage 2 classification with FWHM(G) cross-check
- **Defect type classification** — sp³ / vacancy / edge via I_D/I_D′ [Eckmann 2012]
- **Layer number estimation** — from I_2D/I_G with FWHM(2D) reliability guard
- **eV-based laser dispersion** — all peak windows shift with laser energy per Cançado 2011
- **Batch processing** — analyse entire folders at once
- **Publication-quality plots** — 4 plots per spectrum (300 dpi)
- **CSV + text reports** — structured output ready for further analysis

---

## Peak Windows (532 nm laser)

| Peak | Search Range | Fit Window | Line Shape | Physical Origin |
|------|-------------|------------|------------|-----------------|
| CN_triazine | 670–715 cm⁻¹ | fixed | Lorentzian | g-C₃N₄ triazine ring mode (~691 cm⁻¹) |
| CN_bending  | 960–1010 cm⁻¹ | fixed | Lorentzian | g-C₃N₄ C–N bending / ring-related mode (~988 cm⁻¹) |
| D*   | 1080–1230 cm⁻¹ | fixed | Lorentzian | sp³/sp² boundary; OH/epoxy groups in GO/rGO [Lee 2021] |
| D    | 1270–1450 cm⁻¹ | fixed | Lorentzian | K-point phonon, intervalley double resonance [Ferrari 2001] |
| G    | 1540–1680 cm⁻¹ (search) | adaptive ±50 cm⁻¹ | Lorentzian | E₂g in-plane C–C stretch [Tuinstra 1970] |
| D′   | global fit with D+G | — | Lorentzian | Intravalley defect-induced [Eckmann 2012] |
| 2D   | 2580–2780 cm⁻¹ | fixed | Lorentzian (or dual) | Second-order overtone; layer-sensitive [Ferrari 2006] |
| D+G  | 2850–2960 cm⁻¹ | fixed | Pseudo-Voigt | Combination band |


> **G-band strategy:** `find_peaks` in 1540–1680 cm⁻¹ → adaptive ±50 cm⁻¹ window → single Lorentzian. If R² < 0.60, dual-Lorentzian G+D′ deconvolution is triggered automatically. The D′ component is stored in `PeakResult.deconv_partner` and promoted to the `D_prime` result slot if it outperforms the standalone fit.

> Peak positions shift with laser wavelength for dispersive graphene bands. Dispersive peaks: D ~53 cm⁻¹/eV, D′ ~37 cm⁻¹/eV, 2D ~106 cm⁻¹/eV (Cançado 2011, eV-based). The g-C₃N₄ `CN_triazine` and `CN_bending` windows are treated as non-dispersive auxiliary modes.

---
---

## Quantitative Metrics

### Defect Density — L_D

Inter-defect distance L_D (nm) from the laser-energy-corrected Cançado 2011 formula:

```
L_D² (nm²) = (4.3 × 10³ / E_L⁴) × (A_G / A_D)
```

where E_L is the laser energy in eV. Valid in **Stage 1 only** (L_D ≳ 10 nm). Applied to B-doped graphene, where substitutional B atoms behave as vacancy-like scatterers, this yields L_D ≈ 4.76 nm for 0.22 at% B [Kim 2012]. In Stage 2, L_D is set to `NaN` with a warning.

### Disorder Stage Classification

| Stage | I_D/I_G trend | FWHM(G) | L_D | Regime |
|-------|--------------|---------|-----|--------|
| Stage 1 | Increases with disorder | Broadens moderately | ≳ 10 nm | Crystalline graphene with point defects (Tuinstra–Koenig) |
| Stage 2 | Decreases with disorder | Strongly broadened | < 10 nm | Amorphous / nanocrystalline carbon |

> **Important:** High I_D/I_G alone does not mean Stage 2. In B-doped graphene, I_D/I_G ≈ 7 while FWHM(G) remains narrow and the G position is constant — chemical disorder, not structural amorphisation [Kim 2012]. The analyser cross-checks FWHM(G) before assigning Stage 2.

### Defect Type — I_D / I_D′

The ratio I_D/I_D′ discriminates between defect types [Eckmann 2012]:

| I_D/I_D′ | Defect type | Example |
|-----------|-------------|---------|
| ~13 | sp³ (homoatomic) | H, O functional groups (GO, rGO) |
| ~7  | Vacancy-like | Ion bombardment, substitutional B or N |
| ~3.5 | Grain boundary / edge | Polycrystalline graphene, nanoflakes |

### Layer Number — I_2D/I_G

| I_2D/I_G | 2D FWHM | Layer count |
|----------|---------|-------------|
| > 2 | ~25 cm⁻¹ (single Lorentzian) | Monolayer [Ferrari 2006] |
| ≈ 1 | Broader, 4-component | Bilayer |
| < 1 | Broad, graphite-like | Few-layer (≥ 3) |
| Distorted / absent | — | rGO / functionalised (unreliable) |

---

## Installation

```bash
git clone https://github.com/Hj1308/Raman-analysis.git
cd Raman-analysis
pip install -r requirements.txt
```

**Dependencies:** `numpy`, `scipy`, `matplotlib`, `pandas`, `openpyxl`

`lmfit` is optional — install only if you need Voigt / AsymLorentzian fits or uncertainty reporting via the `band_config` API.

---

## Usage

```bash
# Single file (532 nm laser)
python main.py --file spectrum.txt --laser 532

# Single file with custom output directory
python main.py --file spectrum.txt --laser 633 --output ./my_results/

# Batch mode — all .txt/.csv/.xlsx files in a folder
python main.py --folder ./spectra/ --laser 785 --batch
```

### Advanced: `band_config`

```python
from src.peak_fitter import PeakFitter

config = {
    "G":   {"method": "adaptive"},
    "D":   {"lineshape": "Lorentzian"},
    "2D":  {"method": "deconvolve"},      # force dual-Lorentzian (bilayer)
    "D+G": {"lineshape": "Voigt"},        # requires lmfit
}
results = fitter.fit_all_peaks(band_config=config)
```

---

## Input File Format

Plain text or CSV — two columns (wavenumber, intensity):
```
# Optional comment lines
1000.0   120.5
1001.0   122.3
```

Excel (`.xlsx`): wavenumber in column B, intensity in column C, data from row 4.

---

## Output

For each spectrum:

| File | Content |
|------|---------|
| `*_baseline.png` | Raw spectrum + ALS baseline overlay |
| `*_peaks.png` | Fitted peaks on corrected spectrum |
| `*_individual.png` | Individual peak fits with residuals |
| `*_ratios.png` | Intensity ratio bar chart |
| `*_report.txt` | Full text report (all metrics) |
| `raman_results.csv` | All parameters in one CSV (batch-compatible) |

---

## PeakResult Fields

| Field | Type | Description |
|-------|------|-------------|
| `center` | float | Peak position (cm⁻¹) |
| `amplitude` | float | Peak height (a.u.) |
| `fwhm` | float | Full width at half maximum (cm⁻¹) |
| `area` | float | Integrated area (Thompson–Cox–Hastings for Pseudo-Voigt) |
| `r_squared` | float | Goodness of fit |
| `snr` | float | Signal-to-noise ratio (amplitude / MAD noise) |
| `found` | bool | True if R² > 0.75 and SNR > 3 |
| `is_deconvolved` | bool | True when G was separated from D′ by dual-Lorentzian |
| `deconv_partner` | PeakResult \| None | D′ component from G deconvolution |
| `is_split_2D` | bool | True when 2D fitted with dual-Lorentzian (bilayer) |
| `twoD_fwhm_warning` | bool | True when FWHM(2D) > 35 cm⁻¹ (layer count unreliable) |
| `center_stderr` | float \| None | Standard error on center (lmfit backend only) |
| `fwhm_stderr` | float \| None | Standard error on FWHM (lmfit backend only) |

---

## Known Limitations

### 2D Band — Bilayer / Trilayer
Single Lorentzian for 2D is appropriate for monolayer only. In bilayer graphene, 2D splits into 4 sub-components; single-peak fitting underestimates FWHM and may misassign layer count. Multi-component 2D fitting is on the roadmap.
> Ferrari et al. (2006) *Phys. Rev. Lett.* **97**, 187401

### L_D in Stage 2
Cançado 2011 formula is valid in Stage 1 only. When Stage 2 is assigned, L_D is set to `NaN` with a warning in `L_D_note`. Use FWHM(G) as the primary disorder metric for Stage 2 samples.

### g-C₃N₄ and Visible Excitation
Visible-excitation spectra of g-C₃N₄ are dominated by fluorescence. The apparent 1357/1560 cm⁻¹ bands carry no reliable structural information [Zinin 2009]. The analyser now includes dedicated UV/NIR CN-mode support through the 691/988 cm⁻¹ windows, but visible-range spectra remain fundamentally fluorescence-limited and should be interpreted with caution.

### Layer Number in rGO / Functionalised Graphene
I_2D/I_G is not a reliable layer indicator for rGO or functionalised graphene; the 2D band depends on preparation method, not layer count [Mohan 2017].

### D* Band
The D* peak window (1080–1230 cm⁻¹) is defined, but quantitative I_D*/I_G analysis and C/O ratio correlation are not yet implemented. See roadmap.

### G+D′ Overlap Extremes
Dual-Lorentzian deconvolution may not fully separate G and D′ in extreme cases. More complex multi-band models are available via the `band_config` API.

---

## Roadmap

### v2.4 — In Development

| Feature | Scientific Basis | Priority |
|---------|-----------------|----------|
| **D* band quantification** | I_D*/I_G ↔ C/O ratio in GO/rGO; oxidation degree tracking [Lee 2021] | 🔴 High |
| **B-doping fingerprint flag** | Auto-detect: constant G position + I_D/I_D′ ≈ 7 + I_D/I_G ≫ 1 [Kim 2012] | ✅ Implemented |
| **Fitting uncertainty (scipy)** | Error bars from covariance matrix for all users (no lmfit required) | 🔴 High |
| **Multi-Lorentzian 2D fitting** | 4-component decomposition for bilayer/trilayer [Ferrari 2006] | 🟡 Medium |
| **Carrier-density estimator refinement** | G-band shift → carrier density → n/p doping type; scaling bug corrected in current main branch | ✅ Implemented (further refinement possible) |
| **Stage boundary refinement** | FWHM(G) + A_D/A_G combined metric; 0D vs. 1D defect discrimination [Wu 2018] | 🟡 Medium |
| **Dispersion slope validator** | Multi-wavelength: D slope ≈ 53 cm⁻¹/eV, 2D ≈ 106 cm⁻¹/eV; deviation flags contamination [Cançado 2011] | ✅ Implemented |
| **arPLS baseline** | Asymmetrically reweighted PLS for fluorescence-heavy spectra (GO, g-C₃N₄) | 🟡 Medium |
| **NIR/UV mode for g-C₃N₄** | Dedicated windows at 691/988 cm⁻¹; analyzer warning/note for visible excitation [Zinin 2009] | ✅ Implemented |
| **Batch statistics** | Mean ± std across all samples; ratio heatmap for batch runs | 🟢 Planned |
| **Substrate effect report** | Flag I_D/I_G contrast on hBN vs. SiO₂; L_D comparison [Dierke 2022] | 🟢 Planned |
| **Electrochemical doping tracker** | G-band shift vs. gate voltage; cyclic voltammetry-compatible input | 🟢 Planned |

---

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
    ├── baseline.py          ← ALS baseline correction (no clipping)
    ├── peak_fitter.py       ← global D/G/D′ fit, adaptive G, G+D′ deconvolution, SNR gate
    ├── analyzer.py          ← ratio calculations & defect analysis (L_D E_L⁴-corrected)
    ├── plotter.py           ← matplotlib visualisation (300 dpi)
    └── exporter.py          ← CSV + text report export
```

---

## Citation

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

Plain text: H.J (2026). *Raman Spectrum Analyzer*. Zenodo. https://doi.org/10.5281/zenodo.21109057

---

## Scientific References

### Core Raman Methods
- Tuinstra, F. & Koenig, J.L. (1970) *J. Chem. Phys.* **53**, 1126
- Ferrari, A.C. & Robertson, J. (2001) *Phys. Rev. B* **64**, 075414 — disorder stage classification
- Ferrari, A.C. et al. (2006) *Phys. Rev. Lett.* **97**, 187401 — 2D band and layer number
- - Lucchese, M.M. et al. (2010) *Carbon* **48**, 1592 — defect-density / inter-defect-distance framework for disordered graphene
- Cançado, L.G. et al. (2011) *Nano Lett.* **11**, 3190–3196 — E_L⁴-corrected L_D; eV-based dispersion
- Ferrari, A.C. & Basko, D.M. (2013) *Nature Nanotechnology* **8**, 235–246 — peak conventions & line shapes

### Defect Characterisation
- Eckmann, A. et al. (2012) *Nano Lett.* **12**, 3925–3930 — I_D/I_D′ defect type discrimination
- Pimenta, M.A. et al. (2007) *Phys. Chem. Chem. Phys.* **9**, 1276–1290 — disorder in graphite-based systems
- Claramunt, S. et al. (2015) *Sci. Rep.* **5**, 19491 — G+D′ overlap in GO/rGO

### Doped Graphene
- Kim, Y.A. et al. (2012) *ACS Nano* **6**(7), 6293–6300 — B-doped graphene; G invariance, D×7, L_D = 4.76 nm
- Pisana, S. et al. (2007) *Nature Materials* **6**, 198–201 — G-band shift and Kohn anomaly

### Graphitic C₃N₄
- Zinin, P.V. et al. (2009) *Chem. Phys. Lett.* **472**, 69–73 — triazine modes at 691/988 cm⁻¹

### Reduced Graphene Oxide
- Mohan, V.B. et al. (2017) *Graphene Technology* **2**(4–7), 6–21 — rGO structural evolution; conductivity vs. I_D/I_G
- Lee, J. et al. (2021) *Carbon* **183**, 814–822 — D* band in GO/rGO; I_D*/I_G vs. C/O ratio
- Dierke, T. et al. (2022) *ACS Appl. Nano Mater.* **5**, 4966–4971 — substrate-dependent functionalization
- Hummers, W.S. & Offeman, R.E. (1958) *J. Am. Chem. Soc.* **80**, 1339 — GO synthesis

### Line Shape
- Thompson, P., Cox, D.E. & Hastings, J.B. (1987) *J. Appl. Crystallogr.* **20**, 79–83 — Pseudo-Voigt profile

---

## Changelog

### v2.4 — 2026-07-03
- Feature #9: dedicated g-C₃N₄ CN-mode support via `CN_triazine` (670–715 cm⁻¹) and `CN_bending` (960–1010 cm⁻¹)
- Added `gcn4_detected` and `gcn4_mode_note` to analyzer output
- Added g-C₃N₄ CN-mode section to formatted report
- Visible-excitation warning for g-C₃N₄ CN-mode detection; UV/NIR-friendly note for non-visible excitation
- Fixed `carrier_density_cm2` scaling in `_estimate_doping` by removing an erroneous extra factor of `10^12`
- Full test suite passes after Feature #9 and carrier-density fix

### v2.3 — 2026-07-02
- Global D / G / D′ simultaneous fit (`_fit_D_G_Dp_global`)
- SNR gate: R² > 0.75 **and** SNR > 3; MAD-based noise estimate
- Pseudo-Voigt corrected: Thompson–Cox–Hastings area definition
- L_D: full E_L⁴ Cançado 2011 formula; Stage 2 → `NaN` + warning
- FWHM(2D) guard: `twoD_fwhm_warning` when FWHM(2D) > 35 cm⁻¹
- Removed `graphitization_pct` (no literature basis)
- Baseline clipping removed (negative residuals preserved)
- eV-based peak dispersion in `get_peak_windows`

### v2.2 — 2026-07-01
- Adaptive G-band window + dual-Lorentzian G+D′ deconvolution
- `PeakResult` extended: `is_deconvolved`, `deconv_partner`, `is_split_2D`
- Removed `lmfit` hard dependency

---

## Author

H.J — Researcher, Chemistry
