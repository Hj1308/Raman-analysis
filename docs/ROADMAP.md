# Raman-Analysis — Scientific Roadmap

> Derived from systematic review of reference papers covering:
> graphene disorder metrics, graphitized-SiC Raman, 6H-SiC polytype identification,
> amorphous-SiC characterisation, substrate effects, doping fingerprints, and
> g-C₃N₄ CN-mode Raman behaviour under visible/UV/NIR excitation.

---

## Priority Legend

| Symbol | Meaning |
|--------|---------|
| 🔴 **P1** | Critical / currently wrong or missing |
| 🟠 **P2** | High value, unique scientific contribution |
| 🟡 **P3** | Nice-to-have, publication-quality polish |
| 🟢 **P4** | Future / research-stage |
| ✅ **Done** | Implemented in `main` |

---

## 1 — Fixes (Scientific correctness)

### 1.1 D/G ratio — use integrated area, not peak height ✅ Done

All disorder metrics that the literature defines quantitatively are based on
**integrated area ratios** `A_D/A_G`, not amplitude ratios `I_D/I_G`.
Using height ratios introduces a systematic error that scales with FWHM.

**Implemented:**
- `L_D` now uses `A_D/A_G`
- boron-doping logic now uses `A_D/A_G`
- tests were updated to explicitly distinguish area ratio from height ratio

> Reference: Lucchese et al. 2010; Ferrari & Robertson 2004

---

### 1.2 Two-stage disorder model — Stage 1 vs Stage 2 decision ✅ Done

A single `L_D` value is not physically valid across all disorder regimes.
The code now distinguishes disorder stage and suppresses `L_D` outside
its valid interpretation window.

**Implemented:**
- disorder-stage logic
- refined stage boundary using `FWHM(G)` and `A_D/A_G`
- Stage-2 suppression of `L_D` when the Cançado/Lucchese relation is not valid

> Reference: Ferrari & Robertson 2004; Lucchese et al. 2010; Wu et al. 2018

---

### 1.3 SiC substrate subtraction — mandatory for graphene-on-SiC 🔴 P1

For graphitized SiC (g-SiC, g-BSiC, g-NBSiC), the SiC substrate contributes
bands at **767, 789, 967 cm⁻¹** (1st-order) and **150, 263, 503 cm⁻¹**
(2nd-order) that bleed into the graphene analysis window.
The 1524–1709 cm⁻¹ region of 6H-SiC second-order bands directly overlaps G.

**Action:**
1. Load a virgin/reference SiC spectrum (same polytype, same laser)
2. Scale it to match the 789 cm⁻¹ peak in the sample spectrum
3. Subtract before fitting D, G, D′, 2D

**Status:** substrate-aware interpretation is implemented, but **true SiC spectral subtraction is still missing**.

> Reference: Madito et al. 2021, §3; Lin et al. 2012, Table 1

---

### 1.4 LOPC coupling — nitrogen-doped SiC (g-NBSiC) 🔴 P1

In N-doped 6H-SiC the A₁(LO) mode couples with conduction-band plasmons forming
the **LO phonon–plasmon coupled (LOPC) mode**. Its frequency shifts upward from
967 cm⁻¹ depending on carrier concentration. A fixed Lorentzian near 967 cm⁻¹
is therefore insufficient for g-NBSiC and can contaminate adjacent interpretation.

**Action:**
- detect LOPC shift in the 900–1100 cm⁻¹ window
- if the peak is > 975 cm⁻¹, flag `lopc_active = True`
- integrate this with future SiC-subtraction logic

> Reference: Lin et al. 2012, §3

---

### 1.5 Carrier-density scaling in `_estimate_doping` ✅ Done

The reported `carrier_density_cm2` previously included an erroneous extra factor
of `10^12`, producing values inflated by `10^12×` relative to the analytical
formula used in the tests.

**Implemented:**
- removed the extra `× 1e12` scaling
- `carrier_density_cm2` now matches the expression used by the test suite
- full test suite passes after correction

> Reference: internal regression test `tests/test_doping_estimator.py`

---

## 2 — New Features (Unique Scientific Value)

### 2.1 Quantitative defect density — three independent metrics 🟠 P2

Output all three concurrently and let the user compare them:

| Metric | Formula | Valid range | Reference |
|--------|---------|-------------|----------|
| Inter-defect distance | `L_D² = (1.8±0.5)×10⁻⁹ λ_L⁴ (A_D/A_G)⁻¹` | Stage 2 only | Lucchese 2010 |
| Defect density | `n_D = (1.8±0.5)×10²² / (λ_L⁴ L_D²)` | Stage 2 only | Lucchese 2010 |
| sp³ fraction proxy | `I_D/I_G` trend with `E_laser` | qualitative | Ferrari 2004 |

**Status:** `L_D` is implemented; `n_D` and consolidated multi-metric output are still pending.

> Reference: Lucchese et al. 2010, Eq. 1–3

---

### 2.2 Laser wavelength input + λ⁴ correction ✅ Done

`L_D` and related disorder metrics depend strongly on `λ_L` (fourth power).
Without the excitation wavelength, the values are physically meaningless.

**Implemented:**
- `laser_nm` input in analysis and fitting paths
- wavelength-aware `L_D` calculation
- wavelength-aware monolayer threshold logic
- multi-wavelength D-band dispersion validation

> Reference: Lucchese et al. 2010, Eq. 2; Ferrari & Basko 2013

---

### 2.3 Graphitization degree index for SiC samples 🟠 P2

A unique metric for graphene-on-SiC not found in general-purpose tools:

```text
G_index = A_G(graphene) / [A_G(graphene) + A_789(SiC)]
```

This quantifies how much of the carbon signal is graphitic vs. still bonded in SiC.
Useful for comparing different implant fluences / annealing temperatures.

> Reference: Madito et al. 2021, Fig. 3c

---

### 2.4 Amorphous SiC detection mode 🟠 P2

For ion-implanted or radiation-damaged samples, the spectrum changes dramatically:

- 1st-order SiC peaks (767, 789, 967 cm⁻¹) **disappear**
- broad Si-Si bands appear at **186, 266, 480 cm⁻¹**
- broad Si-C bands appear at **670, 766, 849, 923 cm⁻¹**
- broad C-C band near **1400 cm⁻¹** (amorphous sp²)

**Action:**
1. Check whether the 789 cm⁻¹ peak is absent or `FWHM > 50 cm⁻¹`
2. Fit the broad Si-Si bands
3. Report an **amorphization index** = broad-band area / total area

> Reference: Madito et al. 2021, Fig. 5; Sorieul et al. 2006

---

### 2.5 Polytype identification (6H vs 4H vs 3C vs 15R) 🟠 P2

Different SiC polytypes have characteristic folded-mode positions:

| Polytype | FTA (cm⁻¹) | FLA (cm⁻¹) | FTO (cm⁻¹) | FLO (cm⁻¹) |
|---------|-----------|-----------|-----------|----------|
| 6H-SiC | 150 | 504 | 765–789 | 964 |
| 4H-SiC | — | — | 776–796 | 964 |
| 3C-SiC | — | — | 796 | 972 |
| 15R-SiC | 145–167 | — | 766–788 | 963 |

**Action:** Add an auto-detect routine that identifies the polytype from FTO/FLO positions
before doing any graphene analysis.

> Reference: Lin et al. 2012, Fig. 4–5

---

### 2.6 Impurity doping detection via LOPC frequency 🟠 P2

From Lin et al. 2012: the LO peak shift correlates with carrier concentration.
N-doping shifts A₁(LO) **upward** from 964 cm⁻¹; V-doping shifts it **downward**.

**Action:** Add a `carrier_concentration_proxy` output based on LO peak position,
with a lookup table calibrated from Lin et al.

> Reference: Lin et al. 2012, Table 1 & §3

---

### 2.7 2D-band splitting detection (graphene layer count) 🟡 P3

For graphene grown on SiC the 2D band shape reports layer number:

- **Monolayer:** single Lorentzian ~2680 cm⁻¹
- **Bilayer:** four-component fit
- **Few-layer / turbostratic:** single broadened peak

**Status:** basic layer-count logic is already implemented from `I_2D/I_G` and `FWHM(2D)`, but explicit 2D-component splitting analysis remains future work.

> Reference: Ferrari 2007; graphene-on-SiC Raman literature

---

### 2.8 Strain vs. doping separation (2D–G correlation plot) 🟡 P3

Plotting `ω_2D` vs `ω_G` separates biaxial strain (slope 2.2) from charge doping
(slope 0.7). Essential for graphene-on-SiC where both are present.

**Action:** If batch mode is active (multiple files), auto-generate this scatter plot.

> Reference: Lee et al. 2012

---

### 2.9 g-C₃N₄ NIR/UV CN-mode support ✅ Done

Visible Raman of g‑C₃N₄ is often dominated by fluorescence, burying the
fingerprint region. Dedicated CN-mode support is now implemented.

**Implemented:**
- `CN_triazine` window: **670–715 cm⁻¹** (~691 cm⁻¹)
- `CN_bending` window: **960–1010 cm⁻¹** (~988 cm⁻¹)
- independent fitting support in `peak_fitter.py`
- analyzer fields:
  - `gcn4_detected`
  - `gcn4_mode_note`
- report section:
  - `g-C3N4 CN MODES (Feature #9)`
- visible-excitation warning
- UV/NIR-friendly note for non-visible excitation (e.g. 785 nm)

**Scientific value:** this extends the tool beyond graphene-only logic into a
photocatalyst-relevant material family while preserving explicit warning logic
for fluorescence-limited visible Raman.

> Reference: UV/NIR Raman literature for g‑C₃N₄ CN modes near 691 and 988 cm⁻¹

---

## 3 — Scientific Reference Directory

A `docs/references/` directory is included in this repository with annotated
summaries of core reference papers. See:

- `docs/references/INDEX.md`

---

## 4 — Implementation Status Summary

### Already implemented in `main`

- area-ratio-based `L_D` / boron-doping logic
- disorder stage and refined stage boundary
- laser-dependent analysis logic
- D* oxidation proxy
- boron-doping fingerprint
- doping estimator
- substrate-aware doping suppression
- dispersion-slope validator
- g‑C₃N₄ CN-mode support
- corrected `carrier_density_cm2` scaling

### Still high priority

- true SiC substrate subtraction
- LOPC-aware SiC handling
- polytype auto-detection
- amorphous-SiC mode
- quantitative SiC graphitization index

---

## 5 — Proposed Implementation Order (updated)

```text
Phase A:  SiC subtraction + polytype detection + LOPC-aware handling
Phase B:  quantitative SiC-specific metrics (G-index, amorphization index)
Phase C:  advanced layer/strain-doping analysis
Phase D:  expanded mixed-material logic (graphene + g-C3N4 / non-graphitic hybrids)
```