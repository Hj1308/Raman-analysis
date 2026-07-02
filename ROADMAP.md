# Raman-Analysis — Scientific Roadmap

> Derived from systematic review of 4 core reference papers covering:
> graphene disorder metrics, graphitized-SiC Raman, 6H-SiC polytype identification, and amorphous-SiC characterisation.

---

## Priority Legend

| Symbol | Meaning |
|--------|---------|
| 🔴 **P1** | Critical / currently wrong or missing |
| 🟠 **P2** | High value, unique scientific contribution |
| 🟡 **P3** | Nice-to-have, publication-quality polish |
| 🟢 **P4** | Future / research-stage |

---

## 1 — Fixes (Must Do Before Publication)

### 1.1 D/G ratio — use integrated area, not peak height 🔴 P1

All three disorder metrics (L_D, n_D, Γ_D) that the literature defines are based on
**integrated area ratios** `A_D/A_G`, not amplitude ratios `I_D/I_G`.
Using height ratios introduces a systematic error that scales with FWHM.

**Action:** Replace `I_D/I_G = height_D / height_G` with
`A_D/A_G = area_D / area_G` in `app.py` and `src/peak_fitting.py`.

> Reference: Lucchese et al. 2010 (Stage-2 disorder); Ferrari & Robertson 2004

---

### 1.2 Two-stage disorder model — Stage 1 vs Stage 2 decision 🔴 P1

The current code outputs a single `L_D` value without checking which stage
of disorder the sample is in. The Tuinstra–Koenig relation only applies in
**Stage 2** (`L_D > ~3 nm`). In Stage 1 the relation *inverts*.

**Action:** Add a `disorder_stage` classifier:
- If `I_D/I_G` is rising with increasing defect dose → **Stage 2** (use T-K)
- If `I_D/I_G` is falling → **Stage 1** (use Lucchese 2010 formula)
- Emit a warning badge in the UI when the sample may be in Stage 1.

> Reference: Ferrari & Robertson 2004, §2; Lucchese et al. 2010, Fig. 3

---

### 1.3 SiC substrate subtraction — mandatory for graphene-on-SiC 🔴 P1

For graphitized SiC (g-SiC, g-BSiC, g-NBSiC), the SiC substrate contributes
bands at **767, 789, 967 cm⁻¹** (1st-order) and **150, 263, 503 cm⁻¹**
(2nd-order) that bleed into the graphene analysis window.
The 1524–1709 cm⁻¹ region of 6H-SiC second-order bands directly overlaps G.

**Action:**
1. Load a virgin/reference SiC spectrum (same polytype, same laser).
2. Scale it to match the 789 cm⁻¹ peak in the sample spectrum.
3. Subtract before fitting D, G, D′, 2D.

> Reference: Madito et al. 2021, §3 (True Component Analysis); Lin et al. 2012, Table 1

---

### 1.4 LOPC coupling — nitrogen-doped SiC (g-NBSiC) 🔴 P1

In N-doped 6H-SiC the A₁(LO) mode couples with conduction-band plasmons forming
the **LO phonon–plasmon coupled (LOPC) mode**. Its frequency shifts upward from
967 cm⁻¹ depending on carrier concentration. The current code fits a fixed
Lorentzian at 967 cm⁻¹ — this fails for g-NBSiC and contaminates the G-band fit.

**Action:** Detect LOPC shift in the 900–1100 cm⁻¹ window.
If the peak is > 975 cm⁻¹, flag `lopc_active = True` and adjust substrate subtraction.

> Reference: Lin et al. 2012, §3 (carrier concentration → LO peak shift)

---

## 2 — New Features (Unique Scientific Value)

### 2.1 Quantitative defect density — three independent metrics 🟠 P2

Output all three concurrently and let the user compare them:

| Metric | Formula | Valid range | Reference |
|--------|---------|-------------|----------|
| Inter-defect distance | `L_D² = (1.8±0.5)×10⁻⁹ λ_L⁴ (A_D/A_G)⁻¹` | Stage 2 only | Lucchese 2010 |
| Defect density | `n_D = (1.8±0.5)×10²² / (λ_L⁴ L_D²)` | Stage 2 only | Lucchese 2010 |
| sp³ fraction proxy | `I_D/I_G` trend with `E_laser` | qualitative | Ferrari 2004 |

> Reference: Lucchese et al. 2010, Eq. 1–3

---

### 2.2 Laser wavelength input + λ⁴ correction 🟠 P2

L_D and n_D depend strongly on λ_L (4th power). Without knowing the excitation
wavelength the values are meaningless.

**Action:** Add a `laser_wavelength_nm` field to the UI (default: 532 nm).
Recalculate all disorder metrics using the actual λ_L.

> Reference: Lucchese et al. 2010, Eq. 2; raman-jjj-3.pdf §2.2

---

### 2.3 Graphitization degree index for SiC samples 🟠 P2

A unique metric for graphene-on-SiC not found in general-purpose tools:

```
G_index = A_G(graphene) / [A_G(graphene) + A_{789}(SiC)]
```

This quantifies how much of the carbon signal is graphitic vs. still bonded in SiC.
Useful for comparing different implant fluences / annealing temperatures.

> Reference: Madito et al. 2021, Fig. 3c; raman-jjj-3.pdf §3

---

### 2.4 Amorphous SiC detection mode 🟠 P2

For ion-implanted or radiation-damaged samples, the spectrum changes dramatically:
- 1st-order SiC peaks (767, 789, 967 cm⁻¹) **disappear**
- Broad Si-Si bands appear at **186, 266, 480 cm⁻¹**
- Broad Si-C bands appear at **670, 766, 849, 923 cm⁻¹**
- Broad C-C band near **1400 cm⁻¹** (amorphous sp²)

**Action:** Add an `amorphous_SiC` analysis mode that:
1. Checks if the 789 cm⁻¹ peak is absent or FWHM > 50 cm⁻¹
2. Fits the three broad Si-Si bands
3. Reports an **amorphization index** = ratio of broad-band area to total area

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
before doing any graphene analysis (critical for correct subtraction template).

> Reference: Lin et al. 2012, Fig. 4–5; raman-jjj-3.pdf Table 2

---

### 2.6 Impurity doping detection via LOPC frequency 🟠 P2

From Lin et al. 2012: the LO peak shift correlates with carrier concentration.
N-doping shifts A₁(LO) **upward** from 964 cm⁻¹; V-doping shifts it **downward**.

**Action:** Add a `carrier_concentration_proxy` output based on LO peak position,
with a lookup table calibrated from Lin et al. Table 1.

> Reference: Lin et al. 2012, Table 1 & §3

---

### 2.7 2D-band splitting detection (graphene layer count) 🟡 P3

For graphene grown on SiC the 2D band shape reports layer number:
- **Monolayer:** single Lorentzian ~2680 cm⁻¹
- **Bilayer:** four-component fit (2iB, 2iA, 2oA, 2oB)
- **Few-layer / turbostratic:** single broadened peak

> Reference: raman-jjj-3.pdf §2.3; Ferrari 2007

---

### 2.8 Strain vs. doping separation (2D–G correlation plot) 🟡 P3

Plotting ω_2D vs. ω_G separates biaxial strain (slope 2.2) from charge doping
(slope 0.7). Essential for graphene-on-SiC where both are present.

**Action:** If batch mode is active (multiple files), auto-generate this scatter plot.

> Reference: raman-jjj-3.pdf §4; Lee et al. 2012

---

## 3 — Scientific Reference Directory

A `docs/references/` directory is included in this repository with annotated
summaries of the four core reference papers. See [`docs/references/INDEX.md`](docs/references/INDEX.md).

---

## 4 — Implementation Order

```
Week 1:  Fix 1.1 (area ratio) + Fix 1.2 (disorder stage)
Week 2:  Fix 1.3 (SiC subtraction) + Fix 1.4 (LOPC)
Week 3:  Feature 2.1 (three metrics) + Feature 2.2 (λ input)
Week 4:  Feature 2.3 (G-index) + Feature 2.5 (polytype ID)
Week 5:  Feature 2.4 (amorphous mode) + Feature 2.6 (LOPC doping)
Week 6:  Features 2.7, 2.8 (layer count, strain/doping plot)
```
