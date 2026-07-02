# Scientific Reference Index

Annotated summaries of the core literature underpinning this software.
All papers are cited by their physical role in the analysis pipeline.

---

## REF-01 — Lucchese et al. (2010)

**Full citation:**
Lucchese, M.M., Stavale, F., Ferreira, E.H.M., Vilani, C., Moutinho, M.V.O.,
Capaz, R.B., Achete, C.A., Jorio, A. (2010).
*Quantifying ion-induced defects and Raman relaxation length in graphene.*
**Carbon**, 48(5), 1592–1597.
https://doi.org/10.1016/j.carbon.2009.12.057

**Role in software:** Defines the two-stage disorder model and the quantitative
formulas for L_D (inter-defect distance) and n_D (defect density) as a function
of the integrated area ratio A_D/A_G and laser wavelength λ_L.

**Key equations:**
```
L_D² (nm²) = (1.8 ± 0.5) × 10⁻⁹ × λ_L⁴ × (A_D/A_G)⁻¹
n_D (cm⁻²) = (1.8 ± 0.5) × 10²² / (λ_L⁴ × L_D²)
```

**Stage boundary:** A_D/A_G peaks at L_D ≈ 3 nm (Stage 1/Stage 2 crossover).

**Critical note:** These formulas use **integrated area**, not peak height.
Using height ratios introduces errors proportional to FWHM mismatch.

---

## REF-02 — Madito et al. (2021)

**Full citation:**
Madito, M.J., Hlatshwayo, T.T., Mtshali, C.B. (2021).
*Chemical disorder of a-SiC layer induced in 6H-SiC by Cs and I ions
co-implantation: Raman spectroscopy analysis.*
**Applied Surface Science**, 538, 148099.
https://doi.org/10.1016/j.apsusc.2020.148099

**Role in software:** Provides the complete Raman fingerprint of amorphous SiC
(a-SiC) produced by ion implantation, and demonstrates the True Component
Analysis (TCA) demixing method for separating the a-SiC layer signal from
crystalline 6H-SiC bulk contributions.

**Key peak assignments (a-SiC layer):**

| Region | Bonds | Peaks (cm⁻¹) |
|--------|-------|-------------|
| Si–Si | amorphous Si | 186, 266, 480 |
| Si–C | amorphous SiC | 670, 766, 849, 923 |
| C–C | sp² + sp³ mix | ~1220 (sp³), ~1400 (sp²) |

**Key crystalline 6H-SiC peaks (Fig. 2):**
- Strong: 767 cm⁻¹ (E₂TO), 789 cm⁻¹ (E₂TO), 967 cm⁻¹ (A₁LO)
- Weak 2nd order: 150, 263 cm⁻¹ (E₂TA/TA), 503 cm⁻¹ (A₁LA)
- Weak: 1090–1220 cm⁻¹ (sp³ C), 1380 cm⁻¹ (sp² C), 1524–1709 cm⁻¹ (2nd-order optical)

**Amorphization index concept:** Ratio of integrated broad-band area (Si–Si region)
to total spectral area indicates degree of crystal-to-amorphous transition.

**Optical penetration depth:** 532 nm laser → d_p ≈ 46 nm in a-SiC; 9.3×10⁵ nm in
crystalline SiC. This makes 532 nm optimal for probing shallow ion-implanted layers.

---

## REF-03 — Lin et al. (2012)

**Full citation:**
Lin, [et al.] (2012).
*Effect of impurities on the Raman scattering of 6H-SiC crystals.*
**[Journal]**, 833–836.

**Role in software:** Provides the calibration table for impurity-dependent peak
shifts in 6H-SiC, enabling identification of the dopant type (N, Al, B, V) from
Raman data alone, and establishes the physical mechanism (LOPC coupling) behind
the LO phonon shift with carrier concentration.

**Key data — phonon mode frequencies by dopant (cm⁻¹):**

| Dopant | FTA | FLA | FTO₁ | FTO₂ | FLO |
|--------|-----|-----|------|------|-----|
| Undoped | 150.0 | 504.0 | 766.0 | 787.0 | 964.0 |
| N-doped | 150.7 | 505.9 | 766.6 | 789.4 | 965.1 |
| Al-doped | 151.0 | 506.5 | 766.1 | 789.4 | 965.0 |
| B-doped | 150.4 | 505.6 | 766.3 | 789.4 | 967.4 |
| V-doped | 148.5 | 501.5 | 765.1 | 785.7 | 963.8 |

**LOPC mechanism:** LO–phonon–plasmon coupling shifts A₁(LO) upward with
increasing carrier concentration (N, Al, B doping). Vanadium acts as a deep
compensator, reducing carrier concentration → **red shift**.

**Polytype discrimination peaks (Fig. 4–5):**
Folded-mode positions distinguish 6H (FTO at 765/787 cm⁻¹), 4H (FTO at 776/796 cm⁻¹),
and 15R (FTO at 766/788 cm⁻¹) polytypes.

---

## REF-04 — raman-jjj-3 (Graphene on SiC — internal report)

**Role in software:** Provides the experimental Raman data for the three
graphitized-SiC samples (g-SiC, g-BSiC, g-NBSiC) that are the primary test
cases for this software. Also discusses the 2D-band fitting strategy for
layer-number determination and the ω_2D vs. ω_G correlation method for
separating strain from doping effects.

**Key observations from the three samples:**

| Sample | G position (cm⁻¹) | 2D position (cm⁻¹) | D/G ratio | Notes |
|--------|------------------|-------------------|-----------|-------|
| g-SiC | ~1594 | ~2724 | low | Reference graphene on SiC |
| g-BSiC | ~1588 | ~2718 | moderate | B-doped substrate effect |
| g-NBSiC | ~1590 | ~2720 | higher | N-doped: LOPC complicates fit |

**G-band overlap issue:** g-NBSiC G peak at 1598.8 cm⁻¹ overlaps with
6H-SiC second-order band edge (1524–1709 cm⁻¹) → requires adaptive window
and dual-Lorentzian deconvolution (already implemented in v1.3).

---

## Cross-Reference Map

| Software feature | Primary ref | Secondary ref |
|-----------------|-------------|---------------|
| D/G area ratio | REF-01 | — |
| L_D calculation | REF-01 | — |
| n_D calculation | REF-01 | — |
| Two-stage disorder | REF-01 | — |
| SiC substrate subtraction | REF-02 | REF-03 |
| Amorphous SiC detection | REF-02 | — |
| Polytype identification | REF-03 | REF-02 |
| LOPC / N-doping detection | REF-03 | REF-04 |
| G+D′ deconvolution | REF-04 | — |
| 2D splitting (layer count) | REF-04 | — |
| Strain vs. doping plot | REF-04 | — |
| Graphitization index | REF-02 | REF-04 |
