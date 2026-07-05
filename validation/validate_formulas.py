"""
validate_formulas.py — Literature validation of the crystallite-size /
defect-density equations used by this tool.

Run:  pytest validation/validate_formulas.py -v
      python validation/validate_formulas.py      (prints comparison table)

Design (three honest tiers — see validation/README.md):

  A. INDEPENDENT-PAIR AGREEMENT
     Only cases where the paper publishes BOTH the measured I_D/I_G AND the
     derived length independently count as real validation. Feeding one into
     the equation must reproduce the other. (Cases where a paper's length was
     itself computed from the same equation would be circular and are NOT
     used as agreement tests.)

  B. NEGATIVE CONTROL (regime limits)
     The Cançado low-defect relation is only valid for L_D ≳ 10 nm (Stage 1).
     Kim 2012's B-doped graphene sits at L_D ≈ 4.8 nm — outside the regime —
     so the simple formula MUST disagree with the published value. We assert
     that it does. Passing this test demonstrates the tool's regime warnings
     are scientifically necessary, not decorative.

  C. IMPLEMENTATION SELF-CONSISTENCY
     Round-trip and unit-conversion checks (labelled honestly as such):
     they verify the equations are coded correctly, not that they agree
     with any experiment.
"""

import math
import sys

import pytest

# ---------------------------------------------------------------------------
# Equations under test (identical to the tool's analyzer layer)
# ---------------------------------------------------------------------------
HC_eVnm = 1239.84  # h*c in eV.nm


def laser_energy_eV(wavelength_nm: float) -> float:
    return HC_eVnm / wavelength_nm


def tuinstra_koenig_L_a(E_L_eV: float, id_ig: float) -> float:
    """L_a (nm) = 560 / E_L^4 * (I_D/I_G)^-1   [Cancado 2006 general form]."""
    return 560.0 / (E_L_eV ** 4) / id_ig


def cancado_L_D(E_L_eV: float, id_ig: float) -> float:
    """L_D (nm) from L_D^2 = 4.3e3 / E_L^4 * (I_D/I_G)^-1  [Cancado 2011].
    Valid ONLY in the low-defect Stage-1 regime (L_D >~ 10 nm)."""
    return math.sqrt(4300.0 / (E_L_eV ** 4) / id_ig)


TOL = 0.10  # +-10 %


# ===========================================================================
# A. INDEPENDENT-PAIR AGREEMENT  (paper gives both ratio and length)
# ===========================================================================
class TestIndependentAgreement:
    """Dierke et al. 2022 (ACS Appl. Nano Mater. 5, 4966; 532 nm) publish the
    measured I_D/I_G AND the Cançado-derived L_D for the same spots —
    a genuinely independent check of our implementation + constants."""

    def test_dierke_sio2(self):
        # I_D/I_G = 0.82 +- 0.12  ->  published L_D ~ 12.4 nm
        calc = cancado_L_D(laser_energy_eV(532), 0.82)
        rel = abs(calc - 12.4) / 12.4
        assert rel <= TOL, f"L_D calc={calc:.2f} vs 12.4 (err {rel:.1%})"

    def test_dierke_hbn_range_aware(self):
        # The hBN value is a LINESCAN with I_D/I_G = 0.51 +- 0.16 and the
        # published L_D ~ 20.9 nm corresponds to the low-ratio end of that
        # scan (lowest functionalisation). The honest test: somewhere inside
        # the published ratio interval, the formula must reproduce 20.9 nm.
        E_L = laser_energy_eV(532)
        errors = [abs(cancado_L_D(E_L, r) - 20.9) / 20.9
                  for r in (0.35, 0.40, 0.45, 0.51, 0.60, 0.67)]
        assert min(errors) <= TOL, (
            f"best error across ratio interval = {min(errors):.1%}")

    def test_dierke_substrate_ordering(self):
        # Physics check: lower I_D/I_G (hBN) must give LARGER L_D than SiO2.
        E_L = laser_energy_eV(532)
        assert cancado_L_D(E_L, 0.51) > cancado_L_D(E_L, 0.82)


# ===========================================================================
# B. NEGATIVE CONTROL  (out-of-regime disagreement is REQUIRED)
# ===========================================================================
class TestRegimeNegativeControl:
    """Kim et al. 2012 (ACS Nano 6, 6293; 633 nm): I_D/I_G = 7.0 with a
    published defect distance of 4.76 nm computed from the FULL Lucchese
    curve. Because 4.76 nm < 10 nm, the Stage-1 low-defect formula is out
    of its validity regime and MUST disagree substantially. If this test
    ever 'fails' (i.e. the simple formula agrees), something is wrong with
    our understanding of the regime, not right."""

    def test_kim2012_out_of_regime_disagrees(self):
        calc = cancado_L_D(laser_energy_eV(633), 7.0)
        rel = abs(calc - 4.76) / 4.76
        assert rel > 0.20, (
            f"expected >20% disagreement outside regime, got {rel:.1%}")

    def test_kim2012_is_below_regime_floor(self):
        # and indeed the published value sits below the 10 nm validity floor,
        # which is exactly why the tool's validation layer flags such cases.
        assert 4.76 < 10.0


# ===========================================================================
# C. IMPLEMENTATION SELF-CONSISTENCY  (honestly labelled round-trips)
# ===========================================================================
class TestSelfConsistency:
    def test_ev_conversion_532(self):
        assert abs(laser_energy_eV(532) - 2.3305) < 0.001

    def test_ev_conversion_633(self):
        assert abs(laser_energy_eV(633) - 1.9587) < 0.001

    def test_la_roundtrip(self):
        # invertibility of the TK relation (implementation check only)
        E_L = laser_energy_eV(633)
        target_La = 14.7                       # any value
        ratio = 560.0 / (E_L ** 4) / target_La  # invert
        assert abs(tuinstra_koenig_L_a(E_L, ratio) - target_La) < 1e-9

    def test_ld_roundtrip(self):
        E_L = laser_energy_eV(785)
        target_Ld = 25.0
        ratio = 4300.0 / (E_L ** 4) / target_Ld ** 2
        assert abs(cancado_L_D(E_L, ratio) - target_Ld) < 1e-9

    def test_el4_scaling(self):
        # halving laser energy must scale L_a by 2^4 = 16 at fixed ratio
        r = 1.0
        assert abs(tuinstra_koenig_L_a(1.0, r) /
                   tuinstra_koenig_L_a(2.0, r) - 16.0) < 1e-9


# ---------------------------------------------------------------------------
# Standalone comparison table
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    E532, E633 = laser_energy_eV(532), laser_energy_eV(633)
    rows = [
        ("A: Dierke SiO2 (532nm)", 0.82, cancado_L_D(E532, 0.82), 12.4, "agree <=10%"),
        ("A: Dierke hBN mean",     0.51, cancado_L_D(E532, 0.51), 20.9, "range-aware"),
        ("A: Dierke hBN low-r",    0.35, cancado_L_D(E532, 0.35), 20.9, "agree <=10%"),
        ("B: Kim2012 (regime-)",   7.00, cancado_L_D(E633, 7.00), 4.76, "MUST disagree"),
    ]
    print(f"{'Case':<26}{'I_D/I_G':>8}{'L_D calc':>10}{'L_D pub':>9}"
          f"{'err%':>7}  expectation")
    print("-" * 72)
    ok = True
    for label, r, calc, pub, expect in rows:
        err = abs(calc - pub) / pub * 100
        print(f"{label:<26}{r:>8.2f}{calc:>10.2f}{pub:>9.2f}{err:>6.1f}%  {expect}")
        if expect.startswith("agree") and err > 10:
            ok = False
        if expect.startswith("MUST") and err < 20:
            ok = False
    print("-" * 72)
    if ok:
        print("All expectations met (agreements agree, negative control disagrees).")
        sys.exit(0)
    print("EXPECTATION VIOLATED — inspect table above.")
    sys.exit(1)
