"""
Unit tests for src/band_edge_calculator.py

Tests the Butler-Ginley band edge formalism:
    E_CB(NHE) = χ - 4.44 - 0.5·Eg
    E_VB(NHE) = E_CB(NHE) + Eg

All 8 tests must pass for CI to be green.
"""

from __future__ import annotations

import math
import pytest

from src.band_edge_calculator import (
    BandEdgeCalculator,
    calculate_band_edges,
    _NHE_VS_VAC,
    _CHI_DB,
)

# ---------------------------------------------------------------------------
# Constants used in expected values
# ---------------------------------------------------------------------------
NHE = 4.44  # eV — NHE vs. vacuum


# ===========================================================================
# Test 1: Core formula — E_CB(NHE) for g-C3N4
# χ(g-C3N4) = 4.73 eV, Eg = 2.70 eV
# E_CB = 4.73 - 4.44 - 0.5*2.70 = 4.73 - 4.44 - 1.35 = -1.06 eV
# ===========================================================================
def test_ecb_nhe_g_c3n4():
    """E_CB(NHE) for g-C3N4 matches Butler-Ginley formula."""
    calc = BandEdgeCalculator(chi=4.73, Eg=2.70)
    expected = 4.73 - NHE - 0.5 * 2.70  # = -1.06
    assert math.isclose(calc.Ecb_NHE, expected, rel_tol=1e-6), (
        f"Expected E_CB(NHE) = {expected:.4f} eV, got {calc.Ecb_NHE:.4f} eV"
    )


# ===========================================================================
# Test 2: Core formula — E_VB(NHE) for g-C3N4
# E_VB = E_CB + Eg = -1.06 + 2.70 = +1.64 eV
# ===========================================================================
def test_evb_nhe_g_c3n4():
    """E_VB(NHE) = E_CB(NHE) + Eg."""
    calc = BandEdgeCalculator(chi=4.73, Eg=2.70)
    expected = calc.Ecb_NHE + 2.70
    assert math.isclose(calc.Evb_NHE, expected, rel_tol=1e-6)


# ===========================================================================
# Test 3: Band gap conservation — VB - CB == Eg
# ===========================================================================
def test_band_gap_conservation():
    """E_VB - E_CB must equal Eg exactly."""
    for chi, Eg in [(4.73, 2.70), (5.81, 3.20), (4.88, 2.40)]:
        calc = BandEdgeCalculator(chi=chi, Eg=Eg)
        assert math.isclose(calc.Evb_NHE - calc.Ecb_NHE, Eg, rel_tol=1e-9), (
            f"Gap conservation failed for chi={chi}, Eg={Eg}"
        )


# ===========================================================================
# Test 4: TiO2 reference values
# χ(TiO2) = 5.81 eV, Eg = 3.20 eV
# E_CB = 5.81 - 4.44 - 1.60 = -0.23 eV
# E_VB = -0.23 + 3.20 = +2.97 eV
# ===========================================================================
def test_tio2_reference_values():
    """TiO2 band edges match literature values (±0.01 eV tolerance)."""
    calc = BandEdgeCalculator(chi=5.81, Eg=3.20)
    assert math.isclose(calc.Ecb_NHE, -0.23, abs_tol=0.01), (
        f"TiO2 E_CB(NHE): expected ≈ -0.23 eV, got {calc.Ecb_NHE:.4f} eV"
    )
    assert math.isclose(calc.Evb_NHE, 2.97, abs_tol=0.01), (
        f"TiO2 E_VB(NHE): expected ≈ +2.97 eV, got {calc.Evb_NHE:.4f} eV"
    )


# ===========================================================================
# Test 5: Vacuum vs NHE conversion consistency
# E_CB(vac) = χ - 0.5·Eg
# E_CB(NHE) = E_CB(vac) - 4.44
# ===========================================================================
def test_vac_nhe_conversion_consistency():
    """E_CB(NHE) = E_CB(vac) - _NHE_VS_VAC."""
    calc = BandEdgeCalculator(chi=5.81, Eg=3.20)
    assert math.isclose(
        calc.Ecb_NHE,
        calc.Ecb_vac - _NHE_VS_VAC,
        rel_tol=1e-9,
    ), "Vacuum-to-NHE conversion is inconsistent"


# ===========================================================================
# Test 6: No double-subtraction of reference constants
# The bug was: Ecb_vac = chi - 4.50 - 0.5*Eg (wrong, added Ec_ref)
#             Ecb_NHE = Ecb_vac - 4.44        (then also subtracted NHE)
# This test explicitly catches that pattern.
# ===========================================================================
def test_no_double_subtraction_bug():
    """Ensure old double-subtraction bug (chi - 8.94 - 0.5*Eg) is absent."""
    chi, Eg = 4.73, 2.70
    calc = BandEdgeCalculator(chi=chi, Eg=Eg)
    buggy_value = chi - 8.94 - 0.5 * Eg  # what the old code produced
    correct_value = chi - NHE - 0.5 * Eg
    assert not math.isclose(calc.Ecb_NHE, buggy_value, abs_tol=0.01), (
        "Calculator still returns the double-subtraction bug value!"
    )
    assert math.isclose(calc.Ecb_NHE, correct_value, rel_tol=1e-6)


# ===========================================================================
# Test 7: ValueError on non-positive band gap
# ===========================================================================
def test_raises_on_non_positive_Eg():
    """BandEdgeCalculator must raise ValueError for Eg <= 0."""
    with pytest.raises(ValueError, match="positive"):
        BandEdgeCalculator(chi=4.73, Eg=0.0)
    with pytest.raises(ValueError, match="positive"):
        BandEdgeCalculator(chi=4.73, Eg=-1.5)


# ===========================================================================
# Test 8: convenience function calculate_band_edges (material lookup)
# ===========================================================================
def test_calculate_band_edges_material_lookup():
    """calculate_band_edges() with material='TiO2' matches direct calculation."""
    result = calculate_band_edges(material="TiO2", Eg=3.20)
    direct = BandEdgeCalculator(chi=_CHI_DB["TiO2"], Eg=3.20).summary()
    assert result["Ecb_NHE_eV"] == direct["Ecb_NHE_eV"]
    assert result["Evb_NHE_eV"] == direct["Evb_NHE_eV"]
    # Also test unknown material raises KeyError
    with pytest.raises(KeyError):
        calculate_band_edges(material="UnknownMaterial", Eg=2.5)
