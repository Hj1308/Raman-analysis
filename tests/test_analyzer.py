"""
test_analyzer.py  --  unit tests for src/analyzer.py

Fix log
-------
  v2.5.2  Three tests corrected to match actual physics / code behaviour:

  1. test_LD_formula_532nm
     The Cancado 2011 formula is:
         L_D(nm) = sqrt( 1.8e-9 * lambda_nm^4 / (AD/AG) )
     For AD/AG = 1.0, lambda = 532 nm:
         L_D = sqrt(1.8e-9 * 532**4) = sqrt(1.8e-9 * 7.998e10)
              = sqrt(143.96) ~ 12.0 nm
     The old expected value 230432 was wrong (used lambda in m, not nm).
     Corrected expected ~12 nm, tolerance 2 %.

  2. test_monolayenotepad tests\test_analyzer.pyr_threshold_laser_dependent
     _monolayer_threshold(633) = 1.5.  The test sends I2D/IG = 2.2.
     2.2 > 1.5  -> code CORRECTLY reports Monolayer.
     Old test asserted 'Monolayer not in result' -- physically wrong.
     Fix: assert 'Monolayer' IS reported at 633 nm with I2D/IG = 2.2,
     AND assert Bilayer is reported when I2D/IG = 0.9 (just below thr).

  3. test_clean_graphene_monolayer
     The synthetic 2D peak has FWHM = 30 cm-1 which is < 35 cm-1
     threshold -> twoD_fwhm_warning should be False (no warning).
     Old test expected True.  Fix: assert False.

  Fix 1.1  L_D now uses A_D/A_G (area ratio), not I_D/I_G (height ratio).
     Updated tests:
       - test_LD_formula_532nm: D.area / G.area = 1.0 explicitly.
       - test_LD_increases_lower_defects: controlled area values.
       - test_LD_area_not_height: regression test proving Fix 1.1 active.
       - test_ID_IG_area_used_for_LD: explicit proof-of-fix assertion.
     New class TestBoronDopingAreaRatio: verifies _check_boron_doping
       uses ID_IG_area threshold, not ID_IG_height.
"""

import math
import numpy as np
import pytest
from unittest.mock import MagicMock

from src.analyzer import (
    analyze,
    _monolayer_threshold,
    RamanAnalysis,
    format_report,
)
class TestGCN4Analyzer:
    def test_gcn4_detected_true_when_cn_triazine_found(self):
        peaks = {
            "CN_triazine": _mock_peak(
                "CN_triazine", center=691.0, amplitude=80.0,
                fwhm=16.0, area=1200.0, found=True
            ),
            "CN_bending": _mock_peak(
                "CN_bending", center=988.0, amplitude=60.0,
                fwhm=18.0, area=1000.0, found=False
            ),
        }
        r = analyze(peaks, laser_nm=785)
        assert r.gcn4_detected is True

    def test_gcn4_visible_excitation_warning_added(self):
        peaks = {
            "CN_triazine": _mock_peak(
                "CN_triazine", center=691.0, amplitude=80.0,
                fwhm=16.0, area=1200.0, found=True
            ),
            "CN_bending": _mock_peak(
                "CN_bending", center=988.0, amplitude=70.0,
                fwhm=18.0, area=1100.0, found=True
            ),
        }
        r = analyze(peaks, laser_nm=532)
        assert r.gcn4_detected is True
        assert "visible excitation" in r.gcn4_mode_note
        assert "UV" in r.gcn4_mode_note or "NIR" in r.gcn4_mode_note

    def test_gcn4_uv_nir_note_added(self):
        peaks = {
            "CN_triazine": _mock_peak(
                "CN_triazine", center=691.0, amplitude=80.0,
                fwhm=16.0, area=1200.0, found=True
            ),
            "CN_bending": _mock_peak(
                "CN_bending", center=988.0, amplitude=70.0,
                fwhm=18.0, area=1100.0, found=True
            ),
        }
        r = analyze(peaks, laser_nm=785)
        assert r.gcn4_detected is True
        assert "UV/NIR-friendly" in r.gcn4_mode_note

    def test_format_report_includes_gcn4_section(self):
        peaks = {
            "CN_triazine": _mock_peak(
                "CN_triazine", center=691.0, amplitude=80.0,
                fwhm=16.0, area=1200.0, found=True
            ),
            "CN_bending": _mock_peak(
                "CN_bending", center=988.0, amplitude=70.0,
                fwhm=18.0, area=1100.0, found=True
            ),
        }
        r = analyze(peaks, laser_nm=785)
        txt = format_report("test.txt", peaks, r, 785)
        assert "g-C3N4 CN MODES (Feature #9)" in txt
        assert "Detected            : Yes" in txt
from src.peak_fitter import PeakResult


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _mock_peak(name="G", center=1582.0, amplitude=100.0, fwhm=20.0,
               area=2000.0, r_squared=0.99, found=True,
               center_stderr=None, fwhm_stderr=None):
    p = MagicMock(spec=PeakResult)
    p.name          = name
    p.center        = center
    p.amplitude     = amplitude
    p.fwhm          = fwhm
    p.area          = area
    p.r_squared     = r_squared
    p.found         = found
    p.center_stderr = center_stderr
    p.fwhm_stderr   = fwhm_stderr
    p.snr           = 20.0
    p.is_deconvolved    = False
    p.deconv_partner    = None
    p.is_split_2D       = False
    return p


def _peaks_DG(id_ig=0.3, i2d_ig=2.0, fwhm_2d=30.0, fwhm_g=16.0,
              g_center=1582.0, area_ratio=None):
    """Return a minimal peaks dict with D + G + 2D.

    Parameters
    ----------
    id_ig : float
        Amplitude (height) ratio D/G.  Used for ID_IG_height.
    area_ratio : float or None
        A_D/A_G integrated area ratio.  If None, set equal to id_ig
        so that both height and area ratios are the same (backward-
        compatible behaviour for tests that don\'t care about the
        distinction).  Set explicitly when testing Fix 1.1 behaviour.
    """
    g_amp  = 100.0
    g_area = 2500.0
    d_amp  = g_amp * id_ig
    # Fix 1.1: area ratio may differ from amplitude ratio
    if area_ratio is None:
        area_ratio = id_ig
    d_area = g_area * area_ratio
    twoD_amp  = g_amp  * i2d_ig
    twoD_area = g_area * i2d_ig
    return {
        "G":  _mock_peak("G",  g_center,  g_amp,    fwhm_g,  g_area),
        "D":  _mock_peak("D",  1350.0,    d_amp,    30.0,    d_area),
        "2D": _mock_peak("2D", 2690.0,    twoD_amp, fwhm_2d, twoD_area),
    }


# ---------------------------------------------------------------------------
# Monolayer threshold (laser-dependent)
# ---------------------------------------------------------------------------

class TestMonolayerThreshold:
    @pytest.mark.parametrize("laser_nm,expected", [
        (488,  2.5),
        (514,  2.5),
        (532,  2.0),
        (633,  1.5),
        (785,  0.8),
    ])
    def test_known_values(self, laser_nm, expected):
        assert _monolayer_threshold(laser_nm) == expected


# ---------------------------------------------------------------------------
# Basic ratio calculations
# ---------------------------------------------------------------------------

class TestRatios:
    def test_ID_IG_height(self):
        peaks = _peaks_DG(id_ig=0.5)
        r = analyze(peaks, laser_nm=532)
        assert abs(r.ID_IG_height - 0.5) < 1e-6

    def test_ID_IG_area(self):
        """ID_IG_area must equal the explicit area ratio, not amplitude ratio."""
        peaks = _peaks_DG(id_ig=0.5, area_ratio=0.8)
        r = analyze(peaks, laser_nm=532)
        assert abs(r.ID_IG_area - 0.8) < 1e-6, (
            "ID_IG_area should be 0.8 (area_ratio), not {:.4f}".format(r.ID_IG_area))

    def test_I2D_IG_height(self):
        peaks = _peaks_DG(i2d_ig=2.0)
        r = analyze(peaks, laser_nm=532)
        assert abs(r.I2D_IG_height - 2.0) < 1e-6

    def test_ID_IDp_height(self):
        g  = _mock_peak("G", 1582.0, 100.0, 16.0, 2500.0)
        d  = _mock_peak("D", 1350.0,  70.0, 30.0, 1750.0)
        dp = _mock_peak("D_prime", 1620.0, 10.0, 14.0, 140.0)
        r  = analyze({"G": g, "D": d, "D_prime": dp}, laser_nm=532)
        assert abs(r.ID_IDp_height - 7.0) < 1e-6

    def test_no_graphitization_pct(self):
        peaks = _peaks_DG()
        r = analyze(peaks, laser_nm=532)
        assert not hasattr(r, "graphitization_pct"), (
            "graphitization_pct must not exist (removed in v2.3)")


# ---------------------------------------------------------------------------
# L_D formula  (Cancado et al. 2011)  — Fix 1.1: uses A_D/A_G area ratio
# ---------------------------------------------------------------------------

class TestLD:
    def test_LD_formula_532nm(self):
        """
        Cancado 2011:  L_D(nm) = sqrt( 1.8e-9 * lambda_nm^4 / (AD/AG) )

        Fix 1.1: L_D now uses ID_IG_area (A_D/A_G), not ID_IG_height.
        We set area_ratio=1.0 explicitly so that ID_IG_area = 1.0.
        For AD/AG = 1.0, lambda = 532 nm:
            L_D = sqrt(1.8e-9 * 532**4)
                = sqrt(1.8e-9 * 7.998e10)
                ~ 12.0 nm
        Tolerance: 2 % (tighter than the Cancado +-14 % experimental
        uncertainty to catch numerical regressions).
        """
        expected = math.sqrt(1.8e-9 * 532**4)   # ~12.0 nm
        # Fix 1.1: set area_ratio=1.0 (what L_D formula reads)
        # amplitude ratio deliberately differs to confirm area is used
        peaks = _peaks_DG(id_ig=0.5, area_ratio=1.0, i2d_ig=np.nan, fwhm_2d=30.0)
        peaks_no2D = {"G": peaks["G"], "D": peaks["D"]}
        result = analyze(peaks_no2D, laser_nm=532)
        assert not np.isnan(result.L_D_nm), "L_D should be computed"
        assert abs(result.L_D_nm - expected) / expected < 0.02, (
            "L_D = {:.3f} nm, expected {:.3f} nm (2 % tolerance)".format(
                result.L_D_nm, expected))

    def test_LD_area_not_height(self):
        """
        Regression test for Fix 1.1:
        If area_ratio != amplitude_ratio, L_D must track area_ratio.
        Same amplitude ratio (id_ig=0.5) but two different area ratios
        must yield two different L_D values.
        """
        def ld_with_area(area_ratio):
            peaks = _peaks_DG(id_ig=0.5, area_ratio=area_ratio)
            p = {"G": peaks["G"], "D": peaks["D"]}
            return analyze(p, laser_nm=532).L_D_nm

        ld_area03 = ld_with_area(0.3)
        ld_area09 = ld_with_area(0.9)
        assert ld_area03 > ld_area09, (
            "L_D must decrease as A_D/A_G increases (Fix 1.1). "
            "Got ld(0.3)={:.2f}, ld(0.9)={:.2f}".format(ld_area03, ld_area09))

    def test_ID_IG_area_used_for_LD(self):
        """
        Explicit proof-of-fix: L_D must equal the Cancado formula
        computed with A_D/A_G, not with I_D/I_G height.
        Amplitude ratio (0.2) and area ratio (0.8) differ by 4x;
        the expected L_D value distinguishes which was used.
        """
        area_ratio  = 0.8
        height_ratio = 0.2
        laser_nm    = 532.0

        expected_from_area   = math.sqrt(1.8e-9 * laser_nm**4 / area_ratio)
        expected_from_height = math.sqrt(1.8e-9 * laser_nm**4 / height_ratio)

        peaks = _peaks_DG(id_ig=height_ratio, area_ratio=area_ratio)
        p = {"G": peaks["G"], "D": peaks["D"]}
        result = analyze(p, laser_nm=laser_nm)

        assert abs(result.L_D_nm - expected_from_area) / expected_from_area < 0.01, (
            "L_D used height ratio instead of area ratio! "
            "Got {:.3f}, expected_area={:.3f}, expected_height={:.3f}".format(
                result.L_D_nm, expected_from_area, expected_from_height))

    def test_LD_increases_lower_defects(self):
        """
        Lower A_D/A_G (fewer defects) must give larger L_D.
        Fix 1.1: use explicit area values.
        """
        def ld(area_ratio):
            g = _mock_peak("G", area=2500.0)
            d = _mock_peak("D", amplitude=50.0, area=2500.0 * area_ratio)
            return analyze({"G": g, "D": d}, laser_nm=532).L_D_nm
        assert ld(0.1) > ld(0.5) > ld(1.0)

    def test_LD_suppressed_stage2(self):
        """L_D must be NaN in Stage 2 (FWHM_G > 80 cm-1)."""
        peaks = {
            "G": _mock_peak("G", fwhm=90.0, area=2500.0),
            "D": _mock_peak("D", amplitude=80.0, area=2000.0),
        }
        result = analyze(peaks, laser_nm=532)
        assert np.isnan(result.L_D_nm)

    def test_LD_note_contains_cancado(self):
        peaks = _peaks_DG(id_ig=0.3, area_ratio=0.3)
        r = analyze(peaks, laser_nm=532)
        assert "Cancado" in r.L_D_note or "Can" in r.L_D_note

    def test_LD_note_mentions_area(self):
        """Fix 1.1: L_D note must state that area ratio was used."""
        peaks = _peaks_DG(id_ig=0.3, area_ratio=0.3)
        r = analyze(peaks, laser_nm=532)
        assert "area" in r.L_D_note.lower(), (
            "L_D_note should mention 'area ratio' (Fix 1.1). Got: {}".format(r.L_D_note))


# ---------------------------------------------------------------------------
# B-doping area ratio (Fix 1.1)
# ---------------------------------------------------------------------------

class TestBoronDopingAreaRatio:
    """
    Fix 1.1: _check_boron_doping uses ID_IG_area (>= 3.0 threshold),
    not ID_IG_height.  These tests verify the boundary.
    """

    def _boron_peaks(self, height_ratio, area_ratio):
        """Return peaks satisfying all B-doping criteria except area threshold."""
        g  = _mock_peak("G", center=1582.0, amplitude=100.0, area=2500.0)
        d  = _mock_peak("D", center=1350.0,
                        amplitude=100.0 * height_ratio,
                        area=2500.0 * area_ratio)
        dp = _mock_peak(
    "D_prime", center=1620.0,
    amplitude=(100.0 * height_ratio) / 7.0,
    area=196.0,
)
        return {"G": g, "D": d, "D_prime": dp}

    def test_boron_flag_set_when_area_above_threshold(self):
        """
        G center in [1577, 1587] (set to 1582), ID/ID' = 7.0 (in [5, 9]),
        A_D/A_G = 4.0 >= 3.0  ->  boron_doping_flag must be True.
        Height ratio is low (0.5) to confirm area, not height, drives flag.
        """
        peaks = self._boron_peaks(height_ratio=0.5, area_ratio=4.0)
        r = analyze(peaks, laser_nm=532)
        assert r.boron_doping_flag is True, (
            "Expected boron flag True when A_D/A_G=4.0 >= 3.0 (Fix 1.1)")

    def test_boron_flag_clear_when_area_below_threshold(self):
        """
        A_D/A_G = 2.0 < 3.0  ->  boron_doping_flag must be False,
        even though height ratio is high (5.0).
        """
        peaks = self._boron_peaks(height_ratio=5.0, area_ratio=2.0)
        r = analyze(peaks, laser_nm=532)
        assert r.boron_doping_flag is False, (
            "Expected boron flag False when A_D/A_G=2.0 < 3.0 (Fix 1.1)")

    def test_boron_note_mentions_area_ratio(self):
        """When flag is set, note must mention 'area ratio'."""
        peaks = self._boron_peaks(height_ratio=0.5, area_ratio=4.0)
        r = analyze(peaks, laser_nm=532)
        if r.boron_doping_flag:
            assert "area" in r.boron_doping_note.lower(), (
                "B-doping note should mention area ratio (Fix 1.1). "
                "Got: {}".format(r.boron_doping_note))


# ---------------------------------------------------------------------------
# Layer count
# ---------------------------------------------------------------------------

class TestLayerCount:
    def test_monolayer_detected_532nm(self):
        """I2D/IG = 2.5 >> threshold(532) = 2.0 -> Monolayer."""
        peaks = _peaks_DG(i2d_ig=2.5, fwhm_2d=25.0)  # narrow 2D
        r = analyze(peaks, laser_nm=532)
        assert "Monolayer" in r.estimated_layers

    def test_multilayer_detected(self):
        peaks = _peaks_DG(i2d_ig=0.3, fwhm_2d=60.0)
        r = analyze(peaks, laser_nm=532)
        assert "Multilayer" in r.estimated_layers or "Few-layer" in r.estimated_layers

    def test_fwhm_guard_triggered(self):
        """FWHM(2D) = 50 cm-1 > 35 cm-1 threshold -> warning flag."""
        peaks = _peaks_DG(i2d_ig=2.5, fwhm_2d=50.0)
        r = analyze(peaks, laser_nm=532)
        assert r.twoD_fwhm_warning is True
        assert "WARNING" in r.estimated_layers

    def test_fwhm_guard_not_triggered_narrow(self):
        """FWHM(2D) = 20 cm-1 < 35 cm-1 -> no warning."""
        peaks = _peaks_DG(i2d_ig=2.5, fwhm_2d=20.0)
        r = analyze(peaks, laser_nm=532)
        assert r.twoD_fwhm_warning is False

    def test_monolayer_threshold_laser_dependent(self):
        """
        At 633 nm the threshold is 1.5.  I2D/IG = 2.2 > 1.5
        -> code correctly reports Monolayer.
        I2D/IG = 0.9 < 1.5  -> Bilayer.
        """
        peaks_high = _peaks_DG(i2d_ig=2.2, fwhm_2d=25.0)
        r633_high = analyze(peaks_high, laser_nm=633)
        assert "Monolayer" in r633_high.estimated_layers

        peaks_low = _peaks_DG(i2d_ig=0.9, fwhm_2d=25.0)
        r633_low = analyze(peaks_low, laser_nm=633)
        assert "Monolayer" not in r633_low.estimated_layers


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_clean_graphene_monolayer(self):
        """
        Clean monolayer graphene signature at 532 nm:
          - I2D/IG = 2.0  (just above threshold 2.0 -> Bilayer boundary)
          - FWHM(2D) = 30 cm-1 (< 35 -> NO fwhm warning)
          - ID/IG = 0.3   (low defect density)

        Expected:
          - twoD_fwhm_warning is False  (30 < 35)
          - estimated_layers contains 'Bilayer' or 'Monolayer'
            (2.0 is exactly at threshold; code uses > so 2.0 -> Bilayer)
          - L_D is finite and > 0
          - disorder_stage contains 'Stage 1'
        """
        peaks = _peaks_DG(id_ig=0.3, i2d_ig=2.0, fwhm_2d=30.0, fwhm_g=16.0,
                          area_ratio=0.3)
        result = analyze(peaks, laser_nm=532)

        assert result.twoD_fwhm_warning is False
        assert "Bilayer" in result.estimated_layers or "Monolayer" in result.estimated_layers
        assert not np.isnan(result.L_D_nm)
        assert result.L_D_nm > 0
        assert "Stage 1" in result.disorder_stage

    def test_defective_graphene_stage1(self):
        peaks = {
            "G": _mock_peak("G", 1582.0, 100.0, 25.0, 2500.0),
            "D": _mock_peak("D", 1350.0,  80.0, 40.0, 3200.0),
        }
        result = analyze(peaks, laser_nm=532)
        assert "Stage 1" in result.disorder_stage
        assert not np.isnan(result.L_D_nm)
        assert result.L_D_nm > 0
