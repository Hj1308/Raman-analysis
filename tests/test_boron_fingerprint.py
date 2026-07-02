"""
Unit tests for B-doping fingerprint flag (v2.4 Feature #2).

Kim et al. (2012) ACS Nano 6, 8203 criteria:
  - G center constant at 1577-1587 cm^-1 (no N-doping blue-shift)
  - 5 <= I_D/I_D' <= 9  (sp3 substitutional B, not sp2 N)
  - I_D/I_G > 3

Tests cover:
  1. All three criteria met          -> flag=True
  2. G shifted out of B-doping range -> flag=False
  3. I_D/I_D' too low (sp2 defects)  -> flag=False
  4. I_D/I_G below threshold         -> flag=False
  5. D' not detected                 -> flag=False
  6. Note content when flag=True
"""
import math
import numpy as np
import pytest
from src.peak_fitter import PeakResult
from src.analyzer import analyze, _BORON_G_CENTER_MIN, _BORON_G_CENTER_MAX


def _make_peak(name, center, amplitude, fwhm=20.0, r2=0.99, snr=50.0):
    """Helper: build a fully populated PeakResult."""
    p = PeakResult(
        name=name,
        center=center,
        amplitude=amplitude,
        fwhm=fwhm,
        area=math.pi * amplitude * (fwhm / 2.0),
        r_squared=r2,
        snr=snr,
        found=True,
    )
    return p


def _boron_peaks(
    g_center=1582.0,
    id_ig=4.0,
    id_idp=7.0,
):
    """
    Build peaks dict that satisfies B-doping criteria by construction.

    Given id_ig and id_idp, D amplitude = g_amp * id_ig,
    D' amplitude = D_amp / id_idp.
    """
    g_amp  = 10.0
    d_amp  = g_amp * id_ig
    dp_amp = d_amp / id_idp

    return {
        "G":       _make_peak("G",  g_center,  g_amp),
        "D":       _make_peak("D",  1350.0,    d_amp,  fwhm=30.0),
        "D_prime": _make_peak("D'", 1622.0,    dp_amp, fwhm=15.0),
    }


class TestBoronFingerprintPositive:
    def test_flag_true_for_canonical_b_doping(self):
        """All three Kim 2012 criteria met: flag must be True."""
        peaks = _boron_peaks(g_center=1582.0, id_ig=4.0, id_idp=7.0)
        analysis = analyze(peaks, laser_nm=532.0)
        assert analysis.boron_doping_flag is True

    def test_note_contains_kim_citation(self):
        peaks = _boron_peaks()
        analysis = analyze(peaks, laser_nm=532.0)
        assert analysis.boron_doping_flag is True
        assert "Kim et al. 2012" in analysis.boron_doping_note

    def test_note_contains_g_center(self):
        peaks = _boron_peaks(g_center=1582.0)
        analysis = analyze(peaks, laser_nm=532.0)
        assert "1582" in analysis.boron_doping_note or "1582.0" in analysis.boron_doping_note

    def test_note_contains_id_idp_value(self):
        peaks = _boron_peaks(id_ig=4.0, id_idp=7.0)
        analysis = analyze(peaks, laser_nm=532.0)
        assert "7.0" in analysis.boron_doping_note or "7" in analysis.boron_doping_note

    @pytest.mark.parametrize("g_center", [1577.0, 1582.0, 1587.0])
    def test_flag_true_across_G_center_range(self, g_center):
        """Flag must trigger for all valid G positions in B-doping range."""
        peaks = _boron_peaks(g_center=g_center, id_ig=4.0, id_idp=7.0)
        analysis = analyze(peaks, laser_nm=532.0)
        assert analysis.boron_doping_flag is True, (
            "Expected flag=True for G at {:.1f} cm^-1".format(g_center)
        )

    @pytest.mark.parametrize("id_idp", [5.0, 7.0, 9.0])
    def test_flag_true_across_ID_IDp_range(self, id_idp):
        peaks = _boron_peaks(id_ig=4.0, id_idp=id_idp)
        analysis = analyze(peaks, laser_nm=532.0)
        assert analysis.boron_doping_flag is True


class TestBoronFingerprintNegative:
    def test_flag_false_G_shifted_high(self):
        """G at 1598 cm^-1 (N-doping shift): flag must be False."""
        peaks = _boron_peaks(g_center=1598.0)
        analysis = analyze(peaks, laser_nm=532.0)
        assert analysis.boron_doping_flag is False

    def test_flag_false_G_shifted_low(self):
        """G at 1570 cm^-1: below B-doping G range."""
        peaks = _boron_peaks(g_center=1570.0)
        analysis = analyze(peaks, laser_nm=532.0)
        assert analysis.boron_doping_flag is False

    def test_flag_false_ID_IDp_too_low(self):
        """I_D/I_D' = 3 < 5 threshold: sp2-type defect, not B."""
        peaks = _boron_peaks(id_ig=4.0, id_idp=3.0)
        analysis = analyze(peaks, laser_nm=532.0)
        assert analysis.boron_doping_flag is False

    def test_flag_false_ID_IDp_too_high(self):
        """I_D/I_D' = 12 > 9 threshold."""
        peaks = _boron_peaks(id_ig=4.0, id_idp=12.0)
        analysis = analyze(peaks, laser_nm=532.0)
        assert analysis.boron_doping_flag is False

    def test_flag_false_ID_IG_too_low(self):
        """I_D/I_G = 1.5 < 3.0 threshold."""
        peaks = _boron_peaks(id_ig=1.5, id_idp=7.0)
        analysis = analyze(peaks, laser_nm=532.0)
        assert analysis.boron_doping_flag is False

    def test_flag_false_no_D_prime(self):
        """D' not detected: cannot compute I_D/I_D', flag must be False."""
        peaks = _boron_peaks()
        dp = peaks["D_prime"]
        dp.found = False          # mark as not detected
        analysis = analyze(peaks, laser_nm=532.0)
        assert analysis.boron_doping_flag is False

    def test_flag_false_no_D_band(self):
        peaks = _boron_peaks()
        peaks["D"].found = False
        analysis = analyze(peaks, laser_nm=532.0)
        assert analysis.boron_doping_flag is False

    def test_flag_false_no_G_band(self):
        peaks = _boron_peaks()
        peaks["G"].found = False
        analysis = analyze(peaks, laser_nm=532.0)
        assert analysis.boron_doping_flag is False

    def test_empty_note_when_flag_false(self):
        """boron_doping_note must be empty string when flag is False."""
        peaks = _boron_peaks(g_center=1598.0)
        analysis = analyze(peaks, laser_nm=532.0)
        assert analysis.boron_doping_flag is False
        assert analysis.boron_doping_note == ""


class TestBoronFingerprintEdgeCases:
    def test_boundary_G_center_exactly_at_min(self):
        """G exactly at _BORON_G_CENTER_MIN (1577.0): flag=True."""
        peaks = _boron_peaks(g_center=_BORON_G_CENTER_MIN, id_ig=4.0, id_idp=7.0)
        analysis = analyze(peaks, laser_nm=532.0)
        assert analysis.boron_doping_flag is True

    def test_boundary_G_center_exactly_at_max(self):
        """G exactly at _BORON_G_CENTER_MAX (1587.0): flag=True."""
        peaks = _boron_peaks(g_center=_BORON_G_CENTER_MAX, id_ig=4.0, id_idp=7.0)
        analysis = analyze(peaks, laser_nm=532.0)
        assert analysis.boron_doping_flag is True

    def test_boundary_G_center_just_above_max(self):
        """G at 1587.1: just outside range, flag=False."""
        peaks = _boron_peaks(g_center=1587.1, id_ig=4.0, id_idp=7.0)
        analysis = analyze(peaks, laser_nm=532.0)
        assert analysis.boron_doping_flag is False
