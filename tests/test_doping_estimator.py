"""
Unit tests for doping level estimator (v2.5 Feature #4).

Based on:
  Pisana et al. (2007) Nature Mater. 6, 198
  Das et al.   (2008) Nat. Nanotechnol. 3, 210

Logic in analyzer._estimate_doping:
  - |delta_G| < 3 cm^-1            -> undoped
  - delta_G >= +3 cm^-1, I2D/IG < 0.5  -> n-type
  - delta_G >= +3 cm^-1, I2D/IG >= 0.5 -> p-type
  - carrier density from: n = (|delta_G| / alpha)^2
    where alpha = 2.2e-12 cm^-1 per (cm^-2)^0.5
    (returned as raw cm^-2, NOT divided by 1e12)

Tests:
  1. Undoped: G at 1582 (no shift)
  2. Undoped: G at 1584 (shift = 2 < 3 threshold)
  3. n-type: G blue-shifted, I2D/IG low
  4. p-type: G blue-shifted, I2D/IG high
  5. Carrier density increases with larger shift (monotonic)
  6. Returns N/A when G not found
  7. Note text contains expected references
"""
import math
import numpy as np
import pytest
from src.peak_fitter import PeakResult
from src.analyzer import analyze, _G0_UNDOPED, _ALPHA_PISANA, _DOPING_NOISE


def _make_G(center, fwhm=18.0):
    return PeakResult(
        name="G",
        center=center,
        amplitude=10.0,
        fwhm=fwhm,
        area=math.pi * 10.0 * (fwhm / 2.0),
        r_squared=0.99,
        snr=50.0,
        found=True,
    )


def _make_2D(amplitude, g_amp=10.0):
    """I2D/IG = amplitude / g_amp."""
    return PeakResult(
        name="2D",
        center=2700.0,
        amplitude=amplitude,
        fwhm=30.0,
        area=math.pi * amplitude * 15.0,
        r_squared=0.98,
        snr=40.0,
        found=True,
    )


class TestDopingTypeClassification:
    def test_undoped_exact_reference_position(self):
        """G at 1582.0 (=G0_UNDOPED): must be classified as undoped."""
        peaks = {"G": _make_G(_G0_UNDOPED)}
        analysis = analyze(peaks, laser_nm=532.0)
        assert analysis.doping_type == "undoped"
        assert analysis.carrier_density_cm2 == 0.0

    def test_undoped_within_noise_threshold(self):
        """Shift of 2 cm^-1 is below 3 cm^-1 noise floor: undoped."""
        peaks = {"G": _make_G(_G0_UNDOPED + 2.0)}
        analysis = analyze(peaks, laser_nm=532.0)
        assert analysis.doping_type == "undoped"

    def test_undoped_negative_shift_within_noise(self):
        peaks = {"G": _make_G(_G0_UNDOPED - 2.0)}
        analysis = analyze(peaks, laser_nm=532.0)
        assert analysis.doping_type == "undoped"

    def test_n_type_blue_shift_low_I2D_IG(self):
        """G at +8 cm^-1, I2D/IG = 0.2: n-type."""
        peaks = {
            "G":  _make_G(_G0_UNDOPED + 8.0),
            "2D": _make_2D(amplitude=2.0),   # I2D/IG = 0.2 < 0.5
        }
        analysis = analyze(peaks, laser_nm=532.0)
        assert analysis.doping_type == "n-type"

    def test_p_type_blue_shift_high_I2D_IG(self):
        """G at +8 cm^-1, I2D/IG = 0.8: p-type."""
        peaks = {
            "G":  _make_G(_G0_UNDOPED + 8.0),
            "2D": _make_2D(amplitude=8.0),   # I2D/IG = 0.8 >= 0.5
        }
        analysis = analyze(peaks, laser_nm=532.0)
        assert analysis.doping_type == "p-type"

    def test_n_type_when_no_2D_present(self):
        """No 2D band -> I2D/IG = NaN -> treated as n-type (< 0.5 branch)."""
        peaks = {"G": _make_G(_G0_UNDOPED + 8.0)}
        analysis = analyze(peaks, laser_nm=532.0)
        assert analysis.doping_type == "n-type"

    @pytest.mark.parametrize("shift", [4.0, 8.0, 15.0, 25.0])
    def test_doping_type_set_for_significant_shifts(self, shift):
        """Any shift > noise floor must yield a non-N/A, non-undoped type."""
        peaks = {"G": _make_G(_G0_UNDOPED + shift)}
        analysis = analyze(peaks, laser_nm=532.0)
        assert analysis.doping_type in ("n-type", "p-type")


class TestCarrierDensity:
    def test_carrier_density_positive(self):
        """Carrier density must be positive for a real shift."""
        peaks = {"G": _make_G(_G0_UNDOPED + 10.0)}
        analysis = analyze(peaks, laser_nm=532.0)
        assert analysis.carrier_density_cm2 > 0

    def test_carrier_density_formula(self):
        """
        Manual calculation matching analyzer._estimate_doping:
          n_abs = (|delta_G| / _ALPHA_PISANA) ** 2
        carrier_density_cm2 is stored as n_abs (raw cm^-2, not /1e12).
        """
        delta = 10.0
        peaks = {"G": _make_G(_G0_UNDOPED + delta)}
        analysis = analyze(peaks, laser_nm=532.0)
        expected_n = (delta / _ALPHA_PISANA) ** 2
        rel_err = abs(analysis.carrier_density_cm2 - expected_n) / expected_n
        assert rel_err < 1e-9, (
            "Carrier density formula mismatch: {:.3e} vs {:.3e}".format(
                analysis.carrier_density_cm2, expected_n)
        )

    def test_carrier_density_increases_with_shift(self):
        """Larger G shift -> higher carrier density (monotonic)."""
        shifts = [4.0, 8.0, 15.0]
        densities = []
        for sh in shifts:
            peaks = {"G": _make_G(_G0_UNDOPED + sh)}
            analysis = analyze(peaks, laser_nm=532.0)
            densities.append(analysis.carrier_density_cm2)
        assert densities[0] < densities[1] < densities[2], (
            "Expected monotonic increase: {}".format(densities)
        )

    def test_carrier_density_zero_for_undoped(self):
        peaks = {"G": _make_G(_G0_UNDOPED)}
        analysis = analyze(peaks, laser_nm=532.0)
        assert analysis.carrier_density_cm2 == 0.0


class TestDopingNoteContent:
    def test_note_contains_pisana_citation(self):
        peaks = {"G": _make_G(_G0_UNDOPED + 8.0)}
        analysis = analyze(peaks, laser_nm=532.0)
        assert "Pisana" in analysis.doping_note
        assert "2007" in analysis.doping_note

    def test_note_contains_shift_value(self):
        delta = 8.0
        peaks = {"G": _make_G(_G0_UNDOPED + delta)}
        analysis = analyze(peaks, laser_nm=532.0)
        # delta_G = +8.0 should appear in the note
        assert "+8" in analysis.doping_note or "8.0" in analysis.doping_note

    def test_note_contains_strain_warning(self):
        """Note must warn that strain can mimic doping."""
        peaks = {"G": _make_G(_G0_UNDOPED + 10.0)}
        analysis = analyze(peaks, laser_nm=532.0)
        assert "strain" in analysis.doping_note.lower()

    def test_note_undoped_contains_threshold(self):
        peaks = {"G": _make_G(_G0_UNDOPED)}
        analysis = analyze(peaks, laser_nm=532.0)
        assert "noise threshold" in analysis.doping_note.lower() or "undoped" in analysis.doping_note.lower()

    def test_note_empty_when_G_absent(self):
        analysis = analyze({}, laser_nm=532.0)
        assert analysis.doping_note == ""
        assert analysis.doping_type == "N/A"


class TestDopingEstimatorNoG:
    def test_na_when_no_peaks(self):
        analysis = analyze({}, laser_nm=532.0)
        assert analysis.doping_type == "N/A"
        assert np.isnan(analysis.carrier_density_cm2)

    def test_na_when_G_not_found(self):
        g = _make_G(_G0_UNDOPED + 10.0)
        g.found = False
        analysis = analyze({"G": g}, laser_nm=532.0)
        assert analysis.doping_type == "N/A"
        assert np.isnan(analysis.carrier_density_cm2)
