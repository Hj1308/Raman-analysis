"""
Unit tests for D* band (v2.4 Feature #1).

Checks:
  - fit_all_peaks detects a synthetic D* peak (found=True)
  - I_D*/I_G ratio is correctly computed by analyzer.analyze
  - dstar_co_note contains the C-O warning when I_D*/I_G > 0.15
  - dstar_co_note reports 'low oxidation' when I_D*/I_G <= 0.15

Reference: Lee et al. (2021) Carbon 183, 814-822
"""
import numpy as np
import pytest
from src.peak_fitter import fit_all_peaks
from src.analyzer import analyze


def _lorentzian(x, center, amp, gamma):
    return amp / (1.0 + ((x - center) / gamma) ** 2)


@pytest.fixture
def wn_wide():
    """Wavenumber axis 1000-3000 cm^-1, 1 cm^-1 step."""
    return np.arange(1000.0, 3000.0, 1.0)


@pytest.fixture
def spectrum_with_strong_dstar(wn_wide):
    """
    Synthetic spectrum with:
      G  @ 1580 cm^-1, amp = 100
      D  @ 1350 cm^-1, amp =  30  (needed for global fit)
      D* @ 1150 cm^-1, amp =  25  -> I_D*/I_G ~ 0.25 (> 0.15 threshold)
    Plus small Gaussian noise (seed fixed for reproducibility).
    """
    y = np.zeros_like(wn_wide)
    y += _lorentzian(wn_wide, 1580.0, 100.0, 12.0)  # G
    y += _lorentzian(wn_wide, 1350.0,  30.0, 25.0)  # D
    y += _lorentzian(wn_wide, 1150.0,  25.0, 15.0)  # D* (strong)
    rng = np.random.default_rng(42)
    y += rng.normal(scale=0.4, size=wn_wide.shape)
    return np.clip(y, 0, None)


@pytest.fixture
def spectrum_with_weak_dstar(wn_wide):
    """
    Synthetic spectrum with:
      G  @ 1580 cm^-1, amp = 100
      D* @ 1150 cm^-1, amp =  10  -> I_D*/I_G ~ 0.10 (<= 0.15 threshold)
    """
    y = np.zeros_like(wn_wide)
    y += _lorentzian(wn_wide, 1580.0, 100.0, 12.0)  # G
    y += _lorentzian(wn_wide, 1350.0,  30.0, 25.0)  # D
    y += _lorentzian(wn_wide, 1150.0,  10.0, 15.0)  # D* (weak)
    rng = np.random.default_rng(7)
    y += rng.normal(scale=0.4, size=wn_wide.shape)
    return np.clip(y, 0, None)


class TestDstarDetection:
    def test_dstar_peak_detected(self, wn_wide, spectrum_with_strong_dstar):
        """fit_all_peaks must return D_star with found=True."""
        peaks = fit_all_peaks(wn_wide, spectrum_with_strong_dstar, laser_nm=532.0)
        dstar = peaks.get("D_star")
        assert dstar is not None, "D_star key must exist in peaks dict"
        assert dstar.found, "D* peak should be detected for strong signal"

    def test_dstar_center_in_expected_window(self, wn_wide, spectrum_with_strong_dstar):
        """Detected D* center must fall inside the 1080-1230 cm^-1 window."""
        peaks = fit_all_peaks(wn_wide, spectrum_with_strong_dstar, laser_nm=532.0)
        dstar = peaks.get("D_star")
        assert dstar is not None and dstar.found
        assert 1080.0 <= dstar.center <= 1230.0, (
            "D* center {:.1f} cm^-1 outside window 1080-1230 cm^-1".format(dstar.center)
        )

    def test_dstar_r_squared_above_threshold(self, wn_wide, spectrum_with_strong_dstar):
        """R^2 for D* fit must meet the SNR gate threshold."""
        peaks = fit_all_peaks(wn_wide, spectrum_with_strong_dstar, laser_nm=532.0)
        dstar = peaks.get("D_star")
        assert dstar is not None and dstar.found
        assert dstar.r_squared >= 0.75

    def test_dstar_uncertainty_populated(self, wn_wide, spectrum_with_strong_dstar):
        """center_stderr and fwhm_stderr should be finite when set."""
        peaks = fit_all_peaks(wn_wide, spectrum_with_strong_dstar, laser_nm=532.0)
        dstar = peaks.get("D_star")
        assert dstar is not None and dstar.found
        if dstar.center_stderr is not None:
            assert np.isfinite(dstar.center_stderr)
        if dstar.fwhm_stderr is not None:
            assert np.isfinite(dstar.fwhm_stderr)


class TestDstarRatio:
    def test_IDstar_IG_ratio_strong_signal(self, wn_wide, spectrum_with_strong_dstar):
        """I_D*/I_G should be ~0.25, wide tolerance +-0.15 to allow for noise."""
        peaks = fit_all_peaks(wn_wide, spectrum_with_strong_dstar, laser_nm=532.0)
        analysis = analyze(peaks, laser_nm=532.0)
        assert not np.isnan(analysis.IDstar_IG_height), "IDstar_IG_height should not be NaN"
        assert 0.10 < analysis.IDstar_IG_height < 0.45, (
            "Expected I_D*/I_G ~ 0.25, got {:.3f}".format(analysis.IDstar_IG_height)
        )

    def test_dstar_co_note_high_oxidation(self, wn_wide, spectrum_with_strong_dstar):
        """dstar_co_note must warn about C-O groups when I_D*/I_G > 0.15."""
        peaks = fit_all_peaks(wn_wide, spectrum_with_strong_dstar, laser_nm=532.0)
        analysis = analyze(peaks, laser_nm=532.0)
        assert analysis.IDstar_IG_height > 0.15
        assert "High D*" in analysis.dstar_co_note
        assert "C" in analysis.dstar_co_note
        assert "Lee et al." in analysis.dstar_co_note

    def test_dstar_co_note_low_oxidation(self, wn_wide, spectrum_with_weak_dstar):
        """For weak D*, the note should indicate low oxidation (<= 0.15)."""
        peaks = fit_all_peaks(wn_wide, spectrum_with_weak_dstar, laser_nm=532.0)
        analysis = analyze(peaks, laser_nm=532.0)
        if not np.isnan(analysis.IDstar_IG_height) and analysis.IDstar_IG_height <= 0.15:
            assert "low oxidation" in analysis.dstar_co_note.lower()

    def test_IDstar_IG_nan_when_G_absent(self):
        """No G band -> IDstar_IG_height must remain NaN."""
        from src.peak_fitter import PeakResult
        dstar = PeakResult(
            name="D*", center=1150.0, amplitude=10.0, fwhm=20.0,
            area=100.0, r_squared=0.90, snr=8.0, found=True,
        )
        analysis = analyze({"D_star": dstar}, laser_nm=532.0)
        assert np.isnan(analysis.IDstar_IG_height)


class TestDstarDispersion:
    @pytest.mark.parametrize("laser_nm,expected_shift_sign", [
        (532.0,  0.0),
        (633.0, -1.0),
        (785.0, -1.0),
        (488.0, +1.0),
    ])
    def test_dstar_window_shifts_with_laser(self, laser_nm, expected_shift_sign):
        """D* window must shift in correct direction relative to 532 nm reference."""
        from src.peak_fitter import get_peak_windows, PEAK_WINDOWS_532
        ref_lo, _ = PEAK_WINDOWS_532["D_star"]
        windows = get_peak_windows(laser_nm)
        lo, _ = windows["D_star"]
        shift = lo - ref_lo
        if expected_shift_sign > 0:
            assert shift > 0
        elif expected_shift_sign < 0:
            assert shift < 0
        else:
            assert abs(shift) < 1e-9
