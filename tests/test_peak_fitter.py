"""
Tests for src/peak_fitter.py
Covers: get_peak_windows, fit_all_peaks, SNR gate, adaptive G,
        G+D' deconvolution, pseudo-Voigt, PeakResult fields.
"""
import numpy as np
import pytest
from src.peak_fitter import (
    fit_all_peaks,
    get_peak_windows,
    PeakResult,
    _find_G_peak,
    _r2,
    _noise_sigma,
    _is_detected,
)


# ── get_peak_windows ──────────────────────────────────────────────────────────

class TestGetPeakWindows:
    def test_532nm_reference_unchanged(self):
        """At 532 nm the windows must equal PEAK_WINDOWS_532 exactly."""
        from src.peak_fitter import PEAK_WINDOWS_532
        w = get_peak_windows(532.0)
        for peak, (lo, hi) in PEAK_WINDOWS_532.items():
            assert abs(w[peak][0] - lo) < 0.01
            assert abs(w[peak][1] - hi) < 0.01

    def test_D_window_shifts_lower_at_633nm(self):
        """D band is dispersive: higher λ → lower wavenumber."""
        w532 = get_peak_windows(532.0)
        w633 = get_peak_windows(633.0)
        assert w633["D"][0] < w532["D"][0]
        assert w633["D"][1] < w532["D"][1]

    def test_2D_window_shifts_lower_at_785nm(self):
        w532 = get_peak_windows(532.0)
        w785 = get_peak_windows(785.0)
        assert w785["2D"][0] < w532["2D"][0]

    def test_G_window_non_dispersive(self):
        """G band is non-dispersive — window must not shift."""
        w532 = get_peak_windows(532.0)
        w785 = get_peak_windows(785.0)
        assert abs(w532["G"][0] - w785["G"][0]) < 0.01
        assert abs(w532["G"][1] - w785["G"][1]) < 0.01

    def test_invalid_laser_raises(self):
        with pytest.raises(ValueError):
            get_peak_windows(-1.0)


# ── Helper functions ──────────────────────────────────────────────────────────

class TestHelpers:
    def test_r2_perfect_fit(self):
        x = np.linspace(0, 10, 100)
        assert abs(_r2(x, x) - 1.0) < 1e-10

    def test_r2_zero_for_mean(self):
        y = np.linspace(1, 5, 50)
        assert abs(_r2(y, np.full_like(y, y.mean()))) < 1e-10

    def test_noise_sigma_gaussian(self):
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 1.0, 5000)
        sigma = _noise_sigma(noise, np.zeros_like(noise))
        assert 0.85 < sigma < 1.15

    def test_is_detected_both_gates(self):
        """
        _is_detected needs a realistic signal above a noisy background.
        A flat array has zero noise-sigma (MAD=0), so SNR=inf or undefined.
        Use a noisy background with a clear peak to get a finite SNR > 3.
        """
        rng = np.random.default_rng(42)
        # Background noise sigma ≈ 1.0
        y_obs = rng.normal(0.0, 1.0, 200)
        # Perfect Lorentzian fit with amplitude 20 (SNR ≈ 20 >> 3)
        x = np.linspace(-5, 5, 200)
        y_fit = 20.0 / (1.0 + x**2)
        det, snr = _is_detected(20.0, y_obs, y_fit, r2=0.95)
        assert det is True
        assert snr > 3.0

    def test_is_detected_fails_low_r2(self):
        y_obs = np.ones(50)
        y_fit = np.ones(50)
        det, snr = _is_detected(10.0, y_obs, y_fit, r2=0.50)
        assert det is False


# ── _find_G_peak ──────────────────────────────────────────────────────────────

class TestFindGPeak:
    def test_finds_standard_G(self, wavenumbers, graphene_spectrum):
        centre = _find_G_peak(wavenumbers, graphene_spectrum)
        assert 1560 < centre < 1610

    def test_finds_shifted_G_doped(self, wavenumbers, doped_spectrum):
        """Doped sample: G near 1598 — must still be found."""
        centre = _find_G_peak(wavenumbers, doped_spectrum)
        assert 1570 < centre < 1630


# ── fit_all_peaks — clean graphene ────────────────────────────────────────────

class TestFitCleanGraphene:
    @pytest.fixture(autouse=True)
    def _fit(self, wavenumbers, graphene_spectrum):
        self.peaks = fit_all_peaks(wavenumbers, graphene_spectrum, laser_nm=532)

    def test_G_found(self):
        assert self.peaks["G"].found

    def test_G_center_accuracy(self):
        assert abs(self.peaks["G"].center - 1582) < 5

    def test_G_r2_high(self):
        assert self.peaks["G"].r_squared > 0.90

    def test_D_found(self):
        assert self.peaks["D"].found

    def test_D_center_accuracy(self):
        assert abs(self.peaks["D"].center - 1350) < 10

    def test_2D_found(self):
        assert self.peaks["2D"].found

    def test_2D_center_accuracy(self):
        assert abs(self.peaks["2D"].center - 2690) < 15

    def test_peak_result_has_snr(self):
        assert not np.isnan(self.peaks["G"].snr)
        assert self.peaks["G"].snr > 3.0

    def test_fwhm_positive(self):
        for name in ["D", "G", "2D"]:
            p = self.peaks[name]
            if p.found:
                assert p.fwhm > 0, f"{name}.fwhm should be positive"

    def test_area_positive(self):
        for name in ["D", "G", "2D"]:
            p = self.peaks[name]
            if p.found:
                assert p.area > 0, f"{name}.area should be positive"

    def test_no_graphitization_pct(self):
        """Removed in v2.2 — accessing it must raise AttributeError."""
        from src.analyzer import RamanAnalysis
        ra = RamanAnalysis()
        assert not hasattr(ra, "graphitization_pct")


# ── SNR gate ──────────────────────────────────────────────────────────────────

class TestSNRGate:
    def test_noisy_weak_G_rejected(self, wavenumbers, noisy_spectrum):
        """Very noisy spectrum: weak G should be rejected by SNR gate."""
        peaks = fit_all_peaks(wavenumbers, noisy_spectrum, laser_nm=532)
        g = peaks["G"]
        if g.found:
            assert g.snr >= 3.0


# ── Adaptive G & doped samples ────────────────────────────────────────────────

class TestAdaptiveG:
    def test_doped_G_found(self, wavenumbers, doped_spectrum):
        peaks = fit_all_peaks(wavenumbers, doped_spectrum, laser_nm=532)
        assert peaks["G"].found

    def test_doped_G_center_near_1598(self, wavenumbers, doped_spectrum):
        peaks = fit_all_peaks(wavenumbers, doped_spectrum, laser_nm=532)
        assert abs(peaks["G"].center - 1598) < 15


# ── Pseudo-Voigt (D+G band) ───────────────────────────────────────────────────

class TestPseudoVoigt:
    def test_DG_area_positive(self, wavenumbers, graphene_spectrum):
        peaks = fit_all_peaks(wavenumbers, graphene_spectrum, laser_nm=532)
        dg = peaks.get("DG")
        if dg and dg.found:
            assert dg.area > 0

    def test_DG_fwhm_reasonable(self, wavenumbers, graphene_spectrum):
        """FWHM for D+G should be between 5 and 200 cm⁻¹."""
        peaks = fit_all_peaks(wavenumbers, graphene_spectrum, laser_nm=532)
        dg = peaks.get("DG")
        if dg and dg.found:
            assert 5 < dg.fwhm < 200


# ── PeakResult dataclass ──────────────────────────────────────────────────────

class TestPeakResult:
    def test_default_not_found(self):
        p = PeakResult(name="X")
        assert p.found is False
        assert np.isnan(p.center)
        assert np.isnan(p.fwhm)

    def test_is_deconvolved_false_by_default(self):
        p = PeakResult(name="G")
        assert p.is_deconvolved is False
        assert p.deconv_partner is None

    def test_center_stderr_none_by_default(self):
        p = PeakResult(name="G")
        assert p.center_stderr is None
        assert p.fwhm_stderr is None

    def test_twoD_fwhm_warning_field_exists(self):
        """twoD_fwhm_warning is on RamanAnalysis, not PeakResult."""
        from src.analyzer import RamanAnalysis
        ra = RamanAnalysis()
        assert ra.twoD_fwhm_warning is False
