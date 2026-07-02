"""
Tests for src/analyzer.py
Covers: ratio calculations, L_D formula, disorder stage,
        layer count, FWHM(2D) guard, Stage-2 L_D suppression.
"""
import math
import numpy as np
import pytest
from src.peak_fitter import fit_all_peaks, PeakResult
from src.analyzer import analyze, RamanAnalysis, _monolayer_threshold


# ── Monolayer threshold ───────────────────────────────────────────────────────

class TestMonolayerThreshold:
    @pytest.mark.parametrize("nm,expected", [
        (488, 2.5), (514, 2.5), (532, 2.0), (633, 1.5), (785, 0.8),
    ])
    def test_known_values(self, nm, expected):
        assert _monolayer_threshold(nm) == expected


# ── Ratio calculations ────────────────────────────────────────────────────────

class TestRatios:
    def _make_peaks(self, id_ig=0.3, i2d_ig=2.5, idp_ig=0.05):
        """Build minimal mock peaks dict."""
        def _peak(name, amp, fwhm=20):
            p = PeakResult(name=name)
            p.found = True
            p.amplitude = amp
            p.fwhm = fwhm
            p.area = math.pi * amp * (fwhm / 2)
            p.r_squared = 0.98
            p.snr = 10.0
            p.center = {"D": 1350, "G": 1582, "D_prime": 1622, "2D": 2690}[name]
            return p

        g_amp = 1.0
        peaks = {
            "D":       _peak("D",       g_amp * id_ig,  fwhm=30),
            "G":       _peak("G",       g_amp,          fwhm=16),
            "D_prime": _peak("D_prime", g_amp * idp_ig, fwhm=12),
            "2D":      _peak("2D",      g_amp * i2d_ig, fwhm=28),
        }
        return peaks

    def test_ID_IG_height(self):
        peaks = self._make_peaks(id_ig=0.4)
        result = analyze(peaks, laser_nm=532)
        assert abs(result.ID_IG_height - 0.4) < 0.01

    def test_I2D_IG_height(self):
        peaks = self._make_peaks(i2d_ig=2.1)
        result = analyze(peaks, laser_nm=532)
        assert abs(result.I2D_IG_height - 2.1) < 0.01

    def test_ID_IDp_height(self):
        peaks = self._make_peaks(id_ig=0.5, idp_ig=0.1)
        result = analyze(peaks, laser_nm=532)
        assert abs(result.ID_IDp_height - 5.0) < 0.1

    def test_no_graphitization_pct(self):
        result = RamanAnalysis()
        assert not hasattr(result, "graphitization_pct")


# ── L_D formula ───────────────────────────────────────────────────────────────

class TestLD:
    def _stage1_peaks(self, id_ig=0.3):
        """Stage 1 sample (FWHM_G < 50) with given ID/IG."""
        p_D = PeakResult(name="D"); p_D.found=True; p_D.amplitude=id_ig
        p_D.fwhm=25; p_D.area=math.pi*id_ig*12.5; p_D.r_squared=0.95; p_D.snr=8; p_D.center=1350
        p_G = PeakResult(name="G"); p_G.found=True; p_G.amplitude=1.0
        p_G.fwhm=16; p_G.area=math.pi*1.0*8; p_G.r_squared=0.97; p_G.snr=12; p_G.center=1582
        return {"D": p_D, "G": p_G}

    def test_LD_formula_532nm(self):
        """L_D² = 1.8e-9 × λ⁴ / (ID/IG)."""
        peaks = self._stage1_peaks(id_ig=0.3)
        result = analyze(peaks, laser_nm=532)
        expected = math.sqrt(1.8e-9 * 532**4 / 0.3)
        assert abs(result.L_D_nm - expected) / expected < 0.01

    def test_LD_increases_lower_defects(self):
        """Lower ID/IG → larger L_D."""
        r_high = analyze(self._stage1_peaks(id_ig=0.5), laser_nm=532)
        r_low  = analyze(self._stage1_peaks(id_ig=0.1), laser_nm=532)
        assert r_low.L_D_nm > r_high.L_D_nm

    def test_LD_suppressed_stage2(self):
        """Stage 2 sample (FWHM_G > 80): L_D must be NaN."""
        p_D = PeakResult(name="D"); p_D.found=True; p_D.amplitude=0.5
        p_D.fwhm=60; p_D.area=1.0; p_D.r_squared=0.95; p_D.snr=8; p_D.center=1350
        p_G = PeakResult(name="G"); p_G.found=True; p_G.amplitude=1.0
        p_G.fwhm=90; p_G.area=2.0; p_G.r_squared=0.97; p_G.snr=12; p_G.center=1582
        result = analyze({"D": p_D, "G": p_G}, laser_nm=532)
        assert np.isnan(result.L_D_nm)
        assert "Stage 2" in result.L_D_note

    def test_LD_note_contains_cancado(self):
        peaks = self._stage1_peaks(id_ig=0.3)
        result = analyze(peaks, laser_nm=532)
        assert "ado" in result.L_D_note   # Cançado


# ── Layer count & FWHM guard ──────────────────────────────────────────────────

class TestLayerCount:
    def _peaks_with_2D(self, i2d_ig, fwhm_2d=26):
        p_G = PeakResult(name="G"); p_G.found=True; p_G.amplitude=1.0
        p_G.fwhm=16; p_G.area=2.0; p_G.r_squared=0.97; p_G.snr=12; p_G.center=1582
        p_2D = PeakResult(name="2D"); p_2D.found=True; p_2D.amplitude=i2d_ig
        p_2D.fwhm=fwhm_2d; p_2D.area=math.pi*i2d_ig*14; p_2D.r_squared=0.96; p_2D.snr=10; p_2D.center=2690
        return {"G": p_G, "2D": p_2D}

    def test_monolayer_detected_532nm(self):
        result = analyze(self._peaks_with_2D(i2d_ig=2.5), laser_nm=532)
        assert "Monolayer" in result.estimated_layers

    def test_multilayer_detected(self):
        result = analyze(self._peaks_with_2D(i2d_ig=0.3), laser_nm=532)
        assert "Multilayer" in result.estimated_layers or "Few-layer" in result.estimated_layers

    def test_fwhm_guard_triggered(self):
        """FWHM(2D) = 40 cm⁻¹ > 35 → twoD_fwhm_warning = True."""
        result = analyze(self._peaks_with_2D(i2d_ig=2.5, fwhm_2d=40), laser_nm=532)
        assert result.twoD_fwhm_warning is True
        assert "WARNING" in result.estimated_layers

    def test_fwhm_guard_not_triggered_narrow(self):
        result = analyze(self._peaks_with_2D(i2d_ig=2.5, fwhm_2d=26), laser_nm=532)
        assert result.twoD_fwhm_warning is False

    def test_monolayer_threshold_laser_dependent(self):
        """Same spectrum: monolayer at 532 nm might not be at 785 nm."""
        peaks = self._peaks_with_2D(i2d_ig=1.0)
        r532 = analyze(peaks, laser_nm=532)
        r785 = analyze(peaks, laser_nm=785)
        # At 785 nm threshold is 0.8, so 1.0 should still be monolayer
        assert "Monolayer" in r785.estimated_layers


# ── Full pipeline integration ─────────────────────────────────────────────────

class TestIntegration:
    def test_clean_graphene_monolayer(self, wavenumbers, graphene_spectrum):
        peaks  = fit_all_peaks(wavenumbers, graphene_spectrum, laser_nm=532)
        result = analyze(peaks, laser_nm=532)
        assert result.G_found
        assert result.twoD_found
        assert not np.isnan(result.I2D_IG_height)
        assert result.I2D_IG_height > 1.0
        assert "Monolayer" in result.estimated_layers

    def test_defective_graphene_stage1(self, wavenumbers, defective_spectrum):
        peaks  = fit_all_peaks(wavenumbers, defective_spectrum, laser_nm=532)
        result = analyze(peaks, laser_nm=532)
        assert result.G_found
        assert result.D_found
        assert not np.isnan(result.ID_IG_height)
        assert "Stage 1" in result.disorder_stage
