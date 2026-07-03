"""Tests for src/validation.py — post-fit quality-control warnings."""

import numpy as np
import pytest

from src.validation import validate, Severity, ValidationReport
from src.analyzer import RamanAnalysis
from src.peak_fitter import (
    PeakResult, _G_CENTER_MIN, _DP_CENTER_MIN,
)


def _peak(name, center, found=True, snr=10.0, fwhm=30.0,
          r2=0.99, amp=1.0, area=1.0):
    return PeakResult(
        name=name, center=center, amplitude=amp, fwhm=fwhm, area=area,
        r_squared=r2, snr=snr, found=found,
        model_x=np.array([]), model_y=np.array([]),
    )


def _good_peaks():
    return {
        "D": _peak("D", 1350.0),
        "G": _peak("G", 1585.0),
        "D_prime": _peak("D'", 1618.0),
    }


class TestCleanFit:
    def test_no_flags_on_good_fit(self):
        rep = validate(_good_peaks(), RamanAnalysis())
        assert isinstance(rep, ValidationReport)
        assert rep.ok
        assert not rep.has_critical

    def test_none_peaks_is_critical(self):
        rep = validate(None, RamanAnalysis())
        assert rep.has_critical


class TestCoreBands:
    def test_missing_G_is_critical(self):
        peaks = _good_peaks()
        peaks["G"] = _peak("G", 1585.0, found=False)
        rep = validate(peaks, RamanAnalysis())
        assert rep.has_critical
        assert any(f.code == "G_not_found" for f in rep.flags)

    def test_low_snr_D_warns(self):
        peaks = _good_peaks()
        peaks["D"] = _peak("D", 1350.0, snr=1.5)
        rep = validate(peaks, RamanAnalysis())
        assert any(f.code == "D_low_snr" and f.severity == Severity.WARNING
                   for f in rep.flags)


class TestCenterPinned:
    def test_G_pinned_to_lower_bound_warns(self):
        peaks = _good_peaks()
        peaks["G"] = _peak("G", _G_CENTER_MIN)  # exactly on the bound
        rep = validate(peaks, RamanAnalysis())
        assert any(f.code == "G_center_pinned" and f.severity == Severity.WARNING
                   for f in rep.flags)

    def test_G_not_pinned_when_centered(self):
        peaks = _good_peaks()
        peaks["G"] = _peak("G", 1580.0)  # comfortably inside
        rep = validate(peaks, RamanAnalysis())
        assert not any(f.code == "G_center_pinned" for f in rep.flags)


class TestFitQuality:
    def test_low_r2_warns(self):
        peaks = _good_peaks()
        peaks["D"] = _peak("D", 1350.0, r2=0.80)
        rep = validate(peaks, RamanAnalysis())
        assert any(f.code == "low_global_r2" for f in rep.flags)


class TestLDRegime:
    def test_LD_in_stage2_warns(self):
        an = RamanAnalysis()
        an.disorder_stage = "Stage 2 (amorphous/nanocrystalline carbon)"
        an.L_D_nm = 8.0
        rep = validate(_good_peaks(), an)
        assert any(f.code == "LD_out_of_regime" for f in rep.flags)

    def test_LD_in_stage1_ok(self):
        an = RamanAnalysis()
        an.disorder_stage = "Stage 1 (nanocrystalline graphite — L_D formula valid)"
        an.L_D_nm = 20.0
        rep = validate(_good_peaks(), an)
        assert not any(f.code == "LD_out_of_regime" for f in rep.flags)


class TestHeightAreaDivergence:
    def test_large_divergence_flags_info(self):
        an = RamanAnalysis()
        an.ID_IG_height = 1.0
        an.ID_IG_area = 2.5  # >50% apart
        rep = validate(_good_peaks(), an)
        assert any(f.code == "height_area_divergence"
                   and f.severity == Severity.INFO for f in rep.flags)

    def test_close_values_no_flag(self):
        an = RamanAnalysis()
        an.ID_IG_height = 1.0
        an.ID_IG_area = 1.1
        rep = validate(_good_peaks(), an)
        assert not any(f.code == "height_area_divergence" for f in rep.flags)
