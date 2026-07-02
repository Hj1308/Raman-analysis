"""
Unit tests for Stage boundary refinement (v2.5 Feature #5).

Reference: Wu et al. (2018) Carbon 127, 418-428

Criteria implemented in analyzer._refine_stage:
  Stage 2:              FWHM(G) >= 80 cm^-1
                        OR FWHM(G) >= 50 AND A_D/A_G >= 3.0
  Stage 1->2 transition: 50 <= FWHM(G) < 80
                         OR A_D/A_G >= 2.5
  Stage 1:              FWHM(G) < 50 AND A_D/A_G < 2.5

Test groups:
  A. Stage 1 positive cases
  B. Stage 2 positive cases
  C. Stage 1->2 transition cases
  D. Boundary / edge cases
  E. Note content checks
  F. No G band
  G. Integration with analyze()
"""
import math
import numpy as np
import pytest
from src.peak_fitter import PeakResult
from src.analyzer import analyze, _refine_stage


# ────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────

def _make_G(center=1582.0, fwhm=16.0, amplitude=10.0, area=None):
    if area is None:
        area = math.pi * amplitude * (fwhm / 2.0)
    return PeakResult(
        name="G",
        center=center,
        amplitude=amplitude,
        fwhm=fwhm,
        area=area,
        r_squared=0.99,
        snr=50.0,
        found=True,
    )


def _make_D(amplitude=5.0, fwhm=30.0, area=None):
    if area is None:
        area = math.pi * amplitude * (fwhm / 2.0)
    return PeakResult(
        name="D",
        center=1350.0,
        amplitude=amplitude,
        fwhm=fwhm,
        area=area,
        r_squared=0.97,
        snr=20.0,
        found=True,
    )


def _ad_ag_peaks(ad_ag_ratio, g_fwhm=20.0, g_amp=10.0):
    """Build G and D with a specific A_D/A_G ratio."""
    g_area = math.pi * g_amp * (g_fwhm / 2.0)
    d_area = ad_ag_ratio * g_area
    # back-calculate D amplitude from area (area = pi * amp * gamma, gamma = fwhm/2)
    d_fwhm = 30.0
    d_amp  = d_area / (math.pi * (d_fwhm / 2.0))
    G = _make_G(fwhm=g_fwhm, amplitude=g_amp, area=g_area)
    D = _make_D(amplitude=d_amp, fwhm=d_fwhm, area=d_area)
    return G, D


# ────────────────────────────────────────────────
# A. Stage 1 — should pass all criteria
# ────────────────────────────────────────────────

class TestStage1:
    def test_clean_graphene_stage1(self):
        """FWHM(G)=16, A_D/A_G small: must be Stage 1."""
        G, D = _ad_ag_peaks(0.3, g_fwhm=16.0)
        label, note = _refine_stage(G, D)
        assert "Stage 1" in label
        assert "Stage 2" not in label
        assert "transition" not in label.lower()

    def test_stage1_no_D_band(self):
        """No D band: Stage 1 if FWHM(G) < 50."""
        G = _make_G(fwhm=20.0)
        label, note = _refine_stage(G, None)
        assert "Stage 1" in label

    @pytest.mark.parametrize("fwhm_g", [10.0, 20.0, 35.0, 49.9])
    def test_stage1_across_narrow_fwhm_range(self, fwhm_g):
        G, D = _ad_ag_peaks(0.5, g_fwhm=fwhm_g)
        label, _ = _refine_stage(G, D)
        assert "Stage 1" in label
        assert "Stage 2" not in label

    def test_stage1_note_contains_wu2018(self):
        G, D = _ad_ag_peaks(0.3, g_fwhm=16.0)
        _, note = _refine_stage(G, D)
        assert "Wu 2018" in note or "Wu" in note

    def test_stage1_note_contains_fwhm_value(self):
        G, D = _ad_ag_peaks(0.3, g_fwhm=25.0)
        _, note = _refine_stage(G, D)
        assert "25.0" in note or "25" in note

    def test_stage1_note_contains_ad_ag_value(self):
        G, D = _ad_ag_peaks(1.2, g_fwhm=20.0)
        _, note = _refine_stage(G, D)
        assert "A_D/A_G" in note


# ────────────────────────────────────────────────
# B. Stage 2 — two independent triggers
# ────────────────────────────────────────────────

class TestStage2:
    def test_stage2_fwhm_trigger(self):
        """FWHM(G) >= 80: must be Stage 2 regardless of A_D/A_G."""
        G, D = _ad_ag_peaks(0.5, g_fwhm=85.0)
        label, _ = _refine_stage(G, D)
        assert "Stage 2" in label

    def test_stage2_area_trigger(self):
        """FWHM(G) = 55 (>= 50) and A_D/A_G = 3.5 (>= 3.0): Stage 2."""
        G, D = _ad_ag_peaks(3.5, g_fwhm=55.0)
        label, _ = _refine_stage(G, D)
        assert "Stage 2" in label

    def test_stage2_fwhm_exactly_80(self):
        """Boundary: FWHM(G) = 80 exactly -> Stage 2."""
        G, D = _ad_ag_peaks(0.3, g_fwhm=80.0)
        label, _ = _refine_stage(G, D)
        assert "Stage 2" in label

    @pytest.mark.parametrize("fwhm_g", [80.0, 90.0, 120.0, 200.0])
    def test_stage2_for_large_fwhm(self, fwhm_g):
        G, D = _ad_ag_peaks(0.5, g_fwhm=fwhm_g)
        label, _ = _refine_stage(G, D)
        assert "Stage 2" in label

    def test_stage2_area_no_D_does_not_trigger(self):
        """A_D/A_G cannot be computed without D -> area criterion not met."""
        G = _make_G(fwhm=55.0)   # fwhm >= 50 but no D band
        label, _ = _refine_stage(G, None)
        # Without D, area criterion is N/A -> must be transition or Stage 1
        # (fwhm=55 in 50-80 range: transition expected)
        assert "Stage 2" not in label or "transition" in label.lower()

    def test_stage2_flag_propagates_to_LD(self):
        """Stage 2 -> L_D must be NaN (Cancado formula not valid)."""
        G, D = _ad_ag_peaks(0.5, g_fwhm=90.0)
        peaks = {"G": G, "D": D}
        analysis = analyze(peaks, laser_nm=532.0)
        assert "Stage 2" in analysis.stage_refined
        assert np.isnan(analysis.L_D_nm)


# ────────────────────────────────────────────────
# C. Stage 1->2 transition
# ────────────────────────────────────────────────

class TestStageTransition:
    def test_transition_fwhm_trigger(self):
        """50 <= FWHM(G) < 80 with low A_D/A_G: transition."""
        G, D = _ad_ag_peaks(0.5, g_fwhm=60.0)
        label, _ = _refine_stage(G, D)
        assert "transition" in label.lower() or "Stage 2" in label

    def test_transition_area_trigger(self):
        """A_D/A_G = 2.7 (>= 2.5) with FWHM(G) = 30: transition."""
        G, D = _ad_ag_peaks(2.7, g_fwhm=30.0)
        label, _ = _refine_stage(G, D)
        assert "transition" in label.lower() or "Stage 2" in label

    @pytest.mark.parametrize("fwhm_g", [50.0, 60.0, 70.0, 79.9])
    def test_transition_range(self, fwhm_g):
        """50 <= FWHM(G) < 80 must not be Stage 1."""
        G, D = _ad_ag_peaks(0.5, g_fwhm=fwhm_g)
        label, _ = _refine_stage(G, D)
        assert "Stage 1" not in label or "transition" in label.lower()

    def test_transition_fwhm_exactly_50(self):
        """Boundary: FWHM(G) = 50 exactly -> transition (not Stage 1)."""
        G, D = _ad_ag_peaks(0.5, g_fwhm=50.0)
        label, _ = _refine_stage(G, D)
        assert "transition" in label.lower() or "Stage 2" in label


# ────────────────────────────────────────────────
# D. Boundary / edge values
# ────────────────────────────────────────────────

class TestStageBoundaries:
    def test_boundary_fwhm_just_below_50(self):
        """FWHM(G) = 49.9: Stage 1."""
        G, D = _ad_ag_peaks(0.5, g_fwhm=49.9)
        label, _ = _refine_stage(G, D)
        assert "Stage 1" in label
        assert "transition" not in label.lower()

    def test_boundary_fwhm_just_above_80(self):
        """FWHM(G) = 80.1: Stage 2."""
        G, D = _ad_ag_peaks(0.5, g_fwhm=80.1)
        label, _ = _refine_stage(G, D)
        assert "Stage 2" in label

    def test_boundary_ad_ag_exactly_2p5(self):
        """A_D/A_G = 2.5: transition."""
        G, D = _ad_ag_peaks(2.5, g_fwhm=20.0)
        label, _ = _refine_stage(G, D)
        assert "transition" in label.lower() or "Stage 2" in label

    def test_boundary_ad_ag_exactly_3p0_fwhm50(self):
        """A_D/A_G = 3.0 and FWHM(G) = 50: Stage 2 (area trigger)."""
        G, D = _ad_ag_peaks(3.0, g_fwhm=50.0)
        label, _ = _refine_stage(G, D)
        assert "Stage 2" in label

    def test_boundary_ad_ag_just_below_2p5(self):
        """A_D/A_G = 2.49 with FWHM(G) = 20: Stage 1."""
        G, D = _ad_ag_peaks(2.49, g_fwhm=20.0)
        label, _ = _refine_stage(G, D)
        assert "Stage 1" in label
        assert "transition" not in label.lower()

    def test_g_area_zero_does_not_crash(self):
        """G.area = 0 -> A_D/A_G division guarded; must not raise."""
        G = _make_G(fwhm=20.0, area=0.0)
        D = _make_D(area=10.0)
        label, note = _refine_stage(G, D)   # must not raise
        assert isinstance(label, str)
        assert isinstance(note, str)


# ────────────────────────────────────────────────
# E. Note content
# ────────────────────────────────────────────────

class TestStageNoteContent:
    def test_note_always_contains_fwhm_g(self):
        for fwhm in [16.0, 55.0, 90.0]:
            G = _make_G(fwhm=fwhm)
            _, note = _refine_stage(G, None)
            assert "FWHM" in note, "FWHM not in note for fwhm={}".format(fwhm)

    def test_note_contains_ad_ag_when_D_present(self):
        G, D = _ad_ag_peaks(1.2, g_fwhm=20.0)
        _, note = _refine_stage(G, D)
        assert "A_D/A_G" in note

    def test_note_na_ad_ag_when_D_absent(self):
        G = _make_G(fwhm=20.0)
        _, note = _refine_stage(G, None)
        assert "N/A" in note

    def test_note_not_found_when_D_not_found(self):
        G = _make_G(fwhm=20.0)
        D = _make_D()
        D.found = False
        _, note = _refine_stage(G, D)
        assert "N/A" in note


# ────────────────────────────────────────────────
# F. No G band
# ────────────────────────────────────────────────

class TestNoGBand:
    def test_na_when_G_none(self):
        label, note = _refine_stage(None, None)
        assert label == "N/A"

    def test_na_when_G_not_found(self):
        G = _make_G()
        G.found = False
        label, note = _refine_stage(G, None)
        assert label == "N/A"

    def test_analyze_returns_na_stage_refined_without_G(self):
        analysis = analyze({}, laser_nm=532.0)
        assert analysis.stage_refined == "N/A"


# ────────────────────────────────────────────────
# G. Integration with analyze()
# ────────────────────────────────────────────────

class TestStageRefinedIntegration:
    def test_stage_refined_field_exists(self):
        G, D = _ad_ag_peaks(0.3, g_fwhm=16.0)
        analysis = analyze({"G": G, "D": D}, laser_nm=532.0)
        assert hasattr(analysis, "stage_refined")
        assert hasattr(analysis, "stage_refined_note")

    def test_stage_refined_stage1_clean_graphene(self):
        G, D = _ad_ag_peaks(0.3, g_fwhm=16.0)
        analysis = analyze({"G": G, "D": D}, laser_nm=532.0)
        assert "Stage 1" in analysis.stage_refined
        assert not np.isnan(analysis.L_D_nm)   # L_D valid in Stage 1

    def test_stage_refined_stage2_suppresses_LD(self):
        """Stage 2 from _refine_stage must propagate to L_D = NaN."""
        G, D = _ad_ag_peaks(4.0, g_fwhm=90.0)
        analysis = analyze({"G": G, "D": D}, laser_nm=532.0)
        assert "Stage 2" in analysis.stage_refined
        assert np.isnan(analysis.L_D_nm)
        assert "Stage 2" in analysis.L_D_note or "suppressed" in analysis.L_D_note.lower()

    def test_stage_refined_note_in_analysis(self):
        G, D = _ad_ag_peaks(0.3, g_fwhm=16.0)
        analysis = analyze({"G": G, "D": D}, laser_nm=532.0)
        assert isinstance(analysis.stage_refined_note, str)
        assert len(analysis.stage_refined_note) > 0

    def test_stage_refined_consistent_with_disorder_stage(self):
        """
        When both metrics agree it's Stage 1, the fields should not contradict.
        (disorder_stage is the legacy Ferrari 2001 field; stage_refined is Wu 2018.)
        """
        G, D = _ad_ag_peaks(0.3, g_fwhm=16.0)
        analysis = analyze({"G": G, "D": D}, laser_nm=532.0)
        if "Stage 1" in analysis.stage_refined:
            assert "Stage 2" not in analysis.disorder_stage

    def test_stage2_from_transition_flag_also_suppresses_LD(self):
        """Transition stage -> L_D may or may not be NaN depending on severity;
        ensure it is NaN when stage_refined says Stage 2."""
        G, D = _ad_ag_peaks(3.5, g_fwhm=55.0)  # area trigger -> Stage 2
        analysis = analyze({"G": G, "D": D}, laser_nm=532.0)
        if "Stage 2" in analysis.stage_refined:
            assert np.isnan(analysis.L_D_nm)
