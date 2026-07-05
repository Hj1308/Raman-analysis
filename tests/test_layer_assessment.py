"""Tests for assess_layer_number (Fix #4) — three-tier FWHM(2D) logic with
a noise-robust symmetry test (single- vs double-Lorentzian model comparison).

The critical regression here: a fixed R² threshold used to misclassify a
NOISY but genuinely symmetric monolayer 2D band as "asymmetric/few-layer".
The model-comparison test must keep these two cases separated.
"""

import numpy as np
import pytest

from src.analyzer import assess_layer_number


def _lor(x, c, f, a):
    return a * (f / 2) ** 2 / ((x - c) ** 2 + (f / 2) ** 2)


@pytest.fixture
def axis():
    return np.linspace(2550, 2850, 600)


class TestTiers:
    def test_narrow_is_monolayer(self, axis):
        y = _lor(axis, 2680, 25, 1.0)
        v, c, d = assess_layer_number(axis, y, 2680, 25.0)
        assert "monolayer" in v
        assert c == "high"

    def test_broad_is_multilayer(self, axis):
        y = _lor(axis, 2695, 55, 0.8)
        v, c, d = assess_layer_number(axis, y, 2695, 55.0)
        assert "multilayer" in v or "turbostratic" in v
        assert c == "high"

    def test_clean_borderline_confirmed_high(self, axis):
        rng = np.random.default_rng(1)
        y = _lor(axis, 2680, 35, 1.0) + rng.normal(0, 0.003, axis.size)
        v, c, d = assess_layer_number(axis, y, 2680, 35.0)
        assert "monolayer" in v
        assert c == "high"


class TestNoiseVsAsymmetry:
    """The regression that motivated the model-comparison redesign."""

    def test_noisy_true_monolayer_not_called_asymmetric(self, axis):
        rng = np.random.default_rng(7)
        y = _lor(axis, 2680, 35, 1.0) + rng.normal(0, 0.03, axis.size)  # SNR~30
        v, c, d = assess_layer_number(axis, y, 2680, 35.0)
        assert "monolayer" in v          # NOT "asymmetric ... few-layer"
        assert "noise" in d.lower()

    def test_true_ab_bilayer_flagged_asymmetric(self, axis):
        rng = np.random.default_rng(7)
        # four 2D components of AB bilayer (Ferrari 2006 relative positions)
        y = (_lor(axis, 2636, 28, 0.15) + _lor(axis, 2670, 28, 0.60)
             + _lor(axis, 2690, 28, 0.55) + _lor(axis, 2705, 28, 0.25))
        y += rng.normal(0, 0.005, axis.size)
        v, c, d = assess_layer_number(axis, y, 2682, 40.0)
        assert "asymmetric" in v
        assert "few-layer" in v


class TestEdgeCases:
    def test_nan_fwhm(self, axis):
        y = _lor(axis, 2680, 30, 1.0)
        v, c, d = assess_layer_number(axis, y, 2680, float("nan"))
        assert v == "indeterminate" and c == "low"

    def test_too_few_points(self):
        x = np.linspace(2670, 2690, 5)   # <10 points in window
        y = _lor(x, 2680, 35, 1.0)
        v, c, d = assess_layer_number(x, y, 2680, 35.0)
        assert v == "indeterminate"
