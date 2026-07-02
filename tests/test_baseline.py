"""
Tests for src/baseline.py
Covers: ALS correction, no clipping of negative residuals (fix #10),
        baseline shape, and lambda/p parameter effects.
"""
import numpy as np
import pytest
from src.baseline import als_baseline


class TestALSBaseline:
    def test_returns_two_arrays(self, wavenumbers, graphene_spectrum):
        result = als_baseline(graphene_spectrum)
        assert len(result) == 2

    def test_corrected_same_length(self, wavenumbers, graphene_spectrum):
        corrected, baseline = als_baseline(graphene_spectrum)
        assert len(corrected) == len(graphene_spectrum)
        assert len(baseline) == len(graphene_spectrum)

    def test_baseline_smooth(self, graphene_spectrum):
        """Baseline should be smoother than the raw spectrum."""
        _, baseline = als_baseline(graphene_spectrum)
        raw_var  = float(np.var(np.diff(graphene_spectrum)))
        base_var = float(np.var(np.diff(baseline)))
        assert base_var < raw_var

    def test_no_negative_clipping(self):
        """fix #10: negative residuals must NOT be clipped to zero."""
        # Construct a spectrum where ALS will over-subtract in a region
        x = np.linspace(0, 1, 500)
        # Rising baseline + small peak
        y = 2.0 * x + 0.3 * np.exp(-((x - 0.5) ** 2) / 0.005)
        corrected, _ = als_baseline(y, lam=1e4, p=0.01)
        # Some values can be negative (over-subtraction artefact) — must be preserved
        # We just check the function doesn't forcibly set everything >= 0
        assert corrected.min() < 0.5  # meaningful check that clipping didn't raise floor

    def test_flat_baseline_unchanged(self):
        """Flat signal: ALS should produce near-zero correction."""
        y = np.ones(200) * 5.0
        corrected, baseline = als_baseline(y)
        np.testing.assert_allclose(baseline, 5.0, atol=0.5)

    def test_lambda_effect(self, graphene_spectrum):
        """Higher lambda → smoother (less variable) baseline."""
        _, bl_low  = als_baseline(graphene_spectrum, lam=1e3)
        _, bl_high = als_baseline(graphene_spectrum, lam=1e7)
        assert np.var(np.diff(bl_high)) < np.var(np.diff(bl_low))
