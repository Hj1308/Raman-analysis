"""
Shared fixtures for all Raman-analysis tests.
All spectra are synthetic Lorentzian/noise signals — no real data files required.
"""
import numpy as np
import pytest


def _lorentzian(x, center, amplitude, gamma):
    return amplitude / (1.0 + ((x - center) / gamma) ** 2)


def _pseudo_voigt(x, center, amplitude, fwhm, eta):
    sigma = fwhm / 2.3548206
    gamma = fwhm / 2.0
    gauss = amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)
    loren = amplitude / (1.0 + ((x - center) / gamma) ** 2)
    return eta * loren + (1.0 - eta) * gauss


@pytest.fixture
def wavenumbers():
    return np.linspace(800, 3200, 2400)


@pytest.fixture
def graphene_spectrum(wavenumbers):
    """Clean monolayer graphene: low D, strong G, strong 2D, tiny D+G."""
    rng = np.random.default_rng(42)
    y = np.zeros_like(wavenumbers)
    y += _lorentzian(wavenumbers, 1350, 0.30, 25)   # D  (weak)
    y += _lorentzian(wavenumbers, 1582, 1.00, 14)   # G  (strong)
    y += _lorentzian(wavenumbers, 1622, 0.07, 10)   # D' (tiny)
    y += _lorentzian(wavenumbers, 2690, 2.00, 28)   # 2D (strong, monolayer)
    y += _lorentzian(wavenumbers, 2940, 0.12, 20)   # D+G
    y += rng.normal(0, 0.015, size=len(wavenumbers))
    return np.clip(y, 0, None)


@pytest.fixture
def defective_spectrum(wavenumbers):
    """Defective graphene: strong D, moderate G, weak 2D."""
    rng = np.random.default_rng(7)
    y = np.zeros_like(wavenumbers)
    y += _lorentzian(wavenumbers, 1348, 0.90, 35)   # D  (strong)
    y += _lorentzian(wavenumbers, 1582, 0.80, 18)   # G
    y += _lorentzian(wavenumbers, 1622, 0.18, 12)   # D'
    y += _lorentzian(wavenumbers, 2690, 0.40, 40)   # 2D (weak)
    y += rng.normal(0, 0.015, size=len(wavenumbers))
    return np.clip(y, 0, None)


@pytest.fixture
def doped_spectrum(wavenumbers):
    """N-doped graphene: G shifted to ~1598, overlapping D'."""
    rng = np.random.default_rng(99)
    y = np.zeros_like(wavenumbers)
    y += _lorentzian(wavenumbers, 1348, 0.80, 30)
    y += _lorentzian(wavenumbers, 1598, 1.00, 18)   # G shifted up
    y += _lorentzian(wavenumbers, 1625, 0.50, 12)   # D' overlapping G
    y += _lorentzian(wavenumbers, 2688, 0.50, 40)
    y += rng.normal(0, 0.015, size=len(wavenumbers))
    return np.clip(y, 0, None)


@pytest.fixture
def noisy_spectrum(wavenumbers):
    """High-noise spectrum: tests SNR gate (should reject weak peaks)."""
    rng = np.random.default_rng(13)
    y = np.zeros_like(wavenumbers)
    y += _lorentzian(wavenumbers, 1582, 0.20, 14)   # G barely above noise
    y += rng.normal(0, 0.18, size=len(wavenumbers))
    return y
@pytest.fixture
def gcn4_wavenumbers():
    return np.linspace(600, 1200, 1200)


@pytest.fixture
def gcn4_spectrum(gcn4_wavenumbers):
    """Synthetic g-C3N4-like spectrum with CN modes at ~691 and ~988 cm⁻1."""
    rng = np.random.default_rng(123)
    y = np.zeros_like(gcn4_wavenumbers)
    y += _lorentzian(gcn4_wavenumbers, 691.0, 1.00, 8.0)
    y += _lorentzian(gcn4_wavenumbers, 988.0, 0.85, 10.0)
    y += rng.normal(0, 0.01, size=len(gcn4_wavenumbers))
    return np.clip(y, 0, None)