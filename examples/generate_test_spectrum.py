"""
Generate a synthetic graphene Raman spectrum for testing.
Peaks: D(1350), G(1580), D'(1620), 2D(2700), D+G(2940)
"""
import numpy as np

np.random.seed(42)
wavenumber = np.linspace(1000, 3200, 2200)

def lorentzian(x, center, amplitude, fwhm):
    sigma = fwhm / 2
    return amplitude / (1 + ((x - center) / sigma) ** 2)

def pseudo_voigt(x, center, amplitude, fwhm, eta=0.5):
    sigma = fwhm / 2
    L = amplitude / (1 + ((x - center) / sigma) ** 2)
    G = amplitude * np.exp(-np.log(2) * ((x - center) / sigma) ** 2)
    return eta * L + (1 - eta) * G

spectrum = (
    lorentzian(wavenumber, 1350, 800,  45)    +  # D
    lorentzian(wavenumber, 1580, 2000, 22)    +  # G
    lorentzian(wavenumber, 1620, 200,  18)    +  # D'
    lorentzian(wavenumber, 2700, 3500, 55)    +  # 2D (monolayer-like)
    pseudo_voigt(wavenumber, 2940, 180, 60)   +  # D+G
    50 + 0.01 * wavenumber                    +  # linear background
    np.random.normal(0, 15, len(wavenumber))      # noise
)

data = np.column_stack([wavenumber, np.clip(spectrum, 0, None)])
np.savetxt("examples/graphene_test.txt", data, fmt="%.4f", delimiter="  ",
           header="Wavenumber(cm-1)  Intensity(a.u.)\nSynthetic graphene Raman spectrum (532nm)")
print(f"Test spectrum saved: examples/graphene_test.txt ({len(wavenumber)} points)")
