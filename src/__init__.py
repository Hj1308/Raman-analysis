"""Raman Spectrum Analyzer for graphene and graphene-like materials."""
__version__ = "2.5.3"
__author__  = "Hoda Jaafari"

from .analyzer  import analyze, RamanAnalysis, format_report
from .peak_fitter import fit_all_peaks, get_peak_windows, PeakResult
from .baseline  import als_baseline, correct_baseline
from .loader    import load_spectrum

# Public alias expected by CI check
class RamanAnalyzer:
    """
    High-level facade that bundles the full analysis pipeline.

    Usage
    -----
    >>> ra = RamanAnalyzer(laser_nm=532)
    >>> peaks    = ra.fit(wn, intensity)
    >>> analysis = ra.analyze(peaks)
    >>> report   = ra.report("sample.txt", peaks, analysis)
    """

    def __init__(self, laser_nm: float = 532.0):
        self.laser_nm = float(laser_nm)

    def fit(self, wn, intensity, band_config=None):
        """Return dict[str, PeakResult] from raw (baseline-subtracted) spectrum."""
        return fit_all_peaks(wn, intensity,
                             laser_nm=self.laser_nm,
                             band_config=band_config)

    def analyze(self, peaks):
        """Return RamanAnalysis from a peaks dict."""
        return analyze(peaks, laser_nm=self.laser_nm)

    def report(self, filename, peaks, analysis):
        """Return formatted text report."""
        return format_report(filename, peaks, analysis, self.laser_nm)


__all__ = [
    "RamanAnalyzer",
    "RamanAnalysis",
    "PeakResult",
    "fit_all_peaks",
    "get_peak_windows",
    "analyze",
    "format_report",
    "als_baseline",
    "correct_baseline",
    "load_spectrum",
]
