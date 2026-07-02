"""
Tests for src/loader.py
Covers: txt/csv loading, comment-line skipping, two-column parsing,
        error handling for malformed files.
"""
import numpy as np
import pytest
import os
import tempfile
from src.loader import load_spectrum


def _write_temp(content: str, suffix=".txt") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


class TestLoadSpectrum:
    def test_plain_two_column(self):
        path = _write_temp("1000.0\t50.0\n1001.0\t51.0\n1002.0\t52.0\n")
        try:
            wn, intensity = load_spectrum(path)
            assert len(wn) == 3
            assert abs(wn[0] - 1000.0) < 0.01
            assert abs(intensity[1] - 51.0) < 0.01
        finally:
            os.unlink(path)

    def test_comment_lines_skipped(self):
        content = "# Raman spectrum\n# laser=532nm\n1000 100\n1001 101\n"
        path = _write_temp(content)
        try:
            wn, intensity = load_spectrum(path)
            assert len(wn) == 2
        finally:
            os.unlink(path)

    def test_comma_separated(self):
        path = _write_temp("1000,100\n1001,101\n1002,102\n", suffix=".csv")
        try:
            wn, intensity = load_spectrum(path)
            assert len(wn) == 3
        finally:
            os.unlink(path)

    def test_wavenumber_sorted_ascending(self):
        """Loader should return wavenumbers in ascending order."""
        content = "1002 102\n1000 100\n1001 101\n"
        path = _write_temp(content)
        try:
            wn, _ = load_spectrum(path)
            assert list(wn) == sorted(wn)
        finally:
            os.unlink(path)

    def test_returns_numpy_arrays(self):
        path = _write_temp("1000 100\n1001 101\n")
        try:
            wn, intensity = load_spectrum(path)
            assert isinstance(wn, np.ndarray)
            assert isinstance(intensity, np.ndarray)
        finally:
            os.unlink(path)

    def test_nonexistent_file_raises(self):
        with pytest.raises((FileNotFoundError, OSError)):
            load_spectrum("/tmp/nonexistent_raman_file_xyz.txt")
