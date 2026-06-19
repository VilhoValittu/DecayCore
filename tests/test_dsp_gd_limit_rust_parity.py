"""
Parity tests for Rust GD limit implementation.
Compares gd_smooth_loop_rs against Python reference.
"""

import numpy as np
import pytest


def get_decaycore_dsp():
    """Attempt to import Rust DSP module."""
    try:
        import decaycore_dsp
        return decaycore_dsp
    except ImportError:
        return None


# Mark as requiring Rust extension
pytestmark = pytest.mark.requires_rust


class TestGdSmoothLoopRsParity:
    """Verify Rust gd_smooth_loop_rs matches Python _gd_smooth_loop_numba."""

    @pytest.fixture
    def dsp(self):
        """Ensure Rust module is available."""
        dsp = get_decaycore_dsp()
        if dsp is None:
            pytest.skip("decaycore_dsp not available")
        return dsp

    @pytest.fixture
    def python_gd_smooth_loop(self):
        """Import Python numba reference implementation."""
        from src.decaycore.dsp.dsp_ops import _gd_smooth_loop_numba
        return _gd_smooth_loop_numba

    def _make_log_freq_axis(self, f_min, f_max, n):
        """Create log-spaced frequency axis."""
        return np.log2(np.logspace(np.log10(f_min), np.log10(f_max), n, dtype=np.float64))

    def test_gd_smooth_loop_rs_flat(self, dsp, python_gd_smooth_loop):
        """Test with constant GD (no variation)."""
        n = 300
        gd_l = np.ones(n, dtype=np.float64) * 5.0
        log2f = self._make_log_freq_axis(20, 20000, n)
        lim_arr = np.ones(n, dtype=np.float64) * 30.0
        curv_lim_arr = np.ones(n, dtype=np.float64) * 120.0
        sigma = 0.8

        result_rs = dsp.gd_smooth_loop_rs(gd_l, log2f, lim_arr, curv_lim_arr, sigma)
        result_py = python_gd_smooth_loop(gd_l, log2f, lim_arr, curv_lim_arr, sigma)

        assert result_rs.shape == result_py.shape
        np.testing.assert_allclose(result_rs, result_py, rtol=1e-9, atol=1e-9)

    def test_gd_smooth_loop_rs_peak(self, dsp, python_gd_smooth_loop):
        """Test with GD peak that needs smoothing."""
        n = 500
        freq_hz = np.logspace(np.log10(20), np.log10(20000), n, dtype=np.float64)

        # Create a sharp peak around 200 Hz
        peak_width = 50
        gd_l = np.where(np.abs(freq_hz - 200) < peak_width, 20.0, 5.0).astype(np.float64)

        log2f = np.log2(freq_hz)
        lim_arr = np.ones(n, dtype=np.float64) * 30.0
        curv_lim_arr = np.ones(n, dtype=np.float64) * 120.0
        sigma = 0.8

        result_rs = dsp.gd_smooth_loop_rs(gd_l, log2f, lim_arr, curv_lim_arr, sigma)
        result_py = python_gd_smooth_loop(gd_l, log2f, lim_arr, curv_lim_arr, sigma)

        assert result_rs.shape == result_py.shape
        # Peak should be attenuated by both implementations
        assert np.max(result_rs) < np.max(gd_l)
        assert np.max(result_py) < np.max(gd_l)
        np.testing.assert_allclose(result_rs, result_py, rtol=1e-8, atol=1e-8)

    def test_gd_smooth_loop_rs_empty(self, dsp, python_gd_smooth_loop):
        """Test edge case: empty input."""
        gd_l = np.array([], dtype=np.float64)
        log2f = np.array([], dtype=np.float64)
        lim_arr = np.array([], dtype=np.float64)
        curv_lim_arr = np.array([], dtype=np.float64)
        sigma = 0.8

        result_rs = dsp.gd_smooth_loop_rs(gd_l, log2f, lim_arr, curv_lim_arr, sigma)
        result_py = python_gd_smooth_loop(gd_l, log2f, lim_arr, curv_lim_arr, sigma)

        assert result_rs.shape == (0,)
        assert result_py.shape == (0,)

    def test_gd_smooth_loop_rs_violation(self, dsp, python_gd_smooth_loop):
        """Test with gradient violating the limit."""
        n = 100
        freq_hz = np.logspace(np.log10(20), np.log10(20000), n, dtype=np.float64)
        log2f = np.log2(freq_hz)

        # Create a noisy input with sharp peaks that violate limit
        np.random.seed(42)
        gd_l = 10 + 5 * np.sin(4 * np.linspace(0, 2*np.pi, n)) + np.random.randn(n) * 2
        gd_l = gd_l.astype(np.float64)

        # Very strict limits to trigger smoothing
        lim_arr = np.ones(n, dtype=np.float64) * 1.0  # Very restrictive
        curv_lim_arr = np.ones(n, dtype=np.float64) * 5.0
        sigma = 0.8

        result_rs = dsp.gd_smooth_loop_rs(gd_l, log2f, lim_arr, curv_lim_arr, sigma)
        result_py = python_gd_smooth_loop(gd_l, log2f, lim_arr, curv_lim_arr, sigma)

        assert result_rs.shape == result_py.shape
        # Both should be smoothed (less variation than original)
        assert np.std(np.diff(result_rs)) < np.std(np.diff(gd_l))
        assert np.std(np.diff(result_py)) < np.std(np.diff(gd_l))
        np.testing.assert_allclose(result_rs, result_py, rtol=1e-8, atol=1e-8)

    def test_gd_smooth_loop_rs_narrow_window(self, dsp, python_gd_smooth_loop):
        """Test with very small array."""
        n = 16  # Minimal valid size
        freq_hz = np.logspace(np.log10(20), np.log10(20000), n, dtype=np.float64)
        gd_l = np.random.RandomState(42).randn(n).astype(np.float64) * 10 + 10
        log2f = np.log2(freq_hz)
        lim_arr = np.ones(n, dtype=np.float64) * 30.0
        curv_lim_arr = np.ones(n, dtype=np.float64) * 120.0
        sigma = 0.8

        result_rs = dsp.gd_smooth_loop_rs(gd_l, log2f, lim_arr, curv_lim_arr, sigma)
        result_py = python_gd_smooth_loop(gd_l, log2f, lim_arr, curv_lim_arr, sigma)

        assert result_rs.shape == result_py.shape
        np.testing.assert_allclose(result_rs, result_py, rtol=1e-8, atol=1e-8)

    def test_gd_smooth_loop_rs_sigma_scaling(self, dsp, python_gd_smooth_loop):
        """Test with different sigma values."""
        n = 200
        freq_hz = np.logspace(np.log10(20), np.log10(20000), n, dtype=np.float64)
        gd_l = np.sin(np.log2(freq_hz)).astype(np.float64) * 10 + 10
        log2f = np.log2(freq_hz)
        lim_arr = np.ones(n, dtype=np.float64) * 15.0
        curv_lim_arr = np.ones(n, dtype=np.float64) * 60.0

        for sigma in [0.25, 0.6, 1.2, 2.0]:
            result_rs = dsp.gd_smooth_loop_rs(gd_l, log2f, lim_arr, curv_lim_arr, sigma)
            result_py = python_gd_smooth_loop(gd_l, log2f, lim_arr, curv_lim_arr, sigma)

            np.testing.assert_allclose(result_rs, result_py, rtol=1e-8, atol=1e-8,
                                      err_msg=f"Mismatch at sigma={sigma}")

    def test_gd_smooth_loop_rs_variable_limits(self, dsp, python_gd_smooth_loop):
        """Test with frequency-dependent limits."""
        n = 250
        freq_hz = np.logspace(np.log10(20), np.log10(20000), n, dtype=np.float64)
        gd_l = 10 + 5 * np.sin(np.log2(freq_hz) / 2).astype(np.float64)
        log2f = np.log2(freq_hz)

        # Tighter limits in bass, looser in treble (mimic real usage)
        lim_arr = np.where(freq_hz < 100, 15.0, np.where(freq_hz < 1000, 25.0, 35.0)).astype(np.float64)
        curv_lim_arr = lim_arr * 4
        sigma = 0.8

        result_rs = dsp.gd_smooth_loop_rs(gd_l, log2f, lim_arr, curv_lim_arr, sigma)
        result_py = python_gd_smooth_loop(gd_l, log2f, lim_arr, curv_lim_arr, sigma)

        assert result_rs.shape == result_py.shape
        np.testing.assert_allclose(result_rs, result_py, rtol=1e-8, atol=1e-8)
