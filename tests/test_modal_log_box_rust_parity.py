"""
Parity tests for Rust log-box smoothing implementation.
Compares smooth_log_box_kernel_rs against the pure-Python reference.
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


pytestmark = pytest.mark.requires_rust


class TestSmoothLogBoxKernelRsParity:
    """Verify Rust smooth_log_box_kernel_rs matches the pure-Python kernel."""

    @pytest.fixture
    def dsp(self):
        dsp = get_decaycore_dsp()
        if dsp is None:
            pytest.skip("decaycore_dsp not available")
        return dsp

    @pytest.fixture
    def py_kernel(self):
        from src.decaycore.dsp.modal_analysis_parts.modal_preparation import (
            _smooth_log_box_kernel,
        )
        return _smooth_log_box_kernel

    def _log_axis(self, n, f_min=20.0, f_max=20000.0):
        f = np.logspace(np.log10(f_min), np.log10(f_max), n, dtype=np.float64)
        return np.ascontiguousarray(np.log2(f))

    @pytest.mark.parametrize("n,half", [(64, 0.1), (300, 0.25), (513, 1.0 / 6.0)])
    def test_smooth_parity(self, dsp, py_kernel, n, half):
        rng = np.random.default_rng(n)
        x = self._log_axis(n)
        y = np.ascontiguousarray(rng.standard_normal(n) * 4.0)

        result_rs = dsp.smooth_log_box_kernel_rs(x, y, half)
        result_py = py_kernel(x, y, half)

        assert result_rs.shape == result_py.shape
        np.testing.assert_allclose(result_rs, result_py, rtol=1e-9, atol=1e-9)

    def test_with_nan_values(self, dsp, py_kernel):
        n = 200
        x = self._log_axis(n)
        rng = np.random.default_rng(99)
        y = rng.standard_normal(n) * 5.0
        y[::7] = np.nan  # scattered missing values dropped from the box average
        y = np.ascontiguousarray(y)

        result_rs = dsp.smooth_log_box_kernel_rs(x, y, 0.2)
        result_py = py_kernel(x, y, 0.2)
        np.testing.assert_allclose(result_rs, result_py, rtol=1e-9, atol=1e-9)

    def test_rejects_mismatched_lengths(self, dsp):
        x = np.array([0.0, 1.0], dtype=np.float64)
        y = np.array([1.0], dtype=np.float64)

        with pytest.raises(ValueError, match="same length"):
            dsp.smooth_log_box_kernel_rs(x, y, 0.1)

    def test_nonuniform_axis(self, dsp, py_kernel):
        # Irregularly spaced (still sorted) log axis to exercise the binary search.
        rng = np.random.default_rng(3)
        steps = np.abs(rng.standard_normal(150)) + 0.01
        x = np.ascontiguousarray(np.cumsum(steps))
        y = np.ascontiguousarray(rng.standard_normal(150) * 2.0)

        result_rs = dsp.smooth_log_box_kernel_rs(x, y, 0.5)
        result_py = py_kernel(x, y, 0.5)
        np.testing.assert_allclose(result_rs, result_py, rtol=1e-9, atol=1e-9)

    def test_wide_window_covers_all(self, dsp, py_kernel):
        n = 80
        x = self._log_axis(n)
        rng = np.random.default_rng(11)
        y = np.ascontiguousarray(rng.standard_normal(n))

        result_rs = dsp.smooth_log_box_kernel_rs(x, y, 100.0)  # window covers everything
        result_py = py_kernel(x, y, 100.0)
        np.testing.assert_allclose(result_rs, result_py, rtol=1e-9, atol=1e-9)
