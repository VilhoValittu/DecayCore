"""
Parity tests for Rust robust per-row RMS implementation.
Compares rms_rows_kernel_rs against the pure-Python reference.
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


class TestRmsRowsKernelRsParity:
    """Verify Rust rms_rows_kernel_rs matches the pure-Python kernel."""

    @pytest.fixture
    def dsp(self):
        dsp = get_decaycore_dsp()
        if dsp is None:
            pytest.skip("decaycore_dsp not available")
        return dsp

    @pytest.fixture
    def py_kernel(self):
        from src.decaycore.measurement.repeat_analysis_parts.repeat_representative import (
            _rms_rows_kernel,
        )
        return _rms_rows_kernel

    @pytest.mark.parametrize("shape", [(1, 1), (3, 8), (5, 9), (16, 64), (4, 2)])
    def test_random_parity(self, dsp, py_kernel, shape):
        rng = np.random.default_rng(sum(shape))
        arr = np.ascontiguousarray(rng.standard_normal(shape) * 3.0)

        result_rs = dsp.rms_rows_kernel_rs(arr)
        result_py = py_kernel(arr)

        assert result_rs.shape == result_py.shape
        np.testing.assert_allclose(result_rs, result_py, rtol=1e-9, atol=1e-9)

    def test_nan_inf_rows(self, dsp, py_kernel):
        arr = np.array(
            [
                [1.0, 2.0, np.nan, 4.0],          # even count after filtering -> 3 finite
                [np.nan, np.nan, np.nan, np.nan],  # all non-finite -> 0.0
                [np.inf, 2.0, -np.inf, 6.0],       # filter infs -> 2 finite
                [5.0, 5.0, 5.0, 5.0],              # constant
            ],
            dtype=np.float64,
        )
        arr = np.ascontiguousarray(arr)

        result_rs = dsp.rms_rows_kernel_rs(arr)
        result_py = py_kernel(arr)
        np.testing.assert_allclose(result_rs, result_py, rtol=1e-9, atol=1e-9)

    def test_even_odd_counts(self, dsp, py_kernel):
        # NaN padding yields different finite counts per row:
        # row 0 -> 3 finite (odd median), row 1 -> 4 finite (even median).
        arr = np.array(
            [
                [1.0, 9.0, 4.0, np.nan],
                [1.0, 9.0, 4.0, 16.0],
            ],
            dtype=np.float64,
        )
        arr = np.ascontiguousarray(arr)
        result_rs = dsp.rms_rows_kernel_rs(arr)
        result_py = py_kernel(arr)
        np.testing.assert_allclose(result_rs, result_py, rtol=1e-9, atol=1e-9)
