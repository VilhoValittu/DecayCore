"""
Parity tests for Rust slope-limiter implementation.
Compares slope_passes_rs / slope_passes_asym_rs against the pure-Python reference.
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


class TestSlopePassesRsParity:
    """Verify Rust slope limiters match the pure-Python kernels."""

    @pytest.fixture
    def dsp(self):
        dsp = get_decaycore_dsp()
        if dsp is None:
            pytest.skip("decaycore_dsp not available")
        return dsp

    @pytest.fixture
    def py_slope_passes(self):
        from src.decaycore.dsp.limits import _slope_passes
        return _slope_passes

    @pytest.fixture
    def py_slope_passes_asym(self):
        from src.decaycore.dsp.limits import _slope_passes_asym
        return _slope_passes_asym

    def _log_axis(self, n):
        return np.log2(np.logspace(np.log10(20.0), np.log10(20000.0), n, dtype=np.float64))

    @pytest.mark.parametrize("seed", [0, 1, 7, 42])
    def test_slope_passes_parity(self, dsp, py_slope_passes, seed):
        rng = np.random.default_rng(seed)
        n = 257
        x = self._log_axis(n)
        g = rng.standard_normal(n) * 6.0

        result_rs = dsp.slope_passes_rs(g.copy(), x, 6.0)
        result_py = py_slope_passes(g.copy(), x, 6.0)

        assert result_rs.shape == result_py.shape
        np.testing.assert_allclose(result_rs, result_py, rtol=1e-9, atol=1e-9)

    def test_slope_passes_steep_step(self, dsp, py_slope_passes):
        n = 128
        x = self._log_axis(n)
        g = np.zeros(n, dtype=np.float64)
        g[n // 2:] = 30.0  # sharp step that must be slope-limited

        result_rs = dsp.slope_passes_rs(g.copy(), x, 3.0)
        result_py = py_slope_passes(g.copy(), x, 3.0)
        np.testing.assert_allclose(result_rs, result_py, rtol=1e-9, atol=1e-9)

    def test_slope_passes_rejects_mismatched_lengths(self, dsp):
        g = np.array([0.0], dtype=np.float64)
        x = np.array([0.0, 1.0], dtype=np.float64)

        with pytest.raises(ValueError, match="same length"):
            dsp.slope_passes_rs(g, x, 6.0)

        with pytest.raises(ValueError, match="same length"):
            dsp.slope_passes_asym_rs(g, x, 6.0, 12.0)

    @pytest.mark.parametrize("boost,cut", [(6.0, 12.0), (3.0, 0.0), (0.0, 8.0), (4.0, 4.0)])
    def test_slope_passes_asym_parity(self, dsp, py_slope_passes_asym, boost, cut):
        rng = np.random.default_rng(123)
        n = 211
        x = self._log_axis(n)
        g = rng.standard_normal(n) * 8.0

        result_rs = dsp.slope_passes_asym_rs(g.copy(), x, boost, cut)
        result_py = py_slope_passes_asym(g.copy(), x, boost, cut)

        assert result_rs.shape == result_py.shape
        np.testing.assert_allclose(result_rs, result_py, rtol=1e-9, atol=1e-9)
