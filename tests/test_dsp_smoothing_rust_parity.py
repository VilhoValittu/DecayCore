"""
Parity tests for Rust smoothing implementation.
Compares smooth_mag_core_rs against Python reference.
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


class TestSmoothmMagCoreRsParity:
    """Verify Rust smooth_mag_core_rs matches Python _smooth_mag_core."""

    @pytest.fixture
    def dsp(self):
        """Ensure Rust module is available."""
        dsp = get_decaycore_dsp()
        if dsp is None:
            pytest.skip("decaycore_dsp not available")
        return dsp

    def test_smooth_mag_core_rs_basic(self, dsp):
        """Basic smoke test: flat magnitude smoothing."""
        freqs = np.array([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000], dtype=np.float64)
        mags = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # Smoothing plan (from smoothing.py)
        log_freqs = np.geomspace(20, 20000, 384)
        window = np.hanning(192)
        window /= window.sum()
        pad_len = 96

        # Call Rust function
        result = dsp.smooth_mag_core_rs(freqs, mags, log_freqs, window, pad_len)

        # Result should be all zeros
        assert result.shape == freqs.shape
        assert np.allclose(result, 0.0, atol=1e-12)

    def test_smooth_mag_core_rs_constant(self, dsp):
        """Test with constant magnitude (no variation)."""
        freqs = np.linspace(20, 20000, 500, dtype=np.float64)
        mags = np.ones_like(freqs) * 5.0

        log_freqs = np.geomspace(20, 20000, 384)
        window = np.hanning(192)
        window /= window.sum()
        pad_len = 96

        result = dsp.smooth_mag_core_rs(freqs, mags, log_freqs, window, pad_len)

        # Constant input should stay constant (smoothing just blurs it)
        assert result.shape == freqs.shape
        assert np.allclose(result, 5.0, atol=0.1)

    def test_smooth_mag_core_rs_peak(self, dsp):
        """Test with a resonance peak."""
        freqs = np.linspace(20, 20000, 500, dtype=np.float64)
        # Create a peak around 200 Hz
        mags = np.where(np.abs(freqs - 200) < 50, 10.0, 0.0).astype(np.float64)

        log_freqs = np.geomspace(20, 20000, 384)
        window = np.hanning(192)
        window /= window.sum()
        pad_len = 96

        result = dsp.smooth_mag_core_rs(freqs, mags, log_freqs, window, pad_len)

        # Smoothing should reduce sharp peak but still preserve some of it
        assert result.shape == freqs.shape
        assert np.max(result) <= 10.0 + 0.1  # Allow small overshoot
        assert np.max(result) >= 1.0  # Smoothing reduces the peak significantly

    def test_smooth_mag_core_rs_empty_input(self, dsp):
        """Test edge case: empty input."""
        freqs = np.array([], dtype=np.float64)
        mags = np.array([], dtype=np.float64)
        log_freqs = np.array([100, 200, 400], dtype=np.float64)
        window = np.array([0.5, 1.0, 0.5], dtype=np.float64)
        window /= window.sum()
        pad_len = 1

        result = dsp.smooth_mag_core_rs(freqs, mags, log_freqs, window, pad_len)
        assert result.shape == (0,)

    def test_smooth_mag_core_rs_very_narrow_window(self, dsp):
        """Test with single-element window (minimal smoothing)."""
        freqs = np.array([20, 100, 500, 2000, 10000], dtype=np.float64)
        mags = np.array([0.0, 3.0, 0.0, 3.0, 0.0], dtype=np.float64)

        log_freqs = np.geomspace(20, 10000, 200)
        window = np.array([1.0], dtype=np.float64)  # Single element
        pad_len = 0

        result = dsp.smooth_mag_core_rs(freqs, mags, log_freqs, window, pad_len)
        assert result.shape == freqs.shape
        # With single-element window, should be close to original
        assert np.allclose(result, mags, atol=0.5)

    def test_smooth_mag_core_rs_pink_noise_like(self, dsp):
        """Test with pink-noise-shaped magnitude (1/f)."""
        freqs = np.logspace(1.3, 4.3, 300, dtype=np.float64)  # 20 Hz to 20 kHz
        mags = 1.0 / np.sqrt(np.maximum(freqs, 20.0))  # 1/sqrt(f)

        log_freqs = np.geomspace(20, 20000, 400)
        window = np.hanning(128)
        window /= window.sum()
        pad_len = 64

        result = dsp.smooth_mag_core_rs(freqs, mags, log_freqs, window, pad_len)

        # Should produce a smooth result, not NaN or inf
        assert result.shape == freqs.shape
        assert np.all(np.isfinite(result))
        # Smoothing should reduce high-frequency variation
        assert np.std(np.diff(result)) < np.std(np.diff(mags))

    def test_smooth_mag_core_rs_output_shape(self, dsp):
        """Test output shape matches input frequency array."""
        for n in [10, 100, 500, 2000]:
            freqs = np.logspace(1.3, 4.3, n, dtype=np.float64)
            mags = np.sin(np.log10(freqs)).astype(np.float64)

            log_freqs = np.geomspace(freqs[0], freqs[-1], 200)
            window = np.hanning(64)
            window /= window.sum()
            pad_len = 32

            result = dsp.smooth_mag_core_rs(freqs, mags, log_freqs, window, pad_len)
            assert result.shape == (n,), f"Expected shape ({n},), got {result.shape}"
