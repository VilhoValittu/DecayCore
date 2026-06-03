"""Unit tests for measurement_features.py — RT60 and harmonic risk helpers."""

import math
from typing import Any

import numpy as np
import pytest

from decaycore.common.measurement_features import (
    normalize_rt60_value,
    normalize_rt60_bands,
    serialize_rt60_bands,
    summarize_rt60_bands,
    _harmonic_boost_risk_collect_orders,
    _harmonic_boost_risk_raw_curve,
    build_harmonic_boost_risk_curve,
)


class TestNormalizeRT60Value:
    """Tests for normalize_rt60_value()."""

    def test_valid_float(self):
        assert normalize_rt60_value(1.5) == 1.5
        assert normalize_rt60_value("2.5") == 2.5
        assert normalize_rt60_value(0.1) == 0.1

    def test_zero_and_negative(self):
        assert normalize_rt60_value(0.0) is None
        assert normalize_rt60_value(-1.0) is None

    def test_nan_and_inf(self):
        assert normalize_rt60_value(float("nan")) is None
        assert normalize_rt60_value(float("inf")) is None
        assert normalize_rt60_value(float("-inf")) is None

    def test_invalid_types(self):
        assert normalize_rt60_value(None) is None
        assert normalize_rt60_value("invalid") is None
        assert normalize_rt60_value([1.0]) is None


class TestNormalizeRT60Bands:
    """Tests for normalize_rt60_bands()."""

    def test_valid_float_keyed(self):
        bands = {125.0: 0.5, 250.0: 0.6, 500.0: 0.7}
        result = normalize_rt60_bands(bands)
        assert result == {125.0: 0.5, 250.0: 0.6, 500.0: 0.7}

    def test_string_keyed(self):
        bands = {"125": "0.5", "250": "0.6"}
        result = normalize_rt60_bands(bands)
        assert result == {125.0: 0.5, 250.0: 0.6}

    def test_mixed_keyed(self):
        bands = {125.0: 0.5, "250": 0.6}
        result = normalize_rt60_bands(bands)
        assert result == {125.0: 0.5, 250.0: 0.6}

    def test_invalid_entries_filtered(self):
        bands = {
            125.0: 0.5,
            "bad_key": 0.6,
            250.0: "not_a_float",
            500.0: float("nan"),
            1000.0: -0.5,
        }
        result = normalize_rt60_bands(bands)
        assert result == {125.0: 0.5}

    def test_empty_dict(self):
        assert normalize_rt60_bands({}) is None
        assert normalize_rt60_bands(None) is None
        assert normalize_rt60_bands("not_dict") is None

    def test_sorted_output(self):
        bands = {1000.0: 0.7, 125.0: 0.5, 500.0: 0.6}
        result = normalize_rt60_bands(bands)
        assert list(result.keys()) == [125.0, 500.0, 1000.0]

    def test_zero_freq_and_rt_filtered(self):
        bands = {0.0: 0.5, 125.0: 0.0, 250.0: 0.6}
        result = normalize_rt60_bands(bands)
        assert result == {250.0: 0.6}


class TestSerializeRT60Bands:
    """Tests for serialize_rt60_bands()."""

    def test_converts_to_string_keys(self):
        bands = {125.0: 0.5, 250.5: 0.6}
        result = serialize_rt60_bands(bands)
        assert result == {"125": 0.5, "250.5": 0.6}

    def test_invalid_bands_return_none(self):
        assert serialize_rt60_bands({}) is None
        assert serialize_rt60_bands({"bad": "data"}) is None

    def test_normalizes_input(self):
        bands = {"125": "0.5", "250": "0.6"}
        result = serialize_rt60_bands(bands)
        assert isinstance(result, dict)
        assert all(isinstance(k, str) for k in result.keys())


class TestSummarizeRT60Bands:
    """Tests for summarize_rt60_bands()."""

    def test_empty_bands(self):
        result = summarize_rt60_bands({})
        assert result["valid"] is False
        assert result["source_band_count"] == 0

    def test_single_band_in_range(self):
        bands = {250.0: 0.5}
        result = summarize_rt60_bands(bands)
        assert result["source_band_count"] == 1
        assert result["mid_400_2000_s"] is None
        assert result["lowmid_120_400_s"] is not None

    def test_bands_in_multiple_ranges(self):
        bands = {
            63.0: 0.4,      # LF
            250.0: 0.5,     # lowmid
            1000.0: 0.6,    # mid
            4000.0: 0.55,   # HF
        }
        result = summarize_rt60_bands(bands)
        assert result["valid"] is True
        assert result["lf_20_120_s"] is not None
        assert result["lowmid_120_400_s"] is not None
        assert result["mid_400_2000_s"] is not None
        assert result["hf_2000_8000_s"] is not None

    def test_wideband_override(self):
        bands = {250.0: 0.5}
        result = summarize_rt60_bands(bands, wideband=0.8)
        assert result["wideband_s"] == 0.8

    def test_invalid_wideband_ignored(self):
        bands = {250.0: 0.5}
        result = summarize_rt60_bands(bands, wideband=float("nan"))
        assert result["wideband_s"] is not None  # Uses median of bands

    def test_values_outside_valid_range_filtered(self):
        bands = {
            250.0: 0.01,    # Too low (< 0.05)
            500.0: 0.5,     # Valid
            1000.0: 10.0,   # Too high (> 5.0)
        }
        result = summarize_rt60_bands(bands)
        # Only 500 Hz is valid
        assert result["valid"] is True
        assert result["source_band_count"] == 3


class TestHarmonicBoostRiskCollectOrders:
    """Tests for _harmonic_boost_risk_collect_orders()."""

    def test_h2_only(self):
        freq = np.array([20.0, 50.0, 100.0])
        h_mags = {2: [1.0, 2.0, 3.0]}
        result = _harmonic_boost_risk_collect_orders(freq, h_mags)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], [1.0, 2.0, 3.0])

    def test_h2_h3_h4(self):
        freq = np.array([20.0, 50.0, 100.0])
        h_mags = {
            2: [1.0, 2.0, 3.0],
            3: [0.5, 1.0, 1.5],
            4: [0.25, 0.5, 0.75],
        }
        result = _harmonic_boost_risk_collect_orders(freq, h_mags)
        assert len(result) == 3

    def test_h3_only_rejected(self):
        """H3-only data (no H2) should be rejected — requires H2 first."""
        freq = np.array([20.0, 50.0, 100.0])
        h_mags = {3: [0.5, 1.0, 1.5]}  # No H2
        result = _harmonic_boost_risk_collect_orders(freq, h_mags)
        assert len(result) == 0  # H3 rejected because H2 missing

    def test_h4_only_rejected(self):
        """H4-only data should be rejected."""
        freq = np.array([20.0, 50.0, 100.0])
        h_mags = {4: [0.25, 0.5, 0.75]}
        result = _harmonic_boost_risk_collect_orders(freq, h_mags)
        assert len(result) == 0

    def test_mismatched_size_skipped(self):
        freq = np.array([20.0, 50.0, 100.0])
        h_mags = {
            2: [1.0, 2.0],  # Wrong size
            3: [0.5, 1.0, 1.5],  # Wrong size too
        }
        result = _harmonic_boost_risk_collect_orders(freq, h_mags)
        assert len(result) == 0

    def test_nan_values_skipped(self):
        freq = np.array([20.0, 50.0, 100.0])
        h_mags = {
            2: [float("nan"), float("nan"), float("nan")],
        }
        result = _harmonic_boost_risk_collect_orders(freq, h_mags)
        assert len(result) == 0

    def test_missing_order_skipped(self):
        freq = np.array([20.0, 50.0, 100.0])
        h_mags = {2: [1.0, 2.0, 3.0]}  # No H3
        result = _harmonic_boost_risk_collect_orders(freq, h_mags)
        assert len(result) == 1


class TestHarmonicBoostRiskRawCurve:
    """Tests for _harmonic_boost_risk_raw_curve()."""

    def test_empty_h_arrays(self):
        freq = np.array([20.0, 100.0, 1000.0])
        result = _harmonic_boost_risk_raw_curve(
            freq, [], use_relative=False, fund_at_harm=None
        )
        assert result is None

    def test_absolute_risk(self):
        freq = np.array([20.0, 100.0, 1000.0])
        h2 = np.array([-40.0, -30.0, -20.0])
        result = _harmonic_boost_risk_raw_curve(
            freq, [h2], use_relative=False, fund_at_harm=None
        )
        assert result is not None
        assert result.shape == freq.shape
        assert np.all((result >= 0.0) & (result <= 1.0))

    def test_relative_risk_with_fundamental(self):
        freq = np.array([20.0, 100.0, 1000.0])
        h2 = np.array([-30.0, -20.0, -10.0])
        fund = np.array([-20.0, -10.0, 0.0])
        result = _harmonic_boost_risk_raw_curve(
            freq, [h2], use_relative=True, fund_at_harm=fund
        )
        assert result is not None
        assert np.all((result >= 0.0) & (result <= 1.0))

    def test_all_nan_returns_none(self):
        freq = np.array([20.0, 100.0, 1000.0])
        h2 = np.array([float("nan"), float("nan"), float("nan")])
        result = _harmonic_boost_risk_raw_curve(
            freq, [h2], use_relative=False, fund_at_harm=None
        )
        assert result is None

    def test_mixed_finite_nan(self):
        freq = np.array([20.0, 100.0, 1000.0])
        h2 = np.array([-40.0, float("nan"), -20.0])
        result = _harmonic_boost_risk_raw_curve(
            freq, [h2], use_relative=False, fund_at_harm=None
        )
        assert result is not None
        assert np.isfinite(result[0])
        assert np.isfinite(result[2])


class TestBuildHarmonicBoostRiskCurve:
    """Integration tests for build_harmonic_boost_risk_curve()."""

    def test_valid_input(self):
        freq = np.linspace(20.0, 20000.0, 256)
        h_mags = {
            2: np.random.uniform(-40, -10, 256),
            3: np.random.uniform(-50, -20, 256),
        }
        freq_out, risk_out, summary = build_harmonic_boost_risk_curve(
            freq, h_mags, fundamental_freq_hz=freq / 2
        )
        assert freq_out is not None
        assert risk_out is not None
        assert isinstance(summary, dict)
        assert risk_out.shape == freq.shape
        assert np.all((risk_out >= 0.0) & (risk_out <= 1.0))
        assert summary.get("valid") in (True, False)

    def test_missing_harmonic_data(self):
        freq = np.linspace(20.0, 20000.0, 256)
        freq_out, risk_out, summary = build_harmonic_boost_risk_curve(
            freq, {}, fundamental_freq_hz=freq / 2
        )
        assert freq_out is None
        assert risk_out is None

    def test_h3_only_no_fundamental(self):
        """H3-only data without H2 should return None."""
        freq = np.linspace(20.0, 20000.0, 256)
        h_mags = {3: np.random.uniform(-50, -20, 256)}
        freq_out, risk_out, summary = build_harmonic_boost_risk_curve(freq, h_mags)
        assert freq_out is None
        assert risk_out is None

    def test_short_frequency_array(self):
        freq = np.array([20.0, 100.0])  # Only 2 points
        h_mags = {2: [1.0, 2.0]}
        freq_out, risk_out, summary = build_harmonic_boost_risk_curve(freq, h_mags)
        assert freq_out is None
        assert risk_out is None
