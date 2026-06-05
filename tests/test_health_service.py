# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Tests for application health check service."""

from __future__ import annotations

import pytest

from src.decaycore.application.health_service import (
    HealthResult,
    Issue,
    compute_health,
    format_health_summary,
)


def minimal_config(mode: str = "BASIC") -> dict:
    """Minimal valid config for testing."""
    return {
        "bass_integration_enable": False,
        "file_l": None,
        "local_path_l": None,
        "file_r": None,
        "local_path_r": None,
        "hc_mode": "parametric",
        "mag_correct": True,
        "mag_c_min": 20.0,
        "mag_c_max": 20000.0,
        "fs": 48000,
        "taps": 4096,
        "lvl_min": 60.0,
        "lvl_max": 90.0,
        "max_boost": 12.0,
        "phase_limit": 150.0,
        "exc_prot": True,
        "hpf_enable": False,
        "filter_type": "linear",
        "mixed_freq": None,
        "trans_width": None,
    }


class TestComputeHealthBasic:
    """Tests for compute_health happy path and basic cases."""

    def test_minimal_valid_config_warn(self):
        """Valid config but missing measurements → overall warn, not blocked."""
        data = minimal_config()
        hr = compute_health(data, mode="BASIC")

        # Missing measurements cause warn level
        assert hr.overall == "warn"
        assert hr.blocked is False
        assert len(hr.issues) > 0

    def test_missing_measurements_warn(self):
        """Missing measurement sources → warn level."""
        data = minimal_config()
        # Both l and r measurement sources missing
        hr = compute_health(data, mode="BASIC")

        # Should have at least one warning about measurements
        measurement_issues = [i for i in hr.issues if "measurement" in i.title.lower()]
        assert any(i.level in ("warn", "crit") for i in measurement_issues)

    def test_invalid_fs_not_blocked(self):
        """fs=0 → warning level (not critical in health_engine_issues)."""
        data = minimal_config("BASIC")
        data["fs"] = 0

        hr = compute_health(data, "BASIC")

        # Missing fs produces warn, not crit
        assert any(i.level == "warn" and "metric" in i.title.lower() for i in hr.issues)
        assert hr.blocked is False

    def test_correction_range_invalid_critical(self):
        """mag_c_min >= mag_c_max → critical."""
        data = minimal_config()
        data["mag_c_min"] = 100.0
        data["mag_c_max"] = 50.0

        hr = compute_health(data, "BASIC")

        assert any(i.level == "crit" and "range" in i.title.lower() for i in hr.issues)
        assert hr.blocked is True

    def test_leveling_range_invalid_critical(self):
        """lvl_min >= lvl_max → critical."""
        data = minimal_config()
        data["lvl_min"] = 90.0
        data["lvl_max"] = 60.0

        hr = compute_health(data, "BASIC")

        assert any(i.level == "crit" for i in hr.issues)
        assert hr.blocked is True


class TestComputeHealthBassIntegration:
    """Tests for bass integration health checks."""

    def test_bass_integration_off(self):
        """Bass integration off → uses standard measurement checks."""
        data = minimal_config()
        data["bass_integration_enable"] = False

        hr = compute_health(data, "BASIC")

        # Should check standard measurements, not bass integration
        assert any("measurement" in i.title.lower() for i in hr.issues)

    def test_bass_integration_auto_mode_only_critical(self):
        """Bass integration in BASIC mode → critical."""
        data = minimal_config("BASIC")
        data["bass_integration_enable"] = True

        hr = compute_health(data, "BASIC")

        assert any(i.level == "crit" and "auto" in i.title.lower() for i in hr.issues)

    def test_bass_integration_auto_mode_ok(self):
        """Bass integration in AUTO mode with valid config → issues present but OK."""
        data = minimal_config("AUTO")
        data["bass_integration_enable"] = True
        data["avr_crossover_hz"] = 80.0

        hr = compute_health(data, "AUTO")

        # May have crit due to missing measurement sources, but not auto-mode issue
        bass_issues = [i for i in hr.issues if "auto" in i.title.lower()]
        assert len(bass_issues) == 0 or all(i.level != "crit" for i in bass_issues)

    def test_bass_integration_invalid_hpf_critical(self):
        """Bass integration with invalid AVR HPF → critical."""
        data = minimal_config("AUTO")
        data["bass_integration_enable"] = True
        data["avr_crossover_hz"] = 0.0

        hr = compute_health(data, "AUTO")

        assert any(i.level == "crit" and "hpf" in i.title.lower() for i in hr.issues)

    def test_bass_integration_unusual_hpf_warn(self):
        """Bass integration with unusual AVR HPF (>250 Hz) → warn."""
        data = minimal_config("AUTO")
        data["bass_integration_enable"] = True
        data["avr_crossover_hz"] = 300.0

        hr = compute_health(data, "AUTO")

        assert any(i.level == "warn" and "hpf" in i.title.lower() for i in hr.issues)


class TestComputeHealthEngine:
    """Tests for engine (DSP) parameter checks."""

    def test_engine_valid_fs_taps(self):
        """Valid fs and taps → OK."""
        data = minimal_config()
        data["fs"] = 48000
        data["taps"] = 4096

        hr = compute_health(data, "BASIC")

        engine_issues = [i for i in hr.issues if "latency" in i.title.lower()]
        assert len(engine_issues) > 0
        assert engine_issues[0].level == "ok"

    def test_engine_high_taps_warn(self):
        """Taps > 150ms latency without short window → warn."""
        data = minimal_config()
        data["fs"] = 48000
        data["taps"] = 15000  # >150ms at 48kHz (150.25ms)
        data["ir_window_left"] = 200.0  # Not short (<120ms) - use correct key

        hr = compute_health(data, "BASIC")

        assert any(i.level == "warn" and "taps" in i.title.lower() for i in hr.issues)

    def test_engine_missing_metrics_warn(self):
        """Missing fs or taps → warn."""
        data = minimal_config()
        data["fs"] = None
        data["taps"] = 4096

        hr = compute_health(data, "BASIC")

        assert any(i.level == "warn" and "metric" in i.title.lower() for i in hr.issues)


class TestComputeHealthMixed:
    """Tests for mixed filter configuration checks."""

    def test_mixed_filter_off(self):
        """Mixed filter disabled → no mixed issues."""
        data = minimal_config()
        data["filter_type"] = "linear"

        hr = compute_health(data, "BASIC")

        mixed_issues = [i for i in hr.issues if "mixed" in i.title.lower()]
        assert len(mixed_issues) == 0

    def test_mixed_filter_invalid_freq_critical(self):
        """Mixed filter with invalid split freq → critical."""
        data = minimal_config()
        data["filter_type"] = "mixed"
        data["mixed_freq"] = 0.0
        data["trans_width"] = 10.0

        hr = compute_health(data, "BASIC")

        assert any(i.level == "crit" and "mixed" in i.title.lower() for i in hr.issues)

    def test_mixed_filter_low_freq_warn(self):
        """Mixed filter with low split freq (<40 Hz) → warn."""
        data = minimal_config()
        data["filter_type"] = "mixed"
        data["mixed_freq"] = 20.0
        data["trans_width"] = 10.0

        hr = compute_health(data, "BASIC")

        assert any(i.level == "warn" and "split" in i.title.lower() for i in hr.issues)


class TestFormatHealthSummary:
    """Tests for health result summary formatting."""

    def test_format_empty_issues(self):
        """No issues → empty string."""
        hr = HealthResult(overall="ok", blocked=False, issues=[])
        summary = format_health_summary(hr)

        assert summary == ""

    def test_format_warnings_only(self):
        """Warnings without errors → warnings summary."""
        issues = [
            Issue("warn", "Test Warning 1"),
            Issue("warn", "Test Warning 2"),
        ]
        hr = HealthResult(overall="warn", blocked=False, issues=issues)
        summary = format_health_summary(hr)

        assert "warning" in summary.lower()
        assert "Test Warning 1" in summary

    def test_format_errors_with_more(self):
        """Errors exceeding max_items → shows count of additional."""
        issues = [
            Issue("crit", "Error 1"),
            Issue("crit", "Error 2"),
            Issue("crit", "Error 3"),
            Issue("crit", "Error 4"),
        ]
        hr = HealthResult(overall="crit", blocked=True, issues=issues)
        summary = format_health_summary(hr, max_items=3)

        assert "Error 1" in summary
        assert "Error 2" in summary
        assert "Error 3" in summary
        assert "1 more" in summary or "more" in summary.lower()

    def test_format_errors_truncate(self):
        """Errors within max_items → no truncation."""
        issues = [
            Issue("crit", "Error 1"),
            Issue("crit", "Error 2"),
        ]
        hr = HealthResult(overall="crit", blocked=True, issues=issues)
        summary = format_health_summary(hr, max_items=3)

        assert "Error 1" in summary
        assert "Error 2" in summary
        assert "more" not in summary.lower()


class TestHealthDataValidation:
    """Tests for internal data validation helpers."""

    def test_as_float_valid(self):
        """Valid float conversion."""
        from src.decaycore.application.health_service import _as_float

        assert _as_float(3.14) == pytest.approx(3.14)
        assert _as_float("42.0") == pytest.approx(42.0)
        assert _as_float(100) == pytest.approx(100.0)

    def test_as_float_invalid(self):
        """Invalid inputs return None."""
        from src.decaycore.application.health_service import _as_float

        assert _as_float(None) is None
        assert _as_float("") is None
        assert _as_float("not a number") is None
        assert _as_float([]) is None

    def test_as_int_valid(self):
        """Valid int conversion."""
        from src.decaycore.application.health_service import _as_int

        assert _as_int(42) == 42
        assert _as_int("100") == 100
        assert _as_int(3.9) == 3

    def test_as_int_invalid(self):
        """Invalid inputs return None."""
        from src.decaycore.application.health_service import _as_int

        assert _as_int(None) is None
        assert _as_int("") is None
        assert _as_int("not an int") is None
        assert _as_int({}) is None

    def test_has_uploaded_file(self):
        """Test uploaded file detection."""
        from src.decaycore.application.health_service import _has_uploaded_file

        assert _has_uploaded_file({"filename": "test.wav", "content": b"RIFF"}) is True
        assert _has_uploaded_file({"name": "test.wav", "content": ""}) is False
        assert _has_uploaded_file({}) is False
        assert _has_uploaded_file(None) is False

    def test_has_wav_measurement_source(self):
        """Test WAV file source detection."""
        from src.decaycore.application.health_service import _has_wav_measurement_source

        # Uploaded WAV
        data = {"file": {"filename": "test.wav", "content": b"RIFF"}, "path": None}
        assert _has_wav_measurement_source(data, file_key="file", path_key="path") is True

        # Non-WAV upload
        data = {"file": {"filename": "test.txt", "content": b"text"}, "path": None}
        assert _has_wav_measurement_source(data, file_key="file", path_key="path") is False

        # No source
        data = {"file": None, "path": None}
        assert _has_wav_measurement_source(data, file_key="file", path_key="path") is False
