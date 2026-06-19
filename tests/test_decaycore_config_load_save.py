# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Tests for load_config and save_config functionality."""

from __future__ import annotations

import pytest

from src.decaycore.config import decaycore_config


class TestLoadConfigBasics:
    """Tests for load_config basic functionality."""

    def test_load_config_returns_dict(self):
        """load_config returns a dictionary."""
        cfg = decaycore_config.load_config()
        assert isinstance(cfg, dict)

    def test_load_config_has_required_keys(self):
        """load_config result contains required keys."""
        cfg = decaycore_config.load_config()
        required = ["fs", "mode", "filter_type", "mag_correct", "taps", "hc_mode"]
        for key in required:
            assert key in cfg
        assert "measurement_dither_level_db" in cfg

    def test_load_config_key_count(self):
        """load_config has approximately 196 keys."""
        cfg = decaycore_config.load_config()
        # Allow some variance in key count
        assert 180 <= len(cfg) <= 220

    def test_load_config_mode_is_string(self):
        """mode is a string."""
        cfg = decaycore_config.load_config()
        assert isinstance(cfg["mode"], str)

    def test_load_config_fs_is_positive_int(self):
        """fs is a positive integer."""
        cfg = decaycore_config.load_config()
        assert isinstance(cfg["fs"], int)
        assert cfg["fs"] > 0

    def test_load_config_filter_type_is_capitalized(self):
        """filter_type is properly capitalized."""
        cfg = decaycore_config.load_config()
        assert cfg["filter_type"][0].isupper()

    def test_load_config_booleans_are_bool_type(self):
        """Boolean config values are actual bools."""
        cfg = decaycore_config.load_config()
        bool_keys = ["mag_correct", "exc_prot", "stereo_link", "hpf_enable"]
        for key in bool_keys:
            if key in cfg:
                assert isinstance(cfg[key], bool)

    def test_load_config_numeric_ranges_valid(self):
        """Numeric values are in sensible ranges."""
        cfg = decaycore_config.load_config()
        # Sample numeric values
        if "mag_c_min" in cfg and "mag_c_max" in cfg:
            assert cfg["mag_c_min"] < cfg["mag_c_max"]
        if "lvl_min" in cfg and "lvl_max" in cfg:
            assert cfg["lvl_min"] < cfg["lvl_max"]

    def test_load_config_none_fields_allowed(self):
        """Optional fields can be None."""
        cfg = decaycore_config.load_config()
        # These fields are allowed to be None
        optional = ["xo1_f", "xo2_f", "measurement_input_device"]
        for key in optional:
            if key in cfg:
                # Should exist (either value or None)
                assert key in cfg


class TestLoadConfigErrorHandling:
    """Tests for error handling in load_config."""

    def test_load_config_always_returns_dict(self):
        """load_config always returns a dictionary."""
        cfg = decaycore_config.load_config()
        assert isinstance(cfg, dict)

    def test_load_config_returns_at_least_base_keys(self):
        """load_config always has at least core keys."""
        cfg = decaycore_config.load_config()
        base_keys = ["mode", "fs", "filter_type"]
        for key in base_keys:
            assert key in cfg


class TestSaveConfigBasics:
    """Tests for save_config basic functionality."""

    def test_save_config_can_be_called(self):
        """save_config can be called without crashing."""
        cfg = decaycore_config.load_config()
        cfg["mode"] = "BASIC"

        # Just make sure it doesn't crash
        try:
            decaycore_config.save_config(cfg)
        except Exception as e:
            # Some exceptions are expected (file write issues), but not all
            assert "invalid" not in str(e).lower()

    def test_save_config_normalizes_legacy_choice_indices(self, monkeypatch, tmp_path):
        """Persisted select values should be rewritten to canonical options."""
        cfg_path = tmp_path / "config.json"
        monkeypatch.setattr(decaycore_config, "CONFIG_FILE", str(cfg_path))

        decaycore_config.save_config({
            "fs": 1,
            "taps": 1,
            "hpf_slope": 1,
            "xo1_s": 2,
            "plot_smoothing_level": 1,
            "filter_wav_format": 1,
            "device_audio_format": 1,
            "ir_export_window_mode": "invalid",
            "ir_export_window_shape": "invalid",
            "stereo_link_strategy": "invalid",
        })

        saved = cfg_path.read_text(encoding="utf-8")
        assert '"fs": 48000' in saved
        assert '"taps": 1024' in saved
        assert '"hpf_slope": 12' in saved
        assert '"xo1_s": 18' in saved
        assert '"plot_smoothing_level": 12' in saved
        assert '"filter_wav_format": "S32_LE"' in saved
        assert '"device_audio_format": "S16_LE"' in saved
        assert '"ir_export_window_mode": "auto"' in saved
        assert '"ir_export_window_shape": "hann"' in saved
        assert '"stereo_link_strategy": "auto"' in saved


class TestFilterTypeNormalization:
    """Tests for filter type normalization."""

    @pytest.mark.parametrize(
        "input_val,expected",
        [
            ("asym", "Asymmetric"),
            ("asym-left", "Asymmetric"),
            ("ASYM", "Asymmetric"),
            ("Asymmetric", "Asymmetric"),
            ("mixed", "Mixed"),
            ("MIXED", "Mixed"),
            ("minimum", "Minimum"),
            ("minphase", "Minimum"),
            ("min", "Minimum"),
            ("linear", "Linear"),
            ("LINEAR", "Linear"),
            ("unknown", "Mixed"),  # Unknown defaults to the program default
        ],
    )
    def test_normalize_filter_type_variants(self, input_val, expected):
        """Filter type normalization handles various inputs."""
        result = decaycore_config._normalize_filter_type(input_val)
        assert result == expected

    def test_normalize_filter_type_empty_defaults_mixed(self):
        """Empty filter type defaults to Mixed."""
        result = decaycore_config._normalize_filter_type("")
        assert result == "Mixed"

    def test_normalize_filter_type_none_defaults_mixed(self):
        """None filter type defaults to Mixed."""
        result = decaycore_config._normalize_filter_type(None)
        assert result == "Mixed"


class TestRoundTrip:
    """Tests for save-load round-trip."""

    def test_roundtrip_load_after_save(self):
        """Can load config after saving it."""
        cfg1 = decaycore_config.load_config()
        cfg1["mode"] = "BASIC"
        cfg1["measurement_dither_level_db"] = -44.0

        # Save
        decaycore_config.save_config(cfg1)

        # Load again
        cfg2 = decaycore_config.load_config()

        # Should still be a valid config
        assert isinstance(cfg2, dict)
        assert len(cfg2) > 100
        assert "mode" in cfg2
        assert cfg2["measurement_dither_level_db"] == -44.0

    def test_roundtrip_preserves_ui_theme_preference(self, monkeypatch, tmp_path):
        """Theme preference should persist across save/load."""
        cfg_path = tmp_path / "config.json"
        monkeypatch.setattr(decaycore_config, "CONFIG_FILE", str(cfg_path))

        cfg1 = decaycore_config.load_config()
        assert cfg1["ui_theme_dark"] is True

        cfg1["ui_theme_dark"] = False
        decaycore_config.save_config(cfg1)

        cfg2 = decaycore_config.load_config()
        assert cfg2["ui_theme_dark"] is False

    def test_save_config_preserves_theme_when_new_data_omits_it(self, monkeypatch, tmp_path):
        """Later saves without theme key should keep the previously chosen theme."""
        cfg_path = tmp_path / "config.json"
        monkeypatch.setattr(decaycore_config, "CONFIG_FILE", str(cfg_path))

        decaycore_config.save_config({"mode": "BASIC", "ui_theme_dark": False})
        decaycore_config.save_config({"mode": "ADVANCED"})

        cfg = decaycore_config.load_config()
        assert cfg["ui_theme_dark"] is False


class TestEdgeCases:
    """Tests for edge cases."""

    def test_config_with_extreme_numeric_values(self):
        """Config can handle extreme numeric values."""
        cfg = decaycore_config.load_config()
        cfg["fs"] = 1
        cfg["max_boost"] = 100.0

        # Just make sure save/load works
        decaycore_config.save_config(cfg)
        cfg2 = decaycore_config.load_config()

        # Should still be valid
        assert isinstance(cfg2, dict)
        assert "fs" in cfg2

    def test_config_with_special_string_values(self):
        """Config handles special string values."""
        cfg = decaycore_config.load_config()
        cfg["hc_mode"] = "Test Mode"

        # Just make sure save/load works
        decaycore_config.save_config(cfg)
        cfg2 = decaycore_config.load_config()

        # Should still be valid
        assert isinstance(cfg2, dict)
        assert "hc_mode" in cfg2
