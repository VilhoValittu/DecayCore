"""Unit tests for config migration system in decaycore_config.py."""

import json

import decaycore.config.decaycore_config as decaycore_config


class TestConfigMigrations:
    """Test versioned migration system."""

    def test_no_migrations_needed_when_current(self, monkeypatch, tmp_path):
        """Config at current version should not trigger migrations."""
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(
            json.dumps({
                "_config_version": decaycore_config._CONFIG_CURRENT_VERSION,
                "filter_type": "Asymmetric",
            }),
            encoding="utf-8",
        )
        monkeypatch.setattr(decaycore_config, "CONFIG_FILE", str(cfg_path))

        loaded = decaycore_config.load_config()
        assert loaded["filter_type"] == "Asymmetric"

    def test_migration_v1_coerce_boolean_lists_from_old_config(self, monkeypatch, tmp_path):
        """Old configs with list-valued booleans should be converted to bool."""
        cfg_path = tmp_path / "config.json"
        # Simulate an old config with list-valued booleans
        cfg_path.write_text(
            json.dumps({
                # No _config_version means version 0, triggers migration
                "filter_type": "Asymmetric",
                "mode": "BASIC",  # Use BASIC to avoid AUTO mode overrides
                "mag_correct": [True],  # Was list in old format
                "multi_rate_ultra_high_opt": [True],
                "comparison_mode": [False],
                "bass_integration_enable": [],  # Empty list -> False
            }),
            encoding="utf-8",
        )
        monkeypatch.setattr(decaycore_config, "CONFIG_FILE", str(cfg_path))

        loaded = decaycore_config.load_config()
        assert isinstance(loaded["mag_correct"], bool)
        assert loaded["mag_correct"] is True
        assert isinstance(loaded["multi_rate_ultra_high_opt"], bool)
        assert loaded["multi_rate_ultra_high_opt"] is True
        assert isinstance(loaded["comparison_mode"], bool)
        assert loaded["comparison_mode"] is False
        assert isinstance(loaded["bass_integration_enable"], bool)
        assert loaded["bass_integration_enable"] is False

    def test_migration_v1_lvl_manual_db_shift(self, monkeypatch, tmp_path):
        """Old lvl_manual_db values in 40–110 range should shift by -75 dB."""
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(
            json.dumps({
                "filter_type": "Asymmetric",
                "lvl_manual_db": 80.0,  # Old range: 40–110
            }),
            encoding="utf-8",
        )
        monkeypatch.setattr(decaycore_config, "CONFIG_FILE", str(cfg_path))

        loaded = decaycore_config.load_config()
        assert loaded["lvl_manual_db"] == 5.0  # 80 - 75

    def test_migration_v1_lvl_manual_db_shift_boundary_low(self, monkeypatch, tmp_path):
        """Boundary value at low end: 40.0 should shift."""
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(
            json.dumps({
                "filter_type": "Asymmetric",
                "lvl_manual_db": 40.0,
            }),
            encoding="utf-8",
        )
        monkeypatch.setattr(decaycore_config, "CONFIG_FILE", str(cfg_path))

        loaded = decaycore_config.load_config()
        assert loaded["lvl_manual_db"] == -35.0  # 40 - 75

    def test_migration_v1_lvl_manual_db_shift_boundary_high(self, monkeypatch, tmp_path):
        """Boundary value at high end: 110.0 should shift."""
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(
            json.dumps({
                "filter_type": "Asymmetric",
                "lvl_manual_db": 110.0,
            }),
            encoding="utf-8",
        )
        monkeypatch.setattr(decaycore_config, "CONFIG_FILE", str(cfg_path))

        loaded = decaycore_config.load_config()
        assert loaded["lvl_manual_db"] == 35.0  # 110 - 75

    def test_migration_v1_lvl_manual_db_outside_range_not_shifted(self, monkeypatch, tmp_path):
        """Values outside 40–110 range should not be shifted."""
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(
            json.dumps({
                "filter_type": "Asymmetric",
                "lvl_manual_db": 5.0,  # New range: already relative, < 40
            }),
            encoding="utf-8",
        )
        monkeypatch.setattr(decaycore_config, "CONFIG_FILE", str(cfg_path))

        loaded = decaycore_config.load_config()
        assert loaded["lvl_manual_db"] == 5.0  # No shift

    def test_migration_v1_lvl_manual_db_high_value_not_shifted(self, monkeypatch, tmp_path):
        """High values (> 110) should not be shifted (already relative)."""
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(
            json.dumps({
                "filter_type": "Asymmetric",
                "lvl_manual_db": 150.0,
            }),
            encoding="utf-8",
        )
        monkeypatch.setattr(decaycore_config, "CONFIG_FILE", str(cfg_path))

        loaded = decaycore_config.load_config()
        assert loaded["lvl_manual_db"] == 150.0  # No shift

    def test_migration_version_updated_after_migration(self, monkeypatch, tmp_path):
        """After migration, _config_version should be updated to current."""
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(
            json.dumps({
                "filter_type": "Asymmetric",
                "mag_correct": [True],
            }),
            encoding="utf-8",
        )
        monkeypatch.setattr(decaycore_config, "CONFIG_FILE", str(cfg_path))

        loaded = decaycore_config.load_config()
        # _config_version should be in the loaded config (set by migration)
        assert loaded.get("_config_version") == decaycore_config._CONFIG_CURRENT_VERSION

    def test_migration_not_applied_twice(self, monkeypatch, tmp_path):
        """Migrations should not be applied twice if config already migrated."""
        cfg_path = tmp_path / "config.json"
        # First load: old config with list boolean
        cfg_path.write_text(
            json.dumps({
                "filter_type": "Asymmetric",
                "mag_correct": [True],
            }),
            encoding="utf-8",
        )
        monkeypatch.setattr(decaycore_config, "CONFIG_FILE", str(cfg_path))

        # First load triggers migration
        loaded1 = decaycore_config.load_config()
        assert loaded1["mag_correct"] is True

        # Manually reset to pre-migration state (version 0) with converted boolean
        cfg_path.write_text(
            json.dumps({
                "filter_type": "Asymmetric",
                "mag_correct": True,  # Already converted
                "_config_version": decaycore_config._CONFIG_CURRENT_VERSION,  # Mark as migrated
            }),
            encoding="utf-8",
        )

        # Second load should not re-migrate
        loaded2 = decaycore_config.load_config()
        assert loaded2["mag_correct"] is True

    def test_save_config_excludes_internal_version(self, monkeypatch, tmp_path):
        """_config_version should not be saved to disk."""
        cfg_path = tmp_path / "config.json"
        monkeypatch.setattr(decaycore_config, "CONFIG_FILE", str(cfg_path))

        data = {
            "filter_type": "Asymmetric",
            "mode": "AUTO",
            "_config_version": 999,  # Internal marker
        }
        decaycore_config.save_config(data)

        saved = json.loads(cfg_path.read_text(encoding="utf-8"))
        assert "_config_version" not in saved
        assert saved["filter_type"] == "Asymmetric"

    def test_combined_migrations_in_old_config(self, monkeypatch, tmp_path):
        """Multiple migrations should be applied together."""
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(
            json.dumps({
                "filter_type": "Asymmetric",
                "mag_correct": [True],  # Migration 1: coerce boolean
                "lvl_manual_db": 85.0,  # Migration 1: shift
            }),
            encoding="utf-8",
        )
        monkeypatch.setattr(decaycore_config, "CONFIG_FILE", str(cfg_path))

        loaded = decaycore_config.load_config()
        assert loaded["mag_correct"] is True
        assert loaded["lvl_manual_db"] == 10.0  # 85 - 75
        assert loaded["_config_version"] == decaycore_config._CONFIG_CURRENT_VERSION

    def test_load_config_normalizes_legacy_choice_indices(self, monkeypatch, tmp_path):
        """Old configs with indexed select values should map to current choice values."""
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(
            json.dumps({
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
            }),
            encoding="utf-8",
        )
        monkeypatch.setattr(decaycore_config, "CONFIG_FILE", str(cfg_path))

        loaded = decaycore_config.load_config()

        assert loaded["fs"] == 48000
        assert loaded["taps"] == 1024
        assert loaded["hpf_slope"] == 12
        assert loaded["xo1_s"] == 18
        assert loaded["plot_smoothing_level"] == 12
        assert loaded["filter_wav_format"] == "S32_LE"
        assert loaded["device_audio_format"] == "S16_LE"
        assert loaded["ir_export_window_mode"] == "auto"
        assert loaded["ir_export_window_shape"] == "hann"
        assert loaded["stereo_link_strategy"] == "auto"
