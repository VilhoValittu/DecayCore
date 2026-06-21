from __future__ import annotations

import json

import pytest

from decaycore.engine import build_config, build_config_from_snapshot
from decaycore.config import decaycore_config
from decaycore.config.mode_policy import MODE_CLAMPS, MODE_DEFAULTS
from decaycore.config.pipeline_parts.ui_data import collect_ui_config, collect_ui_data
from decaycore.config.schema import (
    LIST_BOOL_KEYS,
    MODE_CLAMPS as SCHEMA_MODE_CLAMPS,
    MODE_DEFAULTS as SCHEMA_MODE_DEFAULTS,
    REQUEST_RUNTIME_DEFAULTS,
    AppConfigSnapshot,
    RunConfigSnapshot,
    default_config_dict,
    normalize_flat_config,
    normalize_list_backed_booleans,
)


def test_schema_default_snapshot_contains_core_groups() -> None:
    snapshot = decaycore_config.load_config_snapshot()
    data = snapshot.to_flat_dict()

    assert isinstance(snapshot, AppConfigSnapshot)
    for key in (
        "fs",
        "taps",
        "filter_type",
        "mode",
        "mag_c_min",
        "mag_c_max",
        "auto_goal",
        "measurement_dither_level_db",
    ):
        assert key in data


def test_schema_normalizes_legacy_booleans_choices_and_filter_type() -> None:
    data = default_config_dict()
    data.update(
        {
            "mag_correct": [False],
            "fs": 1,
            "hpf_slope": "2",
            "filter_type": "minphase",
            "ir_export_window_shape": "invalid",
        }
    )
    normalize_list_backed_booleans(data)
    normalized = normalize_flat_config(data)

    assert "mag_correct" in LIST_BOOL_KEYS
    assert normalized["mag_correct"] is False
    assert normalized["fs"] == 48000
    assert normalized["hpf_slope"] == 18
    assert normalized["filter_type"] == "Minimum"
    assert normalized["ir_export_window_shape"] == "hann"


def test_save_config_keeps_runtime_only_fields_out_of_persisted_json(monkeypatch, tmp_path) -> None:
    cfg_path = tmp_path / "config.json"
    monkeypatch.setattr(decaycore_config, "CONFIG_FILE", str(cfg_path))

    data = {"mode": "AUTO", "filter_type": "mixed", "unsafe_raw_dsp": True, "_config_version": 99}
    data.update(REQUEST_RUNTIME_DEFAULTS)
    decaycore_config.save_config(data)

    saved = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert saved["mode"] == "AUTO"
    assert saved["filter_type"] == "Mixed"
    assert "unsafe_raw_dsp" not in saved
    assert "_config_version" not in saved
    for key in REQUEST_RUNTIME_DEFAULTS:
        assert key not in saved


def test_mode_policy_exports_are_schema_derived() -> None:
    assert MODE_DEFAULTS == SCHEMA_MODE_DEFAULTS
    assert MODE_CLAMPS == SCHEMA_MODE_CLAMPS
    assert MODE_DEFAULTS["AUTO"]["stereo_link_strategy"] == "auto"


def test_collect_ui_config_matches_legacy_collect_ui_data() -> None:
    pin = {
        "mode": "ADVANCED",
        "filter_type": "Mixed",
        "mixed_freq": 180.0,
        "lvl_mode": "Manual",
        "lvl_algo": "Median",
    }

    assert collect_ui_config(pin).to_flat_dict() == collect_ui_data(pin)


@pytest.mark.parametrize(
    ("mode", "filter_type", "bass_integration_enable"),
    [
        ("BASIC", "Mixed", False),
        ("AUTO", "Mixed", False),
        ("ADVANCED", "Asymmetric", False),
        ("AUTO", "Mixed", True),
    ],
)
def test_build_config_from_snapshot_matches_legacy_build_config(mode, filter_type, bass_integration_enable) -> None:
    data = decaycore_config.load_config()
    data.update(
        {
            "mode": mode,
            "camillafir_automatic_mode": mode == "AUTO",
            "filter_type": filter_type,
            "bass_integration_enable": bass_integration_enable,
            "bass_integration_mode": "direct_dac",
            "mixed_freq": 180.0,
            "mag_c_min": 20.0,
            "mag_c_max": 220.0,
            "max_boost": 4.0,
            "max_cut_db": 15.0,
        }
    )
    kwargs = {
        "fs_v": 44100,
        "taps_v": 4096,
        "xos": [],
        "hpf": {"enabled": True, "freq": 25.0, "order": 2},
    }

    legacy = build_config(data, **kwargs)
    typed = build_config_from_snapshot(RunConfigSnapshot(values=dict(data)), **kwargs)

    for attr in (
        "fs",
        "num_taps",
        "filter_type_str",
        "mixed_split_freq",
        "mag_c_min",
        "mag_c_max",
        "max_boost_db",
        "max_cut_db",
        "bass_integration_enable",
    ):
        assert getattr(typed, attr) == getattr(legacy, attr)
