import logging
import json
from types import SimpleNamespace

import decaycore.auto_mode.orchestrator_finalize as orchestrator_finalize
import decaycore.auto_mode.orchestrator_refine as orchestrator_refine
import decaycore.auto_mode.orchestrator_target as orchestrator_target
import decaycore.auto_mode.cache_signature as cache_signature
import numpy as np
import pytest
from decaycore.application.house_curve_service import load_house_curve
from decaycore.config.decaycore_config import load_config
from decaycore.config.decaycore_pipeline import build_filter_config
from decaycore.config.decaycore_pipeline import build_xos_hpf
from decaycore.config.decaycore_pipeline import collect_ui_data
from decaycore.config.mode_policy import apply_mode_to_cfg
from decaycore.config.models import FilterConfig
from decaycore.auto_mode.cache_measurement_sig import (
    _auto_get_measurement_signature,
    _auto_measurement_metadata_identity,
    _auto_target_study_sig,
)
from decaycore.auto_mode.cache_signature import _auto_cache_empty, _auto_signature, _auto_signature_payload
from decaycore.auto_mode.search_v2.input_model import build_auto_search_input
from decaycore.auto_mode.search_v2.plan import AutoSearchPlan
from decaycore.auto_mode.search_v2.planner import determine_auto_search_plan
from decaycore.auto_mode.search_v2.signature import compute_auto_search_signature
from decaycore.auto_mode.shared import AUTO_MODE_CACHE_SCHEMA_VERSION
from decaycore.engine_build import build_config
from decaycore.auto_mode.orchestrator_target import (
    _TargetTrialSetup,
    _run_target_phase1_trials,
    _target_eval_one,
)
from decaycore.auto_mode.api import _auto_select_target_curve_with_trials
from decaycore.auto_mode.api import _run_auto_mode_search
from decaycore.workflow.auto_flow_parts.seed_phases import _try_cached_target_pick_before_search


def test_auto_mode_search_ignores_v2_disable_flags(monkeypatch):
    import decaycore.auto_mode.api as auto_api

    search_entrypoints = auto_api._search_entrypoints
    calls = []

    def _run_v2_probe(**kwargs):
        calls.append(dict(kwargs))
        return {"ok": True}

    monkeypatch.setattr(search_entrypoints, "_run_auto_search_v2", _run_v2_probe)

    result = _run_auto_mode_search(
        base_data={
            "use_auto_search_v2": False,
            "auto_mode_use_auto_search_v2": False,
        },
        measurements={"f_l": [20.0], "m_l": [0.0]},
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_f=[20.0],
        hc_m=[0.0],
        pin_obj=None,
        status_cb=None,
        n_trials=3,
    )

    assert result == {"ok": True}
    assert len(calls) == 1
    assert calls[0]["base_data"]["use_auto_search_v2"] is False
    assert calls[0]["base_data"]["auto_mode_use_auto_search_v2"] is False
    assert calls[0]["n_trials"] == 3


def test_auto_mode_search_impl_compat_shim_uses_v2(monkeypatch):
    import decaycore.auto_mode.api as auto_api

    search_entrypoints = auto_api._search_entrypoints
    calls = []

    def _run_v2_probe(**kwargs):
        calls.append(dict(kwargs))
        return {"ok": True}

    monkeypatch.setattr(search_entrypoints, "_run_auto_search_v2", _run_v2_probe)

    result = auto_api._run_auto_mode_search_impl(
        base_data={"use_auto_search_v2": False},
        measurements={"f_l": [20.0], "m_l": [0.0]},
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_f=[20.0],
        hc_m=[0.0],
        pin_obj=None,
        status_cb=None,
        n_trials=2,
    )

    assert result == {"ok": True}
    assert len(calls) == 1
    assert calls[0]["base_data"]["use_auto_search_v2"] is False
    assert calls[0]["n_trials"] == 2


def test_collect_ui_data_normalizes_auto_target_mode_to_auto_by_default():
    pin = {
        "mode": "AUTO",
        "camillafir_automatic_mode": True,
        "auto_target_mode": "invalid-value",
    }
    data = collect_ui_data(pin)
    assert str(data.get("auto_target_mode")) == "auto"


def test_load_config_defaults_auto_target_mode_to_auto_on_first_launch(monkeypatch, tmp_path):
    import decaycore.config.decaycore_config as decaycore_config

    monkeypatch.setattr(decaycore_config, "CONFIG_FILE", str(tmp_path / "missing_config.json"))

    data = decaycore_config.load_config()

    assert str(data.get("auto_target_mode")) == "auto"
    assert bool(data.get("hpf_enable")) is True


def test_collect_ui_data_accepts_selected_aliases_for_auto_target_mode():
    pin = {
        "mode": "AUTO",
        "camillafir_automatic_mode": True,
        "auto_target_mode": "manual",
    }
    data = collect_ui_data(pin)
    assert str(data.get("auto_target_mode")) == "selected"


def test_collect_ui_data_preserves_explicit_auto_target_mode():
    pin = {
        "mode": "AUTO",
        "camillafir_automatic_mode": True,
        "auto_target_mode": "auto",
    }
    data = collect_ui_data(pin)
    assert str(data.get("auto_target_mode")) == "auto"


def test_collect_ui_data_preserves_explicit_adaptive_auto_target_mode():
    pin = {
        "mode": "AUTO",
        "camillafir_automatic_mode": True,
        "auto_target_mode": "adaptive",
    }
    data = collect_ui_data(pin)
    assert str(data.get("auto_target_mode")) == "adaptive"


def test_collect_ui_data_forces_selected_target_for_prefer_bass():
    pin = {
        "mode": "AUTO",
        "camillafir_automatic_mode": True,
        "auto_goal": "prefer bass",
        "auto_target_mode": "auto",
    }
    data = collect_ui_data(pin)
    assert str(data.get("auto_target_mode")) == "selected"


def test_collect_ui_data_normalizes_layout_from_stable_ui_keys():
    data = collect_ui_data({"layout": "stereo"})
    assert str(data.get("layout")) == "stereo"

    data = collect_ui_data({"layout": "mono"})
    assert str(data.get("layout")) == "mono"


def test_collect_ui_data_normalizes_legacy_layout_labels_to_stable_keys():
    data = collect_ui_data({"layout": "Stereo"})
    assert str(data.get("layout")) == "stereo"

    data = collect_ui_data({"layout": "Mono"})
    assert str(data.get("layout")) == "mono"


def test_collect_ui_data_normalizes_level_mode_and_algo_to_stable_keys():
    data = collect_ui_data({"mode": "ADVANCED", "lvl_mode": "Manual", "lvl_algo": "Average"})
    assert str(data.get("lvl_mode")) == "manual"
    assert str(data.get("lvl_algo")) == "average"


def test_collect_ui_data_auto_mode_preserves_allowed_inputs_but_forces_managed_settings(monkeypatch):
    monkeypatch.setattr(
        "decaycore.config.decaycore_pipeline.get_auto_mode_filter_auto_defaults",
        lambda filter_type: {
            "filter_type_str": str(filter_type),
            "phase_limit": 407.2,
            "enable_tdc": True,
            "enable_afdw": True,
            "filter_smooth": 96,
            "max_boost_db": 4.11,
            "mixed_split_freq": 177.3,
            "comparison_mode": True,
        },
    )

    pin = {
        "mode": "AUTO",
        "camillafir_automatic_mode": True,
        "filter_type": "Mixed Phase",
        "auto_goal": "low-ripple",
        "auto_target_mode": "selected",
        "hc_mode": "Cinema",
        "fs": 96000,
        "taps": 131072,
        "multi_rate_opt": [True],
        "gain": 9.5,
        "phase_limit": 999.0,
        "enable_tdc": [],
        "enable_afdw": [],
        "filter_smooth": 96,
        "max_boost": 12.0,
        "comparison_mode": [],
        "hpf_enable": [],
        "hpf_freq": 27.5,
        "hpf_slope": 18,
        "xo1_f": 80.0,
        "xo1_s": 24,
        "xo2_f": 2200.0,
        "xo2_s": 12,
    }

    data = collect_ui_data(pin)

    assert str(data.get("filter_type")) == "Mixed Phase"
    assert str(data.get("auto_goal")) == "low-ripple"
    assert str(data.get("auto_target_mode")) == "selected"
    assert str(data.get("hc_mode")) == "Cinema"
    assert int(data.get("fs")) == 96000
    assert int(data.get("taps")) == 131072
    assert bool(data.get("multi_rate_opt")) is True
    assert float(data.get("phase_limit")) == 407.2
    assert bool(data.get("enable_tdc")) is True
    assert bool(data.get("enable_afdw")) is False
    assert int(data.get("filter_smooth")) == 96
    assert float(data.get("max_boost")) == 4.11
    assert float(data.get("mixed_freq")) == 177.3
    assert bool(data.get("comparison_mode")) is True
    assert bool(data.get("hpf_enable")) is True
    assert bool(data.get("low_bass_cut_enable")) is False
    assert float(data.get("hpf_freq")) == 27.5
    assert int(data.get("hpf_slope")) == 18
    assert float(data.get("gain")) == 0.10
    assert float(data.get("xo1_f")) == 80.0
    assert int(data.get("xo1_s")) == 24
    assert float(data.get("xo2_f")) == 2200.0
    assert int(data.get("xo2_s")) == 12
    assert str(data.get("output_tilt_source")) == "off"
    assert float(data.get("output_tilt_db_per_oct")) == 0.0


@pytest.mark.parametrize(
    "filter_type",
    ["Linear Phase", "Minimum Phase", "Mixed Phase", "Asymmetric"],
)
def test_collect_ui_data_auto_mode_does_not_force_filter_type_to_asymmetric(filter_type):
    data = collect_ui_data(
        {
            "mode": "AUTO",
            "camillafir_automatic_mode": True,
            "filter_type": filter_type,
        }
    )

    assert str(data.get("filter_type")) == filter_type


def test_collect_ui_data_trusts_explicit_advanced_mode_over_legacy_auto_flag():
    data = collect_ui_data(
        {
            "mode": "ADVANCED",
            "camillafir_automatic_mode": True,
            "auto_goal": "flat",
            "unsafe_raw_dsp": False,
        }
    )

    assert str(data.get("mode")) == "ADVANCED"
    assert bool(data.get("camillafir_automatic_mode")) is False
    assert bool(data.get("unsafe_raw_dsp")) is False


def test_load_config_trusts_saved_advanced_mode_over_legacy_auto_flag(monkeypatch, tmp_path):
    import decaycore.config.decaycore_config as decaycore_config

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "mode": "ADVANCED",
                "camillafir_automatic_mode": True,
                "unsafe_raw_dsp": False,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(decaycore_config, "CONFIG_FILE", str(config_path))

    data = decaycore_config.load_config()

    assert str(data.get("mode")) == "ADVANCED"
    assert bool(data.get("camillafir_automatic_mode")) is False
    assert bool(data.get("unsafe_raw_dsp")) is False


def test_low_bass_cut_policy_by_mode():
    basic = apply_mode_to_cfg(FilterConfig(low_bass_cut_enable=False), "BASIC")
    auto = apply_mode_to_cfg(FilterConfig(low_bass_cut_enable=True), "AUTO")
    advanced = apply_mode_to_cfg(FilterConfig(low_bass_cut_enable=True), "ADVANCED")

    assert bool(basic.low_bass_cut_enable) is True
    assert bool(auto.low_bass_cut_enable) is False
    assert bool(advanced.low_bass_cut_enable) is False


def test_basic_mode_does_not_force_afdw_on():
    cfg = apply_mode_to_cfg(FilterConfig(enable_afdw=False), "BASIC", apply_defaults=False)

    assert bool(cfg.enable_afdw) is False


def test_build_xos_hpf_forces_hpf_enabled_in_auto_mode():
    _xos, hpf = build_xos_hpf(
        {
            "mode": "AUTO",
            "camillafir_automatic_mode": True,
            "filter_type": "Asymmetric",
            "hpf_enable": False,
            "hpf_freq": 27.5,
            "hpf_slope": 18,
        }
    )

    assert isinstance(hpf, dict)
    assert bool(hpf.get("enabled", False)) is True
    assert float(hpf.get("freq", 0.0)) == 27.5
    assert int(hpf.get("order", 0)) == 3


def test_build_config_honors_auto_hpf_runtime_override_off():
    data = load_config()
    data.update(
        {
            "mode": "AUTO",
            "camillafir_automatic_mode": True,
            "bass_integration_enable": False,
            "auto_goal": "balanced",
            "filter_type": "Asymmetric",
            "hpf_enable": True,
            "hpf_freq": 27.5,
            "hpf_slope": 18,
            "_auto_hpf_runtime_override": {"enabled": False, "freq": 27.5, "order": 3},
        }
    )

    cfg = build_config(
        data,
        fs_v=44100,
        taps_v=4096,
        xos=[],
        hpf={"enabled": True, "freq": 27.5, "order": 3},
    )

    assert isinstance(getattr(cfg, "hpf_settings", None), dict)
    assert bool(cfg.hpf_settings.get("enabled", True)) is False
    assert float(cfg.hpf_settings.get("freq", 0.0)) == pytest.approx(27.5, abs=1e-9)
    assert int(cfg.hpf_settings.get("order", 0)) == 3


def test_build_config_forces_hpf_on_for_prefer_bass_runtime_override_off():
    data = load_config()
    data.update(
        {
            "mode": "AUTO",
            "camillafir_automatic_mode": True,
            "bass_integration_enable": False,
            "auto_goal": "prefer bass",
            "filter_type": "Asymmetric",
            "hpf_enable": False,
            "hpf_freq": 27.5,
            "hpf_slope": 18,
            "_auto_hpf_runtime_override": {"enabled": False, "freq": 27.5, "order": 3},
        }
    )

    cfg = build_config(
        data,
        fs_v=44100,
        taps_v=4096,
        xos=[],
        hpf={"enabled": True, "freq": 27.5, "order": 3},
    )

    assert isinstance(getattr(cfg, "hpf_settings", None), dict)
    assert bool(cfg.hpf_settings.get("enabled", False)) is True
    assert float(cfg.hpf_settings.get("freq", 0.0)) == pytest.approx(27.5, abs=1e-9)
    assert int(cfg.hpf_settings.get("order", 0)) == 3


def test_prefer_bass_house_curve_boost_stops_at_hpf_frequency():
    hc_f, hc_m, _source = load_house_curve(
        {
            "mode": "AUTO",
            "camillafir_automatic_mode": True,
            "auto_goal": "prefer bass",
            "hc_mode": "Harman10",
            "hpf_freq": 27.5,
            "hpf_slope": 18,
        }
    )

    assert float(np.interp(20.0, hc_f, hc_m)) == pytest.approx(0.0, abs=1e-9)
    assert float(np.interp(25.0, hc_f, hc_m)) == pytest.approx(0.0, abs=1e-9)
    assert float(np.interp(27.5, hc_f, hc_m)) == pytest.approx(0.0, abs=1e-9)
    assert float(np.interp(31.5, hc_f, hc_m)) > 0.0


def test_balanced_house_curve_keeps_original_bass_sentinel():
    hc_f, hc_m, _source = load_house_curve(
        {
            "mode": "AUTO",
            "camillafir_automatic_mode": True,
            "auto_goal": "balanced",
            "hc_mode": "Harman10",
            "hpf_freq": 27.5,
            "hpf_slope": 18,
        }
    )

    assert float(np.interp(0.0, hc_f, hc_m)) == pytest.approx(10.0, abs=1e-9)
    assert float(np.interp(20.0, hc_f, hc_m)) == pytest.approx(10.0, abs=1e-9)


def test_build_xos_hpf_ignores_auto_hpf_runtime_override_for_direct_dac():
    _xos, hpf = build_xos_hpf(
        {
            "mode": "AUTO",
            "camillafir_automatic_mode": True,
            "filter_type": "Asymmetric",
            "bass_integration_enable": True,
            "bass_integration_mode": "direct_dac",
            "sub_crossover_hz": 80.0,
            "sub_crossover_slope": 24,
            "_auto_hpf_runtime_override": {"enabled": False, "freq": 27.5, "order": 3},
        }
    )

    assert hpf is None


def test_build_filter_config_auto_mode_does_not_crash_and_uses_locked_data_values():
    class _Pin:
        def __init__(self):
            self._d = {
                "enable_tdc": [],
                "enable_afdw": [],
                "filter_smooth": 96,
                "df_smoothing": [True],
            }

        def get(self, key, default=None):
            return self._d.get(key, default)

        def __getitem__(self, key):
            if key in self._d:
                return self._d[key]
            raise KeyError(key)

    data = load_config()
    data.update(
        {
            "mode": "AUTO",
            "camillafir_automatic_mode": True,
            "filter_type": "Linear Phase",
            "mixed_freq": 180.0,
            "mag_c_min": 15.0,
            "mag_c_max": 196.1,
            "max_boost": 3.22,
            "phase_limit": 448.0,
            "mag_correct": True,
            "reg_strength": 22.5,
            "normalize_opt": False,
            "exc_prot": True,
            "exc_freq": 24.0,
            "low_bass_cut_hz": 18.0,
            "low_bass_cut_enable": True,
            "ir_window_right": 500.0,
            "ir_window_left": 85.0,
            "lvl_manual_db": 0.0,
            "lvl_min": 200.0,
            "lvl_max": 3000.0,
            "lvl_algo": "Median",
            "trans_width": 139.2,
            "enable_tdc": True,
            "enable_afdw": True,
            "tdc_strength": 63.9,
            "tdc_max_reduction_db": 23.7,
            "tdc_slope_db_per_oct": 12.0,
            "filter_smooth": 96,
            "df_smoothing": False,
        }
    )

    cfg = build_filter_config(
        FilterConfig_cls=FilterConfig,
        fs_v=44100,
        taps_v=65536,
        data=data,
        xos=[],
        hpf=None,
        hc_f=None,
        hc_m=None,
        pin=_Pin(),
    )

    assert bool(getattr(cfg, "enable_tdc", False)) is True
    assert bool(getattr(cfg, "enable_afdw", False)) is True
    assert int(getattr(cfg, "filter_smooth", 0)) == 96
    assert bool(getattr(cfg, "df_smoothing", True)) is False


def test_collect_ui_data_preserves_mixed_phase_degree_budgets():
    data = collect_ui_data(
        {
            "mixed_phase_budget_lf_deg": 75.0,
            "mixed_phase_budget_hf_deg": 30.0,
        }
    )

    assert float(data["mixed_phase_budget_lf_deg"]) == 75.0
    assert float(data["mixed_phase_budget_hf_deg"]) == 30.0


def test_build_filter_config_reads_mixed_phase_degree_budgets():
    class _Pin:
        def get(self, key, default=None):
            return default

    data = load_config()
    data.update(
        {
            "mode": "ADVANCED",
            "filter_type": "Mixed Phase",
            "mixed_freq": 180.0,
            "phase_limit": 400.0,
            "mixed_phase_budget_lf_deg": 75.0,
            "mixed_phase_budget_hf_deg": 30.0,
        }
    )

    cfg = build_filter_config(
        FilterConfig_cls=FilterConfig,
        fs_v=44100,
        taps_v=65536,
        data=data,
        xos=[],
        hpf=None,
        hc_f=None,
        hc_m=None,
        pin=_Pin(),
    )

    assert float(getattr(cfg, "mixed_phase_budget_lf_deg")) == 75.0
    assert float(getattr(cfg, "mixed_phase_budget_hf_deg")) == 30.0


def test_auto_target_curve_selection_uses_exact_signature_cache_without_recomputing(monkeypatch):
    f = [20.0, 100.0, 1000.0, 10000.0]
    m = [0.0, 0.0, 0.0, 0.0]

    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode.get_house_curve_by_name",
        lambda name: (f, m),
    )
    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_cache_get_target_for_measurements",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_cache_get_entry",
        lambda *args, **kwargs: {
            "best_target_curve": "Harman8",
            "best_preset": {"preset_id": "signature-final"},
            "target_seed_preset": {
                "preset_id": "signature-seed",
                "max_slope_boost_db_per_oct": 6.0,
                "max_slope_cut_db_per_oct": 24.0,
                "conf_pull_max_hz": 130.0,
            },
        },
    )

    def _unexpected_quick_preselect(*args, **kwargs):
        raise AssertionError("quick target preselect should be skipped on exact cache hit")

    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_select_builtin_target_curve",
        _unexpected_quick_preselect,
    )

    result = _auto_select_target_curve_with_trials(
        base_data={
            "filter_type": "Asymmetric",
            "auto_goal": "balanced",
            "auto_target_mode": "selected",
            "hc_mode": "Harman8",
            "program_version": "test-version",
        },
        measurements={
            "f_l": f,
            "m_l": m,
            "f_r": f,
            "m_r": m,
        },
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        pin_obj=None,
        status_cb=None,
    )

    assert result is not None
    assert str(result.get("selected_hc_mode")) == "Harman8"
    assert str(result.get("selection_method")) == "cache_signature_hit"
    assert int(result.get("top_n", -1)) == 0
    assert int(result.get("trials_per_curve", -1)) == 0
    assert dict(result.get("best_preset", {}) or {}).get("preset_id") == "signature-seed"
    assert dict(result.get("best_preset", {}) or {}).get("max_slope_boost_db_per_oct") == 6.0
    assert dict(result.get("best_preset", {}) or {}).get("max_slope_cut_db_per_oct") == 24.0
    assert dict(result.get("best_preset", {}) or {}).get("conf_pull_max_hz") == 130.0


def test_auto_target_curve_selection_uses_exact_measurement_cache_without_recomputing(monkeypatch):
    f = [20.0, 100.0, 1000.0, 10000.0]
    m = [0.0, 0.0, 0.0, 0.0]

    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode.get_house_curve_by_name",
        lambda name: (f, m),
    )
    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_cache_get_target_for_measurements",
        lambda *args, **kwargs: {
            "best_target_curve": "Harman6",
            "best_preset": {"preset_id": "measurement-final"},
            "target_seed_preset": {
                "preset_id": "measurement-seed",
                "max_slope_boost_db_per_oct": 6.0,
                "max_slope_cut_db_per_oct": 24.0,
                "conf_pull_max_hz": 130.0,
            },
        },
    )
    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_cache_get_entry",
        lambda *args, **kwargs: None,
    )

    def _unexpected_quick_preselect(*args, **kwargs):
        raise AssertionError("quick target preselect should be skipped on measurement cache hit")

    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_select_builtin_target_curve",
        _unexpected_quick_preselect,
    )

    result = _auto_select_target_curve_with_trials(
        base_data={
            "filter_type": "Asymmetric",
            "auto_goal": "balanced",
            "auto_target_mode": "selected",
            "hc_mode": "Harman6",
            "program_version": "test-version",
        },
        measurements={
            "f_l": f,
            "m_l": m,
            "f_r": f,
            "m_r": m,
        },
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        pin_obj=None,
        status_cb=None,
    )

    assert result is not None
    assert str(result.get("selected_hc_mode")) == "Harman6"
    assert str(result.get("selection_method")) == "cache_measurement_hit"
    assert int(result.get("top_n", -1)) == 0
    assert int(result.get("trials_per_curve", -1)) == 0
    assert dict(result.get("best_preset", {}) or {}).get("preset_id") == "measurement-seed"
    assert dict(result.get("best_preset", {}) or {}).get("max_slope_boost_db_per_oct") == 6.0
    assert dict(result.get("best_preset", {}) or {}).get("max_slope_cut_db_per_oct") == 24.0
    assert dict(result.get("best_preset", {}) or {}).get("conf_pull_max_hz") == 130.0


def test_auto_target_measurement_cache_hit_bypasses_comparison_in_auto_mode(monkeypatch):
    """Same measurements with a complete cached seed must reuse the selected target."""
    f = [20.0, 100.0, 1000.0, 10000.0]
    m = [0.0, 0.0, 0.0, 0.0]
    cached_entry = {
        "best_target_curve": "Harman8",
        "best_metrics": {"rank_score": 91.0, "avg_score": 88.0},
        "target_seed_preset": {
            "preset_id": "measurement-seed",
            "max_slope_boost_db_per_oct": 6.0,
            "max_slope_cut_db_per_oct": 24.0,
            "conf_pull_max_hz": 130.0,
        },
    }
    runtime = SimpleNamespace(
        get_house_curve_by_name=lambda name: (f, m),
        auto_cache_get_target_for_measurements=lambda *args, **kwargs: dict(cached_entry),
        auto_cache_get_entry=lambda *args, **kwargs: None,
        auto_optuna_module_ready=lambda *args, **kwargs: False,
    )
    setup = SimpleNamespace(
        runtime=runtime,
        goal="balanced",
        filter_key="minimum",
        compat_version="test-version",
        optimizer_backend="builtin",
        cfg=SimpleNamespace(optuna_persistent_study=False),
        optuna_mod=None,
    )

    state = orchestrator_target._resolve_cached_target_state(
        setup=setup,
        base_data={"auto_target_mode": "auto", "hc_mode": "Harman10"},
        measurements={"f_l": f, "m_l": m, "f_r": f, "m_r": m},
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        status_cb=None,
    )

    assert str(state.cached_target_source or "") == "cache_measurement"
    assert str(state.cached_target_hc or "") == "Harman8"
    assert dict(state.cached_target_preset or {}).get("preset_id") == "measurement-seed"

    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode.get_house_curve_by_name",
        lambda name: (f, m),
    )
    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_cache_get_target_for_measurements",
        lambda *args, **kwargs: dict(cached_entry),
    )
    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_cache_get_entry",
        lambda *args, **kwargs: None,
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("quick target preselect should be skipped on measurement cache hit")

    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_select_builtin_target_curve",
        fail_if_called,
    )
    result = _auto_select_target_curve_with_trials(
        base_data={
            "filter_type": "Asymmetric",
            "auto_goal": "balanced",
            "auto_target_mode": "auto",
            "hc_mode": "Harman10",
            "program_version": "test-version",
        },
        measurements={
            "f_l": f,
            "m_l": m,
            "f_r": f,
            "m_r": m,
        },
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        pin_obj=None,
        status_cb=None,
    )

    assert result is not None
    assert str(result.get("selection_method")) == "cache_measurement_hit"
    assert str(result.get("selected_hc_mode")) == "Harman8"
    assert int(result.get("top_n", -1)) == 0
    assert int(result.get("trials_per_curve", -1)) == 0
    assert dict(result.get("best_preset", {}) or {}).get("preset_id") == "measurement-seed"
    assert dict(result.get("best_metrics", {}) or {}).get("rank_score") == 91.0


def test_auto_target_global_cache_without_current_filter_seed_falls_through_to_filter_cache(monkeypatch):
    f = [20.0, 100.0, 1000.0, 10000.0]
    m = [0.0, 0.0, 0.0, 0.0]
    cached_entry = {
        "best_target_curve": "Harman8",
        "filter_seed_presets": {
            "asym": {"preset_id": "asym-seed"},
        },
        "filter_seed_metrics": {
            "asym": {"rank_score": 91.0},
        },
    }
    filter_entry = {
        "best_target_curve": "Harman6",
        "target_seed_preset": {
            "preset_id": "minimum-seed",
            "max_slope_boost_db_per_oct": 6.0,
            "max_slope_cut_db_per_oct": 24.0,
            "conf_pull_max_hz": 130.0,
        },
        "best_metrics": {"rank_score": 93.0},
    }
    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode.get_house_curve_by_name",
        lambda name: (f, m),
    )
    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_cache_get_target_for_measurements_global",
        lambda *args, **kwargs: dict(cached_entry),
    )
    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_cache_get_target_for_measurements",
        lambda *args, **kwargs: dict(filter_entry),
    )
    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_cache_get_entry",
        lambda *args, **kwargs: None,
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("target comparison should be skipped on filter-specific cache hit")

    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_select_builtin_target_curve",
        fail_if_called,
    )
    result = _auto_select_target_curve_with_trials(
        base_data={
            "filter_type": "Minimum",
            "auto_goal": "balanced",
            "auto_target_mode": "auto",
            "program_version": "test-version",
        },
        measurements={"f_l": f, "m_l": m, "f_r": f, "m_r": m},
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        pin_obj=None,
        status_cb=None,
    )

    assert result is not None
    assert str(result.get("selected_hc_mode")) == "Harman6"
    assert str(result.get("selection_method")) == "cache_measurement_hit"
    assert dict(result.get("best_preset", {}) or {}).get("preset_id") == "minimum-seed"
    assert dict(result.get("best_metrics", {}) or {}).get("rank_score") == 93.0


def test_auto_target_global_cache_uses_seed_only_for_current_filter(monkeypatch):
    f = [20.0, 100.0, 1000.0, 10000.0]
    m = [0.0, 0.0, 0.0, 0.0]
    cached_entry = {
        "best_target_curve": "Harman8",
        "filter_seed_presets": {
            "minimum": {"preset_id": "minimum-seed"},
        },
        "filter_seed_metrics": {
            "minimum": {"rank_score": 92.0},
        },
    }
    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode.get_house_curve_by_name",
        lambda name: (f, m),
    )
    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_cache_get_target_for_measurements_global",
        lambda *args, **kwargs: dict(cached_entry),
    )
    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_cache_get_target_for_measurements",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_cache_get_entry",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_select_builtin_target_curve",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("target comparison should be skipped on global cache hit")
        ),
    )
    result = _auto_select_target_curve_with_trials(
        base_data={
            "filter_type": "Minimum",
            "auto_goal": "balanced",
            "auto_target_mode": "auto",
            "program_version": "test-version",
        },
        measurements={"f_l": f, "m_l": m, "f_r": f, "m_r": m},
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        pin_obj=None,
        status_cb=None,
    )

    assert result is not None
    assert str(result.get("selection_method")) == "cache_measurement_global_filter_seed_hit"
    assert dict(result.get("best_preset", {}) or {}).get("preset_id") == "minimum-seed"
    assert dict(result.get("best_metrics", {}) or {}).get("rank_score") == 92.0


def test_auto_target_legacy_measurement_cache_without_hidden_keys_stays_seed_only(monkeypatch):
    f = [20.0, 100.0, 1000.0, 10000.0]
    m = [0.0, 0.0, 0.0, 0.0]
    cached_entry = {
        "best_target_curve": "Harman8",
        "target_seed_preset": {"preset_id": "legacy-seed"},
    }
    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode.get_house_curve_by_name",
        lambda name: (f, m),
    )
    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_cache_get_target_for_measurements",
        lambda *args, **kwargs: dict(cached_entry),
    )
    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_cache_get_entry",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_select_builtin_target_curve",
        lambda *args, **kwargs: {
            "selected_hc_mode": "Harman10",
            "fit_rms_db": 4.1,
            "offset_db": 0.0,
            "candidates": [{"hc_mode": "Harman10", "fit_rms_db": 4.1}],
            "candidates_all": [{"hc_mode": "Harman10", "fit_rms_db": 4.1}],
        },
    )

    result = _auto_select_target_curve_with_trials(
        base_data={
            "filter_type": "Asymmetric",
            "auto_goal": "balanced",
            "auto_target_mode": "auto",
            "hc_mode": "Harman10",
            "program_version": "test-version",
        },
        measurements={
            "f_l": f,
            "m_l": m,
            "f_r": f,
            "m_r": m,
        },
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        pin_obj=None,
        status_cb=None,
    )

    assert result is not None
    assert str(result.get("selection_method")) != "cache_measurement_hit"
    assert str(result.get("selection_method")) != "cache_signature_hit"


def test_auto_target_exact_signature_cache_hit_bypasses_comparison_in_auto_mode(monkeypatch):
    """Verify that exact cache_signature sources bypass target comparison trials in auto mode.

    When cached_target_source is 'cache_signature' (an exact match), the target should be
    loaded directly from cache and skip all comparison trials, even in auto_target_mode='auto'.
    """
    f = [20.0, 100.0, 1000.0, 10000.0]
    m = [0.0, 0.0, 0.0, 0.0]
    cached_entry = {
        "best_target_curve": "Harman8",
        "target_seed_preset": {
            "preset_id": "exact-cache-preset",
            "max_slope_boost_db_per_oct": 2.0,
            "max_slope_cut_db_per_oct": 2.0,
            "conf_pull_max_hz": 80.0,
        },
        "cached_target_source": "cache_signature",
    }
    runtime = SimpleNamespace(
        get_house_curve_by_name=lambda name: (f, m),
        auto_cache_get_target_for_measurements=lambda *args, **kwargs: None,
        auto_cache_get_entry=lambda *args, **kwargs: dict(cached_entry),
        auto_optuna_module_ready=lambda *args, **kwargs: False,
    )
    setup = SimpleNamespace(
        runtime=runtime,
        goal="balanced",
        filter_key="minimum",
        compat_version="test-version",
        optimizer_backend="builtin",
        cfg=SimpleNamespace(optuna_persistent_study=False),
        optuna_mod=None,
    )

    state = orchestrator_target._resolve_cached_target_state(
        setup=setup,
        base_data={"auto_target_mode": "auto", "hc_mode": "Harman10"},
        measurements={"f_l": f, "m_l": m, "f_r": f, "m_r": m},
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        status_cb=None,
    )

    assert str(state.cached_target_source or "") == "cache_signature"
    assert str(state.cached_target_hc or "") == "Harman8"
    assert dict(state.cached_target_preset or {}).get("preset_id") == "exact-cache-preset"

    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode.get_house_curve_by_name",
        lambda name: (f, m),
    )
    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_cache_get_target_for_measurements",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_cache_get_entry",
        lambda *args, **kwargs: dict(cached_entry),
    )
    def fail_if_called(*args, **kwargs):
        raise AssertionError("_auto_select_builtin_target_curve should not be called")

    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_select_builtin_target_curve",
        fail_if_called,
    )
    result = _auto_select_target_curve_with_trials(
        base_data={
            "filter_type": "Asymmetric",
            "auto_goal": "balanced",
            "auto_target_mode": "auto",
            "hc_mode": "Harman10",
            "program_version": "test-version",
        },
        measurements={
            "f_l": f,
            "m_l": m,
            "f_r": f,
            "m_r": m,
        },
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        pin_obj=None,
        status_cb=None,
    )

    assert result is not None
    assert str(result.get("selection_method")) == "cache_signature_hit"
    assert str(result.get("selected_hc_mode")) == "Harman8"
    assert result.get("top_n") == 0
    assert result.get("trials_per_curve") == 0


def test_selected_target_cache_hit_skips_mismatched_current_hc_mode():
    f = [20.0, 100.0, 1000.0, 10000.0]
    m = [0.0, 0.0, 0.0, 0.0]
    cached_entry = {
        "best_target_curve": "Harman8",
        "target_seed_preset": {"preset_id": "stale-seed"},
    }
    runtime = SimpleNamespace(
        get_house_curve_by_name=lambda name: (f, m),
        auto_cache_get_target_for_measurements=lambda *args, **kwargs: dict(cached_entry),
        auto_cache_get_entry=lambda *args, **kwargs: dict(cached_entry),
        auto_optuna_module_ready=lambda *args, **kwargs: False,
    )
    setup = SimpleNamespace(
        runtime=runtime,
        goal="balanced",
        filter_key="minimum",
        compat_version="test-version",
        optimizer_backend="builtin",
        cfg=SimpleNamespace(optuna_persistent_study=False),
        optuna_mod=None,
    )

    state = orchestrator_target._resolve_cached_target_state(
        setup=setup,
        base_data={"auto_target_mode": "selected", "hc_mode": "Harman10"},
        measurements={"f_l": f, "m_l": m, "f_r": f, "m_r": m},
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        status_cb=None,
    )

    assert str(state.cached_target_source or "") == ""
    assert str(state.cached_target_hc or "") == ""
    assert dict(state.cached_target_preset or {}) == {}


def test_auto_target_already_shortlisted_cache_seed_does_not_bypass_trials():
    f = [20.0, 100.0, 1000.0, 10000.0]
    m = [0.0, 0.0, 0.0, 0.0]
    runtime = SimpleNamespace(get_house_curve_by_name=lambda name: (f, m))
    setup = SimpleNamespace(runtime=runtime)
    cache_state = orchestrator_target._TargetCacheState(
        cached_target_hc="Harman10",
        cached_target_preset={
            "preset_id": "cached-target-seed",
            "max_slope_boost_db_per_oct": 6.0,
            "max_slope_cut_db_per_oct": 24.0,
            "conf_pull_max_hz": 130.0,
        },
        cached_target_source="cache_optuna_target",
    )
    shortlist_state = orchestrator_target._TargetShortlistState(
        quick={"selected_hc_mode": "Harman10"},
        quick_candidates=[
            {"hc_mode": "Harman10", "fit_rms_db": 4.2},
            {"hc_mode": "Harman12", "fit_rms_db": 4.2},
        ],
        shortlisted=[
            {"hc_mode": "Harman10", "fit_rms_db": 4.2},
            {"hc_mode": "Harman12", "fit_rms_db": 4.2},
        ],
        trials_eff=10,
        f6_hz=20.0,
        f6_txt=" (-6 dB 20.0 Hz)",
    )

    result = orchestrator_target._apply_target_shortlist_modifiers(
        setup=setup,
        cache_state=cache_state,
        shortlist_state=shortlist_state,
        base_data={"auto_target_mode": "auto", "hc_mode": "Harman8"},
        measurements={"f_l": f, "m_l": m, "f_r": f, "m_r": m},
        status_cb=None,
    )

    assert isinstance(result, orchestrator_target._TargetShortlistState)
    assert [str(tc.get("hc_mode")) for tc in result.shortlisted] == ["Harman10", "Harman12"]


def test_selected_target_already_shortlisted_cache_seed_can_bypass_trials():
    f = [20.0, 100.0, 1000.0, 10000.0]
    m = [0.0, 0.0, 0.0, 0.0]
    runtime = SimpleNamespace(get_house_curve_by_name=lambda name: (f, m))
    setup = SimpleNamespace(
        runtime=runtime,
        goal="balanced",
        rank_basis="rank_score",
    )
    cache_state = orchestrator_target._TargetCacheState(
        cached_target_hc="Harman10",
        cached_target_preset={
            "preset_id": "cached-target-seed",
            "max_slope_boost_db_per_oct": 6.0,
            "max_slope_cut_db_per_oct": 24.0,
            "conf_pull_max_hz": 130.0,
        },
        cached_target_source="cache_optuna_target",
    )
    shortlist_state = orchestrator_target._TargetShortlistState(
        quick={"selected_hc_mode": "Harman10"},
        quick_candidates=[{"hc_mode": "Harman10", "fit_rms_db": 4.2}],
        shortlisted=[{"hc_mode": "Harman10", "fit_rms_db": 4.2}],
        trials_eff=10,
        f6_hz=20.0,
        f6_txt=" (-6 dB 20.0 Hz)",
    )

    result = orchestrator_target._apply_target_shortlist_modifiers(
        setup=setup,
        cache_state=cache_state,
        shortlist_state=shortlist_state,
        base_data={"auto_target_mode": "selected", "hc_mode": "Harman10"},
        measurements={"f_l": f, "m_l": m, "f_r": f, "m_r": m},
        status_cb=None,
    )

    assert isinstance(result, dict)
    assert str(result.get("selected_hc_mode")) == "Harman10"
    assert str(result.get("selection_method")) == "cache_optuna_target"


def test_target_by_measurement_cache_misses_when_measurements_change(monkeypatch, tmp_path):
    cache_path = tmp_path / "auto_cache.json"
    monkeypatch.setattr(
        cache_signature,
        "_auto_cache_path",
        lambda *args, **kwargs: str(cache_path),
    )
    cache_signature.clear_auto_mode_runtime_caches()
    try:
        m1 = {
            "f_l": [20.0, 100.0],
            "m_l": [0.0, 0.0],
            "f_r": [20.0, 100.0],
            "m_r": [0.0, 0.0],
        }
        m2 = {
            "f_l": [20.0, 100.0],
            "m_l": [0.0, 1.0],
            "f_r": [20.0, 100.0],
            "m_r": [0.0, 0.0],
        }
        cache_signature._auto_cache_put_target_for_measurements(
            measurements=m1,
            best_hc_mode="Harman8",
            best_preset={
                "preset_id": "cached-target",
                "max_slope_boost_db_per_oct": 6.0,
                "max_slope_cut_db_per_oct": 24.0,
                "conf_pull_max_hz": 130.0,
            },
            best_metrics={"rank_score": 91.0},
            goal="balanced",
            filter_key="minimum",
            compat_version="test-version",
        )

        hit = cache_signature._auto_cache_get_target_for_measurements(
            m1,
            goal="balanced",
            filter_key="minimum",
            compat_version="test-version",
        )
        miss = cache_signature._auto_cache_get_target_for_measurements(
            m2,
            goal="balanced",
            filter_key="minimum",
            compat_version="test-version",
        )
    finally:
        cache_signature.clear_auto_mode_runtime_caches()

    assert dict(hit or {}).get("best_hc_mode") == "Harman8"
    assert miss is None


def test_cached_target_second_run_skips_target_trials_but_not_filter_search(monkeypatch, tmp_path):
    import decaycore.auto_mode.api as auto_api

    cache_path = tmp_path / "auto_cache.json"
    monkeypatch.setattr(
        cache_signature,
        "_auto_cache_path",
        lambda *args, **kwargs: str(cache_path),
    )
    cache_signature.clear_auto_mode_runtime_caches()
    f = [20.0, 100.0, 1000.0, 10000.0]
    m = [0.0, 0.0, 0.0, 0.0]
    measurements = {"f_l": f, "m_l": m, "f_r": f, "m_r": m}
    try:
        cache_signature._auto_cache_put_target_for_measurements(
            measurements=measurements,
            best_hc_mode="Harman8",
            best_preset={
                "preset_id": "first-run-target",
                "max_slope_boost_db_per_oct": 6.0,
                "max_slope_cut_db_per_oct": 24.0,
                "conf_pull_max_hz": 130.0,
            },
            best_metrics={"rank_score": 91.0},
            goal="balanced",
            filter_key="minimum",
            compat_version="test-version",
        )

        monkeypatch.setattr(
            "decaycore.io.decaycore_automatic_mode.get_house_curve_by_name",
            lambda name: (f, m),
        )

        def fail_if_called(*args, **kwargs):
            raise AssertionError("_auto_select_builtin_target_curve must not be called")

        monkeypatch.setattr(auto_api, "_auto_select_builtin_target_curve", fail_if_called)

        base_data = {
            "filter_type": "Minimum",
            "auto_goal": "balanced",
            "auto_target_mode": "auto",
            "program_version": "test-version",
            "auto_mode_cache_enabled": True,
            "hc_mode": "Harman6",
            "mag_c_min": 20.0,
            "mag_c_max": 250.0,
        }
        result = _auto_select_target_curve_with_trials(
            base_data=dict(base_data),
            measurements=measurements,
            fs_v=44100,
            taps_v=65536,
            xos=[],
            hpf=None,
            pin_obj=None,
            status_cb=None,
        )
        assert result is not None
        assert str(result.get("selection_method")) in {
            "cache_measurement_hit",
            "cache_measurement_global_filter_seed_hit",
            "cache_signature_hit",
            "cache_optuna_target_hit",
        }

        planned_data = {
            **base_data,
            "hc_mode": str(result.get("selected_hc_mode")),
            "_auto_target_seed_preset": dict(result.get("best_preset", {}) or {}),
            "_auto_target_seed_metrics": dict(result.get("best_metrics", {}) or {}),
            "_auto_target_seed_source": str(result.get("selection_method")),
        }
        ctx = {"fs_v": 44100, "taps_v": 65536, "xos": [], "hpf": None}
        search_input = build_auto_search_input(planned_data, measurements, ctx)
        decision = determine_auto_search_plan(
            search_input,
            planned_data,
            options={
                "signature": compute_auto_search_signature(search_input),
                "filter_key": "minimum",
                "goal": "balanced",
                "compat_version": "test-version",
            },
        )

        assert decision.plan == AutoSearchPlan.FIRST_RUN_FULL_SEARCH
        assert decision.enabled_phases == ("target_search", "phase1", "phase2", "phase3")
        assert decision.skipped_phases == ("phase4",)
        assert decision.seed_source == "cache_measurement_target_seed"
    finally:
        cache_signature.clear_auto_mode_runtime_caches()


@pytest.mark.parametrize(
    ("filter_type", "filter_key"),
    [
        ("Linear", "linear"),
        ("Minimum", "minimum"),
        ("Mixed", "mixed"),
        ("Asymmetric", "asym"),
    ],
)
def test_global_cached_target_seed_skips_target_trials_for_each_filter_type(
    monkeypatch,
    tmp_path,
    filter_type,
    filter_key,
):
    import decaycore.auto_mode.api as auto_api

    cache_path = tmp_path / "auto_cache.json"
    monkeypatch.setattr(
        cache_signature,
        "_auto_cache_path",
        lambda *args, **kwargs: str(cache_path),
    )
    cache_signature.clear_auto_mode_runtime_caches()
    f = [20.0, 100.0, 1000.0, 10000.0]
    m = [0.0, 0.0, 0.0, 0.0]
    measurements = {"f_l": f, "m_l": m, "f_r": f, "m_r": m}
    try:
        cache_signature._auto_cache_put_target_for_measurements_global(
            measurements=measurements,
            best_hc_mode="Harman8",
            goal="balanced",
            compat_version="test-version",
            filter_key=filter_key,
            filter_seed_preset={
                "preset_id": f"{filter_key}-seed",
                "max_slope_boost_db_per_oct": 6.0,
                "max_slope_cut_db_per_oct": 24.0,
                "conf_pull_max_hz": 130.0,
            },
            filter_seed_metrics={"rank_score": 91.0},
        )

        monkeypatch.setattr(
            "decaycore.io.decaycore_automatic_mode.get_house_curve_by_name",
            lambda name: (f, m),
        )

        def fail_if_called(*args, **kwargs):
            raise AssertionError("_auto_select_builtin_target_curve must not be called")

        monkeypatch.setattr(auto_api, "_auto_select_builtin_target_curve", fail_if_called)

        result = _auto_select_target_curve_with_trials(
            base_data={
                "filter_type": filter_type,
                "auto_goal": "balanced",
                "auto_target_mode": "auto",
                "program_version": "test-version",
                "auto_mode_cache_enabled": True,
                "hc_mode": "Harman6",
            },
            measurements=measurements,
            fs_v=44100,
            taps_v=65536,
            xos=[],
            hpf=None,
            pin_obj=None,
            status_cb=None,
        )

        assert result is not None
        assert str(result.get("selection_method")) == "cache_measurement_global_filter_seed_hit"
        assert str(result.get("selected_hc_mode")) == "Harman8"
        assert dict(result.get("best_preset", {}) or {}).get("preset_id") == f"{filter_key}-seed"
    finally:
        cache_signature.clear_auto_mode_runtime_caches()


@pytest.mark.parametrize(
    ("filter_type", "filter_key"),
    [
        ("Linear", "linear"),
        ("Minimum", "minimum"),
        ("Mixed", "mixed"),
        ("Asymmetric", "asym"),
    ],
)
def test_seed_phase_cache_only_target_pick_handles_each_filter_type(
    monkeypatch,
    tmp_path,
    filter_type,
    filter_key,
):
    cache_path = tmp_path / "auto_cache.json"
    monkeypatch.setattr(
        cache_signature,
        "_auto_cache_path",
        lambda *args, **kwargs: str(cache_path),
    )
    cache_signature.clear_auto_mode_runtime_caches()
    measurements = {
        "f_l": [20.0, 100.0, 1000.0, 10000.0],
        "m_l": [0.0, 0.0, 0.0, 0.0],
        "f_r": [20.0, 100.0, 1000.0, 10000.0],
        "m_r": [0.0, 0.0, 0.0, 0.0],
    }
    try:
        cache_signature._auto_cache_put_target_for_measurements_global(
            measurements=measurements,
            best_hc_mode="Harman8",
            goal="balanced",
            compat_version="test-version",
            filter_key=filter_key,
            filter_seed_preset={
                "preset_id": f"{filter_key}-seed",
                "max_slope_boost_db_per_oct": 6.0,
                "max_slope_cut_db_per_oct": 24.0,
                "conf_pull_max_hz": 130.0,
            },
            filter_seed_metrics={"rank_score": 91.0},
        )
        pick = _try_cached_target_pick_before_search(
            data={
                "filter_type": filter_type,
                "auto_mode_compat_version": "test-version",
            },
            measurements=measurements,
            fs_v=44100,
            taps_v=65536,
            xos=[],
            hpf=None,
            goal="balanced",
        )

        assert pick is not None
        assert str(pick.get("selection_method")) == "cache_measurement_global_filter_seed_hit"
        assert str(pick.get("selected_hc_mode")) == "Harman8"
        assert dict(pick.get("best_preset", {}) or {}).get("preset_id") == f"{filter_key}-seed"
    finally:
        cache_signature.clear_auto_mode_runtime_caches()


def test_cached_best_persists_current_target_when_seed_had_prior_hc(monkeypatch):
    put_best_calls = []
    put_target_calls = []
    last_used_calls = []

    import decaycore.auto_mode.orchestrator_finalize_polish as _polish_mod

    monkeypatch.setattr(
        _polish_mod,
        "_auto_measurement_signature",
        lambda measurements: "measurement-sig",
    )
    monkeypatch.setattr(
        _polish_mod,
        "_auto_signature",
        lambda **kwargs: "sig-with-hc" if kwargs.get("include_hc_mode") else "sig-target",
    )
    monkeypatch.setattr(
        _polish_mod,
        "_auto_cache_put_best",
        lambda sig, **kwargs: put_best_calls.append({"sig": sig, **kwargs}),
    )
    monkeypatch.setattr(
        _polish_mod,
        "_auto_cache_put_target_for_measurements",
        lambda **kwargs: put_target_calls.append(dict(kwargs)),
    )
    monkeypatch.setattr(
        _polish_mod,
        "_auto_cache_put_last_used_best",
        lambda **kwargs: last_used_calls.append(dict(kwargs)),
    )

    orchestrator_finalize._save_cached_best(
        cache_base_data={
            "hc_mode": "Harman10",
            "_auto_target_seed_preset": {
                "preset_id": "cached-seed",
                "hc_mode": "Harman8",
                "max_slope_boost_db_per_oct": 6.0,
            },
        },
        measurements={"f_l": [20.0, 100.0], "m_l": [0.0, 0.0]},
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        cfg=SimpleNamespace(cache_enabled=True),
        goal="balanced",
        filter_key="minimum",
        compat_version="test-version",
        best_preset={"preset_id": "winner", "hc_mode": "Harman8"},
        best_metrics={"rank_score": 1.0},
        best_hc_mode="Harman10",
    )

    assert len(put_best_calls) == 2
    assert len(put_target_calls) == 1
    assert len(last_used_calls) == 1
    assert {call["best_hc_mode"] for call in put_best_calls} == {"Harman10"}
    assert put_target_calls[0]["best_hc_mode"] == "Harman10"
    assert last_used_calls[0]["best_hc_mode"] == "Harman10"
    for call in put_best_calls:
        assert dict(call["best_preset"])["hc_mode"] == "Harman10"
        assert dict(call["target_seed_preset"])["hc_mode"] == "Harman10"
    assert dict(put_target_calls[0]["best_preset"])["hc_mode"] == "Harman10"
    assert dict(put_target_calls[0]["target_seed_preset"])["hc_mode"] == "Harman10"


def test_cache_refine_winner_text_is_source_aware():
    assert (
        orchestrator_finalize._cache_refine_winner_summary(
            "target_preselect",
            improved_any=True,
        )
        == "Loaded target preselect seed and ran micro-refine trials."
    )
    assert "exact cached preset" not in orchestrator_finalize._cache_refine_winner_summary(
        "target_preselect",
        improved_any=False,
    )
    assert (
        orchestrator_finalize._cache_refine_winner_phase_label("target_preselect")
        == "target preselect + micro refine"
    )
    assert (
        orchestrator_finalize._cache_refine_winner_phase_label("exact_cache")
        == "exact cache hit + micro refine"
    )
    assert (
        orchestrator_finalize._cache_refine_winner_phase_label("cache_signature_target_seed")
        == "cached target seed + micro refine"
    )
    assert "exact cached preset" not in orchestrator_finalize._cache_refine_winner_summary(
        "cache_signature_target_seed",
        improved_any=False,
    )


def test_target_preselect_seed_does_not_bypass_phase1_search(monkeypatch):
    import decaycore.auto_mode.api as auto_api

    search_entrypoints = auto_api._search_entrypoints
    calls = {}
    call_order = []
    target_seed = {
        "preset_id": "target-preselect",
        "phase_limit": 432.1,
        "tdc_strength": 54.5,
    }
    target_metrics = {"rank_score": 87.5, "avg_score": 81.7}

    def _cache_refine_probe(**kwargs):
        raise AssertionError("pre-Phase-1 cache micro-refine must not run")

    def _search_refine_probe(**kwargs):
        call_order.append("search_refine")
        calls["search_refine"] = dict(kwargs)
        return {
            "phase1_ok": 1,
            "phase2_ok": 0,
            "phase1_tried": 1,
            "phase2_tried": 0,
            "phase1_plateau_hit": False,
            "phase2_plateau_hit": False,
            "phase1_optuna_tel": {},
            "phase2_local_optuna_tels": [],
            "phase3_micro_optuna_tel": {},
            "phase2_rollup_tel": {},
        }

    def _finalize_probe(**kwargs):
        call_order.append("finalize")
        calls["finalize"] = dict(kwargs)
        return {"ok": True}

    monkeypatch.setattr(
        search_entrypoints.orchestrator_refine,
        "run_exact_cache_micro_refine",
        _cache_refine_probe,
    )
    monkeypatch.setattr(
        search_entrypoints.orchestrator_refine,
        "run_search_refine_stages",
        _search_refine_probe,
    )
    monkeypatch.setattr(
        search_entrypoints.orchestrator_finalize,
        "finalize_search_result",
        _finalize_probe,
    )

    result = _run_auto_mode_search(
        base_data={
            "program_version": "test-version",
            "hc_mode": "Harman8",
            "filter_type": "Asymmetric",
            "auto_goal": "balanced",
            "_auto_target_seed_preset": dict(target_seed),
            "_auto_target_seed_metrics": dict(target_metrics),
        },
        measurements={
            "f_l": [20.0, 100.0],
            "m_l": [0.0, 0.0],
            "f_r": [20.0, 100.0],
            "m_r": [0.0, 0.0],
        },
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_f=[20.0, 100.0],
        hc_m=[0.0, 0.0],
        pin_obj=None,
        status_cb=None,
        n_trials=1,
    )

    assert result == {"ok": True}
    assert call_order == ["search_refine", "finalize"]
    assert "search_refine" in calls
    assert dict(calls["search_refine"].get("prior_seed_preset", {}) or {}) == target_seed


def test_search_refine_stages_use_canonical_order_with_seed(monkeypatch):
    calls = []

    def _phase1_search_probe(**kwargs):
        calls.append("build_context")
        return orchestrator_refine._SearchRefineContext(
            params={
                "prior_seed_preset": dict(kwargs.get("prior_seed_preset", {}) or {}),
                "search_state": kwargs.get("search_state"),
            }
        )

    def _phase1_probe(*, context):
        calls.append("phase1")
        params = dict(context.params or {})
        assert params.get("prior_seed_preset") == {"seeded": True}
        params["_phase1_state"] = "phase1-state"
        return orchestrator_refine._SearchRefineContext(params=params)

    def _phase2_probe(*, context):
        calls.append("phase2")
        params = dict(context.params or {})
        assert params.get("_phase1_state") == "phase1-state"
        params["_phase2_state"] = "phase2-state"
        return orchestrator_refine._SearchRefineContext(params=params)

    def _micro_probe(*, context):
        calls.append("micro")
        params = dict(context.params or {})
        assert params.get("_phase2_state") == "phase2-state"
        return orchestrator_refine._SearchRefineSummary(result={"ok": True})

    def _summary_probe(*, summary):
        calls.append("summary")
        return dict(summary.result or {})

    monkeypatch.setattr(orchestrator_refine, "_run_phase1_search", _phase1_search_probe)
    monkeypatch.setattr(orchestrator_refine, "_run_phase1_coarse_search", _phase1_probe)
    monkeypatch.setattr(orchestrator_refine, "_run_phase2_local_refine", _phase2_probe)
    monkeypatch.setattr(orchestrator_refine, "_run_phase3_micro_refine", _micro_probe)
    monkeypatch.setattr(orchestrator_refine, "_assemble_refine_summary", _summary_probe)

    result = orchestrator_refine.run_search_refine_stages(
        search_base_data={},
        measurements={},
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_f=None,
        hc_m=None,
        pin_obj=None,
        status_cb=None,
        cfg=SimpleNamespace(),
        goal="balanced",
        filter_key="asym",
        optimizer_backend="builtin",
        optuna_mod=None,
        seed=1,
        optuna_search_sig="sig",
        status_prefix="DecayCore automatic mode [Harman8]",
        winner_target_name="Harman8",
        search_state=SimpleNamespace(),
        n_trials_eff=1,
        candidates=[],
        prior_seed_preset={"seeded": True},
        use_optuna_trials=False,
        runtime=None,
    )

    assert result == {"ok": True}
    assert calls == ["build_context", "phase1", "phase2", "micro", "summary"]


def test_search_refine_stages_emit_phase3_skip_notice_once(monkeypatch):
    calls = []
    messages = []
    search_state = SimpleNamespace(best_preset={"preset_id": "best"})

    def _phase1_search_probe(**kwargs):
        return orchestrator_refine._SearchRefineContext(
            params={
                "cfg": kwargs.get("cfg"),
                "goal": kwargs.get("goal"),
                "search_state": kwargs.get("search_state"),
                "status_cb": kwargs.get("status_cb"),
                "runtime": kwargs.get("runtime"),
            }
        )

    def _phase1_probe(*, context):
        calls.append("phase1")
        params = dict(context.params or {})
        params["_phase1_state"] = orchestrator_refine._SearchPhase1State(ctx=SimpleNamespace())
        return orchestrator_refine._SearchRefineContext(params=params)

    def _phase2_probe(*, context):
        calls.append("phase2")
        params = dict(context.params or {})
        params["_phase2_state"] = orchestrator_refine._SearchPhase2State(phase2_improved_any=False)
        return orchestrator_refine._SearchRefineContext(params=params)

    def _summary_probe(*, summary):
        calls.append("summary")
        return dict(summary.result or {})

    monkeypatch.setattr(orchestrator_refine, "_run_phase1_search", _phase1_search_probe)
    monkeypatch.setattr(orchestrator_refine, "_run_phase1_coarse_search", _phase1_probe)
    monkeypatch.setattr(orchestrator_refine, "_run_phase2_local_refine", _phase2_probe)
    monkeypatch.setattr(orchestrator_refine, "_assemble_refine_summary", _summary_probe)

    result = orchestrator_refine.run_search_refine_stages(
        search_base_data={},
        measurements={},
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_f=None,
        hc_m=None,
        pin_obj=None,
        status_cb=messages.append,
        cfg=SimpleNamespace(phase3_micro_enabled=True, adaptive_shrink_max=0.5),
        goal="balanced",
        filter_key="asym",
        optimizer_backend="builtin",
        optuna_mod=None,
        seed=1,
        optuna_search_sig="sig",
        status_prefix="DecayCore automatic mode [Harman8]",
        winner_target_name="Harman8",
        search_state=search_state,
        n_trials_eff=1,
        candidates=[],
        prior_seed_preset=None,
        use_optuna_trials=False,
        runtime=SimpleNamespace(
            auto_optuna_telemetry_rollup=lambda items: {},
            auto_optuna_telemetry_text=lambda tel: "",
        ),
    )

    assert result["phase1_ok"] == 0
    assert result["phase2_ok"] == 0
    assert result["phase3_micro_optuna_tel"] == {}
    assert calls == ["phase1", "phase2", "summary"]
    assert messages == ["DecayCore automatic mode: phase 3 skipped"]


def test_auto_target_curve_selection_uses_optuna_target_study_without_recomputing(monkeypatch):
    f = [20.0, 100.0, 1000.0, 10000.0]
    m = [0.0, 0.0, 0.0, 0.0]
    base_data = {
        "filter_type": "Asymmetric",
        "auto_goal": "balanced",
        "program_version": "test-version",
        "auto_mode_optuna": True,
        "auto_mode_optuna_persistent_study": True,
        "auto_target_mode": "selected",
        "hc_mode": "Harman6",
    }
    measurements = {
        "f_l": f,
        "m_l": m,
        "f_r": f,
        "m_r": m,
    }
    sig_target = _auto_target_study_sig(
        _auto_get_measurement_signature(measurements),
        "balanced",
        "asym",
    )

    fake_optuna = SimpleNamespace()
    fake_optuna.get_all_study_summaries = lambda storage=None: [
        SimpleNamespace(
            study_name=("decaycore-target-harman6-phase1-0aff3be544aa-" + str(sig_target)[:32]),
            best_trial=SimpleNamespace(value=94.5),
            user_attrs={
                "decaycore_kind": "target_search",
                "decaycore_target_name": "Harman6",
                "decaycore_target_study_sig": str(sig_target),
                "decaycore_target_cache_version": 2,
                "decaycore_filter_key": "asym",
            },
        )
    ]
    fake_optuna.load_study = lambda study_name=None, storage=None: SimpleNamespace(
        user_attrs={
            "decaycore_kind": "target_search",
            "decaycore_target_name": "Harman6",
            "decaycore_target_study_sig": str(sig_target),
            "decaycore_target_cache_version": 2,
            "decaycore_filter_key": "asym",
        },
        best_trial=SimpleNamespace(
            value=94.5,
            params={
                "preset_id": "optuna-target",
                "max_slope_boost_db_per_oct": 6.0,
                "max_slope_cut_db_per_oct": 24.0,
                "conf_pull_max_hz": 130.0,
            },
            user_attrs={
                "camillafir_out": {
                    "preset": {
                        "preset_id": "optuna-target",
                        "max_slope_boost_db_per_oct": 6.0,
                        "max_slope_cut_db_per_oct": 24.0,
                        "conf_pull_max_hz": 130.0,
                    },
                    "metrics": {"rank_score": 94.5},
                }
            },
        )
    )

    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode.get_house_curve_by_name",
        lambda name: (f, m),
    )
    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_cache_get_target_for_measurements",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_cache_get_entry",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_import_optuna",
        lambda *args, **kwargs: fake_optuna,
    )
    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_optuna_module_ready",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_optuna_create_storage",
        lambda *args, **kwargs: object(),
    )

    def _fake_run_optuna_eval_loop(
        *,
        n_total,
        seed_presets=None,
        eval_one=None,
        consume_one=None,
        **_kwargs,
    ):
        seed = dict((list(seed_presets or [])[:1] or [{}])[0] or {})
        for idx in range(1, int(n_total) + 1):
            out = eval_one(int(idx), dict(seed or {}))
            consume_one(int(idx), dict(out or {}))
        return {}

    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_run_optuna_eval_loop",
        _fake_run_optuna_eval_loop,
    )

    def _unexpected_quick_preselect(*args, **kwargs):
        raise AssertionError("quick target preselect should be skipped on Optuna target study hit")

    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_select_builtin_target_curve",
        _unexpected_quick_preselect,
    )

    result = _auto_select_target_curve_with_trials(
        base_data=base_data,
        measurements=measurements,
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        pin_obj=None,
        status_cb=None,
    )

    assert result is not None
    assert str(result.get("selected_hc_mode")) == "Harman6"
    assert str(result.get("selection_method")) == "cache_optuna_target_hit"
    assert dict(result.get("best_preset", {}) or {}).get("preset_id") == "optuna-target"
    assert dict(result.get("best_preset", {}) or {}).get("max_slope_boost_db_per_oct") == 6.0
    assert dict(result.get("best_preset", {}) or {}).get("max_slope_cut_db_per_oct") == 24.0
    assert dict(result.get("best_preset", {}) or {}).get("conf_pull_max_hz") == 130.0


def test_auto_target_curve_selection_rejects_legacy_optuna_target_cache_hit_without_hidden_seed_keys(monkeypatch):
    f = [20.0, 100.0, 1000.0, 10000.0]
    m = [0.0, 0.0, 0.0, 0.0]
    base_data = {
        "filter_type": "Asymmetric",
        "auto_goal": "balanced",
        "auto_target_mode": "auto",
        "program_version": "test-version",
        "auto_mode_optuna": True,
        "auto_mode_optuna_persistent_study": True,
    }
    sig_target = _auto_signature(
        base_data=base_data,
        measurements={
            "f_l": f,
            "m_l": m,
            "f_r": f,
            "m_r": m,
        },
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_mode=None,
        include_hc_mode=False,
    )

    fake_optuna = SimpleNamespace()
    fake_optuna.get_all_study_summaries = lambda storage=None: [
        SimpleNamespace(
            study_name=(
                "camillafir-target-harman6-phase1-0aff3be544aa-"
                + str(sig_target)[:32]
            ),
            best_trial=SimpleNamespace(value=94.5),
        )
    ]
    fake_optuna.load_study = lambda study_name=None, storage=None: SimpleNamespace(
        best_trial=SimpleNamespace(
            value=94.5,
            params={"preset_id": "legacy-optuna-target"},
            user_attrs={
                "camillafir_out": {
                    "preset": {"preset_id": "legacy-optuna-target"},
                    "metrics": {"rank_score": 94.5},
                }
            },
        )
    )

    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode.get_house_curve_by_name",
        lambda name: (f, m),
    )
    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_cache_get_target_for_measurements",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_cache_get_entry",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_import_optuna",
        lambda *args, **kwargs: fake_optuna,
    )
    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_optuna_module_ready",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_optuna_create_storage",
        lambda *args, **kwargs: object(),
    )

    fallback_called = {"n": 0}

    def _fallback_shortlist(**_kwargs):
        fallback_called["n"] += 1
        return {
            "selected_hc_mode": "Harman4",
            "fit_rms_db": 4.2,
            "offset_db": 0.0,
            "selection_method": "top3x10_trials",
            "selection_basis": "rank_score",
            "auto_goal": "balanced",
            "top_n": 3,
            "trials_per_curve": 10,
            "candidates": [],
            "evaluated": [],
            "best_preset": {
                "preset_id": "fresh-target-seed",
                "max_slope_boost_db_per_oct": 6.0,
                "max_slope_cut_db_per_oct": 24.0,
                "conf_pull_max_hz": 130.0,
            },
        }

    import decaycore.auto_mode.orchestrator_target_selection as _ot_sel
    monkeypatch.setattr(_ot_sel, "_load_quick_target_selection", _fallback_shortlist)

    result = _auto_select_target_curve_with_trials(
        base_data=base_data,
        measurements={
            "f_l": f,
            "m_l": m,
            "f_r": f,
            "m_r": m,
        },
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        pin_obj=None,
        status_cb=None,
    )

    assert result is not None
    assert int(fallback_called["n"]) == 1
    assert str(result.get("selection_method")) == "top3x10_trials"
    assert str(result.get("selection_method")) != "cache_optuna_target_hit"
    assert dict(result.get("best_preset", {}) or {}).get("preset_id") == "fresh-target-seed"


def test_auto_mode_search_ignores_legacy_exact_signature_seed(monkeypatch):
    import decaycore.auto_mode.api as auto_api

    search_entrypoints = auto_api._search_entrypoints
    calls = []
    captured = {}

    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_cache_get_entry",
        lambda *args, **kwargs: {
            "best_preset": {
                "phase_limit": 432.1,
                "tdc_strength": 54.5,
                "preset_id": "exact-cache",
                "_auto_exc_freq_hz": 31.5,
            },
            "best_metrics": {"rank_score": 87.5, "avg_score": 81.7},
        },
    )
    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_cache_get_best",
        lambda *args, **kwargs: {
            "phase_limit": 432.1,
            "tdc_strength": 54.5,
            "preset_id": "exact-cache",
            "_auto_exc_freq_hz": 31.5,
            "_auto_hpf_runtime_override": {"enabled": False, "freq": 27.5, "order": 4},
            "hpf_enable": False,
            "hpf_freq": 27.5,
            "hpf_slope": 24,
        },
    )
    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._build_auto_mode_candidates",
        lambda *args, **kwargs: [{"preset_id": "phase1-trial"}],
    )

    monkeypatch.setattr(
        search_entrypoints.orchestrator_refine,
        "run_search_refine_stages",
        lambda **kwargs: (
            calls.append("search_refine")
            or captured.setdefault("search_refine", dict(kwargs))
            or {
                "phase1_ok": 1,
                "phase2_ok": 0,
                "phase1_tried": 1,
                "phase2_tried": 0,
                "phase1_plateau_hit": False,
                "phase2_plateau_hit": False,
                "phase1_optuna_tel": {},
                "phase2_local_optuna_tels": [],
                "phase3_micro_optuna_tel": {},
                "phase2_rollup_tel": {},
            }
        ),
    )
    monkeypatch.setattr(
        search_entrypoints.orchestrator_finalize,
        "finalize_search_result",
        lambda **kwargs: calls.append("finalize") or {"ok": True},
    )

    result = _run_auto_mode_search(
        base_data={
            "program_version": "test-version",
            "hc_mode": "Harman8",
            "filter_type": "Asymmetric",
            "auto_goal": "balanced",
            "auto_mode_optuna": False,
            "auto_mode_optuna_persistent_study": False,
            "use_auto_search_v2": False,
            "auto_mode_use_auto_search_v2": False,
        },
        measurements={
            "f_l": [20.0, 100.0],
            "m_l": [0.0, 0.0],
            "f_r": [20.0, 100.0],
            "m_r": [0.0, 0.0],
        },
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_f=[20.0, 100.0],
        hc_m=[0.0, 0.0],
        pin_obj=None,
        status_cb=None,
    )

    assert result == {"ok": True}
    assert calls == ["search_refine", "finalize"]
    assert dict(captured["search_refine"].get("prior_seed_preset", {}) or {}).get("preset_id") != "exact-cache"


def test_auto_mode_search_ignores_legacy_optuna_phase1_study_seed(monkeypatch):
    import decaycore.auto_mode.api as auto_api

    search_entrypoints = auto_api._search_entrypoints
    calls = []
    captured = {}

    fake_optuna = SimpleNamespace()
    fake_optuna.load_study = lambda study_name=None, storage=None: SimpleNamespace(
        best_trial=SimpleNamespace(
            value=87.5,
            params={"preset_id": "optuna-phase1"},
            user_attrs={
                "camillafir_out": {
                    "preset": {
                        "phase_limit": 432.1,
                        "tdc_strength": 54.5,
                        "preset_id": "optuna-phase1",
                        "_auto_exc_freq_hz": 31.5,
                    },
                    "metrics": {"rank_score": 87.5, "avg_score": 81.7},
                }
            },
        )
    )

    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_cache_get_entry",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_cache_get_best",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_import_optuna",
        lambda *args, **kwargs: fake_optuna,
    )
    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_optuna_module_ready",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_optuna_create_storage",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        search_entrypoints.orchestrator_refine,
        "run_search_refine_stages",
        lambda **kwargs: (
            calls.append("search_refine")
            or captured.setdefault("search_refine", dict(kwargs))
            or {
                "phase1_ok": 1,
                "phase2_ok": 0,
                "phase1_tried": 1,
                "phase2_tried": 0,
                "phase1_plateau_hit": False,
                "phase2_plateau_hit": False,
                "phase1_optuna_tel": {},
                "phase2_local_optuna_tels": [],
                "phase3_micro_optuna_tel": {},
                "phase2_rollup_tel": {},
            }
        ),
    )
    monkeypatch.setattr(
        search_entrypoints.orchestrator_finalize,
        "finalize_search_result",
        lambda **kwargs: calls.append("finalize") or {"ok": True},
    )

    result = _run_auto_mode_search(
        base_data={
            "program_version": "test-version",
            "hc_mode": "Harman8",
            "filter_type": "Asymmetric",
            "auto_goal": "balanced",
            "auto_mode_optuna": True,
            "auto_mode_optuna_persistent_study": True,
            "use_auto_search_v2": False,
            "auto_mode_use_auto_search_v2": False,
        },
        measurements={
            "f_l": [20.0, 100.0],
            "m_l": [0.0, 0.0],
            "f_r": [20.0, 100.0],
            "m_r": [0.0, 0.0],
        },
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_f=[20.0, 100.0],
        hc_m=[0.0, 0.0],
        pin_obj=None,
        status_cb=None,
    )

    assert result == {"ok": True}
    assert calls == ["search_refine", "finalize"]
    assert dict(captured["search_refine"].get("prior_seed_preset", {}) or {}).get("preset_id") != "optuna-phase1"


def test_target_phase1_pruned_trial_logs_info_instead_of_warning(monkeypatch, caplog):
    monkeypatch.setattr(
        "decaycore.auto_mode.orchestrator_target_phase1._run_target_trials",
        lambda **kwargs: [
            {
                "idx": 1,
                "ok": False,
                "error": "optuna trial pruned",
                "pruned": True,
            }
        ],
    )

    setup = _TargetTrialSetup(
        hc_name="Harman10",
        hc_f=[],
        hc_m=[],
        seed_tc=123,
        base_tc={},
        use_optuna_curve_trials=True,
        candidates=[],
        phase1_seed_presets=[],
        phase1_trial_total=1,
    )

    with caplog.at_level(logging.INFO, logger="DecayCore"):
        accumulator = _run_target_phase1_trials(
            runtime=SimpleNamespace(),
            cfg=None,
            optimizer_backend="optuna",
            optuna_mod=object(),
            target_study_sig="sig",
            seed_target=123,
            measurements={},
            fs_v=48000,
            taps_v=65536,
            xos=[],
            hpf=None,
            pin_obj=None,
            filter_key="linear",
            shortlisted=[{"hc_mode": "Harman10"}],
            status_cb=None,
            f6_txt="",
            goal="balanced",
            tc={},
            t_idx=1,
            emit_status=False,
            setup=setup,
        )

    assert accumulator.ok_n == 0
    assert "Automatic mode target trial failed" not in caplog.text
    assert "Automatic mode target trial pruned: target=Harman10 1/1 (optuna trial pruned)" in caplog.text


def test_auto_mode_search_does_not_replay_exact_cache_before_canonical_search(monkeypatch):
    import decaycore.auto_mode.api as auto_api

    search_entrypoints = auto_api._search_entrypoints
    remembered = []
    captured = {}

    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_cache_get_entry",
        lambda *args, **kwargs: {
            "best_preset": {
                "phase_limit": 432.1,
                "tdc_strength": 54.5,
                "preset_id": "exact-cache",
                "_auto_exc_freq_hz": 31.5,
            },
            "best_metrics": {"rank_score": 87.5, "avg_score": 81.7},
        },
    )
    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_cache_get_best",
        lambda *args, **kwargs: {
            "phase_limit": 432.1,
            "tdc_strength": 54.5,
            "preset_id": "exact-cache",
            "_auto_exc_freq_hz": 31.5,
        },
    )
    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode.build_config",
        lambda ui_data, preset=None, *, fs_v=None, taps_v=None, xos=None, hpf=None, hc_f=None, hc_m=None, filter_config_cls=None, max_safe_boost=8.0: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode.run_pipeline",
        lambda *args, **kwargs: SimpleNamespace(l_st={}, r_st={}, metrics={}),
    )

    def _score_result(result, **kwargs):
        base = dict(kwargs.get("base_data", {}) or {})
        preset_id = str(base.get("preset_id", "exact-cache"))
        if preset_id == "micro-1":
            return {"rank_score": 87.7, "avg_score": 81.6}
        return {"rank_score": 87.5, "avg_score": 81.7}

    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode.summarize_run",
        lambda result: "cached-summary",
    )
    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_score_result",
        _score_result,
    )
    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._build_auto_mode_candidates_micro",
        lambda *args, **kwargs: [
            {"phase_limit": 432.1, "tdc_strength": 54.5, "preset_id": "exact-cache", "_auto_exc_freq_hz": 31.5},
            {"phase_limit": 430.0, "tdc_strength": 53.0, "preset_id": "micro-1", "_auto_exc_freq_hz": 31.5},
        ],
    )
    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_import_optuna",
        lambda: object(),
    )
    monkeypatch.setattr(
        "decaycore.io.decaycore_automatic_mode._auto_optuna_remember_result",
        lambda *args, **kwargs: remembered.append(dict(kwargs or {})) or True,
    )
    monkeypatch.setattr(
        search_entrypoints.orchestrator_refine,
        "run_search_refine_stages",
        lambda **kwargs: (
            captured.setdefault("search_refine", dict(kwargs))
            or {
                "phase1_ok": 1,
                "phase2_ok": 0,
                "phase1_tried": 1,
                "phase2_tried": 0,
                "phase1_plateau_hit": False,
                "phase2_plateau_hit": False,
                "phase1_optuna_tel": {},
                "phase2_local_optuna_tels": [],
                "phase3_micro_optuna_tel": {},
                "phase2_rollup_tel": {},
            }
        ),
    )
    monkeypatch.setattr(
        search_entrypoints.orchestrator_finalize,
        "finalize_search_result",
        lambda **kwargs: {"ok": True},
    )

    result = _run_auto_mode_search(
        base_data={
            "program_version": "test-version",
            "hc_mode": "Harman8",
            "filter_type": "Asymmetric",
            "auto_goal": "balanced",
            "auto_mode_optuna": True,
            "auto_mode_optuna_persistent_study": True,
            "use_auto_search_v2": False,
            "auto_mode_use_auto_search_v2": False,
        },
        measurements={
            "f_l": [20.0, 100.0],
            "m_l": [0.0, 0.0],
            "f_r": [20.0, 100.0],
            "m_r": [0.0, 0.0],
        },
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_f=[20.0, 100.0],
        hc_m=[0.0, 0.0],
        pin_obj=None,
        status_cb=None,
    )

    assert result == {"ok": True}
    assert remembered == []
    assert dict(captured["search_refine"].get("prior_seed_preset", {}) or {}).get("preset_id") != "exact-cache"


def test_auto_signature_changes_when_bi_crossover_changes():
    base = {
        "avr_crossover_hz": 80.0,
        "sub_crossover_slope": 24,
        "sub_hpf_freq": 20.0,
        "sub_hpf_slope": 12,
        "direct_dac_sub_lpf_hz": 100.0,
        "bass_integration_mode": "direct_dac",
        "bass_integration_profile": "default",
    }
    base_alt = dict(base)
    base_alt["avr_crossover_hz"] = 120.0
    m = {"f_l": [20.0, 100.0], "m_l": [0.0, 0.0]}
    sig1 = _auto_signature(
        base_data=base,
        measurements=m,
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_mode=None,
        include_hc_mode=False,
    )
    sig2 = _auto_signature(
        base_data=base_alt,
        measurements=m,
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_mode=None,
        include_hc_mode=False,
    )
    assert sig1 != sig2


def test_auto_signature_normalizes_legacy_bi_mode_to_direct_dac():
    base = {
        "bass_integration_mode": "direct_dac",
        "bass_integration_profile": "default",
        "avr_crossover_hz": 80.0,
    }
    base_alt = dict(base)
    base_alt["bass_integration_mode"] = "avr_lfe"
    m = {"f_l": [20.0, 100.0], "m_l": [0.0, 0.0]}
    sig1 = _auto_signature(
        base_data=base,
        measurements=m,
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_mode=None,
        include_hc_mode=False,
    )
    sig2 = _auto_signature(
        base_data=base_alt,
        measurements=m,
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_mode=None,
        include_hc_mode=False,
    )
    assert sig1 == sig2


def test_auto_signature_changes_when_target_curve_changes():
    base = {"filter_type": "mixed", "hc_mode": "Harman8"}
    m = {"f_l": [20.0, 100.0], "m_l": [0.0, 0.0]}
    sig1 = _auto_signature(
        base_data=base,
        measurements=m,
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_mode="Harman8",
        include_hc_mode=True,
    )
    sig2 = _auto_signature(
        base_data={**base, "hc_mode": "Harman10"},
        measurements=m,
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_mode="Harman10",
        include_hc_mode=True,
    )
    assert sig1 != sig2


def test_auto_signature_changes_when_tdc_settings_change():
    base = {"filter_type": "mixed", "tdc_strength": 45.0, "tdc_max_reduction_db": 8.0}
    m = {"f_l": [20.0, 100.0], "m_l": [0.0, 0.0]}
    sig1 = _auto_signature(
        base_data=base,
        measurements=m,
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_mode=None,
        include_hc_mode=False,
    )
    sig2 = _auto_signature(
        base_data={**base, "tdc_strength": 55.0},
        measurements=m,
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_mode=None,
        include_hc_mode=False,
    )
    assert sig1 != sig2


def test_auto_signature_changes_when_peak_awareness_scoring_settings_change():
    base = {"filter_type": "mixed", "auto_mode_residual_peak_threshold_db": 4.0}
    m = {"f_l": [20.0, 100.0], "m_l": [0.0, 0.0]}
    sig1 = _auto_signature(
        base_data=base,
        measurements=m,
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_mode=None,
        include_hc_mode=False,
    )
    sig2 = _auto_signature(
        base_data={**base, "auto_mode_residual_peak_threshold_db": 5.0},
        measurements=m,
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_mode=None,
        include_hc_mode=False,
    )
    assert sig1 != sig2


def test_auto_signature_changes_when_acoustic_authority_limit_settings_change():
    base = {"filter_type": "mixed", "acoustic_authority_limits_enable": True}
    m = {"f_l": [20.0, 100.0], "m_l": [0.0, 0.0]}
    sig1 = _auto_signature(
        base_data=base,
        measurements=m,
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_mode=None,
        include_hc_mode=False,
    )
    sig2 = _auto_signature(
        base_data={**base, "authority_boost_gamma": 1.55},
        measurements=m,
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_mode=None,
        include_hc_mode=False,
    )
    assert sig1 != sig2


def test_auto_signature_changes_when_voice_clarity_scoring_settings_change():
    base = {"filter_type": "mixed", "auto_voice_clarity_penalty_enable": True}
    m = {"f_l": [20.0, 100.0], "m_l": [0.0, 0.0]}
    sig1 = _auto_signature(
        base_data=base,
        measurements=m,
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_mode=None,
        include_hc_mode=False,
    )
    sig2 = _auto_signature(
        base_data={**base, "auto_voice_clarity_penalty_weight": 1.25},
        measurements=m,
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_mode=None,
        include_hc_mode=False,
    )
    assert sig1 != sig2


def test_auto_signature_payload_exposes_policy_versions_and_metadata_identity():
    base = {
        "filter_type": "mixed",
        "max_boost_db": 3.0,
        "conf_pull_gamma_boost": 1.35,
        "phase_guard_max_gd_gradient_ms_per_oct": 50.0,
        "bass_integration_guard_lo_ratio": 0.60,
    }
    measurements = {
        "f_l": [20.0, 100.0],
        "m_l": [0.0, 0.0],
        "measured_rt60_l": 0.42,
        "measurement_snr_db_l": 31.0,
        "harmonic_risk_summary_l": {"peak": 0.2},
    }

    payload = _auto_signature_payload(
        base_data=base,
        measurements=measurements,
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_mode=None,
        include_hc_mode=False,
    )

    assert payload["measurement_metadata_identity"] == _auto_measurement_metadata_identity(measurements)
    assert payload["signature_policy_versions"]["gain_authority_policy_v"] >= 1
    assert payload["gain_authority_policy"]["max_boost_db"] == 3.0
    assert payload["confidence_model"]["policy_v"] >= 2
    assert payload["confidence_model"]["conf_pull_ceil"] == 0.85
    assert payload["confidence_model"]["conf_pull_gamma_cut"] == 0.45
    assert payload["confidence_model"]["conf_pull_gamma_boost"] == 1.35
    assert payload["confidence_model"]["conf_pull_bass_boost_floor_min"] == 0.55
    assert payload["confidence_model"]["conf_pull_bass_boost_restore"] == 0.70
    assert payload["hybrid_iir"]["policy_v"] >= 3
    assert payload["hybrid_iir"]["max_freq_hz"] == 200.0
    assert payload["hybrid_iir"]["min_confidence"] == 0.30
    assert payload["hybrid_iir"]["min_cut_priority"] == 0.0
    assert payload["residual_peak_scorer"]["scorer_v"] >= 2
    assert payload["bass_integration_feasibility"]["policy_v"] >= 1
    assert payload["phase_gd_guard"]["policy_v"] >= 4


def test_auto_signature_changes_when_measurement_metadata_changes():
    base = {"filter_type": "mixed"}
    measurements = {
        "f_l": [20.0, 100.0],
        "m_l": [0.0, 0.0],
        "measurement_snr_db_l": 31.0,
    }
    changed = dict(measurements)
    changed["measurement_snr_db_l"] = 24.0

    sig1 = _auto_signature(
        base_data=base,
        measurements=measurements,
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_mode=None,
        include_hc_mode=False,
    )
    sig2 = _auto_signature(
        base_data=base,
        measurements=changed,
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_mode=None,
        include_hc_mode=False,
    )
    assert sig1 != sig2


def test_auto_signature_changes_when_phase_gd_guard_policy_changes():
    base = {"filter_type": "mixed", "phase_guard_max_gd_gradient_ms_per_oct": 50.0}
    m = {"f_l": [20.0, 100.0], "m_l": [0.0, 0.0]}

    sig1 = _auto_signature(
        base_data=base,
        measurements=m,
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_mode=None,
        include_hc_mode=False,
    )
    sig2 = _auto_signature(
        base_data={**base, "phase_guard_max_gd_gradient_ms_per_oct": 40.0},
        measurements=m,
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_mode=None,
        include_hc_mode=False,
    )
    assert sig1 != sig2


def test_auto_signature_ignores_plot_only_settings():
    base = {"filter_type": "mixed", "tdc_strength": 45.0, "plot_smoothing_level": "none"}
    m = {"f_l": [20.0, 100.0], "m_l": [0.0, 0.0]}
    sig1 = _auto_signature(
        base_data=base,
        measurements=m,
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_mode=None,
        include_hc_mode=False,
    )
    sig2 = _auto_signature(
        base_data={**base, "plot_smoothing_level": "psy", "export_filename": "x.wav"},
        measurements=m,
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_mode=None,
        include_hc_mode=False,
    )
    assert sig1 == sig2


def test_auto_cache_empty_uses_schema_version_constant():
    cache = _auto_cache_empty(compat_version="test")
    assert int(cache["v"]) == int(AUTO_MODE_CACHE_SCHEMA_VERSION)
    assert int(cache["schema_version"]) == int(AUTO_MODE_CACHE_SCHEMA_VERSION)


def test_bi_enabled_candidates_contain_bi_fields():
    from decaycore.auto_mode.candidate_generation import _build_auto_mode_candidates

    base_data = {
        "bass_integration_enabled": True,
        "avr_crossover_hz": 80.0,
        "bass_integration_sub_delay_ms": 0.0,
        "bass_integration_sub_polarity_invert": False,
        "bass_integration_sub_gain_trim_db": 0.0,
        "direct_dac_sub_lpf_hz": 100.0,
        "bass_integration_mode": "direct_dac",
    }
    candidates = _build_auto_mode_candidates(base_data, n_trials=20, seed=42)
    assert len(candidates) > 0
    bi_keys = {"avr_crossover_hz", "bass_integration_sub_delay_ms", "bass_integration_sub_polarity_invert", "bass_integration_sub_gain_trim_db"}
    assert any(bi_keys.issubset(set(c.keys())) for c in candidates), (
        f"No candidate contained all BI keys. Sample keys: {list(candidates[0].keys())[:10]}"
    )


def test_bi_disabled_candidates_do_not_contain_bi_fields():
    from decaycore.auto_mode.candidate_generation import _build_auto_mode_candidates

    base_data = {
        "bass_integration_enabled": False,
        "avr_crossover_hz": 80.0,
    }
    candidates = _build_auto_mode_candidates(base_data, n_trials=10, seed=42)
    assert len(candidates) > 0
    bi_keys = {"avr_crossover_hz", "bass_integration_sub_delay_ms"}
    assert all(not bi_keys.issubset(set(c.keys())) for c in candidates), (
        "BI-disabled candidates should not contain BI fields"
    )


def test_finalize_injects_phase1_top_into_empty_phase2_pool(monkeypatch):
    pareto_received = []

    import decaycore.auto_mode.orchestrator_finalize as fin
    import decaycore.auto_mode.orchestrator_finalize_run as _run_mod
    import decaycore.auto_mode.orchestrator_finalize_polish as _polish_mod

    original_pareto_front = _run_mod._auto_phase2_pareto_front

    def _capture_pareto(pool):
        pareto_received.extend(list(pool or []))
        return original_pareto_front(pool)

    monkeypatch.setattr(_run_mod, "_auto_phase2_pareto_front", _capture_pareto)

    for fn in (
        "apply_phase_limit_winner_polish",
        "apply_mag_c_min_winner_polish",
        "apply_low_bass_cut_winner_polish",
        "apply_hpf_winner_polish",
        "apply_excess_phase_strength_winner_polish",
        "apply_residual_peak_winner_polish",
    ):
        monkeypatch.setattr(
            _polish_mod,
            fn,
            lambda best_preset, best_metrics, **_kw: (best_preset, best_metrics, False, {}),
        )

    monkeypatch.setattr(
        _polish_mod,
        "apply_stereo_policy_refine",
        lambda best_preset, best_metrics, **_kw: (best_preset, best_metrics, False, {}),
    )

    phase1_preset = {"phase_limit": 400.0, "tdc_strength": 50.0}
    phase1_metrics = {"rank_score": 88.0, "avg_score": 82.0, "phase": "phase 1/2"}

    runtime = SimpleNamespace(
        phase_limit_winner_polish_enabled=False,
        phase_limit_winner_polish_offsets_hz=(),
        mag_c_min_winner_polish_enabled=False,
        mag_c_min_winner_polish_step_hz=1.0,
        mag_c_min_winner_polish_max_down_hz=0.0,
        mag_c_min_winner_polish_max_up_hz=4.0,
        hpf_winner_polish_enabled=False,
        excess_phase_strength_winner_polish_enabled=False,
        excess_phase_strength_winner_polish_step=0.1,
        excess_phase_strength_winner_polish_max_delta=0.0,
        residual_peak_winner_polish_enabled=False,
        residual_peak_winner_polish_max_variants=0,
        residual_peak_winner_polish_min_improvement_db=99.0,
    )

    search_state = SimpleNamespace(
        phase2_pool=[],
        scored=[{"preset": phase1_preset, "metrics": phase1_metrics}],
        best_preset=dict(phase1_preset),
        best_metrics=dict(phase1_metrics),
        best_result=None,
        winner_explanation={},
    )

    cfg = SimpleNamespace(
        phase2_pareto_rank_window=2.0,
        phase2_pareto_pool_max=20,
        phase2_hard_gate_enabled=False,
        phase2_hard_gate_min_keep=3,
        phase2_hard_gate_keep_event_fraction=0.5,
        phase2_hard_gate_keep_ripple_fraction=0.5,
        phase2_hard_gate_keep_peak_fraction=0.5,
        phase2_hard_gate_abs_max_peak_db=6.0,
        phase2_hard_gate_fallback_to_rank=True,
        phase2_pareto_pool_min=1,
        phase2_pareto_acoustic_drop=0.35,
        local_refine_top_k=3,
        cache_enabled=False,
        residual_peak_winner_polish_enabled=False,
        residual_peak_winner_polish_max_variants=8,
        residual_peak_winner_polish_min_improvement_db=0.75,
    )

    def _fake_materialize(preset, include_response_arrays=False, summarize=False, base_data_override=None):
        return None, dict(preset), dict(preset)

    def _fake_cache_ready(preset, best_metrics=None):
        return dict(preset)

    def _fake_residual_tiebreak(best_preset, best_metrics, candidate_items=None, base_data_ref=None, phase_label=""):
        return best_preset, best_metrics, False

    fin.finalize_search_result(
        search_base_data={},
        cache_base_data={},
        measurements={},
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_f=None,
        hc_m=None,
        pin_obj=None,
        cfg=cfg,
        goal="balanced",
        rank_basis="rank_score",
        filter_key="linear",
        compat_version="test",
        optimizer_backend="builtin",
        status_cb=None,
        optuna_mod=None,
        optuna_search_sig="sig",
        seed=1,
        search_state=search_state,
        winner_target_name=None,
        phase1_ok=1,
        phase2_ok=0,
        phase1_tried=1,
        phase2_tried=0,
        phase1_plateau_hit=False,
        phase2_plateau_hit=False,
        phase1_optuna_tel={},
        phase2_local_optuna_tels=[],
        phase3_micro_optuna_tel={},
        phase2_rollup_tel={},
        _cache_ready_preset=_fake_cache_ready,
        _materialize_preset_result=_fake_materialize,
        _maybe_apply_residual_tiebreak=_fake_residual_tiebreak,
        cache_refine_result=None,
        runtime=runtime,
    )

    assert len(pareto_received) >= 1, "Pareto front should have received injected phase1 candidate"
    assert any(
        dict(it.get("metrics", {}) or {}).get("phase") == "phase 1/2"
        for it in pareto_received
    ), "Injected phase1 entry should appear in pareto pool"


def test_target_eval_one_uses_current_build_config_signature_without_pin():
    from types import SimpleNamespace

    calls = []

    def _build_config(
        ui_data,
        preset=None,
        *,
        fs_v=None,
        taps_v=None,
        xos=None,
        hpf=None,
        hc_f=None,
        hc_m=None,
        filter_config_cls=None,
        max_safe_boost=8.0,
    ):
        calls.append(
            {
                "fs_v": fs_v,
                "taps_v": taps_v,
                "max_safe_boost": max_safe_boost,
                "ui_data": dict(ui_data or {}),
            }
        )
        return SimpleNamespace()

    runtime = SimpleNamespace(
        build_config=_build_config,
        run_pipeline=lambda cfg, measurements, include_response_arrays=False: SimpleNamespace(
            metrics={},
            ui_data=dict(measurements.get("ui_data", {}) or {}),
        ),
        auto_score_result=lambda result, **kwargs: {"rank_score": 84.0, "avg_score": 80.0},
    )

    out = _target_eval_one(
        runtime=runtime,
        preset={"phase_limit": 420.0},
        base_tc={"filter_type": "Linear Phase", "comparison_mode": True},
        measurements={
            "f_l": [20.0, 100.0],
            "m_l": [0.0, 0.0],
            "p_l": [0.0, 0.0],
            "f_r": [20.0, 100.0],
            "m_r": [0.0, 0.0],
            "p_r": [0.0, 0.0],
        },
        fs_v=44100,
        taps_v=65536,
        xos=[],
        hpf=None,
        hc_f_arr=[20.0, 100.0],
        hc_m_arr=[0.0, 0.0],
        pin_obj=object(),
        filter_key="linear",
    )

    assert out["ok"] is True
    assert out["metrics"]["rank_score"] == 84.0
    assert len(calls) == 1


# --- calibrated_auto_quality regression tests ---

from decaycore.auto_mode.rank_score import calibrated_auto_quality, _quality_band


def test_calibrated_auto_quality_known_good_case():
    # Known-good case: internal rank_score ~48 should map to ~78-83
    score = calibrated_auto_quality(48.0)
    assert 76.0 <= score <= 83.0, f"Expected 76-83 for rank=48, got {score:.1f}"


def test_calibrated_auto_quality_hard_gate_capped_below_60():
    metrics = {
        "stereo_policy_gate_failed": True,
        "max_net_boost_db": 4.0,
        "events_total": 0,
    }
    score = calibrated_auto_quality(70.0, metrics)
    assert score <= 59.0, f"Hard gate should cap at 59, got {score:.1f}"


def test_calibrated_auto_quality_residual_peak_hard_gate_capped():
    metrics = {
        "worst_residual_peak_db": 8.0,
        "residual_peak_hard_gate_db": 7.0,
    }
    score = calibrated_auto_quality(70.0, metrics)
    assert score <= 59.0, f"Residual peak hard gate should cap at 59, got {score:.1f}"


def test_calibrated_auto_quality_severe_boost_capped_at_69():
    metrics = {
        "max_net_boost_db": 8.0,
        "events_total": 0,
    }
    score = calibrated_auto_quality(60.0, metrics)
    assert score <= 69.0, f"Severe boost should cap at 69, got {score:.1f}"


def test_calibrated_auto_quality_severe_events_can_reach_100_cap():
    metrics = {
        "event_penalty": 4.0,
        "max_net_boost_db": 3.0,
    }
    score = calibrated_auto_quality(65.0, metrics)
    uncapped = calibrated_auto_quality(65.0, {"event_penalty": 0.0, "max_net_boost_db": 3.0})
    assert score == uncapped, f"Severe event_penalty should not cap below 100, got {score:.1f}"


def test_calibrated_auto_quality_residual_peak_penalty_can_reach_100_cap():
    metrics = {
        "residual_peak_penalty": 2.0,
        "max_net_boost_db": 3.0,
    }
    score = calibrated_auto_quality(65.0, metrics)
    uncapped = calibrated_auto_quality(65.0, {"residual_peak_penalty": 0.0, "max_net_boost_db": 3.0})
    assert score == uncapped, f"Residual peak penalty should not cap below 100, got {score:.1f}"


def test_calibrated_auto_quality_small_event_count_not_capped():
    # Many events but low penalty → no cap
    metrics = {
        "events_total": 10,
        "event_penalty": 1.0,
        "max_net_boost_db": 3.0,
    }
    score = calibrated_auto_quality(65.0, metrics)
    assert score > 72.0, f"Low event_penalty should not cap at 72, got {score:.1f}"


def test_calibrated_auto_quality_excellent_result_above_90():
    score = calibrated_auto_quality(92.0)
    assert score >= 90.0, f"Excellent rank should give >90, got {score:.1f}"


def test_calibrated_auto_quality_monotonic():
    # Higher rank_score always gives higher display score (no caps active)
    scores = [calibrated_auto_quality(r) for r in [30.0, 40.0, 50.0, 60.0, 75.0, 90.0]]
    for i in range(len(scores) - 1):
        assert scores[i] < scores[i + 1], f"Not monotonic at index {i}: {scores}"


def test_quality_band_labels():
    assert _quality_band(95.0) == "Excellent"
    assert _quality_band(85.0) == "Good"
    assert _quality_band(75.0) == "Usable"
    assert _quality_band(65.0) == "Weak"
    assert _quality_band(50.0) == "Poor"


def test_calibrated_auto_quality_nan_input():
    import math
    score = calibrated_auto_quality(float("nan"))
    assert math.isnan(score)


def test_calibrated_auto_quality_bass_cancellation_capped_at_70():
    metrics = {
        "bass_cancellation_risk": 0.8,
    }
    score = calibrated_auto_quality(75.0, metrics)
    assert score <= 70.0, f"High cancellation risk should cap at 70, got {score:.1f}"
