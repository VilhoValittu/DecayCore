from __future__ import annotations

from decaycore.config.decaycore_pipeline import collect_ui_data
from decaycore.engine_run_parts.engine_run_02 import _stats_level_comp_factor
from decaycore.measurement.devices import get_default_input_device_index
from decaycore.ui.export_outputs import _direct_dac_yaml_export_settings
from decaycore.ui.results_formatters import safe_float


def test_collect_ui_data_invalid_numeric_inputs_fallback_to_safe_defaults() -> None:
    data = collect_ui_data(
        {
            "gain": "not-a-number",
            "auto_mode_workers": "xx",
            "bass_integration_sub_delay_ms": object(),
            "bass_integration_sub_gain_trim_db": object(),
            "bass_integration_allpass_freq_hz": "bad",
            "bass_integration_allpass_q": "bad",
            "ir_export_tukey_alpha": "bad",
        }
    )

    assert float(data["gain"]) == 0.0
    assert int(data["auto_mode_workers"]) == 0
    assert float(data["bass_integration_sub_delay_ms"]) == 0.0
    assert float(data["bass_integration_sub_gain_trim_db"]) == 0.0
    assert float(data["bass_integration_allpass_freq_hz"]) == 0.0
    assert float(data["bass_integration_allpass_q"]) == 0.707
    assert float(data["ir_export_tukey_alpha"]) == 0.25


def test_results_safe_float_returns_default_for_non_floatable_value() -> None:
    class _BadFloat:
        def __float__(self):
            raise ValueError("nope")

    assert safe_float(_BadFloat(), 12.5) == 12.5


def test_stats_level_comp_factor_invalid_stats_defaults_to_unity() -> None:
    st = {
        "auto_global_gain_db": "bad",
        "auto_headroom_db": object(),
    }
    assert _stats_level_comp_factor(st) == 1.0


def test_direct_dac_yaml_export_settings_bad_values_use_safe_defaults() -> None:
    out = _direct_dac_yaml_export_settings(
        {
            "bass_integration_sub_delay_ms": "bad",
            "bass_integration_sub_gain_trim_db": "bad",
            "sub_crossover_hz": "bad",
            "sub_hpf_freq": "bad",
            "direct_dac_sub_lpf_hz": "bad",
            "sub_crossover_slope": "bad",
            "sub_hpf_slope": "bad",
        },
        include_sub=True,
    )

    assert float(out["sub_delay_ms"]) == 0.0
    assert float(out["sub_gain_trim_db"]) == 0.0
    assert float(out["main_hpf_hz"]) == 80.0
    assert float(out["sub_hpf_hz"]) == 20.0
    assert float(out["sub_lpf_hz"]) >= 80.0
    assert int(out["main_hpf_order"]) == 2
    assert int(out["sub_hpf_order"]) == 2


def test_measurement_default_input_device_handles_nonstandard_default_shape(monkeypatch) -> None:
    class _WeirdSoundDevice:
        default = object()

    monkeypatch.setattr("decaycore.measurement.devices._get_sounddevice", lambda: _WeirdSoundDevice())
    assert get_default_input_device_index() is None
