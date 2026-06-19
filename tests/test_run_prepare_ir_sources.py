from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from decaycore.application.run_request import RunRequest
from decaycore.ui.ng_bridge import ProcessRunCallbacks
from decaycore.workflow.run_prepare import (
    _prepare_target_curve_and_run_context,
    _prepare_ui_and_measurements,
)
from decaycore.workflow.run_prepare_parts.target_context import (
    _build_bass_integration_metadata_unified,
    _prepare_target_curve_bass_integration_context,
    _safe_float_from_dict,
)


class _DummyUiBridge:
    def ensure_progress_bar(self) -> None:
        return None

    def toast_health_gate_result(self, hr, mode) -> bool:
        return False


class _DummySupport:
    def __init__(self) -> None:
        self.force_single_plot_fs_hz = 0
        self.pick_target_curve_label = lambda data: "Flat"
        self.slugify_filename_token = lambda text, default="target": "flat"
        self.ui_bridge = _DummyUiBridge()


def _callbacks() -> ProcessRunCallbacks:
    return ProcessRunCallbacks(
        status=lambda msg: None,
        set_auto_selected_bar=lambda msg="": None,
    )


def _measurement_tuple():
    f = np.asarray([20.0, 100.0], dtype=float)
    m = np.asarray([0.0, -3.0], dtype=float)
    p = np.asarray([0.0, 0.0], dtype=float)
    return f, m, p, f, m, p


def _patch_common(monkeypatch) -> None:
    monkeypatch.setattr("decaycore.workflow.run_prepare.compute_health", lambda data, mode: object())
    monkeypatch.setattr("decaycore.workflow.run_prepare.save_config", lambda data: None)


def test_safe_float_from_dict_rejects_invalid_positive_values() -> None:
    data = {
        "valid": "82.5",
        "zero": 0.0,
        "nan_value": float("nan"),
        "bad": object(),
    }

    assert _safe_float_from_dict(data, "valid", 80.0, positive=True) == 82.5
    assert _safe_float_from_dict(data, "zero", 80.0, positive=True) == 80.0
    assert _safe_float_from_dict(data, "nan_value", 80.0, positive=True) == 80.0
    assert _safe_float_from_dict(data, "bad", 1.5) == 1.5


def test_build_bass_integration_metadata_unified_keeps_measurement_fields_in_sync() -> None:
    data = {
        "bass_integration_profile": "safe",
        "bass_integration_sub_combine_mode": "average",
        "avr_crossover_hz": 82.5,
        "sub_crossover_hz": 82.5,
        "direct_dac_sub_lpf_hz": 96.5,
        "bass_integration_alignment_auto_applied": True,
        "bass_integration_sub_delay_ms": 3.25,
        "bass_integration_sub_polarity_invert": True,
        "bass_integration_sub_gain_trim_db": -1.5,
        "bass_integration_alignment_reason": "Alignment test",
        "bass_integration_allpass_auto_enable": True,
        "bass_integration_allpass_auto_applied": True,
        "bass_integration_allpass_freq_hz": 77.5,
        "bass_integration_allpass_q": 0.9,
        "bass_integration_allpass_reason": "Allpass test",
        "sub_crossover_manual_override": False,
        "local_path_l_main": "L-main.wav",
        "local_path_r_main": "R-main.wav",
        "local_path_l_sub": "L-sub.wav",
        "local_path_r_sub": "R-sub.wav",
    }
    bi_state = {
        "bi_recommended_xo_hz": 82.5,
        "bi_recommended_sub_lpf_hz": 96.5,
        "bi_rec_xo_l": 81.0,
        "bi_rec_xo_r": 84.0,
        "bi_selected_diagnostics": {"selected_only": 2.5},
        "bi_alignment_recommendation": {
            "improvement_score": 0.75,
            "baseline": {"overlap_ripple_db": 4.5},
            "optimized": {"overlap_ripple_db": 2.0},
        },
        "bi_allpass_recommendation": {
            "improvement_score": 0.22,
            "baseline": {"xo_gd_mismatch_ms": 1.4},
            "optimized": {"xo_gd_mismatch_ms": 0.7},
        },
    }

    meta, measurements_updates = _build_bass_integration_metadata_unified(
        data=data,
        bi_state=bi_state,
        bundle_diagnostics={"sub_combine_mode": "dual_sub_peak_aligned_average", "bundle_only": 1.25},
    )

    assert meta["sub_combine_mode"] == "dual_sub_peak_aligned_average"
    assert meta["diagnostics"]["bundle_only"] == 1.25
    assert meta["diagnostics"]["selected_only"] == 2.5
    assert meta["alignment"]["delay_ms"] == measurements_updates["bass_integration_sub_delay_ms"] == 3.25
    assert meta["alignment"]["polarity_invert"] == measurements_updates["bass_integration_sub_polarity_invert"] is True
    assert meta["alignment"]["gain_trim_db"] == measurements_updates["bass_integration_sub_gain_trim_db"] == -1.5
    assert meta["avr_crossover_hz"] == measurements_updates["avr_crossover_hz"] == 82.5
    assert meta["direct_dac_sub_lpf_hz"] == measurements_updates["direct_dac_sub_lpf_hz"] == 96.5
    assert meta["recommended_allpass"]["enabled"] == measurements_updates["bass_integration_allpass_auto_applied"] is True
    assert meta["recommended_allpass"]["freq_hz"] == measurements_updates["bass_integration_allpass_freq_hz"] == 77.5
    assert meta["recommended_allpass"]["q"] == measurements_updates["bass_integration_allpass_q"] == 0.9


def test_prepare_target_curve_context_keeps_bass_integration_metadata_consistent(monkeypatch) -> None:
    _patch_common(monkeypatch)
    bundle = SimpleNamespace(
        diagnostics={
            "sub_combine_mode": "dual_sub_peak_aligned_average",
            "bundle_only": 1.25,
        }
    )

    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.load_measurements_lr",
        lambda data, logger=None: (_ for _ in ()).throw(AssertionError("unexpected regular measurement load")),
    )
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.load_bass_integration_measurements",
        lambda data, logger=None: (bundle, *_measurement_tuple()),
    )
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.load_raw_irs_lr",
        lambda data, logger=None, **kwargs: (
            np.asarray([1.0], dtype=float),
            48000,
            np.asarray([2.0], dtype=float),
            48000,
        ),
    )
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.load_raw_ir_sub",
        lambda data, logger=None: (np.asarray([3.0], dtype=float), 48000),
    )
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.load_house_curve",
        lambda data, parse_measurements_from_path=None: (None, None, "none"),
    )
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.build_xos_hpf",
        lambda data: ([], {"enabled": False}),
    )
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.choose_target_rates",
        lambda data: [48000],
    )
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.choose_dash_fs",
        lambda target_rates, *, multi_rate_on=False, forced_plot_fs_hz=0: 48000,
    )
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.detect_is_wav_source",
        lambda data: False,
    )
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.filter_type_short",
        lambda value: "Linear",
    )
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.log_df_smoothing_toggle",
        lambda data, logger: None,
    )

    def _fake_prepare_bi(*, ctx, data, callbacks):
        data["bass_integration_mode"] = "direct_dac"
        data["bass_integration_sub_combine_mode"] = "average"
        data["bass_integration_profile"] = "safe"
        data["avr_crossover_hz"] = 82.5
        data["sub_crossover_hz"] = 82.5
        data["direct_dac_sub_lpf_hz"] = 96.5
        data["bass_integration_alignment_auto_applied"] = True
        data["bass_integration_sub_delay_ms"] = 3.25
        data["bass_integration_sub_polarity_invert"] = True
        data["bass_integration_sub_gain_trim_db"] = -1.5
        data["bass_integration_alignment_reason"] = "Alignment test"
        data["bass_integration_allpass_auto_enable"] = True
        data["bass_integration_allpass_auto_applied"] = True
        data["bass_integration_allpass_freq_hz"] = 77.5
        data["bass_integration_allpass_q"] = 0.9
        data["bass_integration_allpass_reason"] = "Allpass test"
        return {
            "bi_recommended_xo_hz": 82.5,
            "bi_recommended_sub_lpf_hz": 96.5,
            "bi_rec_xo_l": 81.0,
            "bi_rec_xo_r": 84.0,
            "bi_selected_diagnostics": {"selected_only": 2.5},
            "bi_alignment_recommendation": {
                "applied": True,
                "sub_delay_ms": 3.25,
                "sub_polarity_invert": True,
                "sub_gain_trim_db": -1.5,
                "reason": "Alignment test",
                "improvement_score": 0.75,
                "baseline": {"overlap_ripple_db": 4.5},
                "optimized": {"overlap_ripple_db": 2.0},
            },
            "bi_allpass_recommendation": {
                "enabled": True,
                "freq_hz": 77.5,
                "q": 0.9,
                "reason": "Allpass test",
                "improvement_score": 0.22,
                "baseline": {"xo_gd_mismatch_ms": 1.4},
                "optimized": {"xo_gd_mismatch_ms": 0.7},
            },
        }

    monkeypatch.setattr(
        "decaycore.workflow.run_prepare._prepare_target_curve_bass_integration_context",
        _fake_prepare_bi,
    )

    support = _DummySupport()
    ctx = _prepare_ui_and_measurements(
        request=RunRequest(
            raw_ui_data={
                "mode": "AUTO",
                "bass_integration_enable": True,
                "filter_type": "Linear",
            }
        ),
        callbacks=_callbacks(),
        support=support,
    )

    assert ctx is not None

    _prepare_target_curve_and_run_context(
        ctx,
        support=support,
    )

    meta = ctx["data"]["_bass_integration_meta"]
    measurements = ctx["measurements"]

    assert meta["diagnostics"]["bundle_only"] == 1.25
    assert meta["diagnostics"]["selected_only"] == 2.5
    assert meta["sub_combine_mode"] == measurements["bass_integration_sub_combine_mode"] == "dual_sub_peak_aligned_average"
    assert meta["avr_crossover_hz"] == measurements["avr_crossover_hz"] == 82.5
    assert meta["direct_dac_sub_lpf_hz"] == measurements["direct_dac_sub_lpf_hz"] == 96.5
    assert meta["alignment"]["delay_ms"] == measurements["bass_integration_sub_delay_ms"] == 3.25
    assert meta["alignment"]["polarity_invert"] == measurements["bass_integration_sub_polarity_invert"] is True
    assert meta["alignment"]["gain_trim_db"] == measurements["bass_integration_sub_gain_trim_db"] == -1.5
    assert meta["recommended_allpass"]["enabled"] == measurements["bass_integration_allpass_auto_applied"] is True
    assert meta["recommended_allpass"]["freq_hz"] == measurements["bass_integration_allpass_freq_hz"] == 77.5
    assert meta["recommended_allpass"]["q"] == measurements["bass_integration_allpass_q"] == 0.9


def test_prepare_target_curve_bass_integration_context_applies_prepare_recommendation(monkeypatch) -> None:
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare_parts.target_context._compute_direct_dac_prepare_recommendation",
        lambda bundle, data, callbacks=None: {
            "applied": True,
            "sub_delay_ms": -2.5,
            "sub_array_delay_ms": -2.5,
            "main_l_delay_ms": 2.5,
            "main_r_delay_ms": 2.5,
            "sub_polarity_invert": False,
            "sub_gain_trim_db": -1.25,
            "recommended_hz": 80.0,
            "recommended_sub_lpf_hz": 100.0,
            "baseline": {"overlap_ripple_db": 4.5},
            "optimized": {"overlap_ripple_db": 2.0},
            "improvement_score": 0.75,
            "reason": "Applied shared Direct-DAC sub-array polarity/delay/gain alignment.",
            "allpass_enabled": True,
            "allpass_freq_hz": 77.5,
            "allpass_q": 0.9,
            "allpass_reason": "Allpass test",
        },
    )
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare_parts.target_context._refresh_target_curve_bass_integration_diagnostics",
        lambda **kwargs: {"selected_only": 2.5},
    )

    data = {
        "mode": "AUTO",
        "bass_integration_enable": True,
        "bass_integration_profile": "safe",
        "bass_integration_sub_combine_mode": "average",
        "bass_integration_allpass_auto_enable": True,
        "avr_crossover_hz": 80.0,
        "sub_crossover_hz": 80.0,
        "direct_dac_sub_lpf_hz": 100.0,
        "sub_crossover_slope": 24,
        "sub_hpf_slope": 12,
        "sub_hpf_freq": 20.0,
    }
    ctx = {"bass_integration_bundle": SimpleNamespace(diagnostics={})}

    result = _prepare_target_curve_bass_integration_context(
        ctx=ctx,
        data=data,
        callbacks=_callbacks(),
    )

    assert result["bi_alignment_recommendation"]["applied"] is True
    assert data["bass_integration_sub_delay_ms"] == -2.5
    assert data["bass_integration_sub_array_delay_ms"] == -2.5
    assert data["bass_integration_main_l_delay_ms"] == 2.5
    assert data["bass_integration_main_r_delay_ms"] == 2.5
    assert data["bass_integration_sub_gain_trim_db"] == -1.25
    assert data["bass_integration_alignment_auto_applied"] is True
    assert data["bass_integration_allpass_auto_enable"] is True
    assert data["bass_integration_allpass_auto_applied"] is True
    assert data["bass_integration_allpass_freq_hz"] == 77.5
    assert data["bass_integration_allpass_q"] == 0.9


def test_prepare_ui_reads_lr_raw_ir_from_regular_slots(monkeypatch) -> None:
    _patch_common(monkeypatch)
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.load_measurements_lr",
        lambda data, logger=None: _measurement_tuple(),
    )
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.load_bass_integration_measurements",
        lambda data, logger=None: (_ for _ in ()).throw(AssertionError("unexpected bass integration load")),
    )

    def _fake_load_raw_irs_lr(data, *, logger=None, **kwargs):
        calls["lr_kwargs"] = dict(kwargs)
        return np.asarray([1.0], dtype=float), 48000, np.asarray([2.0], dtype=float), 48000

    monkeypatch.setattr("decaycore.workflow.run_prepare.load_raw_irs_lr", _fake_load_raw_irs_lr)
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.load_raw_ir_sub",
        lambda data, logger=None: (_ for _ in ()).throw(AssertionError("sub IR should stay inactive")),
    )

    ctx = _prepare_ui_and_measurements(
        request=RunRequest(raw_ui_data={"mode": "AUTO", "bass_integration_enable": False}),
        callbacks=_callbacks(),
        support=_DummySupport(),
    )

    assert ctx is not None
    assert calls["lr_kwargs"] == {}
    assert np.array_equal(ctx["raw_ir_l"], np.asarray([1.0], dtype=float))
    assert np.array_equal(ctx["raw_ir_r"], np.asarray([2.0], dtype=float))
    assert ctx["raw_ir_sub"] is None
    assert ctx["raw_ir_fs_sub"] == 0


def test_prepare_ui_reads_bass_integration_raw_ir_from_main_and_sub_slots(monkeypatch) -> None:
    _patch_common(monkeypatch)
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.load_measurements_lr",
        lambda data, logger=None: (_ for _ in ()).throw(AssertionError("unexpected regular measurement load")),
    )
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.load_bass_integration_measurements",
        lambda data, logger=None: ("bundle", *_measurement_tuple()),
    )

    def _fake_load_raw_irs_lr(data, *, logger=None, **kwargs):
        calls["lr_kwargs"] = dict(kwargs)
        return np.asarray([3.0], dtype=float), 48000, np.asarray([4.0], dtype=float), 48000

    def _fake_load_raw_ir_sub(data, *, logger=None):
        calls["sub_called"] = True
        return np.asarray([5.0], dtype=float), 48000

    monkeypatch.setattr("decaycore.workflow.run_prepare.load_raw_irs_lr", _fake_load_raw_irs_lr)
    monkeypatch.setattr("decaycore.workflow.run_prepare.load_raw_ir_sub", _fake_load_raw_ir_sub)

    ctx = _prepare_ui_and_measurements(
        request=RunRequest(raw_ui_data={"mode": "AUTO", "bass_integration_enable": True}),
        callbacks=_callbacks(),
        support=_DummySupport(),
    )

    assert ctx is not None
    assert calls["lr_kwargs"] == {
        "file_key_l": "file_l_main",
        "path_key_l": "local_path_l_main",
        "file_key_r": "file_r_main",
        "path_key_r": "local_path_r_main",
    }
    assert calls["sub_called"] is True
    assert ctx["bass_integration_bundle"] == "bundle"
    assert np.array_equal(ctx["raw_ir_l"], np.asarray([3.0], dtype=float))
    assert np.array_equal(ctx["raw_ir_r"], np.asarray([4.0], dtype=float))
    assert np.array_equal(ctx["raw_ir_sub"], np.asarray([5.0], dtype=float))
    assert ctx["raw_ir_fs_sub"] == 48000


def test_prepare_ui_prefers_generated_measurement_pair_when_available(monkeypatch) -> None:
    _patch_common(monkeypatch)
    generated = (
        np.asarray([20.0, 100.0], dtype=float),
        np.asarray([0.0, -1.0], dtype=float),
        np.asarray([0.0, 0.0], dtype=float),
        np.asarray([20.0, 100.0], dtype=float),
        np.asarray([0.0, -1.5], dtype=float),
        np.asarray([0.0, 0.0], dtype=float),
        np.asarray([6.0], dtype=float),
        48000,
        np.asarray([7.0], dtype=float),
        48000,
        None,
        None,
        None,
        None,
    )

    monkeypatch.setattr(
        "decaycore.workflow.run_prepare._load_generated_measurement_pair",
        lambda data: generated,
    )
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.load_measurements_lr",
        lambda data, logger=None: (_ for _ in ()).throw(AssertionError("regular measurement load should be skipped")),
    )
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.load_bass_integration_measurements",
        lambda data, logger=None: (_ for _ in ()).throw(AssertionError("bass integration load should stay inactive")),
    )
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.load_raw_irs_lr",
        lambda data, logger=None, **kwargs: (_ for _ in ()).throw(AssertionError("generated pair already supplies raw IR")),
    )
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.load_raw_ir_sub",
        lambda data, logger=None: (_ for _ in ()).throw(AssertionError("sub IR should stay inactive")),
    )

    ctx = _prepare_ui_and_measurements(
        request=RunRequest(raw_ui_data={"mode": "AUTO", "bass_integration_enable": False}),
        callbacks=_callbacks(),
        support=_DummySupport(),
    )

    assert ctx is not None
    assert np.array_equal(ctx["raw_ir_l"], np.asarray([6.0], dtype=float))
    assert np.array_equal(ctx["raw_ir_r"], np.asarray([7.0], dtype=float))
    assert ctx["raw_ir_fs_l"] == 48000
    assert ctx["raw_ir_fs_r"] == 48000


def test_prepare_target_curve_context_preserves_harmonic_sidecars(monkeypatch) -> None:
    _patch_common(monkeypatch)

    harmonic_left = (
        np.asarray([50.0, 100.0], dtype=float),
        np.asarray([-60.0, -70.0], dtype=float),
    )
    harmonic_right = (
        np.asarray([60.0, 120.0], dtype=float),
        np.asarray([-55.0, -68.0], dtype=float),
    )

    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.load_measurements_lr",
        lambda data, logger=None: _measurement_tuple(),
    )
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.load_bass_integration_measurements",
        lambda data, logger=None: (_ for _ in ()).throw(AssertionError("unexpected bass integration load")),
    )
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.load_raw_irs_lr",
        lambda data, logger=None, **kwargs: (
            np.asarray([1.0], dtype=float),
            48000,
            np.asarray([2.0], dtype=float),
            48000,
        ),
    )
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.load_raw_ir_sub",
        lambda data, logger=None: (_ for _ in ()).throw(AssertionError("sub IR should stay inactive")),
    )

    def _fake_try_load_harmonic_sidecar(path: str):
        if path == "left.txt":
            return harmonic_left
        if path == "right.txt":
            return harmonic_right
        return None, None

    monkeypatch.setattr(
        "decaycore.workflow.run_prepare._try_load_harmonic_sidecar",
        _fake_try_load_harmonic_sidecar,
    )
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.load_house_curve",
        lambda data, parse_measurements_from_path=None: (None, None, "none"),
    )
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.build_xos_hpf",
        lambda data: ([], {"enabled": False}),
    )
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.choose_target_rates",
        lambda data: [48000],
    )
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.choose_dash_fs",
        lambda target_rates, *, multi_rate_on=False, forced_plot_fs_hz=0: 48000,
    )
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.detect_is_wav_source",
        lambda data: False,
    )
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.filter_type_short",
        lambda value: "Linear",
    )
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.log_df_smoothing_toggle",
        lambda data, logger: None,
    )

    support = _DummySupport()
    ctx = _prepare_ui_and_measurements(
        request=RunRequest(
            raw_ui_data={
                "mode": "AUTO",
                "bass_integration_enable": False,
                "filter_type": "Linear",
                "local_path_l": "left.txt",
                "local_path_r": "right.txt",
            }
        ),
        callbacks=_callbacks(),
        support=support,
    )

    assert ctx is not None

    _prepare_target_curve_and_run_context(
        ctx,
        support=support,
    )

    measurements = ctx["measurements"]
    assert np.array_equal(measurements["harmonic_freq_hz_l"], harmonic_left[0])
    assert np.array_equal(measurements["harmonic_magnitudes_db_l"], harmonic_left[1])
    assert np.array_equal(measurements["harmonic_freq_hz_r"], harmonic_right[0])
    assert np.array_equal(measurements["harmonic_magnitudes_db_r"], harmonic_right[1])


def test_prepare_target_curve_context_keeps_saved_measurement_rt60_for_txt_sources(monkeypatch) -> None:
    _patch_common(monkeypatch)

    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.load_measurements_lr",
        lambda data, logger=None: _measurement_tuple(),
    )
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.load_bass_integration_measurements",
        lambda data, logger=None: (_ for _ in ()).throw(AssertionError("unexpected bass integration load")),
    )
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.load_raw_irs_lr",
        lambda data, logger=None, **kwargs: (
            np.asarray([1.0], dtype=float),
            48000,
            np.asarray([2.0], dtype=float),
            48000,
        ),
    )
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.load_raw_ir_sub",
        lambda data, logger=None: (_ for _ in ()).throw(AssertionError("sub IR should stay inactive")),
    )
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.load_house_curve",
        lambda data, parse_measurements_from_path=None: (None, None, "none"),
    )
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.build_xos_hpf",
        lambda data: ([], {"enabled": False}),
    )
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.choose_target_rates",
        lambda data: [48000],
    )
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.choose_dash_fs",
        lambda target_rates, *, multi_rate_on=False, forced_plot_fs_hz=0: 48000,
    )
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.detect_is_wav_source",
        lambda data: False,
    )
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.filter_type_short",
        lambda value: "Linear",
    )
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.log_df_smoothing_toggle",
        lambda data, logger: None,
    )

    support = _DummySupport()
    ctx = _prepare_ui_and_measurements(
        request=RunRequest(
            raw_ui_data={
                "mode": "AUTO",
                "bass_integration_enable": False,
                "filter_type": "Linear",
                "local_path_l": "left.txt",
                "local_path_r": "right.txt",
                "generated_measurement_l": {
                    "measured_rt60": 0.41,
                    "measured_rt60_bands": {"125.0": 0.35, "250.0": 0.45},
                },
                "generated_measurement_r": {
                    "measured_rt60": 0.53,
                    "measured_rt60_bands": {"250.0": 0.5, "500.0": 0.6},
                },
            }
        ),
        callbacks=_callbacks(),
        support=support,
    )

    assert ctx is not None
    assert ctx["measured_rt60_l"] == 0.41
    assert ctx["measured_rt60_bands_l"] == {125.0: 0.35, 250.0: 0.45}
    assert ctx["measured_rt60_r"] == 0.53
    assert ctx["measured_rt60_bands_r"] == {250.0: 0.5, 500.0: 0.6}

    _prepare_target_curve_and_run_context(
        ctx,
        support=support,
    )

    measurements = ctx["measurements"]
    assert measurements["measured_rt60_l"] == 0.41
    assert measurements["measured_rt60_bands_l"] == {125.0: 0.35, 250.0: 0.45}
    assert measurements["measured_rt60_r"] == 0.53
    assert measurements["measured_rt60_bands_r"] == {250.0: 0.5, 500.0: 0.6}


def test_prepare_ui_loads_rt60_from_local_measurement_metadata_sidecar(monkeypatch, tmp_path) -> None:
    _patch_common(monkeypatch)

    left_path = tmp_path / "left_final__ir.wav"
    right_path = tmp_path / "right_final__ir.wav"
    left_path.write_bytes(b"RIFF")
    right_path.write_bytes(b"RIFF")
    (tmp_path / "left_final__metadata.json").write_text(
        '{"rt60_val": 0.44, "rt60_bands": {"125.0": 0.4, "250.0": 0.48}}',
        encoding="utf-8",
    )
    (tmp_path / "right_final__metadata.json").write_text(
        '{"rt60_val": 0.51, "rt60_bands": {"250.0": 0.49, "500.0": 0.57}}',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.load_measurements_lr",
        lambda data, logger=None: _measurement_tuple(),
    )
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.load_bass_integration_measurements",
        lambda data, logger=None: (_ for _ in ()).throw(AssertionError("unexpected bass integration load")),
    )
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.load_raw_irs_lr",
        lambda data, logger=None, **kwargs: (
            np.asarray([1.0], dtype=float),
            48000,
            np.asarray([2.0], dtype=float),
            48000,
        ),
    )
    monkeypatch.setattr(
        "decaycore.workflow.run_prepare.load_raw_ir_sub",
        lambda data, logger=None: (_ for _ in ()).throw(AssertionError("sub IR should stay inactive")),
    )

    ctx = _prepare_ui_and_measurements(
        request=RunRequest(
            raw_ui_data={
                "mode": "AUTO",
                "bass_integration_enable": False,
                "local_path_l": str(left_path),
                "local_path_r": str(right_path),
            }
        ),
        callbacks=_callbacks(),
        support=_DummySupport(),
    )

    assert ctx is not None
    assert ctx["measured_rt60_l"] == 0.44
    assert ctx["measured_rt60_bands_l"] == {125.0: 0.4, 250.0: 0.48}
    assert ctx["measured_rt60_r"] == 0.51
    assert ctx["measured_rt60_bands_r"] == {250.0: 0.49, 500.0: 0.57}
