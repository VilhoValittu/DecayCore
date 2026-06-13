from types import SimpleNamespace

import numpy as np

from decaycore.dsp.correction_types import _MagPostProcessInputs
from decaycore.dsp.correction_types import MeasurementSideContext
from decaycore.dsp._measurement_ctx_local import clear_measurement_ctx, set_measurement_ctx
from decaycore.dsp.dsp_utils import cfg_float_allow_zero
from decaycore.dsp.mag_authority_trace import (
    REASON_LOW_BASS_CUTS_ONLY,
    REASON_LOW_BASS_FLOOR_REAPPLIED,
    REASON_USER_BOOST_CAP,
    build_mag_authority_stage,
    summarize_mag_authority_trace,
)
from decaycore.dsp.mag_post_limits import apply_post_limits_and_metrics


class _NullLogger:
    def info(self, *_args, **_kwargs):
        return None


def _identity_mid_refit(gain_db, *_args, **_kwargs):
    return np.asarray(gain_db, dtype=float)


def _stage_probe(*args, **_kwargs):
    return {"stage": str(args[0]) if args else "unknown"}


def test_mag_authority_trace_noop_has_no_active_reason():
    freq = np.geomspace(20.0, 200.0, 16)
    curve = np.zeros_like(freq)

    stage = build_mag_authority_stage(
        "noop",
        curve,
        curve,
        freq,
        np.ones_like(freq, dtype=bool),
        reason_codes=[REASON_USER_BOOST_CAP],
    )

    assert stage["changed_bins"] == 0
    assert stage["reason_codes"] == []
    assert stage["max_delta_db"] == 0.0


def test_mag_authority_trace_boost_clamp_reports_reason():
    freq = np.asarray([20.0, 40.0, 80.0, 160.0], dtype=float)
    before = np.asarray([0.0, 4.0, 5.0, -1.0], dtype=float)
    after = np.asarray([0.0, 2.0, 2.0, -1.0], dtype=float)

    stage = build_mag_authority_stage(
        "after_hardclamp",
        before,
        after,
        freq,
        np.ones_like(freq, dtype=bool),
        reason_codes=[REASON_USER_BOOST_CAP],
    )

    assert stage["reason_codes"] == [REASON_USER_BOOST_CAP]
    assert stage["changed_bins"] == 2
    assert stage["authority_only_reduced"] is True
    assert stage["boost_peak_before_db"] == 5.0
    assert stage["boost_peak_after_db"] == 2.0


def test_mag_authority_trace_low_bass_reapply_reports_reasons():
    freq = np.asarray([20.0, 25.0, 40.0, 80.0], dtype=float)
    before = np.asarray([1.0, 0.5, -0.5, 0.0], dtype=float)
    after = np.asarray([0.0, -1.0, -0.5, 0.0], dtype=float)

    stage = build_mag_authority_stage(
        "after_lowbass_hard_reapply",
        before,
        after,
        freq,
        np.ones_like(freq, dtype=bool),
        reason_codes=[REASON_LOW_BASS_CUTS_ONLY, REASON_LOW_BASS_FLOOR_REAPPLIED],
    )
    summary = summarize_mag_authority_trace([stage])

    assert stage["bass_changed_bins_20_200"] == 2
    assert REASON_LOW_BASS_CUTS_ONLY in summary["mag_authority_trace_active_reasons"]
    assert REASON_LOW_BASS_FLOOR_REAPPLIED in summary["mag_authority_trace_active_reasons"]


def test_mag_authority_trace_sanitizes_non_finite_json_values():
    import json

    freq = np.asarray([20.0, 40.0, 80.0], dtype=float)
    before = np.asarray([np.nan, np.inf, -np.inf], dtype=float)
    after = np.asarray([0.0, 1.0, -1.0], dtype=float)

    stage = build_mag_authority_stage(
        "sanitize",
        before,
        after,
        freq,
        np.ones_like(freq, dtype=bool),
        reason_codes=["sanitize"],
    )
    summary = summarize_mag_authority_trace([stage])

    json.dumps({"stage": stage, "summary": summary}, allow_nan=False)


def _make_inputs(
    *,
    gain_apply=None,
    gain_db=None,
    pre_bass_adapt_g=None,
    st=None,
    cfg=None,
    n: int = 128,
):
    freq_axis = np.geomspace(10.0, 1000.0, n).astype(float)
    mask_c = np.ones_like(freq_axis, dtype=bool)
    gain_apply_arr = np.zeros_like(freq_axis) if gain_apply is None else np.asarray(gain_apply, dtype=float)
    gain_db_arr = np.zeros_like(freq_axis) if gain_db is None else np.asarray(gain_db, dtype=float)
    pre_bass = None if pre_bass_adapt_g is None else np.asarray(pre_bass_adapt_g, dtype=float)
    cfg_obj = cfg or SimpleNamespace(
        max_boost_db=2.0,
        max_cut_db=15.0,
        low_bass_cut_enable=True,
        low_bass_cut_hz=40.0,
        low_bass_cut_strength=1.0,
        exc_prot=False,
        exc_freq=0.0,
        bass_boost_cap_enable=False,
        bass_boost_post_restore_enable=False,
        reg_strength=0.0,
        is_wav_source=False,
        mag_c_min=20.0,
        mag_c_max=400.0,
        trans_width=80.0,
        max_slope_db_per_oct=0.0,
        max_slope_boost_db_per_oct=0.0,
        max_slope_cut_db_per_oct=0.0,
        conf_pull_floor=0.05,
        conf_pull_gamma_cut=0.55,
        mid_refit_hz_lo=200.0,
        mid_refit_hz_hi=2000.0,
        do_normalize=False,
        global_gain_db=0.0,
    )
    stats = {} if st is None else st
    return (
        freq_axis,
        stats,
        _MagPostProcessInputs(
            cfg=cfg_obj,
            freq_axis=freq_axis,
            st=stats,
            logger=_NullLogger(),
            stage_probe=_stage_probe,
            cfg_float_allow_zero=cfg_float_allow_zero,
            mask_c=mask_c,
            gain_db=gain_db_arr.copy(),
            gain_apply=gain_apply_arr.copy(),
            raw_g=np.asarray(gain_apply_arr, dtype=float).copy(),
            final_g=np.asarray(gain_apply_arr, dtype=float).copy(),
            pre_bass_adapt_g=pre_bass,
            raw_safe_ref=np.zeros_like(freq_axis),
            conf_mask=np.ones_like(freq_axis),
            filter_smooth=12.0,
            debug_stage_stats=False,
            stage_probes={},
            apply_confidence_weighted_target_pull=lambda **kwargs: (
                np.asarray(kwargs["target_db"], dtype=float),
                {},
            ),
            m_anal=np.zeros_like(freq_axis),
            target_mags=np.zeros_like(freq_axis),
            calc_offset_db=0.0,
        ),
    )


def _transition_cfg(*, trans_width: float) -> SimpleNamespace:
    return SimpleNamespace(
        max_boost_db=12.0,
        max_cut_db=24.0,
        low_bass_cut_enable=False,
        low_bass_cut_hz=0.0,
        low_bass_cut_strength=0.0,
        exc_prot=False,
        exc_freq=0.0,
        bass_boost_cap_enable=False,
        bass_boost_post_restore_enable=False,
        reg_strength=0.0,
        is_wav_source=False,
        mag_c_min=20.0,
        mag_c_max=400.0,
        trans_width=trans_width,
        max_slope_db_per_oct=0.0,
        max_slope_boost_db_per_oct=0.0,
        max_slope_cut_db_per_oct=0.0,
        conf_pull_floor=0.05,
        conf_pull_gamma_cut=0.55,
        mid_refit_hz_lo=200.0,
        mid_refit_hz_hi=2000.0,
        do_normalize=False,
        global_gain_db=0.0,
    )


def test_apply_post_limits_and_metrics_preserves_array_shape():
    freq_axis, st, inputs = _make_inputs(gain_apply=np.linspace(-1.0, 1.0, 128))

    out = apply_post_limits_and_metrics(inputs, apply_mid_refit_pre_slope=_identity_mid_refit)

    assert out.gain_db.shape == freq_axis.shape
    assert out.stage_probes
    assert isinstance(st, dict)


def test_transition_fade_wide_width_covers_full_correction_band():
    # When trans_width exceeds the correction band, the fade covers the full band.
    # The effective fade width = max(trans_width, 0.25 * band_width), so trans_width=1000
    # on a 380 Hz band gives a full-band fade (f_start ≈ mag_c_min).
    freq_axis = np.geomspace(10.0, 1000.0, 512)
    gain_apply = np.ones_like(freq_axis, dtype=float)
    cfg = _transition_cfg(trans_width=1000.0)
    _, st, inputs = _make_inputs(gain_apply=gain_apply, cfg=cfg, n=freq_axis.size)

    out = apply_post_limits_and_metrics(inputs, apply_mid_refit_pre_slope=_identity_mid_refit)

    # Fade covers the whole correction band: gain should be reduced well below mag_c_max
    mid_band = (freq_axis > 100.0) & (freq_axis < 350.0)
    assert np.any(out.gain_db[mid_band] < 1.0)
    assert st["mag_transition_fade_applied"] is True
    assert "mag_transition_slope_abs_max_db_per_oct" in st


def test_transition_fade_preserves_normal_width_start_frequency():
    freq_axis = np.geomspace(10.0, 1000.0, 512)
    gain_apply = np.ones_like(freq_axis, dtype=float)
    cfg = _transition_cfg(trans_width=80.0)
    _, _, inputs = _make_inputs(gain_apply=gain_apply, cfg=cfg, n=freq_axis.size)

    out = apply_post_limits_and_metrics(inputs, apply_mid_refit_pre_slope=_identity_mid_refit)

    # With trans_width=80 and band_width=380, effective width = max(80, 0.25*380=95) = 95.
    # f_start = 400 - 95 = 305 Hz.
    below_new_start = freq_axis <= 305.0
    normal_fade = (freq_axis > 305.0) & (freq_axis < 400.0)
    assert np.allclose(out.gain_db[below_new_start], 1.0, atol=1e-9, rtol=1e-9)
    assert np.any(out.gain_db[normal_fade] < 1.0)


def test_low_frequency_guard_blocks_forbidden_boost_cases():
    freq_axis = np.geomspace(10.0, 1000.0, 128)
    gain_apply = np.where(freq_axis <= 40.0, 6.0, 0.5)
    _, st, inputs = _make_inputs(gain_apply=gain_apply)

    out = apply_post_limits_and_metrics(inputs, apply_mid_refit_pre_slope=_identity_mid_refit)
    lf_mask = freq_axis <= 40.0

    assert np.any(lf_mask)
    assert float(np.max(out.gain_db[lf_mask])) <= 1e-9
    assert st["mag_authority_trace_version"] == 1
    trace = st["mag_authority_trace"]
    assert any(
        item["stage"] == "after_lowbass_policy"
        and "low_bass_cuts_only" in item["reason_codes"]
        for item in trace
    )


def test_realized_metric_fields_are_emitted_when_expected():
    st = {
        "mid_refit_err_rms_before_stage_local": 1.5,
        "bass_adaptive_smoothing_delta_rms_db_20_200_stage_local": 0.25,
        "bass_adaptive_smoothing_delta_max_db_20_200_stage_local": 0.5,
    }
    gain_apply = np.linspace(0.0, 1.0, 128)
    pre_bass = np.zeros(128, dtype=float)
    _, _, inputs = _make_inputs(gain_apply=gain_apply, pre_bass_adapt_g=pre_bass, st=st)

    _ = apply_post_limits_and_metrics(inputs, apply_mid_refit_pre_slope=_identity_mid_refit)

    assert "bass_adaptive_smoothing_delta_rms_db_20_200_realized_pre_ir" in st
    assert "bass_adaptive_smoothing_delta_max_db_20_200_realized_pre_ir" in st
    assert "mid_refit_err_rms_after_realized_pre_ir" in st


def test_minimal_synthetic_inputs_do_not_crash():
    _, _, inputs = _make_inputs(gain_apply=np.zeros(24), gain_db=np.zeros(24), n=24)

    out = apply_post_limits_and_metrics(inputs, apply_mid_refit_pre_slope=_identity_mid_refit)

    assert out is not None
    assert out.gain_db.shape == (24,)


def test_post_limit_stages_do_not_increase_boost_after_clamp_barrier():
    gain_apply = np.full(128, 8.0, dtype=float)
    cfg = SimpleNamespace(
        max_boost_db=1.0,
        max_cut_db=15.0,
        low_bass_cut_enable=False,
        low_bass_cut_hz=0.0,
        low_bass_cut_strength=0.0,
        exc_prot=False,
        exc_freq=0.0,
        bass_boost_cap_enable=False,
        bass_boost_post_restore_enable=False,
        reg_strength=0.0,
        is_wav_source=False,
        mag_c_min=20.0,
        mag_c_max=400.0,
        trans_width=80.0,
        max_slope_db_per_oct=0.0,
        max_slope_boost_db_per_oct=0.0,
        max_slope_cut_db_per_oct=0.0,
        conf_pull_floor=0.05,
        conf_pull_gamma_cut=0.55,
        mid_refit_hz_lo=200.0,
        mid_refit_hz_hi=2000.0,
        do_normalize=False,
        global_gain_db=0.0,
    )
    _, st, inputs = _make_inputs(gain_apply=gain_apply, cfg=cfg)

    out = apply_post_limits_and_metrics(inputs, apply_mid_refit_pre_slope=_identity_mid_refit)

    assert float(np.max(out.gain_db)) <= 1.0 + 1e-6
    trace = st["mag_authority_trace"]
    assert any(
        item["stage"] == "after_hardclamp"
        and "hardclamp_boost" in item["reason_codes"]
        and "user_boost_cap" in item["reason_codes"]
        for item in trace
    )


def test_softclip_trace_reports_reason_when_tanh_reduces_boost():
    st = {}
    gain_apply = np.full(128, 18.0, dtype=float)
    cfg = SimpleNamespace(
        max_boost_db=3.0,
        max_cut_db=24.0,
        low_bass_cut_enable=False,
        low_bass_cut_hz=0.0,
        low_bass_cut_strength=0.0,
        exc_prot=False,
        exc_freq=0.0,
        bass_boost_cap_enable=False,
        bass_boost_post_restore_enable=False,
        reg_strength=0.0,
        is_wav_source=False,
        mag_c_min=20.0,
        mag_c_max=400.0,
        trans_width=80.0,
        max_slope_db_per_oct=0.0,
        max_slope_boost_db_per_oct=0.0,
        max_slope_cut_db_per_oct=0.0,
        conf_pull_floor=0.05,
        conf_pull_gamma_cut=0.55,
        mid_refit_hz_lo=200.0,
        mid_refit_hz_hi=2000.0,
        do_normalize=False,
        global_gain_db=0.0,
    )
    _, _, inputs = _make_inputs(gain_apply=gain_apply, cfg=cfg, st=st)

    _ = apply_post_limits_and_metrics(inputs, apply_mid_refit_pre_slope=_identity_mid_refit)

    softclip = [item for item in st["mag_authority_trace"] if item["stage"] == "after_softclip"]
    assert softclip
    assert "softclip_boost" in softclip[0]["reason_codes"]
    assert "user_boost_cap" in softclip[0]["reason_codes"]
    assert int(softclip[0]["changed_bins"]) > 0


def test_local_bass_cap_does_not_exceed_user_max_boost():
    st = {}
    gain_apply = np.full(128, 5.0, dtype=float)
    cfg = SimpleNamespace(
        max_boost_db=2.0,
        max_cut_db=15.0,
        low_bass_cut_enable=False,
        low_bass_cut_hz=0.0,
        low_bass_cut_strength=0.0,
        exc_prot=False,
        exc_freq=0.0,
        bass_boost_cap_enable=True,
        bass_boost_cap_extra_db=5.0,
        bass_boost_cap_hz=200.0,
        bass_boost_cap_conf_min=0.55,
        bass_boost_post_restore_enable=False,
        reg_strength=0.0,
        is_wav_source=False,
        mag_c_min=20.0,
        mag_c_max=400.0,
        trans_width=80.0,
        max_slope_db_per_oct=0.0,
        max_slope_boost_db_per_oct=0.0,
        max_slope_cut_db_per_oct=0.0,
        conf_pull_floor=0.05,
        conf_pull_gamma_cut=0.55,
        mid_refit_hz_lo=200.0,
        mid_refit_hz_hi=2000.0,
        do_normalize=False,
        global_gain_db=0.0,
    )
    _, _, inputs = _make_inputs(gain_apply=gain_apply, cfg=cfg, st=st)

    out = apply_post_limits_and_metrics(inputs, apply_mid_refit_pre_slope=_identity_mid_refit)

    assert st["bass_boost_cap_enabled"] is False
    assert st["bass_boost_cap_extra_db"] == 5.0
    assert st["bass_boost_cap_max_extra_db_20_200"] == 0.0
    assert float(np.max(out.gain_db)) <= 2.0 + 1e-6


def test_boost_limit_local_cap_allow_above_user_does_not_bypass_user_cap():
    st = {}
    gain_apply = np.full(128, 5.0, dtype=float)
    cfg = SimpleNamespace(
        max_boost_db=2.0,
        max_cut_db=15.0,
        low_bass_cut_enable=False,
        low_bass_cut_hz=0.0,
        low_bass_cut_strength=0.0,
        exc_prot=False,
        exc_freq=0.0,
        bass_boost_cap_enable=True,
        bass_boost_cap_extra_db=5.0,
        bass_boost_cap_allow_above_user=True,
        bass_boost_cap_hz=200.0,
        bass_boost_cap_conf_min=0.55,
        bass_boost_post_restore_enable=False,
        reg_strength=0.0,
        is_wav_source=False,
        mag_c_min=20.0,
        mag_c_max=400.0,
        trans_width=80.0,
        max_slope_db_per_oct=0.0,
        max_slope_boost_db_per_oct=0.0,
        max_slope_cut_db_per_oct=0.0,
        conf_pull_floor=0.05,
        conf_pull_gamma_cut=0.55,
        mid_refit_hz_lo=200.0,
        mid_refit_hz_hi=2000.0,
        do_normalize=False,
        global_gain_db=0.0,
    )
    _, _, inputs = _make_inputs(gain_apply=gain_apply, cfg=cfg, st=st)

    out = apply_post_limits_and_metrics(inputs, apply_mid_refit_pre_slope=_identity_mid_refit)

    assert "bass_boost_cap_allow_above_user" not in st
    assert st["bass_boost_cap_enabled"] is False
    assert st["bass_boost_cap_max_extra_db_20_200"] == 0.0
    assert float(np.max(out.gain_db)) <= 2.0 + 1e-6


def test_small_user_allowed_bass_boost_still_passes_without_local_extra():
    st = {}
    gain_apply = np.full(128, 4.0, dtype=float)
    cfg = SimpleNamespace(
        max_boost_db=4.0,
        max_cut_db=15.0,
        low_bass_cut_enable=False,
        low_bass_cut_hz=0.0,
        low_bass_cut_strength=0.0,
        exc_prot=False,
        exc_freq=0.0,
        bass_boost_cap_enable=True,
        bass_boost_cap_extra_db=5.0,
        bass_boost_cap_hz=200.0,
        bass_boost_cap_conf_min=0.55,
        bass_boost_post_restore_enable=False,
        reg_strength=0.0,
        is_wav_source=False,
        mag_c_min=20.0,
        mag_c_max=400.0,
        trans_width=80.0,
        max_slope_db_per_oct=0.0,
        max_slope_boost_db_per_oct=0.0,
        max_slope_cut_db_per_oct=0.0,
        conf_pull_floor=0.05,
        conf_pull_gamma_cut=0.55,
        mid_refit_hz_lo=200.0,
        mid_refit_hz_hi=2000.0,
        do_normalize=False,
        global_gain_db=0.0,
    )
    _, _, inputs = _make_inputs(gain_apply=gain_apply, cfg=cfg, st=st)

    out = apply_post_limits_and_metrics(inputs, apply_mid_refit_pre_slope=_identity_mid_refit)

    assert float(np.max(out.gain_db)) <= 4.0 + 1e-6
    assert float(np.max(out.gain_db)) > 3.0
    assert st["bass_boost_cap_enabled"] is False


def test_boost_limit_telemetry_records_disabled_local_cap():
    st = {}
    gain_apply = np.full(128, 5.0, dtype=float)
    _, _, inputs = _make_inputs(gain_apply=gain_apply, st=st)

    out = apply_post_limits_and_metrics(inputs, apply_mid_refit_pre_slope=_identity_mid_refit)

    assert st["bass_boost_cap_enabled"] is False
    assert st["bass_boost_cap_max_extra_db_20_200"] == 0.0
    assert st.get("harmonic_risk_cap_enabled", False) is False
    assert float(np.max(out.gain_db)) <= 2.0 + 1e-6


def test_unsafe_raw_bypasses_harmonic_risk_boost_cap_in_manual_post_limits_path():
    st = {}
    gain_apply = np.full(128, 10.0, dtype=float)
    cfg = SimpleNamespace(
        max_boost_db=10.0,
        max_cut_db=15.0,
        low_bass_cut_enable=False,
        low_bass_cut_hz=0.0,
        low_bass_cut_strength=0.0,
        exc_prot=False,
        exc_freq=0.0,
        bass_boost_cap_enable=False,
        bass_boost_post_restore_enable=False,
        acoustic_authority_limits_enable=False,
        unsafe_raw_dsp=True,
        reg_strength=0.0,
        is_wav_source=False,
        mag_c_min=20.0,
        mag_c_max=400.0,
        trans_width=80.0,
        max_slope_db_per_oct=0.0,
        max_slope_boost_db_per_oct=0.0,
        max_slope_cut_db_per_oct=0.0,
        conf_pull_floor=0.05,
        conf_pull_gamma_cut=0.55,
        mid_refit_hz_lo=200.0,
        mid_refit_hz_hi=2000.0,
        do_normalize=False,
        global_gain_db=0.0,
    )
    _, _, inputs = _make_inputs(gain_apply=gain_apply, cfg=cfg, st=st)
    mctx = MeasurementSideContext(
        harmonic_risk_freq_hz=np.asarray([20.0, 80.0, 200.0, 800.0], dtype=float),
        harmonic_risk_curve=np.ones(4, dtype=float),
    )

    try:
        set_measurement_ctx(mctx)
        out = apply_post_limits_and_metrics(inputs, apply_mid_refit_pre_slope=_identity_mid_refit)
    finally:
        clear_measurement_ctx()

    assert float(np.max(out.gain_db)) >= 9.9
    assert st["harmonic_risk_cap_enabled"] is False
    assert st["harmonic_risk_cap_bypassed_by_unsafe_raw"] is True
    assert float(st["harmonic_risk_cap_max_reduction_20_200"]) == 0.0


def test_bass_boost_restore_trace_stays_within_caps():
    st = {}
    gain_apply = np.full(128, 2.0, dtype=float)
    cfg = SimpleNamespace(
        max_boost_db=2.0,
        max_cut_db=15.0,
        low_bass_cut_enable=False,
        low_bass_cut_hz=0.0,
        low_bass_cut_strength=0.0,
        exc_prot=False,
        exc_freq=0.0,
        bass_boost_cap_enable=False,
        bass_boost_post_restore_enable=True,
        bass_boost_post_restore_strength=1.0,
        reg_strength=0.0,
        is_wav_source=False,
        mag_c_min=20.0,
        mag_c_max=400.0,
        trans_width=80.0,
        max_slope_db_per_oct=0.0,
        max_slope_boost_db_per_oct=0.0,
        max_slope_cut_db_per_oct=0.0,
        conf_pull_floor=0.05,
        conf_pull_gamma_cut=0.55,
        mid_refit_hz_lo=200.0,
        mid_refit_hz_hi=2000.0,
        do_normalize=False,
        global_gain_db=0.0,
    )
    _, _, inputs = _make_inputs(gain_apply=gain_apply, cfg=cfg, st=st)

    out = apply_post_limits_and_metrics(inputs, apply_mid_refit_pre_slope=lambda gain_db, *_args, **_kwargs: gain_db * 0.25)

    restore = [item for item in st["mag_authority_trace"] if item["stage"] == "after_bass_boost_restore"]
    assert restore
    assert restore[0]["restored_allowed_correction"] is True
    assert float(np.max(out.gain_db)) <= 2.0 + 1e-6


def test_bass_boost_restore_does_not_relax_bass_cuts():
    st = {}
    gain_apply = np.full(128, -2.0, dtype=float)
    cfg = SimpleNamespace(
        max_boost_db=2.0,
        max_cut_db=15.0,
        low_bass_cut_enable=False,
        low_bass_cut_hz=0.0,
        low_bass_cut_strength=0.0,
        exc_prot=False,
        exc_freq=0.0,
        bass_boost_cap_enable=False,
        bass_boost_post_restore_enable=True,
        bass_boost_post_restore_strength=1.0,
        reg_strength=0.0,
        is_wav_source=False,
        mag_c_min=20.0,
        mag_c_max=400.0,
        trans_width=80.0,
        max_slope_db_per_oct=0.0,
        max_slope_boost_db_per_oct=0.0,
        max_slope_cut_db_per_oct=0.0,
        conf_pull_floor=0.05,
        conf_pull_gamma_cut=0.55,
        mid_refit_hz_lo=200.0,
        mid_refit_hz_hi=2000.0,
        do_normalize=False,
        global_gain_db=0.0,
    )
    freq_axis, _, inputs = _make_inputs(gain_apply=gain_apply, cfg=cfg, st=st)

    out = apply_post_limits_and_metrics(inputs, apply_mid_refit_pre_slope=lambda gain_db, *_args, **_kwargs: gain_db * 2.0)

    bass_mask = (freq_axis >= 25.0) & (freq_axis <= 180.0)
    assert np.any(bass_mask)
    assert float(np.max(out.gain_db[bass_mask])) < -3.9
    assert st["bass_boost_post_restore_bins"] == 0
    restore = [item for item in st["mag_authority_trace"] if item["stage"] == "after_bass_boost_restore"]
    assert restore
    assert restore[0]["changed_bins"] == 0
