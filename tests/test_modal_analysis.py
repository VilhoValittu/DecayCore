import math
from types import SimpleNamespace

import numpy as np
import pytest

from decaycore.auto_mode.scoring_metrics import _auto_score_result
from decaycore.auto_mode.scoring_metrics import compute_modal_intelligence_metrics
from decaycore.auto_mode.scoring_metrics import score_residual_peaks
from decaycore.auto_mode.cache_signature import _auto_signature_payload
import decaycore.dsp.filter_pipeline as filter_pipeline
from decaycore.config.models import FilterConfig
from decaycore.dsp.hybrid_iir import HybridIIRPolicy, design_hybrid_iir, peaking_eq_response
from decaycore.auto_mode.orchestrator_finalize import _build_modal_intelligence_debug
from decaycore.auto_mode.winner_polish import apply_residual_peak_winner_polish
from decaycore.dsp.modal_analysis import RoomModeEvent, detect_room_modes


def _bump(freq_axis, center_hz, gain_db, width_oct):
    return gain_db * np.exp(-0.5 * (np.log2(freq_axis / float(center_hz)) / float(width_oct)) ** 2)


def _assert_jsonish(value):
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        assert math.isfinite(value)
        return
    if isinstance(value, list):
        for item in value:
            _assert_jsonish(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            assert isinstance(key, str)
            _assert_jsonish(item)
        return
    raise AssertionError(f"non JSON-friendly value: {type(value)!r}")


def _room_mode_event(**overrides):
    data = {
        "freq_hz": 41.5,
        "peak_db": 8.0,
        "width_hz": 6.0,
        "width_oct": 0.12,
        "q_estimate": 6.9,
        "area_db_oct": 0.8,
        "gd_excess_ms": 35.0,
        "confidence": 0.9,
        "severity": 0.8,
        "correction_priority": 0.85,
        "cut_priority": 0.9,
        "safe_cut_db": 4.5,
        "voice_clarity_risk": 0.1,
        "kind": "room_mode",
    }
    data.update(overrides)
    return RoomModeEvent(**data)


def _hybrid_signature_payload(**overrides):
    base = {
        "filter_type": "Asymmetric",
        "auto_goal": "balanced",
        "hc_mode": "Harman6",
        "hybrid_iir_enabled": False,
    }
    base.update(overrides)
    return _auto_signature_payload(
        base_data=base,
        measurements={},
        fs_v=48000,
        taps_v=4096,
        xos=[],
        hpf=None,
    )


def test_hybrid_iir_default_min_gd_excess_is_10ms():
    cfg = FilterConfig()
    policy = HybridIIRPolicy.from_config(cfg)
    payload = _hybrid_signature_payload(hybrid_iir_enabled=True)

    assert cfg.hybrid_iir_min_gd_excess_ms == pytest.approx(10.0)
    assert policy.min_gd_excess_ms == pytest.approx(10.0)
    assert payload["hybrid_iir"]["min_gd_excess_ms"] == pytest.approx(10.0)


def test_hybrid_iir_selects_safe_room_mode_and_uses_safe_cut():
    freq = np.linspace(20.0, 200.0, 512)
    result = design_hybrid_iir(
        [_room_mode_event(peak_db=10.0, safe_cut_db=3.25)],
        freq,
        48000,
        HybridIIRPolicy(enabled=True),
    )

    assert result.enabled is True
    assert len(result.biquads) == 1
    biquad = result.biquads[0]
    assert biquad.gain_db == -3.25
    assert biquad.gain_db <= 0.0
    assert np.all(np.isfinite(result.mag_db))
    assert np.all(np.isfinite(result.phase_rad))


def test_hybrid_iir_rejects_uncertain_or_broad_candidates():
    freq = np.linspace(20.0, 200.0, 512)
    events = [
        _room_mode_event(confidence=0.2),
        _room_mode_event(q_estimate=1.2),
        _room_mode_event(freq_hz=220.0),
        _room_mode_event(peak_db=2.0),
    ]
    result = design_hybrid_iir(events, freq, 48000, HybridIIRPolicy(enabled=True))

    assert result.biquads == ()
    reasons = {item["reason"] for item in result.rejected}
    assert "confidence_below_threshold" in reasons
    assert "q_below_threshold" in reasons
    assert "outside_frequency_range" in reasons
    assert "peak_below_threshold" in reasons


def test_hybrid_iir_strong_modal_evidence_can_rescue_moderate_confidence():
    freq = np.linspace(20.0, 200.0, 512)
    policy = HybridIIRPolicy(enabled=True, min_confidence=0.40)
    result = design_hybrid_iir(
        [_room_mode_event(confidence=0.28, cut_priority=0.9, gd_excess_ms=35.0)],
        freq,
        48000,
        policy,
    )

    assert len(result.biquads) == 1
    assert result.biquads[0].confidence == pytest.approx(0.28)


def test_hybrid_iir_default_range_includes_upper_bass_schroeder_modes():
    freq = np.linspace(20.0, 220.0, 512)
    result = design_hybrid_iir(
        [_room_mode_event(freq_hz=181.0, q_estimate=7.0, width_hz=12.0)],
        freq,
        48000,
        HybridIIRPolicy(enabled=True),
    )

    assert len(result.biquads) == 1
    assert result.biquads[0].freq_hz == pytest.approx(181.0)


def test_hybrid_iir_clamps_q_and_cut():
    freq = np.linspace(20.0, 200.0, 512)
    policy = HybridIIRPolicy(enabled=True, max_cut_db=6.0, max_q=8.0)
    result = design_hybrid_iir(
        [_room_mode_event(q_estimate=20.0, safe_cut_db=12.0, peak_db=14.0)],
        freq,
        48000,
        policy,
    )

    assert len(result.biquads) == 1
    assert result.biquads[0].q == 8.0
    assert result.biquads[0].gain_db == -6.0


def test_peaking_eq_response_is_finite_cut_response():
    freq = np.linspace(20.0, 200.0, 512)
    response = peaking_eq_response(freq, 48000, 50.0, 6.0, -4.0)
    mag_db = 20.0 * np.log10(np.maximum(np.abs(response), 1e-12))

    assert np.all(np.isfinite(response))
    assert np.min(mag_db) < -1.0
    assert np.max(mag_db) <= 0.1


def test_filter_pipeline_hybrid_preconditioning_preserves_raw_and_combines_gain(monkeypatch):
    freq = np.linspace(20.0, 200.0, 512)
    raw = np.zeros_like(freq)
    target = np.zeros_like(freq)
    gain = np.zeros_like(freq)
    conf = np.ones_like(freq)
    event = _room_mode_event(freq_hz=50.0, q_estimate=6.0, safe_cut_db=4.0)
    monkeypatch.setattr(filter_pipeline, "detect_room_modes", lambda *args, **kwargs: SimpleNamespace(events=(event,)))
    cfg = FilterConfig(
        fs=48000,
        hybrid_iir_enabled=True,
        hybrid_iir_min_gd_excess_ms=0.0,
        hybrid_iir_min_peak_db=4.0,
    )

    adjusted, stats = filter_pipeline._apply_hybrid_iir_preconditioning(
        cfg=cfg,
        freq_axis=freq,
        gain_db=gain,
        m_anal=raw,
        calc_offset_db=0.0,
        target_mags=target,
        conf_mask=conf,
        st={},
    )

    iir_mag = np.asarray(stats["hybrid_iir_mag_db"], dtype=float)
    preconditioned = np.asarray(stats["hybrid_iir_preconditioned_mags"], dtype=float)
    assert len(stats["hybrid_iir_biquads"]) == 1
    assert np.allclose(preconditioned, raw + iir_mag)
    assert np.allclose(adjusted, gain - iir_mag)


def test_hybrid_iir_detects_raw_modal_error_not_fir_corrected_error(monkeypatch):
    freq = np.linspace(20.0, 200.0, 512)
    raw = np.zeros_like(freq)
    gain = np.zeros_like(freq)
    event = _room_mode_event(freq_hz=50.0, q_estimate=6.0, safe_cut_db=4.0)
    seen = {}

    def _fake_detect(*args, **kwargs):
        seen["corrected_mag_db"] = kwargs.get("corrected_mag_db")
        return SimpleNamespace(events=(event,), mode_count=1)

    monkeypatch.setattr(filter_pipeline, "detect_room_modes", _fake_detect)
    cfg = FilterConfig(fs=48000, hybrid_iir_enabled=True, hybrid_iir_min_gd_excess_ms=0.0)

    _adjusted, stats = filter_pipeline._apply_hybrid_iir_preconditioning(
        cfg=cfg,
        freq_axis=freq,
        gain_db=gain,
        m_anal=raw,
        calc_offset_db=0.0,
        target_mags=np.zeros_like(freq),
        conf_mask=np.ones_like(freq),
        st={},
    )

    assert seen["corrected_mag_db"] is None
    assert stats["hybrid_iir_modal_event_count"] == 1
    assert len(stats["hybrid_iir_biquads"]) == 1


def test_hybrid_iir_uses_phase_derived_gd_when_stats_gd_missing(monkeypatch):
    freq = np.linspace(20.0, 200.0, 512)
    raw = np.zeros_like(freq)
    gain = np.zeros_like(freq)
    event = _room_mode_event(freq_hz=50.0, q_estimate=6.0, safe_cut_db=4.0)
    seen = {}

    def _fake_detect(*args, **kwargs):
        gd = np.asarray(kwargs.get("group_delay_ms", []), dtype=float)
        seen["gd_size"] = int(gd.size)
        return SimpleNamespace(events=(event,), mode_count=1)

    phase = -2.0 * np.pi * freq * 0.003
    phase += -0.35 * np.exp(-0.5 * (np.log2(freq / 50.0) / 0.08) ** 2)
    monkeypatch.setattr(filter_pipeline, "detect_room_modes", _fake_detect)
    cfg = FilterConfig(fs=48000, hybrid_iir_enabled=True, hybrid_iir_min_gd_excess_ms=15.0)

    _adjusted, stats = filter_pipeline._apply_hybrid_iir_preconditioning(
        cfg=cfg,
        freq_axis=freq,
        gain_db=gain,
        m_anal=raw,
        calc_offset_db=0.0,
        target_mags=np.zeros_like(freq),
        conf_mask=np.ones_like(freq),
        phase_rad=phase,
        st={},
    )

    assert seen["gd_size"] == stats["hybrid_iir_modal_analysis_bin_count"]
    assert seen["gd_size"] > freq.size
    assert stats["hybrid_iir_modal_analysis_taps"] == 262144
    assert stats["hybrid_iir_modal_analysis_axis_source"] == "fixed_262144"
    assert stats["hybrid_iir_gd_source"] == "phase"


def test_hybrid_iir_modal_search_uses_fixed_dense_grid_for_low_taps():
    fs = 48000
    native_freq = np.fft.rfftfreq(4096, d=1.0 / fs)
    native_freq = native_freq[(native_freq >= 20.0) & (native_freq <= 200.0)]
    source_freq = np.geomspace(20.0, 200.0, 4096)
    source_peak = _bump(source_freq, 52.0, 8.0, 0.035)
    native_peak = np.interp(native_freq, source_freq, source_peak)
    cfg = FilterConfig(
        fs=fs,
        num_taps=4096,
        hybrid_iir_enabled=True,
        hybrid_iir_min_peak_db=4.0,
        hybrid_iir_min_gd_excess_ms=0.0,
        hybrid_iir_min_confidence=0.5,
    )

    adjusted, stats = filter_pipeline._apply_hybrid_iir_preconditioning(
        cfg=cfg,
        freq_axis=native_freq,
        gain_db=np.zeros_like(native_freq),
        m_anal=native_peak,
        calc_offset_db=0.0,
        target_mags=np.zeros_like(native_freq),
        conf_mask=np.full_like(native_freq, 0.9),
        f_in=source_freq,
        m_smooth_std=source_peak,
        st={},
    )

    assert stats["hybrid_iir_modal_analysis_taps"] == 262144
    assert stats["hybrid_iir_modal_analysis_axis_source"] == "fixed_262144"
    assert stats["hybrid_iir_modal_analysis_bin_count"] > native_freq.size
    assert stats["hybrid_iir_modal_event_count"] >= 1
    assert stats["hybrid_iir_filter_count"] >= 1
    assert stats["hybrid_iir_biquads"][0]["freq"] == pytest.approx(52.0, rel=0.08)
    assert len(stats["hybrid_iir_mag_db"]) == native_freq.size
    assert adjusted.shape == native_freq.shape


def test_hybrid_iir_dense_phase_gd_allows_default_threshold_for_low_taps():
    fs = 48000
    native_freq = np.fft.rfftfreq(4096, d=1.0 / fs)
    native_freq = native_freq[(native_freq >= 20.0) & (native_freq <= 200.0)]
    source_freq = np.geomspace(20.0, 200.0, 4096)
    source_peak = _bump(source_freq, 52.0, 8.0, 0.035)
    gd_excess_ms = 38.0 * np.exp(-0.5 * (np.log2(source_freq / 52.0) / 0.04) ** 2)
    phase_rad = -np.cumsum(gd_excess_ms / 1000.0 * np.gradient(2.0 * np.pi * source_freq))
    native_peak = np.interp(native_freq, source_freq, source_peak)
    cfg = FilterConfig(
        fs=fs,
        num_taps=4096,
        hybrid_iir_enabled=True,
        hybrid_iir_min_peak_db=4.0,
        hybrid_iir_min_gd_excess_ms=12.0,
        hybrid_iir_min_confidence=0.5,
    )

    adjusted, stats = filter_pipeline._apply_hybrid_iir_preconditioning(
        cfg=cfg,
        freq_axis=native_freq,
        gain_db=np.zeros_like(native_freq),
        m_anal=native_peak,
        calc_offset_db=0.0,
        target_mags=np.zeros_like(native_freq),
        conf_mask=np.full_like(native_freq, 0.9),
        f_in=source_freq,
        m_smooth_std=source_peak,
        p_smooth=np.rad2deg(phase_rad),
        st={},
    )

    assert stats["hybrid_iir_gd_source"] == "preprocessed_phase"
    assert stats["hybrid_iir_modal_gd_source"] == "preprocessed_phase"
    assert stats["hybrid_iir_min_gd_excess_ms_effective"] == pytest.approx(12.0)
    assert stats["hybrid_iir_filter_count"] >= 1
    assert stats["hybrid_iir_biquads"][0]["freq"] == pytest.approx(52.0, rel=0.08)
    assert stats["hybrid_iir_biquads"][0]["gain"] < 0.0
    assert len(stats["hybrid_iir_mag_db"]) == native_freq.size
    assert adjusted.shape == native_freq.shape


def test_hybrid_iir_real_detector_selects_synthetic_modal_peak():
    freq = np.geomspace(20.0, 200.0, 768)
    peak = 8.0 * np.exp(-0.5 * (np.log2(freq / 52.0) / 0.08) ** 2)
    gd_excess_ms = 38.0 * np.exp(-0.5 * (np.log2(freq / 52.0) / 0.09) ** 2)
    phase = -np.cumsum(gd_excess_ms / 1000.0 * np.gradient(2.0 * np.pi * freq))
    cfg = FilterConfig(
        fs=48000,
        hybrid_iir_enabled=True,
        hybrid_iir_min_peak_db=4.0,
        hybrid_iir_min_gd_excess_ms=12.0,
        hybrid_iir_min_confidence=0.5,
    )

    adjusted, stats = filter_pipeline._apply_hybrid_iir_preconditioning(
        cfg=cfg,
        freq_axis=freq,
        gain_db=np.zeros_like(freq),
        m_anal=peak,
        calc_offset_db=0.0,
        target_mags=np.zeros_like(freq),
        conf_mask=np.full_like(freq, 0.9),
        phase_rad=phase,
        st={},
    )

    assert stats["hybrid_iir_modal_event_count"] >= 1
    assert stats["hybrid_iir_filter_count"] >= 1
    assert stats["hybrid_iir_biquads"][0]["freq"] == pytest.approx(52.0, rel=0.08)
    assert float(stats["hybrid_iir_biquads"][0]["gain"]) < 0.0
    assert np.any(np.abs(adjusted) > 0.1)


def test_hybrid_iir_signature_changes_when_enabled():
    off = _hybrid_signature_payload(hybrid_iir_enabled=False)
    on = _hybrid_signature_payload(hybrid_iir_enabled=True)

    assert off["hybrid_iir"]["enabled"] is False
    assert on["hybrid_iir"]["enabled"] is True
    assert off != on


def test_hybrid_iir_signature_tracks_thresholds():
    a = _hybrid_signature_payload(hybrid_iir_enabled=True, hybrid_iir_max_cut_db=6.0)
    b = _hybrid_signature_payload(hybrid_iir_enabled=True, hybrid_iir_max_cut_db=3.0)

    assert a["hybrid_iir"]["max_cut_db"] == 6.0
    assert b["hybrid_iir"]["max_cut_db"] == 3.0
    assert a != b


def test_hybrid_iir_signature_tracks_min_cut_priority():
    a = _hybrid_signature_payload(hybrid_iir_enabled=True, hybrid_iir_min_cut_priority=0.0)
    b = _hybrid_signature_payload(hybrid_iir_enabled=True, hybrid_iir_min_cut_priority=0.35)

    assert a["hybrid_iir"]["min_cut_priority"] == pytest.approx(0.0)
    assert b["hybrid_iir"]["min_cut_priority"] == pytest.approx(0.35)
    assert a != b


def test_detect_room_modes_identifies_clear_modal_peak():
    freq_axis = np.geomspace(20.0, 300.0, 360)
    peak = _bump(freq_axis, 113.0, 6.0, 0.09)
    gd = 35.0 * np.exp(-0.5 * (np.log2(freq_axis / 113.0) / 0.10) ** 2)

    result = detect_room_modes(
        freq_axis,
        peak,
        target_mag_db=np.zeros_like(freq_axis),
        group_delay_ms=gd,
        confidence_mask=np.full_like(freq_axis, 0.9),
    )

    assert result.mode_count >= 1
    assert result.worst_mode_hz == pytest.approx(113.0, rel=0.05)
    assert result.events[0].kind == "room_mode"
    assert result.events[0].correction_priority > 0.5
    assert result.events[0].safe_cut_db > 1.0


def test_detect_room_modes_uses_decay_evidence_when_local_confidence_is_low():
    freq_axis = np.geomspace(20.0, 240.0, 512)
    peak = _bump(freq_axis, 181.0, 6.0, 0.08)
    gd = 42.0 * np.exp(-0.5 * (np.log2(freq_axis / 181.0) / 0.09) ** 2)

    result = detect_room_modes(
        freq_axis,
        peak,
        target_mag_db=np.zeros_like(freq_axis),
        group_delay_ms=gd,
        confidence_mask=np.full_like(freq_axis, 0.03),
        hi_hz=200.0,
    )

    assert result.analysis_version >= 2
    assert result.mode_count >= 1
    assert result.events[0].freq_hz == pytest.approx(181.0, rel=0.05)
    assert result.events[0].kind == "room_mode"
    assert result.events[0].confidence >= 0.35
    assert result.events[0].cut_priority > 0.0


def test_detect_room_modes_low_confidence_without_decay_support_stays_uncertain():
    freq_axis = np.geomspace(20.0, 240.0, 512)
    peak = _bump(freq_axis, 90.0, 6.0, 0.08)

    result = detect_room_modes(
        freq_axis,
        peak,
        target_mag_db=np.zeros_like(freq_axis),
        group_delay_ms=np.zeros_like(freq_axis),
        confidence_mask=np.full_like(freq_axis, 0.03),
        hi_hz=200.0,
    )

    assert result.mode_count >= 1
    assert result.events[0].kind != "room_mode"


def test_detect_room_modes_does_not_promote_narrow_comb_spike():
    freq_axis = np.geomspace(20.0, 300.0, 420)
    spike = _bump(freq_axis, 110.0, 4.0, 0.009)

    result = detect_room_modes(
        freq_axis,
        spike,
        target_mag_db=np.zeros_like(freq_axis),
        confidence_mask=np.full_like(freq_axis, 0.35),
        min_width_oct=1.0 / 96.0,
        detect_smooth_oct=96.0,
    )

    assert result.mode_count >= 1
    assert result.events[0].kind in {"local_comb", "uncertain"}
    assert result.events[0].correction_priority < 0.4


def test_detect_room_modes_classifies_broad_buildup():
    freq_axis = np.geomspace(20.0, 300.0, 420)
    buildup = _bump(freq_axis, 62.0, 4.0, 0.27)

    result = detect_room_modes(
        freq_axis,
        buildup,
        target_mag_db=np.zeros_like(freq_axis),
        confidence_mask=np.full_like(freq_axis, 0.9),
    )

    assert result.mode_count >= 1
    assert result.events[0].kind == "broad_buildup"
    assert result.events[0].area_db_oct > 0.0


def test_detect_room_modes_voice_band_risk_weights_110hz_over_35hz():
    freq_axis = np.geomspace(20.0, 300.0, 360)
    conf = np.full_like(freq_axis, 0.9)
    gd_110 = 28.0 * np.exp(-0.5 * (np.log2(freq_axis / 110.0) / 0.10) ** 2)
    gd_35 = 28.0 * np.exp(-0.5 * (np.log2(freq_axis / 35.0) / 0.10) ** 2)

    result_110 = detect_room_modes(
        freq_axis,
        _bump(freq_axis, 110.0, 5.0, 0.09),
        target_mag_db=np.zeros_like(freq_axis),
        group_delay_ms=gd_110,
        confidence_mask=conf,
    )
    result_35 = detect_room_modes(
        freq_axis,
        _bump(freq_axis, 35.0, 5.0, 0.09),
        target_mag_db=np.zeros_like(freq_axis),
        group_delay_ms=gd_35,
        confidence_mask=conf,
    )

    assert result_110.voice_band_modal_risk > result_35.voice_band_modal_risk


def test_detect_room_modes_bad_inputs_return_empty_result():
    result = detect_room_modes(
        [20.0, float("nan"), float("inf")],
        [1.0, 2.0],
        target_mag_db=[0.0],
        corrected_mag_db=[0.0, float("nan")],
    )

    assert result.mode_count == 0
    assert result.events == ()


def test_modal_severity_increases_with_supporting_evidence():
    freq_axis = np.geomspace(20.0, 300.0, 420)
    measured = _bump(freq_axis, 113.0, 6.0, 0.09)
    target = np.zeros_like(freq_axis)
    gd = 35.0 * np.exp(-0.5 * (np.log2(freq_axis / 113.0) / 0.10) ** 2)

    mag_only = detect_room_modes(
        freq_axis,
        measured,
        target_mag_db=target,
        confidence_mask=np.full_like(freq_axis, 0.9),
    )
    mag_gd = detect_room_modes(
        freq_axis,
        measured,
        target_mag_db=target,
        group_delay_ms=gd,
        confidence_mask=np.full_like(freq_axis, 0.9),
    )
    mag_gd_decay = detect_room_modes(
        freq_axis,
        measured,
        target_mag_db=target,
        group_delay_ms=gd,
        confidence_mask=np.full_like(freq_axis, 0.9),
        rt60_by_band={80.0: 0.55, 113.0: 0.95, 160.0: 0.50},
    )

    assert mag_only.events
    assert mag_gd.events
    assert mag_gd_decay.events
    assert mag_only.events[0].severity < mag_gd.events[0].severity
    assert mag_gd.events[0].severity < mag_gd_decay.events[0].severity


def test_compute_modal_intelligence_metrics_is_json_friendly():
    freq_axis = np.geomspace(20.0, 300.0, 360)
    measured = _bump(freq_axis, 113.0, 6.0, 0.09)
    metrics = compute_modal_intelligence_metrics(
        {
            "freq_axis": freq_axis,
            "measured_mags": measured,
            "target_mags": np.zeros_like(freq_axis),
            "realized_filter_mags": np.zeros_like(freq_axis),
            "group_delay_ms": 35.0 * np.exp(-0.5 * (np.log2(freq_axis / 113.0) / 0.10) ** 2),
            "confidence_mask": np.full_like(freq_axis, 0.9),
        },
        lo_hz=20.0,
        hi_hz=250.0,
    )

    assert metrics["modal_mode_count"] >= 1
    assert metrics["modal_events"][0]["kind"] == "room_mode"
    _assert_jsonish(metrics)


def test_residual_peak_penalty_increases_with_modal_support():
    freq_axis = np.geomspace(20.0, 300.0, 420)
    measured = _bump(freq_axis, 113.0, 6.0, 0.09)
    target = np.zeros_like(freq_axis)
    realized = np.zeros_like(freq_axis)
    base_st = {
        "freq_axis": freq_axis,
        "measured_mags": measured,
        "target_mags": target,
        "realized_filter_mags": realized,
        "confidence_mask": np.full_like(freq_axis, 0.9),
    }
    modal_st = {
        **base_st,
        "group_delay_ms": 35.0 * np.exp(-0.5 * (np.log2(freq_axis / 113.0) / 0.10) ** 2),
    }
    base_data = {
        "mag_c_min": 20.0,
        "mag_c_max": 250.0,
        "auto_mode_residual_peak_threshold_db": 2.0,
        "auto_mode_residual_peak_hard_gate_db": 8.0,
    }

    no_modal = score_residual_peaks(base_st, base_st, base_data=base_data)
    with_modal = score_residual_peaks(modal_st, modal_st, base_data=base_data)

    assert with_modal["residual_peak_modal_priority"] > no_modal["residual_peak_modal_priority"]
    assert with_modal["residual_peak_modal_penalty"] > 0.0
    assert with_modal["residual_peak_penalty"] > no_modal["residual_peak_penalty"]
    assert with_modal["residual_peak_hard_gate_db"] == no_modal["residual_peak_hard_gate_db"]


def test_modal_residual_promotion_penalizes_broad_bass_buildup_before_fallback():
    freq_axis = np.geomspace(20.0, 300.0, 420)
    measured = _bump(freq_axis, 90.0, 5.5, 0.25)
    st = {
        "freq_axis": freq_axis,
        "measured_mags": measured,
        "target_mags": np.zeros_like(freq_axis),
        "realized_filter_mags": np.zeros_like(freq_axis),
        "confidence_mask": np.ones_like(freq_axis),
    }

    scored = score_residual_peaks(
        st,
        st,
        base_data={
            "mag_c_min": 20.0,
            "mag_c_max": 250.0,
            "auto_mode_residual_peak_threshold_db": 3.0,
            "auto_mode_residual_peak_hard_gate_db": 6.0,
        },
    )

    assert int(scored["residual_peak_metrics"]["residual_peak_count"]) >= 1
    assert int(scored["residual_peak_metrics"]["residual_peak_modal_promoted_count"]) >= 1
    assert bool(scored["modal_residual_fallback_used"]) is False
    assert float(scored["residual_peak_metrics"]["worst_residual_peak_hz"]) == pytest.approx(90.0, rel=0.08)
    assert float(scored["worst_residual_peak_raw_db"]) > 3.0
    assert float(scored["residual_peak_penalty"]) > 0.0


def test_modal_residual_fallback_lowers_rank_against_corrected_candidate(monkeypatch):
    monkeypatch.setattr(
        "decaycore.auto_mode.scoring_metrics.calc_ai_summary_from_stats",
        lambda _st, **_kwargs: {"score": 90.0},
    )
    freq_axis = np.geomspace(20.0, 300.0, 420)
    broad = _bump(freq_axis, 90.0, 5.5, 0.25)
    bad = {
        "freq_axis": freq_axis,
        "measured_mags": broad,
        "target_mags": np.zeros_like(freq_axis),
        "realized_filter_mags": np.zeros_like(freq_axis),
        "confidence_mask": np.ones_like(freq_axis),
    }
    good = dict(bad)
    good["realized_filter_mags"] = -0.85 * broad

    base_data = {
        "mag_c_min": 20.0,
        "mag_c_max": 250.0,
        "auto_mode_residual_peak_threshold_db": 3.0,
        "auto_mode_residual_peak_hard_gate_db": 6.0,
    }
    bad_metrics = _auto_score_result(SimpleNamespace(l_st=bad, r_st=bad, metrics={}), base_data=base_data)
    good_metrics = _auto_score_result(SimpleNamespace(l_st=good, r_st=good, metrics={}), base_data=base_data)

    assert bool(bad_metrics["modal_residual_fallback_used"]) is False
    assert int(bad_metrics["residual_peak_modal_promoted_count"]) >= 1
    assert float(bad_metrics["worst_residual_peak_raw_db"]) > 3.0
    assert float(good_metrics["modal_residual_fallback_penalty"]) == pytest.approx(0.0, abs=1e-9)
    assert float(bad_metrics["rank_score_components"]["residual_peak_penalty"]) > 0.0
    assert float(good_metrics["rank_score"]) > float(bad_metrics["rank_score"])


def test_modal_residual_fallback_ignores_low_confidence_and_out_of_band_events():
    freq_axis = np.geomspace(20.0, 300.0, 420)
    broad = _bump(freq_axis, 90.0, 5.5, 0.25)
    base_data = {
        "mag_c_min": 20.0,
        "mag_c_max": 250.0,
        "auto_mode_residual_peak_threshold_db": 3.0,
        "auto_mode_residual_peak_hard_gate_db": 6.0,
    }
    low_conf = {
        "freq_axis": freq_axis,
        "measured_mags": broad,
        "target_mags": np.zeros_like(freq_axis),
        "realized_filter_mags": np.zeros_like(freq_axis),
        "confidence_mask": np.full_like(freq_axis, 0.2),
    }
    out_of_band = dict(low_conf)
    out_of_band["confidence_mask"] = np.ones_like(freq_axis)

    low_conf_metrics = _auto_score_result(SimpleNamespace(l_st=low_conf, r_st=low_conf, metrics={}), base_data=base_data)
    out_of_band_metrics = _auto_score_result(
        SimpleNamespace(l_st=out_of_band, r_st=out_of_band, metrics={}),
        base_data={**base_data, "mag_c_max": 70.0},
    )

    assert bool(low_conf_metrics["modal_residual_fallback_used"]) is False
    assert "residual_peak_hard_gate" not in low_conf_metrics["hard_gate_failures"]
    assert "residual_peak_severity_gate" not in low_conf_metrics["hard_gate_failures"]
    assert bool(out_of_band_metrics["modal_residual_fallback_used"]) is False
    assert "residual_peak_hard_gate" not in out_of_band_metrics["hard_gate_failures"]
    assert "residual_peak_severity_gate" not in out_of_band_metrics["hard_gate_failures"]


def test_residual_peak_winner_polish_modal_priority_limits_variants():
    tested: list[dict] = []

    def materialize(preset, **_kwargs):
        tested.append(dict(preset))
        return None, {
            "rank_score": 9.0,
            "avg_score": 90.0,
            "worst_residual_peak_raw_db": 5.0,
            "max_net_boost_db": 0.0,
            "target_tracking_rms_20_200_db": 0.0,
            "target_tracking_rms_100_500_db": 0.0,
            "bass_under_target_penalty": 0.0,
            "correction_sharpness_penalty": 0.0,
            "dip_fill_risk_penalty": 0.0,
            "channel_overfit_penalty": 0.0,
        }, None

    best_metrics = {
        "rank_score": 10.0,
        "avg_score": 90.0,
        "worst_residual_peak_raw_db": 5.2,
        "worst_residual_peak_hz": 113.0,
        "worst_residual_peak_width_oct": 0.12,
        "residual_peak_threshold_db": 3.0,
        "residual_peak_modal_priority": 0.8,
        "residual_peak_modal_support": 0.85,
        "residual_peak_modal_max_severity": 0.9,
        "residual_peak_modal_confidence": 0.8,
        "residual_peak_modal_dominant_freq_hz": 113.0,
        "max_net_boost_db": 0.0,
    }
    _preset, _metrics, improved, meta = apply_residual_peak_winner_polish(
        best_preset={"tdc_strength": 72.0, "tdc_max_reduction_db": 17.0},
        best_metrics=best_metrics,
        base_data_ref={},
        phase_label="test",
        goal="balanced",
        enabled=True,
        max_variants=8,
        min_improvement_db=0.75,
        status_cb=lambda *_args, **_kwargs: None,
        materialize_preset_result=materialize,
        cache_ready_preset=lambda preset, best_metrics=None: dict(preset),
    )

    assert improved is False
    assert meta["modal_priority"] == pytest.approx(0.8)
    assert meta["min_peak_improvement_effective_db"] < meta["min_peak_improvement_base_db"]
    tdc_variants = [item for item in tested if "tdc_strength" in item or "tdc_max_reduction_db" in item]
    assert tdc_variants
    assert all(float(item.get("tdc_strength", 0.0)) <= 75.0 for item in tdc_variants)
    assert all(float(item.get("tdc_max_reduction_db", 0.0)) <= 18.0 for item in tdc_variants)


def test_auto_mode_modal_debug_structure_marks_used_by():
    metrics = {
        "modal_mode_count": 1,
        "modal_events": [
            {
                "freq_hz": 113.0,
                "severity": 0.82,
                "confidence": 0.76,
                "peak_db": 7.1,
                "gd_excess_ms": 38.5,
                "safe_cut_db": 4.5,
                "kind": "room_mode",
            }
        ],
        "residual_peak_modal_support": 0.8,
        "residual_peak_modal_priority": 0.7,
        "residual_peak_modal_penalty": 0.5,
        "residual_peak_modal_dominant_freq_hz": 113.0,
        "tdc_modal_event_count": 1,
        "tdc_modal_reductions": [
            {
                "freq_hz": 113.0,
                "applied_reduction_db": 4.3,
                "modal_severity": 0.82,
                "confidence": 0.76,
            }
        ],
    }
    debug = _build_modal_intelligence_debug(
        metrics,
        {
            "residual_peak_winner_polish": {
                "applicable": True,
                "modal_priority": 0.7,
                "modal_support": 0.8,
                "modal_dominant_freq_hz": 113.0,
                "applied": False,
            }
        },
    )

    assert debug["enabled"] is True
    assert debug["event_count"] == 1
    assert debug["strongest_events"][0]["used_by"] == ["residual_peak", "tdc", "winner_polish"]
    _assert_jsonish(debug)
