# DecayCore
# Copyright (c) 2026 Vilho Valittu
# All rights reserved.
#
# This file is part of the proprietary DecayCore codebase.
# No copying, redistribution, commercial reuse, or removal of attribution
# is permitted without prior written permission.

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from decaycore.auto_mode.orchestrator_finalize_result import _run_p6_final_validation
from decaycore.dsp.final_ir_validation import (
    FinalIRValidationResult,
    final_ir_validation_to_stats,
    validate_final_fir_against_ir,
)

FS = 48000
FREQ = np.geomspace(10.0, 20000.0, 2048)


class _Cfg:
    fs = FS
    final_ir_validation_enable = True
    final_ir_validation_mode = "warn"
    final_ir_validation_score_weight = 1.0
    final_ir_validation_candidate_count = 3
    final_ir_validation_reject_pre_energy_db = -18.0
    final_ir_validation_warn_pre_energy_db = -24.0
    final_ir_validation_reject_gd_peak_ms = 80.0
    final_ir_validation_warn_gd_peak_ms = 45.0
    final_ir_validation_reject_voice_peak_db = 5.0
    final_ir_validation_warn_voice_peak_db = 3.0
    final_ir_validation_reject_stereo_delta_db = 5.0
    final_ir_validation_warn_stereo_delta_db = 3.0
    final_ir_validation_reject_bass_residual_peak_db = 7.0
    final_ir_validation_warn_bass_residual_peak_db = 4.0
    final_ir_validation_pre_window_ms = 25.0
    final_ir_validation_post_window_ms = 250.0
    final_ir_validation_early_window_ms = 20.0
    final_ir_validation_mag_lo_hz = 20.0
    final_ir_validation_mag_hi_hz = 300.0
    final_ir_validation_voice_lo_hz = 70.0
    final_ir_validation_voice_hi_hz = 180.0


def _dirac(n: int, peak_i: int = 0) -> np.ndarray:
    x = np.zeros(n)
    x[peak_i] = 1.0
    return x


def test_missing_inputs_returns_safe_result():
    result = validate_final_fir_against_ir(sample_rate=48000)
    assert result.valid is True
    assert result.severity == "ok"
    assert "missing_final_ir_validation_inputs" in result.reasons
    assert result.score_penalty == 0.0


def test_missing_sample_rate_returns_safe_result():
    fir = _dirac(4096, 0)
    result = validate_final_fir_against_ir(sample_rate=0, fir_l=fir)
    assert result.valid is True
    assert result.severity == "ok"
    assert "missing_final_ir_validation_inputs" in result.reasons


def test_identity_fir_passes():
    fir = _dirac(4096, 0)
    result = validate_final_fir_against_ir(
        sample_rate=FS,
        fir_l=fir,
        config=_Cfg(),
    )
    assert result.severity == "ok"
    assert result.score_penalty < 0.1
    # Pure dirac at tap 0: no pre-ringing samples exist
    assert np.isfinite(result.pre_energy_ratio_db)


def test_pre_ringing_fir_warns_or_rejects():
    # FIR with significant energy before main tap
    fir = np.zeros(4096)
    fir[0] = 0.4     # pre-ringing energy
    fir[64] = 1.0    # main tap (peak)
    result = validate_final_fir_against_ir(
        sample_rate=FS,
        fir_l=fir,
        config=_Cfg(),
    )
    # pre_energy_ratio_db should be elevated compared to identity FIR
    identity_result = validate_final_fir_against_ir(
        sample_rate=FS,
        fir_l=_dirac(4096, 64),
        config=_Cfg(),
    )
    assert result.pre_energy_ratio_db > identity_result.pre_energy_ratio_db
    assert result.severity in ("warn", "reject")
    assert any("pre_energy" in r for r in result.reasons)


def test_voice_band_excess_detected():
    # measured room response is flat, filter adds +6 dB in voice band
    # corrected = measured + predicted = 0 + 6 = 6 dB residual vs flat target
    freq = FREQ
    measured = np.zeros_like(freq)
    gain = np.zeros_like(freq)
    voice_mask = (freq >= 70.0) & (freq <= 180.0)
    gain[voice_mask] = 6.0  # +6 dB filter gain in voice band

    target = np.zeros_like(freq)  # flat target

    result = validate_final_fir_against_ir(
        sample_rate=FS,
        fir_l=_dirac(4096, 0),
        freq_axis=freq,
        measured_mag_db_l=measured,
        predicted_mag_db_l=gain,
        target_mag_db=target,
        config=_Cfg(),
    )
    assert result.voice_band_peak_excess_db > 4.0
    assert result.severity in ("warn", "reject")


def test_stereo_mismatch_detected():
    # L = flat, R = +6 dB boost at 120 Hz
    freq = FREQ
    gain_l = np.zeros_like(freq)
    gain_r = np.zeros_like(freq)
    voice_mask = (freq >= 70.0) & (freq <= 180.0)
    gain_r[voice_mask] = 6.0

    result = validate_final_fir_against_ir(
        sample_rate=FS,
        fir_l=_dirac(4096, 0),
        fir_r=_dirac(4096, 0),
        freq_axis=freq,
        predicted_mag_db_l=gain_l,
        predicted_mag_db_r=gain_r,
        config=_Cfg(),
    )
    assert result.stereo_delta_peak_db > 4.0
    assert result.severity in ("warn", "reject")


def test_stereo_delta_uses_channel_target_residuals_when_available():
    freq = FREQ
    gain_l = np.zeros_like(freq)
    gain_r = np.zeros_like(freq)
    measured_l = np.zeros_like(freq)
    measured_r = np.full_like(freq, -24.0)
    target_l = np.zeros_like(freq)
    target_r = np.full_like(freq, -24.0)

    result = validate_final_fir_against_ir(
        sample_rate=FS,
        fir_l=_dirac(4096, 0),
        fir_r=_dirac(4096, 0),
        freq_axis=freq,
        target_mag_db=target_l,
        target_mag_db_r=target_r,
        predicted_mag_db_l=gain_l,
        predicted_mag_db_r=gain_r,
        measured_mag_db_l=measured_l,
        measured_mag_db_r=measured_r,
        config=_Cfg(),
    )

    assert result.stereo_delta_peak_db == 0.0


def test_default_stereo_delta_reject_threshold_allows_moderate_warn():
    class _DefaultStereoCfg:
        final_ir_validation_warn_stereo_delta_db = 3.0

    freq = FREQ
    gain_l = np.zeros_like(freq)
    gain_r = np.zeros_like(freq)
    gain_r[(freq >= 80.0) & (freq <= 250.0)] = 5.9

    result = validate_final_fir_against_ir(
        sample_rate=FS,
        fir_l=_dirac(4096, 0),
        fir_r=_dirac(4096, 0),
        freq_axis=freq,
        predicted_mag_db_l=gain_l,
        predicted_mag_db_r=gain_r,
        config=_DefaultStereoCfg(),
    )

    assert result.valid is True
    assert result.severity == "warn"
    assert "stereo_delta_warn" in result.reasons
    assert "stereo_delta_reject" not in result.reasons


def test_stats_conversion_stable():
    result = validate_final_fir_against_ir(
        sample_rate=FS,
        fir_l=_dirac(4096, 0),
        config=_Cfg(),
    )
    stats = final_ir_validation_to_stats(result)

    required_keys = [
        "final_ir_validation_valid",
        "final_ir_validation_severity",
        "final_ir_validation_score_penalty",
        "final_ir_validation_mag_rms_db",
        "final_ir_validation_mag_peak_db",
        "final_ir_validation_pre_energy_ratio_db",
        "final_ir_validation_post_energy_ratio_db",
        "final_ir_validation_early_energy_ratio_db",
        "final_ir_validation_gd_peak_ms",
        "final_ir_validation_voice_band_peak_excess_db",
        "final_ir_validation_stereo_delta_rms_db",
        "final_ir_validation_bass_residual_peak_db",
        "final_ir_validation_reasons",
    ]
    for key in required_keys:
        assert key in stats, f"Missing stats key: {key}"

    for key, val in stats.items():
        if isinstance(val, float):
            assert np.isfinite(val) or np.isnan(val), f"Non-finite non-NaN float for {key}: {val}"


def test_linear_phase_skips_pre_ringing():
    # Linear-phase FIR: peak at center → pre-ringing check should be skipped (NaN)
    n = 4096
    fir = _dirac(n, n // 2)  # peak exactly at center
    result = validate_final_fir_against_ir(
        sample_rate=FS,
        fir_l=fir,
        ir_anchor_mode="peak",
        config=_Cfg(),
    )
    assert not np.isfinite(result.pre_energy_ratio_db), (
        f"Expected NaN pre_energy_ratio_db for linear-phase FIR, got {result.pre_energy_ratio_db}"
    )
    # Linear-phase GD should still be computed (not minimum-phase)
    assert np.isfinite(result.gd_peak_ms)


def test_minimum_phase_skips_pre_ringing_and_gd():
    # Minimum-phase FIR: ir_anchor_mode="min_causal" → skip both pre-ringing and GD
    fir = _dirac(4096, 0)
    result = validate_final_fir_against_ir(
        sample_rate=FS,
        fir_l=fir,
        ir_anchor_mode="min_causal",
        config=_Cfg(),
    )
    assert not np.isfinite(result.pre_energy_ratio_db), (
        f"Expected NaN pre_energy_ratio_db for min-phase FIR, got {result.pre_energy_ratio_db}"
    )
    assert not np.isfinite(result.gd_peak_ms), (
        f"Expected NaN gd_peak_ms for min-phase FIR, got {result.gd_peak_ms}"
    )
    # Should not trigger pre-energy reasons
    assert not any("pre_energy" in r for r in result.reasons)


def test_measured_ir_convolution_used_when_provided():
    # With a measured IR that has a strong early reflection, the pre-energy should change
    fir = _dirac(1024, 0)
    # Measured IR: dirac at 0 with echo at 500 samples
    measured_ir = _dirac(2048, 0)
    measured_ir[500] = 0.5  # reflection

    result_with_ir = validate_final_fir_against_ir(
        sample_rate=FS,
        fir_l=fir,
        measured_ir_l=measured_ir,
        config=_Cfg(),
    )
    result_without_ir = validate_final_fir_against_ir(
        sample_rate=FS,
        fir_l=fir,
        config=_Cfg(),
    )
    # Results should differ when measured IR is provided
    assert result_with_ir.post_energy_ratio_db != result_without_ir.post_energy_ratio_db


def _p6_finalize_stats(
    *,
    predicted: np.ndarray,
    realized: np.ndarray | None = None,
    filter_mags: np.ndarray | None = None,
    comparison: bool = False,
) -> dict:
    st = {
        "freq_axis": FREQ.tolist(),
        "target_mags": np.zeros_like(FREQ).tolist(),
        "measured_mags": np.zeros_like(FREQ).tolist(),
        "predicted_filter_mags": np.asarray(predicted, dtype=float).tolist(),
        "ir_anchor_mode": "min_causal",
    }
    if realized is not None:
        st["realized_filter_mags"] = np.asarray(realized, dtype=float).tolist()
    if filter_mags is not None:
        st["filter_mags"] = np.asarray(filter_mags, dtype=float).tolist()
        st["filter_mags_source"] = "ir_fft_final"
    if comparison:
        st.update(
            {
                "analysis_mode": "comparison",
                "cmp_freq_axis": FREQ.tolist(),
                "cmp_target_mags": np.zeros_like(FREQ).tolist(),
                "cmp_measured_mags": np.zeros_like(FREQ).tolist(),
                "cmp_predicted_filter_mags": np.asarray(predicted, dtype=float).tolist(),
            }
        )
        if realized is not None:
            st["cmp_realized_filter_mags"] = np.asarray(realized, dtype=float).tolist()
        if filter_mags is not None:
            st["cmp_filter_mags"] = np.asarray(filter_mags, dtype=float).tolist()
            st["cmp_filter_mags_source"] = "ir_fft_final"
    return st


def _p6_finalize_state(l_st: dict, r_st: dict):
    result = SimpleNamespace(l_ir=_dirac(4096, 0), r_ir=_dirac(4096, 0), l_st=l_st, r_st=r_st)
    return SimpleNamespace(
        scored=[{"preset": {}, "metrics": {"avg_error": 0.0}}],
        best_result=result,
        best_metrics={},
        best_preset={},
    )


def _unused_p6_materializer(*args, **kwargs):
    raise AssertionError("best_result should be used for candidate #1")


def test_p6_finalize_prefers_realized_filter_mags_over_predicted_filter_mags():
    predicted_l = np.zeros_like(FREQ)
    predicted_r = np.zeros_like(FREQ)
    voice = (FREQ >= 70.0) & (FREQ <= 180.0)
    predicted_l[voice] = 6.0
    predicted_r[voice] = -18.0
    realized = np.zeros_like(FREQ)

    search_state = _p6_finalize_state(
        _p6_finalize_stats(predicted=predicted_l, realized=realized),
        _p6_finalize_stats(predicted=predicted_r, realized=realized),
    )

    _run_p6_final_validation(search_state, _Cfg(), _materialize_preset_result=_unused_p6_materializer)

    assert search_state.best_metrics["final_ir_validation_filter_mag_source_l"] == "realized_filter_mags"
    assert search_state.best_metrics["final_ir_validation_filter_mag_source_r"] == "realized_filter_mags"
    assert search_state.best_metrics["final_ir_validation_voice_band_peak_excess_db"] == 0.0
    assert search_state.best_metrics["final_ir_validation_stereo_delta_peak_db"] == 0.0
    assert search_state.best_metrics["final_ir_validation_bass_residual_peak_db"] == 0.0


def test_p6_finalize_falls_back_to_ir_fft_filter_mags_before_predicted_filter_mags():
    predicted = np.zeros_like(FREQ)
    predicted[(FREQ >= 70.0) & (FREQ <= 180.0)] = 6.0
    final_filter = np.zeros_like(FREQ)

    search_state = _p6_finalize_state(
        _p6_finalize_stats(predicted=predicted, filter_mags=final_filter),
        _p6_finalize_stats(predicted=predicted, filter_mags=final_filter),
    )

    _run_p6_final_validation(search_state, _Cfg(), _materialize_preset_result=_unused_p6_materializer)

    assert search_state.best_metrics["final_ir_validation_filter_mag_source_l"] == "filter_mags:ir_fft_final"
    assert search_state.best_metrics["final_ir_validation_filter_mag_source_r"] == "filter_mags:ir_fft_final"
    assert search_state.best_metrics["final_ir_validation_voice_band_peak_excess_db"] == 0.0


def test_p6_finalize_uses_comparison_axes_when_active():
    predicted = np.zeros_like(FREQ)
    predicted[(FREQ >= 70.0) & (FREQ <= 180.0)] = 6.0
    realized = np.zeros_like(FREQ)
    l_st = _p6_finalize_stats(predicted=predicted, realized=realized, comparison=True)
    r_st = _p6_finalize_stats(predicted=predicted, realized=realized, comparison=True)
    l_st["measured_mags"] = np.zeros_like(FREQ).tolist()
    r_st["measured_mags"] = np.full_like(FREQ, -25.0).tolist()
    l_st["target_mags"] = np.zeros_like(FREQ).tolist()
    r_st["target_mags"] = np.full_like(FREQ, -25.0).tolist()

    search_state = _p6_finalize_state(l_st, r_st)

    _run_p6_final_validation(search_state, _Cfg(), _materialize_preset_result=_unused_p6_materializer)

    assert search_state.best_metrics["final_ir_validation_filter_mag_source_l"] == "cmp_realized_filter_mags"
    assert search_state.best_metrics["final_ir_validation_filter_mag_source_r"] == "cmp_realized_filter_mags"
    assert search_state.best_metrics["final_ir_validation_voice_band_peak_excess_db"] == 0.0
    assert search_state.best_metrics["final_ir_validation_stereo_delta_peak_db"] == 0.0
