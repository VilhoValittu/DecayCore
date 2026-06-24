# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

from __future__ import annotations

import numpy as np

from ... import shared
from ...auto_mode_profile import profiled_section
from ..component_scores import (
    score_acoustic_fit,
    score_bass_integration,
    score_phase_quality,
    score_residual_peaks,
    score_safety_limits,
    score_stereo_policy,
    score_temporal_decay,
)
from ..rank_combiner import combine_rank_score
from ..residual_peaks import _build_modal_tdc_debug
from ..score_result_finalize import finalize_score_result_metrics

BROAD_RESIDUAL_PEAK_SCORING_VERSION = 3
CORRECTION_SHARPNESS_SCORING_VERSION = 1
DIP_FILL_RISK_SCORING_VERSION = 1
CHANNEL_OVERFIT_SCORING_VERSION = 1
MODAL_INTELLIGENCE_METRICS_VERSION = 1
AUTO_TDC_DECAY_SCORING_VERSION = 2





def _auto_score_result(
    result,
    *,
    auto_exc_freq_hz: float | None = None,
    focus_lo_hz: float | None = None,
    focus_hi_hz: float | None = None,
    base_data: dict | None = None,
) -> dict:
    l_st = dict(getattr(result, "l_st", {}) or {})
    r_st = dict(getattr(result, "r_st", {}) or {})
    result_metrics = dict(getattr(result, "metrics", {}) or {})

    with profiled_section("auto_score.acoustic_fit"):
        af = score_acoustic_fit(result, l_st, r_st, base_data=base_data)
    with profiled_section("auto_score.safety_limits"):
        sl = score_safety_limits(
            l_st, r_st, auto_exc_freq_hz=auto_exc_freq_hz, base_data=base_data,
            net_boost_max=af["net_boost_max"], lr_delta=af["lr_delta"],
            l_refs=af["l_refs"], r_refs=af["r_refs"],
        )
    with profiled_section("auto_score.phase_quality"):
        pq = score_phase_quality(l_st, r_st, base_data=base_data)
    with profiled_section("auto_score.temporal_decay"):
        td = score_temporal_decay(l_st, r_st, base_data=base_data)
    with profiled_section("auto_score.residual_peaks"):
        rp = score_residual_peaks(l_st, r_st, base_data=base_data)
    with profiled_section("auto_score.bass_integration"):
        bi = score_bass_integration(
            result, l_st, r_st, base_data=base_data,
            net_boost_max=af["net_boost_max"],
            peak_lo=rp["peak_lo"], peak_hi=rp["peak_hi"],
            exc_penalty=sl["exc_penalty"],
        )
    with profiled_section("auto_score.stereo_policy"):
        sp = score_stereo_policy(l_st, r_st, base_data=base_data)
    with profiled_section("auto_score.combine_rank"):
        cr = combine_rank_score(
            l_st, r_st, result,
            focus_lo_hz=focus_lo_hz, focus_hi_hz=focus_hi_hz, base_data=base_data,
            avg_score=af["avg_score"],
            phase_benefit_bonus=pq["phase_benefit_bonus"],
            boost_pen=sl["boost_pen"],
            event_pen=sl["event_pen"],
            lr_pen=sl["lr_pen"],
            dsp_penalty=sl["dsp_penalty"],
            bass_prering_penalty=sl["bass_prering_penalty"],
            exc_penalty=sl["exc_penalty"],
            bass_integration_penalty=bi["bass_integration_penalty"],
            bass_feasibility_penalty=bi["bass_feasibility_penalty"],
            bass_preference_bonus=bi["bass_preference_bonus"],
            decay_penalty=td["decay_penalty"],
            residual_peak_penalty=rp["residual_peak_penalty"],
            correction_sharpness_penalty=bi["correction_sharpness_penalty"],
            dip_fill_risk_penalty=bi["dip_fill_risk_penalty"],
            channel_overfit_penalty=bi["channel_overfit_penalty"],
            target_tracking_penalty=bi["target_tracking_penalty"],
            voice_clarity_penalty=bi["voice_clarity_penalty"],
            phase_risk_penalty=pq["phase_risk_penalty"],
            phase_limit_penalty=pq["phase_limit_penalty"],
            thd_boost_penalty=td["thd_boost_penalty"],
            stereo_coherence_penalty=sp["stereo_coherence_penalty"],
            phantom_center_stability_penalty=sp["phantom_center_stability_penalty"],
            policy_divergence_penalty=sp["policy_divergence_penalty"],
            asymmetry_budget_overflow_penalty=sp["asymmetry_budget_overflow_penalty"],
            worst_channel_relief_bonus=sp["worst_channel_relief_bonus"],
            shared_preference_bias=sp["shared_preference_bias"],
            rt60_policy_pen=td["rt60_policy_pen"],
            harmonic_local_pen=td["harmonic_local_pen"],
        )

    avg_score = af["avg_score"]
    lr_delta = af["lr_delta"]
    net_boost_max = af["net_boost_max"]
    events_total = af["events_total"]
    events_severity_l = af["events_severity_l"]
    events_severity_r = af["events_severity_r"]
    events_severity_raw = af["events_severity_raw"]
    events_severity = af["events_severity"]
    dsp_pen_l = sl["dsp_pen_l"]
    dsp_pen_r = sl["dsp_pen_r"]
    dsp_dbg_l = sl["dsp_dbg_l"]
    dsp_dbg_r = sl["dsp_dbg_r"]
    dsp_penalty_raw = sl["dsp_penalty_raw"]
    dsp_penalty = sl["dsp_penalty"]
    exc_pen_l = sl["exc_pen_l"]
    exc_pen_r = sl["exc_pen_r"]
    exc_dbg_l = sl["exc_dbg_l"]
    exc_dbg_r = sl["exc_dbg_r"]
    exc_penalty_raw = sl["exc_penalty_raw"]
    exc_penalty_raw_total = sl["exc_penalty_raw_total"]
    exc_penalty_bins_raw = sl["exc_penalty_bins_raw"]
    exc_penalty_bins_waived = sl["exc_penalty_bins_waived"]
    exc_penalty_waived = sl["exc_penalty_waived"]
    exc_penalty = sl["exc_penalty"]
    auto_exc_zero_penalty_hz = sl["auto_exc_zero_penalty_hz"]
    boost_pen = sl["boost_pen"]
    event_pen = sl["event_pen"]
    lr_pen = sl["lr_pen"]
    phase_limit_used_hz = pq["phase_limit_used_hz"]
    phase_limit_penalty = pq["phase_limit_penalty"]
    phase_benefit_bonus = pq["phase_benefit_bonus"]
    phase_risk_penalty = pq["phase_risk_penalty"]
    phase_net_score = pq["phase_net_score"]
    phase_lr_consistency_penalty = pq["phase_lr_consistency_penalty"]
    phase_dbg_l = pq["phase_dbg_l"]
    phase_dbg_r = pq["phase_dbg_r"]
    thd_boost_penalty = td["thd_boost_penalty"]
    _rt60_policy_pen = td["rt60_policy_pen"]
    _harmonic_local_pen = td["harmonic_local_pen"]
    decay_penalty = td["decay_penalty"]
    decay_dbg = td["decay_dbg"]
    residual_peak_metrics = rp["residual_peak_metrics"]
    worst_residual_peak_db = rp["worst_residual_peak_db"]
    worst_residual_peak_raw_db = rp["worst_residual_peak_raw_db"]
    residual_peak_severity = rp["residual_peak_severity"]
    residual_peak_hard_gate_db = rp["residual_peak_hard_gate_db"]
    residual_peak_penalty_cap = rp["residual_peak_penalty_cap"]
    top3_residual_peak_mean_db = rp["top3_residual_peak_mean_db"]
    residual_peak_penalty = rp["residual_peak_penalty"]
    modal_residual_fallback_used = bool(rp.get("modal_residual_fallback_used", False))
    modal_residual_fallback_kind = str(rp.get("modal_residual_fallback_kind", "") or "")
    modal_residual_fallback_hz = shared._auto_safe_float(
        rp.get("modal_residual_fallback_hz", float("nan")),
        float("nan"),
    )
    modal_residual_fallback_peak_db = shared._auto_safe_float(
        rp.get("modal_residual_fallback_peak_db", float("nan")),
        float("nan"),
    )
    modal_residual_fallback_penalty = shared._auto_safe_float(
        rp.get("modal_residual_fallback_penalty", 0.0),
        0.0,
    )
    residual_peak_gate_value_db = shared._auto_safe_float(
        rp.get("residual_peak_gate_value_db", float("nan")),
        float("nan"),
    )
    residual_peak_gate_source = str(rp.get("residual_peak_gate_source", "") or "")
    modal_intelligence_metrics = rp["modal_intelligence_metrics"]
    tdc_modal_debug = _build_modal_tdc_debug(modal_intelligence_metrics, decay_dbg)
    bass_integration_penalty = bi["bass_integration_penalty"]
    bass_feasibility_penalty = bi["bass_feasibility_penalty"]
    bass_dbg = bi["bass_dbg"]
    target_tracking_penalty = bi["target_tracking_penalty"]
    bass_under_target_penalty = bi["bass_under_target_penalty"]
    bass_under_target_metrics = bi["bass_under_target_metrics"]
    bass_under_target_l = bi["bass_under_target_l"]
    bass_under_target_r = bi["bass_under_target_r"]
    target_tracking_metrics = bi["target_tracking_metrics"]
    target_tracking_l = bi["target_tracking_l"]
    target_tracking_r = bi["target_tracking_r"]
    bass_boost_metrics = bi["bass_boost_metrics"]
    bass_boost_l = bi["bass_boost_l"]
    bass_boost_r = bi["bass_boost_r"]
    bass_preference_bonus = bi["bass_preference_bonus"]
    sharpness_metrics = bi["sharpness_metrics"]
    correction_sharpness_penalty = bi["correction_sharpness_penalty"]
    dip_fill_metrics = bi["dip_fill_metrics"]
    dip_fill_risk_penalty = bi["dip_fill_risk_penalty"]
    channel_overfit_metrics = bi["channel_overfit_metrics"]
    channel_overfit_penalty = bi["channel_overfit_penalty"]
    voice_metrics = bi["voice_metrics"]
    voice_clarity_penalty = bi["voice_clarity_penalty"]
    stereo_policy_active = sp["stereo_policy_active"]
    stereo_policy_gate_failed = sp["stereo_policy_gate_failed"]
    stereo_split_hz = sp["stereo_split_hz"]
    stereo_lr_mismatch_below = sp["stereo_lr_mismatch_below"]
    stereo_lr_mismatch_above = sp["stereo_lr_mismatch_above"]
    stereo_lr_mismatch_below_delta = sp["stereo_lr_mismatch_below_delta"]
    stereo_lr_mismatch_above_delta = sp["stereo_lr_mismatch_above_delta"]
    phantom_center_change_db = sp["phantom_center_change_db"]
    phantom_center_change_delta_db = sp["phantom_center_change_delta_db"]
    policy_divergence_score = sp["policy_divergence_score"]
    stereo_coherence_penalty = sp["stereo_coherence_penalty"]
    phantom_center_stability_penalty = sp["phantom_center_stability_penalty"]
    policy_divergence_penalty = sp["policy_divergence_penalty"]
    asymmetry_budget_overflow_penalty = sp["asymmetry_budget_overflow_penalty"]
    worst_channel_relief_bonus = sp["worst_channel_relief_bonus"]
    shared_preference_bias = sp["shared_preference_bias"]
    stereo_worst_channel_relief_db = sp["stereo_worst_channel_relief_db"]
    rank_score = cr["rank_score"]
    rank_score_base = cr["rank_score_base"]
    rank_components = cr["rank_components"]
    focus_ripple = cr["focus_ripple"]
    mode_hz = cr["mode_hz"]
    mode_band_lo = cr["mode_band_lo"]
    mode_band_hi = cr["mode_band_hi"]
    mode_ripple_db = cr["mode_ripple_db"]
    mode2_hz = cr["mode2_hz"]
    mode2_band_lo = cr["mode2_band_lo"]
    mode2_band_hi = cr["mode2_band_hi"]
    mode2_ripple_db = cr["mode2_ripple_db"]
    mode_combined = cr["mode_combined"]
    mode_penalty = cr["mode_penalty"]
    realized_rms_20_200 = cr["realized_rms_20_200"]
    ripple_raw = cr["ripple_raw"]
    pre_post_l_f = cr["pre_post_l_f"]
    pre_post_r_f = cr["pre_post_r_f"]
    pre_post_max = cr["pre_post_max"]

    metrics_out = {
        "rank_score": float(rank_score),
        "rank_score_base": float(rank_score_base),
        "rank_score_official": float(rank_score),
        "rank_score_components": dict(rank_components),
        "avg_score": float(avg_score),
        "focus_ripple_db": float(focus_ripple or 0.0),
        "mode_hz": float(mode_hz) if np.isfinite(mode_hz) else float("nan"),
        "mode_band_lo": float(mode_band_lo) if np.isfinite(mode_band_lo) else float("nan"),
        "mode_band_hi": float(mode_band_hi) if np.isfinite(mode_band_hi) else float("nan"),
        "mode_ripple_db": float(mode_ripple_db) if np.isfinite(mode_ripple_db) else float("nan"),
        "mode2_hz": float(mode2_hz) if np.isfinite(mode2_hz) else float("nan"),
        "mode2_band_lo": float(mode2_band_lo) if np.isfinite(mode2_band_lo) else float("nan"),
        "mode2_band_hi": float(mode2_band_hi) if np.isfinite(mode2_band_hi) else float("nan"),
        "mode2_ripple_db": float(mode2_ripple_db) if np.isfinite(mode2_ripple_db) else float("nan"),
        "mode_ripple_combined_db": float(mode_combined) if np.isfinite(mode_combined) else float("nan"),
        "mode_penalty": float(mode_penalty),
        "decay_penalty": float(decay_penalty),
        "tdc_decay_penalty": float(decay_penalty),
        "tdc_decay_need": float(shared._auto_safe_float(decay_dbg.get("tdc_decay_need", 0.0), 0.0)),
        "tdc_weak_penalty": float(
            shared._auto_safe_float(decay_dbg.get("tdc_weak_penalty", 0.0), 0.0)
        ),
        "tdc_overdamping_penalty": float(
            shared._auto_safe_float(decay_dbg.get("tdc_overdamping_penalty", 0.0), 0.0)
        ),
        "tdc_overreach_penalty": float(
            shared._auto_safe_float(decay_dbg.get("tdc_overreach_penalty", 0.0), 0.0)
        ),
        "tdc_decay_penalty_total": float(
            shared._auto_safe_float(decay_dbg.get("tdc_decay_penalty_total", decay_penalty), decay_penalty)
        ),
        "tdc_decay_optimum_strength": float(
            shared._auto_safe_float(decay_dbg.get("tdc_decay_optimum_strength", float("nan")), float("nan"))
        ),
        "modal_decay_excess_s": float(
            shared._auto_safe_float(decay_dbg.get("modal_decay_excess_s", float("nan")), float("nan"))
        ),
        "modal_decay_improvement_estimate": float(
            shared._auto_safe_float(decay_dbg.get("modal_decay_improvement_estimate", float("nan")), float("nan"))
        ),
        "strongest_modal_decay_hz": float(
            shared._auto_safe_float(decay_dbg.get("strongest_modal_decay_hz", float("nan")), float("nan"))
        ),
        "strongest_modal_decay_excess_s": float(
            shared._auto_safe_float(decay_dbg.get("strongest_modal_decay_excess_s", float("nan")), float("nan"))
        ),
        "tdc_decision": str(decay_dbg.get("tdc_decision", "") or ""),
        "tdc_action_hint": str(decay_dbg.get("tdc_action_hint", "") or ""),
        "tdc_extreme_overreach": bool(decay_dbg.get("tdc_extreme_overreach", False)),
        "tdc_decay_scoring_version": int(decay_dbg.get("tdc_decay_scoring_version", AUTO_TDC_DECAY_SCORING_VERSION)),
        "tdc_peak_reduction_hz": float(
            shared._auto_safe_float(decay_dbg.get("tdc_peak_reduction_hz", float("nan")), float("nan"))
        ),
        "tdc_reduction_band_low_hz": float(
            shared._auto_safe_float(decay_dbg.get("tdc_reduction_band_low_hz", float("nan")), float("nan"))
        ),
        "tdc_reduction_band_high_hz": float(
            shared._auto_safe_float(decay_dbg.get("tdc_reduction_band_high_hz", float("nan")), float("nan"))
        ),
        "tdc_modal_event_count": int(tdc_modal_debug.get("tdc_modal_event_count", 0) or 0),
        "tdc_modal_reductions": list(tdc_modal_debug.get("tdc_modal_reductions", []) or []),
        "tdc_modal_debug_available": bool(tdc_modal_debug.get("tdc_modal_debug_available", False)),
        "worst_residual_peak_db": float(worst_residual_peak_db) if np.isfinite(worst_residual_peak_db) else float("nan"),
        "worst_residual_peak_hz": float(
            shared._auto_safe_float(residual_peak_metrics.get("worst_residual_peak_hz", float("nan")), float("nan"))
        ),
        "worst_residual_peak_raw_db": float(worst_residual_peak_raw_db) if np.isfinite(worst_residual_peak_raw_db) else float("nan"),
        "worst_residual_peak_width_hz": float(
            shared._auto_safe_float(residual_peak_metrics.get("worst_residual_peak_width_hz", float("nan")), float("nan"))
        ),
        "worst_residual_peak_width_oct": float(
            shared._auto_safe_float(residual_peak_metrics.get("worst_residual_peak_width_oct", float("nan")), float("nan"))
        ),
        "residual_peak_area_db_oct": float(
            shared._auto_safe_float(residual_peak_metrics.get("residual_peak_area_db_oct", 0.0), 0.0)
        ),
        "broad_residual_peak_scoring_version": int(
            residual_peak_metrics.get("broad_residual_peak_scoring_version", BROAD_RESIDUAL_PEAK_SCORING_VERSION)
        ),
        "residual_peak_severity": float(residual_peak_severity) if np.isfinite(residual_peak_severity) else float("nan"),
        "residual_peak_threshold_db": float(
            shared._auto_safe_float(residual_peak_metrics.get("residual_peak_threshold_db", shared.AUTO_MODE_RESIDUAL_PEAK_THRESHOLD_DB), shared.AUTO_MODE_RESIDUAL_PEAK_THRESHOLD_DB)
        ),
        "residual_peak_hard_gate_db": float(
            shared._auto_safe_float(residual_peak_metrics.get("residual_peak_hard_gate_db", residual_peak_hard_gate_db), residual_peak_hard_gate_db)
        ),
        "top3_residual_peak_mean_db": float(top3_residual_peak_mean_db) if np.isfinite(top3_residual_peak_mean_db) else float("nan"),
        "residual_peak_count": int(residual_peak_metrics.get("residual_peak_count", 0) or 0),
        "residual_peak_modal_promoted_count": int(
            residual_peak_metrics.get("residual_peak_modal_promoted_count", 0) or 0
        ),
        "residual_peak_penalty": float(residual_peak_penalty),
        "residual_peak_penalty_cap": float(residual_peak_penalty_cap),
        "residual_peak_candidates": list(residual_peak_metrics.get("residual_peak_candidates", []) or []),
        "residual_peak_modal_support": float(
            shared._auto_safe_float(rp.get("residual_peak_modal_support", 0.0), 0.0)
        ),
        "residual_peak_modal_max_severity": float(
            shared._auto_safe_float(rp.get("residual_peak_modal_max_severity", 0.0), 0.0)
        ),
        "residual_peak_modal_confidence": float(
            shared._auto_safe_float(rp.get("residual_peak_modal_confidence", 0.0), 0.0)
        ),
        "residual_peak_modal_priority": float(
            shared._auto_safe_float(rp.get("residual_peak_modal_priority", 0.0), 0.0)
        ),
        "residual_peak_modal_dominant_freq_hz": (
            float(rp.get("residual_peak_modal_dominant_freq_hz"))
            if rp.get("residual_peak_modal_dominant_freq_hz") is not None
            else None
        ),
        "residual_peak_modal_event_count": int(rp.get("residual_peak_modal_event_count", 0) or 0),
        "residual_peak_modal_penalty": float(
            shared._auto_safe_float(rp.get("residual_peak_modal_penalty", 0.0), 0.0)
        ),
        "modal_residual_fallback_used": bool(modal_residual_fallback_used),
        "modal_residual_fallback_kind": str(modal_residual_fallback_kind),
        "modal_residual_fallback_hz": float(modal_residual_fallback_hz)
        if np.isfinite(modal_residual_fallback_hz) else float("nan"),
        "modal_residual_fallback_peak_db": float(modal_residual_fallback_peak_db)
        if np.isfinite(modal_residual_fallback_peak_db) else float("nan"),
        "modal_residual_fallback_penalty": float(max(0.0, modal_residual_fallback_penalty)),
        "residual_peak_gate_value_db": float(residual_peak_gate_value_db)
        if np.isfinite(residual_peak_gate_value_db) else float("nan"),
        "residual_peak_gate_source": str(residual_peak_gate_source),
        "modal_analysis_version": int(modal_intelligence_metrics.get("modal_analysis_version", 1) or 1),
        "modal_metrics_version": int(modal_intelligence_metrics.get("modal_metrics_version", MODAL_INTELLIGENCE_METRICS_VERSION) or MODAL_INTELLIGENCE_METRICS_VERSION),
        "modal_mode_count": int(modal_intelligence_metrics.get("modal_mode_count", 0) or 0),
        "modal_worst_mode_hz": (
            float(modal_intelligence_metrics.get("modal_worst_mode_hz"))
            if modal_intelligence_metrics.get("modal_worst_mode_hz") is not None
            else None
        ),
        "modal_worst_mode_severity": float(
            shared._auto_safe_float(modal_intelligence_metrics.get("modal_worst_mode_severity", 0.0), 0.0)
        ),
        "modal_area_db_oct": float(
            shared._auto_safe_float(modal_intelligence_metrics.get("modal_area_db_oct", 0.0), 0.0)
        ),
        "modal_voice_band_risk": float(
            shared._auto_safe_float(modal_intelligence_metrics.get("modal_voice_band_risk", 0.0), 0.0)
        ),
        "modal_events": list(modal_intelligence_metrics.get("modal_events", []) or []),
        "target_tracking_penalty": float(target_tracking_penalty),
        "bass_under_target_penalty": float(bass_under_target_penalty),
        "bass_under_target_rms_20_200_db": float(
            shared._auto_safe_float(bass_under_target_metrics.get("bass_under_target_rms_20_200_db", float("nan")), float("nan"))
        ),
        "bass_under_target_max_20_200_db": float(
            shared._auto_safe_float(bass_under_target_metrics.get("bass_under_target_max_20_200_db", float("nan")), float("nan"))
        ),
        "bass_under_target_rms_20_200_l_db": float(
            shared._auto_safe_float(bass_under_target_l.get("bass_under_target_rms_20_200_db", float("nan")), float("nan"))
        ),
        "bass_under_target_max_20_200_l_db": float(
            shared._auto_safe_float(bass_under_target_l.get("bass_under_target_max_20_200_db", float("nan")), float("nan"))
        ),
        "bass_under_target_rms_20_200_r_db": float(
            shared._auto_safe_float(bass_under_target_r.get("bass_under_target_rms_20_200_db", float("nan")), float("nan"))
        ),
        "bass_under_target_max_20_200_r_db": float(
            shared._auto_safe_float(bass_under_target_r.get("bass_under_target_max_20_200_db", float("nan")), float("nan"))
        ),
        "correction_max_abs_slope_db_per_oct": float(
            shared._auto_safe_float(sharpness_metrics.get("correction_max_abs_slope_db_per_oct", 0.0), 0.0)
        ),
        "correction_rms_slope_db_per_oct": float(
            shared._auto_safe_float(sharpness_metrics.get("correction_rms_slope_db_per_oct", 0.0), 0.0)
        ),
        "correction_curvature_score": float(
            shared._auto_safe_float(sharpness_metrics.get("correction_curvature_score", 0.0), 0.0)
        ),
        "narrow_notch_count": int(sharpness_metrics.get("narrow_notch_count", 0) or 0),
        "correction_event_count": int(sharpness_metrics.get("correction_event_count", 0) or 0),
        "direction_change_count": int(sharpness_metrics.get("direction_change_count", 0) or 0),
        "correction_sharpness_penalty": float(correction_sharpness_penalty),
        "correction_sharpness_scoring_version": int(CORRECTION_SHARPNESS_SCORING_VERSION),
        "dip_fill_risk_score": float(shared._auto_safe_float(dip_fill_metrics.get("dip_fill_risk_score", 0.0), 0.0)),
        "dip_fill_risk_penalty": float(dip_fill_risk_penalty),
        "dip_fill_boost_peak_db": float(shared._auto_safe_float(dip_fill_metrics.get("dip_fill_boost_peak_db", 0.0), 0.0)),
        "dip_fill_deep_narrow_count": int(dip_fill_metrics.get("dip_fill_deep_narrow_count", 0) or 0),
        "dip_fill_risk_scoring_version": int(DIP_FILL_RISK_SCORING_VERSION),
        "correction_lr_delta_rms_20_250": float(
            shared._auto_safe_float(channel_overfit_metrics.get("correction_lr_delta_rms_20_250", 0.0), 0.0)
        ),
        "correction_lr_delta_max_20_250": float(
            shared._auto_safe_float(channel_overfit_metrics.get("correction_lr_delta_max_20_250", 0.0), 0.0)
        ),
        "narrow_lr_divergence_count": int(channel_overfit_metrics.get("narrow_lr_divergence_count", 0) or 0),
        "channel_overfit_penalty": float(channel_overfit_penalty),
        "channel_overfit_scoring_version": int(CHANNEL_OVERFIT_SCORING_VERSION),
        "voice_clarity_penalty": float(voice_clarity_penalty),
        "voice_band_rms_error_db": float(
            shared._auto_safe_float(voice_metrics.get("voice_band_rms_error_db", 0.0), 0.0)
        ),
        "voice_band_peak_excess_db": float(
            shared._auto_safe_float(voice_metrics.get("voice_band_peak_excess_db", 0.0), 0.0)
        ),
        "voice_band_energy_excess_db": float(
            shared._auto_safe_float(voice_metrics.get("voice_band_energy_excess_db", 0.0), 0.0)
        ),
        "voice_band_lr_mismatch_db": float(
            shared._auto_safe_float(voice_metrics.get("voice_band_lr_mismatch_db", 0.0), 0.0)
        ),
        "voice_band_gd_peak_ms": float(
            shared._auto_safe_float(voice_metrics.get("voice_band_gd_peak_ms", 0.0), 0.0)
        ),
        "voice_band_risk_mean": float(
            shared._auto_safe_float(voice_metrics.get("voice_band_risk_mean", 0.0), 0.0)
        ),
        "voice_band_risk_peak": float(
            shared._auto_safe_float(voice_metrics.get("voice_band_risk_peak", 0.0), 0.0)
        ),
        "target_tracking_rms_20_200_db": float(
            shared._auto_safe_float(target_tracking_metrics.get("target_tracking_rms_20_200_db", float("nan")), float("nan"))
        ),
        "target_tracking_max_20_200_db": float(
            shared._auto_safe_float(target_tracking_metrics.get("target_tracking_max_20_200_db", float("nan")), float("nan"))
        ),
        "target_tracking_rms_100_500_db": float(
            shared._auto_safe_float(target_tracking_metrics.get("target_tracking_rms_100_500_db", float("nan")), float("nan"))
        ),
        "target_tracking_max_100_500_db": float(
            shared._auto_safe_float(target_tracking_metrics.get("target_tracking_max_100_500_db", float("nan")), float("nan"))
        ),
        "target_tracking_rms_20_200_l_db": float(
            shared._auto_safe_float(target_tracking_l.get("target_tracking_rms_20_200_db", float("nan")), float("nan"))
        ),
        "target_tracking_max_20_200_l_db": float(
            shared._auto_safe_float(target_tracking_l.get("target_tracking_max_20_200_db", float("nan")), float("nan"))
        ),
        "target_tracking_rms_100_500_l_db": float(
            shared._auto_safe_float(target_tracking_l.get("target_tracking_rms_100_500_db", float("nan")), float("nan"))
        ),
        "target_tracking_max_100_500_l_db": float(
            shared._auto_safe_float(target_tracking_l.get("target_tracking_max_100_500_db", float("nan")), float("nan"))
        ),
        "target_tracking_rms_20_200_r_db": float(
            shared._auto_safe_float(target_tracking_r.get("target_tracking_rms_20_200_db", float("nan")), float("nan"))
        ),
        "target_tracking_max_20_200_r_db": float(
            shared._auto_safe_float(target_tracking_r.get("target_tracking_max_20_200_db", float("nan")), float("nan"))
        ),
        "target_tracking_rms_100_500_r_db": float(
            shared._auto_safe_float(target_tracking_r.get("target_tracking_rms_100_500_db", float("nan")), float("nan"))
        ),
        "target_tracking_max_100_500_r_db": float(
            shared._auto_safe_float(target_tracking_r.get("target_tracking_max_100_500_db", float("nan")), float("nan"))
        ),
        "realized_rms_20_200_db": float(realized_rms_20_200) if np.isfinite(realized_rms_20_200) else float("nan"),
        "ir_pre_post_energy_ratio_l": float(pre_post_l_f) if np.isfinite(pre_post_l_f) else float("nan"),
        "ir_pre_post_energy_ratio_r": float(pre_post_r_f) if np.isfinite(pre_post_r_f) else float("nan"),
        "ir_pre_post_energy_ratio_max": float(pre_post_max) if np.isfinite(pre_post_max) else float("nan"),
        "ripple_rms": float(ripple_raw) if np.isfinite(ripple_raw) else float("nan"),
        "lr_delta_score": float(lr_delta),
        "max_net_boost_db": float(net_boost_max),
        "bass_preference_bonus": float(bass_preference_bonus),
        "bass_boost_20_200_db": float(shared._auto_safe_float(bass_boost_metrics.get("bass_boost_20_200_db", 0.0), 0.0)),
        "bass_boost_20_200_mean_db": float(shared._auto_safe_float(bass_boost_metrics.get("bass_boost_20_200_mean_db", 0.0), 0.0)),
        "bass_boost_20_200_max_db": float(shared._auto_safe_float(bass_boost_metrics.get("bass_boost_20_200_max_db", 0.0), 0.0)),
        "bass_boost_peak_20_200_db": float(shared._auto_safe_float(bass_boost_metrics.get("bass_boost_peak_20_200_db", 0.0), 0.0)),
        "bass_boost_20_200_l_db": float(shared._auto_safe_float(bass_boost_l.get("bass_boost_20_200_db", 0.0), 0.0)),
        "bass_boost_20_200_r_db": float(shared._auto_safe_float(bass_boost_r.get("bass_boost_20_200_db", 0.0), 0.0)),
        "post_filter_boost_peak_db": float(
            max(
                shared._auto_safe_float(l_st.get("boost_peak_db", 0.0), 0.0),
                shared._auto_safe_float(r_st.get("boost_peak_db", 0.0), 0.0),
            )
        ),
        "lf_boost_max_db": float(
            max(
                shared._auto_safe_float(l_st.get("lf_boost_max_db", 0.0), 0.0),
                shared._auto_safe_float(r_st.get("lf_boost_max_db", 0.0), 0.0),
            )
        ),
        "boost_penalty": float(boost_pen),
        "events_total": int(events_total),
        "events_severity": float(events_severity),
        "events_severity_raw": float(events_severity_raw),
        "events_severity_l": float(events_severity_l),
        "events_severity_r": float(events_severity_r),
        "event_penalty": float(event_pen),
        "lr_delta_penalty": float(lr_pen),
        "dsp_penalty": float(dsp_penalty),
        "dsp_penalty_raw": float(dsp_penalty_raw),
        "dsp_penalty_l": float(dsp_pen_l),
        "dsp_penalty_r": float(dsp_pen_r),
        "exc_penalty": float(exc_penalty),
        "bass_integration_penalty": float(bass_integration_penalty),
        "bass_feasibility_penalty": float(bass_feasibility_penalty),
        "bass_cancellation_risk": float(shared._auto_safe_float(bass_dbg.get("bass_cancellation_risk", float("nan")), float("nan"))),
        "bass_overlap_ripple": float(shared._auto_safe_float(bass_dbg.get("bass_overlap_ripple", float("nan")), float("nan"))),
        "bass_sub_dominance": float(shared._auto_safe_float(bass_dbg.get("bass_sub_dominance", float("nan")), float("nan"))),
        "bass_overlap_extension_active": bool(bass_dbg.get("bass_overlap_extension_active", False)),
        "bass_overlap_extension_flatness_db": float(
            shared._auto_safe_float(bass_dbg.get("bass_overlap_extension_flatness_db", float("nan")), float("nan"))
        ),
        "bass_overlap_extension_cancellation_risk": float(
            shared._auto_safe_float(bass_dbg.get("bass_overlap_extension_cancellation_risk", float("nan")), float("nan"))
        ),
        "bass_overlap_extension_peak_excess_db": float(
            shared._auto_safe_float(bass_dbg.get("bass_overlap_extension_peak_excess_db", float("nan")), float("nan"))
        ),
        "bass_overlap_extension_sub_dominance_db": float(
            shared._auto_safe_float(bass_dbg.get("bass_overlap_extension_sub_dominance_db", float("nan")), float("nan"))
        ),
        "bass_overlap_ripple_delta_db": float(shared._auto_safe_float(bass_dbg.get("bass_overlap_ripple_delta_db", float("nan")), float("nan"))),
        "bass_sub_dominance_delta_db": float(shared._auto_safe_float(bass_dbg.get("bass_sub_dominance_delta_db", float("nan")), float("nan"))),
        "bass_xo_gd_mismatch_delta_ms": float(shared._auto_safe_float(bass_dbg.get("bass_xo_gd_mismatch_delta_ms", float("nan")), float("nan"))),
        "bass_dominant_channel": str(bass_dbg.get("bass_dominant_channel", result_metrics.get("bass_dominant_channel", "unknown")) or "unknown"),
        "bass_feasibility_class": str(bass_dbg.get("bass_feasibility_class", result_metrics.get("bass_feasibility_class", "unknown")) or "unknown"),
        "bass_feasibility_reason": str(bass_dbg.get("bass_feasibility_reason", result_metrics.get("bass_feasibility_reason", "")) or ""),
        "bass_xo_gd_mismatch_ms": float(shared._auto_safe_float(result_metrics.get("bass_xo_gd_mismatch_ms", float("nan")), float("nan"))),
        "bass_xo_gd_rms_mismatch_ms": float(shared._auto_safe_float(bass_dbg.get("bass_xo_gd_rms_mismatch_ms", float("nan")), float("nan"))),
        "bass_xo_main_gd_ms": float(shared._auto_safe_float(result_metrics.get("bass_xo_main_gd_ms", float("nan")), float("nan"))),
        "bass_xo_sub_gd_ms": float(shared._auto_safe_float(result_metrics.get("bass_xo_sub_gd_ms", float("nan")), float("nan"))),
        "bass_guard_lo_hz": float(shared._auto_safe_float(result_metrics.get("bass_guard_lo_hz", float("nan")), float("nan"))),
        "bass_guard_hi_hz": float(shared._auto_safe_float(result_metrics.get("bass_guard_hi_hz", float("nan")), float("nan"))),
        "bass_sub_combined_level_delta_db_20_120": float(shared._auto_safe_float(result_metrics.get("bass_sub_combined_level_delta_db_20_120", float("nan")), float("nan"))),
        "bass_sub_combined_level_delta_db_30_90": float(shared._auto_safe_float(result_metrics.get("bass_sub_combined_level_delta_db_30_90", float("nan")), float("nan"))),
        "exc_penalty_raw": float(exc_penalty_raw),
        "exc_penalty_raw_total": float(exc_penalty_raw_total),
        "exc_penalty_bins_raw": float(exc_penalty_bins_raw),
        "exc_penalty_bins_waived": bool(exc_penalty_bins_waived),
        "exc_penalty_waived": bool(exc_penalty_waived),
        "exc_penalty_l": float(exc_pen_l),
        "exc_penalty_r": float(exc_pen_r),
        "auto_exc_zero_penalty_hz": float(auto_exc_zero_penalty_hz) if np.isfinite(auto_exc_zero_penalty_hz) else float("nan"),
        "phase_benefit_bonus": float(phase_benefit_bonus),
        "phase_risk_penalty": float(phase_risk_penalty),
        "phase_net_score": float(phase_net_score),
        "phase_lr_consistency_penalty": float(phase_lr_consistency_penalty),
        "phase_limit_hz": float(phase_limit_used_hz) if np.isfinite(phase_limit_used_hz) else float("nan"),
        "phase_limit_penalty": float(phase_limit_penalty),
        "thd_boost_penalty": float(thd_boost_penalty),
        "stereo_policy_active": bool(stereo_policy_active),
        "stereo_policy_gate_failed": bool(stereo_policy_gate_failed),
        "stereo_policy_split_hz": float(stereo_split_hz),
        "stereo_lr_mismatch_below_split_db": float(stereo_lr_mismatch_below) if np.isfinite(stereo_lr_mismatch_below) else float("nan"),
        "stereo_lr_mismatch_above_split_db": float(stereo_lr_mismatch_above) if np.isfinite(stereo_lr_mismatch_above) else float("nan"),
        "stereo_lr_mismatch_below_split_delta_vs_shared_db": float(stereo_lr_mismatch_below_delta) if np.isfinite(stereo_lr_mismatch_below_delta) else float("nan"),
        "stereo_lr_mismatch_above_split_delta_vs_shared_db": float(stereo_lr_mismatch_above_delta) if np.isfinite(stereo_lr_mismatch_above_delta) else float("nan"),
        "phantom_center_mismatch_change_db": float(phantom_center_change_db) if np.isfinite(phantom_center_change_db) else float("nan"),
        "phantom_center_mismatch_change_delta_vs_shared_db": float(phantom_center_change_delta_db) if np.isfinite(phantom_center_change_delta_db) else float("nan"),
        "policy_divergence_score": float(policy_divergence_score),
        "stereo_coherence_penalty": float(stereo_coherence_penalty),
        "phantom_center_stability_penalty": float(phantom_center_stability_penalty),
        "policy_divergence_penalty": float(policy_divergence_penalty),
        "asymmetry_budget_overflow_penalty": float(asymmetry_budget_overflow_penalty),
        "stereo_worst_channel_relief_db": float(stereo_worst_channel_relief_db) if np.isfinite(stereo_worst_channel_relief_db) else float("nan"),
        "worst_channel_relief_bonus": float(worst_channel_relief_bonus),
        "shared_preference_bias": float(shared_preference_bias),
        "rt60_policy_penalty": float(_rt60_policy_pen),
        "harmonic_local_boost_penalty": float(_harmonic_local_pen),
        "tdc_decay_dbg": dict(decay_dbg),
        "dsp_dbg_l": dict(dsp_dbg_l),
        "dsp_dbg_r": dict(dsp_dbg_r),
        "phase_dbg_l": dict(phase_dbg_l),
        "phase_dbg_r": dict(phase_dbg_r),
        "exc_dbg_l": dict(exc_dbg_l),
        "exc_dbg_r": dict(exc_dbg_r),
        "bass_dbg": dict(bass_dbg),
    }
    return finalize_score_result_metrics(
        metrics_out,
        base_data=base_data,
        worst_residual_peak_raw_db=worst_residual_peak_raw_db,
        worst_residual_peak_db=worst_residual_peak_db,
        stereo_policy_gate_failed=stereo_policy_gate_failed,
        rank_score=rank_score,
        rank_components=rank_components,
    )


__all__ = ['_auto_score_result']

