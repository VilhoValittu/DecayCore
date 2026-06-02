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

import inspect
import logging
import math
import sys

import numpy as np

_logger = logging.getLogger(__name__)

from ...common.acoustic_stats import calc_acoustic_score, calc_ai_summary_from_stats
from ...config.models import StereoAutoPolicyConfig, StereoResolvedAutoPolicies
from ...dsp.quality_metrics import (
    band_lr_mismatch_change_from_stats,
    band_lr_mismatch_rms_from_stats,
    normalized_policy_divergence_score,
    worst_channel_relief_db,
)
from ...dsp.modal_analysis import ModalAnalysisResult, RoomModeEvent, detect_room_modes
from ...dsp.smoothing import smooth_gain_fractional_octave
from ...dsp.target_match import target_match_from_stats
from .. import shared
from ..rank_score import (
    OFFICIAL_RANK_SCORE_CONTEXT,
    attach_official_rank_score,
    calibrated_auto_quality,
    compute_rank_score_components,
)
from ..runtime_context import (
    _auto_collect_reflections,
    _auto_event_penalty_weighted,
    _auto_event_severity,
    _auto_get_top_modes_hz,
    _auto_get_worst_mode_hz,
    _auto_mode_band,
    _auto_pick_metric,
)

from .metric_penalties import _auto_focus_ripple_from_stats

def _rank_scale(v: float) -> float:
    g = float(shared._auto_safe_float(shared.AUTO_MODE_RANK_SCORE_GAIN, 1.0))
    b = float(shared._auto_safe_float(shared.AUTO_MODE_RANK_SCORE_BIAS, 0.0))
    return float(np.clip(float(g) * float(v) + float(b), 0.0, 100.0))


def _collect_rank_focus_metrics(l_st, r_st, *, focus_lo_hz, focus_hi_hz) -> dict:
    focus_ripple_l = None
    focus_ripple_r = None
    flo = shared._auto_safe_float(focus_lo_hz, float("nan"))
    fhi = shared._auto_safe_float(focus_hi_hz, float("nan"))
    if np.isfinite(flo) and np.isfinite(fhi) and float(fhi) > float(flo):
        focus_ripple_l = _auto_focus_ripple_from_stats(l_st, focus_lo_hz=float(flo), focus_hi_hz=float(fhi))
        focus_ripple_r = _auto_focus_ripple_from_stats(r_st, focus_lo_hz=float(flo), focus_hi_hz=float(fhi))
    if not (
        np.isfinite(shared._auto_safe_float(focus_ripple_l, float("nan")))
        or np.isfinite(shared._auto_safe_float(focus_ripple_r, float("nan")))
    ):
        focus_ripple_keys = (
            "post_to_ir_staged_shape_delta_rms_20_200_db",
            "post_to_ir_shape_delta_rms_20_200_db",
            "post_to_ir_delta_rms_20_200_db",
            "ripple_rms",
        )
        focus_ripple_l = _auto_pick_metric(l_st, focus_ripple_keys, abs_value=True, nonneg=True)
        focus_ripple_r = _auto_pick_metric(r_st, focus_ripple_keys, abs_value=True, nonneg=True)
    focus_ripple_vals = []
    for v in (focus_ripple_l, focus_ripple_r):
        x = shared._auto_safe_float(v, float("nan"))
        if np.isfinite(x):
            focus_ripple_vals.append(float(x))
    focus_ripple = float(np.mean(np.asarray(focus_ripple_vals, dtype=float))) if focus_ripple_vals else 0.0
    return {
        "focus_ripple": focus_ripple,
        "focus_ripple_l": focus_ripple_l,
        "focus_ripple_r": focus_ripple_r,
    }


def _collect_rank_base_metrics(base_rank_components: dict, mode_penalty: float) -> tuple[float, float]:
    rank_raw = float(base_rank_components.get("rank_score_raw", 0.0))
    rank_score = float(base_rank_components.get("rank_score", 0.0))
    if mode_penalty > 0.0:
        rank_raw = float(rank_raw - float(mode_penalty))
        rank_score = float(_rank_scale(rank_raw))
    return rank_raw, rank_score


def _collect_rank_top_modes(result) -> list[float]:
    try:
        if bool(shared.AUTO_MODE_DUAL_MODE_ENABLED):
            return _auto_get_top_modes_hz(result, top_n=int(shared.AUTO_MODE_DUAL_MODE_TOP_N))
        m1 = _auto_get_worst_mode_hz(result)
        return [float(m1)] if m1 is not None else []
    except (
        AttributeError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        RuntimeError,
        OSError,
        ImportError,
        ModuleNotFoundError,
        NameError,
    ):
        return []


def _collect_rank_cached_focus_ripple(st: dict, lo: float, hi: float) -> float | None:
    key = (id(st), lo, hi)
    if key not in _RANK_MODE_RIPPLE_MEMO:
        _RANK_MODE_RIPPLE_MEMO[key] = _auto_focus_ripple_from_stats(st, focus_lo_hz=lo, focus_hi_hz=hi)
    return _RANK_MODE_RIPPLE_MEMO[key]


def _collect_rank_band_ripple(st: dict, band: tuple[float, float] | None, fallback: float) -> float:
    if not (isinstance(band, tuple) and len(band) == 2):
        return float(shared._auto_safe_float(fallback, float("nan")))

    band_lo = float(shared._auto_safe_float(band[0], float("nan")))
    band_hi = float(shared._auto_safe_float(band[1], float("nan")))
    if not (np.isfinite(band_lo) and np.isfinite(band_hi) and band_hi > band_lo):
        return float(shared._auto_safe_float(fallback, float("nan")))

    ripple = _collect_rank_cached_focus_ripple(st, float(band_lo), float(band_hi))
    ripple_f = shared._auto_safe_float(ripple, float("nan"))
    if np.isfinite(ripple_f):
        return float(ripple_f)
    return float(shared._auto_safe_float(fallback, float("nan")))


def _collect_rank_mode_band_info(
    st: dict,
    mode_hz: float,
    *,
    base_data,
    fallback_ripple: float,
) -> tuple[float, float, float]:
    band = _auto_mode_band(mode_hz, base_data=base_data) if np.isfinite(mode_hz) else None
    band_lo = float(shared._auto_safe_float(band[0], float("nan"))) if isinstance(band, tuple) and len(band) == 2 else float("nan")
    band_hi = float(shared._auto_safe_float(band[1], float("nan"))) if isinstance(band, tuple) and len(band) == 2 else float("nan")
    ripple_db = _collect_rank_band_ripple(st, band, fallback_ripple)
    return band_lo, band_hi, ripple_db


def _collect_rank_mode_combined_penalty(mode_ripple_db: float, mode2_ripple_db: float) -> tuple[float, float]:
    mode_r1 = shared._auto_safe_float(mode_ripple_db, float("nan"))
    mode_r2 = shared._auto_safe_float(mode2_ripple_db, float("nan"))
    mode_combined = float("nan")
    if np.isfinite(mode_r1) and np.isfinite(mode_r2):
        mode_combined = max(float(mode_r1), float(shared.AUTO_MODE_MODE_RIPPLE_SECONDARY_W) * float(mode_r2))
    elif np.isfinite(mode_r1):
        mode_combined = float(mode_r1)
    elif np.isfinite(mode_r2):
        mode_combined = float(mode_r2)

    mode_penalty = 0.0
    if np.isfinite(mode_combined):
        mode_penalty = float(shared.AUTO_MODE_MODE_RIPPLE_PENALTY_W) * max(
            0.0,
            float(mode_combined) - float(shared.AUTO_MODE_MODE_RIPPLE_OK_DB),
        )
        mode_penalty = float(np.clip(mode_penalty, 0.0, 3.5))
    return mode_combined, mode_penalty


def _collect_rank_mode_band_metrics(l_st, r_st, result, *, base_data, focus_ripple: float) -> dict:
    top_modes = _collect_rank_top_modes(result)
    mode_hz = shared._auto_safe_float((top_modes[0] if top_modes else _auto_get_worst_mode_hz(result)), float("nan"))
    mode2_hz = shared._auto_safe_float((top_modes[1] if len(top_modes) >= 2 else float("nan")), float("nan"))
    mode_band_lo, mode_band_hi, mode_ripple_db = _collect_rank_mode_band_info(
        l_st,
        mode_hz,
        base_data=base_data,
        fallback_ripple=focus_ripple,
    )
    mode2_band_lo, mode2_band_hi, mode2_ripple_db = _collect_rank_mode_band_info(
        r_st,
        mode2_hz,
        base_data=base_data,
        fallback_ripple=focus_ripple,
    )
    mode_combined, mode_penalty = _collect_rank_mode_combined_penalty(mode_ripple_db, mode2_ripple_db)

    return {
        "mode_hz": mode_hz,
        "mode_band_lo": mode_band_lo,
        "mode_band_hi": mode_band_hi,
        "mode_ripple_db": mode_ripple_db,
        "mode2_hz": mode2_hz,
        "mode2_band_lo": mode2_band_lo,
        "mode2_band_hi": mode2_band_hi,
        "mode2_ripple_db": mode2_ripple_db,
        "mode_combined": mode_combined,
        "mode_penalty": mode_penalty,
    }


def _collect_rank_mode_metrics(
    l_st,
    r_st,
    result,
    *,
    focus_lo_hz,
    focus_hi_hz,
    base_data,
    base_rank_components: dict,
) -> dict:
    focus_metrics = _collect_rank_focus_metrics(l_st, r_st, focus_lo_hz=focus_lo_hz, focus_hi_hz=focus_hi_hz)
    mode_metrics = _collect_rank_mode_band_metrics(
        l_st,
        r_st,
        result,
        base_data=base_data,
        focus_ripple=float(focus_metrics["focus_ripple"]),
    )
    rank_raw, rank_score = _collect_rank_base_metrics(base_rank_components, float(mode_metrics["mode_penalty"]))
    return {
        **focus_metrics,
        **mode_metrics,
        "rank_raw": rank_raw,
        "rank_score": rank_score,
    }


def _collect_rank_post_metrics(l_st, r_st) -> dict:
    realized_keys = (
        "post_to_ir_staged_shape_delta_rms_20_200_db",
        "post_to_ir_shape_delta_rms_20_200_db",
        "post_to_ir_delta_rms_20_200_db",
    )
    realized_l = _auto_pick_metric(l_st, realized_keys, abs_value=True, nonneg=True)
    realized_r = _auto_pick_metric(r_st, realized_keys, abs_value=True, nonneg=True)
    realized_vals = []
    for rv in (realized_l, realized_r):
        x = shared._auto_safe_float(rv, float("nan"))
        if np.isfinite(x):
            realized_vals.append(float(x))
    realized_rms_20_200 = float(np.mean(np.asarray(realized_vals, dtype=float))) if realized_vals else float("nan")

    ripple_raw_l = _auto_pick_metric(l_st, ("ripple_rms",), abs_value=True, nonneg=True)
    ripple_raw_r = _auto_pick_metric(r_st, ("ripple_rms",), abs_value=True, nonneg=True)
    ripple_raw_vals = []
    for rv in (ripple_raw_l, ripple_raw_r):
        x = shared._auto_safe_float(rv, float("nan"))
        if np.isfinite(x):
            ripple_raw_vals.append(float(x))
    ripple_raw = float(np.mean(np.asarray(ripple_raw_vals, dtype=float))) if ripple_raw_vals else float("nan")

    pre_post_keys = (
        "ir_pre_post_ratio",
        "ir_pre_energy_guard_after_ratio",
        "ir_pre_energy_guard_before_ratio",
    )
    pre_post_l = None if bool(l_st.get("pre_energy_metric_suspect", False)) else _auto_pick_metric(
        l_st,
        pre_post_keys,
        nonneg=True,
    )
    pre_post_r = None if bool(r_st.get("pre_energy_metric_suspect", False)) else _auto_pick_metric(
        r_st,
        pre_post_keys,
        nonneg=True,
    )
    pre_post_l_f = shared._auto_safe_float(pre_post_l, float("nan"))
    pre_post_r_f = shared._auto_safe_float(pre_post_r, float("nan"))
    pre_post_max = float("nan")
    pre_post_vals = []
    if np.isfinite(pre_post_l_f):
        pre_post_vals.append(float(pre_post_l_f))
    if np.isfinite(pre_post_r_f):
        pre_post_vals.append(float(pre_post_r_f))
    if pre_post_vals:
        pre_post_max = float(max(pre_post_vals))

    return {
        "realized_rms_20_200": realized_rms_20_200,
        "ripple_raw": ripple_raw,
        "pre_post_l_f": pre_post_l_f,
        "pre_post_r_f": pre_post_r_f,
        "pre_post_max": pre_post_max,
    }

def combine_rank_score(
    l_st,
    r_st,
    result,
    *,
    focus_lo_hz,
    focus_hi_hz,
    base_data,
    avg_score,
    phase_benefit_bonus,
    boost_pen,
    event_pen,
    lr_pen,
    dsp_penalty,
    bass_prering_penalty,
    exc_penalty,
    bass_integration_penalty,
    bass_feasibility_penalty,
    bass_preference_bonus,
    decay_penalty,
    residual_peak_penalty,
    correction_sharpness_penalty,
    dip_fill_risk_penalty,
    channel_overfit_penalty,
    target_tracking_penalty,
    voice_clarity_penalty,
    phase_risk_penalty,
    phase_limit_penalty,
    thd_boost_penalty,
    stereo_coherence_penalty,
    phantom_center_stability_penalty,
    policy_divergence_penalty,
    asymmetry_budget_overflow_penalty,
    worst_channel_relief_bonus,
    shared_preference_bias,
    rt60_policy_pen,
    harmonic_local_pen,
) -> dict:
    _rt60_policy_pen = rt60_policy_pen
    _harmonic_local_pen = harmonic_local_pen
    base_rank_components = compute_rank_score_components(
        avg_score=avg_score,
        phase_benefit_bonus=phase_benefit_bonus,
        boost_penalty=boost_pen,
        event_penalty=event_pen,
        lr_delta_penalty=lr_pen,
        dsp_penalty=dsp_penalty,
        bass_prering_penalty=bass_prering_penalty,
        exc_penalty=exc_penalty,
        bass_integration_penalty=bass_integration_penalty,
        bass_feasibility_penalty=bass_feasibility_penalty,
        bass_preference_bonus=bass_preference_bonus,
        decay_penalty=decay_penalty,
        residual_peak_penalty=residual_peak_penalty,
        correction_sharpness_penalty=correction_sharpness_penalty,
        dip_fill_risk_penalty=dip_fill_risk_penalty,
        channel_overfit_penalty=channel_overfit_penalty,
        target_tracking_penalty=target_tracking_penalty,
        voice_clarity_penalty=voice_clarity_penalty,
        phase_risk_penalty=phase_risk_penalty,
        phase_limit_penalty=phase_limit_penalty,
        stereo_coherence_penalty=stereo_coherence_penalty,
        phantom_center_stability_penalty=phantom_center_stability_penalty,
        policy_divergence_penalty=policy_divergence_penalty,
        asymmetry_budget_overflow_penalty=asymmetry_budget_overflow_penalty,
        worst_channel_relief_bonus=worst_channel_relief_bonus,
        shared_preference_bias=shared_preference_bias,
        gain=shared._auto_safe_float(shared.AUTO_MODE_RANK_SCORE_GAIN, 1.0),
        bias=shared._auto_safe_float(shared.AUTO_MODE_RANK_SCORE_BIAS, 0.0),
        context=OFFICIAL_RANK_SCORE_CONTEXT,
    )
    mode_metrics = _collect_rank_mode_metrics(
        l_st,
        r_st,
        result,
        focus_lo_hz=focus_lo_hz,
        focus_hi_hz=focus_hi_hz,
        base_data=base_data,
        base_rank_components=base_rank_components,
    )
    post_metrics = _collect_rank_post_metrics(l_st, r_st)
    rank_raw = float(mode_metrics["rank_raw"])
    rank_score_base = float(base_rank_components.get("rank_score", 0.0))
    rank_score = float(mode_metrics["rank_score"])
    focus_ripple = float(mode_metrics["focus_ripple"])
    focus_ripple_l = mode_metrics["focus_ripple_l"]
    focus_ripple_r = mode_metrics["focus_ripple_r"]
    mode_hz = mode_metrics["mode_hz"]
    mode_band_lo = mode_metrics["mode_band_lo"]
    mode_band_hi = mode_metrics["mode_band_hi"]
    mode_ripple_db = mode_metrics["mode_ripple_db"]
    mode2_hz = mode_metrics["mode2_hz"]
    mode2_band_lo = mode_metrics["mode2_band_lo"]
    mode2_band_hi = mode_metrics["mode2_band_hi"]
    mode2_ripple_db = mode_metrics["mode2_ripple_db"]
    mode_combined = mode_metrics["mode_combined"]
    mode_penalty = mode_metrics["mode_penalty"]
    rank_components = compute_rank_score_components(
        avg_score=avg_score,
        phase_benefit_bonus=phase_benefit_bonus,
        boost_penalty=boost_pen,
        event_penalty=event_pen,
        lr_delta_penalty=lr_pen,
        dsp_penalty=dsp_penalty,
        bass_prering_penalty=bass_prering_penalty,
        exc_penalty=exc_penalty,
        bass_integration_penalty=bass_integration_penalty,
        bass_feasibility_penalty=bass_feasibility_penalty,
        bass_preference_bonus=bass_preference_bonus,
        mode_penalty=mode_penalty,
        decay_penalty=decay_penalty,
        residual_peak_penalty=residual_peak_penalty,
        correction_sharpness_penalty=correction_sharpness_penalty,
        dip_fill_risk_penalty=dip_fill_risk_penalty,
        channel_overfit_penalty=channel_overfit_penalty,
        target_tracking_penalty=target_tracking_penalty,
        voice_clarity_penalty=voice_clarity_penalty,
        phase_risk_penalty=phase_risk_penalty,
        phase_limit_penalty=phase_limit_penalty,
        thd_boost_penalty=thd_boost_penalty,
        stereo_coherence_penalty=stereo_coherence_penalty,
        phantom_center_stability_penalty=phantom_center_stability_penalty,
        policy_divergence_penalty=policy_divergence_penalty,
        asymmetry_budget_overflow_penalty=asymmetry_budget_overflow_penalty,
        worst_channel_relief_bonus=worst_channel_relief_bonus,
        shared_preference_bias=shared_preference_bias,
        rt60_policy_penalty=_rt60_policy_pen,
        harmonic_local_boost_penalty=_harmonic_local_pen,
        gain=shared._auto_safe_float(shared.AUTO_MODE_RANK_SCORE_GAIN, 1.0),
        bias=shared._auto_safe_float(shared.AUTO_MODE_RANK_SCORE_BIAS, 0.0),
        context=OFFICIAL_RANK_SCORE_CONTEXT,
    )
    rank_score = float(rank_components.get("rank_score", rank_score))

    return {
        "rank_score": rank_score,
        "rank_score_base": rank_score_base,
        "rank_components": rank_components,
        "base_rank_components": base_rank_components,
        "focus_ripple": focus_ripple,
        "focus_ripple_l": focus_ripple_l,
        "focus_ripple_r": focus_ripple_r,
        "mode_hz": mode_hz,
        "mode_band_lo": mode_band_lo,
        "mode_band_hi": mode_band_hi,
        "mode_ripple_db": mode_ripple_db,
        "mode2_hz": mode2_hz,
        "mode2_band_lo": mode2_band_lo,
        "mode2_band_hi": mode2_band_hi,
        "mode2_ripple_db": mode2_ripple_db,
        "mode_combined": mode_combined,
        "mode_penalty": mode_penalty,
        "realized_rms_20_200": post_metrics["realized_rms_20_200"],
        "ripple_raw": post_metrics["ripple_raw"],
        "pre_post_l_f": post_metrics["pre_post_l_f"],
        "pre_post_r_f": post_metrics["pre_post_r_f"],
        "pre_post_max": post_metrics["pre_post_max"],
    }



__all__ = ["_rank_scale", "combine_rank_score"]
