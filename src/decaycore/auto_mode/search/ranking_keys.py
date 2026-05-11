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

import logging

import numpy as np

from ..shared import (
    AUTO_MODE_ADAPTIVE_SHRINK_ENABLED,
    AUTO_MODE_ADAPTIVE_SHRINK_MAX,
    AUTO_MODE_ADAPTIVE_SHRINK_MIN,
    AUTO_MODE_GOAL_DEFAULT,
    AUTO_MODE_GOAL_FLAT,
    AUTO_MODE_GOAL_LOW_RIPPLE,
    AUTO_MODE_GOAL_ROOM_SAFE,
    AUTO_MODE_GOAL_SUBWOOFERS,
    AUTO_MODE_HYBRID_MIXED_FREQ_SOFT_DEN_HZ,
    AUTO_MODE_HYBRID_MIXED_FREQ_SOFT_MAX_HZ,
    AUTO_MODE_MAG_C_MAX_MIN_HZ,
    AUTO_MODE_MAX_AVG_SCORE_LOSS_FOR_SAFETY_OVERRIDE,
    AUTO_MODE_PHASE2_HARD_GATE_ABS_MAX_PEAK_DB,
    AUTO_MODE_PHASE2_HARD_GATE_FALLBACK_TO_RANK,
    AUTO_MODE_PHASE2_HARD_GATE_KEEP_EVENT_FRACTION,
    AUTO_MODE_PHASE2_HARD_GATE_KEEP_PEAK_FRACTION,
    AUTO_MODE_PHASE2_HARD_GATE_KEEP_RIPPLE_FRACTION,
    AUTO_MODE_PHASE2_HARD_GATE_MIN_KEEP,
    AUTO_MODE_PHASE2_PARETO_ACOUSTIC_DROP,
    AUTO_MODE_PHASE2_PARETO_BOOST_EPS,
    AUTO_MODE_PHASE2_PARETO_MODE_RIPPLE_EPS,
    AUTO_MODE_PHASE2_PARETO_PREPOST_EPS,
    AUTO_MODE_PHASE2_PARETO_RMS20_200_EPS,
    AUTO_MODE_REFINE_MODE_BOOST_GUARD_MIN_RIPPLE_GAIN_DB,
    AUTO_MODE_REFINE_TIEBREAK_ENABLE,
    AUTO_MODE_REFINE_TIEBREAK_RANK_EPS,
    AUTO_MODE_REFINE_TIEBREAK_RIPPLE_EPS,
    AUTO_MODE_REFINE_TIEBREAK_PHASE_EPS,
    AUTO_MODE_REFINE_TIEBREAK_TRACKING_EPS,
    AUTO_MODE_TARGET_BASS_FORWARD_MAX_RANK_DROP,
    AUTO_MODE_TARGET_BEST_RANK_TIE_EPS,
    MAX_SAFE_BOOST,
    _auto_goal_norm,
    _auto_safe_float,
    _m,
)

logger = logging.getLogger("DecayCore")
AUTO_MODE_PREFER_BASS_MAX_NET_BOOST_HARD_GATE_DB = 12.0

from .ranking_gates import _auto_rank_value

def _auto_rank_key(metrics: dict) -> tuple:
    return (
        -_auto_rank_value(metrics),
        _auto_target_tracking_for_pareto(metrics),
        -_auto_safe_float(metrics.get("avg_score"), 0.0),
        -_auto_bass_boost_for_rank(metrics),
        _auto_phase_risk_for_rank(metrics),
        -_auto_phase_net_for_rank(metrics),
        _auto_safe_float(metrics.get("max_net_boost_db"), 0.0),
        _auto_safe_float(metrics.get("events_severity"), 0.0),
        int(metrics.get("events_total", 0) or 0),
        _auto_safe_float(metrics.get("lr_delta_score"), 0.0),
    )


def _auto_rank_key_room_safe(metrics: dict) -> tuple:
    return (
        -_auto_rank_value(metrics),
        -_auto_bass_boost_for_rank(metrics),
        _auto_phase_risk_for_rank(metrics),
        -_auto_phase_net_for_rank(metrics),
        _auto_safe_float(metrics.get("max_net_boost_db"), 0.0),
        _auto_safe_float(metrics.get("events_severity"), 0.0),
        int(metrics.get("events_total", 0) or 0),
        _auto_safe_float(metrics.get("dsp_penalty_raw"), 0.0),
        _auto_safe_float(metrics.get("exc_penalty_raw"), 0.0),
        _auto_safe_float(metrics.get("lr_delta_score"), 0.0),
        -_auto_safe_float(metrics.get("avg_score"), 0.0),
    )


def _auto_mode_ripple_for_pareto(metrics: dict | None) -> float:
    for k in ("mode_ripple_db", "focus_ripple_db", "ripple_rms"):
        v = _m(metrics, k, float("nan"))
        if np.isfinite(v):
            return float(max(0.0, v))
    return float("inf")


def _auto_realized_rms_20_200_for_pareto(metrics: dict | None) -> float:
    v = _m(metrics, "realized_rms_20_200_db", float("nan"))
    if np.isfinite(v):
        return float(max(0.0, v))
    return float("inf")


def _auto_target_tracking_for_pareto(metrics: dict | None) -> float:
    vals = []
    for k in ("target_tracking_rms_20_200_db", "target_tracking_rms_100_500_db"):
        v = _m(metrics, k, float("nan"))
        if np.isfinite(v):
            vals.append(float(max(0.0, v)))
    if vals:
        return float(max(vals))
    return _auto_realized_rms_20_200_for_pareto(metrics)


def _auto_bass_boost_for_rank(metrics: dict | None) -> float:
    vals = []
    for key in ("bass_boost_20_200_db", "post_filter_boost_peak_db", "lf_boost_max_db"):
        v = _m(metrics, key, float("nan"))
        if np.isfinite(v):
            vals.append(float(max(0.0, v)))
    return float(max(vals)) if vals else 0.0


def _auto_prefer_bass_adjusted_avg_for_rank(metrics: dict | None) -> float:
    met = dict(metrics or {})
    avg = _auto_safe_float(met.get("avg_score"), 0.0)
    bass = _auto_bass_boost_for_rank(met)
    if not np.isfinite(bass) or float(bass) <= 0.10:
        return float(avg)
    bonus = min(24.0, 3.80 * max(0.0, float(bass)))
    net_boost = _m(met, "max_net_boost_db", float("nan"))
    if np.isfinite(net_boost):
        bonus -= 3.0 * max(0.0, float(net_boost) - 8.0)
    bonus -= 0.02 * max(0.0, _m(met, "exc_penalty_raw", 0.0))
    bonus -= 0.05 * max(0.0, _m(met, "bass_feasibility_penalty", 0.0))
    return float(avg + max(0.0, bonus))


def _auto_phase_benefit_for_rank(metrics: dict | None) -> float:
    v = _m(metrics, "phase_benefit_bonus", float("nan"))
    if np.isfinite(v):
        return float(max(0.0, v))
    return 0.0


def _auto_phase_risk_for_rank(metrics: dict | None) -> float:
    v = _m(metrics, "phase_risk_penalty", float("nan"))
    if np.isfinite(v):
        return float(max(0.0, v))
    return 0.0


def _auto_phase_net_for_rank(metrics: dict | None) -> float:
    return float(_auto_phase_benefit_for_rank(metrics) - _auto_phase_risk_for_rank(metrics))


def _auto_rank_key_low_ripple(metrics: dict) -> tuple:
    mode_ripple = _auto_mode_ripple_for_pareto(metrics)
    ripple_fallback = _auto_safe_float(metrics.get("focus_ripple_db"), float("inf"))
    target_tracking = _auto_target_tracking_for_pareto(metrics)
    return (
        -_auto_rank_value(metrics),
        mode_ripple if np.isfinite(mode_ripple) else ripple_fallback,
        _auto_phase_risk_for_rank(metrics),
        -_auto_phase_net_for_rank(metrics),
        target_tracking,
        -_auto_bass_boost_for_rank(metrics),
        _auto_safe_float(metrics.get("events_severity"), 0.0),
        _auto_safe_float(metrics.get("max_net_boost_db"), 0.0),
        _auto_safe_float(metrics.get("mixed_freq_penalty"), 0.0),
        -_auto_safe_float(metrics.get("avg_score"), 0.0),
        _auto_safe_float(metrics.get("lr_delta_score"), 0.0),
    )


def _auto_rank_key_prefer_bass(metrics: dict) -> tuple:
    return (
        -_auto_prefer_bass_adjusted_avg_for_rank(metrics),
        -_auto_bass_boost_for_rank(metrics),
        _auto_phase_risk_for_rank(metrics),
        -_auto_phase_net_for_rank(metrics),
        _auto_safe_float(metrics.get("events_severity"), 0.0),
        int(metrics.get("events_total", 0) or 0),
        _auto_target_tracking_for_pareto(metrics),
        -_auto_safe_float(metrics.get("avg_score"), 0.0),
        _auto_safe_float(metrics.get("lr_delta_score"), 0.0),
        _auto_safe_float(metrics.get("dsp_penalty_raw"), 0.0),
        _auto_safe_float(metrics.get("max_net_boost_db"), 0.0),
        _auto_safe_float(metrics.get("exc_penalty_raw"), 0.0),
        -_auto_rank_value(metrics),
    )


def _auto_rank_key_flat(metrics: dict) -> tuple:
    return _auto_rank_key_prefer_bass(metrics)


def _auto_rank_key_acoustic(metrics: dict) -> tuple:
    return (
        -_auto_safe_float(metrics.get("avg_score"), 0.0),
        _auto_target_tracking_for_pareto(metrics),
        _auto_mode_ripple_for_pareto(metrics),
        _auto_phase_risk_for_rank(metrics),
        -_auto_phase_net_for_rank(metrics),
        _auto_safe_float(metrics.get("events_severity"), 0.0),
        _auto_safe_float(metrics.get("max_net_boost_db"), 0.0),
        _auto_safe_float(metrics.get("lr_delta_score"), 0.0),
        _auto_safe_float(metrics.get("dsp_penalty_raw"), 0.0),
        _auto_safe_float(metrics.get("exc_penalty_raw"), 0.0),
        -_auto_rank_value(metrics),
    )


def _auto_rank_key_hybrid(metrics: dict) -> tuple:
    mode_ripple = _auto_mode_ripple_for_pareto(metrics)
    ripple_fallback = _auto_safe_float(metrics.get("focus_ripple_db"), float("inf"))
    target_tracking = _auto_target_tracking_for_pareto(metrics)
    return (
        -_auto_rank_value(metrics),
        mode_ripple if np.isfinite(mode_ripple) else ripple_fallback,
        _auto_safe_float(metrics.get("mixed_freq_penalty"), 0.0),
        -_auto_safe_float(metrics.get("avg_score"), 0.0),
        _auto_phase_risk_for_rank(metrics),
        -_auto_phase_net_for_rank(metrics),
        target_tracking,
        -_auto_bass_boost_for_rank(metrics),
        _auto_safe_float(metrics.get("events_severity"), 0.0),
        _auto_safe_float(metrics.get("max_net_boost_db"), 0.0),
        _auto_safe_float(metrics.get("lr_delta_score"), 0.0),
    )


def _auto_hybrid_mixed_freq_penalty(
    preset: dict | None,
    *,
    base_data: dict | None = None,
    goal: str = AUTO_MODE_GOAL_DEFAULT,
) -> float:
    if _auto_goal_norm(goal) != AUTO_MODE_GOAL_LOW_RIPPLE:
        return 0.0

    p = dict(base_data or {})
    p.update(dict(preset or {}))
    ft = str(p.get("filter_type", "") or "").strip().lower()
    if "mixed" not in ft:
        return 0.0
    if not bool(p.get("bass_first_ai", True)):
        return 0.0

    mixed_freq = _auto_safe_float(p.get("mixed_freq", float("nan")), float("nan"))
    if not np.isfinite(mixed_freq):
        return 0.0
    pen = max(
        0.0,
        (float(mixed_freq) - float(AUTO_MODE_HYBRID_MIXED_FREQ_SOFT_MAX_HZ))
        / float(max(1e-6, AUTO_MODE_HYBRID_MIXED_FREQ_SOFT_DEN_HZ)),
    )
    return float(np.clip(pen, 0.0, 2.5))


def _auto_apply_goal_tiebreak_metrics(
    metrics: dict,
    *,
    preset: dict | None,
    base_data: dict | None,
    goal: str,
) -> dict:
    out = dict(metrics or {})
    pen = float(_auto_hybrid_mixed_freq_penalty(preset, base_data=base_data, goal=goal))
    out["mixed_freq_penalty"] = pen
    if pen > 0.0:
        p = dict(base_data or {})
        p.update(dict(preset or {}))
        logger.debug(
            "mixed_freq_penalty=%.3f (mixed_freq=%.1f Hz, goal=%s)",
            pen,
            float(_auto_safe_float(p.get("mixed_freq", float("nan")), float("nan"))),
            str(goal),
        )
    return out



__all__ = ["_auto_rank_key", "_auto_rank_key_room_safe", "_auto_mode_ripple_for_pareto", "_auto_realized_rms_20_200_for_pareto", "_auto_target_tracking_for_pareto", "_auto_bass_boost_for_rank", "_auto_prefer_bass_adjusted_avg_for_rank", "_auto_phase_benefit_for_rank", "_auto_phase_risk_for_rank", "_auto_phase_net_for_rank", "_auto_rank_key_low_ripple", "_auto_rank_key_prefer_bass", "_auto_rank_key_flat", "_auto_rank_key_acoustic", "_auto_rank_key_hybrid", "_auto_hybrid_mixed_freq_penalty", "_auto_apply_goal_tiebreak_metrics"]
