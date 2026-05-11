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

def _auto_harmonic_boost_penalty(
    st: dict | None,
    harmonic_freq_hz: np.ndarray | None,
    harmonic_magnitudes_db: dict | None,
    *,
    lo_hz: float = 20.0,
    hi_hz: float = 800.0,
    thd_threshold_db: float = -35.0,
    thd_range_db: float = 15.0,
    scale: float = 0.15,
    fundamental_freq_hz: np.ndarray | None = None,
    fundamental_mag_db: np.ndarray | None = None,
) -> float:
    """Penaltti boostille korkean harmonisen säröön alueilla.

    Jos ``fundamental_freq_hz`` ja ``fundamental_mag_db`` annetaan, harmoniat
    normalisoidaan suhteessa fundamentaaliin (dBr) ennen kynnysvertailua, ja
    käytetään suhteellista kynnystä -30 dBr. Muuten verrataan absoluuttiseen
    dBFS-kynnykseen ``thd_threshold_db``.
    Palauttaa 0.0 jos harmonidata puuttuu tai mikään ehto ei täyty.
    """
    if harmonic_freq_hz is None or harmonic_magnitudes_db is None:
        return 0.0
    try:
        st = dict(st or {})
        net_boost = float(max(0.0, shared._auto_safe_float(st.get("net_boost_peak_db", 0.0), 0.0)))
        if net_boost <= 0.0:
            return 0.0

        freq = np.asarray(harmonic_freq_hz, dtype=float).reshape(-1)
        band_mask = (freq >= float(lo_hz)) & (freq <= float(hi_hz)) & np.isfinite(freq)
        if int(np.count_nonzero(band_mask)) < 4:
            return 0.0

        use_relative = (
            fundamental_freq_hz is not None
            and fundamental_mag_db is not None
        )
        if use_relative:
            fund_f = np.asarray(fundamental_freq_hz, dtype=float).reshape(-1)
            fund_m = np.asarray(fundamental_mag_db, dtype=float).reshape(-1)
            valid = np.isfinite(fund_f) & np.isfinite(fund_m)
            if valid.sum() >= 4:
                fund_at_harm = np.interp(freq, fund_f[valid], fund_m[valid])
            else:
                use_relative = False

        h_arrays = []
        for order in (2, 3):
            arr = (harmonic_magnitudes_db or {}).get(order)
            if arr is None:
                continue
            a = np.asarray(arr, dtype=float).reshape(-1)
            if a.size != freq.size:
                continue
            band_vals = a[band_mask]
            if use_relative:
                band_vals = band_vals - fund_at_harm[band_mask]
            h_arrays.append(band_vals)
        if not h_arrays:
            return 0.0

        effective_threshold = -30.0 if use_relative else float(thd_threshold_db)

        thd_proxy = np.maximum.reduce(h_arrays)
        thd_proxy = thd_proxy[np.isfinite(thd_proxy)]
        if thd_proxy.size == 0:
            return 0.0

        severity_vals = np.clip(
            (thd_proxy - effective_threshold) / max(float(thd_range_db), 1e-6),
            0.0,
            1.0,
        )
        severity = float(np.mean(severity_vals))
        if severity <= 0.0:
            return 0.0

        penalty = float(net_boost) * float(severity) * float(scale)
        return float(max(0.0, penalty))
    except Exception:
        return 0.0


def _auto_rt60_policy_penalty(
    base_data: dict | None,
    l_st: dict | None,
    r_st: dict | None,
    *,
    rt60_high_threshold_s: float = 0.65,
    rt60_extreme_threshold_s: float = 1.0,
    scale: float = 0.05,
) -> float:
    """Small penalty when high RT60 meets high net boost (reverberation risk).

    Conservative secondary signal - max contribution ~0.05 * net_boost.
    """
    try:
        bd = dict(base_data or {})
        rt60_l = shared._auto_safe_float(
            (bd.get("rt60_summary_l") or {}).get("rt60_val", float("nan")),
            float("nan"),
        )
        rt60_r = shared._auto_safe_float(
            (bd.get("rt60_summary_r") or {}).get("rt60_val", float("nan")),
            float("nan"),
        )
        rt60_vals = [v for v in (rt60_l, rt60_r) if np.isfinite(v) and v > 0.0]
        if not rt60_vals:
            return 0.0
        rt60_max = float(max(rt60_vals))
        excess = max(0.0, rt60_max - float(rt60_high_threshold_s))
        if excess <= 0.0:
            return 0.0
        net_boost_l = float(max(0.0, shared._auto_safe_float((l_st or {}).get("net_boost_peak_db", 0.0), 0.0)))
        net_boost_r = float(max(0.0, shared._auto_safe_float((r_st or {}).get("net_boost_peak_db", 0.0), 0.0)))
        net_boost = max(net_boost_l, net_boost_r)
        if net_boost <= 0.0:
            return 0.0
        severity = float(np.clip(
            excess / max(float(rt60_extreme_threshold_s) - float(rt60_high_threshold_s), 0.1),
            0.0,
            1.0,
        ))
        return float(max(0.0, float(net_boost) * float(severity) * float(scale)))
    except Exception:
        return 0.0


AUTO_TDC_DECAY_SCORING_VERSION = 2


def _auto_rt60_band_pairs(base_data: dict | None) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    bd = dict(base_data or {})
    for key in ("measured_rt60_bands_l", "measured_rt60_bands_r"):
        bands = bd.get(key)
        if not isinstance(bands, dict):
            continue
        for freq, rt60 in bands.items():
            f = shared._auto_safe_float(freq, float("nan"))
            v = shared._auto_safe_float(rt60, float("nan"))
            if np.isfinite(f) and np.isfinite(v) and f > 0.0 and v > 0.0:
                pairs.append((float(f), float(v)))
    return pairs


def _auto_tdc_decay_tradeoff(
    base_data: dict | None,
    l_st: dict | None,
    r_st: dict | None,
) -> tuple[float, dict]:
    """Band-limited AUTO TDC decay tradeoff penalty.

    This uses cheap metadata already available during AUTO scoring. It does not
    estimate new impulse responses inside the optimization loop.
    """
    bd = dict(base_data or {})
    penalty_cap = max(
        0.0,
        shared._auto_safe_float(
            bd.get("auto_mode_tdc_decay_penalty_cap", shared.AUTO_MODE_TDC_DECAY_PENALTY_CAP),
            shared.AUTO_MODE_TDC_DECAY_PENALTY_CAP,
        ),
    )
    penalty_weight = max(
        0.0,
        shared._auto_safe_float(
            bd.get("auto_mode_tdc_decay_penalty_weight", shared.AUTO_MODE_TDC_DECAY_PENALTY_WEIGHT),
            shared.AUTO_MODE_TDC_DECAY_PENALTY_WEIGHT,
        ),
    )
    extreme_peak_db = max(
        0.0,
        shared._auto_safe_float(
            bd.get("auto_mode_tdc_extreme_peak_reduction_db", shared.AUTO_MODE_TDC_EXTREME_PEAK_REDUCTION_DB),
            shared.AUTO_MODE_TDC_EXTREME_PEAK_REDUCTION_DB,
        ),
    )
    low_need_threshold = float(
        np.clip(
            shared._auto_safe_float(
                bd.get("auto_mode_tdc_low_need_threshold", shared.AUTO_MODE_TDC_LOW_NEED_THRESHOLD),
                shared.AUTO_MODE_TDC_LOW_NEED_THRESHOLD,
            ),
            0.0,
            1.0,
        )
    )
    target_low_s = max(
        0.01,
        shared._auto_safe_float(
            bd.get("auto_mode_tdc_rt60_target_low_s", shared.AUTO_MODE_TDC_RT60_TARGET_LOW_S),
            shared.AUTO_MODE_TDC_RT60_TARGET_LOW_S,
        ),
    )
    target_upper_s = max(
        0.01,
        shared._auto_safe_float(
            bd.get("auto_mode_tdc_rt60_target_upper_s", shared.AUTO_MODE_TDC_RT60_TARGET_UPPER_S),
            shared.AUTO_MODE_TDC_RT60_TARGET_UPPER_S,
        ),
    )
    low_max_hz = max(
        20.0,
        shared._auto_safe_float(
            bd.get("auto_mode_tdc_rt60_low_max_hz", shared.AUTO_MODE_TDC_RT60_LOW_MAX_HZ),
            shared.AUTO_MODE_TDC_RT60_LOW_MAX_HZ,
        ),
    )
    eval_max_hz = max(
        20.0,
        shared._auto_safe_float(
            bd.get("auto_mode_tdc_rt60_eval_max_hz", shared.AUTO_MODE_TDC_RT60_EVAL_MAX_HZ),
            shared.AUTO_MODE_TDC_RT60_EVAL_MAX_HZ,
        ),
    )
    if not bool(bd.get("enable_tdc", True)):
        return 0.0, {
            "decay_metadata_available": False,
            "tdc_decision": "tdc_disabled",
            "tdc_action_hint": "tdc_disabled",
            "tdc_weak_penalty": 0.0,
            "tdc_overdamping_penalty": 0.0,
            "tdc_overreach_penalty": 0.0,
            "tdc_decay_penalty_total": 0.0,
            "tdc_decay_penalty": 0.0,
            "tdc_extreme_overreach": False,
            "tdc_decay_scoring_version": int(AUTO_TDC_DECAY_SCORING_VERSION),
        }

    strength_pct = shared._auto_safe_float(bd.get("tdc_strength", float("nan")), float("nan"))
    if not np.isfinite(strength_pct):
        strength_pct = 50.0
    strength = float(np.clip(float(strength_pct) / 100.0, 0.0, 1.0))
    pairs = _auto_rt60_band_pairs(bd)

    weighted_excess = 0.0
    weight_sum = 0.0
    strongest_freq = float("nan")
    strongest_excess = 0.0
    for freq, rt60 in pairs:
        if freq < 20.0 or freq > eval_max_hz:
            continue
        if freq <= low_max_hz:
            weight = 1.0
            target = target_low_s
        else:
            weight = 0.40
            target = target_upper_s
        excess = max(0.0, float(rt60) - float(target))
        weighted_excess += float(weight) * float(excess)
        weight_sum += float(weight)
        if excess > strongest_excess:
            strongest_excess = float(excess)
            strongest_freq = float(freq)

    modal_excess_s = float(weighted_excess / weight_sum) if weight_sum > 0.0 else 0.0
    metadata_available = bool(weight_sum > 0.0)

    stats = [dict(l_st or {}), dict(r_st or {})]
    peak_vals = [
        shared._auto_safe_float(st.get("tdc_peak_reduction_db", float("nan")), float("nan"))
        for st in stats
    ]
    peak_vals = [float(v) for v in peak_vals if np.isfinite(v) and v > 0.0]
    tdc_peak_db = float(max(peak_vals)) if peak_vals else 0.0
    peak_hz_vals = [
        shared._auto_safe_float(st.get("tdc_peak_reduction_hz", float("nan")), float("nan"))
        for st in stats
    ]
    peak_hz_vals = [float(v) for v in peak_hz_vals if np.isfinite(v) and v > 0.0]
    tdc_peak_hz = float(peak_hz_vals[0]) if peak_hz_vals else float("nan")
    area_vals = [
        shared._auto_safe_float(st.get("tdc_reduction_area_db_hz", float("nan")), float("nan"))
        for st in stats
    ]
    area_vals = [float(v) for v in area_vals if np.isfinite(v) and v > 0.0]
    tdc_area = float(max(area_vals)) if area_vals else 0.0
    event_vals = [
        shared._auto_safe_float(st.get("tdc_events_used", 0.0), 0.0)
        for st in stats
    ]
    tdc_events = float(max([float(v) for v in event_vals if np.isfinite(v)] or [0.0]))
    band_los = [
        shared._auto_safe_float(st.get("tdc_reduction_band_low_hz", float("nan")), float("nan"))
        for st in stats
    ]
    band_his = [
        shared._auto_safe_float(st.get("tdc_reduction_band_high_hz", float("nan")), float("nan"))
        for st in stats
    ]
    band_los = [float(v) for v in band_los if np.isfinite(v) and v > 0.0]
    band_his = [float(v) for v in band_his if np.isfinite(v) and v > 0.0]
    tdc_band_lo = float(min(band_los)) if band_los else float("nan")
    tdc_band_hi = float(max(band_his)) if band_his else float("nan")

    need = float(np.clip(modal_excess_s / 0.35, 0.0, 1.0))
    if tdc_events > 0.0 and not metadata_available:
        need = max(need, float(np.clip(tdc_peak_db / 5.0, 0.0, 0.65)))
    if not metadata_available and tdc_events <= 0.0:
        return 0.0, {
            "decay_metadata_available": False,
            "tdc_strength": float(strength),
            "tdc_decay_need": 0.0,
            "tdc_weak_penalty": 0.0,
            "tdc_overdamping_penalty": 0.0,
            "tdc_overreach_penalty": 0.0,
            "tdc_decay_penalty_total": 0.0,
            "tdc_decay_penalty": 0.0,
            "tdc_decision": "neutral_no_decay_metadata",
            "tdc_action_hint": "neutral_no_metadata",
            "tdc_extreme_overreach": False,
            "tdc_decay_scoring_version": int(AUTO_TDC_DECAY_SCORING_VERSION),
        }

    optimum = float(np.clip(0.24 + 0.36 * need, 0.20, 0.75))
    weak_pen = 6.0 * need * max(0.0, optimum - strength) ** 2 / max(optimum * optimum, 1e-6)
    dry_need = max(0.0, strength - optimum)
    dry_pen = (1.0 - 0.55 * need) * 5.0 * dry_need * dry_need / max((1.0 - optimum) ** 2, 1e-6)
    overreach_pen = 0.0
    if tdc_peak_db > 0.0:
        expected_peak = 1.0 + 6.5 * max(need, 0.15)
        overreach_pen += 0.18 * max(0.0, tdc_peak_db - expected_peak)
    overreach_pen += 0.0015 * max(0.0, tdc_area - (120.0 + 380.0 * need))
    extreme_overreach = bool(extreme_peak_db > 0.0 and tdc_peak_db >= extreme_peak_db and need <= low_need_threshold)
    if extreme_overreach:
        overreach_pen += 4.0 + 0.25 * max(0.0, tdc_peak_db - extreme_peak_db)
    raw_penalty = float(max(0.0, weak_pen + dry_pen + overreach_pen))
    penalty = float(np.clip(raw_penalty * penalty_weight, 0.0, penalty_cap))

    improvement = 0.0
    if need > 0.0:
        improvement = float(np.clip((min(strength, optimum) / max(optimum, 1e-6)) * need, 0.0, 1.0))
    keep_tol = 0.08
    if need <= 0.12:
        decision = "reduced_by_optimizer" if strength <= 0.35 else "possible_overdamping_risk"
    elif abs(strength - optimum) <= keep_tol:
        decision = "tdc_helped"
    elif strength < optimum:
        decision = "weak_tdc_left_decay"
    else:
        decision = "strong_tdc_overdamping_risk"
    if need <= 0.12:
        action_hint = "keep_tdc" if strength <= 0.35 else "decrease_tdc"
    elif strength < (optimum - keep_tol):
        action_hint = "increase_tdc"
    elif strength > (optimum + keep_tol):
        action_hint = "decrease_tdc"
    else:
        action_hint = "keep_tdc"

    return penalty, {
        "decay_metadata_available": bool(metadata_available),
        "tdc_strength": float(strength),
        "tdc_decay_need": float(need),
        "tdc_weak_penalty": float(weak_pen * penalty_weight),
        "tdc_overdamping_penalty": float(dry_pen * penalty_weight),
        "tdc_overreach_penalty": float(overreach_pen * penalty_weight),
        "tdc_decay_penalty_total": float(penalty),
        "tdc_decay_penalty": float(penalty),
        "tdc_decay_optimum_strength": float(optimum),
        "modal_decay_excess_s": float(modal_excess_s),
        "strongest_modal_decay_hz": float(strongest_freq) if np.isfinite(strongest_freq) else float("nan"),
        "strongest_modal_decay_excess_s": float(strongest_excess),
        "modal_decay_improvement_estimate": float(improvement),
        "tdc_peak_reduction_db": float(tdc_peak_db),
        "tdc_peak_reduction_hz": float(tdc_peak_hz) if np.isfinite(tdc_peak_hz) else float("nan"),
        "tdc_reduction_area_db_hz": float(tdc_area),
        "tdc_reduction_band_low_hz": float(tdc_band_lo) if np.isfinite(tdc_band_lo) else float("nan"),
        "tdc_reduction_band_high_hz": float(tdc_band_hi) if np.isfinite(tdc_band_hi) else float("nan"),
        "tdc_decision": str(decision),
        "tdc_action_hint": str(action_hint),
        "tdc_extreme_overreach": bool(extreme_overreach),
        "tdc_decay_penalty_cap": float(penalty_cap),
        "tdc_decay_penalty_weight": float(penalty_weight),
        "tdc_decay_scoring_version": int(AUTO_TDC_DECAY_SCORING_VERSION),
    }


def _auto_harmonic_local_boost_penalty(
    base_data: dict | None,
    l_st: dict | None,
    r_st: dict | None,
    *,
    risk_threshold: float = 0.5,
    scale: float = 0.04,
) -> float:
    """Small secondary penalty from harmonic risk summary peak.

    Distinct from _auto_harmonic_boost_penalty which does per-bin analysis.
    Uses the compact risk summary aggregated from repeat/multi-position analysis.
    """
    try:
        bd = dict(base_data or {})
        risk_l = shared._auto_safe_float(
            (bd.get("harmonic_risk_summary_l") or {}).get("peak_risk", float("nan")),
            float("nan"),
        )
        risk_r = shared._auto_safe_float(
            (bd.get("harmonic_risk_summary_r") or {}).get("peak_risk", float("nan")),
            float("nan"),
        )
        risk_vals = [v for v in (risk_l, risk_r) if np.isfinite(v) and v > 0.0]
        if not risk_vals:
            return 0.0
        risk_max = float(max(risk_vals))
        excess = max(0.0, risk_max - float(risk_threshold))
        if excess <= 0.0:
            return 0.0
        net_boost_l = float(max(0.0, shared._auto_safe_float((l_st or {}).get("net_boost_peak_db", 0.0), 0.0)))
        net_boost_r = float(max(0.0, shared._auto_safe_float((r_st or {}).get("net_boost_peak_db", 0.0), 0.0)))
        net_boost = max(net_boost_l, net_boost_r)
        if net_boost <= 0.0:
            return 0.0
        severity = float(np.clip(excess / max(1.0 - float(risk_threshold), 0.1), 0.0, 1.0))
        return float(max(0.0, float(net_boost) * float(severity) * float(scale)))
    except Exception:
        return 0.0



__all__ = ["CORRECTION_SHARPNESS_SCORING_VERSION", "DIP_FILL_RISK_SCORING_VERSION", "CHANNEL_OVERFIT_SCORING_VERSION", "AUTO_TDC_DECAY_SCORING_VERSION", "_auto_dsp_quality_penalty", "_auto_exc_penalty_bins_from_dbg", "_auto_exc_zero_penalty_freq_hz_from_stats", "_auto_excursion_penalty", "_auto_bass_integration_penalty", "_auto_focus_ripple_from_stats", "_auto_target_tracking_metrics_from_stats", "_auto_merge_target_tracking_metrics", "_auto_target_tracking_penalty", "_auto_bass_under_target_metrics_from_stats", "_auto_merge_bass_under_target_metrics", "_auto_bass_boost_metrics_from_stats", "_auto_merge_bass_boost_metrics", "_auto_bass_preference_bonus", "_auto_correction_sharpness_metrics_from_stats", "_auto_merge_correction_sharpness_metrics", "_auto_dip_fill_risk_metrics_from_stats", "_auto_merge_dip_fill_risk_metrics", "_auto_channel_overfit_metrics_from_stats", "_auto_harmonic_boost_penalty", "_auto_rt60_policy_penalty", "_auto_tdc_decay_tradeoff", "_auto_harmonic_local_boost_penalty"]

__all__ = ["AUTO_TDC_DECAY_SCORING_VERSION", "_auto_harmonic_boost_penalty", "_auto_rt60_policy_penalty", "_auto_tdc_decay_tradeoff", "_auto_harmonic_local_boost_penalty"]
