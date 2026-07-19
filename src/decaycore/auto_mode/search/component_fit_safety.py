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
import math

import numpy as np

_logger = logging.getLogger(__name__)

from .. import shared
from ..runtime_context import (
    _auto_collect_reflections,
    _auto_event_penalty_weighted,
    _auto_event_severity,
)

from .metrics_common import _ai_score_with_fallback, _calc_ai_summary_from_stats_auto, _get_auto_scoring_range
from .metric_penalties import (
    _auto_dsp_quality_penalty, _auto_exc_penalty_bins_from_dbg,
    _auto_exc_zero_penalty_freq_hz_from_stats, _auto_excursion_penalty,
    _auto_harmonic_boost_penalty, _auto_harmonic_local_boost_penalty,
    _auto_phase_quality, _auto_rt60_policy_penalty, _auto_tdc_decay_tradeoff,
)

def score_acoustic_fit(result, l_st, r_st, *, base_data) -> dict:
    scoring_range = _get_auto_scoring_range(l_st, r_st, base_data)
    if scoring_range is not None:
        _logger.debug(
            "auto scoring range: %.1f–%.1f Hz (mag_c_min=%.1f, mag_c_max=%.1f)",
            scoring_range[0],
            scoring_range[1],
            shared._auto_safe_float(l_st.get("mag_c_min", float("nan")), float("nan")),
            shared._auto_safe_float(l_st.get("mag_c_max", float("nan")), float("nan")),
        )
    l_ai = _calc_ai_summary_from_stats_auto(l_st, scoring_range)
    r_ai = _calc_ai_summary_from_stats_auto(r_st, scoring_range)

    l_score = _ai_score_with_fallback(l_st, l_ai, scoring_range=scoring_range)
    r_score = _ai_score_with_fallback(r_st, r_ai, scoring_range=scoring_range)
    avg_score = (l_score + r_score) / 2.0
    lr_delta = abs(l_score - r_score)

    net_boost_max = max(
        shared._auto_safe_float(l_st.get("net_boost_peak_db", 0.0), 0.0),
        shared._auto_safe_float(r_st.get("net_boost_peak_db", 0.0), 0.0),
    )
    l_refs = _auto_collect_reflections(l_st)
    r_refs = _auto_collect_reflections(r_st)
    events_total = int(len(l_refs) + len(r_refs))
    events_severity_l = _auto_event_severity(l_refs)
    events_severity_r = _auto_event_severity(r_refs)
    events_severity_raw = float(events_severity_l + events_severity_r)
    events_severity = float(math.log1p(max(0.0, events_severity_raw) / 6.0))
    return {
        "avg_score": avg_score,
        "l_score": l_score,
        "r_score": r_score,
        "lr_delta": lr_delta,
        "net_boost_max": net_boost_max,
        "l_refs": l_refs,
        "r_refs": r_refs,
        "events_total": events_total,
        "events_severity_l": events_severity_l,
        "events_severity_r": events_severity_r,
        "events_severity_raw": events_severity_raw,
        "events_severity": events_severity,
        "scoring_range": scoring_range,
    }


def score_safety_limits(l_st, r_st, *, auto_exc_freq_hz, base_data, net_boost_max, lr_delta, l_refs, r_refs) -> dict:
    dsp_pen_l, bass_prering_raw_l, dsp_dbg_l = _auto_dsp_quality_penalty(l_st)
    dsp_pen_r, bass_prering_raw_r, dsp_dbg_r = _auto_dsp_quality_penalty(r_st)
    dsp_penalty_raw = 0.5 * (float(dsp_pen_l) + float(dsp_pen_r))
    exc_pen_l, exc_dbg_l = _auto_excursion_penalty(l_st)
    exc_pen_r, exc_dbg_r = _auto_excursion_penalty(r_st)
    exc_penalty_raw_total = 0.5 * (float(exc_pen_l) + float(exc_pen_r))
    exc_penalty_bins_raw = 0.5 * (
        float(_auto_exc_penalty_bins_from_dbg(exc_dbg_l))
        + float(_auto_exc_penalty_bins_from_dbg(exc_dbg_r))
    )
    exc_penalty_raw = float(exc_penalty_raw_total)
    exc_penalty_waived = bool(np.isfinite(shared._auto_safe_float(auto_exc_freq_hz, float("nan"))))
    auto_exc_zero_l = _auto_exc_zero_penalty_freq_hz_from_stats(l_st)
    auto_exc_zero_r = _auto_exc_zero_penalty_freq_hz_from_stats(r_st)
    auto_exc_zero_vals = [float(v) for v in (auto_exc_zero_l, auto_exc_zero_r) if np.isfinite(v)]
    auto_exc_zero_penalty_hz = float(min(auto_exc_zero_vals)) if auto_exc_zero_vals else float("nan")
    auto_exc_hz_now = shared._auto_safe_float(auto_exc_freq_hz, float("nan"))
    exc_penalty_bins_waived = False
    if (
        bool(exc_penalty_waived)
        and np.isfinite(auto_exc_zero_penalty_hz)
        and np.isfinite(auto_exc_hz_now)
        and (float(auto_exc_hz_now) + 1e-6) >= float(auto_exc_zero_penalty_hz)
    ):
        exc_penalty_raw = max(0.0, float(exc_penalty_raw_total) - float(exc_penalty_bins_raw))
        exc_penalty_bins_waived = bool(float(exc_penalty_bins_raw) > 1e-9)
    exc_penalty = float(exc_penalty_raw) * (0.35 if exc_penalty_waived else 1.0)
    optuna_exc_penalty_disabled = bool(
        str(shared._auto_optimizer_backend(base_data, default_optuna_enabled=False) or "").strip().lower() == "optuna"
    )
    if bool(optuna_exc_penalty_disabled):
        exc_pen_l = 0.0
        exc_pen_r = 0.0
        exc_penalty_raw_total = 0.0
        exc_penalty_bins_raw = 0.0
        exc_penalty_raw = 0.0
        exc_penalty_bins_waived = False
        exc_penalty_waived = False
        exc_penalty = 0.0

    if shared._auto_goal_is_flat_family(shared._auto_goal(base_data)):
        boost_pen = min(8.0, 0.15 * max(0.0, float(net_boost_max) - 5.0))
    else:
        boost_pen = min(8.0, 0.40 * max(0.0, float(net_boost_max) - 6.0))
    bass_prering_avg_raw = 0.5 * (float(bass_prering_raw_l) + float(bass_prering_raw_r))
    dsp_penalty_raw_excl = max(0.0, float(dsp_penalty_raw) - float(bass_prering_avg_raw))
    dsp_penalty = min(8.0, 0.35 * float(dsp_penalty_raw_excl))
    bass_prering_penalty = min(4.0, 0.35 * float(bass_prering_avg_raw))
    all_events = list(l_refs) + list(r_refs)
    event_pen_raw = _auto_event_penalty_weighted(
        all_events,
        base_per_event=float(shared._auto_safe_float(shared.AUTO_MODE_EVENT_PEN_BASE_PER_EVENT, 0.5)),
        dt_weight=float(shared._auto_safe_float(shared.AUTO_MODE_EVENT_PEN_DT_WEIGHT, 0.02)),
        power=float(shared._auto_safe_float(shared.AUTO_MODE_EVENT_PEN_DT_POWER, 2.0)),
        dt_ref_ms=float(shared._auto_safe_float(shared.AUTO_MODE_EVENT_PEN_DT_REF_MS, 100.0)),
    )
    event_pen_conf_scale = 1.0
    if bool(shared.AUTO_MODE_EVENT_PEN_CONF_GATE_ENABLE):
        conf_vals = []
        for st in (l_st, r_st):
            c = shared._auto_safe_float(
                st.get("cmp_avg_confidence", st.get("avg_confidence", float("nan"))),
                float("nan"),
            )
            if not np.isfinite(c):
                continue
            c01 = float(c / 100.0) if float(c) > 1.5 else float(c)
            c01 = float(np.clip(c01, 0.0, 1.0))
            conf_vals.append(float(c01))
        if conf_vals:
            conf_mean = float(np.mean(np.asarray(conf_vals, dtype=float)))
            min_scale = float(np.clip(shared._auto_safe_float(shared.AUTO_MODE_EVENT_PEN_CONF_GATE_MIN_SCALE, 0.45), 0.0, 1.0))
            full_conf = float(np.clip(shared._auto_safe_float(shared.AUTO_MODE_EVENT_PEN_CONF_GATE_FULL_CONF, 0.85), 1e-6, 1.0))
            conf_norm = float(np.clip(conf_mean / full_conf, 0.0, 1.0))
            event_pen_conf_scale = float(min_scale + (1.0 - min_scale) * conf_norm)
    event_pen_raw *= float(event_pen_conf_scale)
    event_pen = min(8.0, max(0.0, event_pen_raw))
    lr_pen = min(4.0, 0.03 * lr_delta)
    exc_penalty = min(12.0, float(exc_penalty))
    return {
        "dsp_pen_l": dsp_pen_l,
        "dsp_pen_r": dsp_pen_r,
        "dsp_dbg_l": dsp_dbg_l,
        "dsp_dbg_r": dsp_dbg_r,
        "dsp_penalty_raw": dsp_penalty_raw,
        "dsp_penalty": dsp_penalty,
        "bass_prering_penalty": bass_prering_penalty,
        "exc_pen_l": exc_pen_l,
        "exc_pen_r": exc_pen_r,
        "exc_dbg_l": exc_dbg_l,
        "exc_dbg_r": exc_dbg_r,
        "exc_penalty_raw": exc_penalty_raw,
        "exc_penalty_raw_total": exc_penalty_raw_total,
        "exc_penalty_bins_raw": exc_penalty_bins_raw,
        "exc_penalty_bins_waived": exc_penalty_bins_waived,
        "exc_penalty_waived": exc_penalty_waived,
        "exc_penalty": exc_penalty,
        "auto_exc_zero_penalty_hz": auto_exc_zero_penalty_hz,
        "boost_pen": boost_pen,
        "event_pen": event_pen,
        "lr_pen": lr_pen,
    }


def score_phase_quality(l_st, r_st, *, base_data) -> dict:
    filter_key = shared._auto_filter_cache_key(base_data)
    phase_limit_used_hz = shared._auto_safe_float((base_data or {}).get("phase_limit", float("nan")), float("nan"))
    if shared._auto_is_phase_search_filter(filter_key):
        phase_limit_used_hz = float(shared._auto_phase_limit_clip(phase_limit_used_hz, default=shared.AUTO_MODE_PHASE_LIMIT_DEFAULT_HZ))
    phase_limit_penalty = float(
        shared._auto_phase_limit_prior_penalty(phase_limit_used_hz, filter_key=filter_key)
    )
    phase_benefit_l, phase_risk_l, phase_dbg_l = _auto_phase_quality(l_st)
    phase_benefit_r, phase_risk_r, phase_dbg_r = _auto_phase_quality(r_st)
    phase_lr_consistency_penalty = 0.0
    for weight, key in (
        (0.80, "useful_lf"),
        (0.60, "useful_xo"),
        (0.30, "useful_audible"),
    ):
        lv = shared._auto_safe_float(phase_dbg_l.get(key, float("nan")), float("nan"))
        rv = shared._auto_safe_float(phase_dbg_r.get(key, float("nan")), float("nan"))
        if np.isfinite(lv) and np.isfinite(rv):
            phase_lr_consistency_penalty += float(weight) * abs(float(lv) - float(rv))
    phase_benefit_bonus = float(
        np.clip(
            0.5 * (float(phase_benefit_l) + float(phase_benefit_r))
            - 0.55 * float(phase_lr_consistency_penalty),
            0.0,
            5.0,
        )
    )
    phase_risk_penalty = float(
        np.clip(
            0.5 * (float(phase_risk_l) + float(phase_risk_r))
            + 1.10 * float(phase_lr_consistency_penalty),
            0.0,
            6.0,
        )
    )
    phase_net_score = float(phase_benefit_bonus - phase_risk_penalty)
    return {
        "phase_limit_used_hz": phase_limit_used_hz,
        "phase_limit_penalty": phase_limit_penalty,
        "phase_benefit_bonus": phase_benefit_bonus,
        "phase_risk_penalty": phase_risk_penalty,
        "phase_net_score": phase_net_score,
        "phase_lr_consistency_penalty": phase_lr_consistency_penalty,
        "phase_dbg_l": phase_dbg_l,
        "phase_dbg_r": phase_dbg_r,
    }


def score_temporal_decay(l_st, r_st, *, base_data) -> dict:
    _thd_pen_l = _auto_harmonic_boost_penalty(
        l_st,
        (base_data or {}).get("harmonic_freq_hz_l"),
        (base_data or {}).get("harmonic_magnitudes_db_l"),
        fundamental_freq_hz=(base_data or {}).get("f_l"),
        fundamental_mag_db=(base_data or {}).get("m_l"),
    )
    _thd_pen_r = _auto_harmonic_boost_penalty(
        r_st,
        (base_data or {}).get("harmonic_freq_hz_r"),
        (base_data or {}).get("harmonic_magnitudes_db_r"),
        fundamental_freq_hz=(base_data or {}).get("f_r"),
        fundamental_mag_db=(base_data or {}).get("m_r"),
    )
    thd_boost_penalty = 0.5 * (float(_thd_pen_l) + float(_thd_pen_r))
    _rt60_policy_pen = _auto_rt60_policy_penalty(base_data, l_st, r_st)
    _harmonic_local_pen = _auto_harmonic_local_boost_penalty(base_data, l_st, r_st)
    decay_penalty, decay_dbg = _auto_tdc_decay_tradeoff(base_data, l_st, r_st)
    return {
        "thd_boost_penalty": thd_boost_penalty,
        "rt60_policy_pen": _rt60_policy_pen,
        "harmonic_local_pen": _harmonic_local_pen,
        "decay_penalty": decay_penalty,
        "decay_dbg": decay_dbg,
    }



__all__ = ["score_acoustic_fit", "score_safety_limits", "score_phase_quality", "score_temporal_decay"]
