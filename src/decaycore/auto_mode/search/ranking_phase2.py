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

from .ranking_gates import _auto_hard_gate_reasons, _auto_rank_value, filter_hard_failed_candidates
from .ranking_keys import (_auto_bass_boost_for_rank, _auto_mode_ripple_for_pareto, _auto_phase_benefit_for_rank, _auto_phase_risk_for_rank, _auto_rank_key, _auto_target_tracking_for_pareto)

def _auto_prepost_lr_for_pareto(metrics: dict | None) -> tuple[float, float, float]:
    m = dict(metrics or {})
    l = _m(m, "ir_pre_post_energy_ratio_l", float("nan"))
    r = _m(m, "ir_pre_post_energy_ratio_r", float("nan"))
    if not np.isfinite(l):
        l = _m(m, "ir_pre_post_ratio_l", float("nan"))
    if not np.isfinite(r):
        r = _m(m, "ir_pre_post_ratio_r", float("nan"))
    if not np.isfinite(l):
        l = _m(dict(m.get("dsp_dbg_l", {}) or {}), "ir_pre_post_ratio", float("nan"))
    if not np.isfinite(r):
        r = _m(dict(m.get("dsp_dbg_r", {}) or {}), "ir_pre_post_ratio", float("nan"))
    if not np.isfinite(l):
        global_v = _m(m, "ir_pre_post_energy_ratio_max", float("nan"))
        if np.isfinite(global_v):
            l = float(global_v)
    if not np.isfinite(r):
        global_v = _m(m, "ir_pre_post_energy_ratio_max", float("nan"))
        if np.isfinite(global_v):
            r = float(global_v)

    vals = [float(v) for v in (l, r) if np.isfinite(v)]
    mx = float(max(vals)) if vals else float("inf")
    return (
        float(l) if np.isfinite(l) else float("nan"),
        float(r) if np.isfinite(r) else float("nan"),
        float(mx),
    )


def _auto_prepost_for_pareto(metrics: dict | None) -> float:
    _, _, mx = _auto_prepost_lr_for_pareto(metrics)
    if np.isfinite(mx):
        return float(max(0.0, mx))
    return float("inf")


def _auto_ripple_metric_for_gate(metrics: dict | None) -> float:
    tracking = _auto_target_tracking_for_pareto(metrics)
    ripple = float("inf")
    for k in ("focus_ripple_db", "mode_ripple_db", "ripple_rms"):
        v = _m(metrics, k, float("nan"))
        if np.isfinite(v):
            ripple = float(max(0.0, v))
            break
    if np.isfinite(tracking) and np.isfinite(ripple):
        return float(max(float(tracking), float(ripple)))
    if np.isfinite(tracking):
        return float(tracking)
    return float(ripple)


def _auto_peak_metric_for_gate(metrics: dict | None) -> float:
    for k in ("worst_residual_peak_raw_db", "worst_residual_peak_db", "top3_residual_peak_mean_db"):
        v = _m(metrics, k, float("nan"))
        if np.isfinite(v):
            return float(max(0.0, v))
    return float("inf")


def _auto_gate_threshold(values: list[float], keep_fraction: float) -> float:
    vals = [float(v) for v in (values or []) if np.isfinite(v)]
    if not vals:
        return float("inf")
    kf = float(np.clip(_auto_safe_float(keep_fraction, 1.0), 0.05, 1.0))
    vals = sorted(vals)
    idx = int(np.floor((len(vals) - 1) * kf))
    idx = int(np.clip(idx, 0, len(vals) - 1))
    return float(vals[idx])


def _auto_phase2_hard_gate_pool(
    pool: list[dict],
    *,
    min_keep: int = AUTO_MODE_PHASE2_HARD_GATE_MIN_KEEP,
    keep_event_fraction: float = AUTO_MODE_PHASE2_HARD_GATE_KEEP_EVENT_FRACTION,
    keep_ripple_fraction: float = AUTO_MODE_PHASE2_HARD_GATE_KEEP_RIPPLE_FRACTION,
    keep_peak_fraction: float = AUTO_MODE_PHASE2_HARD_GATE_KEEP_PEAK_FRACTION,
    abs_max_peak_db: float = AUTO_MODE_PHASE2_HARD_GATE_ABS_MAX_PEAK_DB,
    fallback_to_rank: bool = AUTO_MODE_PHASE2_HARD_GATE_FALLBACK_TO_RANK,
) -> tuple[list[dict], float, float, float]:
    if not isinstance(pool, list) or not pool:
        return [], float("inf"), float("inf"), float("inf")
    n_in = int(len(pool))
    min_keep = int(max(1, min_keep))
    if n_in <= (min_keep + 2):
        return [dict(x or {}) for x in pool], float("inf"), float("inf"), float("inf")

    abs_peak = float(max(0.0, _auto_safe_float(abs_max_peak_db, AUTO_MODE_PHASE2_HARD_GATE_ABS_MAX_PEAK_DB)))
    abs_pool = []
    abs_rejected = 0
    for it in pool:
        m = dict((it or {}).get("metrics", {}) or {})
        pk_i = _auto_peak_metric_for_gate(m)
        if np.isfinite(pk_i) and float(pk_i) > float(abs_peak):
            abs_rejected += 1
            continue
        abs_pool.append(dict(it or {}))
    if abs_pool:
        pool_for_gate = abs_pool
    else:
        least_unsafe = sorted(
            pool,
            key=lambda it: (
                _auto_peak_metric_for_gate(dict((it or {}).get("metrics", {}) or {})),
                _auto_rank_key(dict((it or {}).get("metrics", {}) or {})),
            ),
        )[0]
        lu = dict(least_unsafe or {})
        lu_pk = _auto_peak_metric_for_gate(dict(lu.get("metrics", {}) or {}))
        lu["unsafe_fallback"] = True
        lu["unsafe_fallback_reason"] = "all_candidates_failed_absolute_residual_peak_gate"
        lu["unsafe_fallback_abs_max_peak_db"] = float(abs_peak)
        lu["unsafe_fallback_residual_peak_db"] = float(lu_pk)
        logger.warning(
            "Phase2 hard-gate: ALL %d candidates failed absolute residual peak gate (threshold=%.2f dB); "
            "returning least-unsafe fallback candidate with peak=%.2f dB.",
            n_in, float(abs_peak), float(lu_pk),
        )
        return [lu], float("inf"), float("inf"), float("inf")

    ev = []
    rp = []
    pk = []
    for it in pool_for_gate:
        m = dict((it or {}).get("metrics", {}) or {})
        ev.append(_m(m, "events_severity", float("nan")))
        rp.append(_auto_ripple_metric_for_gate(m))
        pk.append(_auto_peak_metric_for_gate(m))

    ev_thr = _auto_gate_threshold(ev, float(keep_event_fraction))
    rp_thr = _auto_gate_threshold(rp, float(keep_ripple_fraction))
    pk_thr = _auto_gate_threshold(pk, float(keep_peak_fraction))
    gated = []
    gated_or = []
    reject_counts: dict[str, int] = {}
    for it in pool_for_gate:
        m = dict((it or {}).get("metrics", {}) or {})
        ev_i = _m(m, "events_severity", float("inf"))
        rp_i = _auto_ripple_metric_for_gate(m)
        pk_i = _auto_peak_metric_for_gate(m)
        ok_ev = bool(np.isfinite(ev_i) and float(ev_i) <= float(ev_thr))
        ok_rp = bool(np.isfinite(rp_i) and float(rp_i) <= float(rp_thr))
        ok_pk = bool(np.isfinite(pk_i) and float(pk_i) <= float(pk_thr))
        if ok_ev and ok_rp and ok_pk:
            gated.append(dict(it or {}))
        if (ok_ev and ok_pk) or (ok_rp and ok_pk):
            gated_or.append(dict(it or {}))
        if not (ok_ev and ok_rp and ok_pk):
            reasons = []
            if not ok_ev:
                reasons.append("phase2_event_gate")
            if not ok_rp:
                reasons.append("phase2_ripple_gate")
            if not ok_pk:
                reasons.append("phase2_residual_peak_gate")
            reasons.extend(_auto_hard_gate_reasons(m))
            for reason in dict.fromkeys(reasons):
                reject_counts[str(reason)] = int(reject_counts.get(str(reason), 0) or 0) + 1

    if len(gated) >= min_keep:
        if abs_rejected or reject_counts:
            logger.info(
                "Phase2 hard-gate rejected %d absolute-peak candidate(s); soft reject reasons=%s",
                int(abs_rejected),
                dict(sorted(reject_counts.items())),
            )
        return gated, float(ev_thr), float(rp_thr), float(pk_thr)
    if len(gated_or) >= min_keep:
        if abs_rejected or reject_counts:
            logger.info(
                "Phase2 hard-gate relaxed to OR keep; rejected %d absolute-peak candidate(s); soft reject reasons=%s",
                int(abs_rejected),
                dict(sorted(reject_counts.items())),
            )
        return gated_or, float(ev_thr), float(rp_thr), float(pk_thr)
    if bool(fallback_to_rank):
        kept = sorted(
            [dict(x or {}) for x in pool_for_gate],
            key=lambda it: (
                -_auto_rank_value(dict(it.get("metrics", {}) or {})),
                _auto_rank_key(dict(it.get("metrics", {}) or {})),
            ),
        )[:min_keep]
        if abs_rejected or reject_counts:
            logger.info(
                "Phase2 hard-gate fell back to rank; rejected %d absolute-peak candidate(s); soft reject reasons=%s",
                int(abs_rejected),
                dict(sorted(reject_counts.items())),
            )
        return kept, float(ev_thr), float(rp_thr), float(pk_thr)
    kept = gated_or or gated or [dict(x or {}) for x in pool_for_gate]
    return kept, float(ev_thr), float(rp_thr), float(pk_thr)


def _pareto_dominates(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    if len(a) != len(b):
        return False
    le_all = True
    lt_any = False
    for ai, bi in zip(a, b):
        if ai > bi:
            le_all = False
            break
        if ai < bi:
            lt_any = True
    return bool(le_all and lt_any)


def _auto_phase2_pareto_vector(metrics: dict | None) -> tuple[float, float, float, float, float, float, float]:
    avg = _m(metrics, "avg_score", float("nan"))
    neg_avg = -float(avg) if np.isfinite(avg) else float("inf")
    mode_ripple = _auto_mode_ripple_for_pareto(metrics)
    target_tracking = _auto_target_tracking_for_pareto(metrics)
    bass_boost = _auto_bass_boost_for_rank(metrics)
    net_boost = _m(metrics, "max_net_boost_db", float("nan"))
    net_boost = float(net_boost) if np.isfinite(net_boost) else float("inf")
    prepost = _auto_prepost_for_pareto(metrics)
    phase_value = float(_auto_phase_risk_for_rank(metrics) - _auto_phase_benefit_for_rank(metrics))
    return (
        float(neg_avg),
        float(prepost),
        float(phase_value),
        float(mode_ripple),
        float(target_tracking),
        float(-bass_boost),
        float(net_boost),
    )


def _auto_phase2_pareto_front(pool: list[dict]) -> list[dict]:
    front = []
    if not isinstance(pool, list) or not pool:
        return front
    vectors = [_auto_phase2_pareto_vector(dict(it.get("metrics", {}) or {})) for it in pool]
    try:
        vec = np.asarray(vectors, dtype=float)
        if vec.ndim == 2 and vec.shape[0] == len(pool) and vec.shape[1] == 7:
            le_all = np.all(vec[:, None, :] <= vec[None, :, :], axis=2)
            lt_any = np.any(vec[:, None, :] < vec[None, :, :], axis=2)
            dominated = np.any(le_all & lt_any, axis=0)
            return [dict(pool[int(i)] or {}) for i in np.flatnonzero(~dominated)]
    except (TypeError, ValueError, FloatingPointError, MemoryError):
        pass
    for i, cand in enumerate(pool):
        dominated = False
        for j, other in enumerate(pool):
            if i == j:
                continue
            if _pareto_dominates(vectors[j], vectors[i]):
                dominated = True
                break
        if not dominated:
            front.append(dict(cand or {}))
    return front


def _auto_phase2_pick_pareto_winner(
    front: list[dict],
    pool: list[dict],
    *,
    acoustic_drop: float = AUTO_MODE_PHASE2_PARETO_ACOUSTIC_DROP,
    goal: str = AUTO_MODE_GOAL_DEFAULT,
) -> dict | None:
    def _lex_better(a: dict, b: dict) -> bool:
        ma = dict(a.get("metrics", {}) or {})
        mb = dict(b.get("metrics", {}) or {})
        avg_a = _m(ma, "avg_score", float("-inf"))
        avg_b = _m(mb, "avg_score", float("-inf"))
        if float(avg_a) > float(avg_b):
            return True
        if float(avg_a) < float(avg_b):
            return False

        prepost_eps = float(max(0.0, _auto_safe_float(AUTO_MODE_PHASE2_PARETO_PREPOST_EPS, 0.002)))
        prepost_a = _auto_prepost_for_pareto(ma)
        prepost_b = _auto_prepost_for_pareto(mb)
        if float(prepost_a) < float(prepost_b) - float(prepost_eps):
            return True
        if float(prepost_b) < float(prepost_a) - float(prepost_eps):
            return False

        phase_eps = float(max(0.0, _auto_safe_float(AUTO_MODE_REFINE_TIEBREAK_PHASE_EPS, 0.10)))
        phase_a = float(_auto_phase_risk_for_rank(ma) - _auto_phase_benefit_for_rank(ma))
        phase_b = float(_auto_phase_risk_for_rank(mb) - _auto_phase_benefit_for_rank(mb))
        if float(phase_a) < float(phase_b) - float(phase_eps):
            return True
        if float(phase_b) < float(phase_a) - float(phase_eps):
            return False

        mode_eps = float(max(0.0, _auto_safe_float(AUTO_MODE_PHASE2_PARETO_MODE_RIPPLE_EPS, 0.005)))
        mode_a = _auto_mode_ripple_for_pareto(ma)
        mode_b = _auto_mode_ripple_for_pareto(mb)
        if float(mode_a) < float(mode_b) - float(mode_eps):
            return True
        if float(mode_b) < float(mode_a) - float(mode_eps):
            return False

        rms_eps = float(max(0.0, _auto_safe_float(AUTO_MODE_PHASE2_PARETO_RMS20_200_EPS, 0.003)))
        rms_a = _auto_target_tracking_for_pareto(ma)
        rms_b = _auto_target_tracking_for_pareto(mb)
        if float(rms_a) < float(rms_b) - float(rms_eps):
            return True
        if float(rms_b) < float(rms_a) - float(rms_eps):
            return False

        bass_eps = 0.05
        bass_a = _auto_bass_boost_for_rank(ma)
        bass_b = _auto_bass_boost_for_rank(mb)
        if float(bass_a) > float(bass_b) + float(bass_eps):
            return True
        if float(bass_b) > float(bass_a) + float(bass_eps):
            return False

        boost_eps = float(max(0.0, _auto_safe_float(AUTO_MODE_PHASE2_PARETO_BOOST_EPS, 0.02)))
        boost_a = _m(ma, "max_net_boost_db", float("inf"))
        boost_b = _m(mb, "max_net_boost_db", float("inf"))
        if float(boost_a) < float(boost_b) - float(boost_eps):
            return True
        if float(boost_b) < float(boost_a) - float(boost_eps):
            return False

        return bool(_auto_rank_key(ma) < _auto_rank_key(mb))

    front_list = [dict(x or {}) for x in (front or []) if isinstance(x, dict)]
    pool_list = [dict(x or {}) for x in (pool or []) if isinstance(x, dict)]
    safe_front, _front_diag = filter_hard_failed_candidates(front_list, goal=goal)
    safe_pool, _pool_diag = filter_hard_failed_candidates(pool_list, goal=goal)
    if safe_front:
        front_list = safe_front
    elif safe_pool:
        front_list = _auto_phase2_pareto_front(safe_pool)
        pool_list = safe_pool
    else:
        logger.warning(
            "Pareto winner selection has no non-hard-failed candidates; falling back to least-bad hard-failed front."
        )
    if not front_list:
        return None

    avg_vals = [
        _m(dict(it.get("metrics", {}) or {}), "avg_score", float("nan"))
        for it in pool_list
    ]
    avg_vals = [float(v) for v in avg_vals if np.isfinite(v)]
    best_avg = max(avg_vals) if avg_vals else float("nan")
    drop = float(max(0.0, _auto_safe_float(acoustic_drop, AUTO_MODE_PHASE2_PARETO_ACOUSTIC_DROP)))

    acceptable: list[dict] = []
    if np.isfinite(best_avg):
        for it in front_list:
            avg = _m(dict(it.get("metrics", {}) or {}), "avg_score", float("nan"))
            if np.isfinite(avg) and float(avg) >= float(best_avg) - float(drop):
                acceptable.append(dict(it))
    choose_from = acceptable

    if not choose_from:
        front_with_avg = []
        for it in front_list:
            avg = _m(dict(it.get("metrics", {}) or {}), "avg_score", float("nan"))
            if np.isfinite(avg):
                front_with_avg.append((float(avg), dict(it)))
        if front_with_avg:
            front_with_avg = sorted(
                front_with_avg,
                key=lambda t: (
                    -float(t[0]),
                    _auto_rank_key(dict((t[1] or {}).get("metrics", {}) or {})),
                ),
            )
            choose_from = [dict(front_with_avg[0][1])]
        else:
            choose_from = list(front_list)

    if choose_from:
        winner = dict(choose_from[0])
        for cand in choose_from[1:]:
            cand_d = dict(cand or {})
            if _lex_better(cand_d, winner):
                winner = cand_d
        return dict(winner)

    if pool_list:
        pool_sorted = sorted(
            pool_list,
            key=lambda it: (
                -_m(dict(it.get("metrics", {}) or {}), "avg_score", float("-inf")),
                _auto_rank_key(dict(it.get("metrics", {}) or {})),
            ),
        )
        return dict(pool_sorted[0])
    return None



__all__ = ["_auto_prepost_lr_for_pareto", "_auto_prepost_for_pareto", "_auto_ripple_metric_for_gate", "_auto_peak_metric_for_gate", "_auto_gate_threshold", "_auto_phase2_hard_gate_pool", "_pareto_dominates", "_auto_phase2_pareto_vector", "_auto_phase2_pareto_front", "_auto_phase2_pick_pareto_winner"]
