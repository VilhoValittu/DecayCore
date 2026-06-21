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
    AUTO_MODE_GOAL_DEFAULT,
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
    AUTO_MODE_REFINE_TIEBREAK_PHASE_EPS,
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


def _phase2_abs_peak_gate_filter(
    *,
    pool: list[dict],
    abs_peak: float,
) -> tuple[list[dict], int, dict | None]:
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
        return list(abs_pool), int(abs_rejected), None
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
        int(len(pool)),
        float(abs_peak),
        float(lu_pk),
    )
    return [], int(abs_rejected), dict(lu)


def _phase2_collect_gate_metrics(pool_for_gate: list[dict]) -> tuple[list[float], list[float], list[float]]:
    ev = []
    rp = []
    pk = []
    for it in pool_for_gate:
        m = dict((it or {}).get("metrics", {}) or {})
        ev.append(_m(m, "events_severity", float("nan")))
        rp.append(_auto_ripple_metric_for_gate(m))
        pk.append(_auto_peak_metric_for_gate(m))
    return list(ev), list(rp), list(pk)


def _phase2_apply_soft_gates(
    *,
    pool_for_gate: list[dict],
    ev_thr: float,
    rp_thr: float,
    pk_thr: float,
) -> tuple[list[dict], list[dict], dict[str, int]]:
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
    return list(gated), list(gated_or), dict(reject_counts)


def _phase2_log_gate_rejects(*, abs_rejected: int, reject_counts: dict[str, int], msg_prefix: str) -> None:
    if not (abs_rejected or reject_counts):
        return
    logger.info(
        "%s rejected %d absolute-peak candidate(s); soft reject reasons=%s",
        str(msg_prefix),
        int(abs_rejected),
        dict(sorted((reject_counts or {}).items())),
    )


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
    pool_for_gate, abs_rejected, unsafe_fallback = _phase2_abs_peak_gate_filter(
        pool=pool,
        abs_peak=float(abs_peak),
    )
    if isinstance(unsafe_fallback, dict):
        return [dict(unsafe_fallback)], float("inf"), float("inf"), float("inf")

    ev, rp, pk = _phase2_collect_gate_metrics(pool_for_gate)

    ev_thr = _auto_gate_threshold(ev, float(keep_event_fraction))
    rp_thr = _auto_gate_threshold(rp, float(keep_ripple_fraction))
    pk_thr = _auto_gate_threshold(pk, float(keep_peak_fraction))
    gated, gated_or, reject_counts = _phase2_apply_soft_gates(
        pool_for_gate=pool_for_gate,
        ev_thr=float(ev_thr),
        rp_thr=float(rp_thr),
        pk_thr=float(pk_thr),
    )

    if len(gated) >= min_keep:
        _phase2_log_gate_rejects(
            abs_rejected=int(abs_rejected),
            reject_counts=reject_counts,
            msg_prefix="Phase2 hard-gate",
        )
        return gated, float(ev_thr), float(rp_thr), float(pk_thr)
    if len(gated_or) >= min_keep:
        _phase2_log_gate_rejects(
            abs_rejected=int(abs_rejected),
            reject_counts=reject_counts,
            msg_prefix="Phase2 hard-gate relaxed to OR keep;",
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
        _phase2_log_gate_rejects(
            abs_rejected=int(abs_rejected),
            reject_counts=reject_counts,
            msg_prefix="Phase2 hard-gate fell back to rank;",
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


def _pareto_candidate_metrics(candidate: dict) -> dict:
    return dict((candidate or {}).get("metrics", {}) or {})


def _prefer_higher(lhs: float, rhs: float, eps: float = 0.0) -> int:
    if float(lhs) > float(rhs) + float(eps):
        return 1
    if float(rhs) > float(lhs) + float(eps):
        return -1
    return 0


def _prefer_lower(lhs: float, rhs: float, eps: float = 0.0) -> int:
    if float(lhs) < float(rhs) - float(eps):
        return 1
    if float(rhs) < float(lhs) - float(eps):
        return -1
    return 0


def _auto_phase2_lex_compare(a: dict, b: dict) -> int:
    ma = _pareto_candidate_metrics(a)
    mb = _pareto_candidate_metrics(b)

    avg_cmp = _prefer_higher(
        _m(ma, "avg_score", float("-inf")),
        _m(mb, "avg_score", float("-inf")),
    )
    if avg_cmp != 0:
        return avg_cmp

    prepost_cmp = _prefer_lower(
        _auto_prepost_for_pareto(ma),
        _auto_prepost_for_pareto(mb),
        float(max(0.0, _auto_safe_float(AUTO_MODE_PHASE2_PARETO_PREPOST_EPS, 0.002))),
    )
    if prepost_cmp != 0:
        return prepost_cmp

    phase_cmp = _prefer_lower(
        float(_auto_phase_risk_for_rank(ma) - _auto_phase_benefit_for_rank(ma)),
        float(_auto_phase_risk_for_rank(mb) - _auto_phase_benefit_for_rank(mb)),
        float(max(0.0, _auto_safe_float(AUTO_MODE_REFINE_TIEBREAK_PHASE_EPS, 0.10))),
    )
    if phase_cmp != 0:
        return phase_cmp

    mode_cmp = _prefer_lower(
        _auto_mode_ripple_for_pareto(ma),
        _auto_mode_ripple_for_pareto(mb),
        float(max(0.0, _auto_safe_float(AUTO_MODE_PHASE2_PARETO_MODE_RIPPLE_EPS, 0.005))),
    )
    if mode_cmp != 0:
        return mode_cmp

    tracking_cmp = _prefer_lower(
        _auto_target_tracking_for_pareto(ma),
        _auto_target_tracking_for_pareto(mb),
        float(max(0.0, _auto_safe_float(AUTO_MODE_PHASE2_PARETO_RMS20_200_EPS, 0.003))),
    )
    if tracking_cmp != 0:
        return tracking_cmp

    bass_cmp = _prefer_higher(
        _auto_bass_boost_for_rank(ma),
        _auto_bass_boost_for_rank(mb),
        0.05,
    )
    if bass_cmp != 0:
        return bass_cmp

    boost_cmp = _prefer_lower(
        _m(ma, "max_net_boost_db", float("inf")),
        _m(mb, "max_net_boost_db", float("inf")),
        float(max(0.0, _auto_safe_float(AUTO_MODE_PHASE2_PARETO_BOOST_EPS, 0.02))),
    )
    if boost_cmp != 0:
        return boost_cmp

    return 1 if bool(_auto_rank_key(ma) < _auto_rank_key(mb)) else -1


def _sanitize_candidate_list(items: list[dict] | None) -> list[dict]:
    return [dict(x or {}) for x in (items or []) if isinstance(x, dict)]


def _resolve_safe_front_and_pool(front: list[dict], pool: list[dict], goal: str) -> tuple[list[dict], list[dict]]:
    front_list = _sanitize_candidate_list(front)
    pool_list = _sanitize_candidate_list(pool)
    safe_front, _front_diag = filter_hard_failed_candidates(front_list, goal=goal)
    safe_pool, _pool_diag = filter_hard_failed_candidates(pool_list, goal=goal)
    if safe_front:
        return safe_front, pool_list
    if safe_pool:
        return _auto_phase2_pareto_front(safe_pool), safe_pool
    logger.warning(
        "Pareto winner selection has no non-hard-failed candidates; falling back to least-bad hard-failed front."
    )
    return front_list, pool_list


def _build_acceptable_front_candidates(front: list[dict], pool: list[dict], drop: float) -> list[dict]:
    avg_vals = [_m(_pareto_candidate_metrics(it), "avg_score", float("nan")) for it in pool]
    avg_vals = [float(v) for v in avg_vals if np.isfinite(v)]
    best_avg = max(avg_vals) if avg_vals else float("nan")
    if not np.isfinite(best_avg):
        return []
    out: list[dict] = []
    for item in front:
        avg = _m(_pareto_candidate_metrics(item), "avg_score", float("nan"))
        if np.isfinite(avg) and float(avg) >= float(best_avg) - float(drop):
            out.append(dict(item))
    return out


def _fallback_front_choice(front: list[dict]) -> list[dict]:
    front_with_avg = []
    for item in front:
        avg = _m(_pareto_candidate_metrics(item), "avg_score", float("nan"))
        if np.isfinite(avg):
            front_with_avg.append((float(avg), dict(item)))
    if not front_with_avg:
        return list(front)
    front_with_avg = sorted(
        front_with_avg,
        key=lambda t: (
            -float(t[0]),
            _auto_rank_key(_pareto_candidate_metrics(t[1])),
        ),
    )
    return [dict(front_with_avg[0][1])]


def _pick_lexicographic_winner(candidates: list[dict]) -> dict | None:
    if not candidates:
        return None
    winner = dict(candidates[0] or {})
    for cand in candidates[1:]:
        cand_d = dict(cand or {})
        if _auto_phase2_lex_compare(cand_d, winner) > 0:
            winner = cand_d
    return dict(winner)


def _fallback_pool_winner(pool: list[dict]) -> dict | None:
    if not pool:
        return None
    pool_sorted = sorted(
        pool,
        key=lambda it: (
            -_m(_pareto_candidate_metrics(it), "avg_score", float("-inf")),
            _auto_rank_key(_pareto_candidate_metrics(it)),
        ),
    )
    return dict(pool_sorted[0])


def _auto_phase2_pick_pareto_winner(
    front: list[dict],
    pool: list[dict],
    *,
    acoustic_drop: float = AUTO_MODE_PHASE2_PARETO_ACOUSTIC_DROP,
    goal: str = AUTO_MODE_GOAL_DEFAULT,
) -> dict | None:
    front_list, pool_list = _resolve_safe_front_and_pool(front, pool, goal)
    if not front_list:
        return None

    drop = float(max(0.0, _auto_safe_float(acoustic_drop, AUTO_MODE_PHASE2_PARETO_ACOUSTIC_DROP)))
    choose_from = _build_acceptable_front_candidates(front_list, pool_list, drop)
    if not choose_from:
        choose_from = _fallback_front_choice(front_list)
    winner = _pick_lexicographic_winner(choose_from)
    if winner is not None:
        return winner
    return _fallback_pool_winner(pool_list)



__all__ = ["_auto_prepost_lr_for_pareto", "_auto_prepost_for_pareto", "_auto_ripple_metric_for_gate", "_auto_peak_metric_for_gate", "_auto_gate_threshold", "_auto_phase2_hard_gate_pool", "_pareto_dominates", "_auto_phase2_pareto_vector", "_auto_phase2_pareto_front", "_auto_phase2_pick_pareto_winner"]
