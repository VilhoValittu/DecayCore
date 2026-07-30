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

from ..shared_parts import (
    AUTO_MODE_GOAL_DEFAULT,
    AUTO_MODE_MAX_AVG_SCORE_LOSS_FOR_SAFETY_OVERRIDE,
    _auto_goal_is_flat_family,
    _auto_safe_float,
)

logger = logging.getLogger("DecayCore")
AUTO_MODE_PREFER_BASS_MAX_NET_BOOST_HARD_GATE_DB = 12.0
AUTO_MODE_PREFER_BASS_MAX_SUPPORTED_NET_BOOST_HARD_GATE_DB = 18.0
AUTO_MODE_PREFER_BASS_RESIDUAL_PEAK_HARD_GATE_MAX_DB = 12.0

def _auto_pick_metric(st: dict | None, keys: tuple[str, ...], *, abs_value: bool = False, nonneg: bool = False):
    st = st or {}
    for k in keys:
        v = _auto_safe_float(st.get(k, None), default=float("nan"))
        if not np.isfinite(v):
            continue
        if abs_value:
            v = abs(v)
        if nonneg and v < 0.0:
            continue
        return float(v)
    return None


def _auto_rank_value(metrics: dict | None, *, default: float = float("-inf")) -> float:
    value = _auto_safe_float(dict(metrics or {}).get("rank_score", float("nan")), float("nan"))
    return float(value) if np.isfinite(value) else float(default)


def _auto_candidate_metrics(candidate_or_metrics: dict | None) -> dict:
    src = dict(candidate_or_metrics or {})
    nested = src.get("metrics")
    if isinstance(nested, dict):
        return dict(nested or {})
    best_nested = src.get("best_metrics")
    if isinstance(best_nested, dict):
        return dict(best_nested or {})
    return dict(src or {})


def get_residual_peak_db(candidate_or_metrics: dict | None) -> float:
    metrics = _auto_candidate_metrics(candidate_or_metrics)
    for key in ("worst_residual_peak_db", "worst_residual_peak_raw_db", "top3_residual_peak_mean_db"):
        value = _auto_safe_float(metrics.get(key, float("nan")), float("nan"))
        if np.isfinite(value):
            return float(max(0.0, value))
    return float("nan")


def get_residual_peak_hard_gate_db(candidate_or_metrics: dict | None) -> float:
    metrics = _auto_candidate_metrics(candidate_or_metrics)
    value = _auto_safe_float(metrics.get("residual_peak_hard_gate_db", float("nan")), float("nan"))
    return float(value) if np.isfinite(value) else float("nan")


def _auto_bass_boost_support_db(metrics: dict | None) -> float:
    m = dict(metrics or {})
    vals = []
    for key in ("bass_boost_20_200_db", "post_filter_boost_peak_db", "lf_boost_max_db"):
        value = _auto_safe_float(m.get(key, float("nan")), float("nan"))
        if np.isfinite(value):
            vals.append(float(max(0.0, value)))
    return float(max(vals)) if vals else 0.0


_ZONE_GATE_MODAL_DB = 7.5       # <100 Hz: modal region, loosened
_ZONE_GATE_TRANSITION_DB = 6.0  # 100–250 Hz: baseline
_ZONE_GATE_MID_DB = 5.0         # >250 Hz: correctable, tightened


def _residual_peak_freq_zone_gate_db(gate: float, peak_hz: float) -> float:
    """Return the absolute residual peak hard gate for the given frequency zone.

    Modal region (<100 Hz): 7.5 dB — room modes are physically hard to
    fully correct, so candidates with modest residual still compete (scoring
    penalises them separately).

    Transition (100–250 Hz): linearly blended 7.5 → 5.0 dB.

    Mid-range (>250 Hz): 5.0 dB — these peaks are correctable, so
    any candidate leaving >5 dB here is rejected outright.
    """
    if not (np.isfinite(peak_hz) and np.isfinite(gate) and peak_hz > 0.0):
        return float(gate)
    if peak_hz < 100.0:
        return float(min(_ZONE_GATE_MODAL_DB, float(AUTO_MODE_PREFER_BASS_RESIDUAL_PEAK_HARD_GATE_MAX_DB)))
    if peak_hz > 250.0:
        return float(_ZONE_GATE_MID_DB)
    # Linear blend in transition zone 100–250 Hz
    t = (peak_hz - 100.0) / 150.0
    return float(_ZONE_GATE_MODAL_DB + t * (_ZONE_GATE_MID_DB - _ZONE_GATE_MODAL_DB))


def get_residual_peak_hard_gate_effective_db(
    candidate_or_metrics: dict | None,
    *,
    goal: str = AUTO_MODE_GOAL_DEFAULT,
) -> float:
    metrics = _auto_candidate_metrics(candidate_or_metrics)
    finalized_gate = _auto_safe_float(
        metrics.get("residual_peak_hard_gate_effective_db", float("nan")),
        float("nan"),
    )
    if np.isfinite(finalized_gate):
        return float(finalized_gate)
    gate = get_residual_peak_hard_gate_db(metrics)
    if not np.isfinite(gate):
        return float("nan")

    peak_hz = _auto_safe_float(metrics.get("worst_residual_peak_hz", float("nan")), float("nan"))
    gate = _residual_peak_freq_zone_gate_db(gate, peak_hz)

    if not _auto_goal_is_flat_family(goal):
        return float(gate)
    bass_boost = _auto_bass_boost_support_db(metrics)
    if float(bass_boost) < 3.0:
        return float(gate)
    return float(
        min(
            float(AUTO_MODE_PREFER_BASS_RESIDUAL_PEAK_HARD_GATE_MAX_DB),
            float(gate) + float(bass_boost),
        )
    )


def _get_residual_peak_gate_value_and_source(candidate_or_metrics: dict | None) -> tuple[float, str]:
    metrics = _auto_candidate_metrics(candidate_or_metrics)
    explicit_source = str(metrics.get("residual_peak_gate_source", "") or "").strip().lower()
    explicit_value = _auto_safe_float(metrics.get("residual_peak_gate_value_db", float("nan")), float("nan"))
    if explicit_source in {"raw_db", "severity"} and np.isfinite(explicit_value):
        return float(max(0.0, explicit_value)), str(explicit_source)

    raw = _auto_safe_float(metrics.get("worst_residual_peak_raw_db", float("nan")), float("nan"))
    if np.isfinite(raw):
        return float(max(0.0, raw)), "raw_db"

    # Legacy metrics often only carried worst_residual_peak_db. Keep those
    # hard-gate compatible while newer metrics can mark severity explicitly.
    value = get_residual_peak_db(metrics)
    if np.isfinite(value):
        return float(max(0.0, value)), "raw_db"
    return float("nan"), ""


def _append_explicit_hard_gate_reasons(reasons: list[str], metrics: dict) -> None:
    for reason in list(metrics.get("hard_gate_failures", metrics.get("hard_gate_reasons", [])) or []):
        reason_txt = str(reason or "").strip()
        if not reason_txt:
            continue
        if reason_txt in {"residual_peak_hard_gate", "residual_peak_severity_gate"}:
            continue
        reasons.append(reason_txt)


def _append_residual_peak_gate_reason(reasons: list[str], metrics: dict, *, goal: str) -> None:
    rp_worst, rp_gate_source = _get_residual_peak_gate_value_and_source(metrics)
    rp_gate = get_residual_peak_hard_gate_effective_db(metrics, goal=goal)
    if np.isfinite(rp_worst) and np.isfinite(rp_gate) and float(rp_worst) > float(rp_gate):
        reasons.append("residual_peak_hard_gate" if rp_gate_source == "raw_db" else "residual_peak_severity_gate")


def _append_bass_integration_gate_reason(reasons: list[str], metrics: dict) -> None:
    bass_gate_explicit = bool(metrics.get("bass_integration_hard_gate_failed", False))
    bass_gate_from_metrics = bool(
        bool(metrics.get("bass_integration_enable", False))
        and (
            str(metrics.get("bass_feasibility_class", "") or "").strip().lower() == "infeasible"
            or bool(metrics.get("bass_direct_dac_reject_reasons", []) or [])
        )
    )
    if bass_gate_explicit or bass_gate_from_metrics:
        reasons.append("bass_integration_infeasible_hard_gate")


def _append_flat_goal_safety_reasons(reasons: list[str], metrics: dict, *, st_l: dict | None, st_r: dict | None) -> None:
    net_boost = _auto_safe_float(metrics.get("max_net_boost_db"), 0.0)
    bass_boost = _auto_safe_float(
        metrics.get("bass_boost_20_200_db", metrics.get("lf_boost_max_db", float("nan"))),
        float("nan"),
    )
    boost_gate = float(AUTO_MODE_PREFER_BASS_MAX_NET_BOOST_HARD_GATE_DB)
    if np.isfinite(bass_boost) and float(bass_boost) >= 3.0:
        boost_gate = min(
            float(AUTO_MODE_PREFER_BASS_MAX_SUPPORTED_NET_BOOST_HARD_GATE_DB),
            max(boost_gate, float(bass_boost) + 10.0),
        )
    if float(net_boost) > float(boost_gate):
        reasons.append("excessive_net_boost")

    ratio_keys = (
        "ir_pre_post_ratio",
        "ir_pre_energy_guard_after_ratio",
        "ir_pre_energy_guard_before_ratio",
    )
    gd_keys = (
        "gd_grad_limiter_after_max_ms_per_oct",
        "gd_grad_limiter_before_max_ms_per_oct",
        "gd_limiter_max_grad_ms_per_oct",
        "gd_grad_limiter_max_grad_ms_per_oct",
        "gd_limiter_max_grad_after_ms_per_oct",
        "gd_grad_limiter_max_grad_after_ms_per_oct",
        "gd_limiter_max_grad_before_ms_per_oct",
        "gd_grad_limiter_max_grad_before_ms_per_oct",
    )
    for channel, st in (("l", dict(st_l or {})), ("r", dict(st_r or {}))):
        pre_suspect = bool(st.get("pre_energy_metric_suspect", False))
        if not pre_suspect:
            ratio = _auto_pick_metric(st, ratio_keys, nonneg=True)
            if ratio is not None and float(ratio) > 0.05:
                reasons.append(f"unsafe_prepost_{channel}")
        gd_grad = _auto_pick_metric(st, gd_keys, abs_value=True, nonneg=True)
        if gd_grad is not None and float(gd_grad) > 45.0:
            reasons.append(f"unsafe_gd_gradient_{channel}")


def _auto_hard_gate_reasons(
    metrics: dict | None,
    st_l: dict | None = None,
    st_r: dict | None = None,
    goal: str = AUTO_MODE_GOAL_DEFAULT,
) -> list[str]:
    m = dict(metrics or {})
    reasons: list[str] = []
    _append_explicit_hard_gate_reasons(reasons, m)

    rank = _auto_safe_float(m.get("rank_score", float("nan")), float("nan"))
    if not np.isfinite(rank):
        reasons.append("non_finite_rank_score")

    _append_residual_peak_gate_reason(reasons, m, goal=goal)

    if bool(m.get("stereo_policy_gate_failed", False)):
        reasons.append("stereo_policy_gate_failed")

    _append_bass_integration_gate_reason(reasons, m)

    if _auto_goal_is_flat_family(goal):
        _append_flat_goal_safety_reasons(reasons, m, st_l=st_l, st_r=st_r)

    return list(dict.fromkeys(reasons))


def is_hard_failed(candidate_or_metrics: dict | None, *, goal: str = AUTO_MODE_GOAL_DEFAULT) -> bool:
    return bool(_auto_hard_gate_reasons(_auto_candidate_metrics(candidate_or_metrics), goal=goal))


def _auto_hard_gate_diagnostic(candidate_or_metrics: dict | None, *, goal: str = AUTO_MODE_GOAL_DEFAULT) -> dict:
    metrics = _auto_candidate_metrics(candidate_or_metrics)
    reasons = _auto_hard_gate_reasons(metrics, goal=goal)
    return {
        "residual_peak_db": float(get_residual_peak_db(metrics)),
        "residual_peak_hard_gate_db": float(get_residual_peak_hard_gate_effective_db(metrics, goal=goal)),
        "bass_integration_hard_gate_failed": bool(
            "bass_integration_infeasible_hard_gate" in reasons
        ),
        "bass_feasibility_class": str(metrics.get("bass_feasibility_class", "") or ""),
        "bass_feasibility_reason": str(
            metrics.get(
                "bass_integration_hard_gate_reason",
                metrics.get("bass_feasibility_reason", ""),
            )
            or ""
        ),
        "hard_gate_failed": bool(reasons),
        "hard_gate_reasons": list(reasons),
        "rank_score": float(_auto_rank_value(metrics, default=float("nan"))),
        "avg_score": float(_auto_safe_float(metrics.get("avg_score", float("nan")), float("nan"))),
    }


def filter_hard_failed_candidates(
    candidates: list[dict] | tuple[dict, ...] | None,
    *,
    goal: str = AUTO_MODE_GOAL_DEFAULT,
) -> tuple[list[dict], list[dict]]:
    safe: list[dict] = []
    diagnostics: list[dict] = []
    for cand in list(candidates or []):
        if not isinstance(cand, dict):
            continue
        item = dict(cand or {})
        diag = _auto_hard_gate_diagnostic(item, goal=goal)
        diagnostics.append(dict(diag))
        if bool(diag.get("hard_gate_failed", False)):
            logger.info(
                "Auto-mode hard-gate rejected candidate: residual_peak_db=%.3f, residual_peak_hard_gate_db=%.3f, hard_gate_failed=%s, reason=%s",
                float(_auto_safe_float(diag.get("residual_peak_db"), float("nan"))),
                float(_auto_safe_float(diag.get("residual_peak_hard_gate_db"), float("nan"))),
                bool(diag.get("hard_gate_failed", False)),
                ",".join(str(x) for x in list(diag.get("hard_gate_reasons", []) or [])) or "unknown",
            )
            continue
        safe.append(item)
    return safe, diagnostics


def _rank_key_for_safe_select(metrics: dict) -> tuple:
    from .ranking_keys import _auto_rank_key

    return _auto_rank_key(metrics)


def select_best_safe_candidate(
    candidates: list[dict] | tuple[dict, ...] | None,
    *,
    goal: str = AUTO_MODE_GOAL_DEFAULT,
) -> dict | None:
    pool = [dict(x or {}) for x in list(candidates or []) if isinstance(x, dict)]
    if not pool:
        return None
    safe, diagnostics = filter_hard_failed_candidates(pool, goal=goal)
    choose_from = safe
    if not choose_from:
        logger.warning(
            "Auto-mode selection has no non-hard-failed candidates; fallback selected least-bad hard-failed candidate (n=%d).",
            int(len(pool)),
        )
        choose_from = pool
    return dict(
        sorted(
            choose_from,
            key=lambda it: _rank_key_for_safe_select(dict(it.get("metrics", _auto_candidate_metrics(it)) or {})),
        )[0]
    )


def maybe_override_hard_failed_winner(
    current_winner: dict | None,
    candidates: list[dict] | tuple[dict, ...] | None,
    cfg=None,
    *,
    goal: str = AUTO_MODE_GOAL_DEFAULT,
) -> tuple[dict | None, dict]:
    if not isinstance(current_winner, dict):
        return current_winner, {"applied": False, "reason": "no_current_winner"}
    cur = dict(current_winner or {})
    cur_metrics = _auto_candidate_metrics(cur)
    cur_diag = _auto_hard_gate_diagnostic(cur, goal=goal)
    if not bool(cur_diag.get("hard_gate_failed", False)):
        return cur, {"applied": False, "reason": "winner_not_hard_failed", "current": dict(cur_diag)}

    pool = [dict(x or {}) for x in list(candidates or []) if isinstance(x, dict)]
    safe, diagnostics = filter_hard_failed_candidates(pool, goal=goal)
    meta = {
        "applied": False,
        "reason": "no_safe_candidate",
        "current": dict(cur_diag),
        "candidate_count": int(len(pool)),
        "safe_candidate_count": int(len(safe)),
        "diagnostics": list(diagnostics),
    }
    if not safe:
        logger.warning(
            "Auto-mode residual peak safety override could not find a non-hard-failed candidate; keeping least-bad fallback."
        )
        return cur, meta

    replacement = select_best_safe_candidate(safe, goal=goal)
    if not isinstance(replacement, dict):
        return cur, meta
    rep_metrics = _auto_candidate_metrics(replacement)
    cur_avg = _auto_safe_float(cur_metrics.get("avg_score", float("nan")), float("nan"))
    rep_avg = _auto_safe_float(rep_metrics.get("avg_score", float("nan")), float("nan"))
    avg_loss = float(cur_avg - rep_avg) if np.isfinite(cur_avg) and np.isfinite(rep_avg) else 0.0
    max_loss = float(
        max(
            0.0,
            _auto_safe_float(
                getattr(cfg, "max_avg_score_loss_for_safety_override", AUTO_MODE_MAX_AVG_SCORE_LOSS_FOR_SAFETY_OVERRIDE),
                AUTO_MODE_MAX_AVG_SCORE_LOSS_FOR_SAFETY_OVERRIDE,
            ),
        )
    )
    meta.update(
        {
            "reason": "avg_score_loss_guard",
            "previous_winner_score": float(_auto_rank_value(cur_metrics, default=float("nan"))),
            "replacement_winner_score": float(_auto_rank_value(rep_metrics, default=float("nan"))),
            "previous_avg_score": float(cur_avg),
            "replacement_avg_score": float(rep_avg),
            "average_score_loss": float(avg_loss),
            "max_avg_score_loss_for_safety_override": float(max_loss),
            "replacement": _auto_hard_gate_diagnostic(replacement, goal=goal),
        }
    )
    if float(avg_loss) > float(max_loss):
        logger.warning(
            "Auto-mode residual peak safety override skipped: avg loss %.3f exceeds limit %.3f (rank %.3f -> %.3f).",
            float(avg_loss),
            float(max_loss),
            float(meta["previous_winner_score"]),
            float(meta["replacement_winner_score"]),
        )
        return cur, meta

    meta["applied"] = True
    meta["reason"] = "residual_peak_hard_gate_override"
    logger.warning(
        "Auto-mode residual peak safety override applied: rank %.3f -> %.3f, avg %.3f -> %.3f, avg_loss=%.3f, residual_peak_db=%.3f, gate=%.3f.",
        float(meta["previous_winner_score"]),
        float(meta["replacement_winner_score"]),
        float(meta["previous_avg_score"]),
        float(meta["replacement_avg_score"]),
        float(avg_loss),
        float(_auto_safe_float(cur_diag.get("residual_peak_db"), float("nan"))),
        float(_auto_safe_float(cur_diag.get("residual_peak_hard_gate_db"), float("nan"))),
    )
    return dict(replacement), dict(meta)



__all__ = ["_auto_pick_metric", "_auto_rank_value", "_auto_candidate_metrics", "get_residual_peak_db", "get_residual_peak_hard_gate_db", "_residual_peak_freq_zone_gate_db", "get_residual_peak_hard_gate_effective_db", "_get_residual_peak_gate_value_and_source", "_auto_hard_gate_reasons", "is_hard_failed", "_auto_hard_gate_diagnostic", "filter_hard_failed_candidates", "select_best_safe_candidate", "maybe_override_hard_failed_winner"]
