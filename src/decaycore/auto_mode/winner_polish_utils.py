# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Shared utility helpers for winner-polish functions."""

from __future__ import annotations

import logging

import numpy as np

from .rank_score import official_rank_score
from .scoring_ranking import _auto_hard_gate_reasons, get_residual_peak_db, get_residual_peak_hard_gate_db
from .shared_parts import _auto_safe_float

logger = logging.getLogger("DecayCore")


def _polish_metric_delta(prev: dict | None, new: dict | None) -> dict:
    prev_m = dict(prev or {})
    new_m = dict(new or {})

    def _metric(src: dict, key: str) -> float:
        return float(_auto_safe_float(src.get(key, float("nan")), float("nan")))

    return {
        "rank": float(_metric(new_m, "rank_score") - _metric(prev_m, "rank_score"))
        if np.isfinite(_metric(new_m, "rank_score")) and np.isfinite(_metric(prev_m, "rank_score"))
        else float("nan"),
        "avg": float(_metric(new_m, "avg_score") - _metric(prev_m, "avg_score"))
        if np.isfinite(_metric(new_m, "avg_score")) and np.isfinite(_metric(prev_m, "avg_score"))
        else float("nan"),
        "residual_peak": float(_metric(new_m, "worst_residual_peak_db") - _metric(prev_m, "worst_residual_peak_db"))
        if np.isfinite(_metric(new_m, "worst_residual_peak_db")) and np.isfinite(_metric(prev_m, "worst_residual_peak_db"))
        else float("nan"),
        "boost": float(_metric(new_m, "max_net_boost_db") - _metric(prev_m, "max_net_boost_db"))
        if np.isfinite(_metric(new_m, "max_net_boost_db")) and np.isfinite(_metric(prev_m, "max_net_boost_db"))
        else float("nan"),
        "phase_risk": float(_metric(new_m, "phase_risk_penalty") - _metric(prev_m, "phase_risk_penalty"))
        if np.isfinite(_metric(new_m, "phase_risk_penalty")) and np.isfinite(_metric(prev_m, "phase_risk_penalty"))
        else float("nan"),
    }


def _polish_rank_status(metrics: dict | None) -> str:
    m = dict(metrics or {})
    display_rank = float(_auto_safe_float(official_rank_score(m), float("nan")))
    raw_rank = float(_auto_safe_float(m.get("rank_score", float("nan")), float("nan")))
    if not np.isfinite(display_rank):
        return "nan"
    text = f"{display_rank:.3f}"
    hard_gate_reasons = list(m.get("hard_gate_failures", m.get("hard_gate_reasons", [])) or [])
    if bool(m.get("hard_gate_failed", False)) and not hard_gate_reasons:
        hard_gate_reasons = ["hard_gate"]
    if hard_gate_reasons and np.isfinite(raw_rank) and abs(float(raw_rank) - float(display_rank)) > 0.05:
        text += f" (capped by {','.join(str(r) for r in hard_gate_reasons)}; raw {raw_rank:.3f})"
    return text


def _polish_rank_transition_status(prev: dict | None, new: dict | None) -> str:
    prev_m = dict(prev or {})
    new_m = dict(new or {})
    text = f"{_polish_rank_status(prev_m)} -> {_polish_rank_status(new_m)}"
    prev_reasons = list(prev_m.get("hard_gate_failures", prev_m.get("hard_gate_reasons", [])) or [])
    new_reasons = list(new_m.get("hard_gate_failures", new_m.get("hard_gate_reasons", [])) or [])
    prev_raw = float(_auto_safe_float(prev_m.get("rank_score", float("nan")), float("nan")))
    new_raw = float(_auto_safe_float(new_m.get("rank_score", float("nan")), float("nan")))
    if (
        new_reasons
        and prev_reasons == new_reasons
        and np.isfinite(prev_raw)
        and np.isfinite(new_raw)
        and abs(float(new_raw) - float(prev_raw)) > 0.005
    ):
        text += f" (raw {prev_raw:.3f} -> {new_raw:.3f})"
    return text


def _winner_polish_acceptance(
    *,
    candidate_metrics: dict | None,
    current_metrics: dict | None,
    goal: str,
    auto_is_better_refine,
    min_rank_gain: float = 0.01,
    max_avg_drop: float = 3.0,
) -> tuple[bool, str, dict]:
    cand = dict(candidate_metrics or {})
    cur = dict(current_metrics or {})
    worst_peak_hz = _auto_safe_float(cand.get("worst_residual_peak_hz", float("nan")), float("nan"))
    if np.isfinite(worst_peak_hz):
        if worst_peak_hz < 100.0:
            min_rank_gain = min(min_rank_gain, 0.006)
        elif worst_peak_hz < 200.0:
            min_rank_gain = min(min_rank_gain, 0.010)
        else:
            min_rank_gain = min(min_rank_gain, 0.015)
    better, reason = auto_is_better_refine(cand, cur, goal, return_reason=True)
    hard_gate_reasons = _auto_hard_gate_reasons(cand, goal=goal)
    rank = _auto_safe_float(cand.get("rank_score", float("nan")), float("nan"))
    if not np.isfinite(rank):
        better = False
        reason = "non_finite_rank_score"
    if hard_gate_reasons:
        better = False
        reason = "hard_gate:" + ",".join(hard_gate_reasons)
    if bool(better):
        prev_rank = _auto_safe_float(cur.get("rank_score"), float("nan"))
        new_rank = _auto_safe_float(cand.get("rank_score"), float("nan"))
        if np.isfinite(prev_rank) and np.isfinite(new_rank) and float(new_rank - prev_rank) < float(min_rank_gain):
            better = False
            reason = "rank_score_guard"
    if bool(better):
        prev_avg = _auto_safe_float(cur.get("avg_score"), float("nan"))
        new_avg = _auto_safe_float(cand.get("avg_score"), float("nan"))
        if np.isfinite(prev_avg) and np.isfinite(new_avg) and float(prev_avg - new_avg) > float(max_avg_drop):
            better = False
            reason = "avg_score_guard"
    return bool(better), str(reason), {
        "hard_gate_reasons": list(hard_gate_reasons),
        "residual_peak_db": float(get_residual_peak_db(cand)),
        "residual_peak_hard_gate_db": float(get_residual_peak_hard_gate_db(cand)),
        "hard_gate_failed": bool(hard_gate_reasons),
        "delta": _polish_metric_delta(cur, cand),
    }


def _has_target_tracking_metrics(metrics: dict | None) -> bool:
    m = dict(metrics or {})
    for key in ("target_tracking_rms_20_200_db", "target_tracking_rms_100_500_db"):
        v = _auto_safe_float(m.get(key, float("nan")), float("nan"))
        if np.isfinite(v):
            return True
    return False


def _enrich_target_tracking_metrics(
    *,
    preset: dict | None,
    metrics: dict | None,
    base_data_ref: dict | None,
    materialize_preset_result,
) -> dict:
    out = dict(metrics or {})
    if _has_target_tracking_metrics(out):
        return out
    try:
        _result, enriched, _data = materialize_preset_result(
            dict(preset or {}),
            include_response_arrays=True,
            summarize=False,
            base_data_override=base_data_ref,
        )
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
    ) as exc:
        logger.warning(
            "Automatic mode target-tracking enrichment failed: %s",
            f"{type(exc).__name__}: {exc}",
        )
        return out
    enriched = dict(enriched or {})
    if not enriched:
        return out
    return enriched
