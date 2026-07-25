# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Excess-phase-strength winner-polish for auto-mode finalize."""

from __future__ import annotations

import logging

import numpy as np

from .auto_mode_profile import profiled_section
from .shared_parts import _auto_safe_float
from .winner_polish_utils import _polish_rank_status, _winner_polish_acceptance

logger = logging.getLogger("DecayCore")

_RECOVERABLE_EXCESS_POLISH_EXCEPTIONS = (
    AttributeError,
    TypeError,
    ValueError,
    KeyError,
    IndexError,
    RuntimeError,
    OSError,
    ImportError,
    ModuleNotFoundError,
)


def _build_excess_phase_candidate_values(initial_value: float, step: float, max_delta: float) -> list[float]:
    step_eff = max(0.01, float(_auto_safe_float(step, 0.05)))
    max_delta_eff = max(0.0, float(_auto_safe_float(max_delta, 0.15)))
    max_steps = int(max(0, np.floor(float(max_delta_eff) / float(step_eff) + 1e-9)))
    tested_values: set[float] = {float(initial_value)}
    candidate_values: list[float] = []
    for step_idx in range(1, max_steps + 1):
        for sign in (-1.0, 1.0):
            cand = round(float(np.clip(float(initial_value) + sign * float(step_idx) * float(step_eff), 0.0, 1.0)), 4)
            if cand not in tested_values:
                tested_values.add(cand)
                candidate_values.append(cand)
    return candidate_values


def _update_excess_phase_last_candidate(meta: dict, *, accept_meta: dict) -> None:
    meta["last_candidate_delta"] = dict(accept_meta.get("delta", {}) or {})
    meta["last_candidate_hard_gate"] = {
        "hard_gate_failed": bool(accept_meta.get("hard_gate_failed", False)),
        "hard_gate_reasons": list(accept_meta.get("hard_gate_reasons", []) or []),
        "residual_peak_db": float(_auto_safe_float(accept_meta.get("residual_peak_db"), float("nan"))),
        "residual_peak_hard_gate_db": float(_auto_safe_float(accept_meta.get("residual_peak_hard_gate_db"), float("nan"))),
    }

def _record_excess_phase_rejection(meta: dict, *, reason: str) -> None:
    meta.setdefault("reject_reasons", {})
    reject_reasons = dict(meta.get("reject_reasons", {}) or {})
    reject_reasons[str(reason)] = int(reject_reasons.get(str(reason), 0) or 0) + 1
    meta["reject_reasons"] = reject_reasons


def apply_excess_phase_strength_winner_polish(
    *,
    best_preset: dict | None,
    best_metrics: dict | None,
    base_data_ref: dict | None,
    phase_label: str,
    goal: str,
    enabled: bool,
    step: float,
    max_delta: float,
    status_cb,
    materialize_preset_result,
    cache_ready_preset,
    auto_is_better_refine,
) -> tuple[dict, dict, bool, dict]:
    cur_best_preset = dict(best_preset or {})
    cur_best_metrics = dict(best_metrics or {})
    meta = {
        "enabled": bool(enabled),
        "applicable": True,
        "phase_label": str(phase_label),
        "tested_values": [],
        "accepted_values": [],
        "tested_count": 0,
        "applied": False,
        "start_value": float("nan"),
        "final_value": float("nan"),
        "rank_before": float("nan"),
        "rank_after": float("nan"),
        "avg_before": float("nan"),
        "avg_after": float("nan"),
    }
    if not bool(enabled):
        return cur_best_preset, cur_best_metrics, False, meta
    if not isinstance(cur_best_metrics, dict) or not cur_best_metrics:
        return cur_best_preset, cur_best_metrics, False, meta

    base_value = _auto_safe_float(
        cur_best_preset.get(
            "excess_phase_strength",
            dict(base_data_ref or {}).get("excess_phase_strength", float("nan")),
        ),
        float("nan"),
    )
    if not np.isfinite(base_value):
        base_value = 0.9
    initial_value = round(float(np.clip(float(base_value), 0.0, 1.0)), 4)
    meta["start_value"] = float(initial_value)
    meta["rank_before"] = float(_auto_safe_float(cur_best_metrics.get("rank_score"), float("nan")))
    meta["avg_before"] = float(_auto_safe_float(cur_best_metrics.get("avg_score"), float("nan")))

    candidate_values = _build_excess_phase_candidate_values(initial_value, step, max_delta)

    meta["tested_values"] = [float(v) for v in candidate_values]
    meta["tested_count"] = int(len(candidate_values))
    if not candidate_values:
        meta["final_value"] = float(initial_value)
        meta["rank_after"] = float(_auto_safe_float(cur_best_metrics.get("rank_score"), float("nan")))
        meta["avg_after"] = float(_auto_safe_float(cur_best_metrics.get("avg_score"), float("nan")))
        return cur_best_preset, cur_best_metrics, False, meta

    improved = False
    with profiled_section("winner_polish.excess_phase_strength"):
        logger.info(
            "Automatic mode %s: testing %d excess_phase_strength winner-polish candidate(s) around %.4f.",
            str(phase_label),
            int(len(candidate_values)),
            float(initial_value),
        )
        current_base = float(initial_value)
        for idx, cand_value in enumerate(candidate_values, start=1):
            cand_test = dict(cur_best_preset or {})
            cand_test["excess_phase_strength"] = float(cand_value)
            try:
                _result, cand_metrics, _data = materialize_preset_result(
                    cand_test,
                    include_response_arrays=False,
                    summarize=False,
                    base_data_override=base_data_ref,
                )
            except _RECOVERABLE_EXCESS_POLISH_EXCEPTIONS as exc:
                logger.warning(
                    "Automatic mode %s failed for candidate %d/%d (excess_phase_strength=%.4f): %s",
                    str(phase_label),
                    int(idx),
                    int(len(candidate_values)),
                    float(cand_value),
                    f"{type(exc).__name__}: {exc}",
                )
                continue

            cand_metrics = dict(cand_metrics or {})
            better, reason, accept_meta = _winner_polish_acceptance(
                candidate_metrics=cand_metrics,
                current_metrics=cur_best_metrics,
                goal=goal,
                auto_is_better_refine=auto_is_better_refine,
            )
            _update_excess_phase_last_candidate(meta, accept_meta=dict(accept_meta or {}))
            if not bool(better):
                _record_excess_phase_rejection(meta, reason=str(reason))
            logger.info(
                "Automatic mode %s candidate %d/%d: excess_phase_strength=%.4f, rank=%.3f, decision=%s (%s)",
                str(phase_label),
                int(idx),
                int(len(candidate_values)),
                float(cand_value),
                _auto_safe_float(cand_metrics.get("rank_score"), 0.0),
                "accept" if bool(better) else "reject",
                str(reason),
            )
            if not bool(better):
                continue

            prev_best = dict(cur_best_metrics or {})
            cur_best_metrics = dict(cand_metrics or {})
            cur_best_preset = cache_ready_preset(cand_test, best_metrics=cur_best_metrics)
            improved = True
            accepted = list(meta.get("accepted_values", []) or [])
            accepted.append(float(cand_value))
            meta["accepted_values"] = accepted
            logger.info(
                "Automatic mode %s accepted candidate %d/%d: excess_phase_strength %.4f -> %.4f, rank %.3f -> %.3f, avg %.3f -> %.3f",
                str(phase_label),
                int(idx),
                int(len(candidate_values)),
                float(current_base),
                float(cand_value),
                _auto_safe_float(prev_best.get("rank_score"), 0.0),
                _auto_safe_float(cur_best_metrics.get("rank_score"), 0.0),
                _auto_safe_float(prev_best.get("avg_score"), 0.0),
                _auto_safe_float(cur_best_metrics.get("avg_score"), 0.0),
            )
            current_base = float(cand_value)
            if callable(status_cb):
                status_cb(
                    "DecayCore automatic mode: excess_phase_strength winner polish improved "
                    f"(excess_phase_strength {float(cand_value):.4f}, "
                    f"rank {_polish_rank_status(cur_best_metrics)}, "
                    f"avg {_auto_safe_float(cur_best_metrics.get('avg_score'), 0.0):.3f})"
                )

    meta["applied"] = bool(improved)
    final_value = _auto_safe_float(
        cur_best_preset.get(
            "excess_phase_strength",
            dict(base_data_ref or {}).get("excess_phase_strength", initial_value),
        ),
        initial_value,
    )
    meta["final_value"] = float(round(float(np.clip(float(final_value), 0.0, 1.0)), 4))
    meta["rank_after"] = float(_auto_safe_float(cur_best_metrics.get("rank_score"), float("nan")))
    meta["avg_after"] = float(_auto_safe_float(cur_best_metrics.get("avg_score"), float("nan")))
    return cur_best_preset, cur_best_metrics, bool(improved), dict(meta or {})
