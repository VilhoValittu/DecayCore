# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""TDC-strength winner-polish for auto-mode finalize."""

from __future__ import annotations

import logging

import numpy as np

from .auto_mode_profile import profiled_section
from .shared_parts import _auto_safe_float
from .winner_polish_utils import _polish_rank_status, _winner_polish_acceptance

logger = logging.getLogger("DecayCore")

_TDC_STRENGTH_POLISH_MIN = 0.1
_TDC_STRENGTH_POLISH_MAX = 75.0

_RECOVERABLE_TDC_POLISH_EXCEPTIONS = (
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


def _tdc_polish_applicability(
    *,
    cur_best_preset: dict,
    cur_best_metrics: dict,
    base_data_ref: dict | None,
) -> tuple[bool, str, float, float]:
    if not isinstance(cur_best_metrics, dict) or not cur_best_metrics:
        return False, "no_metrics", float("nan"), float("nan")
    enable_tdc = bool(cur_best_preset.get("enable_tdc", dict(base_data_ref or {}).get("enable_tdc", True)))
    if not enable_tdc:
        return False, "tdc_disabled", float("nan"), float("nan")
    action_hint = str(cur_best_metrics.get("tdc_action_hint", "") or "")
    if action_hint != "decrease_tdc":
        return False, f"action_hint={action_hint!r}", float("nan"), float("nan")
    optimum_hint = _auto_safe_float(cur_best_metrics.get("tdc_decay_optimum_strength", float("nan")), float("nan"))
    if not np.isfinite(optimum_hint):
        return False, "no_optimum_hint", float("nan"), float("nan")
    current_strength = _auto_safe_float(
        cur_best_preset.get("tdc_strength", dict(base_data_ref or {}).get("tdc_strength", float("nan"))),
        float("nan"),
    )
    if not np.isfinite(current_strength):
        return False, "no_current_strength", float("nan"), float("nan")
    return True, "", float(current_strength), float(optimum_hint)


def _build_tdc_polish_candidates(
    *,
    current_strength: float,
    optimum_hint: float,
    step: float,
    max_delta: float,
) -> tuple[float, list[float]]:
    step_eff = max(0.1, float(_auto_safe_float(step, 1.0)))
    max_delta_eff = max(0.0, float(_auto_safe_float(max_delta, 20.0)))
    max_candidates = int(max(0, round(max_delta_eff / step_eff + 0.5)))
    raw = np.arange(
        float(current_strength) - step_eff,
        float(optimum_hint) - step_eff / 2.0,
        -step_eff,
    )
    seen: set[float] = set()
    candidate_values: list[float] = []
    for v in raw:
        c = round(float(np.clip(v, _TDC_STRENGTH_POLISH_MIN, float(current_strength) - step_eff)), 1)
        if c not in seen:
            seen.add(c)
            candidate_values.append(c)
        if len(candidate_values) >= max_candidates:
            break
    return float(step_eff), list(candidate_values)


def _record_tdc_polish_rejection(meta: dict, *, reason: str) -> None:
    meta.setdefault("reject_reasons", {})
    reject_reasons = dict(meta.get("reject_reasons", {}) or {})
    reject_reasons[str(reason)] = int(reject_reasons.get(str(reason), 0) or 0) + 1
    meta["reject_reasons"] = reject_reasons


def apply_tdc_strength_winner_polish(
    *,
    best_preset: dict | None,
    best_metrics: dict | None,
    base_data_ref: dict | None,
    phase_label: str,
    goal: str,
    enabled: bool,
    step: float = 1.0,
    max_delta: float = 20.0,
    status_cb,
    materialize_preset_result,
    cache_ready_preset,
    auto_is_better_refine,
) -> tuple[dict, dict, bool, dict]:
    cur_best_preset = dict(best_preset or {})
    cur_best_metrics = dict(best_metrics or {})
    meta: dict = {
        "enabled": bool(enabled),
        "applicable": False,
        "not_applicable_reason": "",
        "phase_label": str(phase_label),
        "tested_strengths": [],
        "accepted_strengths": [],
        "tested_count": 0,
        "applied": False,
        "start_strength": float("nan"),
        "final_strength": float("nan"),
        "optimum_strength_hint": float("nan"),
        "rank_before": float("nan"),
        "rank_after": float("nan"),
        "avg_before": float("nan"),
        "avg_after": float("nan"),
    }
    if not bool(enabled):
        meta["not_applicable_reason"] = "disabled"
        return cur_best_preset, cur_best_metrics, False, meta

    applicable, reason, current_strength, optimum_hint = _tdc_polish_applicability(
        cur_best_preset=dict(cur_best_preset or {}),
        cur_best_metrics=dict(cur_best_metrics or {}),
        base_data_ref=base_data_ref,
    )
    if not applicable:
        meta["not_applicable_reason"] = str(reason)
        return cur_best_preset, cur_best_metrics, False, meta

    meta["applicable"] = True
    meta["start_strength"] = float(current_strength)
    meta["optimum_strength_hint"] = float(optimum_hint)
    meta["rank_before"] = float(_auto_safe_float(cur_best_metrics.get("rank_score"), float("nan")))
    meta["avg_before"] = float(_auto_safe_float(cur_best_metrics.get("avg_score"), float("nan")))

    step_eff, candidate_values = _build_tdc_polish_candidates(
        current_strength=float(current_strength),
        optimum_hint=float(optimum_hint),
        step=float(step),
        max_delta=float(max_delta),
    )

    meta["tested_strengths"] = [float(v) for v in candidate_values]
    meta["tested_count"] = int(len(candidate_values))
    if not candidate_values:
        meta["final_strength"] = float(current_strength)
        meta["rank_after"] = float(_auto_safe_float(cur_best_metrics.get("rank_score"), float("nan")))
        meta["avg_after"] = float(_auto_safe_float(cur_best_metrics.get("avg_score"), float("nan")))
        return cur_best_preset, cur_best_metrics, False, meta

    improved = False
    with profiled_section("winner_polish.tdc_strength"):
        logger.info(
            "Automatic mode %s: testing %d tdc_strength winner-polish candidate(s) "
            "from %.1f toward optimum %.1f (step %.1f).",
            str(phase_label),
            int(len(candidate_values)),
            float(current_strength),
            float(optimum_hint),
            float(step_eff),
        )
        current_base = float(current_strength)
        for idx, cand_value in enumerate(candidate_values, start=1):
            cand_test = dict(cur_best_preset or {})
            cand_test["tdc_strength"] = float(cand_value)
            try:
                _result, cand_metrics, _data = materialize_preset_result(
                    cand_test,
                    include_response_arrays=False,
                    summarize=False,
                    base_data_override=base_data_ref,
                )
            except _RECOVERABLE_TDC_POLISH_EXCEPTIONS as exc:
                logger.warning(
                    "Automatic mode %s failed for candidate %d/%d (tdc_strength=%.1f): %s",
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
            if not bool(better):
                _record_tdc_polish_rejection(meta, reason=str(reason))
            logger.info(
                "Automatic mode %s candidate %d/%d: tdc_strength=%.1f, rank=%.3f, decision=%s (%s)",
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
            accepted = list(meta.get("accepted_strengths", []) or [])
            accepted.append(float(cand_value))
            meta["accepted_strengths"] = accepted
            logger.info(
                "Automatic mode %s accepted candidate %d/%d: tdc_strength %.1f -> %.1f, "
                "rank %.3f -> %.3f, avg %.3f -> %.3f",
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
                    "DecayCore automatic mode: tdc_strength winner polish improved "
                    f"(tdc_strength {float(cand_value):.1f}%, "
                    f"rank {_polish_rank_status(cur_best_metrics)}, "
                    f"avg {_auto_safe_float(cur_best_metrics.get('avg_score'), 0.0):.3f})"
                )

    meta["applied"] = bool(improved)
    final_strength = _auto_safe_float(
        cur_best_preset.get("tdc_strength", current_strength),
        current_strength,
    )
    meta["final_strength"] = float(round(float(np.clip(final_strength, _TDC_STRENGTH_POLISH_MIN, _TDC_STRENGTH_POLISH_MAX)), 1))
    meta["rank_after"] = float(_auto_safe_float(cur_best_metrics.get("rank_score"), float("nan")))
    meta["avg_after"] = float(_auto_safe_float(cur_best_metrics.get("avg_score"), float("nan")))
    return cur_best_preset, cur_best_metrics, bool(improved), dict(meta or {})
