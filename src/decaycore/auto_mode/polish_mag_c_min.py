# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Magnitude C-min winner-polish for auto-mode finalize."""

from __future__ import annotations

import json
import logging

import numpy as np

from .auto_mode_profile import profiled_section
from .rank_score import official_rank_score
from .shared import AUTO_MODE_MAG_C_MIN_MAX_HZ, AUTO_MODE_MAG_C_MIN_MIN_HZ, _auto_safe_float
from .winner_polish_utils import _enrich_target_tracking_metrics, _winner_polish_acceptance

logger = logging.getLogger("DecayCore")


def _mag_c_min_init_meta(*, enabled: bool, phase_label: str) -> dict:
    return {
        "enabled": bool(enabled),
        "applicable": True,
        "phase_label": str(phase_label),
        "tested_mag_c_min_hz": [],
        "accepted_mag_c_min_hz": [],
        "tested_count": 0,
        "applied": False,
        "start_mag_c_min_hz": float("nan"),
        "final_mag_c_min_hz": float("nan"),
        "rank_before": float("nan"),
        "rank_after": float("nan"),
        "avg_before": float("nan"),
        "avg_after": float("nan"),
    }


def _mag_c_min_candidates(
    *,
    initial_mag_c_min: float,
    step_hz,
    max_down_hz,
    max_up_hz,
) -> list[float]:
    step_hz_eff = max(0.1, float(_auto_safe_float(step_hz, 1.0)))
    max_down_hz_eff = max(0.0, float(_auto_safe_float(max_down_hz, 15.0)))
    max_up_hz_eff = max(0.0, float(_auto_safe_float(max_up_hz, 0.0)))
    min_mag_c_min = max(
        float(AUTO_MODE_MAG_C_MIN_MIN_HZ),
        float(initial_mag_c_min - max_down_hz_eff),
    )
    max_mag_c_min = min(
        float(AUTO_MODE_MAG_C_MIN_MAX_HZ),
        float(initial_mag_c_min + max_up_hz_eff),
    )
    candidate_mag_c_mins: list[float] = []
    tested_mag_c_mins: set[float] = {float(initial_mag_c_min)}
    max_span_hz = max(float(max_down_hz_eff), float(max_up_hz_eff))
    max_steps = int(max(0, np.floor((float(max_span_hz) / float(step_hz_eff)) + 1e-9)))
    for step_idx in range(1, int(max_steps) + 1):
        down_value = float(initial_mag_c_min - float(step_idx) * float(step_hz_eff))
        if down_value >= (float(min_mag_c_min) - 1e-9):
            cand_mag_c_min = round(
                float(
                    np.clip(
                        down_value,
                        float(AUTO_MODE_MAG_C_MIN_MIN_HZ),
                        float(AUTO_MODE_MAG_C_MIN_MAX_HZ),
                    )
                ),
                1,
            )
            if cand_mag_c_min not in tested_mag_c_mins:
                tested_mag_c_mins.add(cand_mag_c_min)
                candidate_mag_c_mins.append(float(cand_mag_c_min))
        up_value = float(initial_mag_c_min + float(step_idx) * float(step_hz_eff))
        if up_value <= (float(max_mag_c_min) + 1e-9):
            cand_mag_c_min = round(
                float(
                    np.clip(
                        up_value,
                        float(AUTO_MODE_MAG_C_MIN_MIN_HZ),
                        float(AUTO_MODE_MAG_C_MIN_MAX_HZ),
                    )
                ),
                1,
            )
            if cand_mag_c_min not in tested_mag_c_mins:
                tested_mag_c_mins.add(cand_mag_c_min)
                candidate_mag_c_mins.append(float(cand_mag_c_min))
    return list(candidate_mag_c_mins)


def _mag_c_min_preset_signature(*, preset: dict | None, metrics: dict | None, cache_ready_preset) -> str:
    ready = cache_ready_preset(
        dict(preset or {}),
        best_metrics=dict(metrics or {}),
    )
    try:
        return str(
            json.dumps(
                dict(ready or {}),
                sort_keys=True,
                separators=(",", ":"),
            )
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
        NameError,
    ):
        return str(sorted(dict(ready or {}).items()))


def _mag_c_min_candidate_lookup(
    *,
    candidate_items: list[dict] | None,
    cache_ready_preset,
) -> dict[str, dict]:
    candidate_metrics_lookup: dict[str, dict] = {}
    for item in list(candidate_items or []):
        if not isinstance(item, dict):
            continue
        item_preset = dict(item.get("preset", {}) or {})
        item_metrics = dict(item.get("metrics", {}) or {})
        if not item_preset or not item_metrics:
            continue
        sig = _mag_c_min_preset_signature(
            preset=item_preset,
            metrics=item_metrics,
            cache_ready_preset=cache_ready_preset,
        )
        prev = dict(candidate_metrics_lookup.get(sig, {}) or {})
        prev_rank = _auto_safe_float(prev.get("metrics", {}).get("rank_score"), float("-inf"))
        next_rank = _auto_safe_float(item_metrics.get("rank_score"), float("-inf"))
        if (sig not in candidate_metrics_lookup) or (float(next_rank) > float(prev_rank)):
            candidate_metrics_lookup[sig] = {
                "preset": dict(item_preset or {}),
                "metrics": dict(item_metrics or {}),
                "source": str(item.get("phase", item.get("source", "search_candidate")) or "search_candidate"),
            }
    return dict(candidate_metrics_lookup)


def _mag_c_min_materialize_candidate(
    *,
    candidate_hz: float,
    cur_best_preset: dict,
    cur_best_metrics: dict,
    cache_ready_preset,
    candidate_metrics_lookup: dict[str, dict],
    materialize_preset_result,
    base_data_ref: dict | None,
) -> tuple[dict, dict | None, dict | None]:
    cand_test = dict(cur_best_preset or {})
    cand_test["mag_c_min"] = float(candidate_hz)
    reuse_entry = candidate_metrics_lookup.get(
        _mag_c_min_preset_signature(
            preset=cand_test,
            metrics=cur_best_metrics,
            cache_ready_preset=cache_ready_preset,
        )
    )
    if isinstance(reuse_entry, dict) and reuse_entry:
        return dict(cand_test), dict(reuse_entry.get("metrics", {}) or {}), dict(reuse_entry)
    try:
        _mag_c_min_result, mag_c_min_metrics, _mag_c_min_data = materialize_preset_result(
            cand_test,
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
        NameError,
    ):
        return dict(cand_test), None, None
    return dict(cand_test), dict(mag_c_min_metrics or {}), reuse_entry


def _mag_c_min_update_accept_meta(meta: dict, *, better: bool, reason: str, accept_meta: dict) -> None:
    meta.setdefault("reject_reasons", {})
    if not bool(better):
        reject_reasons = dict(meta.get("reject_reasons", {}) or {})
        reject_reasons[str(reason)] = int(reject_reasons.get(str(reason), 0) or 0) + 1
        meta["reject_reasons"] = reject_reasons
    meta["last_candidate_delta"] = dict(accept_meta.get("delta", {}) or {})
    meta["last_candidate_hard_gate"] = {
        "hard_gate_failed": bool(accept_meta.get("hard_gate_failed", False)),
        "hard_gate_reasons": list(accept_meta.get("hard_gate_reasons", []) or []),
        "residual_peak_db": float(_auto_safe_float(accept_meta.get("residual_peak_db"), float("nan"))),
        "residual_peak_hard_gate_db": float(_auto_safe_float(accept_meta.get("residual_peak_hard_gate_db"), float("nan"))),
    }


def _mag_c_min_emit_status(
    *,
    status_cb,
    prev_mag_c_min_base: float,
    cand_mag_c_min: float,
    prev_best: dict,
    cur_best_metrics: dict,
) -> None:
    if not callable(status_cb):
        return
    status_cb(
        "DecayCore automatic mode: mag_c_min winner polish improved "
        f"(mag_c_min {float(prev_mag_c_min_base):.1f} -> {float(cand_mag_c_min):.1f} Hz, "
        f"rank {official_rank_score(prev_best):.3f} -> {official_rank_score(cur_best_metrics):.3f}, "
        f"avg {_auto_safe_float(prev_best.get('avg_score'), 0.0):.3f} -> {_auto_safe_float(cur_best_metrics.get('avg_score'), 0.0):.3f})"
    )


def _mag_c_min_finalize(
    *,
    cur_best_preset: dict,
    cur_best_metrics: dict,
    base_data_ref: dict | None,
    mag_c_min_meta: dict,
    improved: bool,
) -> tuple[dict, dict, bool, dict]:
    mag_c_min_meta["applied"] = bool(improved)
    mag_c_min_meta["final_mag_c_min_hz"] = float(
        round(
            float(
                np.clip(
                    cur_best_preset.get(
                        "mag_c_min",
                        dict(base_data_ref or {}).get("mag_c_min", 25.0),
                    ),
                    float(AUTO_MODE_MAG_C_MIN_MIN_HZ),
                    float(AUTO_MODE_MAG_C_MIN_MAX_HZ),
                )
            ),
            1,
        )
    )
    mag_c_min_meta["rank_after"] = float(_auto_safe_float(cur_best_metrics.get("rank_score"), float("nan")))
    mag_c_min_meta["avg_after"] = float(_auto_safe_float(cur_best_metrics.get("avg_score"), float("nan")))
    return cur_best_preset, cur_best_metrics, bool(improved), dict(mag_c_min_meta or {})


def apply_mag_c_min_winner_polish(
    *,
    best_preset: dict | None,
    best_metrics: dict | None,
    base_data_ref: dict | None,
    phase_label: str,
    goal: str,
    enabled: bool,
    step_hz,
    max_down_hz,
    max_up_hz=None,
    status_cb,
    materialize_preset_result,
    cache_ready_preset,
    auto_is_better_refine,
    candidate_items: list[dict] | None = None,
) -> tuple[dict, dict, bool, dict]:
    cur_best_preset = dict(best_preset or {})
    cur_best_metrics = dict(best_metrics or {})
    mag_c_min_meta = _mag_c_min_init_meta(enabled=bool(enabled), phase_label=str(phase_label))
    if not bool(enabled):
        return cur_best_preset, cur_best_metrics, False, mag_c_min_meta
    if not isinstance(cur_best_metrics, dict) or not cur_best_metrics:
        return cur_best_preset, cur_best_metrics, False, mag_c_min_meta

    mag_c_min_base = _auto_safe_float(
        cur_best_preset.get(
            "mag_c_min",
            dict(base_data_ref or {}).get("mag_c_min", float("nan")),
        ),
        float("nan"),
    )
    if not np.isfinite(mag_c_min_base):
        return cur_best_preset, cur_best_metrics, False, mag_c_min_meta
    initial_mag_c_min = round(
        float(
            np.clip(
                float(mag_c_min_base),
                float(AUTO_MODE_MAG_C_MIN_MIN_HZ),
                float(AUTO_MODE_MAG_C_MIN_MAX_HZ),
            )
        ),
        1,
    )
    mag_c_min_meta["start_mag_c_min_hz"] = float(initial_mag_c_min)
    mag_c_min_meta["rank_before"] = float(
        _auto_safe_float(cur_best_metrics.get("rank_score"), float("nan"))
    )
    mag_c_min_meta["avg_before"] = float(
        _auto_safe_float(cur_best_metrics.get("avg_score"), float("nan"))
    )

    candidate_mag_c_mins = _mag_c_min_candidates(
        initial_mag_c_min=float(initial_mag_c_min),
        step_hz=step_hz,
        max_down_hz=max_down_hz,
        max_up_hz=max_up_hz,
    )
    mag_c_min_meta["tested_mag_c_min_hz"] = [float(v) for v in candidate_mag_c_mins]
    mag_c_min_meta["tested_count"] = int(len(candidate_mag_c_mins))
    if not candidate_mag_c_mins:
        return _mag_c_min_finalize(
            cur_best_preset=cur_best_preset,
            cur_best_metrics=cur_best_metrics,
            base_data_ref=base_data_ref,
            mag_c_min_meta=mag_c_min_meta,
            improved=False,
        )

    candidate_metrics_lookup = _mag_c_min_candidate_lookup(
        candidate_items=candidate_items,
        cache_ready_preset=cache_ready_preset,
    )

    improved = False
    with profiled_section("winner_polish.mag_c_min"):
        logger.info(
            "Automatic mode %s: testing %d mag_c_min winner-polish candidate(s) around %.1f Hz.",
            str(phase_label),
            int(len(candidate_mag_c_mins)),
            float(initial_mag_c_min),
        )
        for idx, cand_mag_c_min in enumerate(candidate_mag_c_mins, start=1):
            cand_test, mag_c_min_metrics, reuse_entry = _mag_c_min_materialize_candidate(
                candidate_hz=float(cand_mag_c_min),
                cur_best_preset=cur_best_preset,
                cur_best_metrics=cur_best_metrics,
                cache_ready_preset=cache_ready_preset,
                candidate_metrics_lookup=candidate_metrics_lookup,
                materialize_preset_result=materialize_preset_result,
                base_data_ref=base_data_ref,
            )
            if not isinstance(mag_c_min_metrics, dict):
                logger.warning(
                    "Automatic mode %s failed for candidate %d/%d (mag_c_min=%.1f Hz): materialize_failed",
                    str(phase_label),
                    int(idx),
                    int(len(candidate_mag_c_mins)),
                    float(cand_mag_c_min),
                )
                continue

            mag_c_min_metrics = dict(mag_c_min_metrics or {})
            cur_best_metrics = _enrich_target_tracking_metrics(
                preset=cur_best_preset,
                metrics=cur_best_metrics,
                base_data_ref=base_data_ref,
                materialize_preset_result=materialize_preset_result,
            )
            mag_c_min_metrics = _enrich_target_tracking_metrics(
                preset=cand_test,
                metrics=mag_c_min_metrics,
                base_data_ref=base_data_ref,
                materialize_preset_result=materialize_preset_result,
            )
            better, reason, accept_meta = _winner_polish_acceptance(
                candidate_metrics=mag_c_min_metrics,
                current_metrics=cur_best_metrics,
                goal=goal,
                auto_is_better_refine=auto_is_better_refine,
            )
            _mag_c_min_update_accept_meta(
                mag_c_min_meta,
                better=bool(better),
                reason=str(reason),
                accept_meta=dict(accept_meta or {}),
            )
            logger.info(
                "Automatic mode %s candidate %d/%d: mag_c_min=%.1f Hz, rank=%.3f, decision=%s (%s)%s",
                str(phase_label),
                int(idx),
                int(len(candidate_mag_c_mins)),
                float(cand_mag_c_min),
                _auto_safe_float(mag_c_min_metrics.get("rank_score"), 0.0),
                "accept" if bool(better) else "reject",
                str(reason),
                (
                    ""
                    if not isinstance(reuse_entry, dict)
                    else f", reused={str(reuse_entry.get('source', 'search_candidate'))}"
                ),
            )
            if not bool(better):
                continue

            prev_best = dict(cur_best_metrics or {})
            prev_mag_c_min_base = float(mag_c_min_base)
            cur_best_metrics = dict(mag_c_min_metrics or {})
            cur_best_preset = cache_ready_preset(cand_test, best_metrics=cur_best_metrics)
            improved = True
            accepted = list(mag_c_min_meta.get("accepted_mag_c_min_hz", []) or [])
            accepted.append(float(cand_mag_c_min))
            mag_c_min_meta["accepted_mag_c_min_hz"] = accepted
            logger.info(
                "Automatic mode %s accepted candidate %d/%d: mag_c_min %.1f -> %.1f Hz, rank %.3f -> %.3f, avg %.3f -> %.3f",
                str(phase_label),
                int(idx),
                int(len(candidate_mag_c_mins)),
                float(mag_c_min_base),
                float(cand_mag_c_min),
                _auto_safe_float(prev_best.get("rank_score"), 0.0),
                _auto_safe_float(cur_best_metrics.get("rank_score"), 0.0),
                _auto_safe_float(prev_best.get("avg_score"), 0.0),
                _auto_safe_float(cur_best_metrics.get("avg_score"), 0.0),
            )
            mag_c_min_base = float(cand_mag_c_min)
            _mag_c_min_emit_status(
                status_cb=status_cb,
                prev_mag_c_min_base=float(prev_mag_c_min_base),
                cand_mag_c_min=float(cand_mag_c_min),
                prev_best=prev_best,
                cur_best_metrics=cur_best_metrics,
            )

    return _mag_c_min_finalize(
        cur_best_preset=cur_best_preset,
        cur_best_metrics=cur_best_metrics,
        base_data_ref=base_data_ref,
        mag_c_min_meta=mag_c_min_meta,
        improved=improved,
    )
