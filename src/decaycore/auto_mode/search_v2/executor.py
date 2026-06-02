# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Auto Search v2 plan executor."""

from __future__ import annotations

from ..shared import logger
from .legacy_adapter import (
    build_execution_context,
    finalize_from_cache_refine,
    finalize_from_refine_stats,
    record_auto_search_fallback,
    run_micro_refine_from_seed,
    run_refine_stages,
)
from .plan import AutoSearchPlan


def _base_data_with_plan_seed(base_data: dict, decision) -> dict:
    data = dict(base_data or {})
    plan = getattr(decision, "plan", AutoSearchPlan.FIRST_RUN_FULL_SEARCH)
    if plan in (
        AutoSearchPlan.CACHE_MICRO_REFINE,
        AutoSearchPlan.LAST_BEST_MICRO_REFINE,
        AutoSearchPlan.REUSE_VALID_RESULT,
    ):
        return data
    seed = dict(getattr(decision, "seed_preset", {}) or {})
    if not seed:
        return data
    data["_auto_target_seed_preset"] = dict(seed)
    _hc_mode_before = data.get("hc_mode")
    data.update(seed)
    if _hc_mode_before is not None:
        data["hc_mode"] = _hc_mode_before
    cache_record = dict(getattr(decision, "cache_record", {}) or {})
    seed_metrics = dict(cache_record.get("winner_metrics", cache_record.get("best_metrics", {})) or {})
    if seed_metrics:
        data["_auto_target_seed_metrics"] = dict(seed_metrics)
    seed_source = str(getattr(decision, "seed_source", "") or "").strip()
    if seed_source:
        data["_auto_target_seed_source"] = str(seed_source)
    return data


def run_target_search_stage(base_data: dict, decision, *, status_cb=None) -> dict:
    """Attach target-search stage metadata without trusting it as completion proof.

    The workflow still performs the expensive target comparison before this
    search entry point. This adapter centralizes the stage boundary for
    search_v2 and preserves cached/fresh target seeds as refine seeds.
    """

    data = dict(base_data or {})
    seed = dict(getattr(decision, "seed_preset", {}) or {})
    if seed and not isinstance(data.get("_auto_target_seed_preset"), dict):
        data["_auto_target_seed_preset"] = dict(seed)
    seed_source = str(getattr(decision, "seed_source", "") or "").strip()
    if seed_source:
        data["_auto_target_seed_source"] = str(seed_source)
    return data


def _fall_back_to_full_search(context, *, reason: str) -> dict | None:
    record_auto_search_fallback(context.search_base_data, str(reason))
    logger.info("Automatic mode search v2: %s", str(reason))
    stats = run_refine_stages(context, skip_phase1=False)
    return finalize_from_refine_stats(context, stats)


def _reuse_valid_result(context, decision) -> dict | None:
    cache_record = dict(getattr(decision, "cache_record", {}) or {})
    winner_preset = dict(cache_record.get("winner_preset", {}) or {})
    winner_metrics = dict(cache_record.get("winner_metrics", {}) or {})
    if not winner_preset:
        winner_preset = dict(cache_record.get("best_preset", {}) or {})
    if not winner_metrics:
        winner_metrics = dict(cache_record.get("best_metrics", {}) or {})
    if not winner_preset or not winner_metrics:
        record_auto_search_fallback(
            context.search_base_data,
            "reuse valid result skipped: missing winner preset or metrics",
        )
        return None
    reuse_refine_result = {
        "best_preset": winner_preset,
        "best_metrics": winner_metrics,
        "seed_source": "reuse_valid_result",
        "cache_target_name": str(cache_record.get("search_input_summary", {}).get("hc_mode", "") or "n/a"),
        "improved_any": False,
        "improved_count_total": 0,
        "executed_micro_trials_total": 0,
        "cache_refine_rollup_tel": {},
        "stop_reason": "reuse_valid_result",
    }
    result = finalize_from_cache_refine(context, reuse_refine_result)
    if isinstance(result, dict):
        logger.info("search_v2 REUSE_VALID_RESULT: returning cached result directly.")
    return result


def execute_auto_search_plan(
    *,
    decision,
    base_data: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    hc_f,
    hc_m,
    pin_obj,
    status_cb,
    n_trials: int,
) -> dict | None:
    base_data = _base_data_with_plan_seed(base_data, decision)
    if "target_search" in tuple(getattr(decision, "enabled_phases", ()) or ()):
        base_data = run_target_search_stage(base_data, decision, status_cb=status_cb)
    context = build_execution_context(
        base_data=base_data,
        measurements=measurements,
        fs_v=int(fs_v),
        taps_v=int(taps_v),
        xos=list(xos or []),
        hpf=hpf,
        hc_f=hc_f,
        hc_m=hc_m,
        pin_obj=pin_obj,
        status_cb=status_cb,
        n_trials=int(n_trials),
        allow_legacy_cache_seeds=False,
        canonical_signature=str(getattr(decision, "signature", "") or "").strip(),
    )
    plan = getattr(decision, "plan", AutoSearchPlan.FIRST_RUN_FULL_SEARCH)
    if plan in (AutoSearchPlan.CACHE_MICRO_REFINE, AutoSearchPlan.LAST_BEST_MICRO_REFINE):
        cache_refine_result = run_micro_refine_from_seed(context, decision)
        if isinstance(cache_refine_result, dict):
            result = finalize_from_cache_refine(context, dict(cache_refine_result or {}))
            if isinstance(result, dict):
                return result
            return _fall_back_to_full_search(
                context,
                reason="cache micro-refine finalize returned no result; falling back to full search",
            )
        return _fall_back_to_full_search(
            context,
            reason=f"{str(plan.value)} unavailable: falling back to full search",
        )

    if plan == AutoSearchPlan.REUSE_VALID_RESULT:
        result = _reuse_valid_result(context, decision)
        if isinstance(result, dict):
            return result
        stats = run_refine_stages(context, skip_phase1=False)
        return finalize_from_refine_stats(context, stats)

    stats = run_refine_stages(context, skip_phase1=False)
    return finalize_from_refine_stats(context, stats)
