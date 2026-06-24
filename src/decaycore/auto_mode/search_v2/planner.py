# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Auto Search v2 planning."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..shared import _auto_filter_cache_key, _auto_goal, _auto_safe_bool
from .cache import (
    read_exact_cache_seed_with_reason,
    read_exact_cache_with_reason,
    read_last_used_best_with_reason,
)
from .input_model import AutoSearchInput
from .plan import AutoSearchPlan, AutoSearchPlanDecision
from .signature import AutoSearchSignature, compute_auto_search_signature_object


def _cache_status(reason: str, *, hit: bool = False, disabled: bool = False) -> str:
    if bool(disabled):
        return "disabled"
    if bool(hit):
        return "hit"
    reason_s = str(reason or "").strip().lower()
    if (
        "missing cache record" in reason_s
        or "missing cache schema" in reason_s
        or "missing winner preset" in reason_s
    ):
        return "miss"
    return "invalid"


def _cache_proof_status_from_reason(reason: str) -> str:
    reason_s = str(reason or "").strip().lower()
    if "missing cache record" in reason_s:
        return "missing"
    if "schema" in reason_s or "completion missing" in reason_s:
        return "old_schema"
    if "completed stages incomplete" in reason_s:
        return "partial"
    if "filter" in reason_s:
        return "wrong_filter"
    if "signature" in reason_s:
        return "stale_signature"
    if "read failed" in reason_s:
        return "read_failed"
    return "seed_only"


def _build_cache_decision_report(
    *,
    signature_obj: AutoSearchSignature,
    signature: str,
    exact_status: str,
    exact_reason: str,
    last_best_status: str,
    last_best_reason: str,
    selected_plan: AutoSearchPlan,
    cache_proof_status: str = "missing",
    filter_key: str = "",
    target_mode: str = "",
    compat_version: str = "",
) -> dict:
    return {
        "signature": str(signature),
        "signature_source": "identity_snapshot",
        "exact_cache_lookup_signature": str(signature),
        "filter_key": str(filter_key),
        "target_mode": str(target_mode),
        "compat_version": str(compat_version),
        "cache_proof_status": str(cache_proof_status),
        "measurement_identity": str(signature_obj.measurement_identity),
        "frequency_grid_identity": str(signature_obj.frequency_grid_identity),
        "diff_reason": str(signature_obj.diff_reason),
        "exact_cache": {
            "status": str(exact_status),
            "reason": str(exact_reason),
        },
        "last_best": {
            "status": str(last_best_status),
            "reason": str(last_best_reason),
        },
        "selected_plan": str(selected_plan.value),
    }


def _decision(
    *,
    plan: AutoSearchPlan,
    reason: str,
    signature: str,
    seed_preset: dict | None = None,
    cache_record: dict | None = None,
    fallback_reasons: list[str] | tuple[str, ...] | None = None,
    cache_decision_report: dict | None = None,
    seed_source: str | None = None,
) -> AutoSearchPlanDecision:
    if plan in (AutoSearchPlan.CACHE_MICRO_REFINE, AutoSearchPlan.LAST_BEST_MICRO_REFINE):
        skipped = ("target_search", "phase1", "phase2", "phase3")
        enabled = ("phase4",)
    elif plan == AutoSearchPlan.PRESELECTED_TARGET_REFINE or plan == AutoSearchPlan.MANUAL_PRESET_REFINE:
        skipped = ("target_search",)
        enabled = ("phase1", "phase2", "phase3", "phase4")
    elif plan == AutoSearchPlan.REUSE_VALID_RESULT:
        skipped = ("target_search", "phase1", "phase2", "phase3", "phase4")
        enabled = ()
    else:
        skipped = ("phase4",)
        enabled = ("target_search", "phase1", "phase2", "phase3")
    return AutoSearchPlanDecision(
        plan=plan,
        reason=str(reason),
        signature=str(signature),
        seed_preset=dict(seed_preset or {}) if isinstance(seed_preset, dict) and seed_preset else None,
        cache_record=dict(cache_record or {}) if isinstance(cache_record, dict) and cache_record else None,
        skipped_phases=tuple(skipped),
        enabled_phases=tuple(enabled),
        fallback_reasons=tuple(
            str(item)
            for item in list(fallback_reasons or [])
            if str(item or "").strip()
        ),
        cache_decision_report=dict(cache_decision_report or {}) if isinstance(cache_decision_report, dict) else None,
        seed_source=str(seed_source or "").strip() or None,
    )


def _wants_target_preselect(raw_data: dict) -> bool:
    mode = str(dict(raw_data or {}).get("auto_target_mode", "auto") or "auto").strip().lower()
    return bool(mode in ("auto", "best", "find_best", "find-best", "builtin", "built-in"))


def _wants_manual_preset_refine(raw_data: dict) -> bool:
    mode = str(dict(raw_data or {}).get("auto_target_mode", "") or "").strip().lower()
    return bool(mode in ("selected", "manual", "fixed", "user"))


def _has_target_preselect_seed(raw_data: dict) -> bool:
    raw = dict(raw_data or {})
    return bool(
        isinstance(raw.get("_auto_target_seed_preset"), dict)
        and dict(raw.get("_auto_target_seed_preset") or {})
    )


def _target_seed_source(raw_data: dict) -> str:
    return str(dict(raw_data or {}).get("_auto_target_seed_source", "") or "").strip().lower()


def _has_fresh_target_seed(raw_data: dict) -> bool:
    if not _has_target_preselect_seed(raw_data):
        return False
    source = _target_seed_source(raw_data)
    return bool(source in ("", "fresh_target_search", "target_preselect", "target_search"))


def _has_cached_target_seed(raw_data: dict) -> bool:
    if not _has_target_preselect_seed(raw_data):
        return False
    source = _target_seed_source(raw_data)
    return bool(
        source in (
            "cached_target_seed",
            "cache_signature",
            "cache_signature_hit",
            "cache_measurement",
            "cache_measurement_hit",
            "cache_measurement_global",
            "cache_measurement_global_hit",
            "cache_measurement_global_filter_seed",
            "cache_measurement_global_filter_seed_hit",
            "cache_optuna_target",
            "cache_optuna_target_hit",
        )
    )


def _cached_target_seed_label(raw_data: dict) -> str:
    source = _target_seed_source(raw_data)
    if source in ("cache_signature", "cache_signature_hit"):
        return "cache_signature_target_seed"
    if source in ("cache_measurement", "cache_measurement_hit"):
        return "cache_measurement_target_seed"
    if source in ("cache_measurement_global", "cache_measurement_global_hit"):
        return "cache_measurement_global_target"
    if source in ("cache_measurement_global_filter_seed", "cache_measurement_global_filter_seed_hit"):
        return "cache_measurement_global_filter_seed"
    if source in ("cache_optuna_target", "cache_optuna_target_hit"):
        return "cache_optuna_target_seed"
    return "cached_target_seed"


@dataclass(frozen=True)
class _PlanContext:
    """Identity values that stay fixed for a single planning call."""

    signature_obj: AutoSearchSignature
    signature: str
    filter_key: str
    target_mode: str
    compat_version: str


@dataclass
class _CacheStatus:
    """Cache lookup state that evolves as the resolution funnel runs."""

    exact_status: str
    exact_reason: str
    last_best_status: str
    last_best_reason: str
    fallback_reasons: list[str] = field(default_factory=list)
    cache_read_failed: bool = False


def _emit(
    ctx: _PlanContext,
    status: _CacheStatus,
    *,
    plan: AutoSearchPlan,
    reason: str,
    cache_proof_status: str,
    seed_preset: dict | None = None,
    cache_record: dict | None = None,
    seed_source: str | None = None,
    exact_status: str | None = None,
    last_best_status: str | None = None,
    last_best_reason: str | None = None,
) -> AutoSearchPlanDecision:
    """Build the cache-decision report and return a plan decision in one place.

    ``exact_status`` / ``last_best_status`` / ``last_best_reason`` override the
    accumulated ``status`` values for the report only, covering the cases where
    a stage records a per-decision label (e.g. ``"disabled"`` or ``"seed_only"``)
    that differs from the running funnel state.
    """
    report = _build_cache_decision_report(
        signature_obj=ctx.signature_obj,
        signature=ctx.signature,
        exact_status=status.exact_status if exact_status is None else exact_status,
        exact_reason=status.exact_reason,
        last_best_status=status.last_best_status if last_best_status is None else last_best_status,
        last_best_reason=status.last_best_reason if last_best_reason is None else last_best_reason,
        selected_plan=plan,
        cache_proof_status=cache_proof_status,
        filter_key=ctx.filter_key,
        target_mode=ctx.target_mode,
        compat_version=ctx.compat_version,
    )
    return _decision(
        plan=plan,
        reason=reason,
        signature=ctx.signature,
        seed_preset=seed_preset,
        cache_record=cache_record,
        fallback_reasons=status.fallback_reasons,
        cache_decision_report=report,
        seed_source=seed_source,
    )


def _try_exact_cache(
    ctx: _PlanContext,
    status: _CacheStatus,
    search_input: AutoSearchInput,
) -> AutoSearchPlanDecision | None:
    exact, exact_reason = read_exact_cache_with_reason(
        signature=ctx.signature,
        search_input=search_input,
        filter_key=ctx.filter_key,
        compat_version=ctx.compat_version,
    )
    status.exact_reason = exact_reason
    if isinstance(exact, dict) and exact:
        status.exact_status = _cache_status(exact_reason, hit=True)
        seed = dict(exact.get("winner_preset", exact.get("best_preset", {})) or {})
        return _emit(
            ctx,
            status,
            plan=AutoSearchPlan.CACHE_MICRO_REFINE,
            reason="exact canonical cache hit; running Phase 4 only",
            cache_proof_status="completed",
            seed_preset=seed,
            cache_record=exact,
            seed_source="exact_cache",
            last_best_status="disabled",
            last_best_reason="exact cache hit",
        )
    status.exact_status = _cache_status(exact_reason)
    status.fallback_reasons.append(f"exact cache skipped: {exact_reason!s}")
    status.cache_read_failed = status.cache_read_failed or "read failed" in str(exact_reason).lower()
    return None


def _try_exact_seed(
    ctx: _PlanContext,
    status: _CacheStatus,
    search_input: AutoSearchInput,
) -> AutoSearchPlanDecision | None:
    exact_seed, exact_seed_reason = read_exact_cache_seed_with_reason(
        signature=ctx.signature,
        search_input=search_input,
        filter_key=ctx.filter_key,
        compat_version=ctx.compat_version,
    )
    if isinstance(exact_seed, dict) and exact_seed:
        seed = dict(exact_seed.get("winner_preset", exact_seed.get("best_preset", {})) or {})
        if seed:
            return _emit(
                ctx,
                status,
                plan=AutoSearchPlan.FIRST_RUN_FULL_SEARCH,
                reason="exact cache is seed-only; running first-run stages because completion proof is missing",
                cache_proof_status=_cache_proof_status_from_reason(status.exact_reason),
                seed_preset=seed,
                cache_record=exact_seed,
                seed_source="old_exact_cache",
                exact_status="seed_only",
                last_best_status="disabled",
                last_best_reason="exact cache seed is not a completed filter result",
            )
    elif str(exact_seed_reason or "").strip() and str(exact_seed_reason) != str(status.exact_reason):
        status.fallback_reasons.append(f"exact cache seed skipped: {exact_seed_reason!s}")
    return None


def _try_cached_target_seed(
    ctx: _PlanContext,
    status: _CacheStatus,
    raw: dict,
) -> AutoSearchPlanDecision | None:
    if not _has_cached_target_seed(raw):
        return None
    seed = dict(raw.get("_auto_target_seed_preset", {}) or {})
    seed_metrics = dict(raw.get("_auto_target_seed_metrics", {}) or {})
    return _emit(
        ctx,
        status,
        plan=AutoSearchPlan.FIRST_RUN_FULL_SEARCH,
        reason=(
            "cached target seed only; running first-run stages because filter optimization is not proven complete"
        ),
        cache_proof_status="seed_only",
        seed_preset=seed,
        cache_record={"best_metrics": seed_metrics} if seed_metrics else None,
        seed_source=_cached_target_seed_label(raw),
        last_best_status="disabled",
        last_best_reason="cached target seed is not a completed filter result",
    )


def _try_last_best(
    ctx: _PlanContext,
    status: _CacheStatus,
    search_input: AutoSearchInput,
    goal: str,
) -> AutoSearchPlanDecision | None:
    last, last_reason = read_last_used_best_with_reason(
        search_input=search_input,
        goal=goal,
        filter_key=ctx.filter_key,
        compat_version=ctx.compat_version,
    )
    if isinstance(last, dict) and last:
        last_best_status = _cache_status(last_reason, hit=True)
        seed = dict(last.get("winner_preset", last.get("best_preset", {})) or {})
        status.fallback_reasons.append(
            "last-best seed available; running full Phase 1 -> Phase 2 -> Phase 3 -> Phase 4 search"
        )
        if seed:
            return _emit(
                ctx,
                status,
                plan=AutoSearchPlan.FIRST_RUN_FULL_SEARCH,
                reason="compatible last-used best seed; running first-run stages because exact completion proof is missing",
                cache_proof_status="seed_only",
                seed_preset=seed,
                cache_record=last,
                seed_source="last_best",
                last_best_status=last_best_status,
                last_best_reason=last_reason,
            )
    status.last_best_status = _cache_status(last_reason)
    status.last_best_reason = str(last_reason)
    status.cache_read_failed = status.cache_read_failed or "read failed" in str(last_reason).lower()
    status.fallback_reasons.append(f"last-best skipped: {last_reason!s}")
    return None


def _decide_without_cache_seed(
    ctx: _PlanContext,
    status: _CacheStatus,
    raw: dict,
) -> AutoSearchPlanDecision:
    if status.cache_read_failed:
        return _emit(
            ctx,
            status,
            plan=AutoSearchPlan.FALLBACK_FULL_SEARCH,
            reason="no valid cache and no compatible last-best seed",
            cache_proof_status="missing",
        )
    has_target_seed = _has_target_preselect_seed(raw)
    has_fresh_target_seed = _has_fresh_target_seed(raw)
    if has_target_seed or _wants_target_preselect(raw):
        plan = (
            AutoSearchPlan.PRESELECTED_TARGET_REFINE
            if has_fresh_target_seed
            else AutoSearchPlan.FIRST_RUN_FULL_SEARCH
        )
        return _emit(
            ctx,
            status,
            plan=plan,
            reason="using target preselect seed; running refine stages"
            if has_target_seed
            else "using automatic target search; running first-run stages",
            cache_proof_status="seed_only" if has_target_seed else "missing",
            seed_preset=dict(raw.get("_auto_target_seed_preset", {}) or {}) if has_target_seed else None,
            seed_source="fresh_target_search" if has_fresh_target_seed else None,
        )
    if _wants_manual_preset_refine(raw):
        return _emit(
            ctx,
            status,
            plan=AutoSearchPlan.MANUAL_PRESET_REFINE,
            reason="manual target selected and no valid exact cache",
            cache_proof_status="missing",
        )
    return _emit(
        ctx,
        status,
        plan=AutoSearchPlan.FIRST_RUN_FULL_SEARCH,
        reason="no valid cache and no compatible last-best seed",
        cache_proof_status="missing",
    )


def determine_auto_search_plan(
    search_input: AutoSearchInput,
    raw_data: dict,
    cache_store=None,
    options=None,
) -> AutoSearchPlanDecision:
    opts = dict(options or {})
    signature_obj = opts.get("signature_object")
    if not isinstance(signature_obj, AutoSearchSignature):
        signature_obj = compute_auto_search_signature_object(search_input)
    signature = str(opts.get("signature") or signature_obj.signature)
    raw = dict(raw_data or {})
    compat_version = str(opts.get("compat_version", raw.get("auto_mode_compat_version", "")) or "")
    filter_key = str(opts.get("filter_key") or _auto_filter_cache_key(raw))
    goal = str(opts.get("goal") or _auto_goal(raw))
    target_mode = str(raw.get("auto_target_mode", "") or "").strip().lower()
    cache_enabled = _auto_safe_bool(
        opts.get("cache_enabled", raw.get("auto_mode_cache_enabled", True)),
        True,
    )

    ctx = _PlanContext(
        signature_obj=signature_obj,
        signature=signature,
        filter_key=filter_key,
        target_mode=target_mode,
        compat_version=compat_version,
    )
    status = _CacheStatus(
        exact_status="disabled" if not cache_enabled else "miss",
        exact_reason="disabled" if not cache_enabled else "not checked",
        last_best_status="disabled" if not cache_enabled else "miss",
        last_best_reason="disabled" if not cache_enabled else "not checked",
    )

    if cache_enabled:
        decision = _try_exact_cache(ctx, status, search_input)
        if decision is None:
            decision = _try_exact_seed(ctx, status, search_input)
        if decision is None:
            decision = _try_cached_target_seed(ctx, status, raw)
        if decision is None:
            decision = _try_last_best(ctx, status, search_input, goal)
        if decision is not None:
            return decision
    else:
        status.fallback_reasons.append("cache skipped: disabled")

    return _decide_without_cache_seed(ctx, status, raw)
