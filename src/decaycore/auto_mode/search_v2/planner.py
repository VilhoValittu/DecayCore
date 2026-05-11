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

from ..shared import _auto_filter_cache_key, _auto_goal, _auto_safe_bool
from .cache import read_exact_cache_with_reason, read_last_used_best_with_reason
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


def _build_cache_decision_report(
    *,
    signature_obj: AutoSearchSignature,
    signature: str,
    exact_status: str,
    exact_reason: str,
    last_best_status: str,
    last_best_reason: str,
    selected_plan: AutoSearchPlan,
) -> dict:
    return {
        "signature": str(signature),
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
) -> AutoSearchPlanDecision:
    if plan in (AutoSearchPlan.CACHE_MICRO_REFINE, AutoSearchPlan.LAST_BEST_MICRO_REFINE):
        skipped = ("phase1", "phase2")
        enabled = ("micro_refine",)
    elif plan == AutoSearchPlan.PRESELECTED_TARGET_REFINE:
        skipped = ()
        enabled = ("phase1","phase2", "micro_refine")
    elif plan == AutoSearchPlan.MANUAL_PRESET_REFINE:
        skipped = ()
        enabled = ("phase1", "phase2", "micro_refine")
    elif plan == AutoSearchPlan.REUSE_VALID_RESULT:
        skipped = ("phase1", "phase2", "micro_refine")
        enabled = ("reuse",)
    else:
        skipped = ()
        enabled = ("phase1", "phase2", "micro_refine")
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
    )


def _wants_target_preselect(raw_data: dict) -> bool:
    mode = str(dict(raw_data or {}).get("auto_target_mode", "auto") or "auto").strip().lower()
    return bool(mode in ("auto", "best", "find_best", "find-best", "builtin", "built-in"))


def _wants_manual_preset_refine(raw_data: dict) -> bool:
    mode = str(dict(raw_data or {}).get("auto_target_mode", "") or "").strip().lower()
    return bool(mode in ("selected", "manual", "fixed", "user"))


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
    cache_enabled = _auto_safe_bool(
        opts.get("cache_enabled", raw.get("auto_mode_cache_enabled", True)),
        True,
    )
    fallback_reasons: list[str] = []
    exact_status = "disabled" if not cache_enabled else "miss"
    exact_reason = "disabled" if not cache_enabled else "not checked"
    last_best_status = "disabled" if not cache_enabled else "miss"
    last_best_reason = "disabled" if not cache_enabled else "not checked"

    if cache_enabled:
        exact, exact_reason = read_exact_cache_with_reason(
            signature=signature,
            search_input=search_input,
            filter_key=filter_key,
            compat_version=compat_version,
        )
        if isinstance(exact, dict) and exact:
            exact_status = _cache_status(exact_reason, hit=True)
            report = _build_cache_decision_report(
                signature_obj=signature_obj,
                signature=signature,
                exact_status=exact_status,
                exact_reason=exact_reason,
                last_best_status="disabled",
                last_best_reason="exact cache hit",
                selected_plan=AutoSearchPlan.CACHE_MICRO_REFINE,
            )
            seed = dict(exact.get("winner_preset", exact.get("best_preset", {})) or {})
            return _decision(
                plan=AutoSearchPlan.CACHE_MICRO_REFINE,
                reason="exact canonical cache hit",
                signature=signature,
                seed_preset=seed,
                cache_record=exact,
                fallback_reasons=fallback_reasons,
                cache_decision_report=report,
            )
        exact_status = _cache_status(exact_reason)
        fallback_reasons.append(f"exact cache skipped: {str(exact_reason)}")

        last, last_reason = read_last_used_best_with_reason(
            search_input=search_input,
            goal=goal,
            filter_key=filter_key,
            compat_version=compat_version,
        )
        if isinstance(last, dict) and last:
            last_best_status = _cache_status(last_reason, hit=True)
            report = _build_cache_decision_report(
                signature_obj=signature_obj,
                signature=signature,
                exact_status=exact_status,
                exact_reason=exact_reason,
                last_best_status=last_best_status,
                last_best_reason=last_reason,
                selected_plan=AutoSearchPlan.LAST_BEST_MICRO_REFINE,
            )
            seed = dict(last.get("winner_preset", last.get("best_preset", {})) or {})
            return _decision(
                plan=AutoSearchPlan.LAST_BEST_MICRO_REFINE,
                reason="compatible last-used best seed",
                signature=signature,
                seed_preset=seed,
                cache_record=last,
                fallback_reasons=fallback_reasons,
                cache_decision_report=report,
            )
        last_best_status = _cache_status(last_reason)
        last_best_reason = str(last_reason)
        fallback_reasons.append(f"last-best skipped: {str(last_reason)}")
    else:
        fallback_reasons.append("cache skipped: disabled")

    if _wants_target_preselect(raw):
        plan = AutoSearchPlan.PRESELECTED_TARGET_REFINE
        return _decision(
            plan=plan,
            reason="using preselected target; running refine stages",
            signature=signature,
            fallback_reasons=fallback_reasons,
            cache_decision_report=_build_cache_decision_report(
                signature_obj=signature_obj,
                signature=signature,
                exact_status=exact_status,
                exact_reason=exact_reason,
                last_best_status=last_best_status,
                last_best_reason=last_best_reason,
                selected_plan=plan,
            ),
        )
    if _wants_manual_preset_refine(raw):
        plan = AutoSearchPlan.MANUAL_PRESET_REFINE
        return _decision(
            plan=plan,
            reason="manual target selected and no valid exact cache",
            signature=signature,
            fallback_reasons=fallback_reasons,
            cache_decision_report=_build_cache_decision_report(
                signature_obj=signature_obj,
                signature=signature,
                exact_status=exact_status,
                exact_reason=exact_reason,
                last_best_status=last_best_status,
                last_best_reason=last_best_reason,
                selected_plan=plan,
            ),
        )
    plan = AutoSearchPlan.FULL_SEARCH
    return _decision(
        plan=plan,
        reason="no valid cache and no compatible last-best seed",
        signature=signature,
        fallback_reasons=fallback_reasons,
        cache_decision_report=_build_cache_decision_report(
            signature_obj=signature_obj,
            signature=signature,
            exact_status=exact_status,
            exact_reason=exact_reason,
            last_best_status=last_best_status,
            last_best_reason=last_best_reason,
            selected_plan=plan,
        ),
    )
