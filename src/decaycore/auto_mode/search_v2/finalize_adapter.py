# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Finalization bridge from v2 execution result to legacy orchestrator output."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .. import orchestrator_finalize
from ..auto_mode_profile import active_profiler_scope

if TYPE_CHECKING:
    from .context import AutoSearchExecutionContext

from .context import _nullctx

logger = logging.getLogger("DecayCore")


def attach_auto_search_fallbacks(result: dict | None, *sources: dict | None) -> dict | None:
    if not isinstance(result, dict):
        return result
    reasons = []
    seen = set()
    debug = dict(result.get("auto_mode_debug", {}) or {})
    for item in list(debug.get("fallback_reasons", []) or []):
        reason = str(item or "").strip()
        if reason and reason not in seen:
            reasons.append(reason)
            seen.add(reason)
    for source in sources:
        for item in list(
            dict(source or {}).get("_auto_search_fallback_reasons", []) or []
        ):
            reason = str(item or "").strip()
            if reason and reason not in seen:
                reasons.append(reason)
                seen.add(reason)
    if not reasons:
        return result
    out = dict(result)
    debug["fallback_reasons"] = list(reasons)
    out["auto_mode_debug"] = debug
    return out


def finalize_from_refine_stats(
    context: AutoSearchExecutionContext,
    stats: dict,
) -> dict | None:
    with active_profiler_scope(context.profiler):
        with (context.profiler.section("finalize") if context.profiler else _nullctx()):
            result = orchestrator_finalize.finalize_search_result(
                search_base_data=context.search_base_data,
                cache_base_data=context.cache_base_data,
                measurements=context.measurements,
                fs_v=int(context.fs_v),
                taps_v=int(context.taps_v),
                xos=context.xos,
                hpf=context.hpf,
                hc_f=context.hc_f,
                hc_m=context.hc_m,
                pin_obj=context.pin_obj,
                cfg=context.cfg,
                goal=context.goal,
                rank_basis=context.rank_basis,
                filter_key=context.filter_key,
                compat_version=context.compat_version,
                optimizer_backend=context.optimizer_backend,
                status_cb=context.status_cb,
                optuna_mod=context.optuna_mod,
                optuna_search_sig=context.optuna_search_sig,
                seed=int(context.seed),
                search_state=context.search_state,
                winner_target_name=context.winner_target_name,
                phase1_ok=int(dict(stats or {}).get("phase1_ok", 0) or 0),
                phase2_ok=int(dict(stats or {}).get("phase2_ok", 0) or 0),
                phase1_tried=int(dict(stats or {}).get("phase1_tried", 0) or 0),
                phase2_tried=int(dict(stats or {}).get("phase2_tried", 0) or 0),
                phase1_plateau_hit=bool(dict(stats or {}).get("phase1_plateau_hit", False)),
                phase2_plateau_hit=bool(dict(stats or {}).get("phase2_plateau_hit", False)),
                phase1_optuna_tel=dict(dict(stats or {}).get("phase1_optuna_tel", {}) or {}),
                phase2_local_optuna_tels=list(dict(stats or {}).get("phase2_local_optuna_tels", []) or []),
                phase3_micro_optuna_tel=dict(dict(stats or {}).get("phase3_micro_optuna_tel", {}) or {}),
                phase2_rollup_tel=dict(dict(stats or {}).get("phase2_rollup_tel", {}) or {}),
                _cache_ready_preset=context.cache_ready_preset,
                _materialize_preset_result=context.materialize_preset_result,
                _maybe_apply_residual_tiebreak=context.maybe_apply_residual_tiebreak,
                runtime=context.runtime,
            )
    if context.profiler:
        context.profiler.log_summary(logger, label="auto-mode search")
    return attach_auto_search_fallbacks(result, context.search_base_data, context.cache_base_data)


def finalize_from_cache_refine(
    context: AutoSearchExecutionContext,
    cache_refine_result: dict,
) -> dict | None:
    with active_profiler_scope(context.profiler):
        with (context.profiler.section("finalize") if context.profiler else _nullctx()):
            result = orchestrator_finalize.finalize_search_result(
                search_base_data=context.cache_base_data,
                cache_base_data=context.cache_base_data,
                measurements=context.measurements,
                fs_v=int(context.fs_v),
                taps_v=int(context.taps_v),
                xos=context.xos,
                hpf=context.hpf,
                hc_f=context.hc_f,
                hc_m=context.hc_m,
                pin_obj=context.pin_obj,
                cfg=context.cfg,
                goal=context.goal,
                rank_basis=context.rank_basis,
                filter_key=context.filter_key,
                compat_version=context.compat_version,
                optimizer_backend=context.optimizer_backend,
                status_cb=context.status_cb,
                optuna_mod=context.optuna_mod,
                optuna_search_sig=context.optuna_search_sig,
                seed=int(context.seed),
                search_state=None,
                winner_target_name=str(context.cache_base_data.get("hc_mode", "") or "").strip() or None,
                phase1_ok=0,
                phase2_ok=0,
                phase1_tried=0,
                phase2_tried=0,
                phase1_plateau_hit=False,
                phase2_plateau_hit=False,
                phase1_optuna_tel={},
                phase2_local_optuna_tels=[],
                phase3_micro_optuna_tel={},
                phase2_rollup_tel={},
                _cache_ready_preset=context.cache_ready_preset,
                _materialize_preset_result=context.materialize_preset_result,
                _maybe_apply_residual_tiebreak=context.maybe_apply_residual_tiebreak,
                cache_refine_result=dict(cache_refine_result or {}),
                runtime=context.runtime,
            )
    if context.profiler:
        context.profiler.log_summary(logger, label="auto-mode search")
    return attach_auto_search_fallbacks(result, context.search_base_data, context.cache_base_data)
