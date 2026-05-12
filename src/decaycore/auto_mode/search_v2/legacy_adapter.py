# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Auto Search v2 adapter for the existing automatic-mode orchestrators.

Public symbols are re-exported from focused sub-modules; this file owns only
the three run_* orchestration wrappers that are not yet split.
"""

from __future__ import annotations

import logging

from .. import orchestrator_refine
from ..auto_mode_profile import active_profiler_scope
from .context import AutoSearchExecutionContext, _nullctx, build_execution_context
from .finalize_adapter import (
    attach_auto_search_fallbacks,
    finalize_from_cache_refine,
    finalize_from_refine_stats,
)
from .runtime import build_auto_mode_orchestrator_runtime
from .seeds import record_auto_search_fallback

__all__ = [
    "AutoSearchExecutionContext",
    "attach_auto_search_fallbacks",
    "build_auto_mode_orchestrator_runtime",
    "build_execution_context",
    "finalize_from_cache_refine",
    "finalize_from_refine_stats",
    "record_auto_search_fallback",
    "run_legacy_full_search",
    "run_micro_refine_from_seed",
    "run_refine_stages",
]

logger = logging.getLogger("DecayCore")


def run_micro_refine_from_seed(
    context: AutoSearchExecutionContext,
    decision,
) -> dict | None:
    seed_preset = dict(getattr(decision, "seed_preset", {}) or {})
    cache_record = dict(getattr(decision, "cache_record", {}) or {})
    seed_metrics = dict(
        cache_record.get(
            "winner_metrics",
            cache_record.get("best_metrics", {}),
        )
        or {}
    )
    seed_source = str(getattr(decision, "seed_source", "") or "").strip()
    if not seed_source:
        seed_source = "last_best" if str(getattr(getattr(decision, "plan", None), "value", "")) == "last_best_micro_refine" else "exact_cache"
    return orchestrator_refine.run_exact_cache_micro_refine(
        cache_base_data=context.cache_base_data,
        measurements=context.measurements,
        fs_v=int(context.fs_v),
        taps_v=int(context.taps_v),
        xos=context.xos,
        hpf=context.hpf,
        status_cb=context.status_cb,
        cfg=context.cfg,
        goal=context.goal,
        filter_key=context.filter_key,
        compat_version=context.compat_version,
        optimizer_backend=context.optimizer_backend,
        optuna_mod=context.optuna_mod,
        seed=int(context.seed),
        optuna_search_sig=context.optuna_search_sig,
        _cache_ready_preset=context.cache_ready_preset,
        _materialize_preset_result=context.materialize_preset_result,
        seed_preset=seed_preset,
        seed_metrics=seed_metrics,
        seed_source=str(seed_source),
        runtime=context.runtime,
    )


def run_refine_stages(
    context: AutoSearchExecutionContext,
    *,
    skip_phase1: bool = False,
) -> dict:
    with active_profiler_scope(context.profiler):
        with (context.profiler.section("search_refine_stages") if context.profiler else _nullctx()):
            return orchestrator_refine.run_search_refine_stages(
                search_base_data=context.search_base_data,
                measurements=context.measurements,
                fs_v=int(context.fs_v),
                taps_v=int(context.taps_v),
                xos=context.xos,
                hpf=context.hpf,
                hc_f=context.hc_f,
                hc_m=context.hc_m,
                pin_obj=context.pin_obj,
                status_cb=context.status_cb,
                cfg=context.cfg,
                goal=context.goal,
                filter_key=context.filter_key,
                optimizer_backend=context.optimizer_backend,
                optuna_mod=context.optuna_mod,
                seed=int(context.seed),
                optuna_search_sig=context.optuna_search_sig,
                status_prefix=context.status_prefix,
                winner_target_name=context.winner_target_name,
                search_state=context.search_state,
                n_trials_eff=int(context.n_trials_eff),
                candidates=list(context.candidates or []),
                prior_seed_preset=dict(context.prior_seed_preset or {}),
                use_optuna_trials=bool(context.use_optuna_trials),
                runtime=context.runtime,
                skip_phase1=bool(skip_phase1),
            )


def run_legacy_full_search(
    *,
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
        allow_legacy_cache_seeds=True,
    )
    stats = run_refine_stages(context, skip_phase1=False)
    return finalize_from_refine_stats(context, stats)
