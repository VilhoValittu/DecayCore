# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Search-refine pipeline assembly and public API."""

from __future__ import annotations

import logging

from .refine_eval_parts import build_phase2_rollup_telemetry
from .runtime_context import coerce_orchestrator_runtime
from ._refine_types import (
    _SearchPhase1State,
    _SearchPhase2State,
    _SearchRefineContext,
    _SearchRefineSummary,
)
from ._refine_search_core import (
    _build_search_refine_eval_context,
    _carry_forward_phase1_best_core,
    _run_search_refine_micro_core,
    _run_search_refine_phase1_core,
    _run_search_refine_phase2_local_core,
)

logger = logging.getLogger("DecayCore")


def _build_search_refine_result(
    *,
    runtime,
    status_cb,
    phase1: _SearchPhase1State,
    phase2: _SearchPhase2State,
) -> dict:
    phase2_rollup_tel = build_phase2_rollup_telemetry(
        phase2_local_optuna_tels=phase2.phase2_local_optuna_tels,
        phase3_micro_optuna_tel=phase2.phase3_micro_optuna_tel,
        telemetry_rollup_fn=runtime.auto_optuna_telemetry_rollup,
    )
    phase2_rollup_txt = runtime.auto_optuna_telemetry_text(phase2_rollup_tel)
    if phase2_rollup_txt:
        logger.info("Automatic mode Phase2 summary: %s", str(phase2_rollup_txt))
        if callable(status_cb):
            status_cb(f"DecayCore automatic mode: Phase2 summary {phase2_rollup_txt}")
    return {
        "phase1_ok": int(phase1.phase1_ok),
        "phase2_ok": int(phase2.phase2_ok),
        "phase1_tried": int(phase1.phase1_tried),
        "phase2_tried": int(phase2.phase2_tried),
        "phase1_plateau_hit": bool(phase1.phase1_plateau_hit),
        "phase2_plateau_hit": bool(phase2.phase2_plateau_hit),
        "phase1_optuna_tel": dict(phase1.phase1_optuna_tel or {}),
        "phase2_local_optuna_tels": list(phase2.phase2_local_optuna_tels or []),
        "phase3_micro_optuna_tel": dict(phase2.phase3_micro_optuna_tel or {}),
        "phase2_rollup_tel": dict(phase2_rollup_tel or {}),
    }


def _run_phase1_search(
    *,
    search_base_data: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    hc_f,
    hc_m,
    pin_obj,
    status_cb,
    cfg,
    goal: str,
    filter_key: str,
    optimizer_backend: str,
    optuna_mod,
    seed: int,
    optuna_search_sig: str,
    status_prefix: str,
    winner_target_name: str | None,
    search_state,
    n_trials_eff: int,
    candidates: list[dict],
    prior_seed_preset: dict | None,
    use_optuna_trials: bool,
    runtime=None,
) -> _SearchRefineContext:
    return _SearchRefineContext(
        params={
            "search_base_data": dict(search_base_data or {}),
            "measurements": dict(measurements or {}),
            "fs_v": int(fs_v),
            "taps_v": int(taps_v),
            "xos": list(xos or []),
            "hpf": hpf,
            "hc_f": hc_f,
            "hc_m": hc_m,
            "pin_obj": pin_obj,
            "status_cb": status_cb,
            "cfg": cfg,
            "goal": str(goal),
            "filter_key": str(filter_key),
            "optimizer_backend": str(optimizer_backend),
            "optuna_mod": optuna_mod,
            "seed": int(seed),
            "optuna_search_sig": str(optuna_search_sig),
            "status_prefix": str(status_prefix),
            "winner_target_name": winner_target_name,
            "search_state": search_state,
            "n_trials_eff": int(n_trials_eff),
            "candidates": list(candidates or []),
            "prior_seed_preset": dict(prior_seed_preset or {}) if isinstance(prior_seed_preset, dict) else prior_seed_preset,
            "use_optuna_trials": bool(use_optuna_trials),
            "runtime": runtime,
        }
    )


def _run_phase1_coarse_search(
    *,
    context: _SearchRefineContext,
    skip: bool = False,
) -> _SearchRefineContext:
    params = dict(context.params or {})
    runtime = coerce_orchestrator_runtime(params.get("runtime"))
    ctx = _build_search_refine_eval_context(
        search_base_data=dict(params.get("search_base_data", {}) or {}),
        measurements=dict(params.get("measurements", {}) or {}),
        fs_v=int(params.get("fs_v", 0) or 0),
        taps_v=int(params.get("taps_v", 0) or 0),
        xos=list(params.get("xos", []) or []),
        hpf=params.get("hpf"),
        hc_f=params.get("hc_f"),
        hc_m=params.get("hc_m"),
        pin_obj=params.get("pin_obj"),
        status_cb=params.get("status_cb"),
        cfg=params.get("cfg"),
        goal=str(params.get("goal", "") or ""),
        filter_key=str(params.get("filter_key", "") or ""),
        optimizer_backend=str(params.get("optimizer_backend", "") or ""),
        optuna_mod=params.get("optuna_mod"),
        seed=int(params.get("seed", 0) or 0),
        optuna_search_sig=str(params.get("optuna_search_sig", "") or ""),
        status_prefix=str(params.get("status_prefix", "") or ""),
        winner_target_name=params.get("winner_target_name"),
        search_state=params.get("search_state"),
        runtime=runtime,
    )
    params["runtime"] = runtime
    if skip:
        # Phase 1 trials skipped per search_v2 plan; ctx still built for micro refine.
        logger.info("Automatic mode: Phase 1 skipped (search_v2 plan decision).")
        params["_phase1_state"] = _SearchPhase1State(ctx=ctx)
    else:
        params["_phase1_state"] = _run_search_refine_phase1_core(
            search_base_data=dict(params.get("search_base_data", {}) or {}),
            candidates=list(params.get("candidates", []) or []),
            prior_seed_preset=params.get("prior_seed_preset"),
            use_optuna_trials=bool(params.get("use_optuna_trials", False)),
            cfg=params.get("cfg"),
            filter_key=str(params.get("filter_key", "") or ""),
            seed=int(params.get("seed", 0) or 0),
            n_trials_eff=int(params.get("n_trials_eff", 0) or 0),
            runtime=runtime,
            status_cb=params.get("status_cb"),
            ctx=ctx,
        )
    return _SearchRefineContext(params=params)


def _run_phase2_local_refine(
    *,
    context: _SearchRefineContext,
) -> _SearchRefineContext:
    params = dict(context.params or {})
    runtime = coerce_orchestrator_runtime(params.get("runtime"))
    phase1 = params.get("_phase1_state")
    phase2 = _run_search_refine_phase2_local_core(
        search_base_data=dict(params.get("search_base_data", {}) or {}),
        cfg=params.get("cfg"),
        goal=str(params.get("goal", "") or ""),
        filter_key=str(params.get("filter_key", "") or ""),
        seed=int(params.get("seed", 0) or 0),
        winner_target_name=params.get("winner_target_name"),
        use_optuna_trials=bool(params.get("use_optuna_trials", False)),
        status_cb=params.get("status_cb"),
        search_state=params.get("search_state"),
        runtime=runtime,
        phase1=phase1,
    )
    _carry_forward_phase1_best_core(
        cfg=params.get("cfg"),
        winner_target_name=params.get("winner_target_name"),
        search_state=params.get("search_state"),
        phase1=phase1,
    )
    params["runtime"] = runtime
    params["_phase2_state"] = phase2
    return _SearchRefineContext(params=params)


def _run_phase3_micro_refine(
    *,
    context: _SearchRefineContext,
) -> _SearchRefineSummary:
    params = dict(context.params or {})
    runtime = coerce_orchestrator_runtime(params.get("runtime"))
    phase1 = params.get("_phase1_state")
    phase2 = params.get("_phase2_state")
    phase2 = _run_search_refine_micro_core(
        search_base_data=dict(params.get("search_base_data", {}) or {}),
        cfg=params.get("cfg"),
        goal=str(params.get("goal", "") or ""),
        filter_key=str(params.get("filter_key", "") or ""),
        winner_target_name=params.get("winner_target_name"),
        use_optuna_trials=bool(params.get("use_optuna_trials", False)),
        status_cb=params.get("status_cb"),
        search_state=params.get("search_state"),
        runtime=runtime,
        phase1=phase1,
        phase2=phase2,
    )
    result = _build_search_refine_result(
        runtime=runtime,
        status_cb=params.get("status_cb"),
        phase1=phase1,
        phase2=phase2,
    )
    return _SearchRefineSummary(result=dict(result or {}))


def _assemble_refine_summary(
    *,
    summary: _SearchRefineSummary,
) -> dict:
    return dict(summary.result or {})


def run_search_refine_stages(
    *,
    search_base_data: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    hc_f,
    hc_m,
    pin_obj,
    status_cb,
    cfg,
    goal: str,
    filter_key: str,
    optimizer_backend: str,
    optuna_mod,
    seed: int,
    optuna_search_sig: str,
    status_prefix: str,
    winner_target_name: str | None,
    search_state,
    n_trials_eff: int,
    candidates: list[dict],
    prior_seed_preset: dict | None,
    use_optuna_trials: bool,
    runtime=None,
    skip_phase1: bool = False,
) -> dict:
    phase_context = _run_phase1_search(
        search_base_data=search_base_data,
        measurements=measurements,
        fs_v=int(fs_v),
        taps_v=int(taps_v),
        xos=xos,
        hpf=hpf,
        hc_f=hc_f,
        hc_m=hc_m,
        pin_obj=pin_obj,
        status_cb=status_cb,
        cfg=cfg,
        goal=str(goal),
        filter_key=str(filter_key),
        optimizer_backend=str(optimizer_backend),
        optuna_mod=optuna_mod,
        seed=int(seed),
        optuna_search_sig=str(optuna_search_sig),
        status_prefix=str(status_prefix),
        winner_target_name=winner_target_name,
        search_state=search_state,
        n_trials_eff=int(n_trials_eff),
        candidates=list(candidates or []),
        prior_seed_preset=prior_seed_preset,
        use_optuna_trials=bool(use_optuna_trials),
        runtime=runtime,
    )
    if bool(skip_phase1):
        phase1 = _run_phase1_coarse_search(context=phase_context, skip=True)
    else:
        phase1 = _run_phase1_coarse_search(context=phase_context)
    phase2 = _run_phase2_local_refine(context=phase1)
    micro = _run_phase3_micro_refine(context=phase2)
    return _assemble_refine_summary(summary=micro)
