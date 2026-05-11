# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Cache-refine round execution and public API."""

from __future__ import annotations

import logging

from .candidate_generation import (
    _seed_auto_mode_candidate_micro_optuna_params,
    _suggest_auto_mode_candidate_micro_optuna,
)
from .runtime_context import coerce_orchestrator_runtime
from .shared import _auto_safe_float
from ._refine_types import (
    _CacheRefineContext,
    _CacheRefineOutcome,
    _CacheRefineProgress,
    _CacheRefineRound,
    _CacheRefineSeed,
)
from ._refine_cache_setup import (
    _build_exact_cache_refine_context,
    _build_cache_refine_progress,
    _cache_trial_issue_label,
    _cache_trial_log_method,
    _load_exact_cache_seed,
    _log_cache_refine_round_summary,
    _prepare_cache_refine_round,
    _record_cache_refine_candidate,
    _remember_cache_refine_round_seed,
)

logger = logging.getLogger("DecayCore")


def _run_optuna_cache_refine_round(
    *,
    context: _CacheRefineContext,
    progress: _CacheRefineProgress,
    round_state: _CacheRefineRound,
    runtime,
) -> _CacheRefineRound:
    params = dict(context.params or {})
    cache_base_data = dict(params.get("cache_base_data", {}) or {})
    status_cb = params.get("status_cb")
    cfg = params.get("cfg")
    goal = str(params.get("goal", "") or "")
    optuna_mod = params.get("optuna_mod")
    seed = int(params.get("seed", 0) or 0)
    optuna_search_sig = str(params.get("optuna_search_sig", "") or "")
    _cache_ready_preset = params.get("_cache_ready_preset")
    _materialize_preset_result = params.get("_materialize_preset_result")

    if callable(status_cb):
        status_cb(
            "DecayCore automatic mode: cache refine round "
            f"{int(round_state.round_idx)}/{int(runtime.cache_refine_max_rounds)} "
            f"(optuna {int(progress.micro_trials)} trials)"
        )

    def _cache_eval_one(idx: int, preset: dict) -> dict:
        _res_i, met_i, _data_i = _materialize_preset_result(
            preset,
            include_response_arrays=False,
            summarize=False,
        )
        return {
            "idx": int(idx),
            "ok": True,
            "metrics": dict(met_i or {}),
            "trial_preset": dict(preset or {}),
            "phase": (
                "target_preselect_micro_refine"
                if str(progress.seed_source or "") == "target_preselect"
                else "exact_cache_micro_refine"
            ),
            "round": int(round_state.round_idx),
            "seed_source": str(progress.seed_source or "exact_cache"),
        }

    def _cache_consume_one(idx: int, out: dict) -> bool:
        round_state.round_executed += 1
        progress.executed_micro_trials_total += 1
        if not bool(dict(out or {}).get("ok", False)):
            err_txt = str((out or {}).get("error", "unknown error") or "unknown error")
            _cache_trial_log_method(out=dict(out or {}))(
                "Automatic mode cache refine round %d trial %d/%d %s: %s",
                int(round_state.round_idx),
                int(idx),
                int(progress.micro_trials),
                _cache_trial_issue_label(out=dict(out or {})),
                str(err_txt),
            )
            return False
        met_i = dict((out or {}).get("metrics", {}) or {})
        cand = dict((out or {}).get("trial_preset", {}) or {})
        _record_cache_refine_candidate(
            progress=progress,
            round_state=round_state,
            cand=cand,
            metrics=met_i,
            goal=goal,
            total_trials=int(progress.micro_trials),
            status_cb=status_cb,
            _cache_ready_preset=_cache_ready_preset,
        )
        return False

    round_seed = int(seed + 700000 + round_state.round_idx * 1009)
    round_state.round_tel = dict(
        runtime.auto_run_optuna_eval_loop(
            optuna_mod=optuna_mod,
            cfg=cfg,
            n_total=int(progress.micro_trials),
            seed=int(round_seed),
            base_data=dict(cache_base_data or {}),
            seed_presets=list(round_state.round_seed_presets or []),
            build_preset=(
                lambda tr,
                _base=dict(cache_base_data),
                _center=dict(round_state.round_start_preset or {}): _suggest_auto_mode_candidate_micro_optuna(
                    _base,
                    _center,
                    tr,
                    shrink=1.0,
                )
            ),
            eval_one=_cache_eval_one,
            consume_one=_cache_consume_one,
            objective_value=lambda out, _goal=str((cache_base_data or {}).get("auto_goal", "") or ""): runtime.auto_optuna_objective_value(
                dict((out or {}).get("metrics", {}) or {}),
                use_refine_tiebreak=True,
                goal=_goal,
            ),
            workers=int(runtime.auto_trial_workers(cache_base_data, int(progress.micro_trials))),
            seed_to_params=(
                lambda preset,
                _base=dict(cache_base_data),
                _center=dict(round_state.round_start_preset or {}): _seed_auto_mode_candidate_micro_optuna_params(
                    _base,
                    _center,
                    preset,
                    shrink=1.0,
                )
            ),
            study_name=runtime.auto_optuna_study_name(
                study_sig=optuna_search_sig,
                scope=runtime.auto_optuna_effective_scope(
                    cache_base_data,
                    round_state.raw_scope,
                    phase_kind="micro",
                ),
            ),
            study_scope=round_state.raw_scope,
            phase_label=f"cache refine round {int(round_state.round_idx)}/{int(runtime.cache_refine_max_rounds)}",
            phase_kind="micro",
        )
        or {}
    )
    if round_state.round_tel:
        progress.cache_refine_optuna_tels.append(dict(round_state.round_tel))
    return round_state


def _run_builtin_cache_refine_round(
    *,
    context: _CacheRefineContext,
    progress: _CacheRefineProgress,
    round_state: _CacheRefineRound,
    runtime,
) -> _CacheRefineRound:
    params = dict(context.params or {})
    cache_base_data = dict(params.get("cache_base_data", {}) or {})
    status_cb = params.get("status_cb")
    goal = str(params.get("goal", "") or "")
    optimizer_backend = str(params.get("optimizer_backend", "") or "")
    optuna_mod = params.get("optuna_mod")
    seed = int(params.get("seed", 0) or 0)
    optuna_search_sig = str(params.get("optuna_search_sig", "") or "")
    _cache_ready_preset = params.get("_cache_ready_preset")
    _materialize_preset_result = params.get("_materialize_preset_result")

    micro_candidates = runtime.build_auto_mode_candidates_micro(
        cache_base_data,
        dict(round_state.round_start_preset or {}),
        n_trials=int(progress.micro_trials + 1),
        shrink=1.0,
    )
    micro_candidates = [
        dict(cand or {})
        for cand in list(micro_candidates or [])
        if isinstance(cand, dict)
        and dict(cand or {}) != dict(round_state.round_start_preset or {})
    ][: int(progress.micro_trials)]
    if len(micro_candidates) < int(progress.micro_trials):
        logger.info(
            "Automatic mode cache refine round %d: generated %d/%d micro candidates.",
            int(round_state.round_idx),
            int(len(micro_candidates)),
            int(progress.micro_trials),
        )
    if callable(status_cb):
        status_cb(
            "DecayCore automatic mode: cache refine round "
            f"{int(round_state.round_idx)}/{int(runtime.cache_refine_max_rounds)} "
            f"({int(len(micro_candidates))}/{int(progress.micro_trials)} candidates)"
        )

    for idx, cand in enumerate(micro_candidates, start=1):
        _res_i, met_i, _data_i = _materialize_preset_result(
            cand,
            include_response_arrays=False,
            summarize=False,
        )
        if bool(str(optimizer_backend) == "optuna" and optuna_mod is not None):
            runtime.auto_optuna_remember_result(
                optuna_mod,
                base_data=dict(cache_base_data or {}),
                study_name=runtime.auto_optuna_study_name(
                    study_sig=optuna_search_sig,
                    scope=runtime.auto_optuna_effective_scope(
                        cache_base_data,
                        round_state.raw_scope,
                        phase_kind="micro",
                    ),
                ),
                study_scope=round_state.raw_scope,
                phase_kind="micro",
                seed=int(seed + 700000 + round_state.round_idx * 1009 + idx),
                preset=dict(cand or {}),
                metrics=dict(met_i or {}),
                seed_to_params=(
                    lambda preset,
                    _base=dict(cache_base_data),
                    _center=dict(round_state.round_start_preset or {}): _seed_auto_mode_candidate_micro_optuna_params(
                        _base,
                        _center,
                        preset,
                        shrink=1.0,
                    )
                ),
                use_refine_tiebreak=True,
                out_payload={
                    "idx": int(idx),
                    "ok": True,
                    "metrics": dict(met_i or {}),
                    "trial_preset": dict(cand or {}),
                    "phase": (
                        "target_preselect_micro_refine"
                        if str(progress.seed_source or "") == "target_preselect"
                        else "exact_cache_micro_refine"
                    ),
                    "round": int(round_state.round_idx),
                    "seed_source": str(progress.seed_source or "exact_cache"),
                },
            )
        round_state.round_executed += 1
        progress.executed_micro_trials_total += 1
        _record_cache_refine_candidate(
            progress=progress,
            round_state=round_state,
            cand=dict(cand or {}),
            metrics=dict(met_i or {}),
            goal=goal,
            total_trials=int(max(1, len(micro_candidates))),
            status_cb=status_cb,
            _cache_ready_preset=_cache_ready_preset,
        )
    return round_state


def _run_cache_refine_rounds(
    *,
    context: _CacheRefineContext,
    seed_state: _CacheRefineSeed,
) -> _CacheRefineOutcome:
    params = dict(context.params or {})
    runtime = coerce_orchestrator_runtime(params.get("runtime"))
    status_cb = params.get("status_cb")
    optimizer_backend = str(params.get("optimizer_backend", "") or "")
    optuna_mod = params.get("optuna_mod")

    progress = _build_cache_refine_progress(
        seed_state=seed_state,
        runtime=runtime,
    )
    if callable(status_cb):
        status_cb(
            "DecayCore automatic mode: cache refine init "
            f"(rounds up to {int(runtime.cache_refine_max_rounds)}, "
            f"{int(progress.micro_trials)} trials/round, "
            f"min rank improvement {float(progress.min_round_improvement):.2f})"
        )

    for round_idx in range(1, int(max(1, runtime.cache_refine_max_rounds)) + 1):
        progress.rounds_executed = int(round_idx)
        round_state = _prepare_cache_refine_round(
            context=context,
            progress=progress,
            round_idx=int(round_idx),
            runtime=runtime,
        )
        _remember_cache_refine_round_seed(
            context=context,
            progress=progress,
            round_state=round_state,
            runtime=runtime,
        )
        if bool(str(optimizer_backend) == "optuna" and runtime.auto_optuna_module_ready(optuna_mod)):
            round_state = _run_optuna_cache_refine_round(
                context=context,
                progress=progress,
                round_state=round_state,
                runtime=runtime,
            )
        else:
            round_state = _run_builtin_cache_refine_round(
                context=context,
                progress=progress,
                round_state=round_state,
                runtime=runtime,
            )
        round_delta = _log_cache_refine_round_summary(
            progress=progress,
            round_state=round_state,
            runtime=runtime,
            status_cb=status_cb,
        )
        if round_state.round_improved_count <= 0:
            progress.stop_reason = "no_improvement"
            break
        if float(round_delta) < float(progress.min_round_improvement):
            progress.stop_reason = "below_threshold"
            break

    winner_changed = bool(dict(progress.best_preset or {}) != dict(progress.initial_best_preset or {}))
    cache_refine_rollup_tel = runtime.auto_optuna_telemetry_rollup(progress.cache_refine_optuna_tels)
    logger.info(
        "Automatic mode cache refine summary: rounds=%d/%d, executed %d/%d micro-trials, improvements=%d, winner_changed=%s, stop_reason=%s, final_rank=%.3f, final_avg=%.3f",
        int(progress.rounds_executed),
        int(runtime.cache_refine_max_rounds),
        int(progress.executed_micro_trials_total),
        int(progress.micro_trials * max(1, progress.rounds_executed)),
        int(progress.improved_count_total),
        str(bool(winner_changed)).lower(),
        str(progress.stop_reason),
        _auto_safe_float(dict(progress.best_metrics or {}).get("rank_score"), 0.0),
        _auto_safe_float(dict(progress.best_metrics or {}).get("avg_score"), 0.0),
    )
    if callable(status_cb):
        status_cb(
            "DecayCore automatic mode: cache refine summary "
            f"(rounds {int(progress.rounds_executed)}/{int(runtime.cache_refine_max_rounds)}, "
            f"executed {int(progress.executed_micro_trials_total)} trials, "
            f"improvements {int(progress.improved_count_total)}, "
            f"winner {'changed' if bool(winner_changed) else 'unchanged'}, "
            f"stop {str(progress.stop_reason)})"
        )
    result = {
        "cache_target_name": str(progress.cache_target_name),
        "seed_source": str(progress.seed_source or "exact_cache"),
        "best_preset": dict(progress.best_preset or {}),
        "best_metrics": dict(progress.best_metrics or {}),
        "improved_any": bool(progress.improved_any),
        "improved_count_total": int(progress.improved_count_total),
        "executed_micro_trials_total": int(progress.executed_micro_trials_total),
        "cache_refine_rollup_tel": dict(cache_refine_rollup_tel or {}),
        "stop_reason": str(progress.stop_reason),
    }
    return _CacheRefineOutcome(
        result=result,
        best_preset=dict(progress.best_preset or {}),
        best_metrics=dict(progress.best_metrics or {}),
    )


def _select_cache_refine_best(
    *,
    outcome: _CacheRefineOutcome,
) -> _CacheRefineOutcome:
    return outcome


def _finalize_cache_refine_result(
    *,
    outcome: _CacheRefineOutcome,
) -> dict | None:
    if not isinstance(outcome.result, dict):
        return None
    return dict(outcome.result or {})


def _execute_exact_cache_refine(
    *,
    context: _CacheRefineContext,
) -> dict | None:
    seed_state = _load_exact_cache_seed(context=context)
    if seed_state is None:
        return None
    try:
        rounds = _run_cache_refine_rounds(
            context=context,
            seed_state=seed_state,
        )
        winner = _select_cache_refine_best(outcome=rounds)
        return _finalize_cache_refine_result(outcome=winner)
    except Exception as exc:
        logger.warning(
            "Automatic mode: exact preset cache refine failed, "
            f"falling back to search ({type(exc).__name__}: {exc})"
        )
        return None


def run_exact_cache_micro_refine(
    *,
    cache_base_data: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    status_cb,
    cfg,
    goal: str,
    filter_key: str,
    compat_version: str,
    optimizer_backend: str,
    optuna_mod,
    seed: int,
    optuna_search_sig: str,
    _cache_ready_preset,
    _materialize_preset_result,
    seed_preset: dict | None = None,
    seed_metrics: dict | None = None,
    seed_source: str = "",
    runtime=None,
) -> dict | None:
    context = _build_exact_cache_refine_context(
        cache_base_data=cache_base_data,
        measurements=measurements,
        fs_v=int(fs_v),
        taps_v=int(taps_v),
        xos=xos,
        hpf=hpf,
        status_cb=status_cb,
        cfg=cfg,
        goal=str(goal),
        filter_key=str(filter_key),
        compat_version=str(compat_version),
        optimizer_backend=str(optimizer_backend),
        optuna_mod=optuna_mod,
        seed=int(seed),
        optuna_search_sig=str(optuna_search_sig),
        seed_preset=dict(seed_preset or {}),
        seed_metrics=dict(seed_metrics or {}),
        seed_source=str(seed_source or ""),
        _cache_ready_preset=_cache_ready_preset,
        _materialize_preset_result=_materialize_preset_result,
        runtime=runtime,
    )
    return _execute_exact_cache_refine(context=context)
