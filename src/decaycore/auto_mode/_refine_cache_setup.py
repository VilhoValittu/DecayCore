# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Cache-refine context building, seed loading, and round preparation."""

from __future__ import annotations

import logging

from .cache_signature import _auto_signature
from .candidate_generation import _seed_auto_mode_candidate_micro_optuna_params
from .rank_score import official_rank_score
from .runtime_context import coerce_orchestrator_runtime
from .scoring_ranking import _auto_is_better_refine
from .shared import _auto_safe_float
from ._refine_types import (
    _CacheRefineContext,
    _CacheRefineProgress,
    _CacheRefineRound,
    _CacheRefineSeed,
)

logger = logging.getLogger("DecayCore")


def _cache_trial_log_method(*, out: dict):
    return logger.info if bool(dict(out or {}).get("pruned", False)) else logger.warning


def _cache_trial_issue_label(*, out: dict) -> str:
    return "pruned" if bool(dict(out or {}).get("pruned", False)) else "failed"


def _build_exact_cache_refine_context(
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
) -> _CacheRefineContext:
    return _CacheRefineContext(
        params={
            "cache_base_data": dict(cache_base_data or {}),
            "measurements": dict(measurements or {}),
            "fs_v": int(fs_v),
            "taps_v": int(taps_v),
            "xos": list(xos or []),
            "hpf": hpf,
            "status_cb": status_cb,
            "cfg": cfg,
            "goal": str(goal),
            "filter_key": str(filter_key),
            "compat_version": str(compat_version),
            "optimizer_backend": str(optimizer_backend),
            "optuna_mod": optuna_mod,
            "seed": int(seed),
            "optuna_search_sig": str(optuna_search_sig),
            "seed_preset": dict(seed_preset or {}),
            "seed_metrics": dict(seed_metrics or {}),
            "seed_source": str(seed_source or ""),
            "_cache_ready_preset": _cache_ready_preset,
            "_materialize_preset_result": _materialize_preset_result,
            "runtime": runtime,
        }
    )


def _load_exact_cache_seed(
    *,
    context: _CacheRefineContext,
) -> _CacheRefineSeed | None:
    params = dict(context.params or {})
    runtime = coerce_orchestrator_runtime(params.get("runtime"))
    cache_base_data = dict(params.get("cache_base_data", {}) or {})
    measurements = dict(params.get("measurements", {}) or {})
    fs_v = int(params.get("fs_v", 0) or 0)
    taps_v = int(params.get("taps_v", 0) or 0)
    xos = list(params.get("xos", []) or [])
    hpf = params.get("hpf")
    status_cb = params.get("status_cb")
    cfg = params.get("cfg")
    filter_key = str(params.get("filter_key", "") or "")
    compat_version = str(params.get("compat_version", "") or "")
    _cache_ready_preset = params.get("_cache_ready_preset")
    _materialize_preset_result = params.get("_materialize_preset_result")
    optimizer_backend = str(params.get("optimizer_backend", "") or "")
    optuna_mod = params.get("optuna_mod")
    optuna_search_sig = str(params.get("optuna_search_sig", "") or "")

    explicit_seed_preset = dict(params.get("seed_preset", {}) or {})
    explicit_seed_source = str(params.get("seed_source", "") or "").strip() or "target_preselect"
    if explicit_seed_preset:
        try:
            explicit_seed_metrics = dict(params.get("seed_metrics", {}) or {})
            best_preset = _cache_ready_preset(
                explicit_seed_preset,
                best_metrics=explicit_seed_metrics,
            )
            best_metrics = dict(explicit_seed_metrics or {})
            if not best_metrics:
                _best_result, best_metrics, _best_data = _materialize_preset_result(
                    best_preset,
                    include_response_arrays=False,
                    summarize=False,
                )
            if not (isinstance(best_preset, dict) and best_preset and isinstance(best_metrics, dict) and best_metrics):
                raise ValueError("empty target seed preset or metrics")
            cache_target_name = str(cache_base_data.get("hc_mode", "n/a") or "n/a").strip() or "n/a"
            logger.info(
                "Automatic mode: target preselect seed loaded for target=%s, running up to %d x %d micro-trials around selected winner.",
                str(cache_target_name),
                int(runtime.cache_refine_max_rounds),
                int(runtime.cache_refine_micro_trials),
            )
            if callable(status_cb):
                status_cb(
                    "DecayCore automatic mode: target preselect seed loaded "
                    f"(target {cache_target_name}, running up to "
                    f"{int(runtime.cache_refine_max_rounds)} x "
                    f"{int(runtime.cache_refine_micro_trials)} micro-trials)"
                )
            return _CacheRefineSeed(
                cache_target_name=str(cache_target_name),
                best_preset=dict(best_preset or {}),
                best_metrics=dict(best_metrics or {}),
                seed_source=str(explicit_seed_source),
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
        ) as exc:
            logger.warning(
                "Automatic mode: target preselect seed micro-refine setup failed, "
                f"falling back to cache/search ({type(exc).__name__}: {exc})"
            )

    exact_cached_preset = {}
    exact_cached_metrics = {}
    seed_source = "exact_cache"
    if bool(getattr(cfg, "cache_enabled", False)):
        try:
            exact_cache_sig = _auto_signature(
                base_data=cache_base_data,
                measurements=measurements,
                fs_v=int(fs_v),
                taps_v=int(taps_v),
                xos=xos,
                hpf=hpf,
                hc_mode=str(cache_base_data.get("hc_mode", "") or "").strip() or None,
                include_hc_mode=True,
            )
            exact_cached_entry = runtime.auto_cache_get_entry(
                exact_cache_sig,
                filter_key=filter_key,
                compat_version=compat_version,
            ) or {}
            exact_cached_preset = runtime.auto_cache_get_best(
                exact_cache_sig,
                filter_key=filter_key,
                compat_version=compat_version,
            ) or {}
            exact_cached_metrics = dict((exact_cached_entry or {}).get("best_metrics", {}) or {})
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
        ) as exc:
            logger.debug(
                "Auto-mode cache read failed, disabling fast path: %s: %s",
                type(exc).__name__,
                exc,
            )
            exact_cached_preset = {}
            exact_cached_metrics = {}

    if not (isinstance(exact_cached_preset, dict) and exact_cached_preset):
        if not (
            str(optimizer_backend) == "optuna"
            and bool(getattr(cfg, "optuna_persistent_study", False))
            and optuna_mod is not None
            and runtime.auto_optuna_module_ready(optuna_mod)
        ):
            return None
        try:
            storage = runtime.auto_optuna_create_storage(
                optuna_mod,
                base_data=dict(cache_base_data or {}),
            )
            study_name = runtime.auto_optuna_study_name(
                study_sig=str(optuna_search_sig),
                scope=runtime.auto_optuna_effective_scope(
                    cache_base_data,
                    "phase1",
                    phase_kind="phase1",
                ),
            )
            study = optuna_mod.load_study(study_name=str(study_name), storage=storage)
            best_trial = getattr(study, "best_trial", None)
            exact_cached_preset = runtime.auto_optuna_trial_payload_preset(
                dict(getattr(best_trial, "user_attrs", {}) or {})
            )
            if not isinstance(exact_cached_preset, dict) or not exact_cached_preset:
                exact_cached_preset = dict(getattr(best_trial, "params", {}) or {})
            best_out = runtime.auto_optuna_trial_out_payload(best_trial)
            exact_cached_metrics = dict((best_out or {}).get("metrics", {}) or {})
            if isinstance(exact_cached_preset, dict) and exact_cached_preset:
                logger.info(
                    "Automatic mode: Optuna phase1 study cache hit for same measurements + settings, using cached target=%s from %s and running cache refine.",
                    str(cache_base_data.get("hc_mode", "n/a") or "n/a"),
                    str(study_name),
                )
                seed_source = "optuna_phase1_study"
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
            exact_cached_preset = {}
            exact_cached_metrics = {}
        if not (isinstance(exact_cached_preset, dict) and exact_cached_preset):
            return None
    try:
        cache_target_name = str(cache_base_data.get("hc_mode", "n/a") or "n/a").strip() or "n/a"
        best_preset = _cache_ready_preset(
            exact_cached_preset,
            best_metrics=exact_cached_metrics,
        )
        logger.info(
            "Automatic mode: exact preset cache hit for same measurements + settings, using cached target=%s and running up to %d x %d extra micro-trials around cached winner.",
            cache_target_name,
            int(runtime.cache_refine_max_rounds),
            int(runtime.cache_refine_micro_trials),
        )
        if callable(status_cb):
            status_cb(
                "DecayCore automatic mode: preset loaded from cache "
                f"(same measurements + settings, target {cache_target_name}, "
                f"running up to {int(runtime.cache_refine_max_rounds)} x "
                f"{int(runtime.cache_refine_micro_trials)} extra micro-trials)"
            )
        best_metrics = dict(exact_cached_metrics or {})
        if not best_metrics:
            _best_result, best_metrics, _best_data = _materialize_preset_result(
                best_preset,
                include_response_arrays=False,
                summarize=False,
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
    ) as exc:
        logger.warning(
            "Automatic mode: exact preset cache materialization failed, "
            f"falling back to search ({type(exc).__name__}: {exc})"
        )
        return None
    return _CacheRefineSeed(
        cache_target_name=str(cache_target_name),
        best_preset=dict(best_preset or {}),
        best_metrics=dict(best_metrics or {}),
        seed_source=str(seed_source),
    )


def _build_cache_refine_progress(
    *,
    seed_state: _CacheRefineSeed,
    runtime,
) -> _CacheRefineProgress:
    return _CacheRefineProgress(
        cache_target_name=str(seed_state.cache_target_name),
        seed_source=str(seed_state.seed_source or "exact_cache"),
        best_preset=dict(seed_state.best_preset or {}),
        best_metrics=dict(seed_state.best_metrics or {}),
        initial_best_preset=dict(seed_state.best_preset or {}),
        micro_trials=int(max(1, runtime.cache_refine_micro_trials)),
        min_round_improvement=float(
            max(0.0, _auto_safe_float(runtime.cache_refine_min_rank_improvement, 0.02))
        ),
    )


def _prepare_cache_refine_round(
    *,
    context: _CacheRefineContext,
    progress: _CacheRefineProgress,
    round_idx: int,
    runtime,
) -> _CacheRefineRound:
    params = dict(context.params or {})
    cache_base_data = dict(params.get("cache_base_data", {}) or {})
    filter_key = str(params.get("filter_key", "") or "")
    round_start_preset = dict(progress.best_preset or {})
    scope_name = (
        "phase3-micro-u1-target-preselect"
        if str(progress.seed_source or "") == "target_preselect"
        else "phase3-micro-u1-cache"
    )
    raw_scope = runtime.auto_optuna_scope_with_context(
        str(scope_name),
        center=dict(round_start_preset or {}),
        shrink=1.0,
        extra={
            "filter_key": str(filter_key),
            "round": int(round_idx),
            "seed_source": str(progress.seed_source or "exact_cache"),
        },
    )
    round_seed_presets = runtime.build_auto_mode_candidates_micro(
        cache_base_data,
        dict(round_start_preset or {}),
        n_trials=1,
        shrink=1.0,
    )
    return _CacheRefineRound(
        round_idx=int(round_idx),
        round_start_metrics=dict(progress.best_metrics or {}),
        round_start_preset=dict(round_start_preset or {}),
        raw_scope=raw_scope,
        round_seed_presets=list(round_seed_presets or []),
    )


def _remember_cache_refine_round_seed(
    *,
    context: _CacheRefineContext,
    progress: _CacheRefineProgress,
    round_state: _CacheRefineRound,
    runtime,
) -> None:
    params = dict(context.params or {})
    optimizer_backend = str(params.get("optimizer_backend", "") or "")
    optuna_mod = params.get("optuna_mod")
    if not (str(optimizer_backend) == "optuna" and optuna_mod is not None):
        return
    cache_base_data = dict(params.get("cache_base_data", {}) or {})
    optuna_search_sig = str(params.get("optuna_search_sig", "") or "")
    seed = int(params.get("seed", 0) or 0)
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
        seed=int(seed + 700000 + round_state.round_idx * 1009),
        preset=dict(round_state.round_start_preset or {}),
        metrics=dict(round_state.round_start_metrics or {}),
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
            "idx": 0,
            "ok": True,
            "metrics": dict(round_state.round_start_metrics or {}),
            "trial_preset": dict(round_state.round_start_preset or {}),
            "phase": (
                "target_preselect_micro_refine_seed"
                if str(progress.seed_source or "") == "target_preselect"
                else "exact_cache_micro_refine_seed"
            ),
            "round": int(round_state.round_idx),
            "seed_source": str(progress.seed_source or "exact_cache"),
        },
    )


def _record_cache_refine_candidate(
    *,
    progress: _CacheRefineProgress,
    round_state: _CacheRefineRound,
    cand: dict,
    metrics: dict,
    goal: str,
    total_trials: int,
    status_cb,
    _cache_ready_preset,
) -> bool:
    better, _reason = _auto_is_better_refine(
        dict(metrics or {}),
        dict(progress.best_metrics or {}),
        goal,
        return_reason=True,
    )
    if not better:
        return False
    prev_best = dict(progress.best_metrics or {})
    progress.best_metrics = dict(metrics or {})
    progress.best_preset = _cache_ready_preset(
        dict(cand or {}),
        best_metrics=progress.best_metrics,
    )
    progress.improved_any = True
    progress.improved_count_total += 1
    round_state.round_improved_count += 1
    logger.info(
        "Automatic mode cache refine improved: round %d trial %d/%d, rank_score %.3f -> %.3f, avg_score %.3f -> %.3f",
        int(round_state.round_idx),
        int(round_state.round_executed),
        int(total_trials),
        _auto_safe_float(prev_best.get("rank_score"), 0.0),
        _auto_safe_float(progress.best_metrics.get("rank_score"), 0.0),
        _auto_safe_float(prev_best.get("avg_score"), 0.0),
        _auto_safe_float(progress.best_metrics.get("avg_score"), 0.0),
    )
    if callable(status_cb):
        status_cb(
            "DecayCore automatic mode: cache refine best improved "
            f"(round {int(round_state.round_idx)}, {int(round_state.round_executed)}/{int(total_trials)}, "
            f"rank {official_rank_score(progress.best_metrics):.3f}, "
            f"avg {_auto_safe_float(progress.best_metrics.get('avg_score'), 0.0):.3f})"
        )
    return True


def _log_cache_refine_round_summary(
    *,
    progress: _CacheRefineProgress,
    round_state: _CacheRefineRound,
    runtime,
    status_cb,
) -> float:
    round_end_rank = _auto_safe_float(dict(progress.best_metrics or {}).get("rank_score"), 0.0)
    round_delta = float(
        round_end_rank - _auto_safe_float(round_state.round_start_metrics.get("rank_score"), 0.0)
    )
    round_winner_changed = bool(dict(progress.best_preset or {}) != dict(round_state.round_start_preset or {}))
    logger.info(
        "Automatic mode cache refine round %d summary: executed %d/%d, improvements=%d, winner_changed=%s, rank_delta=%.3f, final_rank=%.3f%s",
        int(round_state.round_idx),
        int(round_state.round_executed),
        int(progress.micro_trials),
        int(round_state.round_improved_count),
        str(bool(round_winner_changed)).lower(),
        float(round_delta),
        float(round_end_rank),
        "" if not round_state.round_tel else f", {runtime.auto_optuna_telemetry_text(round_state.round_tel)}",
    )
    if callable(status_cb):
        status_cb(
            "DecayCore automatic mode: cache refine round summary "
            f"(round {int(round_state.round_idx)}, executed {int(round_state.round_executed)}/{int(progress.micro_trials)}, "
            f"improvements {int(round_state.round_improved_count)}, delta {float(round_delta):.3f}"
            f"{'' if not round_state.round_tel else f', {runtime.auto_optuna_telemetry_text(round_state.round_tel)}'})"
        )
    return float(round_delta)
