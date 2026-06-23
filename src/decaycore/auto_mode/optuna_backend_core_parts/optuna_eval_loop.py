# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Optuna backend — core eval loop implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any

import numpy as np

from ...dsp._pruning import (
    clear_pruning_hook as _clear_pruning_hook,
    set_pruning_hook as _set_pruning_hook,
)
from ..cache_signature import _auto_cache_stats_snapshot
from ..shared import (
    AUTO_MODE_OPTUNA_CROSS_STUDY_SEEDS,
    AUTO_MODE_OPTUNA_CROSS_STUDY_SEEDS_TOP_N,
    AUTO_MODE_OPTUNA_DUPLICATE_MAX_ATTEMPTS,
    AUTO_MODE_OPTUNA_PRUNING_ENABLED,
    AUTO_MODE_OPTUNA_PRUNING_N_STARTUP,
    AUTO_MODE_OPTUNA_TELEMETRY,
    AUTO_MODE_OPTUNA_TELEMETRY_LOG_SUMMARY,
    AUTO_MODE_OPTUNA_USER_ATTR_OUT,
    AutoModeConfig,
    _auto_safe_bool,
    _auto_safe_float,
    _auto_safe_int,
    _auto_optuna_sampler_kwargs,
    _auto_trial_chunk_size,
)
from ..optuna_telemetry import _auto_optuna_log_run_telemetry
from ..optuna_backend_params import (
    _auto_optuna_cross_study_best_params,
    _auto_optuna_jsonable,
    _auto_optuna_param_signature,
    _auto_optuna_sanitize_enqueued_params,
    _auto_optuna_trial_params,
)
from ..optuna_backend_constraints import (
    _auto_optuna_constraint_thresholds,
    _auto_optuna_constraints_func,
    _auto_optuna_effective_scope,
    _auto_optuna_startup_for_phase_kind,
    _auto_optuna_use_events_constraint,
)
from ..optuna_backend_scoring import (
    _auto_optuna_attach_out_telemetry,
    _auto_optuna_build_run_telemetry,
    _auto_optuna_run_token,
    _auto_optuna_study_scan_stats_snapshot,
)
from ..optuna_backend_records import _auto_optuna_study_records
from ..optuna_backend_storage import (
    _OPTUNA_KNOWN_RECORDS,
    _OPTUNA_KNOWN_SIGNATURES_PRIMED,
    _auto_optuna_cached_study_records,
    _auto_optuna_create_study,
    _auto_optuna_get_known_signatures,
    _auto_optuna_module_ready,
    _auto_optuna_update_known_record,
)
from ..optuna_backend_loop import (
    _log_optuna_duplicate_summary,
    _run_optuna_seed_trials,
    _run_optuna_serial_trials,
    _run_optuna_parallel_trials,
)

logger = logging.getLogger("DecayCore")


@dataclass
class _OptunaRunSetup:
    total: int
    cfg_optuna: AutoModeConfig
    scope_eff: str
    startup_effective: int
    run_token: str
    sampler_kwargs: dict
    constraint_fn: object
    sampler: object
    pruner: object
    trial_pruned_cls: object
    pruned_state: object
    fail_state: object
    duplicate_guard: bool
    cache_stats_start: dict
    study_scan_stats_start: dict


@dataclass
class _OptunaDuplicateState:
    known_records: dict[str, dict] = field(default_factory=dict)
    reserved_signatures: set[str] = field(default_factory=set)
    duplicate_skips: int = 0
    duplicate_replays: int = 0
    duplicate_reserved: int = 0


@dataclass
class _OptunaEvalLoopContext:
    study: object
    base_data: dict | None
    study_name: str | None
    scope_eff: str
    phase_kind: str | None
    run_token: str
    phase_label: str | None
    total: int
    startup_effective: int
    cache_stats_start: dict
    study_scan_stats_start: dict
    trial_pruned_cls: object
    pruned_state: object
    fail_state: object
    pruner: object
    seed_to_params: Any
    build_preset: Any
    eval_one: Any
    consume_one: Any
    objective_value: Any
    duplicate_guard: bool
    duplicate_state: _OptunaDuplicateState
    known_records: dict[str, dict]
    reserved_signatures: set[str]


class _AutoOptunaPerRunStartupPruner:
    """Keep a per-run startup window when a persistent study already has trials."""

    def __init__(self, delegate, *, startup_trials: int):
        self.delegate = delegate
        self.startup_trials = int(max(1, startup_trials))
        self.existing_trials = 0

    def set_existing_trials(self, count: int) -> None:
        self.existing_trials = int(max(0, count))

    def prune(self, study, trial) -> bool:
        try:
            trial_number = int(getattr(trial, "number", -1))
        except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError, NameError):
            trial_number = -1
        if 0 <= int(trial_number) < int(self.existing_trials + self.startup_trials):
            return False
        return bool(self.delegate.prune(study, trial))


def _auto_optuna_existing_trial_count(study) -> int:
    try:
        trials = study.get_trials(deepcopy=False)
    except TypeError:
        trials = study.get_trials()
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError, NameError):
        trials = getattr(study, "trials", [])
    return int(len(list(trials or [])))


def _auto_optuna_prepare_run_setup(
    *,
    optuna_mod,
    cfg: AutoModeConfig | None,
    n_total: int,
    seed: int,
    startup_trials: int | None,
    base_data: dict | None,
    workers: int,
    study_name: str | None,
    study_scope: str | None,
    phase_kind: str | None,
) -> _OptunaRunSetup:
    total = int(max(0, n_total))
    cfg_optuna = cfg if isinstance(cfg, AutoModeConfig) else AutoModeConfig.from_base_data(base_data)
    scope_eff = _auto_optuna_effective_scope(base_data, study_scope or study_name, phase_kind=phase_kind)
    startup_effective = _auto_optuna_startup_for_phase_kind(
        cfg_optuna,
        phase_kind=phase_kind,
        total=int(total),
    )
    if startup_trials is not None and not str(phase_kind or "").strip():
        startup_effective = int(max(1, min(int(total), _auto_safe_int(startup_trials, startup_effective))))
    logger.info(
        "Automatic mode Optuna startup policy: phase_kind=%s scope=%s total=%d startup=%d",
        str(phase_kind or ""),
        str(study_scope or study_name or ""),
        int(total),
        int(startup_effective),
    )
    run_token = _auto_optuna_run_token(
        study_name=study_name,
        study_scope=scope_eff,
        seed=int(seed),
        total=int(total),
        startup_trials=int(startup_effective),
    )
    sampler_kwargs = dict(_auto_optuna_sampler_kwargs(base_data, workers=int(workers)) or {})
    constraint_fn = _auto_optuna_constraints_func(
        base_data=base_data,
        scope=scope_eff,
        phase_kind=phase_kind,
    )
    if callable(constraint_fn):
        sampler_kwargs["constraints_func"] = constraint_fn
    sampler = optuna_mod.samplers.TPESampler(
        seed=int(seed),
        n_startup_trials=int(startup_effective),
        **sampler_kwargs,
    )
    pruning_enabled = bool(
        _auto_safe_bool(
            (base_data or {}).get("auto_mode_optuna_pruning_enabled", AUTO_MODE_OPTUNA_PRUNING_ENABLED),
            AUTO_MODE_OPTUNA_PRUNING_ENABLED,
        )
    )
    pruner = None
    if bool(pruning_enabled):
        pruning_n_startup = max(
            1,
            _auto_safe_int(
                (base_data or {}).get("auto_mode_optuna_pruning_n_startup", AUTO_MODE_OPTUNA_PRUNING_N_STARTUP),
                AUTO_MODE_OPTUNA_PRUNING_N_STARTUP,
            ),
        )
        pruners_mod = getattr(optuna_mod, "pruners", None)
        median_pruner_cls = getattr(pruners_mod, "MedianPruner", None) if pruners_mod is not None else None
        if callable(median_pruner_cls):
            try:
                pruner = median_pruner_cls(
                    n_startup_trials=int(pruning_n_startup),
                    n_warmup_steps=0,
                    interval_steps=1,
                )
                if (
                    study_name
                    and _auto_safe_bool(
                        (base_data or {}).get("auto_mode_optuna_persistent_study", True),
                        True,
                    )
                ):
                    pruner = _AutoOptunaPerRunStartupPruner(
                        pruner,
                        startup_trials=int(startup_effective),
                    )
            except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError, NameError):
                pruner = None
    logger.info(
        "Automatic mode Optuna study %s: startup=%d total=%d pruning=%s",
        str(study_name or "in-memory"),
        int(startup_effective),
        int(total),
        "on" if pruner is not None else "off",
    )
    logger.info(
        "Automatic mode Optuna phase=%s scope=%s startup=%d total=%d",
        str(phase_kind or ""),
        str(study_scope or study_name or ""),
        int(startup_effective),
        int(total),
    )
    if callable(constraint_fn):
        thr = _auto_optuna_constraint_thresholds(base_data, scope_eff)
        use_events = _auto_optuna_use_events_constraint(
            base_data,
            phase_kind=phase_kind,
        )
        logger.info(
            "Automatic mode Optuna constraints enabled: scope=%s ripple<=%.3f events=%s boost<=%.3f",
            str(scope_eff),
            float(thr["max_mode_ripple_db"]),
            "off" if not bool(use_events) else f"{float(thr['max_events_severity']):.3f}",
            float(thr["max_net_boost_db"]),
        )
    trial_pruned_cls = getattr(optuna_mod, "TrialPruned", None)
    trial_pruned_state = getattr(
        getattr(optuna_mod, "trial", None),
        "TrialState",
        None,
    )
    pruned_state = getattr(trial_pruned_state, "PRUNED", None) if trial_pruned_state is not None else None
    fail_state = optuna_mod.trial.TrialState.FAIL
    duplicate_guard = bool(
        _auto_safe_bool((base_data or {}).get("auto_mode_optuna_avoid_duplicates", True), True)
    )
    return _OptunaRunSetup(
        total=int(total),
        cfg_optuna=cfg_optuna,
        scope_eff=str(scope_eff or ""),
        startup_effective=int(startup_effective),
        run_token=str(run_token),
        sampler_kwargs=dict(sampler_kwargs or {}),
        constraint_fn=constraint_fn,
        sampler=sampler,
        pruner=pruner,
        trial_pruned_cls=trial_pruned_cls,
        pruned_state=pruned_state,
        fail_state=fail_state,
        duplicate_guard=bool(duplicate_guard),
        cache_stats_start=_auto_cache_stats_snapshot(),
        study_scan_stats_start=_auto_optuna_study_scan_stats_snapshot(),
    )


def _auto_optuna_prepare_study(
    *,
    optuna_mod,
    setup: _OptunaRunSetup,
    base_data: dict | None,
    study_name: str | None,
    study_user_attrs: dict | None,
) -> object:
    study = _auto_optuna_create_study(
        optuna_mod,
        sampler=setup.sampler,
        pruner=setup.pruner,
        base_data=base_data,
        study_name=study_name,
    )
    if isinstance(setup.pruner, _AutoOptunaPerRunStartupPruner):
        existing_trials = _auto_optuna_existing_trial_count(study)
        setup.pruner.set_existing_trials(int(existing_trials))
        if int(existing_trials) > 0:
            logger.info(
                "Automatic mode Optuna pruning startup adjusted for persistent study: existing=%d current_startup=%d",
                int(existing_trials),
                int(setup.startup_effective),
            )
    _auto_optuna_apply_study_user_attrs(
        study=study,
        study_user_attrs=study_user_attrs,
    )
    _auto_optuna_maybe_enqueue_cross_study_seeds(
        study=study,
        optuna_mod=optuna_mod,
        setup=setup,
        base_data=base_data,
        study_name=study_name,
    )
    return study


def _auto_optuna_apply_study_user_attrs(
    *,
    study,
    study_user_attrs: dict | None,
) -> None:
    for attr_key, attr_value in dict(study_user_attrs or {}).items():
        try:
            study.set_user_attr(str(attr_key), attr_value)
        except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError, NameError):
            logger.debug("Optuna study user_attr set failed", exc_info=True)


def _auto_optuna_cross_seeding_enabled(*, study, base_data: dict | None, study_name: str | None) -> bool:
    return bool(
        _auto_safe_bool(
            (base_data or {}).get("auto_mode_optuna_cross_study_seeds", AUTO_MODE_OPTUNA_CROSS_STUDY_SEEDS),
            AUTO_MODE_OPTUNA_CROSS_STUDY_SEEDS,
        )
        and study_name
        and hasattr(study, "enqueue_trial")
    )


def _auto_optuna_has_existing_completed_trials(study) -> bool:
    try:
        existing_complete = [
            tr for tr in study.get_trials(deepcopy=False)
            if getattr(tr, "value", None) is not None
        ]
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError, NameError):
        existing_complete = []
    return bool(existing_complete)


def _auto_optuna_maybe_enqueue_cross_study_seeds(
    *,
    study,
    optuna_mod,
    setup: _OptunaRunSetup,
    base_data: dict | None,
    study_name: str | None,
) -> None:
    if not _auto_optuna_cross_seeding_enabled(study=study, base_data=base_data, study_name=study_name):
        return
    if _auto_optuna_has_existing_completed_trials(study):
        return
    top_n_cross = max(
        1,
        _auto_safe_int(
            (base_data or {}).get(
                "auto_mode_optuna_cross_study_seeds_top_n",
                AUTO_MODE_OPTUNA_CROSS_STUDY_SEEDS_TOP_N,
            ),
            AUTO_MODE_OPTUNA_CROSS_STUDY_SEEDS_TOP_N,
        ),
    )
    cross_params = _auto_optuna_cross_study_best_params(
        optuna_mod,
        base_data=base_data,
        scope=str(setup.scope_eff or ""),
        current_study_name=str(study_name),
        top_n=int(top_n_cross),
    )
    cross_enqueued = 0
    for cp in cross_params:
        try:
            study.enqueue_trial(_auto_optuna_sanitize_enqueued_params(dict(cp), base_data=base_data))
            cross_enqueued += 1
        except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError, NameError):
            logger.exception("optuna cross-study trial enqueue")
    if cross_enqueued:
        logger.info(
            "Automatic mode cross-study seeds: enqueued %d trials from sibling studies (scope=%s)",
            cross_enqueued,
            str(setup.scope_eff or ""),
        )


def _auto_optuna_prepare_duplicate_state(
    *,
    setup: _OptunaRunSetup,
    study,
    study_name: str | None,
    seed_to_params,
) -> _OptunaDuplicateState:
    state = _OptunaDuplicateState()
    if not bool(setup.duplicate_guard):
        return state
    cached_records = _auto_optuna_cached_study_records(study_name)
    if cached_records is not None:
        state.known_records = dict(cached_records or {})
        return state
    state.known_records = _auto_optuna_study_records(study, seed_to_params=seed_to_params)
    if study_name:
        _OPTUNA_KNOWN_RECORDS[str(study_name)] = dict(state.known_records or {})
        _OPTUNA_KNOWN_SIGNATURES_PRIMED.add(str(study_name))
        _auto_optuna_get_known_signatures(str(study_name)).update(state.known_records.keys())
    return state


def _auto_optuna_make_pruning_hook(*, trial_obj_ref, trial_pruned_cls):
    """Return a hook that reports a partial score and raises TrialPruned if warranted."""
    step_counter = [0]

    def _hook(partial_score: float) -> None:
        try:
            trial_obj_ref.report(float(partial_score), step=step_counter[0])
            step_counter[0] += 1
            should = trial_obj_ref.should_prune()
        except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError, NameError):
            return
        if bool(should) and trial_pruned_cls is not None:
            raise trial_pruned_cls()

    return _hook


def _auto_optuna_finalize_run_telemetry(
    *,
    context: _OptunaEvalLoopContext,
) -> dict:
    if not bool(
        _auto_safe_bool(
            (context.base_data or {}).get("auto_mode_optuna_telemetry", AUTO_MODE_OPTUNA_TELEMETRY),
            AUTO_MODE_OPTUNA_TELEMETRY,
        )
    ):
        return {}
    telemetry = _auto_optuna_build_run_telemetry(
        context.study,
        base_data=context.base_data,
        study_name=context.study_name,
        study_scope=context.scope_eff,
        phase_kind=context.phase_kind,
        run_token=context.run_token,
        requested_total=int(context.total),
        startup_trials=int(context.startup_effective),
        duplicate_skips=int(context.duplicate_state.duplicate_skips),
        duplicate_replays=int(context.duplicate_state.duplicate_replays),
        duplicate_reserved=int(context.duplicate_state.duplicate_reserved),
    )
    cache_stats_now = _auto_cache_stats_snapshot()
    scan_stats_now = _auto_optuna_study_scan_stats_snapshot()
    telemetry["cache_loads"] = int(cache_stats_now.get("loads", 0) or 0) - int(
        context.cache_stats_start.get("loads", 0) or 0
    )
    telemetry["cache_load_hits"] = int(cache_stats_now.get("load_hits", 0) or 0) - int(
        context.cache_stats_start.get("load_hits", 0) or 0
    )
    telemetry["cache_saves"] = int(cache_stats_now.get("saves", 0) or 0) - int(
        context.cache_stats_start.get("saves", 0) or 0
    )
    telemetry["cache_entry_hits"] = int(cache_stats_now.get("entry_hits", 0) or 0) - int(
        context.cache_stats_start.get("entry_hits", 0) or 0
    )
    telemetry["cache_entry_misses"] = int(cache_stats_now.get("entry_misses", 0) or 0) - int(
        context.cache_stats_start.get("entry_misses", 0) or 0
    )
    telemetry["study_scans"] = int(scan_stats_now.get("study_scans", 0) or 0) - int(
        context.study_scan_stats_start.get("study_scans", 0) or 0
    )
    telemetry["study_trials_scanned"] = int(scan_stats_now.get("study_trials_scanned", 0) or 0) - int(
        context.study_scan_stats_start.get("study_trials_scanned", 0) or 0
    )
    if bool(
        _auto_safe_bool(
            (context.base_data or {}).get("auto_mode_optuna_telemetry_log_summary", AUTO_MODE_OPTUNA_TELEMETRY_LOG_SUMMARY),
            AUTO_MODE_OPTUNA_TELEMETRY_LOG_SUMMARY,
        )
    ):
        _auto_optuna_log_run_telemetry(
            logger,
            phase_label=str(context.phase_label or context.scope_eff or "optuna"),
            tel=telemetry,
        )
    return dict(telemetry or {})


def _auto_optuna_objective_value_from_out(
    *,
    context: _OptunaEvalLoopContext,
    out_payload: dict,
) -> float | None:
    if not bool(dict(out_payload or {}).get("ok", False)):
        return None
    try:
        value = float(context.objective_value(dict(out_payload or {})))
        if not np.isfinite(value):
            return 0.0
        return float(value)
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError, NameError):
        return 0.0


def _auto_optuna_set_trial_out_payload(
    *,
    trial_obj,
    out_payload: dict,
) -> None:
    try:
        if hasattr(trial_obj, "set_user_attr"):
            trial_obj.set_user_attr(
                AUTO_MODE_OPTUNA_USER_ATTR_OUT,
                _auto_optuna_jsonable(dict(out_payload or {})),
            )
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError, NameError):
        logger.exception("optuna tell_trial user attr set")


def _auto_optuna_tell_study_trial(
    *,
    context: _OptunaEvalLoopContext,
    trial_obj,
    out_payload: dict,
    value: float | None,
) -> None:
    try:
        if bool(dict(out_payload or {}).get("ok", False)):
            context.study.tell(trial_obj, float(value))
            if value is not None and np.isfinite(value) and hasattr(trial_obj, "intermediate_values"):
                _iv = dict(trial_obj.intermediate_values or {})
                if _iv:
                    _proxy = _iv.get(0)
                    if _proxy is not None:
                        logger.debug(
                            "optuna pruning proxy=%.3f final_obj=%.4f (proxy=-(p90+clip_pen), obj=rank_score+bass_bonus)",
                            float(_proxy),
                            float(value),
                        )
        else:
            context.study.tell(trial_obj, state=context.fail_state)
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError, NameError):
        logger.exception("optuna tell_trial tell")


def _auto_optuna_store_known_record(
    *,
    context: _OptunaEvalLoopContext,
    params_sig: str,
    out_payload: dict,
    value: float | None,
) -> None:
    if not params_sig:
        return
    context.reserved_signatures.discard(str(params_sig))
    rec = {"params_sig": str(params_sig)}
    if bool(dict(out_payload or {}).get("ok", False)) and value is not None and np.isfinite(value):
        rec["value"] = float(value)
    else:
        rec["state"] = context.fail_state
    if isinstance(out_payload, dict) and out_payload:
        rec["out"] = dict(out_payload or {})
    context.known_records[str(params_sig)] = rec
    _auto_optuna_update_known_record(context.study_name, str(params_sig), rec)


def _auto_optuna_tell_trial(
    *,
    context: _OptunaEvalLoopContext,
    trial_obj,
    out: dict,
    params_sig: str = "",
    source: str = "optuna",
) -> None:
    out_payload = dict(out or {})
    value = _auto_optuna_objective_value_from_out(
        context=context,
        out_payload=out_payload,
    )
    out_payload = _auto_optuna_attach_out_telemetry(
        out_payload,
        base_data=context.base_data,
        study_name=context.study_name,
        study_scope=context.scope_eff,
        phase_kind=context.phase_kind,
        run_token=context.run_token,
        source=str(source or "optuna"),
        objective_value_num=value,
    )
    _auto_optuna_set_trial_out_payload(
        trial_obj=trial_obj,
        out_payload=out_payload,
    )
    _auto_optuna_tell_study_trial(
        context=context,
        trial_obj=trial_obj,
        out_payload=out_payload,
        value=value,
    )
    _auto_optuna_store_known_record(
        context=context,
        params_sig=str(params_sig or ""),
        out_payload=out_payload,
        value=value,
    )


def _auto_optuna_reuse_duplicate_trial(
    *,
    context: _OptunaEvalLoopContext,
    trial_obj,
    params_sig: str,
    replay_idx: int,
) -> None:
    rec = dict(context.known_records.get(str(params_sig), {}) or {})
    out_prev = dict(rec.get("out", {}) or {})
    val = rec.get("value")
    out_payload = _auto_optuna_attach_out_telemetry(
        out_prev,
        base_data=context.base_data,
        study_name=context.study_name,
        study_scope=context.scope_eff,
        phase_kind=context.phase_kind,
        run_token=context.run_token,
        source="replayed",
        objective_value_num=(
            float(val)
            if val is not None and np.isfinite(_auto_safe_float(val, float("nan")))
            else None
        ),
    )
    if out_payload and hasattr(trial_obj, "set_user_attr"):
        try:
            trial_obj.set_user_attr(
                AUTO_MODE_OPTUNA_USER_ATTR_OUT,
                _auto_optuna_jsonable(out_payload),
            )
        except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError, NameError):
            logger.exception("optuna duplicate trial user attr set")
    try:
        if val is not None and np.isfinite(float(val)):
            context.study.tell(trial_obj, float(val))
        else:
            context.study.tell(trial_obj, state=context.fail_state)
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError, NameError):
        logger.exception("optuna duplicate trial tell")
    if bool(dict(out_prev or {}).get("ok", False)):
        try:
            context.consume_one(int(replay_idx), dict(out_payload or {}))
            logger.debug(
                "Automatic mode Optuna duplicate replay consumed into current search state (sig=%.12s)",
                str(params_sig),
            )
        except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError, NameError):
            logger.debug("Automatic mode Optuna duplicate replay consume failed", exc_info=True)


def _auto_optuna_ask_new_trial(
    *,
    context: _OptunaEvalLoopContext,
):
    attempts = int(max(1, AUTO_MODE_OPTUNA_DUPLICATE_MAX_ATTEMPTS))
    last_error = None
    for _ in range(attempts):
        try:
            trial_obj = context.study.ask()
            preset = dict(context.build_preset(trial_obj) or {})
        except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError, NameError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            break
        params = _auto_optuna_sanitize_enqueued_params(
            _auto_optuna_trial_params(
                trial_obj=trial_obj,
                preset=preset,
                seed_to_params=context.seed_to_params,
            ),
            base_data=context.base_data,
        )
        params_sig = _auto_optuna_param_signature(params)
        if (not bool(context.duplicate_guard)) or (not params_sig):
            if params_sig:
                context.reserved_signatures.add(str(params_sig))
            return trial_obj, preset, str(params_sig), None
        if params_sig in context.reserved_signatures:
            context.duplicate_state.duplicate_skips += 1
            context.duplicate_state.duplicate_reserved += 1
            reserved_out = _auto_optuna_attach_out_telemetry(
                {
                    "ok": False,
                    "error": "duplicate suggestion reserved in current batch",
                },
                base_data=context.base_data,
                study_name=context.study_name,
                study_scope=context.scope_eff,
                phase_kind=context.phase_kind,
                run_token=context.run_token,
                source="reserved",
                objective_value_num=None,
            )
            try:
                if hasattr(trial_obj, "set_user_attr"):
                    trial_obj.set_user_attr(
                        AUTO_MODE_OPTUNA_USER_ATTR_OUT,
                        _auto_optuna_jsonable(dict(reserved_out or {})),
                    )
                context.study.tell(trial_obj, state=context.fail_state)
            except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError, NameError):
                logger.exception("optuna reserved trial fail-tell")
            continue
        if params_sig in context.known_records:
            context.duplicate_state.duplicate_skips += 1
            context.duplicate_state.duplicate_replays += 1
            _auto_optuna_reuse_duplicate_trial(
                context=context,
                trial_obj=trial_obj,
                params_sig=str(params_sig),
                replay_idx=int(context.duplicate_state.duplicate_replays),
            )
            continue
        context.reserved_signatures.add(str(params_sig))
        return trial_obj, preset, str(params_sig), None
    return None, {}, "", str(last_error or "no unique optuna candidate available")


def _auto_optuna_filter_and_enqueue_seed_items(
    *,
    context: _OptunaEvalLoopContext,
    seed_items: list[dict],
) -> list[dict]:
    if not (callable(context.seed_to_params) and hasattr(context.study, "enqueue_trial")):
        return list(seed_items or [])
    seed_items_filtered = []
    enqueued_signatures: set[str] = set()
    for preset in list(seed_items):
        try:
            params = _auto_optuna_sanitize_enqueued_params(
                dict(context.seed_to_params(dict(preset or {})) or {}),
                base_data=context.base_data,
            )
        except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError, NameError):
            params = {}
        params_sig = _auto_optuna_param_signature(params)
        if bool(context.duplicate_guard) and params_sig and (
            params_sig in context.known_records or params_sig in enqueued_signatures
        ):
            context.duplicate_state.duplicate_skips += 1
            continue
        if params:
            try:
                context.study.enqueue_trial(dict(params))
                if params_sig:
                    enqueued_signatures.add(str(params_sig))
            except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError, NameError):
                logger.exception("optuna seed trial enqueue")
        seed_items_filtered.append(dict(preset or {}))
    return list(seed_items_filtered)


def _auto_run_optuna_eval_loop_core(
    *,
    optuna_mod,
    cfg: AutoModeConfig | None = None,
    n_total: int,
    seed: int,
    startup_trials: int | None = None,
    base_data: dict | None,
    seed_presets: list[dict] | None,
    build_preset,
    eval_one,
    consume_one,
    objective_value,
    workers: int,
    seed_to_params=None,
    study_name: str | None = None,
    study_scope: str | None = None,
    phase_label: str | None = None,
    phase_kind: str | None = None,
    study_user_attrs: dict | None = None,
) -> dict:
    if not _auto_optuna_module_ready(optuna_mod):
        return {}
    setup = _auto_optuna_prepare_run_setup(
        optuna_mod=optuna_mod,
        cfg=cfg,
        n_total=int(n_total),
        seed=int(seed),
        startup_trials=startup_trials,
        base_data=base_data,
        workers=int(workers),
        study_name=study_name,
        study_scope=study_scope,
        phase_kind=phase_kind,
    )
    total = int(setup.total)
    if total <= 0:
        return {}
    scope_eff = setup.scope_eff
    startup_effective = setup.startup_effective
    run_token = setup.run_token
    pruner = setup.pruner
    _trial_pruned_cls = setup.trial_pruned_cls
    _pruned_state = setup.pruned_state
    fail_state = setup.fail_state
    duplicate_guard = setup.duplicate_guard
    cache_stats_start = dict(setup.cache_stats_start or {})
    study_scan_stats_start = dict(setup.study_scan_stats_start or {})
    study = _auto_optuna_prepare_study(
        optuna_mod=optuna_mod,
        setup=setup,
        base_data=base_data,
        study_name=study_name,
        study_user_attrs=study_user_attrs,
    )
    duplicate_state = _auto_optuna_prepare_duplicate_state(
        setup=setup,
        study=study,
        study_name=study_name,
        seed_to_params=seed_to_params,
    )
    known_records = duplicate_state.known_records
    reserved_signatures = duplicate_state.reserved_signatures
    context = _OptunaEvalLoopContext(
        study=study,
        base_data=base_data,
        study_name=study_name,
        scope_eff=scope_eff,
        phase_kind=phase_kind,
        run_token=run_token,
        phase_label=phase_label,
        total=int(total),
        startup_effective=int(startup_effective),
        cache_stats_start=dict(cache_stats_start or {}),
        study_scan_stats_start=dict(study_scan_stats_start or {}),
        trial_pruned_cls=_trial_pruned_cls,
        pruned_state=_pruned_state,
        fail_state=fail_state,
        pruner=pruner,
        seed_to_params=seed_to_params,
        build_preset=build_preset,
        eval_one=eval_one,
        consume_one=consume_one,
        objective_value=objective_value,
        duplicate_guard=bool(duplicate_guard),
        duplicate_state=duplicate_state,
        known_records=known_records,
        reserved_signatures=reserved_signatures,
    )

    seed_items = list(seed_presets or [])[: int(total)]
    seed_items = _auto_optuna_filter_and_enqueue_seed_items(
        context=context,
        seed_items=list(seed_items or []),
    )

    idx_next, seed_telemetry = _run_optuna_seed_trials(
        total=int(total),
        seed_items=list(seed_items or []),
        seed_to_params=seed_to_params,
        ask_new_trial=lambda: _auto_optuna_ask_new_trial(context=context),
        eval_one=eval_one,
        consume_one=consume_one,
        tell_trial=lambda trial_obj, out, **kwargs: _auto_optuna_tell_trial(
            context=context,
            trial_obj=trial_obj,
            out=out,
            **kwargs,
        ),
        finalize_telemetry=lambda: _auto_optuna_finalize_run_telemetry(context=context),
    )
    if seed_telemetry is not None:
        _log_optuna_duplicate_summary(
            duplicate_skips=int(duplicate_state.duplicate_skips),
            study_name=study_name,
        )
        return dict(seed_telemetry or {})
    if idx_next > total:
        _log_optuna_duplicate_summary(
            duplicate_skips=int(duplicate_state.duplicate_skips),
            study_name=study_name,
        )
        return _auto_optuna_finalize_run_telemetry(context=context)

    remaining = int(total - idx_next + 1)
    if workers <= 1 or remaining <= 1:
        serial_telemetry = _run_optuna_serial_trials(
            idx_next=int(idx_next),
            total=int(total),
            ask_new_trial=lambda: _auto_optuna_ask_new_trial(context=context),
            eval_one=eval_one,
            consume_one=consume_one,
            tell_trial=lambda trial_obj, out, **kwargs: _auto_optuna_tell_trial(
                context=context,
                trial_obj=trial_obj,
                out=out,
                **kwargs,
            ),
            finalize_telemetry=lambda: _auto_optuna_finalize_run_telemetry(context=context),
            pruner=pruner,
            make_pruning_hook=lambda trial_obj_ref: _auto_optuna_make_pruning_hook(
                trial_obj_ref=trial_obj_ref,
                trial_pruned_cls=_trial_pruned_cls,
            ),
            trial_pruned_cls=_trial_pruned_cls,
            pruned_state=_pruned_state,
            study=study,
            reserved_signatures=reserved_signatures,
        )
        _log_optuna_duplicate_summary(
            duplicate_skips=int(duplicate_state.duplicate_skips),
            study_name=study_name,
        )
        return dict(serial_telemetry or _auto_optuna_finalize_run_telemetry(context=context) or {})

    chunk_size = int(_auto_trial_chunk_size(workers))

    def _eval_with_hook(idx, preset, trial_obj_ref):
        if pruner is not None and trial_obj_ref is not None:
            _set_pruning_hook(
                _auto_optuna_make_pruning_hook(
                    trial_obj_ref=trial_obj_ref,
                    trial_pruned_cls=_trial_pruned_cls,
                )
            )
        try:
            return eval_one(int(idx), dict(preset))
        finally:
            _clear_pruning_hook()

    parallel_telemetry = _run_optuna_parallel_trials(
        idx_next=int(idx_next),
        total=int(total),
        workers=int(workers),
        chunk_size=int(chunk_size),
        ask_new_trial=lambda: _auto_optuna_ask_new_trial(context=context),
        eval_with_hook=_eval_with_hook,
        consume_one=consume_one,
        tell_trial=lambda trial_obj, out, **kwargs: _auto_optuna_tell_trial(
            context=context,
            trial_obj=trial_obj,
            out=out,
            **kwargs,
        ),
        finalize_telemetry=lambda: _auto_optuna_finalize_run_telemetry(context=context),
        trial_pruned_cls=_trial_pruned_cls,
        pruned_state=_pruned_state,
        study=study,
        reserved_signatures=reserved_signatures,
    )
    _log_optuna_duplicate_summary(
        duplicate_skips=int(duplicate_state.duplicate_skips),
        study_name=study_name,
    )
    return dict(parallel_telemetry or _auto_optuna_finalize_run_telemetry(context=context) or {})


__all__ = ['_auto_run_optuna_eval_loop_core']

