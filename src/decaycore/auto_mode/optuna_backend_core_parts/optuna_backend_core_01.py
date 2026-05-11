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

import logging

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
    _optuna_pruned_result,
    _run_optuna_seed_trials,
    _run_optuna_serial_trials,
    _run_optuna_parallel_trials,
)

logger = logging.getLogger("DecayCore")

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
) -> dict:
    if not _auto_optuna_module_ready(optuna_mod):
        return {}
    total = int(max(0, n_total))
    if total <= 0:
        return {}
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
            except Exception:
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
    study = _auto_optuna_create_study(
        optuna_mod,
        sampler=sampler,
        pruner=pruner,
        base_data=base_data,
        study_name=study_name,
    )
    if (
        _auto_safe_bool(
            (base_data or {}).get("auto_mode_optuna_cross_study_seeds", AUTO_MODE_OPTUNA_CROSS_STUDY_SEEDS),
            AUTO_MODE_OPTUNA_CROSS_STUDY_SEEDS,
        )
        and study_name
        and hasattr(study, "enqueue_trial")
    ):
        try:
            _existing_complete = [
                tr for tr in study.get_trials(deepcopy=False)
                if getattr(tr, "value", None) is not None
            ]
        except Exception:
            _existing_complete = []
        if not _existing_complete:
            _top_n_cross = max(
                1,
                _auto_safe_int(
                    (base_data or {}).get(
                        "auto_mode_optuna_cross_study_seeds_top_n",
                        AUTO_MODE_OPTUNA_CROSS_STUDY_SEEDS_TOP_N,
                    ),
                    AUTO_MODE_OPTUNA_CROSS_STUDY_SEEDS_TOP_N,
                ),
            )
            _cross_params = _auto_optuna_cross_study_best_params(
                optuna_mod,
                base_data=base_data,
                scope=str(scope_eff or ""),
                current_study_name=str(study_name),
                top_n=int(_top_n_cross),
            )
            _cross_enqueued = 0
            for _cp in _cross_params:
                try:
                    study.enqueue_trial(_auto_optuna_sanitize_enqueued_params(dict(_cp), base_data=base_data))
                    _cross_enqueued += 1
                except Exception:
                    logger.exception("optuna cross-study trial enqueue")
            if _cross_enqueued:
                logger.info(
                    "Automatic mode cross-study seeds: enqueued %d trials from sibling studies (scope=%s)",
                    _cross_enqueued,
                    str(scope_eff or ""),
                )
    _trial_pruned_cls = getattr(optuna_mod, "TrialPruned", None)
    _trial_pruned_state = getattr(
        getattr(optuna_mod, "trial", None),
        "TrialState",
        None,
    )
    _pruned_state = getattr(_trial_pruned_state, "PRUNED", None) if _trial_pruned_state is not None else None
    fail_state = optuna_mod.trial.TrialState.FAIL
    duplicate_guard = bool(
        _auto_safe_bool((base_data or {}).get("auto_mode_optuna_avoid_duplicates", True), True)
    )
    cache_stats_start = _auto_cache_stats_snapshot()
    study_scan_stats_start = _auto_optuna_study_scan_stats_snapshot()
    known_records = {}
    if bool(duplicate_guard):
        cached_records = _auto_optuna_cached_study_records(study_name)
        if cached_records is not None:
            known_records = dict(cached_records or {})
        else:
            known_records = _auto_optuna_study_records(study, seed_to_params=seed_to_params)
            if study_name:
                _OPTUNA_KNOWN_RECORDS[str(study_name)] = dict(known_records or {})
                _OPTUNA_KNOWN_SIGNATURES_PRIMED.add(str(study_name))
                _auto_optuna_get_known_signatures(str(study_name)).update(known_records.keys())
    reserved_signatures: set[str] = set()
    duplicate_skips = 0
    duplicate_replays = 0
    duplicate_reserved = 0

    def _make_pruning_hook(trial_obj_ref):
        """Return a hook that reports a partial score and raises TrialPruned if warranted."""
        step_counter = [0]
        def _hook(partial_score: float) -> None:
            try:
                trial_obj_ref.report(float(partial_score), step=step_counter[0])
                step_counter[0] += 1
                should = trial_obj_ref.should_prune()
            except Exception:
                return
            if bool(should) and _trial_pruned_cls is not None:
                raise _trial_pruned_cls()
        return _hook

    def _finalize_telemetry() -> dict:
        if not bool(
            _auto_safe_bool(
                (base_data or {}).get("auto_mode_optuna_telemetry", AUTO_MODE_OPTUNA_TELEMETRY),
                AUTO_MODE_OPTUNA_TELEMETRY,
            )
        ):
            return {}
        telemetry = _auto_optuna_build_run_telemetry(
            study,
            base_data=base_data,
            study_name=study_name,
            study_scope=scope_eff,
            phase_kind=phase_kind,
            run_token=run_token,
            requested_total=int(total),
            startup_trials=int(startup_effective),
            duplicate_skips=int(duplicate_skips),
            duplicate_replays=int(duplicate_replays),
            duplicate_reserved=int(duplicate_reserved),
        )
        cache_stats_now = _auto_cache_stats_snapshot()
        scan_stats_now = _auto_optuna_study_scan_stats_snapshot()
        telemetry["cache_loads"] = int(cache_stats_now.get("loads", 0) or 0) - int(
            cache_stats_start.get("loads", 0) or 0
        )
        telemetry["cache_load_hits"] = int(cache_stats_now.get("load_hits", 0) or 0) - int(
            cache_stats_start.get("load_hits", 0) or 0
        )
        telemetry["cache_saves"] = int(cache_stats_now.get("saves", 0) or 0) - int(
            cache_stats_start.get("saves", 0) or 0
        )
        telemetry["cache_entry_hits"] = int(cache_stats_now.get("entry_hits", 0) or 0) - int(
            cache_stats_start.get("entry_hits", 0) or 0
        )
        telemetry["cache_entry_misses"] = int(cache_stats_now.get("entry_misses", 0) or 0) - int(
            cache_stats_start.get("entry_misses", 0) or 0
        )
        telemetry["study_scans"] = int(scan_stats_now.get("study_scans", 0) or 0) - int(
            study_scan_stats_start.get("study_scans", 0) or 0
        )
        telemetry["study_trials_scanned"] = int(scan_stats_now.get("study_trials_scanned", 0) or 0) - int(
            study_scan_stats_start.get("study_trials_scanned", 0) or 0
        )
        if bool(
            _auto_safe_bool(
                (base_data or {}).get("auto_mode_optuna_telemetry_log_summary", AUTO_MODE_OPTUNA_TELEMETRY_LOG_SUMMARY),
                AUTO_MODE_OPTUNA_TELEMETRY_LOG_SUMMARY,
            )
        ):
            _auto_optuna_log_run_telemetry(
                logger,
                phase_label=str(phase_label or scope_eff or "optuna"),
                tel=telemetry,
            )
        return dict(telemetry or {})

    def _tell(trial_obj, out: dict, *, params_sig: str = "", source: str = "optuna") -> None:
        value = None
        out_payload = dict(out or {})
        if bool(out_payload.get("ok", False)):
            try:
                value = float(objective_value(dict(out_payload or {})))
                if not np.isfinite(value):
                    value = 0.0
            except Exception:
                value = 0.0
        out_payload = _auto_optuna_attach_out_telemetry(
            out_payload,
            base_data=base_data,
            study_name=study_name,
            study_scope=scope_eff,
            phase_kind=phase_kind,
            run_token=run_token,
            source=str(source or "optuna"),
            objective_value_num=value,
        )
        try:
            if hasattr(trial_obj, "set_user_attr"):
                trial_obj.set_user_attr(
                    AUTO_MODE_OPTUNA_USER_ATTR_OUT,
                    _auto_optuna_jsonable(dict(out_payload or {})),
                )
        except Exception:
            logger.exception("optuna tell_trial user attr set")
        try:
            if bool(dict(out_payload or {}).get("ok", False)):
                study.tell(trial_obj, float(value))
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
                study.tell(trial_obj, state=fail_state)
        except Exception:
            logger.exception("optuna tell_trial tell")
        if params_sig:
            reserved_signatures.discard(str(params_sig))
            rec = {"params_sig": str(params_sig)}
            if bool(dict(out_payload or {}).get("ok", False)) and value is not None and np.isfinite(value):
                rec["value"] = float(value)
            else:
                rec["state"] = fail_state
            if isinstance(out_payload, dict) and out_payload:
                rec["out"] = dict(out_payload or {})
            known_records[str(params_sig)] = rec
            _auto_optuna_update_known_record(study_name, str(params_sig), rec)

    def _reuse_duplicate_trial(trial_obj, params_sig: str, replay_idx: int) -> None:
        rec = dict(known_records.get(str(params_sig), {}) or {})
        out_prev = dict(rec.get("out", {}) or {})
        val = rec.get("value", None)
        out_payload = _auto_optuna_attach_out_telemetry(
            out_prev,
            base_data=base_data,
            study_name=study_name,
            study_scope=scope_eff,
            phase_kind=phase_kind,
            run_token=run_token,
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
            except Exception:
                logger.exception("optuna duplicate trial user attr set")
        try:
            if val is not None and np.isfinite(float(val)):
                study.tell(trial_obj, float(val))
            else:
                study.tell(trial_obj, state=fail_state)
        except Exception:
            logger.exception("optuna duplicate trial tell")
        # Feed replayed result into current search state so it can affect winner selection.
        if bool(dict(out_prev or {}).get("ok", False)):
            try:
                consume_one(int(replay_idx), dict(out_payload or {}))
                logger.debug(
                    "Automatic mode Optuna duplicate replay consumed into current search state (sig=%.12s)",
                    str(params_sig),
                )
            except Exception:
                logger.debug("Automatic mode Optuna duplicate replay consume failed", exc_info=True)

    def _ask_new_trial():
        nonlocal duplicate_reserved, duplicate_replays, duplicate_skips
        attempts = int(max(1, AUTO_MODE_OPTUNA_DUPLICATE_MAX_ATTEMPTS))
        last_error = None
        for _ in range(attempts):
            try:
                trial_obj = study.ask()
                preset = dict(build_preset(trial_obj) or {})
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                break
            params = _auto_optuna_sanitize_enqueued_params(
                _auto_optuna_trial_params(
                    trial_obj=trial_obj,
                    preset=preset,
                    seed_to_params=seed_to_params,
                ),
                base_data=base_data,
            )
            params_sig = _auto_optuna_param_signature(params)
            if (not bool(duplicate_guard)) or (not params_sig):
                if params_sig:
                    reserved_signatures.add(str(params_sig))
                return trial_obj, preset, str(params_sig), None
            if params_sig in reserved_signatures:
                duplicate_skips += 1
                duplicate_reserved += 1
                reserved_out = _auto_optuna_attach_out_telemetry(
                    {
                        "ok": False,
                        "error": "duplicate suggestion reserved in current batch",
                    },
                    base_data=base_data,
                    study_name=study_name,
                    study_scope=scope_eff,
                    phase_kind=phase_kind,
                    run_token=run_token,
                    source="reserved",
                    objective_value_num=None,
                )
                try:
                    if hasattr(trial_obj, "set_user_attr"):
                        trial_obj.set_user_attr(
                            AUTO_MODE_OPTUNA_USER_ATTR_OUT,
                            _auto_optuna_jsonable(dict(reserved_out or {})),
                        )
                    study.tell(trial_obj, state=fail_state)
                except Exception:
                    logger.exception("optuna reserved trial fail-tell")
                continue
            if params_sig in known_records:
                duplicate_skips += 1
                duplicate_replays += 1
                _reuse_duplicate_trial(trial_obj, str(params_sig), int(duplicate_replays))
                continue
            reserved_signatures.add(str(params_sig))
            return trial_obj, preset, str(params_sig), None
        return None, {}, "", str(last_error or "no unique optuna candidate available")

    seed_items = list(seed_presets or [])[: int(total)]
    if callable(seed_to_params) and hasattr(study, "enqueue_trial"):
        seed_items_filtered = []
        enqueued_signatures: set[str] = set()
        for preset in list(seed_items):
            try:
                params = _auto_optuna_sanitize_enqueued_params(
                    dict(seed_to_params(dict(preset or {})) or {}),
                    base_data=base_data,
                )
            except Exception:
                params = {}
            params_sig = _auto_optuna_param_signature(params)
            if bool(duplicate_guard) and params_sig and (
                params_sig in known_records or params_sig in enqueued_signatures
            ):
                duplicate_skips += 1
                continue
            if params:
                try:
                    study.enqueue_trial(dict(params))
                    if params_sig:
                        enqueued_signatures.add(str(params_sig))
                except Exception:
                    logger.exception("optuna seed trial enqueue")
            seed_items_filtered.append(dict(preset or {}))
        seed_items = list(seed_items_filtered)

    idx_next, seed_telemetry = _run_optuna_seed_trials(
        total=int(total),
        seed_items=list(seed_items or []),
        seed_to_params=seed_to_params,
        ask_new_trial=_ask_new_trial,
        eval_one=eval_one,
        consume_one=consume_one,
        tell_trial=_tell,
        finalize_telemetry=_finalize_telemetry,
    )
    if seed_telemetry is not None:
        _log_optuna_duplicate_summary(
            duplicate_skips=int(duplicate_skips),
            study_name=study_name,
        )
        return dict(seed_telemetry or {})
    if idx_next > total:
        _log_optuna_duplicate_summary(
            duplicate_skips=int(duplicate_skips),
            study_name=study_name,
        )
        return _finalize_telemetry()

    remaining = int(total - idx_next + 1)
    if workers <= 1 or remaining <= 1:
        serial_telemetry = _run_optuna_serial_trials(
            idx_next=int(idx_next),
            total=int(total),
            ask_new_trial=_ask_new_trial,
            eval_one=eval_one,
            consume_one=consume_one,
            tell_trial=_tell,
            finalize_telemetry=_finalize_telemetry,
            pruner=pruner,
            make_pruning_hook=_make_pruning_hook,
            trial_pruned_cls=_trial_pruned_cls,
            pruned_state=_pruned_state,
            study=study,
            reserved_signatures=reserved_signatures,
        )
        _log_optuna_duplicate_summary(
            duplicate_skips=int(duplicate_skips),
            study_name=study_name,
        )
        return dict(serial_telemetry or _finalize_telemetry() or {})

    chunk_size = int(_auto_trial_chunk_size(workers))

    def _eval_with_hook(idx, preset, trial_obj_ref):
        if pruner is not None and trial_obj_ref is not None:
            _set_pruning_hook(_make_pruning_hook(trial_obj_ref))
        try:
            return eval_one(int(idx), dict(preset))
        finally:
            _clear_pruning_hook()

    parallel_telemetry = _run_optuna_parallel_trials(
        idx_next=int(idx_next),
        total=int(total),
        workers=int(workers),
        chunk_size=int(chunk_size),
        ask_new_trial=_ask_new_trial,
        eval_with_hook=_eval_with_hook,
        consume_one=consume_one,
        tell_trial=_tell,
        finalize_telemetry=_finalize_telemetry,
        trial_pruned_cls=_trial_pruned_cls,
        pruned_state=_pruned_state,
        study=study,
        reserved_signatures=reserved_signatures,
    )
    _log_optuna_duplicate_summary(
        duplicate_skips=int(duplicate_skips),
        study_name=study_name,
    )
    return dict(parallel_telemetry or _finalize_telemetry() or {})


__all__ = ['_auto_run_optuna_eval_loop_core']


def _load_sibling_symbols() -> None:
    import importlib
    package = __package__
    for module_name in ['optuna_backend_core_01']:
        if module_name == __name__.rsplit('.', 1)[-1]:
            continue
        module = importlib.import_module(f"{package}.{module_name}")
        for symbol in getattr(module, "__all__", ()):
            globals().setdefault(symbol, getattr(module, symbol))


_load_sibling_symbols()
