# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Optuna backend — eval loop orchestration (context, state, seed/serial/parallel runners)."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np

from ..dsp._pruning import (
    clear_pruning_hook as _clear_pruning_hook,
    set_pruning_hook as _set_pruning_hook,
)
from .shared import AutoModeConfig

logger = logging.getLogger("DecayCore")

_OPTUNA_RECOVERABLE_EXC_TYPES = (
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
)


@dataclass(slots=True)
class _OptunaEvalContext:
    params: dict
    total: int
    workers: int
    phase_label: str


@dataclass(slots=True)
class _OptunaEvalState:
    context: _OptunaEvalContext
    telemetry: dict = field(default_factory=dict)


def _prepare_optuna_eval_context(
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
) -> _OptunaEvalContext:
    return _OptunaEvalContext(
        params={
            "optuna_mod": optuna_mod,
            "cfg": cfg,
            "n_total": int(n_total),
            "seed": int(seed),
            "startup_trials": startup_trials,
            "base_data": dict(base_data or {}) if isinstance(base_data, dict) else base_data,
            "seed_presets": list(seed_presets or []) if seed_presets is not None else None,
            "build_preset": build_preset,
            "eval_one": eval_one,
            "consume_one": consume_one,
            "objective_value": objective_value,
            "workers": int(workers),
            "seed_to_params": seed_to_params,
            "study_name": study_name,
            "study_scope": study_scope,
            "phase_label": phase_label,
            "phase_kind": phase_kind,
            "study_user_attrs": dict(study_user_attrs or {}) if isinstance(study_user_attrs, dict) else None,
        },
        total=int(max(0, n_total)),
        workers=int(workers),
        phase_label=str(phase_label or study_scope or study_name or "optuna"),
    )


def _submit_or_schedule_trials(
    *,
    context: _OptunaEvalContext,
) -> _OptunaEvalState:
    from .optuna_backend_core import _auto_run_optuna_eval_loop_core
    telemetry = _auto_run_optuna_eval_loop_core(**dict(context.params or {}))
    return _OptunaEvalState(
        context=context,
        telemetry=dict(telemetry or {}),
    )


def _consume_completed_trial(
    *,
    state: _OptunaEvalState,
) -> _OptunaEvalState:
    return state


def _update_best_and_telemetry(
    *,
    state: _OptunaEvalState,
) -> _OptunaEvalState:
    return _OptunaEvalState(
        context=state.context,
        telemetry=dict(state.telemetry or {}),
    )


def _finalize_optuna_eval_loop(
    *,
    state: _OptunaEvalState,
) -> dict:
    return dict(state.telemetry or {})


def _log_optuna_duplicate_summary(*, duplicate_skips: int, study_name: str | None) -> None:
    if int(duplicate_skips) <= 0:
        return
    logger.info(
        "Automatic mode Optuna duplicate guard skipped %d duplicate suggestions in study %s.",
        int(duplicate_skips),
        str(study_name or "in-memory"),
    )


def _optuna_pruned_result(*, idx: int) -> dict:
    return {
        "idx": int(idx),
        "ok": False,
        "error": "optuna trial pruned",
        "pruned": True,
    }


def _optuna_parallel_recoverable_exc_types(*, trial_pruned_cls):
    if trial_pruned_cls is None:
        return _OPTUNA_RECOVERABLE_EXC_TYPES
    return (trial_pruned_cls,) + _OPTUNA_RECOVERABLE_EXC_TYPES


def _run_optuna_seed_trials(
    *,
    total: int,
    seed_items: list[dict],
    seed_to_params,
    ask_new_trial,
    eval_one,
    consume_one,
    tell_trial,
    finalize_telemetry,
) -> tuple[int, dict | None]:
    idx_next = 1
    for preset in list(seed_items):
        if idx_next > int(total):
            return int(idx_next), dict(finalize_telemetry() or {})
        trial_obj = None
        params_sig = ""
        # Always evaluate the original seed preset directly.
        # When seed_to_params is callable, ask Optuna for a trial_obj (for tell/model update),
        # but do NOT replace the seed preset with whatever Optuna suggested — the enqueued seed
        # should be returned, but if it was a duplicate or reordered, we still evaluate the
        # intended seed rather than an unrelated Optuna suggestion.
        preset_eval = dict(preset or {})
        if callable(seed_to_params):
            trial_obj, _preset_ask, params_sig, ask_error = ask_new_trial()
            if trial_obj is None:
                logger.debug(
                    "Automatic mode seed trial skipped: no Optuna trial available (%s)",
                    str(ask_error or ""),
                )
                out = {
                    "idx": int(idx_next),
                    "ok": False,
                    "error": str(ask_error or "no unique optuna candidate available"),
                }
                if consume_one(int(idx_next), dict(out or {})):
                    return int(idx_next), dict(finalize_telemetry() or {})
                idx_next += 1
                continue
            logger.debug(
                "Automatic mode seed trial evaluated via original seed preset (trial_obj from Optuna ask)"
            )
        try:
            out = eval_one(int(idx_next), dict(preset_eval or {}))
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
            out = {
                "idx": int(idx_next),
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        finally:
            _clear_pruning_hook()
        if trial_obj is not None:
            tell_trial(trial_obj, out, params_sig=params_sig, source="seed")
        if consume_one(int(idx_next), dict(out or {})):
            return int(idx_next), dict(finalize_telemetry() or {})
        idx_next += 1
    return int(idx_next), None


def _run_optuna_serial_trials(
    *,
    idx_next: int,
    total: int,
    ask_new_trial,
    eval_one,
    consume_one,
    tell_trial,
    finalize_telemetry,
    pruner,
    make_pruning_hook,
    trial_pruned_cls,
    pruned_state,
    study,
    reserved_signatures: set[str],
) -> dict | None:
    for idx in range(int(idx_next), int(total) + 1):
        trial_obj, preset, params_sig, ask_error = ask_new_trial()
        if trial_obj is None:
            if _emit_optuna_ask_failure(
                idx=int(idx),
                ask_error=ask_error,
                consume_one=consume_one,
                finalize_telemetry=finalize_telemetry,
            ):
                return dict(finalize_telemetry() or {})
            continue
        _maybe_set_pruning_hook(pruner=pruner, make_pruning_hook=make_pruning_hook, trial_obj=trial_obj)
        try:
            out = eval_one(int(idx), dict(preset))
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
            prune_state = _handle_serial_pruned_trial(
                idx=int(idx),
                exc=exc,
                trial_pruned_cls=trial_pruned_cls,
                pruned_state=pruned_state,
                study=study,
                trial_obj=trial_obj,
                params_sig=params_sig,
                reserved_signatures=reserved_signatures,
                consume_one=consume_one,
                finalize_telemetry=finalize_telemetry,
            )
            if prune_state == "consumed":
                return dict(finalize_telemetry() or {})
            if prune_state == "continue":
                continue
            out = {
                "idx": int(idx),
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        finally:
            _clear_pruning_hook()
        tell_trial(trial_obj, out, params_sig=params_sig, source="optuna")
        if consume_one(int(idx), dict(out or {})):
            return dict(finalize_telemetry() or {})
    return None


def _emit_optuna_ask_failure(
    *,
    idx: int,
    ask_error,
    consume_one,
    finalize_telemetry,
) -> bool:
    out = {
        "idx": int(idx),
        "ok": False,
        "error": str(ask_error or "no unique optuna candidate available"),
    }
    if consume_one(int(idx), dict(out or {})):
        return True
    return False


def _maybe_set_pruning_hook(*, pruner, make_pruning_hook, trial_obj) -> None:
    if pruner is not None:
        _set_pruning_hook(make_pruning_hook(trial_obj))


def _handle_serial_pruned_trial(
    *,
    idx: int,
    exc: Exception,
    trial_pruned_cls,
    pruned_state,
    study,
    trial_obj,
    params_sig: str | None,
    reserved_signatures: set[str],
    consume_one,
    finalize_telemetry,
):
    if trial_pruned_cls is None or not isinstance(exc, trial_pruned_cls):
        return "not_pruned"
    if pruned_state is not None:
        try:
            study.tell(trial_obj, state=pruned_state)
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
            logger.exception("optuna tell pruned trial")
    if params_sig:
        reserved_signatures.discard(str(params_sig))
    out = _optuna_pruned_result(idx=int(idx))
    if consume_one(int(idx), dict(out or {})):
        return "consumed"
    return "continue"


def _optuna_parallel_build_chunk(
    *,
    idx_cursor: int,
    total: int,
    chunk_size: int,
    ask_new_trial,
):
    chunk_items = []
    idx_cursor_eff = int(idx_cursor)
    while idx_cursor_eff <= int(total) and len(chunk_items) < int(chunk_size):
        trial_obj, preset, params_sig, ask_error = ask_new_trial()
        if trial_obj is None:
            chunk_items.append(
                (
                    int(idx_cursor_eff),
                    None,
                    {},
                    "",
                    {
                        "idx": int(idx_cursor_eff),
                        "ok": False,
                        "error": str(ask_error or "no unique optuna candidate available"),
                    },
                )
            )
            idx_cursor_eff += 1
            continue
        chunk_items.append((int(idx_cursor_eff), trial_obj, dict(preset), str(params_sig), None))
        idx_cursor_eff += 1
    return chunk_items, int(idx_cursor_eff)


def _optuna_parallel_future_out(
    *,
    fut,
    idx: int,
    trial_obj,
    params_sig: str,
    trial_pruned_cls,
):
    out = None
    exc = None
    try:
        out = fut.result()
    except _optuna_parallel_recoverable_exc_types(trial_pruned_cls=trial_pruned_cls) as recoverable_exc:
        exc = recoverable_exc
    if exc is None:
        if not isinstance(out, dict):
            out = {"idx": int(idx), "ok": False, "error": "invalid worker result"}
        return out, None
    if trial_pruned_cls is not None and isinstance(exc, trial_pruned_cls):
        return None, exc
    return {"idx": int(idx), "ok": False, "error": f"{type(exc).__name__}: {exc}"}, None


def _optuna_parallel_consume_chunk(
    *,
    chunk_items,
    chunk_out: dict[int, dict],
    consume_one,
    finalize_telemetry,
) -> dict | None:
    for idx, _trial_obj, _preset, _params_sig, pre_out in chunk_items:
        if isinstance(pre_out, dict):
            out = dict(pre_out or {})
        else:
            out = dict(
                chunk_out.get(
                    int(idx),
                    {"idx": int(idx), "ok": False, "error": "missing worker result"},
                )
                or {}
            )
        if consume_one(int(idx), out):
            return dict(finalize_telemetry() or {})
    return None


def _run_optuna_parallel_trials(
    *,
    idx_next: int,
    total: int,
    workers: int,
    chunk_size: int,
    ask_new_trial,
    eval_with_hook,
    consume_one,
    tell_trial,
    finalize_telemetry,
    trial_pruned_cls,
    pruned_state,
    study,
    reserved_signatures: set[str],
) -> dict | None:
    with ThreadPoolExecutor(max_workers=int(workers)) as ex:
        idx_cursor = int(idx_next)
        while idx_cursor <= int(total):
            chunk_items, idx_cursor = _optuna_parallel_build_chunk(
                idx_cursor=idx_cursor,
                total=total,
                chunk_size=chunk_size,
                ask_new_trial=ask_new_trial,
            )
            if not chunk_items:
                break

            fut_map = {
                ex.submit(eval_with_hook, int(idx), dict(preset), trial_obj): (int(idx), trial_obj, str(params_sig))
                for idx, trial_obj, preset, params_sig, pre_out in chunk_items
                if trial_obj is not None and pre_out is None
            }
            chunk_out: dict[int, dict] = {}
            for fut in as_completed(list(fut_map.keys())):
                idx, trial_obj, params_sig = fut_map.get(fut, (0, None, ""))
                out, exc = _optuna_parallel_future_out(
                    fut=fut,
                    idx=int(idx),
                    trial_obj=trial_obj,
                    params_sig=str(params_sig),
                    trial_pruned_cls=trial_pruned_cls,
                )
                if exc is not None:
                    if trial_obj is not None and pruned_state is not None:
                        try:
                            study.tell(trial_obj, state=pruned_state)
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
                            logger.exception("optuna tell pruned trial in parallel")
                    if params_sig:
                        reserved_signatures.discard(str(params_sig))
                    chunk_out[int(idx)] = _optuna_pruned_result(idx=int(idx))
                    continue
                tell_trial(trial_obj, out, params_sig=params_sig, source="optuna")
                chunk_out[int(idx)] = dict(out or {})

            consumed = _optuna_parallel_consume_chunk(
                chunk_items=chunk_items,
                chunk_out=chunk_out,
                consume_one=consume_one,
                finalize_telemetry=finalize_telemetry,
            )
            if consumed is not None:
                return consumed
    return None


def _auto_run_optuna_eval_loop(
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
    context = _prepare_optuna_eval_context(
        optuna_mod=optuna_mod,
        cfg=cfg,
        n_total=int(n_total),
        seed=int(seed),
        startup_trials=startup_trials,
        base_data=base_data,
        seed_presets=seed_presets,
        build_preset=build_preset,
        eval_one=eval_one,
        consume_one=consume_one,
        objective_value=objective_value,
        workers=int(workers),
        seed_to_params=seed_to_params,
        study_name=study_name,
        study_scope=study_scope,
        phase_label=phase_label,
        phase_kind=phase_kind,
        study_user_attrs=study_user_attrs,
    )
    scheduled = _submit_or_schedule_trials(context=context)
    consumed = _consume_completed_trial(state=scheduled)
    telemetry = _update_best_and_telemetry(state=consumed)
    return _finalize_optuna_eval_loop(state=telemetry)
