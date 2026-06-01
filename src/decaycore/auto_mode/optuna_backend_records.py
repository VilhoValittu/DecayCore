# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Optuna backend — study records, result persistence, and objective value."""

from __future__ import annotations

import logging

import numpy as np

from .shared import (
    AUTO_MODE_GOAL_FLAT,
    AUTO_MODE_OPTUNA_USER_ATTR_OUT,
    _auto_goal_norm,
    _auto_safe_float,
    _auto_safe_bool,
    _auto_optuna_sampler_kwargs,
)
from .optuna_backend_params import (
    _auto_optuna_jsonable,
    _auto_optuna_param_signature,
    _auto_optuna_sanitize_enqueued_params,
    _auto_optuna_trial_payload_preset,
    _auto_optuna_build_completed_trial,
)
from .optuna_backend_constraints import (
    _auto_optuna_constraint_thresholds,
    _auto_optuna_constraint_vector_from_metrics,
    _auto_optuna_constraints_func,
    _auto_optuna_effective_scope,
    _auto_optuna_use_events_constraint,
)
from .optuna_backend_scoring import (
    _AUTO_OPTUNA_BASS_PREFERENCE_MAX_BONUS,
    _auto_optuna_attach_out_telemetry,
    _auto_optuna_bass_preference_bonus,
    _auto_optuna_run_token,
)
from .optuna_backend_storage import (
    _auto_optuna_create_study,
    _auto_optuna_get_known_signatures,
    _auto_optuna_module_ready,
    _auto_optuna_note_trial_scan,
    _auto_optuna_prime_known_signatures_from_study,
    _auto_optuna_update_known_record,
)

logger = logging.getLogger("DecayCore")

_RECOVERABLE_OPTUNA_RECORD_EXCEPTIONS = (
    AttributeError,
    KeyError,
    TypeError,
    ValueError,
    RuntimeError,
    OSError,
)


def _auto_optuna_trial_params_from_trial(tr, *, seed_to_params=None) -> dict:
    try:
        user_attrs = dict(getattr(tr, "user_attrs", {}) or {})
    except _RECOVERABLE_OPTUNA_RECORD_EXCEPTIONS:
        user_attrs = {}
    payload_preset = _auto_optuna_trial_payload_preset(user_attrs)
    try:
        params = dict(getattr(tr, "params", {}) or {})
    except _RECOVERABLE_OPTUNA_RECORD_EXCEPTIONS:
        params = {}
    if callable(seed_to_params) and payload_preset:
        try:
            canonical_params = dict(seed_to_params(dict(payload_preset)) or {})
        except _RECOVERABLE_OPTUNA_RECORD_EXCEPTIONS:
            canonical_params = {}
        if canonical_params:
            params = dict(canonical_params)
    return dict(params)


def _auto_optuna_trial_value_from_trial(tr) -> float:
    val = getattr(tr, "value", None)
    if val is None:
        vals = getattr(tr, "values", None)
        if isinstance(vals, (list, tuple)) and vals:
            val = vals[0]
    try:
        val_f = float(val)
    except _RECOVERABLE_OPTUNA_RECORD_EXCEPTIONS:
        return float("nan")
    return float(val_f)


def _auto_optuna_record_from_trial(tr, *, seed_to_params=None) -> tuple[str, dict] | None:
    params = _auto_optuna_trial_params_from_trial(tr, seed_to_params=seed_to_params)
    sig = _auto_optuna_param_signature(params)
    if not sig:
        return None
    rec = {"params": dict(params)}
    val_f = _auto_optuna_trial_value_from_trial(tr)
    if np.isfinite(val_f):
        rec["value"] = float(val_f)
    state = getattr(tr, "state", None)
    if state is not None:
        rec["state"] = state
    try:
        user_attrs = dict(getattr(tr, "user_attrs", {}) or {})
    except _RECOVERABLE_OPTUNA_RECORD_EXCEPTIONS:
        user_attrs = {}
    cached_out = dict(user_attrs.get(AUTO_MODE_OPTUNA_USER_ATTR_OUT, {}) or {})
    if cached_out:
        rec["out"] = cached_out
    return sig, rec


def _auto_optuna_study_records(study, *, seed_to_params=None) -> dict[str, dict]:
    try:
        trials = study.get_trials(deepcopy=False)
    except TypeError:
        trials = study.get_trials()
    except _RECOVERABLE_OPTUNA_RECORD_EXCEPTIONS:
        trials = getattr(study, "trials", [])
    trials = list(trials or [])
    _auto_optuna_note_trial_scan(len(trials))
    out: dict[str, dict] = {}
    for tr in trials:
        row = _auto_optuna_record_from_trial(tr, seed_to_params=seed_to_params)
        if row is None:
            continue
        sig, rec = row
        out[sig] = rec
    return out


def _auto_optuna_build_trial_system_attrs(
    *,
    base_data: dict | None,
    scope_eff: str,
    phase_kind: str | None,
    metrics: dict | None,
    constraint_fn,
) -> dict | None:
    if not callable(constraint_fn):
        return None
    try:
        thr = _auto_optuna_constraint_thresholds(base_data, scope_eff)
        use_events = _auto_optuna_use_events_constraint(base_data, phase_kind=phase_kind)
        cv = _auto_optuna_constraint_vector_from_metrics(
            dict(metrics or {}),
            max_mode_ripple_db=float(thr["max_mode_ripple_db"]),
            max_events_severity=float(thr["max_events_severity"]),
            max_net_boost_db=float(thr["max_net_boost_db"]),
            use_events=bool(use_events),
        )
        return {"constraints": list(cv)}
    except _RECOVERABLE_OPTUNA_RECORD_EXCEPTIONS:
        return None


def _auto_optuna_remember_with_add_trial(
    *,
    optuna_mod,
    study,
    params: dict,
    value: float,
    payload_json: dict,
    base_data: dict | None,
    study_name: str,
    params_sig: str,
    trial_system_attrs: dict | None,
) -> bool:
    add_trial_obj = _auto_optuna_build_completed_trial(
        optuna_mod,
        params=params,
        value=float(value),
        user_attrs={AUTO_MODE_OPTUNA_USER_ATTR_OUT: payload_json},
        base_data=base_data,
        system_attrs=trial_system_attrs,
    )
    if add_trial_obj is None:
        return False
    try:
        study.add_trial(add_trial_obj)
        _auto_optuna_update_known_record(
            study_name,
            params_sig,
            {
                "params": dict(params),
                "value": float(value),
                "out": dict(payload_json or {}),
            },
        )
        return True
    except _RECOVERABLE_OPTUNA_RECORD_EXCEPTIONS:
        logger.exception("optuna add_trial")
        return False


def _auto_optuna_remember_with_enqueue_tell(
    *,
    study,
    params: dict,
    value: float,
    payload_json: dict,
    study_name: str,
    params_sig: str,
) -> bool:
    if not hasattr(study, "enqueue_trial") or not hasattr(study, "ask") or not hasattr(study, "tell"):
        return False
    try:
        study.enqueue_trial(dict(params))
        trial_obj = study.ask()
    except _RECOVERABLE_OPTUNA_RECORD_EXCEPTIONS:
        return False
    try:
        if hasattr(trial_obj, "set_user_attr"):
            trial_obj.set_user_attr(
                AUTO_MODE_OPTUNA_USER_ATTR_OUT,
                payload_json,
            )
    except _RECOVERABLE_OPTUNA_RECORD_EXCEPTIONS:
        logger.exception("optuna trial user attr set")
    try:
        study.tell(trial_obj, float(value))
    except _RECOVERABLE_OPTUNA_RECORD_EXCEPTIONS:
        return False
    _auto_optuna_update_known_record(
        study_name,
        params_sig,
        {
            "params": dict(params),
            "value": float(value),
            "out": dict(payload_json or {}),
        },
    )
    return True

def _auto_optuna_remember_result(
    optuna_mod,
    *,
    base_data: dict | None,
    study_name: str | None,
    study_scope: str | None = None,
    phase_kind: str | None = None,
    seed: int,
    preset: dict | None,
    metrics: dict | None,
    seed_to_params=None,
    use_refine_tiebreak: bool = False,
    out_payload: dict | None = None,
) -> bool:
    if (not _auto_optuna_module_ready(optuna_mod)) or not study_name or not callable(seed_to_params):
        return False
    params = {}
    try:
        params = _auto_optuna_sanitize_enqueued_params(
            dict(seed_to_params(dict(preset or {})) or {}),
            base_data=base_data,
        )
    except _RECOVERABLE_OPTUNA_RECORD_EXCEPTIONS:
        params = {}
    params_sig = _auto_optuna_param_signature(params)
    if not params_sig:
        return False
    scope_eff = _auto_optuna_effective_scope(base_data, study_scope or study_name, phase_kind=phase_kind)
    run_token = _auto_optuna_run_token(
        study_name=study_name,
        study_scope=scope_eff,
        seed=int(seed),
        total=1,
        startup_trials=1,
    )
    sampler_kwargs = dict(_auto_optuna_sampler_kwargs(base_data, workers=1) or {})
    constraint_fn = _auto_optuna_constraints_func(
        base_data=base_data,
        scope=scope_eff,
        phase_kind=phase_kind,
    )
    if callable(constraint_fn):
        sampler_kwargs["constraints_func"] = constraint_fn
    sampler = optuna_mod.samplers.TPESampler(
        seed=int(seed),
        n_startup_trials=1,
        **sampler_kwargs,
    )
    study = _auto_optuna_create_study(
        optuna_mod,
        sampler=sampler,
        base_data=base_data,
        study_name=study_name,
    )
    # Duplicate guard: prime signature set once, then check from set (avoids O(N²) scan)
    _auto_optuna_prime_known_signatures_from_study(study_name, study, seed_to_params=seed_to_params)
    if params_sig in _auto_optuna_get_known_signatures(study_name):
        return False
    value = _auto_optuna_objective_value(
        dict(metrics or {}),
        use_refine_tiebreak=bool(use_refine_tiebreak),
        goal=str((base_data or {}).get("auto_goal", "") or ""),
    )
    payload = _auto_optuna_attach_out_telemetry(
        out_payload
        or {
            "ok": True,
            "metrics": dict(metrics or {}),
            "trial_preset": dict(preset or {}),
            "replayed_from_cache": True,
        },
        base_data=base_data,
        study_name=study_name,
        study_scope=scope_eff,
        phase_kind=phase_kind,
        run_token=run_token,
        source="remembered",
        objective_value_num=float(value) if np.isfinite(value) else None,
    )
    payload_json = _auto_optuna_jsonable(dict(payload or {}))
    if hasattr(study, "add_trial"):
        trial_system_attrs = _auto_optuna_build_trial_system_attrs(
            base_data=base_data,
            scope_eff=scope_eff,
            phase_kind=phase_kind,
            metrics=metrics,
            constraint_fn=constraint_fn,
        )
        if _auto_optuna_remember_with_add_trial(
            optuna_mod=optuna_mod,
            study=study,
            params=params,
            value=float(value),
            payload_json=dict(payload_json or {}),
            base_data=base_data,
            study_name=str(study_name),
            params_sig=str(params_sig),
            trial_system_attrs=trial_system_attrs,
        ):
            return True
    return _auto_optuna_remember_with_enqueue_tell(
        study=study,
        params=params,
        value=float(value),
        payload_json=dict(payload_json or {}),
        study_name=str(study_name),
        params_sig=str(params_sig),
    )

def _auto_optuna_objective_value(metrics: dict | None, *, use_refine_tiebreak: bool = False, goal: str | None = None) -> float:
    met = dict(metrics or {})
    key = "rank_score_refine" if bool(use_refine_tiebreak) else "rank_score"
    value = _auto_safe_float(met.get(key, float("nan")), float("nan"))
    if (not np.isfinite(value)) and bool(use_refine_tiebreak):
        value = _auto_safe_float(met.get("rank_score", float("nan")), float("nan"))
    if np.isfinite(value):
        bass_bonus = _auto_optuna_bass_preference_bonus(met)
        if _auto_goal_norm(goal) == AUTO_MODE_GOAL_FLAT:
            bass_bonus = float(np.clip(bass_bonus * 3.0, 0.0, _AUTO_OPTUNA_BASS_PREFERENCE_MAX_BONUS * 3.0))
        return float(value + bass_bonus)
    return 0.0
