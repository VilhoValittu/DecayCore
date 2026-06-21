# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Optuna backend — scoring, objective value, telemetry building."""

from __future__ import annotations

import hashlib
import json
import logging

import numpy as np

from .optuna_telemetry import _auto_metric_summary
from .shared import (
    AUTO_MODE_CACHE_SCHEMA_VERSION,
    AUTO_MODE_OPTUNA_CONSTRAINTS_ZERO_FEASIBLE_FALLBACK,
    AUTO_MODE_OPTUNA_USER_ATTR_OUT,
    _auto_safe_bool,
    _auto_safe_float,
)
from .optuna_backend_constraints import (
    _auto_optuna_constraint_thresholds,
    _auto_optuna_constraint_vector_from_metrics,
    _auto_optuna_constraints_enabled_for_scope,
    _auto_optuna_is_refine_phase_kind,
    _auto_optuna_use_events_constraint,
)
from .optuna_backend_storage import (
    _auto_optuna_note_trial_scan,
    _auto_optuna_study_scan_stats_snapshot as _auto_optuna_study_scan_stats_snapshot,
)

logger = logging.getLogger("DecayCore")

_AUTO_OPTUNA_BASS_PREFERENCE_LF_BOOST_TARGET_DB = 5.0
_AUTO_OPTUNA_BASS_PREFERENCE_LF_BOOST_WEIGHT = 0.75
_AUTO_OPTUNA_BASS_PREFERENCE_REALIZED_RMS_OK_DB = 0.45
_AUTO_OPTUNA_BASS_PREFERENCE_REALIZED_RMS_WEIGHT = 0.35
_AUTO_OPTUNA_BASS_PREFERENCE_MAX_BONUS = 1.50


def _auto_optuna_run_token(
    *,
    study_name: str | None,
    study_scope: str | None,
    seed: int,
    total: int,
    startup_trials: int,
) -> str:
    payload = {
        "study_name": str(study_name or ""),
        "study_scope": str(study_scope or ""),
        "seed": int(seed),
        "total": int(total),
        "startup_trials": int(startup_trials),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8", "ignore")).hexdigest()[:16]

def _auto_optuna_constraint_info_for_metrics(
    *,
    base_data: dict | None,
    scope: str | None,
    metrics: dict | None,
    phase_kind: str | None = None,
) -> dict:
    enabled = bool(_auto_optuna_constraints_enabled_for_scope(base_data, scope, phase_kind=phase_kind))
    if not enabled:
        return {
            "constraints_active": False,
            "feasible": None,
            "violations": {},
            "constraint_flags": {},
        }

    thr = _auto_optuna_constraint_thresholds(base_data, scope)
    use_events = _auto_optuna_use_events_constraint(
        base_data,
        phase_kind=phase_kind,
    )
    vec = _auto_optuna_constraint_vector_from_metrics(
        dict(metrics or {}),
        max_mode_ripple_db=float(thr["max_mode_ripple_db"]),
        max_events_severity=float(thr["max_events_severity"]),
        max_net_boost_db=float(thr["max_net_boost_db"]),
        use_events=bool(use_events),
    )
    ripple_v, events_v, boost_v = vec
    feasible = bool(
        float(ripple_v) <= 0.0
        and float(events_v) <= 0.0
        and float(boost_v) <= 0.0
    )
    return {
        "constraints_active": True,
        "feasible": bool(feasible),
        "violations": {
            "ripple": float(ripple_v),
            "events": float(events_v),
            "boost": float(boost_v),
        },
        "constraint_flags": {
            "use_events": bool(use_events),
        },
    }

def _auto_optuna_trial_objective_value(trial, out_payload: dict | None = None) -> float | None:
    out = dict(out_payload or {})
    opt_meta = dict(out.get("optuna", {}) or {})
    val = opt_meta.get("objective_value", None)
    try:
        if val is not None and np.isfinite(float(val)):
            return float(val)
    except (TypeError, ValueError):
        pass

    direct_val = getattr(trial, "value", None)
    try:
        if direct_val is not None and np.isfinite(float(direct_val)):
            return float(direct_val)
    except (TypeError, ValueError):
        pass

    vals = getattr(trial, "values", None)
    try:
        if vals and np.isfinite(float(vals[0])):
            return float(vals[0])
    except (TypeError, ValueError, IndexError):
        pass

    return None


def _auto_optuna_bass_preference_bonus(metrics: dict | None) -> float:
    """Keep Optuna slightly on the bass-forward side without overriding safety penalties."""
    met = dict(metrics or {})
    lf_boost = _auto_safe_float(
        met.get(
            "bass_boost_20_200_db",
            met.get("lf_boost_max_db", met.get("post_filter_boost_peak_db", float("nan"))),
        ),
        float("nan"),
    )
    realized_rms = _auto_safe_float(
        met.get("target_tracking_rms_20_200_db", met.get("realized_rms_20_200_db", float("nan"))),
        float("nan"),
    )
    bass_pen = max(0.0, _auto_safe_float(met.get("bass_integration_penalty", 0.0), 0.0))
    feas_pen = max(0.0, _auto_safe_float(met.get("bass_feasibility_penalty", 0.0), 0.0))
    net_boost = _auto_safe_float(met.get("max_net_boost_db", float("nan")), float("nan"))

    lf_bonus = 0.0
    if np.isfinite(lf_boost):
        lf_norm = float(
            np.clip(
                float(lf_boost) / float(_AUTO_OPTUNA_BASS_PREFERENCE_LF_BOOST_TARGET_DB),
                0.0,
                1.0,
            )
        )
        lf_bonus = float(_AUTO_OPTUNA_BASS_PREFERENCE_LF_BOOST_WEIGHT) * float(lf_norm)

    realized_bonus = 0.0
    if np.isfinite(realized_rms):
        realized_norm = float(
            1.0
            - np.clip(
                float(realized_rms) / float(_AUTO_OPTUNA_BASS_PREFERENCE_REALIZED_RMS_OK_DB),
                0.0,
                1.0,
            )
        )
        realized_bonus = float(_AUTO_OPTUNA_BASS_PREFERENCE_REALIZED_RMS_WEIGHT) * float(realized_norm)

    penalty_scale = float(1.0 / (1.0 + 0.12 * float(bass_pen) + 0.28 * float(feas_pen)))
    if np.isfinite(net_boost):
        penalty_scale *= float(1.0 / (1.0 + 0.30 * max(0.0, float(net_boost) - 5.0)))

    bonus = float(lf_bonus + realized_bonus) * float(penalty_scale)
    return float(np.clip(bonus, 0.0, _AUTO_OPTUNA_BASS_PREFERENCE_MAX_BONUS))

def _auto_optuna_attach_out_telemetry(
    out: dict | None,
    *,
    base_data: dict | None,
    study_name: str | None,
    study_scope: str | None,
    phase_kind: str | None = None,
    run_token: str,
    source: str,
    objective_value_num: float | None,
) -> dict:
    out2 = dict(out or {})
    metrics = dict(out2.get("metrics", {}) or {})
    bass_preference_bonus = _auto_optuna_bass_preference_bonus(metrics)
    cinfo = _auto_optuna_constraint_info_for_metrics(
        base_data=base_data,
        scope=study_scope,
        metrics=metrics,
        phase_kind=phase_kind,
    )
    out2["optuna"] = {
        "study_name": str(study_name or ""),
        "study_scope": str(study_scope or ""),
        "phase_kind": str(phase_kind or ""),
        "run_token": str(run_token),
        "source": str(source or ""),
        "objective_value": (
            None
            if objective_value_num is None or not np.isfinite(float(objective_value_num))
            else float(objective_value_num)
        ),
        "bass_preference_bonus": float(bass_preference_bonus),
        "constraints_active": bool(cinfo.get("constraints_active", False)),
        "feasible": cinfo.get("feasible", None),
        "violations": dict(cinfo.get("violations", {}) or {}),
        "constraint_flags": dict(cinfo.get("constraint_flags", {}) or {}),
    }
    return out2

def _auto_optuna_base_data_without_constraints(base_data: dict | None) -> dict:
    data = dict(base_data or {})
    data["auto_mode_optuna_constraints"] = False
    return data

def _auto_optuna_needs_zero_feasible_rescue(
    *,
    base_data: dict | None,
    phase_kind: str | None,
    telemetry: dict | None,
) -> bool:
    tel = dict(telemetry or {})
    enabled = _auto_safe_bool(
        (base_data or {}).get(
            "auto_mode_optuna_constraints_zero_feasible_fallback",
            AUTO_MODE_OPTUNA_CONSTRAINTS_ZERO_FEASIBLE_FALLBACK,
        ),
        AUTO_MODE_OPTUNA_CONSTRAINTS_ZERO_FEASIBLE_FALLBACK,
    )
    if not enabled:
        return False

    kind = str(phase_kind or "").strip().lower()
    if kind not in {"local", "micro"}:
        return False
    if not bool(tel.get("constraints_active", False)):
        return False
    if _auto_optuna_is_refine_phase_kind(phase_kind):
        cflags = dict(tel.get("constraint_flags", {}) or {})
        if not bool(cflags.get("use_events", True)):
            vc = dict(tel.get("violation_counts", {}) or {})
            ripple_bad = int(vc.get("ripple", 0) or 0)
            boost_bad = int(vc.get("boost", 0) or 0)
            if ripple_bad == 0 and boost_bad == 0:
                return False

    complete_n = int(tel.get("complete_trials", 0) or 0)
    feasible_n = int(tel.get("feasible_trials", 0) or 0)
    infeasible_n = int(tel.get("infeasible_trials", 0) or 0)
    return bool(complete_n > 0 and feasible_n == 0 and infeasible_n > 0)


def _auto_optuna_collect_run_trials(study, *, run_token: str) -> list[tuple]:
    try:
        trials = list(
            study.get_trials(deepcopy=False)
            if hasattr(study, "get_trials")
            else getattr(study, "trials", [])
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
    ):
        trials = []
    _auto_optuna_note_trial_scan(len(trials))

    run_trials = []
    for tr in list(trials or []):
        try:
            user_attrs = dict(getattr(tr, "user_attrs", {}) or {})
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
            user_attrs = {}
        out = dict(user_attrs.get(AUTO_MODE_OPTUNA_USER_ATTR_OUT, {}) or {})
        opt_meta = dict(out.get("optuna", {}) or {})
        if str(opt_meta.get("run_token", "")) == str(run_token):
            run_trials.append((tr, out, opt_meta))
    return run_trials


def _auto_optuna_init_run_telemetry_state() -> dict:
    return {
        "state_counts": {},
        "complete_n": 0,
        "fail_n": 0,
        "feasible_n": 0,
        "infeasible_n": 0,
        "best_raw_value": None,
        "best_raw_trial": None,
        "best_feasible_value": None,
        "best_feasible_trial": None,
        "violation_counts": {"ripple": 0, "events": 0, "boost": 0},
        "violation_max": {"ripple": 0.0, "events": 0.0, "boost": 0.0},
        "source_counts": {},
        "events_all": [],
        "events_feasible": [],
        "events_infeasible": [],
        "constraint_flags": {},
        "run_phase_kind": "",
    }


def _auto_optuna_update_run_telemetry_state_counts(state: dict, tr, opt_meta: dict) -> None:
    state_obj = getattr(tr, "state", None)
    state_name = str(getattr(state_obj, "name", state_obj or "UNKNOWN"))
    state_counts = state["state_counts"]
    state_counts[state_name] = int(state_counts.get(state_name, 0) or 0) + 1

    source = str(opt_meta.get("source", "") or "")
    if source:
        source_counts = state["source_counts"]
        source_counts[source] = int(source_counts.get(source, 0) or 0) + 1
    if not state["run_phase_kind"]:
        state["run_phase_kind"] = str(opt_meta.get("phase_kind", "") or "")

    if state_name == "COMPLETE":
        state["complete_n"] += 1
    elif state_name == "FAIL":
        state["fail_n"] += 1
    state["_state_name"] = state_name


def _auto_optuna_update_run_telemetry_state_constraints(state: dict, opt_meta: dict) -> None:
    constraints_active = bool(opt_meta.get("constraints_active", False))
    feasible = opt_meta.get("feasible", None)
    violations = dict(opt_meta.get("violations", {}) or {})
    trial_constraint_flags = dict(opt_meta.get("constraint_flags", {}) or {})
    if trial_constraint_flags and not state["constraint_flags"]:
        state["constraint_flags"] = dict(trial_constraint_flags)

    if constraints_active and feasible is True:
        state["feasible_n"] += 1
    elif constraints_active and feasible is False:
        state["infeasible_n"] += 1

    violation_counts = state["violation_counts"]
    violation_max = state["violation_max"]
    for key in ("ripple", "events", "boost"):
        v = _auto_safe_float(violations.get(key, 0.0), 0.0)
        if float(v) > 0.0:
            violation_counts[key] = int(violation_counts.get(key, 0) or 0) + 1
            violation_max[key] = float(max(float(violation_max.get(key, 0.0) or 0.0), float(v)))


def _auto_optuna_update_run_telemetry_state_complete(state: dict, tr, out: dict, opt_meta: dict) -> None:
    if state.get("_state_name") != "COMPLETE":
        return

    metrics = dict(out.get("metrics", {}) or {})
    events_val = _auto_safe_float(metrics.get("events_severity", float("nan")), float("nan"))
    if np.isfinite(events_val):
        state["events_all"].append(float(events_val))
        constraints_active = bool(opt_meta.get("constraints_active", False))
        feasible = opt_meta.get("feasible", None)
        if constraints_active and feasible is True:
            state["events_feasible"].append(float(events_val))
        elif constraints_active and feasible is False:
            state["events_infeasible"].append(float(events_val))

    obj_val = _auto_optuna_trial_objective_value(tr, out)
    if obj_val is not None:
        if state["best_raw_value"] is None or float(obj_val) > float(state["best_raw_value"]):
            state["best_raw_value"] = float(obj_val)
            state["best_raw_trial"] = int(getattr(tr, "number", -1))

        feasible_ok = opt_meta.get("feasible", None)
        if feasible_ok is True:
            if state["best_feasible_value"] is None or float(obj_val) > float(state["best_feasible_value"]):
                state["best_feasible_value"] = float(obj_val)
                state["best_feasible_trial"] = int(getattr(tr, "number", -1))


def _auto_optuna_constraint_threshold_summary(base_data: dict | None, study_scope: str | None) -> dict:
    constraint_thresholds = {}
    try:
        thr = _auto_optuna_constraint_thresholds(base_data, study_scope)
        constraint_thresholds = {
            "max_events_severity": float(thr["max_events_severity"]),
            "max_mode_ripple_db": float(thr["max_mode_ripple_db"]),
            "max_net_boost_db": float(thr["max_net_boost_db"]),
        }
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
        constraint_thresholds = {}
    return constraint_thresholds

def _auto_optuna_build_run_telemetry(
    study,
    *,
    base_data: dict | None,
    study_name: str | None,
    study_scope: str | None,
    phase_kind: str | None,
    run_token: str,
    requested_total: int,
    startup_trials: int,
    duplicate_skips: int,
    duplicate_replays: int,
    duplicate_reserved: int,
) -> dict:
    run_trials = _auto_optuna_collect_run_trials(study, run_token=run_token)
    state = _auto_optuna_init_run_telemetry_state()
    for tr, out, opt_meta in list(run_trials):
        _auto_optuna_update_run_telemetry_state_counts(state, tr, opt_meta)
        _auto_optuna_update_run_telemetry_state_constraints(state, opt_meta)
        _auto_optuna_update_run_telemetry_state_complete(state, tr, out, opt_meta)

    startup_complete = int(min(max(1, int(startup_trials)), int(state["complete_n"]))) if state["complete_n"] > 0 else 0
    model_complete = int(max(0, int(state["complete_n"]) - int(startup_complete)))
    run_phase_kind = str(state["run_phase_kind"] or phase_kind or "")
    constraints_active_any = bool(_auto_optuna_constraints_enabled_for_scope(base_data, study_scope, phase_kind=phase_kind or run_phase_kind))
    constraint_thresholds = _auto_optuna_constraint_threshold_summary(base_data, study_scope)

    return {
        "study_name": str(study_name or "in-memory"),
        "study_scope": str(study_scope or ""),
        "phase_kind": str(run_phase_kind or phase_kind or ""),
        "run_token": str(run_token),
        "cache_schema_version": int(AUTO_MODE_CACHE_SCHEMA_VERSION),
        "requested_total": int(requested_total),
        "run_trials": int(len(run_trials)),
        "complete_trials": int(state["complete_n"]),
        "failed_trials": int(state["fail_n"]),
        "state_counts": dict(state["state_counts"] or {}),
        "startup_trials": int(startup_trials),
        "startup_complete": int(startup_complete),
        "model_complete": int(model_complete),
        "duplicate_skips": int(duplicate_skips),
        "duplicate_replays": int(duplicate_replays),
        "duplicate_reserved": int(duplicate_reserved),
        "constraints_active": bool(constraints_active_any),
        "feasible_trials": int(state["feasible_n"]) if bool(constraints_active_any) else 0,
        "infeasible_trials": int(state["infeasible_n"]) if bool(constraints_active_any) else 0,
        "best_raw_value": state["best_raw_value"],
        "best_raw_trial": state["best_raw_trial"],
        "best_feasible_value": state["best_feasible_value"] if bool(constraints_active_any) else None,
        "best_feasible_trial": state["best_feasible_trial"] if bool(constraints_active_any) else None,
        "violation_counts": dict(state["violation_counts"] or {}),
        "violation_max": dict(state["violation_max"] or {}),
        "events_summary": _auto_metric_summary(state["events_all"]),
        "events_feasible_summary": _auto_metric_summary(state["events_feasible"]),
        "events_infeasible_summary": _auto_metric_summary(state["events_infeasible"]),
        "constraint_thresholds": dict(constraint_thresholds or {}),
        "constraint_flags": dict(state["constraint_flags"] or {}),
        "source_counts": dict(state["source_counts"] or {}),
    }
