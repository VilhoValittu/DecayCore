# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Optuna backend — constraint functions and phase/scope helpers."""

from __future__ import annotations

import logging
import re

import numpy as np

from .scoring_ranking import _auto_ripple_metric_for_gate
from .shared import (
    AUTO_MODE_OPTUNA_CONSTRAINTS_ENABLED,
    AUTO_MODE_OPTUNA_CONSTRAINTS_MAX_EVENTS_SEVERITY,
    AUTO_MODE_OPTUNA_CONSTRAINTS_MAX_MODE_RIPPLE_DB,
    AUTO_MODE_OPTUNA_CONSTRAINTS_MAX_NET_BOOST_DB,
    AUTO_MODE_OPTUNA_CONSTRAINTS_REFINE_ONLY,
    AUTO_MODE_OPTUNA_CONSTRAINTS_USE_EVENTS_IN_REFINE,
    AUTO_MODE_OPTUNA_USER_ATTR_OUT,
    _auto_filter_cache_key,
    AutoModeConfig,
    _auto_safe_bool,
    _auto_safe_float,
)

logger = logging.getLogger("DecayCore")


def _auto_optuna_startup_for_phase_kind(cfg, *, phase_kind: str | None, total: int) -> int:
    kind = str(phase_kind or "").strip().lower()
    total_i = int(max(1, total))

    if kind == "phase1":
        base = int(getattr(cfg, "optuna_startup_phase1", getattr(cfg, "optuna_pilot_startup_trials", 12)))
    elif kind == "target":
        base = int(getattr(cfg, "optuna_startup_target", getattr(cfg, "optuna_pilot_startup_trials", 12)))
    elif kind == "local":
        base = int(getattr(cfg, "optuna_startup_local", getattr(cfg, "optuna_pilot_startup_trials", 12)))
    elif kind == "micro":
        base = int(getattr(cfg, "optuna_startup_micro", getattr(cfg, "optuna_pilot_startup_trials", 12)))
    else:
        base = int(getattr(cfg, "optuna_pilot_startup_trials", 12))

    return int(max(1, min(base, total_i)))

def _auto_optuna_is_refine_phase_kind(phase_kind: str | None) -> bool:
    kind = str(phase_kind or "").strip().lower()
    return kind in {"local", "micro"}

def _auto_optuna_constraint_scope_kind(scope: str | None) -> str:
    scope_txt = str(scope or "").strip().lower()
    if not scope_txt:
        return ""
    if "phase2-local" in scope_txt or "local_center_" in scope_txt:
        return "local"
    if "phase3-micro" in scope_txt or "cache-micro" in scope_txt or "cache_micro" in scope_txt:
        return "micro"
    return ""

def _auto_optuna_constraints_enabled_for_scope(
    base_data: dict | None,
    scope: str | None,
    *,
    phase_kind: str | None = None,
) -> bool:
    data = dict(base_data or {})
    enabled = _auto_safe_bool(
        data.get("auto_mode_optuna_constraints", AUTO_MODE_OPTUNA_CONSTRAINTS_ENABLED),
        AUTO_MODE_OPTUNA_CONSTRAINTS_ENABLED,
    )
    if not enabled:
        return False
    refine_only = _auto_safe_bool(
        data.get("auto_mode_optuna_constraints_refine_only", AUTO_MODE_OPTUNA_CONSTRAINTS_REFINE_ONLY),
        AUTO_MODE_OPTUNA_CONSTRAINTS_REFINE_ONLY,
    )
    if not refine_only:
        return True
    if str(phase_kind or "").strip():
        return bool(_auto_optuna_is_refine_phase_kind(phase_kind))
    return bool(_auto_optuna_constraint_scope_kind(scope))


def _auto_optuna_scope_for_filter(
    base_data: dict | None,
    scope: str | None,
    *,
    filter_key: str | None = None,
) -> str:
    scope_txt = str(scope or "study").strip() or "study"
    fk = str(
        _auto_filter_cache_key(
            base_data if filter_key is None else None,
            filter_type=filter_key,
        )
    )
    suffix = f"filter-{fk}"
    if re.search(r"(?:^|-)filter-(?:linear|mixed|minimum|asym)(?:-|$)", scope_txt.lower()):
        return str(scope_txt)
    if suffix in scope_txt.lower():
        return str(scope_txt)
    return f"{scope_txt}-{suffix}"


def _auto_optuna_effective_scope(
    base_data: dict | None,
    scope: str | None,
    *,
    phase_kind: str | None = None,
) -> str:
    scope_txt = _auto_optuna_scope_for_filter(base_data, scope)
    if _auto_optuna_is_refine_phase_kind(phase_kind) and str(phase_kind or "").strip().lower() == "local":
        if not str(scope_txt).lower().endswith("-locv2") and "-locv2-" not in str(scope_txt).lower():
            scope_txt = f"{scope_txt}-locv2"
    if str(scope_txt).lower().endswith("-c1"):
        return str(scope_txt)
    if _auto_optuna_constraints_enabled_for_scope(base_data, scope_txt, phase_kind=phase_kind):
        return f"{scope_txt}-c1"
    return str(scope_txt)

def _auto_optuna_constraint_thresholds(base_data: dict | None, scope: str | None) -> dict:
    data = dict(base_data or {})
    kind = _auto_optuna_constraint_scope_kind(scope)

    max_mode_ripple = max(
        0.0,
        _auto_safe_float(
            data.get(
                "auto_mode_optuna_constraints_max_mode_ripple_db",
                AUTO_MODE_OPTUNA_CONSTRAINTS_MAX_MODE_RIPPLE_DB,
            ),
            AUTO_MODE_OPTUNA_CONSTRAINTS_MAX_MODE_RIPPLE_DB,
        ),
    )
    max_events = max(
        0.0,
        _auto_safe_float(
            data.get(
                "auto_mode_optuna_constraints_max_events_severity",
                AUTO_MODE_OPTUNA_CONSTRAINTS_MAX_EVENTS_SEVERITY,
            ),
            AUTO_MODE_OPTUNA_CONSTRAINTS_MAX_EVENTS_SEVERITY,
        ),
    )
    max_boost = max(
        0.0,
        _auto_safe_float(
            data.get(
                "auto_mode_optuna_constraints_max_net_boost_db",
                AUTO_MODE_OPTUNA_CONSTRAINTS_MAX_NET_BOOST_DB,
            ),
            AUTO_MODE_OPTUNA_CONSTRAINTS_MAX_NET_BOOST_DB,
        ),
    )

    return {
        "kind": str(kind),
        "max_mode_ripple_db": float(max_mode_ripple),
        "max_events_severity": float(max_events),
        "max_net_boost_db": float(max_boost),
    }

def _auto_optuna_trial_out_payload(trial) -> dict:
    try:
        user_attrs = dict(getattr(trial, "user_attrs", {}) or {})
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
    out = user_attrs.get(AUTO_MODE_OPTUNA_USER_ATTR_OUT, {})
    if isinstance(out, dict):
        return dict(out or {})
    return {}

def _auto_optuna_constraint_vector_from_metrics(
    metrics: dict | None,
    *,
    max_mode_ripple_db: float,
    max_events_severity: float,
    max_net_boost_db: float,
    use_events: bool = True,
) -> tuple[float, float, float]:
    met = dict(metrics or {})

    ripple = _auto_ripple_metric_for_gate(met)
    events = _auto_safe_float(met.get("events_severity", float("nan")), float("nan"))
    boost = _auto_safe_float(met.get("max_net_boost_db", float("nan")), float("nan"))

    ripple_violation = 0.0
    event_violation = 0.0
    boost_violation = 0.0

    if np.isfinite(ripple):
        ripple_violation = float(max(0.0, float(ripple) - float(max_mode_ripple_db)))
    if bool(use_events) and np.isfinite(events):
        event_violation = float(max(0.0, float(events) - float(max_events_severity)))
    if np.isfinite(boost):
        boost_violation = float(max(0.0, float(boost) - float(max_net_boost_db)))

    return (
        float(ripple_violation),
        float(event_violation),
        float(boost_violation),
    )

def _auto_optuna_use_events_constraint(
    base_data: dict | None,
    *,
    phase_kind: str | None,
) -> bool:
    data = dict(base_data or {})
    kind = str(phase_kind or "").strip().lower()

    if kind in {"local", "micro"}:
        return _auto_safe_bool(
            data.get(
                "auto_mode_optuna_constraints_use_events_in_refine",
                AUTO_MODE_OPTUNA_CONSTRAINTS_USE_EVENTS_IN_REFINE,
            ),
            AUTO_MODE_OPTUNA_CONSTRAINTS_USE_EVENTS_IN_REFINE,
        )

    return True

def _auto_optuna_constraints_func(
    *,
    base_data: dict | None,
    scope: str | None,
    phase_kind: str | None = None,
):
    if not _auto_optuna_constraints_enabled_for_scope(base_data, scope, phase_kind=phase_kind):
        return None

    thr = _auto_optuna_constraint_thresholds(base_data, scope)
    use_events = _auto_optuna_use_events_constraint(
        base_data,
        phase_kind=phase_kind,
    )
    logger.info(
        "Automatic mode Optuna constraints: phase_kind=%s use_events=%s scope=%s",
        str(phase_kind or ""),
        str(bool(use_events)),
        str(scope or ""),
    )

    def _constraints(trial):
        out = _auto_optuna_trial_out_payload(trial)
        metrics = dict(out.get("metrics", {}) or {})
        return _auto_optuna_constraint_vector_from_metrics(
            metrics,
            max_mode_ripple_db=float(thr["max_mode_ripple_db"]),
            max_events_severity=float(thr["max_events_severity"]),
            max_net_boost_db=float(thr["max_net_boost_db"]),
            use_events=bool(use_events),
        )

    return _constraints
