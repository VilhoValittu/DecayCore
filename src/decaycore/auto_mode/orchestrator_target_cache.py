# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Cache management helpers for target-curve selection orchestration."""

from __future__ import annotations

import logging

import numpy as np

from .cache_signature import (
    _auto_apply_seed,
    _auto_signature,
)
from .cache_measurement_sig import _auto_get_measurement_signature
from .shared import (
    AUTO_MODE_CACHE_ENABLED,
    AUTO_MODE_CACHE_SCHEMA_VERSION,
    AUTO_MODE_SYNTH_TARGET_NAME,
    _auto_builtin_target_name,
    _auto_safe_float,
)
from .orchestrator_target_types import (
    _TARGET_CACHE_REQUIRED_HIDDEN_KEYS,
    _TargetCacheState,
    _TargetSelectionSetup,
)

logger = logging.getLogger("DecayCore")


def _cache_target_valid(runtime, hc_name: str | None) -> bool:
    if str(hc_name or "").strip() == AUTO_MODE_SYNTH_TARGET_NAME:
        return False  # synthesized curves are measurement-specific, never cache
    hc = _auto_builtin_target_name(hc_name)
    if not hc:
        return False
    try:
        c_f, c_m = runtime.get_house_curve_by_name(hc)
        c_f = np.asarray(c_f, dtype=float).reshape(-1)
        c_m = np.asarray(c_m, dtype=float).reshape(-1)
        return bool(c_f.size >= 4 and c_m.size == c_f.size)
    except Exception:
        # Cache validation must stay resilient to malformed or missing house-curve data.
        return False


def _cached_target_fit(
    *,
    runtime,
    base_data: dict,
    measurements: dict,
    hc_name: str,
) -> tuple[float, float]:
    fit_rms_db = float("nan")
    offset_db = 0.0
    try:
        quick_fit = runtime.auto_select_builtin_target_curve(
            dict(base_data or {}),
            f_l=measurements.get("f_l"),
            m_l=measurements.get("m_l"),
            f_r=measurements.get("f_r"),
            m_r=measurements.get("m_r"),
        )
        candidates = list(
            (quick_fit or {}).get(
                "candidates_all",
                (quick_fit or {}).get("candidates", []),
            )
            or []
        )
        target_candidate = None
        for cand in candidates:
            if str((cand or {}).get("hc_mode", "") or "").strip() == str(hc_name):
                target_candidate = dict(cand or {})
                break
        if isinstance(target_candidate, dict):
            fit_rms_db = float(
                _auto_safe_float(
                    target_candidate.get("fit_rms_db", float("nan")),
                    float("nan"),
                )
            )
            offset_db = float(
                _auto_safe_float(target_candidate.get("offset_db", 0.0), 0.0)
            )
    except Exception:
        # Cached targets are best-effort metadata and should not abort target selection.
        pass
    return float(fit_rms_db), float(offset_db)


def _cached_target_return(
    *,
    runtime,
    cached_hc_mode: str | None,
    cached_preset: dict,
    selection_method: str,
    base_data: dict,
    measurements: dict,
    goal: str,
    rank_basis: str,
    cached_metrics: dict | None = None,
) -> dict | None:
    if not _cache_target_valid(runtime, cached_hc_mode):
        return None
    metrics = dict(cached_metrics or {}) if isinstance(cached_metrics, dict) else {}
    fit_rms_db = float(
        _auto_safe_float(
            metrics.get("fit_rms_db", metrics.get("preselect_score", float("nan"))),
            float("nan"),
        )
    )
    offset_db = float(
        _auto_safe_float(
            metrics.get("offset_db", 0.0),
            0.0,
        )
    )
    return {
        "selected_hc_mode": str(cached_hc_mode),
        "fit_rms_db": float(fit_rms_db),
        "offset_db": float(offset_db),
        "selection_method": str(selection_method),
        "selection_basis": str(rank_basis),
        "auto_goal": str(goal),
        "top_n": 0,
        "trials_per_curve": 0,
        "candidates": [],
        "evaluated": [],
        "best_preset": dict(cached_preset or {}),
        "best_metrics": dict(metrics or {}),
    }


def _cached_target_seed_preset(entry: dict | None) -> dict:
    payload = dict(entry or {}) if isinstance(entry, dict) else {}
    target_seed = payload.get("target_seed_preset")
    if isinstance(target_seed, dict) and target_seed:
        return dict(target_seed)
    best = payload.get("best_preset")
    if isinstance(best, dict) and best:
        return dict(best)
    return {}


def _missing_cached_target_hidden_keys(preset: dict | None) -> list[str]:
    payload = dict(preset or {}) if isinstance(preset, dict) else {}
    missing: list[str] = []
    for key in _TARGET_CACHE_REQUIRED_HIDDEN_KEYS:
        if key not in payload or payload.get(key) is None:
            missing.append(str(key))
    return missing


def _cached_target_has_exact_preset(
    *,
    runtime,
    cache_state: _TargetCacheState,
) -> bool:
    return bool(
        str(cache_state.cached_target_source or "").strip()
        and _cache_target_valid(runtime, cache_state.cached_target_hc)
        and isinstance(cache_state.cached_target_preset, dict)
        and bool(cache_state.cached_target_preset)
    )


def _set_cached_target_state(
    state: _TargetCacheState,
    *,
    runtime,
    cached_hc: str | None,
    cached_preset: dict | None,
    cached_metrics: dict | None = None,
    source: str,
    status_label: str,
    status_cb,
) -> bool:
    hc_name = _auto_builtin_target_name(cached_hc)
    if not _cache_target_valid(runtime, hc_name):
        return False
    state.cached_target_hc = str(hc_name)
    state.cached_target_preset = dict(cached_preset or {})
    state.cached_target_metrics = dict(cached_metrics or {})
    state.cached_target_source = str(source)
    logger.info(
        "Automatic mode target select: cache seed (%s) target=%s",
        str(status_label),
        str(state.cached_target_hc),
    )
    if callable(status_cb) and not bool(state.cached_target_preset):
        status_cb(
            "DecayCore automatic mode: target preselect cache seed "
            f"({str(status_label)} -> {str(state.cached_target_hc)})"
        )
    return True


def _cached_target_matches_current_hc(base_data: dict, cached_hc: str | None) -> bool:
    if not _auto_target_mode_locks_hc(base_data):
        return True
    current_hc = _auto_builtin_target_name(str(dict(base_data or {}).get("hc_mode", "") or ""))
    if not current_hc:
        return True
    cached_name = _auto_builtin_target_name(cached_hc)
    return bool(cached_name and str(cached_name) == str(current_hc))


def _auto_target_mode_locks_hc(base_data: dict | None) -> bool:
    target_mode = str(dict(base_data or {}).get("auto_target_mode", "auto") or "auto").strip().lower()
    return bool(target_mode in ("selected", "manual", "fixed", "user"))


def _cached_target_state_from_optuna_study(
    *,
    setup: _TargetSelectionSetup,
    base_data: dict,
    measurements: dict,
) -> _TargetCacheState | None:
    if not (
        str(setup.optimizer_backend) == "optuna"
        and bool(getattr(setup.cfg, "optuna_persistent_study", False))
        and setup.optuna_mod is not None
        and setup.runtime.auto_optuna_module_ready(setup.optuna_mod)
    ):
        return None
    try:
        storage_base_data = dict(base_data or {})
        storage_base_data["_optuna_measurement_sig"] = _auto_get_measurement_signature(measurements or {})
        storage_base_data["_optuna_journal_kind"] = "target"
        storage_base_data["_optuna_filter_key"] = str(setup.filter_key or "")
        storage = setup.runtime.auto_optuna_create_storage(
            setup.optuna_mod,
            base_data=storage_base_data,
        )
    except Exception:
        storage = None
    if storage is None:
        return None
    get_summaries = getattr(setup.optuna_mod, "get_all_study_summaries", None)
    load_study = getattr(setup.optuna_mod, "load_study", None)
    if not callable(get_summaries) or not callable(load_study):
        return None

    target_study_sig = str(setup.target_study_sig or "").strip()
    if not target_study_sig:
        return None

    best_hc = None
    best_study_name = None
    best_value = float("-inf")
    try:
        summaries = list(get_summaries(storage=storage) or [])
    except Exception:
        summaries = []
    for summary in summaries:
        study = None
        study_name = ""
        try:
            study_name = str(getattr(summary, "study_name", "") or "").strip()
        except Exception:
            study_name = ""
        attrs = dict(getattr(summary, "user_attrs", {}) or {})
        if not attrs and study_name:
            try:
                study = load_study(study_name=str(study_name), storage=storage)
                attrs = dict(getattr(study, "user_attrs", {}) or {})
            except Exception:
                attrs = {}
        if attrs.get("decaycore_kind") != "target_search":
            continue
        if str(attrs.get("decaycore_target_study_sig", "") or "") != target_study_sig:
            continue
        if (
            str(attrs.get("decaycore_filter_key", "") or "").strip().lower()
            != str(setup.filter_key or "").strip().lower()
        ):
            continue
        hc_name = _auto_builtin_target_name(attrs.get("decaycore_target_name", ""))
        if not _cache_target_valid(setup.runtime, hc_name):
            continue
        try:
            trial_obj = getattr(summary, "best_trial", None)
            if trial_obj is None and study is not None:
                trial_obj = getattr(study, "best_trial", None)
            score_value = float(getattr(trial_obj, "value", float("nan")))
        except Exception:
            score_value = float("nan")
        if not np.isfinite(score_value) or float(score_value) < float(best_value):
            continue
        best_hc = str(hc_name)
        best_study_name = str(study_name)
        best_value = float(score_value)

    if not best_hc or not best_study_name:
        return None
    if not _cached_target_matches_current_hc(base_data, best_hc):
        logger.info(
            "Automatic mode target select: Optuna target cache target=%s skipped; current target=%s",
            str(best_hc or "n/a"),
            str(base_data.get("hc_mode", "n/a") or "n/a"),
        )
        return None

    best_preset = {}
    try:
        study = load_study(study_name=str(best_study_name), storage=storage)
        best_trial = getattr(study, "best_trial", None)
        best_preset = setup.runtime.auto_optuna_trial_payload_preset(
            dict(getattr(best_trial, "user_attrs", {}) or {})
        )
        if not isinstance(best_preset, dict) or not best_preset:
            best_preset = dict(getattr(best_trial, "params", {}) or {})
    except Exception:
        best_preset = {}

    state = _TargetCacheState()
    if not _set_cached_target_state(
        state,
        runtime=setup.runtime,
        cached_hc=best_hc,
        cached_preset=best_preset,
        cached_metrics={},
        source="cache_optuna_target",
        status_label="optuna study",
        status_cb=None,
    ):
        return None
    logger.info(
        "Automatic mode target select: Optuna study cache seed target=%s, study=%s, value=%.3f",
        str(state.cached_target_hc),
        str(best_study_name),
        float(best_value),
    )
    return state


def _resolve_cached_target_state(
    *,
    setup: _TargetSelectionSetup,
    base_data: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    status_cb,
) -> _TargetCacheState:
    state = _TargetCacheState()
    cached_target_entry = None
    try:
        cached_target_entry = setup.runtime.auto_cache_get_target_for_measurements_global(
            measurements,
            goal=setup.goal,
            compat_version=setup.compat_version,
        )
    except Exception:
        cached_target_entry = None
    cached_source = "cache_measurement_global"
    cached_status = "measurement global"
    cached_preset = {}
    cached_metrics = {}
    if isinstance(cached_target_entry, dict):
        fk = str(setup.filter_key or "").strip()
        seed_map = cached_target_entry.get("filter_seed_presets", {})
        metric_map = cached_target_entry.get("filter_seed_metrics", {})
        if isinstance(seed_map, dict):
            cached_preset = dict(seed_map.get(fk, {}) or {})
        if isinstance(metric_map, dict):
            cached_metrics = dict(metric_map.get(fk, {}) or {})
        if cached_preset:
            cached_source = "cache_measurement_global_filter_seed"
            cached_status = "measurement global filter seed"
        else:
            cached_target_entry = None
            cached_preset = {}
            cached_metrics = {}
    if not isinstance(cached_target_entry, dict):
        try:
            cached_target_entry = setup.runtime.auto_cache_get_target_for_measurements(
                measurements,
                goal=setup.goal,
                filter_key=setup.filter_key,
                compat_version=setup.compat_version,
            )
        except Exception:
            cached_target_entry = None
        cached_source = "cache_measurement"
        cached_status = "measurement"
        if isinstance(cached_target_entry, dict):
            cached_preset = _cached_target_seed_preset(cached_target_entry)
            cached_metrics = dict(cached_target_entry.get("best_metrics", {}) or {})
    if isinstance(cached_target_entry, dict):
        cached_hc_measurement = str(
            cached_target_entry.get(
                "best_target_curve",
                cached_target_entry.get("best_hc_mode", ""),
            )
            or ""
        ).strip()
        if _cached_target_matches_current_hc(base_data, cached_hc_measurement):
            _set_cached_target_state(
                state,
                runtime=setup.runtime,
                cached_hc=cached_hc_measurement,
                cached_preset=dict(cached_preset or {}),
                cached_metrics=dict(cached_metrics or {}),
                source=str(cached_source),
                status_label=str(cached_status),
                status_cb=status_cb,
            )
        else:
            logger.info(
                "Automatic mode target select: measurement cache target=%s skipped; current target=%s",
                str(cached_hc_measurement or "n/a"),
                str(base_data.get("hc_mode", "n/a") or "n/a"),
            )
    if bool(AUTO_MODE_CACHE_ENABLED):
        try:
            sig_target = _auto_signature(
                base_data=base_data,
                measurements=measurements,
                fs_v=int(fs_v),
                taps_v=int(taps_v),
                xos=xos,
                hpf=hpf,
                hc_mode=None,
                include_hc_mode=False,
            )
            cached_entry = setup.runtime.auto_cache_get_entry(
                sig_target,
                filter_key=setup.filter_key,
                compat_version=setup.compat_version,
            )
            cached_entry = dict(cached_entry or {}) if isinstance(cached_entry, dict) else {}
            cached_hc = _auto_builtin_target_name(
                str(
                    cached_entry.get(
                        "best_target_curve",
                        cached_entry.get("best_hc_mode", ""),
                    )
                    or ""
                ).strip()
            )
            cached_preset = _cached_target_seed_preset(cached_entry)
            if (
                _cache_target_valid(setup.runtime, cached_hc)
                and _cached_target_matches_current_hc(base_data, cached_hc)
                and (
                    bool(cached_preset)
                    or not _cached_target_has_exact_preset(
                        runtime=setup.runtime,
                        cache_state=state,
                    )
                )
            ):
                _set_cached_target_state(
                    state,
                    runtime=setup.runtime,
                    cached_hc=cached_hc,
                    cached_preset=dict(cached_preset or {}),
                    cached_metrics=dict(cached_entry.get("best_metrics", {}) or {}),
                    source="cache_signature",
                    status_label="signature",
                    status_cb=status_cb,
                )
            elif _cache_target_valid(setup.runtime, cached_hc):
                logger.info(
                    "Automatic mode target select: signature cache target=%s skipped; current target=%s",
                    str(cached_hc or "n/a"),
                    str(base_data.get("hc_mode", "n/a") or "n/a"),
                )
        except Exception:
            logger.exception("cache signature target state set")
    if not _cached_target_has_exact_preset(runtime=setup.runtime, cache_state=state):
        optuna_state = _cached_target_state_from_optuna_study(
            setup=setup,
            base_data=base_data,
            measurements=measurements,
        )
        if isinstance(optuna_state, _TargetCacheState) and str(
            optuna_state.cached_target_source or ""
        ).strip():
            state = optuna_state
    return state


def _fallback_to_cached_target(
    *,
    setup: _TargetSelectionSetup,
    cache_state: _TargetCacheState,
    base_data: dict,
    measurements: dict,
) -> dict | None:
    if not _auto_target_mode_locks_hc(base_data):
        return None
    fallback = _cached_target_return(
        runtime=setup.runtime,
        cached_hc_mode=cache_state.cached_target_hc,
        cached_preset=cache_state.cached_target_preset,
        cached_metrics=cache_state.cached_target_metrics,
        selection_method=str(cache_state.cached_target_source or "cache"),
        base_data=base_data,
        measurements=measurements,
        goal=setup.goal,
        rank_basis=setup.rank_basis,
    )
    return dict(fallback) if isinstance(fallback, dict) else None


def _try_exact_cached_target_result(
    *,
    setup: _TargetSelectionSetup,
    cache_state: _TargetCacheState,
    base_data: dict,
    measurements: dict,
    status_cb,
) -> dict | None:
    cached_source = str(cache_state.cached_target_source or "").strip()
    if not (
        cached_source in (
            "cache_measurement",
            "cache_optuna_target",
            "cache_signature",
        )
        and _cached_target_has_exact_preset(
            runtime=setup.runtime,
            cache_state=cache_state,
        )
    ):
        return None
    missing_hidden_keys = _missing_cached_target_hidden_keys(cache_state.cached_target_preset)
    if missing_hidden_keys:
        logger.info(
            "Automatic mode target select: exact cache hit skipped for target=%s (%s missing from cached target preset: %s)",
            str(cache_state.cached_target_hc or "n/a"),
            str(cached_source or "cache"),
            ", ".join(missing_hidden_keys),
        )
        return None
    fallback = _cached_target_return(
        runtime=setup.runtime,
        cached_hc_mode=cache_state.cached_target_hc,
        cached_preset=cache_state.cached_target_preset,
        cached_metrics=cache_state.cached_target_metrics,
        selection_method=f"{cached_source}_hit",
        base_data=base_data,
        measurements=measurements,
        goal=setup.goal,
        rank_basis=setup.rank_basis,
    )
    if not isinstance(fallback, dict):
        return None
    logger.info(
        "Automatic mode target select: cache hit for same measurements, using cached target=%s and skipping all target comparison trials",
        str(cache_state.cached_target_hc),
    )
    if callable(status_cb):
        status_cb(
            "DecayCore automatic mode: target loaded directly from cache "
            f"(same measurements -> {str(cache_state.cached_target_hc)}, "
            "skipping target comparison trials)"
        )
    return dict(fallback)


def _try_measurement_global_target_result(
    *,
    setup: _TargetSelectionSetup,
    cache_state: _TargetCacheState,
    base_data: dict,
    measurements: dict,
    status_cb,
) -> dict | None:
    source = str(cache_state.cached_target_source or "").strip()
    if source != "cache_measurement_global_filter_seed":
        return None
    if not dict(cache_state.cached_target_preset or {}):
        return None
    hc = _auto_builtin_target_name(cache_state.cached_target_hc)
    if not _cache_target_valid(setup.runtime, hc):
        return None
    if not _cached_target_matches_current_hc(base_data, hc):
        return None
    fallback = _cached_target_return(
        runtime=setup.runtime,
        cached_hc_mode=hc,
        cached_preset=cache_state.cached_target_preset,
        cached_metrics=cache_state.cached_target_metrics,
        selection_method=f"{source}_hit",
        base_data=base_data,
        measurements=measurements,
        goal=setup.goal,
        rank_basis=setup.rank_basis,
    )
    if not isinstance(fallback, dict):
        return None
    logger.info(
        "Automatic mode target select: measurement-global cache hit, using cached target=%s and skipping target comparison trials",
        str(hc),
    )
    if callable(status_cb):
        status_cb(
            "DecayCore automatic mode: target cache hit for same measurements; "
            f"using {str(hc)}"
        )
    return dict(fallback)
