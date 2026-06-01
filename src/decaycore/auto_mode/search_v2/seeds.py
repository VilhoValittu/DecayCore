# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Seed and fallback-reason helpers for auto search v2."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .. import api as auto_api
from ..filter_priors import get_auto_mode_filter_seed_preset

if TYPE_CHECKING:
    from .context import AutoSearchExecutionContext

logger = logging.getLogger("DecayCore")


def record_auto_search_fallback(
    search_data: dict,
    reason: str,
    logger_msg: str | None = None,
    *,
    exc_info: bool = False,
) -> None:
    reason = str(reason or "").strip()
    if not reason:
        return
    reasons = list(search_data.get("_auto_search_fallback_reasons", []) or [])
    if reason not in reasons:
        reasons.append(reason)
    search_data["_auto_search_fallback_reasons"] = list(reasons)
    logger.info(
        str(logger_msg or "Automatic mode search fallback: %s"),
        reason,
        exc_info=bool(exc_info),
    )


def _restore_auto_excursion_seed(search_data: dict, seed_data: dict | None) -> None:
    seed_hz = auto_api._auto_safe_float(
        dict(seed_data or {}).get(
            "_auto_exc_seed_freq_hz",
            dict(seed_data or {}).get(
                "_auto_exc_freq_hz",
                dict(seed_data or {}).get("exc_freq", float("nan")),
            ),
        ),
        float("nan"),
    )
    if not np.isfinite(seed_hz):
        return
    seed_hz = float(
        np.clip(
            float(seed_hz),
            float(auto_api.AUTO_MODE_EXC_MIN_HZ),
            float(auto_api.AUTO_MODE_EXC_MAX_HZ),
        )
    )
    seed_hz = float(round(seed_hz, 1))
    search_data["_auto_exc_seed_freq_hz"] = float(seed_hz)
    search_data["_auto_exc_freq_hz"] = float(seed_hz)
    search_data["exc_freq"] = float(seed_hz)


def _apply_explicit_seed(
    *,
    search_base_data: dict,
    cache_base_data: dict,
    measurements: dict | None = None,
) -> dict:
    try:
        seed_preset = dict(search_base_data.get("_auto_target_seed_preset", {}) or {})
    except (TypeError, ValueError):
        seed_preset = {}
    try:
        prior_seed_preset = dict(
            get_auto_mode_filter_seed_preset(
                search_base_data.get("filter_type", cache_base_data.get("filter_type", "")),
                measurements=measurements,
            )
            or {}
        )
    except (TypeError, ValueError):
        logger.debug("Failed to read automatic-mode prior seed preset", exc_info=True)
        prior_seed_preset = {}
    if seed_preset:
        search_base_data.update(seed_preset)
        _restore_auto_excursion_seed(search_base_data, cache_base_data)
        prior_seed_preset = dict(seed_preset)
        logger.info(
            "Automatic mode: target/cache preset seed loaded for canonical "
            "Phase 1 -> Phase 2 -> Phase 3 -> Phase 4 search."
        )
    return prior_seed_preset


def _apply_seed_payload(
    *,
    search_base_data: dict,
    cache_base_data: dict,
    seed_preset: dict,
    seed_metrics: dict | None,
    context: AutoSearchExecutionContext,
    success_log: str,
) -> bool:
    if not (isinstance(seed_preset, dict) and seed_preset):
        return False
    search_base_data["_auto_target_seed_preset"] = dict(seed_preset)
    search_base_data.update(dict(seed_preset))
    _restore_auto_excursion_seed(search_base_data, cache_base_data)
    context.prior_seed_preset = dict(seed_preset)
    if "hc_mode" in cache_base_data:
        search_base_data["hc_mode"] = cache_base_data["hc_mode"]
    if isinstance(seed_metrics, dict) and seed_metrics:
        search_base_data["_auto_target_seed_metrics"] = dict(seed_metrics)
    logger.info(str(success_log))
    return True


def _try_apply_cache_signature_seed(context: AutoSearchExecutionContext) -> None:
    search_base_data = context.search_base_data
    cache_base_data = context.cache_base_data
    if not (bool(context.cfg.cache_enabled) and not dict(search_base_data.get("_auto_target_seed_preset", {}) or {})):
        return
    try:
        cached_entry_sig = auto_api._auto_cache_get_entry(
            context.optuna_search_sig,
            filter_key=context.filter_key,
            compat_version=context.compat_version,
        )
        cached = dict((cached_entry_sig or {}).get("best_preset", {}) or {}) if isinstance(cached_entry_sig, dict) else {}
        cached_metrics_seed = dict((cached_entry_sig or {}).get("best_metrics", {}) or {}) if isinstance(cached_entry_sig, dict) else {}
        _apply_seed_payload(
            search_base_data=search_base_data,
            cache_base_data=cache_base_data,
            seed_preset=dict(cached or {}),
            seed_metrics=dict(cached_metrics_seed or {}),
            context=context,
            success_log="Automatic mode: loaded cached best preset seed.",
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
        record_auto_search_fallback(
            search_base_data,
            f"cache seed skipped: {type(exc).__name__}",
            exc_info=True,
        )


def _try_apply_optuna_phase1_seed(context: AutoSearchExecutionContext) -> None:
    search_base_data = context.search_base_data
    cache_base_data = context.cache_base_data
    if dict(search_base_data.get("_auto_target_seed_preset", {}) or {}):
        return
    if not (
        str(context.optimizer_backend) == "optuna"
        and bool(getattr(context.cfg, "optuna_persistent_study", False))
        and context.optuna_mod is not None
        and auto_api._auto_optuna_module_ready(context.optuna_mod)
    ):
        return
    try:
        storage = context.runtime["auto_optuna_create_storage"](
            context.optuna_mod,
            base_data=dict(cache_base_data or {}),
        )
        study_name = context.runtime["auto_optuna_study_name"](
            study_sig=str(context.optuna_search_sig),
            scope=context.runtime["auto_optuna_effective_scope"](
                cache_base_data,
                "phase1",
                phase_kind="phase1",
            ),
        )
        study = context.optuna_mod.load_study(study_name=str(study_name), storage=storage)
        best_trial = getattr(study, "best_trial", None)
        study_seed_preset = context.runtime["auto_optuna_trial_payload_preset"](
            dict(getattr(best_trial, "user_attrs", {}) or {})
        )
        if not isinstance(study_seed_preset, dict) or not study_seed_preset:
            study_seed_preset = dict(getattr(best_trial, "params", {}) or {})
        study_seed_metrics = dict(
            (context.runtime["auto_optuna_trial_out_payload"](best_trial) or {}).get("metrics", {}) or {}
        ) if isinstance(study_seed_preset, dict) and study_seed_preset else {}
        _apply_seed_payload(
            search_base_data=search_base_data,
            cache_base_data=cache_base_data,
            seed_preset=dict(study_seed_preset or {}),
            seed_metrics=dict(study_seed_metrics or {}),
            context=context,
            success_log=(
                "Automatic mode: loaded Optuna phase1 study preset seed for "
                "canonical Phase 1 -> Phase 2 -> Micro refine search."
            ),
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
        record_auto_search_fallback(
            search_base_data,
            f"optuna study seed skipped: {type(exc).__name__}",
            exc_info=True,
        )


def _try_apply_last_used_seed(context: AutoSearchExecutionContext) -> None:
    search_base_data = context.search_base_data
    cache_base_data = context.cache_base_data
    if not (bool(context.cfg.cache_enabled) and not dict(search_base_data.get("_auto_target_seed_preset", {}) or {})):
        return
    try:
        cached_entry_last = auto_api._auto_cache_get_last_used_best(
            goal=context.goal,
            filter_key=context.filter_key,
            compat_version=context.compat_version,
        )
        cached = dict((cached_entry_last or {}).get("best_preset", {}) or {})
        cached_metrics_seed = dict((cached_entry_last or {}).get("best_metrics", {}) or {})
        _apply_seed_payload(
            search_base_data=search_base_data,
            cache_base_data=cache_base_data,
            seed_preset=dict(cached or {}),
            seed_metrics=dict(cached_metrics_seed or {}),
            context=context,
            success_log="Automatic mode: loaded filter-specific last-used preset seed.",
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
        record_auto_search_fallback(
            search_base_data,
            f"last-best seed skipped: {type(exc).__name__}",
            exc_info=True,
        )


def _apply_legacy_opportunistic_seeds(context: AutoSearchExecutionContext) -> None:
    search_base_data = context.search_base_data
    cache_base_data = context.cache_base_data
    _try_apply_cache_signature_seed(context)
    _try_apply_optuna_phase1_seed(context)
    _try_apply_last_used_seed(context)
