# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Finalize-stage cache helpers and formatting utilities for automatic mode."""

from __future__ import annotations

import logging

import numpy as np

from ..auto_mode_profile import profiled_section
from ..cache_signature import (
    _auto_cache_stats_snapshot,
    _auto_cache_put_best,
    _auto_cache_put_last_used_best,
    _auto_cache_put_target_for_measurements,
    _auto_cache_put_target_for_measurements_global,
    _auto_measurement_signature,
    _auto_signature,
)
from ..candidate_generation import _seed_auto_mode_candidate_optuna_params
from ..rank_score import attach_official_rank_score, official_rank_score
from ..scoring_ranking import (
    _auto_is_better_refine,
    _auto_mode_ripple_for_pareto,
    _auto_prepost_for_pareto,
    maybe_override_hard_failed_winner,
)
from ..stereo_policy_refine import apply_stereo_policy_refine
from ..shared import AUTO_MODE_CACHE_SCHEMA_VERSION, _auto_builtin_target_name, _auto_safe_float, _m
from ..winner_polish import (
    apply_excess_phase_strength_winner_polish,
    apply_hpf_winner_polish,
    apply_low_bass_cut_winner_polish,
    apply_mag_c_min_winner_polish,
    apply_phase_limit_winner_polish,
    apply_residual_peak_winner_polish,
    apply_tdc_strength_winner_polish,
)

logger = logging.getLogger("DecayCore")

_LOW_BASS_CUT_WINNER_POLISH_STEP_HZ = 2.0
_LOW_BASS_CUT_WINNER_POLISH_MAX_DELTA_HZ = 8.0

def _apply_residual_peak_safety_override(
    *,
    search_state,
    cfg,
    goal: str,
    winner_target_name: str | None,
    phase_label: str,
    candidate_items: list[dict] | None = None,
) -> dict:
    from ..search_state import _auto_set_search_winner

    current = {
        "metrics": dict(getattr(search_state, "best_metrics", {}) or {}),
        "preset": dict(getattr(search_state, "best_preset", {}) or {}),
    }
    replacement, meta = maybe_override_hard_failed_winner(
        current,
        list(candidate_items or _override_candidates(search_state)),
        cfg,
        goal=goal,
    )
    if bool(meta.get("applied", False)) and isinstance(replacement, dict):
        prev_best = dict(getattr(search_state, "best_metrics", {}) or {})
        _auto_set_search_winner(
            search_state,
            dict(replacement.get("metrics", {}) or {}),
            dict(replacement.get("preset", {}) or {}),
            prev_metrics=prev_best,
            phase_label=phase_label,
            target_name=winner_target_name,
        )
    return dict(meta or {})

def _resolve_winner_auto_exc_hz(
    *,
    optimizer_backend: str,
    materialized_best_preset: dict | None,
    search_base_data: dict | None,
    best_metrics: dict | None,
) -> float:
    _ = optimizer_backend
    preset = dict(materialized_best_preset or {})
    base = dict(search_base_data or {})
    metrics = dict(best_metrics or {})
    for source, keys in (
        (base, ("_auto_exc_seed_freq_hz",)),
        (preset, ("_auto_exc_seed_freq_hz", "_auto_exc_freq_hz", "best_auto_exc_freq_hz", "exc_freq")),
        (base, ("_auto_exc_freq_hz", "exc_freq")),
    ):
        for key in keys:
            resolved = _auto_safe_float(source.get(key, float("nan")), float("nan"))
            if np.isfinite(resolved):
                return float(resolved)
    return float(
        _auto_safe_float(
            metrics.get("auto_exc_zero_penalty_hz", float("nan")),
            float("nan"),
        )
    )

def _resolve_target_seed_preset(cache_base_data: dict | None) -> dict:
    base = dict(cache_base_data or {})
    direct = base.get("_auto_target_seed_preset")
    if isinstance(direct, dict) and direct:
        return dict(direct)
    meta = base.get("_auto_target_curve_meta")
    if isinstance(meta, dict):
        fallback = meta.get("best_preset")
        if isinstance(fallback, dict) and fallback:
            return dict(fallback)
    return {}

def _preset_with_target_hc_mode(preset: dict | None, hc_mode: str | None) -> dict:
    out = dict(preset or {})
    hc = str(hc_mode or "").strip()
    if hc:
        out["hc_mode"] = str(hc)
    return dict(out)

def _save_cached_best(
    *,
    cache_base_data: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    cfg,
    goal: str,
    filter_key: str,
    compat_version: str,
    best_preset: dict,
    best_metrics: dict | None,
    best_hc_mode: str | None,
    canonical_signature: str | None = None,
) -> None:
    if not bool(cfg.cache_enabled):
        return
    best_hc_mode_raw = str(best_hc_mode or "").strip()
    best_hc_mode_builtin = _auto_builtin_target_name(best_hc_mode)
    best_preset_for_run = _preset_with_target_hc_mode(best_preset, best_hc_mode_raw)
    target_seed_preset_for_run = _preset_with_target_hc_mode(
        _resolve_target_seed_preset(cache_base_data),
        best_hc_mode_raw,
    )
    best_preset_for_target = _preset_with_target_hc_mode(
        best_preset,
        best_hc_mode_builtin or best_hc_mode_raw,
    )
    target_seed_preset_for_target = _preset_with_target_hc_mode(
        _resolve_target_seed_preset(cache_base_data),
        best_hc_mode_builtin or best_hc_mode_raw,
    )
    measurement_sig = _auto_measurement_signature(measurements)
    sig = _auto_signature(
        base_data=cache_base_data,
        measurements=measurements,
        fs_v=int(fs_v),
        taps_v=int(taps_v),
        xos=xos,
        hpf=hpf,
        hc_mode=best_hc_mode,
        include_hc_mode=True,
    )
    sig_target = _auto_signature(
        base_data=cache_base_data,
        measurements=measurements,
        fs_v=int(fs_v),
        taps_v=int(taps_v),
        xos=xos,
        hpf=hpf,
        hc_mode=None,
        include_hc_mode=False,
    )
    canonical_sig = str(canonical_signature or "").strip()
    primary_sig = canonical_sig or sig
    _auto_cache_put_best(
        primary_sig,
        best_preset=dict(best_preset_for_run or {}),
        target_seed_preset=dict(target_seed_preset_for_run or {}),
        best_metrics=dict(best_metrics or {}),
        best_hc_mode=best_hc_mode_raw or None,
        measurement_sig=measurement_sig,
        goal=goal,
        filter_key=filter_key,
        compat_version=compat_version,
        first_run_complete=True,
    )
    if canonical_sig and canonical_sig != sig:
        _auto_cache_put_best(
            sig,
            best_preset=dict(best_preset_for_run or {}),
            target_seed_preset=dict(target_seed_preset_for_run or {}),
            best_metrics=dict(best_metrics or {}),
            best_hc_mode=best_hc_mode_raw or None,
            measurement_sig=measurement_sig,
            goal=goal,
            filter_key=filter_key,
            compat_version=compat_version,
            first_run_complete=False,
        )
    _auto_cache_put_best(
        sig_target,
        best_preset=dict(best_preset_for_target or {}),
        target_seed_preset=dict(target_seed_preset_for_target or {}),
        best_metrics=dict(best_metrics or {}),
        best_hc_mode=best_hc_mode_builtin,
        measurement_sig=measurement_sig,
        goal=goal,
        filter_key=filter_key,
        compat_version=compat_version,
        first_run_complete=False,
    )
    _auto_cache_put_target_for_measurements(
        measurements=measurements,
        best_hc_mode=best_hc_mode_builtin,
        best_preset=dict(best_preset_for_target or {}),
        target_seed_preset=dict(target_seed_preset_for_target or {}),
        best_metrics=dict(best_metrics or {}),
        goal=goal,
        filter_key=filter_key,
        compat_version=compat_version,
    )
    _auto_cache_put_target_for_measurements_global(
        measurements=measurements,
        best_hc_mode=best_hc_mode_builtin,
        goal=goal,
        compat_version=compat_version,
        target_selection_meta={
            "source": "auto_finalize",
            "filter_key_that_updated_seed": str(filter_key),
        },
        filter_key=filter_key,
        filter_seed_preset=dict(best_preset_for_target or {}),
        filter_seed_metrics=dict(best_metrics or {}),
    )
    _auto_cache_put_last_used_best(
        best_preset=dict(best_preset_for_run or {}),
        best_metrics=dict(best_metrics or {}),
        best_hc_mode=best_hc_mode_raw or None,
        measurement_sig=measurement_sig,
        goal=goal,
        filter_key=filter_key,
        compat_version=compat_version,
    )

def _validate_cached_result(cache_refine_result: dict | None) -> dict:
    result = dict(cache_refine_result or {})
    return {
        "cache_target_name": str(result.get("cache_target_name", "n/a") or "n/a"),
        "seed_source": str(result.get("seed_source", "exact_cache") or "exact_cache"),
        "best_preset": dict(result.get("best_preset", {}) or {}),
        "best_metrics": dict(result.get("best_metrics", {}) or {}),
        "improved_any": bool(result.get("improved_any", False)),
        "improved_count_total": int(result.get("improved_count_total", 0) or 0),
        "executed_micro_trials_total": int(result.get("executed_micro_trials_total", 0) or 0),
        "cache_refine_rollup_tel": dict(result.get("cache_refine_rollup_tel", {}) or {}),
        "stop_reason": str(result.get("stop_reason", "max_rounds") or "max_rounds"),
    }

def _score_cached_result(
    *,
    best_metrics: dict,
    best_applied_preset: dict,
) -> dict:
    cached_best_auto_exc_hz = _auto_safe_float(
        best_applied_preset.get(
            "_auto_exc_freq_hz",
            best_applied_preset.get("best_auto_exc_freq_hz", float("nan")),
        ),
        float("nan"),
    )
    winner_rank = float(
        _auto_safe_float(
            best_metrics.get("rank_score_official", best_metrics.get("rank_score", float("nan"))),
            float("nan"),
        )
    )
    return {
        "winner_rank": float(winner_rank) if np.isfinite(winner_rank) else float("nan"),
        "winner_components": dict(best_metrics.get("rank_score_components", {}) or {}),
        "cached_best_auto_exc_hz": (
            float(cached_best_auto_exc_hz) if np.isfinite(cached_best_auto_exc_hz) else float("nan")
        ),
    }

def _attach_cached_debug(
    *,
    best_metrics: dict,
    residual_peak_safety_override_meta: dict | None,
) -> dict:
    return {
        "cache_schema_version": int(AUTO_MODE_CACHE_SCHEMA_VERSION),
        "cache_stats": _auto_cache_stats_snapshot(),
        "winning_score_breakdown": dict(best_metrics.get("rank_score_breakdown", {}) or {}),
        "residual_peak_safety_override": dict(residual_peak_safety_override_meta or {}),
    }


__all__ = ['_apply_residual_peak_safety_override', '_resolve_winner_auto_exc_hz', '_resolve_target_seed_preset', '_preset_with_target_hc_mode', '_save_cached_best', '_validate_cached_result', '_score_cached_result', '_attach_cached_debug']


def _load_sibling_symbols() -> None:
    import importlib
    package = __package__
    for module_name in ['orchestrator_finalize_cache_01', 'orchestrator_finalize_cache_02', 'orchestrator_finalize_cache_03']:
        if module_name == __name__.rsplit('.', 1)[-1]:
            continue
        module = importlib.import_module(f"{package}.{module_name}")
        for symbol in getattr(module, "__all__", ()):
            globals().setdefault(symbol, getattr(module, symbol))


_load_sibling_symbols()
