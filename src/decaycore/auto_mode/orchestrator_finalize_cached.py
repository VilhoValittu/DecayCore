# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Finalize-stage cached-result handling for automatic mode.

Contains _finalize_cached_result, the function that handles the cache-hit
fast-path (with optional winner polish) in the finalize stage.
"""

from __future__ import annotations

import logging

from .auto_mode_profile import profiled_section
from .scoring_ranking import _auto_is_better_refine
from .winner_polish import (
    apply_excess_phase_strength_winner_polish,
    apply_hpf_winner_polish,
    apply_low_bass_cut_winner_polish,
    apply_mag_c_min_winner_polish,
    apply_phase_limit_winner_polish,
    apply_residual_peak_winner_polish,
    apply_tdc_strength_winner_polish,
)
from .stereo_policy_refine import apply_stereo_policy_refine

from .orchestrator_finalize_cache import (
    _LOW_BASS_CUT_WINNER_POLISH_STEP_HZ,
    _LOW_BASS_CUT_WINNER_POLISH_MAX_DELTA_HZ,
    _cache_refine_winner_phase_label,
    _validate_cached_result,
    _score_cached_result,
    _materialize_cached_result,
    _return_cached_result,
)

logger = logging.getLogger("DecayCore")

__all__ = ["_finalize_cached_result"]

_RECOVERABLE_CACHE_FINALIZE_EXCEPTIONS = (
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


def _apply_cache_improved_state(
    *,
    cached_state: dict,
    improved_count_total: int,
    winner_phase_label: str,
    improved: bool,
    phase_label: str,
) -> tuple[int, str]:
    if not bool(improved):
        return int(improved_count_total), str(winner_phase_label)
    cached_state["improved_any"] = True
    return int(improved_count_total) + 1, str(phase_label)


def _finalize_cached_result(
    *,
    runtime,
    search_base_data: dict,
    cache_base_data: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    cfg,
    goal: str,
    rank_basis: str,
    filter_key: str,
    compat_version: str,
    optimizer_backend: str,
    status_cb,
    optuna_mod,
    optuna_search_sig: str,
    seed: int,
    _cache_ready_preset,
    _materialize_preset_result,
    _maybe_apply_residual_tiebreak,
    cache_refine_result: dict,
    canonical_signature: str | None = None,
) -> dict | None:
    cached_state = _validate_cached_result(cache_refine_result)
    seed_source = str(cached_state.get("seed_source", "exact_cache") or "exact_cache")
    best_preset = dict(cached_state.get("best_preset", {}) or {})
    best_metrics = dict(cached_state.get("best_metrics", {}) or {})
    improved_count_total = int(cached_state.get("improved_count_total", 0) or 0)
    winner_phase_label = _cache_refine_winner_phase_label(seed_source)
    if not bool(getattr(runtime, "cache_winner_polish_enabled", False)):
        skipped_polish_meta = {
            "enabled": False,
            "applicable": False,
            "applied": False,
            "reason": "cache_fast_finalize_skips_winner_polish",
            "phase_label": "cache fast finalize",
        }
        logger.info(
            "Automatic mode cache finalize: skipping repeated winner polish and materializing cached/refined winner."
        )
        try:
            materialized = _materialize_cached_result(
                runtime=runtime,
                cache_base_data=cache_base_data,
                measurements=measurements,
                fs_v=int(fs_v),
                taps_v=int(taps_v),
                xos=xos,
                hpf=hpf,
                cfg=cfg,
                goal=goal,
                filter_key=filter_key,
                compat_version=compat_version,
                optimizer_backend=optimizer_backend,
                optuna_mod=optuna_mod,
                optuna_search_sig=optuna_search_sig,
                seed=int(seed),
                best_preset=dict(best_preset or {}),
                best_metrics=dict(best_metrics or {}),
                stereo_cache_meta={},
                _cache_ready_preset=_cache_ready_preset,
                _materialize_preset_result=_materialize_preset_result,
                canonical_signature=canonical_signature,
            )
            score = _score_cached_result(
                best_metrics=dict(materialized.get("best_metrics", {}) or {}),
                best_applied_preset=dict(materialized.get("best_applied_preset", {}) or {}),
            )
            return _return_cached_result(
                materialized=materialized,
                cached_state=cached_state,
                score=score,
                rank_basis=rank_basis,
                optimizer_backend=optimizer_backend,
                goal=goal,
                winner_phase_label=winner_phase_label,
                phase_limit_cache_meta=dict(skipped_polish_meta),
                mag_c_min_cache_meta=dict(skipped_polish_meta),
                low_bass_cut_cache_meta=dict(skipped_polish_meta),
                hpf_cache_meta=dict(skipped_polish_meta),
                eps_cache_meta=dict(skipped_polish_meta),
                residual_peak_cache_meta=dict(skipped_polish_meta),
                tdc_strength_cache_meta=dict(skipped_polish_meta),
                stereo_cache_meta={},
                fs_v=int(fs_v),
                taps_v=int(taps_v),
            )
        except _RECOVERABLE_CACHE_FINALIZE_EXCEPTIONS as exc:
            logger.warning(
                "Automatic mode: exact preset cache materialization failed, "
                f"falling back to search ({type(exc).__name__}: {exc})"
            )
            return None
    try:
        with profiled_section("finalize.cache_residual_tiebreak"):
            best_preset, best_metrics, residual_cache_improved = _maybe_apply_residual_tiebreak(
                best_preset=best_preset,
                best_metrics=best_metrics,
                candidate_items=None,
                base_data_ref=cache_base_data,
                phase_label="cache residual tie-break",
            )
        improved_count_total, winner_phase_label = _apply_cache_improved_state(
            cached_state=cached_state,
            improved_count_total=improved_count_total,
            winner_phase_label=winner_phase_label,
            improved=bool(residual_cache_improved),
            phase_label="cache residual tie-break",
        )

        with profiled_section("finalize.cache_phase_limit_winner_polish"):
            best_preset, best_metrics, phase_limit_cache_improved, phase_limit_cache_meta = (
                apply_phase_limit_winner_polish(
                    best_preset=best_preset,
                    best_metrics=best_metrics,
                    base_data_ref=cache_base_data,
                    phase_label="cache phase_limit winner polish",
                    goal=goal,
                    filter_key=filter_key,
                    enabled=bool(runtime.phase_limit_winner_polish_enabled),
                    offsets_hz=tuple(runtime.phase_limit_winner_polish_offsets_hz),
                    status_cb=status_cb,
                    materialize_preset_result=_materialize_preset_result,
                    cache_ready_preset=_cache_ready_preset,
                    auto_is_better_refine=_auto_is_better_refine,
                )
            )
        improved_count_total, winner_phase_label = _apply_cache_improved_state(
            cached_state=cached_state,
            improved_count_total=improved_count_total,
            winner_phase_label=winner_phase_label,
            improved=bool(phase_limit_cache_improved),
            phase_label="cache phase_limit winner polish",
        )

        with profiled_section("finalize.cache_mag_c_min_winner_polish"):
            best_preset, best_metrics, mag_c_min_cache_improved, mag_c_min_cache_meta = apply_mag_c_min_winner_polish(
                best_preset=best_preset,
                best_metrics=best_metrics,
                base_data_ref=cache_base_data,
                phase_label="cache mag_c_min winner polish",
                goal=goal,
                enabled=bool(runtime.mag_c_min_winner_polish_enabled),
                step_hz=float(runtime.mag_c_min_winner_polish_step_hz),
                max_down_hz=float(runtime.mag_c_min_winner_polish_max_down_hz),
                max_up_hz=float(runtime.mag_c_min_winner_polish_max_up_hz),
                status_cb=status_cb,
                materialize_preset_result=_materialize_preset_result,
                cache_ready_preset=_cache_ready_preset,
                auto_is_better_refine=_auto_is_better_refine,
            )
        improved_count_total, winner_phase_label = _apply_cache_improved_state(
            cached_state=cached_state,
            improved_count_total=improved_count_total,
            winner_phase_label=winner_phase_label,
            improved=bool(mag_c_min_cache_improved),
            phase_label="cache mag_c_min winner polish",
        )

        with profiled_section("finalize.cache_low_bass_cut_winner_polish"):
            best_preset, best_metrics, low_bass_cut_cache_improved, low_bass_cut_cache_meta = (
                apply_low_bass_cut_winner_polish(
                    best_preset=best_preset,
                    best_metrics=best_metrics,
                    base_data_ref=cache_base_data,
                    phase_label="cache low_bass_cut winner polish",
                    goal=goal,
                    enabled=True,
                    step_hz=float(_LOW_BASS_CUT_WINNER_POLISH_STEP_HZ),
                    max_delta_hz=float(_LOW_BASS_CUT_WINNER_POLISH_MAX_DELTA_HZ),
                    status_cb=status_cb,
                    materialize_preset_result=_materialize_preset_result,
                    cache_ready_preset=_cache_ready_preset,
                    auto_is_better_refine=_auto_is_better_refine,
                )
            )
        improved_count_total, winner_phase_label = _apply_cache_improved_state(
            cached_state=cached_state,
            improved_count_total=improved_count_total,
            winner_phase_label=winner_phase_label,
            improved=bool(low_bass_cut_cache_improved),
            phase_label="cache low_bass_cut winner polish",
        )

        with profiled_section("finalize.cache_hpf_winner_polish"):
            best_preset, best_metrics, hpf_cache_improved, hpf_cache_meta = apply_hpf_winner_polish(
                best_preset=best_preset,
                best_metrics=best_metrics,
                base_data_ref=cache_base_data,
                phase_label="cache hpf winner polish",
                goal=goal,
                enabled=bool(getattr(runtime, "hpf_winner_polish_enabled", True)),
                status_cb=status_cb,
                materialize_preset_result=_materialize_preset_result,
                cache_ready_preset=_cache_ready_preset,
                auto_is_better_refine=_auto_is_better_refine,
            )
        improved_count_total, winner_phase_label = _apply_cache_improved_state(
            cached_state=cached_state,
            improved_count_total=improved_count_total,
            winner_phase_label=winner_phase_label,
            improved=bool(hpf_cache_improved),
            phase_label="cache hpf winner polish",
        )

        with profiled_section("finalize.cache_excess_phase_strength_winner_polish"):
            best_preset, best_metrics, eps_cache_improved, eps_cache_meta = apply_excess_phase_strength_winner_polish(
                best_preset=best_preset,
                best_metrics=best_metrics,
                base_data_ref=cache_base_data,
                phase_label="cache excess_phase_strength winner polish",
                goal=goal,
                enabled=bool(runtime.excess_phase_strength_winner_polish_enabled),
                step=float(runtime.excess_phase_strength_winner_polish_step),
                max_delta=float(runtime.excess_phase_strength_winner_polish_max_delta),
                status_cb=status_cb,
                materialize_preset_result=_materialize_preset_result,
                cache_ready_preset=_cache_ready_preset,
                auto_is_better_refine=_auto_is_better_refine,
            )
        improved_count_total, winner_phase_label = _apply_cache_improved_state(
            cached_state=cached_state,
            improved_count_total=improved_count_total,
            winner_phase_label=winner_phase_label,
            improved=bool(eps_cache_improved),
            phase_label="cache excess_phase_strength winner polish",
        )

        with profiled_section("finalize.cache_residual_peak_winner_polish"):
            best_preset, best_metrics, residual_peak_cache_improved, residual_peak_cache_meta = (
                apply_residual_peak_winner_polish(
                    best_preset=best_preset,
                    best_metrics=best_metrics,
                    base_data_ref=cache_base_data,
                    phase_label="cache residual_peak winner polish",
                    goal=goal,
                    enabled=bool(
                        getattr(
                            cfg,
                            "residual_peak_winner_polish_enabled",
                            getattr(runtime, "residual_peak_winner_polish_enabled", True),
                        )
                    ),
                    max_variants=int(
                        getattr(
                            cfg,
                            "residual_peak_winner_polish_max_variants",
                            getattr(runtime, "residual_peak_winner_polish_max_variants", 8),
                        )
                    ),
                    min_improvement_db=float(
                        getattr(
                            cfg,
                            "residual_peak_winner_polish_min_improvement_db",
                            getattr(runtime, "residual_peak_winner_polish_min_improvement_db", 0.75),
                        )
                    ),
                    status_cb=status_cb,
                    materialize_preset_result=_materialize_preset_result,
                    cache_ready_preset=_cache_ready_preset,
                    auto_is_better_refine=_auto_is_better_refine,
                )
            )
        improved_count_total, winner_phase_label = _apply_cache_improved_state(
            cached_state=cached_state,
            improved_count_total=improved_count_total,
            winner_phase_label=winner_phase_label,
            improved=bool(residual_peak_cache_improved),
            phase_label="cache residual_peak winner polish",
        )

        with profiled_section("finalize.cache_tdc_strength_winner_polish"):
            best_preset, best_metrics, tdc_strength_cache_improved, tdc_strength_cache_meta = (
                apply_tdc_strength_winner_polish(
                    best_preset=best_preset,
                    best_metrics=best_metrics,
                    base_data_ref=cache_base_data,
                    phase_label="cache tdc_strength winner polish",
                    goal=goal,
                    enabled=True,
                    status_cb=status_cb,
                    materialize_preset_result=_materialize_preset_result,
                    cache_ready_preset=_cache_ready_preset,
                    auto_is_better_refine=_auto_is_better_refine,
                )
            )
        improved_count_total, winner_phase_label = _apply_cache_improved_state(
            cached_state=cached_state,
            improved_count_total=improved_count_total,
            winner_phase_label=winner_phase_label,
            improved=bool(tdc_strength_cache_improved),
            phase_label="cache tdc_strength winner polish",
        )

        with profiled_section("finalize.cache_stereo_policy_refine"):
            best_preset, best_metrics, stereo_cache_improved, stereo_cache_meta = apply_stereo_policy_refine(
                best_preset=best_preset,
                best_metrics=best_metrics,
                base_data_ref=cache_base_data,
                goal=goal,
                phase_label="cache stereo policy refine",
                status_cb=status_cb,
                materialize_preset_result=_materialize_preset_result,
                auto_is_better_refine=_auto_is_better_refine,
            )
        improved_count_total, winner_phase_label = _apply_cache_improved_state(
            cached_state=cached_state,
            improved_count_total=improved_count_total,
            winner_phase_label=winner_phase_label,
            improved=bool(stereo_cache_improved),
            phase_label="cache stereo policy refine",
        )
        cached_state["improved_count_total"] = int(improved_count_total)
        materialized = _materialize_cached_result(
            runtime=runtime,
            cache_base_data=cache_base_data,
            measurements=measurements,
            fs_v=int(fs_v),
            taps_v=int(taps_v),
            xos=xos,
            hpf=hpf,
            cfg=cfg,
            goal=goal,
            filter_key=filter_key,
            compat_version=compat_version,
            optimizer_backend=optimizer_backend,
            optuna_mod=optuna_mod,
            optuna_search_sig=optuna_search_sig,
            seed=int(seed),
            best_preset=dict(best_preset or {}),
            best_metrics=dict(best_metrics or {}),
            stereo_cache_meta=stereo_cache_meta,
            _cache_ready_preset=_cache_ready_preset,
            _materialize_preset_result=_materialize_preset_result,
            canonical_signature=canonical_signature,
        )
        score = _score_cached_result(
            best_metrics=dict(materialized.get("best_metrics", {}) or {}),
            best_applied_preset=dict(materialized.get("best_applied_preset", {}) or {}),
        )
        return _return_cached_result(
            materialized=materialized,
            cached_state=cached_state,
            score=score,
            rank_basis=rank_basis,
            optimizer_backend=optimizer_backend,
            goal=goal,
            winner_phase_label=winner_phase_label,
            phase_limit_cache_meta=phase_limit_cache_meta,
            mag_c_min_cache_meta=mag_c_min_cache_meta,
            low_bass_cut_cache_meta=low_bass_cut_cache_meta,
            hpf_cache_meta=hpf_cache_meta,
            eps_cache_meta=eps_cache_meta,
            residual_peak_cache_meta=residual_peak_cache_meta,
            tdc_strength_cache_meta=tdc_strength_cache_meta,
            stereo_cache_meta=stereo_cache_meta,
            fs_v=int(fs_v),
            taps_v=int(taps_v),
        )
    except _RECOVERABLE_CACHE_FINALIZE_EXCEPTIONS as exc:
        # Exact-cache finalize is a best-effort fast path; search fallback remains authoritative.
        logger.warning(
            "Automatic mode: exact preset cache materialization failed, "
            f"falling back to search ({type(exc).__name__}: {exc})"
        )
        return None
