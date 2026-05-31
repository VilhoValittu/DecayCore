# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Finalize-stage polish orchestration for automatic mode.

Contains the winner-polish orchestrating functions split out from
orchestrator_finalize.py to keep individual file sizes manageable:

- _save_cached_best
- _save_final_search_cache
- _set_search_winner_if_improved
- _apply_search_winner_polish_pipeline

The Pareto finalize and top-level entry points live in
orchestrator_finalize_run.py.
"""

from __future__ import annotations

import logging

import numpy as np

from .auto_mode_profile import profiled_section
from .cache_signature import (
    _auto_cache_put_best,
    _auto_cache_put_last_used_best,
    _auto_cache_put_target_for_measurements,
    _auto_cache_put_target_for_measurements_global,
    _auto_measurement_signature,
    _auto_signature,
)
from .scoring_ranking import (
    _auto_is_better_refine,
)
from .stereo_policy_refine import apply_stereo_policy_refine
from .shared import AUTO_MODE_CACHE_SCHEMA_VERSION, _auto_builtin_target_name, _auto_safe_float, _m
from .winner_polish import (
    apply_excess_phase_strength_winner_polish,
    apply_hpf_winner_polish,
    apply_low_bass_cut_winner_polish,
    apply_mag_c_min_winner_polish,
    apply_phase_limit_winner_polish,
    apply_residual_peak_winner_polish,
    apply_tdc_strength_winner_polish,
)

from .orchestrator_finalize_cache import (
    _LOW_BASS_CUT_WINNER_POLISH_STEP_HZ,
    _LOW_BASS_CUT_WINNER_POLISH_MAX_DELTA_HZ,
    _public_stereo_policy_refine_meta,
    _override_candidates,
    _apply_residual_peak_safety_override,
    _resolve_target_seed_preset,
    _preset_with_target_hc_mode,
)
from .orchestrator_finalize_cross_polish import _apply_cross_residual_phase_polish

logger = logging.getLogger("DecayCore")

__all__ = [
    "_save_cached_best",
    "_save_final_search_cache",
    "_set_search_winner_if_improved",
    "_apply_search_winner_polish_pipeline",
]


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


def _save_final_search_cache(
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
    cached_best_preset: dict,
    best_metrics: dict,
    materialized_best_preset: dict,
    canonical_signature: str | None = None,
) -> None:
    if not bool(cfg.cache_enabled):
        return
    try:
        _save_cached_best(
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
            best_preset=dict(cached_best_preset or {}),
            best_metrics=dict(best_metrics or {}),
            best_hc_mode=str(cache_base_data.get("hc_mode", "") or "").strip() or None,
            canonical_signature=canonical_signature,
        )
        logger.info("Automatic mode: saved best preset to cache.")
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
        # Cache persistence must not change the selected winner when the cache backend misbehaves.
        pass

    # If HPF winner polish changed the HPF freq, also save under the post-polish HPF
    # signature. On the next rerun the user's HPF setting will be the polished value,
    # so the cache lookup must match that new value to avoid a cold-start search.
    try:
        polished_hpf_freq = _auto_safe_float(
            materialized_best_preset.get("hpf_freq", float("nan")),
            float("nan"),
        )
        orig_hpf_freq = float(_auto_safe_float((hpf or {}).get("freq", float("nan")), float("nan")))
        if (
            np.isfinite(polished_hpf_freq)
            and np.isfinite(orig_hpf_freq)
            and abs(polished_hpf_freq - orig_hpf_freq) > 0.05
        ):
            polished_hpf_order = max(
                1,
                int(round(_auto_safe_float(materialized_best_preset.get("hpf_slope", 24), 24.0) / 6.0)),
            )
            polished_hpf = {
                "enabled": bool(materialized_best_preset.get("hpf_enable", True)),
                "freq": float(polished_hpf_freq),
                "order": polished_hpf_order,
            }
            _save_cached_best(
                cache_base_data=cache_base_data,
                measurements=measurements,
                fs_v=int(fs_v),
                taps_v=int(taps_v),
                xos=xos,
                hpf=polished_hpf,
                cfg=cfg,
                goal=goal,
                filter_key=filter_key,
                compat_version=compat_version,
                best_preset=dict(cached_best_preset or {}),
                best_metrics=dict(best_metrics or {}),
                best_hc_mode=str(cache_base_data.get("hc_mode", "") or "").strip() or None,
            )
            logger.info(
                "Automatic mode: saved best preset to cache under post-polish HPF "
                f"({orig_hpf_freq:.1f} -> {polished_hpf_freq:.1f} Hz)."
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
        pass


def _set_search_winner_if_improved(
    *,
    search_state,
    improved: bool,
    metrics: dict,
    preset: dict,
    phase_label: str,
    target_name: str | None,
) -> None:
    if not bool(improved):
        return
    from .search_state import _auto_set_search_winner

    prev_best = dict(search_state.best_metrics or {})
    _auto_set_search_winner(
        search_state,
        metrics,
        preset,
        prev_metrics=prev_best,
        phase_label=phase_label,
        target_name=target_name,
    )


def _apply_search_winner_polish_pipeline(
    *,
    runtime,
    search_state,
    search_base_data: dict,
    cfg,
    goal: str,
    filter_key: str,
    status_cb,
    winner_target_name: str | None,
    _cache_ready_preset,
    _materialize_preset_result,
    _maybe_apply_residual_tiebreak,
) -> tuple[dict, dict]:
    residual_peak_safety_override_meta = {
        "applied": False,
        "reason": "not_evaluated",
    }
    residual_candidate_items = list(search_state.phase2_pool or search_state.scored or [])
    with profiled_section("finalize.residual_tiebreak"):
        residual_best_preset, residual_best_metrics, residual_improved = _maybe_apply_residual_tiebreak(
            best_preset=search_state.best_preset,
            best_metrics=search_state.best_metrics,
            candidate_items=residual_candidate_items,
            base_data_ref=search_base_data,
            phase_label="residual tie-break",
        )
    _set_search_winner_if_improved(
        search_state=search_state,
        improved=bool(residual_improved),
        metrics=residual_best_metrics,
        preset=residual_best_preset,
        phase_label="residual tie-break",
        target_name=winner_target_name,
    )
    residual_peak_safety_override_meta = _apply_residual_peak_safety_override(
        search_state=search_state,
        cfg=cfg,
        goal=goal,
        winner_target_name=winner_target_name,
        phase_label="residual_peak safety override",
        candidate_items=residual_candidate_items,
    )

    with profiled_section("finalize.phase_limit_winner_polish"):
        polished_best_preset, polished_best_metrics, phase_limit_polish_improved, phase_limit_polish_meta = (
            apply_phase_limit_winner_polish(
                best_preset=search_state.best_preset,
                best_metrics=search_state.best_metrics,
                base_data_ref=search_base_data,
                phase_label="phase_limit winner polish",
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
    _set_search_winner_if_improved(
        search_state=search_state,
        improved=bool(phase_limit_polish_improved),
        metrics=polished_best_metrics,
        preset=polished_best_preset,
        phase_label="phase_limit winner polish",
        target_name=winner_target_name,
    )

    with profiled_section("finalize.mag_c_min_winner_polish"):
        mag_c_min_best_preset, mag_c_min_best_metrics, mag_c_min_polish_improved, mag_c_min_polish_meta = (
            apply_mag_c_min_winner_polish(
                best_preset=search_state.best_preset,
                best_metrics=search_state.best_metrics,
                base_data_ref=search_base_data,
                phase_label="mag_c_min winner polish",
                goal=goal,
                enabled=bool(runtime.mag_c_min_winner_polish_enabled),
                step_hz=float(runtime.mag_c_min_winner_polish_step_hz),
                max_down_hz=float(runtime.mag_c_min_winner_polish_max_down_hz),
                status_cb=status_cb,
                materialize_preset_result=_materialize_preset_result,
                cache_ready_preset=_cache_ready_preset,
                auto_is_better_refine=_auto_is_better_refine,
                candidate_items=list(search_state.phase2_pool or search_state.scored or []),
            )
        )
    _set_search_winner_if_improved(
        search_state=search_state,
        improved=bool(mag_c_min_polish_improved),
        metrics=mag_c_min_best_metrics,
        preset=mag_c_min_best_preset,
        phase_label="mag_c_min winner polish",
        target_name=winner_target_name,
    )

    with profiled_section("finalize.low_bass_cut_winner_polish"):
        low_bass_cut_best_preset, low_bass_cut_best_metrics, low_bass_cut_polish_improved, low_bass_cut_polish_meta = (
            apply_low_bass_cut_winner_polish(
                best_preset=search_state.best_preset,
                best_metrics=search_state.best_metrics,
                base_data_ref=search_base_data,
                phase_label="low_bass_cut winner polish",
                goal=goal,
                enabled=True,
                step_hz=float(_LOW_BASS_CUT_WINNER_POLISH_STEP_HZ),
                max_delta_hz=float(_LOW_BASS_CUT_WINNER_POLISH_MAX_DELTA_HZ),
                status_cb=status_cb,
                materialize_preset_result=_materialize_preset_result,
                cache_ready_preset=_cache_ready_preset,
                auto_is_better_refine=_auto_is_better_refine,
                candidate_items=list(search_state.phase2_pool or search_state.scored or []),
            )
        )
    _set_search_winner_if_improved(
        search_state=search_state,
        improved=bool(low_bass_cut_polish_improved),
        metrics=low_bass_cut_best_metrics,
        preset=low_bass_cut_best_preset,
        phase_label="low_bass_cut winner polish",
        target_name=winner_target_name,
    )

    with profiled_section("finalize.hpf_winner_polish"):
        hpf_best_preset, hpf_best_metrics, hpf_polish_improved, hpf_polish_meta = apply_hpf_winner_polish(
            best_preset=search_state.best_preset,
            best_metrics=search_state.best_metrics,
            base_data_ref=search_base_data,
            phase_label="hpf winner polish",
            goal=goal,
            enabled=bool(getattr(runtime, "hpf_winner_polish_enabled", True)),
            status_cb=status_cb,
            materialize_preset_result=_materialize_preset_result,
            cache_ready_preset=_cache_ready_preset,
            auto_is_better_refine=_auto_is_better_refine,
        )
    _set_search_winner_if_improved(
        search_state=search_state,
        improved=bool(hpf_polish_improved),
        metrics=hpf_best_metrics,
        preset=hpf_best_preset,
        phase_label="hpf winner polish",
        target_name=winner_target_name,
    )

    with profiled_section("finalize.excess_phase_strength_winner_polish"):
        eps_best_preset, eps_best_metrics, eps_polish_improved, eps_polish_meta = (
            apply_excess_phase_strength_winner_polish(
                best_preset=search_state.best_preset,
                best_metrics=search_state.best_metrics,
                base_data_ref=search_base_data,
                phase_label="excess_phase_strength winner polish",
                goal=goal,
                enabled=bool(runtime.excess_phase_strength_winner_polish_enabled),
                step=float(runtime.excess_phase_strength_winner_polish_step),
                max_delta=float(runtime.excess_phase_strength_winner_polish_max_delta),
                status_cb=status_cb,
                materialize_preset_result=_materialize_preset_result,
                cache_ready_preset=_cache_ready_preset,
                auto_is_better_refine=_auto_is_better_refine,
            )
        )
    _set_search_winner_if_improved(
        search_state=search_state,
        improved=bool(eps_polish_improved),
        metrics=eps_best_metrics,
        preset=eps_best_preset,
        phase_label="excess_phase_strength winner polish",
        target_name=winner_target_name,
    )

    with profiled_section("finalize.residual_peak_winner_polish"):
        (
            residual_peak_best_preset,
            residual_peak_best_metrics,
            residual_peak_polish_improved,
            residual_peak_polish_meta,
        ) = apply_residual_peak_winner_polish(
            best_preset=search_state.best_preset,
            best_metrics=search_state.best_metrics,
            base_data_ref=search_base_data,
            phase_label="residual_peak winner polish",
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
    _set_search_winner_if_improved(
        search_state=search_state,
        improved=bool(residual_peak_polish_improved),
        metrics=residual_peak_best_metrics,
        preset=residual_peak_best_preset,
        phase_label="residual_peak winner polish",
        target_name=winner_target_name,
    )

    cross_polish_meta = _apply_cross_residual_phase_polish(
        runtime=runtime,
        search_state=search_state,
        search_base_data=search_base_data,
        cfg=cfg,
        goal=goal,
        filter_key=filter_key,
        status_cb=status_cb,
        winner_target_name=winner_target_name,
        _cache_ready_preset=_cache_ready_preset,
        _materialize_preset_result=_materialize_preset_result,
        _set_search_winner_if_improved=_set_search_winner_if_improved,
        _auto_is_better_refine=_auto_is_better_refine,
        first_residual_meta=residual_peak_polish_meta,
    )

    with profiled_section("finalize.tdc_strength_winner_polish"):
        tdc_strength_best_preset, tdc_strength_best_metrics, tdc_strength_polish_improved, tdc_strength_polish_meta = (
            apply_tdc_strength_winner_polish(
                best_preset=search_state.best_preset,
                best_metrics=search_state.best_metrics,
                base_data_ref=search_base_data,
                phase_label="tdc_strength winner polish",
                goal=goal,
                enabled=True,
                status_cb=status_cb,
                materialize_preset_result=_materialize_preset_result,
                cache_ready_preset=_cache_ready_preset,
                auto_is_better_refine=_auto_is_better_refine,
            )
        )
    _set_search_winner_if_improved(
        search_state=search_state,
        improved=bool(tdc_strength_polish_improved),
        metrics=tdc_strength_best_metrics,
        preset=tdc_strength_best_preset,
        phase_label="tdc_strength winner polish",
        target_name=winner_target_name,
    )

    with profiled_section("finalize.stereo_policy_refine"):
        stereo_best_preset, stereo_best_metrics, stereo_refine_improved, stereo_refine_meta = (
            apply_stereo_policy_refine(
                best_preset=search_state.best_preset,
                best_metrics=search_state.best_metrics,
                base_data_ref=search_base_data,
                goal=goal,
                phase_label="stereo policy refine",
                status_cb=status_cb,
                materialize_preset_result=_materialize_preset_result,
                auto_is_better_refine=_auto_is_better_refine,
            )
        )
    _set_search_winner_if_improved(
        search_state=search_state,
        improved=bool(stereo_refine_improved),
        metrics=stereo_best_metrics,
        preset=stereo_best_preset,
        phase_label="stereo policy refine",
        target_name=winner_target_name,
    )

    residual_peak_safety_override_meta_after_polish = _apply_residual_peak_safety_override(
        search_state=search_state,
        cfg=cfg,
        goal=goal,
        winner_target_name=winner_target_name,
        phase_label="post-polish residual_peak safety override",
        candidate_items=_override_candidates(search_state),
    )
    if bool(residual_peak_safety_override_meta_after_polish.get("applied", False)):
        residual_peak_safety_override_meta = dict(residual_peak_safety_override_meta_after_polish)

    polish_meta = {
        "phase_limit_winner_polish": dict(phase_limit_polish_meta or {}),
        "mag_c_min_winner_polish": dict(mag_c_min_polish_meta or {}),
        "low_bass_cut_winner_polish": dict(low_bass_cut_polish_meta or {}),
        "hpf_winner_polish": dict(hpf_polish_meta or {}),
        "excess_phase_strength_winner_polish": dict(eps_polish_meta or {}),
        "residual_peak_winner_polish": dict(residual_peak_polish_meta or {}),
        "cross_residual_phase_polish": dict(cross_polish_meta or {}),
        "tdc_strength_winner_polish": dict(tdc_strength_polish_meta or {}),
        "stereo_policy_refine": _public_stereo_policy_refine_meta(stereo_refine_meta),
        "_stereo_refine_meta": dict(stereo_refine_meta or {}),
    }
    return polish_meta, dict(residual_peak_safety_override_meta or {})
