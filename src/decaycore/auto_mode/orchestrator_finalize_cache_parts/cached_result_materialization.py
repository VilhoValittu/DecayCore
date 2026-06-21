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

from ..auto_mode_profile import profiled_section
from ..candidate_generation import _seed_auto_mode_candidate_optuna_params
from ..rank_score import attach_official_rank_score
from ..scoring_ranking import maybe_override_hard_failed_winner

from .cache_finalize_status import (
    _cache_refine_winner_summary,
    _public_stereo_policy_refine_meta,
    _stereo_refine_materialize_base_data,
)
from .cached_result_scoring import (
    _attach_cached_debug,
    _save_cached_best,
)

_LOW_BASS_CUT_WINNER_POLISH_STEP_HZ = 2.0
_LOW_BASS_CUT_WINNER_POLISH_MAX_DELTA_HZ = 8.0

def _materialize_cached_result(
    *,
    runtime,
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
    optimizer_backend: str,
    optuna_mod,
    optuna_search_sig: str,
    seed: int,
    best_preset: dict,
    best_metrics: dict,
    stereo_cache_meta: dict | None,
    _cache_ready_preset,
    _materialize_preset_result,
    canonical_signature: str | None = None,
) -> dict:
    cache_materialize_base_data = _stereo_refine_materialize_base_data(
        cache_base_data,
        stereo_cache_meta,
    )

    with profiled_section("finalize.cache_materialize_best"):
        best_result, best_metrics_recalc, best_data = _materialize_preset_result(
            best_preset,
            include_response_arrays=True,
            summarize=True,
            base_data_override=cache_materialize_base_data,
        )
    best_metrics = attach_official_rank_score(best_metrics_recalc or best_metrics)
    best_applied_preset = dict(best_data or best_preset or {})
    _cache_override_winner, residual_peak_safety_override_meta = maybe_override_hard_failed_winner(
        {"metrics": dict(best_metrics or {}), "preset": dict(best_preset or {})},
        [{"metrics": dict(best_metrics or {}), "preset": dict(best_preset or {})}],
        cfg,
        goal=goal,
    )
    best_cache_preset = _cache_ready_preset(best_preset, best_metrics=best_metrics)
    if bool(str(optimizer_backend) == "optuna" and optuna_mod is not None):
        raw_scope = "phase1"
        scope_eff = runtime.auto_optuna_effective_scope(
            cache_base_data,
            raw_scope,
            phase_kind="phase1",
        )
        runtime.auto_optuna_remember_result(
            optuna_mod,
            base_data=dict(cache_base_data or {}),
            study_name=runtime.auto_optuna_study_name(
                study_sig=optuna_search_sig,
                scope=scope_eff,
            ),
            study_scope=scope_eff,
            phase_kind="phase1",
            seed=int(seed + 500001),
            preset=dict(best_preset or {}),
            metrics=dict(best_metrics or {}),
            seed_to_params=(
                lambda preset, _base=dict(cache_base_data): _seed_auto_mode_candidate_optuna_params(
                    _base,
                    preset,
                )
            ),
            use_refine_tiebreak=False,
            out_payload={
                "idx": 1,
                "ok": True,
                "metrics": dict(best_metrics or {}),
                "trial_preset": dict(best_preset or {}),
                "phase": "exact_cache_replay",
            },
        )
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
        best_preset=dict(best_cache_preset or {}),
        best_metrics=dict(best_metrics or {}),
        best_hc_mode=str(cache_base_data.get("hc_mode", "") or "").strip() or None,
        canonical_signature=canonical_signature,
    )
    return {
        "best_result": best_result,
        "best_metrics": dict(best_metrics or {}),
        "best_preset": dict(best_cache_preset or {}),
        "best_applied_preset": dict(best_applied_preset or {}),
        "residual_peak_safety_override": dict(residual_peak_safety_override_meta or {}),
    }

def _return_cached_result(
    *,
    materialized: dict,
    cached_state: dict,
    score: dict,
    rank_basis: str,
    optimizer_backend: str,
    goal: str,
    winner_phase_label: str,
    phase_limit_cache_meta: dict | None,
    mag_c_min_cache_meta: dict | None,
    low_bass_cut_cache_meta: dict | None,
    hpf_cache_meta: dict | None,
    eps_cache_meta: dict | None,
    residual_peak_cache_meta: dict | None,
    tdc_strength_cache_meta: dict | None,
    stereo_cache_meta: dict | None,
    fs_v: int,
    taps_v: int,
) -> dict:
    best_metrics = dict(materialized.get("best_metrics", {}) or {})
    residual_peak_safety_override_meta = dict(materialized.get("residual_peak_safety_override", {}) or {})
    executed_micro_trials_total = int(cached_state.get("executed_micro_trials_total", 0) or 0)
    cache_refine_rollup_tel = dict(cached_state.get("cache_refine_rollup_tel", {}) or {})
    stop_reason = str(cached_state.get("stop_reason", "max_rounds") or "max_rounds")
    seed_source = str(cached_state.get("seed_source", "exact_cache") or "exact_cache")
    polish_meta = (
        dict(phase_limit_cache_meta or {}),
        dict(mag_c_min_cache_meta or {}),
        dict(low_bass_cut_cache_meta or {}),
        dict(hpf_cache_meta or {}),
        dict(eps_cache_meta or {}),
        dict(residual_peak_cache_meta or {}),
        dict(tdc_strength_cache_meta or {}),
    )
    winner_polish_ran = any(
        str(meta.get("reason", "") or "") != "cache_fast_finalize_skips_winner_polish"
        for meta in polish_meta
    )
    return {
        "best_result": materialized.get("best_result"),
        "best_metrics": dict(best_metrics or {}),
        "best_preset": dict(materialized.get("best_preset", {}) or {}),
        "best_applied_preset": dict(materialized.get("best_applied_preset", {}) or {}),
        "winner": {
            "rank_score_official": float(score.get("winner_rank", float("nan"))),
            "rank_score_components": dict(score.get("winner_components", {}) or {}),
            "rank_score_breakdown": dict(best_metrics.get("rank_score_breakdown", {}) or {}),
        },
        "auto_mode_debug": _attach_cached_debug(
            best_metrics=best_metrics,
            residual_peak_safety_override_meta=residual_peak_safety_override_meta,
        ),
        "residual_peak_safety_override": dict(residual_peak_safety_override_meta or {}),
        "winner_explanation": {
            "summary": _cache_refine_winner_summary(
                seed_source,
                improved_any=bool(cached_state.get("improved_any", False)),
            ),
            "reasons": [],
            "deltas": {},
            "phase_label": str(winner_phase_label),
            "target_name": str(cached_state.get("cache_target_name", "n/a") or "n/a"),
        },
        "best_auto_exc_freq_hz": float(score.get("cached_best_auto_exc_hz", float("nan"))),
        "phase_limit_winner_polish": dict(phase_limit_cache_meta or {}),
        "mag_c_min_winner_polish": dict(mag_c_min_cache_meta or {}),
        "low_bass_cut_winner_polish": dict(low_bass_cut_cache_meta or {}),
        "hpf_winner_polish": dict(hpf_cache_meta or {}),
        "excess_phase_strength_winner_polish": dict(eps_cache_meta or {}),
        "residual_peak_winner_polish": dict(residual_peak_cache_meta or {}),
        "tdc_strength_winner_polish": dict(tdc_strength_cache_meta or {}),
        "stereo_policy_refine": _public_stereo_policy_refine_meta(stereo_cache_meta),
        "optimizer_backend": str(optimizer_backend or "builtin"),
        "auto_goal": str(goal),
        "selection_basis": str(rank_basis),
        "top": [],
        "trials_total": int(executed_micro_trials_total),
        "trials_ok": int(executed_micro_trials_total),
        "trials_phase1_total": 0,
        "trials_phase1_ok": 0,
        "trials_phase2_total": int(executed_micro_trials_total),
        "trials_phase2_ok": int(executed_micro_trials_total),
        "trials_phase3_total": int(executed_micro_trials_total),
        "trials_phase3_ok": int(executed_micro_trials_total),
        "phase4_finalize": True,
        "phase4_steps": {
            "pareto_finalize": False,
            "winner_polish": bool(winner_polish_ran),
            "final_validation": False,
            "cache_save": True,
            "cache_materialize": True,
        },
        "optuna_phase1_telemetry": {},
        "optuna_phase2_local_telemetry": [],
        "optuna_phase3_micro_telemetry": dict(cache_refine_rollup_tel or {}),
        "optuna_phase2_rollup_telemetry": dict(cache_refine_rollup_tel or {}),
        "phase1_plateau_hit": False,
        "phase2_plateau_hit": bool(str(stop_reason) in ("no_improvement", "below_threshold")),
        "search_fs": int(fs_v),
        "search_taps": int(taps_v),
    }


__all__ = ['_materialize_cached_result', '_return_cached_result']

