# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Runtime function map for auto-mode orchestrators."""

from __future__ import annotations

from .. import api as auto_api
from ..auto_mode_profile import AutoModeProfiler


def build_auto_mode_orchestrator_runtime() -> dict:
    return {
        "auto_cache_get_best": auto_api._auto_cache_get_best,
        "auto_cache_get_best_target": auto_api._auto_cache_get_best_target,
        "auto_cache_get_entry": auto_api._auto_cache_get_entry,
        "auto_cache_get_target_for_measurements": auto_api._auto_cache_get_target_for_measurements,
        "auto_cache_get_target_for_measurements_global": auto_api._auto_cache_get_target_for_measurements_global,
        "auto_cache_put_target_for_measurements_global": auto_api._auto_cache_put_target_for_measurements_global,
        "auto_score_result": auto_api._auto_score_result,
        "auto_select_builtin_target_curve": auto_api._auto_select_builtin_target_curve,
        "auto_trial_workers": auto_api._auto_trial_workers,
        "auto_optuna_base_data_without_constraints": auto_api._auto_optuna_base_data_without_constraints,
        "auto_import_optuna": auto_api._auto_import_optuna,
        "auto_optuna_create_storage": auto_api._auto_optuna_create_storage,
        "auto_optuna_effective_scope": auto_api._auto_optuna_effective_scope,
        "auto_optuna_fallback_summary_text": auto_api._auto_optuna_fallback_summary_text,
        "auto_optuna_fmt_value": auto_api._auto_optuna_fmt_value,
        "auto_optuna_module_ready": auto_api._auto_optuna_module_ready,
        "auto_optuna_needs_zero_feasible_rescue": auto_api._auto_optuna_needs_zero_feasible_rescue,
        "auto_optuna_objective_value": auto_api._auto_optuna_objective_value,
        "auto_optuna_remember_result": auto_api._auto_optuna_remember_result,
        "auto_optuna_scope_with_context": auto_api._auto_optuna_scope_with_context,
        "auto_optuna_scope_for_filter": auto_api._auto_optuna_scope_for_filter,
        "auto_optuna_study_name": auto_api._auto_optuna_study_name,
        "auto_optuna_telemetry_rollup": auto_api._auto_optuna_telemetry_rollup,
        "auto_optuna_telemetry_text": auto_api._auto_optuna_telemetry_text,
        "auto_optuna_trial_out_payload": auto_api._auto_optuna_trial_out_payload,
        "auto_optuna_trial_payload_preset": auto_api._auto_optuna_trial_payload_preset,
        "auto_run_optuna_eval_loop": auto_api._auto_run_optuna_eval_loop,
        "build_config": auto_api.build_config,
        "build_auto_mode_candidates_micro": auto_api._build_auto_mode_candidates_micro,
        "cache_refine_max_rounds": auto_api.AUTO_MODE_CACHE_REFINE_MAX_ROUNDS,
        "cache_refine_micro_trials": auto_api.AUTO_MODE_CACHE_REFINE_MICRO_TRIALS,
        "cache_refine_min_rank_improvement": auto_api.AUTO_MODE_CACHE_REFINE_MIN_RANK_IMPROVEMENT,
        "phase2_hard_gate_keep_peak_fraction": auto_api.AUTO_MODE_PHASE2_HARD_GATE_KEEP_PEAK_FRACTION,
        "phase2_hard_gate_abs_max_peak_db": auto_api.AUTO_MODE_PHASE2_HARD_GATE_ABS_MAX_PEAK_DB,
        "get_house_curve_by_name": auto_api.get_house_curve_by_name,
        "residual_peak_threshold_db": auto_api.AUTO_MODE_RESIDUAL_PEAK_THRESHOLD_DB,
        "residual_peak_hard_gate_db": auto_api.AUTO_MODE_RESIDUAL_PEAK_HARD_GATE_DB,
        "residual_peak_penalty_cap": auto_api.AUTO_MODE_RESIDUAL_PEAK_PENALTY_CAP,
        "auto_mode_tdc_decay_penalty_cap": auto_api.AUTO_MODE_TDC_DECAY_PENALTY_CAP,
        "auto_mode_tdc_decay_penalty_weight": auto_api.AUTO_MODE_TDC_DECAY_PENALTY_WEIGHT,
        "auto_mode_tdc_extreme_peak_reduction_db": auto_api.AUTO_MODE_TDC_EXTREME_PEAK_REDUCTION_DB,
        "auto_mode_tdc_low_need_threshold": auto_api.AUTO_MODE_TDC_LOW_NEED_THRESHOLD,
        "auto_mode_tdc_rt60_target_low_s": auto_api.AUTO_MODE_TDC_RT60_TARGET_LOW_S,
        "auto_mode_tdc_rt60_target_upper_s": auto_api.AUTO_MODE_TDC_RT60_TARGET_UPPER_S,
        "auto_mode_tdc_rt60_low_max_hz": auto_api.AUTO_MODE_TDC_RT60_LOW_MAX_HZ,
        "auto_mode_tdc_rt60_eval_max_hz": auto_api.AUTO_MODE_TDC_RT60_EVAL_MAX_HZ,
        "max_avg_score_loss_for_safety_override": auto_api.AUTO_MODE_MAX_AVG_SCORE_LOSS_FOR_SAFETY_OVERRIDE,
        "residual_peak_winner_polish_enabled": auto_api.AUTO_MODE_RESIDUAL_PEAK_WINNER_POLISH_ENABLED,
        "residual_peak_winner_polish_max_variants": auto_api.AUTO_MODE_RESIDUAL_PEAK_WINNER_POLISH_MAX_VARIANTS,
        "residual_peak_winner_polish_min_improvement_db": auto_api.AUTO_MODE_RESIDUAL_PEAK_WINNER_POLISH_MIN_IMPROVEMENT_DB,
        "mag_c_min_winner_polish_enabled": auto_api.AUTO_MODE_MAG_C_MIN_WINNER_POLISH_ENABLED,
        "mag_c_min_winner_polish_max_down_hz": auto_api.AUTO_MODE_MAG_C_MIN_WINNER_POLISH_MAX_DOWN_HZ,
        "mag_c_min_winner_polish_max_up_hz": auto_api.AUTO_MODE_MAG_C_MIN_WINNER_POLISH_MAX_UP_HZ,
        "mag_c_min_winner_polish_step_hz": auto_api.AUTO_MODE_MAG_C_MIN_WINNER_POLISH_STEP_HZ,
        "hpf_winner_polish_enabled": auto_api.AUTO_MODE_HPF_WINNER_POLISH_ENABLED,
        "excess_phase_strength_winner_polish_enabled": auto_api.AUTO_MODE_EXCESS_PHASE_STRENGTH_WINNER_POLISH_ENABLED,
        "excess_phase_strength_winner_polish_step": auto_api.AUTO_MODE_EXCESS_PHASE_STRENGTH_WINNER_POLISH_STEP,
        "excess_phase_strength_winner_polish_max_delta": auto_api.AUTO_MODE_EXCESS_PHASE_STRENGTH_WINNER_POLISH_MAX_DELTA,
        "phase_limit_winner_polish_enabled": auto_api.AUTO_MODE_PHASE_LIMIT_WINNER_POLISH_ENABLED,
        "phase_limit_winner_polish_offsets_hz": auto_api.AUTO_MODE_PHASE_LIMIT_WINNER_POLISH_OFFSETS_HZ,
        "cache_winner_polish_enabled": True,
        "run_pipeline": auto_api.run_pipeline,
        "suggest_auto_mode_candidate_optuna": auto_api._suggest_auto_mode_candidate_optuna,
        "summarize_run": auto_api.summarize_run,
    }


def _runtime(prof: AutoModeProfiler | None) -> dict:
    rt = build_auto_mode_orchestrator_runtime()
    if prof is not None:
        rt["run_pipeline"] = prof.wrap(rt["run_pipeline"], "run_pipeline")
        rt["summarize_run"] = prof.wrap(rt["summarize_run"], "summarize_run")
        rt["build_config"] = prof.wrap(rt["build_config"], "build_config")
        rt["auto_score_result"] = prof.wrap(rt["auto_score_result"], "auto_score_result")
    return rt
