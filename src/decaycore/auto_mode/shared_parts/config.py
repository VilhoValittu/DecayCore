# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

import logging
from dataclasses import dataclass

import numpy as np

from ...config.schema import normalize_flat_config
from ._constants import *

logger = logging.getLogger("DecayCore")

@dataclass(frozen=True)
class AutoModeConfig:
    trials: int = AUTO_MODE_TRIALS
    refine_trials: int = AUTO_MODE_REFINE_TRIALS
    phase1_plateau_rounds: int = AUTO_MODE_PHASE1_PLATEAU_ROUNDS
    local_refine_enabled: bool = AUTO_MODE_LOCAL_REFINE_ENABLED
    local_refine_top_k: int = AUTO_MODE_LOCAL_REFINE_TOP_K
    local_refine_trials_per_top: int = AUTO_MODE_LOCAL_REFINE_TRIALS_PER_TOP
    local_refine_shrink: float = AUTO_MODE_LOCAL_REFINE_SHRINK
    local_refine_keep_best_phase1: bool = AUTO_MODE_LOCAL_REFINE_KEEP_BEST_PHASE1
    phase3_micro_enabled: bool = AUTO_MODE_PHASE3_MICRO_ENABLED
    phase3_micro_trials: int = AUTO_MODE_PHASE3_MICRO_TRIALS
    adaptive_shrink_max: float = AUTO_MODE_ADAPTIVE_SHRINK_MAX
    phase2_pareto_pool_min: int = AUTO_MODE_PHASE2_PARETO_POOL_MIN
    phase2_pareto_pool_max: int = AUTO_MODE_PHASE2_PARETO_POOL_MAX
    phase2_pareto_rank_window: float = AUTO_MODE_PHASE2_PARETO_RANK_WINDOW
    phase2_pareto_acoustic_drop: float = AUTO_MODE_PHASE2_PARETO_ACOUSTIC_DROP
    phase2_hard_gate_enabled: bool = AUTO_MODE_PHASE2_HARD_GATE_ENABLED
    phase2_hard_gate_min_keep: int = AUTO_MODE_PHASE2_HARD_GATE_MIN_KEEP
    phase2_hard_gate_keep_event_fraction: float = (
        AUTO_MODE_PHASE2_HARD_GATE_KEEP_EVENT_FRACTION
    )
    phase2_hard_gate_keep_ripple_fraction: float = (
        AUTO_MODE_PHASE2_HARD_GATE_KEEP_RIPPLE_FRACTION
    )
    phase2_hard_gate_keep_peak_fraction: float = (
        AUTO_MODE_PHASE2_HARD_GATE_KEEP_PEAK_FRACTION
    )
    phase2_hard_gate_abs_max_peak_db: float = AUTO_MODE_PHASE2_HARD_GATE_ABS_MAX_PEAK_DB
    phase2_hard_gate_fallback_to_rank: bool = (
        AUTO_MODE_PHASE2_HARD_GATE_FALLBACK_TO_RANK
    )
    residual_peak_threshold_db: float = AUTO_MODE_RESIDUAL_PEAK_THRESHOLD_DB
    residual_peak_hard_gate_db: float = AUTO_MODE_RESIDUAL_PEAK_HARD_GATE_DB
    residual_peak_penalty_cap: float = AUTO_MODE_RESIDUAL_PEAK_PENALTY_CAP
    tdc_decay_penalty_cap: float = AUTO_MODE_TDC_DECAY_PENALTY_CAP
    tdc_decay_penalty_weight: float = AUTO_MODE_TDC_DECAY_PENALTY_WEIGHT
    tdc_extreme_peak_reduction_db: float = AUTO_MODE_TDC_EXTREME_PEAK_REDUCTION_DB
    tdc_low_need_threshold: float = AUTO_MODE_TDC_LOW_NEED_THRESHOLD
    tdc_rt60_target_low_s: float = AUTO_MODE_TDC_RT60_TARGET_LOW_S
    tdc_rt60_target_upper_s: float = AUTO_MODE_TDC_RT60_TARGET_UPPER_S
    tdc_rt60_low_max_hz: float = AUTO_MODE_TDC_RT60_LOW_MAX_HZ
    tdc_rt60_eval_max_hz: float = AUTO_MODE_TDC_RT60_EVAL_MAX_HZ
    max_avg_score_loss_for_safety_override: float = (
        AUTO_MODE_MAX_AVG_SCORE_LOSS_FOR_SAFETY_OVERRIDE
    )
    residual_peak_winner_polish_enabled: bool = (
        AUTO_MODE_RESIDUAL_PEAK_WINNER_POLISH_ENABLED
    )
    residual_peak_winner_polish_max_variants: int = (
        AUTO_MODE_RESIDUAL_PEAK_WINNER_POLISH_MAX_VARIANTS
    )
    residual_peak_winner_polish_min_improvement_db: float = (
        AUTO_MODE_RESIDUAL_PEAK_WINNER_POLISH_MIN_IMPROVEMENT_DB
    )
    refine_mode_soft_k: float = AUTO_MODE_REFINE_MODE_SOFT_K
    refine_tiebreak_rank_eps: float = AUTO_MODE_REFINE_TIEBREAK_RANK_EPS
    exc_min_hz: float = AUTO_MODE_EXC_MIN_HZ
    exc_max_hz: float = AUTO_MODE_EXC_MAX_HZ
    cache_enabled: bool = AUTO_MODE_CACHE_ENABLED
    optuna_pilot_enabled: bool = AUTO_MODE_OPTUNA_PILOT_ENABLED
    optuna_pilot_min_trials: int = AUTO_MODE_OPTUNA_PILOT_MIN_TRIALS
    optuna_pilot_startup_trials: int = AUTO_MODE_OPTUNA_PILOT_STARTUP_TRIALS
    optuna_startup_phase1: int = AUTO_MODE_OPTUNA_STARTUP_PHASE1
    optuna_startup_target: int = AUTO_MODE_OPTUNA_STARTUP_TARGET
    optuna_startup_local: int = AUTO_MODE_OPTUNA_STARTUP_LOCAL
    optuna_startup_micro: int = AUTO_MODE_OPTUNA_STARTUP_MICRO
    optuna_constraints_enabled: bool = AUTO_MODE_OPTUNA_CONSTRAINTS_ENABLED
    optuna_constraints_refine_only: bool = AUTO_MODE_OPTUNA_CONSTRAINTS_REFINE_ONLY
    optuna_constraints_max_mode_ripple_db: float = (
        AUTO_MODE_OPTUNA_CONSTRAINTS_MAX_MODE_RIPPLE_DB
    )
    optuna_constraints_max_events_severity: float = (
        AUTO_MODE_OPTUNA_CONSTRAINTS_MAX_EVENTS_SEVERITY
    )
    optuna_constraints_max_net_boost_db: float = (
        AUTO_MODE_OPTUNA_CONSTRAINTS_MAX_NET_BOOST_DB
    )
    optuna_constraints_use_events_in_refine: bool = (
        AUTO_MODE_OPTUNA_CONSTRAINTS_USE_EVENTS_IN_REFINE
    )
    optuna_telemetry: bool = AUTO_MODE_OPTUNA_TELEMETRY
    optuna_telemetry_log_summary: bool = AUTO_MODE_OPTUNA_TELEMETRY_LOG_SUMMARY
    optuna_constraints_zero_feasible_fallback: bool = (
        AUTO_MODE_OPTUNA_CONSTRAINTS_ZERO_FEASIBLE_FALLBACK
    )
    optuna_multivariate: bool = AUTO_MODE_OPTUNA_MULTIVARIATE
    optuna_group: bool = AUTO_MODE_OPTUNA_GROUP
    optuna_constant_liar: bool = AUTO_MODE_OPTUNA_CONSTANT_LIAR
    optuna_persistent_study: bool = AUTO_MODE_OPTUNA_PERSISTENT_STUDY
    optuna_avoid_duplicates: bool = AUTO_MODE_OPTUNA_AVOID_DUPLICATES
    optuna_cross_study_seeds: bool = AUTO_MODE_OPTUNA_CROSS_STUDY_SEEDS
    optuna_cross_study_seeds_top_n: int = AUTO_MODE_OPTUNA_CROSS_STUDY_SEEDS_TOP_N
    optuna_pruning_enabled: bool = AUTO_MODE_OPTUNA_PRUNING_ENABLED
    optuna_pruning_n_startup: int = AUTO_MODE_OPTUNA_PRUNING_N_STARTUP
    seed_revalidate_rank_gap: float = 3.0

    @classmethod
    def from_base_data(cls, base_data: dict | None) -> "AutoModeConfig":
        data = normalize_flat_config(dict(base_data or {}), include_runtime=True)
        legacy_startup = max(
            1,
            _auto_safe_int(
                data.get(
                    "auto_mode_optuna_startup_trials",
                    AUTO_MODE_OPTUNA_PILOT_STARTUP_TRIALS,
                ),
                AUTO_MODE_OPTUNA_PILOT_STARTUP_TRIALS,
            ),
        )
        return cls(
            trials=max(
                1,
                _auto_safe_int(
                    data.get("auto_mode_trials", AUTO_MODE_TRIALS), AUTO_MODE_TRIALS
                ),
            ),
            refine_trials=max(
                1,
                _auto_safe_int(
                    data.get("auto_mode_refine_trials", AUTO_MODE_REFINE_TRIALS),
                    AUTO_MODE_REFINE_TRIALS,
                ),
            ),
            phase1_plateau_rounds=max(
                1,
                _auto_safe_int(
                    data.get(
                        "auto_mode_phase1_plateau_rounds",
                        AUTO_MODE_PHASE1_PLATEAU_ROUNDS,
                    ),
                    AUTO_MODE_PHASE1_PLATEAU_ROUNDS,
                ),
            ),
            local_refine_enabled=_auto_safe_bool(
                data.get(
                    "auto_mode_local_refine_enabled", AUTO_MODE_LOCAL_REFINE_ENABLED
                ),
                AUTO_MODE_LOCAL_REFINE_ENABLED,
            ),
            local_refine_top_k=max(
                1,
                _auto_safe_int(
                    data.get(
                        "auto_mode_local_refine_top_k", AUTO_MODE_LOCAL_REFINE_TOP_K
                    ),
                    AUTO_MODE_LOCAL_REFINE_TOP_K,
                ),
            ),
            local_refine_trials_per_top=max(
                1,
                _auto_safe_int(
                    data.get(
                        "auto_mode_local_refine_trials_per_top",
                        AUTO_MODE_LOCAL_REFINE_TRIALS_PER_TOP,
                    ),
                    AUTO_MODE_LOCAL_REFINE_TRIALS_PER_TOP,
                ),
            ),
            local_refine_shrink=float(
                np.clip(
                    _auto_safe_float(
                        data.get(
                            "auto_mode_local_refine_shrink",
                            AUTO_MODE_LOCAL_REFINE_SHRINK,
                        ),
                        AUTO_MODE_LOCAL_REFINE_SHRINK,
                    ),
                    0.05,
                    1.50,
                )
            ),
            local_refine_keep_best_phase1=_auto_safe_bool(
                data.get(
                    "auto_mode_local_refine_keep_phase1",
                    AUTO_MODE_LOCAL_REFINE_KEEP_BEST_PHASE1,
                ),
                AUTO_MODE_LOCAL_REFINE_KEEP_BEST_PHASE1,
            ),
            phase3_micro_enabled=_auto_safe_bool(
                data.get(
                    "auto_mode_phase3_micro_enabled", AUTO_MODE_PHASE3_MICRO_ENABLED
                ),
                AUTO_MODE_PHASE3_MICRO_ENABLED,
            ),
            phase3_micro_trials=max(
                1,
                _auto_safe_int(
                    data.get(
                        "auto_mode_phase3_micro_trials", AUTO_MODE_PHASE3_MICRO_TRIALS
                    ),
                    AUTO_MODE_PHASE3_MICRO_TRIALS,
                ),
            ),
            adaptive_shrink_max=float(
                np.clip(
                    _auto_safe_float(
                        data.get(
                            "auto_mode_adaptive_shrink_max",
                            AUTO_MODE_ADAPTIVE_SHRINK_MAX,
                        ),
                        AUTO_MODE_ADAPTIVE_SHRINK_MAX,
                    ),
                    0.05,
                    1.0,
                )
            ),
            phase2_pareto_pool_min=max(
                1,
                _auto_safe_int(
                    data.get(
                        "auto_mode_phase2_pareto_pool_min",
                        AUTO_MODE_PHASE2_PARETO_POOL_MIN,
                    ),
                    AUTO_MODE_PHASE2_PARETO_POOL_MIN,
                ),
            ),
            phase2_pareto_pool_max=max(
                1,
                _auto_safe_int(
                    data.get(
                        "auto_mode_phase2_pareto_pool_max",
                        AUTO_MODE_PHASE2_PARETO_POOL_MAX,
                    ),
                    AUTO_MODE_PHASE2_PARETO_POOL_MAX,
                ),
            ),
            phase2_pareto_rank_window=max(
                0.0,
                _auto_safe_float(
                    data.get(
                        "auto_mode_phase2_pareto_rank_window",
                        AUTO_MODE_PHASE2_PARETO_RANK_WINDOW,
                    ),
                    AUTO_MODE_PHASE2_PARETO_RANK_WINDOW,
                ),
            ),
            phase2_pareto_acoustic_drop=max(
                0.0,
                _auto_safe_float(
                    data.get(
                        "auto_mode_phase2_pareto_acoustic_drop",
                        AUTO_MODE_PHASE2_PARETO_ACOUSTIC_DROP,
                    ),
                    AUTO_MODE_PHASE2_PARETO_ACOUSTIC_DROP,
                ),
            ),
            phase2_hard_gate_enabled=_auto_safe_bool(
                data.get(
                    "auto_mode_phase2_hard_gate_enabled",
                    AUTO_MODE_PHASE2_HARD_GATE_ENABLED,
                ),
                AUTO_MODE_PHASE2_HARD_GATE_ENABLED,
            ),
            phase2_hard_gate_min_keep=max(
                1,
                _auto_safe_int(
                    data.get(
                        "auto_mode_phase2_hard_gate_min_keep",
                        AUTO_MODE_PHASE2_HARD_GATE_MIN_KEEP,
                    ),
                    AUTO_MODE_PHASE2_HARD_GATE_MIN_KEEP,
                ),
            ),
            phase2_hard_gate_keep_event_fraction=float(
                np.clip(
                    _auto_safe_float(
                        data.get(
                            "auto_mode_phase2_hard_gate_keep_event_fraction",
                            AUTO_MODE_PHASE2_HARD_GATE_KEEP_EVENT_FRACTION,
                        ),
                        AUTO_MODE_PHASE2_HARD_GATE_KEEP_EVENT_FRACTION,
                    ),
                    0.05,
                    1.0,
                )
            ),
            phase2_hard_gate_keep_ripple_fraction=float(
                np.clip(
                    _auto_safe_float(
                        data.get(
                            "auto_mode_phase2_hard_gate_keep_ripple_fraction",
                            AUTO_MODE_PHASE2_HARD_GATE_KEEP_RIPPLE_FRACTION,
                        ),
                        AUTO_MODE_PHASE2_HARD_GATE_KEEP_RIPPLE_FRACTION,
                    ),
                    0.05,
                    1.0,
                )
            ),
            phase2_hard_gate_keep_peak_fraction=float(
                np.clip(
                    _auto_safe_float(
                        data.get(
                            "auto_mode_phase2_hard_gate_keep_peak_fraction",
                            AUTO_MODE_PHASE2_HARD_GATE_KEEP_PEAK_FRACTION,
                        ),
                        AUTO_MODE_PHASE2_HARD_GATE_KEEP_PEAK_FRACTION,
                    ),
                    0.05,
                    1.0,
                )
            ),
            phase2_hard_gate_abs_max_peak_db=max(
                0.0,
                _auto_safe_float(
                    data.get(
                        "auto_mode_phase2_hard_gate_abs_max_peak_db",
                        AUTO_MODE_PHASE2_HARD_GATE_ABS_MAX_PEAK_DB,
                    ),
                    AUTO_MODE_PHASE2_HARD_GATE_ABS_MAX_PEAK_DB,
                ),
            ),
            phase2_hard_gate_fallback_to_rank=_auto_safe_bool(
                data.get(
                    "auto_mode_phase2_hard_gate_fallback_to_rank",
                    AUTO_MODE_PHASE2_HARD_GATE_FALLBACK_TO_RANK,
                ),
                AUTO_MODE_PHASE2_HARD_GATE_FALLBACK_TO_RANK,
            ),
            residual_peak_threshold_db=max(
                0.0,
                _auto_safe_float(
                    data.get(
                        "auto_mode_residual_peak_threshold_db",
                        AUTO_MODE_RESIDUAL_PEAK_THRESHOLD_DB,
                    ),
                    AUTO_MODE_RESIDUAL_PEAK_THRESHOLD_DB,
                ),
            ),
            residual_peak_hard_gate_db=max(
                0.0,
                _auto_safe_float(
                    data.get(
                        "auto_mode_residual_peak_hard_gate_db",
                        AUTO_MODE_RESIDUAL_PEAK_HARD_GATE_DB,
                    ),
                    AUTO_MODE_RESIDUAL_PEAK_HARD_GATE_DB,
                ),
            ),
            residual_peak_penalty_cap=max(
                0.0,
                _auto_safe_float(
                    data.get(
                        "auto_mode_residual_peak_penalty_cap",
                        data.get("residual_peak_penalty_cap", AUTO_MODE_RESIDUAL_PEAK_PENALTY_CAP),
                    ),
                    AUTO_MODE_RESIDUAL_PEAK_PENALTY_CAP,
                ),
            ),
            tdc_decay_penalty_cap=max(
                0.0,
                _auto_safe_float(
                    data.get(
                        "auto_mode_tdc_decay_penalty_cap",
                        AUTO_MODE_TDC_DECAY_PENALTY_CAP,
                    ),
                    AUTO_MODE_TDC_DECAY_PENALTY_CAP,
                ),
            ),
            tdc_decay_penalty_weight=max(
                0.0,
                _auto_safe_float(
                    data.get(
                        "auto_mode_tdc_decay_penalty_weight",
                        AUTO_MODE_TDC_DECAY_PENALTY_WEIGHT,
                    ),
                    AUTO_MODE_TDC_DECAY_PENALTY_WEIGHT,
                ),
            ),
            tdc_extreme_peak_reduction_db=max(
                0.0,
                _auto_safe_float(
                    data.get(
                        "auto_mode_tdc_extreme_peak_reduction_db",
                        AUTO_MODE_TDC_EXTREME_PEAK_REDUCTION_DB,
                    ),
                    AUTO_MODE_TDC_EXTREME_PEAK_REDUCTION_DB,
                ),
            ),
            tdc_low_need_threshold=float(
                np.clip(
                    _auto_safe_float(
                        data.get(
                            "auto_mode_tdc_low_need_threshold",
                            AUTO_MODE_TDC_LOW_NEED_THRESHOLD,
                        ),
                        AUTO_MODE_TDC_LOW_NEED_THRESHOLD,
                    ),
                    0.0,
                    1.0,
                )
            ),
            tdc_rt60_target_low_s=max(
                0.01,
                _auto_safe_float(
                    data.get(
                        "auto_mode_tdc_rt60_target_low_s",
                        AUTO_MODE_TDC_RT60_TARGET_LOW_S,
                    ),
                    AUTO_MODE_TDC_RT60_TARGET_LOW_S,
                ),
            ),
            tdc_rt60_target_upper_s=max(
                0.01,
                _auto_safe_float(
                    data.get(
                        "auto_mode_tdc_rt60_target_upper_s",
                        AUTO_MODE_TDC_RT60_TARGET_UPPER_S,
                    ),
                    AUTO_MODE_TDC_RT60_TARGET_UPPER_S,
                ),
            ),
            tdc_rt60_low_max_hz=max(
                20.0,
                _auto_safe_float(
                    data.get(
                        "auto_mode_tdc_rt60_low_max_hz",
                        AUTO_MODE_TDC_RT60_LOW_MAX_HZ,
                    ),
                    AUTO_MODE_TDC_RT60_LOW_MAX_HZ,
                ),
            ),
            tdc_rt60_eval_max_hz=max(
                20.0,
                _auto_safe_float(
                    data.get(
                        "auto_mode_tdc_rt60_eval_max_hz",
                        AUTO_MODE_TDC_RT60_EVAL_MAX_HZ,
                    ),
                    AUTO_MODE_TDC_RT60_EVAL_MAX_HZ,
                ),
            ),
            max_avg_score_loss_for_safety_override=max(
                0.0,
                _auto_safe_float(
                    data.get(
                        "auto_mode_max_avg_score_loss_for_safety_override",
                        data.get(
                            "max_avg_score_loss_for_safety_override",
                            AUTO_MODE_MAX_AVG_SCORE_LOSS_FOR_SAFETY_OVERRIDE,
                        ),
                    ),
                    AUTO_MODE_MAX_AVG_SCORE_LOSS_FOR_SAFETY_OVERRIDE,
                ),
            ),
            residual_peak_winner_polish_enabled=_auto_safe_bool(
                data.get(
                    "auto_mode_residual_peak_winner_polish_enabled",
                    AUTO_MODE_RESIDUAL_PEAK_WINNER_POLISH_ENABLED,
                ),
                AUTO_MODE_RESIDUAL_PEAK_WINNER_POLISH_ENABLED,
            ),
            residual_peak_winner_polish_max_variants=max(
                1,
                _auto_safe_int(
                    data.get(
                        "auto_mode_residual_peak_winner_polish_max_variants",
                        AUTO_MODE_RESIDUAL_PEAK_WINNER_POLISH_MAX_VARIANTS,
                    ),
                    AUTO_MODE_RESIDUAL_PEAK_WINNER_POLISH_MAX_VARIANTS,
                ),
            ),
            residual_peak_winner_polish_min_improvement_db=max(
                0.0,
                _auto_safe_float(
                    data.get(
                        "auto_mode_residual_peak_winner_polish_min_improvement_db",
                        AUTO_MODE_RESIDUAL_PEAK_WINNER_POLISH_MIN_IMPROVEMENT_DB,
                    ),
                    AUTO_MODE_RESIDUAL_PEAK_WINNER_POLISH_MIN_IMPROVEMENT_DB,
                ),
            ),
            refine_mode_soft_k=max(
                0.0,
                _auto_safe_float(
                    data.get(
                        "auto_mode_refine_mode_soft_k", AUTO_MODE_REFINE_MODE_SOFT_K
                    ),
                    AUTO_MODE_REFINE_MODE_SOFT_K,
                ),
            ),
            refine_tiebreak_rank_eps=max(
                0.0,
                _auto_safe_float(
                    data.get(
                        "auto_mode_refine_tiebreak_rank_eps",
                        AUTO_MODE_REFINE_TIEBREAK_RANK_EPS,
                    ),
                    AUTO_MODE_REFINE_TIEBREAK_RANK_EPS,
                ),
            ),
            exc_min_hz=max(
                1.0,
                _auto_safe_float(
                    data.get("auto_mode_exc_min_hz", AUTO_MODE_EXC_MIN_HZ),
                    AUTO_MODE_EXC_MIN_HZ,
                ),
            ),
            exc_max_hz=max(
                1.0,
                _auto_safe_float(
                    data.get("auto_mode_exc_max_hz", AUTO_MODE_EXC_MAX_HZ),
                    AUTO_MODE_EXC_MAX_HZ,
                ),
            ),
            cache_enabled=_auto_safe_bool(
                data.get("auto_mode_cache_enabled", AUTO_MODE_CACHE_ENABLED),
                AUTO_MODE_CACHE_ENABLED,
            ),
            optuna_pilot_enabled=_auto_safe_bool(
                data.get("auto_mode_optuna", AUTO_MODE_OPTUNA_PILOT_ENABLED),
                AUTO_MODE_OPTUNA_PILOT_ENABLED,
            ),
            optuna_pilot_min_trials=max(
                1,
                _auto_safe_int(
                    data.get(
                        "auto_mode_optuna_min_trials", AUTO_MODE_OPTUNA_PILOT_MIN_TRIALS
                    ),
                    AUTO_MODE_OPTUNA_PILOT_MIN_TRIALS,
                ),
            ),
            optuna_pilot_startup_trials=max(
                1,
                _auto_safe_int(
                    data.get("auto_mode_optuna_startup_trials", legacy_startup),
                    legacy_startup,
                ),
            ),
            optuna_startup_phase1=max(
                1,
                _auto_safe_int(
                    data.get(
                        "auto_mode_optuna_startup_phase1",
                        AUTO_MODE_OPTUNA_STARTUP_PHASE1,
                    ),
                    AUTO_MODE_OPTUNA_STARTUP_PHASE1,
                ),
            ),
            optuna_startup_target=max(
                1,
                _auto_safe_int(
                    data.get(
                        "auto_mode_optuna_startup_target",
                        AUTO_MODE_OPTUNA_STARTUP_TARGET,
                    ),
                    AUTO_MODE_OPTUNA_STARTUP_TARGET,
                ),
            ),
            optuna_startup_local=max(
                1,
                _auto_safe_int(
                    data.get(
                        "auto_mode_optuna_startup_local", AUTO_MODE_OPTUNA_STARTUP_LOCAL
                    ),
                    AUTO_MODE_OPTUNA_STARTUP_LOCAL,
                ),
            ),
            optuna_startup_micro=max(
                1,
                _auto_safe_int(
                    data.get(
                        "auto_mode_optuna_startup_micro", AUTO_MODE_OPTUNA_STARTUP_MICRO
                    ),
                    AUTO_MODE_OPTUNA_STARTUP_MICRO,
                ),
            ),
            optuna_constraints_enabled=_auto_safe_bool(
                data.get(
                    "auto_mode_optuna_constraints", AUTO_MODE_OPTUNA_CONSTRAINTS_ENABLED
                ),
                AUTO_MODE_OPTUNA_CONSTRAINTS_ENABLED,
            ),
            optuna_constraints_refine_only=_auto_safe_bool(
                data.get(
                    "auto_mode_optuna_constraints_refine_only",
                    AUTO_MODE_OPTUNA_CONSTRAINTS_REFINE_ONLY,
                ),
                AUTO_MODE_OPTUNA_CONSTRAINTS_REFINE_ONLY,
            ),
            optuna_constraints_max_mode_ripple_db=max(
                0.0,
                _auto_safe_float(
                    data.get(
                        "auto_mode_optuna_constraints_max_mode_ripple_db",
                        AUTO_MODE_OPTUNA_CONSTRAINTS_MAX_MODE_RIPPLE_DB,
                    ),
                    AUTO_MODE_OPTUNA_CONSTRAINTS_MAX_MODE_RIPPLE_DB,
                ),
            ),
            optuna_constraints_max_events_severity=max(
                0.0,
                _auto_safe_float(
                    data.get(
                        "auto_mode_optuna_constraints_max_events_severity",
                        AUTO_MODE_OPTUNA_CONSTRAINTS_MAX_EVENTS_SEVERITY,
                    ),
                    AUTO_MODE_OPTUNA_CONSTRAINTS_MAX_EVENTS_SEVERITY,
                ),
            ),
            optuna_constraints_max_net_boost_db=max(
                0.0,
                _auto_safe_float(
                    data.get(
                        "auto_mode_optuna_constraints_max_net_boost_db",
                        AUTO_MODE_OPTUNA_CONSTRAINTS_MAX_NET_BOOST_DB,
                    ),
                    AUTO_MODE_OPTUNA_CONSTRAINTS_MAX_NET_BOOST_DB,
                ),
            ),
            optuna_constraints_use_events_in_refine=_auto_safe_bool(
                data.get(
                    "auto_mode_optuna_constraints_use_events_in_refine",
                    AUTO_MODE_OPTUNA_CONSTRAINTS_USE_EVENTS_IN_REFINE,
                ),
                AUTO_MODE_OPTUNA_CONSTRAINTS_USE_EVENTS_IN_REFINE,
            ),
            optuna_telemetry=_auto_safe_bool(
                data.get("auto_mode_optuna_telemetry", AUTO_MODE_OPTUNA_TELEMETRY),
                AUTO_MODE_OPTUNA_TELEMETRY,
            ),
            optuna_telemetry_log_summary=_auto_safe_bool(
                data.get(
                    "auto_mode_optuna_telemetry_log_summary",
                    AUTO_MODE_OPTUNA_TELEMETRY_LOG_SUMMARY,
                ),
                AUTO_MODE_OPTUNA_TELEMETRY_LOG_SUMMARY,
            ),
            optuna_constraints_zero_feasible_fallback=_auto_safe_bool(
                data.get(
                    "auto_mode_optuna_constraints_zero_feasible_fallback",
                    AUTO_MODE_OPTUNA_CONSTRAINTS_ZERO_FEASIBLE_FALLBACK,
                ),
                AUTO_MODE_OPTUNA_CONSTRAINTS_ZERO_FEASIBLE_FALLBACK,
            ),
            optuna_multivariate=_auto_safe_bool(
                data.get(
                    "auto_mode_optuna_multivariate", AUTO_MODE_OPTUNA_MULTIVARIATE
                ),
                AUTO_MODE_OPTUNA_MULTIVARIATE,
            ),
            optuna_group=_auto_safe_bool(
                data.get("auto_mode_optuna_group", AUTO_MODE_OPTUNA_GROUP),
                AUTO_MODE_OPTUNA_GROUP,
            ),
            optuna_constant_liar=_auto_safe_bool(
                data.get(
                    "auto_mode_optuna_constant_liar", AUTO_MODE_OPTUNA_CONSTANT_LIAR
                ),
                AUTO_MODE_OPTUNA_CONSTANT_LIAR,
            ),
            optuna_persistent_study=_auto_safe_bool(
                data.get(
                    "auto_mode_optuna_persistent_study",
                    AUTO_MODE_OPTUNA_PERSISTENT_STUDY,
                ),
                AUTO_MODE_OPTUNA_PERSISTENT_STUDY,
            ),
            optuna_avoid_duplicates=_auto_safe_bool(
                data.get(
                    "auto_mode_optuna_avoid_duplicates",
                    AUTO_MODE_OPTUNA_AVOID_DUPLICATES,
                ),
                AUTO_MODE_OPTUNA_AVOID_DUPLICATES,
            ),
            optuna_cross_study_seeds=_auto_safe_bool(
                data.get(
                    "auto_mode_optuna_cross_study_seeds",
                    AUTO_MODE_OPTUNA_CROSS_STUDY_SEEDS,
                ),
                AUTO_MODE_OPTUNA_CROSS_STUDY_SEEDS,
            ),
            optuna_cross_study_seeds_top_n=max(
                1,
                _auto_safe_int(
                    data.get(
                        "auto_mode_optuna_cross_study_seeds_top_n",
                        AUTO_MODE_OPTUNA_CROSS_STUDY_SEEDS_TOP_N,
                    ),
                    AUTO_MODE_OPTUNA_CROSS_STUDY_SEEDS_TOP_N,
                ),
            ),
            optuna_pruning_enabled=_auto_safe_bool(
                data.get(
                    "auto_mode_optuna_pruning_enabled", AUTO_MODE_OPTUNA_PRUNING_ENABLED
                ),
                AUTO_MODE_OPTUNA_PRUNING_ENABLED,
            ),
            optuna_pruning_n_startup=max(
                1,
                _auto_safe_int(
                    data.get(
                        "auto_mode_optuna_pruning_n_startup",
                        AUTO_MODE_OPTUNA_PRUNING_N_STARTUP,
                    ),
                    AUTO_MODE_OPTUNA_PRUNING_N_STARTUP,
                ),
            ),
        )

    def refine_trial_hint(self, goal: str | None) -> int:
        goal_norm = _auto_goal_norm(goal)
        hint = int(max(1, self.refine_trials))
        if bool(self.local_refine_enabled) and goal_norm in (
            AUTO_MODE_GOAL_DEFAULT,
            AUTO_MODE_GOAL_ROOM_SAFE,
            AUTO_MODE_GOAL_LOW_RIPPLE,
            AUTO_MODE_GOAL_SUBWOOFERS,
            AUTO_MODE_GOAL_ACOUSTIC,
            AUTO_MODE_GOAL_HYBRID,
        ):
            hint = int(
                max(1, self.local_refine_top_k)
                * max(1, self.local_refine_trials_per_top)
            )
        return int(max(1, hint))


__all__ = ['AutoModeConfig']


def _link_sibling_exports() -> None:
    import importlib
    package = __package__
    for module_name in ['cache_hash', 'goal_profile', 'safe_values', 'backend', 'config', 'phase_sampling']:
        if module_name == __name__.rsplit('.', 1)[-1]:
            continue
        module = importlib.import_module(f"{package}.{module_name}")
        for symbol in getattr(module, "__all__", ()):
            globals().setdefault(symbol, getattr(module, symbol))


_link_sibling_exports()
