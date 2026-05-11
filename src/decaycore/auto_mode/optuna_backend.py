# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Optuna backend helpers for automatic mode — coordinator re-exporting all sub-modules."""

from __future__ import annotations

import logging

from .optuna_backend_storage import (
    _OPTUNA_KNOWN_SIGNATURES,
    _OPTUNA_KNOWN_SIGNATURES_PRIMED,
    _OPTUNA_KNOWN_RECORDS,
    _OPTUNA_CROSS_STUDY_BEST_PARAMS,
    _OPTUNA_STUDY_SCAN_STATS,
    _auto_import_optuna,
    _auto_optuna_cached_study_records,
    _auto_optuna_create_storage,
    _auto_optuna_create_study,
    _auto_optuna_get_known_signatures,
    _auto_optuna_mark_signature_seen,
    _auto_optuna_module_ready,
    _auto_optuna_note_trial_scan,
    _auto_optuna_prime_known_signatures_from_study,
    _auto_optuna_storage_filename,
    _auto_optuna_storage_path,
    _auto_optuna_study_name,
    _auto_optuna_study_scan_stats_snapshot,
    _auto_optuna_update_known_record,
)

from .optuna_backend_params import (
    _auto_optuna_adaptive_freq_bounds,
    _auto_optuna_build_completed_trial,
    _auto_optuna_cross_study_best_params,
    _auto_optuna_jsonable,
    _auto_optuna_param_signature,
    _auto_optuna_sanitize_enqueued_params,
    _auto_optuna_scope_context_hash,
    _auto_optuna_scope_with_context,
    _auto_optuna_tdc_min,
    _auto_optuna_trial_distributions,
    _auto_optuna_trial_params,
    _auto_optuna_trial_payload_preset,
)

from .optuna_backend_constraints import (
    _auto_optuna_constraint_scope_kind,
    _auto_optuna_constraint_thresholds,
    _auto_optuna_constraint_vector_from_metrics,
    _auto_optuna_constraints_enabled_for_scope,
    _auto_optuna_constraints_func,
    _auto_optuna_effective_scope,
    _auto_optuna_is_refine_phase_kind,
    _auto_optuna_startup_for_phase_kind,
    _auto_optuna_trial_out_payload,
    _auto_optuna_use_events_constraint,
)

from .optuna_backend_scoring import (
    _AUTO_OPTUNA_BASS_PREFERENCE_LF_BOOST_TARGET_DB,
    _AUTO_OPTUNA_BASS_PREFERENCE_LF_BOOST_WEIGHT,
    _AUTO_OPTUNA_BASS_PREFERENCE_MAX_BONUS,
    _AUTO_OPTUNA_BASS_PREFERENCE_REALIZED_RMS_OK_DB,
    _AUTO_OPTUNA_BASS_PREFERENCE_REALIZED_RMS_WEIGHT,
    _auto_optuna_attach_out_telemetry,
    _auto_optuna_base_data_without_constraints,
    _auto_optuna_bass_preference_bonus,
    _auto_optuna_build_run_telemetry,
    _auto_optuna_constraint_info_for_metrics,
    _auto_optuna_needs_zero_feasible_rescue,
    _auto_optuna_run_token,
    _auto_optuna_trial_objective_value,
)

from .optuna_backend_records import (
    _auto_optuna_objective_value,
    _auto_optuna_remember_result,
    _auto_optuna_study_records,
)

from .optuna_backend_loop import (
    _OptunaEvalContext,
    _OptunaEvalState,
    _auto_run_optuna_eval_loop,
    _consume_completed_trial,
    _finalize_optuna_eval_loop,
    _log_optuna_duplicate_summary,
    _optuna_pruned_result,
    _prepare_optuna_eval_context,
    _run_optuna_parallel_trials,
    _run_optuna_seed_trials,
    _run_optuna_serial_trials,
    _submit_or_schedule_trials,
    _update_best_and_telemetry,
)

from .optuna_backend_core import _auto_run_optuna_eval_loop_core

logger = logging.getLogger("DecayCore")
