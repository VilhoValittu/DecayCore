# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Target-curve selection orchestration for auto mode."""

from __future__ import annotations

import logging

from .shared_parts import (
    AUTO_MODE_TARGET_TOP_N,
    AUTO_MODE_TARGET_TRIALS_PER_CURVE,
)

# Re-export all sub-module symbols for backward compatibility
from .orchestrator_target_types import (  # noqa: F401
    _TARGET_CACHE_REQUIRED_HIDDEN_KEYS,
    _TargetEvalMaterialization,
    _TargetEvalSummary,
    _TargetTrialSetup,
    _TargetTrialAccumulator,
    _TargetSelectionSetup,
    _TargetCacheState,
    _TargetShortlistState,
    _TargetSelectionContext,
    _TargetSelectionOutcome,
    _target_trial_log_method,
    _target_trial_issue_label,
)

from .orchestrator_target_cache import (  # noqa: F401
    _cache_target_valid,
    _cached_target_fit,
    _cached_target_return,
    _cached_target_seed_preset,
    _missing_cached_target_hidden_keys,
    _cached_target_has_exact_preset,
    _set_cached_target_state,
    _cached_target_matches_current_hc,
    _auto_target_mode_locks_hc,
    _cached_target_state_from_optuna_study,
    _resolve_cached_target_state,
    _fallback_to_cached_target,
    _try_exact_cached_target_result,
)

from .orchestrator_target_trials import (  # noqa: F401
    _target_eval_one,
    _run_target_trials,
    _materialize_target_candidate,
    _run_target_eval_trials,
    _summarize_target_eval,
    _build_target_eval_result,
    _load_target_curve_arrays,
    _prepare_target_trial_setup,
    _evaluate_target_curve,
)

from .orchestrator_target_phase1 import (  # noqa: F401
    _run_target_phase1_trials,
    _run_target_local_refine_trials,
    _build_target_eval_core_result,
    _run_target_eval_trials_core,
)

from .orchestrator_target_shortlist import (  # noqa: F401
    _load_quick_target_selection,
    _apply_target_shortlist_modifiers,
    _evaluate_target_shortlist_core,
)

from .orchestrator_target_selection import (  # noqa: F401
    _prepare_target_selection_setup,
    _try_adaptive_target_fast_path,
    _finalize_target_selection_result,
    _resolve_target_cache_seed,
    _build_target_candidate_list,
    _run_target_shortlist_trials,
    _choose_target_winner,
    _finalize_target_selection_metadata,
    _select_target_curve_with_trials_impl,
)

logger = logging.getLogger("DecayCore")
__all__ = ["select_target_curve_with_trials"]


def select_target_curve_with_trials(
    *,
    base_data: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    pin_obj,
    status_cb=None,
    top_n: int = AUTO_MODE_TARGET_TOP_N,
    trials_per_curve: int = AUTO_MODE_TARGET_TRIALS_PER_CURVE,
    runtime=None,
) -> dict | None:
    return _select_target_curve_with_trials_impl(
        base_data=base_data,
        measurements=measurements,
        fs_v=int(fs_v),
        taps_v=int(taps_v),
        xos=list(xos or []),
        hpf=hpf,
        pin_obj=pin_obj,
        status_cb=status_cb,
        top_n=int(top_n),
        trials_per_curve=int(trials_per_curve),
        runtime=runtime,
    )
