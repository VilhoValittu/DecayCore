# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Finalize-stage orchestration for automatic mode.

All implementation is split across:
- orchestrator_finalize_polish.py  — winner-polish helpers
- orchestrator_finalize_run.py     — Pareto finalize and top-level entry point

This module re-exports the public API and the names that monkeypatch-based
tests patch directly on this module's namespace.
"""

from __future__ import annotations

from .orchestrator_finalize_polish import (
    _apply_search_winner_polish_pipeline,
    _save_cached_best,
    _save_final_search_cache,
    _set_search_winner_if_improved,
)
from .orchestrator_finalize_run import (
    _finalize_cached_result,
    _run_phase2_pareto_finalize,
    finalize_search_result,
)

# Re-export the names that tests import or patch directly on this module's
# namespace so nothing breaks without touching test files.
from .orchestrator_finalize_cache_parts import (
    _build_phase2_pareto_status,
    _build_modal_intelligence_debug,
    _cache_refine_winner_phase_label as _cache_refine_winner_phase_label,
    _cache_refine_winner_summary as _cache_refine_winner_summary,
    _resolve_winner_auto_exc_hz,
)
from .cache_signature import (
    _auto_cache_put_best as _auto_cache_put_best,
    _auto_cache_put_last_used_best as _auto_cache_put_last_used_best,
    _auto_cache_put_target_for_measurements as _auto_cache_put_target_for_measurements,
    _auto_measurement_signature as _auto_measurement_signature,
    _auto_signature as _auto_signature,
)
from .orchestrator_finalize_polish import (
    _save_cached_best as _save_cached_best,  # needed so tests can call it via this module
)
from .scoring_ranking import (
    _auto_phase2_pareto_front,
    _auto_select_best_scored,
)
from .stereo_policy_refine import apply_stereo_policy_refine
from .winner_polish import (
    apply_excess_phase_strength_winner_polish,
    apply_hpf_winner_polish,
    apply_low_bass_cut_winner_polish,
    apply_mag_c_min_winner_polish,
    apply_phase_limit_winner_polish,
    apply_residual_peak_winner_polish,
    apply_tdc_strength_winner_polish as apply_tdc_strength_winner_polish,
)

__all__ = [
    "finalize_search_result",
    "_save_cached_best",
    "_save_final_search_cache",
    "_set_search_winner_if_improved",
    "_apply_search_winner_polish_pipeline",
    "_run_phase2_pareto_finalize",
    "_finalize_cached_result",
    # re-exported for test imports and monkeypatching
    "_build_phase2_pareto_status",
    "_build_modal_intelligence_debug",
    "_cache_refine_winner_phase_label",
    "_cache_refine_winner_summary",
    "_resolve_winner_auto_exc_hz",
    "_auto_cache_put_best",
    "_auto_cache_put_last_used_best",
    "_auto_cache_put_target_for_measurements",
    "_auto_measurement_signature",
    "_auto_signature",
    "_auto_phase2_pareto_front",
    "_auto_select_best_scored",
    "apply_stereo_policy_refine",
    "apply_excess_phase_strength_winner_polish",
    "apply_hpf_winner_polish",
    "apply_low_bass_cut_winner_polish",
    "apply_mag_c_min_winner_polish",
    "apply_phase_limit_winner_polish",
    "apply_residual_peak_winner_polish",
    "apply_tdc_strength_winner_polish",
]
