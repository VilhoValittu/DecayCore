# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Refine-stage orchestration for automatic mode."""

from __future__ import annotations

from ._refine_cache_rounds import (
    _run_builtin_cache_refine_round as _run_builtin_cache_refine_round,
    _run_optuna_cache_refine_round as _run_optuna_cache_refine_round,
    run_exact_cache_micro_refine,
)
from ._refine_cache_setup import (
    _build_cache_refine_progress as _build_cache_refine_progress,
    _build_exact_cache_refine_context as _build_exact_cache_refine_context,
    _cache_trial_issue_label as _cache_trial_issue_label,
    _cache_trial_log_method as _cache_trial_log_method,
    _load_exact_cache_seed as _load_exact_cache_seed,
    _log_cache_refine_round_summary as _log_cache_refine_round_summary,
    _prepare_cache_refine_round as _prepare_cache_refine_round,
    _record_cache_refine_candidate as _record_cache_refine_candidate,
    _remember_cache_refine_round_seed as _remember_cache_refine_round_seed,
)
from . import _refine_search_pipeline as _search_pipeline
from ._refine_search_pipeline import (
    _assemble_refine_summary,
    _run_phase1_coarse_search,
    _run_phase1_search,
    _run_phase2_local_refine,
    _run_phase3_micro_refine,
)
from ._refine_types import (
    _CacheRefineContext as _CacheRefineContext,
    _CacheRefineOutcome as _CacheRefineOutcome,
    _CacheRefineProgress as _CacheRefineProgress,
    _CacheRefineRound as _CacheRefineRound,
    _CacheRefineSeed as _CacheRefineSeed,
    _SearchPhase1State as _SearchPhase1State,
    _SearchPhase2State as _SearchPhase2State,
    _SearchRefineContext as _SearchRefineContext,
    _SearchRefineSummary as _SearchRefineSummary,
)

def run_search_refine_stages(**kwargs):
    for _name in (
        "_run_phase1_search",
        "_run_phase1_coarse_search",
        "_run_phase2_local_refine",
        "_run_phase3_micro_refine",
        "_assemble_refine_summary",
    ):
        setattr(_search_pipeline, _name, globals()[_name])
    return _search_pipeline.run_search_refine_stages(**kwargs)


__all__ = [
    "run_exact_cache_micro_refine",
    "run_search_refine_stages",
    "_run_phase1_search",
    "_run_phase1_coarse_search",
    "_run_phase2_local_refine",
    "_run_phase3_micro_refine",
    "_assemble_refine_summary",
    "_run_builtin_cache_refine_round",
    "_run_optuna_cache_refine_round",
    "_build_cache_refine_progress",
    "_build_exact_cache_refine_context",
    "_cache_trial_issue_label",
    "_cache_trial_log_method",
    "_load_exact_cache_seed",
    "_log_cache_refine_round_summary",
    "_prepare_cache_refine_round",
    "_record_cache_refine_candidate",
    "_remember_cache_refine_round_seed",
    "_CacheRefineContext",
    "_CacheRefineOutcome",
    "_CacheRefineProgress",
    "_CacheRefineRound",
    "_CacheRefineSeed",
    "_SearchPhase1State",
    "_SearchPhase2State",
    "_SearchRefineContext",
    "_SearchRefineSummary",
]
