# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Shared dataclasses for refine-stage orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field

from .refine_eval import RefineEvalContext


@dataclass(slots=True)
class _CacheRefineContext:
    params: dict


@dataclass(slots=True)
class _CacheRefineOutcome:
    result: dict | None = None
    best_preset: dict = field(default_factory=dict)
    best_metrics: dict = field(default_factory=dict)


@dataclass(slots=True)
class _CacheRefineSeed:
    cache_target_name: str
    best_preset: dict = field(default_factory=dict)
    best_metrics: dict = field(default_factory=dict)
    seed_source: str = "exact_cache"


@dataclass(slots=True)
class _CacheRefineProgress:
    cache_target_name: str
    seed_source: str = "exact_cache"
    best_preset: dict = field(default_factory=dict)
    best_metrics: dict = field(default_factory=dict)
    initial_best_preset: dict = field(default_factory=dict)
    micro_trials: int = 1
    min_round_improvement: float = 0.0
    improved_any: bool = False
    improved_count_total: int = 0
    executed_micro_trials_total: int = 0
    rounds_executed: int = 0
    stop_reason: str = "max_rounds"
    cache_refine_optuna_tels: list[dict] = field(default_factory=list)


@dataclass(slots=True)
class _CacheRefineRound:
    round_idx: int
    round_start_metrics: dict = field(default_factory=dict)
    round_start_preset: dict = field(default_factory=dict)
    raw_scope: object | None = None
    round_seed_presets: list[dict] = field(default_factory=list)
    round_improved_count: int = 0
    round_executed: int = 0
    round_tel: dict = field(default_factory=dict)


@dataclass(slots=True)
class _SearchRefineContext:
    params: dict


@dataclass(slots=True)
class _SearchRefineSummary:
    result: dict = field(default_factory=dict)


@dataclass(slots=True)
class _SearchPhase1State:
    ctx: RefineEvalContext
    phase1_ok: int = 0
    phase1_tried: int = 0
    phase1_plateau_hit: bool = False
    phase1_optuna_tel: dict = field(default_factory=dict)
    phase1_top: list[dict] = field(default_factory=list)
    phase1_best_metrics: dict | None = None
    phase1_best_preset: dict | None = None


@dataclass(slots=True)
class _SearchPhase2State:
    phase2_ok: int = 0
    phase2_tried: int = 0
    phase2_plateau_hit: bool = False
    phase2_focus_lo: float = float("nan")
    phase2_focus_hi: float = float("nan")
    phase2_local_optuna_tels: list[dict] = field(default_factory=list)
    phase3_micro_optuna_tel: dict = field(default_factory=dict)
