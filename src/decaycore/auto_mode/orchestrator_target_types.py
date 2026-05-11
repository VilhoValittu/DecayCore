# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Dataclasses and helper functions for target-curve selection orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("DecayCore")

_TARGET_CACHE_REQUIRED_HIDDEN_KEYS = (
    "max_slope_boost_db_per_oct",
    "max_slope_cut_db_per_oct",
    "conf_pull_max_hz",
)


@dataclass(slots=True)
class _TargetEvalMaterialization:
    tc: dict
    hc_name: str


@dataclass(slots=True)
class _TargetEvalSummary:
    item: dict | None = None
    best_metrics: dict = field(default_factory=dict)
    best_preset: dict = field(default_factory=dict)


@dataclass(slots=True)
class _TargetTrialSetup:
    hc_name: str
    hc_f: np.ndarray
    hc_m: np.ndarray
    seed_tc: int
    base_tc: dict
    use_optuna_curve_trials: bool
    candidates: list[dict] = field(default_factory=list)
    phase1_seed_presets: list[dict] = field(default_factory=list)
    phase1_trial_total: int = 0


@dataclass(slots=True)
class _TargetTrialAccumulator:
    best_metrics: dict | None = None
    best_preset: dict | None = None
    ok_n: int = 0
    rank_sum: float = 0.0
    avg_score_sum: float = 0.0
    trials_total_count: int = 0
    phase1_scored: list[dict] = field(default_factory=list)
    curve_scored: list[dict] = field(default_factory=list)


@dataclass(slots=True)
class _TargetSelectionSetup:
    runtime: object
    cfg: object
    goal: str
    compat_version: str
    filter_key: str
    rank_basis: str
    optimizer_backend: str
    optuna_mod: object | None
    seed_target: int
    target_study_sig: str


@dataclass(slots=True)
class _TargetCacheState:
    cached_target_hc: str | None = None
    cached_target_preset: dict = field(default_factory=dict)
    cached_target_metrics: dict = field(default_factory=dict)
    cached_target_source: str | None = None


@dataclass(slots=True)
class _TargetShortlistState:
    quick: dict
    quick_candidates: list[dict]
    shortlisted: list[dict]
    trials_eff: int
    f6_hz: float
    f6_txt: str
    cache_wildcard_participated: bool = False


@dataclass(slots=True)
class _TargetSelectionContext:
    params: dict


@dataclass(slots=True)
class _TargetSelectionOutcome:
    result: dict | None = None
    candidates: list[dict] = field(default_factory=list)
    evaluated: list[dict] = field(default_factory=list)
    winner: dict | None = None


def _target_trial_log_method(*, out: dict):
    return logger.info if bool(dict(out or {}).get("pruned", False)) else logger.warning


def _target_trial_issue_label(*, out: dict) -> str:
    return "pruned" if bool(dict(out or {}).get("pruned", False)) else "failed"
