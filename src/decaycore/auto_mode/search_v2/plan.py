# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Auto Search v2 plan models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AutoSearchPlan(Enum):
    FULL_SEARCH = "full_search"
    PRESELECTED_TARGET_REFINE = "preselected_target_refine"
    CACHE_MICRO_REFINE = "cache_micro_refine"
    LAST_BEST_MICRO_REFINE = "last_best_micro_refine"
    MANUAL_PRESET_REFINE = "manual_preset_refine"
    REUSE_VALID_RESULT = "reuse_valid_result"


@dataclass(frozen=True)
class AutoSearchPlanDecision:
    plan: AutoSearchPlan
    reason: str
    signature: str
    seed_preset: dict | None
    cache_record: dict | None
    skipped_phases: tuple[str, ...]
    enabled_phases: tuple[str, ...]
    fallback_reasons: tuple[str, ...]
    cache_decision_report: dict | None = None
    seed_source: str | None = None
