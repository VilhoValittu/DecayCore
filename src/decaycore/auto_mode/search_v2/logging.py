# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Auto Search v2 decision logging."""

from __future__ import annotations

from ..shared import logger
from .cache import AUTO_SEARCH_CACHE_SCHEMA_VERSION
from .plan import AutoSearchPlanDecision


def log_plan_decision(decision: AutoSearchPlanDecision, *, seed_source: str | None = None) -> None:
    report = dict(decision.cache_decision_report or {})
    exact = dict(report.get("exact_cache", {}) or {})
    last_best = dict(report.get("last_best", {}) or {})
    logger.info(
        "=== AUTO SEARCH V2 DECISION ===\n"
        "Signature: %s\n"
        "Plan: %s\n"
        "Reason: %s\n"
        "Skipped phases: %s\n"
        "Running phases: %s\n"
        "Seed source: %s\n"
        "Fallback reasons: %s\n"
        "Exact cache: %s (%s)\n"
        "Last-best cache: %s (%s)\n"
        "Signature inputs: %s\n"
        "Cache schema: %d",
        str(decision.signature),
        str(decision.plan.name),
        str(decision.reason),
        ", ".join(decision.skipped_phases) if decision.skipped_phases else "none",
        ", ".join(decision.enabled_phases) if decision.enabled_phases else "none",
        str(seed_source or "none"),
        "; ".join(decision.fallback_reasons) if decision.fallback_reasons else "none",
        str(exact.get("status", "unknown") or "unknown"),
        str(exact.get("reason", "") or ""),
        str(last_best.get("status", "unknown") or "unknown"),
        str(last_best.get("reason", "") or ""),
        str(report.get("diff_reason", "") or ""),
        int(AUTO_SEARCH_CACHE_SCHEMA_VERSION),
    )
