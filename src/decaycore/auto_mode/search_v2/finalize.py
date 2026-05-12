# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Auto Search v2 finalization helpers."""

from __future__ import annotations


def _merge_fallback_reasons(*groups) -> list[str]:
    out = []
    seen = set()
    for group in groups:
        for item in list(group or []):
            reason = str(item or "").strip()
            if reason and reason not in seen:
                out.append(reason)
                seen.add(reason)
    return out


def attach_search_v2_debug(result: dict | None, *, decision) -> dict | None:
    if not isinstance(result, dict):
        return result
    if "auto_mode_debug" not in result:
        return result
    out = dict(result)
    debug = dict(out.get("auto_mode_debug", {}) or {})
    fallback_reasons = _merge_fallback_reasons(
        debug.get("fallback_reasons", []),
        getattr(decision, "fallback_reasons", ()),
    )
    debug["auto_search_v2"] = {
        "plan": str(decision.plan.value),
        "reason": str(decision.reason),
        "signature": str(decision.signature),
        "skipped_phases": list(decision.skipped_phases),
        "enabled_phases": list(decision.enabled_phases),
        "fallback_reasons": list(fallback_reasons),
        "cache_decision_report": dict(getattr(decision, "cache_decision_report", {}) or {}),
    }
    debug["fallback_reasons"] = list(fallback_reasons)
    out["auto_mode_debug"] = debug
    out["enabled_phases"] = list(decision.enabled_phases)
    out["skipped_phases"] = list(decision.skipped_phases)
    return out
