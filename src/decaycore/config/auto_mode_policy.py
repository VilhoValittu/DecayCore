# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

from __future__ import annotations


AUTO_MODE_GOAL_DEFAULT = "balanced"
AUTO_MODE_GOAL_ROOM_SAFE = "room-safe"
AUTO_MODE_GOAL_LOW_RIPPLE = "low-ripple"
AUTO_MODE_GOAL_FLAT = "flat"
AUTO_MODE_GOAL_SUBWOOFERS = "subwoofers"
AUTO_MODE_GOAL_ACOUSTIC = "acoustic"
AUTO_MODE_GOAL_HYBRID = "hybrid"
AUTO_MODE_GOAL_PREFER_BASS = "prefer-bass"
AUTO_MODE_COMPAT_VERSION = "am52"
AUTO_MODE_SUBWOOFERS_LEVEL_MIN_HZ = 20.0
AUTO_MODE_SUBWOOFERS_LEVEL_MAX_HZ = 200.0

AUTO_MODE_GOALS = frozenset(
    {
        AUTO_MODE_GOAL_DEFAULT,
        AUTO_MODE_GOAL_ROOM_SAFE,
        AUTO_MODE_GOAL_LOW_RIPPLE,
        AUTO_MODE_GOAL_FLAT,
        AUTO_MODE_GOAL_SUBWOOFERS,
        AUTO_MODE_GOAL_ACOUSTIC,
        AUTO_MODE_GOAL_HYBRID,
        AUTO_MODE_GOAL_PREFER_BASS,
    }
)

AUTO_MODE_CACHE_FILTER_KEYS = ("linear", "mixed", "minimum", "asym")


def normalize_auto_goal(goal: str | None) -> str:
    goal_norm = str(goal or AUTO_MODE_GOAL_DEFAULT).strip().lower()
    aliases = {
        "c": AUTO_MODE_GOAL_FLAT,
        "bass": AUTO_MODE_GOAL_PREFER_BASS,
        "prefer bass": AUTO_MODE_GOAL_PREFER_BASS,
        "prefer_bass": AUTO_MODE_GOAL_PREFER_BASS,
        "room_safe": AUTO_MODE_GOAL_ROOM_SAFE,
        "roomsafe": AUTO_MODE_GOAL_ROOM_SAFE,
        "low_ripple": AUTO_MODE_GOAL_LOW_RIPPLE,
        "lowripple": AUTO_MODE_GOAL_LOW_RIPPLE,
        "subwoofer": AUTO_MODE_GOAL_SUBWOOFERS,
        "subs": AUTO_MODE_GOAL_SUBWOOFERS,
    }
    goal_norm = str(aliases.get(goal_norm, goal_norm))
    return goal_norm if goal_norm in AUTO_MODE_GOALS else AUTO_MODE_GOAL_DEFAULT


def auto_goal_is_flat_family(goal: str | None) -> bool:
    return normalize_auto_goal(goal) in {
        AUTO_MODE_GOAL_FLAT,
        AUTO_MODE_GOAL_PREFER_BASS,
        AUTO_MODE_GOAL_ACOUSTIC,
    }


def auto_goal_is_low_ripple_family(goal: str | None) -> bool:
    return normalize_auto_goal(goal) in {
        AUTO_MODE_GOAL_LOW_RIPPLE,
        AUTO_MODE_GOAL_HYBRID,
    }


def auto_goal_forced_level_window(goal: str | None) -> tuple[float, float] | None:
    if normalize_auto_goal(goal) == AUTO_MODE_GOAL_SUBWOOFERS:
        return (
            AUTO_MODE_SUBWOOFERS_LEVEL_MIN_HZ,
            AUTO_MODE_SUBWOOFERS_LEVEL_MAX_HZ,
        )
    return None


def auto_filter_cache_key(
    base_data: dict | None = None,
    *,
    filter_type: str | None = None,
) -> str:
    value = str(
        filter_type
        if filter_type is not None
        else (base_data or {}).get("filter_type", "") or ""
    ).strip().lower()
    if value in AUTO_MODE_CACHE_FILTER_KEYS:
        return value
    if "asym" in value:
        return "asym"
    if "mixed" in value:
        return "mixed"
    if "minimum" in value or "minphase" in value or ("min" in value and "phase" in value):
        return "minimum"
    if "linear" in value:
        return "linear"
    return "asym"


def auto_filter_type_for_key(filter_key: str | None) -> str:
    key = auto_filter_cache_key(filter_type=str(filter_key or ""))
    if key == "asym":
        return "Asymmetric"
    if key == "linear":
        return "Linear Phase"
    if key == "minimum":
        return "Minimum Phase"
    return "Mixed"


__all__ = [
    "AUTO_MODE_CACHE_FILTER_KEYS",
    "AUTO_MODE_COMPAT_VERSION",
    "AUTO_MODE_GOAL_ACOUSTIC",
    "AUTO_MODE_GOAL_DEFAULT",
    "AUTO_MODE_GOAL_FLAT",
    "AUTO_MODE_GOAL_HYBRID",
    "AUTO_MODE_GOAL_LOW_RIPPLE",
    "AUTO_MODE_GOAL_PREFER_BASS",
    "AUTO_MODE_GOAL_ROOM_SAFE",
    "AUTO_MODE_GOAL_SUBWOOFERS",
    "AUTO_MODE_GOALS",
    "AUTO_MODE_SUBWOOFERS_LEVEL_MAX_HZ",
    "AUTO_MODE_SUBWOOFERS_LEVEL_MIN_HZ",
    "auto_filter_cache_key",
    "auto_filter_type_for_key",
    "auto_goal_is_flat_family",
    "auto_goal_forced_level_window",
    "auto_goal_is_low_ripple_family",
    "normalize_auto_goal",
]
