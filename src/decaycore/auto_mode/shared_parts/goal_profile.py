# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

import logging

import numpy as np

from ._constants import *

logger = logging.getLogger("DecayCore")


def _auto_goal_norm(goal: str | None) -> str:
    goal_norm = str(goal or AUTO_MODE_GOAL_DEFAULT).strip().lower()
    goal_aliases = {
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
    goal_norm = str(goal_aliases.get(goal_norm, goal_norm))
    if goal_norm not in (
        AUTO_MODE_GOAL_DEFAULT,
        AUTO_MODE_GOAL_ROOM_SAFE,
        AUTO_MODE_GOAL_LOW_RIPPLE,
        AUTO_MODE_GOAL_FLAT,
        AUTO_MODE_GOAL_SUBWOOFERS,
        AUTO_MODE_GOAL_ACOUSTIC,
        AUTO_MODE_GOAL_HYBRID,
        AUTO_MODE_GOAL_PREFER_BASS,
    ):
        goal_norm = AUTO_MODE_GOAL_DEFAULT
    return str(goal_norm)


def _auto_goal_is_flat_family(goal: str | None) -> bool:
    """True for goals that share the flat objective's semantics (weights, gates,
    prefer-bass behavior). prefer-bass and acoustic historically normalized to
    flat; they stay in the family so canonicalizing them changes only the
    goal-specific rank-key dispatch, nothing else."""
    return _auto_goal_norm(goal) in (
        AUTO_MODE_GOAL_FLAT,
        AUTO_MODE_GOAL_PREFER_BASS,
        AUTO_MODE_GOAL_ACOUSTIC,
    )


def _auto_goal_is_low_ripple_family(goal: str | None) -> bool:
    """True for goals that share the low-ripple objective's semantics. hybrid
    historically normalized to low-ripple; it stays in the family."""
    return _auto_goal_norm(goal) in (
        AUTO_MODE_GOAL_LOW_RIPPLE,
        AUTO_MODE_GOAL_HYBRID,
    )


def _auto_bass_integration_profile_norm(profile: str | None) -> str:
    value = str(profile or "safe").strip().lower()
    if value not in ("safe", "normal", "assertive"):
        value = "safe"
    return str(value)


def _auto_bass_integration_profile_weights(profile: str | None) -> dict[str, float]:
    profile_norm = _auto_bass_integration_profile_norm(profile)
    return dict(
        AUTO_MODE_BASS_INTEGRATION_PROFILE_WEIGHTS.get(
            profile_norm, AUTO_MODE_BASS_INTEGRATION_PROFILE_WEIGHTS["safe"]
        )
    )


def _auto_goal_forced_level_window(goal: str | None) -> tuple[float, float] | None:
    if _auto_goal_norm(goal) == AUTO_MODE_GOAL_SUBWOOFERS:
        return (
            float(AUTO_MODE_SUBWOOFERS_LEVEL_MIN_HZ),
            float(AUTO_MODE_SUBWOOFERS_LEVEL_MAX_HZ),
        )
    return None


def _auto_builtin_target_name(hc_mode: str | None) -> str | None:
    key = str(hc_mode or "").strip().lower()
    if not key:
        return None
    return AUTO_MODE_BUILTIN_TARGET_LOOKUP.get(key)


def _auto_goal(base_data: dict | None, default: str = AUTO_MODE_GOAL_DEFAULT) -> str:
    g = str((base_data or {}).get("auto_goal", default) or default).strip().lower()
    return str(_auto_goal_norm(g))


def _auto_goal_basis_text(goal: str) -> str:
    return "preset_objective_score"


def _auto_metric_text(metrics: dict | None, goal: str) -> str:
    m = dict(metrics or {})
    rank = _auto_safe_float(m.get("rank_score_official", float("nan")), float("nan"))
    if not np.isfinite(rank):
        rank = _auto_safe_float(m.get("rank_score"), 0.0)
    return f"rank={float(rank):.3f}"


def _m(metrics: dict | None, key: str, default=float("nan")) -> float:
    try:
        v = float((metrics or {}).get(key, default))
    except (
        AttributeError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        RuntimeError,
        OSError,
        ImportError,
        ModuleNotFoundError,
        NameError,
    ):
        v = float(default)
    return float(v)


__all__ = [
    '_auto_goal_norm',
    '_auto_bass_integration_profile_norm',
    '_auto_bass_integration_profile_weights',
    '_auto_goal_forced_level_window',
    '_auto_builtin_target_name',
    '_auto_goal',
    '_auto_goal_basis_text',
    '_auto_metric_text',
    '_m',
]

