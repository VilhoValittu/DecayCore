# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Auto-mode last-used-best cache operations."""

from __future__ import annotations

import time

from .cache_io import _auto_cache_guard, _auto_cache_load, _auto_cache_save
from .cache_structure import _auto_cache_bucket
from .shared_parts import (
    AUTO_MODE_CACHE_SCHEMA_VERSION,
    AUTO_MODE_GOAL_DEFAULT,
    _auto_filter_cache_key,
    _auto_goal_norm,
    _auto_safe_float,
)


@_auto_cache_guard
def _auto_cache_get_last_used_best(
    *,
    goal: str = AUTO_MODE_GOAL_DEFAULT,
    filter_key: str | None = None,
    compat_version: str | None = None,
) -> dict | None:
    goal_norm = _auto_goal_norm(goal)
    cache = _auto_cache_load(compat_version=compat_version)
    bucket = _auto_cache_bucket(cache, filter_key=filter_key, create=False)
    last_map = {}
    if isinstance(bucket, dict):
        raw_last_map = bucket.get("last_used_best", {})
        if isinstance(raw_last_map, dict):
            last_map = raw_last_map
    if isinstance(last_map, dict):
        direct = last_map.get(str(goal_norm))
        if isinstance(direct, dict):
            return dict(direct)
        if len(last_map) > 0:
            return None
    legacy_map = cache.get("last_used_best", {})
    if not isinstance(legacy_map, dict):
        return None
    direct = legacy_map.get(str(goal_norm))
    return dict(direct) if isinstance(direct, dict) else None


@_auto_cache_guard
def _auto_cache_put_last_used_best(
    *,
    best_preset: dict,
    best_metrics: dict | None = None,
    best_hc_mode: str | None = None,
    measurement_sig: str | None = None,
    goal: str = AUTO_MODE_GOAL_DEFAULT,
    filter_key: str | None = None,
    compat_version: str | None = None,
) -> None:
    if not isinstance(best_preset, dict) or not best_preset:
        return
    goal_norm = _auto_goal_norm(goal)
    cache = _auto_cache_load(compat_version=compat_version)
    bucket = _auto_cache_bucket(cache, filter_key=filter_key, create=True)
    if not isinstance(bucket, dict):
        return
    last_map = bucket.get("last_used_best", {})
    if not isinstance(last_map, dict):
        last_map = {}
    entry = {
        "t": int(time.time()),
        "schema_version": int(AUTO_MODE_CACHE_SCHEMA_VERSION),
        "auto_goal": str(goal_norm),
        "filter_key": str(_auto_filter_cache_key(filter_type=filter_key)),
        "best_preset": dict(best_preset or {}),
        "best_metrics": dict(best_metrics or {}),
        "best_rank": float(_auto_safe_float((best_metrics or {}).get("rank_score", float("nan")), float("nan"))),
    }
    hc_val = str(best_hc_mode or "").strip()
    if hc_val:
        entry["best_target_curve"] = hc_val
        entry["best_hc_mode"] = hc_val
    msig = str(measurement_sig or "").strip()
    if msig:
        entry["measurement_sig"] = msig
        entry["measurement_identity"] = msig
    last_map[str(goal_norm)] = entry
    bucket["last_used_best"] = last_map
    cache["v"] = int(AUTO_MODE_CACHE_SCHEMA_VERSION)
    cache["schema_version"] = int(AUTO_MODE_CACHE_SCHEMA_VERSION)
    _auto_cache_save(cache, compat_version=compat_version)
