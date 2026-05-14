# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Auto-mode cache get/put operations."""

from __future__ import annotations

import time

from .cache_io import _AUTO_CACHE_STATS, _auto_cache_guard, _auto_cache_load, _auto_cache_save
from .cache_measurement_sig import _auto_get_measurement_signature
from .cache_structure import _auto_cache_bucket
from .shared import (
    AUTO_MODE_CACHE_MAX_ITEMS,
    AUTO_MODE_CACHE_SCHEMA_VERSION,
    AUTO_MODE_GOAL_DEFAULT,
    _auto_builtin_target_name,
    _auto_filter_cache_key,
    _auto_goal_norm,
    _auto_safe_float,
    logger,
)


@_auto_cache_guard
def _auto_cache_get_entry(
    sig: str,
    *,
    filter_key: str | None = None,
    compat_version: str | None = None,
) -> dict | None:
    if not sig:
        return None
    cache = _auto_cache_load(compat_version=compat_version)
    bucket = _auto_cache_bucket(cache, filter_key=filter_key, create=False)
    bucket_items = {}
    if isinstance(bucket, dict):
        raw_bucket_items = bucket.get("items", {})
        if isinstance(raw_bucket_items, dict):
            bucket_items = raw_bucket_items
    if isinstance(bucket_items, dict):
        entry = bucket_items.get(sig)
        if isinstance(entry, dict):
            _AUTO_CACHE_STATS["entry_hits"] = int(_AUTO_CACHE_STATS.get("entry_hits", 0) or 0) + 1
            return dict(entry)
        if len(bucket_items) > 0:
            _AUTO_CACHE_STATS["entry_misses"] = int(_AUTO_CACHE_STATS.get("entry_misses", 0) or 0) + 1
            return None
    items = cache.get("items", {})
    if isinstance(items, dict):
        entry = items.get(sig)
        if isinstance(entry, dict):
            _AUTO_CACHE_STATS["entry_hits"] = int(_AUTO_CACHE_STATS.get("entry_hits", 0) or 0) + 1
        else:
            _AUTO_CACHE_STATS["entry_misses"] = int(_AUTO_CACHE_STATS.get("entry_misses", 0) or 0) + 1
        return dict(entry) if isinstance(entry, dict) else None
    _AUTO_CACHE_STATS["entry_misses"] = int(_AUTO_CACHE_STATS.get("entry_misses", 0) or 0) + 1
    return None


def _auto_cache_get_best(
    sig: str,
    *,
    filter_key: str | None = None,
    compat_version: str | None = None,
) -> dict | None:
    entry = _auto_cache_get_entry(sig, filter_key=filter_key, compat_version=compat_version)
    if not isinstance(entry, dict):
        return None
    best = entry.get("best_preset", {})
    return dict(best) if isinstance(best, dict) else None


def _auto_cache_get_best_target(
    sig: str,
    *,
    filter_key: str | None = None,
    compat_version: str | None = None,
) -> str | None:
    entry = _auto_cache_get_entry(sig, filter_key=filter_key, compat_version=compat_version)
    if not isinstance(entry, dict):
        return None
    hc = str(entry.get("best_target_curve", entry.get("best_hc_mode", "")) or "").strip()
    return _auto_builtin_target_name(hc)


def _auto_target_measurement_cache_key(measurements: dict, goal: str) -> str:
    msig = _auto_get_measurement_signature(measurements or {})
    if not msig:
        return ""
    goal_norm = _auto_goal_norm(goal)
    return f"{msig}|{goal_norm}|target-v2"


@_auto_cache_guard
def _auto_cache_get_target_for_measurements_global(
    measurements: dict,
    *,
    goal: str = AUTO_MODE_GOAL_DEFAULT,
    compat_version: str | None = None,
) -> dict | None:
    """Return measurement-global target cache entry only.

    Important:
    This function must not fall back to filter-specific or legacy target caches.
    It has no filter_key argument, so it cannot safely decide whether a cached
    target/seed belongs to the currently selected filter type.

    Caller must validate that the returned global entry contains a usable
    filter_seed_presets[filter_key] before using it to skip target search.
    """
    key = _auto_target_measurement_cache_key(measurements, goal)
    if not key:
        return None

    cache = _auto_cache_load(compat_version=compat_version)
    target_map = cache.get("target_by_measurement_global", {})
    if not isinstance(target_map, dict):
        return None

    direct = target_map.get(key)
    if not isinstance(direct, dict):
        return None

    goal_norm = _auto_goal_norm(goal)
    entry_goal = _auto_goal_norm(
        str(direct.get("auto_goal", AUTO_MODE_GOAL_DEFAULT) or AUTO_MODE_GOAL_DEFAULT)
    )

    if entry_goal != goal_norm:
        return None

    return dict(direct)


@_auto_cache_guard
def _auto_cache_put_target_for_measurements_global(
    *,
    measurements: dict,
    best_hc_mode: str | None,
    goal: str = AUTO_MODE_GOAL_DEFAULT,
    compat_version: str | None = None,
    target_selection_meta: dict | None = None,
    filter_seed_preset: dict | None = None,
    filter_seed_metrics: dict | None = None,
    filter_key: str | None = None,
) -> None:
    hc_val = str(best_hc_mode or "").strip()
    if not hc_val:
        return
    key = _auto_target_measurement_cache_key(measurements, goal)
    msig = _auto_get_measurement_signature(measurements or {})
    if not key or not msig:
        return
    cache = _auto_cache_load(compat_version=compat_version)
    target_map = cache.get("target_by_measurement_global", {})
    if not isinstance(target_map, dict):
        target_map = {}

    old = target_map.get(key)
    entry = dict(old or {}) if isinstance(old, dict) else {}
    entry.update(
        {
            "t": int(time.time()),
            "schema_version": int(AUTO_MODE_CACHE_SCHEMA_VERSION),
            "target_cache_version": 2,
            "measurement_sig": str(msig),
            "measurement_identity": str(msig),
            "auto_goal": str(_auto_goal_norm(goal)),
            "best_target_curve": hc_val,
            "best_hc_mode": hc_val,
            "target_selection_meta": dict(target_selection_meta or {}),
        }
    )

    seed_map = entry.get("filter_seed_presets", {})
    if not isinstance(seed_map, dict):
        seed_map = {}
    metric_map = entry.get("filter_seed_metrics", {})
    if not isinstance(metric_map, dict):
        metric_map = {}

    fk = _auto_filter_cache_key(filter_type=filter_key)
    if fk and isinstance(filter_seed_preset, dict) and filter_seed_preset:
        seed_map[str(fk)] = dict(filter_seed_preset)
    if fk and isinstance(filter_seed_metrics, dict) and filter_seed_metrics:
        metric_map[str(fk)] = dict(filter_seed_metrics)

    if seed_map:
        entry["filter_seed_presets"] = seed_map
    if metric_map:
        entry["filter_seed_metrics"] = metric_map

    target_map[str(key)] = entry
    try:
        if len(target_map) > int(AUTO_MODE_CACHE_MAX_ITEMS):
            sorted_items = sorted(
                target_map.items(),
                key=lambda kv: int((kv[1] or {}).get("t", 0) or 0),
                reverse=True,
            )
            target_map = dict(sorted_items[: int(AUTO_MODE_CACHE_MAX_ITEMS)])
    except Exception:
        logger.exception("global target_map eviction")
    cache["target_by_measurement_global"] = target_map
    cache["v"] = int(AUTO_MODE_CACHE_SCHEMA_VERSION)
    cache["schema_version"] = int(AUTO_MODE_CACHE_SCHEMA_VERSION)
    _auto_cache_save(cache, compat_version=compat_version)


@_auto_cache_guard
def _auto_cache_get_target_for_measurements(
    measurements: dict,
    *,
    goal: str = AUTO_MODE_GOAL_DEFAULT,
    filter_key: str | None = None,
    compat_version: str | None = None,
) -> dict | None:
    goal_norm = _auto_goal_norm(goal)
    msig = _auto_get_measurement_signature(measurements or {})
    if not msig:
        return None
    cache = _auto_cache_load(compat_version=compat_version)

    bucket = _auto_cache_bucket(cache, filter_key=filter_key, create=False)
    target_map = {}
    if isinstance(bucket, dict):
        raw_target_map = bucket.get("target_by_measurement", {})
        if isinstance(raw_target_map, dict):
            target_map = raw_target_map
    if isinstance(target_map, dict):
        direct = target_map.get(f"{msig}|{goal_norm}")
        if isinstance(direct, dict):
            return dict(direct)
        direct_legacy = target_map.get(msig)
        if isinstance(direct_legacy, dict):
            entry_goal = _auto_goal_norm(str(direct_legacy.get("auto_goal", AUTO_MODE_GOAL_DEFAULT) or AUTO_MODE_GOAL_DEFAULT))
            if entry_goal == goal_norm:
                return dict(direct_legacy)
        if len(target_map) > 0:
            return None

    target_map_legacy = cache.get("target_by_measurement", {})
    if isinstance(target_map_legacy, dict):
        direct = target_map_legacy.get(f"{msig}|{goal_norm}")
        if isinstance(direct, dict):
            return dict(direct)
        direct_legacy = target_map_legacy.get(msig)
        if isinstance(direct_legacy, dict):
            entry_goal = _auto_goal_norm(str(direct_legacy.get("auto_goal", AUTO_MODE_GOAL_DEFAULT) or AUTO_MODE_GOAL_DEFAULT))
            if entry_goal == goal_norm:
                return dict(direct_legacy)

    items = {}
    if isinstance(bucket, dict):
        raw_items = bucket.get("items", {})
        if isinstance(raw_items, dict):
            items = raw_items
    if not items:
        items = cache.get("items", {})
    if not isinstance(items, dict):
        return None
    best = None
    best_t = -1
    for entry in items.values():
        if not isinstance(entry, dict):
            continue
        if str(entry.get("measurement_sig", "") or "") != str(msig):
            continue
        entry_goal = _auto_goal_norm(str(entry.get("auto_goal", AUTO_MODE_GOAL_DEFAULT) or AUTO_MODE_GOAL_DEFAULT))
        if entry_goal != goal_norm:
            continue
        try:
            t = int(entry.get("t", 0) or 0)
        except Exception:
            t = 0
        if t >= best_t:
            best_t = int(t)
            best = dict(entry)
    return dict(best) if isinstance(best, dict) else None


@_auto_cache_guard
def _auto_cache_put_target_for_measurements(
    *,
    measurements: dict,
    best_hc_mode: str | None,
    best_preset: dict,
    target_seed_preset: dict | None = None,
    best_metrics: dict | None = None,
    goal: str = AUTO_MODE_GOAL_DEFAULT,
    filter_key: str | None = None,
    compat_version: str | None = None,
) -> None:
    hc_val = str(best_hc_mode or "").strip()
    if not hc_val:
        return
    msig = _auto_get_measurement_signature(measurements or {})
    if not msig:
        return
    cache = _auto_cache_load(compat_version=compat_version)
    bucket = _auto_cache_bucket(cache, filter_key=filter_key, create=True)
    if not isinstance(bucket, dict):
        return
    target_map = bucket.get("target_by_measurement", {})
    if not isinstance(target_map, dict):
        target_map = {}
    goal_norm = _auto_goal_norm(goal)
    scoped_key = f"{msig}|{goal_norm}"
    entry = {
        "t": int(time.time()),
        "schema_version": int(AUTO_MODE_CACHE_SCHEMA_VERSION),
        "measurement_sig": str(msig),
        "measurement_identity": str(msig),
        "auto_goal": str(goal_norm),
        "filter_key": str(_auto_filter_cache_key(filter_type=filter_key)),
        "best_target_curve": hc_val,
        "best_hc_mode": hc_val,
        "best_preset": dict(best_preset or {}),
        "best_metrics": dict(best_metrics or {}),
        "best_rank": float(_auto_safe_float((best_metrics or {}).get("rank_score", float("nan")), float("nan"))),
    }
    if isinstance(target_seed_preset, dict) and target_seed_preset:
        entry["target_seed_preset"] = dict(target_seed_preset)
    target_map[str(scoped_key)] = entry
    try:
        if len(target_map) > int(AUTO_MODE_CACHE_MAX_ITEMS):
            sorted_items = sorted(
                target_map.items(),
                key=lambda kv: int((kv[1] or {}).get("t", 0) or 0),
                reverse=True,
            )
            target_map = dict(sorted_items[: int(AUTO_MODE_CACHE_MAX_ITEMS)])
    except Exception:
        logger.exception("cache target_map eviction")
    bucket["target_by_measurement"] = target_map
    cache["v"] = int(AUTO_MODE_CACHE_SCHEMA_VERSION)
    cache["schema_version"] = int(AUTO_MODE_CACHE_SCHEMA_VERSION)
    _auto_cache_save(cache, compat_version=compat_version)


@_auto_cache_guard
def _auto_cache_put_best(
    sig: str,
    *,
    best_preset: dict,
    target_seed_preset: dict | None = None,
    best_metrics: dict | None = None,
    best_hc_mode: str | None = None,
    measurement_sig: str | None = None,
    goal: str = AUTO_MODE_GOAL_DEFAULT,
    filter_key: str | None = None,
    compat_version: str | None = None,
    first_run_complete: bool = True,
    completed_stages: list[str] | tuple[str, ...] | None = None,
) -> None:
    if not isinstance(best_preset, dict) or not best_preset:
        return
    sig = str(sig or "").strip()
    if not sig:
        return
    goal_norm = _auto_goal_norm(goal)
    cache = _auto_cache_load(compat_version=compat_version)
    bucket = _auto_cache_bucket(cache, filter_key=filter_key, create=True)
    if not isinstance(bucket, dict):
        return
    items = bucket.get("items", {})
    if not isinstance(items, dict):
        items = {}
    entry = {
        "t": int(time.time()),
        "schema_version": int(AUTO_MODE_CACHE_SCHEMA_VERSION),
        "signature": str(sig),
        "compat_version": str(compat_version or ""),
        "auto_goal": str(goal_norm),
        "filter_key": str(_auto_filter_cache_key(filter_type=filter_key)),
        "first_run_complete": bool(first_run_complete),
        "completed_stages": list(completed_stages or (["target_search", "phase1", "phase2", "phase3"] if bool(first_run_complete) else [])),
        "best_preset": dict(best_preset or {}),
        "best_metrics": dict(best_metrics or {}),
        "best_rank": float(_auto_safe_float((best_metrics or {}).get("rank_score", float("nan")), float("nan"))),
    }
    if isinstance(target_seed_preset, dict) and target_seed_preset:
        entry["target_seed_preset"] = dict(target_seed_preset)
    hc_val = str(best_hc_mode or "").strip()
    if hc_val:
        entry["best_target_curve"] = hc_val
        entry["best_hc_mode"] = hc_val
    msig = str(measurement_sig or "").strip()
    if msig:
        entry["measurement_sig"] = msig
        entry["measurement_identity"] = msig
    items[str(sig)] = entry
    try:
        if len(items) > int(AUTO_MODE_CACHE_MAX_ITEMS):
            sorted_items = sorted(
                items.items(),
                key=lambda kv: int((kv[1] or {}).get("t", 0) or 0),
                reverse=True,
            )
            items = dict(sorted_items[: int(AUTO_MODE_CACHE_MAX_ITEMS)])
    except Exception:
        logger.exception("cache items eviction")
    bucket["items"] = items
    cache["v"] = int(AUTO_MODE_CACHE_SCHEMA_VERSION)
    cache["schema_version"] = int(AUTO_MODE_CACHE_SCHEMA_VERSION)
    _auto_cache_save(cache, compat_version=compat_version)
