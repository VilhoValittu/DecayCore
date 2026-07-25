# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Auto-mode cache structure: compat version, bucket template, empty cache, bucket accessor."""

from __future__ import annotations

from .shared_parts import (
    AUTO_MODE_CACHE_FILTER_KEYS,
    AUTO_MODE_CACHE_SCHEMA_VERSION,
    AUTO_MODE_COMPAT_VERSION,
    _auto_filter_cache_key,
)


def _auto_compat_version(base_data: dict | None) -> str:
    try:
        raw = str(
            (base_data or {}).get("auto_mode_compat_version", AUTO_MODE_COMPAT_VERSION)
            or AUTO_MODE_COMPAT_VERSION
        ).strip()
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
    ):
        raw = str(AUTO_MODE_COMPAT_VERSION)
    return str(raw or AUTO_MODE_COMPAT_VERSION)


def _auto_cache_bucket_template() -> dict:
    return {
        "items": {},
        "target_by_measurement": {},
        "last_used_best": {},
    }


def _auto_cache_empty(
    *,
    compat_version: str | None = None,
    program_version: str | None = None,
) -> dict:
    out = {
        "v": int(AUTO_MODE_CACHE_SCHEMA_VERSION),
        "schema_version": int(AUTO_MODE_CACHE_SCHEMA_VERSION),
        "items": {},
        "target_by_measurement": {},
        "target_by_measurement_global": {},
        "by_filter": {},
    }
    ver = str(compat_version or AUTO_MODE_COMPAT_VERSION).strip()
    if ver:
        out["auto_mode_compat_version"] = str(ver)
    prog_ver = str(program_version or "").strip()
    if prog_ver:
        out["program_version"] = prog_ver
    for k in AUTO_MODE_CACHE_FILTER_KEYS:
        out["by_filter"][str(k)] = _auto_cache_bucket_template()
    return out


def _auto_cache_bucket(
    cache: dict,
    *,
    filter_key: str | None,
    create: bool = False,
) -> dict | None:
    if not isinstance(cache, dict):
        return None
    by_filter = cache.get("by_filter", {})
    if not isinstance(by_filter, dict):
        if not bool(create):
            return None
        by_filter = {}
        cache["by_filter"] = by_filter
    fk = _auto_filter_cache_key(filter_type=str(filter_key or ""))
    bucket = by_filter.get(fk)
    if not isinstance(bucket, dict):
        if not bool(create):
            return None
        bucket = _auto_cache_bucket_template()
        by_filter[fk] = bucket
    if bool(create):
        if not isinstance(bucket.get("items", {}), dict):
            bucket["items"] = {}
        if not isinstance(bucket.get("target_by_measurement", {}), dict):
            bucket["target_by_measurement"] = {}
        if not isinstance(bucket.get("last_used_best", {}), dict):
            bucket["last_used_best"] = {}
    return bucket
