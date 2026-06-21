# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Auto-mode cache I/O: global state, guard decorator, load and save."""

from __future__ import annotations

import logging
import json
import os
from functools import wraps
from threading import RLock

from .cache_structure import _auto_cache_bucket_template, _auto_cache_empty
from .shared import (
    AUTO_MODE_CACHE_FILTER_KEYS,
    AUTO_MODE_CACHE_SCHEMA_VERSION,
    AUTO_MODE_COMPAT_VERSION,
)

logger = logging.getLogger("DecayCore")

_AUTO_CACHE_LOCK = RLock()
_AUTO_CACHE_RUNTIME: dict[str, dict] = {}
_AUTO_CACHE_STATS = {
    "loads": 0,
    "load_hits": 0,
    "entry_hits": 0,
    "entry_misses": 0,
    "saves": 0,
    "save_failures": 0,
    "last_save_error": "",
    "last_save_error_path": "",
}

_RECOVERABLE_CACHE_IO_EXCEPTIONS = (
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
)


def _auto_cache_guard(func):
    @wraps(func)
    def _wrapped(*args, **kwargs):
        with _AUTO_CACHE_LOCK:
            return func(*args, **kwargs)

    return _wrapped


def _auto_cache_stats_snapshot() -> dict:
    with _AUTO_CACHE_LOCK:
        return dict(_AUTO_CACHE_STATS)


def _auto_cache_set_schema_version(cache_obj: dict) -> None:
    try:
        cache_obj["v"] = int(max(int(AUTO_MODE_CACHE_SCHEMA_VERSION), int(cache_obj.get("v", 0) or 0)))
    except _RECOVERABLE_CACHE_IO_EXCEPTIONS:
        cache_obj["v"] = int(AUTO_MODE_CACHE_SCHEMA_VERSION)
    cache_obj["schema_version"] = int(AUTO_MODE_CACHE_SCHEMA_VERSION)


def _auto_cache_normalize_filter_buckets(cache_obj: dict) -> None:
    by_filter = cache_obj.get("by_filter", {})
    if not isinstance(by_filter, dict):
        by_filter = {}
    for k in AUTO_MODE_CACHE_FILTER_KEYS:
        if not isinstance(by_filter.get(k), dict):
            by_filter[k] = _auto_cache_bucket_template()
            continue
        if not isinstance(by_filter[k].get("items", {}), dict):
            by_filter[k]["items"] = {}
        if not isinstance(by_filter[k].get("target_by_measurement", {}), dict):
            by_filter[k]["target_by_measurement"] = {}
        if not isinstance(by_filter[k].get("last_used_best", {}), dict):
            by_filter[k]["last_used_best"] = {}
    cache_obj["by_filter"] = by_filter


def _auto_cache_apply_version_fields(
    cache_obj: dict,
    *,
    compat_version: str | None,
    program_version: str | None,
) -> None:
    ver = str(
        compat_version
        or cache_obj.get("auto_mode_compat_version", "")
        or AUTO_MODE_COMPAT_VERSION
    ).strip()
    if ver:
        cache_obj["auto_mode_compat_version"] = str(ver)
    prog_ver = str(program_version or cache_obj.get("program_version", "") or "").strip()
    if prog_ver:
        cache_obj["program_version"] = prog_ver


def _auto_cache_prepare_for_save(
    cache: dict,
    *,
    compat_version: str | None,
    program_version: str | None,
) -> dict:
    cache_obj = dict(cache or {})
    _auto_cache_set_schema_version(cache_obj)
    _auto_cache_normalize_filter_buckets(cache_obj)
    _auto_cache_apply_version_fields(
        cache_obj,
        compat_version=compat_version,
        program_version=program_version,
    )
    return cache_obj


def _auto_cache_record_save_failure(*, path: str, exc: Exception) -> None:
    reason = f"{type(exc).__name__}: {exc}"
    _AUTO_CACHE_STATS["save_failures"] = int(_AUTO_CACHE_STATS.get("save_failures", 0) or 0) + 1
    _AUTO_CACHE_STATS["last_save_error"] = reason
    _AUTO_CACHE_STATS["last_save_error_path"] = str(path)
    logger.debug("Automatic mode cache save failed: path=%s reason=%s", path, reason)


@_auto_cache_guard
def _auto_cache_load(
    *,
    compat_version: str | None = None,
    program_version: str | None = None,
) -> dict:
    from . import cache_signature as _cs
    with _AUTO_CACHE_LOCK:
        path = _cs._auto_cache_resolve_path(compat_version=compat_version)
        runtime_key = os.path.abspath(path)
        cached = _AUTO_CACHE_RUNTIME.get(runtime_key)
        if isinstance(cached, dict):
            _AUTO_CACHE_STATS["load_hits"] = int(_AUTO_CACHE_STATS.get("load_hits", 0) or 0) + 1
            return cached
        _AUTO_CACHE_STATS["loads"] = int(_AUTO_CACHE_STATS.get("loads", 0) or 0) + 1
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if not isinstance(obj, dict):
                obj = {}
            expected_ver = str(compat_version or AUTO_MODE_COMPAT_VERSION).strip()
            actual_ver = str(obj.get("auto_mode_compat_version", "") or "").strip()
            if expected_ver and actual_ver and actual_ver != expected_ver:
                obj = _auto_cache_empty(
                    compat_version=expected_ver,
                    program_version=program_version,
                )
                _AUTO_CACHE_RUNTIME[runtime_key] = obj
                return obj
            if expected_ver and not actual_ver:
                obj["auto_mode_compat_version"] = str(expected_ver)
            _AUTO_CACHE_RUNTIME[runtime_key] = obj
            return obj
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
            expected_ver = str(compat_version or AUTO_MODE_COMPAT_VERSION).strip()
            if expected_ver:
                obj = _auto_cache_empty(
                    compat_version=expected_ver,
                    program_version=program_version,
                )
                _AUTO_CACHE_RUNTIME[runtime_key] = obj
                return obj
            obj = {}
            _AUTO_CACHE_RUNTIME[runtime_key] = obj
            return obj


@_auto_cache_guard
def _auto_cache_save(
    cache: dict,
    *,
    compat_version: str | None = None,
    program_version: str | None = None,
) -> None:
    from . import cache_signature as _cs
    with _AUTO_CACHE_LOCK:
        path = _cs._auto_cache_resolve_path(compat_version=compat_version)
        try:
            cache_obj = _auto_cache_prepare_for_save(
                cache,
                compat_version=compat_version,
                program_version=program_version,
            )
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(cache_obj, f, indent=2, sort_keys=True)
            os.replace(tmp, path)
            _AUTO_CACHE_RUNTIME[os.path.abspath(path)] = cache_obj
            _AUTO_CACHE_STATS["saves"] = int(_AUTO_CACHE_STATS.get("saves", 0) or 0) + 1
            _AUTO_CACHE_STATS["last_save_error"] = ""
            _AUTO_CACHE_STATS["last_save_error_path"] = ""
        except _RECOVERABLE_CACHE_IO_EXCEPTIONS as exc:
            _auto_cache_record_save_failure(path=str(path), exc=exc)
            return
