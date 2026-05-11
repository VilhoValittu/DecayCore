# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Auto Search v2 cache adapters."""

from __future__ import annotations

import time

from .. import api as auto_api
from ..shared import logger
from .input_model import AutoSearchInput

AUTO_SEARCH_CACHE_SCHEMA_VERSION = 2


def _valid_record(record: dict | None, *, signature: str | None, measurement_identity: str | None) -> tuple[bool, str]:
    if not isinstance(record, dict) or not record:
        return False, "missing cache record"
    schema = int(record.get("schema_version", record.get("v", 0)) or 0)
    if schema <= 0:
        return False, "missing cache schema"
    if signature and str(record.get("signature", signature) or signature) != str(signature):
        return False, "cache signature mismatch"
    preset = record.get("winner_preset", record.get("best_preset", {}))
    if not isinstance(preset, dict) or not preset:
        return False, "cache record missing winner preset"
    if measurement_identity:
        record_msig = str(record.get("measurement_identity", record.get("measurement_sig", "")) or "")
        if record_msig and record_msig != str(measurement_identity):
            return False, "cache measurement mismatch"
    return True, "ok"


def read_exact_cache_with_reason(
    *,
    signature: str,
    search_input: AutoSearchInput,
    filter_key: str,
    compat_version: str,
) -> tuple[dict | None, str]:
    try:
        record = auto_api._auto_cache_get_entry(
            signature,
            filter_key=filter_key,
            compat_version=compat_version,
        )
    except Exception as exc:
        reason = f"exact cache read failed: {type(exc).__name__}"
        logger.info("Automatic mode search v2: %s", reason, exc_info=True)
        return None, reason
    ok, reason = _valid_record(
        record,
        signature=str(signature),
        measurement_identity=str(search_input.measurement_identity),
    )
    if not ok:
        if isinstance(record, dict) and record:
            logger.info("Automatic mode search v2: exact cache invalidated: %s", str(reason))
        return None, str(reason)
    out = dict(record or {})
    out.setdefault("signature", str(signature))
    out.setdefault("measurement_identity", str(search_input.measurement_identity))
    out.setdefault("winner_preset", dict(out.get("best_preset", {}) or {}))
    out.setdefault("winner_metrics", dict(out.get("best_metrics", {}) or {}))
    return out, "ok"


def read_exact_cache(
    *,
    signature: str,
    search_input: AutoSearchInput,
    filter_key: str,
    compat_version: str,
) -> dict | None:
    record, _reason = read_exact_cache_with_reason(
        signature=signature,
        search_input=search_input,
        filter_key=filter_key,
        compat_version=compat_version,
    )
    return record


def read_last_used_best_with_reason(
    *,
    search_input: AutoSearchInput,
    goal: str,
    filter_key: str,
    compat_version: str,
) -> tuple[dict | None, str]:
    try:
        record = auto_api._auto_cache_get_last_used_best(
            goal=goal,
            filter_key=filter_key,
            compat_version=compat_version,
        )
    except Exception as exc:
        reason = f"last-best cache read failed: {type(exc).__name__}"
        logger.info("Automatic mode search v2: %s", reason, exc_info=True)
        return None, reason
    ok, reason = _valid_record(
        record,
        signature=None,
        measurement_identity=str(search_input.measurement_identity),
    )
    if not ok:
        if isinstance(record, dict) and record:
            logger.info("Automatic mode search v2: last-best cache invalidated: %s", str(reason))
        return None, str(reason)
    out = dict(record or {})
    out.setdefault("measurement_identity", str(search_input.measurement_identity))
    out.setdefault("winner_preset", dict(out.get("best_preset", {}) or {}))
    out.setdefault("winner_metrics", dict(out.get("best_metrics", {}) or {}))
    return out, "ok"


def read_last_used_best(
    *,
    search_input: AutoSearchInput,
    goal: str,
    filter_key: str,
    compat_version: str,
) -> dict | None:
    record, _reason = read_last_used_best_with_reason(
        search_input=search_input,
        goal=goal,
        filter_key=filter_key,
        compat_version=compat_version,
    )
    return record


