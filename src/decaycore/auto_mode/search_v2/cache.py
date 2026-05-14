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
from types import SimpleNamespace

AUTO_SEARCH_CACHE_SCHEMA_VERSION = 2

from ..cache_signature import _auto_cache_get_entry, _auto_cache_get_last_used_best
from ..shared import _auto_filter_cache_key
from ..shared import logger
from .input_model import AutoSearchInput

auto_api = SimpleNamespace(
    _auto_cache_get_entry=_auto_cache_get_entry,
    _auto_cache_get_last_used_best=_auto_cache_get_last_used_best,
)


_FIRST_RUN_STAGES = frozenset(("target_search", "phase1", "phase2", "phase3"))


def _record_preset(record: dict | None) -> dict:
    if not isinstance(record, dict):
        return {}
    preset = record.get("winner_preset", record.get("best_preset", {}))
    return dict(preset or {}) if isinstance(preset, dict) and preset else {}


def _record_measurement_matches(record: dict, measurement_identity: str | None) -> tuple[bool, str]:
    if not measurement_identity:
        return True, "ok"
    record_msig = str(record.get("measurement_identity", record.get("measurement_sig", "")) or "")
    if record_msig and record_msig != str(measurement_identity):
        return False, "cache measurement mismatch"
    return True, "ok"


def _record_filter_matches(record: dict, filter_key: str | None) -> tuple[bool, str]:
    wanted = str(_auto_filter_cache_key(filter_type=filter_key) or "").strip()
    if not wanted:
        return True, "ok"
    raw = str(record.get("filter_key", record.get("_optuna_filter_key", record.get("filter_type", ""))) or "").strip()
    if not raw:
        return False, "cache filter missing"
    got = str(_auto_filter_cache_key(filter_type=raw) or "").strip()
    if got != wanted:
        return False, "cache filter mismatch"
    return True, "ok"


def _valid_completed_record(
    record: dict | None,
    *,
    signature: str | None,
    measurement_identity: str | None,
    filter_key: str | None,
    compat_version: str | None,
) -> tuple[bool, str]:
    if not isinstance(record, dict) or not record:
        return False, "missing cache record"
    schema = int(record.get("schema_version", record.get("v", 0)) or 0)
    if schema <= 0:
        return False, "missing cache schema"
    if signature and str(record.get("signature", signature) or signature) != str(signature):
        return False, "cache signature mismatch"
    if not _record_preset(record):
        return False, "cache record missing winner preset"
    ok, reason = _record_measurement_matches(record, measurement_identity)
    if not ok:
        return False, reason
    ok, reason = _record_filter_matches(record, filter_key)
    if not ok:
        return False, reason
    record_compat = str(record.get("compat_version", "") or "").strip()
    if str(compat_version or "").strip() and record_compat != str(compat_version or "").strip():
        return False, "cache compat mismatch"
    if record.get("first_run_complete") is not True:
        return False, "cache first-run completion missing"
    completed = {str(item) for item in list(record.get("completed_stages", []) or [])}
    if not _FIRST_RUN_STAGES.issubset(completed):
        return False, "cache completed stages incomplete"
    return True, "ok"


def _seedable_record(
    record: dict | None,
    *,
    signature: str | None,
    measurement_identity: str | None,
) -> tuple[bool, str]:
    if not isinstance(record, dict) or not record:
        return False, "missing cache record"
    schema = int(record.get("schema_version", record.get("v", 0)) or 0)
    if schema <= 0:
        return False, "missing cache schema"
    if signature and str(record.get("signature", signature) or signature) != str(signature):
        return False, "cache signature mismatch"
    if not _record_preset(record):
        return False, "cache record missing winner preset"
    ok, reason = _record_measurement_matches(record, measurement_identity)
    if not ok:
        return False, reason
    return True, "seed_only"


def _normalize_record(record: dict, *, measurement_identity: str, signature: str | None = None) -> dict:
    out = dict(record or {})
    if signature:
        out.setdefault("signature", str(signature))
    out.setdefault("measurement_identity", str(measurement_identity))
    out.setdefault("winner_preset", _record_preset(out))
    out.setdefault("winner_metrics", dict(out.get("best_metrics", {}) or {}))
    return out


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
    ok, reason = _valid_completed_record(
        record,
        signature=str(signature),
        measurement_identity=str(search_input.measurement_identity),
        filter_key=filter_key,
        compat_version=compat_version,
    )
    if not ok:
        if isinstance(record, dict) and record:
            logger.info("Automatic mode search v2: exact cache invalidated: %s", str(reason))
        return None, str(reason)
    return _normalize_record(
        dict(record or {}),
        signature=str(signature),
        measurement_identity=str(search_input.measurement_identity),
    ), "ok"


def read_exact_cache_seed_with_reason(
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
        reason = f"exact cache seed read failed: {type(exc).__name__}"
        logger.info("Automatic mode search v2: %s", reason, exc_info=True)
        return None, reason
    ok, reason = _seedable_record(
        record,
        signature=str(signature),
        measurement_identity=str(search_input.measurement_identity),
    )
    if not ok:
        return None, str(reason)
    return _normalize_record(
        dict(record or {}),
        signature=str(signature),
        measurement_identity=str(search_input.measurement_identity),
    ), "seed_only"


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
    ok, reason = _seedable_record(
        record,
        signature=None,
        measurement_identity=str(search_input.measurement_identity),
    )
    if not ok:
        if isinstance(record, dict) and record:
            logger.info("Automatic mode search v2: last-best cache invalidated: %s", str(reason))
        return None, str(reason)
    return _normalize_record(
        dict(record or {}),
        measurement_identity=str(search_input.measurement_identity),
    ), "ok"


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
