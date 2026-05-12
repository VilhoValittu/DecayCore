# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Optuna backend — storage, study creation, and signature tracking."""

from __future__ import annotations

import hashlib
import logging
import os
import re

from ..app_paths import decaycore_data_dir, program_version_token
from .cache_signature import _auto_compat_version
from .shared import (
    AUTO_MODE_OPTUNA_STORAGE_FILENAME,
    _auto_filter_cache_key,
    _auto_safe_bool,
)

logger = logging.getLogger("DecayCore")

# Per-study known-signature sets for the duplicate guard in _auto_optuna_remember_result.
# Avoids O(N²) full trial scan: prime once per study, then check/update a set.
_OPTUNA_KNOWN_SIGNATURES: dict[str, set[str]] = {}
_OPTUNA_KNOWN_SIGNATURES_PRIMED: set[str] = set()
_OPTUNA_KNOWN_RECORDS: dict[str, dict[str, dict]] = {}
_OPTUNA_CROSS_STUDY_BEST_PARAMS: dict[str, list[dict]] = {}
_OPTUNA_STUDY_SCAN_STATS = {
    "study_scans": 0,
    "study_trials_scanned": 0,
}


def _auto_optuna_get_known_signatures(study_name: str) -> set[str]:
    if study_name not in _OPTUNA_KNOWN_SIGNATURES:
        _OPTUNA_KNOWN_SIGNATURES[study_name] = set()
    return _OPTUNA_KNOWN_SIGNATURES[study_name]


def _auto_optuna_prime_known_signatures_from_study(
    study_name: str,
    study,
    *,
    seed_to_params=None,
) -> None:
    """Scan study trials once and populate the known-signature set."""
    if study_name in _OPTUNA_KNOWN_SIGNATURES_PRIMED:
        return
    try:
        from .optuna_backend_records import _auto_optuna_study_records
        records = _auto_optuna_study_records(study, seed_to_params=seed_to_params)
        sigs = _auto_optuna_get_known_signatures(study_name)
        sigs.update(records.keys())
        _OPTUNA_KNOWN_RECORDS[study_name] = dict(records or {})
    except Exception:
        logger.exception("optuna study signature prime")
    _OPTUNA_KNOWN_SIGNATURES_PRIMED.add(study_name)


def _auto_optuna_mark_signature_seen(study_name: str, sig: str) -> None:
    if sig:
        _auto_optuna_get_known_signatures(study_name).add(str(sig))


def _auto_optuna_cached_study_records(study_name: str | None) -> dict[str, dict] | None:
    key = str(study_name or "")
    if not key or key not in _OPTUNA_KNOWN_SIGNATURES_PRIMED:
        return None
    return dict(_OPTUNA_KNOWN_RECORDS.get(key, {}) or {})


def _auto_optuna_update_known_record(study_name: str | None, sig: str, rec: dict | None) -> None:
    key = str(study_name or "")
    if not key or not sig:
        return
    _auto_optuna_mark_signature_seen(key, str(sig))
    records = _OPTUNA_KNOWN_RECORDS.setdefault(key, {})
    records[str(sig)] = dict(rec or {})


def _auto_optuna_study_scan_stats_snapshot() -> dict:
    return dict(_OPTUNA_STUDY_SCAN_STATS)


def _auto_optuna_note_trial_scan(count: int) -> None:
    _OPTUNA_STUDY_SCAN_STATS["study_scans"] = int(_OPTUNA_STUDY_SCAN_STATS.get("study_scans", 0) or 0) + 1
    _OPTUNA_STUDY_SCAN_STATS["study_trials_scanned"] = (
        int(_OPTUNA_STUDY_SCAN_STATS.get("study_trials_scanned", 0) or 0)
        + int(max(0, count))
    )


def _auto_import_optuna():
    try:
        import optuna  # type: ignore
    except Exception:
        # Optional dependency: fall back to builtin search if Optuna is absent or unusable.
        logger.debug("Optuna not available; automatic mode will use builtin backend", exc_info=True)
        return None
    return optuna

def _auto_optuna_module_ready(optuna_mod) -> bool:
    if optuna_mod is None:
        return False
    try:
        sampler_cls = getattr(getattr(optuna_mod, "samplers", None), "TPESampler", None)
        create_study = getattr(optuna_mod, "create_study", None)
        trial_state = getattr(getattr(optuna_mod, "trial", None), "TrialState", None)
    except (AttributeError, TypeError):
        return False
    return bool(
        callable(sampler_cls)
        and callable(create_study)
        and trial_state is not None
        and hasattr(trial_state, "FAIL")
    )

_OPTUNA_VALID_FILTER_KEYS = frozenset(("asym", "linear", "minimum", "mixed"))
_OPTUNA_FILTER_FILENAME_LABELS = {
    "asym": "asymmetric",
    "linear": "linear",
    "minimum": "minimum",
    "mixed": "mixed",
}


def _auto_optuna_measurement_token(measurement_identity: str | None) -> str:
    txt = str(measurement_identity or "").strip().lower()
    if not txt:
        return "nomeasurement"
    return re.sub(r"[^a-z0-9]+", "", txt)[:16] or "measurement"


def _auto_optuna_storage_filename(
    *,
    compat_version: str | None = None,
    filter_key: str | None = None,
    measurement_identity: str | None = None,
    journal_kind: str | None = None,
) -> str:
    token = str(program_version_token(compat_version, default="") or "").strip()
    kind = str(journal_kind or "filter").strip().lower()
    fk = str(filter_key or "").strip().lower()
    _, ext = os.path.splitext(str(AUTO_MODE_OPTUNA_STORAGE_FILENAME))
    if kind == "target":
        parts = [
            "decaycore",
            "optuna",
            "target",
            _auto_optuna_measurement_token(measurement_identity),
        ]
    else:
        fk_label = _OPTUNA_FILTER_FILENAME_LABELS.get(fk, fk if fk in _OPTUNA_VALID_FILTER_KEYS else "unknown")
        parts = [
            "decaycore",
            "optuna",
            fk_label,
            _auto_optuna_measurement_token(measurement_identity),
        ]
    if token:
        parts.append(token)
    return "_".join(parts) + (ext or ".log")


def _auto_optuna_storage_path(
    *,
    compat_version: str | None = None,
    filter_key: str | None = None,
    measurement_identity: str | None = None,
    journal_kind: str | None = None,
) -> str:
    filename = _auto_optuna_storage_filename(
        compat_version=compat_version,
        filter_key=filter_key,
        measurement_identity=measurement_identity,
        journal_kind=journal_kind,
    )
    preferred_base = os.fspath(decaycore_data_dir())
    preferred_path = os.path.join(preferred_base, filename)
    legacy_base = os.path.join(os.path.expanduser("~"), ".camillafir")
    legacy_path = os.path.join(legacy_base, filename)

    try:
        os.makedirs(preferred_base, exist_ok=True)
    except OSError:
        try:
            os.makedirs(legacy_base, exist_ok=True)
        except OSError:
            logger.debug("Failed to create legacy Optuna storage directory", exc_info=True)
            pass
        logger.debug("Falling back to legacy Optuna storage directory", exc_info=True)
        return legacy_path
    try:
        source_candidates = [legacy_path]
        if str(filename) != str(AUTO_MODE_OPTUNA_STORAGE_FILENAME):
            source_candidates.extend(
                (
                    os.path.join(preferred_base, AUTO_MODE_OPTUNA_STORAGE_FILENAME),
                    os.path.join(legacy_base, AUTO_MODE_OPTUNA_STORAGE_FILENAME),
                )
            )
        source_path = next(
            (
                path
                for path in source_candidates
                if path != preferred_path and os.path.isfile(path)
            ),
            None,
        )
        if (not os.path.isfile(preferred_path)) and source_path:
            try:
                os.replace(source_path, preferred_path)
            except OSError:
                with open(source_path, "rb") as src_f:
                    payload = src_f.read()
                with open(preferred_path, "wb") as dst_f:
                    dst_f.write(payload)
                try:
                    os.remove(source_path)
                except OSError:
                    logger.debug("Failed to remove migrated Optuna storage source file", exc_info=True)
                    pass
            logger.info(f"Automatic mode Optuna storage migrated to: {preferred_path}")
    except OSError:
        logger.debug("Optuna storage setup failed; falling back to legacy path", exc_info=True)
        return legacy_path
    return preferred_path

def _auto_optuna_study_name(*, study_sig: str | None, scope: str | None) -> str:
    sig_txt = str(study_sig or "").strip().lower()
    scope_txt = str(scope or "study").strip().lower()
    scope_tok = re.sub(r"[^a-z0-9._-]+", "-", scope_txt).strip("-") or "study"
    scope_hash = hashlib.sha1(scope_txt.encode("utf-8", "ignore")).hexdigest()[:12]
    sig_tok = sig_txt[:32] if sig_txt else "nosig"
    return f"decaycore-{scope_tok[:48]}-{scope_hash}-{sig_tok}"

def _auto_optuna_create_storage(optuna_mod, *, base_data: dict | None):
    if not _auto_safe_bool((base_data or {}).get("auto_mode_optuna_persistent_study", True), True):
        return None
    storages_mod = getattr(optuna_mod, "storages", None)
    if storages_mod is None:
        return None
    explicit_fk = str((base_data or {}).get("_optuna_filter_key", "")).strip().lower()
    fk = explicit_fk if explicit_fk in _OPTUNA_VALID_FILTER_KEYS else _auto_filter_cache_key(base_data)
    measurement_identity = str((base_data or {}).get("_optuna_measurement_sig", "") or "")
    journal_kind = str((base_data or {}).get("_optuna_journal_kind", "filter") or "filter")
    path = _auto_optuna_storage_path(
        compat_version=_auto_compat_version(base_data),
        filter_key=str(fk),
        measurement_identity=measurement_identity,
        journal_kind=journal_kind,
    )
    candidates = [
        (
            getattr(getattr(storages_mod, "journal", None), "JournalStorage", None),
            getattr(getattr(storages_mod, "journal", None), "JournalFileBackend", None),
            getattr(getattr(storages_mod, "journal", None), "JournalFileOpenLock", None),
        ),
        (
            getattr(storages_mod, "JournalStorage", None),
            getattr(storages_mod, "JournalFileStorage", None),
            getattr(storages_mod, "JournalFileOpenLock", None),
        ),
        (
            getattr(getattr(storages_mod, "journal", None), "JournalStorage", None),
            getattr(getattr(storages_mod, "journal", None), "JournalFileBackend", None),
            None,
        ),
    ]
    for storage_cls, backend_cls, open_lock_cls in candidates:
        if not callable(storage_cls) or not callable(backend_cls):
            continue
        try:
            if callable(open_lock_cls):
                return storage_cls(backend_cls(path, lock_obj=open_lock_cls(path)))
            return storage_cls(backend_cls(path))
        except Exception:
            # Third-party Optuna journal backends differ across versions; try the next compatible variant.
            logger.debug("Optuna journal storage candidate initialization failed", exc_info=True)
            continue
    return None

def _auto_optuna_create_study(
    optuna_mod,
    *,
    sampler,
    pruner=None,
    base_data: dict | None,
    study_name: str | None,
):
    storage = _auto_optuna_create_storage(optuna_mod, base_data=base_data)
    create_kwargs = {"direction": "maximize", "sampler": sampler}
    if pruner is not None:
        create_kwargs["pruner"] = pruner
    if storage is not None and study_name:
        try:
            return optuna_mod.create_study(
                **create_kwargs,
                storage=storage,
                study_name=str(study_name),
                load_if_exists=True,
            )
        except TypeError:
            pass
        except Exception as exc:
            logger.warning(
                "Automatic mode Optuna storage unavailable for study %s (%s: %s). "
                "Falling back to in-memory study.",
                str(study_name),
                type(exc).__name__,
                exc,
            )
    return optuna_mod.create_study(**create_kwargs)
