# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""NiceGUI Files tab builder.

Replaces build_input_section() from layout_builders.py.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from .file_slot_helpers import (
    _normalize_local_path_value,
)
from .measurement_token_matching import (
    _measurement_entry_mtime_ns,
    _measurement_hint_tokens,
    _token_has_numeric_suffix,
    _token_is_leftish,
    _token_is_rightish,
    _token_is_sub1ish,
    _token_is_sub2ish,
    _token_is_subish,
)

logger = logging.getLogger("DecayCore")

from ...config.decaycore_config import load_config, save_config
from ...app_paths import default_measurements_dir

_MEASUREMENT_LIBRARY_EXTENSIONS = frozenset({".txt", ".wav"})
_MEASUREMENT_SLOT_UPLOAD_KEYS = {
    "local_path_l": "file_l",
    "local_path_r": "file_r",
    "local_path_l_main": "file_l_main",
    "local_path_r_main": "file_r_main",
    "local_path_l_sub": "file_l_sub",
    "local_path_r_sub": "file_r_sub",
}


_SLOT_FILTER_THRESHOLD = -50.0
_SUB_SLOT_KEYS = frozenset({"local_path_l_sub", "local_path_r_sub"})
_SUB_FILENAME_PREFIXES = ("sub", "lfe", "sw")


def _persist_measurement_library_dir(value: Any) -> str:
    persisted_value = _normalize_local_path_value(value) or str(default_measurements_dir())
    try:
        cfg = dict(load_config() or {})
        cfg["measurement_library_dir"] = persisted_value
        save_config(cfg)
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
        logger.exception("measurement library dir persist failed")
    return persisted_value


def _score_left_slot(
    tokens: list[str],
    *,
    slot_key: str,
    scale: float,
    first: str,
    last: str,
    has_main: bool,
    has_rightish: bool,
    has_subish: bool,
) -> float:
    score = 0.0
    if "left" in tokens:
        score += 180.0 * scale
    if any(token in {"fl", "frontleft", "frontl"} or _token_has_numeric_suffix(token, "fl") for token in tokens):
        score += 140.0 * scale
    if "lmain" in tokens:
        score += 120.0 * scale
    if first == "l" or last == "l":
        score += 90.0 * scale
    if slot_key == "local_path_l_main" and has_main:
        score += 45.0 * scale
    if slot_key == "local_path_l" and has_main:
        score -= 10.0 * scale
    if has_rightish:
        score -= 160.0 * scale
    if has_subish:
        score -= 150.0 * scale
    return score


def _score_right_slot(
    tokens: list[str],
    *,
    slot_key: str,
    scale: float,
    first: str,
    last: str,
    has_main: bool,
    has_leftish: bool,
    has_subish: bool,
) -> float:
    score = 0.0
    if "right" in tokens:
        score += 180.0 * scale
    if any(token in {"fr", "frontright", "frontr"} or _token_has_numeric_suffix(token, "fr") for token in tokens):
        score += 140.0 * scale
    if "rmain" in tokens:
        score += 120.0 * scale
    if first == "r" or last == "r":
        score += 90.0 * scale
    if slot_key == "local_path_r_main" and has_main:
        score += 45.0 * scale
    if slot_key == "local_path_r" and has_main:
        score -= 10.0 * scale
    if has_leftish:
        score -= 160.0 * scale
    if has_subish:
        score -= 150.0 * scale
    return score


def _score_left_sub_slot(
    *,
    scale: float,
    first: str,
    last: str,
    has_main: bool,
    has_subish: bool,
    has_sub1ish: bool,
    has_sub2ish: bool,
    has_left_token: bool,
    has_right_token: bool,
) -> float:
    score = 0.0
    if has_subish:
        score += 140.0 * scale
    if has_sub1ish:
        score += 120.0 * scale
    if has_left_token or first == "l" or last == "l":
        score += 35.0 * scale
    if has_sub2ish and not has_sub1ish:
        score -= 200.0 * scale
    if has_right_token or first == "r" or last == "r":
        score -= 30.0 * scale
    if has_main:
        score -= 110.0 * scale
    return score


def _score_right_sub_slot(
    *,
    scale: float,
    first: str,
    last: str,
    has_main: bool,
    has_subish: bool,
    has_sub1ish: bool,
    has_sub2ish: bool,
    has_left_token: bool,
    has_right_token: bool,
) -> float:
    score = 0.0
    if has_subish:
        score += 140.0 * scale
    if has_sub2ish:
        score += 120.0 * scale
    if has_right_token or first == "r" or last == "r":
        score += 35.0 * scale
    if has_sub1ish and not has_sub2ish:
        score -= 200.0 * scale
    if has_left_token or first == "l" or last == "l":
        score -= 30.0 * scale
    if has_main:
        score -= 110.0 * scale
    return score


def _score_measurement_tokens(tokens: list[str], slot_key: str, *, scale: float) -> float:
    if not tokens:
        return 0.0

    first = tokens[0]
    last = tokens[-1]
    has_main = any(token in {"main", "mainonly", "mains"} for token in tokens)
    has_leftish = any(_token_is_leftish(token) for token in tokens)
    has_rightish = any(_token_is_rightish(token) for token in tokens)
    has_subish = any(_token_is_subish(token) for token in tokens)
    has_sub1ish = any(_token_is_sub1ish(token) for token in tokens)
    has_sub2ish = any(_token_is_sub2ish(token) for token in tokens)
    has_left_token = "left" in tokens
    has_right_token = "right" in tokens

    if slot_key in {"local_path_l", "local_path_l_main"}:
        return _score_left_slot(
            tokens,
            slot_key=slot_key,
            scale=scale,
            first=first,
            last=last,
            has_main=has_main,
            has_rightish=has_rightish,
            has_subish=has_subish,
        )

    if slot_key in {"local_path_r", "local_path_r_main"}:
        return _score_right_slot(
            tokens,
            slot_key=slot_key,
            scale=scale,
            first=first,
            last=last,
            has_main=has_main,
            has_leftish=has_leftish,
            has_subish=has_subish,
        )

    if slot_key == "local_path_l_sub":
        return _score_left_sub_slot(
            scale=scale,
            first=first,
            last=last,
            has_main=has_main,
            has_subish=has_subish,
            has_sub1ish=has_sub1ish,
            has_sub2ish=has_sub2ish,
            has_left_token=has_left_token,
            has_right_token=has_right_token,
        )

    if slot_key == "local_path_r_sub":
        return _score_right_sub_slot(
            scale=scale,
            first=first,
            last=last,
            has_main=has_main,
            has_subish=has_subish,
            has_sub1ish=has_sub1ish,
            has_sub2ish=has_sub2ish,
            has_left_token=has_left_token,
            has_right_token=has_right_token,
        )

    return 0.0


def _score_measurement_candidate(entry: dict[str, Any], slot_key: str) -> float:
    name_tokens = list(entry.get("name_tokens") or [])
    path_tokens = list(entry.get("path_tokens") or [])
    return _score_measurement_tokens(name_tokens, slot_key, scale=1.0) + _score_measurement_tokens(
        path_tokens, slot_key, scale=0.65
    )


def _scan_measurement_library(path_raw: Any) -> list[dict[str, Any]]:
    root = _normalize_local_path_value(path_raw)
    if not root:
        return []
    try:
        if not os.path.isdir(root):
            return []
    except OSError:
        return []

    root_abs = os.path.abspath(root)
    _POSITION_DIR_RE = re.compile(r"^position_\d+$", re.IGNORECASE)
    entries: list[dict[str, Any]] = []
    for current_root, dirs, files in os.walk(root_abs):
        dirs[:] = [d for d in dirs if not _POSITION_DIR_RE.match(d)]
        for filename in sorted(files):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in _MEASUREMENT_LIBRARY_EXTENSIONS:
                continue
            full_path = os.path.abspath(os.path.join(current_root, filename))
            try:
                stat_result = os.stat(full_path)
                mtime_ns = int(getattr(stat_result, "st_mtime_ns", int(stat_result.st_mtime * 1_000_000_000)))
            except OSError:
                mtime_ns = 0
            relative_path = os.path.relpath(full_path, root_abs).replace("\\", "/")
            entries.append(
                {
                    "path": full_path,
                    "format": ext[1:].upper(),
                    "display_label": f"{relative_path} [{ext[1:].upper()}]",
                    "mtime_ns": mtime_ns,
                    "name_tokens": _measurement_hint_tokens(os.path.splitext(filename)[0]),
                    "path_tokens": _measurement_hint_tokens(relative_path),
                }
            )
    entries.sort(key=lambda entry: (-_measurement_entry_mtime_ns(entry), str(entry.get("display_label", "")).lower()))
    return entries


def _build_measurement_library_options(entries: list[dict[str, Any]]) -> dict[str, str]:
    return {
        str(entry.get("path", "")): str(entry.get("display_label", "") or entry.get("path", ""))
        for entry in entries
        if str(entry.get("path", "")).strip()
    }


def _entry_passes_slot_filter(entry: dict[str, Any], slot_key: str) -> bool:
    """Kova whitelist-tarkistus sub-sloteille.

    Sub-sloteissa näytetään vain tiedostot joiden ensimmäinen nimitokeni
    alkaa jollakin sub-etuliitteellä (sub, lfe, sw). Muissa sloteissa
    kaikki tiedostot läpäisevät tämän tarkistuksen.
    """
    if slot_key not in _SUB_SLOT_KEYS:
        return True
    name_tokens = entry.get("name_tokens") or []
    if not name_tokens:
        return False
    return any(name_tokens[0].startswith(prefix) for prefix in _SUB_FILENAME_PREFIXES)


def _build_slot_options(entries: list[dict[str, Any]], slot_key: str) -> dict[str, str]:
    """Palauttaa vain slotin kannalta relevantit tiedostot.

    Sub-sloteissa sovelletaan ensin whitelist-ehtoa (tiedoston nimi alkaa
    sub/lfe/sw), sen jälkeen scoring-kynnnystä väärän kanavan karsimiseksi.
    Muissa sloteissa vain scoring-kynnys.
    """
    filtered = [
        entry
        for entry in entries
        if _entry_passes_slot_filter(entry, slot_key)
        and _score_measurement_candidate(entry, slot_key) >= _SLOT_FILTER_THRESHOLD
    ]
    return _build_measurement_library_options(filtered)


def _suggest_measurement_library_matches(
    entries: list[dict[str, Any]],
    *,
    path_keys: list[str],
    min_score: float = 80.0,
) -> dict[str, str]:
    suggestions: dict[str, str] = {}
    used_paths: set[str] = set()

    for path_key in path_keys:
        ranked = sorted(
            (
                (
                    _score_measurement_candidate(entry, path_key),
                    _measurement_entry_mtime_ns(entry),
                    str(entry.get("path", "")),
                )
                for entry in entries
                if str(entry.get("path", "")).strip()
            ),
            key=lambda item: (-float(item[0]), -int(item[1]), item[2].lower()),
        )
        for score, _mtime_ns, candidate_path in ranked:
            if score < float(min_score) or candidate_path in used_paths:
                continue
            suggestions[path_key] = candidate_path
            used_paths.add(candidate_path)
            break
    return suggestions


def _build_measurement_library_slot_options(
    entries: list[dict[str, Any]],
    *,
    path_keys: list[str],
) -> dict[str, dict[str, str]]:
    return {path_key: _build_slot_options(entries, path_key) for path_key in path_keys}


def _build_measurement_library_state(
    entries: list[dict[str, Any]],
    *,
    path_keys: list[str],
) -> dict[str, Any]:
    entries_list = list(entries)
    return {
        "entries": entries_list,
        "options": _build_measurement_library_options(entries_list),
        "slot_options": _build_measurement_library_slot_options(entries_list, path_keys=path_keys),
    }


def _build_measurement_library_refresh_payload(
    path_raw: Any,
    *,
    path_keys: list[str],
) -> dict[str, Any]:
    dir_value = _normalize_local_path_value(path_raw)
    exists = False
    if dir_value:
        try:
            exists = os.path.isdir(dir_value)
        except OSError:
            exists = False
    entries = _scan_measurement_library(dir_value) if exists else []
    payload = _build_measurement_library_state(entries, path_keys=path_keys)
    payload["dir_value"] = dir_value
    payload["exists"] = exists
    return payload


def _measurement_library_refresh_payload_for_token(
    payload: dict[str, Any],
    *,
    token: int,
    current_token: int,
) -> dict[str, Any] | None:
    if int(token) != int(current_token):
        return None
    payload_copy = dict(payload)
    payload_copy["token"] = int(token)
    return payload_copy


def _measurement_library_status_key(
    *,
    dir_value: str,
    exists: bool,
    entry_count: int,
    is_scanning: bool,
) -> str:
    if is_scanning:
        return "measurement_library_status_scanning"
    if not dir_value:
        return "measurement_library_status_idle"
    if not exists:
        return "measurement_library_status_missing"
    if entry_count:
        return "measurement_library_status_found"
    return "measurement_library_status_empty"


def _suggest_measurement_library_matches_if_ready(
    entries: list[dict[str, Any]],
    *,
    path_keys: list[str],
    is_scanning: bool,
    min_score: float = 80.0,
) -> dict[str, str]:
    if is_scanning:
        return {}
    return _suggest_measurement_library_matches(entries, path_keys=path_keys, min_score=min_score)
