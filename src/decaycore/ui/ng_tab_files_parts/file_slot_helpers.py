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

import hashlib
import logging
import os
from typing import Any, Callable

logger = logging.getLogger("DecayCore")

from ...io.measurements_loader_parts.lr_measurement_loader import (
    _try_load_harmonic_sidecar,
    _try_load_rt60_sidecar,
)
from ...ui_i18n import normalize_layout_value

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


def _normalize_layout_value(value: Any, t: Callable[[str], str] | None = None) -> str:
    return normalize_layout_value(value, t)


def _guess_upload_format(file_data: dict[str, Any] | None) -> str:
    if not isinstance(file_data, dict):
        return "Unknown"
    name = str(file_data.get("filename", "") or "").strip().lower()
    content = file_data.get("content", b"")
    if name.endswith(".wav") or (
        isinstance(content, (bytes, bytearray)) and len(content) >= 4 and bytes(content[:4]) == b"RIFF"
    ):
        return "WAV"
    if name.endswith(".txt"):
        return "TXT"
    return "Unknown"


def _normalize_local_path_value(value: Any) -> str:
    try:
        return str(value or "").strip().strip('"').strip("'")
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
        return ""


def _describe_local_path(path_raw: Any) -> dict[str, Any]:
    path = _normalize_local_path_value(path_raw)
    if not path:
        return {
            "entered": False,
            "exists": False,
            "path": "",
            "filename": "",
            "format": "Unknown",
            "size_bytes": 0,
        }

    try:
        exists = bool(os.path.isfile(path))
    except OSError:
        exists = False

    size_bytes = 0
    if exists:
        try:
            size_bytes = int(os.path.getsize(path))
        except OSError:
            size_bytes = 0
    has_harmonics = False
    rt60_val = None
    if exists and path.lower().endswith(".wav"):
        harmonic_freq_hz, harmonic_magnitudes_db = _try_load_harmonic_sidecar(path)
        has_harmonics = bool(harmonic_freq_hz is not None and harmonic_magnitudes_db)
        rt60_val, _rt60_bands = _try_load_rt60_sidecar(path)

    return {
        "entered": True,
        "exists": exists,
        "path": path,
        "filename": os.path.basename(path) or path,
        "format": "WAV" if path.lower().endswith(".wav") else ("TXT" if path.lower().endswith(".txt") else "Unknown"),
        "size_bytes": size_bytes,
        "has_harmonics": has_harmonics,
        "rt60_val": rt60_val,
    }


def _format_upload_size(size_bytes: Any) -> str:
    try:
        size = float(size_bytes)
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
        size = 0.0
    if size <= 0:
        return "0 KB"
    if size >= 1024.0 * 1024.0:
        return f"{size / (1024.0 * 1024.0):.2f} MB"
    return f"{size / 1024.0:.1f} KB"


def _build_upload_payload(*, filename: str, content: bytes, mime_type: str = "") -> dict[str, Any]:
    content_bytes = bytes(content or b"")
    return {
        "filename": filename,
        "content": content_bytes,
        "mime_type": str(mime_type or ""),
        "size_bytes": len(content_bytes),
        "content_sha256": hashlib.sha256(content_bytes).hexdigest(),
    }


def _file_slot_scope_name(upload_key: str, slot_variant: str) -> str:
    return f"{upload_key}_status_scope__{slot_variant}"


def _file_slot_input_name(path_key: str, slot_variant: str) -> str:
    return f"{path_key}__{slot_variant}"


__all__ = [
    "_normalize_layout_value",
    "_guess_upload_format",
    "_normalize_local_path_value",
    "_describe_local_path",
    "_format_upload_size",
    "_build_upload_payload",
    "_file_slot_scope_name",
    "_file_slot_input_name",
]
