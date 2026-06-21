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
import re
from typing import Any

logger = logging.getLogger("DecayCore")



















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

def _measurement_hint_tokens(value: Any) -> list[str]:
    try:
        text = str(value or "").strip().lower().replace("\\", "/")
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
        return []
    if not text:
        return []
    return [token for token in re.split(r"[^a-z0-9]+", text) if token]

def _measurement_entry_mtime_ns(entry: dict[str, Any]) -> int:
    try:
        return int(entry.get("mtime_ns") or 0)
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
        return 0

def _token_has_numeric_suffix(token: str, prefix: str) -> bool:
    if not token.startswith(prefix):
        return False
    suffix = token[len(prefix):]
    return bool(suffix and suffix.isdigit())

def _token_is_leftish(token: str) -> bool:
    token_l = str(token or "").strip().lower()
    return token_l in {"left", "fl", "frontleft", "frontl", "lmain"} or _token_has_numeric_suffix(token_l, "fl")

def _token_is_rightish(token: str) -> bool:
    token_l = str(token or "").strip().lower()
    return token_l in {"right", "fr", "frontright", "frontr", "rmain"} or _token_has_numeric_suffix(token_l, "fr")

def _token_is_subish(token: str) -> bool:
    token_l = str(token or "").strip().lower()
    return (
        token_l in {"sub", "subwoofer", "sw", "lfe", "sub1", "sub2", "sw1", "sw2", "lfe1", "lfe2"}
        or token_l.startswith("sub")
        or token_l.startswith("sw")
        or token_l.startswith("lfe")
    )

def _token_is_sub1ish(token: str) -> bool:
    token_l = str(token or "").strip().lower()
    return (
        token_l in {"sub1", "sw1", "lfe1"}
        or token_l.startswith("sub1")
        or token_l.startswith("sw1")
        or token_l.startswith("lfe1")
    )

def _token_is_sub2ish(token: str) -> bool:
    token_l = str(token or "").strip().lower()
    return (
        token_l in {"sub2", "sw2", "lfe2"}
        or token_l.startswith("sub2")
        or token_l.startswith("sw2")
        or token_l.startswith("lfe2")
    )


__all__ = ['_measurement_hint_tokens', '_measurement_entry_mtime_ns', '_token_has_numeric_suffix', '_token_is_leftish', '_token_is_rightish', '_token_is_subish', '_token_is_sub1ish', '_token_is_sub2ish']


def _link_sibling_exports() -> None:
    import importlib
    package = __package__
    for module_name in ['file_slot_helpers', 'measurement_token_matching', 'files_tab_builder']:
        if module_name == __name__.rsplit('.', 1)[-1]:
            continue
        module = importlib.import_module(f"{package}.{module_name}")
        for symbol in getattr(module, "__all__", ()):
            globals().setdefault(symbol, getattr(module, symbol))


_link_sibling_exports()
