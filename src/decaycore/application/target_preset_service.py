# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""User-defined target curve presets: save, list, load, delete.

A preset is a snapshot of the currently displayed target curve, saved under
a user-chosen name so it can be reselected later from the hc_mode dropdown
alongside the built-in house curves (Harman, Toole, BK, ...).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from threading import Lock

import numpy as np

from ..app_paths import decaycore_data_dir
from ..common.house_curves import USER_PRESET_MODE_PREFIX

logger = logging.getLogger("DecayCore")

_PRESET_LOCK = Lock()
_RECOVERABLE_IO_EXCEPTIONS = (
    RuntimeError,
    OSError,
    ImportError,
    TypeError,
    ValueError,
    AttributeError,
    KeyError,
    IndexError,
    OverflowError,
    FloatingPointError,
)


def _presets_dir() -> str:
    path = os.path.join(str(decaycore_data_dir()), "target_presets")
    os.makedirs(path, exist_ok=True)
    return path


def _slugify(name: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return base or "preset"


def _unique_slug(name: str, *, directory: str) -> str:
    base = _slugify(name)
    slug = base
    n = 2
    while os.path.exists(os.path.join(directory, f"{slug}.json")):
        slug = f"{base}_{n}"
        n += 1
    return slug


def mode_key_for_slug(slug: str) -> str:
    return f"{USER_PRESET_MODE_PREFIX}{slug}"


def _slug_from_mode_key(mode_key) -> str | None:
    if not isinstance(mode_key, str) or not mode_key.startswith(USER_PRESET_MODE_PREFIX):
        return None
    slug = mode_key[len(USER_PRESET_MODE_PREFIX) :].strip()
    return slug or None


def preset_name_exists(name: str) -> bool:
    """Case-insensitive check used by the UI before saving, so a duplicate
    name can be rejected with an explicit error instead of silently
    creating a second, differently-slugged preset with the same label."""
    clean_name = str(name or "").strip().lower()
    if not clean_name:
        return False
    return any(p["name"].strip().lower() == clean_name for p in list_user_target_presets())


def save_user_target_preset(name: str, frequency_hz, magnitude_db) -> str | None:
    """Persist *frequency_hz*/*magnitude_db* as a named preset.

    Returns the new hc_mode key on success, or None if the name is empty or
    the curve is invalid. Callers should check preset_name_exists() first to
    surface a duplicate-name error explicitly rather than relying on this
    function's slug-collision suffixing (which only guards against two
    different names slugifying to the same filesystem id).
    """
    clean_name = str(name or "").strip()
    if not clean_name:
        return None
    try:
        freqs = np.asarray(frequency_hz, dtype=float).reshape(-1)
        mags = np.asarray(magnitude_db, dtype=float).reshape(-1)
    except _RECOVERABLE_IO_EXCEPTIONS:
        return None
    valid = np.isfinite(freqs) & np.isfinite(mags) & (freqs > 0.0)
    freqs = freqs[valid]
    mags = mags[valid]
    if freqs.size < 2 or mags.size != freqs.size:
        return None

    directory = _presets_dir()
    payload = {
        "name": clean_name,
        "frequency_hz": freqs.tolist(),
        "magnitude_db": mags.tolist(),
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with _PRESET_LOCK:
        slug = _unique_slug(clean_name, directory=directory)
        path = os.path.join(directory, f"{slug}.json")
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            os.replace(tmp, path)
        except _RECOVERABLE_IO_EXCEPTIONS:
            logger.debug("save_user_target_preset(%r) failed", clean_name, exc_info=True)
            return None
    return mode_key_for_slug(slug)


def list_user_target_presets() -> list[dict]:
    """Return saved presets as [{"mode_key", "name", "saved_at"}, ...], sorted by name."""
    directory = _presets_dir()
    try:
        filenames = sorted(os.listdir(directory))
    except _RECOVERABLE_IO_EXCEPTIONS:
        return []

    presets = []
    for filename in filenames:
        if not filename.endswith(".json"):
            continue
        slug = filename[: -len(".json")]
        path = os.path.join(directory, filename)
        try:
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
            name = str(payload.get("name") or slug)
            saved_at = str(payload.get("saved_at") or "")
        except _RECOVERABLE_IO_EXCEPTIONS:
            logger.debug("list_user_target_presets: skipping unreadable %r", filename, exc_info=True)
            continue
        presets.append({"mode_key": mode_key_for_slug(slug), "name": name, "saved_at": saved_at})

    presets.sort(key=lambda p: p["name"].lower())
    return presets


def load_user_target_preset(mode_key):
    """Return (frequency_hz, magnitude_db) for *mode_key*, or (None, None)."""
    slug = _slug_from_mode_key(mode_key)
    if slug is None:
        return None, None
    path = os.path.join(_presets_dir(), f"{slug}.json")
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        freqs = np.asarray(payload["frequency_hz"], dtype=float).reshape(-1)
        mags = np.asarray(payload["magnitude_db"], dtype=float).reshape(-1)
    except _RECOVERABLE_IO_EXCEPTIONS:
        logger.debug("load_user_target_preset(%r) failed", mode_key, exc_info=True)
        return None, None
    if freqs.size < 2 or mags.size != freqs.size:
        return None, None
    return freqs, mags


def delete_user_target_preset(mode_key) -> bool:
    """Delete the preset file for *mode_key*. Returns True if a file was removed."""
    slug = _slug_from_mode_key(mode_key)
    if slug is None:
        return False
    path = os.path.join(_presets_dir(), f"{slug}.json")
    with _PRESET_LOCK:
        try:
            os.remove(path)
            return True
        except FileNotFoundError:
            return False
        except _RECOVERABLE_IO_EXCEPTIONS:
            logger.debug("delete_user_target_preset(%r) failed", mode_key, exc_info=True)
            return False
