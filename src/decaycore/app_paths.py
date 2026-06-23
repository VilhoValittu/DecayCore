# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

import logging
import os
import re
import sys
from pathlib import Path

logger = logging.getLogger("DecayCore")

PROGRAM_NAME = "DecayCore"
_RECOVERABLE_PATH_EXCEPTIONS = (AttributeError, TypeError, ValueError, OSError, RuntimeError)
_RECOVERABLE_STR_EXCEPTIONS = (AttributeError, TypeError, ValueError, RuntimeError)


def decaycore_data_dir() -> Path:
    """Return platform-correct user data directory for DecayCore."""
    home = Path.home()
    try:
        if os.name == "nt":
            appdata = str(os.environ.get("APPDATA", "") or "").strip()
            if appdata:
                return Path(appdata) / PROGRAM_NAME
            return home / "AppData" / "Roaming" / PROGRAM_NAME
        if sys.platform == "darwin":
            return home / "Library" / "Application Support" / PROGRAM_NAME
        xdg_data_home = str(os.environ.get("XDG_DATA_HOME", "") or "").strip()
        if xdg_data_home:
            return Path(xdg_data_home) / PROGRAM_NAME
        return home / ".local" / "share" / PROGRAM_NAME
    except _RECOVERABLE_PATH_EXCEPTIONS:
        return home / PROGRAM_NAME


camillafir_data_dir = decaycore_data_dir


def decaycore_config_path() -> Path:
    """Return platform-correct path for config.json."""
    home = Path.home()
    try:
        if os.name == "nt":
            appdata = str(os.environ.get("APPDATA", "") or "").strip()
            base = Path(appdata) if appdata else home / "AppData" / "Roaming"
            return base / PROGRAM_NAME / "config.json"
        if sys.platform == "darwin":
            return home / "Library" / "Application Support" / PROGRAM_NAME / "config.json"
        xdg_cfg = str(os.environ.get("XDG_CONFIG_HOME", "") or "").strip()
        base = Path(xdg_cfg) if xdg_cfg else home / ".config"
        return base / PROGRAM_NAME / "config.json"
    except _RECOVERABLE_PATH_EXCEPTIONS:
        return home / PROGRAM_NAME / "config.json"


def default_filters_export_base_dir() -> Path:
    """Return default base directory for exported filters.

    Prefer Documents on all platforms for discoverability.
    """
    home = Path.home()
    try:
        if os.name == "nt":
            userprofile = str(os.environ.get("USERPROFILE", "") or "").strip()
            base_home = Path(userprofile) if userprofile else home
        else:
            base_home = home
        return base_home / "Documents" / PROGRAM_NAME / "filters"
    except _RECOVERABLE_PATH_EXCEPTIONS:
        return home / "Documents" / PROGRAM_NAME / "filters"


def default_measurements_dir() -> Path:
    """Return default directory for saved measurements."""
    home = Path.home()
    try:
        if os.name == "nt":
            userprofile = str(os.environ.get("USERPROFILE", "") or "").strip()
            base_home = Path(userprofile) if userprofile else home
        else:
            base_home = home
        return base_home / "Documents" / PROGRAM_NAME / "measurement"
    except _RECOVERABLE_PATH_EXCEPTIONS:
        return home / "Documents" / PROGRAM_NAME / "measurement"


def safe_measurements_dir(path: str | None, *, fallback: Path | None = None) -> str:
    """Resolve a writable measurement directory with conservative fallbacks.

    Measurement saves prefer Documents for discoverability, but macOS privacy
    prompts or platform-specific folder policies can make that location
    unavailable. In that case we fall back to the app data directory and then
    to a local working-directory directory.
    """
    fallback_base = fallback or (decaycore_data_dir() / "measurement")
    try:
        p_base = Path(path).expanduser() if path else default_measurements_dir()
    except _RECOVERABLE_PATH_EXCEPTIONS:
        p_base = default_measurements_dir()

    try:
        if p_base.is_absolute() and len(p_base.parts) <= 2:
            p_base = default_measurements_dir()
    except _RECOVERABLE_PATH_EXCEPTIONS:
        p_base = default_measurements_dir()

    try:
        p_base.mkdir(parents=True, exist_ok=True)
        return str(p_base.resolve())
    except OSError:
        pass
    except _RECOVERABLE_PATH_EXCEPTIONS:
        pass

    try:
        fallback_base.mkdir(parents=True, exist_ok=True)
        return str(fallback_base.resolve())
    except _RECOVERABLE_PATH_EXCEPTIONS:
        last = Path.cwd() / "measurement"
        try:
            last.mkdir(parents=True, exist_ok=True)
        except _RECOVERABLE_PATH_EXCEPTIONS:
            logger.exception("fallback measurement dir create")
        return str(last)


def program_version_token(version: str | None, *, default: str = "v0") -> str:
    """Create filesystem-safe version token for folder/file names."""
    try:
        raw = str(version or "").strip()
    except _RECOVERABLE_STR_EXCEPTIONS:
        raw = ""
    if not raw:
        return str(default)
    if raw.lower().startswith("v."):
        raw = "v" + raw[2:]
    raw = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip(".-_")
    return str(raw or default)


def safe_filters_dir(path: str | None, *, program_version: str | None = None) -> str:
    """Resolve a writable export filters directory with robust fallbacks.

    Guards:
    - avoid top-level/root paths such as `/filters`
    - fall back to platform user data directory if target is not writable
    """
    version_tag = program_version_token(program_version, default="v0")
    fallback_base = default_filters_export_base_dir()
    fallback = fallback_base / version_tag
    try:
        p_base = Path(path).expanduser() if path else fallback_base
    except _RECOVERABLE_PATH_EXCEPTIONS:
        p_base = fallback_base

    try:
        if p_base.is_absolute() and len(p_base.parts) <= 2:
            p_base = fallback_base
    except _RECOVERABLE_PATH_EXCEPTIONS:
        p_base = fallback_base

    try:
        if str(getattr(p_base, "name", "")).strip() != str(version_tag):
            p = p_base / version_tag
        else:
            p = p_base
    except _RECOVERABLE_PATH_EXCEPTIONS:
        p = fallback

    try:
        p.mkdir(parents=True, exist_ok=True)
        return str(p.resolve())
    except OSError:
        p = fallback
    except _RECOVERABLE_PATH_EXCEPTIONS:
        p = fallback

    try:
        p.mkdir(parents=True, exist_ok=True)
        return str(p.resolve())
    except _RECOVERABLE_PATH_EXCEPTIONS:
        last = Path.cwd() / "filters" / version_tag
        try:
            last.mkdir(parents=True, exist_ok=True)
        except _RECOVERABLE_PATH_EXCEPTIONS:
            logger.exception("fallback filter dir create")
        return str(last)
