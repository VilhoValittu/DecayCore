# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

import os
import re

DEFAULT_VERSION = "v1.2.1"


def normalize_version(value: str | None, *, default: str = DEFAULT_VERSION) -> str:
    """Normalize runtime/build version text to the UI/export format `v.X.Y.Z`."""
    try:
        raw = str(value or "").strip()
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
        raw = ""
    if not raw:
        return str(default)

    if raw.lower().startswith("v."):
        body = raw[2:]
    elif raw.lower().startswith("v"):
        body = raw[1:]
    else:
        body = raw
    body = re.sub(r"[^0-9A-Za-z._-]+", "-", str(body or "")).strip(".-_")
    if not body:
        return str(default)
    return f"v.{body}"


def resolve_version(*, default: str = DEFAULT_VERSION) -> str:
    env_version = str(
        os.environ.get("DECAYCORE_VERSION", os.environ.get("CAMILLAFIR_VERSION", ""))
        or ""
    ).strip()
    if env_version:
        return normalize_version(env_version, default=default)

    try:
        from .build_version import VERSION as build_version

        return normalize_version(build_version, default=default)
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
        return normalize_version(default, default=default)


VERSION = resolve_version()
