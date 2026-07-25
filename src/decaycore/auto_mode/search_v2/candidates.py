# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Candidate seed utilities for Auto Search v2."""

from __future__ import annotations

import json


def preset_identity(preset: dict | None) -> str:
    payload = {
        str(k): v
        for k, v in dict(preset or {}).items()
        if not str(k).startswith("_") and str(k) not in {"preset_id", "selection_method"}
    }
    try:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
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
    ):
        return str(sorted(payload.items()))


def deduplicate_presets(presets: list[dict] | tuple[dict, ...] | None) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for preset in list(presets or []):
        if not isinstance(preset, dict) or not preset:
            continue
        key = preset_identity(preset)
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(preset))
    return out
