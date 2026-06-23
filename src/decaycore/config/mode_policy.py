# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

from __future__ import annotations

import logging
from typing import Any
from decaycore.config.models import FilterConfig
from decaycore.config.schema import MODE_CLAMPS as SCHEMA_MODE_CLAMPS
from decaycore.config.schema import MODE_DEFAULTS as SCHEMA_MODE_DEFAULTS

logger = logging.getLogger("DecayCore")


def _clamp_float(v, lo: float, hi: float) -> float:
    try:
        x = float(v)
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
        return float(lo)
    if x < lo:
        return float(lo)
    if x > hi:
        return float(hi)
    return float(x)


def _apply_defaults(cfg: FilterConfig, d: dict[str, Any]) -> None:
    for k, v in d.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)


def _apply_clamps(cfg: FilterConfig, clamps: dict[str, tuple[Any, Any]]) -> None:
    for k, lim in clamps.items():
        if not hasattr(cfg, k):
            continue

        lo, hi = lim

        if isinstance(lo, bool) and isinstance(hi, bool):
            setattr(cfg, k, bool(lo))
            continue

        if lo == hi:
            setattr(cfg, k, lo)
            continue

        try:
            cur = getattr(cfg, k)
            setattr(cfg, k, _clamp_float(cur, float(lo), float(hi)))
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
            logger.exception("mode clamp apply")


MODE_DEFAULTS: dict[str, dict[str, Any]] = {key: dict(value) for key, value in SCHEMA_MODE_DEFAULTS.items()}
MODE_CLAMPS: dict[str, dict[str, tuple[Any, Any]]] = {key: dict(value) for key, value in SCHEMA_MODE_CLAMPS.items()}


def apply_mode_to_cfg(cfg: FilterConfig, mode: str | None, *, apply_defaults: bool = True) -> FilterConfig:
    """Soveltaa tai paivittaa: apply mode to cfg."""
    m = (mode or "BASIC").upper().strip()
    if m not in MODE_DEFAULTS:
        m = "BASIC"

    if apply_defaults:
        _apply_defaults(cfg, MODE_DEFAULTS[m])
    _apply_clamps(cfg, MODE_CLAMPS.get(m, {}))
    return cfg
