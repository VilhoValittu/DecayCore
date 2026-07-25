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

import numpy as np

from ._constants import *

logger = logging.getLogger("DecayCore")

def _auto_safe_float(value, default=0.0) -> float:
    try:
        x = float(value)
        if np.isfinite(x):
            return float(x)
    except (TypeError, ValueError):
        return float(default)
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
        logger.exception("safe float parse")
    return float(default)

def _auto_safe_bool(value, default=False) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return bool(value)
    try:
        s = str(value or "").strip().lower()
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
        return bool(default)
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return bool(default)

def _auto_safe_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
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
        return int(default)

def _auto_output_tilt_bounds(base_data: dict | None) -> tuple[float, float]:
    try:
        auto_target_mode = (
            str((base_data or {}).get("auto_target_mode", "") or "").strip().lower()
        )
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
        auto_target_mode = ""
    if auto_target_mode == "adaptive":
        return 0.0, 2.0
    return -2.0, 2.0

def _clip(v, lo, hi):
    vlo = _auto_safe_float(lo, 0.0)
    vhi = _auto_safe_float(hi, vlo)
    if vhi < vlo:
        vlo, vhi = vhi, vlo
    return float(np.clip(_auto_safe_float(v, vlo), vlo, vhi))


__all__ = [
    '_auto_safe_float',
    '_auto_safe_bool',
    '_auto_safe_int',
    '_auto_output_tilt_bounds',
    '_clip',
]
