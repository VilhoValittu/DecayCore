# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

import hashlib
import logging

import numpy as np

from ._constants import *

logger = logging.getLogger("DecayCore")

def _auto_filter_cache_key(
    base_data: dict | None = None, *, filter_type: str | None = None
) -> str:
    ft = (
        str(
            filter_type
            if filter_type is not None
            else (base_data or {}).get("filter_type", "") or ""
        )
        .strip()
        .lower()
    )
    if ft in AUTO_MODE_CACHE_FILTER_KEYS:
        return str(ft)
    if "asym" in ft:
        return "asym"
    if "mixed" in ft:
        return "mixed"
    if "minimum" in ft or "minphase" in ft or ("min" in ft and "phase" in ft):
        return "minimum"
    if "linear" in ft:
        return "linear"
    return "mixed"


def _auto_filter_type_for_key(filter_key: str | None) -> str:
    fk = str(_auto_filter_cache_key(filter_type=str(filter_key or "")))
    if fk == "asym":
        return "Asymmetric"
    if fk == "linear":
        return "Linear Phase"
    if fk == "minimum":
        return "Minimum Phase"
    return "Mixed"

def _auto_hash_array(a: np.ndarray, *, decimals: int = 4, max_len: int = 1200) -> str:
    try:
        x = np.asarray(a, dtype=float).reshape(-1)
    except Exception:
        return ""
    if x.size <= 0:
        return ""
    m = np.isfinite(x)
    x = x[m]
    if x.size <= 0:
        return ""
    if x.size > int(max_len):
        idx = np.linspace(0, x.size - 1, int(max_len)).astype(int)
        x = x[idx]
    x = np.round(x, int(decimals))
    b = x.astype(np.float32).tobytes()
    return hashlib.sha256(b).hexdigest()

def _auto_hash_array_full(a: np.ndarray) -> str:
    try:
        raw = np.asarray(a)
    except Exception:
        return ""
    if raw.size <= 0:
        return ""
    try:
        x = raw.reshape(-1)
        mask = np.isfinite(x)
        x = x[mask]
    except Exception:
        return ""
    if x.size <= 0:
        return ""
    h = hashlib.sha256()
    if np.iscomplexobj(x):
        canonical = np.ascontiguousarray(x.astype(np.dtype("<c16"), copy=False))
        dtype_name = "complex128"
    else:
        try:
            canonical = np.ascontiguousarray(x.astype(np.dtype("<f8"), copy=False))
        except Exception:
            return ""
        dtype_name = "float64"
    h.update(str(dtype_name).encode("ascii", "ignore"))
    h.update(str(tuple(raw.shape)).encode("ascii", "ignore"))
    h.update(str(int(canonical.size)).encode("ascii", "ignore"))
    h.update(canonical.tobytes())
    return h.hexdigest()


__all__ = ['_auto_filter_cache_key', '_auto_hash_array', '_auto_hash_array_full']


def _load_sibling_symbols() -> None:
    import importlib
    package = __package__
    for module_name in ['cache_hash', 'goal_profile', 'safe_values', 'backend', 'config', 'phase_sampling']:
        if module_name == __name__.rsplit('.', 1)[-1]:
            continue
        module = importlib.import_module(f"{package}.{module_name}")
        for symbol in getattr(module, "__all__", ()):
            globals().setdefault(symbol, getattr(module, symbol))


_load_sibling_symbols()
