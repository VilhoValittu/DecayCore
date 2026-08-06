# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Python boundary for the packaged-only automatic-mode decision engine."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from ..features import (
    PACKAGED_AUTO_ENGINE_MODULE,
    PACKAGED_AUTO_ENGINE_POLICY_VERSION,
    require_packaged_auto_engine,
)


def _load_engine():
    require_packaged_auto_engine()
    return importlib.import_module(PACKAGED_AUTO_ENGINE_MODULE)


def select_best_index(
    rank_keys: np.ndarray,
    hard_gate_failed: np.ndarray,
) -> dict[str, Any]:
    """Select a safe winner in Rust and validate the native result contract."""
    keys = np.ascontiguousarray(rank_keys, dtype=np.float64)
    gates = np.ascontiguousarray(hard_gate_failed, dtype=np.bool_)
    if keys.ndim != 2 or keys.shape[0] == 0 or keys.shape[1] == 0:
        raise ValueError("rank_keys must have shape (candidates, ranking_fields)")
    if gates.ndim != 1 or gates.shape[0] != keys.shape[0]:
        raise ValueError("hard_gate_failed must have one entry per candidate")

    engine = _load_engine()
    result = dict(engine.select_best_index_rs(keys, gates))
    policy_version = int(result.get("engine_policy_version", 0) or 0)
    if policy_version != int(PACKAGED_AUTO_ENGINE_POLICY_VERSION):
        raise RuntimeError(
            "Packaged automatic-mode engine returned an incompatible policy version "
            f"({policy_version}, expected {PACKAGED_AUTO_ENGINE_POLICY_VERSION})."
        )
    winner_index = int(result.get("winner_index", -1))
    if winner_index < 0 or winner_index >= keys.shape[0]:
        raise RuntimeError("Packaged automatic-mode engine returned an invalid winner index.")
    return result


__all__ = ["select_best_index"]
