# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""P6 — Final FIR impulse-domain validation before accepting auto-mode winner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

_EPS = 1e-12
_SEVERITY_ORDER: dict[str, int] = {"ok": 0, "warn": 1, "reject": 2}


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Per-channel temporal energy metrics
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# FIR spectrum computation (cached for both GD and magnitude metrics)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Group delay from FIR
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Magnitude metrics
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Stereo metrics
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Safe missing result
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FinalIRValidationResult:
    valid: bool
    severity: str  # "ok" | "warn" | "reject"
    score_penalty: float

    mag_rms_db: float
    mag_peak_db: float

    pre_energy_ratio_db: float
    post_energy_ratio_db: float
    early_energy_ratio_db: float

    gd_peak_ms: float
    gd_rms_ms: float

    voice_band_peak_excess_db: float
    voice_band_energy_excess_db: float

    stereo_delta_rms_db: float
    stereo_delta_peak_db: float

    bass_residual_peak_db: float

    reasons: tuple[str, ...]
    metrics: dict[str, float]


def _bump_severity(current: str, candidate: str) -> str:
    if _SEVERITY_ORDER.get(candidate, 0) > _SEVERITY_ORDER.get(current, 0):
        return candidate
    return current


def _next_pow2(n: int) -> int:
    return int(2 ** np.ceil(np.log2(max(n, 2))))


def _main_peak_index(ir: np.ndarray) -> int:
    return int(np.argmax(np.abs(ir)))


def _safe_energy(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sum(x * x))


def _safe_rms(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 0.0
    return float(np.sqrt(np.mean(arr * arr)))


def _safe_arr(value: Any, n_ref: int | None = None) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
        if not np.any(np.isfinite(arr)):
            return None
        if n_ref is not None and arr.size != n_ref:
            return None
        return arr
    except (TypeError, ValueError):
        return None


def _freq_mask(freq: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (freq >= float(lo)) & (freq <= float(hi))


__all__ = [
    "FinalIRValidationResult",
    "_bump_severity",
    "_next_pow2",
    "_main_peak_index",
    "_safe_energy",
    "_safe_rms",
    "_safe_arr",
    "_freq_mask",
]
