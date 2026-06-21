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

from dataclasses import dataclass
from typing import Any

import numpy as np


SUB_TARGET_LPF_HZ = 200.0
SUB_TARGET_LPF_SLOPE_DB_PER_OCT = 96.0
SUB_TARGET_LPF_MAX_ATTENUATION_DB = 80.0
SUB_TARGET_POLICY = "main_target_plus_steep_200hz_lpf"


@dataclass(frozen=True)
class SubwooferTarget:
    house_freqs: list[float] | None
    house_mags: list[float] | None
    applied: bool
    lpf_hz: float = SUB_TARGET_LPF_HZ
    slope_db_per_oct: float = SUB_TARGET_LPF_SLOPE_DB_PER_OCT
    policy: str = SUB_TARGET_POLICY


def build_subwoofer_target_with_lpf(
    house_freqs: Any,
    house_mags: Any,
    *,
    lpf_hz: float = SUB_TARGET_LPF_HZ,
    slope_db_per_oct: float = SUB_TARGET_LPF_SLOPE_DB_PER_OCT,
    max_attenuation_db: float = SUB_TARGET_LPF_MAX_ATTENUATION_DB,
) -> SubwooferTarget:
    """Return a sub-specific house curve with a steep deterministic LPF target."""
    try:
        f_raw = np.asarray(house_freqs, dtype=float).reshape(-1)
        m_raw = np.asarray(house_mags, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return SubwooferTarget(None, None, False, float(lpf_hz), float(slope_db_per_oct))

    n = int(min(f_raw.size, m_raw.size))
    if n < 2:
        return SubwooferTarget(None, None, False, float(lpf_hz), float(slope_db_per_oct))

    f = f_raw[:n]
    m = m_raw[:n]
    valid = np.isfinite(f) & np.isfinite(m)
    f = f[valid]
    m = m[valid]
    if f.size < 2:
        return SubwooferTarget(None, None, False, float(lpf_hz), float(slope_db_per_oct))

    order = np.argsort(f, kind="mergesort")
    f = f[order]
    m = m[order]
    uniq = np.concatenate(([True], np.diff(f) > 0.0))
    f = f[uniq]
    m = m[uniq]
    if f.size < 2:
        return SubwooferTarget(None, None, False, float(lpf_hz), float(slope_db_per_oct))

    try:
        cutoff = float(lpf_hz)
        slope = float(slope_db_per_oct)
        max_att = float(max_attenuation_db)
    except (TypeError, ValueError):
        return SubwooferTarget(f.astype(float).tolist(), m.astype(float).tolist(), False)
    if not (np.isfinite(cutoff) and cutoff > 0.0 and np.isfinite(slope) and slope > 0.0):
        return SubwooferTarget(f.astype(float).tolist(), m.astype(float).tolist(), False)
    if not (np.isfinite(max_att) and max_att > 0.0):
        max_att = SUB_TARGET_LPF_MAX_ATTENUATION_DB

    max_freq = float(max(np.max(f), cutoff * 2.0))
    octave_span = max(1.0, float(np.log2(max_freq / cutoff)))
    n_grid = max(32, int(np.ceil(octave_span * 32.0)) + 1)
    lpf_grid = np.geomspace(cutoff, max_freq, n_grid)
    octave_grid = cutoff * np.power(2.0, np.arange(0.0, np.floor(octave_span) + 1.0))
    out_f = np.unique(np.concatenate((f, np.asarray([cutoff], dtype=float), lpf_grid, octave_grid)))
    out_f = out_f[np.isfinite(out_f)]
    out_f.sort()

    base = np.interp(out_f, f, m)
    with np.errstate(divide="ignore", invalid="ignore"):
        attenuation = slope * np.log2(np.maximum(out_f, cutoff) / cutoff)
    attenuation = np.where(out_f > cutoff, np.minimum(attenuation, max_att), 0.0)
    out_m = base - attenuation
    out_m = np.where(np.isfinite(out_m), out_m, base)

    return SubwooferTarget(
        out_f.astype(float).tolist(),
        out_m.astype(float).tolist(),
        True,
        float(cutoff),
        float(slope),
        SUB_TARGET_POLICY,
    )


def subwoofer_target_metadata(target: SubwooferTarget) -> dict[str, float | str | bool]:
    return {
        "sub_target_policy": str(target.policy),
        "sub_target_lpf_hz": float(target.lpf_hz),
        "sub_target_lpf_slope_db_per_oct": float(target.slope_db_per_oct),
        "sub_target_applied": bool(target.applied),
    }


__all__ = [
    'SUB_TARGET_LPF_HZ',
    'SUB_TARGET_LPF_MAX_ATTENUATION_DB',
    'SUB_TARGET_LPF_SLOPE_DB_PER_OCT',
    'SUB_TARGET_POLICY',
    'SubwooferTarget',
    'build_subwoofer_target_with_lpf',
    'subwoofer_target_metadata',
]
