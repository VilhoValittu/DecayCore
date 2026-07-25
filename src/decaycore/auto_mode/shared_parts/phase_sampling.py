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
from .cache_hash import _auto_filter_cache_key
from .safe_values import _auto_safe_float, _clip

logger = logging.getLogger("DecayCore")

def _auto_is_phase_search_filter(filter_type: str | None) -> bool:
    fk = str(_auto_filter_cache_key(filter_type=filter_type))
    return fk in ("linear", "asym")

def _auto_phase_limit_clip(value, *, default: float = 400.0) -> float:
    v = _auto_safe_float(value, float("nan"))
    if not np.isfinite(v):
        v = _auto_safe_float(default, 400.0)
    return _clip(
        v, float(AUTO_MODE_PHASE_LIMIT_MIN_HZ), float(AUTO_MODE_PHASE_LIMIT_MAX_HZ)
    )

def _auto_phase_limit_center(value, *, default: float | None = None) -> float:
    v = _auto_safe_float(value, float("nan"))
    lo = float(AUTO_MODE_PHASE_LIMIT_MIN_HZ)
    hi = float(AUTO_MODE_PHASE_LIMIT_MAX_HZ)
    if np.isfinite(v) and (lo <= float(v) <= hi):
        return float(v)
    d = _auto_safe_float(
        AUTO_MODE_PHASE_LIMIT_DEFAULT_HZ if default is None else default,
        AUTO_MODE_PHASE_LIMIT_DEFAULT_HZ,
    )
    return float(_clip(d, lo, hi))

def _auto_mag_c_min_center(base_data: dict | None, *, default: float = 25.0) -> float:
    try:
        raw = dict(base_data or {}).get(
            "_auto_mag_c_min_hz",
            dict(base_data or {}).get("mag_c_min", default),
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
        raw = default
    seed = _auto_safe_float(raw, float("nan"))
    if not np.isfinite(seed):
        seed = _auto_safe_float(default, float("nan"))
    if not np.isfinite(seed):
        return float("nan")
    return float(
        _clip(
            seed,
            float(AUTO_MODE_MAG_C_MIN_MIN_HZ),
            float(AUTO_MODE_MAG_C_MIN_MAX_HZ),
        )
    )

def _auto_mag_c_min_search_bounds(base_data: dict | None) -> tuple[float, float]:
    """Keep automatic search close to the measured low-frequency capability.

    The protection seed is the smoothed -6 dB extension estimate. Searching far
    above it lets the optimizer improve unrelated metrics by abandoning bass
    correction. Downward exploration remains open because cuts below the seed
    are safe; upward exploration is limited to the same small allowance used by
    winner polish.
    """
    lo = float(AUTO_MODE_MAG_C_MIN_MIN_HZ)
    hi = float(AUTO_MODE_MAG_C_MIN_MAX_HZ)
    center = _auto_mag_c_min_center(base_data, default=float("nan"))
    if np.isfinite(center):
        max_up_hz = max(
            0.0,
            _auto_safe_float(AUTO_MODE_MAG_C_MIN_SEARCH_MAX_UP_HZ, 4.0),
        )
        hi = min(hi, float(center) + float(max_up_hz))
    return float(lo), float(max(lo, hi))

def _auto_phase_limit_prior_penalty(
    phase_limit_hz: float, *, filter_key: str | None
) -> float:
    if not _auto_is_phase_search_filter(filter_key):
        return 0.0
    pl = _auto_safe_float(phase_limit_hz, float("nan"))
    if not np.isfinite(pl):
        return 0.0
    center = float(
        _clip(
            AUTO_MODE_PHASE_LIMIT_PRIOR_CENTER_HZ,
            AUTO_MODE_PHASE_LIMIT_MIN_HZ,
            AUTO_MODE_PHASE_LIMIT_MAX_HZ,
        )
    )
    tol = float(max(1.0, _auto_safe_float(AUTO_MODE_PHASE_LIMIT_PRIOR_TOL_HZ, 90.0)))
    span = float(max(1.0, _auto_safe_float(AUTO_MODE_PHASE_LIMIT_PRIOR_SPAN_HZ, 70.0)))
    w = float(max(0.0, _auto_safe_float(AUTO_MODE_PHASE_LIMIT_PRIOR_WEIGHT, 1.2)))
    max_pen = float(
        max(0.0, _auto_safe_float(AUTO_MODE_PHASE_LIMIT_PRIOR_MAX_PEN, 4.0))
    )
    excess = max(0.0, abs(float(pl) - center) - tol)
    pen = float(w) * ((float(excess) / float(span)) ** 2.0)
    return float(min(max_pen, max(0.0, pen)))

def _jitter(
    rng,
    v,
    sigma,
    lo,
    hi,
    *,
    base_data: dict | None = None,
    key: str | None = None,
    default=None,
):
    center = _auto_safe_float(v, float("nan"))
    if not np.isfinite(center):
        if key and isinstance(base_data, dict):
            center = _auto_safe_float(base_data.get(key, default), float("nan"))
        if not np.isfinite(center):
            if default is not None:
                center = _auto_safe_float(default, float("nan"))
            if not np.isfinite(center):
                center = 0.5 * (_auto_safe_float(lo, 0.0) + _auto_safe_float(hi, 0.0))
    sig = max(0.0, _auto_safe_float(sigma, 0.0))
    if sig <= 0.0:
        return _clip(center, lo, hi)
    try:
        x = float(rng.normal(loc=float(center), scale=float(sig)))
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
        x = float(center)
    return _clip(x, lo, hi)

def _auto_sample_mag_low_pair(
    rng,
    *,
    mag_center: float,
    low_center: float,
    mag_sigma: float,
    low_sigma: float,
    mag_lo: float | None = None,
    mag_hi: float | None = None,
) -> tuple[float, float]:
    mag_lo_eff = float(
        AUTO_MODE_MAG_C_MIN_MIN_HZ if mag_lo is None else mag_lo
    )
    mag_hi_eff = float(
        AUTO_MODE_MAG_C_MIN_MAX_HZ if mag_hi is None else mag_hi
    )
    mag_hi_eff = max(float(mag_lo_eff), float(mag_hi_eff))
    mag = float(
        _jitter(
            rng,
            mag_center,
            mag_sigma,
            float(mag_lo_eff),
            float(mag_hi_eff),
            default=mag_center,
        )
    )
    low = float(
        _jitter(
            rng,
            low_center,
            low_sigma,
            float(AUTO_MODE_LOW_BASS_MIN_HZ),
            float(AUTO_MODE_LOW_BASS_MAX_HZ),
            default=low_center,
        )
    )
    mag = float(
        _clip(
            mag,
            float(mag_lo_eff),
            float(mag_hi_eff),
        )
    )
    low = float(
        _clip(
            low,
            float(AUTO_MODE_LOW_BASS_MIN_HZ),
            float(AUTO_MODE_LOW_BASS_MAX_HZ),
        )
    )
    return float(round(mag, 1)), float(round(low, 1))


__all__ = [
    '_auto_is_phase_search_filter',
    '_auto_phase_limit_clip',
    '_auto_phase_limit_center',
    '_auto_mag_c_min_center',
    '_auto_mag_c_min_search_bounds',
    '_auto_phase_limit_prior_penalty',
    '_jitter',
    '_auto_sample_mag_low_pair',
]
