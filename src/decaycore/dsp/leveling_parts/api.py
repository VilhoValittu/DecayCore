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
import numpy as np

from ..leveling_compute import compute_leveling_impl
from ..leveling_window import (
    find_shared_stereo_level_window_impl,
    find_stable_level_window_impl,
)

from .state_cache import (
    StereoLinkContext,
    _LEVELING_CACHE,
    _LEVELING_CACHE_LOCK,
    _LEVELING_CACHE_MAX,
    _LEVEL_WINDOW_CACHE,
    _LEVEL_WINDOW_CACHE_LOCK,
    _LEVEL_WINDOW_CACHE_MAX,
    _capture_leveling_state,
    _leveling_cache_key,
    _remember_leveling_error,
    _restore_leveling_state,
    _safe_setattr,
    _stable_level_window_cache_key,
    _shared_stereo_level_window_cache_key,
    _to_bool,
    _to_float,
)
from .tilt_helpers import (
    _is_identity_log_grid,
    _log_median,
    _lower_tail_robust_std_db,
    _resample_log_axis,
    _tilt_fit_offset_and_slope_db_per_oct,
    _window_offset_consistency_score,
)
from .window_scoring import _perceptual_shape_score


def find_stable_level_window(
    freq_axis: np.ndarray,
    magnitudes: np.ndarray,
    target_mags: np.ndarray,
    f_min: float,
    f_max: float,
    window_size_octaves: float = 1.0,
    hpf_freq: float = 0.0,
    tilt_comp: bool = True,
    tilt_max_db_per_oct: float = 2.0,
    perceptual_weighting: bool = False,
    perceptual_strength: float = 0.18,
    perceptual_min_hz: float = 250.0,
    perceptual_max_hz: float = 4000.0,
    perceptual_tie_only: bool = False,
) -> tuple[float, float]:
    cache_key = None
    try:
        cache_key = _stable_level_window_cache_key(
            freq_axis,
            magnitudes,
            target_mags,
            f_min=f_min,
            f_max=f_max,
            window_size_octaves=window_size_octaves,
            hpf_freq=hpf_freq,
            tilt_comp=tilt_comp,
            tilt_max_db_per_oct=tilt_max_db_per_oct,
            perceptual_weighting=perceptual_weighting,
            perceptual_strength=perceptual_strength,
            perceptual_min_hz=perceptual_min_hz,
            perceptual_max_hz=perceptual_max_hz,
            perceptual_tie_only=perceptual_tie_only,
        )
    except (AttributeError, TypeError, ValueError, FloatingPointError, OverflowError):
        cache_key = None

    if cache_key is not None:
        with _LEVEL_WINDOW_CACHE_LOCK:
            cached = _LEVEL_WINDOW_CACHE.get(cache_key)
        if cached is not None:
            return cached

    result = find_stable_level_window_impl(
        freq_axis,
        magnitudes,
        target_mags,
        f_min,
        f_max,
        window_size_octaves=window_size_octaves,
        hpf_freq=hpf_freq,
        tilt_comp=tilt_comp,
        tilt_max_db_per_oct=tilt_max_db_per_oct,
        perceptual_weighting=perceptual_weighting,
        perceptual_strength=perceptual_strength,
        perceptual_min_hz=perceptual_min_hz,
        perceptual_max_hz=perceptual_max_hz,
        perceptual_tie_only=perceptual_tie_only,
        to_float_fn=_to_float,
        to_bool_fn=_to_bool,
        resample_log_axis_fn=_resample_log_axis,
        lower_tail_robust_std_db_fn=_lower_tail_robust_std_db,
        window_offset_consistency_score_fn=_window_offset_consistency_score,
        perceptual_shape_score_fn=_perceptual_shape_score,
        is_identity_log_grid_fn=_is_identity_log_grid,
    )
    cached_result = (float(result[0]), float(result[1]))
    if cache_key is not None:
        with _LEVEL_WINDOW_CACHE_LOCK:
            if cache_key not in _LEVEL_WINDOW_CACHE and len(_LEVEL_WINDOW_CACHE) >= _LEVEL_WINDOW_CACHE_MAX:
                _LEVEL_WINDOW_CACHE.pop(next(iter(_LEVEL_WINDOW_CACHE)))
            _LEVEL_WINDOW_CACHE[cache_key] = cached_result
    return cached_result


def find_shared_stereo_level_window(
    freq_axis_l: np.ndarray,
    magnitudes_l: np.ndarray,
    target_mags_l: np.ndarray,
    freq_axis_r: np.ndarray,
    magnitudes_r: np.ndarray,
    target_mags_r: np.ndarray,
    f_min: float,
    f_max: float,
    window_size_octaves: float = 1.0,
    hpf_freq: float = 0.0,
    tilt_comp: bool = True,
    tilt_max_db_per_oct: float = 2.0,
    perceptual_weighting: bool = False,
    perceptual_strength: float = 0.18,
    perceptual_min_hz: float = 250.0,
    perceptual_max_hz: float = 4000.0,
    perceptual_tie_only: bool = False,
) -> tuple[float, float]:
    cache_key = None
    try:
        cache_key = _shared_stereo_level_window_cache_key(
            freq_axis_l,
            magnitudes_l,
            target_mags_l,
            freq_axis_r,
            magnitudes_r,
            target_mags_r,
            f_min=f_min,
            f_max=f_max,
            window_size_octaves=window_size_octaves,
            hpf_freq=hpf_freq,
            tilt_comp=tilt_comp,
            tilt_max_db_per_oct=tilt_max_db_per_oct,
            perceptual_weighting=perceptual_weighting,
            perceptual_strength=perceptual_strength,
            perceptual_min_hz=perceptual_min_hz,
            perceptual_max_hz=perceptual_max_hz,
            perceptual_tie_only=perceptual_tie_only,
        )
    except (AttributeError, TypeError, ValueError, FloatingPointError, OverflowError):
        cache_key = None

    if cache_key is not None:
        with _LEVEL_WINDOW_CACHE_LOCK:
            cached = _LEVEL_WINDOW_CACHE.get(cache_key)
        if cached is not None:
            return cached

    result = find_shared_stereo_level_window_impl(
        freq_axis_l,
        magnitudes_l,
        target_mags_l,
        freq_axis_r,
        magnitudes_r,
        target_mags_r,
        f_min,
        f_max,
        window_size_octaves=window_size_octaves,
        hpf_freq=hpf_freq,
        tilt_comp=tilt_comp,
        tilt_max_db_per_oct=tilt_max_db_per_oct,
        perceptual_weighting=perceptual_weighting,
        perceptual_strength=perceptual_strength,
        perceptual_min_hz=perceptual_min_hz,
        perceptual_max_hz=perceptual_max_hz,
        perceptual_tie_only=perceptual_tie_only,
        to_float_fn=_to_float,
        to_bool_fn=_to_bool,
        resample_log_axis_fn=_resample_log_axis,
        lower_tail_robust_std_db_fn=_lower_tail_robust_std_db,
        window_offset_consistency_score_fn=_window_offset_consistency_score,
        perceptual_shape_score_fn=_perceptual_shape_score,
        is_identity_log_grid_fn=_is_identity_log_grid,
    )
    cached_result = (float(result[0]), float(result[1]))
    if cache_key is not None:
        with _LEVEL_WINDOW_CACHE_LOCK:
            if cache_key not in _LEVEL_WINDOW_CACHE and len(_LEVEL_WINDOW_CACHE) >= _LEVEL_WINDOW_CACHE_MAX:
                _LEVEL_WINDOW_CACHE.pop(next(iter(_LEVEL_WINDOW_CACHE)))
            _LEVEL_WINDOW_CACHE[cache_key] = cached_result
    return cached_result


def compute_leveling(
    cfg,
    freq_axis: np.ndarray,
    m_anal: np.ndarray,
    target_mags: np.ndarray,
    *,
    stereo_link_ctx: StereoLinkContext | None = None,
):
    cache_key = None
    try:
        cache_key = _leveling_cache_key(
            cfg,
            freq_axis,
            m_anal,
            target_mags,
            stereo_link_ctx=stereo_link_ctx,
        )
    except (AttributeError, TypeError, ValueError, FloatingPointError, OverflowError):
        cache_key = None

    if cache_key is not None:
        with _LEVELING_CACHE_LOCK:
            cached = _LEVELING_CACHE.get(cache_key)
        if cached is not None:
            cached_result, cached_state = cached
            _restore_leveling_state(cfg, cached_state)
            return cached_result

    result = compute_leveling_impl(
        cfg,
        freq_axis,
        m_anal,
        target_mags,
        stereo_link_ctx=stereo_link_ctx,
        to_float_fn=_to_float,
        to_bool_fn=_to_bool,
        remember_leveling_error_fn=_remember_leveling_error,
        safe_setattr_fn=_safe_setattr,
        log_median_fn=_log_median,
        tilt_fit_offset_and_slope_db_per_oct_fn=_tilt_fit_offset_and_slope_db_per_oct,
        perceptual_shape_score_fn=_perceptual_shape_score,
        find_stable_level_window_fn=find_stable_level_window,
    )
    if cache_key is not None:
        cached_result = tuple(result)
        cached_state = _capture_leveling_state(cfg)
        with _LEVELING_CACHE_LOCK:
            if cache_key not in _LEVELING_CACHE and len(_LEVELING_CACHE) >= _LEVELING_CACHE_MAX:
                _LEVELING_CACHE.pop(next(iter(_LEVELING_CACHE)))
            _LEVELING_CACHE[cache_key] = (cached_result, cached_state)
        return cached_result
    return result


__all__ = [
    "find_stable_level_window",
    "find_shared_stereo_level_window",
    "compute_leveling",
    "compute_leveling_impl",
    "find_shared_stereo_level_window_impl",
    "find_stable_level_window_impl",
]
