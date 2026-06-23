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

from typing import Any, Callable

import numpy as np


def _leveling_store_window_debug(
    cfg,
    safe_setattr_fn: Callable[[Any, str, Any], None],
    ss_min_value: float,
    ss_max_value: float,
    offset_method_value: str,
    perceptual_value,
    *,
    lvl_perceptual_weighting: bool,
) -> None:
    safe_setattr_fn(cfg, "_lvl_perceptual_error_rms", perceptual_value)
    try:
        tilt_slope_value = getattr(cfg, "_lvl_tilt_slope_db_per_oct", None)
    except (AttributeError, TypeError, ValueError):
        tilt_slope_value = None
    safe_setattr_fn(
        cfg,
        "_lvl_window_debug",
        {
            "ss_min": float(ss_min_value),
            "ss_max": float(ss_max_value),
            "offset_method": str(offset_method_value),
            "tilt_slope_db_per_oct": tilt_slope_value,
            "perceptual_error_rms": perceptual_value,
            "perceptual_enabled": bool(lvl_perceptual_weighting),
        },
    )


def _leveling_tilt_fit_for_window(
    tilt_fit_offset_and_slope_db_per_oct_fn: Callable[..., tuple[float, float]],
    freq_values: np.ndarray,
    diff_values: np.ndarray,
    *,
    window_hi_hz: float,
    tilt_max_db_per_oct: float,
    subwoofer_goal: bool,
) -> tuple[float, float]:
    prefer_lf_piecewise_tilt = bool(
        subwoofer_goal and np.isfinite(float(window_hi_hz)) and float(window_hi_hz) <= 220.0
    )
    return tilt_fit_offset_and_slope_db_per_oct_fn(
        freq_values,
        diff_values,
        max_db_per_oct=float(tilt_max_db_per_oct),
        prefer_lf_piecewise_tilt=bool(prefer_lf_piecewise_tilt),
    )


def _leveling_perceptual_error(
    *,
    enabled: bool,
    mask: np.ndarray,
    target_arr: np.ndarray | None,
    freq_axis: np.ndarray,
    m_anal: np.ndarray,
    tilt_comp: bool,
    tilt_max_db_per_oct: float,
    lvl_perceptual_min_hz: float,
    lvl_perceptual_max_hz: float,
    perceptual_shape_score_fn: Callable[..., float],
) -> float | None:
    if not bool(enabled) or np.count_nonzero(mask) < 20 or target_arr is None:
        return None
    perceptual_error_rms = perceptual_shape_score_fn(
        freq_axis[mask],
        m_anal[mask],
        target_arr[mask],
        tilt_comp=bool(tilt_comp),
        tilt_max_db_per_oct=float(tilt_max_db_per_oct),
        min_hz=float(lvl_perceptual_min_hz),
        max_hz=float(lvl_perceptual_max_hz),
    )
    if not np.isfinite(float(perceptual_error_rms)):
        return None
    return float(perceptual_error_rms)


def _leveling_metrics_from_mask(
    *,
    cfg,
    freq_axis: np.ndarray,
    m_anal: np.ndarray,
    target_mags: np.ndarray,
    target_arr: np.ndarray | None,
    mask: np.ndarray,
    offset_prefix: str,
    forced_offset,
    window_hi_hz: float,
    tilt_comp: bool,
    tilt_max_db_per_oct: float,
    lvl_perceptual_weighting: bool,
    lvl_perceptual_min_hz: float,
    lvl_perceptual_max_hz: float,
    subwoofer_goal: bool,
    safe_setattr_fn: Callable[[Any, str, Any], None],
    log_median_fn: Callable[[np.ndarray, np.ndarray], float],
    tilt_fit_offset_and_slope_db_per_oct_fn: Callable[..., tuple[float, float]],
    perceptual_shape_score_fn: Callable[..., float],
) -> tuple[float, float, float, str, float | None]:
    if np.any(mask):
        meas_level_db_window = log_median_fn(freq_axis[mask], m_anal[mask])
        target_level_db_window = log_median_fn(freq_axis[mask], target_mags[mask])
    else:
        meas_level_db_window = 0.0
        target_level_db_window = 0.0

    if forced_offset is not None:
        calc_offset_db = float(forced_offset)
        offset_method = "ForcedOffset"
        if tilt_comp and np.any(mask):
            _off_tmp, slope = _leveling_tilt_fit_for_window(
                tilt_fit_offset_and_slope_db_per_oct_fn,
                freq_axis[mask],
                (m_anal[mask] - target_mags[mask]),
                window_hi_hz=float(window_hi_hz),
                tilt_max_db_per_oct=float(tilt_max_db_per_oct),
                subwoofer_goal=bool(subwoofer_goal),
            )
            safe_setattr_fn(cfg, "_lvl_tilt_slope_db_per_oct", float(slope))
    elif np.any(mask):
        diff = m_anal[mask] - target_mags[mask]
        if tilt_comp:
            calc_offset_db, tilt_slope = _leveling_tilt_fit_for_window(
                tilt_fit_offset_and_slope_db_per_oct_fn,
                freq_axis[mask],
                diff,
                window_hi_hz=float(window_hi_hz),
                tilt_max_db_per_oct=float(tilt_max_db_per_oct),
                subwoofer_goal=bool(subwoofer_goal),
            )
            safe_setattr_fn(cfg, "_lvl_tilt_slope_db_per_oct", float(tilt_slope))
            offset_method = f"{offset_prefix}TiltMedian"
        else:
            calc_offset_db = log_median_fn(freq_axis[mask], diff)
            offset_method = f"{offset_prefix}Median"
    else:
        calc_offset_db = 0.0
        offset_method = f"{offset_prefix}NoMask"

    perceptual_error_rms = _leveling_perceptual_error(
        enabled=bool(lvl_perceptual_weighting),
        mask=mask,
        target_arr=target_arr,
        freq_axis=freq_axis,
        m_anal=m_anal,
        tilt_comp=bool(tilt_comp),
        tilt_max_db_per_oct=float(tilt_max_db_per_oct),
        lvl_perceptual_min_hz=float(lvl_perceptual_min_hz),
        lvl_perceptual_max_hz=float(lvl_perceptual_max_hz),
        perceptual_shape_score_fn=perceptual_shape_score_fn,
    )
    return (
        float(meas_level_db_window),
        float(target_level_db_window),
        float(calc_offset_db),
        str(offset_method),
        perceptual_error_rms,
    )


def _leveling_forced_result(
    *,
    cfg,
    freq_axis: np.ndarray,
    m_anal: np.ndarray,
    target_mags: np.ndarray,
    target_arr: np.ndarray | None,
    manual_target_db: float,
    s_min: float,
    s_max: float,
    forced_window,
    forced_offset,
    shared_target_level_db,
    tilt_comp: bool,
    tilt_max_db_per_oct: float,
    lvl_perceptual_weighting: bool,
    lvl_perceptual_min_hz: float,
    lvl_perceptual_max_hz: float,
    subwoofer_goal: bool,
    safe_setattr_fn: Callable[[Any, str, Any], None],
    remember_leveling_error_fn: Callable[[Any, str, Exception | None], None],
    log_median_fn: Callable[[np.ndarray, np.ndarray], float],
    tilt_fit_offset_and_slope_db_per_oct_fn: Callable[..., tuple[float, float]],
    perceptual_shape_score_fn: Callable[..., float],
    to_float_fn: Callable[[object, float], float],
    lvl_perceptual_weighting_value: bool,
) -> tuple[float, float, float, float, str, float, float] | None:
    if forced_window is None and forced_offset is None:
        return None
    try:
        if forced_window is not None:
            fw0, fw1 = forced_window
            ss_min = to_float_fn(fw0, s_min)
            ss_max = to_float_fn(fw1, s_max)
            if (ss_min <= 0) or (ss_max <= 0) or (ss_min >= ss_max):
                ss_min, ss_max = s_min, s_max
            ss_min = max(s_min, ss_min)
            ss_max = min(s_max, ss_max)
        else:
            ss_min, ss_max = s_min, s_max

        mask = (freq_axis >= ss_min) & (freq_axis <= ss_max)
        (
            meas_level_db_window,
            target_level_db_window,
            calc_offset_db,
            offset_method,
            perceptual_error_rms,
        ) = _leveling_metrics_from_mask(
            cfg=cfg,
            freq_axis=freq_axis,
            m_anal=m_anal,
            target_mags=target_mags,
            target_arr=target_arr,
            mask=mask,
            offset_prefix="ForcedWindow",
            forced_offset=forced_offset,
            window_hi_hz=float(ss_max),
            tilt_comp=bool(tilt_comp),
            tilt_max_db_per_oct=float(tilt_max_db_per_oct),
            lvl_perceptual_weighting=bool(lvl_perceptual_weighting),
            lvl_perceptual_min_hz=float(lvl_perceptual_min_hz),
            lvl_perceptual_max_hz=float(lvl_perceptual_max_hz),
            subwoofer_goal=bool(subwoofer_goal),
            safe_setattr_fn=safe_setattr_fn,
            log_median_fn=log_median_fn,
            tilt_fit_offset_and_slope_db_per_oct_fn=tilt_fit_offset_and_slope_db_per_oct_fn,
            perceptual_shape_score_fn=perceptual_shape_score_fn,
        )

        if shared_target_level_db is not None:
            target_level_db = to_float_fn(shared_target_level_db, float(meas_level_db_window))
        else:
            target_level_db = float(meas_level_db_window)
        if not np.isfinite(calc_offset_db):
            calc_offset_db = 0.0
        _leveling_store_window_debug(
            cfg,
            safe_setattr_fn,
            ss_min,
            ss_max,
            offset_method,
            perceptual_error_rms,
            lvl_perceptual_weighting=lvl_perceptual_weighting_value,
        )
        return (
            float(target_level_db),
            float(calc_offset_db),
            float(meas_level_db_window),
            float(target_level_db_window),
            str(offset_method),
            float(ss_min),
            float(ss_max),
        )
    except (TypeError, ValueError, FloatingPointError, ZeroDivisionError) as exc:
        remember_leveling_error_fn(cfg, "forced_window", exc)
        return None


def _leveling_manual_result(
    *,
    cfg,
    freq_axis: np.ndarray,
    m_anal: np.ndarray,
    target_mags: np.ndarray,
    target_arr: np.ndarray | None,
    manual_target_db: float,
    s_min: float,
    s_max: float,
    tilt_comp: bool,
    tilt_max_db_per_oct: float,
    lvl_perceptual_weighting: bool,
    lvl_perceptual_min_hz: float,
    lvl_perceptual_max_hz: float,
    subwoofer_goal: bool,
    safe_setattr_fn: Callable[[Any, str, Any], None],
    log_median_fn: Callable[[np.ndarray, np.ndarray], float],
    tilt_fit_offset_and_slope_db_per_oct_fn: Callable[..., tuple[float, float]],
    perceptual_shape_score_fn: Callable[..., float],
    lvl_perceptual_weighting_value: bool,
) -> tuple[float, float, float, float, str, float, float] | None:
    mask = (freq_axis >= s_min) & (freq_axis <= s_max)
    (
        meas_level_db_window,
        target_level_db_window,
        calc_offset_db,
        offset_method,
        perceptual_error_rms,
    ) = _leveling_metrics_from_mask(
        cfg=cfg,
        freq_axis=freq_axis,
        m_anal=m_anal,
        target_mags=target_mags,
        target_arr=target_arr,
        mask=mask,
        offset_prefix="Manual",
        forced_offset=None,
        window_hi_hz=float(s_max),
        tilt_comp=bool(tilt_comp),
        tilt_max_db_per_oct=float(tilt_max_db_per_oct),
        lvl_perceptual_weighting=bool(lvl_perceptual_weighting),
        lvl_perceptual_min_hz=float(lvl_perceptual_min_hz),
        lvl_perceptual_max_hz=float(lvl_perceptual_max_hz),
        subwoofer_goal=bool(subwoofer_goal),
        safe_setattr_fn=safe_setattr_fn,
        log_median_fn=log_median_fn,
        tilt_fit_offset_and_slope_db_per_oct_fn=tilt_fit_offset_and_slope_db_per_oct_fn,
        perceptual_shape_score_fn=perceptual_shape_score_fn,
    )
    if not np.isfinite(calc_offset_db):
        calc_offset_db = 0.0
    _leveling_store_window_debug(
        cfg,
        safe_setattr_fn,
        s_min,
        s_max,
        offset_method,
        perceptual_error_rms,
        lvl_perceptual_weighting=lvl_perceptual_weighting_value,
    )
    return (
        float(manual_target_db),
        float(calc_offset_db),
        float(meas_level_db_window),
        float(target_level_db_window),
        str(offset_method),
        float(s_min),
        float(s_max),
    )


def _leveling_smart_scan_result(
    *,
    cfg,
    freq_axis: np.ndarray,
    m_anal: np.ndarray,
    target_mags: np.ndarray,
    target_arr: np.ndarray | None,
    shared_target_level_db,
    is_manual: bool,
    manual_target_db: float,
    s_min: float,
    s_max: float,
    tilt_comp: bool,
    tilt_max_db_per_oct: float,
    lvl_perceptual_weighting: bool,
    lvl_perceptual_strength: float,
    lvl_perceptual_min_hz: float,
    lvl_perceptual_max_hz: float,
    lvl_perceptual_tie_only: bool,
    find_stable_level_window_fn: Callable[..., tuple[float, float]],
    safe_setattr_fn: Callable[[Any, str, Any], None],
    log_median_fn: Callable[[np.ndarray, np.ndarray], float],
    tilt_fit_offset_and_slope_db_per_oct_fn: Callable[..., tuple[float, float]],
    perceptual_shape_score_fn: Callable[..., float],
    remember_leveling_error_fn: Callable[[Any, str, Exception | None], None],
    to_float_fn: Callable[[object, float], float],
    lvl_perceptual_weighting_value: bool,
) -> tuple[float, float, float, float, str, float, float]:
    hpf_freq = 0.0
    hpf_settings = getattr(cfg, "hpf_settings", None)
    if hpf_settings:
        try:
            hpf_freq = to_float_fn(hpf_settings.get("freq", 0.0), 0.0)
        except (AttributeError, TypeError, ValueError) as exc:
            remember_leveling_error_fn(cfg, "hpf_settings", exc)
            hpf_freq = 0.0

    ss_min, ss_max = find_stable_level_window_fn(
        freq_axis,
        m_anal,
        target_mags,
        s_min,
        s_max,
        window_size_octaves=1.0,
        hpf_freq=float(hpf_freq),
        tilt_comp=bool(tilt_comp),
        tilt_max_db_per_oct=float(tilt_max_db_per_oct),
        perceptual_weighting=bool(lvl_perceptual_weighting),
        perceptual_strength=float(lvl_perceptual_strength),
        perceptual_min_hz=float(lvl_perceptual_min_hz),
        perceptual_max_hz=float(lvl_perceptual_max_hz),
        perceptual_tie_only=bool(lvl_perceptual_tie_only),
    )

    ss_min = to_float_fn(ss_min, s_min)
    ss_max = to_float_fn(ss_max, s_max)
    if (ss_min <= 0) or (ss_max <= 0) or (ss_min >= ss_max):
        ss_min, ss_max = s_min, s_max
    ss_min = max(s_min, ss_min)
    ss_max = min(s_max, ss_max)

    mask = (freq_axis >= ss_min) & (freq_axis <= ss_max)
    if np.count_nonzero(mask) < 20:
        fb_min = max(s_min, 350.0)
        fb_max = min(s_max, 5000.0)
        if fb_min < fb_max:
            ss_min, ss_max = fb_min, fb_max
            mask = (freq_axis >= ss_min) & (freq_axis <= ss_max)
    if np.count_nonzero(mask) < 20:
        ss_min, ss_max = s_min, s_max
        mask = (freq_axis >= ss_min) & (freq_axis <= ss_max)

    (
        meas_level_db_window,
        target_level_db_window,
        calc_offset_db,
        offset_method,
        perceptual_error_rms,
    ) = _leveling_metrics_from_mask(
        cfg=cfg,
        freq_axis=freq_axis,
        m_anal=m_anal,
        target_mags=target_mags,
        target_arr=target_arr,
        mask=mask,
        offset_prefix="SmartScan",
        forced_offset=None,
        window_hi_hz=float(ss_max),
        tilt_comp=bool(tilt_comp),
        tilt_max_db_per_oct=float(tilt_max_db_per_oct),
        lvl_perceptual_weighting=bool(lvl_perceptual_weighting),
        lvl_perceptual_min_hz=float(lvl_perceptual_min_hz),
        lvl_perceptual_max_hz=float(lvl_perceptual_max_hz),
        subwoofer_goal=False,
        safe_setattr_fn=safe_setattr_fn,
        log_median_fn=log_median_fn,
        tilt_fit_offset_and_slope_db_per_oct_fn=tilt_fit_offset_and_slope_db_per_oct_fn,
        perceptual_shape_score_fn=perceptual_shape_score_fn,
    )

    if is_manual:
        target_level_db = float(manual_target_db)
    elif shared_target_level_db is not None:
        target_level_db = to_float_fn(shared_target_level_db, float(meas_level_db_window))
    else:
        target_level_db = float(meas_level_db_window)
    if not np.isfinite(calc_offset_db):
        calc_offset_db = 0.0
    _leveling_store_window_debug(
        cfg,
        safe_setattr_fn,
        ss_min,
        ss_max,
        offset_method,
        perceptual_error_rms,
        lvl_perceptual_weighting=lvl_perceptual_weighting_value,
    )
    return (
        float(target_level_db),
        float(calc_offset_db),
        float(meas_level_db_window),
        float(target_level_db_window),
        str(offset_method),
        float(ss_min),
        float(ss_max),
    )


def _leveling_setup_context(
    cfg,
    freq_axis: np.ndarray,
    target_mags: np.ndarray,
    *,
    to_float_fn: Callable[[object, float], float],
    to_bool_fn: Callable[[object, bool], bool],
    safe_setattr_fn: Callable[[Any, str, Any], None],
) -> dict[str, Any]:
    manual_target_db = to_float_fn(getattr(cfg, "lvl_manual_db", 0.0), 0.0)
    s_min = to_float_fn(getattr(cfg, "lvl_min", 500.0), 500.0)
    s_max = to_float_fn(getattr(cfg, "lvl_max", 2000.0), 2000.0)
    if s_min <= 0 or s_max <= 0 or s_min >= s_max:
        s_min, s_max = 500.0, 2000.0

    mode = str(getattr(cfg, "lvl_mode", "Auto"))
    tilt_comp = bool(getattr(cfg, "lvl_tilt_comp", True))
    tilt_max_db_per_oct = to_float_fn(getattr(cfg, "lvl_tilt_max_db_per_oct", 2.0), 2.0)
    try:
        auto_goal = str(getattr(cfg, "auto_goal", "") or "").strip().lower()
    except (AttributeError, TypeError, ValueError):
        auto_goal = ""
    subwoofer_goal = auto_goal in ("subwoofer", "subwoofers", "subs")
    lvl_perceptual_weighting = to_bool_fn(getattr(cfg, "lvl_perceptual_weighting", False), False)
    lvl_perceptual_strength = max(0.0, to_float_fn(getattr(cfg, "lvl_perceptual_strength", 0.12), 0.12))
    lvl_perceptual_min_hz = to_float_fn(getattr(cfg, "lvl_perceptual_min_hz", 250.0), 250.0)
    lvl_perceptual_max_hz = to_float_fn(getattr(cfg, "lvl_perceptual_max_hz", 4000.0), 4000.0)
    lvl_perceptual_tie_only = to_bool_fn(getattr(cfg, "lvl_perceptual_tie_only", True), True)
    if lvl_perceptual_min_hz <= 0.0 or lvl_perceptual_max_hz <= lvl_perceptual_min_hz:
        lvl_perceptual_min_hz, lvl_perceptual_max_hz = 250.0, 4000.0

    safe_setattr_fn(cfg, "_lvl_last_error", None)
    safe_setattr_fn(cfg, "_lvl_tilt_slope_db_per_oct", None)
    safe_setattr_fn(cfg, "_lvl_perceptual_enabled", bool(lvl_perceptual_weighting))
    safe_setattr_fn(cfg, "_lvl_perceptual_strength", float(lvl_perceptual_strength))
    safe_setattr_fn(
        cfg,
        "_lvl_perceptual_band_hz",
        (float(lvl_perceptual_min_hz), float(lvl_perceptual_max_hz)),
    )
    safe_setattr_fn(cfg, "_lvl_perceptual_error_rms", None)
    safe_setattr_fn(cfg, "_lvl_window_debug", None)

    try:
        target_arr = np.asarray(target_mags, dtype=float)
        if target_arr.shape != np.asarray(freq_axis).shape:
            target_arr = None
    except (TypeError, ValueError):
        target_arr = None

    return {
        "manual_target_db": float(manual_target_db),
        "s_min": float(s_min),
        "s_max": float(s_max),
        "is_manual": bool("Manual" in mode),
        "tilt_comp": bool(tilt_comp),
        "tilt_max_db_per_oct": float(tilt_max_db_per_oct),
        "subwoofer_goal": bool(subwoofer_goal),
        "lvl_perceptual_weighting": bool(lvl_perceptual_weighting),
        "lvl_perceptual_strength": float(lvl_perceptual_strength),
        "lvl_perceptual_min_hz": float(lvl_perceptual_min_hz),
        "lvl_perceptual_max_hz": float(lvl_perceptual_max_hz),
        "lvl_perceptual_tie_only": bool(lvl_perceptual_tie_only),
        "target_arr": target_arr,
    }


def _leveling_dispatch_result(
    cfg,
    freq_axis: np.ndarray,
    m_anal: np.ndarray,
    target_mags: np.ndarray,
    *,
    stereo_link_ctx: Any | None,
    to_float_fn: Callable[[object, float], float],
    remember_leveling_error_fn: Callable[[Any, str, Exception | None], None],
    safe_setattr_fn: Callable[[Any, str, Any], None],
    log_median_fn: Callable[[np.ndarray, np.ndarray], float],
    tilt_fit_offset_and_slope_db_per_oct_fn: Callable[..., tuple[float, float]],
    perceptual_shape_score_fn: Callable[..., float],
    find_stable_level_window_fn: Callable[..., tuple[float, float]],
    setup: dict[str, Any],
):
    forced_window = getattr(cfg, "lvl_force_window", None)
    forced_offset = getattr(cfg, "lvl_force_offset_db", None)
    shared_target_level_db = None
    if stereo_link_ctx is not None:
        try:
            if stereo_link_ctx.forced_window_hz is not None:
                forced_window = stereo_link_ctx.forced_window_hz
            if stereo_link_ctx.forced_offset_db is not None:
                forced_offset = stereo_link_ctx.forced_offset_db
            if stereo_link_ctx.shared_target_level_db is not None:
                shared_target_level_db = float(stereo_link_ctx.shared_target_level_db)
        except (AttributeError, TypeError, ValueError):
            pass
    if shared_target_level_db is not None and (not np.isfinite(float(shared_target_level_db))):
        shared_target_level_db = None

    forced_result = _leveling_forced_result(
        cfg=cfg,
        freq_axis=freq_axis,
        m_anal=m_anal,
        target_mags=target_mags,
        target_arr=setup["target_arr"],
        manual_target_db=setup["manual_target_db"],
        s_min=setup["s_min"],
        s_max=setup["s_max"],
        forced_window=forced_window,
        forced_offset=forced_offset,
        shared_target_level_db=shared_target_level_db,
        tilt_comp=setup["tilt_comp"],
        tilt_max_db_per_oct=setup["tilt_max_db_per_oct"],
        lvl_perceptual_weighting=setup["lvl_perceptual_weighting"],
        lvl_perceptual_min_hz=setup["lvl_perceptual_min_hz"],
        lvl_perceptual_max_hz=setup["lvl_perceptual_max_hz"],
        subwoofer_goal=setup["subwoofer_goal"],
        safe_setattr_fn=safe_setattr_fn,
        remember_leveling_error_fn=remember_leveling_error_fn,
        log_median_fn=log_median_fn,
        tilt_fit_offset_and_slope_db_per_oct_fn=tilt_fit_offset_and_slope_db_per_oct_fn,
        perceptual_shape_score_fn=perceptual_shape_score_fn,
        to_float_fn=to_float_fn,
        lvl_perceptual_weighting_value=setup["lvl_perceptual_weighting"],
    )
    if forced_result is not None:
        return forced_result

    manual_result = _leveling_manual_result(
        cfg=cfg,
        freq_axis=freq_axis,
        m_anal=m_anal,
        target_mags=target_mags,
        target_arr=setup["target_arr"],
        manual_target_db=setup["manual_target_db"],
        s_min=setup["s_min"],
        s_max=setup["s_max"],
        tilt_comp=setup["tilt_comp"],
        tilt_max_db_per_oct=setup["tilt_max_db_per_oct"],
        lvl_perceptual_weighting=setup["lvl_perceptual_weighting"],
        lvl_perceptual_min_hz=setup["lvl_perceptual_min_hz"],
        lvl_perceptual_max_hz=setup["lvl_perceptual_max_hz"],
        subwoofer_goal=setup["subwoofer_goal"],
        safe_setattr_fn=safe_setattr_fn,
        log_median_fn=log_median_fn,
        tilt_fit_offset_and_slope_db_per_oct_fn=tilt_fit_offset_and_slope_db_per_oct_fn,
        perceptual_shape_score_fn=perceptual_shape_score_fn,
        lvl_perceptual_weighting_value=setup["lvl_perceptual_weighting"],
    )
    if manual_result is not None:
        return manual_result

    return _leveling_smart_scan_result(
        cfg=cfg,
        freq_axis=freq_axis,
        m_anal=m_anal,
        target_mags=target_mags,
        target_arr=setup["target_arr"],
        shared_target_level_db=shared_target_level_db,
        is_manual=setup["is_manual"],
        manual_target_db=setup["manual_target_db"],
        s_min=setup["s_min"],
        s_max=setup["s_max"],
        tilt_comp=setup["tilt_comp"],
        tilt_max_db_per_oct=setup["tilt_max_db_per_oct"],
        lvl_perceptual_weighting=setup["lvl_perceptual_weighting"],
        lvl_perceptual_strength=setup["lvl_perceptual_strength"],
        lvl_perceptual_min_hz=setup["lvl_perceptual_min_hz"],
        lvl_perceptual_max_hz=setup["lvl_perceptual_max_hz"],
        lvl_perceptual_tie_only=setup["lvl_perceptual_tie_only"],
        find_stable_level_window_fn=find_stable_level_window_fn,
        safe_setattr_fn=safe_setattr_fn,
        log_median_fn=log_median_fn,
        tilt_fit_offset_and_slope_db_per_oct_fn=tilt_fit_offset_and_slope_db_per_oct_fn,
        perceptual_shape_score_fn=perceptual_shape_score_fn,
        remember_leveling_error_fn=remember_leveling_error_fn,
        to_float_fn=to_float_fn,
        lvl_perceptual_weighting_value=setup["lvl_perceptual_weighting"],
    )


def compute_leveling_impl(
    cfg,
    freq_axis: np.ndarray,
    m_anal: np.ndarray,
    target_mags: np.ndarray,
    *,
    stereo_link_ctx: Any | None = None,
    to_float_fn: Callable[[object, float], float],
    to_bool_fn: Callable[[object, bool], bool],
    remember_leveling_error_fn: Callable[[Any, str, Exception | None], None],
    safe_setattr_fn: Callable[[Any, str, Any], None],
    log_median_fn: Callable[[np.ndarray, np.ndarray], float],
    tilt_fit_offset_and_slope_db_per_oct_fn: Callable[..., tuple[float, float]],
    perceptual_shape_score_fn: Callable[..., float],
    find_stable_level_window_fn: Callable[..., tuple[float, float]],
):
    setup = _leveling_setup_context(
        cfg=cfg,
        freq_axis=freq_axis,
        target_mags=target_mags,
        to_float_fn=to_float_fn,
        to_bool_fn=to_bool_fn,
        safe_setattr_fn=safe_setattr_fn,
    )
    return _leveling_dispatch_result(
        cfg,
        freq_axis,
        m_anal,
        target_mags,
        stereo_link_ctx=stereo_link_ctx,
        to_float_fn=to_float_fn,
        remember_leveling_error_fn=remember_leveling_error_fn,
        safe_setattr_fn=safe_setattr_fn,
        log_median_fn=log_median_fn,
        tilt_fit_offset_and_slope_db_per_oct_fn=tilt_fit_offset_and_slope_db_per_oct_fn,
        perceptual_shape_score_fn=perceptual_shape_score_fn,
        find_stable_level_window_fn=find_stable_level_window_fn,
        setup=setup,
    )
