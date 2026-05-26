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

import inspect
import logging
import math
import sys

import numpy as np

_logger = logging.getLogger(__name__)

from ...common.acoustic_stats import calc_acoustic_score, calc_ai_summary_from_stats
from ...config.models import StereoAutoPolicyConfig, StereoResolvedAutoPolicies
from ...dsp.quality_metrics import (
    band_lr_mismatch_change_from_stats,
    band_lr_mismatch_rms_from_stats,
    normalized_policy_divergence_score,
    worst_channel_relief_db,
)
from ...dsp.modal_analysis import ModalAnalysisResult, RoomModeEvent, detect_room_modes
from ...dsp.smoothing import smooth_gain_fractional_octave
from ...dsp.target_match import target_match_from_stats
from .. import shared
from ..rank_score import (
    OFFICIAL_RANK_SCORE_CONTEXT,
    attach_official_rank_score,
    calibrated_auto_quality,
    compute_rank_score_components,
)
from ..runtime_context import (
    _auto_collect_reflections,
    _auto_event_penalty_weighted,
    _auto_event_severity,
    _auto_get_top_modes_hz,
    _auto_get_worst_mode_hz,
    _auto_mode_band,
    _auto_pick_metric,
)

from .metrics_common import _auto_stats_pick_arr
from .residual_analysis import (
    _auto_band_weight_for_correction_shape,
    _auto_min_broad_peak_width_oct,
)

CORRECTION_SHARPNESS_SCORING_VERSION = 1
DIP_FILL_RISK_SCORING_VERSION = 1
CHANNEL_OVERFIT_SCORING_VERSION = 1

def _auto_correction_sharpness_metrics_from_stats(
    st: dict | None,
    *,
    lo_hz: float = 20.0,
    hi_hz: float = 250.0,
) -> dict:
    out = {
        "correction_max_abs_slope_db_per_oct": 0.0,
        "correction_rms_slope_db_per_oct": 0.0,
        "correction_curvature_score": 0.0,
        "narrow_notch_count": 0,
        "narrow_peak_count": 0,
        "integrated_correction_db_oct": 0.0,
        "correction_event_count": 0,
        "direction_change_count": 0,
        "correction_sharpness_penalty": 0.0,
        "correction_sharpness_scoring_version": int(CORRECTION_SHARPNESS_SCORING_VERSION),
    }
    f = _auto_stats_pick_arr(st, "freq_axis")
    filt = _auto_stats_pick_arr(st, "predicted_filter_mags", "realized_filter_mags", "filter_mags")
    n = int(min(f.size, filt.size))
    if n < 8:
        return dict(out)
    f = np.asarray(f[:n], dtype=float)
    filt = np.asarray(filt[:n], dtype=float)
    mask = np.isfinite(f) & np.isfinite(filt) & (f > 0.0) & (f >= float(lo_hz)) & (f <= float(hi_hz))
    if int(np.count_nonzero(mask)) < 8:
        return dict(out)
    f_use = np.asarray(f[mask], dtype=float)
    y = np.asarray(filt[mask], dtype=float)
    order = np.argsort(f_use)
    f_use = f_use[order]
    y = y[order]
    x = np.log2(np.maximum(f_use, 1e-9))
    try:
        y_s = np.asarray(smooth_gain_fractional_octave(f_use, y, 12.0), dtype=float).reshape(-1)
        if y_s.size != y.size:
            y_s = y
    except Exception:
        y_s = y
    integrated_correction = float(np.trapezoid(np.abs(y_s), x)) if y_s.size >= 2 else 0.0
    out["integrated_correction_db_oct"] = integrated_correction

    dx = np.diff(x)
    dy = np.diff(y_s)
    valid_dx = np.isfinite(dx) & (np.abs(dx) > 1e-9) & np.isfinite(dy)
    if int(np.count_nonzero(valid_dx)) < 4:
        return dict(out)
    slope = np.asarray(dy[valid_dx] / dx[valid_dx], dtype=float)
    mid_f = np.sqrt(f_use[:-1][valid_dx] * f_use[1:][valid_dx])
    weights = _auto_band_weight_for_correction_shape(mid_f)
    abs_slope = np.abs(slope)
    max_abs_slope = float(np.max(abs_slope)) if abs_slope.size else 0.0
    rms_slope = float(np.sqrt(np.mean((abs_slope * weights) ** 2))) if abs_slope.size else 0.0
    curvature_score = 0.0
    direction_changes = 0
    if slope.size >= 3:
        slope_mid_f = np.asarray(mid_f, dtype=float)
        ds = np.diff(slope)
        dx2 = np.diff(np.log2(np.maximum(slope_mid_f, 1e-9)))
        valid2 = np.isfinite(ds) & np.isfinite(dx2) & (np.abs(dx2) > 1e-9)
        if int(np.count_nonzero(valid2)):
            curv = np.asarray(ds[valid2] / dx2[valid2], dtype=float)
            curv_f = np.asarray(slope_mid_f[1:][valid2], dtype=float)
            curvature_score = float(np.sqrt(np.mean((np.abs(curv) * _auto_band_weight_for_correction_shape(curv_f)) ** 2)))
        signs = np.sign(slope)
        signs[np.abs(slope) < 1.0] = 0.0
        nz = signs[signs != 0.0]
        if nz.size >= 2:
            direction_changes = int(np.count_nonzero(nz[1:] != nz[:-1]))

    narrow_notches = 0
    narrow_peaks = 0
    correction_events = 0
    if y_s.size >= 5:
        active = np.abs(y_s) >= 1.0
        starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
        ends = np.flatnonzero(active & ~np.r_[active[1:], False])
        correction_events = int(len(starts))
        for start, end in zip(starts, ends, strict=False):
            if int(end) <= int(start):
                continue
            width_oct = float(x[int(end)] - x[int(start)])
            seg = y_s[int(start):int(end) + 1]
            depth = float(abs(np.nanmin(seg))) if np.nanmin(seg) < 0.0 else 0.0
            peak_h = float(np.nanmax(seg)) if np.nanmax(seg) > 0.0 else 0.0
            center_f = float(np.sqrt(f_use[int(start)] * f_use[int(end)]))
            min_width = 0.75 * _auto_min_broad_peak_width_oct(center_f)
            if float(width_oct) < float(min_width) and depth >= 2.0:
                narrow_notches += 1
            if float(width_oct) < float(min_width) and peak_h >= 2.5:
                narrow_peaks += 1

    penalty = (
        0.030 * max(0.0, max_abs_slope - 36.0)
        + 0.045 * max(0.0, rms_slope - 14.0)
        + 0.006 * max(0.0, curvature_score - 120.0)
        + 0.35 * max(0, narrow_notches)
        + 0.40 * max(0, narrow_peaks)
        + 0.12 * max(0, correction_events - 4)
        + 0.16 * max(0, direction_changes - 3)
        + 0.015 * max(0.0, integrated_correction - 4.5)
    )
    out.update(
        {
            "correction_max_abs_slope_db_per_oct": float(max_abs_slope),
            "correction_rms_slope_db_per_oct": float(rms_slope),
            "correction_curvature_score": float(curvature_score),
            "narrow_notch_count": int(narrow_notches),
            "narrow_peak_count": int(narrow_peaks),
            "correction_event_count": int(correction_events),
            "direction_change_count": int(direction_changes),
            "correction_sharpness_penalty": float(np.clip(penalty, 0.0, 8.0)),
        }
    )
    return dict(out)


def _auto_merge_correction_sharpness_metrics(left_metrics: dict | None, right_metrics: dict | None) -> dict:
    out: dict[str, float | int] = {
        "correction_sharpness_scoring_version": int(CORRECTION_SHARPNESS_SCORING_VERSION),
    }
    for key in (
        "correction_max_abs_slope_db_per_oct",
        "correction_rms_slope_db_per_oct",
        "correction_curvature_score",
        "correction_sharpness_penalty",
    ):
        vals = [
            shared._auto_safe_float((m or {}).get(key, float("nan")), float("nan"))
            for m in (left_metrics, right_metrics)
        ]
        vals = [float(v) for v in vals if np.isfinite(v)]
        out[key] = float(max(vals)) if vals else 0.0
    for key in ("narrow_notch_count", "correction_event_count", "direction_change_count"):
        out[key] = int(
            sum(int(max(0, shared._auto_safe_float((m or {}).get(key, 0), 0.0))) for m in (left_metrics, right_metrics))
        )
    return dict(out)


def _auto_dip_fill_risk_metrics_from_stats(
    st: dict | None,
    *,
    lo_hz: float = 20.0,
    hi_hz: float = 250.0,
) -> dict:
    out = {
        "dip_fill_risk_score": 0.0,
        "dip_fill_risk_penalty": 0.0,
        "dip_fill_boost_peak_db": 0.0,
        "dip_fill_deep_narrow_count": 0,
        "dip_fill_risk_scoring_version": int(DIP_FILL_RISK_SCORING_VERSION),
    }
    f_full = _auto_stats_pick_arr(st, "freq_axis")
    # Pre-slice to relevant band before extracting other arrays — freq_axis may span
    # 0..Nyquist (100k+ pts) but the band tops out at hi_hz (default 250 Hz).
    _n_band = int(np.searchsorted(f_full, float(hi_hz) * 2.5, side="right")) if f_full.size >= 8 else f_full.size
    _n_lim = _n_band if _n_band >= 8 else f_full.size
    f = f_full[:_n_lim]
    measured = _auto_stats_pick_arr(st, "measured_mags")[:_n_lim]
    target = _auto_stats_pick_arr(st, "target_mags")[:_n_lim]
    filt = _auto_stats_pick_arr(st, "predicted_filter_mags", "realized_filter_mags", "filter_mags")[:_n_lim]
    n = int(min(f.size, measured.size, target.size, filt.size))
    if n < 8:
        return dict(out)
    f = np.asarray(f[:n], dtype=float)
    measured = np.asarray(measured[:n], dtype=float)
    target = np.asarray(target[:n], dtype=float)
    filt = np.asarray(filt[:n], dtype=float)
    mask = np.isfinite(f) & np.isfinite(measured) & np.isfinite(target) & np.isfinite(filt) & (f > 0.0) & (f >= lo_hz) & (f <= hi_hz)
    if int(np.count_nonzero(mask)) < 8:
        return dict(out)
    f_use = f[mask]
    pre_err = measured[mask] - target[mask]
    boost = np.maximum(0.0, filt[mask])
    dip_depth = np.maximum(0.0, -pre_err)
    risky = (dip_depth >= 3.0) & (boost >= 1.0)
    if int(np.count_nonzero(risky)) <= 0:
        return dict(out)
    x = np.log2(np.maximum(f_use, 1e-9))
    risk_vals = np.maximum(0.0, dip_depth - 3.0) * np.maximum(0.0, boost - 0.75)
    active = risky & np.isfinite(risk_vals)
    area = float(np.trapezoid(risk_vals[active], x=x[active])) if int(np.count_nonzero(active)) >= 2 else float(np.sum(risk_vals[active]) * 0.01)
    deep_narrow_count = 0
    starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    ends = np.flatnonzero(active & ~np.r_[active[1:], False])
    for start, end in zip(starts, ends, strict=False):
        if int(end) <= int(start):
            continue
        width_oct = float(x[int(end)] - x[int(start)])
        center_f = float(np.sqrt(f_use[int(start)] * f_use[int(end)]))
        if width_oct < _auto_min_broad_peak_width_oct(center_f) and float(np.max(dip_depth[int(start):int(end) + 1])) >= 6.0:
            deep_narrow_count += 1
    peak_boost = float(np.max(boost[active])) if int(np.count_nonzero(active)) else 0.0
    score = float(max(0.0, area) + 0.65 * max(0.0, peak_boost - 3.0) + 1.25 * deep_narrow_count)
    out.update(
        {
            "dip_fill_risk_score": float(score),
            "dip_fill_risk_penalty": float(np.clip(0.85 * score, 0.0, 10.0)),
            "dip_fill_boost_peak_db": float(peak_boost),
            "dip_fill_deep_narrow_count": int(deep_narrow_count),
        }
    )
    return dict(out)


def _auto_merge_dip_fill_risk_metrics(left_metrics: dict | None, right_metrics: dict | None) -> dict:
    out = {"dip_fill_risk_scoring_version": int(DIP_FILL_RISK_SCORING_VERSION)}
    for key in ("dip_fill_risk_score", "dip_fill_risk_penalty", "dip_fill_boost_peak_db"):
        vals = [
            shared._auto_safe_float((m or {}).get(key, float("nan")), float("nan"))
            for m in (left_metrics, right_metrics)
        ]
        vals = [float(v) for v in vals if np.isfinite(v)]
        out[key] = float(max(vals)) if vals else 0.0
    out["dip_fill_deep_narrow_count"] = int(
        sum(int(max(0, shared._auto_safe_float((m or {}).get("dip_fill_deep_narrow_count", 0), 0.0))) for m in (left_metrics, right_metrics))
    )
    return dict(out)


def _auto_channel_overfit_metrics_from_stats(
    left_st: dict | None,
    right_st: dict | None,
    *,
    lo_hz: float = 20.0,
    hi_hz: float = 250.0,
) -> dict:
    out = {
        "correction_lr_delta_rms_20_250": 0.0,
        "correction_lr_delta_max_20_250": 0.0,
        "narrow_lr_divergence_count": 0,
        "channel_overfit_penalty": 0.0,
        "channel_overfit_scoring_version": int(CHANNEL_OVERFIT_SCORING_VERSION),
    }
    f_l = _auto_stats_pick_arr(left_st, "freq_axis")
    f_r = _auto_stats_pick_arr(right_st, "freq_axis")
    g_l = _auto_stats_pick_arr(left_st, "predicted_filter_mags", "realized_filter_mags", "filter_mags")
    g_r = _auto_stats_pick_arr(right_st, "predicted_filter_mags", "realized_filter_mags", "filter_mags")
    n_l = int(min(f_l.size, g_l.size))
    n_r = int(min(f_r.size, g_r.size))
    if n_l < 8 or n_r < 8:
        return dict(out)
    f_l = np.asarray(f_l[:n_l], dtype=float)
    f_r = np.asarray(f_r[:n_r], dtype=float)
    g_l = np.asarray(g_l[:n_l], dtype=float)
    g_r = np.asarray(g_r[:n_r], dtype=float)
    mask_l = np.isfinite(f_l) & np.isfinite(g_l) & (f_l > 0.0) & (f_l >= lo_hz) & (f_l <= hi_hz)
    mask_r = np.isfinite(f_r) & np.isfinite(g_r) & (f_r > 0.0) & (f_r >= lo_hz) & (f_r <= hi_hz)
    if int(np.count_nonzero(mask_l)) < 8 or int(np.count_nonzero(mask_r)) < 8:
        return dict(out)
    f_grid = np.geomspace(max(lo_hz, max(float(np.min(f_l[mask_l])), float(np.min(f_r[mask_r])))), min(hi_hz, min(float(np.max(f_l[mask_l])), float(np.max(f_r[mask_r])))), 192)
    if f_grid.size < 8 or not np.all(np.isfinite(f_grid)):
        return dict(out)
    x_l = np.log2(np.maximum(f_l[mask_l], 1e-9))
    x_r = np.log2(np.maximum(f_r[mask_r], 1e-9))
    x_g = np.log2(np.maximum(f_grid, 1e-9))
    gl_i = np.interp(x_g, x_l, g_l[mask_l])
    gr_i = np.interp(x_g, x_r, g_r[mask_r])
    delta = gl_i - gr_i
    rms = float(np.sqrt(np.mean(delta * delta)))
    max_delta = float(np.max(np.abs(delta)))
    active = np.abs(delta) >= 3.0
    narrow_count = 0
    starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    ends = np.flatnonzero(active & ~np.r_[active[1:], False])
    x = np.log2(np.maximum(f_grid, 1e-9))
    for start, end in zip(starts, ends, strict=False):
        if int(end) <= int(start):
            continue
        width_oct = float(x[int(end)] - x[int(start)])
        center_f = float(np.sqrt(f_grid[int(start)] * f_grid[int(end)]))
        if width_oct < _auto_min_broad_peak_width_oct(center_f):
            narrow_count += 1
    penalty = float(
        np.clip(
            0.55 * max(0.0, rms - 1.5)
            + 0.30 * max(0.0, max_delta - 4.0)
            + 1.00 * max(0, narrow_count),
            0.0,
            8.0,
        )
    )
    out.update(
        {
            "correction_lr_delta_rms_20_250": float(rms),
            "correction_lr_delta_max_20_250": float(max_delta),
            "narrow_lr_divergence_count": int(narrow_count),
            "channel_overfit_penalty": float(penalty),
        }
    )
    return dict(out)



__all__ = ["CORRECTION_SHARPNESS_SCORING_VERSION", "DIP_FILL_RISK_SCORING_VERSION", "CHANNEL_OVERFIT_SCORING_VERSION", "_auto_correction_sharpness_metrics_from_stats", "_auto_merge_correction_sharpness_metrics", "_auto_dip_fill_risk_metrics_from_stats", "_auto_merge_dip_fill_risk_metrics", "_auto_channel_overfit_metrics_from_stats"]
