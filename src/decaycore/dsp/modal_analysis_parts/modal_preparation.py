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
import math
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("DecayCore.dsp")

# Try to import Rust DSP extension
try:
    from decaycore_dsp import smooth_log_box_kernel_rs as _smooth_log_box_kernel_rs

    _DSP_RUST_AVAILABLE = True
except ImportError:
    _DSP_RUST_AVAILABLE = False


MODAL_ANALYSIS_VERSION = 2


@dataclass(frozen=True)
class RoomModeEvent:
    freq_hz: float
    peak_db: float
    width_hz: float
    width_oct: float
    q_estimate: float
    area_db_oct: float

    gd_excess_ms: float = 0.0
    decay_severity: float = 0.0
    confidence: float = 1.0
    lr_consistency: float = 0.0
    repeatability: float = 0.0

    severity: float = 0.0
    correction_priority: float = 0.0
    cut_priority: float = 0.0
    safe_cut_db: float = 0.0
    safe_width_oct: float = 0.0
    voice_clarity_risk: float = 0.0

    kind: str = "unknown"
    reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ModalAnalysisResult:
    events: tuple[RoomModeEvent, ...]
    worst_mode_hz: float | None
    worst_mode_severity: float
    mode_count: int
    modal_area_db_oct: float
    voice_band_modal_risk: float
    analysis_version: int = MODAL_ANALYSIS_VERSION


def _empty_result() -> ModalAnalysisResult:
    return ModalAnalysisResult(
        events=(),
        worst_mode_hz=None,
        worst_mode_severity=0.0,
        mode_count=0,
        modal_area_db_oct=0.0,
        voice_band_modal_risk=0.0,
    )


def _as_float_array(value) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=float)
    try:
        return np.asarray(value, dtype=float).reshape(-1)
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
        return np.asarray([], dtype=float)


def _smooth_log_box_kernel(x: np.ndarray, y_raw: np.ndarray, half: float) -> np.ndarray:
    n = x.size
    sum_y = np.empty(n + 1)
    sum_w = np.empty(n + 1)
    sum_y[0] = 0.0
    sum_w[0] = 0.0
    for i in range(n):
        if math.isfinite(y_raw[i]):
            sum_y[i + 1] = sum_y[i] + y_raw[i]
            sum_w[i + 1] = sum_w[i] + 1.0
        else:
            sum_y[i + 1] = sum_y[i]
            sum_w[i + 1] = sum_w[i]
    out = np.empty(n)
    for i in range(n):
        lo = x[i] - half
        hi = x[i] + half
        left = 0
        right = n
        while left < right:
            mid = (left + right) >> 1
            if x[mid] < lo:
                left = mid + 1
            else:
                right = mid
        rl = 0
        rr = n
        while rl < rr:
            mid = (rl + rr) >> 1
            if x[mid] <= hi:
                rl = mid + 1
            else:
                rr = mid
        count = sum_w[rl] - sum_w[left]
        out[i] = (sum_y[rl] - sum_y[left]) / count if count > 0.0 else 0.0
    return out


def _smooth_log_box(freq_hz: np.ndarray, values: np.ndarray, width_oct: float) -> np.ndarray:
    f = np.asarray(freq_hz, dtype=float).reshape(-1)
    v = np.asarray(values, dtype=float).reshape(-1)
    if f.size < 3 or v.size != f.size:
        return np.copy(v)
    width = float(np.clip(width_oct, 1.0 / 192.0, 2.0))
    half = 0.5 * width
    log_f = np.log2(np.maximum(f, 1e-9))
    order = np.argsort(log_f)
    x = np.ascontiguousarray(log_f[order], dtype=float)
    y_raw = np.ascontiguousarray(v[order], dtype=float)
    out = np.empty_like(v, dtype=float)
    if _DSP_RUST_AVAILABLE:
        try:
            out[order] = _smooth_log_box_kernel_rs(x, y_raw, float(half))
            return out
        except (TypeError, ValueError) as exc:
            logger.warning("Rust log-box smoothing rejected its input; using Python fallback: %s", exc)
    out[order] = _smooth_log_box_kernel(x, y_raw, half)
    return out


def _safe_confidence(confidence_mask: np.ndarray, n: int) -> np.ndarray:
    if confidence_mask.size < n:
        return np.ones(n, dtype=float)
    conf = np.asarray(confidence_mask[:n], dtype=float)
    finite = conf[np.isfinite(conf)]
    if finite.size and float(np.nanmax(finite)) > 1.5:
        conf = conf / 100.0
    return np.clip(np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)


def _prepare_modal_series(
    freq_axis,
    measured_mag_db,
    target_mag_db,
    corrected_mag_db,
    group_delay_ms,
    confidence_mask,
    left_mag_db,
    right_mag_db,
) -> tuple[np.ndarray, ...]:
    f = _as_float_array(freq_axis)
    measured = _as_float_array(measured_mag_db)
    target = _as_float_array(target_mag_db)
    corrected = _as_float_array(corrected_mag_db)
    gd = _as_float_array(group_delay_ms)
    conf = _as_float_array(confidence_mask)
    left = _as_float_array(left_mag_db)
    right = _as_float_array(right_mag_db)
    return f, measured, target, corrected, gd, conf, left, right


def _prepare_modal_truncate_size(
    *,
    f: np.ndarray,
    measured: np.ndarray,
    target: np.ndarray,
    corrected: np.ndarray,
    gd: np.ndarray,
    conf: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> int:
    sizes = [f.size, measured.size]
    if target.size:
        sizes.append(target.size)
    if corrected.size:
        sizes.append(corrected.size)
    if gd.size:
        sizes.append(gd.size)
    if conf.size:
        sizes.append(conf.size)
    if left.size and right.size:
        sizes.extend([left.size, right.size])
    return int(min(sizes)) if sizes else 0


def _prepare_modal_analysis_array(
    *,
    measured: np.ndarray,
    target: np.ndarray,
    corrected: np.ndarray,
    n: int,
    target_mag_db,
) -> np.ndarray:
    if corrected.size >= n:
        return measured + corrected - target
    if target_mag_db is not None and target.size >= n:
        return measured - target
    return np.asarray(measured, dtype=float)


def _prepare_modal_valid_mask(
    *,
    f: np.ndarray,
    analysis: np.ndarray,
    conf: np.ndarray,
    gd: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    n: int,
    lo_hz: float,
    hi_hz: float,
) -> np.ndarray:
    valid = np.isfinite(f) & np.isfinite(analysis) & np.isfinite(conf) & (f > 0.0)
    valid &= (f >= float(lo_hz)) & (f <= float(hi_hz))
    if gd.size >= n:
        valid &= np.isfinite(gd)
    if left.size >= n and right.size >= n:
        valid &= np.isfinite(left) & np.isfinite(right)
    return valid


def _prepare_arrays(
    freq_axis,
    measured_mag_db,
    target_mag_db,
    corrected_mag_db,
    group_delay_ms,
    confidence_mask,
    left_mag_db,
    right_mag_db,
    lo_hz: float,
    hi_hz: float,
) -> tuple[np.ndarray, ...] | None:
    f, measured, target, corrected, gd, conf, left, right = _prepare_modal_series(
        freq_axis,
        measured_mag_db,
        target_mag_db,
        corrected_mag_db,
        group_delay_ms,
        confidence_mask,
        left_mag_db,
        right_mag_db,
    )
    n = _prepare_modal_truncate_size(
        f=f,
        measured=measured,
        target=target,
        corrected=corrected,
        gd=gd,
        conf=conf,
        left=left,
        right=right,
    )
    if n < 8:
        return None

    f = np.asarray(f[:n], dtype=float)
    measured = np.asarray(measured[:n], dtype=float)
    target = np.asarray(target[:n], dtype=float) if target.size >= n else np.zeros(n, dtype=float)
    corrected = np.asarray(corrected[:n], dtype=float) if corrected.size >= n else np.asarray([], dtype=float)
    gd = np.asarray(gd[:n], dtype=float) if gd.size >= n else np.asarray([], dtype=float)
    conf = _safe_confidence(conf, n)
    left = np.asarray(left[:n], dtype=float) if left.size >= n and right.size >= n else np.asarray([], dtype=float)
    right = np.asarray(right[:n], dtype=float) if right.size >= n and left.size >= n else np.asarray([], dtype=float)

    analysis = _prepare_modal_analysis_array(
        measured=measured,
        target=target,
        corrected=corrected,
        n=n,
        target_mag_db=target_mag_db,
    )
    valid = _prepare_modal_valid_mask(
        f=f,
        analysis=analysis,
        conf=conf,
        gd=gd,
        left=left,
        right=right,
        n=n,
        lo_hz=lo_hz,
        hi_hz=hi_hz,
    )
    if int(np.count_nonzero(valid)) < 8:
        return None

    f_use = np.asarray(f[valid], dtype=float)
    analysis_use = np.asarray(analysis[valid], dtype=float)
    conf_use = np.asarray(conf[valid], dtype=float)
    gd_use = np.asarray(gd[valid], dtype=float) if gd.size >= n else np.asarray([], dtype=float)
    left_use = (
        np.asarray(left[valid], dtype=float) if left.size >= n and right.size >= n else np.asarray([], dtype=float)
    )
    right_use = (
        np.asarray(right[valid], dtype=float) if right.size >= n and left.size >= n else np.asarray([], dtype=float)
    )

    order = np.argsort(f_use)
    return (
        f_use[order],
        analysis_use[order],
        conf_use[order],
        gd_use[order] if gd_use.size else gd_use,
        left_use[order] if left_use.size else left_use,
        right_use[order] if right_use.size else right_use,
    )


def _width_bounds(excess: np.ndarray, peak_idx: int, min_floor: float) -> tuple[int, int]:
    peak = float(max(0.0, excess[int(peak_idx)]))
    threshold = float(max(min_floor, peak - 3.0, 0.5 * peak))
    left = int(peak_idx)
    right = int(peak_idx)
    while left > 0 and np.isfinite(excess[left - 1]) and float(excess[left - 1]) > threshold:
        left -= 1
    while right < excess.size - 1 and np.isfinite(excess[right + 1]) and float(excess[right + 1]) > threshold:
        right += 1
    return int(left), int(right)


__all__ = [
    "RoomModeEvent",
    "ModalAnalysisResult",
    "_empty_result",
    "_as_float_array",
    "_smooth_log_box",
    "_safe_confidence",
    "_prepare_arrays",
    "_width_bounds",
]
