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

from .metrics_common import _auto_stats_band_n, _auto_stats_pick_arr

VOICE_CLARITY_SCORING_VERSION = 1

_VOICE_KEYS = (
    "voice_clarity_penalty",
    "voice_band_rms_error_db",
    "voice_band_peak_excess_db",
    "voice_band_energy_excess_db",
    "voice_band_lr_mismatch_db",
    "voice_band_gd_peak_ms",
    "voice_band_risk_mean",
    "voice_band_risk_peak",
)


def _empty_voice_metrics() -> dict[str, float]:
    return dict.fromkeys(_VOICE_KEYS, 0.0)


def _slice_optional(arr, idx: np.ndarray):
    """Slice an optional array/None by index array; return None if not sliceable."""
    if arr is None:
        return None
    try:
        a = np.asarray(arr, dtype=float).reshape(-1)
        if a.size >= int(idx[-1]) + 1:
            return a[idx]
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
        pass
    return None


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


def _aligned_array(value, n: int, default: float = 0.0) -> np.ndarray:
    arr = _as_float_array(value)
    if arr.size == n:
        return np.nan_to_num(arr.astype(float, copy=False), nan=default, posinf=default, neginf=default)
    if arr.size == 1:
        return np.full(n, float(arr[0]), dtype=float)
    return np.full(n, float(default), dtype=float)


def _valid_array(value, n: int) -> tuple[np.ndarray, bool]:
    arr = _as_float_array(value)
    if arr.size != n:
        return np.zeros(n, dtype=float), False
    return np.nan_to_num(arr.astype(float, copy=False), nan=0.0, posinf=0.0, neginf=0.0), True


def _voice_band_weight(freq_axis: np.ndarray) -> np.ndarray:
    f = np.asarray(freq_axis, dtype=float)
    out = np.zeros_like(f, dtype=float)
    out = np.where((f >= 55.0) & (f < 80.0), (f - 55.0) / 25.0, out)
    out = np.where((f >= 80.0) & (f <= 160.0), 1.0, out)
    out = np.where((f > 160.0) & (f <= 220.0), 1.0 - (f - 160.0) / 60.0, out)
    return np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0.0)
    if int(np.count_nonzero(mask)) <= 0:
        return 0.0
    w_use = np.asarray(w[mask], dtype=float)
    total = float(np.sum(w_use))
    if not np.isfinite(total) or total <= 1e-12:
        return 0.0
    return float(np.sum(np.asarray(v[mask], dtype=float) * w_use) / total)


def _weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    return float(np.sqrt(max(0.0, _weighted_mean(v * v, weights))))


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0.0)
    if int(np.count_nonzero(mask)) <= 0:
        return 0.0
    v_use = np.asarray(v[mask], dtype=float)
    w_use = np.asarray(w[mask], dtype=float)
    order = np.argsort(v_use)
    v_sorted = v_use[order]
    w_sorted = w_use[order]
    total = float(np.sum(w_sorted))
    if not np.isfinite(total) or total <= 1e-12:
        return 0.0
    threshold = float(np.clip(percentile, 0.0, 100.0)) / 100.0 * total
    idx = int(np.searchsorted(np.cumsum(w_sorted), threshold, side="left"))
    idx = int(np.clip(idx, 0, v_sorted.size - 1))
    return float(v_sorted[idx])


def _weighted_positive_peak(values: np.ndarray, weights: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0.0)
    if int(np.count_nonzero(mask)) <= 0:
        return 0.0
    return float(max(0.0, np.max(np.maximum(0.0, v[mask]))))


def compute_voice_clarity_penalty(
    freq_axis,
    corrected_mag_db,
    target_mag_db=None,
    authority_voice_risk=None,
    authority_modal_support=None,
    authority_null_risk=None,
    left_corrected_mag_db=None,
    right_corrected_mag_db=None,
    group_delay_ms=None,
    *,
    lo_hz: float = 70.0,
    hi_hz: float = 180.0,
) -> dict:
    out = _empty_voice_metrics()
    freq = _as_float_array(freq_axis)
    corrected = _as_float_array(corrected_mag_db)
    n = int(min(freq.size, corrected.size))
    if n < 4:
        return dict(out)

    freq = np.asarray(freq[:n], dtype=float)
    corrected = np.asarray(corrected[:n], dtype=float)
    if not (np.all(np.isfinite(freq)) and np.any(np.isfinite(corrected))):
        return dict(out)

    # Pre-slice to the voice band with margin — freq_axis may span 0..Nyquist (100k+ pts)
    # but all useful computation is in a narrow band (typically ~30-250 Hz).
    _lo_guard = max(0.0, float(lo_hz) * 0.5)
    _hi_guard = float(hi_hz) * 2.5
    _band_mask = (freq >= _lo_guard) & (freq <= _hi_guard)
    if int(np.count_nonzero(_band_mask)) >= 4:
        _idx = np.where(_band_mask)[0]
        freq = freq[_idx]
        corrected = corrected[_idx]
        n = freq.size
        target_mag_db = None if target_mag_db is None else np.asarray(target_mag_db, dtype=float).reshape(-1)
        if target_mag_db is not None and target_mag_db.size >= _idx[-1] + 1:
            target_mag_db = target_mag_db[_idx]
        authority_voice_risk = _slice_optional(authority_voice_risk, _idx)
        authority_modal_support = _slice_optional(authority_modal_support, _idx)
        authority_null_risk = _slice_optional(authority_null_risk, _idx)
        left_corrected_mag_db = _slice_optional(left_corrected_mag_db, _idx)
        right_corrected_mag_db = _slice_optional(right_corrected_mag_db, _idx)
        group_delay_ms = _slice_optional(group_delay_ms, _idx)

    target = _aligned_array(target_mag_db, n, 0.0)
    weights = _voice_band_weight(freq)
    weights = np.where((freq >= float(lo_hz)) & (freq <= float(hi_hz)), weights, 0.0)
    weights = np.where(np.isfinite(freq) & np.isfinite(corrected) & np.isfinite(target), weights, 0.0)
    if int(np.count_nonzero(weights > 0.0)) < 2:
        return dict(out)

    err_db = np.nan_to_num(corrected - target, nan=0.0, posinf=0.0, neginf=0.0)
    positive = np.maximum(0.0, err_db)
    rms_error_db = _weighted_rms(err_db, weights)
    peak_excess_db = _weighted_positive_peak(err_db, weights)
    energy_excess_db = _weighted_mean(positive, weights)

    voice_risk, voice_valid = _valid_array(authority_voice_risk, n)
    modal_support, _modal_valid = _valid_array(authority_modal_support, n)
    null_risk, _null_valid = _valid_array(authority_null_risk, n)
    voice_risk = np.clip(voice_risk, 0.0, 1.0)
    modal_support = np.clip(modal_support, 0.0, 1.0)
    null_risk = np.clip(null_risk, 0.0, 1.0)

    risk_mean = _weighted_mean(voice_risk, weights) if voice_valid else 0.0
    risk_peak = _weighted_percentile(voice_risk, weights, 95.0) if voice_valid else 0.0
    risk_factor = 0.65 + 0.70 * risk_peak if voice_valid else 1.0
    modal_factor = 1.0 + 0.35 * _weighted_mean(modal_support, weights)
    null_relief = float(np.clip(1.0 - 0.45 * _weighted_mean(null_risk, weights), 0.55, 1.0))

    excess_penalty = (
        0.35 * max(0.0, rms_error_db - 1.5) / 3.0
        + 0.45 * max(0.0, peak_excess_db - 2.5) / 5.0
        + 0.20 * max(0.0, energy_excess_db - 1.0) / 3.0
    )
    excess_penalty = float(np.clip(excess_penalty, 0.0, 2.0))

    lr_mismatch_db = 0.0
    left, left_valid = _valid_array(left_corrected_mag_db, n)
    right, right_valid = _valid_array(right_corrected_mag_db, n)
    if left_valid and right_valid:
        lr_mismatch_db = _weighted_percentile(np.abs(left - right), weights, 95.0)
    lr_penalty = max(0.0, lr_mismatch_db - 2.5) / 5.0

    gd_peak_ms = 0.0
    gd, gd_valid = _valid_array(group_delay_ms, n)
    if gd_valid:
        gd_peak_ms = _weighted_percentile(np.abs(gd), weights, 95.0)
    gd_penalty = max(0.0, gd_peak_ms - 25.0) / 50.0

    penalty = (
        excess_penalty * float(risk_factor) * float(modal_factor) * float(null_relief)
        + 0.25 * float(lr_penalty)
        + 0.20 * float(gd_penalty)
    )
    out.update(
        {
            "voice_clarity_penalty": float(np.clip(penalty, 0.0, 3.0)),
            "voice_band_rms_error_db": float(max(0.0, rms_error_db)),
            "voice_band_peak_excess_db": float(max(0.0, peak_excess_db)),
            "voice_band_energy_excess_db": float(max(0.0, energy_excess_db)),
            "voice_band_lr_mismatch_db": float(max(0.0, lr_mismatch_db)),
            "voice_band_gd_peak_ms": float(max(0.0, gd_peak_ms)),
            "voice_band_risk_mean": float(np.clip(risk_mean, 0.0, 1.0)),
            "voice_band_risk_peak": float(np.clip(risk_peak, 0.0, 1.0)),
        }
    )
    return {key: float(np.nan_to_num(out[key], nan=0.0, posinf=0.0, neginf=0.0)) for key in _VOICE_KEYS}


def _corrected_response_from_stats(
    st: dict | None, *, _max_n: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    st = dict(st or {})
    freq = _auto_stats_pick_arr(st, "freq_axis", _max_n=_max_n)
    measured = _auto_stats_pick_arr(st, "measured_mags", _max_n=_max_n)
    filt = _auto_stats_pick_arr(st, "realized_filter_mags", "filter_mags", "predicted_filter_mags", _max_n=_max_n)
    target = _auto_stats_pick_arr(st, "target_mags", _max_n=_max_n)
    n = int(min(freq.size, measured.size, filt.size))
    if n < 4:
        return np.asarray([], dtype=float), np.asarray([], dtype=float), np.asarray([], dtype=float)
    freq = np.asarray(freq[:n], dtype=float)
    corrected = np.asarray(measured[:n], dtype=float) + np.asarray(filt[:n], dtype=float)
    target_n = target[:n] if target.size >= n else target
    target_use = _aligned_array(target_n, n, 0.0)
    return freq, corrected, target_use


def voice_clarity_metrics_from_stats(
    st: dict | None,
    *,
    lo_hz: float = 70.0,
    hi_hz: float = 180.0,
) -> dict:
    st = dict(st or {})
    _hi_g = float(hi_hz) * 2.5
    _n: int | None = _auto_stats_band_n(st, _hi_g)

    freq, corrected, target = _corrected_response_from_stats(st, _max_n=_n)

    def _opt(key: str, *fallbacks: str) -> np.ndarray:
        return _auto_stats_pick_arr(st, key, *fallbacks, _max_n=_n)

    gd = _opt("group_delay_ms", "gd_ms", "gd_curve_ms")
    return compute_voice_clarity_penalty(
        freq,
        corrected,
        target_mag_db=target,
        authority_voice_risk=_opt("authority_voice_risk", "voice_risk"),
        authority_modal_support=_opt("authority_modal_support", "modal_support"),
        authority_null_risk=_opt("authority_null_risk", "null_risk"),
        group_delay_ms=gd if gd.size else None,
        lo_hz=lo_hz,
        hi_hz=hi_hz,
    )


def _interp_log_freq(src_freq: np.ndarray, src_values: np.ndarray, dst_freq: np.ndarray) -> np.ndarray:
    if src_freq.size < 2 or src_values.size < 2 or dst_freq.size == 0:
        return np.asarray([], dtype=float)
    n = int(min(src_freq.size, src_values.size))
    f = np.asarray(src_freq[:n], dtype=float)
    v = np.asarray(src_values[:n], dtype=float)
    mask = np.isfinite(f) & np.isfinite(v) & (f > 0.0)
    if int(np.count_nonzero(mask)) < 2:
        return np.asarray([], dtype=float)
    x = np.log2(f[mask])
    order = np.argsort(x)
    return np.interp(np.log2(np.maximum(dst_freq, 1e-9)), x[order], v[mask][order])


def voice_clarity_lr_metrics_from_stats(
    left_st: dict | None,
    right_st: dict | None,
    *,
    lo_hz: float = 70.0,
    hi_hz: float = 180.0,
) -> dict:
    # Quick peek at freq_axis to compute band-limited n before the expensive
    # corrected = measured + filt addition in _corrected_response_from_stats.
    _hi_g = float(hi_hz) * 2.5
    lf, lc, lt = _corrected_response_from_stats(left_st, _max_n=_auto_stats_band_n(left_st, _hi_g))
    rf, rc, rt = _corrected_response_from_stats(right_st, _max_n=_auto_stats_band_n(right_st, _hi_g))
    if lf.size < 4 or rf.size < 4:
        return _empty_voice_metrics()

    lf_finite = lf[np.isfinite(lf)]
    rf_finite = rf[np.isfinite(rf)]
    if lf_finite.size == 0 or rf_finite.size == 0:
        return _empty_voice_metrics()
    lo = max(float(lo_hz), float(np.min(lf_finite)), float(np.min(rf_finite)))
    hi = min(float(hi_hz), float(np.max(lf_finite)), float(np.max(rf_finite)))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return _empty_voice_metrics()
    grid = np.geomspace(max(1e-6, lo), hi, 192)
    li = _interp_log_freq(lf, lc, grid)
    ri = _interp_log_freq(rf, rc, grid)
    lti = _interp_log_freq(lf, lt, grid)
    rti = _interp_log_freq(rf, rt, grid)
    if li.size != grid.size or ri.size != grid.size:
        return _empty_voice_metrics()
    target = 0.5 * (lti + rti) if lti.size == grid.size and rti.size == grid.size else np.zeros_like(grid)
    corrected = 0.5 * (li + ri)
    return compute_voice_clarity_penalty(
        grid,
        corrected,
        target_mag_db=target,
        left_corrected_mag_db=li,
        right_corrected_mag_db=ri,
        lo_hz=lo_hz,
        hi_hz=hi_hz,
    )


def merge_voice_clarity_metrics(*metrics: dict | None) -> dict:
    out = _empty_voice_metrics()
    for key in _VOICE_KEYS:
        vals = []
        for metric in metrics:
            try:
                v = float((metric or {}).get(key, 0.0))
            except (TypeError, ValueError, OverflowError):
                v = 0.0
            if np.isfinite(v):
                vals.append(float(v))
        out[key] = float(max(vals)) if vals else 0.0
    return dict(out)


__all__ = [
    "VOICE_CLARITY_SCORING_VERSION",
    "compute_voice_clarity_penalty",
    "merge_voice_clarity_metrics",
    "voice_clarity_lr_metrics_from_stats",
    "voice_clarity_metrics_from_stats",
]
