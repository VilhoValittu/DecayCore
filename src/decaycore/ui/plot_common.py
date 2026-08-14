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
import scipy.ndimage

from ..dsp.smoothing import apply_smoothing_std, psychoacoustic_smoothing

PHASE_SMOOTH_OCT = 0.1
GD_SMOOTH_OCT = 0.1
GD_SMOOTH_SIGMA = 0.1
logger = logging.getLogger("DecayCore")

_RECOVERABLE_PLOT_EXCEPTIONS = (
    AttributeError,
    TypeError,
    ValueError,
    RuntimeError,
    OverflowError,
    FloatingPointError,
)


def _maybe_shift_to_abs(mags_db, avg_t_db):
    try:
        a = np.asarray(mags_db, dtype=float)
        if a.size == 0:
            return a
        med = float(np.nanmedian(a))
        # Treat strongly negative curves as already absolute low-reference data.
        # Only near-zero relative curves should be lifted to the target level.
        if np.isfinite(med) and (-20.0 < med < 40.0):
            return a + float(avg_t_db)
        return a
    except _RECOVERABLE_PLOT_EXCEPTIONS:
        return np.asarray(mags_db, dtype=float)


def _align_meas_to_target_window(freqs_hz, meas_db, targ_db, f_min_hz, f_max_hz):
    try:
        f = np.asarray(freqs_hz, dtype=float)
        m = np.asarray(meas_db, dtype=float)
        t = np.asarray(targ_db, dtype=float)
        if f.size < 16 or m.size != f.size or t.size != f.size:
            return m
        f_min = float(f_min_hz)
        f_max = float(f_max_hz)
        if not (np.isfinite(f_min) and np.isfinite(f_max) and f_min > 0 and f_max > f_min):
            return m
        mask = (f >= f_min) & (f <= f_max) & np.isfinite(m) & np.isfinite(t)
        if np.count_nonzero(mask) < 20:
            return m
        off = float(np.median(m[mask] - t[mask]))
        if not np.isfinite(off):
            return m
        return m - off
    except _RECOVERABLE_PLOT_EXCEPTIONS:
        return np.asarray(meas_db, dtype=float)


def _prepare_curve_for_target_plot(
    freqs_hz,
    mags_db,
    *,
    avg_t_db,
    target_freqs_hz=None,
    target_mags_db=None,
    f_min_hz=500.0,
    f_max_hz=2000.0,
):
    try:
        f = np.asarray(freqs_hz, dtype=float)
        m = np.asarray(mags_db, dtype=float)
        tf = np.asarray(target_freqs_hz if target_freqs_hz is not None else [], dtype=float)
        tm = np.asarray(target_mags_db if target_mags_db is not None else [], dtype=float)
        if f.size < 2 or m.size != f.size:
            return m
        if tf.size < 2 or tm.size != tf.size:
            return m
        m_abs = _maybe_shift_to_abs(m, avg_t_db)
        t_abs = _maybe_shift_to_abs(tm, avg_t_db)
        t_on_f = np.interp(f, tf, t_abs)
        mask = (f >= float(f_min_hz)) & (f <= float(f_max_hz)) & np.isfinite(m_abs) & np.isfinite(t_on_f)
        if np.count_nonzero(mask) >= 4:
            off = float(np.median(m_abs[mask] - t_on_f[mask]))
            if np.isfinite(off):
                return m_abs - off
        return m_abs
    except _RECOVERABLE_PLOT_EXCEPTIONS:
        return np.asarray(mags_db, dtype=float)


def _confidence_prepare_arrays(freqs, conf_mask):
    try:
        f = np.asarray(freqs, dtype=float)
        c = np.asarray(conf_mask, dtype=float)
    except _RECOVERABLE_PLOT_EXCEPTIONS:
        return None
    if f.size != c.size or f.size < 8:
        return None
    valid = np.isfinite(f) & np.isfinite(c) & (f > 0.0)
    if np.count_nonzero(valid) < 8:
        return None
    return f[valid], c[valid]


def _confidence_raw_segments(f: np.ndarray, bad: np.ndarray) -> list[tuple[float, float]]:
    raw: list[tuple[float, float]] = []
    in_seg = False
    seg_start = None
    for fx, is_bad in zip(f, bad):
        if is_bad and not in_seg:
            in_seg = True
            seg_start = float(fx)
        elif (not is_bad) and in_seg:
            in_seg = False
            seg_end = float(fx)
            if seg_start is not None and seg_end > seg_start:
                raw.append((float(seg_start), float(seg_end)))
    if in_seg and seg_start is not None and float(f[-1]) > float(seg_start):
        raw.append((float(seg_start), float(f[-1])))
    return list(raw)


def _merge_confidence_segments(
    raw: list[tuple[float, float]],
    *,
    min_gap_hz: float,
) -> list[list[float]]:
    merged: list[list[float]] = []
    for start, end in raw:
        if not merged:
            merged.append([float(start), float(end)])
            continue
        prev = merged[-1]
        if float(start - prev[1]) <= float(min_gap_hz):
            prev[1] = max(float(prev[1]), float(end))
        else:
            merged.append([float(start), float(end)])
    return merged


def _filter_confidence_segments(
    merged: list[list[float]],
    *,
    min_width_hz: float,
) -> list[tuple[float, float]]:
    kept = [(float(start), float(end)) for start, end in merged if float(end - start) >= float(min_width_hz)]
    if kept:
        return kept
    return [(float(start), float(end)) for start, end in merged]


def _limit_confidence_segments(
    kept: list[tuple[float, float]],
    *,
    max_segments: int,
) -> list[tuple[float, float]]:
    if len(kept) <= int(max_segments):
        return kept
    limited = sorted(
        kept,
        key=lambda seg: (-(float(seg[1]) - float(seg[0])), float(seg[0])),
    )[: int(max_segments)]
    return sorted(limited, key=lambda seg: float(seg[0]))


def _confidence_bad_segments(
    freqs,
    conf_mask,
    *,
    thr: float = 0.35,
    min_width_hz: float = 30.0,
    min_gap_hz: float = 35.0,
    max_segments: int = 24,
):
    prepared = _confidence_prepare_arrays(freqs, conf_mask)
    if prepared is None:
        return []
    f, c = prepared
    bad = np.asarray(c < float(thr), dtype=bool)
    if not np.any(bad):
        return []

    raw = _confidence_raw_segments(f, bad)
    if not raw:
        return []

    merged = _merge_confidence_segments(raw, min_gap_hz=float(min_gap_hz))
    kept = _filter_confidence_segments(merged, min_width_hz=float(min_width_hz))
    return _limit_confidence_segments(kept, max_segments=int(max_segments))


def smooth_complex(freqs, spec, oct_frac=1.0):
    real_parts = np.nan_to_num(np.real(spec))
    imag_parts = np.nan_to_num(np.imag(spec))
    real_s, _ = apply_smoothing_std(freqs, real_parts, np.zeros_like(freqs), oct_frac)
    imag_s, _ = apply_smoothing_std(freqs, imag_parts, np.zeros_like(freqs), oct_frac)
    return real_s + 1j * imag_s


def calculate_clean_gd(freqs, complex_resp, *, sigma: float = GD_SMOOTH_SIGMA):
    phase_rad = np.unwrap(np.angle(complex_resp))
    df = np.gradient(freqs) + 1e-12
    gd_ms = -np.gradient(phase_rad) / (2 * np.pi * df) * 1000.0
    gd_ms = np.nan_to_num(gd_ms, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        sigma_v = float(sigma)
    except _RECOVERABLE_PLOT_EXCEPTIONS:
        sigma_v = float(GD_SMOOTH_SIGMA)
    if sigma_v > 0.0:
        gd_ms = scipy.ndimage.gaussian_filter1d(gd_ms, sigma=sigma_v)
    return gd_ms


def remove_ir_peak_delay(freqs, complex_resp, ir, fs):
    try:
        f = np.asarray(freqs, dtype=float).reshape(-1)
        h = np.asarray(complex_resp, dtype=complex).reshape(-1)
        x = np.asarray(ir, dtype=float).reshape(-1)
        fs_v = float(fs)
        if f.size == 0 or h.size != f.size or x.size == 0 or fs_v <= 0.0:
            return h, 0.0
        peak_idx = int(np.argmax(np.abs(x)))
        if peak_idx <= 0:
            return h, 0.0
        delay_s = float(peak_idx) / fs_v
        rot = np.exp(1j * 2.0 * np.pi * f * delay_s)
        return h * rot, delay_s * 1000.0
    except _RECOVERABLE_PLOT_EXCEPTIONS:
        return np.asarray(complex_resp, dtype=complex).reshape(-1), 0.0


def _filter_focus_band(freqs, filt_db, *, delta_db: float = 0.75) -> tuple[float, float] | None:
    try:
        f = np.asarray(freqs, dtype=float).reshape(-1)
        g = np.asarray(filt_db, dtype=float).reshape(-1)
        valid = np.isfinite(f) & np.isfinite(g) & (f > 0.0)
        if np.count_nonzero(valid) < 16:
            return None
        fv = f[valid]
        gv = g[valid]
        order = np.argsort(fv, kind="mergesort")
        fv = fv[order]
        gv = gv[order]
        hi_start = int(max(0, round(0.8 * (fv.size - 1))))
        baseline = float(np.median(gv[hi_start:])) if hi_start < gv.size else float(np.median(gv))
        active = np.abs(gv - baseline) >= float(max(0.1, delta_db))
        if np.count_nonzero(active) < 8:
            return None
        lo = float(fv[np.where(active)[0][0]])
        hi = float(fv[np.where(active)[0][-1]])
        lo = max(float(fv[0]), lo / 1.35)
        hi = min(float(fv[-1]), hi * 1.6)
        if hi <= lo:
            return None
        return lo, hi
    except _RECOVERABLE_PLOT_EXCEPTIONS:
        return None


def _axis_valid_mask(
    freqs: np.ndarray,
    values: np.ndarray,
    *,
    focus_band: tuple[float, float] | None,
) -> np.ndarray:
    valid = np.isfinite(freqs) & np.isfinite(values)
    if focus_band is not None:
        lo_f, hi_f = focus_band
        valid &= (freqs >= float(lo_f)) & (freqs <= float(hi_f))
    if np.count_nonzero(valid) < 12:
        valid = np.isfinite(values)
    return valid


def _axis_quantile_bounds(values: np.ndarray, q_lo: float, q_hi: float) -> tuple[float, float]:
    lo = float(np.quantile(values, float(q_lo)))
    hi = float(np.quantile(values, float(q_hi)))
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        return lo, hi
    lo = float(np.min(values))
    hi = float(np.max(values))
    return lo, hi


def _axis_apply_padding(lo: float, hi: float, *, pad_ratio: float) -> tuple[float, float]:
    span = float(hi - lo)
    return lo - span * float(pad_ratio), hi + span * float(pad_ratio)


def _axis_apply_min_span(lo: float, hi: float, *, min_span: float) -> tuple[float, float]:
    if (hi - lo) >= float(min_span):
        return lo, hi
    mid = 0.5 * (hi + lo)
    return mid - 0.5 * float(min_span), mid + 0.5 * float(min_span)


def _axis_include_zero(lo: float, hi: float) -> tuple[float, float]:
    return min(lo, 0.0), max(hi, 0.0)


def _axis_apply_max_span(lo: float, hi: float, *, max_span: float, include_zero: bool) -> tuple[float, float]:
    if (hi - lo) <= float(max_span):
        return lo, hi
    center = 0.0 if include_zero else 0.5 * (hi + lo)
    return center - 0.5 * float(max_span), center + 0.5 * float(max_span)


def _robust_axis_range(
    freqs,
    values,
    *,
    focus_band: tuple[float, float] | None = None,
    q_lo: float = 0.03,
    q_hi: float = 0.97,
    pad_ratio: float = 0.18,
    min_span: float = 10.0,
    max_span: float | None = None,
    include_zero: bool = False,
):
    try:
        f = np.asarray(freqs, dtype=float).reshape(-1)
        y = np.asarray(values, dtype=float).reshape(-1)
        valid = _axis_valid_mask(f, y, focus_band=focus_band)
        if np.count_nonzero(valid) < 4:
            return None
        yy = y[valid]
        lo, hi = _axis_quantile_bounds(yy, q_lo, q_hi)
        if hi <= lo:
            mid = float(lo)
            lo = mid - 0.5 * float(min_span)
            hi = mid + 0.5 * float(min_span)
        lo, hi = _axis_apply_padding(lo, hi, pad_ratio=pad_ratio)
        if include_zero:
            lo, hi = _axis_include_zero(lo, hi)
        lo, hi = _axis_apply_min_span(lo, hi, min_span=min_span)
        if include_zero:
            lo, hi = _axis_include_zero(lo, hi)
        if max_span is not None:
            lo, hi = _axis_apply_max_span(lo, hi, max_span=float(max_span), include_zero=include_zero)
        return [float(lo), float(hi)]
    except _RECOVERABLE_PLOT_EXCEPTIONS:
        return None


def _view_mags_for_plot(freqs, mags, *, plot_smoothing_level="Psychoacoustic"):
    f = np.asarray(freqs, dtype=float)
    m = np.asarray(mags, dtype=float)

    if f.size == 0 or m.size == 0:
        return m

    psl = plot_smoothing_level

    if isinstance(psl, str) and ("psy" in psl.strip().lower()):
        return psychoacoustic_smoothing(f, m)

    try:
        lvl = int(psl)
    except _RECOVERABLE_PLOT_EXCEPTIONS:
        return m

    lvl = max(1, lvl)
    out, _ = apply_smoothing_std(f, m, np.zeros_like(m), 1.0 / float(lvl))
    return out
