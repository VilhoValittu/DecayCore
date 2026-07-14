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

from .state_cache import _to_float


def _resample_series_inputs(
    freq_axis: np.ndarray,
    series: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    f = np.asarray(freq_axis, dtype=float).reshape(-1)
    prepared: list[np.ndarray] = []
    if f.size == 0:
        return f, prepared, np.zeros_like(f, dtype=bool)
    mask = np.isfinite(f) & (f > 0.0)
    for s in series:
        values = np.asarray(s, dtype=float).reshape(-1)
        if values.size != f.size:
            return f[:0], [], np.zeros((0,), dtype=bool)
        prepared.append(values)
        mask &= np.isfinite(values)
    return f, prepared, mask


def _sort_unique_resample_series(freq: np.ndarray, series: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    if freq.size == 0:
        return freq, series
    if freq.size > 1 and bool(np.all(np.diff(freq) > 0.0)):
        # Aidosti kasvavalle akselille stabiili argsort on identiteettipermutaatio
        # ja uniq-maski kaikkialla True, joten lajittelu olisi pelkka kopio.
        return freq, series
    order = np.argsort(freq, kind="mergesort")
    freq_sorted = freq[order]
    series_sorted = [values[order] for values in series]
    if freq_sorted.size > 1:
        uniq = np.concatenate(([True], np.diff(freq_sorted) > 0.0))
        freq_sorted = freq_sorted[uniq]
        series_sorted = [values[uniq] for values in series_sorted]
    return freq_sorted, series_sorted


def _log_resample_axis(
    freq: np.ndarray,
    *,
    points_per_octave: float,
    min_points: int,
) -> np.ndarray | None:
    if freq.size < 2:
        return None
    f_lo = float(freq[0])
    f_hi = float(freq[-1])
    if (not np.isfinite(f_lo)) or (not np.isfinite(f_hi)) or (f_hi <= f_lo):
        return None
    octaves = float(np.log2(f_hi / f_lo))
    if not np.isfinite(octaves) or octaves <= 0.0:
        return None
    ppo = float(points_per_octave)
    if (not np.isfinite(ppo)) or (ppo <= 0.0):
        ppo = 48.0
    n_points = max(int(np.ceil(octaves * ppo)) + 1, int(min_points))
    return np.geomspace(f_lo, f_hi, n_points)


def _is_identity_log_grid(
    freq_axis: np.ndarray,
    *,
    points_per_octave: float = 48.0,
    min_points: int = 32,
) -> bool:
    """True jos freq_axis on tasan _log_resample_axis-hila omista paatepisteistaan.

    Talloin _resample_log_axis samalle akselille (finiteilla sarjoilla) on
    bitti-identtinen identiteetti, ja kutsuja voi valittaa skip_resample=True
    alavirran sovituksiin.
    """
    try:
        f = np.asarray(freq_axis, dtype=float).reshape(-1)
        if f.size < 2 or not bool(np.all(np.isfinite(f))) or float(f[0]) <= 0.0:
            return False
        expected = _log_resample_axis(f, points_per_octave=points_per_octave, min_points=min_points)
        return expected is not None and bool(np.array_equal(expected, f))
    except (TypeError, ValueError, FloatingPointError, OverflowError):
        return False


def _resample_log_axis(
    freq_axis: np.ndarray,
    *series: np.ndarray,
    points_per_octave: float = 48.0,
    min_points: int = 32,
):
    try:
        f, prepared, mask = _resample_series_inputs(freq_axis, series)
        if f.size == 0:
            return f, tuple(np.asarray(s, dtype=float).reshape(-1)[:0] for s in series)

        f = f[mask]
        prepared = [v[mask] for v in prepared]
        if f.size == 0:
            return f, tuple(prepared)

        f, prepared = _sort_unique_resample_series(f, prepared)
        if f.size < 2:
            return f, tuple(prepared)

        f_log = _log_resample_axis(f, points_per_octave=points_per_octave, min_points=min_points)
        if f_log is None:
            return f, tuple(prepared)
        prepared_log = [np.interp(f_log, f, values) for values in prepared]
        return f_log, tuple(prepared_log)
    except (TypeError, ValueError, FloatingPointError, OverflowError):
        f = np.asarray(freq_axis, dtype=float).reshape(-1)
        prepared = tuple(np.asarray(s, dtype=float).reshape(-1) for s in series)
        return f, prepared


def _log_median(freq_axis: np.ndarray, values: np.ndarray, *, skip_resample: bool = False) -> float:
    try:
        if skip_resample:
            # Kutsuja takaa etta freq_axis on tuore _log_resample_axis-hila
            # (ks. _is_identity_log_grid); talloin resample on identiteetti.
            v = np.asarray(values, dtype=float).reshape(-1)
            if (
                v.size > 0
                and v.size == np.asarray(freq_axis, dtype=float).reshape(-1).size
                and bool(np.all(np.isfinite(v)))
            ):
                return float(np.median(v))
        _f_log, (v_log,) = _resample_log_axis(freq_axis, values)
        if v_log.size == 0:
            return 0.0
        return float(np.median(v_log))
    except (TypeError, ValueError, FloatingPointError):
        try:
            return float(np.median(values))
        except (TypeError, ValueError, FloatingPointError):
            return 0.0


def _lower_tail_robust_std_db(values: np.ndarray, *, clip_below_db: float = 6.0) -> float:
    try:
        y = np.asarray(values, dtype=float)
        y = y[np.isfinite(y)]
        if y.size == 0:
            return 0.0

        med = float(np.median(y))
        clip_below_db = _to_float(clip_below_db, 6.0)
        if clip_below_db > 0.0:
            keep = y >= (med - float(clip_below_db))
            if np.count_nonzero(keep) >= max(12, int(np.ceil(y.size * 0.4))):
                y = y[keep]

        center = float(np.median(y))
        mad = float(np.median(np.abs(y - center)))
        robust = 1.4826 * mad
        if np.isfinite(robust) and robust > 1e-9:
            return float(robust)
        return float(np.std(y))
    except (TypeError, ValueError, FloatingPointError):
        try:
            return float(np.std(values))
        except (TypeError, ValueError, FloatingPointError):
            return 0.0


def _centered_rms(values: np.ndarray) -> float:
    try:
        y = np.asarray(values, dtype=float)
        y = y[np.isfinite(y)]
        if y.size == 0:
            return float("inf")
        y = y - float(np.median(y))
        return float(np.sqrt(np.mean(y * y)))
    except (TypeError, ValueError, FloatingPointError):
        return float("inf")


def _tilt_fit_linear_log_axis(
    log_freq: np.ndarray,
    diff_db: np.ndarray,
    *,
    max_db_per_oct: float = 2.0,
) -> tuple[float, float]:
    try:
        x = np.asarray(log_freq, dtype=float).reshape(-1)
        y = np.asarray(diff_db, dtype=float).reshape(-1)
        if x.size != y.size or x.size < 20:
            off = float(np.median(y)) if y.size else 0.0
            return off, 0.0

        x0 = float(np.median(x))
        xc = x - x0

        y_med = float(np.median(y))
        dev = np.abs(y - y_med)
        if dev.size >= 30:
            thr = float(np.quantile(dev, 0.90))
            keep = dev <= thr
            if np.count_nonzero(keep) >= 20:
                xc = xc[keep]
                y = y[keep]

        y_med_kept = float(np.median(y))
        denom = float(np.dot(xc, xc))
        if denom <= 1e-12:
            return y_med_kept, 0.0

        slope = float(np.dot(xc, (y - y_med_kept)) / denom)
        max_db_per_oct = float(max_db_per_oct)
        if not np.isfinite(max_db_per_oct) or max_db_per_oct <= 0.0:
            max_db_per_oct = 2.0
        slope = float(np.clip(slope, -max_db_per_oct, +max_db_per_oct))

        off = float(np.median(y - slope * xc))
        if not np.isfinite(off):
            off = float(np.median(y))
        return float(off), float(slope)
    except (TypeError, ValueError, FloatingPointError):
        try:
            return float(np.median(diff_db)), 0.0
        except (TypeError, ValueError, FloatingPointError):
            return 0.0, 0.0


def _smooth_tilt_fit_series(y: np.ndarray, *, sigma_bins: float) -> np.ndarray:
    try:
        arr = np.asarray(y, dtype=float).reshape(-1)
        if arr.size < 8:
            return arr.copy()
        sigma = float(sigma_bins)
        if not np.isfinite(sigma) or sigma <= 0.0:
            return arr.copy()
        sigma = float(max(1.0, sigma))
        half = int(np.ceil(3.0 * sigma))
        if half < 1:
            return arr.copy()
        xx = np.arange(-half, half + 1, dtype=float)
        kk = np.exp(-0.5 * (xx / sigma) ** 2)
        k_sum = float(np.sum(kk))
        if (not np.isfinite(k_sum)) or k_sum <= 1e-12:
            return arr.copy()
        kk /= k_sum
        arr_pad = np.pad(arr, (half, half), mode="edge")
        return np.convolve(arr_pad, kk, mode="valid")
    except (TypeError, ValueError, FloatingPointError):
        try:
            return np.asarray(y, dtype=float).copy()
        except (TypeError, ValueError):
            return np.asarray([], dtype=float)


def _tilt_fit_lf_piecewise_prepare(
    log_freq: np.ndarray,
    diff_db: np.ndarray,
    *,
    max_db_per_oct: float = 2.0,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    try:
        x = np.asarray(log_freq, dtype=float).reshape(-1)
        y_raw = np.asarray(diff_db, dtype=float).reshape(-1)
        if x.size != y_raw.size or x.size < 24:
            return None

        x_span = float(x[-1] - x[0])
        if (not np.isfinite(x_span)) or x_span < 0.85:
            return None

        x0 = float(np.median(x))
        xl = np.minimum(x - x0, 0.0)
        xr = np.maximum(x - x0, 0.0)
        if np.count_nonzero(xl < 0.0) < 10 or np.count_nonzero(xr > 0.0) < 10:
            return None

        y_fit = y_raw.copy()
        if y_fit.size >= 30:
            y_med = float(np.median(y_fit))
            thr = float(np.quantile(np.abs(y_fit - y_med), 0.90))
            if np.isfinite(thr) and thr > 1e-9:
                y_fit = np.clip(y_fit, y_med - thr, y_med + thr)

        step = float(np.median(np.diff(x)))
        if (not np.isfinite(step)) or step <= 0.0:
            return None
        y_smooth = _smooth_tilt_fit_series(y_fit, sigma_bins=0.55 / max(step, 1e-6))
        if y_smooth.size != x.size:
            return None

        return x, x0, xl, xr, y_raw, y_smooth
    except (TypeError, ValueError, FloatingPointError, IndexError):
        return None


def _tilt_fit_lf_piecewise_evaluate(
    x: np.ndarray,
    x0: float,
    xl: np.ndarray,
    xr: np.ndarray,
    y_raw: np.ndarray,
    y_smooth: np.ndarray,
    *,
    linear_offset: float,
    linear_slope: float,
    max_db_per_oct: float = 2.0,
) -> tuple[float, float] | None:
    design = np.column_stack([np.ones_like(x), xl, xr])
    try:
        coeffs = np.linalg.lstsq(design, y_smooth, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    if np.asarray(coeffs).size != 3 or not np.all(np.isfinite(coeffs)):
        return None

    max_db_per_oct = float(max_db_per_oct)
    if not np.isfinite(max_db_per_oct) or max_db_per_oct <= 0.0:
        max_db_per_oct = 2.0
    slope_lo = float(np.clip(float(coeffs[1]), -max_db_per_oct, +max_db_per_oct))
    slope_hi = float(np.clip(float(coeffs[2]), -max_db_per_oct, +max_db_per_oct))
    if abs(slope_hi - slope_lo) < 0.35:
        return None

    basis_piecewise = (slope_lo * xl) + (slope_hi * xr)
    off_piecewise = float(np.median(y_raw - basis_piecewise))
    if not np.isfinite(off_piecewise):
        return None

    linear_pred = float(linear_offset) + (float(linear_slope) * (x - x0))
    piecewise_pred = off_piecewise + basis_piecewise
    linear_rms = _centered_rms(y_smooth - linear_pred)
    piecewise_rms = _centered_rms(y_smooth - piecewise_pred)
    if (not np.isfinite(piecewise_rms)) or (not np.isfinite(linear_rms)):
        return None
    if piecewise_rms > max(linear_rms * 0.88, linear_rms - 0.08):
        return None
    if abs(off_piecewise - float(linear_offset)) > 1.5:
        return None

    span_lo = max(float(x0 - x[0]), 1e-9)
    span_hi = max(float(x[-1] - x0), 1e-9)
    slope_eq = ((slope_lo * span_lo) + (slope_hi * span_hi)) / (span_lo + span_hi)
    slope_eq = float(np.clip(slope_eq, -max_db_per_oct, +max_db_per_oct))
    return float(off_piecewise), float(slope_eq)


def _tilt_fit_lf_piecewise_log_axis(
    log_freq: np.ndarray,
    diff_db: np.ndarray,
    *,
    linear_offset: float,
    linear_slope: float,
    max_db_per_oct: float = 2.0,
) -> tuple[float, float] | None:
    try:
        prepared = _tilt_fit_lf_piecewise_prepare(
            log_freq,
            diff_db,
            max_db_per_oct=max_db_per_oct,
        )
        if prepared is None:
            return None
        x, x0, xl, xr, y_raw, y_smooth = prepared
        return _tilt_fit_lf_piecewise_evaluate(
            x,
            x0,
            xl,
            xr,
            y_raw,
            y_smooth,
            linear_offset=linear_offset,
            linear_slope=linear_slope,
            max_db_per_oct=max_db_per_oct,
        )
    except (TypeError, ValueError, FloatingPointError, IndexError):
        return None


def _window_offset_consistency_score(
    freq_axis: np.ndarray,
    measured_db: np.ndarray,
    target_db: np.ndarray | None,
    *,
    tilt_comp: bool = True,
    tilt_max_db_per_oct: float = 2.0,
    assume_log_grid: bool = False,
) -> tuple[float, float, float]:
    """Arvioi kuinka luotettava level-offset on ikkunan sisalla.

    Palauttaa kolmen metriikan tuplen:
    1) offset-spread (dB) eri alajaksojen valilla
    2) shape RMS targetiin nahden (offset/tilt poistettuna)
    3) absoluuttinen tilt (dB/okt)
    """
    try:
        f, diff = _window_consistency_prepare_inputs(freq_axis, measured_db, target_db)
        if f.size < 24 or diff.size < 24:
            return float("inf"), float("inf"), float("inf")

        # Identiteettiohitus vain jos valid-maski ei pudottanut yhtaan binia,
        # muuten f ei ole enaa kutsujan takaama tuore log-hila.
        skip = bool(assume_log_grid) and f.size == int(np.size(freq_axis))
        if tilt_comp:
            off_full, slope_full = _tilt_fit_offset_and_slope_db_per_oct(
                f,
                diff,
                max_db_per_oct=float(tilt_max_db_per_oct),
                skip_resample=skip,
            )
        else:
            off_full = _log_median(f, diff, skip_resample=skip)
            slope_full = 0.0

        x = np.log2(np.clip(f, 1e-9, None))
        xc = x - float(np.median(x))
        shape_resid = diff - float(off_full) - (float(slope_full) * xc)
        shape_rms = _centered_rms(shape_resid)

        offsets = _window_offset_candidates(
            f,
            diff,
            tilt_comp=bool(tilt_comp),
            tilt_max_db_per_oct=float(tilt_max_db_per_oct),
            full_offset=float(off_full),
        )

        if len(offsets) >= 3:
            offset_spread = float(np.std(np.asarray(offsets, dtype=float)))
        else:
            offset_spread = 0.0

        return float(offset_spread), float(shape_rms), float(abs(slope_full))
    except (TypeError, ValueError, FloatingPointError, IndexError):
        return float("inf"), float("inf"), float("inf")


def _window_consistency_prepare_inputs(
    freq_axis: np.ndarray,
    measured_db: np.ndarray,
    target_db: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    f = np.asarray(freq_axis, dtype=float).reshape(-1)
    m = np.asarray(measured_db, dtype=float).reshape(-1)
    if target_db is None:
        t = np.zeros_like(m, dtype=float)
    else:
        t = np.asarray(target_db, dtype=float).reshape(-1)
    if f.size < 24 or m.size != f.size or t.size != f.size:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    valid = np.isfinite(f) & np.isfinite(m) & np.isfinite(t) & (f > 0.0)
    if int(np.count_nonzero(valid)) < 24:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    f_valid = f[valid]
    diff = np.asarray(m[valid] - t[valid], dtype=float)
    return f_valid, diff


def _window_offset_candidates(
    freq: np.ndarray,
    diff: np.ndarray,
    *,
    tilt_comp: bool,
    tilt_max_db_per_oct: float,
    full_offset: float,
) -> list[float]:
    log_f = np.log2(np.clip(freq, 1e-9, None))
    log_f_lo = float(log_f[0])
    log_f_span = float(log_f[-1]) - log_f_lo
    parts: list[tuple[float, float]] = [
        (0.0, 0.5),
        (0.5, 1.0),
        (0.0, 2.0 / 3.0),
        (1.0 / 3.0, 1.0),
        (0.2, 0.8),
    ]
    offsets = [float(full_offset)]
    for start_frac, end_frac in parts:
        f_lo_sub = 2.0 ** (log_f_lo + start_frac * log_f_span)
        f_hi_sub = 2.0 ** (log_f_lo + end_frac * log_f_span)
        sub_mask = (freq >= f_lo_sub) & (freq <= f_hi_sub)
        if int(np.count_nonzero(sub_mask)) < 10:
            continue
        f_sub = freq[sub_mask]
        d_sub = diff[sub_mask]
        if tilt_comp:
            off_sub, _ = _tilt_fit_offset_and_slope_db_per_oct(
                f_sub,
                d_sub,
                max_db_per_oct=float(tilt_max_db_per_oct),
            )
        else:
            off_sub = _log_median(f_sub, d_sub)
        if np.isfinite(off_sub):
            offsets.append(float(off_sub))
    return offsets


def _tilt_aware_offset_db(
    freq_axis: np.ndarray,
    diff_db: np.ndarray,
    *,
    max_db_per_oct: float = 2.0,
) -> float:

    try:
        f, (y,) = _resample_log_axis(freq_axis, diff_db)
        if f.size != y.size or f.size < 20:
            return float(np.median(y)) if y.size else 0.0

        x = np.log2(f)

        x0 = float(np.median(x))
        xc = x - x0

        y_med = float(np.median(y))
        dev = np.abs(y - y_med)
        if dev.size >= 30:
            thr = float(np.quantile(dev, 0.90))
            keep = dev <= thr
            if np.count_nonzero(keep) >= 20:
                xc = xc[keep]
                y = y[keep]

        denom = float(np.dot(xc, xc))
        if denom <= 1e-12:
            return float(np.median(y))

        slope = float(np.dot(xc, (y - float(np.median(y)))) / denom)
        max_db_per_oct = float(max_db_per_oct)
        if not np.isfinite(max_db_per_oct) or max_db_per_oct <= 0:
            max_db_per_oct = 2.0
        slope = float(np.clip(slope, -max_db_per_oct, +max_db_per_oct))

        offset = float(np.median(y - slope * xc))
        if not np.isfinite(offset):
            offset = float(np.median(y)) if y.size else 0.0
        return float(offset)
    except (TypeError, ValueError, FloatingPointError):
        try:
            return float(np.median(diff_db))
        except (TypeError, ValueError, FloatingPointError):
            return 0.0


def _tilt_fit_offset_and_slope_db_per_oct(
    freq_axis: np.ndarray,
    diff_db: np.ndarray,
    *,
    max_db_per_oct: float = 2.0,
    prefer_lf_piecewise_tilt: bool = False,
    skip_resample: bool = False,
):
    """Sisainen apufunktio: tilt fit offset and slope db per oct.

    skip_resample: kutsuja takaa etta freq_axis on tuore _log_resample_axis-hila
    (ks. _is_identity_log_grid), jolloin sisainen resample olisi bitti-identtinen
    identiteetti ja voidaan ohittaa. Guardin failaus palauttaa taydelle polulle.
    """
    try:
        f = None
        y = None
        if skip_resample:
            f_in = np.asarray(freq_axis, dtype=float).reshape(-1)
            y_in = np.asarray(diff_db, dtype=float).reshape(-1)
            if f_in.size == y_in.size and bool(np.all(np.isfinite(y_in))):
                f, y = f_in, y_in
        if f is None:
            f, (y,) = _resample_log_axis(freq_axis, diff_db)
        if f.size != y.size or f.size < 20:
            off = float(np.median(y)) if y.size else 0.0
            return off, 0.0

        x = np.log2(f)
        off, slope = _tilt_fit_linear_log_axis(x, y, max_db_per_oct=float(max_db_per_oct))
        if bool(prefer_lf_piecewise_tilt) and float(f[-1]) <= 220.0:
            piecewise = _tilt_fit_lf_piecewise_log_axis(
                x,
                y,
                linear_offset=float(off),
                linear_slope=float(slope),
                max_db_per_oct=float(max_db_per_oct),
            )
            if piecewise is not None:
                return piecewise
        return off, slope
    except (TypeError, ValueError, FloatingPointError):
        try:
            return float(np.median(diff_db)), 0.0
        except (TypeError, ValueError, FloatingPointError):
            return 0.0, 0.0


__all__ = [
    "_is_identity_log_grid",
    "_resample_log_axis",
    "_log_median",
    "_lower_tail_robust_std_db",
    "_centered_rms",
    "_tilt_fit_linear_log_axis",
    "_smooth_tilt_fit_series",
    "_tilt_fit_lf_piecewise_log_axis",
    "_window_offset_consistency_score",
    "_tilt_aware_offset_db",
    "_tilt_fit_offset_and_slope_db_per_oct",
]
