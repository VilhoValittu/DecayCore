# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

from dataclasses import dataclass

import numpy as np
import scipy.signal
import scipy.ndimage
import scipy.stats

from decaycore.dsp.cache_utils import BoundedLruCache
from decaycore.dsp.mode_q import estimate_peak_q


@dataclass(frozen=True)
class Rt60WindowPolicy:
    """RT60-sovitusikkunoiden politiikka.

    Oletusarvot vastaavat ISO 3382 -kaytantoa:
    - EDT 0..-10 dB: early decay time, herkka varhaisille heijastuksille.
    - T20 -5..-25 dB: ohittaa suoran aanen, sietaa korkeahkon pohjakohinan.
    - T30 -5..-35 dB: tarkin, mutta vaatii ~45 dB:n mittausdynamiikan.
    Kandidaateista valitaan T30 > T20 > EDT -jarjestyksessa ensimmainen,
    jonka R^2 >= r2_threshold; muuten paras saatavilla oleva sovitus.

    Ei-standardeille impulssivasteille (esim. korkea pohjakohina, hyvin
    lyhyt hanta) voi antaa matalammat ikkunat - silloin estimaatti on
    karkeampi mutta sovitus onnistuu useammin.
    """

    edt_lo_db: float = 0.0
    edt_hi_db: float = -10.0
    t20_lo_db: float = -5.0
    t20_hi_db: float = -25.0
    t30_lo_db: float = -5.0
    t30_hi_db: float = -35.0
    min_fit_points: int = 12
    r2_threshold: float = 0.90


DEFAULT_RT60_WINDOW_POLICY = Rt60WindowPolicy()

_SOS_CACHE = BoundedLruCache(256)


def clear_analysis_cache() -> None:
    """Clear module-level analysis caches used across runs."""
    _SOS_CACHE.clear()


def _get_bandpass_sos(nyq: float, order: int, fl: float, fh: float) -> np.ndarray:
    key = (round(nyq, 1), int(order), round(fl, 4), round(fh, 4))
    sos = _SOS_CACHE.get(key)
    if sos is None:
        sos = scipy.signal.butter(order, [fl / nyq, fh / nyq], btype="bandpass", output="sos")
        _SOS_CACHE.put(key, sos)
    return sos


def _sigma_bins_from_hz(freq_axis, sigma_hz: float, fallback_bins: float = 3.0) -> float:
    try:
        f = np.asarray(freq_axis, dtype=float)
        if f.size < 4:
            return float(fallback_bins)
        df = np.median(np.diff(f))
        if not np.isfinite(df) or df <= 0:
            return float(fallback_bins)
        s = float(sigma_hz) / float(df)
        if not np.isfinite(s) or s <= 0:
            return float(fallback_bins)
        return float(max(1.0, s))
    except (TypeError, ValueError, FloatingPointError):
        return float(fallback_bins)


def _distance_bins_from_hz(freq_axis, distance_hz: float, fallback_bins: int = 100) -> int:
    try:
        f = np.asarray(freq_axis, dtype=float)
        if f.size < 4:
            return int(fallback_bins)
        df = float(np.median(np.diff(f)))
        if not np.isfinite(df) or df <= 0.0:
            return int(fallback_bins)
        bins = int(round(float(distance_hz) / df))
        return int(max(1, bins))
    except (TypeError, ValueError, FloatingPointError):
        return int(fallback_bins)


# Theil-Sen on O(k^2) pareittain; EDC-ikkuna voi olla kymmenia tuhansia
# naytteita, joten sovitus tehdaan tasavalisesti harvennetulle osajoukolle.
_RT60_FIT_MAX_POINTS = 256


def _moving_average_same(values: np.ndarray, window_size: int) -> np.ndarray:
    """Return an O(n) moving average matching ``np.convolve(..., 'same')``.

    The asymmetric one-sample padding for even windows intentionally mirrors
    NumPy's centering convention.  RT60 inputs always exceed the smoothing
    window; retain NumPy's legacy behavior for other internal callers.
    """
    arr = np.asarray(values, dtype=float).reshape(-1)
    win = int(window_size)
    if arr.size == 0 or win <= 1:
        return arr.copy()
    if win > arr.size:
        return np.convolve(arr, np.ones(win, dtype=float) / float(win), mode="same")

    padded = np.pad(arr, (win // 2, (win - 1) // 2), mode="constant")
    prefix = np.empty(padded.size + 1, dtype=float)
    prefix[0] = 0.0
    np.cumsum(padded, dtype=float, out=prefix[1:])
    return (prefix[win:] - prefix[:-win]) / float(win)


def _fit_rt60_window(t_u: np.ndarray, d_u: np.ndarray, lo_db: float, hi_db: float, *, min_points: int = 12):
    mask = (d_u <= lo_db) & (d_u >= hi_db)
    if np.count_nonzero(mask) < int(min_points):
        return None
    tt = t_u[mask]
    yy = d_u[mask]
    if tt.size > _RT60_FIT_MAX_POINTS:
        sel = np.linspace(0, tt.size - 1, _RT60_FIT_MAX_POINTS).astype(int)
        ts, ys = tt[sel], yy[sel]
    else:
        ts, ys = tt, yy
    try:
        a, b, _lo, _hi = scipy.stats.theilslopes(ys, ts)
    except (ValueError, FloatingPointError):
        return None
    yhat = a * tt + b
    ss_res = float(np.sum((yy - yhat) ** 2))
    ss_tot = float(np.sum((yy - np.mean(yy)) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    if a >= -1e-9:
        return None
    rt60 = -60.0 / a
    return rt60, r2


def _pick_rt60_candidate(candidates: list[tuple[str, float, float]], *, r2_threshold: float = 0.90):
    if not candidates:
        return None
    pref = {"T30": 0, "T20": 1, "EDT": 2}
    candidates.sort(key=lambda x: (pref[x[0]], -x[2]))
    for name, rt60, r2 in candidates:
        if r2 >= float(r2_threshold):
            return rt60, r2, name
    name, rt60, r2 = candidates[0]
    return rt60, r2, name


def calculate_group_delay(freqs, phases_deg):
    phase_rad = np.unwrap(np.deg2rad(phases_deg))
    d_phi_d_f = np.gradient(phase_rad, freqs)
    gd_ms = -d_phi_d_f / (2 * np.pi) * 1000.0
    sigma_bins = _sigma_bins_from_hz(freqs, sigma_hz=2.0, fallback_bins=3.0)
    return scipy.ndimage.gaussian_filter1d(gd_ms, sigma=sigma_bins)


_GD_THR_FLOOR_MS = 2.5
_GD_THR_CEIL_MS = 8.0
_GD_THR_MIN_BAND_SAMPLES = 16


def _local_gd_threshold_curve(freq_axis, gd_diff, valid_idx, global_thr_ms):
    """Taajuuslokaali GD-kynnyskayra: p70 oktaavikaistoittain, log-f-interpolointi.

    Kaistat, joissa on liian vahan naytteita, perivat globaalin kynnyksen.
    Kaistakohtainen kynnys rajataan samaan [2.5, 8.0] ms ikkunaan kuin
    globaali kynnys, joten lokaali painotus ei voi loysata kokonaisrajoja.
    """
    out = np.full(len(freq_axis), float(global_thr_ms), dtype=float)
    if valid_idx.size < 4 * _GD_THR_MIN_BAND_SAMPLES:
        return out
    f_v = np.asarray(freq_axis, dtype=float)[valid_idx]
    g_v = np.asarray(gd_diff, dtype=float)[valid_idx]
    f_lo = max(20.0, float(f_v[0]))
    f_hi = float(f_v[-1])
    if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or f_hi <= f_lo * 1.5:
        return out
    n_bands = int(np.ceil(np.log2(f_hi / f_lo)))
    edges = f_lo * 2.0 ** np.arange(n_bands + 1, dtype=float)
    edges[-1] = f_hi
    centers: list[float] = []
    thrs: list[float] = []
    for e0, e1 in zip(edges[:-1], edges[1:]):
        m = (f_v >= e0) & (f_v <= e1)
        if int(np.count_nonzero(m)) < _GD_THR_MIN_BAND_SAMPLES:
            thr = float(global_thr_ms)
        else:
            p70 = float(np.percentile(g_v[m], 70.0))
            thr = float(np.clip(max(_GD_THR_FLOOR_MS, p70), _GD_THR_FLOOR_MS, _GD_THR_CEIL_MS))
        centers.append(float(np.sqrt(e0 * e1)))
        thrs.append(thr)
    if len(centers) < 2:
        return out
    log_f = np.log2(np.maximum(np.asarray(freq_axis, dtype=float), 1e-3))
    return np.interp(log_f, np.log2(np.asarray(centers)), np.asarray(thrs))


def analyze_acoustic_confidence(freq_axis, complex_meas, fs):
    phase_rad = np.unwrap(np.angle(complex_meas))
    df = np.gradient(freq_axis) + 1e-12
    gd_s = -np.gradient(phase_rad) / (2 * np.pi * df)
    gd_ms = gd_s * 1000.0

    sigma_bins = _sigma_bins_from_hz(freq_axis, sigma_hz=6.7, fallback_bins=10.0)
    gd_smooth = scipy.ndimage.gaussian_filter1d(gd_ms, sigma=sigma_bins)
    gd_diff = np.abs(gd_ms - gd_smooth)

    valid_idx = np.where(freq_axis > 20)[0]
    gd_eval = gd_diff[valid_idx] if valid_idx.size else gd_diff
    if gd_eval.size:
        p70 = float(np.percentile(gd_eval, 70.0))
        threshold_ms = float(np.clip(max(2.5, p70), 2.5, 8.0))
    else:
        threshold_ms = 2.5
    threshold_curve = _local_gd_threshold_curve(freq_axis, gd_diff, valid_idx, threshold_ms)
    x = 1.5 * (gd_diff - threshold_curve)
    x = np.clip(x, -60.0, 60.0)
    confidence_mask = 1.0 / (1.0 + np.exp(x))
    peaks = np.array([], dtype=int)

    reflection_nodes = []
    if valid_idx.size > 0:
        peak_distance = _distance_bins_from_hz(freq_axis[valid_idx], distance_hz=120.0, fallback_bins=100)
        peak_height = max(2.0, 0.8 * threshold_ms)
        peaks, _props = scipy.signal.find_peaks(
            gd_diff[valid_idx],
            height=peak_height,
            distance=peak_distance,
        )

    raw_nodes = []
    q_vals, _bw_hz = estimate_peak_q(
        freq_axis[valid_idx], gd_diff[valid_idx], peaks, min_bw_ratio=0.02
    )
    for p, q_val in zip(peaks, q_vals):
        idx = valid_idx[p]
        f_peak = float(freq_axis[idx])
        peak_val = float(gd_diff[idx])
        q_est = round(float(q_val), 1)

        raw_nodes.append({
            "freq": round(f_peak, 1),
            "gd_error": round(peak_val, 2),
            "dist": round((peak_val / 1000.0 * 343.0) / 2.0, 2),
            "q_est": q_est,
            "type": "Resonance" if f_peak < 200.0 else "Reflection",
        })

    reflection_nodes = sorted(raw_nodes, key=lambda x: x["gd_error"], reverse=True)[:15]
    return confidence_mask, reflection_nodes, gd_ms


def calculate_rt60(impulse, fs, policy: Rt60WindowPolicy | None = None):
    pol = policy if policy is not None else DEFAULT_RT60_WINDOW_POLICY
    try:
        imp = np.asarray(impulse, dtype=float)
        if imp.size < int(0.1 * fs):
            return 0.0

        peak_idx = int(np.argmax(np.abs(imp)))
        x = imp[peak_idx:]
        if x.size < int(0.05 * fs):
            return 0.0

        e = x * x

        tail_n = max(int(0.15 * e.size), int(0.05 * fs))
        tail_n = min(tail_n, e.size)
        noise_power = float(np.mean(e[-tail_n:]))

        E = np.cumsum(e[::-1])[::-1]
        E0 = float(E[0]) + 1e-18

        noise_mult = 20.0
        stop_candidates = np.where(noise_mult * noise_power >= E)[0]
        stop_idx = int(stop_candidates[0]) if stop_candidates.size > 0 else (E.size - 1)
        stop_idx = max(stop_idx, 10)

        t = np.arange(E.size) / fs
        edc_db = 10.0 * np.log10((E / E0) + 1e-30)

        smooth_ms = 10.0
        win = max(1, int((smooth_ms / 1000.0) * fs))
        if win > 1:
            edc_db = _moving_average_same(edc_db, win)

        t_u = t[:stop_idx + 1]
        d_u = edc_db[:stop_idx + 1]

        candidates = []
        r = _fit_rt60_window(t_u, d_u, pol.edt_lo_db, pol.edt_hi_db, min_points=pol.min_fit_points)
        if r:
            candidates.append(("EDT",) + r)
        r = _fit_rt60_window(t_u, d_u, pol.t20_lo_db, pol.t20_hi_db, min_points=pol.min_fit_points)
        if r:
            candidates.append(("T20",) + r)
        r = _fit_rt60_window(t_u, d_u, pol.t30_lo_db, pol.t30_hi_db, min_points=pol.min_fit_points)
        if r:
            candidates.append(("T30",) + r)

        chosen = _pick_rt60_candidate(candidates, r2_threshold=pol.r2_threshold)
        if chosen is None:
            return 0.0

        rt60 = float(chosen[0])
        if 0.05 < rt60 < 5.0:
            return round(rt60, 2)
        return 0.0

    except (TypeError, ValueError, FloatingPointError, IndexError):
        return 0.0


def _third_oct_centers(f_min=31.5, f_max=8000.0):
    centers = []
    f = float(f_min)
    step = 2 ** (1 / 3)
    while f <= f_max * 1.0001:
        centers.append(float(f))
        f *= step
    return centers


def calculate_rt60_bands(impulse, fs, f_min=31.5, f_max=8000.0, order=4, policy: Rt60WindowPolicy | None = None):
    try:
        imp = np.asarray(impulse, dtype=float)
        if imp.size < int(0.1 * fs):
            return {}

        nyq = 0.5 * fs
        centers = _third_oct_centers(f_min, min(f_max, nyq * 0.90))
        out = {}

        for fc in centers:
            fl = fc / (2 ** (1 / 6))
            fh = fc * (2 ** (1 / 6))
            fl = max(1.0, fl)
            fh = min(nyq * 0.98, fh)
            if fh <= fl * 1.05:
                continue

            sos = _get_bandpass_sos(nyq, order, fl, fh)
            x = scipy.signal.sosfiltfilt(sos, imp)
            rt = calculate_rt60(x, fs, policy)
            if 0.05 < rt < 5.0:
                out[float(round(fc, 2))] = float(rt)
        return out
    except (TypeError, ValueError, FloatingPointError, IndexError):
        return {}
