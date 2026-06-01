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
import scipy.signal
import scipy.ndimage

_LOGGER = logging.getLogger(__name__)

def _clamp01(x):
    return np.clip(x, 0.0, 1.0)

_LOG_GRAD_F_FLOOR = 10.0  # Hz; prevents inflation of roughness below this frequency

def _log_grad(x, f):
    df = np.gradient(f) + 1e-12
    dx_df = np.gradient(x) / df
    return dx_df * np.maximum(f, _LOG_GRAD_F_FLOOR)

def _baseline_heavy(freqs, mags_db, sigma_hz=6.0):
    df = np.median(np.diff(freqs[np.isfinite(freqs)])) if len(freqs) > 2 else 1.0
    sigma_bins = max(1.0, float(sigma_hz / max(df, 1e-9)))
    return scipy.ndimage.gaussian_filter1d(mags_db, sigma=sigma_bins)

def _freq_prior(freqs, f1=120.0, f2=200.0):
    p = np.ones_like(freqs, dtype=float)
    if f1 >= f2:
        return p
    p[freqs >= f2] = 0.0
    mid = (freqs >= f1) & (freqs < f2)
    p[mid] = 1.0 - (freqs[mid] - f1) / (f2 - f1)
    return p

def _adaptive_mode_weights(rt60_lf_s: float | None, peak_q_values: list[float]) -> tuple[float, float, float]:
    w_gd, w_mag, w_q = 0.45, 0.35, 0.20

    if rt60_lf_s is not None and np.isfinite(float(rt60_lf_s)) and float(rt60_lf_s) > 0.05:
        _rt60 = float(rt60_lf_s)
        if _rt60 <= 0.4:
            w_gd, w_mag, w_q = 0.35, 0.45, 0.20
        elif _rt60 >= 1.2:
            w_gd, w_mag, w_q = 0.60, 0.25, 0.15
        else:
            t = (_rt60 - 0.4) / (1.2 - 0.4)
            w_gd = 0.45 + t * (0.60 - 0.45)
            w_mag = 0.35 + t * (0.25 - 0.35)
            w_q = 0.20 + t * (0.15 - 0.20)

    if len(peak_q_values) > 0:
        mean_q = float(np.mean(peak_q_values))
        if mean_q > 7.0:
            delta = 0.05 * min(1.0, (mean_q - 7.0) / 3.0)
            w_mag -= delta
            w_q += delta
            total = w_gd + w_mag + w_q + 1e-12
            w_gd /= total
            w_mag /= total
            w_q /= total

    _LOGGER.debug(f"room_mode_weights: gd={w_gd:.3f} mag={w_mag:.3f} q={w_q:.3f} (rt60={rt60_lf_s}, mean_q={np.mean(peak_q_values) if peak_q_values else 'none'})")
    return w_gd, w_mag, w_q

def _wav_source_relaxed_thresholds(
    *,
    is_wav_source: bool,
    rough_r0: float,
    rough_r1: float,
    pj_p0: float,
    pj_p1: float,
    gd_t0: float,
    gd_t1: float,
) -> tuple[float, float, float, float, float, float]:
    if not bool(is_wav_source):
        return float(rough_r0), float(rough_r1), float(pj_p0), float(pj_p1), float(gd_t0), float(gd_t1)
    try:
        return (
            float(rough_r0) * 1.5,
            float(rough_r1) * 1.5,
            float(pj_p0) * 2.0,
            float(pj_p1) * 2.0,
            float(gd_t0) * 1.2,
            float(gd_t1) * 1.2,
        )
    except (TypeError, ValueError):
        return float(rough_r0), float(rough_r1), float(pj_p0), float(pj_p1), float(gd_t0), float(gd_t1)


def _rt60_tightened_gd_thresholds(*, rt60_lf_s: float | None, gd_t0: float, gd_t1: float) -> tuple[float, float]:
    if rt60_lf_s is None:
        return float(gd_t0), float(gd_t1)
    try:
        rt60 = float(rt60_lf_s)
        if np.isfinite(rt60) and rt60 > 0.05:
            tighten = float(np.clip(1.0 - 0.375 * min(1.0, max(0.0, (rt60 - 0.4) / 0.8)), 0.7, 1.0))
            if tighten < 0.999:
                return float(gd_t0) * tighten, float(gd_t1) * tighten
    except (TypeError, ValueError, FloatingPointError):
        pass
    return float(gd_t0), float(gd_t1)


def _mode_q_norm(f: np.ndarray, mag_peak: np.ndarray, *, q0: float, q1: float) -> tuple[np.ndarray, list[float]]:
    q_norm = np.zeros_like(f)
    peak_q_values: list[float] = []
    try:
        peaks, _props = scipy.signal.find_peaks(mag_peak, prominence=1.0, distance=max(3, int(0.02 * len(f))))
        if len(peaks) > 0:
            results_half = scipy.signal.peak_widths(mag_peak, peaks, rel_height=0.5)
            widths_bins = results_half[0]
            for peak_idx, width_bins in zip(peaks, widths_bins):
                lo = max(0, int(peak_idx - width_bins / 2))
                hi = min(len(f) - 1, int(peak_idx + width_bins / 2))
                bw_hz = max(1e-6, f[hi] - f[lo])
                q_value = f[peak_idx] / bw_hz if bw_hz > 0 else 0.0
                peak_q_values.append(q_value)
                q_norm[peak_idx] = _clamp01((q_value - q0) / (q1 - q0 + 1e-12))
            q_norm = scipy.ndimage.gaussian_filter1d(q_norm, sigma=6)
    except (TypeError, ValueError, FloatingPointError, IndexError):
        pass
    return np.asarray(q_norm, dtype=float), list(peak_q_values)


def _apply_room_mode_taper(
    *,
    f: np.ndarray,
    room_mode_mask: np.ndarray,
    rew_asym: bool,
    left_ms: float,
) -> np.ndarray:
    try:
        rew_asym_flag = bool(rew_asym)
        try:
            left_ms_val = float(left_ms)
        except (TypeError, ValueError):
            left_ms_val = 0.0
        if not np.isfinite(left_ms_val):
            left_ms_val = 0.0

        if rew_asym_flag and (left_ms_val < 15.0):
            f1, f2 = 60.0, 80.0
        else:
            f1, f2 = 20.0, 30.0
        taper = np.ones_like(f, dtype=float)
        taper[f <= f1] = 0.0
        mid = (f > f1) & (f < f2)
        taper[mid] = 0.5 - 0.5 * np.cos(np.pi * (f[mid] - f1) / (f2 - f1))
        return np.asarray(room_mode_mask, dtype=float) * taper
    except (TypeError, ValueError, FloatingPointError):
        return np.asarray(room_mode_mask, dtype=float)


def build_bassfirst_masks(freq_axis, m_raw_db, phase_rad_unwrapped, gd_ms, gd_diff,
                          is_wav_source=False, mode_f1=120.0, mode_f2=200.0,
                          rew_asym: bool = False, left_ms: float = 0.0,
                          gd_t0=1.0, gd_t1=6.0,
                          mag_a0=1.5, mag_a1=8.0,
                          q0=2.0, q1=10.0,
                          rough_r0=0.6, rough_r1=2.5,
                          pj_p0=0.0008, pj_p1=0.0040,
                          rt60_lf_s: float | None = None):

    f = np.asarray(freq_axis, dtype=float)
    m = np.asarray(m_raw_db, dtype=float)
    ph = np.asarray(phase_rad_unwrapped, dtype=float)
    rough_r0, rough_r1, pj_p0, pj_p1, gd_t0, gd_t1 = _wav_source_relaxed_thresholds(
        is_wav_source=bool(is_wav_source),
        rough_r0=float(rough_r0),
        rough_r1=float(rough_r1),
        pj_p0=float(pj_p0),
        pj_p1=float(pj_p1),
        gd_t0=float(gd_t0),
        gd_t1=float(gd_t1),
    )
    gd_t0, gd_t1 = _rt60_tightened_gd_thresholds(
        rt60_lf_s=rt60_lf_s,
        gd_t0=float(gd_t0),
        gd_t1=float(gd_t1),
    )

    prior = _freq_prior(f, f1=mode_f1, f2=mode_f2)
    gd_norm = _clamp01((gd_diff - gd_t0) / (gd_t1 - gd_t0 + 1e-12))

    base = _baseline_heavy(f, m, sigma_hz=8.0)
    mag_peak = np.maximum(0.0, m - base)
    mag_norm = _clamp01((mag_peak - mag_a0) / (mag_a1 - mag_a0 + 1e-12))

    q_norm, peak_q_values = _mode_q_norm(
        f,
        mag_peak,
        q0=float(q0),
        q1=float(q1),
    )

    w_gd, w_mag, w_q = _adaptive_mode_weights(rt60_lf_s, peak_q_values)
    mode_score = prior * (w_gd*gd_norm + w_mag*mag_norm + w_q*q_norm)
    room_mode_mask = scipy.ndimage.gaussian_filter1d(_clamp01(mode_score), sigma=4)

    if bool(is_wav_source):
        mag_only_mode = scipy.ndimage.gaussian_filter1d(_clamp01(mag_norm * prior), sigma=4)
        room_mode_mask = np.maximum(room_mode_mask, mag_only_mode)

    room_mode_mask = _apply_room_mode_taper(
        f=f,
        room_mode_mask=room_mode_mask,
        rew_asym=bool(rew_asym),
        left_ms=float(left_ms),
    )

    g = np.abs(_log_grad(m, f))
    rough = scipy.ndimage.gaussian_filter1d(g, sigma=6)
    rough_norm = _clamp01((rough - rough_r0) / (rough_r1 - rough_r0 + 1e-12))

    df = np.gradient(f) + 1e-12
    d1 = np.gradient(ph) / df
    d2 = np.gradient(d1) / df
    pj = scipy.ndimage.gaussian_filter1d(np.abs(d2), sigma=6)
    pj_norm = _clamp01((pj - pj_p0) / (pj_p1 - pj_p0 + 1e-12))

    bad = _clamp01(0.75*rough_norm + 0.25*pj_norm)
    reliability_mask = 1.0 - bad

    dbg = {
        "mag_baseline": base,
        "mag_peak": mag_peak,
        "gd_norm": gd_norm,
        "mag_norm": mag_norm,
        "q_norm": q_norm,
        "rough": rough,
        "pj": pj,
    }
    return reliability_mask, room_mode_mask, dbg

def fuse_conf_for_smoothing(freq_axis, reliability_mask,
                            bass_floor_lo=0.75, bass_floor_hi=0.35,
                            f_lo=80.0, f_hi=200.0):
    """Funktio: fuse conf for smoothing."""
    f = np.asarray(freq_axis, dtype=float)
    rel = np.asarray(reliability_mask, dtype=float)

    floor = np.zeros_like(f) + bass_floor_hi
    floor[f <= f_lo] = bass_floor_lo
    mid = (f > f_lo) & (f < f_hi)
    floor[mid] = bass_floor_lo - (bass_floor_lo - bass_floor_hi) * ((f[mid] - f_lo) / (f_hi - f_lo))

    return np.maximum(rel, floor)

def modulate_gain_bassfirst(gain_db, room_mode_mask,
                            k_mode_cut=0.6, k_mode_boost=0.9,
                            max_cut_db: float | None = None):
    g = np.asarray(gain_db, dtype=float)
    mm = np.asarray(room_mode_mask, dtype=float)

    out = g.copy()
    cut = out < 0.0
    boost = out > 0.0
    out[cut] = out[cut] * (1.0 + k_mode_cut * mm[cut])
    out[boost] = out[boost] * (1.0 - k_mode_boost * mm[boost])
    if max_cut_db is not None and float(max_cut_db) > 0.0:
        out = np.maximum(out, -float(max_cut_db))
    return out
