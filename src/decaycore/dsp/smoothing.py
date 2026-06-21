# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

import hashlib
import numpy as np
import logging

from decaycore.auto_mode.auto_mode_profile import profiled_section
from decaycore.dsp.cache_utils import BoundedLruCache

# Try to import Rust DSP extension
try:
    from decaycore_dsp import smooth_mag_core_rs as _smooth_mag_core_rs
    _DSP_RUST_AVAILABLE = True
except ImportError:
    _DSP_RUST_AVAILABLE = False

logger = logging.getLogger("DecayCore.dsp")

AFDW_BW_MIN_OCT = 1.0 / 96.0
AFDW_BW_MAX_OCT = 1.0 / 2.0

# ---------------------------------------------------------------------------
# Smoothing plan cache
# ---------------------------------------------------------------------------

_SMOOTHING_PLAN_CACHE_MAX = 512
_SMOOTHING_PLAN_CACHE = BoundedLruCache(_SMOOTHING_PLAN_CACHE_MAX)


def _make_smoothing_plan(freqs: np.ndarray, octave_fraction: float) -> dict:
    """Build and return a smoothing plan dict (log grid + window)."""
    f_min = float(max(freqs[0], 1.0))
    f_max = float(freqs[-1])
    points_per_octave = 384
    num_points = int(np.log2(f_max / f_min) * points_per_octave)
    num_points = max(num_points, 10)
    log_freqs = np.geomspace(f_min, f_max, num_points)
    window_size = int(points_per_octave * octave_fraction)
    window_size = max(window_size, 1)
    window = np.hanning(window_size)
    w_sum = window.sum()
    if w_sum > 0:
        window = window / w_sum
    else:
        window = np.ones(window_size) / window_size
    pad_len = window_size // 2
    return {
        "f_min": f_min,
        "f_max": f_max,
        "num_points": num_points,
        "log_freqs": log_freqs,
        "window": window,
        "window_size": window_size,
        "pad_len": pad_len,
        "points_per_octave": points_per_octave,
    }


def _get_smoothing_plan(freqs: np.ndarray, octave_fraction: float) -> dict:
    """Return a cached smoothing plan, building it if needed."""
    n = len(freqs)
    key = (n, float(freqs[0]) if n > 0 else 0.0, float(freqs[-1]) if n > 0 else 0.0, round(octave_fraction, 8))
    plan = _SMOOTHING_PLAN_CACHE.get(key)
    if plan is None:
        plan = _make_smoothing_plan(freqs, octave_fraction)
        _SMOOTHING_PLAN_CACHE.put(key, plan)
    return plan


def clear_smoothing_cache() -> None:
    """Clear module-level smoothing caches shared across runs."""
    _SMOOTHING_PLAN_CACHE.clear()
    _AFDW_STACK_CACHE.clear()


# ---------------------------------------------------------------------------
# Core smoothing
# ---------------------------------------------------------------------------

def psychoacoustic_smoothing(
    freqs,
    mags,
    *,
    low_bw=1/48.0,
    high_bw=1/3.0,
    f_lo=200.0,
    f_hi=2000.0,
):
    f = np.asarray(freqs, dtype=float).ravel()
    m = np.asarray(mags, dtype=float).ravel()
    if f.size < 8 or m.size != f.size:
        return np.copy(m)

    m_low = _apply_smoothing_mag_only(f, m, float(low_bw))
    m_high = _apply_smoothing_mag_only(f, m, float(high_bw))

    ff = np.maximum(f, 1.0)
    lo = float(max(f_lo, 1.0))
    hi = float(max(f_hi, lo * 1.01))
    w = (np.log10(ff) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
    w = np.clip(w, 0.0, 1.0)

    return (1.0 - w) * m_low + w * m_high

def psycho_smooth_safe_gain(freqs, mags):
    """Funktio: psycho smooth safe gain."""
    return psychoacoustic_smoothing(
        freqs, mags,
        low_bw=1/48.0,
        high_bw=1/3.0,
        f_lo=200.0,
        f_hi=2000.0,
    )

def smooth_gain_fractional_octave(freqs, gain_db, filter_smooth, *, mult=1.0):
    """Kasittelee signaalia tai dataa: smooth gain fractional octave."""
    f = np.asarray(freqs, dtype=float).ravel()
    g = np.asarray(gain_db, dtype=float).ravel()
    if f.size < 8 or g.size != f.size:
        return np.copy(g)

    try:
        fs = float(filter_smooth)
    except (TypeError, ValueError, OverflowError):
        fs = 12.0
    if not np.isfinite(fs) or fs <= 0.0:
        fs = 12.0

    try:
        m = float(mult)
    except (TypeError, ValueError, OverflowError):
        m = 1.0
    if not np.isfinite(m) or m <= 0.0:
        m = 1.0

    octave_fraction = float(np.clip(m / fs, 1.0 / 192.0, 1.0))
    return _apply_smoothing_mag_only(f, g, octave_fraction)


# ---------------------------------------------------------------------------
# Frequency-dependent measurement smoothing
# ---------------------------------------------------------------------------

MEAS_SMOOTH_BLEND_LO_HZ: float = 120.0
MEAS_SMOOTH_BLEND_HI_HZ: float = 350.0
MEAS_SMOOTH_BELOW: float = 96.0   # 1/96 oct — tarkka bassossa
MEAS_SMOOTH_ABOVE: float = 3.0    # 1/3 oct — leveä ylätaajuuksilla


def smooth_meas_freq_dep(m_db: np.ndarray, freq_axis: np.ndarray) -> np.ndarray:
    """Tasoittaa mittauksen tai virheen taajuusriippuvaisesti.

    Alle 120 Hz: 1/96 okt, yli 350 Hz: 1/3 okt,
    välillä lineaarinen sekoitus log-taajuusasteikolla.
    """
    f = np.asarray(freq_axis, dtype=float)
    m = np.asarray(m_db, dtype=float)
    sm_lo = np.asarray(smooth_gain_fractional_octave(f, m, MEAS_SMOOTH_BELOW), dtype=float)
    sm_hi = np.asarray(smooth_gain_fractional_octave(f, m, MEAS_SMOOTH_ABOVE), dtype=float)
    log_f = np.log(np.maximum(f, 1e-6))
    log_lo = np.log(MEAS_SMOOTH_BLEND_LO_HZ)
    log_hi = np.log(MEAS_SMOOTH_BLEND_HI_HZ)
    t = np.clip((log_f - log_lo) / (log_hi - log_lo), 0.0, 1.0)
    return sm_lo * (1.0 - t) + sm_hi * t


# ---------------------------------------------------------------------------
# Adaptive FDW
# ---------------------------------------------------------------------------

# Module-level cache for adaptive FDW bandwidth stacks.
# Key: (n_freqs, f0, f_last, full_magnitude_hash) -> sm_stack array
_AFDW_STACK_CACHE_MAX = 64
_AFDW_STACK_CACHE = BoundedLruCache(_AFDW_STACK_CACHE_MAX)


def apply_adaptive_fdw(freqs, mags, confidence_mask, base_cycles=15.0, min_cycles=5.0):
    """Soveltaa tai paivittaa: apply adaptive fdw."""
    with profiled_section("dsp.smoothing.apply_adaptive_fdw"):
        return _apply_adaptive_fdw_impl(
            freqs,
            mags,
            confidence_mask,
            base_cycles=base_cycles,
            min_cycles=min_cycles,
        )


def _apply_adaptive_fdw_impl(freqs, mags, confidence_mask, base_cycles=15.0, min_cycles=5.0):
    f = np.asarray(freqs, dtype=float).ravel()
    m = np.asarray(mags, dtype=float).ravel()
    c = np.asarray(confidence_mask, dtype=float).ravel() if confidence_mask is not None else None

    if f.size < 8 or m.size != f.size:
        return np.copy(mags)

    if c is None or c.size != f.size:
        c = np.ones_like(f)
    c = np.clip(c, 0.0, 1.0)

    base_cycles = float(base_cycles)
    min_cycles = float(min_cycles)
    if base_cycles < 1.0: base_cycles = 1.0
    if min_cycles < 1.0: min_cycles = 1.0
    if min_cycles > base_cycles:
        min_cycles, base_cycles = base_cycles, min_cycles

    adaptive_cycles = min_cycles + (c * (base_cycles - min_cycles))
    oct_widths = 2.0 / np.maximum(adaptive_cycles, 1.0)
    t = np.clip(oct_widths, AFDW_BW_MIN_OCT, AFDW_BW_MAX_OCT)

    if t.size and float(np.max(t) - np.min(t)) <= 1e-12:
        out = _apply_smoothing_mag_only(f, m, float(t[0]))
        _log_afdw_bw_once(f, t)
        return out

    bw_list = np.array([
        1.0/96.0, 1.0/72.0, 1.0/48.0, 1.0/36.0, 1.0/24.0, 1.0/18.0,
        1.0/12.0, 1.0/9.0, 1.0/6.0, 1.0/4.5, 1.0/3.0, 1.0/2.0,
    ], dtype=float)

    n = f.size
    _m_full = np.ascontiguousarray(m, dtype=np.float64)
    _m_hash = hashlib.blake2b(_m_full.view(np.uint8), digest_size=8).hexdigest()
    _stack_key = (n, round(float(f[0]), 4), round(float(f[-1]), 4), _m_hash)
    sm_stack = _AFDW_STACK_CACHE.get(_stack_key)
    if sm_stack is None:
        sm_stack = np.empty((len(bw_list), n), dtype=float)
        for i, bw in enumerate(bw_list):
            sm_stack[i] = _apply_smoothing_mag_only(f, m, float(bw))
        _AFDW_STACK_CACHE.put(_stack_key, sm_stack)

    hi = np.searchsorted(bw_list, t, side='right')
    hi = np.clip(hi, 1, len(bw_list) - 1)
    lo = hi - 1

    bw_lo = bw_list[lo]
    bw_hi = bw_list[hi]
    denom = (bw_hi - bw_lo)
    denom = np.where(denom <= 1e-12, 1.0, denom)
    alpha = (t - bw_lo) / denom
    alpha = np.clip(alpha, 0.0, 1.0)

    idx = np.arange(f.size)
    sm_lo = sm_stack[lo, idx]
    sm_hi = sm_stack[hi, idx]

    out = (1.0 - alpha) * sm_lo + alpha * sm_hi


    _log_afdw_bw_once(f, t)
    return out


def _log_afdw_bw_once(f: np.ndarray, t: np.ndarray) -> None:
    try:
        if not hasattr(apply_adaptive_fdw, "_dbg_printed"):
            apply_adaptive_fdw._dbg_printed = True

            bw_min = float(np.min(t))
            bw_max = float(np.max(t))
            bw_mean = float(np.mean(t))

            f_min_bw = float(f[np.argmin(t)])
            f_max_bw = float(f[np.argmax(t)])

            logger.info(
                "A-FDW effective BW: "
                f"min={bw_min:.4f} oct @ {f_min_bw:.0f} Hz, "
                f"mean={bw_mean:.4f} oct, "
                f"max={bw_max:.4f} oct @ {f_max_bw:.0f} Hz"
            )
    except (AttributeError, TypeError, ValueError, FloatingPointError, IndexError):
        pass


# ---------------------------------------------------------------------------
# Internal smoothing helpers
# ---------------------------------------------------------------------------

def _smooth_mag_core(
    freqs: np.ndarray,
    mags: np.ndarray,
    log_freqs: np.ndarray,
    window: np.ndarray,
    pad_len: int,
) -> np.ndarray:
    # Fused interp → edge-pad → convolve ('same') → interp back.
    # Pure-Python fallback for the Rust smooth_mag_core_rs hot path.
    log_mags = np.interp(log_freqs, freqs, mags)
    n = log_mags.size
    w = window.size
    P = n + 2 * pad_len

    padded = np.empty(P)
    for i in range(pad_len):
        padded[i] = log_mags[0]
    for i in range(n):
        padded[pad_len + i] = log_mags[i]
    for i in range(pad_len):
        padded[pad_len + n + i] = log_mags[n - 1]

    # Convolve padded with window ('same' mode), output indices [pad_len : pad_len+n].
    # output[ii] = sum_k padded[half_w + pad_len + ii - k] * window[k]
    half_w = (w - 1) // 2
    sm = np.empty(n)
    for ii in range(n):
        acc = 0.0
        base = half_w + pad_len + ii
        for k in range(w):
            acc += padded[base - k] * window[k]
        sm[ii] = acc

    return np.interp(freqs, log_freqs, sm)


def _smooth_mag_core_dispatch(freqs_arr: np.ndarray, vals_arr: np.ndarray, plan) -> np.ndarray:
    """Dispatch the core smoothing kernel to Rust if available, else pure Python.

    `freqs_arr`/`vals_arr` must already be contiguous float64; `plan` is the
    smoothing plan from `_get_smoothing_plan`.
    """
    if _DSP_RUST_AVAILABLE:
        try:
            return _smooth_mag_core_rs(
                freqs_arr,
                vals_arr,
                plan["log_freqs"],
                plan["window"],
                plan["pad_len"],
            )
        except Exception:
            # Fallback to pure-Python version on any error
            pass

    # Pure-Python fallback
    return _smooth_mag_core(
        freqs_arr,
        vals_arr,
        plan["log_freqs"],
        plan["window"],
        plan["pad_len"],
    )


def _apply_smoothing_mag_only(freqs: np.ndarray, mags: np.ndarray, octave_fraction: float) -> np.ndarray:
    """Magnitude-only smoothing path; no phase work."""
    if octave_fraction <= 0:
        return mags
    freqs_arr = np.asarray(freqs, dtype=np.float64).ravel()
    if not np.all(np.diff(freqs_arr) > 0):
        raise ValueError("frequency axis must be strictly monotonically increasing")
    plan = _get_smoothing_plan(freqs_arr, octave_fraction)
    mags_arr = np.asarray(mags, dtype=np.float64)
    return _smooth_mag_core_dispatch(freqs_arr, mags_arr, plan)


def apply_smoothing_std(freqs, mags, phases, octave_fraction=1.0):
    """Soveltaa tai paivittaa: apply smoothing std."""
    if octave_fraction <= 0: return mags, phases
    freqs = np.asarray(freqs, dtype=np.float64).ravel()
    mags = np.asarray(mags, dtype=np.float64).ravel()
    phases = np.asarray(phases, dtype=np.float64).ravel()
    if not np.all(np.diff(freqs) > 0):
        raise ValueError("frequency axis must be strictly monotonically increasing")
    phase_unwrap = np.unwrap(np.deg2rad(phases))
    plan = _get_smoothing_plan(freqs, float(octave_fraction))
    sm_mags = _smooth_mag_core_dispatch(freqs, mags, plan)
    sm_phases = _smooth_mag_core_dispatch(freqs, phase_unwrap, plan)
    return sm_mags, np.rad2deg(sm_phases)
