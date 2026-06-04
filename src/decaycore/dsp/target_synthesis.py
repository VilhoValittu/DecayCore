# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

import numpy as np
from .smoothing import smooth_meas_freq_dep

_SYNTH_FREQS = np.array([
    0., 20., 25., 31.5, 40., 50., 63., 80., 100., 125.,
    160., 200., 250., 400., 1000., 2000., 4000., 8000., 16000., 20000.
])
_SYNTH_BASE_MAGS = np.array([
    6., 6., 5.9, 5.8, 5.6, 5.3, 4.9, 4.3, 3.5, 2.5,
    1.4, 0.4, 0., -0.5, -1., -1.8, -2.8, -4., -5.5, -6.
])


def _slope_db_oct(f_hz, mag_db, f_lo=None, f_hi=None) -> float:
    """Least-squares dB/octave slope via polyfit on log10(f)."""
    try:
        ff = np.asarray(f_hz, dtype=float).reshape(-1)
        mm = np.asarray(mag_db, dtype=float).reshape(-1)
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
        NameError,
    ):
        return float("nan")
    m = np.isfinite(ff) & np.isfinite(mm) & (ff > 0.)
    if f_lo is not None:
        m &= ff >= float(f_lo)
    if f_hi is not None:
        m &= ff <= float(f_hi)
    if int(np.count_nonzero(m)) < 6:
        return float("nan")
    x = np.log10(ff[m])
    y = mm[m]
    if float(np.max(x) - np.min(x)) <= 1e-6:
        return float("nan")
    try:
        p = np.polyfit(x, y, 1)
        return float(p[0] * np.log10(2.))
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
        NameError,
    ):
        return float("nan")


# Precompute Harman6 slope the same way we will estimate measurements
_SYNTH_HARMAN6_SLOPE_DB_OCT = _slope_db_oct(_SYNTH_FREQS[1:], _SYNTH_BASE_MAGS[1:])
_RT60_TARGET_DELTA_LIMIT_DB = 4.0


def _coerce_rt60_bands(value) -> dict[float, float]:
    if not isinstance(value, dict):
        return {}
    out: dict[float, float] = {}
    for raw_f, raw_rt in value.items():
        try:
            f_hz = float(raw_f)
            rt_s = float(raw_rt)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f_hz) and np.isfinite(rt_s) and f_hz > 0.0 and 0.05 < rt_s < 5.0:
            out[float(f_hz)] = float(rt_s)
    return out


def _combined_rt60_bands(measurements: dict | None) -> dict[float, float]:
    if not isinstance(measurements, dict):
        return {}
    values: dict[float, list[float]] = {}
    for key in ("measured_rt60_bands_l", "measured_rt60_bands_r"):
        for f_hz, rt_s in _coerce_rt60_bands(measurements.get(key)).items():
            values.setdefault(float(f_hz), []).append(float(rt_s))
    return {
        f_hz: float(np.mean(rt_vals))
        for f_hz, rt_vals in values.items()
        if rt_vals
    }


def _rt60_band_mean(bands: dict[float, float], f_lo: float, f_hi: float, *, min_count: int = 2) -> float:
    vals = [
        float(rt_s)
        for f_hz, rt_s in bands.items()
        if np.isfinite(float(f_hz))
        and np.isfinite(float(rt_s))
        and float(f_lo) <= float(f_hz) <= float(f_hi)
    ]
    return float(np.mean(vals)) if len(vals) >= int(min_count) else float("nan")


def _interp_clamped(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    if not np.isfinite(x):
        return float("nan")
    if float(x1) <= float(x0):
        return float(y0)
    t = float(np.clip((float(x) - float(x0)) / (float(x1) - float(x0)), 0.0, 1.0))
    return float(y0 + (y1 - y0) * t)


def _rt60_adjusted_compensation(
    measurements: dict | None,
    *,
    bass_comp_frac: float,
    tilt_comp_frac: float,
) -> tuple[float, float]:
    bands = _combined_rt60_bands(measurements)
    if not bands:
        return float(bass_comp_frac), float(tilt_comp_frac)

    bass_rt = _rt60_band_mean(bands, 20.0, 125.0)
    mid_rt = _rt60_band_mean(bands, 400.0, 2000.0)
    treble_rt = _rt60_band_mean(bands, 2000.0, 8000.0)

    bass_eff = float(bass_comp_frac)
    tilt_eff = float(tilt_comp_frac)
    if abs(bass_eff) > 1e-9 and np.isfinite(bass_rt) and np.isfinite(mid_rt) and mid_rt > 1e-6:
        bass_to_mid = float(bass_rt / mid_rt)
        bass_target = _interp_clamped(bass_to_mid, 0.8, 1.5, 0.05, 0.40)
        if np.isfinite(bass_target):
            bass_eff = bass_target
    if abs(tilt_eff) > 1e-9 and np.isfinite(treble_rt) and np.isfinite(mid_rt) and mid_rt > 1e-6:
        treble_to_mid = float(treble_rt / mid_rt)
        tilt_target = _interp_clamped(treble_to_mid, 0.70, 1.10, 0.18, 0.42)
        if np.isfinite(tilt_target):
            tilt_eff = tilt_target

    return float(bass_eff), float(tilt_eff)


def _target_synthesis_to_arr(x):
    try:
        return np.asarray(x, dtype=float).reshape(-1)
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
        NameError,
    ):
        return np.array([], dtype=float)


def _target_synthesis_sort_finite(f, m):
    idx = np.argsort(f)
    ff, mm = f[idx], m[idx]
    mask = np.isfinite(ff) & np.isfinite(mm) & (ff > 0.0)
    return ff[mask], mm[mask]


def _target_synthesis_band_mean(arr, fg, flo, fhi):
    mask = np.isfinite(arr) & (fg >= float(flo)) & (fg <= float(fhi))
    return float(np.mean(arr[mask])) if int(np.count_nonzero(mask)) >= 4 else float("nan")


def _target_synthesis_build_work(
    *,
    f_work: np.ndarray,
    fg: np.ndarray,
    bass_excess_db: float,
    meas_slope: float,
    hf_slope: float,
    bass_comp_ref_db: float,
    bass_ref_lo_hz: float,
    hf_break_hz: float,
    bass_frac: float,
    tilt_frac: float,
    hf_comp_frac: float,
) -> np.ndarray:
    m_out = _SYNTH_BASE_MAGS[1:].copy()

    bass_adj = (
        -float(bass_frac)
        * np.tanh(bass_excess_db / float(bass_comp_ref_db))
        * float(bass_comp_ref_db)
    )
    if np.isfinite(bass_adj) and abs(bass_adj) > 1e-4:
        log_lo = np.log10(max(float(bass_ref_lo_hz), 1.0))
        log_hi = np.log10(400.0)
        log_f = np.log10(np.maximum(f_work, 1.0))
        shelf_w = np.where(
            f_work <= float(bass_ref_lo_hz),
            1.0,
            np.where(
                f_work >= 400.0,
                0.0,
                1.0 - (log_f - log_lo) / (log_hi - log_lo),
            ),
        )
        shelf_w = np.clip(shelf_w, 0.0, 1.0)
        m_out += float(bass_adj) * shelf_w

    if np.isfinite(meas_slope) and np.isfinite(_SYNTH_HARMAN6_SLOPE_DB_OCT):
        tilt_adj_per_oct = float(tilt_frac) * float(meas_slope - _SYNTH_HARMAN6_SLOPE_DB_OCT)
        if abs(tilt_adj_per_oct) > 1e-4:
            log2_f_over_1k = np.log2(np.maximum(f_work, 1.0) / 1000.0)
            m_out += tilt_adj_per_oct * log2_f_over_1k

    harman6_hf_ref = np.interp(fg, _SYNTH_FREQS[1:], _SYNTH_BASE_MAGS[1:])
    harman6_hf_slope = _slope_db_oct(fg, harman6_hf_ref, f_lo=float(hf_break_hz))
    if np.isfinite(hf_slope) and np.isfinite(harman6_hf_slope) and abs(hf_slope - harman6_hf_slope) > 0.5:
        hf_excess_slope = float(hf_slope - harman6_hf_slope)
        hf_adj_per_oct = float(hf_comp_frac) * hf_excess_slope
        if abs(hf_adj_per_oct) > 1e-4:
            mask_hf = f_work > float(hf_break_hz)
            log2_f_over_break = np.log2(np.maximum(f_work[mask_hf], 1.0) / float(hf_break_hz))
            m_out[mask_hf] += hf_adj_per_oct * log2_f_over_break
    return m_out


def synthesize_target_from_measurements(
    f_l, m_l, f_r, m_r,
    *,
    bass_comp_frac: float = 0.50,
    bass_comp_ref_db: float = 8.0,
    tilt_comp_frac: float = 0.30,
    hf_comp_frac: float = 0.25,
    smooth_oct: float = 1.0 / 3.0,
    bass_ref_lo_hz: float = 50.0,
    bass_ref_hi_hz: float = 200.0,
    mid_ref_lo_hz: float = 500.0,
    mid_ref_hi_hz: float = 2000.0,
    hf_break_hz: float = 2000.0,
    measurements: dict | None = None,
):
    """Synthesize a custom target curve from L/R room measurements.

    Derives a Harman6-based target adjusted for the room's bass buildup,
    broadband tilt, and HF roll-off.

    Returns (freq_array, mag_array) on the standard 20-point grid,
    or None if inputs are insufficient.
    """
    fl, ml = _target_synthesis_to_arr(f_l), _target_synthesis_to_arr(m_l)
    fr, mr = _target_synthesis_to_arr(f_r), _target_synthesis_to_arr(m_r)

    l_ok = bool(fl.size >= 32 and ml.size == fl.size)
    r_ok = bool(fr.size >= 32 and mr.size == fr.size)
    if not l_ok and not r_ok:
        return None
    if not l_ok:
        fl, ml = fr.copy(), mr.copy()
    if not r_ok:
        fr, mr = fl.copy(), ml.copy()

    fl, ml = _target_synthesis_sort_finite(fl, ml)
    fr, mr = _target_synthesis_sort_finite(fr, mr)
    if fl.size < 32 or fr.size < 32:
        return None

    # Common frequency range
    f_lo = max(20., float(np.min(fl)), float(np.min(fr)))
    f_hi = min(20000., float(np.max(fl)), float(np.max(fr)))
    if not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_hi <= f_lo * 1.5:
        return None

    # 320-pt log grid, average L+R in dB
    fg = np.logspace(np.log10(f_lo), np.log10(f_hi), 320)
    m_avg = 0.5 * (np.interp(fg, fl, ml) + np.interp(fg, fr, mr))

    # freq-dep smooth → macro spectral envelope
    try:
        m_sm = smooth_meas_freq_dep(m_avg, fg)
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
        NameError,
    ):
        m_sm = m_avg.copy()

    # Band averages
    bass_avg = _target_synthesis_band_mean(m_sm, fg, bass_ref_lo_hz, bass_ref_hi_hz)
    mid_avg = _target_synthesis_band_mean(m_sm, fg, mid_ref_lo_hz, mid_ref_hi_hz)
    bass_excess_db = float(bass_avg - mid_avg) if (np.isfinite(bass_avg) and np.isfinite(mid_avg)) else 0.0

    # Slope estimates — use mid-range only to avoid bass-room-mode bias
    meas_slope = _slope_db_oct(fg, m_sm, f_lo=200.0, f_hi=8000.0)
    hf_slope = _slope_db_oct(fg, m_sm, f_lo=float(hf_break_hz))

    f_work = _SYNTH_FREQS[1:].copy()   # 19 points: 20..20000 Hz
    m_base = _target_synthesis_build_work(
        f_work=f_work,
        fg=fg,
        bass_excess_db=bass_excess_db,
        meas_slope=meas_slope,
        hf_slope=hf_slope,
        bass_comp_ref_db=bass_comp_ref_db,
        bass_ref_lo_hz=bass_ref_lo_hz,
        hf_break_hz=hf_break_hz,
        bass_frac=float(bass_comp_frac),
        tilt_frac=float(tilt_comp_frac),
        hf_comp_frac=float(hf_comp_frac),
    )
    bass_eff, tilt_eff = _rt60_adjusted_compensation(
        measurements,
        bass_comp_frac=float(bass_comp_frac),
        tilt_comp_frac=float(tilt_comp_frac),
    )
    if abs(bass_eff - float(bass_comp_frac)) > 1e-9 or abs(tilt_eff - float(tilt_comp_frac)) > 1e-9:
        m_rt60 = _target_synthesis_build_work(
            f_work=f_work,
            fg=fg,
            bass_excess_db=bass_excess_db,
            meas_slope=meas_slope,
            hf_slope=hf_slope,
            bass_comp_ref_db=bass_comp_ref_db,
            bass_ref_lo_hz=bass_ref_lo_hz,
            hf_break_hz=hf_break_hz,
            bass_frac=float(bass_eff),
            tilt_frac=float(tilt_eff),
            hf_comp_frac=float(hf_comp_frac),
        )
        rt60_delta = np.clip(m_rt60 - m_base, -_RT60_TARGET_DELTA_LIMIT_DB, _RT60_TARGET_DELTA_LIMIT_DB)
        m_work = m_base + rt60_delta
    else:
        m_work = m_base

    # Clip and assemble output
    m_work = np.clip(m_work, -12., 14.)
    out_freqs = _SYNTH_FREQS.copy()
    out_mags = np.empty_like(out_freqs)
    out_mags[0] = float(m_work[0])   # 0 Hz sentinel = same as 20 Hz value
    out_mags[1:] = m_work

    return out_freqs, out_mags
