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
from .smoothing import smooth_gain_fractional_octave

_SYNTH_FREQS = np.array(
    [
        0.0,
        20.0,
        25.0,
        31.5,
        40.0,
        50.0,
        63.0,
        80.0,
        100.0,
        125.0,
        160.0,
        200.0,
        250.0,
        400.0,
        1000.0,
        2000.0,
        4000.0,
        8000.0,
        16000.0,
        20000.0,
    ]
)
_SYNTH_BASE_MAGS = np.array(
    [
        6.0,
        6.0,
        5.9,
        5.8,
        5.6,
        5.3,
        4.9,
        4.3,
        3.5,
        2.5,
        1.4,
        0.4,
        0.0,
        -0.5,
        -1.0,
        -1.8,
        -2.8,
        -4.0,
        -5.5,
        -6.0,
    ]
)

# HF slope is estimated and applied only up to this frequency.  Above this
# point the slope-based adjustment is held constant (shelf), preventing
# measurement noise floor above ~10 kHz from inflating the estimate and
# creating spurious target rises at very high frequencies.
_SYNTH_HF_SLOPE_CAP_HZ = 10000.0


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
    ):
        return float("nan")
    m = np.isfinite(ff) & np.isfinite(mm) & (ff > 0.0)
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
        return float(p[0] * np.log10(2.0))
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
        return float("nan")


_SYNTH_POLICY_VERSION = 4
_SYNTH_BASS_CUT_LIMIT_DB = 4.0
_SYNTH_BASS_BOOST_LIMIT_DB = 3.0
_SYNTH_BASS_LIFT_MIN_RT60_CONFIDENCE = 0.20
_SYNTH_HF_DELTA_LIMIT_DB = 0.50
_SYNTH_CHANNEL_DISAGREE_FULL_DB = 4.0
_SYNTH_ADAPT_FADE_START_HZ = 250.0
_SYNTH_ADAPT_MAX_HZ = 500.0
_SYNTH_HF_ADAPT_MIN_SNR_DB = 30.0


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


def _rt60_band_median(
    bands: dict[float, float],
    f_lo: float,
    f_hi: float,
    *,
    min_count: int = 2,
) -> tuple[float, int]:
    vals = [
        float(rt_s)
        for f_hz, rt_s in bands.items()
        if np.isfinite(float(f_hz)) and np.isfinite(float(rt_s)) and float(f_lo) <= float(f_hz) <= float(f_hi)
    ]
    count = int(len(vals))
    if count < int(min_count):
        return float("nan"), count
    return float(np.median(np.asarray(vals, dtype=float))), count


def _interp_clamped(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    if not np.isfinite(x):
        return float("nan")
    if float(x1) <= float(x0):
        return float(y0)
    t = float(np.clip((float(x) - float(x0)) / (float(x1) - float(x0)), 0.0, 1.0))
    return float(y0 + (y1 - y0) * t)


def _adaptive_rt60_guard(measurements: dict | None) -> dict:
    """Return conservative RT60 confidence and a bass-to-mid decay ratio.

    Spectral RT60 variation is physical evidence, not measurement consistency.
    Confidence therefore comes from band coverage and L/R agreement.  The
    result is only used to prevent extra bass target lift in slow-decay rooms;
    it never creates a target adjustment on its own.
    """
    if not isinstance(measurements, dict):
        return {
            "confidence": 0.0,
            "bass_to_mid_ratio": float("nan"),
            "channel_ratio_disagreement": float("nan"),
            "reason": "missing_metadata",
        }

    bands_l = _coerce_rt60_bands(measurements.get("measured_rt60_bands_l"))
    bands_r = _coerce_rt60_bands(measurements.get("measured_rt60_bands_r"))
    bass_l, bass_count_l = _rt60_band_median(bands_l, 20.0, 125.0)
    mid_l, mid_count_l = _rt60_band_median(bands_l, 400.0, 2000.0)
    bass_r, bass_count_r = _rt60_band_median(bands_r, 20.0, 125.0)
    mid_r, mid_count_r = _rt60_band_median(bands_r, 400.0, 2000.0)

    valid_l = bool(np.isfinite(bass_l) and np.isfinite(mid_l) and mid_l > 1e-6)
    valid_r = bool(np.isfinite(bass_r) and np.isfinite(mid_r) and mid_r > 1e-6)
    if not valid_l or not valid_r:
        return {
            "confidence": 0.0,
            "bass_to_mid_ratio": float("nan"),
            "channel_ratio_disagreement": float("nan"),
            "reason": "insufficient_stereo_bands",
            "band_count_l": int(min(bass_count_l, mid_count_l)),
            "band_count_r": int(min(bass_count_r, mid_count_r)),
        }

    ratio_l = float(bass_l / mid_l)
    ratio_r = float(bass_r / mid_r)
    ratio = float(np.sqrt(max(ratio_l, 1e-9) * max(ratio_r, 1e-9)))
    ratio_disagreement = float(abs(np.log(max(ratio_l, 1e-9) / max(ratio_r, 1e-9))))
    coverage = float(
        np.clip(
            min(bass_count_l, mid_count_l, bass_count_r, mid_count_r) / 3.0,
            0.0,
            1.0,
        )
    )
    agreement = float(
        np.clip(
            np.interp(
                ratio_disagreement,
                [np.log(1.10), np.log(1.60)],
                [1.0, 0.0],
            ),
            0.0,
            1.0,
        )
    )
    confidence = float(coverage * agreement)
    return {
        "confidence": float(confidence),
        "bass_to_mid_ratio": float(ratio),
        "channel_ratio_disagreement": float(ratio_disagreement),
        "reason": "ok" if confidence > 0.0 else "channel_disagreement",
        "band_count_l": int(min(bass_count_l, mid_count_l)),
        "band_count_r": int(min(bass_count_r, mid_count_r)),
    }


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
    ):
        return np.array([], dtype=float)


def _target_synthesis_sort_finite(f, m):
    idx = np.argsort(f)
    ff, mm = f[idx], m[idx]
    mask = np.isfinite(ff) & np.isfinite(mm) & (ff > 0.0)
    return ff[mask], mm[mask]


def _target_synthesis_band_median(arr, fg, flo, fhi):
    mask = np.isfinite(arr) & (fg >= float(flo)) & (fg <= float(fhi))
    return float(np.median(arr[mask])) if int(np.count_nonzero(mask)) >= 4 else float("nan")


def _target_synthesis_smooth_macro(mag_db, freq_hz, smooth_oct: float) -> np.ndarray:
    try:
        octave_width = float(smooth_oct)
    except (TypeError, ValueError, OverflowError):
        octave_width = 1.0 / 3.0
    if not np.isfinite(octave_width) or octave_width <= 0.0:
        octave_width = 1.0 / 3.0
    octave_width = float(np.clip(octave_width, 1.0 / 12.0, 1.0))
    fraction = float(1.0 / octave_width)
    try:
        return np.asarray(
            smooth_gain_fractional_octave(freq_hz, mag_db, fraction),
            dtype=float,
        )
    except (TypeError, ValueError, OverflowError, FloatingPointError):
        return np.asarray(mag_db, dtype=float).copy()


def _target_synthesis_hf_snr_confidence(measurements: dict | None) -> float:
    if not isinstance(measurements, dict):
        return 0.0

    values = []
    for side in ("l", "r"):
        raw = None
        for key in (
            f"measurement_snr_db_{side}",
            f"measured_snr_db_{side}",
            f"snr_db_{side}",
        ):
            if measurements.get(key) is not None:
                raw = measurements.get(key)
                break
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError):
            continue
        if np.isfinite(value):
            values.append(value)
    if len(values) < 2:
        return 0.0
    snr_db = float(min(values))
    return float(
        np.clip(
            np.interp(snr_db, [_SYNTH_HF_ADAPT_MIN_SNR_DB, 40.0], [0.0, 1.0]),
            0.0,
            1.0,
        )
    )


def _target_synthesis_prepare_analysis(
    f_l,
    m_l,
    f_r,
    m_r,
    *,
    smooth_oct: float,
    bass_ref_lo_hz: float,
    bass_ref_hi_hz: float,
    mid_ref_lo_hz: float,
    mid_ref_hi_hz: float,
    measurements: dict | None,
) -> dict | None:
    fl, ml = _target_synthesis_to_arr(f_l), _target_synthesis_to_arr(m_l)
    fr, mr = _target_synthesis_to_arr(f_r), _target_synthesis_to_arr(m_r)

    l_ok = bool(fl.size >= 32 and ml.size == fl.size)
    r_ok = bool(fr.size >= 32 and mr.size == fr.size)
    if not l_ok and not r_ok:
        return None
    stereo_input = bool(l_ok and r_ok)
    if not l_ok:
        fl, ml = fr.copy(), mr.copy()
    if not r_ok:
        fr, mr = fl.copy(), ml.copy()

    fl, ml = _target_synthesis_sort_finite(fl, ml)
    fr, mr = _target_synthesis_sort_finite(fr, mr)
    if fl.size < 32 or fr.size < 32:
        return None

    f_lo = max(20.0, float(np.min(fl)), float(np.min(fr)))
    f_hi = min(20000.0, float(np.max(fl)), float(np.max(fr)))
    if not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_hi <= f_lo * 1.5:
        return None

    fg = np.logspace(np.log10(f_lo), np.log10(f_hi), 320)
    ml_g = np.interp(fg, fl, ml)
    mr_g = np.interp(fg, fr, mr)
    ml_sm = _target_synthesis_smooth_macro(ml_g, fg, smooth_oct)
    mr_sm = _target_synthesis_smooth_macro(mr_g, fg, smooth_oct)
    base_g = np.interp(fg, _SYNTH_FREQS[1:], _SYNTH_BASE_MAGS[1:])

    mid_mask = np.isfinite(fg) & (fg >= float(mid_ref_lo_hz)) & (fg <= float(mid_ref_hi_hz))
    if int(np.count_nonzero(mid_mask)) < 8:
        return None
    offset_l = float(np.median((ml_sm - base_g)[mid_mask]))
    offset_r = float(np.median((mr_sm - base_g)[mid_mask]))
    residual_l = ml_sm - (base_g + offset_l)
    residual_r = mr_sm - (base_g + offset_r)

    bass_l = _target_synthesis_band_median(
        residual_l,
        fg,
        bass_ref_lo_hz,
        bass_ref_hi_hz,
    )
    bass_r = _target_synthesis_band_median(
        residual_r,
        fg,
        bass_ref_lo_hz,
        bass_ref_hi_hz,
    )
    bass_residual = 0.5 * (float(bass_l) + float(bass_r)) if np.isfinite(bass_l) and np.isfinite(bass_r) else 0.0
    channel_disagreement = (
        abs(float(bass_l) - float(bass_r))
        if np.isfinite(bass_l) and np.isfinite(bass_r)
        else float(_SYNTH_CHANNEL_DISAGREE_FULL_DB)
    )
    if stereo_input:
        channel_confidence = float(
            np.clip(
                1.0 - channel_disagreement / float(_SYNTH_CHANNEL_DISAGREE_FULL_DB),
                0.0,
                1.0,
            )
        )
    else:
        channel_confidence = 0.50

    tilt_l = _slope_db_oct(fg, residual_l, f_lo=200.0, f_hi=1000.0)
    tilt_r = _slope_db_oct(fg, residual_r, f_lo=200.0, f_hi=1000.0)
    tilt_residual = 0.5 * (float(tilt_l) + float(tilt_r)) if np.isfinite(tilt_l) and np.isfinite(tilt_r) else 0.0
    hf_l = _slope_db_oct(fg, residual_l, f_lo=2000.0, f_hi=_SYNTH_HF_SLOPE_CAP_HZ)
    hf_r = _slope_db_oct(fg, residual_r, f_lo=2000.0, f_hi=_SYNTH_HF_SLOPE_CAP_HZ)
    hf_residual = 0.5 * (float(hf_l) + float(hf_r)) if np.isfinite(hf_l) and np.isfinite(hf_r) else 0.0
    hf_slope_disagreement = abs(float(hf_l) - float(hf_r)) if np.isfinite(hf_l) and np.isfinite(hf_r) else float("inf")
    hf_channel_confidence = float(np.clip(1.0 - hf_slope_disagreement / 1.0, 0.0, 1.0))
    hf_confidence = float(
        channel_confidence * hf_channel_confidence * _target_synthesis_hf_snr_confidence(measurements)
    )

    return {
        "fg": fg,
        "base_g": base_g,
        "residual_l": residual_l,
        "residual_r": residual_r,
        "bass_residual_db": float(bass_residual),
        "bass_residual_l_db": float(bass_l),
        "bass_residual_r_db": float(bass_r),
        "tilt_residual_db_per_oct": float(tilt_residual),
        "hf_residual_db_per_oct": float(hf_residual),
        "channel_disagreement_db": float(channel_disagreement),
        "channel_confidence": float(channel_confidence),
        "hf_confidence": float(hf_confidence),
        "stereo_input": bool(stereo_input),
        "rt60_guard": _adaptive_rt60_guard(measurements),
    }


def _target_synthesis_hf_slope_hi_hz(hf_break_hz: float) -> float | None:
    try:
        hf_break = float(hf_break_hz)
        hf_cap = float(_SYNTH_HF_SLOPE_CAP_HZ)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(hf_break) or not np.isfinite(hf_cap):
        return None
    if hf_break <= 0.0 or hf_cap <= hf_break:
        return None
    return float(hf_cap)


def _target_synthesis_build_work(
    *,
    f_work: np.ndarray,
    bass_residual_db: float,
    tilt_residual_db_per_oct: float,
    hf_residual_db_per_oct: float,
    channel_confidence: float,
    hf_confidence: float,
    rt60_guard: dict,
    bass_comp_ref_db: float,
    bass_ref_lo_hz: float,
    hf_break_hz: float,
    bass_frac: float,
    tilt_frac: float,
    hf_comp_frac: float,
) -> np.ndarray:
    base = _SYNTH_BASE_MAGS[1:].copy()
    bass_adj_db = (
        -float(bass_frac) * np.tanh(float(bass_residual_db) / float(bass_comp_ref_db)) * float(bass_comp_ref_db)
    )
    if not np.isfinite(bass_adj_db):
        bass_adj_db = 0.0
    bass_adj_db *= float(np.clip(channel_confidence, 0.0, 1.0))

    rt60_confidence = float(np.clip(float((rt60_guard or {}).get("confidence", 0.0) or 0.0), 0.0, 1.0))
    rt60_ratio = float((rt60_guard or {}).get("bass_to_mid_ratio", float("nan")))
    if bass_adj_db > 0.0:
        if rt60_confidence < float(_SYNTH_BASS_LIFT_MIN_RT60_CONFIDENCE) or not np.isfinite(rt60_ratio):
            bass_adj_db = 0.0
        else:
            decay_guard = _interp_clamped(rt60_ratio, 0.9, 1.5, 1.0, 0.0)
            bass_adj_db *= float((1.0 - rt60_confidence) + rt60_confidence * float(decay_guard))

    log_lo = np.log10(max(float(bass_ref_lo_hz), 1.0))
    log_hi = np.log10(float(_SYNTH_ADAPT_MAX_HZ))
    log_f = np.log10(np.maximum(f_work, 1.0))
    shelf_w = np.where(
        f_work <= float(bass_ref_lo_hz),
        1.0,
        np.where(
            f_work >= float(_SYNTH_ADAPT_MAX_HZ),
            0.0,
            1.0 - (log_f - log_lo) / (log_hi - log_lo),
        ),
    )
    shelf_w = np.clip(shelf_w, 0.0, 1.0)
    bass_delta = float(bass_adj_db) * shelf_w

    tilt_per_oct = float(tilt_frac) * float(tilt_residual_db_per_oct) * float(np.clip(channel_confidence, 0.0, 1.0))
    if not np.isfinite(tilt_per_oct):
        tilt_per_oct = 0.0
    tilt_per_oct = float(np.clip(tilt_per_oct, -0.25, 0.25))
    tilt_delta = tilt_per_oct * np.log2(np.maximum(f_work, 1.0) / float(_SYNTH_ADAPT_MAX_HZ))
    tilt_delta *= np.where(
        f_work <= float(_SYNTH_ADAPT_FADE_START_HZ),
        1.0,
        np.clip(
            (np.log(float(_SYNTH_ADAPT_MAX_HZ)) - np.log(np.maximum(f_work, 1.0)))
            / (np.log(float(_SYNTH_ADAPT_MAX_HZ)) - np.log(float(_SYNTH_ADAPT_FADE_START_HZ))),
            0.0,
            1.0,
        ),
    )
    tilt_delta = np.where(f_work < float(_SYNTH_ADAPT_MAX_HZ), tilt_delta, 0.0)
    low_delta = np.clip(
        bass_delta + tilt_delta,
        -float(_SYNTH_BASS_CUT_LIMIT_DB) * shelf_w,
        float(_SYNTH_BASS_BOOST_LIMIT_DB) * shelf_w,
    )

    hf_delta = np.zeros_like(f_work, dtype=float)
    hf_slope_hi_hz = _target_synthesis_hf_slope_hi_hz(float(hf_break_hz))
    if (
        float(hf_comp_frac) > 0.0
        and float(hf_confidence) > 0.0
        and hf_slope_hi_hz is not None
        and np.isfinite(hf_residual_db_per_oct)
    ):
        mask_hf = f_work > float(hf_break_hz)
        hf_per_oct = float(hf_comp_frac) * float(hf_residual_db_per_oct) * float(np.clip(hf_confidence, 0.0, 1.0))
        log2_cap = np.log2(float(hf_slope_hi_hz) / float(hf_break_hz))
        log2_f_over_break = np.minimum(
            np.log2(np.maximum(f_work[mask_hf], 1.0) / float(hf_break_hz)),
            log2_cap,
        )
        hf_delta[mask_hf] = np.clip(
            hf_per_oct * log2_f_over_break,
            -float(_SYNTH_HF_DELTA_LIMIT_DB),
            float(_SYNTH_HF_DELTA_LIMIT_DB),
        )

    return base + low_delta + hf_delta


def synthesize_target_from_measurements(
    f_l,
    m_l,
    f_r,
    m_r,
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

    Derives a Harman6-based target from broad, base-relative room evidence.
    Adaptation is confidence-weighted across channels, bounded in the modal
    region, and does not use RT60 to create tonal changes by itself.

    Returns (freq_array, mag_array) on the standard 20-point grid,
    or None if inputs are insufficient.
    """
    analysis = _target_synthesis_prepare_analysis(
        f_l,
        m_l,
        f_r,
        m_r,
        smooth_oct=float(smooth_oct),
        bass_ref_lo_hz=float(bass_ref_lo_hz),
        bass_ref_hi_hz=float(bass_ref_hi_hz),
        mid_ref_lo_hz=float(mid_ref_lo_hz),
        mid_ref_hi_hz=float(mid_ref_hi_hz),
        measurements=measurements,
    )
    if analysis is None:
        return None

    f_work = _SYNTH_FREQS[1:].copy()
    m_work = _target_synthesis_build_work(
        f_work=f_work,
        bass_residual_db=float(analysis["bass_residual_db"]),
        tilt_residual_db_per_oct=float(analysis["tilt_residual_db_per_oct"]),
        hf_residual_db_per_oct=float(analysis["hf_residual_db_per_oct"]),
        channel_confidence=float(analysis["channel_confidence"]),
        hf_confidence=float(analysis["hf_confidence"]),
        rt60_guard=dict(analysis["rt60_guard"]),
        bass_comp_ref_db=bass_comp_ref_db,
        bass_ref_lo_hz=bass_ref_lo_hz,
        hf_break_hz=hf_break_hz,
        bass_frac=float(bass_comp_frac),
        tilt_frac=float(tilt_comp_frac),
        hf_comp_frac=float(hf_comp_frac),
    )

    # Clip and assemble output
    m_work = np.clip(m_work, -12.0, 14.0)
    out_freqs = _SYNTH_FREQS.copy()
    out_mags = np.empty_like(out_freqs)
    out_mags[0] = float(m_work[0])  # 0 Hz sentinel = same as 20 Hz value
    out_mags[1:] = m_work

    return out_freqs, out_mags


def adaptive_target_diagnostics(
    f_l,
    m_l,
    f_r,
    m_r,
    *,
    target_f,
    target_m,
    smooth_oct: float = 1.0 / 3.0,
    bass_ref_lo_hz: float = 50.0,
    bass_ref_hi_hz: float = 200.0,
    mid_ref_lo_hz: float = 500.0,
    mid_ref_hi_hz: float = 2000.0,
    measurements: dict | None = None,
) -> dict:
    """Return compact, JSON-friendly evidence for an adaptive target."""
    analysis = _target_synthesis_prepare_analysis(
        f_l,
        m_l,
        f_r,
        m_r,
        smooth_oct=float(smooth_oct),
        bass_ref_lo_hz=float(bass_ref_lo_hz),
        bass_ref_hi_hz=float(bass_ref_hi_hz),
        mid_ref_lo_hz=float(mid_ref_lo_hz),
        mid_ref_hi_hz=float(mid_ref_hi_hz),
        measurements=measurements,
    )
    if analysis is None:
        return {
            "policy_version": int(_SYNTH_POLICY_VERSION),
            "valid": False,
            "fallback_reason": "insufficient_measurement",
        }

    tf = _target_synthesis_to_arr(target_f)
    tm = _target_synthesis_to_arr(target_m)
    if tf.size != tm.size or tf.size < 2:
        return {
            "policy_version": int(_SYNTH_POLICY_VERSION),
            "valid": False,
            "fallback_reason": "invalid_target",
        }
    tf, tm = _target_synthesis_sort_finite(tf, tm)
    base_t = np.interp(tf, _SYNTH_FREQS[1:], _SYNTH_BASE_MAGS[1:])
    target_delta = tm - base_t
    fg = np.asarray(analysis["fg"], dtype=float)
    target_delta_g = np.interp(fg, tf, target_delta)
    fit_mask = (fg >= 20.0) & (fg <= float(_SYNTH_ADAPT_MAX_HZ))
    fit_l = np.asarray(analysis["residual_l"], dtype=float) - target_delta_g
    fit_r = np.asarray(analysis["residual_r"], dtype=float) - target_delta_g
    fit_rms_db = 0.5 * (
        float(np.sqrt(np.mean(np.square(fit_l[fit_mask])))) + float(np.sqrt(np.mean(np.square(fit_r[fit_mask]))))
    )
    channel_confidence = float(analysis["channel_confidence"])
    rt60_guard = dict(analysis["rt60_guard"])
    rt60_confidence = float(rt60_guard.get("confidence", 0.0) or 0.0)
    rt60_ratio = float(rt60_guard.get("bass_to_mid_ratio", float("nan")))
    bass_lift_requested = bool(float(analysis["bass_residual_db"]) < -0.05)
    bass_lift_metadata_reliable = bool(
        rt60_confidence >= float(_SYNTH_BASS_LIFT_MIN_RT60_CONFIDENCE) and np.isfinite(rt60_ratio)
    )
    bass_lift_guard_reason = ""
    if bass_lift_requested and not bass_lift_metadata_reliable:
        bass_lift_guard_reason = "missing_or_low_confidence_rt60"
    elif bass_lift_requested and rt60_ratio >= 1.5:
        bass_lift_guard_reason = "slow_bass_decay"
    if channel_confidence <= 0.05:
        fallback_reason = "channel_disagreement"
    elif bass_lift_requested and not bass_lift_metadata_reliable and float(np.max(np.abs(target_delta))) <= 0.05:
        fallback_reason = "bass_lift_requires_rt60"
    elif float(np.max(np.abs(target_delta))) <= 0.05:
        fallback_reason = "base_target_preserved"
    else:
        fallback_reason = ""

    return {
        "policy_version": int(_SYNTH_POLICY_VERSION),
        "valid": True,
        "adaptive_applied": bool(float(np.max(np.abs(target_delta))) > 0.05),
        "fallback_reason": str(fallback_reason),
        "fit_rms_db": float(fit_rms_db),
        "bass_residual_db": float(analysis["bass_residual_db"]),
        "bass_residual_l_db": float(analysis["bass_residual_l_db"]),
        "bass_residual_r_db": float(analysis["bass_residual_r_db"]),
        "tilt_residual_db_per_oct": float(analysis["tilt_residual_db_per_oct"]),
        "channel_disagreement_db": float(analysis["channel_disagreement_db"]),
        "channel_confidence": float(channel_confidence),
        "hf_confidence": float(analysis["hf_confidence"]),
        "rt60_confidence": float(rt60_confidence),
        "rt60_bass_to_mid_ratio": (float(rt60_ratio) if np.isfinite(rt60_ratio) else None),
        "rt60_reason": str(rt60_guard.get("reason", "") or ""),
        "bass_lift_requested": bool(bass_lift_requested),
        "bass_lift_metadata_reliable": bool(bass_lift_metadata_reliable),
        "bass_lift_guard_reason": str(bass_lift_guard_reason),
        "target_delta_min_db": float(np.min(target_delta)),
        "target_delta_max_db": float(np.max(target_delta)),
        "target_delta_abs_max_db": float(np.max(np.abs(target_delta))),
    }
