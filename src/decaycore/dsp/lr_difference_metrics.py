# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""L/R difference metrics based on measured/analyzed response data.

Primary metric source: measured magnitude responses from l_st / r_st stats
(the ``measured_mags`` arrays in dB on a common frequency axis).

These metrics are genuinely variable — they change with measurement data and
reflect actual room asymmetry rather than correction filter similarity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LRDifferenceResult:
    """Per-run L/R difference metrics derived from measured response data."""

    # Magnitude RMS mismatch in fixed bands (dB).
    mag_rms_bass_db: float       # 20–200 Hz
    mag_rms_mid_db: float        # 200–2000 Hz
    mag_rms_band_db: float       # correction / analysis band
    mag_maxabs_band_db: float    # peak absolute delta over analysis band

    # Optional group-delay mismatch over analysis band (ms).
    gd_rms_band_ms: float = float("nan")

    # Frequency range actually used for *_band_* metrics.
    band_lo_hz: float = float("nan")
    band_hi_hz: float = float("nan")


_NAN_RESULT = LRDifferenceResult(
    mag_rms_bass_db=float("nan"),
    mag_rms_mid_db=float("nan"),
    mag_rms_band_db=float("nan"),
    mag_maxabs_band_db=float("nan"),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _band_mask(freq: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Boolean mask for bins inside [lo, hi] Hz."""
    f = np.asarray(freq, dtype=float).reshape(-1)
    return np.isfinite(f) & (f >= float(lo)) & (f <= float(hi))


def _safe_rms(x: np.ndarray) -> float:
    """RMS of finite values; returns nan if fewer than 2 finite values."""
    arr = np.asarray(x, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    return float(np.sqrt(np.mean(arr * arr)))


def _safe_maxabs(x: np.ndarray) -> float:
    """Max absolute value of finite elements; returns nan if none."""
    arr = np.asarray(x, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.max(np.abs(arr)))


def _get_mags(st: dict, key: str = "measured_mags") -> np.ndarray | None:
    """Return a float array from stats dict or None if unavailable/too small."""
    raw = st.get(key)
    if raw is None:
        return None
    try:
        arr = np.asarray(raw, dtype=float).reshape(-1)
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
        return None
    if arr.size < 8:
        return None
    return arr


def _get_freq(st: dict) -> np.ndarray | None:
    """Return freq_axis from stats dict or None."""
    raw = st.get("freq_axis")
    if raw is None:
        return None
    try:
        arr = np.asarray(raw, dtype=float).reshape(-1)
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
        return None
    if arr.size < 8:
        return None
    return arr


def _interp_to_axis(
    target_freq: np.ndarray,
    src_freq: np.ndarray,
    src_vals: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate src_vals (on src_freq) onto target_freq."""
    return np.interp(
        np.asarray(target_freq, dtype=float),
        np.asarray(src_freq, dtype=float),
        np.asarray(src_vals, dtype=float),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_lr_difference_metrics(
    l_st: dict,
    r_st: dict,
    *,
    correction_band_lo: float | None = None,
    correction_band_hi: float | None = None,
) -> LRDifferenceResult:
    """Compute L/R magnitude difference metrics from measured response data.

    Parameters
    ----------
    l_st, r_st:
        Per-channel stats dicts containing at minimum ``freq_axis`` and
        ``measured_mags`` (dB arrays of the same or similar length).
    correction_band_lo, correction_band_hi:
        Frequency limits of the correction band in Hz.  If omitted, a
        sensible default of 20–2000 Hz is used for the *_band_* metrics.

    Returns
    -------
    LRDifferenceResult with magnitude RMS/MaxAbs fields.
    Fields are NaN when insufficient data is available.

    """
    try:
        l_st = dict(l_st or {})
        r_st = dict(r_st or {})

        # Prefer comparison-mode mags when present, fall back to native.
        l_mag = _get_mags(l_st, "cmp_measured_mags")
        if l_mag is None:
            l_mag = _get_mags(l_st, "measured_mags")
        r_mag = _get_mags(r_st, "cmp_measured_mags")
        if r_mag is None:
            r_mag = _get_mags(r_st, "measured_mags")

        if l_mag is None or r_mag is None:
            return _NAN_RESULT

        l_freq = _get_freq(l_st)
        r_freq = _get_freq(r_st)

        if l_freq is None or r_freq is None:
            return _NAN_RESULT

        # Bring everything onto l_freq as reference axis.
        if l_freq.size == r_freq.size and np.allclose(l_freq, r_freq, atol=0.5, rtol=1e-4):
            freq = l_freq
            r_on_axis = r_mag if r_mag.size == freq.size else _interp_to_axis(freq, r_freq, r_mag)
            l_on_axis = l_mag if l_mag.size == freq.size else _interp_to_axis(freq, l_freq, l_mag)
        # Different axes — interpolate both onto the shorter/coarser one.
        elif l_freq.size <= r_freq.size:
            freq = l_freq
            l_on_axis = l_mag if l_mag.size == freq.size else _interp_to_axis(freq, l_freq, l_mag)
            r_on_axis = _interp_to_axis(freq, r_freq, r_mag)
        else:
            freq = r_freq
            r_on_axis = r_mag if r_mag.size == freq.size else _interp_to_axis(freq, r_freq, r_mag)
            l_on_axis = _interp_to_axis(freq, l_freq, l_mag)

        # Truncate to common length after axis alignment.
        n = min(freq.size, l_on_axis.size, r_on_axis.size)
        freq = freq[:n]
        l_on_axis = l_on_axis[:n]
        r_on_axis = r_on_axis[:n]

        delta = l_on_axis - r_on_axis

        # Fixed-band metrics.
        bass_mask = _band_mask(freq, 20.0, 200.0)
        mid_mask = _band_mask(freq, 200.0, 2000.0)

        mag_rms_bass = _safe_rms(delta[bass_mask]) if int(np.count_nonzero(bass_mask)) >= 4 else float("nan")
        mag_rms_mid = _safe_rms(delta[mid_mask]) if int(np.count_nonzero(mid_mask)) >= 4 else float("nan")

        # Analysis-band metrics.
        band_lo = float(correction_band_lo) if correction_band_lo is not None else 20.0
        band_hi = float(correction_band_hi) if correction_band_hi is not None else 2000.0
        band_mask = _band_mask(freq, band_lo, band_hi)
        n_band = int(np.count_nonzero(band_mask))

        mag_rms_band = _safe_rms(delta[band_mask]) if n_band >= 4 else float("nan")
        mag_maxabs_band = _safe_maxabs(delta[band_mask]) if n_band >= 4 else float("nan")

        # Optional GD mismatch.
        gd_rms_band = _compute_gd_rms_band(l_st, r_st, freq, band_lo, band_hi)

        return LRDifferenceResult(
            mag_rms_bass_db=mag_rms_bass,
            mag_rms_mid_db=mag_rms_mid,
            mag_rms_band_db=mag_rms_band,
            mag_maxabs_band_db=mag_maxabs_band,
            gd_rms_band_ms=gd_rms_band,
            band_lo_hz=band_lo,
            band_hi_hz=band_hi,
        )
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
    ) as exc:
        logger.debug("compute_lr_difference_metrics failed: %s", exc, exc_info=True)
        return _NAN_RESULT


# ---------------------------------------------------------------------------
# Optional GD mismatch
# ---------------------------------------------------------------------------

def _compute_gd_rms_band(
    l_st: dict,
    r_st: dict,
    freq: np.ndarray,
    band_lo: float,
    band_hi: float,
) -> float:
    """Compute GD RMS mismatch over the band; returns nan if unavailable."""
    try:
        gd_keys = ("group_delay_ms", "gd_ms", "gd_curve_ms")
        l_gd = _first_valid_gd_array(l_st, gd_keys)
        r_gd = _first_valid_gd_array(r_st, gd_keys)
        if l_gd is None or r_gd is None:
            return float("nan")
        return _gd_rms_band_from_arrays(l_gd, r_gd, freq, band_lo, band_hi)
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


def _first_valid_gd_array(stats: dict, keys: tuple[str, ...]) -> np.ndarray | None:
    for key in keys:
        raw = stats.get(key)
        if raw is None:
            continue
        arr = np.asarray(raw, dtype=float).reshape(-1)
        if arr.size >= 8:
            return arr
    return None


def _gd_rms_band_from_arrays(
    left_gd: np.ndarray,
    right_gd: np.ndarray,
    freq: np.ndarray,
    band_lo: float,
    band_hi: float,
) -> float:
    n = min(freq.size, left_gd.size, right_gd.size)
    if n < 8:
        return float("nan")
    freq_used = freq[:n]
    gd_delta = left_gd[:n] - right_gd[:n]
    mask = _band_mask(freq_used, band_lo, band_hi)
    if int(np.count_nonzero(mask)) < 4:
        return float("nan")
    return _safe_rms(gd_delta[mask])


__all__ = ["LRDifferenceResult", "compute_lr_difference_metrics"]
