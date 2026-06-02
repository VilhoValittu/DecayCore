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

# Legacy house-curve anchor frequencies used by older stats payloads.
_HOUSE_FREQS = np.array(
    [
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
    ],
    dtype=float,
)


def _target_match_pick(stats: dict[str, object], base_key: str):
    mode = str(stats.get("analysis_mode", "native") or "native").lower()
    if mode == "comparison":
        ck = "cmp_" + base_key
        if ck in stats and stats.get(ck) is not None:
            return stats.get(ck)
    return stats.get(base_key)


def _target_match_as_arr(v):
    if v is None:
        return None
    try:
        a = np.asarray(v, dtype=float)
    except (TypeError, ValueError):
        return None
    if a.ndim != 1 or a.size == 0:
        return None
    return a


def _target_match_align_to_freq_axis(values, freq_axis):
    arr = _target_match_as_arr(values)
    if arr is None:
        return None
    f = np.asarray(freq_axis, dtype=float)
    if f.ndim != 1 or f.size == 0:
        return arr
    if arr.size == f.size:
        return arr
    if arr.size == _HOUSE_FREQS.size:
        fs = np.maximum(f, 1.0)
        lf_dst = np.log10(fs)
        lf_src = np.log10(_HOUSE_FREQS)
        return np.interp(lf_dst, lf_src, arr, left=arr[0], right=arr[-1])
    x_src = np.linspace(0.0, 1.0, num=arr.size, dtype=float)
    x_dst = np.linspace(0.0, 1.0, num=f.size, dtype=float)
    return np.interp(x_dst, x_src, arr)


def _target_match_prepare_with_freq_axis(stats: dict[str, object], include_filter: bool):
    f = _target_match_as_arr(_target_match_pick(stats, "freq_axis"))
    m = _target_match_as_arr(_target_match_pick(stats, "measured_mags"))
    t = _target_match_as_arr(_target_match_pick(stats, "target_mags"))
    if m is None or t is None:
        return None, None, None, None, None

    c = _target_match_as_arr(_target_match_pick(stats, "confidence_mask"))
    g = _target_match_as_arr(_target_match_pick(stats, "filter_mags"))

    if f is not None:
        if m.size != f.size:
            m = _target_match_align_to_freq_axis(m, f)
        if t.size != f.size:
            t = _target_match_align_to_freq_axis(t, f)
        if include_filter:
            if g is None:
                g = np.zeros_like(m, dtype=float)
            elif g.size != f.size:
                g = _target_match_align_to_freq_axis(g, f)
        if c is not None and c.size != f.size:
            c = _target_match_align_to_freq_axis(c, f)
    return f, m, t, c, g


def _target_match_prepare_without_freq_axis(stats: dict[str, object], include_filter: bool):
    m = _target_match_as_arr(_target_match_pick(stats, "measured_mags"))
    t = _target_match_as_arr(_target_match_pick(stats, "target_mags"))
    if m is None or t is None:
        return None, None, None, None, None

    c = _target_match_as_arr(_target_match_pick(stats, "confidence_mask"))
    g = _target_match_as_arr(_target_match_pick(stats, "filter_mags"))

    if include_filter:
        if g is None:
            g = np.zeros_like(m, dtype=float)
        n = min(m.size, t.size, g.size)
        m, t, g = m[:n], t[:n], g[:n]
    else:
        n = min(m.size, t.size)
        m, t = m[:n], t[:n]
    if c is not None:
        c = c[: min(c.size, m.size)]
    return None, m, t, c, g


def _target_match_apply_range_mask(
    mask: np.ndarray,
    f: np.ndarray | None,
    stats: dict[str, object],
    *,
    freq_range: tuple[float, float] | None,
    use_smart_scan_range: bool,
    default_range_hz: tuple[float, float],
) -> np.ndarray:
    if f is None or f.size != mask.size:
        return mask
    out = np.asarray(mask, dtype=bool).copy()
    if freq_range is not None:
        try:
            fmin = float(freq_range[0])
            fmax = float(freq_range[1])
        except (TypeError, ValueError, OverflowError, IndexError):
            fmin, fmax = float("nan"), float("nan")
        if np.isfinite(fmin) and np.isfinite(fmax) and fmin > 0.0 and fmax > fmin:
            return out & (f >= fmin) & (f <= fmax)
        return out
    if use_smart_scan_range:
        rng = _target_match_pick(stats, "smart_scan_range")
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            try:
                fmin = float(rng[0])
                fmax = float(rng[1])
            except (TypeError, ValueError, OverflowError):
                fmin, fmax = float(default_range_hz[0]), float(default_range_hz[1])
        else:
            fmin, fmax = float(default_range_hz[0]), float(default_range_hz[1])
        if np.isfinite(fmin) and np.isfinite(fmax) and fmin > 0.0 and fmax > fmin:
            return out & (f >= fmin) & (f <= fmax)
    return out


def _target_match_compute_rms(
    pred: np.ndarray,
    t: np.ndarray,
    mask: np.ndarray,
    c: np.ndarray | None,
    *,
    use_confidence: bool,
    confidence_floor: float,
) -> float | None:
    d = (pred - t)[mask]
    if d.size < 2:
        return None
    if use_confidence and c is not None and c.size == pred.size:
        w = np.clip(c[mask], 0.0, 1.0)
        w = np.maximum(w, float(confidence_floor))
        w_sum = float(np.sum(w))
        if np.isfinite(w_sum) and w_sum > 1e-12:
            return float(np.sqrt(np.sum(w * d * d) / w_sum))
    return float(np.sqrt(np.mean(d * d)))


def match_pct_from_rms(
    rms_db: float,
    *,
    m50: float = 3.2,
    slope: float = 0.9,
    cap_rms: float = 0.4,
    cap_match_pct: float = 99.0,
) -> float:
    """Map RMS error (dB) to target-match percentage."""
    try:
        rms = float(rms_db)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not np.isfinite(rms):
        return 0.0
    out = 100.0 / (1.0 + np.exp((rms - float(m50)) / float(slope)))
    out = float(np.clip(out, 0.0, 100.0))
    if rms <= float(cap_rms):
        out = float(cap_match_pct)
    return float(np.clip(out, 0.0, 100.0))


def target_match_from_stats(
    stats: dict | None,
    *,
    include_filter: bool = True,
    use_confidence: bool = True,
    confidence_floor: float = 0.05,
    use_smart_scan_range: bool = True,
    default_range_hz: tuple[float, float] = (200.0, 5000.0),
    freq_range: tuple[float, float] | None = None,
) -> tuple[float | None, float | None]:
    """
    Single source of truth for target-match from stats payload.

    Returns `(rms_db, match_pct)`. If input is insufficient, returns `(None, None)`.

    `freq_range`, when provided, overrides `use_smart_scan_range` and
    `default_range_hz` — use this from auto-mode scoring to lock the
    evaluation window to the actual correction band instead of the
    Smart Scan level-match window.
    """
    st = stats or {}
    if _target_match_pick(st, "freq_axis") is not None:
        f, m, t, c, g = _target_match_prepare_with_freq_axis(st, include_filter)
    else:
        f, m, t, c, g = _target_match_prepare_without_freq_axis(st, include_filter)

    if m is None or t is None or m.size == 0 or t.size == 0:
        return None, None

    if include_filter:
        if g is None:
            g = np.zeros_like(m, dtype=float)
        pred = m + g
    else:
        pred = m

    n = min(pred.size, t.size)
    pred = pred[:n]
    t = t[:n]
    if f is not None:
        f = f[:n]
    if c is not None:
        c = c[:n]

    mask = np.isfinite(pred) & np.isfinite(t)
    mask = _target_match_apply_range_mask(
        mask,
        f,
        st,
        freq_range=freq_range,
        use_smart_scan_range=use_smart_scan_range,
        default_range_hz=default_range_hz,
    )
    if int(np.count_nonzero(mask)) < 10:
        return None, None

    rms = _target_match_compute_rms(
        pred,
        t,
        mask,
        c,
        use_confidence=use_confidence,
        confidence_floor=confidence_floor,
    )
    if rms is None:
        return None, None

    return rms, match_pct_from_rms(rms)
