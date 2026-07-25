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

from dataclasses import dataclass

import numpy as np

from . import decaycore_dsp as dsp


@dataclass(frozen=True)
class LfRolloffEstimate:
    f6_hz: float
    stable_f6_hz: float
    model_f6_hz: float
    ref_db: float
    confidence: float
    method: str
    usable: bool = False
    reason: str = "fallback"
    ref_spread_db: float = float("nan")
    low_coverage_oct: float = float("nan")
    model_delta_oct: float = float("nan")
    reference_band: str = ""


def _sorted_finite_xy(f_hz, mag_db) -> tuple[np.ndarray, np.ndarray]:
    try:
        ff = np.asarray(f_hz, dtype=float).reshape(-1)
        mm = np.asarray(mag_db, dtype=float).reshape(-1)
    except (TypeError, ValueError, AttributeError, KeyError, IndexError, RuntimeError, OSError, ImportError):
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    if ff.size != mm.size or ff.size < 16:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    idx = np.argsort(ff)
    ff = ff[idx]
    mm = mm[idx]
    mask = np.isfinite(ff) & np.isfinite(mm) & (ff > 0.0)
    return np.asarray(ff[mask], dtype=float), np.asarray(mm[mask], dtype=float)


def _smooth_mag(ff: np.ndarray, mm: np.ndarray, smooth_oct: float) -> np.ndarray:
    smooth = float(smooth_oct)
    if not np.isfinite(smooth) or smooth <= 0.0:
        return np.asarray(mm, dtype=float)
    try:
        mm_sm, _phase = dsp.apply_smoothing_std(ff, mm, np.zeros_like(mm), smooth)
        return np.asarray(mm_sm, dtype=float)
    except (TypeError, ValueError, AttributeError, KeyError, IndexError, RuntimeError, OSError, ImportError):
        return np.asarray(mm, dtype=float)


def _robust_reference_db(values: np.ndarray) -> float:
    vv = np.asarray(values, dtype=float).reshape(-1)
    vv = vv[np.isfinite(vv)]
    if vv.size < 6:
        return float("nan")
    if vv.size >= 12:
        lo, hi = np.quantile(vv, [0.10, 0.90])
        trimmed = vv[(vv >= float(lo)) & (vv <= float(hi))]
        if trimmed.size >= 6:
            vv = trimmed
    return float(np.quantile(vv, 0.70))


def _reference_stats(values: np.ndarray) -> tuple[float, float]:
    """Return the robust reference level and its residual spread in dB."""
    vv = np.asarray(values, dtype=float).reshape(-1)
    vv = vv[np.isfinite(vv)]
    if vv.size < 6:
        return float("nan"), float("nan")
    if vv.size >= 12:
        lo, hi = np.quantile(vv, [0.10, 0.90])
        trimmed = vv[(vv >= float(lo)) & (vv <= float(hi))]
        if trimmed.size >= 6:
            vv = trimmed
    return float(np.quantile(vv, 0.70)), float(np.quantile(vv, 0.90) - np.quantile(vv, 0.10))


def _fallback_estimate(
    default_hz: float,
    *,
    ref_db: float = float("nan"),
    ref_spread_db: float = float("nan"),
    reason: str,
    reference_band: str = "",
) -> LfRolloffEstimate:
    return LfRolloffEstimate(
        float(default_hz),
        float("nan"),
        float("nan"),
        float(ref_db),
        0.0,
        "fallback",
        usable=False,
        reason=str(reason),
        ref_spread_db=float(ref_spread_db),
        reference_band=str(reference_band),
    )


def _interpolate_crossing(f_lo: np.ndarray, y_lo: np.ndarray, idx: int, threshold_db: float) -> float:
    if idx <= 0:
        return float(f_lo[0])
    x1, y1 = float(f_lo[idx - 1]), float(y_lo[idx - 1])
    x2, y2 = float(f_lo[idx]), float(y_lo[idx])
    dy = float(y2 - y1)
    if np.isfinite(dy) and abs(dy) > 1e-9:
        return float(x1 + (float(threshold_db) - y1) * (x2 - x1) / dy)
    return float(x2)


def _stable_crossing_hz(
    f_lo: np.ndarray,
    y_lo: np.ndarray,
    threshold_db: float,
    *,
    min_span_oct: float = 0.24,
    near_tolerance_db: float = 1.0,
    min_fraction: float = 0.68,
) -> float | None:
    if f_lo.size < 8 or y_lo.size != f_lo.size:
        return None
    near_threshold = float(threshold_db) - float(max(0.0, near_tolerance_db))
    candidate_idx = np.where(np.asarray(y_lo >= float(threshold_db), dtype=bool))[0]
    if candidate_idx.size <= 0:
        return None
    min_pts = int(np.clip(round(f_lo.size * 0.025), 3, 12))
    span_mul = float(2.0 ** max(0.03, float(min_span_oct)))
    for idx_raw in candidate_idx:
        idx = int(idx_raw)
        hi = float(f_lo[idx]) * span_mul
        j = int(np.searchsorted(f_lo, hi, side="right"))
        j = max(j, idx + min_pts)
        j = min(j, f_lo.size)
        window = np.asarray(y_lo[idx:j], dtype=float)
        window = window[np.isfinite(window)]
        if window.size < min_pts:
            continue
        frac_near = float(np.mean(window >= near_threshold))
        med = float(np.median(window))
        if frac_near >= float(min_fraction) and med >= near_threshold:
            return _interpolate_crossing(f_lo, y_lo, idx, threshold_db)
    return None


def _model_f6_from_crossings(f3_hz: float | None, f6_hz: float | None, f12_hz: float | None) -> tuple[float, float]:
    if f3_hz is None or f6_hz is None or not (np.isfinite(f3_hz) and np.isfinite(f6_hz)):
        return float("nan"), 0.0
    if float(f6_hz) >= float(f3_hz):
        return float("nan"), 0.0
    c6 = float((10.0**0.6) - 1.0)
    c12 = float((10.0**1.2) - 1.0)
    estimates: list[float] = []
    try:
        den6 = float(np.log(float(f3_hz) / float(f6_hz)))
        if np.isfinite(den6) and den6 > 1e-9:
            estimates.append(float(np.log(c6) / (2.0 * den6)))
    except (TypeError, ValueError, FloatingPointError):
        pass
    if f12_hz is not None and np.isfinite(f12_hz) and float(f12_hz) < float(f3_hz):
        try:
            den12 = float(np.log(float(f3_hz) / float(f12_hz)))
            if np.isfinite(den12) and den12 > 1e-9:
                estimates.append(float(np.log(c12) / (2.0 * den12)))
        except (TypeError, ValueError, FloatingPointError):
            pass
    if not estimates:
        return float("nan"), 0.0
    order = float(np.clip(np.median(np.asarray(estimates, dtype=float)), 1.0, 8.0))
    ratio6 = float(np.power(c6, 1.0 / (2.0 * order)))
    model_f6 = float(f3_hz) / max(ratio6, 1e-9)
    confidence = 0.62
    if f12_hz is not None and np.isfinite(f12_hz):
        confidence += 0.18
    if len(estimates) >= 2:
        spread = float(np.max(estimates) - np.min(estimates))
        confidence *= float(np.clip(1.0 - spread / 4.0, 0.55, 1.0))
    return float(model_f6), float(np.clip(confidence, 0.0, 1.0))


def estimate_lf_rolloff_f6(
    f_hz,
    mag_db,
    *,
    min_hz: float,
    max_hz: float,
    ref_min_hz: float,
    ref_max_hz: float,
    search_max_hz: float,
    smooth_oct: float,
    default_hz: float = float("nan"),
) -> LfRolloffEstimate:
    ff, mm = _sorted_finite_xy(f_hz, mag_db)
    default_f = float(default_hz)
    if ff.size < 16:
        return _fallback_estimate(default_f, reason="insufficient_samples")
    mm_use = _smooth_mag(ff, mm, smooth_oct)
    ref_mask = (ff >= float(ref_min_hz)) & (ff <= float(ref_max_hz))
    reference_band = "configured"
    if int(np.count_nonzero(ref_mask)) < 8:
        ref_mask = (ff >= 63.0) & (ff <= 250.0)
        reference_band = "fallback_63_250"
    ref_db, ref_spread_db = _reference_stats(mm_use[ref_mask])
    if not np.isfinite(ref_db):
        return _fallback_estimate(default_f, reason="reference_unavailable", reference_band=reference_band)
    lf_hi = float(min(float(search_max_hz), float(ref_min_hz), float(max_hz)))
    lf_mask = (ff >= float(min_hz)) & (ff <= lf_hi)
    if int(np.count_nonzero(lf_mask)) < 8:
        return _fallback_estimate(
            default_f,
            ref_db=float(ref_db),
            ref_spread_db=float(ref_spread_db),
            reason="insufficient_low_band_samples",
            reference_band=reference_band,
        )
    f_lo = np.asarray(ff[lf_mask], dtype=float)
    m_lo = np.asarray(mm_use[lf_mask], dtype=float)

    f3 = _stable_crossing_hz(f_lo, m_lo, float(ref_db) - 3.0, min_span_oct=0.20, near_tolerance_db=0.8)
    stable_f6 = _stable_crossing_hz(f_lo, m_lo, float(ref_db) - 6.0)
    f12 = _stable_crossing_hz(f_lo, m_lo, float(ref_db) - 12.0, min_span_oct=0.18, near_tolerance_db=1.2)

    if stable_f6 is None or not np.isfinite(stable_f6):
        reason = "no_stable_crossing"
        if not np.isfinite(ref_spread_db) or ref_spread_db > 6.0:
            reason = "unstable_reference,no_stable_crossing"
        return _fallback_estimate(
            default_f,
            ref_db=float(ref_db),
            ref_spread_db=float(ref_spread_db),
            reason=reason,
            reference_band=reference_band,
        )

    model_f6, model_conf = _model_f6_from_crossings(f3, stable_f6, f12)
    low_coverage_oct = float(np.log2(float(stable_f6) / max(float(f_lo[0]), 1e-9)))
    model_delta_oct = (
        float(abs(np.log2(float(model_f6) / max(float(stable_f6), 1e-9))))
        if np.isfinite(model_f6)
        else float("nan")
    )
    candidates = [float(stable_f6)]
    method = "stable_crossing"
    confidence = 0.70
    if np.isfinite(model_f6) and model_conf >= 0.55 and float(model_f6) > (float(stable_f6) + 2.0):
        candidates.append(float(model_f6))
        method = "stable_crossing_model_guard"
        confidence = max(confidence, float(model_conf))
    f6 = float(max(candidates))
    f6 = float(np.clip(f6, float(min_hz), float(max_hz)))
    reference_quality = float(np.clip(1.0 - max(0.0, float(ref_spread_db) - 2.0) / 12.0, 0.0, 1.0))
    coverage_quality = float(np.clip(low_coverage_oct / 0.20, 0.0, 1.0))
    model_quality = (
        float(np.clip(0.30 / max(model_delta_oct, 1e-9), 0.0, 1.0))
        if np.isfinite(model_delta_oct)
        else 1.0
    )
    confidence = float(np.clip(confidence * reference_quality * coverage_quality * model_quality, 0.0, 1.0))
    reasons: list[str] = []
    if low_coverage_oct < 0.20:
        reasons.append("insufficient_low_coverage")
    if not np.isfinite(ref_spread_db) or ref_spread_db > 6.0:
        reasons.append("unstable_reference")
    if np.isfinite(model_delta_oct) and model_delta_oct > 0.30:
        reasons.append("model_disagreement")
    if confidence < 0.55:
        reasons.append("low_confidence")
    usable = not reasons
    return LfRolloffEstimate(
        f6_hz=float(f6),
        stable_f6_hz=float(stable_f6),
        model_f6_hz=float(model_f6) if np.isfinite(model_f6) else float("nan"),
        ref_db=float(ref_db),
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        method=str(method),
        usable=bool(usable),
        reason="ok" if usable else ",".join(reasons),
        ref_spread_db=float(ref_spread_db),
        low_coverage_oct=float(low_coverage_oct),
        model_delta_oct=float(model_delta_oct),
        reference_band=reference_band,
    )
