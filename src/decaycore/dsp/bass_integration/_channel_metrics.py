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

from typing import Any

import numpy as np
import scipy.signal

from ...auto_mode.shared import (
    AUTO_MODE_BASS_INTEGRATION_GUARD_HI_RATIO,
    AUTO_MODE_BASS_INTEGRATION_GUARD_LO_RATIO,
)
from ...io.measurement_bundle import TransferData
from ._utils import _band_mask, _interp_complex_response, _safe_float


def _channel_overlap_metrics(
    main: TransferData,
    sub: TransferData,
    total: TransferData,
    *,
    lo_hz: float,
    hi_hz: float,
) -> dict[str, float]:
    freqs = np.asarray(total.freqs_hz, dtype=float)
    mask = _band_mask(freqs, lo_hz, hi_hz)
    if int(np.count_nonzero(mask)) < 3:
        return {
            "overlap_ratio": float("nan"),
            "overlap_ripple_db": float("nan"),
            "cancellation_risk": float("nan"),
            "sub_dominance_db": float("nan"),
        }

    main_spec = _interp_complex_response(main, freqs)
    sub_spec = _interp_complex_response(sub, freqs)
    total_spec = _interp_complex_response(total, freqs)

    if not (np.all(np.isfinite(main_spec)) and np.all(np.isfinite(sub_spec)) and np.all(np.isfinite(total_spec))):
        return {
            "overlap_ratio": float("nan"),
            "overlap_ripple_db": float("nan"),
            "cancellation_risk": float("nan"),
            "sub_dominance_db": float("nan"),
        }

    main_mag = np.maximum(np.abs(main_spec[mask]), 1e-12)
    sub_mag = np.maximum(np.abs(sub_spec[mask]), 1e-12)
    total_mag = np.maximum(np.abs(total_spec[mask]), 1e-12)
    stronger = np.maximum(main_mag, sub_mag)
    weaker = np.minimum(main_mag, sub_mag)
    overlap_ratio = float(np.mean(weaker / np.maximum(stronger, 1e-12)))

    total_db = 20.0 * np.log10(np.maximum(total_mag, 1e-12))
    if total_db.size >= 5:
        ripple_db = float(np.percentile(total_db, 95.0) - np.percentile(total_db, 5.0))
    else:
        ripple_db = float(np.max(total_db) - np.min(total_db))

    main_phase = np.angle(main_spec[mask])
    sub_phase = np.angle(sub_spec[mask])
    phase_delta = np.angle(np.exp(1j * (sub_phase - main_phase)))
    # Phase opposition: 0 when in phase (≤90°), ramps to 1 at 180°.
    # Pure cosine-based — no arbitrary offset shifting the zero crossing.
    phase_opposition = np.clip(-np.cos(phase_delta), 0.0, 1.0)
    depth_db = 20.0 * np.log10(np.maximum(stronger, 1e-12) / np.maximum(total_mag, 1e-12))
    depth_weight = np.clip(depth_db / 12.0, 0.0, 1.0)
    cancellation_risk = float(np.mean(phase_opposition * depth_weight))

    sub_dominance_db = float(np.median(20.0 * np.log10(sub_mag / np.maximum(main_mag, 1e-12))))
    return {
        "overlap_ratio": float(overlap_ratio),
        "overlap_ripple_db": float(ripple_db),
        "cancellation_risk": float(cancellation_risk),
        "sub_dominance_db": float(sub_dominance_db),
    }


def _fc_guard_ratios(fc: float, lo_ratio: float, hi_ratio: float) -> tuple[float, float]:
    """Scale guard band ratios mildly with XO frequency.

    Higher XO → slightly narrower relative band (modes denser, transitions sharper).
    Lower XO → slightly wider band. Scale clamped to ±10% to stay conservative.
    """
    _scale = float(np.clip(80.0 / max(fc, 1.0), 0.9, 1.1))
    lo_eff = max(0.05, lo_ratio * (2.0 - _scale))
    hi_eff = max(lo_eff + 0.05, hi_ratio * _scale)
    return lo_eff, hi_eff


def compute_overlap_metrics(
    main: TransferData,
    sub: TransferData,
    total: TransferData,
    *,
    fc_hz: float,
    guard_lo_ratio: float = AUTO_MODE_BASS_INTEGRATION_GUARD_LO_RATIO,
    guard_hi_ratio: float = AUTO_MODE_BASS_INTEGRATION_GUARD_HI_RATIO,
) -> dict[str, float]:
    fc = _safe_float(fc_hz, 80.0)
    lo_ratio_eff, hi_ratio_eff = _fc_guard_ratios(fc, max(0.05, _safe_float(guard_lo_ratio, 0.60)), max(0.05, _safe_float(guard_hi_ratio, 1.40)))
    lo = max(5.0, fc * lo_ratio_eff)
    hi = max(lo + 1.0, fc * hi_ratio_eff)
    ch = _channel_overlap_metrics(main, sub, total, lo_hz=lo, hi_hz=hi)
    return {
        "guard_lo_hz": float(lo),
        "guard_hi_hz": float(hi),
        "overlap_ratio": float(ch.get("overlap_ratio", float("nan"))),
        "overlap_ripple_db": float(ch.get("overlap_ripple_db", float("nan"))),
    }


def compute_cancellation_metrics(
    main: TransferData,
    sub: TransferData,
    total: TransferData,
    *,
    fc_hz: float,
    guard_lo_ratio: float = AUTO_MODE_BASS_INTEGRATION_GUARD_LO_RATIO,
    guard_hi_ratio: float = AUTO_MODE_BASS_INTEGRATION_GUARD_HI_RATIO,
) -> dict[str, float]:
    fc = _safe_float(fc_hz, 80.0)
    lo_ratio_eff, hi_ratio_eff = _fc_guard_ratios(fc, max(0.05, _safe_float(guard_lo_ratio, 0.60)), max(0.05, _safe_float(guard_hi_ratio, 1.40)))
    lo = max(5.0, fc * lo_ratio_eff)
    hi = max(lo + 1.0, fc * hi_ratio_eff)
    ch = _channel_overlap_metrics(main, sub, total, lo_hz=lo, hi_hz=hi)
    return {
        "guard_lo_hz": float(lo),
        "guard_hi_hz": float(hi),
        "cancellation_risk": float(ch.get("cancellation_risk", float("nan"))),
    }


def compute_sub_dominance_metrics(
    main: TransferData,
    sub: TransferData,
    *,
    fc_hz: float,
) -> dict[str, float]:
    freqs = np.asarray(main.freqs_hz, dtype=float)
    fc = _safe_float(fc_hz, 80.0)
    lo = max(5.0, 0.35 * fc)
    hi = max(lo + 1.0, 1.00 * fc)
    mask = _band_mask(freqs, lo, hi)
    if int(np.count_nonzero(mask)) < 3:
        return {
            "sub_dominance_db": float("nan"),
            "sub_dominance_lo_hz": float(lo),
            "sub_dominance_hi_hz": float(hi),
        }
    main_spec = _interp_complex_response(main, freqs)
    sub_spec = _interp_complex_response(sub, freqs)
    main_mag = np.maximum(np.abs(main_spec[mask]), 1e-12)
    sub_mag = np.maximum(np.abs(sub_spec[mask]), 1e-12)
    return {
        "sub_dominance_db": float(np.median(20.0 * np.log10(sub_mag / np.maximum(main_mag, 1e-12)))),
        "sub_dominance_lo_hz": float(lo),
        "sub_dominance_hi_hz": float(hi),
    }


def _rolling_percentile(values: np.ndarray, percentile: float, half_window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.asarray([], dtype=float)
    half = max(1, int(half_window))
    window = 2 * half + 1
    padded = np.pad(arr, half, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=window)
    return np.percentile(windows, float(percentile), axis=1)


def _channel_predicted_sum_metrics(
    total: TransferData,
    *,
    lo_hz: float,
    hi_hz: float,
    fc_hz: float,
) -> dict[str, float]:
    freqs = np.asarray(total.freqs_hz, dtype=float)
    mask = _band_mask(freqs, lo_hz, hi_hz)
    if int(np.count_nonzero(mask)) < 5:
        return {
            "predicted_sum_flatness_db": float("nan"),
            "predicted_sum_dip_depth_db": float("nan"),
            "predicted_sum_peak_excess_db": float("nan"),
            "null_depth_db": float("nan"),
            "null_width_hz": float("nan"),
            "null_center_hz": float("nan"),
            "null_severity": float("nan"),
            "dip_p10_db": float("nan"),
        }

    band_freqs = freqs[mask]
    total_spec = _interp_complex_response(total, freqs)
    total_db = 20.0 * np.log10(np.maximum(np.abs(total_spec[mask]), 1e-12))
    kernel_size = 5 if total_db.size >= 5 else 3
    if kernel_size % 2 == 0:
        kernel_size += 1
    stabilized_db = scipy.signal.medfilt(
        total_db,
        kernel_size=max(3, min(kernel_size, int(total_db.size) | 1)),
    )
    half_window = max(2, min(16, int(total_db.size // 10)))
    local_ref_db = _rolling_percentile(stabilized_db, 85.0, half_window)
    local_floor_db = _rolling_percentile(stabilized_db, 50.0, half_window)
    dips_db = np.maximum(0.0, local_ref_db - stabilized_db)
    peak_excess_db = np.maximum(0.0, stabilized_db - local_floor_db)

    flatness_db = float(np.percentile(stabilized_db, 95.0) - np.percentile(stabilized_db, 5.0))
    dip_depth_db = float(np.max(dips_db)) if dips_db.size else float("nan")
    peak_db = float(np.max(peak_excess_db)) if peak_excess_db.size else float("nan")

    max_idx = int(np.argmax(dips_db)) if dips_db.size else 0
    null_center_hz = float(band_freqs[max_idx]) if band_freqs.size else float("nan")
    null_depth_db = float(dips_db[max_idx]) if dips_db.size else float("nan")
    threshold_db = max(1.0, 0.45 * float(null_depth_db)) if np.isfinite(null_depth_db) else 1.0
    above_threshold = np.asarray(dips_db >= threshold_db, dtype=bool)
    if above_threshold.size and above_threshold[max_idx]:
        starts = np.flatnonzero(above_threshold & ~np.r_[False, above_threshold[:-1]])
        ends = np.flatnonzero(above_threshold & ~np.r_[above_threshold[1:], False])
        run_idx = int(np.searchsorted(starts, max_idx, side="right") - 1)
        if 0 <= run_idx < starts.size and int(ends[run_idx]) >= max_idx:
            left_idx = int(starts[run_idx])
            right_idx = int(ends[run_idx])
        else:
            left_idx = max_idx
            right_idx = max_idx
    else:
        left_idx = max_idx
        right_idx = max_idx
    null_width_hz = float(band_freqs[right_idx] - band_freqs[left_idx]) if band_freqs.size else float("nan")
    width_ref_hz = max(5.0, 0.20 * float(max(fc_hz, 1.0)))
    # log2-based width weight: less steep than sqrt, no hard floor that hides narrow deep nulls.
    # Minimum 0.5x so a narrow null is never completely ignored.
    _log_weight = 1.0 + float(np.log2(max(max(float(null_width_hz), 0.0), 1.0) / max(float(width_ref_hz), 1.0)))
    null_severity = float(max(0.0, float(null_depth_db)) * max(0.5, _log_weight)) if np.isfinite(null_depth_db) else float("nan")

    return {
        "predicted_sum_flatness_db": float(flatness_db),
        "predicted_sum_dip_depth_db": float(dip_depth_db),
        "predicted_sum_peak_excess_db": float(peak_db),
        "null_depth_db": float(null_depth_db),
        "null_width_hz": float(null_width_hz),
        "null_center_hz": float(null_center_hz),
        "null_severity": float(null_severity),
        "dip_p10_db": float(np.percentile(stabilized_db, 10.0)),
    }


def _channel_overlap_extension_metrics(
    main: TransferData,
    sub: TransferData,
    total: TransferData,
    *,
    fc_hz: float,
    sub_lpf_hz: float,
) -> dict[str, float | bool]:
    fc = _safe_float(fc_hz, 80.0)
    hi = _safe_float(sub_lpf_hz, fc)
    if not np.isfinite(fc) or not np.isfinite(hi) or hi <= (fc + 1e-6):
        return {
            "overlap_extension_active": False,
            "overlap_extension_lo_hz": float(fc),
            "overlap_extension_hi_hz": float(hi),
            "overlap_extension_flatness_db": float("nan"),
            "overlap_extension_cancellation_risk": float("nan"),
            "overlap_extension_peak_excess_db": float("nan"),
            "overlap_extension_sub_dominance_db": float("nan"),
        }

    overlap_metrics = _channel_overlap_metrics(
        main,
        sub,
        total,
        lo_hz=float(fc),
        hi_hz=float(hi),
    )
    predicted_metrics = _channel_predicted_sum_metrics(
        total,
        lo_hz=float(fc),
        hi_hz=float(hi),
        fc_hz=float(fc),
    )
    return {
        "overlap_extension_active": True,
        "overlap_extension_lo_hz": float(fc),
        "overlap_extension_hi_hz": float(hi),
        "overlap_extension_flatness_db": float(
            _safe_float(predicted_metrics.get("predicted_sum_flatness_db", float("nan")), float("nan"))
        ),
        "overlap_extension_cancellation_risk": float(
            _safe_float(overlap_metrics.get("cancellation_risk", float("nan")), float("nan"))
        ),
        "overlap_extension_peak_excess_db": float(
            _safe_float(predicted_metrics.get("predicted_sum_peak_excess_db", float("nan")), float("nan"))
        ),
        "overlap_extension_sub_dominance_db": float(
            _safe_float(overlap_metrics.get("sub_dominance_db", float("nan")), float("nan"))
        ),
    }


def _channel_metric_summary(
    left_metrics: dict[str, float],
    right_metrics: dict[str, float],
    key: str,
    *,
    absolute: bool = False,
) -> dict[str, float]:
    left = _safe_float((left_metrics or {}).get(key, float("nan")), float("nan"))
    right = _safe_float((right_metrics or {}).get(key, float("nan")), float("nan"))
    valid = [float(v) for v in (left, right) if np.isfinite(v)]
    if absolute:
        avg = float(np.mean(np.asarray([abs(v) for v in valid], dtype=float))) if valid else float("nan")
        worst = float(max((abs(v) for v in valid), default=float("nan")))
    else:
        avg = float(np.mean(np.asarray(valid, dtype=float))) if valid else float("nan")
        worst = float(max(valid)) if valid else float("nan")
    return {
        f"{key}_l": float(left),
        f"{key}_r": float(right),
        f"{key}_avg": float(avg),
        f"{key}_worst": float(worst),
    }


def _metric_delta(left_value: Any, right_value: Any) -> float:
    left = _safe_float(left_value, float("nan"))
    right = _safe_float(right_value, float("nan"))
    if np.isfinite(left) and np.isfinite(right):
        return float(abs(float(left) - float(right)))
    return float("nan")


