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
import scipy.ndimage


def _weighted_percentile(values: np.ndarray, weights: np.ndarray, percentile: float) -> float:
    v = np.asarray(values, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    valid = np.isfinite(v) & np.isfinite(w) & (w > 0.0)
    if int(np.count_nonzero(valid)) < 4:
        return float("nan")
    vv = v[valid]
    ww = w[valid]
    order = np.argsort(vv, kind="mergesort")
    vv = vv[order]
    ww = ww[order]
    cumulative = np.cumsum(ww)
    total = float(cumulative[-1])
    if not np.isfinite(total) or total <= 0.0:
        return float("nan")
    target = float(np.clip(percentile, 0.0, 100.0)) * total / 100.0
    return float(vv[int(np.clip(np.searchsorted(cumulative, target, side="left"), 0, vv.size - 1))])


def _delay_compensated_filter_phase(freq_axis: np.ndarray, impulse: np.ndarray, fs: float) -> np.ndarray:
    f = np.asarray(freq_axis, dtype=float).reshape(-1)
    ir = np.asarray(impulse, dtype=float).reshape(-1)
    if ir.size < 8 or f.size != (ir.size // 2 + 1) or not (np.isfinite(fs) and fs > 0.0):
        return np.asarray([], dtype=float)
    h = np.fft.rfft(ir, n=ir.size)
    peak_idx = int(np.argmax(np.abs(ir)))
    if peak_idx > 0:
        h *= np.exp(1j * 2.0 * np.pi * f * (float(peak_idx) / float(fs)))
    return np.unwrap(np.angle(h))


def _gd_profile_metrics(
    freq_axis: np.ndarray,
    phase_rad: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | np.ndarray]:
    f = np.asarray(freq_axis, dtype=float).reshape(-1)
    p = np.asarray(phase_rad, dtype=float).reshape(-1)
    w = np.asarray(weights, dtype=float).reshape(-1)
    sel = np.asarray(mask, dtype=bool).reshape(-1)
    if not (f.size == p.size == w.size == sel.size) or int(np.count_nonzero(sel)) < 8:
        return {
            "rms_ms": float("nan"),
            "p95_ms": float("nan"),
            "gradient_p95_ms_per_oct": float("nan"),
            "curve_ms": np.asarray([], dtype=float),
        }

    phase_u = np.unwrap(p)
    omega = 2.0 * np.pi * f
    gd_ms = np.nan_to_num(
        -np.gradient(phase_u, omega) * 1000.0,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if gd_ms.size >= 8:
        gd_ms = scipy.ndimage.gaussian_filter1d(gd_ms, sigma=2.0, mode="nearest")

    center = _weighted_percentile(gd_ms[sel], w[sel], 50.0)
    if not np.isfinite(center):
        center = 0.0
    gd_residual = gd_ms - float(center)
    selected_weights = np.asarray(w[sel], dtype=float)
    denom = float(np.sum(selected_weights))
    if not np.isfinite(denom) or denom <= 1e-12:
        return {
            "rms_ms": float("nan"),
            "p95_ms": float("nan"),
            "gradient_p95_ms_per_oct": float("nan"),
            "curve_ms": np.asarray(gd_residual, dtype=float),
        }

    selected_gd = np.asarray(gd_residual[sel], dtype=float)
    rms_ms = float(np.sqrt(np.sum(selected_weights * selected_gd**2) / denom))
    p95_ms = _weighted_percentile(np.abs(selected_gd), selected_weights, 95.0)

    log_f = np.log2(np.maximum(f, 1e-9))
    gd_gradient = np.nan_to_num(
        np.gradient(gd_residual, log_f),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    gradient_p95 = _weighted_percentile(np.abs(gd_gradient[sel]), selected_weights, 95.0)
    return {
        "rms_ms": float(rms_ms),
        "p95_ms": float(p95_ms),
        "gradient_p95_ms_per_oct": float(gradient_p95),
        "curve_ms": np.asarray(gd_residual, dtype=float),
    }


def _relative_improvement(before: float, after: float, *, floor: float) -> float:
    if not (np.isfinite(before) and np.isfinite(after)) or before <= float(floor):
        return float("nan")
    return float(np.clip((float(before) - float(after)) / max(float(before), float(floor)), -1.0, 1.0))


def compute_realized_phase_gd_metrics(
    *,
    freq_axis: np.ndarray,
    measured_phase_rad: np.ndarray,
    impulse: np.ndarray,
    fs: float,
    phase_limit_hz: float,
    confidence_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Measure timing improvement from the final exported FIR realization."""
    f = np.asarray(freq_axis, dtype=float).reshape(-1)
    measured = np.asarray(measured_phase_rad, dtype=float).reshape(-1)
    ir = np.asarray(impulse, dtype=float).reshape(-1)
    if f.size < 16 or measured.size != f.size or ir.size < 16:
        return {}
    if not (np.all(np.isfinite(f)) and np.all(np.isfinite(measured)) and np.all(np.isfinite(ir))):
        return {}

    filter_phase = _delay_compensated_filter_phase(f, ir, float(fs))
    if filter_phase.size != f.size:
        return {}

    try:
        conf = np.asarray(confidence_mask, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        conf = np.asarray([], dtype=float)
    if conf.size != f.size:
        conf = np.ones_like(f, dtype=float)
    conf = np.clip(np.nan_to_num(conf, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)

    hi_hz = float(phase_limit_hz)
    if not np.isfinite(hi_hz) or hi_hz <= 20.0:
        return {}
    mask = np.isfinite(f) & (f >= 20.0) & (f <= hi_hz) & (conf >= 0.05)
    if int(np.count_nonzero(mask)) < 8:
        return {}
    weights = np.where(mask, np.maximum(conf, 0.05), 0.0)

    before = _gd_profile_metrics(f, measured, weights, mask)
    after = _gd_profile_metrics(f, np.unwrap(measured) + filter_phase, weights, mask)
    filter_only = _gd_profile_metrics(f, filter_phase, weights, mask)

    rms_improvement = _relative_improvement(
        float(before["rms_ms"]),
        float(after["rms_ms"]),
        floor=0.10,
    )
    p95_improvement = _relative_improvement(
        float(before["p95_ms"]),
        float(after["p95_ms"]),
        floor=0.20,
    )
    gradient_improvement = _relative_improvement(
        float(before["gradient_p95_ms_per_oct"]),
        float(after["gradient_p95_ms_per_oct"]),
        floor=0.50,
    )
    finite_improvements = [
        (0.40, rms_improvement),
        (0.35, p95_improvement),
        (0.25, gradient_improvement),
    ]
    usable = [(weight, value) for weight, value in finite_improvements if np.isfinite(value)]
    if usable:
        weight_sum = float(sum(weight for weight, _ in usable))
        improvement_score = float(sum(weight * value for weight, value in usable) / max(weight_sum, 1e-12))
    else:
        improvement_score = float("nan")

    return {
        "phase_realized_metrics_source": "measurement_plus_final_fir",
        "phase_realized_band_lo_hz": 20.0,
        "phase_realized_band_hi_hz": float(hi_hz),
        "phase_realized_gd_before_rms_ms": float(before["rms_ms"]),
        "phase_realized_gd_after_rms_ms": float(after["rms_ms"]),
        "phase_realized_gd_before_p95_ms": float(before["p95_ms"]),
        "phase_realized_gd_after_p95_ms": float(after["p95_ms"]),
        "phase_realized_gd_gradient_before_p95_ms_per_oct": float(
            before["gradient_p95_ms_per_oct"]
        ),
        "phase_realized_gd_gradient_after_p95_ms_per_oct": float(
            after["gradient_p95_ms_per_oct"]
        ),
        "phase_realized_filter_gd_gradient_p95_ms_per_oct": float(
            filter_only["gradient_p95_ms_per_oct"]
        ),
        "phase_realized_gd_rms_improvement_frac": float(rms_improvement),
        "phase_realized_gd_p95_improvement_frac": float(p95_improvement),
        "phase_realized_gd_gradient_improvement_frac": float(gradient_improvement),
        "phase_realized_gd_improvement_score": float(improvement_score),
    }


__all__ = ["compute_realized_phase_gd_metrics"]
