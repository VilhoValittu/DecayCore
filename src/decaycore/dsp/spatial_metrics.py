# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""L/R correction filter similarity metric (internal/deprecated as primary stereo metric).

compute_filter_similarity_metric() (formerly compute_iacc()) computes the
normalized cross-correlation peak between the left and right FIR correction
filter IRs.  This is a proxy for "how similarly the two channels are being
corrected", NOT a meaningful room-acoustic or stereo-image IACC metric.

Because correction filters are aligned before this is computed and
full-lag abs(max xcorr) is used, Early/All values gravitate toward ~1.0
and ITD toward ~0 ms regardless of room asymmetry.  This makes the metric
near-constant and unsuitable as a primary user-facing L/R difference report.

Kept internally for backwards compatibility only.  For user-facing L/R
difference reporting use lr_difference_metrics.compute_lr_difference_metrics().
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class IACCResult:
    """Result of an IACC computation."""

    early: float   # IACC_E — cross-correlation magnitude over early window
    late: float    # IACC_L — cross-correlation magnitude over late window
    all: float     # IACC_A — cross-correlation magnitude over full IR
    tau_ms: float  # Time-lag at peak of full-IR cross-correlation (ms)


_IACC_NAN = IACCResult(early=float("nan"), late=float("nan"), all=float("nan"), tau_ms=float("nan"))


def _normalized_xcorr_peak(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Return (max_abs_normalized_xcorr, lag_samples) for 1-D arrays a and b.

    Normalisation: xcorr / sqrt(energy_a * energy_b).
    Returns (nan, nan) if either signal has zero energy.
    """
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan")

    e_a = float(np.dot(a, a))
    e_b = float(np.dot(b, b))
    if e_a <= 0.0 or e_b <= 0.0:
        return float("nan"), float("nan")

    norm = float(np.sqrt(e_a * e_b))
    xcorr = np.correlate(a, b, mode="full")
    peak_idx = int(np.argmax(np.abs(xcorr)))
    peak_val = float(np.abs(xcorr[peak_idx]) / norm)
    # Lag in samples relative to zero-lag centre.
    lag = float(peak_idx - (a.size - 1))
    return float(np.clip(peak_val, 0.0, 1.0)), lag


def _window_around_peak(ir: np.ndarray, fs: int, half_ms: float = 60.0) -> tuple[int, int]:
    """Return (start, end) sample indices for a window centred on the IR peak."""
    half = max(1, int(round(half_ms * 1e-3 * float(fs))))
    if ir.size == 0:
        return 0, 0
    peak = int(np.argmax(np.abs(ir)))
    start = max(0, peak - half)
    end = min(ir.size, peak + half)
    return start, end


def compute_filter_similarity_metric(
    ir_l: np.ndarray,
    ir_r: np.ndarray,
    fs: int,
    *,
    early_ms: float = 80.0,
) -> IACCResult:
    """Compute L/R correction filter similarity from left and right FIR IRs.

    .. deprecated::
        This is an internal/secondary metric only.  It is NOT a meaningful
        room-acoustic IACC or stereo-image metric.  Values typically gravitate
        toward ~1.0 because correction filters are pre-aligned and full-lag
        abs(max xcorr) is used.  Use lr_difference_metrics for user-facing
        L/R mismatch reporting.

    Parameters
    ----------
    ir_l, ir_r:
        1-D float arrays (FIR correction filter coefficients).
    fs:
        Sample rate in Hz.
    early_ms:
        Duration of the "early" window in milliseconds.

    Returns
    -------
    IACCResult with early, late, all, tau_ms fields.
    ``nan`` fields indicate insufficient data.

    """
    try:
        l = np.asarray(ir_l, dtype=float).reshape(-1)
        r = np.asarray(ir_r, dtype=float).reshape(-1)
        if l.size < 4 or r.size < 4:
            return _IACC_NAN
        fs_i = max(1, int(fs))

        # Align lengths to the shorter IR.
        n = min(l.size, r.size)
        l = l[:n]
        r = r[:n]

        # IACC_A — full IR.
        iacc_all, lag_samples = _normalized_xcorr_peak(l, r)
        tau_ms = float(lag_samples) / float(fs_i) * 1000.0 if np.isfinite(lag_samples) else float("nan")

        # Split into early (around peak) and late windows.
        start_l, end_l = _window_around_peak(l, fs_i, half_ms=early_ms / 2.0)
        early_len = max(1, int(round(early_ms * 1e-3 * float(fs_i))))

        # Use a fixed window from the combined peak rather than per-channel peaks
        # so early/late windows are consistent.
        combined_env = np.abs(l) + np.abs(r)
        peak_combined = int(np.argmax(combined_env))
        early_start = max(0, peak_combined - early_len // 4)
        early_end = min(n, early_start + early_len)
        late_start = early_end
        late_end = n

        l_early = l[early_start:early_end]
        r_early = r[early_start:early_end]
        l_late = l[late_start:late_end]
        r_late = r[late_start:late_end]

        iacc_early, _ = _normalized_xcorr_peak(l_early, r_early) if l_early.size >= 4 else (float("nan"), 0.0)
        iacc_late, _ = _normalized_xcorr_peak(l_late, r_late) if l_late.size >= 4 else (float("nan"), 0.0)

        return IACCResult(
            early=float(iacc_early),
            late=float(iacc_late),
            all=float(iacc_all),
            tau_ms=float(tau_ms),
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
    ):
        return _IACC_NAN


def compute_iacc(
    ir_l: np.ndarray,
    ir_r: np.ndarray,
    fs: int,
    *,
    early_ms: float = 80.0,
) -> IACCResult:
    """Compatibility alias for compute_filter_similarity_metric(). Deprecated."""
    return compute_filter_similarity_metric(ir_l, ir_r, fs, early_ms=early_ms)


__all__ = ["IACCResult", "compute_filter_similarity_metric", "compute_iacc"]
