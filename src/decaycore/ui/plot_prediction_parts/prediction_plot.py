# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Channel-level data preparation for interactive Results plots.

This module deliberately contains no HTML generation.  The relatively costly
FFT and smoothing work is performed once per channel and the resulting data is
reused by the compact overview and the lazily rendered diagnostic figures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.fft
import scipy.ndimage

from ...dsp.hybrid_iir import peaking_eq_response
from ..plot_common import (
    GD_SMOOTH_OCT,
    PHASE_SMOOTH_OCT,
    _maybe_shift_to_abs,
    _prepare_curve_for_target_plot,
    _view_mags_for_plot,
    calculate_clean_gd,
    remove_ir_peak_delay,
    smooth_complex,
)

_PLOT_EXCEPTIONS = (
    RuntimeError,
    OSError,
    ImportError,
    TypeError,
    ValueError,
    AttributeError,
    KeyError,
    IndexError,
    OverflowError,
    FloatingPointError,
    NameError,
)


@dataclass(frozen=True)
class ChannelPlotData:
    """Finite, display-ready curves for one output channel."""

    channel_key: str
    fs_hz: int
    freq_hz: np.ndarray
    measured_db: np.ndarray
    predicted_exported_db: np.ndarray
    predicted_compensated_db: np.ndarray
    filter_exported_db: np.ndarray
    filter_compensated_db: np.ndarray
    filter_phase_deg: np.ndarray
    filter_group_delay_ms: np.ndarray
    confidence: np.ndarray | None
    afdw_bw_oct: np.ndarray | None
    target_freq_hz: np.ndarray
    effective_target_db: np.ndarray | None
    requested_target_db: np.ndarray | None
    correction_min_hz: float
    correction_max_hz: float
    lf_guard_hz: float
    filter_delay_ms: float
    auto_global_gain_db: float
    auto_headroom_db: float

    @property
    def full_max_hz(self) -> float:
        return float(min(20000.0, 0.5 * float(self.fs_hz)))

    @property
    def has_valid_correction_range(self) -> bool:
        return bool(
            np.isfinite(self.correction_min_hz)
            and np.isfinite(self.correction_max_hz)
            and 0.0 < self.correction_min_hz < self.correction_max_hz <= self.full_max_hz
        )


def _build_hybrid_iir_response(
    biquads: list[dict],
    f_lin: np.ndarray,
    fs: float,
) -> np.ndarray:
    """Return the stored hybrid-IIR response, or unity when unavailable."""
    response = np.ones(len(f_lin), dtype=complex)
    for biquad in biquads or []:
        try:
            response *= peaking_eq_response(
                f_lin,
                fs,
                float(biquad["freq"]),
                float(biquad["q"]),
                float(biquad["gain"]),
            )
        except _PLOT_EXCEPTIONS:
            continue
    return response


def _prediction_plot_fft_context(*, filt_ir, fs, target_stats) -> dict:
    min_fft_size = 131072
    n_fft = max(len(filt_ir) * 4, min_fft_size)
    f_lin = scipy.fft.rfftfreq(n_fft, d=1 / fs)
    h_filt = scipy.fft.rfft(filt_ir, n=n_fft)
    h_filt_display, filt_delay_ms = remove_ir_peak_delay(f_lin, h_filt, filt_ir, fs)
    return {
        "vis_points": 4000,
        "show_afdw": bool(target_stats.get("afdw_active", True)) if target_stats else True,
        "f_lin": f_lin,
        "h_filt": h_filt,
        "h_filt_display": h_filt_display,
        "filt_delay_ms": float(filt_delay_ms),
    }


def _resolve_magnitude_display_offset_db(
    *,
    target_stats,
    target_curve,
    avg_t,
) -> float:
    """Keep low-reference native measurement plots readable around 0 dB."""
    if not isinstance(target_stats, dict):
        return 0.0

    references: list[float] = []
    for key in ("target_level_db_window", "eff_target_db"):
        try:
            value = float(target_stats.get(key, np.nan))
        except _PLOT_EXCEPTIONS:
            continue
        if np.isfinite(value):
            references.append(value)

    try:
        target_abs = np.asarray(_maybe_shift_to_abs(target_curve, avg_t), dtype=float).reshape(-1)
        target_abs = target_abs[np.isfinite(target_abs)]
        if target_abs.size >= 4:
            references.append(float(np.nanmedian(target_abs)))
    except _PLOT_EXCEPTIONS:
        pass

    for reference_db in references:
        if np.isfinite(reference_db) and reference_db < 40.0:
            return float(reference_db)
    return 0.0


def _target_curve_on_display_grid(
    *,
    raw_curve,
    target_freq_hz: np.ndarray,
    display_freq_hz: np.ndarray,
    avg_target_db: float,
    display_offset_db: float,
) -> np.ndarray | None:
    if raw_curve is None or target_freq_hz.size < 2:
        return None
    values = np.asarray(_maybe_shift_to_abs(raw_curve, avg_target_db), dtype=float).reshape(-1)
    if values.size != target_freq_hz.size:
        return None
    values = values - float(display_offset_db)
    return np.interp(display_freq_hz, target_freq_hz, values)


def _afdw_curve_on_display_grid(
    *,
    target_stats: dict | None,
    display_freq_hz: np.ndarray,
) -> np.ndarray | None:
    if not target_stats or not bool(target_stats.get("afdw_active", True)):
        return None
    mode = str(target_stats.get("analysis_mode", "native")).lower()
    if mode == "comparison":
        freq_raw = target_stats.get("cmp_freq_axis")
        bw_raw = target_stats.get(
            "cmp_afdw_bw_plot_oct",
            target_stats.get("cmp_afdw_bw_oct"),
        )
    else:
        freq_raw = target_stats.get("freq_axis")
        bw_raw = target_stats.get(
            "afdw_bw_plot_oct",
            target_stats.get("afdw_bw_oct"),
        )
    if freq_raw is None or bw_raw is None:
        return None
    freq = np.asarray(freq_raw, dtype=float).reshape(-1)
    bandwidth = np.asarray(bw_raw, dtype=float).reshape(-1)
    if freq.size != bandwidth.size or freq.size <= 16:
        return None
    result = np.interp(display_freq_hz, freq, bandwidth)
    result = scipy.ndimage.gaussian_filter1d(result, sigma=5.0)
    return np.clip(result, 1.0 / 96.0, 1.0 / 3.0)


def compute_channel_plot_data(
    *,
    channel_key: str,
    orig_freqs,
    orig_mags,
    orig_phases,
    filt_ir,
    fs,
    target_stats,
    plot_smoothing_level,
) -> ChannelPlotData:
    """Compute all Results curves for one channel without building a figure."""
    stats = dict(target_stats or {})
    fft_ctx = _prediction_plot_fft_context(
        filt_ir=filt_ir,
        fs=fs,
        target_stats=stats,
    )
    f_lin = np.asarray(fft_ctx["f_lin"], dtype=float)
    h_filt = np.asarray(fft_ctx["h_filt"], dtype=complex)
    h_filt_display = np.asarray(fft_ctx["h_filt_display"], dtype=complex)
    avg_target_db = float(stats.get("eff_target_db", 75.0) or 75.0)

    match_range = stats.get(
        "smart_scan_range",
        stats.get("match_range", [500.0, 2000.0]),
    )
    try:
        match_min_hz = float(match_range[0])
        match_max_hz = float(match_range[1])
    except _PLOT_EXCEPTIONS:
        match_min_hz, match_max_hz = 500.0, 2000.0

    target_freq_hz = np.asarray(stats.get("freq_axis", []), dtype=float).reshape(-1)
    effective_target_raw = stats.get("target_mags")
    requested_target_raw = stats.get("selected_target_mags")
    direct_pred_export = None
    direct_pred_compensated = None

    if "measured_mags" in stats and target_freq_hz.size > 1:
        measured_source = stats.get("measured_mags", [])
        direct_measured = np.asarray(
            stats.get("direct_dac_sum_measured_mags", []),
            dtype=float,
        ).reshape(-1)
        if direct_measured.size == target_freq_hz.size:
            measured_source = direct_measured
        measured_aligned = _prepare_curve_for_target_plot(
            target_freq_hz,
            measured_source,
            avg_t_db=avg_target_db,
            target_freqs_hz=target_freq_hz,
            target_mags_db=effective_target_raw,
            f_min_hz=match_min_hz,
            f_max_hz=match_max_hz,
        )
        measured_interp = np.interp(f_lin, target_freq_hz, measured_aligned)
        measured_smoothed = _view_mags_for_plot(
            f_lin,
            measured_interp,
            plot_smoothing_level=plot_smoothing_level,
        )

        predicted_raw = np.asarray(
            _maybe_shift_to_abs(
                stats.get("direct_dac_sum_predicted_mags", []),
                avg_target_db,
            ),
            dtype=float,
        ).reshape(-1)
        if predicted_raw.size == target_freq_hz.size:
            predicted_aligned = _prepare_curve_for_target_plot(
                target_freq_hz,
                predicted_raw,
                avg_t_db=avg_target_db,
                target_freqs_hz=target_freq_hz,
                target_mags_db=effective_target_raw,
                f_min_hz=match_min_hz,
                f_max_hz=match_max_hz,
            )
            direct_pred_export = _view_mags_for_plot(
                f_lin,
                np.interp(f_lin, target_freq_hz, predicted_aligned),
                plot_smoothing_level=plot_smoothing_level,
            )

            predicted_comp_raw = np.asarray(
                _maybe_shift_to_abs(
                    stats.get("direct_dac_sum_predicted_mags_comp", []),
                    avg_target_db,
                ),
                dtype=float,
            ).reshape(-1)
            if predicted_comp_raw.size == target_freq_hz.size:
                predicted_comp_aligned = _prepare_curve_for_target_plot(
                    target_freq_hz,
                    predicted_comp_raw,
                    avg_t_db=avg_target_db,
                    target_freqs_hz=target_freq_hz,
                    target_mags_db=effective_target_raw,
                    f_min_hz=match_min_hz,
                    f_max_hz=match_max_hz,
                )
                direct_pred_compensated = _view_mags_for_plot(
                    f_lin,
                    np.interp(f_lin, target_freq_hz, predicted_comp_aligned),
                    plot_smoothing_level=plot_smoothing_level,
                )
    else:
        measured_source = _prepare_curve_for_target_plot(
            orig_freqs,
            orig_mags,
            avg_t_db=avg_target_db,
            target_freqs_hz=target_freq_hz,
            target_mags_db=effective_target_raw,
            f_min_hz=match_min_hz,
            f_max_hz=match_max_hz,
        )
        measured_smoothed = _view_mags_for_plot(
            f_lin,
            np.interp(f_lin, orig_freqs, measured_source),
            plot_smoothing_level=plot_smoothing_level,
        )

    auto_global_gain_db = float(stats.get("auto_global_gain_db", 0.0) or 0.0)
    auto_headroom_db = float(stats.get("auto_headroom_db", 0.0) or 0.0)
    level_compensation_db = 0.0
    if np.isfinite(auto_global_gain_db) and np.isfinite(auto_headroom_db):
        level_compensation_db = -(auto_global_gain_db + auto_headroom_db)
    elif np.isfinite(auto_global_gain_db):
        level_compensation_db = -auto_global_gain_db

    phase_interp = np.interp(f_lin, orig_freqs, orig_phases)
    h_iir = _build_hybrid_iir_response(
        stats.get("hybrid_iir_biquads", []),
        f_lin,
        float(fs),
    )
    total_spec = (
        10 ** (np.asarray(measured_smoothed, dtype=float) / 20.0)
        * np.exp(1j * np.deg2rad(phase_interp))
        * h_filt
        * h_iir
    )

    if direct_pred_export is not None:
        predicted_exported = np.asarray(direct_pred_export, dtype=float)
        predicted_compensated = (
            np.asarray(direct_pred_compensated, dtype=float)
            if direct_pred_compensated is not None
            else predicted_exported + float(level_compensation_db)
        )
    else:
        predicted_exported = _view_mags_for_plot(
            f_lin,
            20.0 * np.log10(np.abs(total_spec) + 1e-12),
            plot_smoothing_level=plot_smoothing_level,
        )
        predicted_compensated = predicted_exported + float(level_compensation_db)

    phase_smoothed = smooth_complex(f_lin, h_filt_display, PHASE_SMOOTH_OCT)
    filter_phase_deg = (np.rad2deg(np.angle(phase_smoothed)) + 180.0) % 360.0 - 180.0
    gd_smoothed = smooth_complex(f_lin, h_filt_display, GD_SMOOTH_OCT)
    filter_group_delay_ms = calculate_clean_gd(f_lin, gd_smoothed)
    filter_exported_db = 20.0 * np.log10(np.abs(h_filt) + 1e-12)
    filter_compensated_db = filter_exported_db + float(level_compensation_db)

    display_freq_hz = np.geomspace(10.0, float(fs) / 2.0, int(fft_ctx["vis_points"]))
    measured_display = np.interp(display_freq_hz, f_lin, measured_smoothed)
    predicted_exported_display = np.interp(display_freq_hz, f_lin, predicted_exported)
    predicted_compensated_display = np.interp(
        display_freq_hz,
        f_lin,
        predicted_compensated,
    )

    display_offset_db = _resolve_magnitude_display_offset_db(
        target_stats=stats,
        target_curve=effective_target_raw,
        avg_t=avg_target_db,
    )
    if display_offset_db != 0.0:
        measured_display -= display_offset_db
        predicted_exported_display -= display_offset_db
        predicted_compensated_display -= display_offset_db

    confidence = None
    confidence_raw = np.asarray(stats.get("confidence_mask", []), dtype=float).reshape(-1)
    if target_freq_hz.size > 1 and confidence_raw.size == target_freq_hz.size:
        confidence = np.clip(
            np.interp(display_freq_hz, target_freq_hz, confidence_raw),
            0.0,
            1.0,
        )

    correction_min_hz = float(stats.get("mag_c_min", 0.0) or 0.0)
    correction_max_hz = float(stats.get("mag_c_max", 0.0) or 0.0)
    if confidence is not None and 0.0 < correction_min_hz < correction_max_hz:
        confidence[
            (display_freq_hz < correction_min_hz)
            | (display_freq_hz > correction_max_hz)
        ] = np.nan

    return ChannelPlotData(
        channel_key=str(channel_key),
        fs_hz=int(fs),
        freq_hz=display_freq_hz,
        measured_db=np.asarray(measured_display, dtype=float),
        predicted_exported_db=np.asarray(predicted_exported_display, dtype=float),
        predicted_compensated_db=np.asarray(predicted_compensated_display, dtype=float),
        filter_exported_db=np.interp(display_freq_hz, f_lin, filter_exported_db),
        filter_compensated_db=np.interp(
            display_freq_hz,
            f_lin,
            filter_compensated_db,
        ),
        filter_phase_deg=np.interp(display_freq_hz, f_lin, filter_phase_deg),
        filter_group_delay_ms=np.interp(
            display_freq_hz,
            f_lin,
            filter_group_delay_ms,
        ),
        confidence=confidence,
        afdw_bw_oct=_afdw_curve_on_display_grid(
            target_stats=stats,
            display_freq_hz=display_freq_hz,
        ),
        target_freq_hz=display_freq_hz,
        effective_target_db=_target_curve_on_display_grid(
            raw_curve=effective_target_raw,
            target_freq_hz=target_freq_hz,
            display_freq_hz=display_freq_hz,
            avg_target_db=avg_target_db,
            display_offset_db=display_offset_db,
        ),
        requested_target_db=_target_curve_on_display_grid(
            raw_curve=requested_target_raw,
            target_freq_hz=target_freq_hz,
            display_freq_hz=display_freq_hz,
            avg_target_db=avg_target_db,
            display_offset_db=display_offset_db,
        ),
        correction_min_hz=correction_min_hz,
        correction_max_hz=correction_max_hz,
        lf_guard_hz=float(stats.get("lf_guard_hz", 0.0) or 0.0),
        filter_delay_ms=float(fft_ctx["filt_delay_ms"]),
        auto_global_gain_db=auto_global_gain_db,
        auto_headroom_db=auto_headroom_db,
    )


__all__ = [
    "ChannelPlotData",
    "_prediction_plot_fft_context",
    "_resolve_magnitude_display_offset_db",
    "compute_channel_plot_data",
]
