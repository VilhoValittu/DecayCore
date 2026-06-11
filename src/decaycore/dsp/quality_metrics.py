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

import logging

import numpy as np

logger = logging.getLogger("DecayCore")


def _mag_error_db(target_db, measured_db, gain_db, offset_db=None):
    """Return magnitude error in dB: target - (measured + gain).

    `measured_db` is expected to be on the same aligned level axis as `target_db`.
    The optional `offset_db` argument is ignored and kept only for call-site
    compatibility with older reporting code.
    """
    target = np.asarray(target_db, dtype=float)
    measured = np.asarray(measured_db, dtype=float)
    gain = np.asarray(gain_db, dtype=float)
    predicted = measured + gain
    return target - predicted


def _rms(x):
    arr = np.asarray(x, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(arr * arr)))


def _as_arr(values):
    if values is None:
        return None
    try:
        arr = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if arr.size == 0:
        return None
    return arr


def _align_to_freq_axis(values, freq_axis):
    arr = _as_arr(values)
    freq = _as_arr(freq_axis)
    if arr is None:
        return None
    if freq is None or freq.size == 0 or arr.size == freq.size:
        return arr
    src = np.linspace(0.0, 1.0, num=arr.size, dtype=float)
    dst = np.linspace(0.0, 1.0, num=freq.size, dtype=float)
    return np.interp(dst, src, arr)


def _stats_curve_from_kind(stats: dict | None, kind: str):
    st = dict(stats or {})
    freq = _as_arr(st.get("freq_axis"))
    if freq is None or freq.size < 8:
        return None, None
    measured = _align_to_freq_axis(st.get("measured_mags"), freq)
    if measured is None or measured.size != freq.size:
        return None, None
    if kind == "raw":
        return freq, measured
    if kind == "corrected":
        gain = _align_to_freq_axis(
            st.get("predicted_filter_mags", st.get("filter_mags")),
            freq,
        )
        if gain is None or gain.size != freq.size:
            return None, None
        return freq, measured + gain
    if kind == "target":
        target = _align_to_freq_axis(st.get("target_mags"), freq)
        if target is None or target.size != freq.size:
            return None, None
        return freq, target
    if kind == "confidence":
        conf = _align_to_freq_axis(st.get("confidence_mask"), freq)
        if conf is None or conf.size != freq.size:
            return None, None
        return freq, conf
    return None, None


def _interp_curve(freq_src, values_src, freq_dst):
    fs = _as_arr(freq_src)
    vs = _as_arr(values_src)
    fd = _as_arr(freq_dst)
    if fs is None or vs is None or fd is None or fs.size < 2 or vs.size != fs.size:
        return None
    valid = np.isfinite(fs) & np.isfinite(vs)
    if int(np.count_nonzero(valid)) < 2:
        return None
    fs = np.asarray(fs[valid], dtype=float)
    vs = np.asarray(vs[valid], dtype=float)
    order = np.argsort(fs)
    fs = fs[order]
    vs = vs[order]
    fd_clip = np.clip(fd, float(fs[0]), float(fs[-1]))
    return np.interp(fd_clip, fs, vs)


def band_target_error_rms_from_stats(
    stats: dict | None,
    *,
    lo_hz: float,
    hi_hz: float,
    use_confidence: bool = True,
) -> float:
    freq, corrected = _stats_curve_from_kind(stats, "corrected")
    _freq_t, target = _stats_curve_from_kind(stats, "target")
    if freq is None or corrected is None or target is None:
        return float("nan")
    mask = (
        np.isfinite(freq)
        & np.isfinite(corrected)
        & np.isfinite(target)
        & (freq >= float(lo_hz))
        & (freq <= float(hi_hz))
    )
    if int(np.count_nonzero(mask)) < 8:
        return float("nan")
    error = _mag_error_db(target[mask], corrected[mask], 0.0)
    if not bool(use_confidence):
        return _rms(error)
    _freq_c, conf = _stats_curve_from_kind(stats, "confidence")
    if conf is None or conf.size != freq.size:
        return _rms(error)
    weights = np.clip(np.asarray(conf[mask], dtype=float), 0.0, 1.0)
    weights = np.maximum(weights, 0.05)
    w_sum = float(np.sum(weights))
    if not np.isfinite(w_sum) or w_sum <= 1e-12:
        return _rms(error)
    return float(np.sqrt(np.sum(weights * error * error) / w_sum))


def band_filter_peak_boost_from_stats(
    stats: dict | None,
    *,
    lo_hz: float,
    hi_hz: float,
) -> float:
    st = dict(stats or {})
    freq = _as_arr(st.get("freq_axis"))
    if freq is None or freq.size < 8:
        return float("nan")
    gain = _align_to_freq_axis(
        st.get("predicted_filter_mags", st.get("filter_mags")),
        freq,
    )
    if gain is None or gain.size != freq.size:
        return float("nan")
    mask = (
        np.isfinite(freq)
        & np.isfinite(gain)
        & (freq >= float(lo_hz))
        & (freq <= float(hi_hz))
    )
    if int(np.count_nonzero(mask)) < 8:
        return float("nan")
    band_gain = np.asarray(gain[mask], dtype=float)
    return float(np.max(np.clip(band_gain, 0.0, None))) if band_gain.size else float("nan")


def band_lr_mismatch_rms_from_stats(
    left_stats: dict | None,
    right_stats: dict | None,
    *,
    lo_hz: float,
    hi_hz: float,
    corrected: bool = True,
) -> float:
    kind = "corrected" if bool(corrected) else "raw"
    f_l, y_l = _stats_curve_from_kind(left_stats, kind)
    f_r, y_r = _stats_curve_from_kind(right_stats, kind)
    if f_l is None or y_l is None or f_r is None or y_r is None:
        return float("nan")
    lo = max(float(lo_hz), float(np.min(f_l)), float(np.min(f_r)))
    hi = min(float(hi_hz), float(np.max(f_l)), float(np.max(f_r)))
    mask_l = np.isfinite(f_l) & np.isfinite(y_l) & (f_l >= lo) & (f_l <= hi)
    if int(np.count_nonzero(mask_l)) < 8:
        return float("nan")
    f_eval = np.asarray(f_l[mask_l], dtype=float)
    y_left = np.asarray(y_l[mask_l], dtype=float)
    y_right = _interp_curve(f_r, y_r, f_eval)
    if y_right is None or y_right.size != f_eval.size:
        return float("nan")
    diff = y_left - y_right
    return _rms(diff)


def band_lr_mismatch_change_from_stats(
    left_stats: dict | None,
    right_stats: dict | None,
    *,
    lo_hz: float,
    hi_hz: float,
) -> float:
    corrected = band_lr_mismatch_rms_from_stats(
        left_stats,
        right_stats,
        lo_hz=float(lo_hz),
        hi_hz=float(hi_hz),
        corrected=True,
    )
    raw = band_lr_mismatch_rms_from_stats(
        left_stats,
        right_stats,
        lo_hz=float(lo_hz),
        hi_hz=float(hi_hz),
        corrected=False,
    )
    if not np.isfinite(corrected) or not np.isfinite(raw):
        return float("nan")
    return float(corrected - raw)


def normalized_policy_divergence_score(
    resolved_policies,
    *,
    max_confidence_pull_delta: float = 0.20,
    max_tdc_strength_delta: float = 20.0,
    max_tdc_max_reduction_delta_db: float = 2.0,
    max_bass_first_mode_max_hz_delta: float = 30.0,
    max_low_bass_cut_strength_delta: float = 0.35,
    max_excess_phase_strength_delta: float = 0.15,
) -> float:
    if resolved_policies is None:
        return 0.0
    shared = getattr(resolved_policies, "shared", None)
    left = getattr(resolved_policies, "left", None)
    right = getattr(resolved_policies, "right", None)
    if left is None or right is None:
        return 0.0
    ratios = _normalized_policy_divergence_ratios(
        left=left,
        right=right,
        shared=shared,
        thresholds=(
            ("conf_pull_floor", max_confidence_pull_delta),
            ("tdc_strength", max_tdc_strength_delta),
            ("tdc_max_reduction_db", max_tdc_max_reduction_delta_db),
            ("bass_first_mode_max_hz", max_bass_first_mode_max_hz_delta),
            ("low_bass_cut_strength", max_low_bass_cut_strength_delta),
            ("excess_phase_strength", max_excess_phase_strength_delta),
        ),
    )
    if not ratios:
        return 0.0
    return float(np.mean(np.asarray(ratios, dtype=float)))


def _policy_effective_value(policy, shared, key: str):
    try:
        return policy.effective_value(key, shared)
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
        value = getattr(policy, key, None)
        if value is None and shared is not None:
            value = getattr(shared, key, None)
        return value


def _normalized_policy_divergence_ratios(
    *,
    left,
    right,
    shared,
    thresholds: tuple[tuple[str, float], ...],
) -> list[float]:
    ratios: list[float] = []
    for key, limit in thresholds:
        limit_f = float(limit)
        if not np.isfinite(limit_f) or limit_f <= 0.0:
            continue
        lv = _policy_effective_value(left, shared, key)
        rv = _policy_effective_value(right, shared, key)
        if lv is None or rv is None:
            continue
        try:
            ratio = abs(float(lv) - float(rv)) / limit_f
        except (TypeError, ValueError):
            continue
        if np.isfinite(ratio):
            ratios.append(float(max(0.0, ratio)))
    return ratios


def band_lr_spectral_xcorr_from_stats(
    left_stats: dict | None,
    right_stats: dict | None,
    *,
    lo_hz: float,
    hi_hz: float,
    corrected: bool = True,
    n_lag_bins: int = 16,
) -> tuple[float, float]:
    """Cross-correlation of level-normalised L/R spectral shapes in a band.

    Returns (peak_corr, lag_octaves):
    - peak_corr: Pearson correlation at best log-frequency lag, -1..1
    - lag_octaves: log-frequency shift at best correlation (>0 means L leads R)

    Detects whether channels differ in spectral shape (room-mode pattern) vs.
    just overall level — useful for diagnosing asymmetric room placement.
    NaN is returned when data are insufficient.
    """
    kind = "corrected" if bool(corrected) else "raw"
    f_l, y_l = _stats_curve_from_kind(left_stats, kind)
    f_r, y_r = _stats_curve_from_kind(right_stats, kind)
    if f_l is None or y_l is None or f_r is None or y_r is None:
        return float("nan"), float("nan")

    lo = max(float(lo_hz), float(np.min(f_l)), float(np.min(f_r)))
    hi = min(float(hi_hz), float(np.max(f_l)), float(np.max(f_r)))
    if hi <= lo * 1.1:
        return float("nan"), float("nan")

    n_pts = max(32, int(round(np.log2(hi / lo) * 20)))
    f_grid = np.logspace(np.log10(lo), np.log10(hi), n_pts)

    yl_i = _interp_curve(f_l, y_l, f_grid)
    yr_i = _interp_curve(f_r, y_r, f_grid)
    if yl_i is None or yr_i is None:
        return float("nan"), float("nan")

    # Remove overall level offset so we compare spectral shape only.
    yl_n = yl_i - float(np.nanmean(yl_i))
    yr_n = yr_i - float(np.nanmean(yr_i))

    n_lag = min(int(n_lag_bins), n_pts // 4)
    xcorr = np.correlate(yl_n, yr_n, mode="full")
    mid = len(xcorr) // 2
    norm = float(np.sqrt(np.sum(yl_n ** 2) * np.sum(yr_n ** 2))) + 1e-12
    xcorr_n = xcorr / norm

    window = xcorr_n[mid - n_lag: mid + n_lag + 1]
    best_idx = int(np.argmax(window))
    best_lag_bins = best_idx - n_lag

    oct_per_bin = float(np.log2(hi / lo)) / float(max(n_pts - 1, 1))
    lag_oct = float(best_lag_bins) * oct_per_bin

    peak_corr = float(np.clip(window[best_idx], -1.0, 1.0))
    return peak_corr, lag_oct


def worst_channel_relief_db(
    shared_left_stats: dict | None,
    shared_right_stats: dict | None,
    candidate_left_stats: dict | None,
    candidate_right_stats: dict | None,
    *,
    lo_hz: float,
    hi_hz: float,
) -> float:
    shared_left = band_filter_peak_boost_from_stats(shared_left_stats, lo_hz=lo_hz, hi_hz=hi_hz)
    shared_right = band_filter_peak_boost_from_stats(shared_right_stats, lo_hz=lo_hz, hi_hz=hi_hz)
    cand_left = band_filter_peak_boost_from_stats(candidate_left_stats, lo_hz=lo_hz, hi_hz=hi_hz)
    cand_right = band_filter_peak_boost_from_stats(candidate_right_stats, lo_hz=lo_hz, hi_hz=hi_hz)
    shared_vals = [float(v) for v in (shared_left, shared_right) if np.isfinite(v)]
    cand_vals = [float(v) for v in (cand_left, cand_right) if np.isfinite(v)]
    if not shared_vals or not cand_vals:
        return float("nan")
    return float(max(shared_vals) - max(cand_vals))
