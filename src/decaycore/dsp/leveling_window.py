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

from typing import Callable

import numpy as np


def _prepare_level_window_search(
    freq_axis: np.ndarray,
    magnitudes: np.ndarray,
    target_mags: np.ndarray | None,
    *,
    f_min: float,
    f_max: float,
    hpf_freq: float,
):
    try:
        freq_arr = np.asarray(freq_axis, dtype=float).reshape(-1)
        mag_arr = np.asarray(magnitudes, dtype=float).reshape(-1)
        if freq_arr.size != mag_arr.size or freq_arr.size == 0:
            return None

        safe_f_min = max(float(f_min), float(hpf_freq) * 1.5)
        if safe_f_min >= float(f_max) * 0.8:
            safe_f_min = float(f_min)

        mask = (
            np.isfinite(freq_arr)
            & np.isfinite(mag_arr)
            & (freq_arr >= float(safe_f_min))
            & (freq_arr <= float(f_max))
        )

        target_search = None
        if target_mags is not None:
            try:
                target_arr = np.asarray(target_mags, dtype=float).reshape(-1)
                if target_arr.size == freq_arr.size:
                    mask &= np.isfinite(target_arr)
                    target_search = target_arr
            except (TypeError, ValueError):
                target_search = None

        if int(np.count_nonzero(mask)) < 50:
            return None

        return {
            "freq": freq_arr[mask],
            "magnitudes": mag_arr[mask],
            "safe_f_min": float(safe_f_min),
            "f_max": float(f_max),
            "target": None if target_search is None else np.asarray(target_search[mask], dtype=float),
        }
    except (TypeError, ValueError, FloatingPointError, IndexError):
        return None


def _evaluate_level_window_candidate(
    freq_axis: np.ndarray,
    magnitudes: np.ndarray,
    target_mags: np.ndarray | None,
    *,
    tilt_comp: bool,
    tilt_max_db_per_oct: float,
    perceptual_weighting: bool,
    perceptual_strength: float,
    perceptual_min_hz: float,
    perceptual_max_hz: float,
    perceptual_tie_only: bool,
    resample_log_axis_fn: Callable[..., tuple[np.ndarray, tuple[np.ndarray, ...]]],
    lower_tail_robust_std_db_fn: Callable[..., float],
    window_offset_consistency_score_fn: Callable[..., tuple[float, float, float]],
    perceptual_shape_score_fn: Callable[..., float],
):
    try:
        f_w = np.asarray(freq_axis, dtype=float).reshape(-1)
        m_w = np.asarray(magnitudes, dtype=float).reshape(-1)
        if f_w.size < 20 or m_w.size != f_w.size:
            return None

        if target_mags is not None:
            t_w = np.asarray(target_mags, dtype=float).reshape(-1)
            if t_w.size != f_w.size:
                return None
            f_eval, (y, t_eval) = resample_log_axis_fn(f_w, m_w, t_w)
        else:
            f_eval, (y,) = resample_log_axis_fn(f_w, m_w)
            t_eval = None

        if f_eval.size < 20 or y.size < 20:
            return None

        channel_offset = float(np.median(y - t_eval)) if (t_eval is not None and t_eval.size == y.size) else float(np.median(y))

        x = np.log2(np.clip(f_eval, 1e-9, None))
        x0 = float(np.median(x))
        xc = x - x0
        y_med = float(np.median(y))
        denom = float(np.dot(xc, xc))
        residual = y - (float(np.dot(xc, (y - y_med)) / denom) * xc) if denom > 1e-12 else y

        std = lower_tail_robust_std_db_fn(residual, clip_below_db=6.0)
        f_center = float(np.sqrt(max(float(f_eval[0]), 1e-9) * max(float(f_eval[-1]), 1e-9)))
        weight = 1.0 + 0.05 * abs(np.log10(max(f_center, 1.0) / 1000.0))
        score = float(std * weight)

        target_rms = float("inf")
        offset_spread = float("inf")
        tilt_abs = float("inf")
        perceptual_rms = float("inf")
        try:
            if t_eval is not None and t_eval.size == y.size:
                offset_spread, target_rms, tilt_abs = window_offset_consistency_score_fn(
                    f_eval,
                    y,
                    t_eval,
                    tilt_comp=bool(tilt_comp),
                    tilt_max_db_per_oct=float(tilt_max_db_per_oct),
                )
                if perceptual_weighting:
                    perceptual_rms = perceptual_shape_score_fn(
                        f_eval,
                        y,
                        t_eval,
                        tilt_comp=bool(tilt_comp),
                        tilt_max_db_per_oct=float(tilt_max_db_per_oct),
                        min_hz=float(perceptual_min_hz),
                        max_hz=float(perceptual_max_hz),
                    )
            else:
                offset_spread, _shape_rms, tilt_abs = window_offset_consistency_score_fn(
                    f_eval,
                    y,
                    None,
                    tilt_comp=bool(tilt_comp),
                    tilt_max_db_per_oct=float(tilt_max_db_per_oct),
                )
        except (TypeError, ValueError, FloatingPointError, IndexError):
            pass

        if np.isfinite(offset_spread):
            score += 0.85 * float(offset_spread)
        if np.isfinite(target_rms):
            score += 0.20 * float(target_rms)
        if np.isfinite(tilt_abs):
            score += 0.08 * float(tilt_abs)
        if (not perceptual_tie_only) and np.isfinite(perceptual_rms):
            score += float(perceptual_strength) * float(perceptual_rms)

        return {
            "score": float(score),
            "target_rms": float(target_rms),
            "offset_spread": float(offset_spread),
            "tilt_abs": float(tilt_abs),
            "perceptual_rms": float(perceptual_rms),
            "channel_offset": float(channel_offset),
        }
    except (TypeError, ValueError, FloatingPointError, IndexError):
        return None


def _level_window_ranges(freq_axis: np.ndarray, safe_f_min: float, f_max: float, window_size_octaves: float):
    try:
        f = np.asarray(freq_axis, dtype=float).reshape(-1)
        if f.size == 0:
            return []
        safe = float(safe_f_min)
        hi = float(f_max)
        width = float(window_size_octaves)
        if safe <= 0.0 or hi <= safe or width <= 0.0:
            return []
        step = 2.0 ** (1.0 / 12.0)
        ratio = 2.0 ** width
        max_start = hi / ratio
        if max_start < safe:
            return []
        n_steps = int(np.floor(np.log(max_start / safe) / np.log(step))) + 1
        if n_steps <= 0:
            return []
        starts = safe * np.power(step, np.arange(n_steps, dtype=float))
        ends = starts * ratio
        monotonic = bool(f.size < 2 or np.all(np.diff(f) >= 0.0))
        if monotonic:
            lo_idx = np.searchsorted(f, starts, side="left")
            hi_idx = np.searchsorted(f, ends, side="right")
            return [
                (float(starts[i]), float(ends[i]), int(lo_idx[i]), int(hi_idx[i]), None)
                for i in range(int(starts.size))
            ]
        return [
            (float(w_start), float(w_end), 0, 0, (f >= float(w_start)) & (f <= float(w_end)))
            for w_start, w_end in zip(starts, ends, strict=False)
        ]
    except (TypeError, ValueError, FloatingPointError, OverflowError):
        return []


def _level_window_ranges_with_counts(
    freq_axis: np.ndarray,
    safe_f_min: float,
    f_max: float,
    window_size_octaves: float,
):
    ranges = _level_window_ranges(freq_axis, safe_f_min, f_max, window_size_octaves)
    if not ranges:
        return []
    f = np.asarray(freq_axis, dtype=float).reshape(-1)
    if f.size >= 2 and np.all(np.diff(f) >= 0.0):
        counts = np.asarray(
            [max(0, int(hi_idx) - int(lo_idx)) for _s, _e, lo_idx, hi_idx, _m in ranges],
            dtype=int,
        )
    else:
        counts = np.asarray(
            [
                int(np.count_nonzero(mask))
                if mask is not None
                else max(0, int(hi_idx) - int(lo_idx))
                for _s, _e, lo_idx, hi_idx, mask in ranges
            ],
            dtype=int,
        )
    return [(*item, int(counts[idx])) for idx, item in enumerate(ranges)]


def _slice_level_window(arr: np.ndarray | None, lo_idx: int, hi_idx: int, mask) -> np.ndarray | None:
    if arr is None:
        return None
    a = np.asarray(arr, dtype=float)
    if mask is None:
        return a[int(lo_idx):int(hi_idx)]
    return np.asarray(a[mask], dtype=float)


def find_stable_level_window_impl(
    freq_axis: np.ndarray,
    magnitudes: np.ndarray,
    target_mags: np.ndarray,
    f_min: float,
    f_max: float,
    *,
    window_size_octaves: float,
    hpf_freq: float,
    tilt_comp: bool,
    tilt_max_db_per_oct: float,
    perceptual_weighting: bool,
    perceptual_strength: float,
    perceptual_min_hz: float,
    perceptual_max_hz: float,
    perceptual_tie_only: bool,
    to_float_fn: Callable[[object, float], float],
    to_bool_fn: Callable[[object, bool], bool],
    resample_log_axis_fn: Callable[..., tuple[np.ndarray, tuple[np.ndarray, ...]]],
    lower_tail_robust_std_db_fn: Callable[..., float],
    window_offset_consistency_score_fn: Callable[..., tuple[float, float, float]],
    perceptual_shape_score_fn: Callable[..., float],
):
    try:
        f_min = to_float_fn(f_min, 0.0)
        f_max = to_float_fn(f_max, 0.0)
        hpf_freq = to_float_fn(hpf_freq, 0.0)
        window_size_octaves = to_float_fn(window_size_octaves, 1.0)
        perceptual_weighting = to_bool_fn(perceptual_weighting, False)
        perceptual_strength = max(0.0, to_float_fn(perceptual_strength, 0.12))
        perceptual_min_hz = to_float_fn(perceptual_min_hz, 250.0)
        perceptual_max_hz = to_float_fn(perceptual_max_hz, 4000.0)
        perceptual_tie_only = to_bool_fn(perceptual_tie_only, True)
        if f_min <= 0 or f_max <= 0 or f_min >= f_max:
            return float(f_min), float(f_max)

        prepared = _prepare_level_window_search(
            freq_axis,
            magnitudes,
            target_mags,
            f_min=float(f_min),
            f_max=float(f_max),
            hpf_freq=float(hpf_freq),
        )
        if prepared is None:
            return float(f_min), float(f_max)

        f_search = np.asarray(prepared["freq"], dtype=float)
        m_search = np.asarray(prepared["magnitudes"], dtype=float)
        t_search = prepared.get("target", None)
        safe_f_min = float(prepared["safe_f_min"])
        best_score = float("inf")
        best_target_rms = float("inf")
        best_offset_spread = float("inf")
        best_tilt_abs = float("inf")
        best_perceptual_rms = float("inf")
        res_min, res_max = float(safe_f_min), float(f_max)
        tie_eps_rel = 0.05
        for w_start, w_end, lo_idx, hi_idx, w_mask, count in _level_window_ranges_with_counts(
            f_search,
            safe_f_min,
            f_max,
            window_size_octaves,
        ):
            if count >= 20:
                metrics = _evaluate_level_window_candidate(
                    _slice_level_window(f_search, lo_idx, hi_idx, w_mask),
                    _slice_level_window(m_search, lo_idx, hi_idx, w_mask),
                    _slice_level_window(t_search, lo_idx, hi_idx, w_mask),
                    tilt_comp=bool(tilt_comp),
                    tilt_max_db_per_oct=float(tilt_max_db_per_oct),
                    perceptual_weighting=bool(perceptual_weighting),
                    perceptual_strength=float(perceptual_strength),
                    perceptual_min_hz=float(perceptual_min_hz),
                    perceptual_max_hz=float(perceptual_max_hz),
                    perceptual_tie_only=bool(perceptual_tie_only),
                    resample_log_axis_fn=resample_log_axis_fn,
                    lower_tail_robust_std_db_fn=lower_tail_robust_std_db_fn,
                    window_offset_consistency_score_fn=window_offset_consistency_score_fn,
                    perceptual_shape_score_fn=perceptual_shape_score_fn,
                )
                if metrics is not None:
                    score = float(metrics["score"])
                    target_rms = float(metrics["target_rms"])
                    offset_spread = float(metrics["offset_spread"])
                    tilt_abs = float(metrics["tilt_abs"])
                    perceptual_rms = float(metrics["perceptual_rms"])
                    better_stability = score < (best_score * (1.0 - tie_eps_rel))
                    near_tie = score <= (best_score * (1.0 + tie_eps_rel))
                    better_tie_break = near_tie and (
                        (offset_spread < best_offset_spread)
                        or (
                            offset_spread <= (best_offset_spread + 1e-6)
                            and (
                                (target_rms < best_target_rms)
                                or (
                                    target_rms <= (best_target_rms + 1e-6)
                                    and (
                                        (tilt_abs < best_tilt_abs)
                                        or (
                                            tilt_abs <= (best_tilt_abs + 1e-6)
                                            and perceptual_weighting
                                            and np.isfinite(perceptual_rms)
                                            and (perceptual_rms < best_perceptual_rms)
                                        )
                                    )
                                )
                            )
                        )
                    )
                    if better_stability or better_tie_break:
                        best_score = score
                        best_target_rms = target_rms
                        best_offset_spread = offset_spread
                        best_tilt_abs = tilt_abs
                        best_perceptual_rms = perceptual_rms
                        res_min, res_max = float(w_start), float(w_end)

        return (float(f_min), float(f_max)) if not np.isfinite(best_score) else (float(res_min), float(res_max))
    except (TypeError, ValueError, FloatingPointError, IndexError):
        return float(f_min), float(f_max)


def find_shared_stereo_level_window_impl(
    freq_axis_l: np.ndarray,
    magnitudes_l: np.ndarray,
    target_mags_l: np.ndarray,
    freq_axis_r: np.ndarray,
    magnitudes_r: np.ndarray,
    target_mags_r: np.ndarray,
    f_min: float,
    f_max: float,
    *,
    window_size_octaves: float,
    hpf_freq: float,
    tilt_comp: bool,
    tilt_max_db_per_oct: float,
    perceptual_weighting: bool,
    perceptual_strength: float,
    perceptual_min_hz: float,
    perceptual_max_hz: float,
    perceptual_tie_only: bool,
    to_float_fn: Callable[[object, float], float],
    to_bool_fn: Callable[[object, bool], bool],
    resample_log_axis_fn: Callable[..., tuple[np.ndarray, tuple[np.ndarray, ...]]],
    lower_tail_robust_std_db_fn: Callable[..., float],
    window_offset_consistency_score_fn: Callable[..., tuple[float, float, float]],
    perceptual_shape_score_fn: Callable[..., float],
):
    try:
        f_min = to_float_fn(f_min, 0.0)
        f_max = to_float_fn(f_max, 0.0)
        hpf_freq = to_float_fn(hpf_freq, 0.0)
        window_size_octaves = to_float_fn(window_size_octaves, 1.0)
        perceptual_weighting = to_bool_fn(perceptual_weighting, False)
        perceptual_strength = max(0.0, to_float_fn(perceptual_strength, 0.12))
        perceptual_min_hz = to_float_fn(perceptual_min_hz, 250.0)
        perceptual_max_hz = to_float_fn(perceptual_max_hz, 4000.0)
        perceptual_tie_only = to_bool_fn(perceptual_tie_only, True)
        if f_min <= 0 or f_max <= 0 or f_min >= f_max:
            return float(f_min), float(f_max)

        prep_l = _prepare_level_window_search(freq_axis_l, magnitudes_l, target_mags_l, f_min=float(f_min), f_max=float(f_max), hpf_freq=float(hpf_freq))
        prep_r = _prepare_level_window_search(freq_axis_r, magnitudes_r, target_mags_r, f_min=float(f_min), f_max=float(f_max), hpf_freq=float(hpf_freq))
        if prep_l is None or prep_r is None:
            return float(f_min), float(f_max)

        safe_f_min = max(float(prep_l["safe_f_min"]), float(prep_r["safe_f_min"]))
        if safe_f_min <= 0.0 or safe_f_min >= float(f_max):
            return float(f_min), float(f_max)

        f_l = np.asarray(prep_l["freq"], dtype=float)
        m_l = np.asarray(prep_l["magnitudes"], dtype=float)
        t_l = prep_l.get("target", None)
        f_r = np.asarray(prep_r["freq"], dtype=float)
        m_r = np.asarray(prep_r["magnitudes"], dtype=float)
        t_r = prep_r.get("target", None)

        best_primary = float("inf")
        best_secondary = float("inf")
        best_offset_spread = float("inf")
        best_target_rms = float("inf")
        best_tilt_abs = float("inf")
        best_perceptual_rms = float("inf")
        res_min, res_max = float(safe_f_min), float(f_max)
        tie_eps_rel = 0.05
        ranges_l = _level_window_ranges_with_counts(f_l, safe_f_min, f_max, window_size_octaves)
        ranges_r = _level_window_ranges_with_counts(f_r, safe_f_min, f_max, window_size_octaves)
        for left_range, right_range in zip(ranges_l, ranges_r, strict=False):
            w_start, w_end, lo_l, hi_l, mask_l, count_l = left_range
            _w_start_r, _w_end_r, lo_r, hi_r, mask_r, count_r = right_range
            if count_l >= 20 and count_r >= 20:
                metrics_l = _evaluate_level_window_candidate(
                    _slice_level_window(f_l, lo_l, hi_l, mask_l),
                    _slice_level_window(m_l, lo_l, hi_l, mask_l),
                    _slice_level_window(t_l, lo_l, hi_l, mask_l),
                    tilt_comp=bool(tilt_comp), tilt_max_db_per_oct=float(tilt_max_db_per_oct),
                    perceptual_weighting=bool(perceptual_weighting), perceptual_strength=float(perceptual_strength),
                    perceptual_min_hz=float(perceptual_min_hz), perceptual_max_hz=float(perceptual_max_hz),
                    perceptual_tie_only=bool(perceptual_tie_only), resample_log_axis_fn=resample_log_axis_fn,
                    lower_tail_robust_std_db_fn=lower_tail_robust_std_db_fn,
                    window_offset_consistency_score_fn=window_offset_consistency_score_fn,
                    perceptual_shape_score_fn=perceptual_shape_score_fn,
                )
                metrics_r = _evaluate_level_window_candidate(
                    _slice_level_window(f_r, lo_r, hi_r, mask_r),
                    _slice_level_window(m_r, lo_r, hi_r, mask_r),
                    _slice_level_window(t_r, lo_r, hi_r, mask_r),
                    tilt_comp=bool(tilt_comp), tilt_max_db_per_oct=float(tilt_max_db_per_oct),
                    perceptual_weighting=bool(perceptual_weighting), perceptual_strength=float(perceptual_strength),
                    perceptual_min_hz=float(perceptual_min_hz), perceptual_max_hz=float(perceptual_max_hz),
                    perceptual_tie_only=bool(perceptual_tie_only), resample_log_axis_fn=resample_log_axis_fn,
                    lower_tail_robust_std_db_fn=lower_tail_robust_std_db_fn,
                    window_offset_consistency_score_fn=window_offset_consistency_score_fn,
                    perceptual_shape_score_fn=perceptual_shape_score_fn,
                )
                if metrics_l is not None and metrics_r is not None:
                    score_l = float(metrics_l["score"])
                    score_r = float(metrics_r["score"])
                    offset_diff = abs(float(metrics_l["channel_offset"]) - float(metrics_r["channel_offset"]))
                    primary = max(score_l, score_r) + 0.25 * offset_diff
                    secondary = 0.5 * (score_l + score_r)
                    offset_spread = max(float(metrics_l["offset_spread"]), float(metrics_r["offset_spread"]))
                    target_rms = max(float(metrics_l["target_rms"]), float(metrics_r["target_rms"]))
                    tilt_abs = max(float(metrics_l["tilt_abs"]), float(metrics_r["tilt_abs"]))
                    perceptual_rms = max(float(metrics_l["perceptual_rms"]), float(metrics_r["perceptual_rms"]))
                    better_stability = primary < (best_primary * (1.0 - tie_eps_rel))
                    near_tie = primary <= (best_primary * (1.0 + tie_eps_rel))
                    better_tie_break = near_tie and (
                        (offset_spread < best_offset_spread)
                        or (
                            offset_spread <= (best_offset_spread + 1e-6)
                            and (
                                (secondary < best_secondary)
                                or (
                                    secondary <= (best_secondary + 1e-6)
                                    and (
                                        (target_rms < best_target_rms)
                                        or (
                                            target_rms <= (best_target_rms + 1e-6)
                                            and (
                                                (tilt_abs < best_tilt_abs)
                                                or (
                                                    tilt_abs <= (best_tilt_abs + 1e-6)
                                                    and perceptual_weighting
                                                    and np.isfinite(perceptual_rms)
                                                    and (perceptual_rms < best_perceptual_rms)
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                    if better_stability or better_tie_break:
                        best_primary = float(primary)
                        best_secondary = float(secondary)
                        best_offset_spread = float(offset_spread)
                        best_target_rms = float(target_rms)
                        best_tilt_abs = float(tilt_abs)
                        best_perceptual_rms = float(perceptual_rms)
                        res_min, res_max = float(w_start), float(w_end)

        return (float(f_min), float(f_max)) if not np.isfinite(best_primary) else (float(res_min), float(res_max))
    except (TypeError, ValueError, FloatingPointError, IndexError):
        return float(f_min), float(f_max)
