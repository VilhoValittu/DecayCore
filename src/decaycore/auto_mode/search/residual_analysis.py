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

import inspect
import logging
import math
import sys

import numpy as np

_logger = logging.getLogger(__name__)

from ...common.acoustic_stats import calc_acoustic_score, calc_ai_summary_from_stats
from ...config.models import StereoAutoPolicyConfig, StereoResolvedAutoPolicies
from ...dsp.quality_metrics import (
    band_lr_mismatch_change_from_stats,
    band_lr_mismatch_rms_from_stats,
    normalized_policy_divergence_score,
    worst_channel_relief_db,
)
from ...dsp.modal_analysis import ModalAnalysisResult, RoomModeEvent, detect_room_modes
from ...dsp.smoothing import smooth_gain_fractional_octave
from ...dsp.target_match import target_match_from_stats
from .. import shared
from ..rank_score import (
    OFFICIAL_RANK_SCORE_CONTEXT,
    attach_official_rank_score,
    calibrated_auto_quality,
    compute_rank_score_components,
)
from ..runtime_context import (
    _auto_collect_reflections,
    _auto_event_penalty_weighted,
    _auto_event_severity,
    _auto_get_top_modes_hz,
    _auto_get_worst_mode_hz,
    _auto_mode_band,
    _auto_pick_metric,
)

from .metrics_common import _auto_stats_band_n, _auto_stats_pick_arr, _finite_json_float

BROAD_RESIDUAL_PEAK_SCORING_VERSION = 1
MODAL_INTELLIGENCE_METRICS_VERSION = 1

def _auto_min_broad_peak_width_oct(freq_hz: float) -> float:
    f = float(shared._auto_safe_float(freq_hz, float("nan")))
    if not np.isfinite(f):
        return 1.0 / 6.0
    if f < 80.0:
        return 1.0 / 6.0
    if f < 150.0:
        return 1.0 / 8.0
    return 1.0 / 5.0


def _auto_band_weight_for_correction_shape(freq_hz: np.ndarray) -> np.ndarray:
    f = np.asarray(freq_hz, dtype=float)
    return np.where(f < 80.0, 1.10, np.where(f < 150.0, 1.0, 0.85))


def _run_bounds_from_mask(run_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(run_mask, dtype=bool).reshape(-1)
    n = int(mask.size)
    left = np.arange(n, dtype=int)
    right = np.arange(n, dtype=int)
    if n == 0 or not bool(np.any(mask)):
        return left, right
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    lengths = (ends - starts + 1).astype(int, copy=False)
    idx = np.concatenate(
        [
            np.arange(int(start), int(end) + 1, dtype=int)
            for start, end in zip(starts, ends, strict=False)
        ]
    )
    left[idx] = np.repeat(starts, lengths)
    right[idx] = np.repeat(ends, lengths)
    return left, right


def _run_area_prefix(log_freq: np.ndarray, values: np.ndarray, threshold: float) -> np.ndarray:
    x = np.asarray(log_freq, dtype=float).reshape(-1)
    y = np.maximum(0.0, np.asarray(values, dtype=float).reshape(-1) - float(threshold))
    if x.size != y.size or x.size < 2:
        return np.zeros(x.size + 1, dtype=float)
    contrib = 0.5 * (y[:-1] + y[1:]) * np.maximum(0.0, x[1:] - x[:-1])
    return np.r_[0.0, np.cumsum(np.nan_to_num(contrib, nan=0.0, posinf=0.0, neginf=0.0))]


def _finite_window_means(
    sorted_axis: np.ndarray,
    values: np.ndarray,
    centers: np.ndarray,
    half_width: float,
    *,
    default: float,
) -> np.ndarray:
    axis = np.asarray(sorted_axis, dtype=float).reshape(-1)
    vals = np.asarray(values, dtype=float).reshape(-1)
    ctr = np.asarray(centers, dtype=float).reshape(-1)
    if axis.size == 0 or vals.size != axis.size or ctr.size == 0:
        return np.full(ctr.shape, float(default), dtype=float)
    finite = np.isfinite(vals)
    sums = np.r_[0.0, np.cumsum(np.where(finite, vals, 0.0))]
    counts = np.r_[0, np.cumsum(finite.astype(int))]
    left = np.searchsorted(axis, ctr - float(half_width), side="left")
    right = np.searchsorted(axis, ctr + float(half_width), side="right")
    count = counts[right] - counts[left]
    total = sums[right] - sums[left]
    out = np.full(ctr.shape, float(default), dtype=float)
    valid = count > 0
    out[valid] = total[valid] / count[valid]
    return out


def _min_broad_peak_width_oct_array(freq_hz: np.ndarray) -> np.ndarray:
    f = np.asarray(freq_hz, dtype=float)
    return np.where(
        np.isfinite(f),
        np.where(f < 80.0, 1.0 / 6.0, np.where(f < 150.0, 1.0 / 8.0, 1.0 / 5.0)),
        1.0 / 6.0,
    )


def _broad_residual_thresholds(
    *,
    threshold_db: float | None,
    hard_gate_db: float | None,
) -> tuple[float, float]:
    threshold_eff = float(
        max(
            0.0,
            shared._auto_safe_float(
                shared.AUTO_MODE_RESIDUAL_PEAK_THRESHOLD_DB if threshold_db is None else threshold_db,
                shared.AUTO_MODE_RESIDUAL_PEAK_THRESHOLD_DB,
            ),
        )
    )
    hard_gate_eff = float(
        max(
            0.0,
            shared._auto_safe_float(
                shared.AUTO_MODE_RESIDUAL_PEAK_HARD_GATE_DB if hard_gate_db is None else hard_gate_db,
                shared.AUTO_MODE_RESIDUAL_PEAK_HARD_GATE_DB,
            ),
        )
    )
    return float(threshold_eff), float(hard_gate_eff)


def _broad_residual_empty_result(*, threshold_eff: float, hard_gate_eff: float) -> dict:
    return {
        "worst_residual_peak_db": float("nan"),
        "worst_residual_peak_hz": float("nan"),
        "worst_residual_peak_raw_db": float("nan"),
        "worst_residual_peak_width_hz": float("nan"),
        "worst_residual_peak_width_oct": float("nan"),
        "residual_peak_area_db_oct": 0.0,
        "residual_peak_severity": float("nan"),
        "residual_peak_threshold_db": float(threshold_eff),
        "residual_peak_hard_gate_db": float(hard_gate_eff),
        "top3_residual_peak_mean_db": float("nan"),
        "residual_peak_count": 0,
        "residual_peak_candidates": [],
        "voice_band_worst_residual_peak_db": 0.0,
        "broad_residual_peak_scoring_version": int(BROAD_RESIDUAL_PEAK_SCORING_VERSION),
    }


def _broad_residual_zero_peak_result(out_empty: dict) -> dict:
    out = dict(out_empty)
    out["worst_residual_peak_db"] = 0.0
    out["worst_residual_peak_raw_db"] = 0.0
    out["residual_peak_severity"] = 0.0
    out["top3_residual_peak_mean_db"] = 0.0
    out["residual_peak_area_db_oct"] = 0.0
    return out


def _broad_residual_prepare_series(
    st: dict,
    *,
    lo: float,
    hi: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    _n_lim = _auto_stats_band_n(st, float(hi) * 2.5)
    f = _auto_stats_pick_arr(st, "freq_axis", _max_n=_n_lim)
    measured = _auto_stats_pick_arr(st, "measured_mags", _max_n=_n_lim)
    target = _auto_stats_pick_arr(st, "target_mags", _max_n=_n_lim)
    realized = _auto_stats_pick_arr(st, "realized_filter_mags", "filter_mags", "predicted_filter_mags", _max_n=_n_lim)
    conf = _auto_stats_pick_arr(st, "confidence_mask", _max_n=_n_lim)
    n = int(min(f.size, measured.size, target.size, realized.size))
    if n < 8:
        return None
    f = np.asarray(f[:n], dtype=float)
    measured = np.asarray(measured[:n], dtype=float)
    target = np.asarray(target[:n], dtype=float)
    realized = np.asarray(realized[:n], dtype=float)
    pred = measured + realized
    err = pred - target
    valid = (
        np.isfinite(f)
        & np.isfinite(err)
        & (f > 0.0)
        & (f >= float(lo))
        & (f <= float(hi))
    )
    if int(np.count_nonzero(valid)) < 8:
        return None
    f_use = np.asarray(f[valid], dtype=float)
    err_use = np.asarray(err[valid], dtype=float)
    order = np.argsort(f_use)
    f_use = np.asarray(f_use[order], dtype=float)
    err_use = np.asarray(err_use[order], dtype=float)
    if conf.size >= n:
        conf_use = np.asarray(conf[:n], dtype=float)[valid][order]
        finite_conf = conf_use[np.isfinite(conf_use)]
        if finite_conf.size and float(np.nanmax(finite_conf)) > 1.5:
            conf_use = conf_use / 100.0
        conf_use = np.clip(conf_use, 0.0, 1.0)
    else:
        conf_use = np.ones_like(f_use, dtype=float)
    return np.asarray(f_use, dtype=float), np.asarray(err_use, dtype=float), np.asarray(conf_use, dtype=float)


def _broad_residual_smooth_series(
    *,
    f_use: np.ndarray,
    err_use: np.ndarray,
    detect_smooth_oct: float,
    baseline_smooth_oct: float,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        detect = np.asarray(
            smooth_gain_fractional_octave(
                f_use,
                err_use,
                float(np.clip(shared._auto_safe_float(detect_smooth_oct, 6.0), 1.0, 8.0)),
            ),
            dtype=float,
        ).reshape(-1)
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
        detect = np.asarray(err_use, dtype=float)
    if detect.size != err_use.size:
        detect = np.asarray(err_use, dtype=float)
    try:
        baseline = np.asarray(
            smooth_gain_fractional_octave(
                f_use,
                err_use,
                float(shared._auto_safe_float(baseline_smooth_oct, 2.0)),
            ),
            dtype=float,
        ).reshape(-1)
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
        baseline = np.full_like(err_use, float(np.nanmedian(err_use)))
    if baseline.size != err_use.size:
        baseline = np.full_like(err_use, float(np.nanmedian(err_use)))
    return np.asarray(detect, dtype=float), np.asarray(baseline, dtype=float)


def _broad_residual_peak_indices(
    *,
    f_use: np.ndarray,
    detect: np.ndarray,
    peak_mask: np.ndarray,
    peak_excess: np.ndarray,
) -> np.ndarray:
    if f_use.size >= 3:
        local = np.zeros(f_use.size, dtype=bool)
        local[1:-1] = (detect[1:-1] >= detect[:-2]) & (detect[1:-1] >= detect[2:])
    else:
        local = np.ones(f_use.size, dtype=bool)
    idxs = np.flatnonzero(peak_mask & local)
    if idxs.size == 0:
        idxs = np.asarray([int(np.flatnonzero(peak_mask)[int(np.argmax(peak_excess[peak_mask]))])], dtype=int)
    return np.asarray(idxs, dtype=int)


def _broad_residual_build_raw_peaks(
    *,
    f_use: np.ndarray,
    detect: np.ndarray,
    peak_excess: np.ndarray,
    idxs: np.ndarray,
    conf_use: np.ndarray,
    threshold_eff: float,
    hard_gate_eff: float,
    lo: float,
    hi: float,
    conf_floor: float,
) -> list[dict]:
    run_mask = np.isfinite(detect) & (detect >= float(threshold_eff))
    run_left, run_right = _run_bounds_from_mask(run_mask)
    raw_peaks: list[dict] = []
    conf_floor_eff = float(np.clip(shared._auto_safe_float(conf_floor, 0.25), 0.0, 1.0))
    log_f_use = np.log2(np.maximum(f_use, 1e-9))
    idxs_i = np.asarray(idxs, dtype=int)
    peak_freqs = np.asarray(f_use[idxs_i], dtype=float)
    lefts = np.asarray(run_left[idxs_i], dtype=int)
    rights = np.asarray(run_right[idxs_i], dtype=int)
    widths_hz = np.maximum(0.0, f_use[rights] - f_use[lefts])
    widths_oct = np.where(
        (f_use[lefts] > 0.0) & (f_use[rights] > f_use[lefts]),
        np.log2(f_use[rights] / f_use[lefts]),
        0.0,
    )
    min_widths_oct = _min_broad_peak_width_oct_array(peak_freqs)
    conf_means = _finite_window_means(log_f_use, conf_use, log_f_use[idxs_i], 0.06, default=1.0)
    prom_vals = np.maximum(0.0, peak_excess[idxs_i])
    residual_vals = np.maximum(0.0, detect[idxs_i])
    height_vals = np.maximum(0.0, residual_vals - float(threshold_eff))
    area_prefix = _run_area_prefix(log_f_use, detect, float(threshold_eff))
    area_vals = np.maximum(0.0, area_prefix[rights] - area_prefix[lefts])
    area_vals = np.where(
        (area_vals <= 0.0) & (widths_oct > 0.0) & (height_vals > 0.0),
        height_vals * widths_oct,
        area_vals,
    )
    freq_weights = np.where(
        peak_freqs <= 120.0,
        1.0,
        np.clip(1.0 - 0.35 * ((peak_freqs - 120.0) / 130.0), 0.65, 1.0),
    )
    width_weights = np.clip(0.70 + np.minimum(widths_oct, 0.50) / 0.50, 0.70, 1.70)
    conf_weights = np.maximum(conf_floor_eff, np.clip(conf_means, 0.0, 1.0))
    inside_bands = (float(lo) <= peak_freqs) & (peak_freqs <= float(hi))
    band_weights = np.where(inside_bands, 1.0, 0.50)
    severity_vals = (
        (height_vals + 0.35 * prom_vals + 0.35 * area_vals)
        * width_weights
        * freq_weights
        * conf_weights
        * band_weights
    )
    keep_idxs = np.flatnonzero(
        np.isfinite(peak_freqs)
        & (peak_freqs > 0.0)
        & (widths_oct + 1e-9 >= min_widths_oct)
    )
    for pos in keep_idxs:
        freq = float(peak_freqs[int(pos)])
        if not np.isfinite(freq) or freq <= 0.0:
            continue
        raw_peaks.append(
            {
                "freq_hz": float(freq),
                "excess_db": float(height_vals[int(pos)]),
                "prominence_db": float(prom_vals[int(pos)]),
                "residual_db": float(residual_vals[int(pos)]),
                "confidence_mean": float(conf_means[int(pos)]),
                "width_hz": float(widths_hz[int(pos)]),
                "width_oct": float(widths_oct[int(pos)]),
                "min_width_oct": float(min_widths_oct[int(pos)]),
                "area_db_oct": float(max(0.0, area_vals[int(pos)])),
                "frequency_weight": float(freq_weights[int(pos)]),
                "inside_correction_band": bool(inside_bands[int(pos)]),
                "severity": float(severity_vals[int(pos)]),
                "weighted_peak_db": float(severity_vals[int(pos)]),
                "threshold_db": float(threshold_eff),
                "hard_gate_db": float(hard_gate_eff),
            }
        )
    return list(raw_peaks)


def _broad_residual_merge_peaks(raw_peaks: list[dict]) -> list[dict]:
    raw_peaks = sorted(
        raw_peaks,
        key=lambda p: (
            -float(shared._auto_safe_float(p.get("residual_db"), 0.0)),
            -float(shared._auto_safe_float(p.get("severity"), 0.0)),
            -float(shared._auto_safe_float(p.get("prominence_db"), 0.0)),
        ),
    )
    merged: list[dict] = []
    merge_oct = 0.12
    for peak in raw_peaks:
        freq = float(shared._auto_safe_float(peak.get("freq_hz"), float("nan")))
        if not np.isfinite(freq) or freq <= 0.0:
            continue
        too_close = False
        for kept in merged:
            kept_freq = float(shared._auto_safe_float(kept.get("freq_hz"), float("nan")))
            if np.isfinite(kept_freq) and kept_freq > 0.0:
                if abs(float(np.log2(freq / kept_freq))) < float(merge_oct):
                    too_close = True
                    break
        if not too_close:
            merged.append(dict(peak))
    return list(merged)


def _broad_residual_finalize(
    *,
    merged: list[dict],
    top_n: int,
    threshold_eff: float,
    hard_gate_eff: float,
) -> dict:
    top_n_eff = int(max(1, shared._auto_safe_float(top_n, 3)))
    worst = dict(merged[0])
    top_vals = [
        float(shared._auto_safe_float(p.get("severity", p.get("weighted_peak_db")), 0.0))
        for p in merged[:top_n_eff]
    ]
    total_area = float(
        np.sum(
            np.asarray(
                [max(0.0, shared._auto_safe_float(p.get("area_db_oct", 0.0), 0.0)) for p in merged],
                dtype=float,
            )
        )
    )
    voice_band_peaks = [
        p for p in merged
        if 60.0 <= float(shared._auto_safe_float(p.get("freq_hz"), float("nan"))) <= 160.0
    ]
    voice_band_worst = float(max(
        (shared._auto_safe_float(p.get("severity", p.get("weighted_peak_db")), 0.0) for p in voice_band_peaks),
        default=0.0,
    ))
    return {
        "worst_residual_peak_db": float(shared._auto_safe_float(worst.get("severity", worst.get("weighted_peak_db")), 0.0)),
        "worst_residual_peak_hz": float(shared._auto_safe_float(worst.get("freq_hz"), float("nan"))),
        "worst_residual_peak_raw_db": float(shared._auto_safe_float(worst.get("residual_db"), 0.0)),
        "worst_residual_peak_width_hz": float(shared._auto_safe_float(worst.get("width_hz"), float("nan"))),
        "worst_residual_peak_width_oct": float(shared._auto_safe_float(worst.get("width_oct"), float("nan"))),
        "residual_peak_area_db_oct": float(total_area),
        "residual_peak_severity": float(shared._auto_safe_float(worst.get("severity", worst.get("weighted_peak_db")), 0.0)),
        "residual_peak_threshold_db": float(threshold_eff),
        "residual_peak_hard_gate_db": float(hard_gate_eff),
        "top3_residual_peak_mean_db": float(np.mean(np.asarray(top_vals, dtype=float))) if top_vals else 0.0,
        "residual_peak_count": int(len(merged)),
        "residual_peak_candidates": [dict(p) for p in merged[: max(top_n_eff, 3)]],
        "voice_band_worst_residual_peak_db": float(voice_band_worst),
        "broad_residual_peak_scoring_version": int(BROAD_RESIDUAL_PEAK_SCORING_VERSION),
    }


def compute_broad_residual_peak_metrics(
    st: dict | None,
    *,
    lo_hz: float,
    hi_hz: float,
    baseline_smooth_oct: float = 2.0,
    detect_smooth_oct: float = 6.0,
    min_prom_db: float = 0.75,
    threshold_db: float | None = None,
    hard_gate_db: float | None = None,
    conf_floor: float = 0.25,
    top_n: int = 3,
) -> dict:
    st = dict(st or {})
    threshold_eff, hard_gate_eff = _broad_residual_thresholds(
        threshold_db=threshold_db,
        hard_gate_db=hard_gate_db,
    )
    out_empty = _broad_residual_empty_result(
        threshold_eff=float(threshold_eff),
        hard_gate_eff=float(hard_gate_eff),
    )
    lo = shared._auto_safe_float(lo_hz, float("nan"))
    hi = shared._auto_safe_float(hi_hz, float("nan"))
    if not (np.isfinite(lo) and np.isfinite(hi)) or float(hi) <= float(lo):
        return dict(out_empty)

    prepared = _broad_residual_prepare_series(
        st,
        lo=float(lo),
        hi=float(hi),
    )
    if prepared is None:
        return dict(out_empty)
    f_use, err_use, conf_use = prepared
    detect, baseline = _broad_residual_smooth_series(
        f_use=f_use,
        err_use=err_use,
        detect_smooth_oct=float(detect_smooth_oct),
        baseline_smooth_oct=float(baseline_smooth_oct),
    )

    peak_excess = np.asarray(detect - baseline, dtype=float)
    peak_mask = (
        np.isfinite(f_use)
        & np.isfinite(detect)
        & np.isfinite(peak_excess)
        & (peak_excess > float(max(0.0, min_prom_db)))
        & (detect >= float(threshold_eff))
    )
    if int(np.count_nonzero(peak_mask)) <= 0:
        return _broad_residual_zero_peak_result(out_empty)

    idxs = _broad_residual_peak_indices(
        f_use=f_use,
        detect=detect,
        peak_mask=peak_mask,
        peak_excess=peak_excess,
    )
    raw_peaks = _broad_residual_build_raw_peaks(
        f_use=f_use,
        detect=detect,
        peak_excess=peak_excess,
        idxs=idxs,
        conf_use=conf_use,
        threshold_eff=float(threshold_eff),
        hard_gate_eff=float(hard_gate_eff),
        lo=float(lo),
        hi=float(hi),
        conf_floor=float(conf_floor),
    )
    if not raw_peaks:
        return _broad_residual_zero_peak_result(out_empty)
    merged = _broad_residual_merge_peaks(raw_peaks)
    if not merged:
        return _broad_residual_zero_peak_result(out_empty)
    return _broad_residual_finalize(
        merged=merged,
        top_n=int(top_n),
        threshold_eff=float(threshold_eff),
        hard_gate_eff=float(hard_gate_eff),
    )



__all__ = ["BROAD_RESIDUAL_PEAK_SCORING_VERSION", "compute_broad_residual_peak_metrics"]
