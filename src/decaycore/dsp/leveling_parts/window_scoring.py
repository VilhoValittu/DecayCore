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
import copy
from dataclasses import dataclass
import hashlib
import threading
from typing import Tuple
import numpy as np

from ..leveling_compute import compute_leveling_impl
from ..leveling_window import (
    find_shared_stereo_level_window_impl,
    find_stable_level_window_impl,
)

__all__ = [
    "StereoLinkContext",
    "find_stable_level_window",
    "find_shared_stereo_level_window",
    "compute_leveling",
]











_LEVELING_CACHE: dict[str, tuple[tuple[float, float, float, float, str, float, float], dict[str, object]]] = {}
_LEVELING_CACHE_LOCK = threading.Lock()
_LEVELING_CACHE_MAX = 128
_LEVEL_WINDOW_CACHE: dict[str, tuple[float, float]] = {}
_LEVEL_WINDOW_CACHE_LOCK = threading.Lock()
_LEVEL_WINDOW_CACHE_MAX = 256
_LEVELING_CACHE_ATTRS = (
    "_lvl_last_error",
    "_lvl_tilt_slope_db_per_oct",
    "_lvl_perceptual_enabled",
    "_lvl_perceptual_strength",
    "_lvl_perceptual_band_hz",
    "_lvl_perceptual_error_rms",
    "_lvl_window_debug",
)

def _hz_to_erb_number(freq_hz):
    try:
        f = np.asarray(freq_hz, dtype=float)
        f = np.clip(f, 0.0, None)
        erb = 21.4 * np.log10(1.0 + 4.37e-3 * f)
        if np.ndim(erb) == 0:
            return float(erb)
        return erb
    except (TypeError, ValueError, FloatingPointError, OverflowError):
        try:
            f = max(float(freq_hz), 0.0)
            return float(21.4 * np.log10(1.0 + 4.37e-3 * f))
        except (TypeError, ValueError, FloatingPointError, OverflowError):
            return 0.0

def _perceptual_importance_weights(
    freq_axis,
    *,
    min_hz=250.0,
    max_hz=4000.0,
    min_weight=0.60,
    max_weight=1.35,
    bass_floor_weight=0.85,
):
    try:
        f = np.asarray(freq_axis, dtype=float).reshape(-1)
        if f.size == 0:
            return np.asarray([], dtype=float)

        min_hz = _to_float(min_hz, 250.0)
        max_hz = _to_float(max_hz, 4000.0)
        min_weight = _to_float(min_weight, 0.60)
        max_weight = _to_float(max_weight, 1.35)
        bass_floor_weight = _to_float(bass_floor_weight, 0.85)

        if min_hz <= 0.0:
            min_hz = 250.0
        if max_hz <= min_hz:
            min_hz, max_hz = 250.0, 4000.0
        if max_weight < min_weight:
            max_weight = min_weight

        weights = np.full(f.shape, float(min_weight), dtype=float)
        valid = np.isfinite(f) & (f > 0.0)
        if not np.any(valid):
            return weights

        erb = np.asarray(_hz_to_erb_number(f[valid]), dtype=float)
        erb_lo = float(_hz_to_erb_number(min_hz))
        erb_hi = float(_hz_to_erb_number(max_hz))
        erb_span = max(float(erb_hi - erb_lo), 1e-6)
        erb_mid = 0.5 * (erb_lo + erb_hi)

        shape_sigma = max(0.34 * erb_span, 1e-6)
        gate_sigma = max(0.10 * erb_span, 1e-6)

        mid_shape = np.exp(-0.5 * ((erb - erb_mid) / shape_sigma) ** 2)
        lo_gate = 1.0 / (1.0 + np.exp(-(erb - erb_lo) / gate_sigma))
        hi_gate = 1.0 / (1.0 + np.exp((erb - erb_hi) / gate_sigma))
        band_gate = lo_gate * hi_gate

        w_valid = min_weight + (max_weight - min_weight) * mid_shape * band_gate

        low_mask = f[valid] < min_hz
        if np.any(low_mask):
            w_valid[low_mask] = np.maximum(w_valid[low_mask], float(bass_floor_weight))

        w_lo = min(min_weight, max_weight, bass_floor_weight)
        w_hi = max(min_weight, max_weight, bass_floor_weight)
        weights[valid] = np.clip(w_valid, w_lo, w_hi)
        return weights
    except (TypeError, ValueError, FloatingPointError, OverflowError):
        try:
            f = np.asarray(freq_axis, dtype=float).reshape(-1)
            return np.ones_like(f, dtype=float)
        except (TypeError, ValueError):
            return np.asarray([], dtype=float)

def _weighted_centered_rms(values, weights):
    try:
        y = np.asarray(values, dtype=float).reshape(-1)
        w = np.asarray(weights, dtype=float).reshape(-1)
        if y.size == 0 or w.size != y.size:
            return float("inf")

        valid = np.isfinite(y) & np.isfinite(w) & (w > 0.0)
        if not np.any(valid):
            return float("inf")

        y = y[valid]
        w = w[valid]
        w_sum = float(np.sum(w))
        if (not np.isfinite(w_sum)) or (w_sum <= 1e-12):
            return float("inf")

        y_mean = float(np.sum(w * y) / w_sum)
        resid = y - y_mean
        return float(np.sqrt(np.sum(w * resid * resid) / w_sum))
    except (TypeError, ValueError, FloatingPointError):
        return float("inf")

def _perceptual_shape_score(
    freq_axis,
    measured_db,
    target_db,
    *,
    tilt_comp=True,
    tilt_max_db_per_oct=2.0,
    min_hz=250.0,
    max_hz=4000.0,
):
    # This is a lightweight perceptual-weighting surrogate, not ANSI S3.4 loudness.
    try:
        f = np.asarray(freq_axis, dtype=float).reshape(-1)
        m = np.asarray(measured_db, dtype=float).reshape(-1)
        t = np.asarray(target_db, dtype=float).reshape(-1)
        if f.size < 12 or m.size != f.size or t.size != f.size:
            return float("inf")

        valid = np.isfinite(f) & np.isfinite(m) & np.isfinite(t) & (f > 0.0)
        if int(np.count_nonzero(valid)) < 12:
            return float("inf")

        f = f[valid]
        m = m[valid]
        t = t[valid]
        f, (m, t) = _resample_log_axis(f, m, t)
        if f.size < 12 or m.size != f.size or t.size != f.size:
            return float("inf")

        diff = np.asarray(m - t, dtype=float)
        if tilt_comp:
            off, slope = _tilt_fit_offset_and_slope_db_per_oct(
                f,
                diff,
                max_db_per_oct=float(tilt_max_db_per_oct),
            )
        else:
            off = _log_median(f, diff)
            slope = 0.0

        x = np.log2(np.clip(f, 1e-9, None))
        xc = x - float(np.median(x))
        resid = diff - float(off) - (float(slope) * xc)

        weights = _perceptual_importance_weights(
            f,
            min_hz=min_hz,
            max_hz=max_hz,
        )
        return _weighted_centered_rms(resid, weights)
    except (TypeError, ValueError, FloatingPointError, IndexError):
        return float("inf")

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

        out = {
            "freq": freq_arr[mask],
            "magnitudes": mag_arr[mask],
            "safe_f_min": float(safe_f_min),
            "f_max": float(f_max),
        }
        if target_search is not None:
            out["target"] = np.asarray(target_search[mask], dtype=float)
        else:
            out["target"] = None
        return out
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
):
    try:
        f_w = np.asarray(freq_axis, dtype=float).reshape(-1)
        m_w = np.asarray(magnitudes, dtype=float).reshape(-1)
        if f_w.size < 20 or m_w.size != f_w.size:
            return None

        eval_data = _resample_window_eval_inputs(f_w=f_w, m_w=m_w, target_mags=target_mags)
        if eval_data is None:
            return None
        f_eval, y, t_eval = eval_data

        if f_eval.size < 20 or y.size < 20:
            return None

        x = np.log2(np.clip(f_eval, 1e-9, None))
        x0 = float(np.median(x))
        xc = x - x0
        y_med = float(np.median(y))
        denom = float(np.dot(xc, xc))
        if denom > 1e-12:
            slope = float(np.dot(xc, (y - y_med)) / denom)
            residual = y - (slope * xc)
        else:
            residual = y

        std = _lower_tail_robust_std_db(residual, clip_below_db=6.0)
        weight = 1.0 + 0.05 * abs(np.log10(max(float(f_eval[0]), 1.0) / 1000.0))
        score = float(std * weight)

        target_rms, offset_spread, tilt_abs, perceptual_rms = _window_target_terms(
            f_eval=f_eval,
            y=y,
            t_eval=t_eval,
            tilt_comp=bool(tilt_comp),
            tilt_max_db_per_oct=float(tilt_max_db_per_oct),
            perceptual_weighting=bool(perceptual_weighting),
            perceptual_min_hz=float(perceptual_min_hz),
            perceptual_max_hz=float(perceptual_max_hz),
        )
        score = _window_apply_score_adjustments(
            score=float(score),
            target_rms=float(target_rms),
            offset_spread=float(offset_spread),
            tilt_abs=float(tilt_abs),
            perceptual_rms=float(perceptual_rms),
            perceptual_tie_only=bool(perceptual_tie_only),
            perceptual_strength=float(perceptual_strength),
        )

        return {
            "score": float(score),
            "target_rms": float(target_rms),
            "offset_spread": float(offset_spread),
            "tilt_abs": float(tilt_abs),
            "perceptual_rms": float(perceptual_rms),
        }
    except (TypeError, ValueError, FloatingPointError, IndexError):
        return None


def _resample_window_eval_inputs(
    *,
    f_w: np.ndarray,
    m_w: np.ndarray,
    target_mags: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
    if target_mags is not None:
        t_w = np.asarray(target_mags, dtype=float).reshape(-1)
        if t_w.size != f_w.size:
            return None
        f_eval, (y, t_eval) = _resample_log_axis(f_w, m_w, t_w)
        return f_eval, y, t_eval
    f_eval, (y,) = _resample_log_axis(f_w, m_w)
    return f_eval, y, None


def _window_target_terms(
    *,
    f_eval: np.ndarray,
    y: np.ndarray,
    t_eval: np.ndarray | None,
    tilt_comp: bool,
    tilt_max_db_per_oct: float,
    perceptual_weighting: bool,
    perceptual_min_hz: float,
    perceptual_max_hz: float,
) -> tuple[float, float, float, float]:
    target_rms = float("inf")
    offset_spread = float("inf")
    tilt_abs = float("inf")
    perceptual_rms = float("inf")
    try:
        if t_eval is not None and t_eval.size == y.size:
            offset_spread, target_rms, tilt_abs = _window_offset_consistency_score(
                f_eval,
                y,
                t_eval,
                tilt_comp=bool(tilt_comp),
                tilt_max_db_per_oct=float(tilt_max_db_per_oct),
            )
            if perceptual_weighting:
                perceptual_rms = _perceptual_shape_score(
                    f_eval,
                    y,
                    t_eval,
                    tilt_comp=bool(tilt_comp),
                    tilt_max_db_per_oct=float(tilt_max_db_per_oct),
                    min_hz=float(perceptual_min_hz),
                    max_hz=float(perceptual_max_hz),
                )
        else:
            offset_spread, _shape_rms, tilt_abs = _window_offset_consistency_score(
                f_eval,
                y,
                None,
                tilt_comp=bool(tilt_comp),
                tilt_max_db_per_oct=float(tilt_max_db_per_oct),
            )
    except (TypeError, ValueError, FloatingPointError, IndexError):
        pass
    return float(target_rms), float(offset_spread), float(tilt_abs), float(perceptual_rms)


def _window_apply_score_adjustments(
    *,
    score: float,
    target_rms: float,
    offset_spread: float,
    tilt_abs: float,
    perceptual_rms: float,
    perceptual_tie_only: bool,
    perceptual_strength: float,
) -> float:
    out = float(score)
    if np.isfinite(offset_spread):
        out += 0.85 * float(offset_spread)
    if np.isfinite(target_rms):
        out += 0.20 * float(target_rms)
    if np.isfinite(tilt_abs):
        out += 0.08 * float(tilt_abs)
    if (not perceptual_tie_only) and np.isfinite(perceptual_rms):
        out += float(perceptual_strength) * float(perceptual_rms)
    return float(out)


__all__ = ['_hz_to_erb_number', '_perceptual_importance_weights', '_weighted_centered_rms', '_perceptual_shape_score', '_prepare_level_window_search', '_evaluate_level_window_candidate']


def _link_sibling_exports() -> None:
    import importlib
    package = __package__
    for module_name in ['state_cache', 'tilt_helpers', 'window_scoring', 'api']:
        if module_name == __name__.rsplit('.', 1)[-1]:
            continue
        module = importlib.import_module(f"{package}.{module_name}")
        for symbol in getattr(module, "__all__", ()):
            globals().setdefault(symbol, getattr(module, symbol))


_link_sibling_exports()
