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

from .metrics_common import _auto_stats_pick_arr

def _auto_focus_ripple_from_stats(
    st: dict | None,
    *,
    focus_lo_hz: float,
    focus_hi_hz: float,
) -> float | None:
    st = dict(st or {})
    lo = shared._auto_safe_float(focus_lo_hz, float("nan"))
    hi = shared._auto_safe_float(focus_hi_hz, float("nan"))
    if not (np.isfinite(lo) and np.isfinite(hi)) or float(hi) <= float(lo):
        return None

    mode = str(st.get("analysis_mode", "native") or "native").strip().lower()

    def _pick_arr(base_key: str, *fallback_keys: str) -> np.ndarray:
        keys: list[str] = []
        if mode == "comparison":
            keys.append(f"cmp_{str(base_key)}")
            keys.extend([f"cmp_{str(k)}" for k in fallback_keys])
        keys.append(str(base_key))
        keys.extend([str(k) for k in fallback_keys])
        for key in keys:
            try:
                arr = np.asarray(st.get(key, []), dtype=float).reshape(-1)
            except Exception:
                arr = np.asarray([], dtype=float)
            if arr.size:
                return np.asarray(arr, dtype=float)
        return np.asarray([], dtype=float)

    # Primary metric: local corrected-response RMS vs target inside the detected focus band.
    # This keeps auto-mode decisions tied to residual acoustic error instead of IR realization fidelity.
    f = _pick_arr("freq_axis")
    m_meas = _pick_arr("measured_mags")
    t_tgt = _pick_arr("target_mags")
    g_real = _pick_arr("realized_filter_mags", "filter_mags")
    c_mask = _pick_arr("confidence_mask")
    n = int(min(f.size, m_meas.size, t_tgt.size))
    if n >= 8:
        f = np.asarray(f[:n], dtype=float)
        pred = np.asarray(m_meas[:n], dtype=float)
        if g_real.size >= n:
            pred = pred + np.asarray(g_real[:n], dtype=float)
        err = pred - np.asarray(t_tgt[:n], dtype=float)
        mask = np.isfinite(f) & np.isfinite(err) & (f >= float(lo)) & (f <= float(hi))
        if int(np.count_nonzero(mask)) >= 8:
            err_use = np.asarray(err[mask], dtype=float)
            if c_mask.size >= n:
                w = np.clip(np.asarray(c_mask[:n], dtype=float)[mask], 0.0, 1.0)
                w = np.maximum(w, 0.05)
                w_sum = float(np.sum(w))
                if np.isfinite(w_sum) and w_sum > 1e-12:
                    return float(np.sqrt(np.sum(w * err_use * err_use) / w_sum))
            return float(np.sqrt(np.mean(err_use * err_use)))

    # Fallback: if corrected-response data is incomplete, fall back to filter-realization delta.
    g_pred = _pick_arr("predicted_filter_mags")
    g_real = _pick_arr("realized_filter_mags", "filter_mags")
    n = int(min(f.size, g_pred.size, g_real.size))
    if n < 8:
        return None
    f = np.asarray(f[:n], dtype=float)
    d = np.asarray(g_real[:n], dtype=float) - np.asarray(g_pred[:n], dtype=float)
    m = np.isfinite(f) & np.isfinite(d) & (f >= float(lo)) & (f <= float(hi))
    if int(np.count_nonzero(m)) < 8:
        return None
    dv = np.asarray(d[m], dtype=float)
    off = float(np.median(dv))
    d_shape = np.asarray(dv, dtype=float) - float(off)
    return float(np.sqrt(np.mean(d_shape * d_shape)))


def _auto_target_tracking_metrics_from_stats(st: dict | None) -> dict:
    st = dict(st or {})
    out = {
        "target_tracking_rms_20_200_db": float("nan"),
        "target_tracking_max_20_200_db": float("nan"),
        "target_tracking_rms_100_500_db": float("nan"),
        "target_tracking_max_100_500_db": float("nan"),
    }
    mode = str(st.get("analysis_mode", "native") or "native").strip().lower()

    def _pick_arr(base_key: str, *fallback_keys: str) -> np.ndarray:
        keys: list[str] = []
        if mode == "comparison":
            keys.append(f"cmp_{str(base_key)}")
            keys.extend([f"cmp_{str(k)}" for k in fallback_keys])
        keys.append(str(base_key))
        keys.extend([str(k) for k in fallback_keys])
        for key in keys:
            try:
                arr = np.asarray(st.get(key, []), dtype=float).reshape(-1)
            except Exception:
                arr = np.asarray([], dtype=float)
            if arr.size:
                return np.asarray(arr, dtype=float)
        return np.asarray([], dtype=float)

    f = _pick_arr("freq_axis")
    measured = _pick_arr("measured_mags")
    target = _pick_arr("target_mags")
    filt = _pick_arr("filter_mags", "realized_filter_mags")
    conf = _pick_arr("confidence_mask")
    n = int(min(f.size, measured.size, target.size, filt.size))
    if n < 8:
        return dict(out)

    f = np.asarray(f[:n], dtype=float)
    err = (
        np.asarray(measured[:n], dtype=float)
        + np.asarray(filt[:n], dtype=float)
        - np.asarray(target[:n], dtype=float)
    )
    conf_use = None
    if conf.size >= n:
        conf_use = np.asarray(conf[:n], dtype=float)
        finite_conf = conf_use[np.isfinite(conf_use)]
        if finite_conf.size and float(np.nanmax(finite_conf)) > 1.5:
            conf_use = conf_use / 100.0
        conf_use = np.clip(conf_use, 0.0, 1.0)

    for lo, hi, suffix in (
        (20.0, 200.0, "20_200"),
        (100.0, 500.0, "100_500"),
    ):
        mask = np.isfinite(f) & np.isfinite(err) & (f >= float(lo)) & (f <= float(hi))
        if int(np.count_nonzero(mask)) < 4:
            continue
        err_band = np.asarray(err[mask], dtype=float)
        if conf_use is not None:
            w = np.maximum(np.asarray(conf_use[mask], dtype=float), 0.05)
            w_sum = float(np.sum(w))
            if np.isfinite(w_sum) and w_sum > 1e-12:
                rms = float(np.sqrt(np.sum(w * err_band * err_band) / w_sum))
            else:
                rms = float(np.sqrt(np.mean(err_band * err_band)))
        else:
            rms = float(np.sqrt(np.mean(err_band * err_band)))
        max_err = float(np.max(np.abs(err_band)))
        out[f"target_tracking_rms_{suffix}_db"] = float(rms)
        out[f"target_tracking_max_{suffix}_db"] = float(max_err)
    return dict(out)


def _auto_merge_target_tracking_metrics(l_metrics: dict | None, r_metrics: dict | None) -> dict:
    out: dict[str, float] = {}
    for key in (
        "target_tracking_rms_20_200_db",
        "target_tracking_max_20_200_db",
        "target_tracking_rms_100_500_db",
        "target_tracking_max_100_500_db",
    ):
        vals = []
        for metrics in (l_metrics, r_metrics):
            v = shared._auto_safe_float((metrics or {}).get(key, float("nan")), float("nan"))
            if np.isfinite(v):
                vals.append(float(v))
        out[key] = float(max(vals)) if vals else float("nan")
    return out


def _auto_target_tracking_penalty(metrics: dict | None) -> float:
    m = dict(metrics or {})
    rms_20_200 = shared._auto_safe_float(m.get("target_tracking_rms_20_200_db", float("nan")), float("nan"))
    max_20_200 = shared._auto_safe_float(m.get("target_tracking_max_20_200_db", float("nan")), float("nan"))
    rms_100_500 = shared._auto_safe_float(m.get("target_tracking_rms_100_500_db", float("nan")), float("nan"))
    max_100_500 = shared._auto_safe_float(m.get("target_tracking_max_100_500_db", float("nan")), float("nan"))

    penalty = 0.0
    if np.isfinite(rms_20_200):
        penalty += 3.00 * max(0.0, float(rms_20_200) - 1.20)
    if np.isfinite(max_20_200):
        penalty += 1.00 * max(0.0, float(max_20_200) - 3.50)
    if np.isfinite(rms_100_500):
        penalty += 3.00 * max(0.0, float(rms_100_500) - 1.00)
    if np.isfinite(max_100_500):
        penalty += 0.90 * max(0.0, float(max_100_500) - 3.50)
    return float(np.clip(float(penalty), 0.0, 12.0))


def _auto_bass_under_target_metrics_from_stats(st: dict | None) -> dict:
    st = dict(st or {})
    out = {
        "bass_under_target_rms_20_200_db": float("nan"),
        "bass_under_target_max_20_200_db": float("nan"),
        "bass_under_target_penalty": 0.0,
    }
    mode = str(st.get("analysis_mode", "native") or "native").strip().lower()

    def _pick_arr(base_key: str, *fallback_keys: str) -> np.ndarray:
        keys: list[str] = []
        if mode == "comparison":
            keys.append(f"cmp_{str(base_key)}")
            keys.extend([f"cmp_{str(k)}" for k in fallback_keys])
        keys.append(str(base_key))
        keys.extend([str(k) for k in fallback_keys])
        for key in keys:
            try:
                arr = np.asarray(st.get(key, []), dtype=float).reshape(-1)
            except Exception:
                arr = np.asarray([], dtype=float)
            if arr.size:
                return np.asarray(arr, dtype=float)
        return np.asarray([], dtype=float)

    f = _pick_arr("freq_axis")
    measured = _pick_arr("measured_mags")
    target = _pick_arr("target_mags")
    filt = _pick_arr("realized_filter_mags", "filter_mags")
    conf = _pick_arr("confidence_mask")
    n = int(min(f.size, measured.size, target.size, filt.size))
    if n < 8:
        return dict(out)

    f = np.asarray(f[:n], dtype=float)
    err = (
        np.asarray(measured[:n], dtype=float)
        + np.asarray(filt[:n], dtype=float)
        - np.asarray(target[:n], dtype=float)
    )
    mask = np.isfinite(f) & np.isfinite(err) & (f >= 20.0) & (f <= 200.0)
    if int(np.count_nonzero(mask)) < 4:
        return dict(out)

    under = np.maximum(0.0, -np.asarray(err[mask], dtype=float) - 0.20)
    if conf.size >= n:
        conf_use = np.asarray(conf[:n], dtype=float)
        finite_conf = conf_use[np.isfinite(conf_use)]
        if finite_conf.size and float(np.nanmax(finite_conf)) > 1.5:
            conf_use = conf_use / 100.0
        w = np.maximum(np.clip(np.asarray(conf_use[mask], dtype=float), 0.0, 1.0), 0.35)
        w_sum = float(np.sum(w))
        rms = float(np.sqrt(np.sum(w * under * under) / w_sum)) if w_sum > 1e-12 else float(np.sqrt(np.mean(under * under)))
    else:
        rms = float(np.sqrt(np.mean(under * under)))
    max_under = float(np.max(under)) if under.size else 0.0
    penalty = float((2.2 * max(0.0, rms - 0.10)) + (0.65 * max(0.0, max_under - 0.45)))
    out["bass_under_target_rms_20_200_db"] = float(rms)
    out["bass_under_target_max_20_200_db"] = float(max_under)
    out["bass_under_target_penalty"] = float(np.clip(penalty, 0.0, 5.0))
    return dict(out)


def _auto_merge_bass_under_target_metrics(l_metrics: dict | None, r_metrics: dict | None) -> dict:
    out: dict[str, float] = {}
    for key in (
        "bass_under_target_rms_20_200_db",
        "bass_under_target_max_20_200_db",
        "bass_under_target_penalty",
    ):
        vals = []
        for metrics in (l_metrics, r_metrics):
            v = shared._auto_safe_float((metrics or {}).get(key, float("nan")), float("nan"))
            if np.isfinite(v):
                vals.append(float(v))
        out[key] = float(max(vals)) if vals else float("nan")
    return out


def _auto_bass_boost_metrics_from_stats(st: dict | None) -> dict:
    st = dict(st or {})
    mode = str(st.get("analysis_mode", "native") or "native").strip().lower()

    def _pick_arr(base_key: str, *fallback_keys: str) -> np.ndarray:
        keys: list[str] = []
        if mode == "comparison":
            keys.append(f"cmp_{str(base_key)}")
            keys.extend([f"cmp_{str(k)}" for k in fallback_keys])
        keys.append(str(base_key))
        keys.extend([str(k) for k in fallback_keys])
        for key in keys:
            try:
                arr = np.asarray(st.get(key, []), dtype=float).reshape(-1)
            except Exception:
                arr = np.asarray([], dtype=float)
            if arr.size:
                return np.asarray(arr, dtype=float)
        return np.asarray([], dtype=float)

    f = _pick_arr("freq_axis")
    filt = _pick_arr("filter_mags", "realized_filter_mags")
    n = int(min(f.size, filt.size))
    out = {
        "bass_boost_20_200_db": 0.0,
        "bass_boost_peak_20_200_db": 0.0,
    }
    if n < 4:
        return dict(out)

    f = np.asarray(f[:n], dtype=float)
    filt = np.asarray(filt[:n], dtype=float)
    guard_hi = 20.0
    low_cut = shared._auto_safe_float(st.get("low_bass_cut_hz", float("nan")), float("nan"))
    if np.isfinite(low_cut) and float(low_cut) > 0.0:
        guard_hi = max(float(guard_hi), float(low_cut))
    exc_freq = shared._auto_safe_float(st.get("exc_freq", float("nan")), float("nan"))
    if bool(st.get("exc_prot", False)) and np.isfinite(exc_freq) and float(exc_freq) > 0.0:
        guard_hi = max(float(guard_hi), float(exc_freq) * 1.41)

    mask = np.isfinite(f) & np.isfinite(filt) & (f >= float(guard_hi)) & (f <= 200.0)
    if int(np.count_nonzero(mask)) < 4:
        return dict(out)

    positive = np.maximum(np.asarray(filt[mask], dtype=float), 0.0)
    finite = positive[np.isfinite(positive)]
    if finite.size == 0:
        return dict(out)
    boosted = finite[finite > 0.05]
    if boosted.size == 0:
        return dict(out)
    out["bass_boost_20_200_db"] = float(np.percentile(boosted, 70.0))
    out["bass_boost_peak_20_200_db"] = float(np.max(boosted))
    return dict(out)


def _auto_merge_bass_boost_metrics(l_metrics: dict | None, r_metrics: dict | None) -> dict:
    vals = []
    peaks = []
    for metrics in (l_metrics, r_metrics):
        v = shared._auto_safe_float((metrics or {}).get("bass_boost_20_200_db", float("nan")), float("nan"))
        p = shared._auto_safe_float((metrics or {}).get("bass_boost_peak_20_200_db", float("nan")), float("nan"))
        if np.isfinite(v):
            vals.append(float(max(0.0, v)))
        if np.isfinite(p):
            peaks.append(float(max(0.0, p)))
    shared_bass = float(min(vals)) if len(vals) >= 2 else (float(vals[0]) if vals else 0.0)
    return {
        "bass_boost_20_200_db": float(shared_bass),
        "bass_boost_20_200_mean_db": float(np.mean(vals)) if vals else 0.0,
        "bass_boost_20_200_max_db": float(max(vals)) if vals else 0.0,
        "bass_boost_peak_20_200_db": float(max(peaks)) if peaks else 0.0,
    }


def _auto_bass_preference_bonus(
    *,
    bass_boost_db: float,
    target_tracking_penalty: float,
    exc_penalty: float,
    bass_integration_penalty: float,
    bass_feasibility_penalty: float,
) -> float:
    bass = shared._auto_safe_float(bass_boost_db, 0.0)
    if not np.isfinite(bass) or float(bass) <= 0.5:
        return 0.0
    norm = float(np.clip((float(bass) - 0.5) / 4.5, 0.0, 1.0))
    raw = 3.0 * float(norm)
    safety_scale = 1.0 / (
        1.0
        + 0.10 * max(0.0, shared._auto_safe_float(exc_penalty, 0.0))
        + 0.15 * max(0.0, shared._auto_safe_float(bass_integration_penalty, 0.0))
        + 0.25 * max(0.0, shared._auto_safe_float(bass_feasibility_penalty, 0.0))
        + 0.03 * max(0.0, shared._auto_safe_float(target_tracking_penalty, 0.0))
    )
    return float(np.clip(float(raw) * float(safety_scale), 0.0, 3.0))



__all__ = ["_auto_focus_ripple_from_stats", "_auto_target_tracking_metrics_from_stats", "_auto_merge_target_tracking_metrics", "_auto_target_tracking_penalty", "_auto_bass_under_target_metrics_from_stats", "_auto_merge_bass_under_target_metrics", "_auto_bass_boost_metrics_from_stats", "_auto_merge_bass_boost_metrics", "_auto_bass_preference_bonus"]
