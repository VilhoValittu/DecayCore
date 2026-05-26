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

import bisect
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

_AI_SUMMARY_SCORING_RANGE_SUPPORT_CACHE: dict[tuple[int, str], bool] = {}

def _auto_stats_pick_arr(st: dict | None, base_key: str, *fallback_keys: str, _max_n: int | None = None) -> np.ndarray:
    st = dict(st or {})
    mode = str(st.get("analysis_mode", "native") or "native").strip().lower()
    keys: list[str] = []
    if mode == "comparison":
        keys.append(f"cmp_{str(base_key)}")
        keys.extend([f"cmp_{str(k)}" for k in fallback_keys])
    keys.append(str(base_key))
    keys.extend([str(k) for k in fallback_keys])
    for key in keys:
        try:
            raw = st.get(key, [])
            if _max_n is not None and raw is not None:
                raw = raw[:_max_n]
            arr = np.asarray(raw, dtype=float).reshape(-1)
        except Exception:
            arr = np.asarray([], dtype=float)
        if arr.size:
            return np.asarray(arr, dtype=float)
    return np.asarray([], dtype=float)


def _auto_stats_band_n(st: dict | None, hi_guard: float, key: str = "freq_axis") -> int | None:
    """Return the index limit for hi_guard Hz without materializing the full array.

    Uses bisect for Python lists (O(log n)) and searchsorted for numpy arrays (O(log n)).
    Returns None if the array is too small or the limit cannot be determined.
    """
    st = dict(st or {})
    mode = str(st.get("analysis_mode", "native") or "native").strip().lower()
    raw = st.get(f"cmp_{key}") if mode == "comparison" else None
    if raw is None:
        raw = st.get(key)
    if raw is None:
        return None
    try:
        if isinstance(raw, np.ndarray):
            n = int(np.searchsorted(raw, hi_guard, side="right"))
        else:
            n = bisect.bisect_right(raw, hi_guard)
        return n if n >= 8 else None
    except Exception:
        return None


def _finite_json_float(value, default: float | None = 0.0):
    try:
        v = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not np.isfinite(v):
        return default
    return float(v)


def _calc_ai_summary_from_stats_auto(stats: dict, scoring_range: tuple[float, float] | None) -> dict:
    func = getattr(sys.modules.get("decaycore.auto_mode.scoring_metrics"), "calc_ai_summary_from_stats", calc_ai_summary_from_stats)
    if scoring_range is None:
        return func(stats)

    key = (id(func), str(getattr(func, "__qualname__", "")))
    accepts_scoring_range = _AI_SUMMARY_SCORING_RANGE_SUPPORT_CACHE.get(key)
    if accepts_scoring_range is None:
        try:
            params = inspect.signature(func).parameters
            accepts_scoring_range = "scoring_range" in params or any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
            )
        except (TypeError, ValueError):
            accepts_scoring_range = True
        if len(_AI_SUMMARY_SCORING_RANGE_SUPPORT_CACHE) > 16:
            _AI_SUMMARY_SCORING_RANGE_SUPPORT_CACHE.clear()
        _AI_SUMMARY_SCORING_RANGE_SUPPORT_CACHE[key] = bool(accepts_scoring_range)

    if bool(accepts_scoring_range):
        return func(stats, scoring_range=scoring_range)
    return func(stats)


def _get_auto_scoring_range(
    st_l: dict,
    st_r: dict,
    base_data: dict | None,
) -> tuple[float, float] | None:
    if not shared.AUTO_MODE_CORRECTION_RANGE_SCORING:
        return None
    sources = [st_l, st_r, base_data or {}]

    def _pick(key: str, fallback: float) -> float:
        for s in sources:
            v = shared._auto_safe_float((s or {}).get(key, float("nan")), float("nan"))
            if np.isfinite(v) and v > 0.0:
                return float(v)
        return float(fallback)

    mag_c_min = _pick("mag_c_min", 20.0)
    mag_c_max = _pick("mag_c_max", 300.0)
    trans_w = _pick("trans_width", 100.0)
    lo = float(mag_c_min)
    hi = max(float(mag_c_max) + float(trans_w), 1500.0)
    return (lo, hi)


def _ai_score_with_fallback(st: dict, ai: dict, *, scoring_range) -> float:
    score = shared._auto_safe_float((ai or {}).get("score"), float("nan"))
    if np.isfinite(score):
        return float(score)
    try:
        conf = shared._auto_safe_float(
            st.get("cmp_avg_confidence", st.get("avg_confidence", 0.0)),
            0.0,
        )
        _rms_fb, match_fb = target_match_from_stats(
            st,
            include_filter=False,
            use_confidence=True,
            use_smart_scan_range=False,
            freq_range=scoring_range,
        )
        if match_fb is None:
            return 0.0
        rt60 = st.get("rt60_val", None)
        rt_rel = st.get("rt60_reliability", None)
        return shared._auto_safe_float(
            calc_acoustic_score(conf, float(match_fb), rt60_s=rt60, rt60_rel=rt_rel),
            0.0,
        )
    except Exception:
        return 0.0



__all__ = ["calc_ai_summary_from_stats", "_auto_stats_pick_arr", "_auto_stats_band_n", "_finite_json_float", "_calc_ai_summary_from_stats_auto", "_get_auto_scoring_range", "_ai_score_with_fallback"]
