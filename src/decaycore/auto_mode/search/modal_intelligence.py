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

from .metrics_common import _auto_stats_pick_arr, _finite_json_float

MODAL_INTELLIGENCE_METRICS_VERSION = 1

def _room_mode_event_to_json(event: RoomModeEvent) -> dict:
    return {
        "freq_hz": float(_finite_json_float(event.freq_hz, 0.0)),
        "peak_db": float(_finite_json_float(event.peak_db, 0.0)),
        "width_hz": float(_finite_json_float(event.width_hz, 0.0)),
        "width_oct": float(_finite_json_float(event.width_oct, 0.0)),
        "q_estimate": float(_finite_json_float(event.q_estimate, 0.0)),
        "area_db_oct": float(_finite_json_float(event.area_db_oct, 0.0)),
        "gd_excess_ms": float(_finite_json_float(event.gd_excess_ms, 0.0)),
        "decay_severity": float(_finite_json_float(event.decay_severity, 0.0)),
        "confidence": float(_finite_json_float(event.confidence, 0.0)),
        "lr_consistency": float(_finite_json_float(event.lr_consistency, 0.0)),
        "repeatability": float(_finite_json_float(event.repeatability, 0.0)),
        "kind": str(event.kind or "unknown"),
        "severity": float(_finite_json_float(event.severity, 0.0)),
        "correction_priority": float(_finite_json_float(event.correction_priority, 0.0)),
        "cut_priority": float(_finite_json_float(event.cut_priority, event.correction_priority)),
        "safe_cut_db": float(_finite_json_float(event.safe_cut_db, 0.0)),
        "safe_width_oct": float(_finite_json_float(event.safe_width_oct, 0.0)),
        "voice_clarity_risk": float(_finite_json_float(event.voice_clarity_risk, 0.0)),
        "reasons": [str(reason) for reason in tuple(event.reasons or ())],
    }


def _modal_result_to_metrics(result: ModalAnalysisResult, *, top_n: int = 6) -> dict:
    events = tuple(result.events or ())
    return {
        "modal_analysis_version": int(result.analysis_version),
        "modal_metrics_version": int(MODAL_INTELLIGENCE_METRICS_VERSION),
        "modal_mode_count": int(max(0, result.mode_count)),
        "modal_worst_mode_hz": _finite_json_float(result.worst_mode_hz, None),
        "modal_worst_mode_severity": float(_finite_json_float(result.worst_mode_severity, 0.0)),
        "modal_area_db_oct": float(_finite_json_float(result.modal_area_db_oct, 0.0)),
        "modal_voice_band_risk": float(_finite_json_float(result.voice_band_modal_risk, 0.0)),
        "modal_events": [_room_mode_event_to_json(event) for event in events[: max(1, int(top_n))]],
    }


def compute_modal_intelligence_metrics(st: dict | None, *, lo_hz: float, hi_hz: float, top_n: int = 6) -> dict:
    st = dict(st or {})
    f = _auto_stats_pick_arr(st, "freq_axis")
    measured = _auto_stats_pick_arr(st, "measured_mags")
    target = _auto_stats_pick_arr(st, "target_mags")
    corrected = _auto_stats_pick_arr(st, "realized_filter_mags", "filter_mags", "predicted_filter_mags")
    conf = _auto_stats_pick_arr(st, "confidence_mask")
    gd = _auto_stats_pick_arr(st, "group_delay_ms", "gd_ms", "gd_curve_ms")
    left = _auto_stats_pick_arr(st, "left_mag_db", "l_measured_mags")
    right = _auto_stats_pick_arr(st, "right_mag_db", "r_measured_mags")
    result = detect_room_modes(
        f,
        measured,
        target_mag_db=target if target.size else None,
        corrected_mag_db=corrected if corrected.size else None,
        group_delay_ms=gd if gd.size else None,
        confidence_mask=conf if conf.size else None,
        left_mag_db=left if left.size else None,
        right_mag_db=right if right.size else None,
        lo_hz=float(lo_hz),
        hi_hz=float(hi_hz),
    )
    return _modal_result_to_metrics(result, top_n=top_n)


def _auto_merge_modal_intelligence_metrics(left_metrics, right_metrics) -> dict:
    events: list[dict] = []
    count = 0
    for channel, src in (("L", left_metrics), ("R", right_metrics)):
        m = dict(src or {}) if isinstance(src, dict) else {}
        count += int(max(0, shared._auto_safe_float(m.get("modal_mode_count", 0), 0.0)))
        for event in list(m.get("modal_events", []) or []):
            if not isinstance(event, dict):
                continue
            p = dict(event)
            p["channel"] = str(channel)
            events.append(p)

    events = sorted(
        events,
        key=lambda p: (
            -float(shared._auto_safe_float(p.get("severity"), 0.0)),
            -float(shared._auto_safe_float(p.get("peak_db"), 0.0)),
            float(shared._auto_safe_float(p.get("freq_hz"), 0.0)),
        ),
    )
    total_area = float(
        np.sum(
            np.asarray(
                [max(0.0, shared._auto_safe_float(p.get("area_db_oct", 0.0), 0.0)) for p in events],
                dtype=float,
            )
        )
    )
    voice_risk = float(
        np.clip(
            np.sum(
                np.asarray(
                    [
                        max(0.0, shared._auto_safe_float(p.get("severity", 0.0), 0.0))
                        * max(0.0, shared._auto_safe_float(p.get("confidence", 0.0), 0.0))
                        for p in events
                        if 80.0 <= shared._auto_safe_float(p.get("freq_hz", 0.0), 0.0) <= 160.0
                    ],
                    dtype=float,
                )
            ),
            0.0,
            1.0,
        )
    )
    worst = dict(events[0]) if events else {}
    return {
        "modal_analysis_version": int(worst.get("modal_analysis_version", 1) or 1),
        "modal_metrics_version": int(MODAL_INTELLIGENCE_METRICS_VERSION),
        "modal_mode_count": int(count),
        "modal_worst_mode_hz": _finite_json_float(worst.get("freq_hz"), None) if worst else None,
        "modal_worst_mode_severity": float(_finite_json_float(worst.get("severity"), 0.0)) if worst else 0.0,
        "modal_area_db_oct": float(total_area),
        "modal_voice_band_risk": float(voice_risk),
        "modal_events": [dict(p) for p in events[:6]],
    }


def _modal_support_for_band_from_metrics(
    modal_metrics: dict | None,
    *,
    f_min: float,
    f_max: float,
    channel: str | None = None,
) -> dict:
    try:
        lo = float(f_min)
        hi = float(f_max)
    except (TypeError, ValueError, OverflowError):
        lo = float("nan")
        hi = float("nan")
    empty = {
        "support": 0.0,
        "max_severity": 0.0,
        "dominant_freq_hz": None,
        "event_count": 0,
        "confidence": 0.0,
        "safe_cut_db": 0.0,
        "cut_priority": 0.0,
    }
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo <= 0.0 or hi <= lo:
        return dict(empty)
    events = list(dict(modal_metrics or {}).get("modal_events", []) or [])
    if not events:
        return dict(empty)
    center = float(np.sqrt(lo * hi))
    band_half_oct = max(0.02, 0.5 * float(np.log2(hi / lo)))
    matches: list[tuple[float, dict]] = []
    channel_eff = str(channel or "").strip()
    for raw in events:
        if not isinstance(raw, dict):
            continue
        event = dict(raw)
        if channel_eff and str(event.get("channel", "") or "").strip() not in ("", channel_eff):
            continue
        freq = shared._auto_safe_float(event.get("freq_hz"), float("nan"))
        if not np.isfinite(freq) or float(freq) <= 0.0:
            continue
        width_oct = max(
            0.0,
            shared._auto_safe_float(
                event.get("safe_width_oct", event.get("width_oct", 0.0)),
                0.0,
            ),
        )
        event_half_oct = max(0.03, 0.5 * float(width_oct))
        distance_oct = abs(float(np.log2(float(freq) / center)))
        overlap = max(0.0, 1.0 - distance_oct / max(1e-9, band_half_oct + event_half_oct))
        if overlap <= 0.0:
            continue
        severity = float(np.clip(shared._auto_safe_float(event.get("severity"), 0.0), 0.0, 1.0))
        confidence = float(np.clip(shared._auto_safe_float(event.get("confidence"), 0.0), 0.0, 1.0))
        priority = float(
            np.clip(
                max(
                    shared._auto_safe_float(event.get("cut_priority"), 0.0),
                    shared._auto_safe_float(event.get("correction_priority"), 0.0),
                    severity * confidence,
                ),
                0.0,
                1.0,
            )
        )
        matches.append((float(overlap * priority), event))
    if not matches:
        return dict(empty)
    matches = sorted(matches, key=lambda item: -float(item[0]))
    dominant = dict(matches[0][1])
    return {
        "support": float(np.clip(matches[0][0], 0.0, 1.0)),
        "max_severity": float(
            np.clip(
                max(shared._auto_safe_float(event.get("severity"), 0.0) for _score, event in matches),
                0.0,
                1.0,
            )
        ),
        "dominant_freq_hz": float(shared._auto_safe_float(dominant.get("freq_hz"), float("nan"))),
        "event_count": int(len(matches)),
        "confidence": float(np.clip(shared._auto_safe_float(dominant.get("confidence"), 0.0), 0.0, 1.0)),
        "safe_cut_db": float(max(0.0, shared._auto_safe_float(dominant.get("safe_cut_db"), 0.0))),
        "cut_priority": float(
            np.clip(
                max(
                    shared._auto_safe_float(dominant.get("cut_priority"), 0.0),
                    shared._auto_safe_float(dominant.get("correction_priority"), 0.0),
                ),
                0.0,
                1.0,
            )
        ),
    }


def _attach_modal_support_to_residual_metrics(residual_metrics: dict, modal_metrics: dict) -> dict:
    out = dict(residual_metrics or {})
    peaks: list[dict] = []
    strongest = {
        "support": 0.0,
        "max_severity": 0.0,
        "dominant_freq_hz": None,
        "event_count": 0,
        "confidence": 0.0,
        "safe_cut_db": 0.0,
        "cut_priority": 0.0,
        "modal_priority": 0.0,
    }
    for raw in list(out.get("residual_peak_candidates", []) or []):
        if not isinstance(raw, dict):
            continue
        peak = dict(raw)
        freq = shared._auto_safe_float(peak.get("freq_hz"), float("nan"))
        width_oct = shared._auto_safe_float(peak.get("width_oct"), float("nan"))
        if np.isfinite(freq) and float(freq) > 0.0:
            if not np.isfinite(width_oct) or float(width_oct) <= 0.0:
                width_oct = 1.0 / 8.0
            half = max(1.0 / 48.0, 0.5 * float(width_oct))
            f_min = float(freq) / float(2.0**half)
            f_max = float(freq) * float(2.0**half)
            support = _modal_support_for_band_from_metrics(
                modal_metrics,
                f_min=f_min,
                f_max=f_max,
                channel=str(peak.get("channel", "") or ""),
            )
        else:
            support = dict(strongest)
        priority = float(
            np.clip(
                float(support.get("support", 0.0) or 0.0)
                * max(float(support.get("max_severity", 0.0) or 0.0), float(support.get("cut_priority", 0.0) or 0.0))
                * max(float(support.get("confidence", 0.0) or 0.0), 0.35),
                0.0,
                1.0,
            )
        )
        peak["modal_support"] = float(support.get("support", 0.0) or 0.0)
        peak["modal_max_severity"] = float(support.get("max_severity", 0.0) or 0.0)
        peak["modal_confidence"] = float(support.get("confidence", 0.0) or 0.0)
        peak["modal_priority"] = float(priority)
        peak["modal_dominant_freq_hz"] = support.get("dominant_freq_hz")
        peak["modal_safe_cut_db"] = float(support.get("safe_cut_db", 0.0) or 0.0)
        peaks.append(peak)
        if priority > float(strongest.get("modal_priority", 0.0) or 0.0):
            strongest = dict(support)
            strongest["modal_priority"] = float(priority)
    out["residual_peak_candidates"] = peaks
    out["residual_peak_modal_support"] = float(strongest.get("support", 0.0) or 0.0)
    out["residual_peak_modal_max_severity"] = float(strongest.get("max_severity", 0.0) or 0.0)
    out["residual_peak_modal_confidence"] = float(strongest.get("confidence", 0.0) or 0.0)
    out["residual_peak_modal_priority"] = float(strongest.get("modal_priority", 0.0) or 0.0)
    out["residual_peak_modal_dominant_freq_hz"] = strongest.get("dominant_freq_hz")
    out["residual_peak_modal_event_count"] = int(strongest.get("event_count", 0) or 0)
    return out


def _modal_residual_fallback_metrics(
    residual_metrics: dict | None,
    modal_metrics: dict | None,
    *,
    lo_hz: float,
    hi_hz: float,
    threshold_db: float,
) -> dict:
    empty = {
        "modal_residual_fallback_used": False,
        "modal_residual_fallback_kind": "",
        "modal_residual_fallback_hz": float("nan"),
        "modal_residual_fallback_peak_db": float("nan"),
        "modal_residual_fallback_penalty": 0.0,
    }
    rm = dict(residual_metrics or {})
    residual_count = int(max(0, shared._auto_safe_float(rm.get("residual_peak_count", 0), 0.0)))
    if residual_count > 0:
        return dict(empty)

    lo = shared._auto_safe_float(lo_hz, float("nan"))
    hi = shared._auto_safe_float(hi_hz, float("nan"))
    threshold = max(0.0, shared._auto_safe_float(threshold_db, shared.AUTO_MODE_RESIDUAL_PEAK_THRESHOLD_DB))
    if not (np.isfinite(lo) and np.isfinite(hi)) or float(hi) <= float(lo):
        return dict(empty)

    candidates: list[tuple[float, dict]] = []
    for raw in list(dict(modal_metrics or {}).get("modal_events", []) or []):
        if not isinstance(raw, dict):
            continue
        event = dict(raw)
        kind = str(event.get("kind", "") or "").strip().lower()
        if kind not in {"room_mode", "broad_buildup"}:
            continue
        freq = shared._auto_safe_float(event.get("freq_hz", float("nan")), float("nan"))
        peak_db = shared._auto_safe_float(event.get("peak_db", float("nan")), float("nan"))
        confidence = float(np.clip(shared._auto_safe_float(event.get("confidence", 0.0), 0.0), 0.0, 1.0))
        priority = float(
            np.clip(
                max(
                    shared._auto_safe_float(event.get("correction_priority", 0.0), 0.0),
                    shared._auto_safe_float(event.get("cut_priority", 0.0), 0.0),
                ),
                0.0,
                1.0,
            )
        )
        severity = float(np.clip(shared._auto_safe_float(event.get("severity", 0.0), 0.0), 0.0, 1.0))
        if not (np.isfinite(freq) and np.isfinite(peak_db)) or float(freq) <= 0.0:
            continue
        if not (float(lo) <= float(freq) <= float(hi)):
            continue
        if float(peak_db) < float(threshold) or confidence < 0.45 or priority < 0.12:
            continue
        score = float(
            max(0.0, float(peak_db) - float(threshold))
            + 1.25 * float(priority)
            + 0.75 * float(severity)
        )
        candidates.append((score, event))

    if not candidates:
        return dict(empty)

    candidates = sorted(
        candidates,
        key=lambda item: (
            -float(item[0]),
            -float(shared._auto_safe_float(item[1].get("peak_db"), 0.0)),
            float(shared._auto_safe_float(item[1].get("freq_hz"), 0.0)),
        ),
    )
    best = dict(candidates[0][1])
    peak_db = shared._auto_safe_float(best.get("peak_db", float("nan")), float("nan"))
    priority = float(
        np.clip(
            max(
                shared._auto_safe_float(best.get("correction_priority", 0.0), 0.0),
                shared._auto_safe_float(best.get("cut_priority", 0.0), 0.0),
            ),
            0.0,
            1.0,
        )
    )
    severity = float(np.clip(shared._auto_safe_float(best.get("severity", 0.0), 0.0), 0.0, 1.0))
    penalty = float(
        np.clip(
            0.80 * max(0.0, float(peak_db) - float(threshold))
            + 1.25 * float(priority)
            + 0.35 * float(severity),
            0.0,
            3.0,
        )
    )
    return {
        "modal_residual_fallback_used": bool(penalty > 0.0),
        "modal_residual_fallback_kind": str(best.get("kind", "") or ""),
        "modal_residual_fallback_hz": float(shared._auto_safe_float(best.get("freq_hz"), float("nan"))),
        "modal_residual_fallback_peak_db": float(peak_db) if np.isfinite(peak_db) else float("nan"),
        "modal_residual_fallback_penalty": float(penalty),
    }


def _build_modal_tdc_debug(modal_metrics: dict | None, decay_dbg: dict | None) -> dict:
    modal = dict(modal_metrics or {})
    dbg = dict(decay_dbg or {})
    events = [dict(event) for event in list(modal.get("modal_events", []) or []) if isinstance(event, dict)]
    tdc_peak_db = float(max(0.0, shared._auto_safe_float(dbg.get("tdc_peak_reduction_db", 0.0), 0.0)))
    tdc_peak_hz = shared._auto_safe_float(dbg.get("tdc_peak_reduction_hz", float("nan")), float("nan"))
    band_lo = shared._auto_safe_float(dbg.get("tdc_reduction_band_low_hz", float("nan")), float("nan"))
    band_hi = shared._auto_safe_float(dbg.get("tdc_reduction_band_high_hz", float("nan")), float("nan"))
    reductions: list[dict] = []
    for event in events:
        freq = shared._auto_safe_float(event.get("freq_hz"), float("nan"))
        if not np.isfinite(freq) or float(freq) <= 0.0:
            continue
        severity = float(np.clip(shared._auto_safe_float(event.get("severity"), 0.0), 0.0, 1.0))
        confidence = float(np.clip(shared._auto_safe_float(event.get("confidence"), 0.0), 0.0, 1.0))
        if severity < 0.35 and confidence < 0.50:
            continue
        support = 0.0
        if np.isfinite(band_lo) and np.isfinite(band_hi) and float(band_hi) > float(band_lo) > 0.0:
            support = float(
                _modal_support_for_band_from_metrics(
                    modal,
                    f_min=float(band_lo),
                    f_max=float(band_hi),
                    channel=str(event.get("channel", "") or ""),
                ).get("support", 0.0)
                or 0.0
            )
        elif np.isfinite(tdc_peak_hz) and float(tdc_peak_hz) > 0.0:
            distance = abs(float(np.log2(float(freq) / float(tdc_peak_hz))))
            support = float(np.clip(1.0 - distance / 0.18, 0.0, 1.0))
        else:
            support = float(severity * confidence)
        if support <= 0.0:
            continue
        applied_reduction = float(tdc_peak_db * np.clip(support, 0.0, 1.0))
        reductions.append(
            {
                "freq_hz": float(freq),
                "reason": "modal_decay",
                "modal_severity": float(severity),
                "gd_excess_ms": float(shared._auto_safe_float(event.get("gd_excess_ms"), 0.0)),
                "peak_db": float(shared._auto_safe_float(event.get("peak_db"), 0.0)),
                "applied_reduction_db": float(applied_reduction),
                "confidence": float(confidence),
                "support": float(np.clip(support, 0.0, 1.0)),
                "channel": str(event.get("channel", "") or ""),
            }
        )
    reductions = sorted(
        reductions,
        key=lambda item: (
            -float(shared._auto_safe_float(item.get("applied_reduction_db"), 0.0)),
            -float(shared._auto_safe_float(item.get("modal_severity"), 0.0)),
            float(shared._auto_safe_float(item.get("freq_hz"), 0.0)),
        ),
    )[:5]
    return {
        "tdc_modal_event_count": int(len(reductions)),
        "tdc_modal_reductions": [dict(item) for item in reductions],
        "tdc_modal_debug_available": bool(reductions),
    }



__all__ = ["MODAL_INTELLIGENCE_METRICS_VERSION", "compute_modal_intelligence_metrics", "_auto_merge_modal_intelligence_metrics", "_modal_support_for_band_from_metrics", "_attach_modal_support_to_residual_metrics", "_modal_residual_fallback_metrics", "_build_modal_tdc_debug"]
