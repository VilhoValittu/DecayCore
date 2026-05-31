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

def _auto_bass_integration_penalty(
    result,
    *,
    base_data: dict | None = None,
    net_boost_max_db: float = float("nan"),
) -> tuple[float, float, dict]:
    data = dict(base_data or {})
    if not bool(data.get("bass_integration_enable", False)):
        return 0.0, 0.0, {}

    result_metrics = dict(getattr(result, "metrics", {}) or {})
    cancellation_risk = shared._auto_safe_float(
        result_metrics.get("bass_cancellation_risk", float("nan")),
        float("nan"),
    )
    overlap_ripple_db = shared._auto_safe_float(
        result_metrics.get("bass_overlap_ripple", float("nan")),
        float("nan"),
    )
    sub_dominance_db = shared._auto_safe_float(
        result_metrics.get("bass_sub_dominance", float("nan")),
        float("nan"),
    )
    null_severity = shared._auto_safe_float(
        result_metrics.get("bass_null_severity", float("nan")),
        float("nan"),
    )
    gd_rms_mismatch_ms = shared._auto_safe_float(
        result_metrics.get(
            "bass_xo_gd_rms_mismatch_ms",
            result_metrics.get("bass_xo_gd_mismatch_ms", float("nan")),
        ),
        float("nan"),
    )
    overlap_extension_active = bool(result_metrics.get("bass_overlap_extension_active", False))
    overlap_extension_flatness_db = shared._auto_safe_float(
        result_metrics.get("bass_overlap_extension_flatness_db", float("nan")),
        float("nan"),
    )
    overlap_extension_cancellation_risk = shared._auto_safe_float(
        result_metrics.get("bass_overlap_extension_cancellation_risk", float("nan")),
        float("nan"),
    )
    overlap_extension_peak_excess_db = shared._auto_safe_float(
        result_metrics.get("bass_overlap_extension_peak_excess_db", float("nan")),
        float("nan"),
    )
    overlap_extension_sub_dominance_db = shared._auto_safe_float(
        result_metrics.get("bass_overlap_extension_sub_dominance_db", float("nan")),
        float("nan"),
    )
    overlap_ripple_delta_db = shared._auto_safe_float(
        result_metrics.get("bass_overlap_ripple_delta_db", float("nan")),
        float("nan"),
    )
    sub_dominance_delta_db = shared._auto_safe_float(
        result_metrics.get("bass_sub_dominance_delta_db", float("nan")),
        float("nan"),
    )
    gd_mismatch_delta_ms = shared._auto_safe_float(
        result_metrics.get("bass_xo_gd_mismatch_delta_ms", float("nan")),
        float("nan"),
    )
    feasibility_class = str(result_metrics.get("bass_feasibility_class", "") or "").strip().lower()
    feasibility_reason = str(result_metrics.get("bass_feasibility_reason", "") or "").strip()
    dominant_channel = str(result_metrics.get("bass_dominant_channel", "unknown") or "unknown").strip().lower()
    weights = shared._auto_bass_integration_profile_weights(
        data.get("bass_integration_profile", result_metrics.get("bass_integration_profile", "safe"))
    )

    l_st = dict(getattr(result, "l_st", {}) or {})
    r_st = dict(getattr(result, "r_st", {}) or {})
    lf_boost = max(
        shared._auto_safe_float(l_st.get("lf_boost_max_db", 0.0), 0.0),
        shared._auto_safe_float(r_st.get("lf_boost_max_db", 0.0), 0.0),
    )
    net_boost = shared._auto_safe_float(net_boost_max_db, float("nan"))
    if not np.isfinite(net_boost):
        net_boost = max(
            shared._auto_safe_float(l_st.get("net_boost_peak_db", 0.0), 0.0),
            shared._auto_safe_float(r_st.get("net_boost_peak_db", 0.0), 0.0),
        )

    pen_cancel = 0.0
    if np.isfinite(cancellation_risk):
        pen_cancel = float(max(0.0, cancellation_risk - 0.12) ** 1.5) * float(weights.get("cancellation", 0.0))

    pen_overlap = 0.0
    if np.isfinite(overlap_ripple_db):
        pen_overlap = float(max(0.0, overlap_ripple_db - 3.0)) * float(weights.get("overlap_ripple", 0.0))

    pen_anti_null = 0.0
    if np.isfinite(cancellation_risk) and np.isfinite(net_boost):
        pen_anti_null = float(max(0.0, cancellation_risk - 0.08) * max(0.0, net_boost - 3.5))
        pen_anti_null *= float(weights.get("anti_null_boost", 0.0))

    pen_sub = 0.0
    if np.isfinite(sub_dominance_db):
        pen_sub = float(max(0.0, sub_dominance_db - 4.0) / 6.0) * float(max(0.0, lf_boost - 1.5))
        pen_sub *= float(weights.get("sub_dominance", 0.0))

    pen_null = 0.0
    if np.isfinite(null_severity):
        pen_null = float(max(0.0, null_severity - 1.2) / 4.0) * float(weights.get("cancellation", 0.0))

    pen_gd = 0.0
    if np.isfinite(gd_rms_mismatch_ms):
        pen_gd = float(max(0.0, gd_rms_mismatch_ms - 0.8) / 2.5) * float(weights.get("xo_gd_continuity", 0.0))

    pen_ext_flatness = 0.0
    if overlap_extension_active and np.isfinite(overlap_extension_flatness_db):
        pen_ext_flatness = (
            float(max(0.0, overlap_extension_flatness_db - 6.0) / 6.0)
            * 0.20
            * float(weights.get("overlap_ripple", 0.0))
        )

    pen_ext_cancel = 0.0
    if overlap_extension_active and np.isfinite(overlap_extension_cancellation_risk):
        pen_ext_cancel = (
            float(max(0.0, overlap_extension_cancellation_risk - 0.15))
            * 0.20
            * float(weights.get("cancellation", 0.0))
        )

    pen_ext_peak = 0.0
    if overlap_extension_active and np.isfinite(overlap_extension_peak_excess_db):
        pen_ext_peak = (
            float(max(0.0, overlap_extension_peak_excess_db - 4.0) / 4.0)
            * 0.15
            * float(weights.get("overlap_ripple", 0.0))
        )

    pen_ext_sub = 0.0
    if overlap_extension_active and np.isfinite(overlap_extension_sub_dominance_db):
        pen_ext_sub = (
            float(max(0.0, overlap_extension_sub_dominance_db - 12.0) / 6.0)
            * 0.12
            * float(weights.get("sub_dominance", 0.0))
        )

    pen_overlap_delta = 0.0
    if np.isfinite(overlap_ripple_delta_db):
        pen_overlap_delta = (
            float(max(0.0, overlap_ripple_delta_db - 2.0) / 4.0)
            * 0.55
            * float(weights.get("overlap_ripple", 0.0))
        )

    pen_sub_delta = 0.0
    if np.isfinite(sub_dominance_delta_db):
        pen_sub_delta = (
            float(max(0.0, sub_dominance_delta_db - 2.0) / 4.0)
            * 0.75
            * float(weights.get("sub_dominance", 0.0))
        )

    pen_gd_delta = 0.0
    if np.isfinite(gd_mismatch_delta_ms):
        pen_gd_delta = (
            float(max(0.0, gd_mismatch_delta_ms - 4.0) / 10.0)
            * 0.75
            * float(weights.get("xo_gd_continuity", 0.0))
        )

    pen_feasibility_base = {
        "good": 0.0,
        "marginal": 0.35,
        "infeasible": 0.9,
    }.get(feasibility_class, 0.0)

    penalty = float(
        np.clip(
            pen_cancel
            + pen_overlap
            + pen_anti_null
            + pen_sub
            + pen_null
            + pen_gd
            + pen_ext_flatness
            + pen_ext_cancel
            + pen_ext_peak
            + pen_ext_sub
            + pen_overlap_delta
            + pen_sub_delta
            + pen_gd_delta
            + pen_feasibility_base,
            0.0,
            12.0,
        )
    )

    def _excess(value: float, threshold: float) -> float:
        if np.isfinite(value):
            return float(max(0.0, float(value) - float(threshold)))
        return 0.0

    feasibility_penalty = 0.0
    if feasibility_class == "marginal":
        feasibility_penalty = (
            0.35
            + 0.18 * _excess(overlap_ripple_delta_db, 4.0)
            + 0.18 * _excess(sub_dominance_delta_db, 4.0)
            + 0.06 * _excess(gd_mismatch_delta_ms, 10.0)
        )
        feasibility_penalty = float(np.clip(feasibility_penalty, 0.0, 2.0))
    elif feasibility_class == "infeasible":
        feasibility_penalty = (
            2.5
            + 0.18 * _excess(overlap_ripple_db, 12.0)
            + 0.22 * _excess(sub_dominance_db, 12.0)
            + 0.08 * _excess(gd_rms_mismatch_ms, 20.0)
            + 0.20 * _excess(overlap_ripple_delta_db, 8.0)
            + 0.24 * _excess(sub_dominance_delta_db, 8.0)
            + 0.06 * _excess(gd_mismatch_delta_ms, 18.0)
        )
        feasibility_penalty = float(np.clip(feasibility_penalty, 0.0, 6.0))

    dbg = {
        "bass_cancellation_risk": float(cancellation_risk) if np.isfinite(cancellation_risk) else float("nan"),
        "bass_overlap_ripple": float(overlap_ripple_db) if np.isfinite(overlap_ripple_db) else float("nan"),
        "bass_sub_dominance": float(sub_dominance_db) if np.isfinite(sub_dominance_db) else float("nan"),
        "bass_null_severity": float(null_severity) if np.isfinite(null_severity) else float("nan"),
        "bass_xo_gd_rms_mismatch_ms": float(gd_rms_mismatch_ms) if np.isfinite(gd_rms_mismatch_ms) else float("nan"),
        "bass_overlap_extension_active": bool(overlap_extension_active),
        "bass_overlap_extension_flatness_db": float(overlap_extension_flatness_db)
        if np.isfinite(overlap_extension_flatness_db) else float("nan"),
        "bass_overlap_extension_cancellation_risk": float(overlap_extension_cancellation_risk)
        if np.isfinite(overlap_extension_cancellation_risk) else float("nan"),
        "bass_overlap_extension_peak_excess_db": float(overlap_extension_peak_excess_db)
        if np.isfinite(overlap_extension_peak_excess_db) else float("nan"),
        "bass_overlap_extension_sub_dominance_db": float(overlap_extension_sub_dominance_db)
        if np.isfinite(overlap_extension_sub_dominance_db) else float("nan"),
        "bass_overlap_ripple_delta_db": float(overlap_ripple_delta_db) if np.isfinite(overlap_ripple_delta_db) else float("nan"),
        "bass_sub_dominance_delta_db": float(sub_dominance_delta_db) if np.isfinite(sub_dominance_delta_db) else float("nan"),
        "bass_xo_gd_mismatch_delta_ms": float(gd_mismatch_delta_ms) if np.isfinite(gd_mismatch_delta_ms) else float("nan"),
        "bass_dominant_channel": str(dominant_channel or "unknown"),
        "bass_feasibility_class": str(feasibility_class or "unknown"),
        "bass_feasibility_reason": str(feasibility_reason),
        "lf_boost_max_db": float(lf_boost),
        "net_boost_max_db": float(net_boost) if np.isfinite(net_boost) else float("nan"),
        "pen_cancel": float(pen_cancel),
        "pen_overlap": float(pen_overlap),
        "pen_anti_null": float(pen_anti_null),
        "pen_sub": float(pen_sub),
        "pen_null": float(pen_null),
        "pen_gd": float(pen_gd),
        "pen_ext_flatness": float(pen_ext_flatness),
        "pen_ext_cancel": float(pen_ext_cancel),
        "pen_ext_peak": float(pen_ext_peak),
        "pen_ext_sub": float(pen_ext_sub),
        "pen_overlap_delta": float(pen_overlap_delta),
        "pen_sub_delta": float(pen_sub_delta),
        "pen_gd_delta": float(pen_gd_delta),
        "pen_feasibility_base": float(pen_feasibility_base),
        "pen_total": float(penalty),
        "pen_feasibility_total": float(feasibility_penalty),
        "profile": shared._auto_bass_integration_profile_norm(
            data.get("bass_integration_profile", result_metrics.get("bass_integration_profile", "safe"))
        ),
    }
    return penalty, feasibility_penalty, dbg


def _auto_exc_penalty_bins_from_dbg(exc_dbg: dict | None) -> float:
    try:
        dbg = dict(exc_dbg or {})
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
        dbg = {}
    v = shared._auto_safe_float(dbg.get("pen_bins", float("nan")), float("nan"))
    if np.isfinite(v):
        return float(max(0.0, v))
    exc_bins = int(shared._auto_safe_float(dbg.get("exc_bins", 0), 0.0))
    return float(min(2.5, 0.10 * max(0, exc_bins)))


def _auto_exc_zero_penalty_freq_hz_from_stats(st: dict | None) -> float:
    st = dict(st or {})
    v = shared._auto_safe_float(st.get("boost_candidate_min_hz", float("nan")), float("nan"))
    if not np.isfinite(v) or float(v) <= 0.0:
        return float("nan")
    return float(
        np.clip(
            float(v),
            float(shared._auto_safe_float(shared.AUTO_MODE_EXC_MIN_HZ, 20.0)),
            float(shared._auto_safe_float(shared.AUTO_MODE_EXC_MAX_HZ, 80.0)),
        )
    )



__all__ = ["_auto_bass_integration_penalty", "_auto_exc_penalty_bins_from_dbg", "_auto_exc_zero_penalty_freq_hz_from_stats"]
