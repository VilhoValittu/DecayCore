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

_logger = logging.getLogger(__name__)

from .. import shared_parts as shared

def _bass_metric(result_metrics: dict, key: str, default: float = float("nan")) -> float:
    return shared._auto_safe_float(result_metrics.get(key, default), default)


def _bass_integration_extract_metrics(result_metrics: dict) -> dict:
    return {
        "cancellation_risk": _bass_metric(result_metrics, "bass_cancellation_risk"),
        "overlap_ripple_db": _bass_metric(result_metrics, "bass_overlap_ripple"),
        "sub_dominance_db": _bass_metric(result_metrics, "bass_sub_dominance"),
        "null_severity": _bass_metric(result_metrics, "bass_null_severity"),
        "gd_rms_mismatch_ms": shared._auto_safe_float(
            result_metrics.get(
                "bass_xo_gd_rms_mismatch_ms",
                result_metrics.get("bass_xo_gd_mismatch_ms", float("nan")),
            ),
            float("nan"),
        ),
        "overlap_extension_active": bool(result_metrics.get("bass_overlap_extension_active", False)),
        "overlap_extension_flatness_db": _bass_metric(result_metrics, "bass_overlap_extension_flatness_db"),
        "overlap_extension_cancellation_risk": _bass_metric(result_metrics, "bass_overlap_extension_cancellation_risk"),
        "overlap_extension_peak_excess_db": _bass_metric(result_metrics, "bass_overlap_extension_peak_excess_db"),
        "overlap_extension_sub_dominance_db": _bass_metric(result_metrics, "bass_overlap_extension_sub_dominance_db"),
        "overlap_ripple_delta_db": _bass_metric(result_metrics, "bass_overlap_ripple_delta_db"),
        "sub_dominance_delta_db": _bass_metric(result_metrics, "bass_sub_dominance_delta_db"),
        "gd_mismatch_delta_ms": _bass_metric(result_metrics, "bass_xo_gd_mismatch_delta_ms"),
        "feasibility_class": str(result_metrics.get("bass_feasibility_class", "") or "").strip().lower(),
        "feasibility_reason": str(result_metrics.get("bass_feasibility_reason", "") or "").strip(),
        "dominant_channel": str(result_metrics.get("bass_dominant_channel", "unknown") or "unknown").strip().lower(),
    }


def _bass_integration_boost_state(result, *, net_boost_max_db: float) -> tuple[float, float]:
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
    return float(lf_boost), float(net_boost)


def _bass_integration_component_penalties(
    *,
    metrics: dict,
    weights: dict,
    lf_boost: float,
    net_boost: float,
) -> tuple[dict, float]:
    cancellation_risk = float(metrics["cancellation_risk"])
    overlap_ripple_db = float(metrics["overlap_ripple_db"])
    sub_dominance_db = float(metrics["sub_dominance_db"])
    null_severity = float(metrics["null_severity"])
    gd_rms_mismatch_ms = float(metrics["gd_rms_mismatch_ms"])
    overlap_extension_active = bool(metrics["overlap_extension_active"])
    overlap_extension_flatness_db = float(metrics["overlap_extension_flatness_db"])
    overlap_extension_cancellation_risk = float(metrics["overlap_extension_cancellation_risk"])
    overlap_extension_peak_excess_db = float(metrics["overlap_extension_peak_excess_db"])
    overlap_extension_sub_dominance_db = float(metrics["overlap_extension_sub_dominance_db"])
    overlap_ripple_delta_db = float(metrics["overlap_ripple_delta_db"])
    sub_dominance_delta_db = float(metrics["sub_dominance_delta_db"])
    gd_mismatch_delta_ms = float(metrics["gd_mismatch_delta_ms"])
    pen_cancel, pen_overlap, pen_anti_null, pen_sub, pen_null, pen_gd = _bass_integration_base_penalties(
        cancellation_risk=cancellation_risk,
        overlap_ripple_db=overlap_ripple_db,
        sub_dominance_db=sub_dominance_db,
        null_severity=null_severity,
        gd_rms_mismatch_ms=gd_rms_mismatch_ms,
        weights=weights,
        lf_boost=lf_boost,
        net_boost=net_boost,
    )
    pen_ext_flatness, pen_ext_cancel, pen_ext_peak, pen_ext_sub = _bass_integration_extension_penalties(
        overlap_extension_active=overlap_extension_active,
        overlap_extension_flatness_db=overlap_extension_flatness_db,
        overlap_extension_cancellation_risk=overlap_extension_cancellation_risk,
        overlap_extension_peak_excess_db=overlap_extension_peak_excess_db,
        overlap_extension_sub_dominance_db=overlap_extension_sub_dominance_db,
        weights=weights,
    )
    pen_overlap_delta, pen_sub_delta, pen_gd_delta = _bass_integration_delta_penalties(
        overlap_ripple_delta_db=overlap_ripple_delta_db,
        sub_dominance_delta_db=sub_dominance_delta_db,
        gd_mismatch_delta_ms=gd_mismatch_delta_ms,
        weights=weights,
    )
    pen_feasibility_base = {
        "good": 0.0,
        "marginal": 0.35,
        "infeasible": 0.9,
    }.get(str(metrics["feasibility_class"]), 0.0)
    components = {
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
    }
    penalty = float(
        np.clip(
            sum(float(v) for v in components.values()),
            0.0,
            12.0,
        )
    )
    return components, penalty


def _bass_integration_base_penalties(
    *,
    cancellation_risk: float,
    overlap_ripple_db: float,
    sub_dominance_db: float,
    null_severity: float,
    gd_rms_mismatch_ms: float,
    weights: dict,
    lf_boost: float,
    net_boost: float,
) -> tuple[float, float, float, float, float, float]:
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
    return float(pen_cancel), float(pen_overlap), float(pen_anti_null), float(pen_sub), float(pen_null), float(pen_gd)


def _bass_integration_extension_penalties(
    *,
    overlap_extension_active: bool,
    overlap_extension_flatness_db: float,
    overlap_extension_cancellation_risk: float,
    overlap_extension_peak_excess_db: float,
    overlap_extension_sub_dominance_db: float,
    weights: dict,
) -> tuple[float, float, float, float]:
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
    return float(pen_ext_flatness), float(pen_ext_cancel), float(pen_ext_peak), float(pen_ext_sub)


def _bass_integration_delta_penalties(
    *,
    overlap_ripple_delta_db: float,
    sub_dominance_delta_db: float,
    gd_mismatch_delta_ms: float,
    weights: dict,
) -> tuple[float, float, float]:
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
    return float(pen_overlap_delta), float(pen_sub_delta), float(pen_gd_delta)


def _bass_integration_excess(value: float, threshold: float) -> float:
    if np.isfinite(value):
        return float(max(0.0, float(value) - float(threshold)))
    return 0.0


def _bass_integration_feasibility_penalty(*, metrics: dict) -> float:
    feasibility_class = str(metrics["feasibility_class"])
    overlap_ripple_delta_db = float(metrics["overlap_ripple_delta_db"])
    sub_dominance_delta_db = float(metrics["sub_dominance_delta_db"])
    gd_mismatch_delta_ms = float(metrics["gd_mismatch_delta_ms"])
    overlap_ripple_db = float(metrics["overlap_ripple_db"])
    sub_dominance_db = float(metrics["sub_dominance_db"])
    gd_rms_mismatch_ms = float(metrics["gd_rms_mismatch_ms"])
    if feasibility_class == "marginal":
        feasibility_penalty = (
            0.35
            + 0.18 * _bass_integration_excess(overlap_ripple_delta_db, 4.0)
            + 0.18 * _bass_integration_excess(sub_dominance_delta_db, 4.0)
            + 0.06 * _bass_integration_excess(gd_mismatch_delta_ms, 10.0)
        )
        return float(np.clip(feasibility_penalty, 0.0, 2.0))
    if feasibility_class == "infeasible":
        feasibility_penalty = (
            2.5
            + 0.18 * _bass_integration_excess(overlap_ripple_db, 12.0)
            + 0.22 * _bass_integration_excess(sub_dominance_db, 12.0)
            + 0.08 * _bass_integration_excess(gd_rms_mismatch_ms, 20.0)
            + 0.20 * _bass_integration_excess(overlap_ripple_delta_db, 8.0)
            + 0.24 * _bass_integration_excess(sub_dominance_delta_db, 8.0)
            + 0.06 * _bass_integration_excess(gd_mismatch_delta_ms, 18.0)
        )
        return float(np.clip(feasibility_penalty, 0.0, 6.0))
    return 0.0


def _bass_integration_debug(
    *,
    metrics: dict,
    components: dict,
    data: dict,
    result_metrics: dict,
    lf_boost: float,
    net_boost: float,
    penalty: float,
    feasibility_penalty: float,
) -> dict:
    return {
        "bass_cancellation_risk": float(metrics["cancellation_risk"]) if np.isfinite(metrics["cancellation_risk"]) else float("nan"),
        "bass_overlap_ripple": float(metrics["overlap_ripple_db"]) if np.isfinite(metrics["overlap_ripple_db"]) else float("nan"),
        "bass_sub_dominance": float(metrics["sub_dominance_db"]) if np.isfinite(metrics["sub_dominance_db"]) else float("nan"),
        "bass_null_severity": float(metrics["null_severity"]) if np.isfinite(metrics["null_severity"]) else float("nan"),
        "bass_xo_gd_rms_mismatch_ms": float(metrics["gd_rms_mismatch_ms"]) if np.isfinite(metrics["gd_rms_mismatch_ms"]) else float("nan"),
        "bass_overlap_extension_active": bool(metrics["overlap_extension_active"]),
        "bass_overlap_extension_flatness_db": float(metrics["overlap_extension_flatness_db"])
        if np.isfinite(metrics["overlap_extension_flatness_db"]) else float("nan"),
        "bass_overlap_extension_cancellation_risk": float(metrics["overlap_extension_cancellation_risk"])
        if np.isfinite(metrics["overlap_extension_cancellation_risk"]) else float("nan"),
        "bass_overlap_extension_peak_excess_db": float(metrics["overlap_extension_peak_excess_db"])
        if np.isfinite(metrics["overlap_extension_peak_excess_db"]) else float("nan"),
        "bass_overlap_extension_sub_dominance_db": float(metrics["overlap_extension_sub_dominance_db"])
        if np.isfinite(metrics["overlap_extension_sub_dominance_db"]) else float("nan"),
        "bass_overlap_ripple_delta_db": float(metrics["overlap_ripple_delta_db"]) if np.isfinite(metrics["overlap_ripple_delta_db"]) else float("nan"),
        "bass_sub_dominance_delta_db": float(metrics["sub_dominance_delta_db"]) if np.isfinite(metrics["sub_dominance_delta_db"]) else float("nan"),
        "bass_xo_gd_mismatch_delta_ms": float(metrics["gd_mismatch_delta_ms"]) if np.isfinite(metrics["gd_mismatch_delta_ms"]) else float("nan"),
        "bass_dominant_channel": str(metrics["dominant_channel"] or "unknown"),
        "bass_feasibility_class": str(metrics["feasibility_class"] or "unknown"),
        "bass_feasibility_reason": str(metrics["feasibility_reason"]),
        "lf_boost_max_db": float(lf_boost),
        "net_boost_max_db": float(net_boost) if np.isfinite(net_boost) else float("nan"),
        "pen_cancel": float(components["pen_cancel"]),
        "pen_overlap": float(components["pen_overlap"]),
        "pen_anti_null": float(components["pen_anti_null"]),
        "pen_sub": float(components["pen_sub"]),
        "pen_null": float(components["pen_null"]),
        "pen_gd": float(components["pen_gd"]),
        "pen_ext_flatness": float(components["pen_ext_flatness"]),
        "pen_ext_cancel": float(components["pen_ext_cancel"]),
        "pen_ext_peak": float(components["pen_ext_peak"]),
        "pen_ext_sub": float(components["pen_ext_sub"]),
        "pen_overlap_delta": float(components["pen_overlap_delta"]),
        "pen_sub_delta": float(components["pen_sub_delta"]),
        "pen_gd_delta": float(components["pen_gd_delta"]),
        "pen_feasibility_base": float(components["pen_feasibility_base"]),
        "pen_total": float(penalty),
        "pen_feasibility_total": float(feasibility_penalty),
        "profile": shared._auto_bass_integration_profile_norm(
            data.get("bass_integration_profile", result_metrics.get("bass_integration_profile", "safe"))
        ),
    }


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
    metrics = _bass_integration_extract_metrics(result_metrics)
    weights = shared._auto_bass_integration_profile_weights(
        data.get("bass_integration_profile", result_metrics.get("bass_integration_profile", "safe"))
    )
    lf_boost, net_boost = _bass_integration_boost_state(
        result,
        net_boost_max_db=net_boost_max_db,
    )
    components, penalty = _bass_integration_component_penalties(
        metrics=metrics,
        weights=dict(weights or {}),
        lf_boost=float(lf_boost),
        net_boost=float(net_boost),
    )
    feasibility_penalty = _bass_integration_feasibility_penalty(metrics=metrics)
    dbg = _bass_integration_debug(
        metrics=metrics,
        components=components,
        data=data,
        result_metrics=result_metrics,
        lf_boost=float(lf_boost),
        net_boost=float(net_boost),
        penalty=float(penalty),
        feasibility_penalty=float(feasibility_penalty),
    )
    return float(penalty), float(feasibility_penalty), dbg


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
