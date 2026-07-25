# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Residual-peak winner-polish for auto-mode finalize."""

from __future__ import annotations

import json
import logging

import numpy as np

from .auto_mode_profile import profiled_section
from .scoring_ranking import _auto_hard_gate_reasons, get_residual_peak_db, get_residual_peak_hard_gate_db
from .shared_parts import _auto_safe_float
from .winner_polish_utils import _polish_metric_delta, _polish_rank_status

logger = logging.getLogger("DecayCore")


def _evaluate_residual_peak_guards(
    cur_best_metrics: dict,
    cand_metrics: dict,
    min_peak_improve: float,
    goal: str,
    _metric,
) -> tuple[bool, str, list[str], list[str]]:
    guard_values = _residual_peak_guard_values(cur_best_metrics, cand_metrics, _metric)
    guard_flags = _residual_peak_guard_flags(guard_values, min_peak_improve=min_peak_improve, goal=goal)
    reason = _residual_peak_guard_reason(
        guard_flags,
        hard_gate_reasons=guard_flags["hard_gate_reasons"],
    )
    return (
        bool(guard_flags["better"]),
        str(reason),
        list(guard_flags["failed_guards"]),
        list(guard_flags["hard_gate_reasons"]),
    )


def _residual_peak_guard_values(cur_best_metrics: dict, cand_metrics: dict, _metric) -> dict:
    cur_rank = _metric(cur_best_metrics, "rank_score")
    cand_rank = _metric(cand_metrics, "rank_score")
    cur_avg = _metric(cur_best_metrics, "avg_score")
    cand_avg = _metric(cand_metrics, "avg_score")
    cur_boost = _metric(cur_best_metrics, "max_net_boost_db")
    cand_boost = _metric(cand_metrics, "max_net_boost_db")
    cur_peak = _metric(cur_best_metrics, "worst_residual_peak_raw_db")
    cand_peak = _metric(cand_metrics, "worst_residual_peak_raw_db")
    if not np.isfinite(cur_peak):
        cur_peak = _metric(cur_best_metrics, "worst_residual_peak_db")
    if not np.isfinite(cand_peak):
        cand_peak = _metric(cand_metrics, "worst_residual_peak_db")
    return {
        "cand_metrics": dict(cand_metrics or {}),
        "cur_rank": float(cur_rank),
        "cand_rank": float(cand_rank),
        "cur_avg": float(cur_avg),
        "cand_avg": float(cand_avg),
        "cur_boost": float(cur_boost),
        "cand_boost": float(cand_boost),
        "cur_peak": float(cur_peak),
        "cand_peak": float(cand_peak),
        "cur_tracking": float(
            max(
                _metric(cur_best_metrics, "target_tracking_rms_20_200_db", 0.0),
                _metric(cur_best_metrics, "target_tracking_rms_100_500_db", 0.0),
            )
        ),
        "cand_tracking": float(
            max(
                _metric(cand_metrics, "target_tracking_rms_20_200_db", 0.0),
                _metric(cand_metrics, "target_tracking_rms_100_500_db", 0.0),
            )
        ),
        "cur_under": float(_metric(cur_best_metrics, "bass_under_target_penalty", 0.0)),
        "cand_under": float(_metric(cand_metrics, "bass_under_target_penalty", 0.0)),
        "cur_sharp": float(_metric(cur_best_metrics, "correction_sharpness_penalty", 0.0)),
        "cand_sharp": float(_metric(cand_metrics, "correction_sharpness_penalty", 0.0)),
        "cur_dip": float(_metric(cur_best_metrics, "dip_fill_risk_penalty", 0.0)),
        "cand_dip": float(_metric(cand_metrics, "dip_fill_risk_penalty", 0.0)),
        "cur_overfit": float(_metric(cur_best_metrics, "channel_overfit_penalty", 0.0)),
        "cand_overfit": float(_metric(cand_metrics, "channel_overfit_penalty", 0.0)),
    }


def _residual_peak_guard_flags(values: dict, *, min_peak_improve: float, goal: str) -> dict:
    rank_delta = float(values["cand_rank"] - values["cur_rank"]) if np.isfinite(values["cand_rank"]) and np.isfinite(values["cur_rank"]) else float("nan")
    peak_improve = float(values["cur_peak"] - values["cand_peak"]) if np.isfinite(values["cur_peak"]) and np.isfinite(values["cand_peak"]) else float("nan")
    rank_clear = bool(np.isfinite(rank_delta) and float(rank_delta) >= 0.05)
    peak_clear = bool(np.isfinite(peak_improve) and float(peak_improve) >= float(min_peak_improve))
    rank_guard = bool((not np.isfinite(rank_delta)) or float(rank_delta) >= -0.15)
    avg_guard = bool((not (np.isfinite(values["cur_avg"]) and np.isfinite(values["cand_avg"]))) or float(values["cur_avg"] - values["cand_avg"]) <= 1.0)
    boost_guard = bool((not (np.isfinite(values["cur_boost"]) and np.isfinite(values["cand_boost"]))) or float(values["cand_boost"]) <= float(values["cur_boost"]) + 1e-6)
    tracking_guard = bool((not (np.isfinite(values["cur_tracking"]) and np.isfinite(values["cand_tracking"]))) or float(values["cand_tracking"]) <= float(values["cur_tracking"]) + 0.35)
    under_guard = bool((not (np.isfinite(values["cur_under"]) and np.isfinite(values["cand_under"]))) or float(values["cand_under"]) <= float(values["cur_under"]) + 0.50)
    sharpness_guard = bool((not (np.isfinite(values["cur_sharp"]) and np.isfinite(values["cand_sharp"]))) or float(values["cand_sharp"]) <= float(values["cur_sharp"]) + 0.35)
    dip_guard = bool((not (np.isfinite(values["cur_dip"]) and np.isfinite(values["cand_dip"]))) or float(values["cand_dip"]) <= float(values["cur_dip"]) + 0.25)
    overfit_guard = bool((not (np.isfinite(values["cur_overfit"]) and np.isfinite(values["cand_overfit"]))) or float(values["cand_overfit"]) <= float(values["cur_overfit"]) + 0.25)
    peak_guard = bool((not (np.isfinite(values["cur_peak"]) and np.isfinite(values["cand_peak"]))) or float(values["cand_peak"]) <= float(values["cur_peak"]) + 0.10)
    hard_gate_reasons = _auto_hard_gate_reasons(dict(values.get("cand_metrics", {})), goal=goal)
    failed_guards = [
        name
        for ok, name in (
            (rank_guard, "rank_drop"),
            (avg_guard, "avg_drop"),
            (boost_guard, "boost"),
            (tracking_guard, "tracking"),
            (under_guard, "bass_under"),
            (sharpness_guard, "sharpness"),
            (dip_guard, "dip_fill"),
            (overfit_guard, "channel_overfit"),
            (peak_guard, "peak_regression"),
        )
        if not ok
    ]
    better = bool(
        (rank_clear or peak_clear)
        and rank_guard
        and avg_guard
        and boost_guard
        and tracking_guard
        and under_guard
        and sharpness_guard
        and dip_guard
        and overfit_guard
        and peak_guard
        and not hard_gate_reasons
    )
    return {
        "rank_delta": rank_delta,
        "peak_improve": peak_improve,
        "rank_clear": rank_clear,
        "peak_clear": peak_clear,
        "better": better,
        "failed_guards": failed_guards,
        "hard_gate_reasons": list(hard_gate_reasons),
    }


def _residual_peak_guard_reason(values: dict, *, hard_gate_reasons: list[str]) -> str:
    if hard_gate_reasons:
        return "hard_gate:" + ",".join(hard_gate_reasons)
    if values["rank_clear"]:
        return "rank_score"
    if values["peak_clear"]:
        return "residual_peak"
    return "guards:" + ",".join(values["failed_guards"] or ["no_clear_improvement"])


def _residual_peak_initial_meta(*, enabled: bool, phase_label: str) -> dict:
    return {
        "enabled": bool(enabled),
        "applicable": True,
        "phase_label": str(phase_label),
        "tested_count": 0,
        "tested_variants": [],
        "accepted_variants": [],
        "applied": False,
        "worst_peak_before_db": float("nan"),
        "worst_peak_after_db": float("nan"),
        "worst_peak_freq_hz": float("nan"),
        "worst_peak_width_hz": float("nan"),
        "worst_peak_width_oct": float("nan"),
        "modal_support": 0.0,
        "modal_priority": 0.0,
        "modal_max_severity": 0.0,
        "modal_confidence": 0.0,
        "modal_dominant_freq_hz": None,
        "threshold_db": float("nan"),
        "top3_before_db": float("nan"),
        "top3_after_db": float("nan"),
        "rank_before": float("nan"),
        "rank_after": float("nan"),
        "avg_before": float("nan"),
        "avg_after": float("nan"),
    }


def _residual_peak_candidate_signature(cand_test: dict) -> str:
    try:
        return json.dumps(cand_test, sort_keys=True, separators=(",", ":"), default=str)
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
        return str(sorted(dict(cand_test or {}).items()))


def _has_modally_supported_residual_peak(metrics: dict | None) -> bool:
    m = dict(metrics or {})
    if _auto_safe_float(m.get("residual_peak_modal_priority", 0.0), 0.0) >= 0.12:
        return True
    if _auto_safe_float(m.get("residual_peak_modal_promoted_count", 0), 0.0) > 0.0:
        return True
    for raw in list(m.get("residual_peak_candidates", []) or []):
        if not isinstance(raw, dict):
            continue
        peak = dict(raw)
        if str(peak.get("source", "") or "") == "modal_residual":
            return True
        if (
            _auto_safe_float(peak.get("modal_priority", 0.0), 0.0) >= 0.12
            and _auto_safe_float(peak.get("modal_confidence", 0.0), 0.0) >= 0.25
        ):
            return True
    return False


def _residual_peak_freq_bounds(cur_best_metrics: dict | None, base_data_ref: dict | None) -> tuple[float, float]:
    data = dict(base_data_ref or {})
    peak_hz = _auto_safe_float(dict(cur_best_metrics or {}).get("worst_residual_peak_hz"), float("nan"))
    width_oct = _auto_safe_float(dict(cur_best_metrics or {}).get("worst_residual_peak_width_oct"), float("nan"))
    base_lo = max(20.0, _auto_safe_float(data.get("mag_c_min", data.get("mag_c_min_hz", 20.0)), 20.0))
    base_hi = min(
        200.0,
        _auto_safe_float(data.get("mag_c_max", data.get("mag_c_max_hz", 200.0)), 200.0),
    )
    if np.isfinite(peak_hz) and float(peak_hz) > 0.0:
        half = float(np.clip(_auto_safe_float(width_oct, 0.16), 0.08, 0.35))
        base_lo = max(20.0, min(base_lo, float(peak_hz) / float(2.0**half)))
        base_hi = min(200.0, max(base_hi, float(peak_hz) * float(2.0**half)))
    if float(base_hi) <= float(base_lo):
        base_hi = min(200.0, float(base_lo) + 1.0)
    return float(base_lo), float(base_hi)


def _residual_peak_variant_factories(
    *,
    cur_best_preset: dict,
    base_data_ref: dict | None,
    cur_best_metrics: dict | None,
    modal_priority: float,
) -> list:
    def _current_value(key: str, default: float) -> float:
        return float(
            _auto_safe_float(
                cur_best_preset.get(key, dict(base_data_ref or {}).get(key, default)),
                default,
            )
        )

    slope_choices = [0.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 24.0, 36.0]

    def _slope_variant(step_idx: int):
        current = _current_value("max_slope_cut_db_per_oct", 0.0)
        higher = [float(v) for v in slope_choices if float(v) > float(current) + 1e-6]
        if len(higher) < int(step_idx):
            return None
        value = float(higher[int(step_idx) - 1])
        cand = dict(cur_best_preset or {})
        cand["max_slope_cut_db_per_oct"] = float(value)
        return cand, {"knob": "max_slope_cut_db_per_oct", "value": float(value)}

    def _tdc_variant(strength_delta: float, reduction_delta: float):
        strength = float(np.clip(_current_value("tdc_strength", 50.0) + float(strength_delta), 20.0, 75.0))
        reduction = float(np.clip(_current_value("tdc_max_reduction_db", 9.0) + float(reduction_delta), 0.0, 18.0))
        cand = dict(cur_best_preset or {})
        cand["tdc_strength"] = round(float(strength), 1)
        cand["tdc_max_reduction_db"] = round(float(reduction), 1)
        if (
            abs(float(cand["tdc_strength"]) - _current_value("tdc_strength", 50.0)) <= 1e-9
            and abs(float(cand["tdc_max_reduction_db"]) - _current_value("tdc_max_reduction_db", 9.0)) <= 1e-9
        ):
            return None
        return cand, {
            "knob": "tdc",
            "tdc_strength": float(cand["tdc_strength"]),
            "tdc_max_reduction_db": float(cand["tdc_max_reduction_db"]),
        }

    def _conf_pull_variant(delta: float):
        value = float(np.clip(_current_value("conf_pull_gamma_cut", 0.45) + float(delta), 0.35, 0.90))
        cand = dict(cur_best_preset or {})
        cand["conf_pull_gamma_cut"] = round(float(value), 4)
        if abs(float(cand["conf_pull_gamma_cut"]) - _current_value("conf_pull_gamma_cut", 0.45)) <= 1e-9:
            return None
        return cand, {"knob": "conf_pull_gamma_cut", "value": float(cand["conf_pull_gamma_cut"])}

    def _hybrid_iir_variant():
        if bool(cur_best_preset.get("hybrid_iir_enabled", dict(base_data_ref or {}).get("hybrid_iir_enabled", False))):
            return None
        if not _has_modally_supported_residual_peak(cur_best_metrics):
            return None
        lo_hz, hi_hz = _residual_peak_freq_bounds(cur_best_metrics, base_data_ref)
        threshold = _auto_safe_float(dict(cur_best_metrics or {}).get("residual_peak_threshold_db", 3.0), 3.0)
        cand = dict(cur_best_preset or {})
        cand["hybrid_iir_enabled"] = True
        cand["hybrid_iir_min_freq_hz"] = float(lo_hz)
        cand["hybrid_iir_max_freq_hz"] = float(hi_hz)
        cand["hybrid_iir_min_peak_db"] = float(np.clip(min(_current_value("hybrid_iir_min_peak_db", 4.0), max(3.0, threshold)), 1.5, 8.0))
        cand["hybrid_iir_min_cut_priority"] = float(np.clip(_current_value("hybrid_iir_min_cut_priority", 0.0), 0.0, 1.0))
        cand["hybrid_iir_max_cut_db"] = float(np.clip(_current_value("hybrid_iir_max_cut_db", 6.0), 0.0, 6.0))
        return cand, {
            "knob": "hybrid_iir_enabled",
            "value": True,
            "min_freq_hz": float(lo_hz),
            "max_freq_hz": float(hi_hz),
        }

    variant_factories = [
        lambda: _slope_variant(1),
        lambda: _slope_variant(2),
        lambda: _conf_pull_variant(-0.06),
        _hybrid_iir_variant,
    ]
    if modal_priority >= 0.25:
        variant_factories.extend(
            [
                lambda: _tdc_variant(8.0, 1.5),
                lambda: _tdc_variant(15.0, 3.0),
            ]
        )
    if modal_priority >= 0.45:
        variant_factories.append(lambda: _conf_pull_variant(-0.12))
    return list(variant_factories)


def _residual_peak_finalize_meta(meta: dict, cur_best_metrics: dict, _metric) -> dict:
    meta["worst_peak_after_db"] = float(_metric(cur_best_metrics, "worst_residual_peak_raw_db", _metric(cur_best_metrics, "worst_residual_peak_db")))
    meta["worst_peak_freq_hz"] = float(_metric(cur_best_metrics, "worst_residual_peak_hz"))
    meta["worst_peak_width_hz"] = float(_metric(cur_best_metrics, "worst_residual_peak_width_hz"))
    meta["worst_peak_width_oct"] = float(_metric(cur_best_metrics, "worst_residual_peak_width_oct"))
    meta["top3_after_db"] = float(_metric(cur_best_metrics, "top3_residual_peak_mean_db"))
    meta["rank_after"] = float(_metric(cur_best_metrics, "rank_score"))
    meta["avg_after"] = float(_metric(cur_best_metrics, "avg_score"))
    return dict(meta or {})


def _apply_residual_peak_candidate_result(
    *,
    phase_label: str,
    cur_best_preset: dict,
    cur_best_metrics: dict,
    cand_test: dict,
    cand_metrics: dict,
    variant_meta: dict,
    meta: dict,
    better: bool,
    reason: str,
    hard_gate_reasons: list[str],
    cache_ready_preset,
    status_cb,
    _metric,
) -> tuple[dict, dict, bool, dict]:
    cur_peak = _metric(cur_best_metrics, "worst_residual_peak_raw_db")
    if not np.isfinite(cur_peak):
        cur_peak = _metric(cur_best_metrics, "worst_residual_peak_db")
    cand_peak = _metric(cand_metrics, "worst_residual_peak_raw_db")
    if not np.isfinite(cand_peak):
        cand_peak = _metric(cand_metrics, "worst_residual_peak_db")
    cur_rank = _metric(cur_best_metrics, "rank_score")
    cand_rank = _metric(cand_metrics, "rank_score")
    if not bool(better):
        reject_reasons = dict(meta.get("reject_reasons", {}) or {})
        reject_reasons[str(reason)] = int(reject_reasons.get(str(reason), 0) or 0) + 1
        meta["reject_reasons"] = reject_reasons
        meta["rejected_reason"] = str(reason)
    meta["last_candidate_delta"] = _polish_metric_delta(cur_best_metrics, cand_metrics)
    meta["last_candidate_hard_gate"] = {
        "hard_gate_failed": bool(hard_gate_reasons),
        "hard_gate_reasons": list(hard_gate_reasons),
        "residual_peak_db": float(get_residual_peak_db(cand_metrics)),
        "residual_peak_hard_gate_db": float(get_residual_peak_hard_gate_db(cand_metrics)),
    }
    logger.info(
        "Automatic mode %s residual-peak candidate %d: %s, peak %.3f -> %.3f dB, rank %.3f -> %.3f, decision=%s (%s)",
        str(phase_label),
        int(meta["tested_count"]),
        str(variant_meta),
        float(cur_peak) if np.isfinite(cur_peak) else float("nan"),
        float(cand_peak) if np.isfinite(cand_peak) else float("nan"),
        float(cur_rank) if np.isfinite(cur_rank) else float("nan"),
        float(cand_rank) if np.isfinite(cand_rank) else float("nan"),
        "accept" if bool(better) else "reject",
        str(reason),
    )
    if not bool(better):
        return cur_best_preset, cur_best_metrics, False, meta

    prev_best = dict(cur_best_metrics or {})
    cur_best_metrics = dict(cand_metrics or {})
    cur_best_preset = cache_ready_preset(cand_test, best_metrics=cur_best_metrics)
    accepted = list(meta.get("accepted_variants", []) or [])
    accepted_meta = dict(variant_meta or {})
    accepted_meta["worst_peak_before_db"] = float(_metric(prev_best, "worst_residual_peak_raw_db", _metric(prev_best, "worst_residual_peak_db")))
    accepted_meta["worst_peak_after_db"] = float(_metric(cur_best_metrics, "worst_residual_peak_raw_db", _metric(cur_best_metrics, "worst_residual_peak_db")))
    accepted_meta["worst_peak_freq_hz"] = float(_metric(cur_best_metrics, "worst_residual_peak_hz"))
    accepted_meta["worst_peak_width_hz"] = float(_metric(cur_best_metrics, "worst_residual_peak_width_hz"))
    accepted_meta["worst_peak_width_oct"] = float(_metric(cur_best_metrics, "worst_residual_peak_width_oct"))
    accepted_meta["rank_before"] = float(_metric(prev_best, "rank_score"))
    accepted_meta["rank_after"] = float(_metric(cur_best_metrics, "rank_score"))
    accepted_meta["reason"] = str(reason)
    accepted.append(accepted_meta)
    meta["accepted_variants"] = accepted
    meta["accepted_reason"] = str(reason)
    if callable(status_cb):
        status_cb(
            "DecayCore automatic mode: residual-peak polish improved "
            f"(peak {_metric(prev_best, 'worst_residual_peak_raw_db', _metric(prev_best, 'worst_residual_peak_db')):.2f} -> "
            f"{_metric(cur_best_metrics, 'worst_residual_peak_raw_db', _metric(cur_best_metrics, 'worst_residual_peak_db')):.2f} dB, "
            f"rank {_polish_rank_status(cur_best_metrics)})"
    )
    return cur_best_preset, cur_best_metrics, True, meta


def _prepare_residual_peak_winner_polish_context(
    *,
    cur_best_metrics: dict,
    cur_best_preset: dict,
    best_metrics: dict | None,
    base_data_ref: dict | None,
    phase_label: str,
    goal: str,
    enabled: bool,
    max_variants: int,
    min_improvement_db: float,
    _metric,
) -> dict | None:
    meta = _residual_peak_initial_meta(enabled=bool(enabled), phase_label=str(phase_label))
    if not bool(enabled):
        return {"meta": meta, "enabled": False}
    if not isinstance(cur_best_metrics, dict) or not cur_best_metrics:
        return {"meta": meta, "enabled": False}

    threshold_db = _metric(cur_best_metrics, "residual_peak_threshold_db", 3.0)
    if not np.isfinite(threshold_db):
        threshold_db = 3.0
    start_worst = _metric(cur_best_metrics, "worst_residual_peak_raw_db")
    if not np.isfinite(start_worst):
        start_worst = _metric(cur_best_metrics, "worst_residual_peak_db")
    start_top3 = _metric(cur_best_metrics, "top3_residual_peak_mean_db")
    meta["worst_peak_before_db"] = float(start_worst) if np.isfinite(start_worst) else float("nan")
    meta["worst_peak_freq_hz"] = _metric(cur_best_metrics, "worst_residual_peak_hz")
    meta["worst_peak_width_hz"] = _metric(cur_best_metrics, "worst_residual_peak_width_hz")
    meta["worst_peak_width_oct"] = _metric(cur_best_metrics, "worst_residual_peak_width_oct")
    modal_priority = float(np.clip(_metric(cur_best_metrics, "residual_peak_modal_priority", 0.0), 0.0, 1.0))
    modal_support = float(np.clip(_metric(cur_best_metrics, "residual_peak_modal_support", 0.0), 0.0, 1.0))
    meta["modal_support"] = float(modal_support)
    meta["modal_priority"] = float(modal_priority)
    meta["modal_max_severity"] = float(np.clip(_metric(cur_best_metrics, "residual_peak_modal_max_severity", 0.0), 0.0, 1.0))
    meta["modal_confidence"] = float(np.clip(_metric(cur_best_metrics, "residual_peak_modal_confidence", 0.0), 0.0, 1.0))
    dom_hz = dict(cur_best_metrics or {}).get("residual_peak_modal_dominant_freq_hz")
    meta["modal_dominant_freq_hz"] = float(dom_hz) if dom_hz is not None and np.isfinite(_auto_safe_float(dom_hz, float("nan"))) else None
    meta["threshold_db"] = float(threshold_db)
    meta["top3_before_db"] = float(start_top3) if np.isfinite(start_top3) else float("nan")
    meta["rank_before"] = _metric(cur_best_metrics, "rank_score")
    meta["avg_before"] = _metric(cur_best_metrics, "avg_score")
    if not np.isfinite(start_worst) or float(start_worst) <= float(threshold_db):
        meta["worst_peak_after_db"] = meta["worst_peak_before_db"]
        meta["top3_after_db"] = meta["top3_before_db"]
        meta["rank_after"] = meta["rank_before"]
        meta["avg_after"] = meta["avg_before"]
        meta["rejected_reason"] = (
            "no_peaks_above_threshold" if not np.isfinite(start_worst) else "worst_peak_below_threshold"
        )
        return {"meta": meta, "enabled": True, "short_circuit": True, "cur_best_preset": cur_best_preset, "cur_best_metrics": cur_best_metrics}

    max_variants_eff = int(max(1, _auto_safe_float(max_variants, 8)))
    min_peak_improve_base = float(max(0.0, _auto_safe_float(min_improvement_db, 0.75)))
    if modal_priority >= 0.45:
        min_peak_improve = float(max(0.35, min_peak_improve_base * (1.0 - 0.30 * modal_priority)))
    elif modal_priority <= 0.10:
        min_peak_improve = float(min_peak_improve_base + 0.10)
    else:
        min_peak_improve = float(min_peak_improve_base)
    meta["min_peak_improvement_base_db"] = float(min_peak_improve_base)
    meta["min_peak_improvement_effective_db"] = float(min_peak_improve)

    variant_factories = _residual_peak_variant_factories(
        cur_best_preset=cur_best_preset,
        base_data_ref=base_data_ref,
        cur_best_metrics=cur_best_metrics,
        modal_priority=float(modal_priority),
    )
    return {
        "meta": meta,
        "enabled": True,
        "short_circuit": False,
        "cur_best_preset": cur_best_preset,
        "cur_best_metrics": cur_best_metrics,
        "threshold_db": float(threshold_db),
        "min_peak_improve": float(min_peak_improve),
        "variant_factories": variant_factories,
        "max_variants_eff": int(max_variants_eff),
    }


def _residual_peak_metric(src: dict | None, key: str, default: float = float("nan")) -> float:
    return float(_auto_safe_float(dict(src or {}).get(key, default), default))


def apply_residual_peak_winner_polish(
    *,
    best_preset: dict | None,
    best_metrics: dict | None,
    base_data_ref: dict | None,
    phase_label: str,
    goal: str,
    enabled: bool,
    max_variants: int,
    min_improvement_db: float,
    status_cb,
    materialize_preset_result,
    cache_ready_preset,
    auto_is_better_refine=None,
) -> tuple[dict, dict, bool, dict]:
    cur_best_preset = dict(best_preset or {})
    cur_best_metrics = dict(best_metrics or {})

    state = _prepare_residual_peak_winner_polish_context(
        cur_best_metrics=cur_best_metrics,
        cur_best_preset=cur_best_preset,
        best_metrics=best_metrics,
        base_data_ref=base_data_ref,
        phase_label=phase_label,
        goal=goal,
        enabled=enabled,
        max_variants=max_variants,
        min_improvement_db=min_improvement_db,
        _metric=_residual_peak_metric,
    )
    meta = dict(state.get("meta", {}) or {}) if isinstance(state, dict) else {}
    if not bool(state and state.get("enabled", False)):
        return cur_best_preset, cur_best_metrics, False, meta
    if bool(state.get("short_circuit", False)):
        return (
            dict(state.get("cur_best_preset", cur_best_preset)),
            dict(state.get("cur_best_metrics", cur_best_metrics)),
            False,
            _residual_peak_finalize_meta(meta, dict(state.get("cur_best_metrics", cur_best_metrics)), _residual_peak_metric),
        )

    min_peak_improve = float(state.get("min_peak_improve", 0.75))
    variant_factories = list(state.get("variant_factories", []) or [])
    max_variants_eff = int(state.get("max_variants_eff", 1))
    start_worst = _residual_peak_metric(cur_best_metrics, "worst_residual_peak_raw_db")
    if not np.isfinite(start_worst):
        start_worst = _residual_peak_metric(cur_best_metrics, "worst_residual_peak_db")

    improved = False
    tested_signatures: set[str] = set()
    with profiled_section("winner_polish.residual_peak"):
        logger.info(
            "Automatic mode %s: residual-peak winner polish starting (worst_peak=%.3f dB).",
            str(phase_label),
            float(start_worst),
        )
        for make_variant in variant_factories:
            if int(meta["tested_count"]) >= int(max_variants_eff):
                break
            made = make_variant()
            if made is None:
                continue
            cand_test, variant_meta = made
            sig = _residual_peak_candidate_signature(cand_test)
            if sig in tested_signatures:
                continue
            tested_signatures.add(str(sig))

            meta["tested_count"] = int(meta["tested_count"]) + 1
            tested = list(meta.get("tested_variants", []) or [])
            tested.append(dict(variant_meta or {}))
            meta["tested_variants"] = tested
            try:
                _result, cand_metrics, _data = materialize_preset_result(
                    cand_test,
                    include_response_arrays=False,
                    summarize=False,
                    base_data_override=base_data_ref,
                )
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
            ) as exc:
                logger.warning(
                    "Automatic mode %s failed for residual-peak candidate %d (%s): %s",
                    str(phase_label),
                    int(meta["tested_count"]),
                    str(variant_meta),
                    f"{type(exc).__name__}: {exc}",
                )
                continue

            cand_metrics = dict(cand_metrics or {})
            better, reason, failed_guards, hard_gate_reasons = _evaluate_residual_peak_guards(
                cur_best_metrics,
                cand_metrics,
                min_peak_improve,
                goal,
                _residual_peak_metric,
            )
            cur_best_preset, cur_best_metrics, accepted, meta = _apply_residual_peak_candidate_result(
                phase_label=phase_label,
                cur_best_preset=cur_best_preset,
                cur_best_metrics=cur_best_metrics,
                cand_test=cand_test,
                cand_metrics=cand_metrics,
                variant_meta=variant_meta,
                meta=meta,
                better=bool(better),
                reason=str(reason),
                hard_gate_reasons=list(hard_gate_reasons),
                cache_ready_preset=cache_ready_preset,
                status_cb=status_cb,
                _metric=_residual_peak_metric,
            )
            if not bool(accepted):
                continue
            improved = True

    meta["applied"] = bool(improved)
    return cur_best_preset, cur_best_metrics, bool(improved), _residual_peak_finalize_meta(meta, cur_best_metrics, _residual_peak_metric)
