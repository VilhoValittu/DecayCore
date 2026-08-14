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

from typing import Any

import numpy as np

from .final_ir_validation_parts import validate_final_fir_against_ir

_SEVERITY_RANK = {"ok": 0, "warn": 1, "reject": 2}
_PRE_ENERGY_REGRESSION_TOLERANCE_DB = 0.5
_MAG_RMS_REGRESSION_TOLERANCE_DB = 0.25
_MAG_PEAK_REGRESSION_TOLERANCE_DB = 1.0


def phase_feedback_strength_field(filter_type: str | None) -> str | None:
    mode = str(filter_type or "").strip().lower()
    if "min" in mode:
        return None
    if "mixed" in mode:
        return "excess_phase_strength"
    return "linear_excess_strength"


def build_phase_feedback_strengths(requested: float, candidate_count: int = 5) -> list[float]:
    value = float(np.clip(float(requested), 0.0, 1.0))
    count = int(np.clip(int(candidate_count), 2, 5))
    values = [round(value * float(multiplier), 6) for multiplier in np.linspace(1.0, 0.0, count)]
    return list(dict.fromkeys(values))


def _stats_array(stats: dict, key: str) -> np.ndarray | None:
    try:
        value = stats.get(key)
        if value is None:
            return None
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size < 4 or not np.any(np.isfinite(arr)):
            return None
        return arr
    except (AttributeError, TypeError, ValueError):
        return None


def _active_stats_array(stats: dict, key: str) -> np.ndarray | None:
    if str(stats.get("analysis_mode", "native") or "native").strip().lower() == "comparison":
        cmp_key = {
            "freq_axis": "cmp_freq_axis",
            "target_mags": "cmp_target_mags",
            "measured_mags": "cmp_measured_mags",
            "filter_mags": "cmp_filter_mags",
        }.get(key)
        if cmp_key:
            arr = _stats_array(stats, cmp_key)
            if arr is not None:
                return arr
    return _stats_array(stats, key)


def _safe_float(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _channel_assessment(
    *,
    impulse,
    stats: dict | None,
    cfg,
    measured_ir=None,
) -> dict[str, Any]:
    st = dict(stats or {})
    ir = np.asarray(impulse, dtype=float).reshape(-1)
    finite_ir = bool(ir.size >= 8 and np.all(np.isfinite(ir)))
    validation = validate_final_fir_against_ir(
        sample_rate=int(getattr(cfg, "fs", 48000) or 48000),
        measured_ir_l=measured_ir,
        fir_l=ir,
        freq_axis=_active_stats_array(st, "freq_axis"),
        target_mag_db=_active_stats_array(st, "target_mags"),
        predicted_mag_db_l=_active_stats_array(st, "filter_mags"),
        measured_mag_db_l=_active_stats_array(st, "measured_mags"),
        ir_anchor_mode=str(st.get("ir_anchor_mode", getattr(cfg, "ir_anchor_mode", "")) or ""),
        filter_type=str(st.get("filter_type", st.get("filter_type_str", getattr(cfg, "filter_type_str", ""))) or ""),
        authority_voice_risk=_stats_array(st, "authority_voice_risk"),
        authority_modal_support=_stats_array(st, "authority_modal_support"),
        authority_null_risk=_stats_array(st, "authority_null_risk"),
        authority_reflection_risk=_stats_array(st, "authority_reflection_risk"),
        config=cfg,
    )
    gd_score = _safe_float(validation.metrics.get("gd_improvement_frac"))
    gd_source = "measured_ir_convolution"
    if not np.isfinite(gd_score):
        gd_score = _safe_float(st.get("phase_realized_gd_improvement_score"))
        gd_source = "measurement_plus_final_fir"
    return {
        "finite": finite_ir,
        "length": int(ir.size),
        "severity": str(validation.severity),
        "severity_rank": int(_SEVERITY_RANK.get(str(validation.severity), 2)),
        "gd_score": gd_score,
        "gd_source": gd_source if np.isfinite(gd_score) else "unavailable",
        "pre_energy_db": _safe_float(validation.pre_energy_ratio_db),
        "gd_peak_ms": _safe_float(validation.gd_peak_ms),
        "mag_rms_db": _safe_float(validation.mag_rms_db),
        "mag_peak_db": _safe_float(validation.mag_peak_db),
        "validation": validation,
    }


def assess_phase_feedback_candidate(
    *,
    strength: float,
    requested_strength: float,
    impulse_l,
    stats_l: dict | None,
    cfg,
    impulse_r=None,
    stats_r: dict | None = None,
    measured_ir_l=None,
    measured_ir_r=None,
    payload=None,
) -> dict[str, Any]:
    channels = [
        _channel_assessment(
            impulse=impulse_l,
            stats=stats_l,
            cfg=cfg,
            measured_ir=measured_ir_l,
        )
    ]
    if impulse_r is not None:
        channels.append(
            _channel_assessment(
                impulse=impulse_r,
                stats=stats_r,
                cfg=cfg,
                measured_ir=measured_ir_r,
            )
        )
    requested = float(np.clip(float(requested_strength), 0.0, 1.0))
    selected = float(np.clip(float(strength), 0.0, requested))
    return {
        "strength": selected,
        "requested_strength": requested,
        "multiplier": (selected / requested) if requested > 1e-12 else 1.0,
        "channels": channels,
        "payload": payload,
    }


def _finite_not_worse(candidate: float, baseline: float, tolerance: float) -> bool:
    if np.isfinite(candidate) and np.isfinite(baseline):
        return bool(candidate <= baseline + float(tolerance))
    return True


def _candidate_is_safe(candidate: dict, baseline: dict) -> bool:
    cand_channels = list(candidate.get("channels", []) or [])
    base_channels = list(baseline.get("channels", []) or [])
    if not cand_channels or len(cand_channels) != len(base_channels):
        return False
    for cand, base in zip(cand_channels, base_channels, strict=True):
        if not bool(cand.get("finite", False)) or int(cand.get("length", 0)) != int(base.get("length", 0)):
            return False
        if int(cand.get("severity_rank", 2)) >= _SEVERITY_RANK["reject"]:
            return False
        if int(cand.get("severity_rank", 2)) > int(base.get("severity_rank", 2)):
            return False
        if not _finite_not_worse(
            _safe_float(cand.get("pre_energy_db")),
            _safe_float(base.get("pre_energy_db")),
            _PRE_ENERGY_REGRESSION_TOLERANCE_DB,
        ):
            return False
        if not _finite_not_worse(
            _safe_float(cand.get("mag_rms_db")),
            _safe_float(base.get("mag_rms_db")),
            _MAG_RMS_REGRESSION_TOLERANCE_DB,
        ):
            return False
        if not _finite_not_worse(
            _safe_float(cand.get("mag_peak_db")),
            _safe_float(base.get("mag_peak_db")),
            _MAG_PEAK_REGRESSION_TOLERANCE_DB,
        ):
            return False
    return True


def _gd_key(candidate: dict) -> tuple[float, float, float] | None:
    scores = np.asarray(
        [_safe_float(channel.get("gd_score")) for channel in candidate.get("channels", [])],
        dtype=float,
    )
    if scores.size == 0 or not np.all(np.isfinite(scores)):
        return None
    return float(np.min(scores)), float(np.mean(scores)), float(candidate.get("strength", 0.0))


def select_phase_feedback_candidate(candidates: list[dict]) -> tuple[dict, str]:
    if not candidates:
        raise ValueError("phase feedback requires at least one candidate")
    baseline = candidates[0]
    baseline_key = _gd_key(baseline)
    if baseline_key is None:
        return baseline, "missing_realized_gd_metric"
    safe = [candidate for candidate in candidates if _candidate_is_safe(candidate, baseline)]
    ranked = [(key, candidate) for candidate in safe if (key := _gd_key(candidate)) is not None]
    if not ranked:
        return baseline, "no_safe_candidate"
    _, selected = max(ranked, key=lambda item: item[0])
    if selected is baseline:
        return baseline, "requested_strength_retained"
    return selected, "best_realized_gd"


def phase_feedback_stats(
    *,
    field: str | None,
    candidates: list[dict],
    selected: dict,
    reason: str,
    applicable: bool = True,
) -> dict[str, Any]:
    baseline = candidates[0] if candidates else selected
    base_channels = list(baseline.get("channels", []) or [])
    selected_channels = list(selected.get("channels", []) or [])
    return {
        "phase_realization_feedback_enabled": True,
        "phase_realization_feedback_applicable": bool(applicable),
        "phase_realization_feedback_field": str(field or ""),
        "phase_realization_feedback_requested_strength": float(selected.get("requested_strength", 0.0)),
        "phase_realization_feedback_selected_strength": float(selected.get("strength", 0.0)),
        "phase_realization_feedback_selected_multiplier": float(selected.get("multiplier", 1.0)),
        "phase_realization_feedback_tested_strengths": [float(item.get("strength", 0.0)) for item in candidates],
        "phase_realization_feedback_reason": str(reason),
        "phase_realization_feedback_applied": bool(selected is not baseline),
        "phase_realization_feedback_gd_before": [_safe_float(ch.get("gd_score")) for ch in base_channels],
        "phase_realization_feedback_gd_selected": [_safe_float(ch.get("gd_score")) for ch in selected_channels],
        "phase_realization_feedback_gd_source": [str(ch.get("gd_source", "unavailable")) for ch in selected_channels],
        "phase_realization_feedback_pre_energy_before_db": [
            _safe_float(ch.get("pre_energy_db")) for ch in base_channels
        ],
        "phase_realization_feedback_pre_energy_selected_db": [
            _safe_float(ch.get("pre_energy_db")) for ch in selected_channels
        ],
        "phase_realization_feedback_mag_rms_selected_db": [
            _safe_float(ch.get("mag_rms_db")) for ch in selected_channels
        ],
        "phase_realization_feedback_mag_peak_selected_db": [
            _safe_float(ch.get("mag_peak_db")) for ch in selected_channels
        ],
    }


__all__ = [
    "assess_phase_feedback_candidate",
    "build_phase_feedback_strengths",
    "phase_feedback_stats",
    "phase_feedback_strength_field",
    "select_phase_feedback_candidate",
]
