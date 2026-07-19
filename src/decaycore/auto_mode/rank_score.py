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
from typing import Any

import numpy as np

from .shared import _auto_safe_float

logger = logging.getLogger("DecayCore")

try:
    import decaycore_scoring as _rust_scoring
    _RUST_SCORING_AVAILABLE = True
except ImportError:
    _rust_scoring = None  # type: ignore[assignment]
    _RUST_SCORING_AVAILABLE = False
_RUST_SCORING_SUPPORTS_VOICE: bool | None = None

OFFICIAL_RANK_SCORE_CONTEXT = "preset_objective_score"
RUN_RANKING_SCORE_CONTEXT = "run_ranking_score"
OFFICIAL_RANK_SCORE_DISPLAY_GAIN = 1.70
OFFICIAL_RANK_SCORE_DISPLAY_BIAS = 0.0
OFFICIAL_RANK_SCORE_DISPLAY_CALIBRATION = "display_x1_700"

RANK_SCORE_BATCH_FIELDS = (
    "avg_score",
    "phase_benefit_bonus",
    "boost_penalty",
    "event_penalty",
    "lr_delta_penalty",
    "dsp_penalty",
    "bass_prering_penalty",
    "exc_penalty",
    "bass_integration_penalty",
    "bass_feasibility_penalty",
    "bass_preference_bonus",
    "mode_penalty",
    "decay_penalty",
    "residual_peak_penalty",
    "correction_sharpness_penalty",
    "dip_fill_risk_penalty",
    "channel_overfit_penalty",
    "target_tracking_penalty",
    "voice_clarity_penalty",
    "phase_risk_penalty",
    "phase_limit_penalty",
    "thd_boost_penalty",
    "stereo_coherence_penalty",
    "phantom_center_stability_penalty",
    "policy_divergence_penalty",
    "asymmetry_budget_overflow_penalty",
    "worst_channel_relief_bonus",
    "shared_preference_bias",
    "rt60_policy_penalty",
    "harmonic_local_boost_penalty",
    "shared_preference_penalty",
)

_RANK_SCORE_BATCH_SIGNS = np.asarray(
    [
        1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        0.0,
        -1.0,
        -1.0,
        0.0,
    ],
    dtype=np.float64,
)

_RANK_COMPONENT_META = {
    "avg_score": {"role": "base", "direction": "higher_is_better", "unit": "score"},
    "phase_benefit_bonus": {"role": "bonus", "direction": "higher_is_better", "unit": "rank_points"},
    "boost_penalty": {"role": "penalty", "direction": "lower_is_better", "unit": "rank_points"},
    "event_penalty": {"role": "penalty", "direction": "lower_is_better", "unit": "rank_points"},
    "lr_delta_penalty": {"role": "penalty", "direction": "lower_is_better", "unit": "rank_points"},
    "dsp_penalty": {"role": "penalty", "direction": "lower_is_better", "unit": "rank_points"},
    "bass_prering_penalty": {"role": "penalty", "direction": "lower_is_better", "unit": "rank_points"},
    "exc_penalty": {"role": "penalty", "direction": "lower_is_better", "unit": "rank_points"},
    "bass_integration_penalty": {"role": "penalty", "direction": "lower_is_better", "unit": "rank_points"},
    "bass_feasibility_penalty": {"role": "penalty", "direction": "lower_is_better", "unit": "rank_points"},
    "bass_preference_bonus": {"role": "bonus", "direction": "higher_is_better", "unit": "rank_points"},
    "mode_penalty": {"role": "penalty", "direction": "lower_is_better", "unit": "rank_points"},
    "decay_penalty": {"role": "penalty", "direction": "lower_is_better", "unit": "rank_points"},
    "residual_peak_penalty": {"role": "penalty", "direction": "lower_is_better", "unit": "rank_points"},
    "correction_sharpness_penalty": {"role": "penalty", "direction": "lower_is_better", "unit": "rank_points"},
    "dip_fill_risk_penalty": {"role": "penalty", "direction": "lower_is_better", "unit": "rank_points"},
    "channel_overfit_penalty": {"role": "penalty", "direction": "lower_is_better", "unit": "rank_points"},
    "target_tracking_penalty": {"role": "penalty", "direction": "lower_is_better", "unit": "rank_points"},
    "voice_clarity_penalty": {"role": "penalty", "direction": "lower_is_better", "unit": "rank_points"},
    "phase_risk_penalty": {"role": "penalty", "direction": "lower_is_better", "unit": "rank_points"},
    "phase_limit_penalty": {"role": "penalty", "direction": "lower_is_better", "unit": "rank_points"},
    "thd_boost_penalty": {"role": "penalty", "direction": "lower_is_better", "unit": "rank_points"},
    "stereo_coherence_penalty": {"role": "penalty", "direction": "lower_is_better", "unit": "rank_points"},
    "phantom_center_stability_penalty": {"role": "penalty", "direction": "lower_is_better", "unit": "rank_points"},
    "policy_divergence_penalty": {"role": "penalty", "direction": "lower_is_better", "unit": "rank_points"},
    "asymmetry_budget_overflow_penalty": {"role": "penalty", "direction": "lower_is_better", "unit": "rank_points"},
    "worst_channel_relief_bonus": {"role": "bonus", "direction": "higher_is_better", "unit": "rank_points"},
    "rt60_policy_penalty": {"role": "penalty", "direction": "lower_is_better", "unit": "rank_points"},
    "harmonic_local_boost_penalty": {"role": "penalty", "direction": "lower_is_better", "unit": "rank_points"},
    "shared_preference_penalty": {"role": "penalty", "direction": "lower_is_better", "unit": "rank_points"},
    "shared_preference_bias": {"role": "penalty", "direction": "lower_is_better", "unit": "rank_points"},
}


def display_rank_score(score: Any) -> float:
    v = float(_auto_safe_float(score, float("nan")))
    if not np.isfinite(v):
        return float("nan")
    return float(
        np.clip(
            (float(OFFICIAL_RANK_SCORE_DISPLAY_GAIN) * float(v))
            + float(OFFICIAL_RANK_SCORE_DISPLAY_BIAS),
            0.0,
            100.0,
        )
    )


def _official_rank_score_is_calibrated(metrics: dict[str, Any], components: dict[str, Any]) -> bool:
    if str(metrics.get("rank_score_official_calibration", "") or "").strip() == OFFICIAL_RANK_SCORE_DISPLAY_CALIBRATION:
        return True
    ctx = dict(components.get("context", {}) or {})
    return str(ctx.get("official_score_calibration", "") or "").strip() == OFFICIAL_RANK_SCORE_DISPLAY_CALIBRATION


def _attach_official_rank_context(components: dict[str, Any]) -> dict[str, Any]:
    comp = dict(components or {})
    ctx = dict(comp.get("context", {}) or {})
    ctx["official_score_calibration"] = str(OFFICIAL_RANK_SCORE_DISPLAY_CALIBRATION)
    ctx["official_score_display_gain"] = float(OFFICIAL_RANK_SCORE_DISPLAY_GAIN)
    ctx["official_score_display_bias"] = float(OFFICIAL_RANK_SCORE_DISPLAY_BIAS)
    comp["context"] = dict(ctx)
    return comp


def rank_score_component_metadata() -> dict[str, dict[str, str]]:
    return {str(k): dict(v) for k, v in _RANK_COMPONENT_META.items()}


def build_rank_score_breakdown(
    components: dict[str, Any] | None,
    *,
    hard_gates: list[str] | tuple[str, ...] | None = None,
    cache_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    comp = dict(components or {})

    def _v(key: str) -> float:
        return float(_auto_safe_float(comp.get(key, 0.0), 0.0))

    phase_gd = float(
        _v("phase_risk_penalty")
        + _v("phase_limit_penalty")
        + _v("dsp_penalty")
        - _v("phase_benefit_bonus")
    )
    bass_component = float(
        _v("bass_integration_penalty")
        + _v("bass_feasibility_penalty")
        - _v("bass_preference_bonus")
    )
    out = {
        "total": float(_auto_safe_float(comp.get("rank_score", float("nan")), float("nan"))),
        "raw": float(_auto_safe_float(comp.get("rank_score_raw", float("nan")), float("nan"))),
        "magnitude_error_component": float(_v("target_tracking_penalty") + _v("mode_penalty")),
        "voice_clarity_component": float(_v("voice_clarity_penalty")),
        "residual_peak_component": float(_v("residual_peak_penalty")),
        "anti_overfit_component": float(
            _v("correction_sharpness_penalty")
            + _v("dip_fill_risk_penalty")
            + _v("channel_overfit_penalty")
        ),
        "phase_gd_component": float(phase_gd),
        "tdc_decay_component": float(_v("decay_penalty")),
        "boost_cut_component": float(_v("boost_penalty") + _v("thd_boost_penalty")),
        "excess_correction_component": float(_v("exc_penalty")),
        "bass_component": float(bass_component),
        "stereo_policy_component": float(
            _v("stereo_coherence_penalty")
            + _v("phantom_center_stability_penalty")
            + _v("policy_divergence_penalty")
            + _v("asymmetry_budget_overflow_penalty")
            - _v("worst_channel_relief_bonus")
            + _v("shared_preference_penalty")
        ),
        "rt60_policy_component": float(_v("rt60_policy_penalty")),
        "harmonic_local_component": float(_v("harmonic_local_boost_penalty")),
        "hard_gates": list(hard_gates or []),
        "components": {
            key: {
                "value": float(_auto_safe_float(comp.get(key, 0.0), 0.0)),
                **dict(meta),
            }
            for key, meta in _RANK_COMPONENT_META.items()
            if key in comp
        },
    }
    if isinstance(cache_info, dict) and cache_info:
        out["cache"] = dict(cache_info)
    return out


def compute_rank_score_components(
    *,
    avg_score: Any,
    phase_benefit_bonus: Any = 0.0,
    boost_penalty: Any = 0.0,
    event_penalty: Any = 0.0,
    lr_delta_penalty: Any = 0.0,
    dsp_penalty: Any = 0.0,
    bass_prering_penalty: Any = 0.0,
    exc_penalty: Any = 0.0,
    bass_integration_penalty: Any = 0.0,
    bass_feasibility_penalty: Any = 0.0,
    bass_preference_bonus: Any = 0.0,
    mode_penalty: Any = 0.0,
    decay_penalty: Any = 0.0,
    residual_peak_penalty: Any = 0.0,
    correction_sharpness_penalty: Any = 0.0,
    dip_fill_risk_penalty: Any = 0.0,
    channel_overfit_penalty: Any = 0.0,
    target_tracking_penalty: Any = 0.0,
    voice_clarity_penalty: Any = 0.0,
    phase_risk_penalty: Any = 0.0,
    phase_limit_penalty: Any = 0.0,
    thd_boost_penalty: Any = 0.0,
    stereo_coherence_penalty: Any = 0.0,
    phantom_center_stability_penalty: Any = 0.0,
    policy_divergence_penalty: Any = 0.0,
    asymmetry_budget_overflow_penalty: Any = 0.0,
    worst_channel_relief_bonus: Any = 0.0,
    shared_preference_bias: Any = 0.0,
    rt60_policy_penalty: Any = 0.0,
    harmonic_local_boost_penalty: Any = 0.0,
    shared_preference_penalty: Any = 0.0,
    gain: Any = 1.0,
    bias: Any = 0.0,
    score_min: Any = 0.0,
    score_max: Any = 100.0,
    context: str | None = None,
) -> dict[str, Any]:
    global _RUST_SCORING_SUPPORTS_VOICE
    if _RUST_SCORING_AVAILABLE:
        rust_kwargs = {
            "avg_score": float(_auto_safe_float(avg_score, 0.0)),
            "phase_benefit_bonus": float(_auto_safe_float(phase_benefit_bonus, 0.0)),
            "boost_penalty": float(_auto_safe_float(boost_penalty, 0.0)),
            "event_penalty": float(_auto_safe_float(event_penalty, 0.0)),
            "lr_delta_penalty": float(_auto_safe_float(lr_delta_penalty, 0.0)),
            "dsp_penalty": float(_auto_safe_float(dsp_penalty, 0.0)),
            "bass_prering_penalty": float(_auto_safe_float(bass_prering_penalty, 0.0)),
            "exc_penalty": float(_auto_safe_float(exc_penalty, 0.0)),
            "bass_integration_penalty": float(_auto_safe_float(bass_integration_penalty, 0.0)),
            "bass_feasibility_penalty": float(_auto_safe_float(bass_feasibility_penalty, 0.0)),
            "bass_preference_bonus": float(_auto_safe_float(bass_preference_bonus, 0.0)),
            "mode_penalty": float(_auto_safe_float(mode_penalty, 0.0)),
            "decay_penalty": float(_auto_safe_float(decay_penalty, 0.0)),
            "residual_peak_penalty": float(_auto_safe_float(residual_peak_penalty, 0.0)),
            "correction_sharpness_penalty": float(_auto_safe_float(correction_sharpness_penalty, 0.0)),
            "dip_fill_risk_penalty": float(_auto_safe_float(dip_fill_risk_penalty, 0.0)),
            "channel_overfit_penalty": float(_auto_safe_float(channel_overfit_penalty, 0.0)),
            "target_tracking_penalty": float(_auto_safe_float(target_tracking_penalty, 0.0)),
            "phase_risk_penalty": float(_auto_safe_float(phase_risk_penalty, 0.0)),
            "phase_limit_penalty": float(_auto_safe_float(phase_limit_penalty, 0.0)),
            "thd_boost_penalty": float(_auto_safe_float(thd_boost_penalty, 0.0)),
            "stereo_coherence_penalty": float(_auto_safe_float(stereo_coherence_penalty, 0.0)),
            "phantom_center_stability_penalty": float(_auto_safe_float(phantom_center_stability_penalty, 0.0)),
            "policy_divergence_penalty": float(_auto_safe_float(policy_divergence_penalty, 0.0)),
            "asymmetry_budget_overflow_penalty": float(_auto_safe_float(asymmetry_budget_overflow_penalty, 0.0)),
            "worst_channel_relief_bonus": float(_auto_safe_float(worst_channel_relief_bonus, 0.0)),
            "shared_preference_bias": float(_auto_safe_float(shared_preference_bias, 0.0)),
            "rt60_policy_penalty": float(_auto_safe_float(rt60_policy_penalty, 0.0)),
            "harmonic_local_boost_penalty": float(_auto_safe_float(harmonic_local_boost_penalty, 0.0)),
            "shared_preference_penalty": float(_auto_safe_float(shared_preference_penalty, 0.0)),
            "gain": float(_auto_safe_float(gain, 1.0)),
            "bias": float(_auto_safe_float(bias, 0.0)),
            "score_min": float(_auto_safe_float(score_min, 0.0)),
            "score_max": float(_auto_safe_float(score_max, 100.0)),
            "context": context,
        }
        voice_pen_rust = float(max(0.0, _auto_safe_float(voice_clarity_penalty, 0.0)))
        if _RUST_SCORING_SUPPORTS_VOICE is not False:
            try:
                out = dict(
                    _rust_scoring.compute_rank_score_components(
                        **rust_kwargs,
                        voice_clarity_penalty=voice_pen_rust,
                    )
                )
                _RUST_SCORING_SUPPORTS_VOICE = True
                return out
            except TypeError:
                _RUST_SCORING_SUPPORTS_VOICE = False
        try:
            legacy = dict(_rust_scoring.compute_rank_score_components(**rust_kwargs))
            if voice_pen_rust > 0.0:
                g_legacy = float(_auto_safe_float(legacy.get("rank_score_gain", gain), gain))
                b_legacy = float(_auto_safe_float(legacy.get("rank_score_bias", bias), bias))
                ctx_legacy = dict(legacy.get("context", {}) or {})
                lo_legacy = float(_auto_safe_float(ctx_legacy.get("score_min", score_min), score_min))
                hi_legacy = float(_auto_safe_float(ctx_legacy.get("score_max", score_max), score_max))
                if hi_legacy < lo_legacy:
                    lo_legacy, hi_legacy = hi_legacy, lo_legacy
                raw_legacy = float(_auto_safe_float(legacy.get("rank_score_raw", 0.0), 0.0)) - voice_pen_rust
                legacy["rank_score_raw"] = float(raw_legacy)
                legacy["rank_score"] = float(np.clip((g_legacy * raw_legacy) + b_legacy, lo_legacy, hi_legacy))
            legacy["voice_clarity_penalty"] = float(voice_pen_rust)
            return legacy
        except TypeError:
            logger.warning("decaycore_scoring extension is older than Python rank scorer; using Python fallback.")
    avg = float(_auto_safe_float(avg_score, 0.0))
    phase_bonus = float(max(0.0, _auto_safe_float(phase_benefit_bonus, 0.0)))
    boost = float(max(0.0, _auto_safe_float(boost_penalty, 0.0)))
    event = float(max(0.0, _auto_safe_float(event_penalty, 0.0)))
    lr_pen = float(max(0.0, _auto_safe_float(lr_delta_penalty, 0.0)))
    dsp_pen = float(max(0.0, _auto_safe_float(dsp_penalty, 0.0)))
    bass_prering_pen = float(max(0.0, _auto_safe_float(bass_prering_penalty, 0.0)))
    exc_pen = float(max(0.0, _auto_safe_float(exc_penalty, 0.0)))
    bass_pen = float(max(0.0, _auto_safe_float(bass_integration_penalty, 0.0)))
    bass_feas_pen = float(max(0.0, _auto_safe_float(bass_feasibility_penalty, 0.0)))
    bass_bonus = float(max(0.0, _auto_safe_float(bass_preference_bonus, 0.0)))
    mode_pen = float(max(0.0, _auto_safe_float(mode_penalty, 0.0)))
    decay_pen = float(max(0.0, _auto_safe_float(decay_penalty, 0.0)))
    residual_peak_pen = float(max(0.0, _auto_safe_float(residual_peak_penalty, 0.0)))
    sharpness_pen = float(max(0.0, _auto_safe_float(correction_sharpness_penalty, 0.0)))
    dip_fill_pen = float(max(0.0, _auto_safe_float(dip_fill_risk_penalty, 0.0)))
    channel_overfit_pen = float(max(0.0, _auto_safe_float(channel_overfit_penalty, 0.0)))
    target_tracking_pen = float(max(0.0, _auto_safe_float(target_tracking_penalty, 0.0)))
    voice_pen = float(max(0.0, _auto_safe_float(voice_clarity_penalty, 0.0)))
    phase_risk = float(max(0.0, _auto_safe_float(phase_risk_penalty, 0.0)))
    phase_pen = float(max(0.0, _auto_safe_float(phase_limit_penalty, 0.0)))
    thd_pen = float(max(0.0, _auto_safe_float(thd_boost_penalty, 0.0)))
    stereo_pen = float(max(0.0, _auto_safe_float(stereo_coherence_penalty, 0.0)))
    center_pen = float(max(0.0, _auto_safe_float(phantom_center_stability_penalty, 0.0)))
    policy_pen = float(max(0.0, _auto_safe_float(policy_divergence_penalty, 0.0)))
    asym_pen = float(max(0.0, _auto_safe_float(asymmetry_budget_overflow_penalty, 0.0)))
    worst_bonus = float(max(0.0, _auto_safe_float(worst_channel_relief_bonus, 0.0)))
    shared_pen = float(
        max(
            max(0.0, _auto_safe_float(shared_preference_bias, 0.0)),
            max(0.0, _auto_safe_float(shared_preference_penalty, 0.0)),
        )
    )
    rt60_pen = float(max(0.0, _auto_safe_float(rt60_policy_penalty, 0.0)))
    harmonic_pen = float(max(0.0, _auto_safe_float(harmonic_local_boost_penalty, 0.0)))
    g = float(_auto_safe_float(gain, 1.0))
    b = float(_auto_safe_float(bias, 0.0))
    lo = float(_auto_safe_float(score_min, 0.0))
    hi = float(_auto_safe_float(score_max, 100.0))
    if hi < lo:
        lo, hi = hi, lo

    rank_raw = float(
        avg
        + phase_bonus
        - boost
        - event
        - lr_pen
        - dsp_pen
        - bass_prering_pen
        - exc_pen
        - bass_pen
        - bass_feas_pen
        + bass_bonus
        - mode_pen
        - decay_pen
        - residual_peak_pen
        - sharpness_pen
        - dip_fill_pen
        - channel_overfit_pen
        - target_tracking_pen
        - voice_pen
        - phase_risk
        - phase_pen
        - thd_pen
        - stereo_pen
        - center_pen
        - policy_pen
        - asym_pen
        + worst_bonus
        - shared_pen
        - rt60_pen
        - harmonic_pen
    )
    rank_score = float(np.clip((g * rank_raw) + b, lo, hi))
    score_kind = str(context or OFFICIAL_RANK_SCORE_CONTEXT).strip() or OFFICIAL_RANK_SCORE_CONTEXT
    score_label = "Best rank score" if score_kind == OFFICIAL_RANK_SCORE_CONTEXT else "Run ranking score"

    return {
        "rank_score": float(rank_score),
        "avg_score": float(avg),
        "phase_benefit_bonus": float(phase_bonus),
        "boost_penalty": float(boost),
        "event_penalty": float(event),
        "lr_delta_penalty": float(lr_pen),
        "dsp_penalty": float(dsp_pen),
        "bass_prering_penalty": float(bass_prering_pen),
        "exc_penalty": float(exc_pen),
        "bass_integration_penalty": float(bass_pen),
        "bass_feasibility_penalty": float(bass_feas_pen),
        "bass_preference_bonus": float(bass_bonus),
        "mode_penalty": float(mode_pen),
        "decay_penalty": float(decay_pen),
        "residual_peak_penalty": float(residual_peak_pen),
        "correction_sharpness_penalty": float(sharpness_pen),
        "dip_fill_risk_penalty": float(dip_fill_pen),
        "channel_overfit_penalty": float(channel_overfit_pen),
        "target_tracking_penalty": float(target_tracking_pen),
        "voice_clarity_penalty": float(voice_pen),
        "phase_risk_penalty": float(phase_risk),
        "phase_limit_penalty": float(phase_pen),
        "thd_boost_penalty": float(thd_pen),
        "stereo_coherence_penalty": float(stereo_pen),
        "phantom_center_stability_penalty": float(center_pen),
        "policy_divergence_penalty": float(policy_pen),
        "asymmetry_budget_overflow_penalty": float(asym_pen),
        "worst_channel_relief_bonus": float(worst_bonus),
        "shared_preference_bias": float(shared_pen),
        "shared_preference_penalty": float(shared_pen),
        "rt60_policy_penalty": float(rt60_pen),
        "harmonic_local_boost_penalty": float(harmonic_pen),
        "rank_score_raw": float(rank_raw),
        "rank_score_gain": float(g),
        "rank_score_bias": float(b),
        "context": {
            "score_kind": str(score_kind),
            "score_label": str(score_label),
            "score_min": float(lo),
            "score_max": float(hi),
        },
    }


def compute_rank_scores_batch(
    component_rows: Any,
    *,
    gain: Any = 1.0,
    bias: Any = 0.0,
    score_min: Any = 0.0,
    score_max: Any = 100.0,
) -> np.ndarray:
    """Combine a ``(candidates, 31)`` component matrix into rank scores.

    Columns follow :data:`RANK_SCORE_BATCH_FIELDS`. This function combines
    numeric terms only; winner-selection hard gates remain mandatory and are
    intentionally not represented in the matrix.
    """
    matrix = np.asarray(component_rows, dtype=np.float64)
    expected_columns = len(RANK_SCORE_BATCH_FIELDS)
    if matrix.ndim != 2 or matrix.shape[1] != expected_columns:
        raise ValueError(
            f"component_rows must have shape (candidates, {expected_columns}), got {matrix.shape}"
        )
    matrix = np.ascontiguousarray(matrix, dtype=np.float64)
    g = float(_auto_safe_float(gain, 1.0))
    b = float(_auto_safe_float(bias, 0.0))
    lo = float(_auto_safe_float(score_min, 0.0))
    hi = float(_auto_safe_float(score_max, 100.0))
    if hi < lo:
        lo, hi = hi, lo

    rust_batch = getattr(_rust_scoring, "compute_rank_scores_batch", None) if _RUST_SCORING_AVAILABLE else None
    if rust_batch is not None:
        try:
            return np.asarray(
                rust_batch(matrix, gain=g, bias=b, score_min=lo, score_max=hi),
                dtype=np.float64,
            )
        except (TypeError, ValueError) as exc:
            logger.warning("Rust batch rank scorer rejected its input; using NumPy fallback: %s", exc)

    clean = np.nan_to_num(matrix, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    positive = np.maximum(clean[:, 1:], 0.0)
    rank_raw = clean[:, 0] + (positive @ _RANK_SCORE_BATCH_SIGNS)
    shared_penalty = np.maximum(positive[:, 26], positive[:, 29])
    rank_raw -= shared_penalty
    return np.asarray(np.clip((g * rank_raw) + b, lo, hi), dtype=np.float64)


def attach_official_rank_score(
    metrics: dict[str, Any] | None,
    *,
    components: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = dict(metrics or {})
    comp = dict(components or out.get("rank_score_components", {}) or {})
    if not comp:
        rank_score = _auto_safe_float(out.get("rank_score", float("nan")), float("nan"))
        if np.isfinite(rank_score):
            comp = {
                "rank_score": float(rank_score),
                "avg_score": float(_auto_safe_float(out.get("avg_score", float("nan")), float("nan"))),
                "phase_benefit_bonus": float(_auto_safe_float(out.get("phase_benefit_bonus", 0.0), 0.0)),
                "boost_penalty": float(_auto_safe_float(out.get("boost_penalty", 0.0), 0.0)),
                "event_penalty": float(_auto_safe_float(out.get("event_penalty", 0.0), 0.0)),
                "lr_delta_penalty": float(_auto_safe_float(out.get("lr_delta_penalty", 0.0), 0.0)),
                "dsp_penalty": float(_auto_safe_float(out.get("dsp_penalty", 0.0), 0.0)),
                "exc_penalty": float(_auto_safe_float(out.get("exc_penalty", 0.0), 0.0)),
                "bass_integration_penalty": float(_auto_safe_float(out.get("bass_integration_penalty", 0.0), 0.0)),
                "bass_feasibility_penalty": float(_auto_safe_float(out.get("bass_feasibility_penalty", 0.0), 0.0)),
                "bass_preference_bonus": float(_auto_safe_float(out.get("bass_preference_bonus", 0.0), 0.0)),
                "mode_penalty": float(_auto_safe_float(out.get("mode_penalty", 0.0), 0.0)),
                "decay_penalty": float(_auto_safe_float(out.get("decay_penalty", 0.0), 0.0)),
                "residual_peak_penalty": float(_auto_safe_float(out.get("residual_peak_penalty", 0.0), 0.0)),
                "correction_sharpness_penalty": float(_auto_safe_float(out.get("correction_sharpness_penalty", 0.0), 0.0)),
                "dip_fill_risk_penalty": float(_auto_safe_float(out.get("dip_fill_risk_penalty", 0.0), 0.0)),
                "channel_overfit_penalty": float(_auto_safe_float(out.get("channel_overfit_penalty", 0.0), 0.0)),
                "target_tracking_penalty": float(_auto_safe_float(out.get("target_tracking_penalty", 0.0), 0.0)),
                "voice_clarity_penalty": float(_auto_safe_float(out.get("voice_clarity_penalty", 0.0), 0.0)),
                "phase_risk_penalty": float(_auto_safe_float(out.get("phase_risk_penalty", 0.0), 0.0)),
                "phase_limit_penalty": float(_auto_safe_float(out.get("phase_limit_penalty", 0.0), 0.0)),
                "stereo_coherence_penalty": float(_auto_safe_float(out.get("stereo_coherence_penalty", 0.0), 0.0)),
                "phantom_center_stability_penalty": float(_auto_safe_float(out.get("phantom_center_stability_penalty", 0.0), 0.0)),
                "policy_divergence_penalty": float(_auto_safe_float(out.get("policy_divergence_penalty", 0.0), 0.0)),
                "asymmetry_budget_overflow_penalty": float(_auto_safe_float(out.get("asymmetry_budget_overflow_penalty", 0.0), 0.0)),
                "worst_channel_relief_bonus": float(_auto_safe_float(out.get("worst_channel_relief_bonus", 0.0), 0.0)),
                "shared_preference_bias": float(_auto_safe_float(out.get("shared_preference_bias", 0.0), 0.0)),
                "context": {
                    "score_kind": str(OFFICIAL_RANK_SCORE_CONTEXT),
                    "score_label": "Best rank score",
                },
            }
    explicit_official = _auto_safe_float(out.get("rank_score_official", float("nan")), float("nan"))
    explicit_official_ok = bool(np.isfinite(explicit_official))
    if explicit_official_ok:
        official = float(explicit_official)
    else:
        official = float(
            _auto_safe_float(
                comp.get("rank_score", out.get("rank_score", float("nan"))),
                float("nan"),
            )
        )
    if np.isfinite(official):
        if not explicit_official_ok and not _official_rank_score_is_calibrated(out, comp):
            official = float(calibrated_auto_quality(official, out))
        comp = _attach_official_rank_context(comp)
        out["rank_score_official"] = float(official)
        out["rank_score_official_calibration"] = str(OFFICIAL_RANK_SCORE_DISPLAY_CALIBRATION)
        out["rank_score_components"] = dict(comp)
        out["rank_score_breakdown"] = build_rank_score_breakdown(
            comp,
            hard_gates=list(out.get("hard_gate_failures", out.get("hard_gate_reasons", [])) or []),
            cache_info=dict(out.get("cache_info", {}) or {}),
        )
    return out


DISPLAY_QUALITY_CALIBRATION = "quality_v1_power2_35"
DISPLAY_QUALITY_EVENT_PENALTY_CAP = 100.0
DISPLAY_QUALITY_RESIDUAL_PEAK_PENALTY_CAP = 100.0


def _quality_band(score: float) -> str:
    s = float(score)
    if s >= 90.0:
        return "Excellent"
    if s >= 80.0:
        return "Good"
    if s >= 70.0:
        return "Usable"
    if s >= 60.0:
        return "Weak"
    return "Poor"


def calibrated_auto_quality(rank_score_0_100: Any, metrics: dict[str, Any] | None = None) -> float:
    """Convert internal rank_score (0-100) to user-facing quality score with soft power curve and hard caps."""
    if _RUST_SCORING_AVAILABLE:
        raw = float(_auto_safe_float(rank_score_0_100, float("nan")))
        return float(_rust_scoring.calibrated_auto_quality(raw, metrics or None))
    raw = float(_auto_safe_float(rank_score_0_100, float("nan")))
    if not np.isfinite(raw):
        return float("nan")
    raw_c = float(np.clip(raw, 0.0, 100.0))
    score = 100.0 * (1.0 - (1.0 - raw_c / 100.0) ** 2.35)
    score = float(np.clip(score, 0.0, 100.0))

    m = dict(metrics or {})

    max_boost = float(_auto_safe_float(m.get("max_net_boost_db", float("nan")), float("nan")))
    if np.isfinite(max_boost) and max_boost > 6.5:
        score = min(score, 69.0)

    event_penalty = float(_auto_safe_float(m.get("event_penalty", 0.0), 0.0))
    if event_penalty >= 3.0:
        score = min(score, DISPLAY_QUALITY_EVENT_PENALTY_CAP)

    hard_gate_failed = bool(m.get("hard_gate_failed", False) or m.get("stereo_policy_gate_failed", False))
    rp_worst = float(_auto_safe_float(m.get("worst_residual_peak_db", float("nan")), float("nan")))
    rp_gate = float(_auto_safe_float(m.get("residual_peak_hard_gate_db", float("nan")), float("nan")))
    if np.isfinite(rp_worst) and np.isfinite(rp_gate) and rp_worst > rp_gate:
        hard_gate_failed = True
    if hard_gate_failed:
        score = min(score, 59.0)

    rp_pen = float(_auto_safe_float(m.get("residual_peak_penalty", 0.0), 0.0))
    if rp_pen > 1.5:
        score = min(score, DISPLAY_QUALITY_RESIDUAL_PEAK_PENALTY_CAP)

    bass_cancel = float(_auto_safe_float(m.get("bass_cancellation_risk", float("nan")), float("nan")))
    if np.isfinite(bass_cancel) and bass_cancel > 0.5:
        score = min(score, 70.0)

    bass_gd_mismatch = float(_auto_safe_float(m.get("bass_xo_gd_mismatch_delta_ms", float("nan")), float("nan")))
    if np.isfinite(bass_gd_mismatch) and abs(bass_gd_mismatch) > 5.0:
        score = min(score, 65.0)

    return float(np.clip(score, 0.0, 100.0))


def compute_run_ranking_score_components(
    *,
    avg_score: Any,
    boost_penalty: Any = 0.0,
    event_penalty: Any = 0.0,
    lr_delta_penalty: Any = 0.0,
    dsp_penalty: Any = 0.0,
) -> dict[str, Any]:
    components = compute_rank_score_components(
        avg_score=avg_score,
        boost_penalty=boost_penalty,
        event_penalty=event_penalty,
        lr_delta_penalty=lr_delta_penalty,
        dsp_penalty=dsp_penalty,
        context=RUN_RANKING_SCORE_CONTEXT,
    )
    out = dict(components)
    out["run_ranking_score"] = float(components.get("rank_score", 0.0))
    out["run_ranking_score_components"] = dict(components)
    return out


def official_rank_score(metrics: dict[str, Any] | None) -> float:
    m = attach_official_rank_score(metrics)
    score = _auto_safe_float(m.get("rank_score_official", float("nan")), float("nan"))
    if np.isfinite(score):
        return float(score)
    comp = dict(m.get("rank_score_components", {}) or {})
    score = _auto_safe_float(comp.get("rank_score", float("nan")), float("nan"))
    if np.isfinite(score):
        return float(calibrated_auto_quality(score, m))
    return float(calibrated_auto_quality(m.get("rank_score", float("nan")), m))
