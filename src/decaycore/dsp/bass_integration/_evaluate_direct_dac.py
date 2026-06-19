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

from ...auto_mode.shared import (
    AUTO_MODE_BASS_INTEGRATION_GUARD_HI_RATIO,
    AUTO_MODE_BASS_INTEGRATION_GUARD_LO_RATIO,
    _auto_bass_integration_profile_norm,
    _auto_bass_integration_profile_weights,
)
from ...io.measurement_bundle import BassIntegrationBundle
from ._candidate import DirectDacCandidate, DirectDacCandidateMetrics, DirectDacChannelMetrics
from ._channel_metrics import (
    _channel_overlap_extension_metrics,
    _channel_overlap_metrics,
    _channel_predicted_sum_metrics,
    _metric_delta,
)
from ._export_filter_model import apply_direct_dac_export_branch_model
from ._gd_feasibility import _classify_bass_integration_feasibility, compute_xo_gd_continuity
from ._sub_combine import build_bundle_combined_sub_transfer, sum_complex_responses
from ._utils import _safe_float, normalize_sub_combine_mode


# Reject reasons that indicate a hardware-safety issue (speaker/hearing).
# These always add +1000 so the candidate is never selected by the optimizer.
_SAFETY_REJECT_REASONS: frozenset[str] = frozenset({
    "sub_gain_trim_above_safe_range",
    "sub_gain_trim_below_safe_range",
    "non_finite_score",
})
# Per-reason score penalty for acoustic-quality rejects (not safety).
# Large enough to prefer any clean candidate, small enough to rank bad candidates.
_ACOUSTIC_REJECT_PENALTY = 5.0


def _finite_or(value: Any, default: float = float("nan")) -> float:
    return float(_safe_float(value, default))


def _branch_bundle(
    bundle: BassIntegrationBundle,
    candidate: DirectDacCandidate,
    *,
    sub_combine_mode: str,
) -> BassIntegrationBundle:
    l_main = apply_direct_dac_export_branch_model(
        bundle.l_main,
        hpf_hz=float(candidate.main_hpf_hz),
        hpf_order=int(candidate.main_hpf_order),
        label="L main CamillaDSP Direct-DAC HPF",
    )
    r_main = apply_direct_dac_export_branch_model(
        bundle.r_main,
        hpf_hz=float(candidate.main_hpf_hz),
        hpf_order=int(candidate.main_hpf_order),
        label="R main CamillaDSP Direct-DAC HPF",
    )
    l_sub_raw, l_combine_diag = build_bundle_combined_sub_transfer(
        bundle,
        channel="l",
        mode=sub_combine_mode,
        label="L Direct-DAC combined sub raw",
    )
    r_sub_raw, r_combine_diag = build_bundle_combined_sub_transfer(
        bundle,
        channel="r",
        mode=sub_combine_mode,
        label="R Direct-DAC combined sub raw",
    )

    allpass_freq = float(candidate.sub_allpass_freq_hz) if bool(candidate.sub_allpass_enabled) else None
    allpass_q = float(candidate.sub_allpass_q) if bool(candidate.sub_allpass_enabled) else None
    l_sub = apply_direct_dac_export_branch_model(
        l_sub_raw,
        hpf_hz=float(candidate.sub_hpf_hz),
        hpf_order=int(candidate.sub_hpf_order),
        lpf_hz=float(candidate.sub_lpf_hz),
        lpf_order=int(candidate.sub_lpf_order),
        gain_trim_db=float(candidate.sub_gain_trim_db),
        delay_ms=float(candidate.sub_delay_ms),
        polarity_invert=bool(candidate.sub_polarity_invert),
        allpass_freq_hz=allpass_freq,
        allpass_q=allpass_q,
        label="L sub CamillaDSP Direct-DAC branch",
    )
    r_sub = apply_direct_dac_export_branch_model(
        r_sub_raw,
        hpf_hz=float(candidate.sub_hpf_hz),
        hpf_order=int(candidate.sub_hpf_order),
        lpf_hz=float(candidate.sub_lpf_hz),
        lpf_order=int(candidate.sub_lpf_order),
        gain_trim_db=float(candidate.sub_gain_trim_db),
        delay_ms=float(candidate.sub_delay_ms),
        polarity_invert=bool(candidate.sub_polarity_invert),
        allpass_freq_hz=allpass_freq,
        allpass_q=allpass_q,
        label="R sub CamillaDSP Direct-DAC branch",
    )
    l_total = sum_complex_responses(l_main, l_sub, label="L Direct-DAC canonical predicted total")
    r_total = sum_complex_responses(r_main, r_sub, label="R Direct-DAC canonical predicted total")
    diagnostics = dict(getattr(bundle, "diagnostics", {}) or {})
    diagnostics.update(dict(l_combine_diag or {}))
    diagnostics.update({f"r_{k}": v for k, v in dict(r_combine_diag or {}).items()})
    diagnostics.update(
        {
            "sub_combine_mode": str(sub_combine_mode),
            "sub_topology": str(candidate.topology or "single_sub_bus"),
            "direct_dac_export_model": "camilladsp_yaml_compatible",
        }
    )
    return BassIntegrationBundle(
        l_main=l_main,
        r_main=r_main,
        l_sub=l_sub,
        r_sub=r_sub,
        l_total=l_total,
        r_total=r_total,
        avr_crossover_hz=float(candidate.main_hpf_hz),
        profile=str(bundle.profile or "safe"),
        diagnostics=diagnostics,
    )


def _channel_score(metrics: dict[str, float], *, profile: str) -> float:
    weights = _auto_bass_integration_profile_weights(profile)
    score = 0.0

    def add(key: str, weight: float, scale: float) -> None:
        nonlocal score
        value = _finite_or(metrics.get(key, float("nan")), float("nan"))
        if np.isfinite(value):
            score += float(weight) * max(0.0, float(value)) / max(float(scale), 1e-9)

    add("cancellation_risk", float(weights.get("cancellation", 8.0)), 1.0)
    add("overlap_ripple_db", float(weights.get("overlap_ripple", 1.8)), 10.0)
    add("sub_dominance_db", float(weights.get("sub_dominance", 0.9)), 8.0)
    add("null_severity", 0.55 * float(weights.get("cancellation", 8.0)), 6.0)
    add("predicted_sum_flatness_db", 0.35 * float(weights.get("overlap_ripple", 1.8)), 8.0)
    add("predicted_sum_dip_depth_db", 0.40 * float(weights.get("cancellation", 8.0)), 6.0)
    add("predicted_sum_peak_excess_db", 0.25 * float(weights.get("overlap_ripple", 1.8)), 6.0)
    add("xo_gd_rms_mismatch_ms", float(weights.get("xo_gd_continuity", 0.8)), 3.0)
    add("xo_gd_max_mismatch_ms", 0.35 * float(weights.get("xo_gd_continuity", 0.8)), 6.0)
    return float(score)


def _reject_reasons_for_channel(metrics: dict[str, float], channel: str, *, fc_hz: float = 80.0) -> tuple[str, ...]:
    prefix = "left" if channel == "left" else "right"
    reasons: list[str] = []
    cancellation_risk = _finite_or(metrics.get("cancellation_risk"), 0.0)
    predicted_dip = _finite_or(metrics.get("predicted_sum_dip_depth_db"), 0.0)
    null_depth = _finite_or(metrics.get("null_depth_db"), 0.0)
    null_width = _finite_or(metrics.get("null_width_hz"), 0.0)
    gd_rms = _finite_or(metrics.get("xo_gd_rms_mismatch_ms"), 0.0)
    gd_max = _finite_or(metrics.get("xo_gd_max_mismatch_ms"), 0.0)
    overlap_ripple = _finite_or(metrics.get("overlap_ripple_db"), 0.0)
    sub_dominance = abs(_finite_or(metrics.get("sub_dominance_db"), 0.0))
    # Null thresholds scale with XO frequency: a 12 dB notch at 40 Hz is far less harmful
    # than the same notch at 150 Hz where hearing sensitivity and comb-filter audibility rise.
    # Scale factor clipped to [0.8, 1.4]: never tighten excessively or loosen dangerously.
    _fc = float(max(_finite_or(fc_hz, 80.0), 1.0))
    _null_scale = float(np.clip(80.0 / _fc, 0.8, 1.4))
    _dip_threshold = 12.0 * _null_scale
    _null_depth_threshold = 12.0 * _null_scale

    if predicted_dip > _dip_threshold:
        reasons.append(f"deep_null_{prefix}")
    if (
        null_depth > _null_depth_threshold
        and null_width >= 5.0
    ):
        reasons.append(f"wide_null_{prefix}")
    if cancellation_risk > 0.70:
        reasons.append(f"high_cancellation_{prefix}")
    if gd_rms > 80.0 and (cancellation_risk > 0.25 or predicted_dip > 8.0):
        reasons.append(f"unsafe_gd_rms_mismatch_{prefix}")
    if gd_max > 180.0 and (cancellation_risk > 0.25 or predicted_dip > 8.0):
        reasons.append(f"unsafe_gd_max_mismatch_{prefix}")
    if overlap_ripple > 32.0 and (cancellation_risk > 0.25 or predicted_dip > 8.0):
        reasons.append(f"unsafe_overlap_ripple_{prefix}")
    if sub_dominance > 24.0 and (cancellation_risk > 0.25 or predicted_dip > 8.0):
        reasons.append(f"unsafe_sub_dominance_{prefix}")
    return tuple(reasons)


def _channel_metrics_obj(
    *,
    channel: str,
    overlap: dict[str, float],
    predicted: dict[str, float],
    extension: dict[str, float],
    gd: dict[str, Any],
    profile: str,
    fc_hz: float = 80.0,
) -> DirectDacChannelMetrics:
    gd_prefix = "l" if channel == "left" else "r"
    values = {
        **dict(overlap or {}),
        **dict(predicted or {}),
        **dict(extension or {}),
        "xo_gd_rms_mismatch_ms": _finite_or(gd.get(f"gd_rms_mismatch_ms_{gd_prefix}", float("nan")), float("nan")),
        "xo_gd_max_mismatch_ms": _finite_or(gd.get(f"gd_max_mismatch_ms_{gd_prefix}", float("nan")), float("nan")),
    }
    score = _channel_score(values, profile=profile)
    reasons = _reject_reasons_for_channel(values, channel, fc_hz=fc_hz)
    if not np.isfinite(score):
        reasons = tuple(dict.fromkeys((*reasons, f"non_finite_score_{channel}")))
        score = float("inf")
    return DirectDacChannelMetrics(
        channel=str(channel),
        score=float(score),
        cancellation_risk=_finite_or(values.get("cancellation_risk"), float("nan")),
        null_depth_db=_finite_or(values.get("null_depth_db"), float("nan")),
        null_width_hz=_finite_or(values.get("null_width_hz"), float("nan")),
        overlap_ripple_db=_finite_or(values.get("overlap_ripple_db"), float("nan")),
        sub_dominance_db=_finite_or(values.get("sub_dominance_db"), float("nan")),
        xo_gd_rms_mismatch_ms=_finite_or(values.get("xo_gd_rms_mismatch_ms"), float("nan")),
        xo_gd_max_mismatch_ms=_finite_or(values.get("xo_gd_max_mismatch_ms"), float("nan")),
        predicted_sum_flatness_db=_finite_or(values.get("predicted_sum_flatness_db"), float("nan")),
        predicted_sum_dip_depth_db=_finite_or(values.get("predicted_sum_dip_depth_db"), float("nan")),
        predicted_sum_peak_excess_db=_finite_or(values.get("predicted_sum_peak_excess_db"), float("nan")),
        feasible=not reasons,
        reject_reasons=tuple(reasons),
    )


def evaluate_direct_dac_candidate(
    bundle: BassIntegrationBundle,
    candidate: DirectDacCandidate,
    *,
    profile: str,
    sub_combine_mode: str = "average",
    guard_lo_ratio: float = AUTO_MODE_BASS_INTEGRATION_GUARD_LO_RATIO,
    guard_hi_ratio: float = AUTO_MODE_BASS_INTEGRATION_GUARD_HI_RATIO,
) -> DirectDacCandidateMetrics:
    """Evaluate the canonical Direct-DAC candidate used by prepare, final metrics, and export verification."""
    combine_mode_norm = normalize_sub_combine_mode(sub_combine_mode)
    fc = max(1.0, _finite_or(candidate.main_hpf_hz, 80.0))
    sub_lpf = max(fc, _finite_or(candidate.sub_lpf_hz, fc))
    lo_ratio = max(0.05, _finite_or(guard_lo_ratio, AUTO_MODE_BASS_INTEGRATION_GUARD_LO_RATIO))
    hi_ratio = max(lo_ratio + 0.05, _finite_or(guard_hi_ratio, AUTO_MODE_BASS_INTEGRATION_GUARD_HI_RATIO))
    guard_lo_hz = max(5.0, fc * lo_ratio)
    guard_hi_hz = max(guard_lo_hz + 1.0, fc * hi_ratio)
    eval_bundle = _branch_bundle(bundle, candidate, sub_combine_mode=combine_mode_norm)

    l_overlap = _channel_overlap_metrics(eval_bundle.l_main, eval_bundle.l_sub, eval_bundle.l_total, lo_hz=guard_lo_hz, hi_hz=guard_hi_hz, fc_hz=fc, sub_lpf_hz=sub_lpf)
    r_overlap = _channel_overlap_metrics(eval_bundle.r_main, eval_bundle.r_sub, eval_bundle.r_total, lo_hz=guard_lo_hz, hi_hz=guard_hi_hz, fc_hz=fc, sub_lpf_hz=sub_lpf)
    l_pred = _channel_predicted_sum_metrics(eval_bundle.l_total, lo_hz=guard_lo_hz, hi_hz=guard_hi_hz, fc_hz=fc)
    r_pred = _channel_predicted_sum_metrics(eval_bundle.r_total, lo_hz=guard_lo_hz, hi_hz=guard_hi_hz, fc_hz=fc)
    l_ext = _channel_overlap_extension_metrics(eval_bundle.l_main, eval_bundle.l_sub, eval_bundle.l_total, fc_hz=fc, sub_lpf_hz=sub_lpf)
    r_ext = _channel_overlap_extension_metrics(eval_bundle.r_main, eval_bundle.r_sub, eval_bundle.r_total, fc_hz=fc, sub_lpf_hz=sub_lpf)
    gd_cont = compute_xo_gd_continuity(eval_bundle, fc, sub_combine_mode=combine_mode_norm)

    left = _channel_metrics_obj(channel="left", overlap=l_overlap, predicted=l_pred, extension=l_ext, gd=gd_cont, profile=profile, fc_hz=fc)
    right = _channel_metrics_obj(channel="right", overlap=r_overlap, predicted=r_pred, extension=r_ext, gd=gd_cont, profile=profile, fc_hz=fc)
    worst = left if left.score >= right.score else right
    dominant_channel = "left" if worst is left else "right"
    if np.isfinite(left.score) and np.isfinite(right.score) and abs(left.score - right.score) < 0.05:
        dominant_channel = "balanced"

    reject_reasons = list(left.reject_reasons + right.reject_reasons)
    if float(candidate.sub_gain_trim_db) > 12.0:
        reject_reasons.append("sub_gain_trim_above_safe_range")
    if float(candidate.sub_gain_trim_db) < -15.0:
        reject_reasons.append("sub_gain_trim_below_safe_range")
    if not np.isfinite(worst.score):
        reject_reasons.append("non_finite_score")

    asymmetry_penalty = min(3.0, 0.20 * abs(float(left.score) - float(right.score))) if np.isfinite(left.score + right.score) else 100.0
    delay_penalty = 0.025 * abs(float(candidate.sub_delay_ms))
    gain_penalty = 0.035 * abs(float(candidate.sub_gain_trim_db))
    polarity_penalty = 0.03 if bool(candidate.sub_polarity_invert) else 0.0
    xo_penalty = 0.01 * abs(float(fc) - 80.0) / 10.0
    allpass_penalty = 0.08 if bool(candidate.sub_allpass_enabled) else 0.0
    score = float(worst.score + asymmetry_penalty + delay_penalty + gain_penalty + polarity_penalty + xo_penalty + allpass_penalty)
    # Safety rejects (gain out of range, non-finite) → hard +1000: must never be selected.
    # Acoustic-quality rejects → continuous per-reason penalty so the optimizer can still
    # rank bad candidates against each other when all candidates have quality issues.
    _unique_reasons = list(dict.fromkeys(reject_reasons))
    _safety = [r for r in _unique_reasons if r in _SAFETY_REJECT_REASONS]
    _acoustic = [r for r in _unique_reasons if r not in _SAFETY_REJECT_REASONS]
    if _safety:
        score += 1000.0
    if _acoustic:
        score += _ACOUSTIC_REJECT_PENALTY * len(_acoustic)

    overlap_ripple_delta = _metric_delta(left.overlap_ripple_db, right.overlap_ripple_db)
    sub_dominance_delta = _metric_delta(left.sub_dominance_db, right.sub_dominance_db)
    gd_delta = _metric_delta(left.xo_gd_rms_mismatch_ms, right.xo_gd_rms_mismatch_ms)
    # Classify feasibility using the worst channel's own metrics, not a synthetic cross-channel
    # max that combines ripple from one channel with GD from the other.  Both channels' reject
    # reasons are still collected above, so genuinely bad channels are not silently passed.
    feasibility_class, feasibility_reason, feasibility_limiting_factor = _classify_bass_integration_feasibility(
        overlap_ripple_worst=worst.overlap_ripple_db,
        sub_dominance_worst=abs(worst.sub_dominance_db),
        xo_gd_rms_worst=worst.xo_gd_rms_mismatch_ms,
        overlap_ripple_delta=overlap_ripple_delta,
        sub_dominance_delta=sub_dominance_delta,
        xo_gd_delta=gd_delta,
        dominant_channel=dominant_channel,
        sub_combine_mode=combine_mode_norm,
        sub_level_delta_db_20_120=dict(getattr(eval_bundle, "diagnostics", {}) or {}).get("sub_combined_level_delta_db_20_120", float("nan")),
        fc_hz=fc,
    )
    if reject_reasons:
        reject_note = " Direct-DAC candidate safety notes: " + ", ".join(tuple(dict.fromkeys(reject_reasons))) + "."
        feasibility_reason = (str(feasibility_reason or "").rstrip(".") + "." + reject_note).strip()

    summary = {
        "profile": _auto_bass_integration_profile_norm(profile),
        "avr_crossover_hz": float(fc),
        "sub_lpf_hz": float(sub_lpf),
        "sub_combine_mode": str(combine_mode_norm),
        "sub_combined_level_delta_db_20_120": _finite_or(
            dict(getattr(eval_bundle, "diagnostics", {}) or {}).get("sub_combined_level_delta_db_20_120"),
            float("nan"),
        ),
        "sub_combined_level_delta_db_30_90": _finite_or(
            dict(getattr(eval_bundle, "diagnostics", {}) or {}).get("sub_combined_level_delta_db_30_90"),
            float("nan"),
        ),
        "metric_channel_mode": "worst_case",
        "guard_lo_ratio": float(lo_ratio),
        "guard_hi_ratio": float(hi_ratio),
        "guard_lo_hz": float(guard_lo_hz),
        "guard_hi_hz": float(guard_hi_hz),
        "cancellation_risk": max(left.cancellation_risk, right.cancellation_risk),
        "overlap_ripple_db": max(left.overlap_ripple_db, right.overlap_ripple_db),
        "sub_dominance_db": max(abs(left.sub_dominance_db), abs(right.sub_dominance_db)),
        "null_severity": max(
            _finite_or(l_pred.get("null_severity"), float("nan")),
            _finite_or(r_pred.get("null_severity"), float("nan")),
        ),
        "predicted_sum_flatness_db": max(left.predicted_sum_flatness_db, right.predicted_sum_flatness_db),
        "predicted_sum_dip_depth_db": max(left.predicted_sum_dip_depth_db, right.predicted_sum_dip_depth_db),
        "predicted_sum_peak_excess_db": max(left.predicted_sum_peak_excess_db, right.predicted_sum_peak_excess_db),
        "null_depth_db": max(left.null_depth_db, right.null_depth_db),
        "null_width_hz": max(left.null_width_hz, right.null_width_hz),
        "overlap_ripple_delta_db": float(overlap_ripple_delta),
        "sub_dominance_delta_db": float(sub_dominance_delta),
        "xo_gd_mismatch_delta_ms": float(gd_delta),
        "dominant_channel": str(dominant_channel),
        "feasibility_class": str(feasibility_class),
        "feasibility_reason": str(feasibility_reason),
        "feasibility_limiting_factor": str(feasibility_limiting_factor),
        "export_model": "camilladsp_yaml_compatible",
        "candidate_score": float(score),
        "objective": float(-score),
        "candidate_feasible": not tuple(dict.fromkeys(reject_reasons)),
        "reject_reasons": tuple(dict.fromkeys(reject_reasons)),
        "left": left,
        "right": right,
        "eval_bundle": eval_bundle,
        "gd_continuity": dict(gd_cont or {}),
        "channels": {"l": {**l_overlap, **l_pred}, "r": {**r_overlap, **r_pred}},
        "channels_extension": {"l": dict(l_ext), "r": dict(r_ext)},
    }
    for key in ("overlap_extension_flatness_db", "overlap_extension_cancellation_risk", "overlap_extension_peak_excess_db", "overlap_extension_sub_dominance_db"):
        summary[key] = max(_finite_or(l_ext.get(key), float("nan")), _finite_or(r_ext.get(key), float("nan")))
        summary[f"{key}_l"] = _finite_or(l_ext.get(key), float("nan"))
        summary[f"{key}_r"] = _finite_or(r_ext.get(key), float("nan"))
    summary["overlap_extension_active"] = bool(l_ext.get("overlap_extension_active", False) or r_ext.get("overlap_extension_active", False))

    feasible = not tuple(dict.fromkeys(reject_reasons))
    return DirectDacCandidateMetrics(
        candidate=candidate,
        score=float(score),
        objective=float(-score),
        applied=bool(feasible),
        feasible=bool(feasible),
        reject_reasons=tuple(dict.fromkeys(reject_reasons)),
        left=left,
        right=right,
        dominant_channel=str(dominant_channel),
        summary=summary,
    )
