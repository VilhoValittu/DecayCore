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

import dataclasses
from typing import Any

import numpy as np

from ...features import require_packaged_bass_engine
from ...auto_mode.shared_parts import (
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
from ._realized_response import (
    ROBUST_DELAY_OFFSETS_MS,
    ROBUST_GAIN_OFFSETS_DB,
    ROBUST_NOMINAL_WEIGHT,
    ROBUST_P90_PERCENTILE,
    ROBUST_PERTURBATION_POLICY_VERSION,
    build_realized_bass_integration_bundle,
)
from ._sub_combine import build_bundle_combined_sub_transfer
from ._utils import (
    _build_transfer_like,
    _interp_complex_response,
    _safe_float,
    normalize_sub_combine_mode,
)

try:
    from decaycore_bass_engine import (
        robust_score_rs as _robust_score_rs,
        transform_sub_and_sum_rs as _transform_sub_and_sum_rs,
    )
except ImportError:
    _robust_score_rs = None
    _transform_sub_and_sum_rs = None


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


def _finite_percentile(values: np.ndarray, percentile: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.percentile(finite, percentile)) if finite.size else float("nan")


def _finite_max(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.max(finite)) if finite.size else float("nan")


def _evaluation_band_bundle(
    bundle: BassIntegrationBundle,
    *,
    fc_hz: float,
    sub_lpf_hz: float,
    guard_lo_ratio: float,
    guard_hi_ratio: float,
) -> BassIntegrationBundle:
    """Return a cached low-frequency view containing every bass metric band plus GD padding."""
    freqs = np.asarray(bundle.l_main.freqs_hz, dtype=float).reshape(-1)
    if freqs.size < 32:
        return bundle
    fc = max(1.0, float(fc_hz))
    lo_hz = max(0.0, min(20.0, fc * float(guard_lo_ratio), fc * 0.90))
    hi_hz = max(120.0, float(sub_lpf_hz), fc * float(guard_hi_ratio), fc * 1.30)
    left = max(0, int(np.searchsorted(freqs, lo_hz, side="left")) - 8)
    right = min(freqs.size, int(np.searchsorted(freqs, hi_hz, side="right")) + 8)
    if left == 0 and right == freqs.size:
        return bundle
    if right - left < 16:
        center = int(np.searchsorted(freqs, fc, side="left"))
        left = max(0, center - 8)
        right = min(freqs.size, center + 8)
    cache_key = (left, right)
    try:
        cache = object.__getattribute__(bundle, "_decaycore_bass_band_view_cache")
    except AttributeError:
        cache = {}
        object.__setattr__(bundle, "_decaycore_bass_band_view_cache", cache)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    def sliced(transfer, label: str):
        transfer_freqs = np.asarray(transfer.freqs_hz, dtype=float).reshape(-1)
        if transfer_freqs.size != freqs.size or not np.array_equal(transfer_freqs, freqs):
            return transfer
        spec = np.asarray(transfer.complex_spec, dtype=np.complex128).reshape(-1)
        if spec.size != freqs.size:
            return transfer
        template = type(transfer)(
            freqs_hz=freqs[left:right],
            complex_spec=spec[left:right],
            mag_db=np.asarray(transfer.mag_db, dtype=float).reshape(-1)[left:right],
            phase_deg=np.asarray(transfer.phase_deg, dtype=float).reshape(-1)[left:right],
            sample_rate=int(transfer.sample_rate),
            label=label,
        )
        return _build_transfer_like(template, spec[left:right], label=label)

    view = BassIntegrationBundle(
        l_main=sliced(bundle.l_main, "L main bass evaluation band"),
        r_main=sliced(bundle.r_main, "R main bass evaluation band"),
        l_sub=sliced(bundle.l_sub, "L sub bass evaluation band"),
        r_sub=sliced(bundle.r_sub, "R sub bass evaluation band"),
        l_total=sliced(bundle.l_total, "L total bass evaluation band"),
        r_total=sliced(bundle.r_total, "R total bass evaluation band"),
        avr_crossover_hz=float(bundle.avr_crossover_hz),
        profile=str(bundle.profile),
        diagnostics=dict(bundle.diagnostics or {}),
    )
    if len(cache) >= 32:
        cache.clear()
    cache[cache_key] = view
    return view


def _perturbed_filtered_bundles(
    nominal_bundle: BassIntegrationBundle,
    *,
    gain_offsets_db: np.ndarray,
    delay_offsets_ms: np.ndarray,
) -> list[BassIntegrationBundle]:
    """Build all robust branch responses with one scenario-axis broadcast."""
    gains = np.asarray(gain_offsets_db, dtype=float).reshape(-1)
    delays = np.asarray(delay_offsets_ms, dtype=float).reshape(-1)
    if gains.shape != delays.shape:
        raise ValueError("Bass perturbation gain and delay arrays must have identical shapes")
    l_freqs = np.asarray(nominal_bundle.l_sub.freqs_hz, dtype=float).reshape(-1)
    r_freqs = np.asarray(nominal_bundle.r_sub.freqs_hz, dtype=float).reshape(-1)
    if l_freqs.shape != r_freqs.shape or not np.array_equal(l_freqs, r_freqs):
        raise ValueError("Bass perturbation channels must share a frequency grid")
    gain_linear = np.power(10.0, gains / 20.0)[:, None]
    delay_phase = np.exp(
        -1j * 2.0 * np.pi * l_freqs[None, :] * (delays[:, None] / 1000.0)
    )
    transform = np.asarray(gain_linear * delay_phase, dtype=np.complex128)
    l_sub_specs = np.asarray(nominal_bundle.l_sub.complex_spec, dtype=np.complex128)[None, :] * transform
    r_sub_specs = np.asarray(nominal_bundle.r_sub.complex_spec, dtype=np.complex128)[None, :] * transform
    l_main_spec = np.asarray(nominal_bundle.l_main.complex_spec, dtype=np.complex128)
    r_main_spec = np.asarray(nominal_bundle.r_main.complex_spec, dtype=np.complex128)
    l_total_specs = l_main_spec[None, :] + l_sub_specs
    r_total_specs = r_main_spec[None, :] + r_sub_specs
    out: list[BassIntegrationBundle] = []
    for idx in range(gains.size):
        l_sub = _build_transfer_like(
            nominal_bundle.l_sub,
            l_sub_specs[idx],
            label="L robust perturbed sub branch",
        )
        r_sub = _build_transfer_like(
            nominal_bundle.r_sub,
            r_sub_specs[idx],
            label="R robust perturbed sub branch",
        )
        out.append(
            BassIntegrationBundle(
                l_main=nominal_bundle.l_main,
                r_main=nominal_bundle.r_main,
                l_sub=l_sub,
                r_sub=r_sub,
                l_total=_build_transfer_like(
                    nominal_bundle.l_total,
                    l_total_specs[idx],
                    label="L robust perturbed total",
                ),
                r_total=_build_transfer_like(
                    nominal_bundle.r_total,
                    r_total_specs[idx],
                    label="R robust perturbed total",
                ),
                avr_crossover_hz=float(nominal_bundle.avr_crossover_hz),
                profile=str(nominal_bundle.profile),
                diagnostics=dict(nominal_bundle.diagnostics or {}),
            )
        )
    return out


def _branch_bundle(
    bundle: BassIntegrationBundle,
    candidate: DirectDacCandidate,
    *,
    sub_combine_mode: str,
) -> BassIntegrationBundle:
    allpass_freq = float(candidate.sub_allpass_freq_hz) if bool(candidate.sub_allpass_enabled) else None
    allpass_q = float(candidate.sub_allpass_q) if bool(candidate.sub_allpass_enabled) else None
    base_key = (
        round(float(candidate.main_hpf_hz), 4),
        int(candidate.main_hpf_order),
        round(float(candidate.sub_hpf_hz), 4),
        int(candidate.sub_hpf_order),
        round(float(candidate.sub_lpf_hz), 4),
        int(candidate.sub_lpf_order),
        bool(candidate.sub_polarity_invert),
        round(float(allpass_freq), 4) if allpass_freq is not None else None,
        round(float(allpass_q), 5) if allpass_q is not None else None,
        str(sub_combine_mode),
    )
    try:
        base_cache = object.__getattribute__(bundle, "_decaycore_direct_dac_branch_base_cache")
    except AttributeError:
        base_cache = {}
        object.__setattr__(bundle, "_decaycore_direct_dac_branch_base_cache", base_cache)
    cached = base_cache.get(base_key)
    if cached is None:
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
        l_sub_base = apply_direct_dac_export_branch_model(
            l_sub_raw,
            hpf_hz=float(candidate.sub_hpf_hz),
            hpf_order=int(candidate.sub_hpf_order),
            lpf_hz=float(candidate.sub_lpf_hz),
            lpf_order=int(candidate.sub_lpf_order),
            polarity_invert=bool(candidate.sub_polarity_invert),
            allpass_freq_hz=allpass_freq,
            allpass_q=allpass_q,
            label="L sub CamillaDSP Direct-DAC filtered base",
        )
        r_sub_base = apply_direct_dac_export_branch_model(
            r_sub_raw,
            hpf_hz=float(candidate.sub_hpf_hz),
            hpf_order=int(candidate.sub_hpf_order),
            lpf_hz=float(candidate.sub_lpf_hz),
            lpf_order=int(candidate.sub_lpf_order),
            polarity_invert=bool(candidate.sub_polarity_invert),
            allpass_freq_hz=allpass_freq,
            allpass_q=allpass_q,
            label="R sub CamillaDSP Direct-DAC filtered base",
        )
        cached = (l_main, r_main, l_sub_base, r_sub_base, l_combine_diag, r_combine_diag)
        if len(base_cache) >= 64:
            base_cache.clear()
        base_cache[base_key] = cached
    l_main, r_main, l_sub_base, r_sub_base, l_combine_diag, r_combine_diag = cached
    if _transform_sub_and_sum_rs is None:
        require_packaged_bass_engine()
        raise RuntimeError("Packaged bass engine did not expose transform_sub_and_sum_rs")

    def _transform_channel(main, sub_base, *, sub_label: str, total_label: str):
        freqs = np.asarray(main.freqs_hz, dtype=np.float64)
        main_spec = _interp_complex_response(main, freqs)
        sub_spec = _interp_complex_response(sub_base, freqs)
        # Polarity is already folded into sub_base by the export branch model, so
        # the engine must not invert again.
        transformed, total = _transform_sub_and_sum_rs(
            np.ascontiguousarray(freqs, dtype=np.float64),
            np.ascontiguousarray(main_spec, dtype=np.complex128),
            np.ascontiguousarray(sub_spec, dtype=np.complex128),
            float(candidate.sub_gain_trim_db),
            float(candidate.sub_delay_ms),
        )
        return (
            _build_transfer_like(main, np.asarray(transformed), label=sub_label),
            _build_transfer_like(main, np.asarray(total), label=total_label),
        )

    l_sub, l_total = _transform_channel(
        l_main,
        l_sub_base,
        sub_label="L sub CamillaDSP Direct-DAC branch",
        total_label="L Direct-DAC canonical predicted total",
    )
    r_sub, r_total = _transform_channel(
        r_main,
        r_sub_base,
        sub_label="R sub CamillaDSP Direct-DAC branch",
        total_label="R Direct-DAC canonical predicted total",
    )
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


def _evaluate_direct_dac_candidate_nominal(
    bundle: BassIntegrationBundle,
    candidate: DirectDacCandidate,
    *,
    profile: str,
    sub_combine_mode: str = "average",
    guard_lo_ratio: float = AUTO_MODE_BASS_INTEGRATION_GUARD_LO_RATIO,
    guard_hi_ratio: float = AUTO_MODE_BASS_INTEGRATION_GUARD_HI_RATIO,
    gd_continuity: dict[str, Any] | None = None,
    filtered_bundle: BassIntegrationBundle | None = None,
) -> DirectDacCandidateMetrics:
    """Evaluate the canonical Direct-DAC candidate used by prepare, final metrics, and export verification."""
    combine_mode_norm = normalize_sub_combine_mode(sub_combine_mode)
    fc = max(1.0, _finite_or(candidate.main_hpf_hz, 80.0))
    sub_lpf = max(fc, _finite_or(candidate.sub_lpf_hz, fc))
    lo_ratio = max(0.05, _finite_or(guard_lo_ratio, AUTO_MODE_BASS_INTEGRATION_GUARD_LO_RATIO))
    hi_ratio = max(lo_ratio + 0.05, _finite_or(guard_hi_ratio, AUTO_MODE_BASS_INTEGRATION_GUARD_HI_RATIO))
    guard_lo_hz = max(5.0, fc * lo_ratio)
    guard_hi_hz = max(guard_lo_hz + 1.0, fc * hi_ratio)
    if filtered_bundle is None:
        bundle = _evaluation_band_bundle(
            bundle,
            fc_hz=fc,
            sub_lpf_hz=sub_lpf,
            guard_lo_ratio=lo_ratio,
            guard_hi_ratio=hi_ratio,
        )
        eval_bundle = _branch_bundle(bundle, candidate, sub_combine_mode=combine_mode_norm)
    else:
        eval_bundle = filtered_bundle

    l_overlap = _channel_overlap_metrics(eval_bundle.l_main, eval_bundle.l_sub, eval_bundle.l_total, lo_hz=guard_lo_hz, hi_hz=guard_hi_hz, fc_hz=fc, sub_lpf_hz=sub_lpf)
    r_overlap = _channel_overlap_metrics(eval_bundle.r_main, eval_bundle.r_sub, eval_bundle.r_total, lo_hz=guard_lo_hz, hi_hz=guard_hi_hz, fc_hz=fc, sub_lpf_hz=sub_lpf)
    l_pred = _channel_predicted_sum_metrics(eval_bundle.l_total, lo_hz=guard_lo_hz, hi_hz=guard_hi_hz, fc_hz=fc)
    r_pred = _channel_predicted_sum_metrics(eval_bundle.r_total, lo_hz=guard_lo_hz, hi_hz=guard_hi_hz, fc_hz=fc)
    l_ext = _channel_overlap_extension_metrics(eval_bundle.l_main, eval_bundle.l_sub, eval_bundle.l_total, fc_hz=fc, sub_lpf_hz=sub_lpf)
    r_ext = _channel_overlap_extension_metrics(eval_bundle.r_main, eval_bundle.r_sub, eval_bundle.r_total, fc_hz=fc, sub_lpf_hz=sub_lpf)
    gd_cont = (
        dict(gd_continuity)
        if gd_continuity is not None
        else compute_xo_gd_continuity(eval_bundle, fc, sub_combine_mode=combine_mode_norm)
    )

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


def evaluate_direct_dac_candidate(
    bundle: BassIntegrationBundle,
    candidate: DirectDacCandidate,
    *,
    profile: str,
    sub_combine_mode: str = "average",
    guard_lo_ratio: float = AUTO_MODE_BASS_INTEGRATION_GUARD_LO_RATIO,
    guard_hi_ratio: float = AUTO_MODE_BASS_INTEGRATION_GUARD_HI_RATIO,
    l_fir: Any | None = None,
    r_fir: Any | None = None,
    sub_fir: Any | None = None,
    fir_sample_rate: int | None = None,
    robust: bool | None = None,
) -> DirectDacCandidateMetrics:
    """Evaluate the exported chain, with deterministic acoustic perturbations when FIRs exist."""
    require_packaged_bass_engine()
    has_realized_firs = any(value is not None for value in (l_fir, r_fir, sub_fir))
    eval_bundle = bundle
    if has_realized_firs:
        if any(value is None for value in (l_fir, r_fir, sub_fir)):
            raise ValueError("Realized bass integration evaluation requires L, R, and sub FIRs")
        fc = max(1.0, _finite_or(candidate.main_hpf_hz, 80.0))
        sub_lpf = max(fc, _finite_or(candidate.sub_lpf_hz, fc))
        lo_ratio = max(0.05, _finite_or(guard_lo_ratio, AUTO_MODE_BASS_INTEGRATION_GUARD_LO_RATIO))
        hi_ratio = max(
            lo_ratio + 0.05,
            _finite_or(guard_hi_ratio, AUTO_MODE_BASS_INTEGRATION_GUARD_HI_RATIO),
        )
        source_bundle = _evaluation_band_bundle(
            bundle,
            fc_hz=fc,
            sub_lpf_hz=sub_lpf,
            guard_lo_ratio=lo_ratio,
            guard_hi_ratio=hi_ratio,
        )
        eval_bundle = build_realized_bass_integration_bundle(
            source_bundle,
            l_fir=l_fir,
            r_fir=r_fir,
            sub_fir=sub_fir,
            sample_rate=fir_sample_rate,
            sub_combine_mode=sub_combine_mode,
        )
    nominal = _evaluate_direct_dac_candidate_nominal(
        eval_bundle,
        candidate,
        profile=profile,
        sub_combine_mode=sub_combine_mode,
        guard_lo_ratio=guard_lo_ratio,
        guard_hi_ratio=guard_hi_ratio,
    )
    use_robust = bool(has_realized_firs if robust is None else robust)
    if not use_robust:
        return nominal

    gain_grid, delay_grid = np.meshgrid(
        np.asarray(ROBUST_GAIN_OFFSETS_DB, dtype=float),
        np.asarray(ROBUST_DELAY_OFFSETS_MS, dtype=float),
        indexing="ij",
    )
    gain_offsets = gain_grid.reshape(-1)
    delay_offsets = delay_grid.reshape(-1)
    nominal_filtered_bundle = nominal.summary.get("eval_bundle")
    if not isinstance(nominal_filtered_bundle, BassIntegrationBundle):
        raise RuntimeError("Nominal bass evaluation did not expose its filtered branch bundle")
    filtered_scenarios = _perturbed_filtered_bundles(
        nominal_filtered_bundle,
        gain_offsets_db=gain_offsets,
        delay_offsets_ms=delay_offsets,
    )
    scenario_results: list[DirectDacCandidateMetrics] = [nominal]
    gd_by_delay_offset: dict[float, dict[str, Any]] = {
        0.0: dict(nominal.summary.get("gd_continuity", {}) or {})
    }
    for gain_offset, delay_offset, filtered_scenario in zip(
        gain_offsets,
        delay_offsets,
        filtered_scenarios,
        strict=True,
    ):
        gain_offset = float(gain_offset)
        delay_offset = float(delay_offset)
        if abs(gain_offset) <= 1e-12 and abs(delay_offset) <= 1e-12:
            continue
        scenario_candidate = dataclasses.replace(
            candidate,
            sub_gain_trim_db=float(candidate.sub_gain_trim_db) + gain_offset,
            sub_delay_ms=float(candidate.sub_delay_ms) + delay_offset,
            source=f"{candidate.source}:robust_v{ROBUST_PERTURBATION_POLICY_VERSION}",
        )
        scenario = _evaluate_direct_dac_candidate_nominal(
            eval_bundle,
            scenario_candidate,
            profile=profile,
            sub_combine_mode=sub_combine_mode,
            guard_lo_ratio=guard_lo_ratio,
            guard_hi_ratio=guard_hi_ratio,
            gd_continuity=gd_by_delay_offset.get(delay_offset),
            filtered_bundle=filtered_scenario,
        )
        gd_by_delay_offset.setdefault(
            delay_offset,
            dict(scenario.summary.get("gd_continuity", {}) or {}),
        )
        scenario_results.append(scenario)

    scores = np.asarray([result.score for result in scenario_results], dtype=float)
    if _robust_score_rs is None:
        raise RuntimeError("Packaged bass engine did not expose robust_score_rs")
    p90_score, robust_score = _robust_score_rs(
        float(nominal.score),
        np.ascontiguousarray(scores, dtype=np.float64),
        float(ROBUST_NOMINAL_WEIGHT),
        float(ROBUST_P90_PERCENTILE),
    )
    cancellation = np.asarray(
        [result.summary.get("cancellation_risk", float("nan")) for result in scenario_results],
        dtype=float,
    )
    ripple = np.asarray(
        [result.summary.get("overlap_ripple_db", float("nan")) for result in scenario_results],
        dtype=float,
    )
    scenario_rejects = tuple(
        dict.fromkeys(reason for result in scenario_results for reason in result.reject_reasons)
    )
    feasible = bool(nominal.feasible and all(result.feasible for result in scenario_results))
    reject_reasons = tuple(nominal.reject_reasons)
    if not feasible and not reject_reasons:
        reject_reasons = tuple(f"robust_{reason}" for reason in scenario_rejects) or (
            "robust_perturbation_infeasible",
        )
    summary = dict(nominal.summary or {})
    summary.update(
        {
            "realized_response": bool(has_realized_firs),
            "export_verification_match": bool(has_realized_firs),
            "robust_perturbation_policy": "gain_±1db_delay_±0.5ms_cartesian",
            "robust_perturbation_policy_v": int(ROBUST_PERTURBATION_POLICY_VERSION),
            "robust_scenario_count": int(len(scenario_results)),
            "robust_nominal_score": float(nominal.score),
            "robust_p90_score": float(p90_score),
            "robust_score": float(robust_score),
            "robust_feasible": bool(feasible),
            "robust_cancellation_risk_p90": _finite_percentile(cancellation, ROBUST_P90_PERCENTILE),
            "robust_cancellation_risk_worst": _finite_max(cancellation),
            "robust_overlap_ripple_db_p90": _finite_percentile(ripple, ROBUST_P90_PERCENTILE),
            "robust_overlap_ripple_db_worst": _finite_max(ripple),
            "sub_scaling_assumption": "single_bus_average_normalized",
            "sub_coherence_assumption": "measured_complex_with_bounded_gain_delay_perturbation",
        }
    )
    return DirectDacCandidateMetrics(
        candidate=candidate,
        score=float(robust_score),
        objective=float(-robust_score),
        applied=bool(feasible),
        feasible=bool(feasible),
        reject_reasons=tuple(dict.fromkeys(reject_reasons)),
        left=nominal.left,
        right=nominal.right,
        dominant_channel=str(nominal.dominant_channel),
        summary=summary,
    )
