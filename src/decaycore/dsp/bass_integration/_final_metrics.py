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

from ...auto_mode.shared_parts import (
    AUTO_MODE_BASS_INTEGRATION_GUARD_HI_RATIO,
    AUTO_MODE_BASS_INTEGRATION_GUARD_LO_RATIO,
    _auto_bass_integration_profile_norm,
)
from ...io.measurement_bundle import BassIntegrationBundle
from ..bass_cache import (
    _get_metrics_cache,
    _metrics_cache_key,
    increment_metrics_cache_hit,
    increment_metrics_cache_miss,
)
from ._candidate import DirectDacCandidate
from ._evaluate_direct_dac import evaluate_direct_dac_candidate
from ._realized_response import realized_fir_signature
from ._utils import _safe_float, normalize_sub_combine_mode


def _direct_dac_eval_to_legacy_metrics(eval_result, *, profile: str, mode_norm: str, combine_mode_norm: str, allpass_enabled: bool, ap_freq_hz: float, ap_q: float) -> dict[str, Any]:
    s = dict(eval_result.summary or {})
    gd_cont = dict(s.get("gd_continuity", {}) or {})
    left = eval_result.left
    right = eval_result.right
    dominant = str(eval_result.dominant_channel)
    metrics = {
        "bass_cancellation_risk": _safe_float(
            s.get("robust_cancellation_risk_p90", s.get("cancellation_risk", float("nan"))),
            float("nan"),
        ),
        "bass_overlap_ripple": _safe_float(
            s.get("robust_overlap_ripple_db_p90", s.get("overlap_ripple_db", float("nan"))),
            float("nan"),
        ),
        "bass_sub_dominance": _safe_float(s.get("sub_dominance_db", float("nan")), float("nan")),
        "bass_null_severity": _safe_float(s.get("null_severity", float("nan")), float("nan")),
        "bass_predicted_sum_flatness_db": _safe_float(s.get("predicted_sum_flatness_db", float("nan")), float("nan")),
        "bass_predicted_sum_dip_depth_db": _safe_float(s.get("predicted_sum_dip_depth_db", float("nan")), float("nan")),
        "bass_predicted_sum_peak_excess_db": _safe_float(s.get("predicted_sum_peak_excess_db", float("nan")), float("nan")),
        "bass_overlap_extension_active": bool(s.get("overlap_extension_active", False)),
        "bass_overlap_extension_flatness_db": _safe_float(s.get("overlap_extension_flatness_db", float("nan")), float("nan")),
        "bass_overlap_extension_cancellation_risk": _safe_float(s.get("overlap_extension_cancellation_risk", float("nan")), float("nan")),
        "bass_overlap_extension_peak_excess_db": _safe_float(s.get("overlap_extension_peak_excess_db", float("nan")), float("nan")),
        "bass_overlap_extension_sub_dominance_db": _safe_float(s.get("overlap_extension_sub_dominance_db", float("nan")), float("nan")),
        "bass_guard_lo_hz": _safe_float(s.get("guard_lo_hz", float("nan")), float("nan")),
        "bass_guard_hi_hz": _safe_float(s.get("guard_hi_hz", float("nan")), float("nan")),
        "bass_integration_profile": str(_auto_bass_integration_profile_norm(profile)),
        "bass_integration_mode": str(mode_norm),
        "bass_metric_channel_mode": "worst_case",
        "bass_xo_gd_mismatch_ms": _safe_float(gd_cont.get("gd_rms_mismatch_ms_worst", float("nan")), float("nan")),
        "bass_xo_gd_rms_mismatch_ms": _safe_float(gd_cont.get("gd_rms_mismatch_ms_worst", float("nan")), float("nan")),
        "bass_xo_gd_max_mismatch_ms": _safe_float(gd_cont.get("gd_max_mismatch_ms_worst", float("nan")), float("nan")),
        "bass_xo_l_gd_mismatch_ms": float(left.xo_gd_rms_mismatch_ms),
        "bass_xo_r_gd_mismatch_ms": float(right.xo_gd_rms_mismatch_ms),
        "bass_xo_main_gd_ms": _safe_float((gd_cont.get("l_main_gd_ms", float("nan")) + gd_cont.get("r_main_gd_ms", float("nan"))) / 2.0, float("nan")),
        "bass_xo_sub_gd_ms": _safe_float(gd_cont.get("sub_gd_ms", float("nan")), float("nan")),
        "bass_allpass_enabled": bool(allpass_enabled),
        "bass_allpass_freq_hz": float(ap_freq_hz) if allpass_enabled else 0.0,
        "bass_allpass_q": float(ap_q) if allpass_enabled else 0.707,
        "bass_sub_combine_mode": str(combine_mode_norm),
        "bass_sub_combined_level_delta_db_20_120": _safe_float(
            s.get("sub_combined_level_delta_db_20_120", float("nan")),
            float("nan"),
        ),
        "bass_sub_combined_level_delta_db_30_90": _safe_float(
            s.get("sub_combined_level_delta_db_30_90", float("nan")),
            float("nan"),
        ),
        "bass_cancellation_risk_l": float(left.cancellation_risk),
        "bass_cancellation_risk_r": float(right.cancellation_risk),
        "bass_cancellation_risk_worst": max(float(left.cancellation_risk), float(right.cancellation_risk)),
        "bass_overlap_ripple_l": float(left.overlap_ripple_db),
        "bass_overlap_ripple_r": float(right.overlap_ripple_db),
        "bass_overlap_ripple_worst": max(float(left.overlap_ripple_db), float(right.overlap_ripple_db)),
        "bass_sub_dominance_l": float(left.sub_dominance_db),
        "bass_sub_dominance_r": float(right.sub_dominance_db),
        "bass_sub_dominance_worst": max(abs(float(left.sub_dominance_db)), abs(float(right.sub_dominance_db))),
        "bass_null_severity_l": _safe_float(dict(s.get("channels", {}).get("l", {}) or {}).get("null_severity", float("nan")), float("nan")),
        "bass_null_severity_r": _safe_float(dict(s.get("channels", {}).get("r", {}) or {}).get("null_severity", float("nan")), float("nan")),
        "bass_null_severity_worst": _safe_float(s.get("null_severity", float("nan")), float("nan")),
        "bass_overlap_ripple_delta_db": _safe_float(s.get("overlap_ripple_delta_db", float("nan")), float("nan")),
        "bass_sub_dominance_delta_db": _safe_float(s.get("sub_dominance_delta_db", float("nan")), float("nan")),
        "bass_xo_gd_mismatch_delta_ms": _safe_float(s.get("xo_gd_mismatch_delta_ms", float("nan")), float("nan")),
        "bass_dominant_channel": str(dominant),
        "bass_feasibility_class": str(s.get("feasibility_class", "infeasible" if not eval_result.feasible else "good")),
        "bass_feasibility_reason": str(s.get("feasibility_reason", "")),
        "bass_feasibility_limiting_factor": str(s.get("feasibility_limiting_factor", "unknown")),
        "bass_direct_dac_candidate_score": float(eval_result.score),
        "bass_direct_dac_objective": float(eval_result.objective),
        "bass_direct_dac_reject_reasons": list(eval_result.reject_reasons),
        "bass_direct_dac_left_score": float(left.score),
        "bass_direct_dac_right_score": float(right.score),
        "bass_direct_dac_worst_channel": str(dominant),
        "bass_direct_dac_export_model": "camilladsp_yaml_compatible",
        "bass_realized_response": bool(s.get("realized_response", False)),
        "bass_export_verification_match": bool(s.get("export_verification_match", False)),
        "bass_robust_perturbation_policy": str(s.get("robust_perturbation_policy", "nominal_only")),
        "bass_robust_perturbation_policy_v": int(s.get("robust_perturbation_policy_v", 0) or 0),
        "bass_robust_scenario_count": int(s.get("robust_scenario_count", 1) or 1),
        "bass_robust_nominal_score": _safe_float(s.get("robust_nominal_score", eval_result.score), eval_result.score),
        "bass_robust_p90_score": _safe_float(s.get("robust_p90_score", eval_result.score), eval_result.score),
        "bass_robust_score": _safe_float(s.get("robust_score", eval_result.score), eval_result.score),
        "bass_robust_cancellation_risk_worst": _safe_float(
            s.get("robust_cancellation_risk_worst", s.get("cancellation_risk", float("nan"))),
            float("nan"),
        ),
        "bass_robust_overlap_ripple_db_worst": _safe_float(
            s.get("robust_overlap_ripple_db_worst", s.get("overlap_ripple_db", float("nan"))),
            float("nan"),
        ),
        "bass_sub_scaling_assumption": str(s.get("sub_scaling_assumption", "single_bus_average_normalized")),
        "bass_sub_coherence_assumption": str(s.get("sub_coherence_assumption", "measured_complex_nominal")),
    }
    metrics["objective"] = float(eval_result.objective)
    diag = dict(s)
    diag.pop("eval_bundle", None)
    diag["dominant_channel"] = str(dominant)
    diag["feasibility_class"] = str(metrics["bass_feasibility_class"])
    diag["feasibility_reason"] = str(metrics["bass_feasibility_reason"])
    diag["feasibility_limiting_factor"] = str(metrics["bass_feasibility_limiting_factor"])
    metrics["diagnostics"] = diag
    metrics["gd_continuity"] = gd_cont
    metrics["snapshot"] = _final_metric_snapshot(metrics)
    return metrics


def _final_metric_snapshot(metrics: dict[str, Any]) -> dict[str, float | bool | str]:
    return {
        "allpass_enabled": bool(metrics.get("bass_allpass_enabled", False)),
        "allpass_freq_hz": float(_safe_float(metrics.get("bass_allpass_freq_hz", 0.0), 0.0)),
        "allpass_q": float(_safe_float(metrics.get("bass_allpass_q", 0.707), 0.707)),
        "cancellation_risk": float(_safe_float(metrics.get("bass_cancellation_risk", float("nan")), float("nan"))),
        "overlap_ripple_db": float(_safe_float(metrics.get("bass_overlap_ripple", float("nan")), float("nan"))),
        "sub_dominance_db": float(_safe_float(metrics.get("bass_sub_dominance", float("nan")), float("nan"))),
        "null_severity": float(_safe_float(metrics.get("bass_null_severity", float("nan")), float("nan"))),
        "predicted_sum_flatness_db": float(
            _safe_float(metrics.get("bass_predicted_sum_flatness_db", float("nan")), float("nan"))
        ),
        "predicted_sum_dip_depth_db": float(
            _safe_float(metrics.get("bass_predicted_sum_dip_depth_db", float("nan")), float("nan"))
        ),
        "predicted_sum_peak_excess_db": float(
            _safe_float(metrics.get("bass_predicted_sum_peak_excess_db", float("nan")), float("nan"))
        ),
        "overlap_extension_active": bool(metrics.get("bass_overlap_extension_active", False)),
        "overlap_extension_flatness_db": float(
            _safe_float(metrics.get("bass_overlap_extension_flatness_db", float("nan")), float("nan"))
        ),
        "overlap_extension_cancellation_risk": float(
            _safe_float(metrics.get("bass_overlap_extension_cancellation_risk", float("nan")), float("nan"))
        ),
        "overlap_extension_peak_excess_db": float(
            _safe_float(metrics.get("bass_overlap_extension_peak_excess_db", float("nan")), float("nan"))
        ),
        "overlap_extension_sub_dominance_db": float(
            _safe_float(metrics.get("bass_overlap_extension_sub_dominance_db", float("nan")), float("nan"))
        ),
        "xo_gd_mismatch_ms": float(_safe_float(metrics.get("bass_xo_gd_rms_mismatch_ms", float("nan")), float("nan"))),
        "xo_gd_rms_mismatch_ms": float(
            _safe_float(metrics.get("bass_xo_gd_rms_mismatch_ms", float("nan")), float("nan"))
        ),
        "xo_gd_max_mismatch_ms": float(
            _safe_float(metrics.get("bass_xo_gd_max_mismatch_ms", float("nan")), float("nan"))
        ),
        "overlap_ripple_delta_db": float(
            _safe_float(metrics.get("bass_overlap_ripple_delta_db", float("nan")), float("nan"))
        ),
        "sub_dominance_delta_db": float(
            _safe_float(metrics.get("bass_sub_dominance_delta_db", float("nan")), float("nan"))
        ),
        "xo_gd_mismatch_delta_ms": float(
            _safe_float(metrics.get("bass_xo_gd_mismatch_delta_ms", float("nan")), float("nan"))
        ),
        "xo_main_gd_ms": float(_safe_float(metrics.get("bass_xo_main_gd_ms", float("nan")), float("nan"))),
        "xo_sub_gd_ms": float(_safe_float(metrics.get("bass_xo_sub_gd_ms", float("nan")), float("nan"))),
        "dominant_channel": str(metrics.get("bass_dominant_channel", "unknown") or "unknown"),
        "feasibility_class": str(metrics.get("bass_feasibility_class", "marginal") or "marginal"),
        "feasibility_reason": str(metrics.get("bass_feasibility_reason", "") or ""),
        "feasibility_limiting_factor": str(metrics.get("bass_feasibility_limiting_factor", "unknown") or "unknown"),
        "metric_channel_mode": str(metrics.get("bass_metric_channel_mode", "worst_case") or "worst_case"),
    }


def compute_final_bass_integration_metrics(
    bundle: BassIntegrationBundle,
    fc_hz: float,
    profile: str,
    *,
    mode: str = "direct_dac",
    main_hpf_order: int = 4,
    sub_lpf_order: int = 4,
    sub_hpf_hz: float = 20.0,
    sub_hpf_order: int = 2,
    sub_combine_mode: str = "average",
    sub_delay_ms: float = 0.0,
    sub_polarity_invert: bool = False,
    sub_gain_trim_db: float = 0.0,
    sub_lpf_hz: float | None = None,
    sub_allpass_freq_hz: float | None = None,
    sub_allpass_q: float | None = None,
    guard_lo_ratio: float = AUTO_MODE_BASS_INTEGRATION_GUARD_LO_RATIO,
    guard_hi_ratio: float = AUTO_MODE_BASS_INTEGRATION_GUARD_HI_RATIO,
    l_fir: Any | None = None,
    r_fir: Any | None = None,
    sub_fir: Any | None = None,
    fir_sample_rate: int | None = None,
    robust: bool | None = None,
) -> dict[str, Any]:
    """Compute Bass Integration metrics through the canonical Direct-DAC evaluator."""
    mode_norm = "direct_dac"
    combine_mode_norm = normalize_sub_combine_mode(sub_combine_mode)
    fc = _safe_float(fc_hz, 80.0)
    has_realized_firs = any(value is not None for value in (l_fir, r_fir, sub_fir))
    fs = int(fir_sample_rate or bundle.l_main.sample_rate)
    fir_sig = (
        realized_fir_signature(l_fir, r_fir, sub_fir, sample_rate=fs)
        if has_realized_firs
        else ()
    )
    robust_enabled = bool(has_realized_firs if robust is None else robust)

    # --- session-scope cache ---
    _cache = _get_metrics_cache(bundle)
    _key = _metrics_cache_key(
        fc,
        profile,
        mode=mode_norm,
        main_hpf_order=int(main_hpf_order),
        sub_lpf_order=int(sub_lpf_order),
        sub_hpf_hz=float(sub_hpf_hz),
        sub_hpf_order=int(sub_hpf_order),
        sub_combine_mode=combine_mode_norm,
        sub_delay_ms=float(sub_delay_ms),
        sub_polarity_invert=bool(sub_polarity_invert),
        sub_gain_trim_db=float(sub_gain_trim_db),
        sub_lpf_hz=sub_lpf_hz,
        sub_allpass_freq_hz=sub_allpass_freq_hz,
        sub_allpass_q=sub_allpass_q,
        guard_lo_ratio=float(guard_lo_ratio),
        guard_hi_ratio=float(guard_hi_ratio),
        realized_fir_signature=fir_sig,
        robust=robust_enabled,
    )
    if _key in _cache:
        increment_metrics_cache_hit(bundle)
        return dict(_cache[_key])
    increment_metrics_cache_miss(bundle)
    # --- end cache check ---

    ap_freq_hz = _safe_float(sub_allpass_freq_hz, float("nan"))
    ap_q = _safe_float(sub_allpass_q, float("nan"))
    allpass_enabled = bool(np.isfinite(ap_freq_hz) and ap_freq_hz > 0.0 and np.isfinite(ap_q) and ap_q > 0.0)
    candidate = DirectDacCandidate(
        main_hpf_hz=float(fc),
        main_hpf_order=int(main_hpf_order),
        sub_hpf_hz=float(sub_hpf_hz),
        sub_hpf_order=int(sub_hpf_order),
        sub_lpf_hz=float(max(float(fc), _safe_float(sub_lpf_hz, float(fc)) if sub_lpf_hz is not None else float(fc))),
        sub_lpf_order=int(sub_lpf_order),
        sub_delay_ms=float(sub_delay_ms),
        sub_gain_trim_db=float(sub_gain_trim_db),
        sub_polarity_invert=bool(sub_polarity_invert),
        sub_allpass_enabled=bool(allpass_enabled),
        sub_allpass_freq_hz=float(ap_freq_hz) if allpass_enabled else 0.0,
        sub_allpass_q=float(ap_q) if allpass_enabled else 0.707,
        topology=str(dict(getattr(bundle, "diagnostics", {}) or {}).get("sub_topology", "single_sub_bus") or "single_sub_bus"),
        source="final_metrics",
    )
    eval_result = evaluate_direct_dac_candidate(
        bundle,
        candidate,
        profile=profile,
        sub_combine_mode=combine_mode_norm,
        guard_lo_ratio=float(guard_lo_ratio),
        guard_hi_ratio=float(guard_hi_ratio),
        l_fir=l_fir,
        r_fir=r_fir,
        sub_fir=sub_fir,
        fir_sample_rate=fs,
        robust=robust_enabled,
    )
    metrics = _direct_dac_eval_to_legacy_metrics(
        eval_result,
        profile=profile,
        mode_norm=mode_norm,
        combine_mode_norm=combine_mode_norm,
        allpass_enabled=allpass_enabled,
        ap_freq_hz=ap_freq_hz,
        ap_q=ap_q,
    )
    if len(_cache) >= 256:
        _cache.clear()
    _cache[_key] = metrics
    return dict(metrics)
