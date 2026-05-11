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
from ...io.measurement_bundle import BassIntegrationBundle, TransferData
from ..bass_cache import (
    _get_metrics_cache,
    _metrics_cache_key,
    increment_metrics_cache_hit,
    increment_metrics_cache_miss,
)
from ._bundles import _build_avr_lfe_main_trial_bundle, _build_direct_dac_trial_bundle
from ._channel_metrics import _metric_delta
from ._constants import BASS_INTEGRATION_FEASIBILITY_OBJECTIVE_PENALTY
from ._diagnostics import compute_bass_integration_diagnostics
from ._gd_feasibility import (
    _classify_bass_integration_feasibility,
    _dominant_bass_channel,
    compute_xo_gd_continuity,
)
from ._utils import _safe_float, normalize_sub_combine_mode


def _score_final_bass_metrics(metrics: dict[str, Any], *, profile: str) -> float:
    weights = _auto_bass_integration_profile_weights(profile)
    penalty = 0.0

    def _add(metric_key: str, *, weight: float, scale: float = 1.0) -> None:
        nonlocal penalty
        value = _safe_float(metrics.get(metric_key, float("nan")), float("nan"))
        if np.isfinite(value):
            penalty += float(weight) * max(0.0, float(value)) / float(max(scale, 1e-9))

    _add("bass_cancellation_risk", weight=float(weights.get("cancellation", 8.0)), scale=1.0)
    _add("bass_overlap_ripple", weight=float(weights.get("overlap_ripple", 1.8)), scale=10.0)
    _add("bass_sub_dominance", weight=float(weights.get("sub_dominance", 0.9)), scale=8.0)
    _add("bass_null_severity", weight=0.55 * float(weights.get("cancellation", 8.0)), scale=6.0)
    _add("bass_xo_gd_rms_mismatch_ms", weight=float(weights.get("xo_gd_continuity", 0.8)), scale=3.0)
    _add("bass_xo_gd_max_mismatch_ms", weight=0.35 * float(weights.get("xo_gd_continuity", 0.8)), scale=6.0)
    _add("bass_predicted_sum_flatness_db", weight=0.35 * float(weights.get("overlap_ripple", 1.8)), scale=8.0)
    _add("bass_predicted_sum_dip_depth_db", weight=0.40 * float(weights.get("cancellation", 8.0)), scale=6.0)
    _add("bass_predicted_sum_peak_excess_db", weight=0.25 * float(weights.get("overlap_ripple", 1.8)), scale=6.0)
    _add(
        "bass_overlap_extension_flatness_db",
        weight=0.20 * float(weights.get("overlap_ripple", 1.8)),
        scale=8.0,
    )
    _add(
        "bass_overlap_extension_cancellation_risk",
        weight=0.20 * float(weights.get("cancellation", 8.0)),
        scale=1.0,
    )
    _add(
        "bass_overlap_extension_peak_excess_db",
        weight=0.15 * float(weights.get("overlap_ripple", 1.8)),
        scale=6.0,
    )
    _add(
        "bass_overlap_ripple_delta_db",
        weight=0.55 * float(weights.get("overlap_ripple", 1.8)),
        scale=4.0,
    )
    _add(
        "bass_sub_dominance_delta_db",
        weight=0.75 * float(weights.get("sub_dominance", 0.9)),
        scale=4.0,
    )
    _add(
        "bass_xo_gd_mismatch_delta_ms",
        weight=0.75 * float(weights.get("xo_gd_continuity", 0.8)),
        scale=10.0,
    )
    overlap_extension_sub_dominance_db = _safe_float(
        metrics.get("bass_overlap_extension_sub_dominance_db", float("nan")),
        float("nan"),
    )
    if np.isfinite(overlap_extension_sub_dominance_db):
        penalty += (
            0.12
            * float(weights.get("sub_dominance", 0.9))
            * max(0.0, float(overlap_extension_sub_dominance_db) - 12.0)
            / 6.0
        )
    feasibility_class = str(metrics.get("bass_feasibility_class", "good") or "good").strip().lower()
    penalty += float(BASS_INTEGRATION_FEASIBILITY_OBJECTIVE_PENALTY.get(feasibility_class, 0.0))
    return float(-penalty)


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
        "metric_channel_mode": str(metrics.get("bass_metric_channel_mode", "worst_case") or "worst_case"),
    }


def compute_final_bass_integration_metrics(
    bundle: BassIntegrationBundle,
    fc_hz: float,
    profile: str,
    *,
    mode: str = "avr_lfe_main_decomposed",
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
) -> dict[str, Any]:
    mode_norm = str(mode or "avr_lfe_main_decomposed").strip().lower()
    combine_mode_norm = normalize_sub_combine_mode(sub_combine_mode)
    fc = _safe_float(fc_hz, 80.0)

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
    )
    if _key in _cache:
        increment_metrics_cache_hit(bundle)
        return dict(_cache[_key])
    increment_metrics_cache_miss(bundle)
    # --- end cache check ---

    eval_bundle = bundle
    ap_freq_hz = _safe_float(sub_allpass_freq_hz, float("nan"))
    ap_q = _safe_float(sub_allpass_q, float("nan"))
    allpass_enabled = bool(np.isfinite(ap_freq_hz) and ap_freq_hz > 0.0 and np.isfinite(ap_q) and ap_q > 0.0)
    if mode_norm == "direct_dac":
        eval_bundle = _build_direct_dac_trial_bundle(
            bundle,
            fc_hz=float(fc),
            main_hpf_order=int(main_hpf_order),
            sub_lpf_order=int(sub_lpf_order),
            sub_hpf_hz=float(sub_hpf_hz),
            sub_hpf_order=int(sub_hpf_order),
            sub_combine_mode=combine_mode_norm,
            sub_delay_ms=float(sub_delay_ms),
            sub_polarity_invert=bool(sub_polarity_invert),
            sub_gain_trim_db=float(sub_gain_trim_db),
            sub_lpf_hz=sub_lpf_hz,
            sub_allpass_freq_hz=(float(ap_freq_hz) if allpass_enabled else None),
            sub_allpass_q=(float(ap_q) if allpass_enabled else None),
        )
    elif mode_norm == "avr_lfe_main_decomposed" and (
        abs(float(sub_delay_ms)) > 1e-9
        or bool(sub_polarity_invert)
        or abs(float(sub_gain_trim_db)) > 1e-9
    ):
        eval_bundle = _build_avr_lfe_main_trial_bundle(
            bundle,
            sub_combine_mode=combine_mode_norm,
            sub_delay_ms=float(sub_delay_ms),
            sub_polarity_invert=bool(sub_polarity_invert),
            sub_gain_trim_db=float(sub_gain_trim_db),
        )
    diag = compute_bass_integration_diagnostics(
        eval_bundle,
        float(fc),
        profile,
        sub_combine_mode=combine_mode_norm,
        sub_lpf_hz=sub_lpf_hz,
        guard_lo_ratio=float(guard_lo_ratio),
        guard_hi_ratio=float(guard_hi_ratio),
    )
    gd_cont = compute_xo_gd_continuity(
        eval_bundle,
        float(fc),
        sub_combine_mode=combine_mode_norm,
    )
    overlap_ripple_delta_db = _metric_delta(
        diag.get("overlap_ripple_db_l", float("nan")),
        diag.get("overlap_ripple_db_r", float("nan")),
    )
    sub_dominance_delta_db = _metric_delta(
        diag.get("sub_dominance_db_l", float("nan")),
        diag.get("sub_dominance_db_r", float("nan")),
    )
    xo_gd_mismatch_delta_ms = _metric_delta(
        gd_cont.get("gd_rms_mismatch_ms_l", float("nan")),
        gd_cont.get("gd_rms_mismatch_ms_r", float("nan")),
    )
    dominant_channel = _dominant_bass_channel(
        overlap_ripple_l=diag.get("overlap_ripple_db_l", float("nan")),
        overlap_ripple_r=diag.get("overlap_ripple_db_r", float("nan")),
        sub_dominance_l=diag.get("sub_dominance_db_l", float("nan")),
        sub_dominance_r=diag.get("sub_dominance_db_r", float("nan")),
        xo_gd_l=gd_cont.get("gd_rms_mismatch_ms_l", float("nan")),
        xo_gd_r=gd_cont.get("gd_rms_mismatch_ms_r", float("nan")),
    )
    feasibility_class, feasibility_reason = _classify_bass_integration_feasibility(
        overlap_ripple_worst=diag.get("overlap_ripple_db_worst", float("nan")),
        sub_dominance_worst=diag.get("sub_dominance_db_worst", float("nan")),
        xo_gd_rms_worst=gd_cont.get("gd_rms_mismatch_ms_worst", float("nan")),
        overlap_ripple_delta=overlap_ripple_delta_db,
        sub_dominance_delta=sub_dominance_delta_db,
        xo_gd_delta=xo_gd_mismatch_delta_ms,
        dominant_channel=dominant_channel,
        sub_combine_mode=combine_mode_norm,
        sub_level_delta_db_20_120=diag.get("sub_combined_level_delta_db_20_120", float("nan")),
    )
    metrics = {
        "bass_cancellation_risk": _safe_float(diag.get("cancellation_risk", float("nan")), float("nan")),
        "bass_overlap_ripple": _safe_float(diag.get("overlap_ripple_db", float("nan")), float("nan")),
        "bass_sub_dominance": _safe_float(diag.get("sub_dominance_db", float("nan")), float("nan")),
        "bass_null_severity": _safe_float(diag.get("null_severity", float("nan")), float("nan")),
        "bass_predicted_sum_flatness_db": _safe_float(diag.get("predicted_sum_flatness_db", float("nan")), float("nan")),
        "bass_predicted_sum_dip_depth_db": _safe_float(diag.get("predicted_sum_dip_depth_db", float("nan")), float("nan")),
        "bass_predicted_sum_peak_excess_db": _safe_float(diag.get("predicted_sum_peak_excess_db", float("nan")), float("nan")),
        "bass_overlap_extension_active": bool(diag.get("overlap_extension_active", False)),
        "bass_overlap_extension_flatness_db": _safe_float(
            diag.get("overlap_extension_flatness_db", float("nan")),
            float("nan"),
        ),
        "bass_overlap_extension_cancellation_risk": _safe_float(
            diag.get("overlap_extension_cancellation_risk", float("nan")),
            float("nan"),
        ),
        "bass_overlap_extension_peak_excess_db": _safe_float(
            diag.get("overlap_extension_peak_excess_db", float("nan")),
            float("nan"),
        ),
        "bass_overlap_extension_sub_dominance_db": _safe_float(
            diag.get("overlap_extension_sub_dominance_db", float("nan")),
            float("nan"),
        ),
        "bass_guard_lo_hz": _safe_float(diag.get("guard_lo_hz", float("nan")), float("nan")),
        "bass_guard_hi_hz": _safe_float(diag.get("guard_hi_hz", float("nan")), float("nan")),
        "bass_integration_profile": str(_auto_bass_integration_profile_norm(profile)),
        "bass_integration_mode": str(mode_norm),
        "bass_metric_channel_mode": "worst_case",
        "bass_xo_gd_mismatch_ms": _safe_float(gd_cont.get("gd_rms_mismatch_ms_worst", float("nan")), float("nan")),
        "bass_xo_gd_rms_mismatch_ms": _safe_float(gd_cont.get("gd_rms_mismatch_ms_worst", float("nan")), float("nan")),
        "bass_xo_gd_max_mismatch_ms": _safe_float(gd_cont.get("gd_max_mismatch_ms_worst", float("nan")), float("nan")),
        "bass_xo_l_gd_mismatch_ms": _safe_float(gd_cont.get("gd_rms_mismatch_ms_l", float("nan")), float("nan")),
        "bass_xo_r_gd_mismatch_ms": _safe_float(gd_cont.get("gd_rms_mismatch_ms_r", float("nan")), float("nan")),
        "bass_xo_main_gd_ms": _safe_float(
            (
                _safe_float(gd_cont.get("l_main_gd_ms", float("nan")), float("nan"))
                + _safe_float(gd_cont.get("r_main_gd_ms", float("nan")), float("nan"))
            )
            / 2.0,
            float("nan"),
        ),
        "bass_xo_sub_gd_ms": _safe_float(gd_cont.get("sub_gd_ms", float("nan")), float("nan")),
        "bass_allpass_enabled": bool(allpass_enabled),
        "bass_allpass_freq_hz": float(ap_freq_hz) if allpass_enabled else 0.0,
        "bass_allpass_q": float(ap_q) if allpass_enabled else 0.707,
        "bass_sub_combine_mode": str(combine_mode_norm),
        "bass_sub_combined_level_delta_db_20_120": _safe_float(
            diag.get("sub_combined_level_delta_db_20_120", float("nan")),
            float("nan"),
        ),
        "bass_sub_combined_level_delta_db_30_90": _safe_float(
            diag.get("sub_combined_level_delta_db_30_90", float("nan")),
            float("nan"),
        ),
        "bass_alignment_applied": bool(diag.get("whether_alignment_applied", False)),
        "bass_alignment_offset_ms": _safe_float(diag.get("alignment_offset_ms", 0.0), 0.0),
        "bass_alignment_confidence": _safe_float(diag.get("alignment_confidence", float("nan")), float("nan")),
        "bass_cancellation_risk_l": _safe_float(diag.get("cancellation_risk_l", float("nan")), float("nan")),
        "bass_cancellation_risk_r": _safe_float(diag.get("cancellation_risk_r", float("nan")), float("nan")),
        "bass_cancellation_risk_worst": _safe_float(diag.get("cancellation_risk_worst", float("nan")), float("nan")),
        "bass_overlap_ripple_l": _safe_float(diag.get("overlap_ripple_db_l", float("nan")), float("nan")),
        "bass_overlap_ripple_r": _safe_float(diag.get("overlap_ripple_db_r", float("nan")), float("nan")),
        "bass_overlap_ripple_worst": _safe_float(diag.get("overlap_ripple_db_worst", float("nan")), float("nan")),
        "bass_sub_dominance_l": _safe_float(diag.get("sub_dominance_db_l", float("nan")), float("nan")),
        "bass_sub_dominance_r": _safe_float(diag.get("sub_dominance_db_r", float("nan")), float("nan")),
        "bass_sub_dominance_worst": _safe_float(diag.get("sub_dominance_db_worst", float("nan")), float("nan")),
        "bass_null_severity_l": _safe_float(diag.get("null_severity_l", float("nan")), float("nan")),
        "bass_null_severity_r": _safe_float(diag.get("null_severity_r", float("nan")), float("nan")),
        "bass_null_severity_worst": _safe_float(diag.get("null_severity_worst", float("nan")), float("nan")),
        "bass_overlap_ripple_delta_db": float(overlap_ripple_delta_db),
        "bass_sub_dominance_delta_db": float(sub_dominance_delta_db),
        "bass_xo_gd_mismatch_delta_ms": float(xo_gd_mismatch_delta_ms),
        "bass_dominant_channel": str(dominant_channel),
        "bass_feasibility_class": str(feasibility_class),
        "bass_feasibility_reason": str(feasibility_reason),
    }
    metrics["objective"] = _score_final_bass_metrics(metrics, profile=profile)
    metrics["diagnostics"] = {
        **dict(diag or {}),
        **dict(gd_cont or {}),
        "overlap_ripple_delta_db": float(overlap_ripple_delta_db),
        "sub_dominance_delta_db": float(sub_dominance_delta_db),
        "gd_mismatch_delta_ms": float(xo_gd_mismatch_delta_ms),
        "dominant_channel": str(dominant_channel),
        "feasibility_class": str(feasibility_class),
        "feasibility_reason": str(feasibility_reason),
        "overlap_extension_active": bool(diag.get("overlap_extension_active", False)),
    }
    metrics["gd_continuity"] = dict(gd_cont or {})
    metrics["snapshot"] = _final_metric_snapshot(metrics)
    if len(_cache) >= 256:
        _cache.clear()
    _cache[_key] = metrics
    return dict(metrics)
