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

from ..dsp.bass_integration import (
    direct_dac_candidate_from_data,
    evaluate_direct_dac_candidate,
    write_direct_dac_candidate_to_data,
)

logger = logging.getLogger("DecayCore")


def apply_direct_dac_bass_integration_result(
    *,
    cfg: Any,
    measurements: dict[str, Any],
    data: dict[str, Any],
    l_imp: np.ndarray,
    r_imp: np.ndarray,
    sub_ir: np.ndarray | None,
    l_st: dict[str, Any] | None,
    r_st: dict[str, Any] | None,
    sub_st: dict[str, Any] | None,
) -> dict[str, Any]:
    if (
        sub_ir is None
        or not bool(getattr(cfg, "bass_integration_enable", False))
        or str(getattr(cfg, "bass_integration_mode", "") or "").strip().lower() != "direct_dac"
    ):
        return {}
    bundle = measurements.get("bass_integration_bundle", None)
    if bundle is None:
        return {}
    try:
        candidate = direct_dac_candidate_from_data(data, cfg)
        metrics = evaluate_direct_dac_candidate(
            bundle,
            candidate,
            profile=str(getattr(cfg, "bass_integration_profile", getattr(bundle, "profile", "safe")) or "safe"),
            sub_combine_mode=str(getattr(cfg, "bass_integration_sub_combine_mode", "average") or "average"),
        )
        write_direct_dac_candidate_to_data(data, candidate, metrics, reason="Direct-DAC final candidate verification.")
        result_dict = {
            "enabled": True,
            "main_hpf_hz": float(candidate.main_hpf_hz),
            "sub_hpf_hz": float(candidate.sub_hpf_hz),
            "sub_lpf_hz": float(candidate.sub_lpf_hz),
            "sub_overlap_hz": float(candidate.sub_lpf_hz - candidate.main_hpf_hz),
            "sub_delay_ms": float(candidate.sub_delay_ms),
            "sub_gain_db": float(candidate.sub_gain_trim_db),
            "sub_polarity_invert": bool(candidate.sub_polarity_invert),
            "score": float(metrics.score),
            "objective": float(metrics.objective),
            "rejected": not bool(metrics.feasible),
            "reject_reason": ", ".join(metrics.reject_reasons),
            "reject_reasons": list(metrics.reject_reasons),
            "worst_channel": str(metrics.dominant_channel),
            "cancellation_score": float(max(metrics.left.cancellation_risk, metrics.right.cancellation_risk)),
            "cancellation_risk": str(metrics.summary.get("feasibility_class", "")),
            "magnitude_ripple_db": float(metrics.summary.get("overlap_ripple_db", float("nan"))),
        }
        data["bass_integration_alignment_auto_applied"] = True
        data["bass_integration_alignment_reason"] = "Direct DAC final candidate verified with canonical evaluator."
        sub_target_policy = str(data.get("sub_target_policy", "") or "").strip()
        sub_target_lpf_hz = data.get("sub_target_lpf_hz", None)
        sub_target_slope = data.get("sub_target_lpf_slope_db_per_oct", None)
        bi_meta = data.setdefault("_bass_integration_meta", {})
        if isinstance(bi_meta, dict):
            bi_meta.update(
                {
                    "enabled": True,
                    "mode": "direct_dac",
                    "optimization": "automatic",
                    "avr_crossover_hz": float(candidate.main_hpf_hz),
                    "direct_dac_sub_lpf_hz": float(candidate.sub_lpf_hz),
                    "recommended_crossover_hz": float(candidate.main_hpf_hz),
                    "recommended_sub_lpf_hz": float(candidate.sub_lpf_hz),
                    "direct_dac_result": dict(result_dict),
                    **(
                        {
                            "sub_target_policy": sub_target_policy,
                            "sub_target_lpf_hz": float(sub_target_lpf_hz),
                            "sub_target_lpf_slope_db_per_oct": float(sub_target_slope),
                        }
                        if sub_target_policy and sub_target_lpf_hz is not None and sub_target_slope is not None
                        else {}
                    ),
                }
            )
            diagnostics = dict(bi_meta.get("diagnostics", {}) or {})
            diagnostics.update(
                {
                    "direct_dac_main_hpf_hz": float(candidate.main_hpf_hz),
                    "direct_dac_sub_hpf_hz": float(candidate.sub_hpf_hz),
                    "direct_dac_sub_lpf_hz": float(candidate.sub_lpf_hz),
                    "direct_dac_sub_overlap_hz": float(candidate.sub_lpf_hz - candidate.main_hpf_hz),
                    "direct_dac_cancellation_score": float(result_dict["cancellation_score"]),
                    "direct_dac_magnitude_ripple_db": float(result_dict["magnitude_ripple_db"]),
                    "direct_dac_candidate_score": float(metrics.score),
                    "direct_dac_worst_channel": str(metrics.dominant_channel),
                }
            )
            bi_meta["diagnostics"] = diagnostics
            bi_meta["alignment"] = {
                "applied": True,
                "delay_ms": float(candidate.sub_delay_ms),
                "polarity_invert": bool(candidate.sub_polarity_invert),
                "gain_trim_db": float(candidate.sub_gain_trim_db),
                "reason": "Direct DAC final candidate verified with canonical evaluator.",
                "improvement_score": float(metrics.objective),
                "baseline_metrics": {},
                "optimized_metrics": dict(result_dict),
            }
        for st in (l_st, r_st, sub_st):
            if isinstance(st, dict):
                st["direct_dac_bass_integration"] = dict(result_dict)
        logger.info(
            "Direct-DAC Bass Integration final: main HPF %.1f Hz order %d, sub HPF %.1f Hz order %d, "
            "sub LPF %.1f Hz order %d, overlap ratio %.2f, sub delay %+.2f ms, gain %+0.2f dB, "
            "polarity %s, allpass %s, score %.3f, worst channel %s, feasibility %s, reject reasons %s, "
            "export model camilladsp_yaml_compatible",
            float(candidate.main_hpf_hz),
            int(candidate.main_hpf_order),
            float(candidate.sub_hpf_hz),
            int(candidate.sub_hpf_order),
            float(candidate.sub_lpf_hz),
            int(candidate.sub_lpf_order),
            float(candidate.sub_lpf_hz) / max(float(candidate.main_hpf_hz), 1e-9),
            float(candidate.sub_delay_ms),
            float(candidate.sub_gain_trim_db),
            "invert" if bool(candidate.sub_polarity_invert) else "normal",
            (f"{candidate.sub_allpass_freq_hz:.1f} Hz Q {candidate.sub_allpass_q:.3f}" if candidate.sub_allpass_enabled else "off"),
            float(metrics.score),
            str(metrics.dominant_channel),
            str(metrics.summary.get("feasibility_class", "")),
            ",".join(metrics.reject_reasons) if metrics.reject_reasons else "none",
        )
        return dict(result_dict)
    except Exception:
        logger.debug("Direct-DAC final candidate verification failed", exc_info=True)
        return {}


__all__ = ["apply_direct_dac_bass_integration_result"]
