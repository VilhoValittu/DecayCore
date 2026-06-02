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

from ...auto_mode.shared import _auto_bass_integration_profile_weights
from ...io.measurement_bundle import BassIntegrationBundle
from ._constants import (
    AVR_CROSSOVER_CANDIDATES,
    DIRECT_DAC_OVERLAP_RATIOS,
    MIN_DIRECT_DAC_OVERLAP_RATIO,
)
from ._recommend_alignment import _evaluate_metric_grid
from ._utils import _get_bass_integration_pkg, _normalize_candidate_frequencies, _safe_float, normalize_sub_combine_mode


def _get_pkg():
    """Return the bass_integration package module for patchable attribute lookup."""
    return _get_bass_integration_pkg(__name__)


def _recommend_direct_dac_int_or_default(value: object, default: int) -> int:
    try:
        return max(1, int(value))
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
        return int(default)


def _recommend_direct_dac_eval_fc(
    *,
    bundle: BassIntegrationBundle,
    fc: float,
    profile: str,
    sub_hpf_hz_f: float,
    hpf_order_i: int,
    lpf_order_i: int,
    sub_hpf_order_i: int,
    combine_mode_norm: str,
    sub_delay_ms: float,
    sub_polarity_invert: bool,
    sub_gain_trim_db: float,
    w_main_act: float,
) -> dict[str, float | str] | None:
    if not np.isfinite(fc) or fc <= 0.0 or fc <= (sub_hpf_hz_f + 1.0):
        return None
    pkg = _get_pkg()
    l_drop = pkg._main_guard_band_drop_db(bundle.l_main, fc)
    r_drop = pkg._main_guard_band_drop_db(bundle.r_main, fc)
    drop_vals = [v for v in (l_drop, r_drop) if np.isfinite(v)]
    avg_drop = float(np.mean(np.asarray(drop_vals, dtype=float))) if drop_vals else float("nan")
    main_drop_norm = max(0.0, avg_drop) / 12.0 if np.isfinite(avg_drop) else float("nan")

    best_trial: dict[str, float | str] | None = None
    for ratio in DIRECT_DAC_OVERLAP_RATIOS:
        sub_lpf = float(fc * ratio)
        metrics = pkg.compute_final_bass_integration_metrics(
            bundle,
            fc,
            profile,
            mode="direct_dac",
            main_hpf_order=hpf_order_i,
            sub_lpf_order=lpf_order_i,
            sub_hpf_hz=sub_hpf_hz_f,
            sub_hpf_order=sub_hpf_order_i,
            sub_combine_mode=combine_mode_norm,
            sub_delay_ms=float(sub_delay_ms),
            sub_polarity_invert=bool(sub_polarity_invert),
            sub_gain_trim_db=float(sub_gain_trim_db),
            sub_lpf_hz=sub_lpf,
        )
        trial_score = _safe_float(metrics.get("objective", float("nan")), float("nan"))
        if np.isfinite(trial_score) and np.isfinite(main_drop_norm):
            trial_score -= w_main_act * float(main_drop_norm)
        if not np.isfinite(trial_score):
            continue
        trial: dict[str, float | str] = {
            "score": float(trial_score),
            "cancellation_risk": float(_safe_float(metrics.get("bass_cancellation_risk", float("nan")), float("nan"))),
            "overlap_ripple_db": float(_safe_float(metrics.get("bass_overlap_ripple", float("nan")), float("nan"))),
            "sub_dominance_db": float(_safe_float(metrics.get("bass_sub_dominance", float("nan")), float("nan"))),
            "null_severity": float(_safe_float(metrics.get("bass_null_severity", float("nan")), float("nan"))),
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
            "xo_gd_rms_mismatch_ms": float(_safe_float(metrics.get("bass_xo_gd_rms_mismatch_ms", float("nan")), float("nan"))),
            "xo_gd_max_mismatch_ms": float(_safe_float(metrics.get("bass_xo_gd_max_mismatch_ms", float("nan")), float("nan"))),
            "predicted_sum_flatness_db": float(_safe_float(metrics.get("bass_predicted_sum_flatness_db", float("nan")), float("nan"))),
            "predicted_sum_dip_depth_db": float(_safe_float(metrics.get("bass_predicted_sum_dip_depth_db", float("nan")), float("nan"))),
            "predicted_sum_peak_excess_db": float(_safe_float(metrics.get("bass_predicted_sum_peak_excess_db", float("nan")), float("nan"))),
            "overlap_ripple_delta_db": float(
                _safe_float(metrics.get("bass_overlap_ripple_delta_db", float("nan")), float("nan"))
            ),
            "sub_dominance_delta_db": float(
                _safe_float(metrics.get("bass_sub_dominance_delta_db", float("nan")), float("nan"))
            ),
            "xo_gd_mismatch_delta_ms": float(
                _safe_float(metrics.get("bass_xo_gd_mismatch_delta_ms", float("nan")), float("nan"))
            ),
            "dominant_channel": str(metrics.get("bass_dominant_channel", "unknown") or "unknown"),
            "feasibility_class": str(metrics.get("bass_feasibility_class", "marginal") or "marginal"),
            "feasibility_reason": str(metrics.get("bass_feasibility_reason", "") or ""),
            "main_activity_drop_db": float(avg_drop),
            "overlap_ratio": float(ratio),
            "sub_lpf_hz": float(sub_lpf),
            "metric_channel_mode": str(metrics.get("bass_metric_channel_mode", "worst_case") or "worst_case"),
        }
        if best_trial is None or float(trial["score"]) > float(best_trial["score"]):
            best_trial = trial
    return best_trial


def _recommend_direct_dac_scan_grid(
    *,
    fc_list: list[float],
    scores: dict[float, dict[str, float | str]],
    eval_fc,
) -> None:
    new_fcs: list[float] = []
    seen: set[float] = set()
    for fc_raw in fc_list:
        fc = round(float(fc_raw), 4)
        if fc in scores or fc in seen:
            continue
        seen.add(fc)
        new_fcs.append(fc)
    rows = _evaluate_metric_grid([(float(fc),) for fc in new_fcs], lambda fc: (0.0, eval_fc(float(fc)) or {}))
    for (candidate, _score, result) in rows:
        fc = round(float(candidate[0]), 4)
        scores[fc] = result or {
            "score": float("nan"),
            "cancellation_risk": float("nan"),
            "overlap_ripple_db": float("nan"),
            "sub_dominance_db": float("nan"),
            "null_severity": float("nan"),
            "overlap_extension_active": False,
            "overlap_extension_flatness_db": float("nan"),
            "overlap_extension_cancellation_risk": float("nan"),
            "overlap_extension_peak_excess_db": float("nan"),
            "overlap_extension_sub_dominance_db": float("nan"),
            "xo_gd_rms_mismatch_ms": float("nan"),
            "xo_gd_max_mismatch_ms": float("nan"),
            "predicted_sum_flatness_db": float("nan"),
            "predicted_sum_dip_depth_db": float("nan"),
            "predicted_sum_peak_excess_db": float("nan"),
            "overlap_ripple_delta_db": float("nan"),
            "sub_dominance_delta_db": float("nan"),
            "xo_gd_mismatch_delta_ms": float("nan"),
            "dominant_channel": "unknown",
            "feasibility_class": "marginal",
            "feasibility_reason": "",
            "main_activity_drop_db": float("nan"),
            "overlap_ratio": MIN_DIRECT_DAC_OVERLAP_RATIO,
            "sub_lpf_hz": fc * MIN_DIRECT_DAC_OVERLAP_RATIO,
            "metric_channel_mode": "worst_case",
        }


def _recommend_direct_dac_best_so_far(
    scores: dict[float, dict[str, float | str]],
    *,
    fallback_hz: float,
) -> float:
    valid = {fc: float(d["score"]) for fc, d in scores.items() if np.isfinite(_safe_float(d.get("score", float("nan")), float("nan")))}
    return float(max(valid, key=lambda x: valid[x])) if valid else float(fallback_hz)


def recommend_direct_dac_crossover(
    bundle: BassIntegrationBundle,
    candidates: tuple[float, ...] | None = None,
    profile: str = "safe",
    *,
    main_hpf_order: int = 4,
    sub_lpf_order: int = 4,
    sub_hpf_hz: float = 20.0,
    sub_hpf_order: int = 2,
    sub_combine_mode: str = "average",
    sub_delay_ms: float = 0.0,
    sub_polarity_invert: bool = False,
    sub_gain_trim_db: float = 0.0,
) -> dict[str, Any]:
    """Recommend a Direct-DAC crossover using coarse->refine->micro-refine search."""
    weights = _auto_bass_integration_profile_weights(profile)
    w_main_act = float(weights.get("main_activity", 6.0))
    hpf_order_i = _recommend_direct_dac_int_or_default(main_hpf_order, 4)
    lpf_order_i = _recommend_direct_dac_int_or_default(sub_lpf_order, hpf_order_i)
    sub_hpf_hz_f = float(sub_hpf_hz) if isinstance(sub_hpf_hz, (int, float)) else 20.0
    if not np.isfinite(sub_hpf_hz_f) or sub_hpf_hz_f < 0.0:
        sub_hpf_hz_f = 20.0
    sub_hpf_order_i = _recommend_direct_dac_int_or_default(sub_hpf_order, 2)

    combine_mode_norm = normalize_sub_combine_mode(sub_combine_mode)

    # If caller passes explicit candidates, use them directly (skip coarse->refine)
    explicit_candidates = _normalize_candidate_frequencies(candidates)

    _fc_lo = float(AVR_CROSSOVER_CANDIDATES[0])
    _fc_hi = float(AVR_CROSSOVER_CANDIDATES[-1])

    scores: dict[float, dict[str, float | str]] = {}

    if explicit_candidates:
        _recommend_direct_dac_scan_grid(
            fc_list=list(explicit_candidates),
            scores=scores,
            eval_fc=lambda fc: _recommend_direct_dac_eval_fc(
                bundle=bundle,
                fc=float(fc),
                profile=profile,
                sub_hpf_hz_f=sub_hpf_hz_f,
                hpf_order_i=hpf_order_i,
                lpf_order_i=lpf_order_i,
                sub_hpf_order_i=sub_hpf_order_i,
                combine_mode_norm=combine_mode_norm,
                sub_delay_ms=float(sub_delay_ms),
                sub_polarity_invert=bool(sub_polarity_invert),
                sub_gain_trim_db=float(sub_gain_trim_db),
                w_main_act=w_main_act,
            ),
        )
    else:
        # Phase A: coarse 5 Hz grid
        coarse_grid = [
            round(float(_fc_lo) + 5.0 * i, 4)
            for i in range(int(round((_fc_hi - _fc_lo) / 5.0)) + 1)
        ]
        _recommend_direct_dac_scan_grid(
            fc_list=coarse_grid,
            scores=scores,
            eval_fc=lambda fc: _recommend_direct_dac_eval_fc(
                bundle=bundle,
                fc=float(fc),
                profile=profile,
                sub_hpf_hz_f=sub_hpf_hz_f,
                hpf_order_i=hpf_order_i,
                lpf_order_i=lpf_order_i,
                sub_hpf_order_i=sub_hpf_order_i,
                combine_mode_norm=combine_mode_norm,
                sub_delay_ms=float(sub_delay_ms),
                sub_polarity_invert=bool(sub_polarity_invert),
                sub_gain_trim_db=float(sub_gain_trim_db),
                w_main_act=w_main_act,
            ),
        )
        best_fc_coarse = _recommend_direct_dac_best_so_far(scores, fallback_hz=bundle.avr_crossover_hz)

        # Phase B: ±10 Hz @ 1 Hz around coarse best
        refine_lo = float(np.clip(best_fc_coarse - 10.0, _fc_lo, _fc_hi))
        refine_hi = float(np.clip(best_fc_coarse + 10.0, _fc_lo, _fc_hi))
        refine_steps = max(0, int(round((refine_hi - refine_lo) / 1.0)))
        refine_grid = [round(refine_lo + 1.0 * i, 4) for i in range(refine_steps + 1)]
        _recommend_direct_dac_scan_grid(
            fc_list=refine_grid,
            scores=scores,
            eval_fc=lambda fc: _recommend_direct_dac_eval_fc(
                bundle=bundle,
                fc=float(fc),
                profile=profile,
                sub_hpf_hz_f=sub_hpf_hz_f,
                hpf_order_i=hpf_order_i,
                lpf_order_i=lpf_order_i,
                sub_hpf_order_i=sub_hpf_order_i,
                combine_mode_norm=combine_mode_norm,
                sub_delay_ms=float(sub_delay_ms),
                sub_polarity_invert=bool(sub_polarity_invert),
                sub_gain_trim_db=float(sub_gain_trim_db),
                w_main_act=w_main_act,
            ),
        )
        best_fc_refine = _recommend_direct_dac_best_so_far(scores, fallback_hz=bundle.avr_crossover_hz)

        # Phase C: micro-refine ±2 Hz @ 0.5 Hz around refined best
        micro_lo = float(np.clip(best_fc_refine - 2.0, _fc_lo, _fc_hi))
        micro_hi = float(np.clip(best_fc_refine + 2.0, _fc_lo, _fc_hi))
        micro_steps = max(0, int(round((micro_hi - micro_lo) / 0.5)))
        micro_grid = [round(micro_lo + 0.5 * i, 4) for i in range(micro_steps + 1)]
        _recommend_direct_dac_scan_grid(
            fc_list=micro_grid,
            scores=scores,
            eval_fc=lambda fc: _recommend_direct_dac_eval_fc(
                bundle=bundle,
                fc=float(fc),
                profile=profile,
                sub_hpf_hz_f=sub_hpf_hz_f,
                hpf_order_i=hpf_order_i,
                lpf_order_i=lpf_order_i,
                sub_hpf_order_i=sub_hpf_order_i,
                combine_mode_norm=combine_mode_norm,
                sub_delay_ms=float(sub_delay_ms),
                sub_polarity_invert=bool(sub_polarity_invert),
                sub_gain_trim_db=float(sub_gain_trim_db),
                w_main_act=w_main_act,
            ),
        )

    best_hz = _recommend_direct_dac_best_so_far(scores, fallback_hz=bundle.avr_crossover_hz)
    best_entry = dict(scores.get(round(best_hz, 4), scores.get(best_hz, {})) or {})
    return {
        "recommended_hz": float(best_hz),
        "recommended_sub_lpf_hz": float(best_entry.get("sub_lpf_hz", best_hz)),
        "scores": scores,
        "feasibility_class": str(best_entry.get("feasibility_class", "marginal") or "marginal"),
        "feasibility_reason": str(best_entry.get("feasibility_reason", "") or ""),
        "dominant_channel": str(best_entry.get("dominant_channel", "unknown") or "unknown"),
    }
