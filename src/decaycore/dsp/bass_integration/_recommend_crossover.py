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

_LOG = logging.getLogger("DecayCore.dsp")

from ...auto_mode.shared import _auto_bass_integration_profile_weights
from ...dsp.lf_rolloff import estimate_lf_rolloff_f6
from ...io.measurement_bundle import BassIntegrationBundle
from ._constants import (
    AVR_CROSSOVER_CANDIDATES,
    DIRECT_DAC_OVERLAP_RATIOS,
    MIN_DIRECT_DAC_OVERLAP_RATIO,
)
from ._recommend_alignment import _evaluate_metric_grid
from ._utils import _get_pkg, _normalize_candidate_frequencies, _safe_float, normalize_sub_combine_mode


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


def _direct_dac_main_rolloff_meta(bundle: BassIntegrationBundle) -> dict[str, float | str]:
    def _estimate(transfer) -> float:
        estimate = estimate_lf_rolloff_f6(
            getattr(transfer, "freqs_hz", []),
            getattr(transfer, "mag_db", []),
            min_hz=15.0,
            max_hz=float(AVR_CROSSOVER_CANDIDATES[-1]),
            ref_min_hz=220.0,
            ref_max_hz=600.0,
            search_max_hz=220.0,
            smooth_oct=1.0,
            default_hz=float("nan"),
        )
        return float(estimate.f6_hz)

    l_f6 = _estimate(bundle.l_main)
    r_f6 = _estimate(bundle.r_main)
    vals = [v for v in (l_f6, r_f6) if np.isfinite(v)]
    worst = float(max(vals)) if vals else float("nan")
    if np.isfinite(l_f6) and np.isfinite(r_f6) and abs(l_f6 - r_f6) > 20.0:
        _LOG.warning(
            "Bass integration: L/R main speaker F6 estimates diverge by %.1f Hz (L=%.1f Hz, R=%.1f Hz) — "
            "measurement asymmetry may affect crossover recommendation",
            abs(l_f6 - r_f6),
            l_f6,
            r_f6,
        )
    return {
        "main_l_f6_hz": float(l_f6) if np.isfinite(l_f6) else float("nan"),
        "main_r_f6_hz": float(r_f6) if np.isfinite(r_f6) else float("nan"),
        "main_f6_worst_hz": float(worst),
    }


def _direct_dac_main_usability_penalty(fc: float, main_rolloff_meta: dict[str, float | str]) -> float:
    worst_f6 = _safe_float(main_rolloff_meta.get("main_f6_worst_hz", float("nan")), float("nan"))
    if not np.isfinite(worst_f6):
        return 0.0
    guard_hz = float(worst_f6) + max(2.0, 0.04 * float(worst_f6))
    deficit_hz = max(0.0, float(guard_hz) - float(fc))
    return float(deficit_hz / 12.0)


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
    main_rolloff_meta: dict[str, float | str],
) -> dict[str, float | str] | None:
    if not np.isfinite(fc) or fc <= 0.0 or fc <= (sub_hpf_hz_f + 1.0):
        return None
    pkg = _get_pkg()
    l_drop = pkg._main_guard_band_drop_db(bundle.l_main, fc)
    r_drop = pkg._main_guard_band_drop_db(bundle.r_main, fc)
    drop_vals = [v for v in (l_drop, r_drop) if np.isfinite(v)]
    avg_drop = float(np.mean(np.asarray(drop_vals, dtype=float))) if drop_vals else float("nan")
    main_drop_norm = max(0.0, avg_drop) / 12.0 if np.isfinite(avg_drop) else float("nan")
    main_usability_penalty = _direct_dac_main_usability_penalty(float(fc), main_rolloff_meta)

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
        if np.isfinite(trial_score) and np.isfinite(main_usability_penalty):
            trial_score -= w_main_act * float(main_usability_penalty)
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
            "main_usability_penalty": float(main_usability_penalty),
            "main_l_f6_hz": float(_safe_float(main_rolloff_meta.get("main_l_f6_hz", float("nan")), float("nan"))),
            "main_r_f6_hz": float(_safe_float(main_rolloff_meta.get("main_r_f6_hz", float("nan")), float("nan"))),
            "main_f6_worst_hz": float(_safe_float(main_rolloff_meta.get("main_f6_worst_hz", float("nan")), float("nan"))),
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
            "main_usability_penalty": float("nan"),
            "main_l_f6_hz": float("nan"),
            "main_r_f6_hz": float("nan"),
            "main_f6_worst_hz": float("nan"),
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
    main_rolloff_meta = _direct_dac_main_rolloff_meta(bundle)

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
                main_rolloff_meta=main_rolloff_meta,
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
                main_rolloff_meta=main_rolloff_meta,
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
                main_rolloff_meta=main_rolloff_meta,
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
                main_rolloff_meta=main_rolloff_meta,
            ),
        )

    best_hz = _recommend_direct_dac_best_so_far(scores, fallback_hz=bundle.avr_crossover_hz)
    best_entry = dict(scores.get(round(best_hz, 4), scores.get(best_hz, {})) or {})

    # Log top-3 candidates and warn on close calls.
    _valid_scores = sorted(
        ((fc, float(d["score"])) for fc, d in scores.items() if np.isfinite(_safe_float(d.get("score"), float("nan")))),
        key=lambda x: x[1],
        reverse=True,
    )
    for _rank, (_fc, _sc) in enumerate(_valid_scores[:3], start=1):
        _e = scores[_fc]
        _LOG.debug(
            "Bass XO candidate #%d: %.1f Hz  score=%.4f  cancel=%.3f  ripple=%.1f dB  gd_rms=%.1f ms  feasibility=%s",
            _rank, _fc, _sc,
            float(_safe_float(_e.get("cancellation_risk"), float("nan"))),
            float(_safe_float(_e.get("overlap_ripple_db"), float("nan"))),
            float(_safe_float(_e.get("xo_gd_rms_mismatch_ms"), float("nan"))),
            _e.get("feasibility_class", "?"),
        )
    if len(_valid_scores) >= 2:
        _best_sc = _valid_scores[0][1]
        _second_sc = _valid_scores[1][1]
        _gap = abs(_best_sc - _second_sc)
        _ref = max(abs(_best_sc), 1e-9)
        if _gap / _ref < 0.05:
            _LOG.warning(
                "Bass XO recommendation uncertain: winner %.1f Hz (score %.4f) vs runner-up %.1f Hz (score %.4f) differ by only %.1f%%",
                _valid_scores[0][0], _best_sc, _valid_scores[1][0], _second_sc, 100.0 * _gap / _ref,
            )

    return {
        "recommended_hz": float(best_hz),
        "recommended_sub_lpf_hz": float(best_entry.get("sub_lpf_hz", best_hz)),
        "scores": scores,
        "feasibility_class": str(best_entry.get("feasibility_class", "marginal") or "marginal"),
        "feasibility_reason": str(best_entry.get("feasibility_reason", "") or ""),
        "dominant_channel": str(best_entry.get("dominant_channel", "unknown") or "unknown"),
    }
