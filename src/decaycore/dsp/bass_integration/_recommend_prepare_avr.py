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

import sys
from typing import Any

import numpy as np

from ...auto_mode.shared import _auto_bass_integration_profile_weights
from ...io.measurement_bundle import BassIntegrationBundle
from ._constants import (
    AVR_CROSSOVER_CANDIDATES,
    AVR_LFE_MAIN_ALIGNMENT_COARSE_DELAYS_MS,
    AVR_LFE_MAIN_ALIGNMENT_COARSE_GAINS_DB,
    AVR_LFE_MAIN_ALIGNMENT_REFINE_DELTA_DB,
    AVR_LFE_MAIN_ALIGNMENT_REFINE_DELTA_MS,
)
from ._final_metrics import _final_metric_snapshot
from ._recommend_alignment import (
    _alignment_subset,
    _best_metric_grid_row,
    _evaluate_metric_grid,
    _feasibility_rank,
    _nearest_from_candidates,
)
from ._utils import _safe_float, normalize_sub_combine_mode


def _get_pkg():
    """Return the bass_integration package module for patchable attribute lookup."""
    return sys.modules[__name__.rsplit(".", 1)[0]]


def recommend_avr_lfe_main_prepare(
    bundle: BassIntegrationBundle,
    candidates: tuple[float, ...] = AVR_CROSSOVER_CANDIDATES,
    profile: str = "safe",
    *,
    sub_combine_mode: str = "average",
) -> dict[str, Any]:
    weights = _auto_bass_integration_profile_weights(profile)
    w_main_act = float(weights.get("main_activity", 6.0))
    combine_mode_norm = normalize_sub_combine_mode(sub_combine_mode)

    def _score(fc: float, metrics: dict[str, Any]) -> float:
        score = _safe_float(metrics.get("objective", float("nan")), float("nan"))
        _pkg = _get_pkg()
        l_drop = _pkg._main_guard_band_drop_db(bundle.l_main, fc)
        r_drop = _pkg._main_guard_band_drop_db(bundle.r_main, fc)
        drop_vals = [v for v in (l_drop, r_drop) if np.isfinite(v)]
        if np.isfinite(score) and drop_vals:
            avg_drop = float(np.mean(np.asarray(drop_vals, dtype=float)))
            score -= w_main_act * max(0.0, avg_drop) / 12.0
        return float(score)

    def _eval(fc: float, delay_ms: float, polarity: bool, gain_db: float) -> tuple[float, dict[str, Any]]:
        metrics = _get_pkg().compute_final_bass_integration_metrics(
            bundle,
            float(fc),
            profile,
            mode="avr_lfe_main_decomposed",
            sub_combine_mode=combine_mode_norm,
            sub_delay_ms=float(delay_ms),
            sub_polarity_invert=bool(polarity),
            sub_gain_trim_db=float(gain_db),
        )
        return _score(float(fc), metrics), dict(metrics)

    baseline_score, baseline_metrics = _eval(float(bundle.avr_crossover_hz or 80.0), 0.0, False, 0.0)
    best_score = float(baseline_score)
    best_metrics = dict(baseline_metrics)
    best_fc = float(bundle.avr_crossover_hz or 80.0)
    best_delay = 0.0
    best_polarity = False
    best_gain = 0.0
    scores: dict[float, dict[str, float | str]] = {}
    _pkg0 = _get_pkg()
    coarse_delays = _alignment_subset(
        _pkg0.DIRECT_DAC_ALIGNMENT_DELAY_CANDIDATES_MS,
        AVR_LFE_MAIN_ALIGNMENT_COARSE_DELAYS_MS,
    )
    coarse_gains = _alignment_subset(
        _pkg0.DIRECT_DAC_ALIGNMENT_GAIN_CANDIDATES_DB,
        AVR_LFE_MAIN_ALIGNMENT_COARSE_GAINS_DB,
    )

    for fc_raw in candidates:
        fc = float(fc_raw)
        best_fc_metrics: dict[str, Any] | None = None
        best_fc_score = float("nan")
        best_fc_delay = 0.0
        best_fc_polarity = False
        best_fc_gain = 0.0
        rows = _evaluate_metric_grid(
            [
                (float(fc), float(delay_ms), bool(polarity), float(gain_db))
                for polarity in (False, True)
                for delay_ms in coarse_delays
                for gain_db in coarse_gains
            ],
            _eval,
        )
        fc_candidate, best_fc_score, best_fc_metrics = _best_metric_grid_row(rows, current_score=float("nan"))
        if fc_candidate is not None and best_fc_metrics is not None:
            best_fc_delay = float(fc_candidate[1])
            best_fc_polarity = bool(fc_candidate[2])
            best_fc_gain = float(fc_candidate[3])
            if (not np.isfinite(best_score)) or float(best_fc_score) > float(best_score):
                best_score = float(best_fc_score)
                best_metrics = dict(best_fc_metrics)
                best_fc = float(fc)
                best_delay = float(best_fc_delay)
                best_polarity = bool(best_fc_polarity)
                best_gain = float(best_fc_gain)
        entry_metrics = dict(best_fc_metrics or {})
        l_drop = _get_pkg()._main_guard_band_drop_db(bundle.l_main, fc)
        r_drop = _get_pkg()._main_guard_band_drop_db(bundle.r_main, fc)
        drop_vals = [v for v in (l_drop, r_drop) if np.isfinite(v)]
        scores[fc] = {
            "score": float(best_fc_score),
            "cancellation_risk": float(_safe_float(entry_metrics.get("bass_cancellation_risk", float("nan")), float("nan"))),
            "overlap_ripple_db": float(_safe_float(entry_metrics.get("bass_overlap_ripple", float("nan")), float("nan"))),
            "sub_dominance_db": float(_safe_float(entry_metrics.get("bass_sub_dominance", float("nan")), float("nan"))),
            "null_severity": float(_safe_float(entry_metrics.get("bass_null_severity", float("nan")), float("nan"))),
            "overlap_ripple_delta_db": float(
                _safe_float(entry_metrics.get("bass_overlap_ripple_delta_db", float("nan")), float("nan"))
            ),
            "sub_dominance_delta_db": float(
                _safe_float(entry_metrics.get("bass_sub_dominance_delta_db", float("nan")), float("nan"))
            ),
            "xo_gd_mismatch_delta_ms": float(
                _safe_float(entry_metrics.get("bass_xo_gd_mismatch_delta_ms", float("nan")), float("nan"))
            ),
            "dominant_channel": str(entry_metrics.get("bass_dominant_channel", "unknown") or "unknown"),
            "feasibility_class": str(entry_metrics.get("bass_feasibility_class", "unknown") or "unknown"),
            "feasibility_reason": str(entry_metrics.get("bass_feasibility_reason", "") or ""),
            "main_activity_drop_db": float(np.mean(np.asarray(drop_vals, dtype=float))) if drop_vals else float("nan"),
            "sub_delay_ms": float(best_fc_delay),
            "sub_polarity_invert": bool(best_fc_polarity),
            "sub_gain_trim_db": float(best_fc_gain),
        }

    _pkg1 = _get_pkg()
    refine_delays = _alignment_subset(
        _pkg1.DIRECT_DAC_ALIGNMENT_DELAY_CANDIDATES_MS,
        tuple(float(best_delay) + float(delta) for delta in AVR_LFE_MAIN_ALIGNMENT_REFINE_DELTA_MS),
    )
    refine_gains = _alignment_subset(
        _pkg1.DIRECT_DAC_ALIGNMENT_GAIN_CANDIDATES_DB,
        tuple(float(best_gain) + float(delta) for delta in AVR_LFE_MAIN_ALIGNMENT_REFINE_DELTA_DB),
    )
    refine_fcs = tuple(
        sorted(
            {
                float(_nearest_from_candidates(best_fc + delta, candidates))
                for delta in (-10.0, -5.0, 0.0, 5.0, 10.0)
            }
        )
    )
    refine_rows = _evaluate_metric_grid(
        [
            (float(fc), float(delay_ms), bool(best_polarity), float(gain_db))
            for fc in refine_fcs
            for delay_ms in refine_delays
            for gain_db in refine_gains
        ],
        _eval,
    )
    refine_candidate, refine_score, refine_metrics = _best_metric_grid_row(refine_rows, current_score=best_score)
    if refine_candidate is not None and refine_metrics is not None:
        best_score = float(refine_score)
        best_metrics = dict(refine_metrics)
        best_fc = float(refine_candidate[0])
        best_delay = float(refine_candidate[1])
        best_polarity = bool(refine_candidate[2])
        best_gain = float(refine_candidate[3])

    baseline_snap = _final_metric_snapshot(baseline_metrics)
    optimized_snap = _final_metric_snapshot(best_metrics)
    improvement_score = (
        float(best_score - baseline_score)
        if np.isfinite(best_score) and np.isfinite(baseline_score)
        else float("nan")
    )
    base_cancel = _safe_float(baseline_snap.get("cancellation_risk", float("nan")), float("nan"))
    opt_cancel = _safe_float(optimized_snap.get("cancellation_risk", float("nan")), float("nan"))
    base_ripple = _safe_float(baseline_snap.get("overlap_ripple_db", float("nan")), float("nan"))
    opt_ripple = _safe_float(optimized_snap.get("overlap_ripple_db", float("nan")), float("nan"))
    base_feas = str(baseline_snap.get("feasibility_class", "unknown") or "unknown")
    opt_feas = str(optimized_snap.get("feasibility_class", "unknown") or "unknown")

    cancel_worsened = np.isfinite(base_cancel) and np.isfinite(opt_cancel) and opt_cancel > base_cancel + 0.15
    ripple_worsened = np.isfinite(base_ripple) and np.isfinite(opt_ripple) and opt_ripple > base_ripple + 1.5
    feasibility_dropped = _feasibility_rank(opt_feas) < _feasibility_rank(base_feas)
    high_ripple_baseline = (
        str(base_feas).strip().lower() == "infeasible"
        or (np.isfinite(base_ripple) and float(base_ripple) > 12.0)
    )
    ripple_improved = np.isfinite(base_ripple) and np.isfinite(opt_ripple) and (base_ripple - opt_ripple) >= 0.5
    feasibility_improved = _feasibility_rank(opt_feas) > _feasibility_rank(base_feas)
    applied = bool(
        np.isfinite(improvement_score)
        and improvement_score >= 0.03
        and not cancel_worsened
        and not ripple_worsened
        and not feasibility_dropped
        and (not high_ripple_baseline or feasibility_improved or ripple_improved)
        and (
            abs(best_delay) > 1e-9
            or bool(best_polarity)
            or abs(best_gain) > 1e-9
            or abs(best_fc - float(bundle.avr_crossover_hz or 80.0)) > 1e-9
        )
    )
    chosen_metrics = best_metrics if applied else baseline_metrics
    chosen_snap = _final_metric_snapshot(chosen_metrics)
    chosen_fc = float(best_fc if applied else float(bundle.avr_crossover_hz or 80.0))
    return {
        "applied": bool(applied),
        "backend": "builtin",
        "recommended_hz": float(chosen_fc),
        "sub_delay_ms": float(best_delay if applied else 0.0),
        "sub_polarity_invert": bool(best_polarity if applied else False),
        "sub_gain_trim_db": float(best_gain if applied else 0.0),
        "baseline": dict(baseline_snap),
        "optimized": dict(chosen_snap),
        "scores": scores,
        "improvement_score": float(improvement_score) if applied and np.isfinite(improvement_score) else 0.0,
        "reason": (
            "Applied AVR LFE+Main shared-sub crossover/polarity/delay/gain alignment."
            if applied
            else "Baseline AVR LFE+Main alignment kept."
        ),
        "feasibility_class": str(chosen_snap.get("feasibility_class", "unknown") or "unknown"),
        "feasibility_reason": str(chosen_snap.get("feasibility_reason", "") or ""),
        "dominant_channel": str(chosen_snap.get("dominant_channel", "unknown") or "unknown"),
    }
