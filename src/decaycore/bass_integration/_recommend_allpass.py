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

from ...io.measurement_bundle import BassIntegrationBundle
from ._constants import (
    DIRECT_DAC_ALLPASS_FREQ_MULTIPLIERS,
    DIRECT_DAC_ALLPASS_MIN_CANCEL_IMPROVEMENT,
    DIRECT_DAC_ALLPASS_MIN_GD_IMPROVEMENT_MS,
    DIRECT_DAC_ALLPASS_MIN_IMPROVEMENT_SCORE,
    DIRECT_DAC_ALLPASS_MIN_RIPPLE_IMPROVEMENT_DB,
    DIRECT_DAC_ALLPASS_Q_CANDIDATES,
    DIRECT_DAC_ALLPASS_REFINE_FREQ_FACTORS,
    DIRECT_DAC_ALLPASS_REFINE_Q_FACTORS,
)
from ._final_metrics import _final_metric_snapshot
from ._utils import (
    _normalize_candidate_frequencies,
    _normalize_candidate_q_values,
    _safe_float,
    normalize_sub_combine_mode,
)


def _get_pkg():
    """Return the bass_integration package module for patchable attribute lookup."""
    return sys.modules[__name__.rsplit(".", 1)[0]]


def recommend_direct_dac_allpass(
    bundle: BassIntegrationBundle,
    *,
    fc_hz: float,
    profile: str,
    main_hpf_order: int,
    sub_lpf_order: int,
    sub_hpf_hz: float,
    sub_hpf_order: int,
    sub_combine_mode: str = "average",
    sub_delay_ms: float = 0.0,
    sub_polarity_invert: bool = False,
    sub_gain_trim_db: float = 0.0,
    sub_lpf_hz: float | None = None,
) -> dict[str, Any]:
    fc = _safe_float(fc_hz, 80.0)
    sub_hp = max(0.0, _safe_float(sub_hpf_hz, 20.0))
    combine_mode_norm = normalize_sub_combine_mode(sub_combine_mode)
    pkg = _get_pkg()

    baseline_metrics = pkg.compute_final_bass_integration_metrics(
        bundle,
        float(fc),
        profile,
        mode="direct_dac",
        main_hpf_order=int(main_hpf_order),
        sub_lpf_order=int(sub_lpf_order),
        sub_hpf_hz=float(sub_hp),
        sub_hpf_order=int(sub_hpf_order),
        sub_combine_mode=combine_mode_norm,
        sub_delay_ms=float(sub_delay_ms),
        sub_polarity_invert=bool(sub_polarity_invert),
        sub_gain_trim_db=float(sub_gain_trim_db),
        sub_lpf_hz=sub_lpf_hz,
    )
    baseline = _final_metric_snapshot(baseline_metrics)
    baseline_score = _safe_float(baseline_metrics.get("objective", float("nan")), float("nan"))
    if (not np.isfinite(fc)) or fc <= 0.0 or fc <= (sub_hp + 1.0):
        return {
            "enabled": False,
            "freq_hz": 0.0,
            "q": 0.707,
            "baseline": baseline,
            "optimized": baseline,
            "improvement_score": 0.0,
            "reason": "No meaningful improvement found.",
        }

    # Early-exit: if baseline is already good enough, skip candidate search
    _bl_cancel = _safe_float(baseline_metrics.get("bass_cancellation_risk", float("nan")), float("nan"))
    _bl_ripple = _safe_float(baseline_metrics.get("bass_overlap_ripple", float("nan")), float("nan"))
    _bl_gd = _safe_float(baseline_metrics.get("bass_xo_gd_mismatch_ms", float("nan")), float("nan"))
    if (
        np.isfinite(_bl_cancel) and _bl_cancel < 0.08
        and np.isfinite(_bl_ripple) and _bl_ripple < 2.0
        and np.isfinite(_bl_gd) and _bl_gd < 0.8
    ):
        return {
            "enabled": False,
            "freq_hz": 0.0,
            "q": 0.707,
            "baseline": baseline,
            "optimized": baseline,
            "improvement_score": 0.0,
            "reason": "Baseline already good enough; auto allpass skipped.",
        }

    def _evaluate(freq_hz: float, q: float) -> dict[str, Any] | None:
        metrics = pkg.compute_final_bass_integration_metrics(
            bundle,
            float(fc),
            profile,
            mode="direct_dac",
            main_hpf_order=int(main_hpf_order),
            sub_lpf_order=int(sub_lpf_order),
            sub_hpf_hz=float(sub_hp),
            sub_hpf_order=int(sub_hpf_order),
            sub_combine_mode=combine_mode_norm,
            sub_delay_ms=float(sub_delay_ms),
            sub_polarity_invert=bool(sub_polarity_invert),
            sub_gain_trim_db=float(sub_gain_trim_db),
            sub_lpf_hz=sub_lpf_hz,
            sub_allpass_freq_hz=float(freq_hz),
            sub_allpass_q=float(q),
        )
        score = _safe_float(metrics.get("objective", float("nan")), float("nan"))
        if not np.isfinite(score):
            return None
        out = dict(metrics)
        out["score"] = float(score)
        return out

    coarse_freqs = _normalize_candidate_frequencies(float(fc) * mul for mul in DIRECT_DAC_ALLPASS_FREQ_MULTIPLIERS)
    coarse_qs = _normalize_candidate_q_values(DIRECT_DAC_ALLPASS_Q_CANDIDATES)
    best_candidate: dict[str, Any] | None = None

    def _consider_candidates(
        freqs: tuple[float, ...],
        qs: tuple[float, ...],
        current_best: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        best = current_best
        best_score = _safe_float((best or {}).get("score", float("nan")), float("nan"))
        for freq_hz in freqs:
            freq_v = _safe_float(freq_hz, float("nan"))
            if (not np.isfinite(freq_v)) or freq_v <= (sub_hp + 1.0):
                continue
            for q in qs:
                cand = _evaluate(freq_v, q)
                if cand is None:
                    continue
                cand_score = _safe_float(cand.get("score", float("nan")), float("nan"))
                if (best is None) or (not np.isfinite(best_score)) or cand_score > best_score:
                    best = cand
                    best_score = cand_score
        return best

    best_candidate = _consider_candidates(coarse_freqs, coarse_qs, None)
    if best_candidate is None:
        return {
            "enabled": False,
            "freq_hz": 0.0,
            "q": 0.707,
            "baseline": baseline,
            "optimized": baseline,
            "improvement_score": 0.0,
            "reason": "No meaningful improvement found.",
        }

    refine_freqs = _normalize_candidate_frequencies(
        float(best_candidate.get("bass_allpass_freq_hz", fc)) * factor
        for factor in DIRECT_DAC_ALLPASS_REFINE_FREQ_FACTORS
    )
    refine_qs = _normalize_candidate_q_values(
        float(
            np.clip(
                float(best_candidate.get("bass_allpass_q", 1.0)) * factor,
                float(min(DIRECT_DAC_ALLPASS_Q_CANDIDATES)),
                float(max(DIRECT_DAC_ALLPASS_Q_CANDIDATES)),
            )
        )
        for factor in DIRECT_DAC_ALLPASS_REFINE_Q_FACTORS
    )
    best_candidate = _consider_candidates(refine_freqs, refine_qs, best_candidate)

    optimized = _final_metric_snapshot(best_candidate or baseline_metrics)
    optimized_score = _safe_float((best_candidate or {}).get("score", float("nan")), float("nan"))
    improvement_score = (
        float(optimized_score - baseline_score)
        if np.isfinite(optimized_score) and np.isfinite(baseline_score)
        else float("nan")
    )
    baseline_cancel = _safe_float(baseline.get("cancellation_risk", float("nan")), float("nan"))
    optimized_cancel = _safe_float(optimized.get("cancellation_risk", float("nan")), float("nan"))
    baseline_ripple = _safe_float(baseline.get("overlap_ripple_db", float("nan")), float("nan"))
    optimized_ripple = _safe_float(optimized.get("overlap_ripple_db", float("nan")), float("nan"))
    baseline_gd = _safe_float(baseline.get("xo_gd_mismatch_ms", float("nan")), float("nan"))
    optimized_gd = _safe_float(optimized.get("xo_gd_mismatch_ms", float("nan")), float("nan"))
    baseline_null = _safe_float(baseline.get("null_severity", float("nan")), float("nan"))
    optimized_null = _safe_float(optimized.get("null_severity", float("nan")), float("nan"))
    cancel_improvement = (
        float(baseline_cancel - optimized_cancel)
        if np.isfinite(baseline_cancel) and np.isfinite(optimized_cancel)
        else float("nan")
    )
    ripple_improvement = (
        float(baseline_ripple - optimized_ripple)
        if np.isfinite(baseline_ripple) and np.isfinite(optimized_ripple)
        else float("nan")
    )
    gd_improvement = (
        float(baseline_gd - optimized_gd)
        if np.isfinite(baseline_gd) and np.isfinite(optimized_gd)
        else float("nan")
    )
    null_improvement = (
        float(baseline_null - optimized_null)
        if np.isfinite(baseline_null) and np.isfinite(optimized_null)
        else float("nan")
    )
    enabled = bool(
        np.isfinite(improvement_score)
        and improvement_score >= float(DIRECT_DAC_ALLPASS_MIN_IMPROVEMENT_SCORE)
        and (
            (np.isfinite(cancel_improvement) and cancel_improvement >= float(DIRECT_DAC_ALLPASS_MIN_CANCEL_IMPROVEMENT))
            or (np.isfinite(ripple_improvement) and ripple_improvement >= float(DIRECT_DAC_ALLPASS_MIN_RIPPLE_IMPROVEMENT_DB))
            or (np.isfinite(gd_improvement) and gd_improvement >= float(DIRECT_DAC_ALLPASS_MIN_GD_IMPROVEMENT_MS))
            or (np.isfinite(null_improvement) and null_improvement >= 0.25)
        )
    )
    return {
        "enabled": bool(enabled),
        "freq_hz": float((best_candidate or {}).get("bass_allpass_freq_hz", 0.0)) if enabled else 0.0,
        "q": float((best_candidate or {}).get("bass_allpass_q", 0.707)) if enabled else 0.707,
        "baseline": baseline,
        "optimized": optimized,
        "improvement_score": float(improvement_score) if np.isfinite(improvement_score) else 0.0,
        "reason": "Applied shared mono-sub allpass." if enabled else "No meaningful improvement found.",
    }
