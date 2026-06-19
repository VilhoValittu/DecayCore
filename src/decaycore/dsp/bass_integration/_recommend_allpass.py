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
    _get_pkg,
    _normalize_candidate_frequencies,
    _normalize_candidate_q_values,
    _safe_float,
    normalize_sub_combine_mode,
)


def _allpass_result(
    *,
    enabled: bool,
    freq_hz: float,
    q: float,
    baseline: dict[str, Any],
    optimized: dict[str, Any],
    improvement_score: float,
    reason: str,
) -> dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "freq_hz": float(freq_hz) if bool(enabled) else 0.0,
        "q": float(q) if bool(enabled) else 0.707,
        "baseline": baseline,
        "optimized": optimized,
        "improvement_score": float(improvement_score) if np.isfinite(improvement_score) else 0.0,
        "reason": str(reason),
    }


def _compute_direct_dac_metrics(
    pkg,
    bundle: BassIntegrationBundle,
    *,
    fc: float,
    profile: str,
    main_hpf_order: int,
    sub_lpf_order: int,
    sub_hp: float,
    sub_hpf_order: int,
    combine_mode_norm: str,
    sub_delay_ms: float,
    sub_polarity_invert: bool,
    sub_gain_trim_db: float,
    sub_lpf_hz: float | None,
    sub_allpass_freq_hz: float | None = None,
    sub_allpass_q: float | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "mode": "direct_dac",
        "main_hpf_order": int(main_hpf_order),
        "sub_lpf_order": int(sub_lpf_order),
        "sub_hpf_hz": float(sub_hp),
        "sub_hpf_order": int(sub_hpf_order),
        "sub_combine_mode": combine_mode_norm,
        "sub_delay_ms": float(sub_delay_ms),
        "sub_polarity_invert": bool(sub_polarity_invert),
        "sub_gain_trim_db": float(sub_gain_trim_db),
        "sub_lpf_hz": sub_lpf_hz,
    }
    if sub_allpass_freq_hz is not None:
        kwargs["sub_allpass_freq_hz"] = float(sub_allpass_freq_hz)
    if sub_allpass_q is not None:
        kwargs["sub_allpass_q"] = float(sub_allpass_q)
    return pkg.compute_final_bass_integration_metrics(
        bundle,
        float(fc),
        profile,
        **kwargs,
    )


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

    baseline_metrics = _compute_direct_dac_metrics(
        pkg,
        bundle,
        fc=float(fc),
        profile=profile,
        main_hpf_order=int(main_hpf_order),
        sub_lpf_order=int(sub_lpf_order),
        sub_hp=float(sub_hp),
        sub_hpf_order=int(sub_hpf_order),
        combine_mode_norm=combine_mode_norm,
        sub_delay_ms=float(sub_delay_ms),
        sub_polarity_invert=bool(sub_polarity_invert),
        sub_gain_trim_db=float(sub_gain_trim_db),
        sub_lpf_hz=sub_lpf_hz,
    )
    baseline = _final_metric_snapshot(baseline_metrics)
    baseline_score = _safe_float(baseline_metrics.get("objective", float("nan")), float("nan"))
    if (not np.isfinite(fc)) or fc <= 0.0 or fc <= (sub_hp + 1.0):
        return _allpass_result(
            enabled=False,
            freq_hz=0.0,
            q=0.707,
            baseline=baseline,
            optimized=baseline,
            improvement_score=0.0,
            reason="No meaningful improvement found.",
        )

    if _allpass_baseline_good_enough(baseline_metrics):
        return _allpass_result(
            enabled=False,
            freq_hz=0.0,
            q=0.707,
            baseline=baseline,
            optimized=baseline,
            improvement_score=0.0,
            reason="Baseline already good enough; auto allpass skipped.",
        )

    def _evaluate(freq_hz: float, q: float) -> dict[str, Any] | None:
        metrics = _compute_direct_dac_metrics(
            pkg,
            bundle,
            fc=float(fc),
            profile=profile,
            main_hpf_order=int(main_hpf_order),
            sub_lpf_order=int(sub_lpf_order),
            sub_hp=float(sub_hp),
            sub_hpf_order=int(sub_hpf_order),
            combine_mode_norm=combine_mode_norm,
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

    best_candidate = _run_allpass_candidate_search(
        fc=float(fc),
        sub_hp=float(sub_hp),
        evaluate=_evaluate,
    )
    if best_candidate is None:
        return _allpass_result(
            enabled=False,
            freq_hz=0.0,
            q=0.707,
            baseline=baseline,
            optimized=baseline,
            improvement_score=0.0,
            reason="No meaningful improvement found.",
        )

    optimized = _final_metric_snapshot(best_candidate or baseline_metrics)
    improvement = _allpass_improvement_metrics(
        baseline=baseline,
        optimized=optimized,
        baseline_score=baseline_score,
        optimized_score=_safe_float((best_candidate or {}).get("score", float("nan")), float("nan")),
    )
    enabled = _allpass_candidate_enabled(improvement)
    return _allpass_result(
        enabled=bool(enabled),
        freq_hz=float((best_candidate or {}).get("bass_allpass_freq_hz", 0.0)),
        q=float((best_candidate or {}).get("bass_allpass_q", 0.707)),
        baseline=baseline,
        optimized=optimized,
        improvement_score=float(improvement.get("improvement_score", 0.0) or 0.0),
        reason="Applied shared mono-sub allpass." if enabled else "No meaningful improvement found.",
    )


def _run_allpass_candidate_search(
    *,
    fc: float,
    sub_hp: float,
    evaluate,
) -> dict[str, Any] | None:
    coarse_freqs = _normalize_candidate_frequencies(float(fc) * mul for mul in DIRECT_DAC_ALLPASS_FREQ_MULTIPLIERS)
    coarse_qs = _normalize_candidate_q_values(DIRECT_DAC_ALLPASS_Q_CANDIDATES)
    best_candidate = _consider_allpass_candidates(
        freqs=coarse_freqs,
        qs=coarse_qs,
        current_best=None,
        evaluate=evaluate,
        sub_hp=float(sub_hp),
    )
    if best_candidate is None:
        return None
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
    best_candidate = _consider_allpass_candidates(
        freqs=refine_freqs,
        qs=refine_qs,
        current_best=best_candidate,
        evaluate=evaluate,
        sub_hp=float(sub_hp),
    )
    if best_candidate is None:
        return None
    freq_v = _safe_float(best_candidate.get("bass_allpass_freq_hz", float("nan")), float("nan"))
    if (not np.isfinite(freq_v)) or freq_v <= (float(sub_hp) + 1.0):
        return None
    return dict(best_candidate)


def _consider_allpass_candidates(
    *,
    freqs: tuple[float, ...],
    qs: tuple[float, ...],
    current_best: dict[str, Any] | None,
    evaluate,
    sub_hp: float,
) -> dict[str, Any] | None:
    best = current_best
    best_score = _safe_float((best or {}).get("score", float("nan")), float("nan"))
    for freq_hz in freqs:
        freq_v = _safe_float(freq_hz, float("nan"))
        if (not np.isfinite(freq_v)) or freq_v <= (float(sub_hp) + 1.0):
            continue
        for q in qs:
            cand = evaluate(freq_v, q)
            if cand is None:
                continue
            cand_score = _safe_float(cand.get("score", float("nan")), float("nan"))
            if (best is None) or (not np.isfinite(best_score)) or cand_score > best_score:
                best = cand
                best_score = cand_score
    return best


def _allpass_baseline_good_enough(baseline_metrics: dict) -> bool:
    baseline_cancel = _safe_float(baseline_metrics.get("bass_cancellation_risk", float("nan")), float("nan"))
    baseline_ripple = _safe_float(baseline_metrics.get("bass_overlap_ripple", float("nan")), float("nan"))
    baseline_gd = _safe_float(baseline_metrics.get("bass_xo_gd_mismatch_ms", float("nan")), float("nan"))
    return bool(
        np.isfinite(baseline_cancel) and baseline_cancel < 0.05
        and np.isfinite(baseline_ripple) and baseline_ripple < 1.5
        and np.isfinite(baseline_gd) and baseline_gd < 0.6
    )


def _allpass_metric_delta(lhs, rhs) -> float:
    left = _safe_float(lhs, float("nan"))
    right = _safe_float(rhs, float("nan"))
    if np.isfinite(left) and np.isfinite(right):
        return float(left - right)
    return float("nan")


def _allpass_improvement_metrics(
    *,
    baseline: dict[str, Any],
    optimized: dict[str, Any],
    baseline_score: float,
    optimized_score: float,
) -> dict[str, float]:
    improvement_score = (
        float(optimized_score - baseline_score)
        if np.isfinite(optimized_score) and np.isfinite(baseline_score)
        else float("nan")
    )
    cancel_improvement = _allpass_metric_delta(
        baseline.get("cancellation_risk", float("nan")),
        optimized.get("cancellation_risk", float("nan")),
    )
    ripple_improvement = _allpass_metric_delta(
        baseline.get("overlap_ripple_db", float("nan")),
        optimized.get("overlap_ripple_db", float("nan")),
    )
    gd_improvement = _allpass_metric_delta(
        baseline.get("xo_gd_mismatch_ms", float("nan")),
        optimized.get("xo_gd_mismatch_ms", float("nan")),
    )
    null_improvement = _allpass_metric_delta(
        baseline.get("null_severity", float("nan")),
        optimized.get("null_severity", float("nan")),
    )
    return {
        "improvement_score": float(improvement_score) if np.isfinite(improvement_score) else 0.0,
        "cancel_improvement": float(cancel_improvement),
        "ripple_improvement": float(ripple_improvement),
        "gd_improvement": float(gd_improvement),
        "null_improvement": float(null_improvement),
    }


def _allpass_candidate_enabled(improvement: dict[str, float]) -> bool:
    improvement_score = _safe_float(improvement.get("improvement_score", float("nan")), float("nan"))
    cancel_improvement = _safe_float(improvement.get("cancel_improvement", float("nan")), float("nan"))
    ripple_improvement = _safe_float(improvement.get("ripple_improvement", float("nan")), float("nan"))
    gd_improvement = _safe_float(improvement.get("gd_improvement", float("nan")), float("nan"))
    null_improvement = _safe_float(improvement.get("null_improvement", float("nan")), float("nan"))
    return bool(
        np.isfinite(improvement_score)
        and improvement_score >= float(DIRECT_DAC_ALLPASS_MIN_IMPROVEMENT_SCORE)
        and (
            (np.isfinite(cancel_improvement) and cancel_improvement >= float(DIRECT_DAC_ALLPASS_MIN_CANCEL_IMPROVEMENT))
            or (np.isfinite(ripple_improvement) and ripple_improvement >= float(DIRECT_DAC_ALLPASS_MIN_RIPPLE_IMPROVEMENT_DB))
            or (np.isfinite(gd_improvement) and gd_improvement >= float(DIRECT_DAC_ALLPASS_MIN_GD_IMPROVEMENT_MS))
            or (np.isfinite(null_improvement) and null_improvement >= 0.25)
        )
    )
