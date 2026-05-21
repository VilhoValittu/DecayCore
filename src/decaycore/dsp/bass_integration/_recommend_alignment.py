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
)
from ...io.measurement_bundle import BassIntegrationBundle
from ._constants import (
    DIRECT_DAC_ALIGNMENT_MIN_IMPROVEMENT_SCORE,
)
from ._final_metrics import _final_metric_snapshot
from ._utils import _get_bass_integration_pkg, _safe_float


def _get_pkg():
    """Return the bass_integration package module for patchable attribute lookup."""
    return _get_bass_integration_pkg(__name__)


def _feasibility_rank(value: Any) -> int:
    return {"unknown": 0, "infeasible": 1, "marginal": 2, "good": 3}.get(
        str(value or "unknown").strip().lower(),
        0,
    )


def _nearest_from_candidates(value: float, candidates: tuple[float, ...]) -> float:
    vals = [float(v) for v in candidates if np.isfinite(_safe_float(v, float("nan")))]
    if not vals:
        return float(value)
    return float(min(vals, key=lambda item: abs(float(item) - float(value))))


def _alignment_subset(
    candidates: tuple[float, ...],
    preferred: tuple[float, ...],
) -> tuple[float, ...]:
    vals = tuple(float(v) for v in candidates if np.isfinite(_safe_float(v, float("nan"))))
    if not vals:
        return (0.0,)
    out = {float(_nearest_from_candidates(0.0, vals))}
    for value in preferred:
        out.add(float(_nearest_from_candidates(float(value), vals)))
    return tuple(sorted(out))


def _evaluate_metric_grid(
    candidates: list[tuple[Any, ...]] | tuple[tuple[Any, ...], ...],
    eval_fn,
) -> list[tuple[tuple[Any, ...], float, dict[str, Any]]]:
    """Evaluate a candidate grid in a stable order and keep scalar tie semantics."""
    rows: list[tuple[tuple[Any, ...], float, dict[str, Any]]] = []
    for candidate in candidates:
        score, metrics = eval_fn(*candidate)
        rows.append((tuple(candidate), float(score), dict(metrics or {})))
    return rows


def _best_metric_grid_row(
    rows: list[tuple[tuple[Any, ...], float, dict[str, Any]]],
    *,
    current_score: float = float("nan"),
) -> tuple[tuple[Any, ...] | None, float, dict[str, Any] | None]:
    best_score = float(current_score)
    if not rows:
        return None, float(best_score), None
    scores = np.asarray([float(score) for _candidate, score, _metrics in rows], dtype=float)
    finite = np.isfinite(scores)
    if not bool(np.any(finite)):
        return None, float(best_score), None
    eligible = finite if not np.isfinite(best_score) else (finite & (scores > float(best_score)))
    if not bool(np.any(eligible)):
        return None, float(best_score), None
    masked_scores = np.where(eligible, scores, float("-inf"))
    best_idx = int(np.argmax(masked_scores))
    candidate, score, metrics = rows[best_idx]
    return tuple(candidate), float(score), dict(metrics)


def recommend_direct_dac_alignment(
    bundle: BassIntegrationBundle,
    *,
    fc_hz: float,
    profile: str,
    main_hpf_order: int,
    sub_lpf_order: int,
    sub_hpf_hz: float,
    sub_hpf_order: int,
    sub_combine_mode: str = "average",
    sub_lpf_hz: float | None = None,
    guard_lo_ratio: float = AUTO_MODE_BASS_INTEGRATION_GUARD_LO_RATIO,
    guard_hi_ratio: float = AUTO_MODE_BASS_INTEGRATION_GUARD_HI_RATIO,
) -> dict[str, Any]:
    pkg = _get_pkg()
    _DELAY_CANDIDATES = pkg.DIRECT_DAC_ALIGNMENT_DELAY_CANDIDATES_MS
    _GAIN_CANDIDATES = pkg.DIRECT_DAC_ALIGNMENT_GAIN_CANDIDATES_DB
    _DELAY_LO = float(min(_DELAY_CANDIDATES))
    _DELAY_HI = float(max(_DELAY_CANDIDATES))
    _GAIN_LO = float(min(_GAIN_CANDIDATES))
    _GAIN_HI = float(max(_GAIN_CANDIDATES))

    def _dedup(seq: list[float]) -> list[float]:
        seen: set[float] = set()
        out: list[float] = []
        for v in seq:
            k = round(float(v), 4)
            if k not in seen:
                seen.add(k)
                out.append(float(v))
        return out

    def _eval(polarity: bool, delay_ms: float, gain_db: float) -> tuple[float, dict]:
        m = pkg.compute_final_bass_integration_metrics(
            bundle,
            fc_hz,
            profile,
            mode="direct_dac",
            main_hpf_order=int(main_hpf_order),
            sub_lpf_order=int(sub_lpf_order),
            sub_hpf_hz=float(sub_hpf_hz),
            sub_hpf_order=int(sub_hpf_order),
            sub_combine_mode=sub_combine_mode,
            sub_delay_ms=float(delay_ms),
            sub_polarity_invert=bool(polarity),
            sub_gain_trim_db=float(gain_db),
            sub_lpf_hz=sub_lpf_hz,
            guard_lo_ratio=float(guard_lo_ratio),
            guard_hi_ratio=float(guard_hi_ratio),
        )
        return _safe_float(m.get("objective", float("nan")), float("nan")), m

    baseline_score, baseline_metrics = _eval(False, 0.0, 0.0)
    best_score = float(baseline_score)
    best_metrics = dict(baseline_metrics)
    best_delay = 0.0
    best_polarity = False
    best_gain = 0.0

    # Phase A: search the documented coarse candidate grid across the full range.
    _COARSE_DELAYS = _DELAY_CANDIDATES
    _COARSE_GAINS = _GAIN_CANDIDATES
    coarse_rows = _evaluate_metric_grid(
        [
            (bool(polarity), float(delay_ms), float(gain_db))
            for polarity in (False, True)
            for delay_ms in _COARSE_DELAYS
            for gain_db in _COARSE_GAINS
        ],
        _eval,
    )
    coarse_candidate, coarse_score, coarse_metrics = _best_metric_grid_row(coarse_rows, current_score=best_score)
    if coarse_candidate is not None and coarse_metrics is not None:
        best_score, best_metrics = float(coarse_score), dict(coarse_metrics)
        best_polarity = bool(coarse_candidate[0])
        best_delay = float(coarse_candidate[1])
        best_gain = float(coarse_candidate[2])

    # Phase B: refine around best coarse result
    refine_delays = _dedup(
        [float(np.clip(best_delay + s, _DELAY_LO, _DELAY_HI)) for s in (-10.0, -5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 10.0)]
    )
    refine_gains = _dedup(
        [float(np.clip(best_gain + s, _GAIN_LO, _GAIN_HI)) for s in (-6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0)]
    )
    refine_rows = _evaluate_metric_grid(
        [(bool(best_polarity), float(delay_ms), float(gain_db)) for delay_ms in refine_delays for gain_db in refine_gains],
        _eval,
    )
    refine_candidate, refine_score, refine_metrics = _best_metric_grid_row(refine_rows, current_score=best_score)
    if refine_candidate is not None and refine_metrics is not None:
        best_score, best_metrics = float(refine_score), dict(refine_metrics)
        best_delay, best_gain = float(refine_candidate[1]), float(refine_candidate[2])

    improvement_score = (
        float(best_score - baseline_score)
        if np.isfinite(best_score) and np.isfinite(baseline_score)
        else float("nan")
    )
    applied = bool(
        np.isfinite(improvement_score)
        and improvement_score >= float(DIRECT_DAC_ALIGNMENT_MIN_IMPROVEMENT_SCORE)
        and (
            abs(best_delay) > 1e-6
            or bool(best_polarity)
            or abs(best_gain) > 1e-6
        )
    )
    chosen_metrics = best_metrics if applied else baseline_metrics
    return {
        "applied": bool(applied),
        "sub_delay_ms": float(best_delay if applied else 0.0),
        "sub_polarity_invert": bool(best_polarity if applied else False),
        "sub_gain_trim_db": float(best_gain if applied else 0.0),
        "baseline": _final_metric_snapshot(baseline_metrics),
        "optimized": _final_metric_snapshot(chosen_metrics),
        "improvement_score": float(improvement_score) if np.isfinite(improvement_score) else 0.0,
        "reason": (
            "Applied shared mono-sub polarity/delay/gain alignment."
            if applied
            else "Baseline shared mono-sub alignment kept."
        ),
    }
