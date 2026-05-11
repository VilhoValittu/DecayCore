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

import numpy as np

from .. import shared
from ..rank_score import attach_official_rank_score, calibrated_auto_quality

AUTO_MODE_PREFER_BASS_RESIDUAL_PEAK_HARD_GATE_MAX_DB = 12.0


def _bass_boost_support_db(metrics: dict | None) -> float:
    vals = []
    for key in ("bass_boost_20_200_db", "post_filter_boost_peak_db", "lf_boost_max_db"):
        value = shared._auto_safe_float(dict(metrics or {}).get(key, float("nan")), float("nan"))
        if np.isfinite(value):
            vals.append(float(max(0.0, value)))
    return float(max(vals)) if vals else 0.0


def _effective_residual_peak_hard_gate_db(
    metrics: dict | None,
    *,
    base_data: dict | None,
    default_gate: float,
) -> float:
    gate = shared._auto_safe_float(default_gate, float("nan"))
    if not np.isfinite(gate):
        return float("nan")
    if shared._auto_goal(base_data) != shared.AUTO_MODE_GOAL_FLAT:
        return float(gate)
    bass_boost = _bass_boost_support_db(metrics)
    if float(bass_boost) < 3.0:
        return float(gate)
    return float(min(AUTO_MODE_PREFER_BASS_RESIDUAL_PEAK_HARD_GATE_MAX_DB, float(gate) + float(bass_boost)))


def finalize_score_result_metrics(
    metrics_out: dict,
    *,
    base_data: dict | None,
    worst_residual_peak_raw_db: float,
    worst_residual_peak_db: float,
    stereo_policy_gate_failed: bool,
    rank_score: float,
    rank_components: dict,
) -> dict:
    bass_failed = bool(
        bool(dict(base_data or {}).get("bass_integration_enable", False))
        and str(metrics_out.get("bass_feasibility_class", "") or "").strip().lower() == "infeasible"
    )
    bass_reason = ""
    if bass_failed:
        bass_reason = str(metrics_out.get("bass_feasibility_reason", "") or "bass integration feasibility is infeasible")

    metrics_out["bass_integration_hard_gate_failed"] = bool(bass_failed)
    metrics_out["bass_integration_hard_gate_reason"] = str(bass_reason)

    hard_gate_failures = []
    rp_gate = _effective_residual_peak_hard_gate_db(
        metrics_out,
        base_data=base_data,
        default_gate=shared._auto_safe_float(metrics_out.get("residual_peak_hard_gate_db", float("nan")), float("nan")),
    )
    metrics_out["residual_peak_hard_gate_effective_db"] = float(rp_gate) if np.isfinite(rp_gate) else float("nan")
    rp_gate_value = shared._auto_safe_float(metrics_out.get("residual_peak_gate_value_db", float("nan")), float("nan"))
    rp_gate_source = str(metrics_out.get("residual_peak_gate_source", "") or "").strip().lower()
    if not np.isfinite(rp_gate_value):
        if np.isfinite(worst_residual_peak_raw_db):
            rp_gate_value = float(worst_residual_peak_raw_db)
            rp_gate_source = "raw_db"
        elif np.isfinite(worst_residual_peak_db):
            rp_gate_value = float(worst_residual_peak_db)
            rp_gate_source = "severity"
    if np.isfinite(rp_gate_value) and np.isfinite(rp_gate) and float(rp_gate_value) > float(rp_gate):
        hard_gate_failures.append("residual_peak_hard_gate" if rp_gate_source == "raw_db" else "residual_peak_severity_gate")
    if bass_failed:
        hard_gate_failures.append("bass_integration_infeasible_hard_gate")
    if bool(stereo_policy_gate_failed):
        hard_gate_failures.append("stereo_policy_gate_failed")
    if not np.isfinite(shared._auto_safe_float(metrics_out.get("rank_score", float("nan")), float("nan"))):
        hard_gate_failures.append("non_finite_rank_score")
    metrics_out["hard_gate_failures"] = list(dict.fromkeys(hard_gate_failures))
    metrics_out["hard_gate_reasons"] = list(metrics_out["hard_gate_failures"])
    metrics_out["hard_gate_failed"] = bool(metrics_out["hard_gate_failures"])
    display_q = calibrated_auto_quality(rank_score, metrics_out)
    metrics_out["display_score_100"] = display_q
    metrics_out["rank_score_official"] = display_q
    return attach_official_rank_score(metrics_out, components=rank_components)


__all__ = ["finalize_score_result_metrics"]
