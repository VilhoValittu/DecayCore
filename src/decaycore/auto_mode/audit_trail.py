# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Structured automatic-mode winner audit trail.

The audit trail is reporting metadata only. It must not alter scoring,
candidate eligibility, cache identity, or DSP output.
"""

from __future__ import annotations

import math
from typing import Any

from .rank_score import attach_official_rank_score, official_rank_score
from .scoring_ranking import _auto_hard_gate_diagnostic
from .shared import _auto_safe_float

AUTO_MODE_AUDIT_TRAIL_SCHEMA_VERSION = 1


def _finite_float(value: Any) -> float:
    out = float(_auto_safe_float(value, float("nan")))
    return float(out) if math.isfinite(out) else float("nan")


def _clean_dict(value: Any) -> dict:
    return dict(value or {}) if isinstance(value, dict) else {}


def _clean_list(value: Any) -> list:
    return list(value or []) if isinstance(value, (list, tuple)) else []


def _dedupe_text(items: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in list(items or []):
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        out.append(text)
        seen.add(text)
    return out


def _candidate_summary(candidate: dict, *, goal: str) -> dict:
    metrics = attach_official_rank_score(_clean_dict(candidate.get("metrics", candidate)))
    diag = _auto_hard_gate_diagnostic(metrics, goal=goal)
    rank = official_rank_score(metrics)
    return {
        "rank_score_official": float(rank) if math.isfinite(rank) else float("nan"),
        "avg_score": _finite_float(metrics.get("avg_score")),
        "residual_peak_db": _finite_float(diag.get("residual_peak_db")),
        "residual_peak_hard_gate_db": _finite_float(diag.get("residual_peak_hard_gate_db")),
        "hard_gate_failed": bool(diag.get("hard_gate_failed", False)),
        "hard_gate_reasons": _dedupe_text(_clean_list(diag.get("hard_gate_reasons"))),
    }


def _build_search_section(
    *,
    optimizer_backend: str,
    phase1_ok: int,
    phase2_ok: int,
    phase1_tried: int,
    phase2_tried: int,
    trials_total: int | None,
    trials_ok: int | None,
    phase1_plateau_hit: bool,
    phase2_plateau_hit: bool,
    phase3_total: int = 0,
    phase3_ok: int = 0,
    phase4_steps: dict | None = None,
    fs_v: int = 0,
    taps_v: int = 0,
) -> dict:
    total = int(trials_total) if trials_total is not None else int(phase1_tried + phase2_tried + phase3_total)
    ok = int(trials_ok) if trials_ok is not None else int(phase1_ok + phase2_ok + phase3_ok)
    return {
        "optimizer_backend": str(optimizer_backend or "builtin"),
        "trials_total": int(total),
        "trials_ok": int(ok),
        "trials_phase1_total": int(phase1_tried),
        "trials_phase1_ok": int(phase1_ok),
        "trials_phase2_total": int(phase2_tried),
        "trials_phase2_ok": int(phase2_ok),
        "trials_phase3_total": int(phase3_total),
        "trials_phase3_ok": int(phase3_ok),
        "phase1_plateau_hit": bool(phase1_plateau_hit),
        "phase2_plateau_hit": bool(phase2_plateau_hit),
        "phase4_steps": _clean_dict(phase4_steps),
        "search_fs": int(fs_v),
        "search_taps": int(taps_v),
    }


def _build_polish_section(polish_meta: dict | None) -> dict:
    meta = _clean_dict(polish_meta)
    keys = (
        "phase_limit_winner_polish",
        "mag_c_min_winner_polish",
        "low_bass_cut_winner_polish",
        "hpf_winner_polish",
        "excess_phase_strength_winner_polish",
        "residual_peak_winner_polish",
        "tdc_strength_winner_polish",
        "stereo_policy_refine",
    )
    out = {key: _clean_dict(meta.get(key)) for key in keys if isinstance(meta.get(key), dict)}
    out["applied"] = [
        key
        for key, value in out.items()
        if isinstance(value, dict) and bool(value.get("applied", False))
    ]
    return out


def _build_hard_gate_section(
    *,
    best_metrics: dict,
    goal: str,
    residual_peak_safety_override_meta: dict | None,
) -> dict:
    diag = _auto_hard_gate_diagnostic(best_metrics, goal=goal)
    explicit_failures = _clean_list(best_metrics.get("hard_gate_failures", best_metrics.get("hard_gate_reasons", [])))
    failures = _dedupe_text(explicit_failures + _clean_list(diag.get("hard_gate_reasons")))
    override = _clean_dict(residual_peak_safety_override_meta)
    hard_failed = bool(diag.get("hard_gate_failed", False) or failures)
    safe_count = int(override.get("safe_candidate_count", 0) or 0)
    candidate_count = int(override.get("candidate_count", 0) or 0)
    fallback = bool(hard_failed and candidate_count > 0 and safe_count == 0)
    status = "failed_fallback" if fallback else ("failed" if hard_failed else "passed")
    return {
        "status": str(status),
        "hard_gate_failed": bool(hard_failed),
        "hard_gate_failures": list(failures),
        "residual_peak_db": _finite_float(diag.get("residual_peak_db")),
        "residual_peak_hard_gate_db": _finite_float(diag.get("residual_peak_hard_gate_db")),
        "bass_integration_hard_gate_failed": bool(diag.get("bass_integration_hard_gate_failed", False)),
        "bass_feasibility_class": str(diag.get("bass_feasibility_class", "") or ""),
        "bass_feasibility_reason": str(diag.get("bass_feasibility_reason", "") or ""),
        "winner_override": dict(override),
        "fallback_reason": str(override.get("reason", "") or "") if fallback else "",
    }


def build_auto_mode_audit_trail(
    *,
    best_metrics: dict | None,
    best_preset: dict | None = None,
    winner_explanation: dict | None = None,
    residual_peak_safety_override_meta: dict | None = None,
    optimizer_backend: str = "builtin",
    goal: str = "balanced",
    selection_basis: str = "rank_score",
    target_name: str | None = None,
    target_meta: dict | None = None,
    top: list | None = None,
    cache_info: dict | None = None,
    polish_meta: dict | None = None,
    phase1_ok: int = 0,
    phase2_ok: int = 0,
    phase1_tried: int = 0,
    phase2_tried: int = 0,
    phase1_plateau_hit: bool = False,
    phase2_plateau_hit: bool = False,
    phase3_total: int = 0,
    phase3_ok: int = 0,
    phase4_steps: dict | None = None,
    fs_v: int = 0,
    taps_v: int = 0,
    trials_total: int | None = None,
    trials_ok: int | None = None,
    source: str = "search",
) -> dict:
    metrics = attach_official_rank_score(best_metrics)
    preset = _clean_dict(best_preset)
    explanation = _clean_dict(winner_explanation)
    rank = official_rank_score(metrics)
    hard_gates = _build_hard_gate_section(
        best_metrics=metrics,
        goal=str(goal or "balanced"),
        residual_peak_safety_override_meta=residual_peak_safety_override_meta,
    )
    reasons = _dedupe_text(_clean_list(explanation.get("reasons")))
    summary = str(explanation.get("summary", "") or "").strip()
    if not summary:
        summary = "Winner selected from available automatic-mode metrics."

    cache = _clean_dict(cache_info)
    candidates = [
        _candidate_summary(dict(item or {}), goal=str(goal or "balanced"))
        for item in _clean_list(top)[:3]
        if isinstance(item, dict)
    ]
    target = _clean_dict(target_meta)
    if target_name:
        target["name"] = str(target_name)

    notes: list[str] = []
    if bool(_clean_dict(residual_peak_safety_override_meta).get("applied", False)):
        notes.append("Winner override applied for safer hard-gate behavior.")
    if str(hard_gates.get("status")) == "failed_fallback":
        notes.append("All checked candidates failed a hard gate; kept least-bad fallback.")

    return {
        "schema_version": int(AUTO_MODE_AUDIT_TRAIL_SCHEMA_VERSION),
        "source": str(source or "search"),
        "selection": {
            "goal": str(goal or "balanced"),
            "basis": str(selection_basis or "rank_score"),
            "optimizer_backend": str(optimizer_backend or "builtin"),
            "target_name": str(target.get("name", "") or ""),
        },
        "winner": {
            "summary": str(summary),
            "reasons": list(reasons),
            "rank_score_official": float(rank) if math.isfinite(rank) else float("nan"),
            "avg_score": _finite_float(metrics.get("avg_score")),
            "residual_peak_db": _finite_float(hard_gates.get("residual_peak_db")),
            "residual_peak_hard_gate_db": _finite_float(hard_gates.get("residual_peak_hard_gate_db")),
            "max_net_boost_db": _finite_float(metrics.get("max_net_boost_db")),
            "selected_preset": dict(preset),
        },
        "scores": {
            "rank_score_breakdown": _clean_dict(metrics.get("rank_score_breakdown")),
            "rank_score_components": _clean_dict(metrics.get("rank_score_components")),
        },
        "hard_gates": dict(hard_gates),
        "candidate_comparison": {
            "top_count": int(len(_clean_list(top))),
            "top3": list(candidates),
        },
        "target": dict(target),
        "search": _build_search_section(
            optimizer_backend=str(optimizer_backend or "builtin"),
            phase1_ok=int(phase1_ok),
            phase2_ok=int(phase2_ok),
            phase1_tried=int(phase1_tried),
            phase2_tried=int(phase2_tried),
            trials_total=trials_total,
            trials_ok=trials_ok,
            phase1_plateau_hit=bool(phase1_plateau_hit),
            phase2_plateau_hit=bool(phase2_plateau_hit),
            phase3_total=int(phase3_total),
            phase3_ok=int(phase3_ok),
            phase4_steps=phase4_steps,
            fs_v=int(fs_v),
            taps_v=int(taps_v),
        ),
        "cache": dict(cache),
        "polish": _build_polish_section(polish_meta),
        "notes": list(notes),
    }


__all__ = ["AUTO_MODE_AUDIT_TRAIL_SCHEMA_VERSION", "build_auto_mode_audit_trail"]
