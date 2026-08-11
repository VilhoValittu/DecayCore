# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Build and render the Automatic-mode audit trail for exports."""

from __future__ import annotations

import math

from ...common.auto_reporting import build_auto_mode_audit_trail
from ..export_scoring import _safe_float


def _audit_dict(value) -> dict:
    return dict(value or {}) if isinstance(value, dict) else {}


def _audit_list(value) -> list:
    return list(value or []) if isinstance(value, (list, tuple)) else []


def _audit_fmt_score(value) -> str:
    v = _safe_float(value, float("nan"))
    return f"{float(v):.3f}" if math.isfinite(float(v)) else "n/a"


def _audit_fmt_avg(value) -> str:
    v = _safe_float(value, float("nan"))
    return f"{float(v):.3f}" if math.isfinite(float(v)) else "n/a"


def _auto_audit_from_meta(
    data: dict, auto_meta: dict, bm: dict, bp: dict, optimizer_backend: str
) -> dict:
    audit = _audit_dict(auto_meta.get("audit_trail"))
    if audit:
        return dict(audit)
    return build_auto_mode_audit_trail(
        best_metrics=bm,
        best_preset=bp,
        winner_explanation=_audit_dict(auto_meta.get("winner_explanation")),
        residual_peak_safety_override_meta=_audit_dict(
            auto_meta.get("residual_peak_safety_override")
        ),
        optimizer_backend=str(
            optimizer_backend
            or auto_meta.get("optimizer_backend", "builtin")
            or "builtin"
        ),
        goal=str(
            auto_meta.get("auto_goal", data.get("auto_goal", "balanced")) or "balanced"
        ),
        selection_basis=str(
            auto_meta.get("selection_basis", "rank_score") or "rank_score"
        ),
        target_name=str(
            _audit_dict(auto_meta.get("winner_explanation")).get("target_name", "")
            or ""
        ),
        target_meta=_audit_dict(data.get("_auto_target_curve_meta")),
        top=_audit_list(auto_meta.get("top")),
        cache_info={},
        polish_meta=auto_meta,
        phase1_ok=int(auto_meta.get("trials_phase1_ok", 0) or 0),
        phase2_ok=int(auto_meta.get("trials_phase2_ok", 0) or 0),
        phase1_tried=int(auto_meta.get("trials_phase1_total", 0) or 0),
        phase2_tried=int(auto_meta.get("trials_phase2_total", 0) or 0),
        phase1_plateau_hit=bool(auto_meta.get("phase1_plateau_hit", False)),
        phase2_plateau_hit=bool(auto_meta.get("phase2_plateau_hit", False)),
        phase3_total=int(auto_meta.get("trials_phase3_total", 0) or 0),
        phase3_ok=int(auto_meta.get("trials_phase3_ok", 0) or 0),
        phase4_steps=_audit_dict(auto_meta.get("phase4_steps")),
        fs_v=int(auto_meta.get("search_fs", 0) or 0),
        taps_v=int(auto_meta.get("search_taps", 0) or 0),
        trials_total=int(auto_meta.get("trials_total", 0) or 0),
        trials_ok=int(auto_meta.get("trials_ok", 0) or 0),
        source="summary_reconstructed",
    )


def _append_dsp_effective_auto_audit_summary(
    summary_content: str,
    data: dict,
    auto_meta: dict,
    bm: dict,
    bp: dict,
    optimizer_backend: str,
) -> str:
    audit = _auto_audit_from_meta(data, auto_meta, bm, bp, optimizer_backend)
    if not audit:
        return summary_content
    selection = _audit_dict(audit.get("selection"))
    winner = _audit_dict(audit.get("winner"))
    hard_gates = _audit_dict(audit.get("hard_gates"))
    search = _audit_dict(audit.get("search"))
    cache = _audit_dict(audit.get("cache"))

    goal = str(
        selection.get("goal", auto_meta.get("auto_goal", "balanced")) or "balanced"
    )
    basis = str(
        selection.get("basis", auto_meta.get("selection_basis", "rank_score"))
        or "rank_score"
    )
    backend = str(
        selection.get("optimizer_backend", optimizer_backend or "builtin") or "builtin"
    )
    summary_content += "AUTO audit trail:\n"
    summary_content += (
        f"- Winner: rank {_audit_fmt_score(winner.get('rank_score_official'))}/100, "
        f"avg {_audit_fmt_avg(winner.get('avg_score'))}, goal {goal}, basis {basis}\n"
    )
    why = str(winner.get("summary", "") or "").strip()
    if why:
        summary_content += f"- Why: {why}\n"

    gate_status = str(hard_gates.get("status", "passed") or "passed").replace("_", " ")
    failures = [
        str(item)
        for item in _audit_list(hard_gates.get("hard_gate_failures"))
        if str(item or "").strip()
    ]
    failure_txt = f" ({', '.join(failures)})" if failures else ""
    summary_content += f"- Safety gates: {gate_status}{failure_txt}\n"

    override = _audit_dict(hard_gates.get("winner_override"))
    if override:
        override_state = (
            "applied" if bool(override.get("applied", False)) else "not applied"
        )
        override_reason = str(override.get("reason", "") or "").strip()
        reason_txt = f" ({override_reason})" if override_reason else ""
        summary_content += f"- Winner override: {override_state}{reason_txt}\n"
    fallback_reason = str(hard_gates.get("fallback_reason", "") or "").strip()
    if fallback_reason:
        summary_content += f"- Fallback: {fallback_reason}\n"

    trials_ok = int(search.get("trials_ok", auto_meta.get("trials_ok", 0)) or 0)
    trials_total = int(
        search.get("trials_total", auto_meta.get("trials_total", 0)) or 0
    )
    phase4 = _audit_dict(search.get("phase4_steps"))
    phase4_used = any(bool(v) for v in phase4.values())
    summary_content += (
        f"- Search: {backend}, {trials_ok}/{trials_total} trials ok, "
        f"phase4={'on' if phase4_used else 'off'}\n"
    )

    target_name = str(
        selection.get("target_name", "")
        or _audit_dict(audit.get("target")).get("name", "")
        or ""
    ).strip()
    if target_name:
        summary_content += f"- Target: {target_name}\n"

    cache_stats = _audit_dict(cache.get("cache_stats"))
    if cache_stats:
        summary_content += (
            "- Cache: "
            f"entry_hits={int(cache_stats.get('entry_hits', 0) or 0)}, "
            f"entry_misses={int(cache_stats.get('entry_misses', 0) or 0)}, "
            f"saves={int(cache_stats.get('saves', 0) or 0)}\n"
        )
    return summary_content


