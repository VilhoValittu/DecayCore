# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

import logging


logger = logging.getLogger(__name__)
from ..export_scoring import _safe_float

_AUTO_ASYM_PHASE1_SEARCH_SPACE_EST = 1877500016615829065655090169509480


def _stereo_policy_fallback_reason(stereo_meta: dict) -> str:
    reason_code = str(stereo_meta.get("gate_reason", "") or "").strip().lower()
    if not reason_code:
        if stereo_meta:
            return "shared winner kept; no specific fallback reason was recorded"
        return "shared winner kept; stereo refine metadata unavailable"
    return {
        "asymmetry_too_small": "shared winner kept; L/R LF asymmetry was too small to justify per-channel refinement",
        "stereo_safety_gate": "shared winner kept; refined candidate failed the stereo safety gate",
        "worst_channel_gain_too_small": "shared winner kept; worst-channel LF improvement stayed below the minimum threshold",
        "no_refined_winner": "shared winner kept; no constrained per-channel candidate beat the shared winner",
        "shared_fallback": "shared winner kept",
    }.get(reason_code, reason_code)


def _append_stereo_penalties(summary_content: str, stereo_meta: dict) -> str:
    for key, label in (
        ("stereo_coherence_penalty", "Stereo coherence penalty"),
        ("phantom_center_stability_penalty", "Phantom center stability penalty"),
        ("policy_divergence_penalty", "Policy divergence penalty"),
    ):
        value = _safe_float(stereo_meta.get(key, float("nan")), float("nan"))
        if value == value:
            summary_content += f"{label}: {float(value):.3f}\n"
    return summary_content


def _policy_delta_lines(stereo_meta: dict) -> list[str]:
    resolved = dict(stereo_meta.get("resolved", {}) or {})
    shared = dict(resolved.get("shared", {}) or {})
    left = dict(resolved.get("left", {}) or {})
    right = dict(resolved.get("right", {}) or {})
    diverged: list[str] = []
    for key, label, fmt in (
        ("conf_pull_floor", "LF confidence floor", "{:.2f}"),
        ("tdc_strength", "TDC strength", "{:.1f}%"),
        ("tdc_max_reduction_db", "TDC max reduction", "{:.1f} dB"),
        ("bass_first_mode_max_hz", "Bass-first LF span", "{:.1f} Hz"),
        ("low_bass_cut_strength", "Low-bass cut strength", "{:.2f}"),
    ):
        s_val = _safe_float(shared.get(key, float("nan")), float("nan"))
        l_val = _safe_float(left.get(key, s_val), s_val)
        r_val = _safe_float(right.get(key, s_val), s_val)
        if not (s_val == s_val and l_val == l_val and r_val == r_val):
            continue
        if abs(float(l_val) - float(s_val)) <= 1e-6 and abs(float(r_val) - float(s_val)) <= 1e-6:
            continue
        diverged.append(
            f"{label}: shared {fmt.format(float(s_val))}, L {fmt.format(float(l_val))}, R {fmt.format(float(r_val))}"
        )
    return diverged


def _append_auto_stereo_policy_summary(
    summary_content: str,
    data: dict | None,
    auto_meta: dict | None,
) -> str:
    ui_data = dict(data or {})
    meta = dict(auto_meta or {})
    stereo_meta = dict(meta.get("stereo_policy_refine", {}) or {})
    enabled = bool(ui_data.get("enable_channel_specific_auto_policy", False))
    summary_content += "AUTO Stereo Policy: "
    if not enabled:
        summary_content += "OFF\n"
        return summary_content

    state = str(stereo_meta.get("state", "shared_fallback") or "shared_fallback").strip().lower()
    split_hz = _safe_float(
        stereo_meta.get("split_hz", ui_data.get("channel_specific_policy_max_hz", 220.0)),
        220.0,
    )
    summary_content += "shared target + per-channel LF restraint refinement\n"
    summary_content += f"Asymmetry allowed below: {float(split_hz):.1f} Hz\n"
    summary_content += (
        f"Refined stereo-safety gate: {'PASSED' if bool(stereo_meta.get('gate_passed', False)) else 'FALLBACK'}\n"
    )
    if state == "applied":
        summary_content += "Refine source: constrained finalize-stage refinement\n"
    else:
        summary_content += f"Fallback reason: {_stereo_policy_fallback_reason(stereo_meta)}\n"

    shared_rank = _safe_float(stereo_meta.get("shared_rank", float("nan")), float("nan"))
    refined_rank = _safe_float(stereo_meta.get("refined_rank", float("nan")), float("nan"))
    if shared_rank == shared_rank and refined_rank == refined_rank:
        summary_content += f"Rank: {float(shared_rank):.3f} -> {float(refined_rank):.3f}\n"

    worse_side = str(stereo_meta.get("worse_side", "") or "").strip().upper()
    if worse_side:
        summary_content += f"Protected / limiting side: {worse_side}\n"

    relief = _safe_float(stereo_meta.get("worst_channel_relief_db", float("nan")), float("nan"))
    if relief == relief and abs(relief) != float("inf"):
        summary_content += f"Worst-channel relief: {float(relief):+.3f} dB\n"
    min_relief = _safe_float(stereo_meta.get("min_worst_channel_improvement_db", float("nan")), float("nan"))
    if min_relief == min_relief and abs(min_relief) != float("inf"):
        summary_content += f"Minimum required worst-channel relief: {float(min_relief):+.3f} dB\n"

    summary_content = _append_stereo_penalties(summary_content, stereo_meta)

    diverged = _policy_delta_lines(stereo_meta)
    if diverged:
        summary_content += "Policy deltas:\n"
        for line in diverged:
            summary_content += f"- {line}\n"
    return summary_content


__all__ = ['_append_auto_stereo_policy_summary']

