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
import math
import sys

from ...config.decaycore_pipeline import filter_type_supports_xo_phase_model

logger = logging.getLogger(__name__)
from ...dsp.smoothing import AFDW_BW_MAX_OCT, AFDW_BW_MIN_OCT
from ...dsp.lr_difference_metrics import compute_lr_difference_metrics
from ...auto_mode.rank_score import attach_official_rank_score, calibrated_auto_quality, display_rank_score, official_rank_score, _quality_band
from ..export_scoring import _pick_metric, _safe_float

_AUTO_ASYM_PHASE1_SEARCH_SPACE_EST = 1877500016615829065655090169509480

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
        reason_code = str(stereo_meta.get("gate_reason", "") or "").strip().lower()
        if not reason_code:
            if stereo_meta:
                reason = "shared winner kept; no specific fallback reason was recorded"
            else:
                reason = "shared winner kept; stereo refine metadata unavailable"
        else:
            reason = {
                "asymmetry_too_small": "shared winner kept; L/R LF asymmetry was too small to justify per-channel refinement",
                "stereo_safety_gate": "shared winner kept; refined candidate failed the stereo safety gate",
                "worst_channel_gain_too_small": "shared winner kept; worst-channel LF improvement stayed below the minimum threshold",
                "no_refined_winner": "shared winner kept; no constrained per-channel candidate beat the shared winner",
                "shared_fallback": "shared winner kept",
            }.get(reason_code, reason_code)
        summary_content += f"Fallback reason: {reason}\n"

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

    coh_pen = _safe_float(stereo_meta.get("stereo_coherence_penalty", float("nan")), float("nan"))
    center_pen = _safe_float(stereo_meta.get("phantom_center_stability_penalty", float("nan")), float("nan"))
    policy_pen = _safe_float(stereo_meta.get("policy_divergence_penalty", float("nan")), float("nan"))
    if coh_pen == coh_pen:
        summary_content += f"Stereo coherence penalty: {float(coh_pen):.3f}\n"
    if center_pen == center_pen:
        summary_content += f"Phantom center stability penalty: {float(center_pen):.3f}\n"
    if policy_pen == policy_pen:
        summary_content += f"Policy divergence penalty: {float(policy_pen):.3f}\n"

    resolved = dict(stereo_meta.get("resolved", {}) or {})
    shared = dict(resolved.get("shared", {}) or {})
    left = dict(resolved.get("left", {}) or {})
    right = dict(resolved.get("right", {}) or {})
    diverged = []
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
    if diverged:
        summary_content += "Policy deltas:\n"
        for line in diverged:
            summary_content += f"- {line}\n"
    return summary_content


__all__ = ['_append_auto_stereo_policy_summary']


def _link_sibling_exports() -> None:
    import importlib
    package = __package__
    for module_name in ['runtime', 'bass_integration', 'stereo_policy', 'dsp_effective', 'events']:
        if module_name == __name__.rsplit('.', 1)[-1]:
            continue
        module = importlib.import_module(f"{package}.{module_name}")
        for symbol in getattr(module, "__all__", ()):
            globals().setdefault(symbol, getattr(module, symbol))


_link_sibling_exports()
