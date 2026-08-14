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

import math

from ..export_scoring import _safe_float
from .runtime import _polish_display_rank


def _append_dsp_effective_phase_limit_polish_summary(summary_content: str, auto_meta: dict) -> str:
    polish = dict(auto_meta.get("phase_limit_winner_polish", {}) or {})
    if not bool(polish.get("applicable", False)):
        return summary_content
    polish_start = _safe_float(polish.get("start_phase_limit_hz", float("nan")), float("nan"))
    polish_final = _safe_float(polish.get("final_phase_limit_hz", float("nan")), float("nan"))
    polish_rank_before = _polish_display_rank(polish, "rank_before", "rank_before_official")
    polish_rank_after = _polish_display_rank(polish, "rank_after", "rank_after_official")
    polish_tested = [float(v) for v in list(polish.get("tested_phase_limits_hz", []) or [])]
    polish_tested_txt = ", ".join([f"{float(v):.1f}" for v in polish_tested]) if polish_tested else "n/a"
    if bool(polish.get("applied", False)):
        summary_content += (
            f"Phase-limit winner polish: applied "
            f"({float(polish_start):.1f} -> {float(polish_final):.1f} Hz, "
            f"rank {float(polish_rank_before):.3f} -> {float(polish_rank_after):.3f}, "
            f"tested [{polish_tested_txt}] Hz)\n"
        )
    else:
        summary_content += (
            f"Phase-limit winner polish: tested, no change "
            f"(kept {float(polish_final):.1f} Hz, tested [{polish_tested_txt}] Hz)\n"
        )
    return summary_content


def _append_dsp_effective_mag_c_min_polish_summary(summary_content: str, auto_meta: dict) -> str:
    polish = dict(auto_meta.get("mag_c_min_winner_polish", {}) or {})
    if not bool(polish.get("applicable", False)):
        return summary_content
    polish_start = _safe_float(polish.get("start_mag_c_min_hz", float("nan")), float("nan"))
    polish_final = _safe_float(polish.get("final_mag_c_min_hz", float("nan")), float("nan"))
    polish_rank_before = _polish_display_rank(polish, "rank_before", "rank_before_official")
    polish_rank_after = _polish_display_rank(polish, "rank_after", "rank_after_official")
    polish_tested = [float(v) for v in list(polish.get("tested_mag_c_min_hz", []) or [])]
    polish_tested_txt = ", ".join([f"{float(v):.1f}" for v in polish_tested]) if polish_tested else "n/a"
    if bool(polish.get("applied", False)):
        summary_content += (
            f"Mag-c-min winner polish: applied "
            f"({float(polish_start):.1f} -> {float(polish_final):.1f} Hz, "
            f"rank {float(polish_rank_before):.3f} -> {float(polish_rank_after):.3f}, "
            f"tested [{polish_tested_txt}] Hz)\n"
        )
    else:
        summary_content += (
            f"Mag-c-min winner polish: tested, no change "
            f"(kept {float(polish_final):.1f} Hz, tested [{polish_tested_txt}] Hz)\n"
        )
    return summary_content


def _append_dsp_effective_low_bass_cut_polish_summary(summary_content: str, auto_meta: dict) -> str:
    polish = dict(auto_meta.get("low_bass_cut_winner_polish", {}) or {})
    if not bool(polish.get("applicable", False)):
        return summary_content
    polish_start = _safe_float(polish.get("start_low_bass_cut_hz", float("nan")), float("nan"))
    polish_final = _safe_float(polish.get("final_low_bass_cut_hz", float("nan")), float("nan"))
    polish_rank_before = _polish_display_rank(polish, "rank_before", "rank_before_official")
    polish_rank_after = _polish_display_rank(polish, "rank_after", "rank_after_official")
    polish_tested = [float(v) for v in list(polish.get("tested_low_bass_cut_hz", []) or [])]
    polish_tested_txt = ", ".join([f"{float(v):.1f}" for v in polish_tested]) if polish_tested else "n/a"
    if bool(polish.get("applied", False)):
        summary_content += (
            f"Low-bass-cut winner polish: applied "
            f"({float(polish_start):.1f} -> {float(polish_final):.1f} Hz, "
            f"rank {float(polish_rank_before):.3f} -> {float(polish_rank_after):.3f}, "
            f"tested [{polish_tested_txt}] Hz)\n"
        )
    else:
        summary_content += (
            f"Low-bass-cut winner polish: tested, no change "
            f"(kept {float(polish_final):.1f} Hz, tested [{polish_tested_txt}] Hz)\n"
        )
    return summary_content


def _append_dsp_effective_hpf_polish_summary(summary_content: str, auto_meta: dict) -> str:
    polish = dict(auto_meta.get("hpf_winner_polish", {}) or {})
    if not bool(polish.get("applicable", False)):
        return summary_content
    start_enabled = bool(polish.get("start_enabled", False))
    start_freq_hz = _safe_float(polish.get("start_freq_hz", float("nan")), float("nan"))
    start_slope_db_oct = int(round(_safe_float(polish.get("start_slope_db_oct", 0.0), 0.0)))
    final_enabled = bool(polish.get("final_enabled", False))
    final_freq_hz = _safe_float(polish.get("final_freq_hz", float("nan")), float("nan"))
    final_slope_db_oct = int(round(_safe_float(polish.get("final_slope_db_oct", 0.0), 0.0)))
    polish_rank_before = _polish_display_rank(polish, "rank_before", "rank_before_official")
    polish_rank_after = _polish_display_rank(polish, "rank_after", "rank_after_official")
    tested_candidates = [
        str(dict(item).get("label", "") or "").strip()
        for item in list(polish.get("tested_candidates", []) or [])
        if isinstance(item, dict)
    ]
    tested_candidates = [txt for txt in tested_candidates if txt]
    tested_txt = ", ".join(tested_candidates) if tested_candidates else "n/a"

    def _fmt_hpf(enabled_state: bool, freq_hz: float, slope_db_oct: int) -> str:
        if not bool(enabled_state):
            return "HPF off"
        return f"HPF {float(freq_hz):.1f} Hz/{int(slope_db_oct)} dB/oct"

    if bool(polish.get("applied", False)):
        summary_content += (
            f"HPF winner polish: applied "
            f"({_fmt_hpf(start_enabled, start_freq_hz, start_slope_db_oct)} -> "
            f"{_fmt_hpf(final_enabled, final_freq_hz, final_slope_db_oct)}, "
            f"rank {float(polish_rank_before):.3f} -> {float(polish_rank_after):.3f}, "
            f"tested [{tested_txt}])\n"
        )
    else:
        summary_content += (
            f"HPF winner polish: tested, no change "
            f"(kept {_fmt_hpf(final_enabled, final_freq_hz, final_slope_db_oct)}, tested [{tested_txt}])\n"
        )
    return summary_content


def _append_dsp_effective_excess_phase_strength_polish_summary(summary_content: str, auto_meta: dict) -> str:
    polish = dict(auto_meta.get("excess_phase_strength_winner_polish", {}) or {})
    if not bool(polish.get("applicable", False)):
        return summary_content
    eps_start = _safe_float(polish.get("start_value", float("nan")), float("nan"))
    eps_final = _safe_float(polish.get("final_value", float("nan")), float("nan"))
    eps_rank_before = _polish_display_rank(polish, "rank_before", "rank_before_official")
    eps_rank_after = _polish_display_rank(polish, "rank_after", "rank_after_official")
    eps_tested = int(polish.get("tested_count", 0) or 0)
    if bool(polish.get("applied", False)):
        summary_content += (
            f"Excess-phase-strength winner polish: applied "
            f"({float(eps_start):.4f} -> {float(eps_final):.4f}, "
            f"rank {float(eps_rank_before):.3f} -> {float(eps_rank_after):.3f}, "
            f"tested {eps_tested})\n"
        )
    else:
        summary_content += (
            f"Excess-phase-strength winner polish: tested, no change "
            f"(kept {float(eps_final):.4f}, tested {eps_tested})\n"
        )
    return summary_content


def _append_dsp_effective_residual_peak_polish_summary(summary_content: str, auto_meta: dict) -> str:
    polish = dict(auto_meta.get("residual_peak_winner_polish", {}) or {})
    if not bool(polish.get("applicable", False)):
        return summary_content
    peak_before = _safe_float(polish.get("worst_peak_before_db", float("nan")), float("nan"))
    peak_after = _safe_float(polish.get("worst_peak_after_db", float("nan")), float("nan"))
    peak_hz = _safe_float(polish.get("worst_peak_freq_hz", float("nan")), float("nan"))
    width_oct = _safe_float(polish.get("worst_peak_width_oct", float("nan")), float("nan"))
    tested_count = int(polish.get("tested_count", 0) or 0)
    peak_pos = f" @ {float(peak_hz):.1f} Hz" if math.isfinite(float(peak_hz)) else ""
    width_txt = f", width={float(width_oct):.3f} oct" if math.isfinite(float(width_oct)) else ""
    if bool(polish.get("applied", False)):
        summary_content += (
            f"Residual-peak winner polish: applied "
            f"({float(peak_before):.2f} -> {float(peak_after):.2f} dB{peak_pos}{width_txt}, "
            f"tested {tested_count})\n"
        )
    elif bool(polish.get("enabled", False)):
        summary_content += (
            f"Residual-peak winner polish: tested, no change "
            f"(peak {float(peak_after):.2f} dB{peak_pos}{width_txt}, tested {tested_count})\n"
        )
    return summary_content


def _append_dsp_effective_tdc_strength_polish_summary(summary_content: str, auto_meta: dict) -> str:
    polish = dict(auto_meta.get("tdc_strength_winner_polish", {}) or {})
    if not bool(polish.get("applicable", False)):
        return summary_content
    tdc_start = _safe_float(polish.get("start_strength", float("nan")), float("nan"))
    tdc_final = _safe_float(polish.get("final_strength", float("nan")), float("nan"))
    tdc_hint = _safe_float(polish.get("optimum_strength_hint", float("nan")), float("nan"))
    tdc_rank_before = _polish_display_rank(polish, "rank_before", "rank_before_official")
    tdc_rank_after = _polish_display_rank(polish, "rank_after", "rank_after_official")
    tdc_tested = [float(v) for v in list(polish.get("tested_strengths", []) or [])]
    tdc_tested_txt = ", ".join(f"{v:.1f}" for v in tdc_tested) if tdc_tested else "n/a"
    if bool(polish.get("applied", False)):
        summary_content += (
            f"TDC-strength winner polish: applied "
            f"({float(tdc_start):.1f}% -> {float(tdc_final):.1f}%, "
            f"hint={float(tdc_hint):.1f}%, "
            f"rank {float(tdc_rank_before):.3f} -> {float(tdc_rank_after):.3f}, "
            f"tested [{tdc_tested_txt}])\n"
        )
    else:
        summary_content += (
            f"TDC-strength winner polish: tested, no change "
            f"(kept {float(tdc_start):.1f}%, hint={float(tdc_hint):.1f}%, tested {len(tdc_tested)})\n"
        )
    return summary_content


def _append_dsp_effective_auto_winner_polish_summary(summary_content: str, auto_meta: dict) -> str:
    for helper in (
        _append_dsp_effective_phase_limit_polish_summary,
        _append_dsp_effective_mag_c_min_polish_summary,
        _append_dsp_effective_low_bass_cut_polish_summary,
        _append_dsp_effective_hpf_polish_summary,
        _append_dsp_effective_excess_phase_strength_polish_summary,
        _append_dsp_effective_residual_peak_polish_summary,
        _append_dsp_effective_tdc_strength_polish_summary,
    ):
        summary_content = helper(summary_content, auto_meta)
    return summary_content
