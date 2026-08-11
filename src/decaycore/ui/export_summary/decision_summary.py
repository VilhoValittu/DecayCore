# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Render the compact export decision and safety summary."""

from __future__ import annotations

import logging
import math

from ...common.auto_reporting import attach_official_rank_score
from ..export_scoring import _safe_float

logger = logging.getLogger(__name__)


def _audit_dict(value) -> dict:
    return dict(value or {}) if isinstance(value, dict) else {}


def _audit_list(value) -> list:
    return list(value or []) if isinstance(value, (list, tuple)) else []


def _decision_fmt_hz(value, *, decimals: int = 1) -> str:
    v = _safe_float(value, float("nan"))
    if not math.isfinite(float(v)):
        return "n/a"
    if abs(float(v) - round(float(v))) <= 1e-6:
        return f"{float(v):.0f} Hz"
    return f"{float(v):.{int(decimals)}f} Hz"


def _decision_fmt_db(value, *, signed: bool = False) -> str:
    v = _safe_float(value, float("nan"))
    if not math.isfinite(float(v)):
        return "n/a"
    sign = "+" if bool(signed) else ""
    return f"{float(v):{sign}.2f} dB"


def _decision_fmt_ms(value) -> str:
    v = _safe_float(value, float("nan"))
    return f"{float(v):+.3f} ms" if math.isfinite(float(v)) else "n/a"


def _decision_pick_float(*values) -> float:
    for value in values:
        v = _safe_float(value, float("nan"))
        if math.isfinite(float(v)):
            return float(v)
    return float("nan")


def _decision_value(data: dict, best_preset: dict, key: str, default=float("nan")):
    for src in (data, best_preset):
        if isinstance(src, dict) and key in src:
            return src.get(key)
    return default


def _decision_filter_label(data: dict) -> str:
    raw = str(data.get("filter_type", data.get("filter_type_str", "")) or "").strip()
    return raw or "n/a"


def _decision_label(value) -> str:
    return str(value).replace("_", " ").strip()


def _decision_correction_range(data: dict, best_preset: dict) -> list[str]:
    mag_min = _decision_pick_float(_decision_value(data, best_preset, "mag_c_min"))
    mag_max = _decision_pick_float(_decision_value(data, best_preset, "mag_c_max"))
    lines = []
    if math.isfinite(mag_min) or math.isfinite(mag_max):
        lines.append(
            f"Magnitude correction band: {_decision_fmt_hz(mag_min)} - {_decision_fmt_hz(mag_max)}"
        )

    full_hz = _decision_pick_float(
        _decision_value(data, best_preset, "low_freq_full_correction_hz")
    )
    no_hz = _decision_pick_float(
        _decision_value(data, best_preset, "high_freq_no_correction_hz")
    )
    if math.isfinite(full_hz) or math.isfinite(no_hz):
        lines.append(
            f"Correction taper: full <= {_decision_fmt_hz(full_hz)}, off by {_decision_fmt_hz(no_hz)}"
        )

    low_cut_enabled = bool(
        _decision_value(data, best_preset, "low_bass_cut_enable", False)
    )
    low_cut_hz = _decision_pick_float(
        _decision_value(data, best_preset, "low_bass_cut_hz")
    )
    if low_cut_enabled and math.isfinite(low_cut_hz) and low_cut_hz > 0.0:
        lines.append(
            f"Low-bass boost lock: cuts only below {_decision_fmt_hz(low_cut_hz)}"
        )
    return lines


def _decision_gain_limits(
    data: dict, best_preset: dict, l_st: dict, r_st: dict
) -> list[str]:
    max_boost = _decision_pick_float(
        l_st.get("max_boost_db_effective"),
        r_st.get("max_boost_db_effective"),
        _decision_value(data, best_preset, "max_boost"),
        _decision_value(data, best_preset, "max_boost_db"),
    )
    max_cut = abs(
        _decision_pick_float(
            l_st.get("max_cut_db"),
            r_st.get("max_cut_db"),
            _decision_value(data, best_preset, "max_cut_db"),
        )
    )
    boost_l = _decision_pick_float(
        l_st.get("boost_peak_db"), l_st.get("net_boost_peak_db")
    )
    boost_r = _decision_pick_float(
        r_st.get("boost_peak_db"), r_st.get("net_boost_peak_db")
    )
    cut_l = _decision_pick_float(l_st.get("cut_peak_db"))
    cut_r = _decision_pick_float(r_st.get("cut_peak_db"))
    net_l = _decision_pick_float(l_st.get("net_boost_peak_db"))
    net_r = _decision_pick_float(r_st.get("net_boost_peak_db"))

    cut_limit = f"-{max_cut:.2f} dB" if math.isfinite(float(max_cut)) else "n/a"
    lines = [
        f"Limits: boost <= {_decision_fmt_db(max_boost)}, attenuation >= {cut_limit}"
    ]
    if any(math.isfinite(v) for v in (boost_l, boost_r, cut_l, cut_r)):
        lines.append(
            "Realized correction: "
            f"boost L/R {_decision_fmt_db(boost_l, signed=True)} / {_decision_fmt_db(boost_r, signed=True)}, "
            f"cut L/R {_decision_fmt_db(cut_l)} / {_decision_fmt_db(cut_r)}"
        )
    if any(math.isfinite(v) for v in (net_l, net_r)):
        lines.append(
            f"Post-gain net boost peak L/R: {_decision_fmt_db(net_l, signed=True)} / {_decision_fmt_db(net_r, signed=True)}"
        )
    return lines


def _decision_phase_tdc(
    data: dict, best_preset: dict, bm: dict, l_st: dict, r_st: dict
) -> list[str]:
    filter_type = _decision_filter_label(data)
    phase_bits = [f"Filter type: {filter_type}"]
    ft_l = filter_type.lower()
    if "mixed" in ft_l:
        mixed_freq = _decision_pick_float(
            _decision_value(data, best_preset, "mixed_freq")
        )
        phase_bits.append(f"mixed transition {_decision_fmt_hz(mixed_freq)}")
    elif "linear" in ft_l or "asym" in ft_l:
        phase_limit = _decision_pick_float(
            _decision_value(data, best_preset, "phase_limit")
        )
        realized = [
            _decision_pick_float(
                l_st.get("mixed_phase_no_correction_hz"), l_st.get("phase_limit_hz")
            ),
            _decision_pick_float(
                r_st.get("mixed_phase_no_correction_hz"), r_st.get("phase_limit_hz")
            ),
        ]
        if any(math.isfinite(v) for v in realized):
            phase_bits.append(
                f"phase limit {_decision_fmt_hz(phase_limit)}, realized L/R {_decision_fmt_hz(realized[0])} / {_decision_fmt_hz(realized[1])}"
            )
        else:
            phase_bits.append(f"phase limit {_decision_fmt_hz(phase_limit)}")

    lines = ["; ".join(phase_bits)]
    gd_before_l = _decision_pick_float(l_st.get("phase_realized_gd_before_rms_ms"))
    gd_after_l = _decision_pick_float(l_st.get("phase_realized_gd_after_rms_ms"))
    gd_before_r = _decision_pick_float(r_st.get("phase_realized_gd_before_rms_ms"))
    gd_after_r = _decision_pick_float(r_st.get("phase_realized_gd_after_rms_ms"))
    if any(
        math.isfinite(v) for v in (gd_before_l, gd_after_l, gd_before_r, gd_after_r)
    ):
        lines.append(
            "Final-FIR system GD RMS before -> after: "
            f"L {_decision_fmt_ms(gd_before_l)} -> {_decision_fmt_ms(gd_after_l)}, "
            f"R {_decision_fmt_ms(gd_before_r)} -> {_decision_fmt_ms(gd_after_r)}"
        )
    feedback_reason = str(
        l_st.get(
            "phase_realization_feedback_reason",
            r_st.get("phase_realization_feedback_reason", ""),
        )
        or ""
    ).strip()
    feedback_requested = _decision_pick_float(
        l_st.get("phase_realization_feedback_requested_strength"),
        r_st.get("phase_realization_feedback_requested_strength"),
    )
    feedback_selected = _decision_pick_float(
        l_st.get("phase_realization_feedback_selected_strength"),
        r_st.get("phase_realization_feedback_selected_strength"),
    )
    if feedback_reason == "not_applicable_minimum_phase":
        lines.append("Phase realization feedback: not applicable (minimum phase)")
    elif feedback_reason == "disabled":
        lines.append("Phase realization feedback: disabled")
    elif feedback_reason:
        if math.isfinite(feedback_requested) and math.isfinite(feedback_selected):
            lines.append(
                "Phase realization feedback: "
                f"strength {feedback_requested:.3f} -> {feedback_selected:.3f}, "
                f"reason {feedback_reason}"
            )
        else:
            lines.append(f"Phase realization feedback: {feedback_reason}")
    if bool(data.get("enable_tdc", False)):
        strength = _decision_pick_float(
            _decision_value(data, best_preset, "tdc_strength")
        )
        decision = str(bm.get("tdc_decision", "") or "").replace("_", " ").strip()
        decay_penalty = _decision_pick_float(
            bm.get("tdc_decay_penalty"), bm.get("decay_penalty")
        )
        tdc_line = f"TDC: ON, strength {strength:.1f}%"
        if decision:
            tdc_line += f", decision {decision}"
        if math.isfinite(decay_penalty):
            tdc_line += f", decay penalty {decay_penalty:.2f}"
        lines.append(tdc_line)
    else:
        lines.append("TDC: OFF")
    return lines


def _decision_bass_status(data: dict, bm: dict) -> list[str]:
    if not bool(data.get("bass_integration_enable", False)):
        return ["Bass integration: OFF"]
    bi_meta = dict(data.get("_bass_integration_meta", {}) or {})
    diag = dict(bi_meta.get("diagnostics", {}) or {})
    align = dict(bi_meta.get("alignment", {}) or {})
    mode = str(data.get("bass_integration_mode", "direct_dac") or "direct_dac")
    feasibility = str(
        diag.get("feasibility_class", bm.get("bass_feasibility_class", "")) or ""
    ).strip()
    reason = str(
        diag.get("feasibility_reason", bm.get("bass_feasibility_reason", "")) or ""
    ).strip()
    delay = _decision_pick_float(
        align.get("delay_ms"), data.get("bass_integration_sub_delay_ms")
    )
    gain = _decision_pick_float(
        align.get("gain_trim_db"), data.get("bass_integration_sub_gain_trim_db")
    )
    polarity = (
        "inverted"
        if bool(data.get("bass_integration_sub_polarity_invert", False))
        else "normal"
    )
    line = f"Bass integration: ON ({mode})"
    if feasibility:
        line += f", feasibility {feasibility.upper()}"
    if reason:
        line += f" - {reason}"
    settings = f"Sub alignment: delay {_decision_fmt_ms(delay)}, gain {_decision_fmt_db(gain, signed=True)}, polarity {polarity}"
    return [line, settings]


def _decision_rejected_risks(auto_meta: dict, bm: dict) -> list[str]:
    audit = _audit_dict(auto_meta.get("audit_trail"))
    hard_gates = _audit_dict(audit.get("hard_gates"))
    failures = [
        str(item).strip()
        for item in _audit_list(
            hard_gates.get(
                "hard_gate_failures",
                bm.get("hard_gate_failures", bm.get("hard_gate_reasons", [])),
            )
        )
        if str(item).strip()
    ]
    risks: list[str] = []
    if failures:
        risks.append(
            "Hard gates: "
            + ", ".join(dict.fromkeys(_decision_label(item) for item in failures))
        )
    override = _audit_dict(hard_gates.get("winner_override"))
    if override:
        state = "applied" if bool(override.get("applied", False)) else "not applied"
        reason = str(override.get("reason", "") or "").strip()
        risks.append(
            f"Winner safety override: {state}" + (f" ({reason})" if reason else "")
        )
    p6_severity = str(bm.get("final_ir_validation_severity", "") or "").strip()
    p6_reasons = [
        _decision_label(item)
        for item in list(bm.get("final_ir_validation_reasons", []) or [])
        if str(item or "").strip()
    ]
    if p6_severity:
        risks.append(
            f"Final IR validation: {p6_severity}"
            + (f" ({', '.join(p6_reasons[:4])})" if p6_reasons else "")
        )
    bass_gate = bool(
        bm.get("bass_integration_hard_gate_failed", False)
        or "bass_integration_infeasible_hard_gate" in failures
    )
    if bass_gate:
        reason = str(
            bm.get(
                "bass_integration_hard_gate_reason",
                bm.get("bass_feasibility_reason", ""),
            )
            or ""
        ).strip()
        risks.append("Bass integration hard gate" + (f": {reason}" if reason else ""))
    return risks or ["No rejected hard-gate risks reported."]


def _append_export_decision_summary(
    summary_content: str,
    data: dict | None,
    fs_v: int,
    l_st: dict | None,
    r_st: dict | None,
) -> str:
    try:
        ui_data = dict(data or {})
        auto_meta = dict(ui_data.get("_auto_mode_meta", {}) or {})
        best_preset = dict(auto_meta.get("best_preset", {}) or {})
        bm = attach_official_rank_score(auto_meta.get("best_metrics", {}) or {})
        left = dict(l_st or {})
        right = dict(r_st or {})
        lines: list[str] = [
            "\n=== DECISION SUMMARY ===",
            f"Sample rate: {int(fs_v)} Hz",
        ]
        for group in (
            _decision_correction_range(ui_data, best_preset),
            _decision_gain_limits(ui_data, best_preset, left, right),
            _decision_phase_tdc(ui_data, best_preset, bm, left, right),
            _decision_bass_status(ui_data, bm),
        ):
            lines.extend(group)
        lines.append(
            "Rejected / contained risks: "
            + "; ".join(_decision_rejected_risks(auto_meta, bm))
        )
        return summary_content + "\n".join(lines) + "\n"
    except (
        AttributeError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        RuntimeError,
        OSError,
        ImportError,
        ModuleNotFoundError,
        NameError,
    ) as exc:
        logger.exception("decision summary section")
        return (
            summary_content
            + f"\n=== DECISION SUMMARY ===\nCould not build decision summary: {type(exc).__name__}: {exc}\n"
        )
