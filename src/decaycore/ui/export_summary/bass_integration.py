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
from ...auto_mode.rank_score import attach_official_rank_score
from ...resources.i8n.decaycore_i18n import t
from ..bass_integration_dsp_settings import build_bass_integration_dsp_settings
from ..export_scoring import _safe_float


def _append_bass_integration_summary(
    summary_content: str, data: dict | None
) -> str:  # noqa: C901 - legacy summary keeps detailed compatibility branches
    try:
        ui_data = dict(data or {})
        bi_meta = dict(ui_data.get("_bass_integration_meta", {}) or {})
        auto_meta = dict(ui_data.get("_auto_mode_meta", {}) or {})
        bm = attach_official_rank_score(auto_meta.get("best_metrics", {}))
        summary_content += "\n=== BASS INTEGRATION ===\n"
        bi_on = bool(ui_data.get("bass_integration_enable", False))
        summary_content += f"State: {'ON' if bi_on else 'OFF'}\n"
        if not bi_on:
            return summary_content
        raw_bi_mode = "direct_dac"
        bi_mode = "Direct DAC / CamillaDSP sub output"
        summary_content += f"Mode: {bi_mode}\n"
        if raw_bi_mode == "direct_dac":
            pass
        summary_content += f"Profile: {bi_meta.get('profile', ui_data.get('bass_integration_profile', 'safe')) or 'safe'!s}\n"
        summary_content += f"Sub combine mode: {bi_meta.get('sub_combine_mode', ui_data.get('bass_integration_sub_combine_mode', 'average')) or 'average'!s}\n"
        dual_sub_diag = dict(bi_meta.get("diagnostics", {}) or {})
        combine_mode = str(
            dual_sub_diag.get(
                "sub_combine_mode",
                bi_meta.get("sub_combine_mode", ui_data.get("bass_integration_sub_combine_mode", "average")),
            )
            or "average"
        ).strip().lower()
        is_dual_sub_shared = bool(dual_sub_diag.get("dual_sub_preprocessing_applied", False)) or combine_mode == "dual_sub_peak_aligned_average"
        if bool(dual_sub_diag.get("dual_sub_preprocessing_applied", False)):
            sub1_peak_ms = _safe_float(
                dual_sub_diag.get("dual_sub_sub1_peak_ms", float("nan")), float("nan")
            )
            sub2_peak_ms = _safe_float(
                dual_sub_diag.get("dual_sub_sub2_peak_ms", float("nan")), float("nan")
            )
            delay_ms = _safe_float(
                dual_sub_diag.get("dual_sub_relative_delay_ms", float("nan")),
                float("nan"),
            )
            summary_content += "Bass Integration dual-sub preprocessing:\n"
            summary_content += "Sub topology: dual-sub vector-average reference\n"
            if sub1_peak_ms == sub1_peak_ms and abs(sub1_peak_ms) != float("inf"):
                summary_content += f"- SUB1 peak: {float(sub1_peak_ms):.3f} ms\n"
            if sub2_peak_ms == sub2_peak_ms and abs(sub2_peak_ms) != float("inf"):
                summary_content += f"- SUB2 peak: {float(sub2_peak_ms):.3f} ms\n"
            if delay_ms == delay_ms and abs(delay_ms) != float("inf"):
                summary_content += (
                    f"- Applied peak-alignment delay: {float(delay_ms):+.3f} ms\n"
                )
            sub1_delay_ms = _safe_float(
                dual_sub_diag.get(
                    "sub1_delay_ms",
                    dual_sub_diag.get("dual_sub_sub1_delay_ms", float("nan")),
                ),
                float("nan"),
            )
            sub2_delay_ms = _safe_float(
                dual_sub_diag.get(
                    "sub2_delay_ms",
                    dual_sub_diag.get("dual_sub_sub2_delay_ms", float("nan")),
                ),
                float("nan"),
            )
            if sub1_delay_ms == sub1_delay_ms and abs(sub1_delay_ms) != float("inf"):
                summary_content += f"- SUB1 preprocessing delay: {float(sub1_delay_ms):+.3f} ms\n"
            if sub2_delay_ms == sub2_delay_ms and abs(sub2_delay_ms) != float("inf"):
                summary_content += f"- SUB2 preprocessing delay: {float(sub2_delay_ms):+.3f} ms\n"
            summary_content += (
                "- Combined sub reference: peak-aligned vector average of SUB1 + SUB2\n"
            )
            summary_content += (
                "- Combined method: complex vector average after peak alignment\n"
            )
            summary_content += "- Output filter: one shared mono Sub FIR for both subwoofers\n"
            summary_content += "- Per-sub optimization: no\n"
            summary_content += "- Reported delay, polarity, gain trim, and allpass values apply to the shared combined sub branch.\n"
        dsp_settings = build_bass_integration_dsp_settings(ui_data)
        if dsp_settings:
            summary_content += "\n=== DSP SETTINGS TO ENTER IN YOUR DSP ===\n"
            for setting in dsp_settings:
                summary_content += f"{t(setting.label_key)}: {setting.value}\n"
            summary_content += "\n=== BASS INTEGRATION DIAGNOSTICS ===\n"
        if is_dual_sub_shared:
            summary_content += "Dual-sub output model: shared mono sub branch for both subwoofers.\n"
        diag = dict(bi_meta.get("diagnostics", {}) or {})
        gd_cont = dict(diag.get("gd_continuity", {}) or {})
        alignment = dict(bi_meta.get("alignment", {}) or {})
        def _first_finite(*values):
            for value in values:
                parsed = _safe_float(value, float("nan"))
                if parsed == parsed and abs(parsed) != float("inf"):
                    return float(parsed)
            return float("nan")
        if raw_bi_mode == "direct_dac":
            summary_content += f"Auto alignment applied: {'YES' if bool(alignment.get('applied', False)) else 'NO'}\n"
            sub_delay = float(
                alignment.get(
                    "delay_ms", ui_data.get("bass_integration_sub_delay_ms", 0.0)
                )
                or 0.0
            )
            main_delay_default = max(0.0, -sub_delay)
            main_l_delay_raw = alignment.get(
                "main_l_delay_ms",
                ui_data.get("bass_integration_main_l_delay_ms", main_delay_default),
            )
            main_r_delay_raw = alignment.get(
                "main_r_delay_ms",
                ui_data.get("bass_integration_main_r_delay_ms", main_delay_default),
            )
            main_l_delay = _safe_float(
                main_l_delay_raw,
                main_delay_default,
            )
            main_r_delay = _safe_float(
                main_r_delay_raw,
                main_delay_default,
            )
            if (
                main_l_delay == main_l_delay
                and abs(main_l_delay) != float("inf")
                and abs(main_l_delay) > 1e-6
            ) or (
                main_r_delay == main_r_delay
                and abs(main_r_delay) != float("inf")
                and abs(main_r_delay) > 1e-6
            ):
                summary_content += (
                    "CamillaDSP main delay: "
                    f"L {float(main_l_delay):+.2f} ms, R {float(main_r_delay):+.2f} ms\n"
                )
            worst_channel = (
                str(
                    bm.get(
                        "bass_direct_dac_worst_channel",
                        diag.get("dominant_channel", "unknown"),
                    )
                    or "unknown"
                )
                .strip()
                .upper()
            )
            if worst_channel:
                summary_content += f"Worst channel: {worst_channel}\n"
        cancel = _first_finite(
            diag.get("cancellation_risk", float("nan")),
            bm.get("bass_cancellation_risk", float("nan")),
        )
        ripple = _first_finite(
            diag.get("overlap_ripple_db", float("nan")),
            bm.get("bass_overlap_ripple", float("nan")),
        )
        gd_rms = _first_finite(
            diag.get("xo_gd_rms_mismatch_ms", float("nan")),
            gd_cont.get("gd_rms_mismatch_ms_worst", float("nan")),
            bm.get("bass_xo_gd_rms_mismatch_ms", float("nan")),
            bm.get("bass_xo_gd_mismatch_ms", float("nan")),
        )
        feasibility_class = (
            str(
                diag.get("feasibility_class", bm.get("bass_feasibility_class", ""))
                or ""
            )
            .strip()
            .lower()
        )
        feasibility_reason = str(
            diag.get("feasibility_reason", bm.get("bass_feasibility_reason", "")) or ""
        ).strip()
        if feasibility_class:
            feasibility_label = feasibility_class.upper()
            if feasibility_class in ("marginal", "infeasible"):
                feasibility_label += " (SYSTEM-LIMITED)"
            summary_content += f"Feasibility: {feasibility_label}\n"
        hard_gate_failed = bool(
            bm.get("bass_integration_hard_gate_failed", False)
            or "bass_integration_infeasible_hard_gate"
            in list(bm.get("hard_gate_failures", bm.get("hard_gate_reasons", [])) or [])
        )
        hard_gate_reason = str(
            bm.get("bass_integration_hard_gate_reason", feasibility_reason)
            or feasibility_reason
        ).strip()
        if hard_gate_failed:
            summary_content += "Safety gate: Bass Integration infeasible hard gate\n"
            if hard_gate_reason:
                summary_content += f"Safety gate reason: {hard_gate_reason}\n"
        if feasibility_reason:
            summary_content += f"Feasibility reason: {feasibility_reason}\n"
        if cancel == cancel and abs(cancel) != float("inf"):
            summary_content += f"Predicted cancellation risk: {float(cancel):.3f}\n"
        if ripple == ripple and abs(ripple) != float("inf"):
            summary_content += f"Overlap ripple: {float(ripple):.3f} dB p2p\n"
        if gd_rms == gd_rms and abs(gd_rms) != float("inf"):
            summary_content += f"GD band mismatch RMS: {float(gd_rms):.3f} ms\n"
    except Exception:
        logger.exception("LR difference summary section")
    return summary_content


def _append_bass_integration_allpass_auto_summary(
    summary_content: str, data: dict | None
) -> str:
    try:
        ui_data = dict(data or {})
        if not bool(ui_data.get("bass_integration_enable", False)):
            return summary_content
        bi_meta = dict(ui_data.get("_bass_integration_meta", {}) or {})
        allpass_meta = dict(bi_meta.get("recommended_allpass", {}) or {})
        if (
            not bool(ui_data.get("bass_integration_allpass_auto_enable", False))
            and not bool(ui_data.get("bass_integration_allpass_auto_applied", False))
            and not bool(allpass_meta.get("enabled", False))
        ):
            return summary_content
        baseline = dict(bi_meta.get("allpass_baseline_metrics", {}) or {})
        optimized = dict(bi_meta.get("allpass_optimized_metrics", {}) or {})

        def _fmt(v, unit: str = "") -> str:
            x = _safe_float(v, float("nan"))
            if x == x and abs(x) != float("inf"):
                return f"{float(x):.3f}{unit}"
            return "n/a"

        summary_content += "\n=== BASS INTEGRATION ALLPASS AUTO ===\n"
        summary_content += (
            f"State: {'ON' if bool(allpass_meta.get('enabled', False)) else 'OFF'}\n"
        )
        summary_content += "Mode: direct_dac only\n"
        if bool(allpass_meta.get("enabled", False)):
            summary_content += (
                f"Freq: {float(allpass_meta.get('freq_hz', 0.0) or 0.0):.1f} Hz\n"
            )
            summary_content += (
                f"Q: {float(allpass_meta.get('q', 0.707) or 0.707):.3f}\n"
            )
        else:
            summary_content += "Freq: n/a\n"
            summary_content += "Q: n/a\n"
        summary_content += "Baseline metrics:\n"
        summary_content += f"- Cancellation risk: {_fmt(baseline.get('cancellation_risk', float('nan')))}\n"
        summary_content += f"- Overlap ripple: {_fmt(baseline.get('overlap_ripple_db', float('nan')), ' dB p2p')}\n"
        summary_content += f"- XO GD mismatch: {_fmt(baseline.get('xo_gd_mismatch_ms', float('nan')), ' ms')}\n"
        summary_content += "Optimized metrics:\n"
        summary_content += f"- Cancellation risk: {_fmt(optimized.get('cancellation_risk', float('nan')))}\n"
        summary_content += f"- Overlap ripple: {_fmt(optimized.get('overlap_ripple_db', float('nan')), ' dB p2p')}\n"
        summary_content += f"- XO GD mismatch: {_fmt(optimized.get('xo_gd_mismatch_ms', float('nan')), ' ms')}\n"
        summary_content += f"Improvement score: {_fmt(allpass_meta.get('improvement_score', float('nan')))}\n"
        summary_content += f"Reason: {allpass_meta.get('reason', ui_data.get('bass_integration_allpass_reason', '')) or ''!s}\n"
        summary_content += "Does not change FIR generation.\n"
        summary_content += "Applied in the exported CamillaDSP Direct DAC sub pipeline when State is ON.\n"
        summary_content += "HLC config export is unchanged.\n"
    except Exception:
        logger.exception("bass integration allpass auto summary section")
    return summary_content


__all__ = ['_append_bass_integration_summary', '_append_bass_integration_allpass_auto_summary']

