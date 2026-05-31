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

def _append_bass_integration_summary(summary_content: str, data: dict | None) -> str:
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
        xo_label = "Main HPF"
        xo_rec_label = "Recommended Main HPF"
        xo_value = float(
            bi_meta.get(
                "avr_crossover_hz",
                ui_data.get(
                    "sub_crossover_hz",
                    ui_data.get("avr_crossover_hz", 80.0),
                ),
            )
            or 80.0
        )
        sub_lpf_value = float(
            bi_meta.get(
                "direct_dac_sub_lpf_hz",
                ui_data.get("direct_dac_sub_lpf_hz", xo_value),
            )
            or xo_value
        )
        playback_note = "Direct-DAC XO is the crossover between the main-speaker HPF and the subwoofer LPF.\n"
        xo_note = "AUTO may use overlap by setting Sub FIR LPF above Main HPF.\n"
        summary_content += (
            f"Mode: {bi_mode}\n"
        )
        summary_content += f"{xo_label}: {float(xo_value):.1f} Hz\n"
        if raw_bi_mode == "direct_dac":
            summary_content += f"Sub LPF: {float(sub_lpf_value):.1f} Hz\n"
            sub_target_lpf = _safe_float(
                bi_meta.get("sub_target_lpf_hz", ui_data.get("sub_target_lpf_hz", float("nan"))),
                float("nan"),
            )
            sub_target_slope = _safe_float(
                bi_meta.get(
                    "sub_target_lpf_slope_db_per_oct",
                    ui_data.get("sub_target_lpf_slope_db_per_oct", float("nan")),
                ),
                float("nan"),
            )
            if (
                sub_target_lpf == sub_target_lpf
                and abs(sub_target_lpf) != float("inf")
                and sub_target_slope == sub_target_slope
                and abs(sub_target_slope) != float("inf")
            ):
                summary_content += (
                    f"Sub target LPF: {float(sub_target_lpf):.0f} Hz, "
                    f"{float(sub_target_slope):.0f} dB/oct\n"
                )
            diag_direct = dict(bi_meta.get("diagnostics", {}) or {})
            sub_hpf_value = _safe_float(
                diag_direct.get("direct_dac_sub_hpf_hz", ui_data.get("sub_hpf_freq", 20.0)),
                20.0,
            )
            overlap_value = _safe_float(
                diag_direct.get("direct_dac_sub_overlap_hz", max(0.0, float(sub_lpf_value) - float(xo_value))),
                max(0.0, float(sub_lpf_value) - float(xo_value)),
            )
            summary_content += "Optimization: automatic\n"
            summary_content += f"Sub HPF: {float(sub_hpf_value):.1f} Hz\n"
            summary_content += f"Sub overlap above main XO: +{float(overlap_value):.1f} Hz\n"
            if float(xo_value) > 0.0:
                summary_content += f"Overlap ratio: {float(sub_lpf_value) / float(xo_value):.2f}\n"
            summary_content += "Export model: CamillaDSP-compatible\n"
        rec_hz = bi_meta.get("recommended_crossover_hz", None)
        rec_sub_lpf_hz = bi_meta.get("recommended_sub_lpf_hz", None)
        rec_hz_l = bi_meta.get("recommended_crossover_hz_l", None)
        rec_hz_r = bi_meta.get("recommended_crossover_hz_r", None)
        if (
            raw_bi_mode == "direct_dac"
            and rec_hz_l is not None
            and rec_hz_r is not None
            and abs(float(rec_hz_l) - float(rec_hz_r)) >= 1.0
        ):
            summary_content += f"Recommended Main HPF L: {_format_recommended_xo_hz(float(rec_hz_l))}\n"
            summary_content += f"Recommended Main HPF R: {_format_recommended_xo_hz(float(rec_hz_r))}\n"
        elif rec_hz is not None:
            summary_content += f"{xo_rec_label}: {_format_recommended_xo_hz(float(rec_hz))}\n"
        if raw_bi_mode == "direct_dac" and rec_sub_lpf_hz is not None:
            summary_content += f"Recommended Sub LPF: {_format_recommended_xo_hz(float(rec_sub_lpf_hz))}\n"
        summary_content += (
            f"Profile: {str(bi_meta.get('profile', ui_data.get('bass_integration_profile', 'safe')) or 'safe')}\n"
        )
        summary_content += (
            f"Sub combine mode: {str(bi_meta.get('sub_combine_mode', ui_data.get('bass_integration_sub_combine_mode', 'average')) or 'average')}\n"
        )
        dual_sub_diag = dict(bi_meta.get("diagnostics", {}) or {})
        if bool(dual_sub_diag.get("dual_sub_preprocessing_applied", False)):
            sub1_peak_ms = _safe_float(dual_sub_diag.get("dual_sub_sub1_peak_ms", float("nan")), float("nan"))
            sub2_peak_ms = _safe_float(dual_sub_diag.get("dual_sub_sub2_peak_ms", float("nan")), float("nan"))
            delay_ms = _safe_float(dual_sub_diag.get("dual_sub_relative_delay_ms", float("nan")), float("nan"))
            summary_content += "Bass Integration dual-sub preprocessing:\n"
            summary_content += "Sub topology: dual-sub vector-average reference\n"
            if sub1_peak_ms == sub1_peak_ms and abs(sub1_peak_ms) != float("inf"):
                summary_content += f"- SUB1 peak: {float(sub1_peak_ms):.3f} ms\n"
            if sub2_peak_ms == sub2_peak_ms and abs(sub2_peak_ms) != float("inf"):
                summary_content += f"- SUB2 peak: {float(sub2_peak_ms):.3f} ms\n"
            if delay_ms == delay_ms and abs(delay_ms) != float("inf"):
                summary_content += f"- Applied relative delay: {float(delay_ms):+.3f} ms\n"
            summary_content += "- Combined sub response: vector average of aligned SUB1 + SUB2\n"
            summary_content += "- Combined method: complex vector average after peak alignment\n"
            summary_content += "- Per-sub optimization: no\n"
            summary_content += "- This is a combined acoustic sub reference, not independent per-sub optimization.\n"
        inputs = dict(bi_meta.get("inputs", {}) or {})
        summary_content += "Inputs used:\n"
        summary_content += f"L main: {str(inputs.get('l_main', '') or 'n/a')}\n"
        summary_content += f"R main: {str(inputs.get('r_main', '') or 'n/a')}\n"
        summary_content += f"L sub: {str(inputs.get('l_sub', '') or 'n/a')}\n"
        summary_content += f"R sub: {str(inputs.get('r_sub', '') or 'n/a')}\n"
        if str(ui_data.get("bass_integration_mode", "") or "").strip().lower() == "direct_dac":
            summary_content += (
                "Main-speaker FIR optimized from main-only measurements. "
                "Sub FIR optimized from sub-only measurements.\n"
            )
        else:
            summary_content += (
                "Predicted totals used for FIR optimization: "
                "L_total_pred = L_main + combined_sub, "
                "R_total_pred = R_main + combined_sub.\n"
            )
        summary_content += playback_note
        summary_content += xo_note
        diag = dict(bi_meta.get("diagnostics", {}) or {})
        alignment = dict(bi_meta.get("alignment", {}) or {})
        metric_channel_mode = str(
            bm.get("bass_metric_channel_mode", diag.get("metric_channel_mode", "worst_case")) or "worst_case"
        )
        summary_content += f"Metric channel mode: {metric_channel_mode}\n"
        delta_20_120 = _safe_float(
            diag.get("sub_combined_level_delta_db_20_120", float("nan")),
            float("nan"),
        )
        delta_30_90 = _safe_float(
            diag.get("sub_combined_level_delta_db_30_90", float("nan")),
            float("nan"),
        )
        if delta_20_120 == delta_20_120 and abs(delta_20_120) != float("inf"):
            summary_content += f"Combined-sub level delta 20-120 Hz: {float(delta_20_120):+.3f} dB\n"
        if delta_30_90 == delta_30_90 and abs(delta_30_90) != float("inf"):
            summary_content += f"Combined-sub level delta 30-90 Hz: {float(delta_30_90):+.3f} dB\n"
        if raw_bi_mode == "direct_dac":
            summary_content += f"Auto alignment applied: {'YES' if bool(alignment.get('applied', False)) else 'NO'}\n"
            summary_content += f"Sub delay: {float(alignment.get('delay_ms', ui_data.get('bass_integration_sub_delay_ms', 0.0)) or 0.0):+.2f} ms\n"
            summary_content += (
                f"Sub polarity: {'INVERTED' if bool(alignment.get('polarity_invert', ui_data.get('bass_integration_sub_polarity_invert', False))) else 'NORMAL'}\n"
            )
            summary_content += f"Sub gain trim: {float(alignment.get('gain_trim_db', ui_data.get('bass_integration_sub_gain_trim_db', 0.0)) or 0.0):+.2f} dB\n"
            worst_channel = str(
                bm.get("bass_direct_dac_worst_channel", diag.get("dominant_channel", "unknown")) or "unknown"
            ).strip().upper()
            if worst_channel:
                summary_content += f"Worst channel: {worst_channel}\n"
            phase_err = _safe_float(diag.get("direct_dac_phase_error_deg", float("nan")), float("nan"))
            direct_gd = _safe_float(diag.get("direct_dac_gd_mismatch_ms", float("nan")), float("nan"))
            direct_cancel = str(diag.get("direct_dac_cancellation_risk", "") or "").strip().lower()
            if phase_err == phase_err and abs(phase_err) != float("inf"):
                summary_content += f"XO phase error: {float(phase_err):.1f} deg\n"
            if direct_gd == direct_gd and abs(direct_gd) != float("inf"):
                summary_content += f"XO GD mismatch: {float(direct_gd):.3f} ms\n"
            if direct_cancel:
                summary_content += f"Cancellation risk: {direct_cancel}\n"
        cancel = _safe_float(bm.get("bass_cancellation_risk", float("nan")), float("nan"))
        ripple = _safe_float(bm.get("bass_overlap_ripple", float("nan")), float("nan"))
        dominance = _safe_float(bm.get("bass_sub_dominance", float("nan")), float("nan"))
        ripple_delta = _safe_float(
            bm.get("bass_overlap_ripple_delta_db", diag.get("overlap_ripple_delta_db", float("nan"))),
            float("nan"),
        )
        dominance_delta = _safe_float(
            bm.get("bass_sub_dominance_delta_db", diag.get("sub_dominance_delta_db", float("nan"))),
            float("nan"),
        )
        null_severity = _safe_float(bm.get("bass_null_severity", float("nan")), float("nan"))
        gd_rms = _safe_float(
            bm.get("bass_xo_gd_rms_mismatch_ms", bm.get("bass_xo_gd_mismatch_ms", float("nan"))),
            float("nan"),
        )
        gd_max = _safe_float(bm.get("bass_xo_gd_max_mismatch_ms", float("nan")), float("nan"))
        gd_delta = _safe_float(
            bm.get("bass_xo_gd_mismatch_delta_ms", diag.get("gd_mismatch_delta_ms", float("nan"))),
            float("nan"),
        )
        dominant_channel = str(
            bm.get("bass_dominant_channel", diag.get("dominant_channel", "unknown")) or "unknown"
        ).strip().lower()
        feasibility_class = str(
            bm.get("bass_feasibility_class", diag.get("feasibility_class", "")) or ""
        ).strip().lower()
        feasibility_reason = str(
            bm.get("bass_feasibility_reason", diag.get("feasibility_reason", "")) or ""
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
            bm.get("bass_integration_hard_gate_reason", feasibility_reason) or feasibility_reason
        ).strip()
        if hard_gate_failed:
            summary_content += "Safety gate: Bass Integration infeasible hard gate\n"
            if hard_gate_reason:
                summary_content += f"Safety gate reason: {hard_gate_reason}\n"
        if dominant_channel and dominant_channel != "unknown":
            summary_content += f"Dominant / limiting channel: {dominant_channel.upper()}\n"
        if feasibility_reason:
            summary_content += f"Feasibility reason: {feasibility_reason}\n"
        if cancel == cancel and abs(cancel) != float("inf"):
            summary_content += f"Predicted cancellation risk: {float(cancel):.3f}\n"
        if ripple == ripple and abs(ripple) != float("inf"):
            summary_content += f"Overlap ripple: {float(ripple):.3f} dB p2p\n"
        if ripple_delta == ripple_delta and abs(ripple_delta) != float("inf"):
            summary_content += f"Overlap ripple delta L/R: {float(ripple_delta):.3f} dB\n"
        if dominance == dominance and abs(dominance) != float("inf"):
            summary_content += f"Sub dominance: {float(dominance):.3f} dB\n"
        if dominance_delta == dominance_delta and abs(dominance_delta) != float("inf"):
            summary_content += f"Sub dominance delta L/R: {float(dominance_delta):.3f} dB\n"
        if null_severity == null_severity and abs(null_severity) != float("inf"):
            summary_content += f"Null severity: {float(null_severity):.3f}\n"
        if gd_rms == gd_rms and abs(gd_rms) != float("inf"):
            summary_content += f"GD band mismatch RMS: {float(gd_rms):.3f} ms\n"
        if gd_max == gd_max and abs(gd_max) != float("inf"):
            summary_content += f"GD band mismatch MAX: {float(gd_max):.3f} ms\n"
        if gd_delta == gd_delta and abs(gd_delta) != float("inf"):
            summary_content += f"GD mismatch delta L/R: {float(gd_delta):.3f} ms\n"
    except Exception:
        logger.exception("LR difference summary section")
    return summary_content

def _append_bass_integration_allpass_auto_summary(summary_content: str, data: dict | None) -> str:
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
        summary_content += f"State: {'ON' if bool(allpass_meta.get('enabled', False)) else 'OFF'}\n"
        summary_content += "Mode: direct_dac only\n"
        if bool(allpass_meta.get("enabled", False)):
            summary_content += f"Freq: {float(allpass_meta.get('freq_hz', 0.0) or 0.0):.1f} Hz\n"
            summary_content += f"Q: {float(allpass_meta.get('q', 0.707) or 0.707):.3f}\n"
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
        summary_content += f"Reason: {str(allpass_meta.get('reason', ui_data.get('bass_integration_allpass_reason', '')) or '')}\n"
        summary_content += "Does not change FIR generation.\n"
        summary_content += "Applied in the exported CamillaDSP Direct DAC sub pipeline when State is ON.\n"
        summary_content += "HLC config export is unchanged.\n"
    except Exception:
        logger.exception("bass integration allpass auto summary section")
    return summary_content


__all__ = ['_append_bass_integration_summary', '_append_bass_integration_allpass_auto_summary']


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
