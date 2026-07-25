# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""NiceGUI results rendering after a DSP run.

Replaces camillafir_ui._render_results() + results_sections.py.

render_results() has the same signature as _render_results() so ng_bridge.py
can call it without changes to workflow code.
"""
from __future__ import annotations

import logging
from typing import Any

from .overview import (
    _format_recommended_xo_hz,
)
from .section import _section

from ...resources.i8n.decaycore_i18n import t
from ..bass_integration_dsp_settings import build_bass_integration_dsp_settings
from ..results_formatters import (
    metric_row,
    safe_float,
)

logger = logging.getLogger("DecayCore")


def _format_main_sub_gd_assessment(main_gd_ms: float, sub_gd_ms: float) -> str:
    if any(value != value or abs(value) == float("inf") for value in (main_gd_ms, sub_gd_ms)):
        return "n/a"

    delta_ms = abs(float(main_gd_ms) - float(sub_gd_ms))
    if delta_ms < 1.5:
        key = "results_value_bass_xo_main_sub_gd_assessment_good"
        color = "#22c55e"
    elif delta_ms < 5.0:
        key = "results_value_bass_xo_main_sub_gd_assessment_average"
        color = "#f59e0b"
    elif delta_ms > 8.0:
        key = "results_value_bass_xo_main_sub_gd_assessment_reposition"
        color = "#ef4444"
    else:
        key = "results_value_bass_xo_main_sub_gd_assessment_elevated"
        color = "#ef4444"
    message = t(key).format(delta=f"{delta_ms:.3f}")
    return f"<span style='color:{color};font-weight:600;'>{message}</span>"


def _render_bass_integration(*, data: dict) -> None:
    if not bool((data or {}).get("bass_integration_enable", False)):
        return

    dsp_settings = build_bass_integration_dsp_settings(data)
    bi_meta = dict((data or {}).get("_bass_integration_meta", {}) or {})
    auto_meta = dict((data or {}).get("_auto_mode_meta", {}) or {})
    best_metrics = dict(auto_meta.get("best_metrics", {}) or {})
    diag = dict(bi_meta.get("diagnostics", {}) or {})
    gd_cont = dict(diag.get("gd_continuity", {}) or {})
    sub_st = dict(bi_meta.get("sub_filter_stats", {}) or {})
    allpass_meta = dict(bi_meta.get("recommended_allpass", {}) or {})
    alignment_meta = dict(bi_meta.get("alignment", {}) or {})
    allpass_baseline = dict(bi_meta.get("allpass_baseline_metrics", {}) or {})
    allpass_optimized = dict(bi_meta.get("allpass_optimized_metrics", {}) or {})
    lf_rolloff = dict(diag.get("lf_rolloff", {}) or {})
    allpass_auto_enabled = bool(data.get("bass_integration_allpass_auto_enable", False))
    allpass_on = bool(allpass_meta.get("enabled", False))
    combine_mode = str(
        diag.get(
            "sub_combine_mode",
            bi_meta.get("sub_combine_mode", data.get("bass_integration_sub_combine_mode", "average")),
        )
        or "average"
    ).strip().lower()
    is_dual_sub_prealigned = bool(diag.get("dual_sub_preprocessing_applied", False)) or combine_mode == "dual_sub_peak_aligned_average"
    metric_channel_mode = str(
        diag.get(
            "metric_channel_mode",
            bi_meta.get("metric_channel_mode", best_metrics.get("bass_metric_channel_mode", "worst_case")),
        )
        or "worst_case"
    )

    def _first_finite(*values: Any) -> float:
        for value in values:
            parsed = safe_float(value, float("nan"))
            if parsed == parsed and abs(parsed) != float("inf"):
                return float(parsed)
        return float("nan")

    cancellation_risk = _first_finite(
        diag.get("cancellation_risk", float("nan")),
        best_metrics.get("bass_cancellation_risk", float("nan")),
    )
    overlap_ripple = _first_finite(
        diag.get("overlap_ripple_db", float("nan")),
        best_metrics.get("bass_overlap_ripple", float("nan")),
    )
    sub_dominance = _first_finite(
        diag.get("sub_dominance_db", float("nan")),
        best_metrics.get("bass_sub_dominance", float("nan")),
    )
    overlap_ripple_delta = _first_finite(
        diag.get("overlap_ripple_delta_db", float("nan")),
        best_metrics.get("bass_overlap_ripple_delta_db", float("nan")),
    )
    sub_dominance_delta = _first_finite(
        diag.get("sub_dominance_delta_db", float("nan")),
        best_metrics.get("bass_sub_dominance_delta_db", float("nan")),
    )
    null_severity = _first_finite(
        diag.get("null_severity", float("nan")),
        best_metrics.get("bass_null_severity", float("nan")),
    )
    xo_gd_mismatch = _first_finite(
        diag.get("xo_gd_rms_mismatch_ms", float("nan")),
        gd_cont.get("gd_rms_mismatch_ms_worst", float("nan")),
        best_metrics.get("bass_xo_gd_rms_mismatch_ms", float("nan")),
        best_metrics.get("bass_xo_gd_mismatch_ms", float("nan")),
    )
    xo_gd_delta = _first_finite(
        diag.get("xo_gd_mismatch_delta_ms", float("nan")),
        best_metrics.get("bass_xo_gd_mismatch_delta_ms", float("nan")),
    )
    xo_gd_max = _first_finite(
        diag.get("xo_gd_max_mismatch_ms", float("nan")),
        gd_cont.get("gd_max_mismatch_ms_worst", float("nan")),
        best_metrics.get("bass_xo_gd_max_mismatch_ms", float("nan")),
    )
    xo_main_gd = _first_finite(
        diag.get("xo_main_gd_ms", float("nan")),
        (
            safe_float(gd_cont.get("l_main_gd_ms", float("nan")), float("nan"))
            + safe_float(gd_cont.get("r_main_gd_ms", float("nan")), float("nan"))
        )
        / 2.0,
        best_metrics.get("bass_xo_main_gd_ms", float("nan")),
    )
    xo_sub_gd = _first_finite(
        diag.get("xo_sub_gd_ms", float("nan")),
        gd_cont.get("sub_gd_ms", float("nan")),
        best_metrics.get("bass_xo_sub_gd_ms", float("nan")),
    )
    main_sub_gd_assessment = _format_main_sub_gd_assessment(xo_main_gd, xo_sub_gd)
    sub_level_delta_20_120 = safe_float(diag.get("sub_combined_level_delta_db_20_120", float("nan")), float("nan"))
    sub_level_delta_30_90 = safe_float(diag.get("sub_combined_level_delta_db_30_90", float("nan")), float("nan"))
    dominant_channel = str(
        diag.get("dominant_channel", best_metrics.get("bass_dominant_channel", "unknown")) or "unknown"
    ).strip().lower()
    feasibility_class = str(
        diag.get("feasibility_class", best_metrics.get("bass_feasibility_class", "unknown")) or "unknown"
    ).strip().lower()
    feasibility_reason = str(
        diag.get("feasibility_reason", best_metrics.get("bass_feasibility_reason", "")) or ""
    ).strip()

    def _fmt(v: float, unit: str = "") -> str:
        if v != v or abs(v) == float("inf"):
            return "n/a"
        return f"{float(v):.3f}{unit}"

    def _fmt_transition(before: float, after: float, unit: str = "") -> str:
        before_ok = before == before and abs(before) != float("inf")
        after_ok = after == after and abs(after) != float("inf")
        if before_ok and after_ok:
            return f"{float(before):.3f}{unit} -> {float(after):.3f}{unit}"
        if after_ok:
            return _fmt(after, unit)
        if before_ok:
            return _fmt(before, unit)
        return "n/a"

    def _translate_feasibility(value: str) -> str:
        key = f"results_value_bass_feasibility_{str(value or 'unknown').strip().lower()}"
        translated = t(key)
        return translated if translated != key else str(value or "unknown")

    def _translate_channel(value: str) -> str:
        key = f"results_value_bass_channel_{str(value or 'unknown').strip().lower()}"
        translated = t(key)
        return translated if translated != key else str(value or "unknown")

    bi_mode = "direct_dac"
    bi_mode_label = t("bi_mode_direct_dac")
    xo_metric_label = t("results_metric_main_hpf")
    xo_rec_metric_label = t("results_metric_main_hpf_recommended")
    sub_lpf_metric_label = t("results_metric_sub_lpf")
    sub_lpf_rec_metric_label = t("results_metric_sub_lpf_recommended")
    playback_note = (
        t("bass_integration_direct_playback_match")
        if bi_mode == "direct_dac"
        else t("bass_integration_playback_match")
    )
    actual_main_hpf_hz = float(bi_meta.get("avr_crossover_hz", data.get("avr_crossover_hz", 80.0)) or 80.0)
    actual_sub_lpf_hz = float(
        bi_meta.get(
            "direct_dac_sub_lpf_hz",
            data.get("direct_dac_sub_lpf_hz", actual_main_hpf_hz),
        )
        or actual_main_hpf_hz
    )
    baseline_cancel = safe_float(allpass_baseline.get("cancellation_risk", float("nan")), float("nan"))
    baseline_ripple = safe_float(allpass_baseline.get("overlap_ripple_db", float("nan")), float("nan"))
    baseline_gd = safe_float(allpass_baseline.get("xo_gd_mismatch_ms", float("nan")), float("nan"))
    optimized_cancel = _first_finite(
        diag.get("cancellation_risk", float("nan")),
        best_metrics.get("bass_cancellation_risk", float("nan")),
        allpass_optimized.get("cancellation_risk", cancellation_risk),
    )
    optimized_ripple = _first_finite(
        diag.get("overlap_ripple_db", float("nan")),
        best_metrics.get("bass_overlap_ripple", float("nan")),
        allpass_optimized.get("overlap_ripple_db", overlap_ripple),
    )
    optimized_gd = _first_finite(
        diag.get("xo_gd_rms_mismatch_ms", float("nan")),
        gd_cont.get("gd_rms_mismatch_ms_worst", float("nan")),
        best_metrics.get("bass_xo_gd_rms_mismatch_ms", float("nan")),
        best_metrics.get("bass_xo_gd_mismatch_ms", float("nan")),
        allpass_optimized.get("xo_gd_mismatch_ms", xo_gd_mismatch),
    )
    sub_delay_value = safe_float(
        alignment_meta.get("delay_ms", data.get("bass_integration_sub_delay_ms", float("nan"))),
        float("nan"),
    )
    sub_array_delay_value = safe_float(
        alignment_meta.get(
            "sub_array_delay_ms",
            data.get("bass_integration_sub_array_delay_ms", alignment_meta.get("delay_ms", float("nan"))),
        ),
        float("nan"),
    )
    main_l_delay_value = safe_float(
        alignment_meta.get("main_l_delay_ms", data.get("bass_integration_main_l_delay_ms", float("nan"))),
        float("nan"),
    )
    main_r_delay_value = safe_float(
        alignment_meta.get("main_r_delay_ms", data.get("bass_integration_main_r_delay_ms", float("nan"))),
        float("nan"),
    )
    main_delay_label = f"L {_fmt(main_l_delay_value, ' ms')}, R {_fmt(main_r_delay_value, ' ms')}"
    shared_branch_label = t("results_value_bass_shared_sub_branch") if is_dual_sub_prealigned else None
    lf_rolloff_used = bool(lf_rolloff.get("used_measurement", False))
    lf_rolloff_f6_hz = safe_float(lf_rolloff.get("f6_hz", float("nan")), float("nan"))
    lf_rolloff_value = _fmt(lf_rolloff_f6_hz, " Hz") if lf_rolloff_used else t("results_value_bass_f6_unavailable")
    lf_rolloff_source = (
        f"{lf_rolloff.get('source', 'n/a') or 'n/a'!s}, "
        f"{safe_float(lf_rolloff.get('confidence', 0.0), 0.0) * 100.0:.0f}%"
        if lf_rolloff_used
        else t("results_value_bass_f6_unavailable")
    )

    if dsp_settings:
        _section(
            t("results_section_bass_dsp_settings"),
            [metric_row(t(setting.label_key), setting.value, setting.value) for setting in dsp_settings],
            summary_lines=[t("bass_integration_dsp_settings_note")],
        )

    _section(
        t("results_section_bass_integration"),
        [
            metric_row(
                t("results_metric_bass_integration_mode"),
                bi_mode_label,
                bi_mode_label,
            ),
            metric_row(
                xo_metric_label,
                f"{actual_main_hpf_hz:.1f} Hz",
                f"{actual_main_hpf_hz:.1f} Hz",
            ),
            *(
                [
                    metric_row(
                        sub_lpf_metric_label,
                        f"{actual_sub_lpf_hz:.1f} Hz",
                        f"{actual_sub_lpf_hz:.1f} Hz",
                    )
                ]
                if bi_mode == "direct_dac"
                else []
            ),
            *(
                (
                    [
                        metric_row(
                            t("results_metric_main_sub_crossover_recommended_l"),
                            _format_recommended_xo_hz(float(bi_meta["recommended_crossover_hz_l"])),
                            _format_recommended_xo_hz(float(bi_meta["recommended_crossover_hz_l"])),
                        ),
                        metric_row(
                            t("results_metric_main_sub_crossover_recommended_r"),
                            _format_recommended_xo_hz(float(bi_meta["recommended_crossover_hz_r"])),
                            _format_recommended_xo_hz(float(bi_meta["recommended_crossover_hz_r"])),
                        ),
                        *(
                            [
                                metric_row(
                                    sub_lpf_rec_metric_label,
                                    _format_recommended_xo_hz(float(bi_meta["recommended_sub_lpf_hz"])),
                                    _format_recommended_xo_hz(float(bi_meta["recommended_sub_lpf_hz"])),
                                )
                            ]
                            if bi_meta.get("recommended_sub_lpf_hz") is not None
                            else []
                        ),
                    ]
                    if (
                        bi_mode == "direct_dac"
                        and bi_meta.get("recommended_crossover_hz_l") is not None
                        and bi_meta.get("recommended_crossover_hz_r") is not None
                        and abs(
                            float(bi_meta["recommended_crossover_hz_l"])
                            - float(bi_meta["recommended_crossover_hz_r"])
                        ) >= 1.0
                    )
                    else [
                        metric_row(
                            xo_rec_metric_label,
                            _format_recommended_xo_hz(float(bi_meta["recommended_crossover_hz"])),
                            _format_recommended_xo_hz(float(bi_meta["recommended_crossover_hz"])),
                        ),
                        *(
                            [
                                metric_row(
                                    sub_lpf_rec_metric_label,
                                    _format_recommended_xo_hz(float(bi_meta["recommended_sub_lpf_hz"])),
                                    _format_recommended_xo_hz(float(bi_meta["recommended_sub_lpf_hz"])),
                                )
                            ]
                            if bi_mode == "direct_dac" and bi_meta.get("recommended_sub_lpf_hz") is not None
                            else []
                        ),
                    ]
                )
                if bi_meta.get("recommended_crossover_hz") is not None
                else []
            ),
            metric_row(
                t("results_metric_bass_integration_profile"),
                str(bi_meta.get("profile", data.get("bass_integration_profile", "safe")) or "safe"),
                str(bi_meta.get("profile", data.get("bass_integration_profile", "safe")) or "safe"),
            ),
            metric_row(
                t("results_metric_bass_sub_combine_mode"),
                str(combine_mode),
                str(combine_mode),
            ),
            metric_row(
                t("results_metric_bass_metric_channel_mode"),
                str(metric_channel_mode),
                str(metric_channel_mode),
            ),
            metric_row(
                t("results_metric_bass_feasibility"),
                _translate_feasibility(feasibility_class),
                _translate_feasibility(feasibility_class),
            ),
            metric_row(
                t("results_metric_bass_feasibility_reason"),
                feasibility_reason or "n/a",
                feasibility_reason or "n/a",
            ),
            *(
                [
                    metric_row(t("results_metric_bass_main_f6"), lf_rolloff_value, lf_rolloff_value),
                    metric_row(t("results_metric_bass_main_f6_source"), lf_rolloff_source, lf_rolloff_source),
                ]
                if bi_mode == "direct_dac"
                else []
            ),
            metric_row(
                t("results_metric_bass_dominant_channel"),
                _translate_channel(dominant_channel),
                _translate_channel(dominant_channel),
            ),
            *(
                [
                    metric_row(
                        t("results_metric_bass_shared_sub_branch"),
                        shared_branch_label or "n/a",
                        shared_branch_label or "n/a",
                    )
                ]
                if is_dual_sub_prealigned
                else []
            ),
            metric_row(
                t(
                    "results_metric_bass_dual_sub_alignment_delta_20_120"
                    if is_dual_sub_prealigned
                    else "results_metric_bass_sub_level_delta_20_120"
                ),
                _fmt(sub_level_delta_20_120, " dB"),
                _fmt(sub_level_delta_20_120, " dB"),
            ),
            metric_row(
                t(
                    "results_metric_bass_dual_sub_alignment_delta_30_90"
                    if is_dual_sub_prealigned
                    else "results_metric_bass_sub_level_delta_30_90"
                ),
                _fmt(sub_level_delta_30_90, " dB"),
                _fmt(sub_level_delta_30_90, " dB"),
            ),
            metric_row(
                t("results_metric_bass_alignment_auto"),
                t("state_on") if bool(alignment_meta.get("applied", False)) else t("state_off"),
                t("state_on") if bool(alignment_meta.get("applied", False)) else t("state_off"),
            ),
            metric_row(
                t(
                    "results_metric_bass_shared_sub_array_delay"
                    if is_dual_sub_prealigned
                    else "results_metric_bass_alignment_delay"
                ),
                _fmt(sub_array_delay_value if is_dual_sub_prealigned else sub_delay_value, " ms"),
                _fmt(sub_array_delay_value if is_dual_sub_prealigned else sub_delay_value, " ms"),
            ),
            metric_row(
                t("results_metric_bass_main_delay"),
                main_delay_label,
                main_delay_label,
            ),
            metric_row(
                t(
                    "results_metric_bass_shared_sub_polarity"
                    if is_dual_sub_prealigned
                    else "results_metric_bass_alignment_polarity"
                ),
                t("ir_align_value_inverted")
                if bool(alignment_meta.get("polarity_invert", data.get("bass_integration_sub_polarity_invert", False)))
                else t("ir_align_value_ok"),
                t("ir_align_value_inverted")
                if bool(alignment_meta.get("polarity_invert", data.get("bass_integration_sub_polarity_invert", False)))
                else t("ir_align_value_ok"),
            ),
            metric_row(
                t(
                    "results_metric_bass_shared_sub_gain"
                    if is_dual_sub_prealigned
                    else "results_metric_bass_alignment_gain"
                ),
                _fmt(safe_float(alignment_meta.get("gain_trim_db", data.get("bass_integration_sub_gain_trim_db", float("nan"))), float("nan")), " dB"),
                _fmt(safe_float(alignment_meta.get("gain_trim_db", data.get("bass_integration_sub_gain_trim_db", float("nan"))), float("nan")), " dB"),
            ),
            metric_row(
                t("results_metric_bass_shared_sub_allpass" if is_dual_sub_prealigned else "results_metric_bass_allpass"),
                t("state_on") if allpass_on else t("state_off"),
                t("state_on") if allpass_on else t("state_off"),
            ),
            metric_row(
                t(
                    "results_metric_bass_shared_sub_allpass_freq"
                    if is_dual_sub_prealigned
                    else "results_metric_bass_allpass_freq"
                ),
                f"{float(allpass_meta.get('freq_hz', 0.0) or 0.0):.1f} Hz" if allpass_on else "n/a",
                f"{float(allpass_meta.get('freq_hz', 0.0) or 0.0):.1f} Hz" if allpass_on else "n/a",
            ),
            metric_row(
                t(
                    "results_metric_bass_shared_sub_allpass_q"
                    if is_dual_sub_prealigned
                    else "results_metric_bass_allpass_q"
                ),
                f"{float(allpass_meta.get('q', 0.707) or 0.707):.3f}" if allpass_on else "n/a",
                f"{float(allpass_meta.get('q', 0.707) or 0.707):.3f}" if allpass_on else "n/a",
            ),
            metric_row(
                t("results_metric_bass_allpass_improvement"),
                _fmt(safe_float(allpass_meta.get("improvement_score", float("nan")), float("nan"))),
                _fmt(safe_float(allpass_meta.get("improvement_score", float("nan")), float("nan"))),
            ),
            metric_row(
                t("results_metric_bass_cancellation_risk"),
                _fmt_transition(baseline_cancel, optimized_cancel),
                _fmt_transition(baseline_cancel, optimized_cancel),
            ),
            metric_row(
                t("results_metric_bass_overlap_smoothness"),
                _fmt_transition(baseline_ripple, optimized_ripple, " dB p2p"),
                _fmt_transition(baseline_ripple, optimized_ripple, " dB p2p"),
            ),
            metric_row(
                t("results_metric_bass_overlap_delta"),
                _fmt(overlap_ripple_delta, " dB"),
                _fmt(overlap_ripple_delta, " dB"),
            ),
            metric_row(
                t("results_metric_bass_sub_dominance"),
                _fmt(sub_dominance, " dB"),
                _fmt(sub_dominance, " dB"),
            ),
            metric_row(
                t("results_metric_bass_sub_dominance_delta"),
                _fmt(sub_dominance_delta, " dB"),
                _fmt(sub_dominance_delta, " dB"),
            ),
            metric_row(
                t("results_metric_bass_null_severity"),
                _fmt(null_severity),
                _fmt(null_severity),
            ),
            metric_row(
                t("results_metric_bass_xo_gd_band_rms"),
                _fmt_transition(baseline_gd, optimized_gd, " ms"),
                _fmt_transition(baseline_gd, optimized_gd, " ms"),
            ),
            metric_row(
                t("results_metric_bass_xo_gd_band_max"),
                _fmt(xo_gd_max, " ms"),
                _fmt(xo_gd_max, " ms"),
            ),
            metric_row(
                t("results_metric_bass_xo_gd_delta"),
                _fmt(xo_gd_delta, " ms"),
                _fmt(xo_gd_delta, " ms"),
            ),
            metric_row(
                t("results_metric_bass_xo_main_gd"),
                _fmt(xo_main_gd, " ms"),
                _fmt(xo_main_gd, " ms"),
            ),
            metric_row(
                t("results_metric_bass_xo_sub_gd"),
                _fmt(xo_sub_gd, " ms"),
                _fmt(xo_sub_gd, " ms"),
            ),
            metric_row(
                t("results_metric_bass_xo_main_sub_gd_assessment"),
                main_sub_gd_assessment,
                main_sub_gd_assessment,
            ),
            *(
                [
                    metric_row(
                        t("results_metric_sub_filter_boost"),
                        _fmt(safe_float(sub_st.get("max_boost_db_effective", sub_st.get("max_boost_db", float("nan"))), float("nan")), " dB"),
                        _fmt(safe_float(sub_st.get("max_boost_db_effective", sub_st.get("max_boost_db", float("nan"))), float("nan")), " dB"),
                    ),
                    metric_row(
                        t("results_metric_sub_filter_cut"),
                        _fmt(safe_float(sub_st.get("max_cut_db", float("nan")), float("nan")), " dB"),
                        _fmt(safe_float(sub_st.get("max_cut_db", float("nan")), float("nan")), " dB"),
                    ),
                    metric_row(
                        t("results_metric_sub_filter_gain_margin"),
                        _fmt(safe_float(sub_st.get("gain_margin_db", float("nan")), float("nan")), " dB"),
                        _fmt(safe_float(sub_st.get("gain_margin_db", float("nan")), float("nan")), " dB"),
                    ),
                    metric_row(
                        t("results_metric_sub_filter_applied_gain"),
                        _fmt(safe_float(sub_st.get("auto_global_gain_db", float("nan")), float("nan")), " dB"),
                        _fmt(safe_float(sub_st.get("auto_global_gain_db", float("nan")), float("nan")), " dB"),
                    ),
                    metric_row(
                        t("results_metric_sub_filter_confidence"),
                        _fmt(safe_float(sub_st.get("avg_confidence", float("nan")), float("nan")), "%"),
                        _fmt(safe_float(sub_st.get("avg_confidence", float("nan")), float("nan")), "%"),
                    ),
                ]
                if sub_st
                else []
            ),
        ],
        summary_lines=[
            playback_note,
            *(
                [t("bass_allpass_no_improvement")]
                if bi_mode == "direct_dac" and allpass_auto_enabled and not allpass_on
                else []
            ),
        ],
    )


__all__ = ['_render_bass_integration']
