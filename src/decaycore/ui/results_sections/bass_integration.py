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

import html
import logging
import math
import time
from typing import Any

# ---------------------------------------------------------------------------
# Interactive plot render cache
# Key: (run_signature, channel, plot_smoothing_level)
# Value: plotly Figure object
# Cleared at the start of each render_results call (per-run semantics).
# ---------------------------------------------------------------------------
_PLOT_RENDER_CACHE: dict = {}

from ...resources.i8n.decaycore_i18n import t
from ...auto_mode.rank_score import calibrated_auto_quality
from .. import decaycore_plot as plots
from .. import ui_state
from ..results_formatters import (
    anchor_label,
    boost_diag,
    fmt_ai_match,
    fmt_ai_score,
    fmt_freq_window,
    fmt_tilt,
    format_ir_window,
    gd_grad_max_label,
    gd_limiter_label,
    hpf_diff_raw_label,
    hpf_model_label,
    metric_row,
    mixed_blend_label,
    phase_clamp_label,
    plot_smoothing_label,
    safe_float,
    shared_window_label,
    stereo_link_mode_label,
    xo_fc_gd_label,
)

from ...dsp.lr_difference_metrics import compute_lr_difference_metrics

logger = logging.getLogger("DecayCore")

def _render_bass_integration(*, data: dict) -> None:
    if not bool((data or {}).get("bass_integration_enable", False)):
        return

    bi_meta = dict((data or {}).get("_bass_integration_meta", {}) or {})
    auto_meta = dict((data or {}).get("_auto_mode_meta", {}) or {})
    best_metrics = dict(auto_meta.get("best_metrics", {}) or {})
    diag = dict(bi_meta.get("diagnostics", {}) or {})
    sub_st = dict(bi_meta.get("sub_filter_stats", {}) or {})
    allpass_meta = dict(bi_meta.get("recommended_allpass", {}) or {})
    alignment_meta = dict(bi_meta.get("alignment", {}) or {})
    allpass_baseline = dict(bi_meta.get("allpass_baseline_metrics", {}) or {})
    allpass_optimized = dict(bi_meta.get("allpass_optimized_metrics", {}) or {})
    allpass_auto_enabled = bool(data.get("bass_integration_allpass_auto_enable", False))
    allpass_on = bool(allpass_meta.get("enabled", False))
    combine_mode = str(
        bi_meta.get("sub_combine_mode", data.get("bass_integration_sub_combine_mode", "average")) or "average"
    )
    metric_channel_mode = str(best_metrics.get("bass_metric_channel_mode", "worst_case") or "worst_case")

    cancellation_risk = safe_float(
        best_metrics.get("bass_cancellation_risk", diag.get("cancellation_risk", float("nan"))),
        float("nan"),
    )
    overlap_ripple = safe_float(
        best_metrics.get("bass_overlap_ripple", diag.get("overlap_ripple_db", float("nan"))),
        float("nan"),
    )
    sub_dominance = safe_float(
        best_metrics.get("bass_sub_dominance", diag.get("sub_dominance_db", float("nan"))),
        float("nan"),
    )
    overlap_ripple_delta = safe_float(
        best_metrics.get("bass_overlap_ripple_delta_db", diag.get("overlap_ripple_delta_db", float("nan"))),
        float("nan"),
    )
    sub_dominance_delta = safe_float(
        best_metrics.get("bass_sub_dominance_delta_db", diag.get("sub_dominance_delta_db", float("nan"))),
        float("nan"),
    )
    null_severity = safe_float(
        best_metrics.get("bass_null_severity", diag.get("null_severity", float("nan"))),
        float("nan"),
    )
    xo_gd_mismatch = safe_float(
        best_metrics.get("bass_xo_gd_rms_mismatch_ms", best_metrics.get("bass_xo_gd_mismatch_ms", float("nan"))),
        float("nan"),
    )
    xo_gd_delta = safe_float(
        best_metrics.get("bass_xo_gd_mismatch_delta_ms", diag.get("gd_mismatch_delta_ms", float("nan"))),
        float("nan"),
    )
    xo_gd_max = safe_float(
        best_metrics.get("bass_xo_gd_max_mismatch_ms", diag.get("gd_max_mismatch_ms_worst", float("nan"))),
        float("nan"),
    )
    xo_main_gd = safe_float(best_metrics.get("bass_xo_main_gd_ms", float("nan")), float("nan"))
    xo_sub_gd = safe_float(best_metrics.get("bass_xo_sub_gd_ms", float("nan")), float("nan"))
    sub_level_delta_20_120 = safe_float(diag.get("sub_combined_level_delta_db_20_120", float("nan")), float("nan"))
    sub_level_delta_30_90 = safe_float(diag.get("sub_combined_level_delta_db_30_90", float("nan")), float("nan"))
    dominant_channel = str(
        best_metrics.get("bass_dominant_channel", diag.get("dominant_channel", "unknown")) or "unknown"
    ).strip().lower()
    feasibility_class = str(
        best_metrics.get("bass_feasibility_class", diag.get("feasibility_class", "unknown")) or "unknown"
    ).strip().lower()
    feasibility_reason = str(
        best_metrics.get("bass_feasibility_reason", diag.get("feasibility_reason", "")) or ""
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

    bi_mode = str(
        bi_meta.get("mode", data.get("bass_integration_mode", "avr_lfe_main_decomposed"))
        or "avr_lfe_main_decomposed"
    ).strip().lower()
    bi_mode_label = (
        t("bi_mode_direct_dac")
        if bi_mode == "direct_dac"
        else t("bass_integration_mode_avr_lfe_main_decomposed")
    )
    xo_metric_label = (
        t("results_metric_main_hpf")
        if bi_mode == "direct_dac"
        else t("results_metric_avr_crossover")
    )
    xo_rec_metric_label = (
        t("results_metric_main_hpf_recommended")
        if bi_mode == "direct_dac"
        else t("results_metric_avr_crossover_recommended")
    )
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
    optimized_cancel = safe_float(
        best_metrics.get("bass_cancellation_risk", allpass_optimized.get("cancellation_risk", cancellation_risk)),
        float("nan"),
    )
    optimized_ripple = safe_float(
        best_metrics.get("bass_overlap_ripple", allpass_optimized.get("overlap_ripple_db", overlap_ripple)),
        float("nan"),
    )
    optimized_gd = safe_float(
        best_metrics.get("bass_xo_gd_mismatch_ms", allpass_optimized.get("xo_gd_mismatch_ms", xo_gd_mismatch)),
        float("nan"),
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
            metric_row(
                t("results_metric_bass_dominant_channel"),
                _translate_channel(dominant_channel),
                _translate_channel(dominant_channel),
            ),
            metric_row(
                t("results_metric_bass_sub_level_delta_20_120"),
                _fmt(sub_level_delta_20_120, " dB"),
                _fmt(sub_level_delta_20_120, " dB"),
            ),
            metric_row(
                t("results_metric_bass_sub_level_delta_30_90"),
                _fmt(sub_level_delta_30_90, " dB"),
                _fmt(sub_level_delta_30_90, " dB"),
            ),
            metric_row(
                t("results_metric_bass_alignment_auto"),
                t("state_on") if bool(alignment_meta.get("applied", False)) else t("state_off"),
                t("state_on") if bool(alignment_meta.get("applied", False)) else t("state_off"),
            ),
            metric_row(
                t("results_metric_bass_alignment_delay"),
                _fmt(safe_float(alignment_meta.get("delay_ms", data.get("bass_integration_sub_delay_ms", float("nan"))), float("nan")), " ms"),
                _fmt(safe_float(alignment_meta.get("delay_ms", data.get("bass_integration_sub_delay_ms", float("nan"))), float("nan")), " ms"),
            ),
            metric_row(
                t("results_metric_bass_alignment_polarity"),
                t("ir_align_value_inverted")
                if bool(alignment_meta.get("polarity_invert", data.get("bass_integration_sub_polarity_invert", False)))
                else t("ir_align_value_ok"),
                t("ir_align_value_inverted")
                if bool(alignment_meta.get("polarity_invert", data.get("bass_integration_sub_polarity_invert", False)))
                else t("ir_align_value_ok"),
            ),
            metric_row(
                t("results_metric_bass_alignment_gain"),
                _fmt(safe_float(alignment_meta.get("gain_trim_db", data.get("bass_integration_sub_gain_trim_db", float("nan"))), float("nan")), " dB"),
                _fmt(safe_float(alignment_meta.get("gain_trim_db", data.get("bass_integration_sub_gain_trim_db", float("nan"))), float("nan")), " dB"),
            ),
            metric_row(
                t("results_metric_bass_allpass"),
                t("state_on") if allpass_on else t("state_off"),
                t("state_on") if allpass_on else t("state_off"),
            ),
            metric_row(
                t("results_metric_bass_allpass_freq"),
                f"{float(allpass_meta.get('freq_hz', 0.0) or 0.0):.1f} Hz" if allpass_on else "n/a",
                f"{float(allpass_meta.get('freq_hz', 0.0) or 0.0):.1f} Hz" if allpass_on else "n/a",
            ),
            metric_row(
                t("results_metric_bass_allpass_q"),
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


def _load_sibling_symbols() -> None:
    import importlib
    package = __package__
    for module_name in ['overview', 'bass_integration', 'quality', 'plots_export']:
        if module_name == __name__.rsplit('.', 1)[-1]:
            continue
        module = importlib.import_module(f"{package}.{module_name}")
        for symbol in getattr(module, "__all__", ()):
            globals().setdefault(symbol, getattr(module, symbol))


_load_sibling_symbols()
