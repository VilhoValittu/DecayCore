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
import math

from .section import _section

from ...resources.i8n.decaycore_i18n import t
from ..results_formatters import (
    format_ir_window,
    gd_grad_max_label,
    gd_limiter_label,
    hpf_diff_raw_label,
    hpf_model_label,
    metric_row,
    mixed_blend_label,
    phase_clamp_label,
    safe_float,
    xo_fc_gd_badge,
    xo_fc_gd_label,
)

from ...dsp.lr_difference_metrics import compute_lr_difference_metrics

logger = logging.getLogger("DecayCore")

def _format_ir_alignment_ms(v) -> str:
    try:
        x = float(v)
        if math.isfinite(x):
            return f"{x:+.2f} ms"
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
    ):
        logger.exception("ir align ms format")
    return "-"


def _format_ir_alignment_db(v) -> str:
    try:
        x = float(v)
        if math.isfinite(x):
            return f"{x:+.1f} dB"
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
    ):
        logger.exception("ir align dB format")
    return "-"


def _format_ir_alignment_deg(v) -> str:
    try:
        x = float(v)
        if math.isfinite(x):
            return f"{x:+.1f}°"
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
    ):
        logger.exception("ir align deg format")
    return "-"


def _format_ir_alignment_pct(v) -> str:
    try:
        x = float(v)
        if math.isfinite(x):
            return f"{x * 100.0:.0f}%"
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
    ):
        logger.exception("ir align pct format")
    return "-"


def _format_ir_alignment_dbfs(v) -> str:
    try:
        x = float(v)
        if math.isfinite(x):
            return f"{x:.1f} dBFS"
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
    ):
        logger.exception("ir align dBFS format")
    return "-"


def _ir_alignment_polarity_label(d: dict) -> str:
    inv = bool(d.get("ir_align_polarity_inverted", False)) or bool(d.get("ir_align_xcorr_polarity_flip", False))
    return t("ir_align_value_inverted") if inv else t("ir_align_value_ok")


def _ir_alignment_xo_label(d: dict) -> str:
    xo = safe_float(d.get("ir_align_xo_hz"), None)
    if xo is not None and math.isfinite(xo):
        return f"{xo:.0f}"
    return "80"


def _build_ir_alignment_rows(ir_align: dict, ir_align_sub: dict) -> list[dict]:
    rows: list[dict] = []
    if ir_align:
        rows.append(metric_row(f"── {t('ir_align_group_lr')} ──", "", ""))
        rows.append(metric_row(
            t("ir_align_metric_xcorr_offset"),
            _format_ir_alignment_ms(ir_align.get("ir_align_xcorr_offset_ms")),
            _format_ir_alignment_ms(ir_align.get("ir_align_xcorr_offset_ms")),
        ))
        rows.append(metric_row(
            t("ir_align_metric_xcorr_confidence"),
            _format_ir_alignment_pct(ir_align.get("ir_align_xcorr_confidence")),
            _format_ir_alignment_pct(ir_align.get("ir_align_xcorr_confidence")),
        ))
        rows.append(metric_row(
            t("ir_align_metric_polarity"),
            _ir_alignment_polarity_label(ir_align),
            _ir_alignment_polarity_label(ir_align),
        ))
        rows.append(metric_row(
            t("ir_align_metric_level_peak"),
            _format_ir_alignment_dbfs(ir_align.get("ir_align_level_peak_a_dbfs")),
            _format_ir_alignment_dbfs(ir_align.get("ir_align_level_peak_b_dbfs")),
        ))
        rows.append(metric_row(
            t("ir_align_metric_level_diff"),
            _format_ir_alignment_db(ir_align.get("ir_align_level_rms_diff_db")),
            _format_ir_alignment_db(ir_align.get("ir_align_level_rms_diff_db")),
        ))

    if ir_align_sub:
        xo_lbl = _ir_alignment_xo_label(ir_align_sub)
        rows.append(metric_row(f"── {t('ir_align_group_sub')} ──", "", ""))
        rows.append(metric_row(
            t("ir_align_metric_xcorr_offset"),
            _format_ir_alignment_ms(ir_align_sub.get("ir_align_xcorr_offset_ms")),
            _format_ir_alignment_ms(ir_align_sub.get("ir_align_xcorr_offset_ms")),
        ))
        rows.append(metric_row(
            t("ir_align_metric_xcorr_confidence"),
            _format_ir_alignment_pct(ir_align_sub.get("ir_align_xcorr_confidence")),
            _format_ir_alignment_pct(ir_align_sub.get("ir_align_xcorr_confidence")),
        ))
        rows.append(metric_row(
            t("ir_align_metric_polarity"),
            _ir_alignment_polarity_label(ir_align_sub),
            _ir_alignment_polarity_label(ir_align_sub),
        ))
        rows.append(metric_row(
            t("ir_align_metric_level_peak"),
            _format_ir_alignment_dbfs(ir_align_sub.get("ir_align_level_peak_a_dbfs")),
            _format_ir_alignment_dbfs(ir_align_sub.get("ir_align_level_peak_b_dbfs")),
        ))
        rows.append(metric_row(
            t("ir_align_metric_level_diff"),
            _format_ir_alignment_db(ir_align_sub.get("ir_align_level_rms_diff_db")),
            _format_ir_alignment_db(ir_align_sub.get("ir_align_level_rms_diff_db")),
        ))
        rows.append(metric_row(
            t("ir_align_metric_phase_diff").format(xo=xo_lbl),
            _format_ir_alignment_deg(ir_align_sub.get("ir_align_phase_diff_deg")),
            _format_ir_alignment_deg(ir_align_sub.get("ir_align_phase_diff_deg")),
        ))
        rows.append(metric_row(
            t("ir_align_metric_gd").format(xo=xo_lbl),
            _format_ir_alignment_ms(ir_align_sub.get("ir_align_gd_a_ms")),
            _format_ir_alignment_ms(ir_align_sub.get("ir_align_gd_b_ms")),
        ))
        rows.append(metric_row(
            t("ir_align_metric_gd_diff"),
            _format_ir_alignment_ms(ir_align_sub.get("ir_align_gd_diff_ms")),
            _format_ir_alignment_ms(ir_align_sub.get("ir_align_gd_diff_ms")),
        ))
    return rows


def _build_ir_alignment_summary(ir_align: dict, ir_align_sub: dict) -> str:
    issues: list[str] = []
    for d in (ir_align, ir_align_sub):
        if not d:
            continue
        if bool(d.get("ir_align_polarity_inverted")) or bool(d.get("ir_align_xcorr_polarity_flip")):
            issues.append(t("ir_align_issue_polarity"))
        offset_value = d.get("ir_align_xcorr_offset_ms")
        off = safe_float(offset_value) if offset_value is not None else float("nan")
        if math.isfinite(off) and abs(off) >= 5.0:
            issues.append(t("ir_align_issue_timing"))
        if not bool(d.get("ir_align_phase_in_phase", True)) and d is ir_align_sub:
            issues.append(t("ir_align_issue_phase"))
        rms_value = d.get("ir_align_level_rms_diff_db")
        rms = safe_float(rms_value) if rms_value is not None else float("nan")
        if math.isfinite(rms) and abs(rms) >= 10.0:
            issues.append(t("ir_align_issue_level"))

    seen: set[str] = set()
    unique_issues = [x for x in issues if not (x in seen or seen.add(x))]
    if unique_issues:
        return t("ir_align_summary_issues").format(issues=", ".join(unique_issues))
    return t("ir_align_summary_ok")


def _render_ir_alignment(*, l_st_f: dict) -> None:
    """Renderöi Mittausten IR-infot -osion jos ir_align- tai ir_align_sub-data löytyy."""
    ir_align = dict(l_st_f.get("ir_align") or {})
    ir_align_sub = dict(l_st_f.get("ir_align_sub") or {})
    if not ir_align and not ir_align_sub:
        return

    rows = _build_ir_alignment_rows(ir_align, ir_align_sub)
    summary_line = _build_ir_alignment_summary(ir_align, ir_align_sub)
    _section(t("results_section_ir_alignment"), rows, summary_lines=[summary_line])

def _render_dsp_quality(*, data: dict, l_st_f: dict, r_st_f: dict, psl_str: str) -> None:
    _section(
        t("results_section_filter_ir"),
        [
            metric_row(
                t("results_metric_filter_type"),
                str(data.get("filter_type", "") or ""),
                str(data.get("filter_type", "") or ""),
            ),
            metric_row(t("results_metric_ir_window"), format_ir_window(l_st_f), format_ir_window(r_st_f)),
            metric_row(
                t("results_metric_correction_range"),
                f"{data.get('mag_c_min', 0):.0f}-{data.get('mag_c_max', 0):.0f} Hz",
                f"{data.get('mag_c_min', 0):.0f}-{data.get('mag_c_max', 0):.0f} Hz",
            ),
            metric_row(t("results_metric_plot_smoothing"), psl_str, psl_str),
        ],
    )
    _section(
        t("results_section_phase_gd"),
        [
            metric_row(t("results_metric_xo_phase_model"), xo_fc_gd_label(l_st_f), xo_fc_gd_label(r_st_f)),
            metric_row("XO severity", xo_fc_gd_badge(l_st_f), xo_fc_gd_badge(r_st_f)),
            metric_row(t("results_metric_phase_clamp"), phase_clamp_label(l_st_f), phase_clamp_label(r_st_f)),
            metric_row(t("results_metric_gd_limiter"), gd_limiter_label(l_st_f), gd_limiter_label(r_st_f)),
            metric_row(t("results_metric_gd_gradient_max"), gd_grad_max_label(l_st_f), gd_grad_max_label(r_st_f)),
            metric_row(t("results_metric_hpf_model"), hpf_model_label(l_st_f), hpf_model_label(r_st_f)),
            metric_row(t("results_metric_hpf_diff_raw"), hpf_diff_raw_label(l_st_f), hpf_diff_raw_label(r_st_f)),
            metric_row(
                t("results_metric_mixed_blend_split"),
                mixed_blend_label(l_st_f, "mixed_blend_split_hz"),
                mixed_blend_label(r_st_f, "mixed_blend_split_hz"),
            ),
            metric_row(
                t("results_metric_mixed_blend_transition"),
                mixed_blend_label(l_st_f, "mixed_blend_transition_hz"),
                mixed_blend_label(r_st_f, "mixed_blend_transition_hz"),
            ),
        ],
    )

def _render_lr_difference(*, l_st_f: dict, r_st_f: dict) -> None:
    """Render L/R difference metrics section based on measured response data."""
    try:
        lr = compute_lr_difference_metrics(l_st_f, r_st_f)

        def _fmtdb(v: float) -> str:
            return f"{v:.2f} dB" if math.isfinite(v) else "n/a"

        has_any = any(
            math.isfinite(v)
            for v in (lr.mag_rms_bass_db, lr.mag_rms_mid_db, lr.mag_rms_band_db, lr.mag_maxabs_band_db)
        )
        if not has_any:
            return

        band_lo = lr.band_lo_hz if math.isfinite(lr.band_lo_hz) else 20.0
        band_hi = lr.band_hi_hz if math.isfinite(lr.band_hi_hz) else 2000.0

        rows = [
            metric_row("L/R Mag RMS  20–200 Hz", _fmtdb(lr.mag_rms_bass_db), _fmtdb(lr.mag_rms_bass_db)),
            metric_row("L/R Mag RMS  200–2000 Hz", _fmtdb(lr.mag_rms_mid_db), _fmtdb(lr.mag_rms_mid_db)),
            metric_row(
                f"L/R Mag RMS  {band_lo:.0f}–{band_hi:.0f} Hz",
                _fmtdb(lr.mag_rms_band_db),
                _fmtdb(lr.mag_rms_band_db),
            ),
            metric_row(
                f"L/R Mag MaxAbs  {band_lo:.0f}–{band_hi:.0f} Hz",
                _fmtdb(lr.mag_maxabs_band_db),
                _fmtdb(lr.mag_maxabs_band_db),
            ),
        ]
        if math.isfinite(lr.gd_rms_band_ms):
            gd_txt = f"{lr.gd_rms_band_ms:.2f} ms"
            rows.append(
                metric_row(f"L/R GD RMS  {band_lo:.0f}–{band_hi:.0f} Hz", gd_txt, gd_txt)
            )

        _section(
            t("results_section_lr_difference"),
            rows,
            summary_lines=["Computed from measured left-right response data. Lower = more symmetric."],
        )
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
    ):
        logger.debug("_render_lr_difference failed", exc_info=True)


def _fmt_biquad(b: dict | None) -> str:
    if b is None:
        return "-"
    freq = float(b.get("freq", 0.0) or 0.0)
    q = float(b.get("q", 0.0) or 0.0)
    gain = float(b.get("gain", 0.0) or 0.0)
    conf = float(b.get("confidence", 0.0) or 0.0)
    return f"{freq:.1f} Hz, Q {q:.2f}, {gain:+.2f} dB (conf {conf:.2f})"


def _fmt_external_iir_hpf(st: dict) -> str:
    if not bool(st.get("external_iir_hpf_enabled", False)):
        return "-"
    try:
        freq_hz = float(st.get("external_iir_hpf_freq_hz", 0.0) or 0.0)
        slope = int(st.get("external_iir_hpf_slope_db_oct", 0) or 0)
    except (TypeError, ValueError, OverflowError):
        return "-"
    if not (math.isfinite(freq_hz) and freq_hz > 0.0 and slope > 0):
        return "-"
    return f"{freq_hz:.1f} Hz, {slope:d} dB/oct, CamillaDSP IIR"


def _rejected_reasons(st: dict) -> str:
    rejected = [d for d in (st.get("hybrid_iir_rejected") or []) if isinstance(d, dict)]
    reasons = sorted({str(d.get("reason", "unknown")) for d in rejected[:6]})
    return ", ".join(reasons)


def _render_hybrid_iir_cuts(*, l_st_f: dict, r_st_f: dict) -> None:
    """Render collapsible Hybrid FIR-IIR modal cuts section."""
    l_enabled = bool(l_st_f.get("hybrid_iir_enabled", False))
    r_enabled = bool(r_st_f.get("hybrid_iir_enabled", False))
    l_hpf_enabled = bool(l_st_f.get("external_iir_hpf_enabled", False))
    r_hpf_enabled = bool(r_st_f.get("external_iir_hpf_enabled", False))
    if not l_enabled and not r_enabled and not l_hpf_enabled and not r_hpf_enabled:
        return

    l_biquads = [dict(b) for b in (l_st_f.get("hybrid_iir_biquads") or []) if isinstance(b, dict)]
    r_biquads = [dict(b) for b in (r_st_f.get("hybrid_iir_biquads") or []) if isinstance(b, dict)]
    l_count = int(l_st_f.get("hybrid_iir_filter_count", len(l_biquads)))
    r_count = int(r_st_f.get("hybrid_iir_filter_count", len(r_biquads)))
    l_events = int(l_st_f.get("hybrid_iir_modal_event_count", 0))
    r_events = int(r_st_f.get("hybrid_iir_modal_event_count", 0))
    l_gd_src = str(l_st_f.get("hybrid_iir_gd_source", "stats") or "stats")
    r_gd_src = str(r_st_f.get("hybrid_iir_gd_source", "stats") or "stats")

    rows: list[dict] = [
        metric_row(t("results_metric_external_iir_hpf"), _fmt_external_iir_hpf(l_st_f), _fmt_external_iir_hpf(r_st_f)),
        metric_row(t("results_metric_hybrid_iir_active_cuts"), str(l_count), str(r_count)),
        metric_row(t("results_metric_hybrid_iir_detected_events"), str(l_events), str(r_events)),
    ]
    n_cuts = max(len(l_biquads), len(r_biquads))
    for i in range(n_cuts):
        lb = l_biquads[i] if i < len(l_biquads) else None
        rb = r_biquads[i] if i < len(r_biquads) else None
        rows.append(metric_row(
            f"{t('results_metric_hybrid_iir_cut')} #{i + 1}",
            _fmt_biquad(lb),
            _fmt_biquad(rb),
        ))
    if l_gd_src != "stats" or r_gd_src != "stats":
        rows.append(metric_row(t("results_metric_hybrid_iir_gd_source"), l_gd_src, r_gd_src))

    summary_lines: list[str] = []
    if l_count == 0 and l_enabled:
        reasons = _rejected_reasons(l_st_f)
        if reasons:
            summary_lines.append(f"L — {t('results_hybrid_iir_no_cuts')}: {reasons}")
    if r_count == 0 and r_enabled:
        reasons = _rejected_reasons(r_st_f)
        if reasons:
            summary_lines.append(f"R — {t('results_hybrid_iir_no_cuts')}: {reasons}")

    _section(t("results_section_hybrid_iir_cuts"), rows, summary_lines or None)


__all__ = [
    '_render_ir_alignment',
    '_render_dsp_quality',
    '_render_lr_difference',
    '_render_hybrid_iir_cuts',
    '_fmt_biquad',
    '_fmt_external_iir_hpf',
    '_rejected_reasons',
]
