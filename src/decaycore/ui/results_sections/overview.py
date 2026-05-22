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

def _format_recommended_xo_hz(value: float) -> str:
    hz = float(value)
    if math.isclose(hz, round(hz), abs_tol=1e-6):
        return f"{hz:.0f} Hz"
    return f"{hz:.1f} Hz"

def render_results(
    data,
    f_l,
    m_l,
    p_l,
    f_r,
    m_r,
    p_r,
    l_imp_f,
    r_imp_f,
    l_st_f,
    r_st_f,
    fname,
    zip_buffer,
    *,
    dash_html_l=None,
    dash_html_r=None,
    run_started_at=None,
    perf_stats=None,
    per_fs_stats=None,
    saved_filters_dir=None,
    auto_cache_path=None,
    optuna_storage_path=None,
    sub_imp_f=None,
    sub_meas_f=None,
    sub_st_f=None,
) -> None:
    from nicegui import ui
    from ..ng_run_section import get_progress_element, get_results_container, set_progress_visual_state  # noqa: PLC0415

    if run_started_at is not None:
        try:
            elapsed = max(0.0, float(time.perf_counter() - float(run_started_at)))
            ui_state.update_status(f"{t('stat_plot')} | {elapsed:.1f} s")
        except Exception:
            ui_state.update_status(t("stat_plot"))
    else:
        ui_state.update_status(t("stat_plot"))

    prog = get_progress_element()
    if prog is not None:
        try:
            prog.set_value(0.96)
        except Exception:
            logger.exception("progress bar set 0.96")

    # Clear the per-run plot cache so stale figures don't survive across runs.
    _PLOT_RENDER_CACHE.clear()

    container = get_results_container()
    if container is None:
        logger.warning("render_results: results container not available")
        return
    if bool(getattr(container, "is_deleted", False)):
        logger.warning("render_results: results container has been deleted")
        return

    container.clear()

    with container:
        if l_st_f is None or r_st_f is None:
            ui.label(t("results_error_no_results")).classes("text-red-500 font-bold")
            return

        psl_str = plot_smoothing_label(data.get("plot_smoothing_level", "Psychoacoustic"))

        _update_crossover_recommendation_label(data)
        # Keep the main plots above all collapsible summaries in the results view.
        _render_plots_and_export(
            data=data,
            f_l=f_l,
            m_l=m_l,
            p_l=p_l,
            f_r=f_r,
            m_r=m_r,
            p_r=p_r,
            l_imp_f=l_imp_f,
            r_imp_f=r_imp_f,
            l_st_f=l_st_f,
            r_st_f=r_st_f,
            fname=fname,
            zip_buffer=zip_buffer,
            dash_html_l=dash_html_l,
            dash_html_r=dash_html_r,
            saved_filters_dir=saved_filters_dir,
            auto_cache_path=auto_cache_path,
            optuna_storage_path=optuna_storage_path,
            sub_imp_f=sub_imp_f,
            sub_meas_f=sub_meas_f,
            sub_st_f=sub_st_f,
        )
        _render_run_overview(data=data, l_st_f=l_st_f, r_st_f=r_st_f)
        _render_auto_diagnostics(data=data)
        _append_auto_polish_to_status_log(data=data)
        _render_bass_integration(data=data)
        _render_ir_alignment(l_st_f=l_st_f)
        _render_dsp_quality(data=data, l_st_f=l_st_f, r_st_f=r_st_f, psl_str=psl_str)
        _render_lr_difference(l_st_f=l_st_f, r_st_f=r_st_f)

    done_msg = t("done_msg")
    done_status = t("stat_done")
    if saved_filters_dir:
        try:
            done_status = done_status.format(path=str(saved_filters_dir))
        except Exception:
            done_status = f"{done_status} {saved_filters_dir}"

    if run_started_at is not None:
        try:
            total_s = max(0.0, float(time.perf_counter() - float(run_started_at)))
            ui_state.update_status_notices(summary_text=done_msg, info_text="")
            ui_state.update_status(f"{done_status} | {total_s:.1f} s")
        except Exception:
            ui_state.update_status_notices(summary_text=done_msg, info_text="")
            ui_state.update_status(done_status)
    else:
        ui_state.update_status_notices(summary_text=done_msg, info_text="")
        ui_state.update_status(done_status)

    if prog is not None:
        try:
            prog.set_value(1.0)
        except Exception:
            logger.exception("progress bar set 1.0")
    set_progress_visual_state(completed=True)

    if l_st_f is not None and r_st_f is not None:
        try:
            l_ai = plots.calc_ai_summary_from_stats(l_st_f)
            r_ai = plots.calc_ai_summary_from_stats(r_st_f)
            l_score = l_ai.get("score")
            r_score = r_ai.get("score")
            l_match = l_ai.get("match")
            r_match = r_ai.get("match")
            l_conf = float(l_st_f.get("avg_confidence") or 0.0)
            r_conf = float(r_st_f.get("avg_confidence") or 0.0)
            ui_state.set_last_run_info(
                {
                    "score": (float(l_score) + float(r_score)) / 2.0
                    if (l_score is not None and r_score is not None)
                    else None,
                    "match": (float(l_match) + float(r_match)) / 2.0
                    if (l_match is not None and r_match is not None)
                    else None,
                    "conf": (l_conf + r_conf) / 2.0,
                }
            )
        except Exception:
            logger.debug("set_last_run_info failed", exc_info=True)

def _esc(v: Any) -> str:
    return html.escape(str(v) if v is not None else "-")

def _metric_table_html(rows: list[dict]) -> str:
    """Build an HTML table from metric_row() / fmt_* dicts."""
    shared: list[tuple[str, str]] = []
    stereo: list[tuple[str, str, str]] = []

    for row in rows:
        left = dict(row.get("left", {}) or {})
        right = dict(row.get("right", {}) or {})
        label = str(row.get("label", "") or "")
        if str(left.get("compare", "")) == str(right.get("compare", "")):
            shared.append((label, left.get("render", "-")))
        else:
            stereo.append((label, left.get("render", "-"), right.get("render", "-")))

    parts = []
    if shared:
        parts.append(
            "<table style='width:100%;border-collapse:collapse;font-size:12px;'>"
            "<thead><tr>"
            f"<th style='text-align:left;padding:4px 8px;background:rgba(255,255,255,0.06);'>{_esc(t('results_table_metric'))}</th>"
            f"<th style='text-align:left;padding:4px 8px;background:rgba(255,255,255,0.06);'>{_esc(t('results_table_value'))}</th>"
            "</tr></thead><tbody>"
            + "".join(
                f"<tr><td style='padding:3px 8px;border-top:1px solid rgba(255,255,255,0.06);'>{_esc(lbl)}</td>"
                f"<td style='padding:3px 8px;border-top:1px solid rgba(255,255,255,0.06);'>{val}</td></tr>"
                for lbl, val in shared
            )
            + "</tbody></table>"
        )
    if stereo:
        parts.append(
            "<table style='width:100%;border-collapse:collapse;font-size:12px;margin-top:8px;'>"
            "<thead><tr>"
            f"<th style='text-align:left;padding:4px 8px;background:rgba(255,255,255,0.06);'>{_esc(t('results_table_metric'))}</th>"
            f"<th style='text-align:left;padding:4px 8px;background:rgba(255,255,255,0.06);'>{_esc(t('results_table_left_short'))}</th>"
            f"<th style='text-align:left;padding:4px 8px;background:rgba(255,255,255,0.06);'>{_esc(t('results_table_right_short'))}</th>"
            "</tr></thead><tbody>"
            + "".join(
                f"<tr><td style='padding:3px 8px;border-top:1px solid rgba(255,255,255,0.06);'>{_esc(lbl)}</td>"
                f"<td style='padding:3px 8px;border-top:1px solid rgba(255,255,255,0.06);'>{lv}</td>"
                f"<td style='padding:3px 8px;border-top:1px solid rgba(255,255,255,0.06);'>{rv}</td></tr>"
                for lbl, lv, rv in stereo
            )
            + "</tbody></table>"
        )
    return "\n".join(parts)

def _section(title: str, rows: list[dict], summary_lines: list[str] | None = None) -> None:
    """Render a collapsible metric section."""
    from nicegui import ui  # noqa: PLC0415

    with ui.expansion(title).classes("w-full"):
        for line in list(summary_lines or []):
            if line:
                ui.markdown(str(line))
        table_html = _metric_table_html(rows)
        if table_html:
            ui.html(table_html)

def _render_run_overview(*, data: dict, l_st_f: dict, r_st_f: dict) -> None:
    l_ai = plots.calc_ai_summary_from_stats(l_st_f)
    r_ai = plots.calc_ai_summary_from_stats(r_st_f)
    l_score = float(l_ai.get("score") or 0.0)
    r_score = float(r_ai.get("score") or 0.0)
    avg_pred = (l_score + r_score) / 2.0
    l_match = l_ai.get("match")
    r_match = r_ai.get("match")
    avg_match = 0.0 if (l_match is None or r_match is None) else (float(l_match) + float(r_match)) / 2.0

    stereo_link_enabled = bool(data.get("stereo_link", False))
    l_window_used = l_st_f.get("stereo_link_window_used", l_st_f.get("smart_scan_range", [0, 0]))
    r_window_used = r_st_f.get("stereo_link_window_used", r_st_f.get("smart_scan_range", [0, 0]))

    acoustic_summary = []
    if l_ai.get("score") is not None and r_ai.get("score") is not None:
        acoustic_summary.append(t("results_summary_avg_acoustic_score").format(score=avg_pred))
    if l_match is not None and r_match is not None:
        acoustic_summary.append(t("results_summary_avg_target_match").format(match=avg_match))

    l_boost_pre, _, l_net_boost = boost_diag(l_st_f)
    r_boost_pre, _, r_net_boost = boost_diag(r_st_f)

    _section(
        t("results_section_acoustic_summary"),
        [
            metric_row(
                t("results_metric_stereo_leveling"),
                stereo_link_mode_label(l_st_f, stereo_link_enabled=stereo_link_enabled),
                stereo_link_mode_label(r_st_f, stereo_link_enabled=stereo_link_enabled),
            ),
            metric_row(
                t("results_metric_target_level"),
                f"{l_st_f.get('eff_target_db', 0):.1f} dB",
                f"{r_st_f.get('eff_target_db', 0):.1f} dB",
            ),
            metric_row(
                t("results_metric_window_used_for_leveling"),
                fmt_freq_window(l_window_used),
                fmt_freq_window(r_window_used),
            ),
            metric_row(
                t("results_metric_shared_leveling_window"),
                shared_window_label(l_st_f),
                shared_window_label(r_st_f),
            ),
            metric_row(
                t("results_metric_leveling_tilt"),
                fmt_tilt(l_st_f),
                fmt_tilt(r_st_f),
                left_compare=safe_float(l_st_f.get("tilt_slope_db_per_oct", float("nan")), float("nan")),
                right_compare=safe_float(r_st_f.get("tilt_slope_db_per_oct", float("nan")), float("nan")),
            ),
            metric_row(
                t("results_metric_offset_to_meas"),
                f"{l_st_f.get('offset_db', 0):.1f} dB",
                f"{r_st_f.get('offset_db', 0):.1f} dB",
            ),
            metric_row(
                t("results_metric_stereo_anchor"),
                anchor_label(l_st_f, stereo_link_enabled=stereo_link_enabled),
                anchor_label(r_st_f, stereo_link_enabled=stereo_link_enabled),
            ),
            {"label": t("results_metric_target_match"), "left": fmt_ai_match(l_ai), "right": fmt_ai_match(r_ai)},
            metric_row(
                t("results_metric_acoustic_confidence"),
                f"{l_st_f.get('avg_confidence', 0):.1f}%",
                f"{r_st_f.get('avg_confidence', 0):.1f}%",
            ),
            {"label": t("results_metric_acoustic_score"), "left": fmt_ai_score(l_ai), "right": fmt_ai_score(r_ai)},
            metric_row(
                t("results_metric_estimated_rt60"),
                f"{l_st_f.get('rt60_val', 0):.2f} s",
                f"{r_st_f.get('rt60_val', 0):.2f} s",
            ),
            metric_row(
                t("results_metric_schroeder_hz"),
                f"{l_st_f.get('schroeder_hz_estimate', 0):.0f} Hz" if l_st_f.get('schroeder_hz_estimate') else "n/a",
                f"{r_st_f.get('schroeder_hz_estimate', 0):.0f} Hz" if r_st_f.get('schroeder_hz_estimate') else "n/a",
            ),
        ],
        summary_lines=acoustic_summary,
    )

    tdc_text = (
        t("results_value_tdc_on").format(
            strength=float(data.get("tdc_strength", 0)),
            max_reduction=float(data.get("tdc_max_reduction_db", 0)),
        )
        if bool(data.get("enable_tdc", False))
        else t("results_value_off")
    )
    _section(
        t("results_section_gain_headroom"),
        [
            metric_row(t("results_metric_tdc"), tdc_text, tdc_text),
            metric_row(
                t("results_metric_auto_gain_margin"),
                f"{float(l_st_f.get('gain_margin_db', 0) or 0):.2f} dB",
                f"{float(r_st_f.get('gain_margin_db', 0) or 0):.2f} dB",
            ),
            metric_row(
                t("results_metric_applied_auto_gain"),
                f"{float(l_st_f.get('auto_global_gain_db', 0) or 0):.2f} dB",
                f"{float(r_st_f.get('auto_global_gain_db', 0) or 0):.2f} dB",
            ),
            metric_row(
                t("results_metric_net_boost_pre_to_net"),
                f"{l_boost_pre:.2f} dB -> {l_net_boost:.2f} dB",
                f"{r_boost_pre:.2f} dB -> {r_net_boost:.2f} dB",
            ),
            metric_row(
                t("results_metric_final_max_post_gain"),
                f"{float(l_st_f.get('final_max_db', 0) or 0):.2f} dB",
                f"{float(r_st_f.get('final_max_db', 0) or 0):.2f} dB",
            ),
        ],
    )

def _build_auto_polish_lines(auto_meta: dict) -> list[str]:
    lines: list[str] = []

    def _display_rank(polish: dict, raw_key: str, official_key: str) -> float:
        official = safe_float(polish.get(official_key, float("nan")), float("nan"))
        if math.isfinite(official):
            return float(official)
        raw = safe_float(polish.get(raw_key, float("nan")), float("nan"))
        return float(calibrated_auto_quality(raw))

    def _rank_delta_display(polish: dict) -> str:
        before = _display_rank(polish, "rank_before", "rank_before_official")
        after = _display_rank(polish, "rank_after", "rank_after_official")
        if math.isfinite(before) and math.isfinite(after):
            return f", rank {before:.3f} → {after:.3f}"
        return ""

    pl = dict(auto_meta.get("phase_limit_winner_polish", {}) or {})
    if bool(pl.get("applicable", False)):
        start = safe_float(pl.get("start_phase_limit_hz", float("nan")), float("nan"))
        final = safe_float(pl.get("final_phase_limit_hz", float("nan")), float("nan"))
        rd = _rank_delta_display(pl)
        if bool(pl.get("applied", False)):
            lines.append(f"✓ Phase-limit: applied ({start:.1f} → {final:.1f} Hz{rd})")
        else:
            lines.append(f"– Phase-limit: tested, no change (kept {final:.1f} Hz)")

    mc = dict(auto_meta.get("mag_c_min_winner_polish", {}) or {})
    if bool(mc.get("applicable", False)):
        start = safe_float(mc.get("start_mag_c_min_hz", float("nan")), float("nan"))
        final = safe_float(mc.get("final_mag_c_min_hz", float("nan")), float("nan"))
        rd = _rank_delta_display(mc)
        if bool(mc.get("applied", False)):
            lines.append(f"✓ Mag-c-min: applied ({start:.1f} → {final:.1f} Hz{rd})")
        else:
            lines.append(f"– Mag-c-min: tested, no change (kept {final:.1f} Hz)")

    lbc = dict(auto_meta.get("low_bass_cut_winner_polish", {}) or {})
    if bool(lbc.get("applicable", False)):
        start = safe_float(lbc.get("start_low_bass_cut_hz", float("nan")), float("nan"))
        final = safe_float(lbc.get("final_low_bass_cut_hz", float("nan")), float("nan"))
        rd = _rank_delta_display(lbc)
        if bool(lbc.get("applied", False)):
            lines.append(f"✓ Low-bass-cut: applied ({start:.1f} → {final:.1f} Hz{rd})")
        else:
            lines.append(f"– Low-bass-cut: tested, no change (kept {final:.1f} Hz)")

    hpf = dict(auto_meta.get("hpf_winner_polish", {}) or {})
    if bool(hpf.get("applicable", False)):
        def _hpf_label(enabled: bool, freq: float, slope: int) -> str:
            return f"HPF off" if not enabled else f"HPF {freq:.1f} Hz/{slope} dB/oct"
        s_en = bool(hpf.get("start_enabled", False))
        s_fr = safe_float(hpf.get("start_freq_hz", float("nan")), float("nan"))
        s_sl = int(round(safe_float(hpf.get("start_slope_db_oct", 0.0), 0.0)))
        f_en = bool(hpf.get("final_enabled", False))
        f_fr = safe_float(hpf.get("final_freq_hz", float("nan")), float("nan"))
        f_sl = int(round(safe_float(hpf.get("final_slope_db_oct", 0.0), 0.0)))
        rd = _rank_delta_display(hpf)
        if bool(hpf.get("applied", False)):
            lines.append(f"✓ HPF: applied ({_hpf_label(s_en, s_fr, s_sl)} → {_hpf_label(f_en, f_fr, f_sl)}{rd})")
        else:
            lines.append(f"– HPF: tested, no change (kept {_hpf_label(f_en, f_fr, f_sl)})")

    eps = dict(auto_meta.get("excess_phase_strength_winner_polish", {}) or {})
    if bool(eps.get("applicable", False)):
        start = safe_float(eps.get("start_value", float("nan")), float("nan"))
        final = safe_float(eps.get("final_value", float("nan")), float("nan"))
        rd = _rank_delta_display(eps)
        if bool(eps.get("applied", False)):
            lines.append(f"✓ Excess-phase-strength: applied ({start:.4f} → {final:.4f}{rd})")
        else:
            lines.append(f"– Excess-phase-strength: tested, no change (kept {final:.4f})")

    rp = dict(auto_meta.get("residual_peak_winner_polish", {}) or {})
    if bool(rp.get("applicable", False)):
        pb = safe_float(rp.get("worst_peak_before_db", float("nan")), float("nan"))
        pa = safe_float(rp.get("worst_peak_after_db", float("nan")), float("nan"))
        ph = safe_float(rp.get("worst_peak_freq_hz", float("nan")), float("nan"))
        peak_pos = f" @ {ph:.1f} Hz" if math.isfinite(ph) else ""
        rd = _rank_delta_display(rp)
        if bool(rp.get("applied", False)):
            lines.append(f"✓ Residual-peak: applied ({pb:.2f} → {pa:.2f} dB{peak_pos}{rd})")
        elif bool(rp.get("enabled", False)):
            lines.append(f"– Residual-peak: tested, no change (peak {pa:.2f} dB{peak_pos})")

    sp = dict(auto_meta.get("stereo_policy_refine", {}) or {})
    if bool(sp.get("applicable", False)):
        sp_state = str(sp.get("state", "") or "")
        if sp_state == "applied":
            lines.append("✓ Stereo-policy: applied")
        elif sp_state:
            lines.append(f"– Stereo-policy: tested, no change ({sp_state})")

    return lines

def _append_auto_polish_to_status_log(*, data: dict) -> None:
    from .. import ui_state  # noqa: PLC0415

    try:
        mode_u = str(data.get("mode", "BASIC") or "BASIC").strip().upper()
    except Exception:
        return
    auto_enabled = bool(mode_u == "AUTO" or data.get("camillafir_automatic_mode", False))
    auto_meta = data.get("_auto_mode_meta", None)
    if not (auto_enabled and isinstance(auto_meta, dict)):
        return

    lines = _build_auto_polish_lines(auto_meta)
    if not lines:
        return

    ui_state.append_auto_status_detail_raw(t("results_auto_diag_polish_header") + ":")
    for line in lines:
        ui_state.append_auto_status_detail_raw("  " + line)

def _build_p6_validation_block(best_metrics: dict) -> str | None:
    severity = str(best_metrics.get("final_ir_validation_severity", "") or "").strip().lower()
    if not severity:
        return None

    pre = safe_float(best_metrics.get("final_ir_validation_pre_energy_ratio_db", float("nan")), float("nan"))
    gd = safe_float(best_metrics.get("final_ir_validation_gd_peak_ms", float("nan")), float("nan"))
    voice = safe_float(best_metrics.get("final_ir_validation_voice_band_peak_excess_db", float("nan")), float("nan"))
    stereo = safe_float(best_metrics.get("final_ir_validation_stereo_delta_peak_db", float("nan")), float("nan"))
    bass = safe_float(best_metrics.get("final_ir_validation_bass_residual_peak_db", float("nan")), float("nan"))
    penalty = safe_float(best_metrics.get("final_ir_validation_score_penalty", float("nan")), float("nan"))
    reasons = list(best_metrics.get("final_ir_validation_reasons", []) or [])

    p6_mode = str(best_metrics.get("final_ir_validation_mode", "warn") or "warn").strip().lower()
    icon = "✓" if severity == "ok" else ("✗" if severity == "reject" else "⚠")
    severity_label = severity
    if severity == "reject" and p6_mode != "reject":
        severity_label = f"{severity} (warn mode — result used)"
    lines = [f"{icon} P6: {severity_label}"]
    if math.isfinite(pre):
        lines.append(f"- Pre-ringing: {pre:.1f} dB")
    if math.isfinite(gd):
        lines.append(f"- Group delay p95: {gd:.0f} ms")
    if math.isfinite(voice):
        lines.append(f"- Voice band excess: {voice:.1f} dB")
    if math.isfinite(stereo):
        lines.append(f"- Stereo delta: {stereo:.1f} dB")
    if math.isfinite(bass):
        lines.append(f"- Bass residual peak: {bass:.1f} dB")
    if math.isfinite(penalty) and penalty > 0.0:
        lines.append(f"- Penalty: {penalty:.2f}")
    if reasons:
        lines.append(f"- Reasons: {', '.join(reasons)}")

    return "\n\n".join(lines)

def _render_auto_diagnostics(*, data: dict) -> None:
    try:
        mode_u = str(data.get("mode", "BASIC") or "BASIC").strip().upper()
    except Exception:
        mode_u = "BASIC"
    auto_enabled = bool(mode_u == "AUTO" or data.get("camillafir_automatic_mode", False))
    auto_meta = data.get("_auto_mode_meta", None)
    if not (auto_enabled and isinstance(auto_meta, dict)):
        return

    from nicegui import ui  # noqa: PLC0415
    from ...auto_mode.rank_score import attach_official_rank_score, official_rank_score  # noqa: PLC0415

    bm = attach_official_rank_score(auto_meta.get("best_metrics", {}))
    rank_sc = safe_float(official_rank_score(bm), 0.0)
    avg_sc = safe_float(bm.get("avg_score", 0.0), 0.0)
    boost_db = safe_float(bm.get("max_net_boost_db", 0.0), 0.0)
    tc_meta = dict(data.get("_auto_target_curve_meta", {}) or {})
    tc_selected = str(tc_meta.get("selected_hc_mode", data.get("hc_mode", "n/a")) or "n/a")
    trials_ok = int(auto_meta.get("trials_ok", 0) or 0)
    trials_total = int(auto_meta.get("trials_total", 0) or 0)

    with ui.expansion(t("results_auto_diag_title")).classes("w-full"):
        ui.markdown(
            t("results_auto_diag_summary").format(
                target=tc_selected,
                rank=rank_sc,
                avg=avg_sc,
                boost=boost_db,
                ok=trials_ok,
                total=trials_total,
            )
        )
        winner_expl = dict(auto_meta.get("winner_explanation", {}) or {})
        expl_summary = str(winner_expl.get("summary", "") or "").strip()
        expl_reasons = list(winner_expl.get("reasons", []) or [])
        if expl_summary or expl_reasons:
            reasons_md = "\n\n".join(expl_reasons) if expl_reasons else ""
            body = "\n\n".join(filter(None, [expl_summary, reasons_md]))
            ui.markdown(f"**{t('results_auto_diag_rationale_header')}**\n\n{body}")

        polish_lines = _build_auto_polish_lines(auto_meta)
        if polish_lines:
            ui.markdown(f"**{t('results_auto_diag_polish_header')}**\n\n" + "\n\n".join(polish_lines))

        p6_block = _build_p6_validation_block(bm)
        if p6_block:
            ui.markdown(f"**{t('results_auto_diag_p6_header')}**\n\n{p6_block}")

def _update_crossover_recommendation_label(data: dict) -> None:
    """Update the crossover recommendation label in the input tab after a run."""
    try:
        from .. import ng_controls as ctrl  # noqa: PLC0415
        label_el = ctrl.get("avr_crossover_hz_recommendation")
        if label_el is None:
            return
        bi_meta = dict((data or {}).get("_bass_integration_meta", {}) or {})
        rec = bi_meta.get("recommended_crossover_hz", None)
        if rec is not None:
            rec_label = t("bass_integration_main_hpf_recommended")
            label_el.set_text(f"{rec_label}: {_format_recommended_xo_hz(float(rec))}")
        else:
            label_el.set_text("")
    except Exception:
        logger.exception("bass integration recommended XO label")


__all__ = ['_format_recommended_xo_hz', 'render_results', '_esc', '_metric_table_html', '_section', '_render_run_overview', '_build_auto_polish_lines', '_append_auto_polish_to_status_log', '_build_p6_validation_block', '_render_auto_diagnostics', '_update_crossover_recommendation_label']


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
