# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

import numpy as np
import plotly.graph_objects as go
import scipy.fft
import scipy.ndimage
from plotly.subplots import make_subplots

from ...resources.i8n.decaycore_i18n import t
from ..plot_common import (
    GD_SMOOTH_OCT,
    PHASE_SMOOTH_OCT,
    _maybe_shift_to_abs,
    _plotly_js_path,
    _prepare_curve_for_target_plot,
    _view_mags_for_plot,
    calculate_clean_gd,
    logger,
    remove_ir_peak_delay,
    smooth_complex,
)

_PLOT_BG           = "#0e1219"
_PLOT_PAPER_BG     = "#0e1219"
_PLOT_FONT         = "#c8cdd5"
_PLOT_GRID         = "rgba(255,255,255,0.07)"
_PLOT_AXIS         = "rgba(255,255,255,0.16)"
_PLOT_PANEL_BG     = "rgba(14,18,25,0.94)"
_PLOT_PANEL_BORDER = "rgba(255,255,255,0.14)"
_PLOT_MUTED_LINE   = "rgba(255,255,255,0.30)"
_PLOT_ACTIVE_BTN_FILL   = "rgba(107,168,240,0.22)"
_PLOT_ACTIVE_BTN_STROKE = "#6ba8f0"

def _prediction_plot_fft_context(*, filt_ir, fs, target_stats) -> dict:
    min_fft_size = 131072
    fft_mul = 4
    max_fft_size = None
    show_afdw = bool(target_stats.get("afdw_active", True)) if target_stats else True

    n_fft = max(len(filt_ir) * fft_mul, min_fft_size)
    if max_fft_size is not None:
        n_fft = min(n_fft, int(max_fft_size))
    f_lin = scipy.fft.rfftfreq(n_fft, d=1 / fs)
    h_filt = scipy.fft.rfft(filt_ir, n=n_fft)
    h_filt_display, filt_delay_ms = remove_ir_peak_delay(f_lin, h_filt, filt_ir, fs)
    return {
        "vis_points": 4000,
        "show_afdw": bool(show_afdw),
        "fig_height": 1520 if show_afdw else 1220,
        "fig_width": 1750,
        "n_fft": int(n_fft),
        "f_lin": f_lin,
        "h_filt": h_filt,
        "h_filt_display": h_filt_display,
        "filt_delay_ms": float(filt_delay_ms),
    }

def _resolve_magnitude_display_offset_db(
    *,
    target_stats,
    target_curve,
    avg_t,
) -> float:
    """Keep low-reference native measurement plots readable around 0 dB."""
    if not isinstance(target_stats, dict):
        return 0.0

    refs: list[float] = []
    for key in ("target_level_db_window", "eff_target_db"):
        try:
            value = float(target_stats.get(key, np.nan))
        except Exception:
            continue
        if np.isfinite(value):
            refs.append(float(value))

    try:
        target_abs = np.asarray(_maybe_shift_to_abs(target_curve, avg_t), dtype=float).reshape(-1)
        target_abs = target_abs[np.isfinite(target_abs)]
        if target_abs.size >= 4:
            refs.append(float(np.nanmedian(target_abs)))
    except Exception:
        pass

    for ref_db in refs:
        if np.isfinite(ref_db) and ref_db < 40.0:
            return float(ref_db)
    return 0.0

def generate_prediction_plot(
    orig_freqs,
    orig_mags,
    orig_phases,
    filt_ir,
    fs,
    title,
    save_filename=None,
    target_stats=None,
    mixed_split=None,
    zoom_hint="",
    create_full_html=True,
    return_fig: bool = False,
    plot_smoothing_level="Psychoacoustic",
    x_max_hz: float = 20000.0,
    y_mag_range_db: float | None = None,
):
    try:
        fft_ctx = _prediction_plot_fft_context(filt_ir=filt_ir, fs=fs, target_stats=target_stats)
        VIS_POINTS = int(fft_ctx["vis_points"])
        show_afdw = bool(fft_ctx["show_afdw"])
        fig_height, fig_width = int(fft_ctx["fig_height"]), int(fft_ctx["fig_width"])
        n_fft = int(fft_ctx["n_fft"])
        f_lin = fft_ctx["f_lin"]
        h_filt = fft_ctx["h_filt"]
        h_filt_display = fft_ctx["h_filt_display"]
        filt_delay_ms = float(fft_ctx["filt_delay_ms"])

        avg_t = target_stats.get("eff_target_db", 75) if target_stats else 75
        if target_stats and "smart_scan_range" in target_stats:
            match_range = target_stats.get("smart_scan_range", [500, 2000])
        else:
            match_range = target_stats.get("match_range", [500, 2000]) if target_stats else [500, 2000]
        try:
            f_win_min = float(match_range[0])
            f_win_max = float(match_range[1])
        except Exception:
            f_win_min, f_win_max = 500.0, 2000.0
        f_target = np.asarray(target_stats.get("freq_axis", []), dtype=float) if target_stats else np.asarray([], dtype=float)
        target_curve = target_stats.get("target_mags", None) if target_stats else None

        direct_pred_export = None
        direct_pred_comp = None

        if target_stats and "measured_mags" in target_stats:
            f_stats = f_target
            m_stats_src = target_stats.get("measured_mags", [])
            direct_m_stats = np.asarray(
                target_stats.get("direct_dac_sum_measured_mags", []),
                dtype=float,
            ).reshape(-1)
            if f_stats.size > 1 and direct_m_stats.size == f_stats.size:
                m_stats_src = direct_m_stats
            m_stats = _prepare_curve_for_target_plot(
                f_stats,
                m_stats_src,
                avg_t_db=avg_t,
                target_freqs_hz=f_target,
                target_mags_db=target_curve,
                f_min_hz=f_win_min,
                f_max_hz=f_win_max,
            )
            t_stats = _maybe_shift_to_abs(target_curve, avg_t) if target_curve is not None else None

            m_interp = np.interp(f_lin, f_stats, m_stats)

            m_lin_clean = _view_mags_for_plot(
                f_lin,
                m_interp,
                plot_smoothing_level=plot_smoothing_level,
            )

            pred_stats = _maybe_shift_to_abs(
                target_stats.get("direct_dac_sum_predicted_mags", []),
                avg_t,
            )
            pred_stats = np.asarray(pred_stats, dtype=float).reshape(-1)
            if f_stats.size > 1 and pred_stats.size == f_stats.size:
                pred_aligned = _prepare_curve_for_target_plot(
                    f_stats,
                    pred_stats,
                    avg_t_db=avg_t,
                    target_freqs_hz=f_target,
                    target_mags_db=target_curve,
                    f_min_hz=f_win_min,
                    f_max_hz=f_win_max,
                )
                pred_interp = np.interp(f_lin, f_stats, pred_aligned)
                direct_pred_export = _view_mags_for_plot(
                    f_lin,
                    pred_interp,
                    plot_smoothing_level=plot_smoothing_level,
                )

                pred_comp_stats = _maybe_shift_to_abs(
                    target_stats.get("direct_dac_sum_predicted_mags_comp", []),
                    avg_t,
                )
                pred_comp_stats = np.asarray(pred_comp_stats, dtype=float).reshape(-1)
                if pred_comp_stats.size == f_stats.size:
                    pred_comp_aligned = _prepare_curve_for_target_plot(
                        f_stats,
                        pred_comp_stats,
                        avg_t_db=avg_t,
                        target_freqs_hz=f_target,
                        target_mags_db=target_curve,
                        f_min_hz=f_win_min,
                        f_max_hz=f_win_max,
                    )
                    pred_comp_interp = np.interp(f_lin, f_stats, pred_comp_aligned)
                    direct_pred_comp = _view_mags_for_plot(
                        f_lin,
                        pred_comp_interp,
                        plot_smoothing_level=plot_smoothing_level,
                    )
        else:
            m_plot_src = _prepare_curve_for_target_plot(
                orig_freqs,
                orig_mags,
                avg_t_db=avg_t,
                target_freqs_hz=f_target,
                target_mags_db=target_curve,
                f_min_hz=f_win_min,
                f_max_hz=f_win_max,
            )
            m_raw = np.interp(f_lin, orig_freqs, m_plot_src)
            m_lin_clean = _view_mags_for_plot(
                f_lin,
                m_raw,
                plot_smoothing_level=plot_smoothing_level,
            )

        p_lin = np.interp(f_lin, orig_freqs, orig_phases)
        total_spec = 10 ** (np.asarray(m_lin_clean, dtype=float) / 20.0) * np.exp(1j * np.deg2rad(p_lin)) * h_filt

        plot_level_comp_db = 0.0
        ag_db = 0.0
        ah_db = 0.0
        try:
            if target_stats is not None:
                ag_db = float(target_stats.get("auto_global_gain_db", 0.0) or 0.0)
                ah_db = float(target_stats.get("auto_headroom_db", 0.0) or 0.0)
                if np.isfinite(ag_db) and np.isfinite(ah_db):
                    plot_level_comp_db = -(ag_db + ah_db)
                elif np.isfinite(ag_db):
                    plot_level_comp_db = -ag_db
        except Exception:
            plot_level_comp_db = 0.0
            ag_db = 0.0
            ah_db = 0.0

        if direct_pred_export is not None:
            p_sm_export = np.asarray(direct_pred_export, dtype=float)
            if direct_pred_comp is not None:
                p_sm_comp = np.asarray(direct_pred_comp, dtype=float)
            else:
                p_sm_comp = p_sm_export.copy()
                if plot_level_comp_db != 0.0:
                    p_sm_comp = p_sm_comp + float(plot_level_comp_db)
        else:
            p_sm_export = _view_mags_for_plot(
                f_lin,
                20.0 * np.log10(np.abs(total_spec) + 1e-12),
                plot_smoothing_level=plot_smoothing_level,
            )
            p_sm_comp = p_sm_export.copy()
            if plot_level_comp_db != 0.0:
                p_sm_comp = p_sm_comp + float(plot_level_comp_db)
        filt_sm_phase = smooth_complex(f_lin, h_filt_display, PHASE_SMOOTH_OCT)
        ph_sm = (np.rad2deg(np.angle(filt_sm_phase)) + 180) % 360 - 180

        filt_sm_gd = smooth_complex(f_lin, h_filt_display, GD_SMOOTH_OCT)
        gd_sm = calculate_clean_gd(f_lin, filt_sm_gd)

        filt_db_export = 20.0 * np.log10(np.abs(h_filt) + 1e-12)
        filt_db_comp = filt_db_export.copy()
        if plot_level_comp_db != 0.0:
            filt_db_comp = filt_db_comp + float(plot_level_comp_db)

        f_vis = np.geomspace(10, fs / 2, VIS_POINTS)

        m_vis = np.interp(f_vis, f_lin, m_lin_clean)
        p_vis_export = np.interp(f_vis, f_lin, p_sm_export)
        p_vis_comp = np.interp(f_vis, f_lin, p_sm_comp)
        mag_display_offset_db = _resolve_magnitude_display_offset_db(
            target_stats=target_stats,
            target_curve=target_curve,
            avg_t=avg_t,
        )
        if mag_display_offset_db != 0.0:
            m_vis = m_vis - float(mag_display_offset_db)
            p_vis_export = p_vis_export - float(mag_display_offset_db)
            p_vis_comp = p_vis_comp - float(mag_display_offset_db)
        ph_vis = np.interp(f_vis, f_lin, ph_sm)
        gd_vis = np.interp(f_vis, f_lin, gd_sm)
        filt_vis_export = np.interp(f_vis, f_lin, filt_db_export)
        filt_vis_comp = np.interp(f_vis, f_lin, filt_db_comp)

        conf_vis = None
        if target_stats and f_target.size > 1:
            conf_raw = np.asarray(target_stats.get("confidence_mask", []), dtype=float)
            if conf_raw.size == f_target.size and conf_raw.size > 1:
                conf_vis = np.interp(f_vis, f_target, conf_raw)
                conf_vis = np.clip(conf_vis, 0.0, 1.0)
                try:
                    c_min = float(target_stats.get("mag_c_min", 0.0) or 0.0)
                    c_max = float(target_stats.get("mag_c_max", 0.0) or 0.0)
                    if np.isfinite(c_min) and np.isfinite(c_max) and c_min > 0 and c_max > c_min:
                        conf_vis[(f_vis < c_min) | (f_vis > c_max)] = np.nan
                except Exception:
                    pass

        lf_guard_hz = 0.0
        if target_stats:
            lf_guard_hz = float(target_stats.get("lf_guard_hz", 0.0) or 0.0)

        _subplot_titles = (
            "<b>Magnitude & Alignment</b>",
            "<b>Filter Phase (delay compensated)</b>",
            "<b>Filter Group Delay (delay compensated)</b>",
            "<b>Filter (dB)</b>",
        )
        if show_afdw:
            _subplot_titles = _subplot_titles + ("<b>A-FDW Effective BW (oct)</b>",)
        fig = make_subplots(
            rows=5 if show_afdw else 4,
            cols=1,
            vertical_spacing=0.045,
            subplot_titles=_subplot_titles,
        )

        fig.add_trace(
            go.Scatter(
                x=f_vis,
                y=m_vis,
                name="Measured",
                line=dict(color="#60a5fa", width=1.6),
            ),
            row=1,
            col=1,
        )

        if target_stats and "target_mags" in target_stats:
            t_mags = _maybe_shift_to_abs(target_stats.get("target_mags", []), avg_t)
            if mag_display_offset_db != 0.0:
                t_mags = np.asarray(t_mags, dtype=float) - float(mag_display_offset_db)
            fig.add_trace(
                go.Scatter(
                    x=target_stats["freq_axis"],
                    y=t_mags,
                    name="Target",
                    line=dict(color="#34d399", dash="dash", width=2.0),
                ),
                row=1,
                col=1,
            )

        idx_pred_export = len(fig.data)
        fig.add_trace(
            go.Scatter(
                x=f_vis,
                y=p_vis_export,
                name="Predicted (exported)",
                line=dict(color="#fb923c", width=1.6),
            ),
            row=1,
            col=1,
        )

        idx_pred_comp = len(fig.data)
        fig.add_trace(
            go.Scatter(
                x=f_vis,
                y=p_vis_comp,
                name="Predicted (compensated)",
                line=dict(color="#fb923c", width=1.6, dash="dot"),
                visible=False,
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        if conf_vis is not None:
            fig.add_trace(
                go.Scatter(
                    x=f_vis,
                    y=conf_vis,
                    name="Confidence",
                    fill="tozeroy",
                    fillcolor="rgba(148,163,184,0.18)",
                    line=dict(color="rgba(148,163,184,0.45)", width=1.8),
                    yaxis="y6",
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

        err_vis = m_vis - p_vis_export
        fig.add_trace(
            go.Scatter(
                x=f_vis,
                y=err_vis,
                name="Error (M\u2212P)",
                line=dict(color="#c084fc", width=0.9),
                visible="legendonly",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=f_vis,
                y=ph_vis,
                name="Filter Phase",
                line=dict(color="#a78bfa", width=1.2),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=f_vis,
                y=gd_vis,
                name="Filter Group Delay",
                line=dict(color="#a78bfa", width=1.2),
                showlegend=False,
            ),
            row=3,
            col=1,
        )

        idx_filter_export = len(fig.data)
        fig.add_trace(
            go.Scatter(
                x=f_vis,
                y=filt_vis_export,
                name="Filter dB (exported)",
                line=dict(color="#f87171", width=1.2),
                showlegend=False,
                visible=True,
            ),
            row=4,
            col=1,
        )
        idx_filter_comp = len(fig.data)
        fig.add_trace(
            go.Scatter(
                x=f_vis,
                y=filt_vis_comp,
                name="Filter dB (compensated)",
                line=dict(color="#f87171", width=1.2, dash="dot"),
                showlegend=False,
                visible=False,
            ),
            row=4,
            col=1,
        )

        try:
            if target_stats is not None:
                ag_txt = float(target_stats.get("auto_global_gain_db", 0.0) or 0.0)
                ah_txt = float(target_stats.get("auto_headroom_db", 0.0) or 0.0)
                if np.isfinite(ag_txt) or np.isfinite(ah_txt):
                    fig.add_annotation(
                        x=0.99,
                        y=1.0,
                        xref="paper",
                        yref="paper",
                        xanchor="right",
                        yanchor="top",
                        text=f"Auto gain: {ag_txt:+.2f} dB | Headroom: {ah_txt:+.2f} dB | Filter delay removed: {filt_delay_ms:.2f} ms",
                        showarrow=False,
                        align="left",
                        font=dict(size=12, color=_PLOT_FONT),
                        bgcolor=_PLOT_PANEL_BG,
                        bordercolor=_PLOT_PANEL_BORDER,
                        borderwidth=1,
                    )
        except Exception:
            pass

        try:
            n_tr = len(fig.data)
            vis_export = [True] * n_tr
            vis_comp = [True] * n_tr
            vis_both = [True] * n_tr

            vis_export[idx_pred_comp] = False
            vis_export[idx_pred_export] = True
            vis_export[idx_filter_comp] = False
            vis_export[idx_filter_export] = True

            vis_comp[idx_pred_export] = False
            vis_comp[idx_pred_comp] = True
            vis_comp[idx_filter_export] = False
            vis_comp[idx_filter_comp] = True

            vis_both[idx_pred_export] = True
            vis_both[idx_pred_comp] = True
            vis_both[idx_filter_export] = True
            vis_both[idx_filter_comp] = True

            fig.update_layout(
                margin=dict(t=120, b=90),
                updatemenus=[
                    dict(
                        type="buttons",
                        direction="right",
                        x=0.01,
                        y=1.15,
                        xanchor="left",
                        yanchor="top",
                        showactive=True,
                        bgcolor=_PLOT_PANEL_BG,
                        bordercolor=_PLOT_PANEL_BORDER,
                        borderwidth=1,
                        font=dict(size=12, color=_PLOT_FONT),
                        pad=dict(t=4, r=6, b=4, l=6),
                        buttons=[
                            dict(
                                label=t("plot_level_exported"),
                                method="update",
                                args=[{"visible": vis_export}],
                            ),
                            dict(
                                label=t("plot_level_compensated"),
                                method="update",
                                args=[{"visible": vis_comp}],
                            ),
                            dict(
                                label=t("plot_level_both"),
                                method="update",
                                args=[{"visible": vis_both}],
                            ),
                        ],
                    )
                ],
            )
        except Exception:
            pass

        try:
            fig.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=11),
                    bgcolor=_PLOT_PANEL_BG,
                    bordercolor=_PLOT_PANEL_BORDER,
                    borderwidth=1,
                )
            )
        except Exception:
            pass

        if target_stats:
            try:
                cmin = float(target_stats.get("mag_c_min", 0.0) or 0.0)
                cmax = float(target_stats.get("mag_c_max", 0.0) or 0.0)
                if np.isfinite(cmin) and np.isfinite(cmax) and cmin > 0 and cmax > cmin:
                    fig.add_shape(
                        type="rect",
                        xref="x",
                        yref="y",
                        x0=cmin,
                        x1=cmax,
                        y0=-15,
                        y1=10,
                        fillcolor="rgba(96,165,250,0.07)",
                        layer="below",
                        line_width=0,
                        row=4,
                        col=1,
                    )
            except Exception:
                pass

        if target_stats:
            try:
                if lf_guard_hz > 10.0 and np.isfinite(lf_guard_hz):
                    fig.add_shape(
                        type="rect",
                        xref="x",
                        yref="y domain",
                        x0=10.0,
                        x1=lf_guard_hz,
                        y0=0,
                        y1=1,
                        fillcolor="rgba(248,113,113,0.08)",
                        layer="below",
                        line_width=0,
                        row=1,
                        col=1,
                    )
                    fig.add_shape(
                        type="line",
                        xref="x",
                        yref="y domain",
                        x0=lf_guard_hz,
                        x1=lf_guard_hz,
                        y0=0,
                        y1=1,
                        line=dict(color="rgba(248,113,113,0.40)", width=1, dash="dot"),
                        layer="below",
                        row=1,
                        col=1,
                    )
            except Exception:
                pass

        bw_vis = None
        bw_vis_smooth = None

        if show_afdw:
            bw_dbg = ""
            mode = "native"
            if target_stats:
                mode = str(target_stats.get("analysis_mode", "native")).lower()

            try:
                if target_stats:
                    if mode == "comparison":
                        fx_raw = target_stats.get("cmp_freq_axis")
                        bw_raw = target_stats.get("cmp_afdw_bw_plot_oct", target_stats.get("cmp_afdw_bw_oct"))
                    else:
                        fx_raw = target_stats.get("freq_axis")
                        bw_raw = target_stats.get("afdw_bw_plot_oct", target_stats.get("afdw_bw_oct"))

                    if fx_raw is not None and bw_raw is not None:
                        fx = np.asarray(fx_raw, dtype=float)
                        bw = np.asarray(bw_raw, dtype=float)

                        if fx.size == bw.size and fx.size > 16:
                            bw_vis = np.interp(f_vis, fx, bw)
                            bw_vis = np.clip(bw_vis, 1.0 / 96.0, 1.0 / 3.0)
                            bw_vis_smooth = scipy.ndimage.gaussian_filter1d(bw_vis, sigma=5.0)
                            bw_vis_smooth = np.clip(bw_vis_smooth, 1.0 / 96.0, 1.0 / 3.0)
                            fig.add_trace(
                                go.Scatter(
                                    x=f_vis,
                                    y=bw_vis_smooth,
                                    mode="lines",
                                    fill="tozeroy",
                                    fillcolor="rgba(56, 189, 248, 0.22)",
                                    opacity=0.6,
                                    line=dict(color="#38bdf8", width=2.2),
                                    showlegend=False,
                                    name="A-FDW BW",
                                ),
                                row=5,
                                col=1,
                            )
                        else:
                            bw_dbg = f"shape mismatch: fx={fx.size} bw={bw.size}"
                    else:
                        bw_dbg = "missing afdw bw data"
                else:
                    bw_dbg = "target_stats is None"
            except Exception as e:
                bw_dbg = f"{type(e).__name__}: {e}"

            if bw_vis is None:
                fig.add_annotation(
                    text=f"No A-FDW BW data ({bw_dbg})",
                    x=0.5,
                    y=0.5,
                    xref="x5 domain",
                    yref="y5 domain",
                    showarrow=False,
                    font=dict(color=_PLOT_FONT),
                    bgcolor=_PLOT_PANEL_BG,
                    bordercolor=_PLOT_PANEL_BORDER,
                    borderwidth=1,
                )

        _t_vals_full = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        t_vals = [v for v in _t_vals_full if v <= x_max_hz]
        _n_rows = 5 if show_afdw else 4
        for r in range(1, _n_rows + 1):
            fig.update_xaxes(matches="x", row=r, col=1)
            fig.update_xaxes(type="log", range=[np.log10(10), np.log10(x_max_hz)], tickvals=t_vals, row=r, col=1)

        if y_mag_range_db is not None:
            fig.update_yaxes(range=[-y_mag_range_db, y_mag_range_db], row=1, col=1)
        else:
            fig.update_yaxes(autorange=True, row=1, col=1)
        fig.update_yaxes(autorange=True, row=2, col=1)
        fig.update_yaxes(autorange=True, row=3, col=1)
        fig.update_yaxes(range=[-30, 12], row=4, col=1)
        if show_afdw:
            if bw_vis_smooth is not None and len(bw_vis_smooth) > 0:
                bw_data_min = float(np.min(bw_vis_smooth))
                bw_data_max = float(np.max(bw_vis_smooth))
                bw_span = bw_data_max - bw_data_min
                margin = max(bw_span * 0.3, 0.01)
                bw_lo = max(0.0, bw_data_min - margin)
                bw_hi = min(1.0 / 3.0, bw_data_max + margin)
                if bw_hi - bw_lo < 0.02:
                    bw_lo = max(0.0, (bw_data_min + bw_data_max) / 2.0 - 0.01)
                    bw_hi = bw_lo + 0.02
                fig.update_yaxes(range=[bw_lo, bw_hi], row=5, col=1)
            else:
                fig.update_yaxes(range=[0.0, 1.0 / 3.0], row=5, col=1)
            fig.update_yaxes(title_text="oct", row=5, col=1)

        fig.update_layout(
            height=fig_height,
            width=fig_width,
            template="plotly_dark",
            paper_bgcolor=_PLOT_PAPER_BG,
            plot_bgcolor=_PLOT_BG,
            font=dict(color=_PLOT_FONT, family="Inter, system-ui, sans-serif"),
            title_text=f"{title} Analysis",
            uirevision="keep",
            yaxis6=dict(
                overlaying="y",
                range=[0, 1],
                side="right",
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                layer="below traces",
            ),
        )

        fig.update_xaxes(
            gridcolor=_PLOT_GRID,
            linecolor=_PLOT_AXIS,
            zerolinecolor=_PLOT_GRID,
        )
        fig.update_yaxes(
            gridcolor=_PLOT_GRID,
            linecolor=_PLOT_AXIS,
            zerolinecolor=_PLOT_GRID,
        )

        if create_full_html:
            if _plotly_js_path():
                js_mode = "assets/plotly.min.js"
            else:
                js_mode = "cdn"
        else:
            if _plotly_js_path():
                js_mode = "assets/plotly.min.js"
            else:
                js_mode = "cdn"

        config = {
            "responsive": True,
            "scrollZoom": True,
            "displaylogo": False,
            "doubleClick": False,
        }

        html = fig.to_html(
            include_plotlyjs=js_mode,
            full_html=create_full_html,
            config=config,
        )

        _active_btn_js = """
<script>
(function() {
  function _fixActiveBtns(root) {
    var rects = (root || document).querySelectorAll('.updatemenu-item-rect');
    rects.forEach(function(r) {
      var fill = r.getAttribute('fill') || r.style.fill || '';
      if (fill && fill !== 'none' && fill !== 'rgba(0,0,0,0)' &&
          fill !== 'transparent' && fill.toLowerCase() !== '#000' &&
          fill.toLowerCase() !== 'black') {
        var isLight = (
          fill.startsWith('rgb(2') || fill.startsWith('rgb(1') ||
          fill.startsWith('#f') || fill.startsWith('#e') ||
          fill.startsWith('#d') || fill.startsWith('#c') ||
          fill === 'white' || fill === '#fff' || fill === '#ffffff'
        );
        if (isLight) {
          r.setAttribute('fill', '__ACTIVE_FILL__');
          r.style.fill = '__ACTIVE_FILL__';
          r.setAttribute('stroke', '__ACTIVE_STROKE__');
        }
      }
    });
  }
  var _obs = new MutationObserver(function(muts) {
    muts.forEach(function(m) {
      if (m.type === 'attributes' && (m.attributeName === 'fill' || m.attributeName === 'style')) {
        _fixActiveBtns(m.target.closest('.updatemenu') || document);
      } else if (m.type === 'childList') {
        _fixActiveBtns(document);
      }
    });
  });
  function _attach() {
    _fixActiveBtns(document);
    document.querySelectorAll('.updatemenu').forEach(function(el) {
      _obs.observe(el, { subtree: true, attributes: true, attributeFilter: ['fill', 'style'], childList: true });
    });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() { setTimeout(_attach, 300); });
  } else {
    setTimeout(_attach, 300);
  }
})();
</script>""".replace("__ACTIVE_FILL__", _PLOT_ACTIVE_BTN_FILL).replace(
            "__ACTIVE_STROKE__",
            _PLOT_ACTIVE_BTN_STROKE,
        )
        if "</body>" in html:
            html = html.replace("</body>", _active_btn_js + "\n</body>", 1)
        else:
            html = html + _active_btn_js

        if bool(return_fig):
            return html, fig
        return html

    except Exception as e:
        msg = f"Visual Engine Error: {str(e)}"
        if bool(return_fig):
            return msg, None
        return msg


__all__ = ['_prediction_plot_fft_context', '_resolve_magnitude_display_offset_db', 'generate_prediction_plot']


def _load_sibling_symbols() -> None:
    import importlib
    package = __package__
    for module_name in ['plot_prediction_01']:
        if module_name == __name__.rsplit('.', 1)[-1]:
            continue
        module = importlib.import_module(f"{package}.{module_name}")
        for symbol in getattr(module, "__all__", ()):
            globals().setdefault(symbol, getattr(module, symbol))


_load_sibling_symbols()
