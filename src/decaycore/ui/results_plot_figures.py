# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Figure-only builders for the NiceGUI Results view."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..resources.i8n.decaycore_i18n import t
from .plot_common import _robust_axis_range
from .plot_prediction_parts import ChannelPlotData, FILTER_IMPULSE_VIEW_RANGE_MS

_LEFT = "#38bdf8"
_RIGHT = "#c084fc"
_SUB = "#f59e0b"
_TARGET = "#34d399"
_FILTER = "#f87171"
_TIMING = "#a78bfa"
_HYBRID_IIR = "#f59e0b"


@dataclass(frozen=True)
class PlotTheme:
    paper: str
    plot: str
    text: str
    grid: str
    axis: str
    panel: str
    border: str
    muted: str


def plot_theme(*, dark: bool) -> PlotTheme:
    if dark:
        return PlotTheme(
            paper="#0e1219",
            plot="#0e1219",
            text="#c8cdd5",
            grid="rgba(255,255,255,0.07)",
            axis="rgba(255,255,255,0.16)",
            panel="rgba(14,18,25,0.94)",
            border="rgba(255,255,255,0.14)",
            muted="rgba(200,205,213,0.42)",
        )
    return PlotTheme(
        paper="#ffffff",
        plot="#ffffff",
        text="#2f2547",
        grid="rgba(47,37,71,0.10)",
        axis="rgba(47,37,71,0.24)",
        panel="rgba(246,243,251,0.96)",
        border="rgba(77,58,120,0.28)",
        muted="rgba(47,37,71,0.42)",
    )


def _channel_color(channel_key: str) -> str:
    return {
        "left": _LEFT,
        "right": _RIGHT,
        "sub": _SUB,
    }.get(str(channel_key).lower(), _LEFT)


def _full_range(data: ChannelPlotData) -> tuple[float, float]:
    return 10.0, max(10.01, float(data.full_max_hz))


def _overview_range(
    left: ChannelPlotData,
    right: ChannelPlotData,
) -> tuple[float, float]:
    valid = [
        data
        for data in (left, right)
        if data.has_valid_correction_range
    ]
    if not valid:
        return 10.0, min(left.full_max_hz, right.full_max_hz)
    return (
        min(data.correction_min_hz for data in valid),
        max(data.correction_max_hz for data in valid),
    )


def _tick_values(max_hz: float) -> list[int]:
    return [
        value
        for value in (10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000)
        if value <= float(max_hz)
    ]


def _base_layout(
    fig: go.Figure,
    *,
    title: str,
    dark: bool,
    height: int,
    show_legend: bool = True,
) -> PlotTheme:
    theme = plot_theme(dark=dark)
    fig.update_layout(
        height=int(height),
        autosize=True,
        template="plotly_dark" if dark else "plotly_white",
        paper_bgcolor=theme.paper,
        plot_bgcolor=theme.plot,
        font=dict(color=theme.text, family="Inter, system-ui, sans-serif"),
        title=dict(text=title, x=0.01, xanchor="left"),
        margin=dict(l=62, r=24, t=92 if show_legend else 68, b=58),
        hovermode="x unified",
        uirevision="keep",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
            bgcolor=theme.panel,
            bordercolor=theme.border,
            borderwidth=1,
        ),
        showlegend=show_legend,
    )
    fig.update_xaxes(
        gridcolor=theme.grid,
        linecolor=theme.axis,
        zerolinecolor=theme.grid,
        title_text=t("results_plot_frequency_axis"),
    )
    fig.update_yaxes(
        gridcolor=theme.grid,
        linecolor=theme.axis,
        zerolinecolor=theme.grid,
    )
    return theme


def _title_with_filter_metadata(title: str, data: ChannelPlotData) -> str:
    """Append the applied level and removed FIR delay to a channel plot title."""
    values = (
        float(data.auto_global_gain_db),
        float(data.auto_headroom_db),
        float(data.filter_delay_ms),
    )
    if not all(np.isfinite(value) for value in values):
        return title
    auto_gain_db, headroom_db, filter_delay_ms = values
    metadata = " | ".join(
        (
            t("results_plot_auto_gain").format(value=auto_gain_db),
            t("results_plot_headroom").format(value=headroom_db),
            t("results_plot_filter_delay_removed").format(value=filter_delay_ms),
        )
    )
    return (
        f"{title}<span style='font-size:0.78em; font-weight:400;'>"
        f"&nbsp;&nbsp;&nbsp;&nbsp;{metadata}</span>"
    )


def _apply_log_x_axis(
    fig: go.Figure,
    *,
    range_hz: tuple[float, float],
    full_max_hz: float,
    row: int | None = None,
) -> None:
    kwargs = {
        "type": "log",
        "range": [np.log10(range_hz[0]), np.log10(range_hz[1])],
        "tickvals": _tick_values(full_max_hz),
    }
    if row is None:
        fig.update_xaxes(**kwargs)
    else:
        fig.update_xaxes(row=row, col=1, **kwargs)


def _add_correction_and_guard_shapes(
    fig: go.Figure,
    data: ChannelPlotData,
    *,
    row: int = 1,
) -> None:
    if data.has_valid_correction_range:
        fig.add_vrect(
            x0=data.correction_min_hz,
            x1=data.correction_max_hz,
            fillcolor="rgba(52,211,153,0.055)",
            line_width=0,
            layer="below",
            row=row,
            col=1,
        )
    if np.isfinite(data.lf_guard_hz) and data.lf_guard_hz > 10.0:
        fig.add_vrect(
            x0=10.0,
            x1=data.lf_guard_hz,
            fillcolor="rgba(248,113,113,0.075)",
            line_width=0,
            layer="below",
            row=row,
            col=1,
        )


def _set_results_plot_controls(
    fig: go.Figure,
    *,
    exported_indexes: list[int] | None = None,
    compensated_indexes: list[int] | None = None,
    correction_range: tuple[float, float] | None = None,
    full_range: tuple[float, float] | None = None,
    default_range_mode: str = "full",
) -> None:
    """Store UI control metadata without placing buttons inside the plot."""
    fig.update_layout(
        meta={
            "results_plot_controls": {
                "base_visibility": [
                    trace.visible if trace.visible is not None else True
                    for trace in fig.data
                ],
                "exported_indexes": list(exported_indexes or []),
                "compensated_indexes": list(compensated_indexes or []),
                "correction_range": list(correction_range) if correction_range else None,
                "full_range": list(full_range) if full_range else None,
                "default_range_mode": str(default_range_mode),
            }
        }
    )


def results_plot_control_metadata(fig: go.Figure) -> dict:
    meta = fig.layout.meta
    if not isinstance(meta, dict):
        return {}
    controls = meta.get("results_plot_controls")
    return dict(controls) if isinstance(controls, dict) else {}


def apply_results_plot_level(fig: go.Figure, mode: str) -> None:
    """Apply exported/compensated visibility while preserving legend-only traces."""
    controls = results_plot_control_metadata(fig)
    base = list(controls.get("base_visibility") or [])
    if len(base) != len(fig.data):
        return
    exported_indexes = [int(index) for index in controls.get("exported_indexes", [])]
    compensated_indexes = [
        int(index) for index in controls.get("compensated_indexes", [])
    ]
    visibility = list(base)
    for index in exported_indexes:
        visibility[index] = mode in {"exported", "both"}
    for index in compensated_indexes:
        visibility[index] = mode in {"compensated", "both"}
    for trace, visible in zip(fig.data, visibility, strict=True):
        trace.visible = visible


def apply_results_plot_range(fig: go.Figure, mode: str) -> None:
    controls = results_plot_control_metadata(fig)
    key = "correction_range" if mode == "correction" else "full_range"
    range_hz = controls.get(key)
    if not isinstance(range_hz, (list, tuple)) or len(range_hz) != 2:
        return
    lo_hz, hi_hz = float(range_hz[0]), float(range_hz[1])
    if not (np.isfinite(lo_hz) and np.isfinite(hi_hz) and 0.0 < lo_hz < hi_hz):
        return
    fig.update_xaxes(range=[np.log10(lo_hz), np.log10(hi_hz)])


def _targets_equal(left: np.ndarray | None, right: np.ndarray | None) -> bool:
    if left is None or right is None or left.shape != right.shape:
        return False
    return bool(np.allclose(left, right, rtol=1e-5, atol=1e-4, equal_nan=True))


def build_lr_overview_figure(
    left: ChannelPlotData,
    right: ChannelPlotData,
    *,
    dark: bool,
) -> go.Figure:
    fig = go.Figure()
    predicted_exported_indexes: list[int] = []
    predicted_compensated_indexes: list[int] = []

    for data, side_key in ((left, "results_left_channel"), (right, "results_right_channel")):
        color = _channel_color(data.channel_key)
        side = t(side_key)
        fig.add_trace(
            go.Scatter(
                x=data.freq_hz,
                y=data.measured_db,
                name=t("results_plot_measured_channel").format(channel=side),
                line=dict(color=color, width=1.2, dash="dot"),
                opacity=0.48,
            )
        )
        predicted_exported_indexes.append(len(fig.data))
        fig.add_trace(
            go.Scatter(
                x=data.freq_hz,
                y=data.predicted_exported_db,
                name=t("results_plot_predicted_channel").format(channel=side),
                line=dict(color=color, width=2.2),
            )
        )
        predicted_compensated_indexes.append(len(fig.data))
        fig.add_trace(
            go.Scatter(
                x=data.freq_hz,
                y=data.predicted_compensated_db,
                name=t("results_plot_predicted_comp_channel").format(channel=side),
                line=dict(color=color, width=2.0, dash="dash"),
                visible=False,
            )
        )

    if _targets_equal(left.effective_target_db, right.effective_target_db):
        fig.add_trace(
            go.Scatter(
                x=left.target_freq_hz,
                y=left.effective_target_db,
                name=t("results_plot_effective_target"),
                line=dict(color=_TARGET, width=1.8, dash="dash"),
            )
        )
    else:
        for data, side_key in ((left, "results_left_channel"), (right, "results_right_channel")):
            if data.effective_target_db is None:
                continue
            fig.add_trace(
                go.Scatter(
                    x=data.target_freq_hz,
                    y=data.effective_target_db,
                    name=t("results_plot_effective_target_channel").format(
                        channel=t(side_key),
                    ),
                    line=dict(
                        color=_channel_color(data.channel_key),
                        width=1.5,
                        dash="dash",
                    ),
                )
            )

    requested = left.requested_target_db
    if requested is not None:
        fig.add_trace(
            go.Scatter(
                x=left.target_freq_hz,
                y=requested,
                name=t("results_plot_requested_target"),
                line=dict(color=_TARGET, width=1.2, dash="dot"),
                visible="legendonly",
            )
        )

    correction_range = _overview_range(left, right)
    full_range = (10.0, min(left.full_max_hz, right.full_max_hz))
    values = np.concatenate(
        (
            left.measured_db,
            left.predicted_exported_db,
            right.measured_db,
            right.predicted_exported_db,
        )
    )
    frequencies = np.tile(left.freq_hz, 4)
    y_range = _robust_axis_range(
        frequencies,
        values,
        focus_band=correction_range,
        min_span=12.0,
    )

    _base_layout(
        fig,
        title=t("results_plot_overview_title"),
        dark=dark,
        height=520,
    )
    _apply_log_x_axis(
        fig,
        range_hz=correction_range,
        full_max_hz=full_range[1],
    )
    fig.update_yaxes(title_text=t("results_plot_magnitude_axis"), range=y_range)
    _set_results_plot_controls(
        fig,
        exported_indexes=predicted_exported_indexes,
        compensated_indexes=predicted_compensated_indexes,
        correction_range=correction_range,
        full_range=full_range,
        default_range_mode="correction",
    )
    _add_correction_and_guard_shapes(fig, left)
    return fig


def build_response_figure(
    data: ChannelPlotData,
    *,
    title: str,
    dark: bool,
    default_full_range: bool = True,
) -> go.Figure:
    has_hybrid_iir = bool(data.hybrid_iir_cuts)
    if has_hybrid_iir:
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.10,
            row_heights=(0.74, 0.26),
            subplot_titles=("", t("results_plot_hybrid_iir_title")),
        )
    else:
        fig = go.Figure()

    def add_response_trace(trace: go.Scatter) -> None:
        if has_hybrid_iir:
            fig.add_trace(trace, row=1, col=1)
        else:
            fig.add_trace(trace)

    add_response_trace(
        go.Scatter(
            x=data.freq_hz,
            y=data.measured_db,
            name=t("results_plot_measured"),
            line=dict(color=_channel_color(data.channel_key), width=1.3, dash="dot"),
            opacity=0.55,
        )
    )
    exported_index = len(fig.data)
    add_response_trace(
        go.Scatter(
            x=data.freq_hz,
            y=data.predicted_exported_db,
            name=t("results_plot_predicted_exported"),
            line=dict(color=_channel_color(data.channel_key), width=2.2),
        )
    )
    compensated_index = len(fig.data)
    add_response_trace(
        go.Scatter(
            x=data.freq_hz,
            y=data.predicted_compensated_db,
            name=t("results_plot_predicted_compensated"),
            line=dict(color=_channel_color(data.channel_key), width=2.0, dash="dash"),
            visible=False,
        )
    )
    if data.effective_target_db is not None:
        add_response_trace(
            go.Scatter(
                x=data.target_freq_hz,
                y=data.effective_target_db,
                name=t("results_plot_effective_target"),
                line=dict(color=_TARGET, width=1.8, dash="dash"),
            )
        )
    if data.requested_target_db is not None:
        add_response_trace(
            go.Scatter(
                x=data.target_freq_hz,
                y=data.requested_target_db,
                name=t("results_plot_requested_target"),
                line=dict(color=_TARGET, width=1.2, dash="dot"),
                visible="legendonly",
            )
        )

    if has_hybrid_iir:
        combined_response_db = np.sum(
            np.stack(
                [cut.response_db for cut in data.hybrid_iir_cuts],
                axis=0,
            ),
            axis=0,
        )
        if len(data.hybrid_iir_cuts) > 1:
            fig.add_trace(
                go.Scatter(
                    x=data.freq_hz,
                    y=combined_response_db,
                    name=t("results_plot_hybrid_iir_total").format(
                        count=len(data.hybrid_iir_cuts),
                    ),
                    line=dict(color=_HYBRID_IIR, width=2.4),
                    legendgroup="hybrid_iir",
                ),
                row=2,
                col=1,
            )
        for index, cut in enumerate(data.hybrid_iir_cuts, start=1):
            fig.add_trace(
                go.Scatter(
                    x=data.freq_hz,
                    y=cut.response_db,
                    name=t("results_plot_hybrid_iir_cut").format(
                        index=index,
                        freq=cut.freq_hz,
                        gain=cut.gain_db,
                        q=cut.q,
                    ),
                    line=dict(
                        color=_HYBRID_IIR,
                        width=2.1 if len(data.hybrid_iir_cuts) == 1 else 1.3,
                        dash="solid" if len(data.hybrid_iir_cuts) == 1 else "dot",
                    ),
                    opacity=1.0 if len(data.hybrid_iir_cuts) == 1 else 0.68,
                    legendgroup="hybrid_iir",
                ),
                row=2,
                col=1,
            )

    full_range = _full_range(data)
    correction_range = (
        (data.correction_min_hz, data.correction_max_hz)
        if data.has_valid_correction_range
        else full_range
    )
    initial_range = full_range if default_full_range else correction_range
    _base_layout(
        fig,
        title=_title_with_filter_metadata(title, data),
        dark=dark,
        height=650 if has_hybrid_iir else 500,
    )
    _apply_log_x_axis(fig, range_hz=initial_range, full_max_hz=full_range[1])
    if has_hybrid_iir:
        minimum_iir_db = float(np.nanmin(combined_response_db))
        iir_axis_min_db = min(-1.0, 1.15 * minimum_iir_db)
        fig.update_xaxes(title_text=None, showticklabels=False, row=1, col=1)
        fig.update_xaxes(
            title_text=t("results_plot_frequency_axis"),
            showticklabels=True,
            row=2,
            col=1,
        )
        fig.update_yaxes(
            title_text=t("results_plot_magnitude_axis"),
            autorange=True,
            row=1,
            col=1,
        )
        fig.update_yaxes(
            title_text=t("results_plot_hybrid_iir_axis"),
            range=[iir_axis_min_db, 0.5],
            row=2,
            col=1,
        )
        fig.add_hline(
            y=0.0,
            line_width=1,
            line_dash="dot",
            line_color=plot_theme(dark=dark).muted,
            row=2,
            col=1,
        )
    else:
        fig.update_yaxes(title_text=t("results_plot_magnitude_axis"), autorange=True)
    _set_results_plot_controls(
        fig,
        exported_indexes=[exported_index],
        compensated_indexes=[compensated_index],
        correction_range=correction_range,
        full_range=full_range,
        default_range_mode="full" if default_full_range else "correction",
    )
    _add_correction_and_guard_shapes(fig, data, row=1)
    return fig


def build_filter_figure(
    data: ChannelPlotData,
    *,
    title: str,
    dark: bool,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data.freq_hz,
            y=data.filter_exported_db,
            name=t("results_plot_filter_exported"),
            line=dict(color=_FILTER, width=1.8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data.freq_hz,
            y=data.filter_compensated_db,
            name=t("results_plot_filter_compensated"),
            line=dict(color=_FILTER, width=1.7, dash="dash"),
            visible=False,
        )
    )
    full_range = _full_range(data)
    correction_range = (
        (data.correction_min_hz, data.correction_max_hz)
        if data.has_valid_correction_range
        else full_range
    )
    _base_layout(
        fig,
        title=_title_with_filter_metadata(title, data),
        dark=dark,
        height=440,
    )
    _apply_log_x_axis(fig, range_hz=full_range, full_max_hz=full_range[1])
    fig.update_yaxes(title_text=t("results_plot_filter_axis"), range=[-30.0, 12.0])
    _set_results_plot_controls(
        fig,
        exported_indexes=[0],
        compensated_indexes=[1],
        correction_range=correction_range,
        full_range=full_range,
        default_range_mode="full",
    )
    _add_correction_and_guard_shapes(fig, data)
    return fig


def build_timing_figure(
    data: ChannelPlotData,
    *,
    title: str,
    dark: bool,
) -> go.Figure:
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.09,
        subplot_titles=(
            t("results_plot_phase_title"),
            t("results_plot_group_delay_title"),
            t("results_plot_impulse_title"),
        ),
    )
    corrected_band_mask = (
        np.isfinite(data.freq_hz)
        & (data.freq_hz >= 10.0)
        & (data.freq_hz <= float(data.phase_display_max_hz))
    )
    fig.add_trace(
        go.Scatter(
            x=data.freq_hz[corrected_band_mask],
            y=data.system_phase_before_deg[corrected_band_mask],
            name=t("results_plot_system_before"),
            line=dict(color=_LEFT, width=1.2, dash="dot"),
            showlegend=True,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data.freq_hz[corrected_band_mask],
            y=data.system_phase_after_deg[corrected_band_mask],
            name=t("results_plot_system_after"),
            line=dict(color=_TIMING, width=1.4),
            showlegend=True,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data.freq_hz[corrected_band_mask],
            y=data.system_group_delay_before_ms[corrected_band_mask],
            name=t("results_plot_system_before"),
            line=dict(color=_LEFT, width=1.2, dash="dot"),
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data.freq_hz[corrected_band_mask],
            y=data.system_group_delay_after_ms[corrected_band_mask],
            name=t("results_plot_system_after"),
            line=dict(color=_TIMING, width=1.4),
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=data.filter_impulse_time_ms,
            y=data.filter_impulse_normalized,
            name=t("results_plot_filter_impulse"),
            line=dict(color=_FILTER, width=1.3),
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    _base_layout(
        fig,
        title=_title_with_filter_metadata(title, data),
        dark=dark,
        height=850,
        show_legend=True,
    )
    _apply_log_x_axis(
        fig,
        range_hz=(10.0, float(data.phase_display_max_hz)),
        full_max_hz=float(data.phase_display_max_hz),
        row=1,
    )
    _apply_log_x_axis(
        fig,
        range_hz=(10.0, float(data.phase_display_max_hz)),
        full_max_hz=float(data.phase_display_max_hz),
        row=2,
    )
    fig.update_yaxes(title_text=t("results_plot_phase_axis"), row=1, col=1)
    fig.update_yaxes(title_text=t("results_plot_group_delay_axis"), row=2, col=1)
    fig.update_xaxes(
        title_text=t("results_plot_impulse_time_axis"),
        range=list(FILTER_IMPULSE_VIEW_RANGE_MS),
        row=3,
        col=1,
    )
    fig.update_yaxes(
        title_text=t("results_plot_impulse_amplitude_axis"),
        range=[-1.05, 1.05],
        row=3,
        col=1,
    )
    fig.add_vline(
        x=0.0,
        line_width=1.0,
        line_dash="dot",
        line_color=plot_theme(dark=dark).muted,
        row=3,
        col=1,
    )
    return fig


def build_quality_figure(
    data: ChannelPlotData,
    *,
    title: str,
    dark: bool,
) -> go.Figure | None:
    rows = int(data.confidence is not None) + int(data.afdw_bw_oct is not None)
    if rows == 0:
        return None
    subplot_titles = []
    if data.confidence is not None:
        subplot_titles.append(t("plot_confidence_title"))
    if data.afdw_bw_oct is not None:
        subplot_titles.append(t("results_plot_afdw_title"))
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.12 if rows > 1 else 0.08,
        subplot_titles=tuple(subplot_titles),
    )
    row = 1
    if data.confidence is not None:
        fig.add_trace(
            go.Scatter(
                x=data.freq_hz,
                y=data.confidence,
                name=t("plot_confidence_title"),
                fill="tozeroy",
                fillcolor="rgba(148,163,184,0.18)",
                line=dict(color="rgba(148,163,184,0.65)", width=1.8),
                showlegend=False,
            ),
            row=row,
            col=1,
        )
        fig.update_yaxes(
            range=[0.0, 1.0],
            tickvals=[0.0, 0.25, 0.5, 0.75, 1.0],
            ticktext=["0%", "25%", "50%", "75%", "100%"],
            row=row,
            col=1,
        )
        row += 1
    if data.afdw_bw_oct is not None:
        fig.add_trace(
            go.Scatter(
                x=data.freq_hz,
                y=data.afdw_bw_oct,
                name=t("results_plot_afdw_title"),
                fill="tozeroy",
                fillcolor="rgba(56,189,248,0.20)",
                line=dict(color=_LEFT, width=2.0),
                showlegend=False,
            ),
            row=row,
            col=1,
        )
        fig.update_yaxes(title_text="oct", row=row, col=1)

    full_range = _full_range(data)
    _base_layout(
        fig,
        title=title,
        dark=dark,
        height=620 if rows > 1 else 430,
        show_legend=False,
    )
    for row_index in range(1, rows + 1):
        _apply_log_x_axis(
            fig,
            range_hz=full_range,
            full_max_hz=full_range[1],
            row=row_index,
        )
    if data.confidence is not None:
        fig.update_xaxes(
            showticklabels=True,
            title_text=t("results_plot_frequency_axis"),
            row=1,
            col=1,
        )
    return fig


def build_combined_bass_figure(
    *,
    freq_hz,
    measured_db,
    predicted_db,
    target_db,
    title: str,
    dark: bool,
) -> go.Figure:
    freq = np.asarray(freq_hz, dtype=float)
    mask = (freq >= 10.0) & (freq <= 500.0)
    offset_db = 0.0
    predicted = np.asarray(predicted_db, dtype=float) if predicted_db is not None else None
    measured = np.asarray(measured_db, dtype=float) if measured_db is not None else None
    target = np.asarray(target_db, dtype=float) if target_db is not None else None
    reference = predicted if predicted is not None else measured
    if reference is not None and target is not None:
        align_mask = (
            mask
            & (freq >= 50.0)
            & (freq <= 200.0)
            & np.isfinite(reference)
            & np.isfinite(target)
        )
        if np.count_nonzero(align_mask) >= 4:
            offset_db = float(np.median(reference[align_mask] - target[align_mask]))

    fig = go.Figure()
    if measured is not None and measured.size == freq.size:
        fig.add_trace(
            go.Scatter(
                x=freq[mask],
                y=(measured - offset_db)[mask],
                name=t("results_plot_measured_combined"),
                line=dict(color="rgba(148,163,184,0.58)", width=1.2, dash="dot"),
            )
        )
    if predicted is not None and predicted.size == freq.size:
        fig.add_trace(
            go.Scatter(
                x=freq[mask],
                y=(predicted - offset_db)[mask],
                name=t("results_plot_predicted_combined"),
                line=dict(color=_LEFT, width=2.2),
            )
        )
    if target is not None and target.size == freq.size:
        fig.add_trace(
            go.Scatter(
                x=freq[mask],
                y=target[mask],
                name=t("results_plot_effective_target"),
                line=dict(color=_TARGET, width=1.7, dash="dash"),
            )
        )
    _base_layout(fig, title=title, dark=dark, height=520)
    _apply_log_x_axis(fig, range_hz=(10.0, 500.0), full_max_hz=500.0)
    fig.update_yaxes(title_text=t("results_plot_magnitude_axis"), autorange=True)
    return fig


def results_plot_theme_javascript(*, dark: bool) -> str:
    """Relayout mounted Results plots after the global theme changes."""
    theme = plot_theme(dark=dark)
    return f"""
document.querySelectorAll('.cf-results-plot .js-plotly-plot').forEach((plot) => {{
  if (window.Plotly) {{
    Plotly.relayout(plot, {{
      paper_bgcolor: '{theme.paper}',
      plot_bgcolor: '{theme.plot}',
      'font.color': '{theme.text}',
      'xaxis.gridcolor': '{theme.grid}',
      'xaxis.linecolor': '{theme.axis}',
      'yaxis.gridcolor': '{theme.grid}',
      'yaxis.linecolor': '{theme.axis}',
      'xaxis2.gridcolor': '{theme.grid}',
      'xaxis2.linecolor': '{theme.axis}',
      'yaxis2.gridcolor': '{theme.grid}',
      'yaxis2.linecolor': '{theme.axis}'
    }});
  }}
}});
"""


__all__ = [
    "PlotTheme",
    "apply_results_plot_level",
    "apply_results_plot_range",
    "build_combined_bass_figure",
    "build_filter_figure",
    "build_lr_overview_figure",
    "build_quality_figure",
    "build_response_figure",
    "build_timing_figure",
    "plot_theme",
    "results_plot_control_metadata",
    "results_plot_theme_javascript",
]
