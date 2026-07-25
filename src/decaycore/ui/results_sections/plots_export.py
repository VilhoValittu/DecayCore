# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Compact Results plots and lazily rendered diagnostics."""

from __future__ import annotations

import logging

from ...resources.i8n.decaycore_i18n import t
from ..plot_prediction_parts import ChannelPlotData, compute_channel_plot_data
from ..results_plot_figures import (
    apply_results_plot_level,
    apply_results_plot_range,
    build_filter_figure,
    build_quality_figure,
    build_response_figure,
    build_timing_figure,
    results_plot_control_metadata,
)

logger = logging.getLogger("DecayCore")

_PLOT_DATA_CACHE: dict[str, ChannelPlotData] = {}
_PLOT_FIGURE_CACHE: dict[tuple[str, str, bool], object] = {}

_DIRECT_DAC_COMBINED_KEYS = frozenset(
    {
        "direct_dac_sum_measured_mags",
        "direct_dac_sum_predicted_mags",
        "direct_dac_sum_predicted_mags_comp",
    }
)

_PLOT_EXCEPTIONS = (
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
)


def clear_plot_render_cache() -> None:
    """Clear all per-run Results plot data and figure objects."""
    _PLOT_DATA_CACHE.clear()
    _PLOT_FIGURE_CACHE.clear()


def _channel_title(channel_key: str) -> str:
    return {
        "left": t("results_left_channel"),
        "right": t("results_right_channel"),
        "sub": t("results_sub_channel"),
    }.get(channel_key, channel_key)


def _compute_channel(
    *,
    channel_key: str,
    freqs,
    mags,
    phases,
    filt_ir,
    stats,
    fs_hz: int,
    smoothing,
) -> ChannelPlotData:
    cached = _PLOT_DATA_CACHE.get(channel_key)
    if cached is not None:
        return cached
    result = compute_channel_plot_data(
        channel_key=channel_key,
        orig_freqs=freqs,
        orig_mags=mags,
        orig_phases=phases,
        filt_ir=filt_ir,
        fs=fs_hz,
        target_stats=stats,
        plot_smoothing_level=smoothing,
    )
    _PLOT_DATA_CACHE[channel_key] = result
    return result


def _diagnostic_figure(
    *,
    channel_data: ChannelPlotData,
    view_key: str,
    dark: bool,
):
    cache_key = (channel_data.channel_key, view_key, bool(dark))
    cached = _PLOT_FIGURE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    channel_title = _channel_title(channel_data.channel_key)
    title = t(f"results_plot_{view_key}_heading").format(channel=channel_title)
    if view_key == "response":
        figure = build_response_figure(
            channel_data,
            title=title,
            dark=dark,
            default_full_range=True,
        )
    elif view_key == "filter":
        figure = build_filter_figure(channel_data, title=title, dark=dark)
    elif view_key == "timing":
        figure = build_timing_figure(channel_data, title=title, dark=dark)
    elif view_key == "quality":
        figure = build_quality_figure(channel_data, title=title, dark=dark)
    else:
        figure = None
    if figure is not None:
        _PLOT_FIGURE_CACHE[cache_key] = figure
    return figure


def _render_plot_widget(figure) -> None:
    from nicegui import ui

    controls = results_plot_control_metadata(figure)
    widget_ref: dict[str, object] = {}

    def refresh_widget() -> None:
        widget = widget_ref.get("widget")
        if widget is None:
            return
        widget.figure = figure
        widget.update()

    def on_level_change(event) -> None:
        mode = _resolve_toggle_value(
            event.value,
            ("exported", "compensated", "both"),
            fallback="exported",
        )
        apply_results_plot_level(figure, mode)
        refresh_widget()

    def on_range_change(event) -> None:
        mode = _resolve_toggle_value(
            event.value,
            ("correction", "full"),
            fallback="full",
        )
        apply_results_plot_range(figure, mode)
        refresh_widget()

    has_level_control = bool(
        controls.get("exported_indexes") and controls.get("compensated_indexes")
    )
    has_range_control = bool(
        controls.get("correction_range") and controls.get("full_range")
    )
    if has_level_control or has_range_control:
        with ui.row().classes(
            "w-full items-center justify-between gap-3 flex-wrap px-2 pt-1"
        ):
            if has_level_control:
                with ui.row().classes("items-center gap-2 flex-wrap"):
                    ui.label(t("results_plot_level_control")).classes(
                        "text-sm text-gray-400"
                    )
                    ui.toggle(
                        {
                            "exported": t("plot_level_exported"),
                            "compensated": t("plot_level_compensated"),
                            "both": t("plot_level_both"),
                        },
                        value="exported",
                        on_change=on_level_change,
                    ).props("dense no-caps").classes("cf-results-choice-toggle")
            if has_range_control:
                with ui.row().classes("items-center gap-2 flex-wrap"):
                    ui.label(t("results_plot_range_control")).classes(
                        "text-sm text-gray-400"
                    )
                    ui.toggle(
                        {
                            "correction": t("results_plot_range_correction"),
                            "full": t("results_plot_range_full"),
                        },
                        value=str(controls.get("default_range_mode", "full")),
                        on_change=on_range_change,
                    ).props("dense no-caps").classes("cf-results-choice-toggle")

    widget_ref["widget"] = ui.plotly(figure).classes("w-full cf-results-plot")


def _resolve_toggle_value(value, options: tuple[str, ...], *, fallback: str) -> str:
    """Normalize NiceGUI 3.13 QBtnToggle's index-valued change events."""
    if isinstance(value, int) and not isinstance(value, bool):
        return options[value] if 0 <= value < len(options) else fallback
    selected = str(value)
    return selected if selected in options else fallback


def _resolve_diagnostic_channel(value, channel_inputs: dict[str, dict]) -> str:
    selected = str(value)
    return selected if selected in channel_inputs else "left"


def _resolve_diagnostic_view(value) -> str:
    selected = str(value)
    return (
        selected
        if selected in {"response", "filter", "timing", "quality"}
        else "response"
    )


def _render_diagnostics(
    *,
    channel_inputs: dict[str, dict],
    dark: bool,
) -> None:
    from nicegui import ui

    state = {
        "open": False,
        "channel": "left",
        "view": "response",
    }
    holder_ref: dict[str, object] = {}

    def render_selected() -> None:
        if not state["open"]:
            return
        holder = holder_ref.get("holder")
        if holder is None:
            return
        holder.clear()
        channel_key = str(state["channel"])
        source = channel_inputs[channel_key]
        try:
            channel_data = _compute_channel(channel_key=channel_key, **source)
            figure = _diagnostic_figure(
                channel_data=channel_data,
                view_key=str(state["view"]),
                dark=dark,
            )
        except _PLOT_EXCEPTIONS:
            logger.debug("Diagnostic plot generation failed", exc_info=True)
            figure = None
        with holder:
            if figure is None:
                ui.label(t("results_plot_quality_unavailable")).classes(
                    "text-sm text-gray-400 py-6"
                )
            else:
                _render_plot_widget(figure)

    def on_channel_change(event) -> None:
        state["channel"] = _resolve_diagnostic_channel(
            event.value,
            channel_inputs,
        )
        render_selected()

    def on_view_change(event) -> None:
        state["view"] = _resolve_diagnostic_view(event.value)
        render_selected()

    def on_expansion_change(event) -> None:
        state["open"] = bool(event.value)
        render_selected()

    with ui.expansion(
        t("results_plot_diagnostics_title"),
        value=False,
        on_value_change=on_expansion_change,
    ).classes("w-full mt-3"):
        with ui.tabs(on_change=on_channel_change).classes("w-full") as channel_tabs:
            ui.tab("left", label=t("results_left_channel"))
            ui.tab("right", label=t("results_right_channel"))
            if "sub" in channel_inputs:
                ui.tab("sub", label=t("results_sub_channel"))
        channel_tabs.set_value("left")

        with ui.tabs(on_change=on_view_change).classes("w-full mt-2") as view_tabs:
            ui.tab("response", label=t("results_plot_view_response"))
            ui.tab("filter", label=t("results_plot_view_filter"))
            ui.tab("timing", label=t("results_plot_view_timing"))
            ui.tab("quality", label=t("results_plot_view_quality"))
        view_tabs.set_value("response")
        holder_ref["holder"] = ui.column().classes("w-full min-w-0")


def _render_plots_and_export(
    *,
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
    saved_filters_dir=None,
    sub_imp_f=None,
    sub_meas_f=None,
    sub_st_f=None,
) -> None:
    from nicegui import ui

    smoothing = data.get("plot_smoothing_level", "Psychoacoustic")
    fs_hz = int(data.get("fs", 48000) or 48000)
    dark = bool(data.get("ui_theme_dark", True))
    sub_measurements = dict(sub_meas_f or {})
    has_sub = bool(
        sub_imp_f is not None
        and sub_measurements.get("f_sub") is not None
        and len(sub_measurements.get("f_sub", [])) > 0
    )
    left_stats = {
        key: value
        for key, value in dict(l_st_f or {}).items()
        if key not in _DIRECT_DAC_COMBINED_KEYS
    }
    right_stats = {
        key: value
        for key, value in dict(r_st_f or {}).items()
        if key not in _DIRECT_DAC_COMBINED_KEYS
    }
    channel_inputs = {
        "left": {
            "freqs": f_l,
            "mags": m_l,
            "phases": p_l,
            "filt_ir": l_imp_f,
            "stats": left_stats,
            "fs_hz": fs_hz,
            "smoothing": smoothing,
        },
        "right": {
            "freqs": f_r,
            "mags": m_r,
            "phases": p_r,
            "filt_ir": r_imp_f,
            "stats": right_stats,
            "fs_hz": fs_hz,
            "smoothing": smoothing,
        },
    }
    if has_sub:
        channel_inputs["sub"] = {
            "freqs": sub_measurements["f_sub"],
            "mags": sub_measurements["m_sub"],
            "phases": sub_measurements["p_sub"],
            "filt_ir": sub_imp_f,
            "stats": dict(sub_st_f or {}),
            "fs_hz": fs_hz,
            "smoothing": smoothing,
        }

    with ui.card().classes("w-full min-w-0"):
        _render_diagnostics(channel_inputs=channel_inputs, dark=dark)

    with ui.row().classes("w-full items-center gap-4 mt-2"):
        if zip_buffer is not None and fname:
            try:
                zip_bytes = zip_buffer.getvalue()
                ui.button(
                    t("results_download_zip").format(fname=fname),
                    on_click=lambda: ui.download(zip_bytes, filename=fname),
                ).props('color="primary" unelevated').classes("font-bold")
            except _PLOT_EXCEPTIONS:
                logger.debug("ZIP download button failed", exc_info=True)
        if saved_filters_dir:
            ui.label(t("results_saved_to").format(path=saved_filters_dir)).classes(
                "text-sm text-gray-400"
            )


__all__ = [
    "_PLOT_DATA_CACHE",
    "_PLOT_FIGURE_CACHE",
    "_diagnostic_figure",
    "_render_plots_and_export",
    "_resolve_diagnostic_channel",
    "_resolve_diagnostic_view",
    "_resolve_toggle_value",
    "clear_plot_render_cache",
]
