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

# ---------------------------------------------------------------------------
# Interactive plot render cache
# Key: (run_signature, channel, plot_smoothing_level)
# Value: plotly Figure object
# Cleared at the start of each render_results call (per-run semantics).
# ---------------------------------------------------------------------------
_PLOT_RENDER_CACHE: dict = {}


def clear_plot_render_cache() -> None:
    """Clear the per-run plotly figure cache. Called by reset_runtime_caches()."""
    _PLOT_RENDER_CACHE.clear()

from ...resources.i8n.decaycore_i18n import t
from .. import decaycore_plot as plots


logger = logging.getLogger("DecayCore")

_DIRECT_DAC_COMBINED_KEYS = frozenset({
    "direct_dac_sum_measured_mags",
    "direct_dac_sum_predicted_mags",
    "direct_dac_sum_predicted_mags_comp",
})


def _build_combined_fig(freq_hz, measured_db, predicted_db, title: str,
                        target_db=None):
    import numpy as np
    import plotly.graph_objects as go

    f = np.asarray(freq_hz, dtype=float)
    mask = (f >= 10.0) & (f <= 500.0)
    f_plot = f[mask]

    # Align measured and predicted to the target's dB scale.
    # measured/predicted are in absolute FFT dB; target is in relative dB.
    # Align from the predicted (post-FIR output) rather than the raw measured:
    # the FIR changes the absolute level, so an offset calibrated from the raw
    # measured would mis-place the predicted curve at bass/sub frequencies.
    # Using the predicted at 50–200 Hz (main-dominated, FIR-corrected region)
    # gives the correct reference for both main and sub ranges.
    offset_db = 0.0
    ref_for_align = predicted_db if predicted_db is not None else measured_db
    if ref_for_align is not None and target_db is not None:
        ref_arr = np.asarray(ref_for_align, dtype=float)
        tgt_arr = np.asarray(target_db, dtype=float)
        if ref_arr.size == f.size and tgt_arr.size == f.size:
            align_mask = (
                mask
                & (f >= 50.0) & (f <= 200.0)
                & np.isfinite(ref_arr) & np.isfinite(tgt_arr)
            )
            if np.count_nonzero(align_mask) >= 4:
                offset_db = float(np.median(ref_arr[align_mask] - tgt_arr[align_mask]))

    fig = go.Figure()
    if measured_db is not None:
        m = np.asarray(measured_db, dtype=float)
        if m.size == f.size:
            fig.add_trace(go.Scatter(
                x=f_plot, y=(m - offset_db)[mask],
                name="Measured (combined)",
                line=dict(color="rgba(148,163,184,0.55)", width=1.2, dash="dot"),
            ))
    if predicted_db is not None:
        p = np.asarray(predicted_db, dtype=float)
        if p.size == f.size:
            fig.add_trace(go.Scatter(
                x=f_plot, y=(p - offset_db)[mask],
                name="Predicted (combined)",
                line=dict(color="#60a5fa", width=2.0),
            ))
    if target_db is not None:
        tgt = np.asarray(target_db, dtype=float)
        if tgt.size == f.size:
            fig.add_trace(go.Scatter(
                x=f_plot, y=tgt[mask],
                name="Target",
                line=dict(color="#34d399", width=1.5, dash="dash"),
            ))
    fig.update_xaxes(
        type="log",
        range=[np.log10(10), np.log10(500)],
        tickvals=[10, 20, 50, 100, 200, 500],
    )
    fig.update_yaxes(autorange=True)
    fig.update_layout(
        title_text=f"{title} Analysis",
        height=350,
        template="plotly_dark",
        paper_bgcolor="#0e1219",
        plot_bgcolor="#0e1219",
        font=dict(color="#c8cdd5", family="Inter, system-ui, sans-serif"),
        uirevision="keep",
    )
    return fig


def _render_channel_plot(
    *,
    channel_key: str,
    title: str,
    f_ch,
    m_ch,
    p_ch,
    imp_f,
    st_f,
    psl,
    fs_v: int,
    mixed_freq,
) -> None:
    from nicegui import ui

    cache_key = (channel_key, psl)
    fig = _PLOT_RENDER_CACHE.get(cache_key)
    if fig is None:
        try:
            is_sub = channel_key == "sub"
            result = plots.generate_prediction_plot(
                f_ch,
                m_ch,
                p_ch,
                imp_f,
                fs_v,
                title,
                None,
                st_f,
                mixed_freq,
                "low",
                False,
                True,
                psl,
                x_max_hz=500.0 if is_sub else 20000.0,
                y_mag_range_db=20.0 if is_sub else None,
            )
            fig = result[1] if isinstance(result, tuple) else None
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
            logger.debug("Plot generation failed", exc_info=True)
            fig = None
        if fig is not None:
            _PLOT_RENDER_CACHE[cache_key] = fig
    if fig is not None:
        ui.plotly(fig).classes("w-full")
    else:
        ui.label(t("results_plot_unavailable")).classes("text-gray-400")


def _render_combined_plot(*, l_st_f: dict, psl, title: str) -> None:
    from nicegui import ui

    cache_key = ("combined", psl)
    fig = _PLOT_RENDER_CACHE.get(cache_key)
    if fig is None:
        try:
            fig = _build_combined_fig(
                freq_hz=l_st_f["freq_axis"],
                measured_db=l_st_f.get("direct_dac_sum_measured_mags"),
                predicted_db=l_st_f.get("direct_dac_sum_predicted_mags"),
                title=title,
                target_db=l_st_f.get("target_mags"),
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
            logger.debug("Combined plot generation failed", exc_info=True)
            fig = None
        if fig is not None:
            _PLOT_RENDER_CACHE[cache_key] = fig
    if fig is not None:
        ui.plotly(fig).classes("w-full")
    else:
        ui.label(t("results_plot_unavailable")).classes("text-gray-400")


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
    dash_html_l=None,
    dash_html_r=None,
    saved_filters_dir=None,
    auto_cache_path=None,
    optuna_storage_path=None,
    sub_imp_f=None,
    sub_meas_f=None,
    sub_st_f=None,
) -> None:
    from nicegui import ui  # noqa: PLC0415

    psl = data.get("plot_smoothing_level", "Psychoacoustic")
    mixed_freq = data.get("mixed_freq", 200.0)
    fs_v = int(data.get("fs", 48000) or 48000)

    sub_meas = dict(sub_meas_f or {})
    has_sub = (
        sub_imp_f is not None
        and sub_meas.get("f_sub") is not None
        and len(sub_meas.get("f_sub", [])) > 0
    )
    has_combined = has_sub and bool(
        l_st_f and l_st_f.get("direct_dac_sum_predicted_mags")
    )

    l_st_speaker = {k: v for k, v in (l_st_f or {}).items() if k not in _DIRECT_DAC_COMBINED_KEYS}
    r_st_speaker = {k: v for k, v in (r_st_f or {}).items() if k not in _DIRECT_DAC_COMBINED_KEYS}

    with ui.card().classes("w-full"):
        with ui.tabs().classes("w-full") as plot_tabs:
            tab_left = ui.tab(t("results_left_channel"))
            tab_right = ui.tab(t("results_right_channel"))
            tab_sub = ui.tab(t("results_sub_channel")) if has_sub else None
            tab_combined = ui.tab(t("results_combined_channel")) if has_combined else None

        with ui.tab_panels(plot_tabs, value=tab_left).classes("w-full"):
            with ui.tab_panel(tab_left).classes("w-full"):
                _render_channel_plot(
                    channel_key="left",
                    title=t("results_left_channel"),
                    f_ch=f_l, m_ch=m_l, p_ch=p_l,
                    imp_f=l_imp_f, st_f=l_st_speaker,
                    psl=psl,
                    fs_v=fs_v,
                    mixed_freq=mixed_freq,
                )

            with ui.tab_panel(tab_right).classes("w-full"):
                _render_channel_plot(
                    channel_key="right",
                    title=t("results_right_channel"),
                    f_ch=f_r, m_ch=m_r, p_ch=p_r,
                    imp_f=r_imp_f, st_f=r_st_speaker,
                    psl=psl,
                    fs_v=fs_v,
                    mixed_freq=mixed_freq,
                )

            if has_sub and tab_sub is not None:
                with ui.tab_panel(tab_sub).classes("w-full"):
                    _render_channel_plot(
                        channel_key="sub",
                        title=t("results_sub_channel"),
                        f_ch=sub_meas["f_sub"], m_ch=sub_meas["m_sub"], p_ch=sub_meas["p_sub"],
                        imp_f=sub_imp_f, st_f=sub_st_f,
                        psl=psl,
                        fs_v=fs_v,
                        mixed_freq=mixed_freq,
                    )

            if has_combined and tab_combined is not None:
                with ui.tab_panel(tab_combined).classes("w-full"):
                    _render_combined_plot(
                        l_st_f=l_st_f,
                        psl=psl,
                        title=t("results_combined_channel"),
                    )

    with ui.row().classes("w-full items-center gap-4 mt-2"):
        if zip_buffer is not None and fname:
            try:
                zip_bytes = zip_buffer.getvalue()
                ui.button(
                    t("results_download_zip").format(fname=fname),
                    on_click=lambda: ui.download(zip_bytes, filename=fname),
                ).props('color="primary" unelevated').classes("font-bold")
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
                logger.debug("ZIP download button failed", exc_info=True)

        if saved_filters_dir:
            ui.label(t("results_saved_to").format(path=saved_filters_dir)).classes("text-sm text-gray-400")


__all__ = ['_render_plots_and_export']


def _link_sibling_exports() -> None:
    import importlib
    package = __package__
    for module_name in ['overview', 'bass_integration', 'quality', 'plots_export']:
        if module_name == __name__.rsplit('.', 1)[-1]:
            continue
        module = importlib.import_module(f"{package}.{module_name}")
        for symbol in getattr(module, "__all__", ()):
            globals().setdefault(symbol, getattr(module, symbol))


_link_sibling_exports()
