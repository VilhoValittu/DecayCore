# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""NiceGUI Target tab builder.

Replaces build_target_section() from layout_builders.py.
"""
from __future__ import annotations

from typing import Callable

from .. import ng_controls as ctrl
from ..ng_sections import page_shell, section_card
from ...ui_i18n import (
    LVL_ALGO_MEDIAN,
    LVL_ALGO_OPTION_LABEL_KEYS,
    LVL_MODE_AUTO,
    LVL_MODE_MANUAL,
    LVL_MODE_OPTION_LABEL_KEYS,
    OUTPUT_TILT_SOURCE_OFF,
    OUTPUT_TILT_SOURCE_OPTION_LABEL_KEYS,
    normalize_lvl_algo_value,
    normalize_lvl_mode_value,
    normalize_output_tilt_source_value,
    tr_options,
)
from .preview_metadata import _render_target_decay_hint, _render_target_preview_metadata
from .preview_refresh import refresh_target_preview
from .preview_state import STATE

_HC_OPTS = {
    "Harman6":   "Harman 6 dB",
    "Harman8":   "Harman 8 dB",
    "Harman4":   "Harman 4 dB",
    "Harman10":  "Harman 10 dB",
    "Harman12":  "Harman 12 dB",
    "Studio":    "Studio Tilt",
    "Nearfield": "Nearfield",
    "HiFi":      "HiFi Loudness",
    "Speech":    "Speech",
    "Toole":     "Toole",
    "BK_Light":  "BK Light",
    "BK_Medium": "BK Medium",
    "BK_Strong": "BK Strong",
    "Flat":      "Flat",
    "Cinema":    "Cinema",
    "Upload":    "Upload Custom",
}


def _step_manual_target(delta_db: float) -> None:
    try:
        cur = float(ctrl.value("lvl_manual_db", 0.0) or 0.0)
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
        cur = 0.0

    nxt = round((float(cur) + float(delta_db)) * 10.0) / 10.0
    ctrl.set_value("lvl_manual_db", float(nxt))
    refresh_target_preview()

def _step_manual_target_tilt(delta_db_per_oct: float) -> None:
    try:
        cur = float(ctrl.value("manual_target_tilt_db_per_oct", 0.0) or 0.0)
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
        cur = 0.0

    nxt = round((float(cur) + float(delta_db_per_oct)) * 10.0) / 10.0
    ctrl.set_value("manual_target_tilt_db_per_oct", float(nxt))
    refresh_target_preview()


def build_target_tab(*, t: Callable, get_val: Callable) -> None:
    STATE.reset()
    STATE.translate = t

    with page_shell(title=t("tab_target"), intro=t("target_page_intro"), wide=True):
        _build_target_preview_section(t=t)
        _build_target_hc_section(t=t, get_val=get_val)
        _build_target_leveling_section(t=t, get_val=get_val)
        _build_target_mag_correction_section(t=t, get_val=get_val)


def _build_target_preview_section(*, t: Callable) -> None:
    from nicegui import ui

    with section_card(title=t("ui_target_preview"), intro=t("target_preview_legend_hint"), hero=True):
        hint_col = ui.column().classes("w-full gap-1")
        ctrl.register_container("target_decay_hint_scope", hint_col)

        preview_col = ui.column().classes("w-full")
        ctrl.register_container("target_preview_scope", preview_col)
        metadata_col = ui.column().classes("w-full")
        ctrl.register_container("target_preview_metadata_scope", metadata_col)
        _render_target_decay_hint()
        _render_target_preview_metadata()


def _build_target_hc_section(*, t: Callable, get_val: Callable) -> None:
    from nicegui import ui
    from ...application.house_curve_service import _normalize_hc_mode_key  # noqa: PLC0415

    _hc_file = ctrl._ValueHolder(get_val("hc_custom_file", None))
    ctrl.register("hc_custom_file", _hc_file)

    hc_value = _normalize_hc_mode_key(get_val("hc_mode", "Harman6"))
    with section_card(title=t("hc_mode")):
        ctrl.register(
            "hc_mode",
            ui.select(
                options=_HC_OPTS,
                value=hc_value,
                label=t("hc_mode"),
            ).props("dense outlined").classes("w-full"),
        )

        with ui.column().classes("w-full") as hc_upload_col:
            ui.label(t("hc_custom")).classes("text-sm font-medium")

            async def _on_hc_upload(e) -> None:
                _hc_file.set_value({
                    "filename": e.file.name,
                    "content": await e.file.read(),
                    "mime_type": getattr(e.file, "content_type", ""),
                })
                refresh_target_preview()

            ui.upload(
                label=t("hc_custom"),
                on_upload=_on_hc_upload,
                auto_upload=True,
            ).props('accept=".txt"').classes("w-full")
        ctrl.register_container("hc_custom_upload_col", hc_upload_col)
        hc_upload_col.set_visibility(hc_value == "Upload")


def _build_target_leveling_section(*, t: Callable, get_val: Callable) -> None:
    from nicegui import ui

    with section_card(title=t("ui_leveling_gain")):
        with ui.row().classes("w-full gap-4"):
            ctrl.register(
                "lvl_algo",
                ui.select(
                    tr_options(t, LVL_ALGO_OPTION_LABEL_KEYS),
                    value=normalize_lvl_algo_value(get_val("lvl_algo", LVL_ALGO_MEDIAN), t),
                    label=t("lvl_algo"),
                ).props("dense outlined").classes("flex-1"),
            )
            ctrl.register(
                "gain",
                ui.number(
                    label=t("gain"),
                    value=float(get_val("gain", 0.0) or 0.0),
                    format="%.1f",
                ).props("dense outlined").classes("flex-1"),
            )

        with ui.row().classes("w-full gap-4"):
            ctrl.register(
                "lvl_min",
                ui.number(
                    label=t("lvl_min"),
                    value=float(get_val("lvl_min", 500.0) or 500.0),
                    format="%.0f",
                ).props("dense outlined").classes("flex-1"),
            )
            ctrl.register(
                "lvl_max",
                ui.number(
                    label=t("lvl_max"),
                    value=float(get_val("lvl_max", 2000.0) or 2000.0),
                    format="%.0f",
                ).props("dense outlined").classes("flex-1"),
            )

        with ui.row().classes("w-full gap-4 items-start"):
            ctrl.register(
                "lvl_mode",
                ui.select(
                    options=tr_options(t, LVL_MODE_OPTION_LABEL_KEYS),
                    value=normalize_lvl_mode_value(get_val("lvl_mode", LVL_MODE_AUTO), t),
                    label=t("lvl_mode"),
                ).props("dense outlined").classes("flex-1"),
            )
            lvl_manual_col = ui.column().classes("flex-1 gap-1")
            ctrl.register_container("lvl_manual_scope", lvl_manual_col)
            with lvl_manual_col:
                with ui.row().classes("w-full gap-2 items-end"):
                    ctrl.register(
                        "lvl_manual_db",
                        ui.number(
                            label=t("lvl_target_db"),
                            value=float(get_val("lvl_manual_db", 0.0) or 0.0),
                            format="%.1f",
                        ).props("dense outlined step=0.1").classes("flex-1"),
                    )
                    ui.button(
                        "+",
                        on_click=lambda: _step_manual_target(+0.1),
                    ).props('color="secondary" outline').style("min-width:34px;")
                    ui.button(
                        "-",
                        on_click=lambda: _step_manual_target(-0.1),
                    ).props('color="secondary" outline').style("min-width:34px;")
                with ui.row().classes("w-full gap-2 items-end"):
                    ctrl.register(
                        "manual_target_tilt_db_per_oct",
                        ui.number(
                            label=t("manual_target_tilt"),
                            value=float(get_val("manual_target_tilt_db_per_oct", 0.0) or 0.0),
                            format="%.1f",
                        ).props("dense outlined step=0.1").classes("flex-1"),
                    )
                    ui.button(
                        "+",
                        on_click=lambda: _step_manual_target_tilt(+0.1),
                    ).props('color="secondary" outline').style("min-width:34px;")
                    ui.button(
                        "-",
                        on_click=lambda: _step_manual_target_tilt(-0.1),
                    ).props('color="secondary" outline').style("min-width:34px;")
                ui.label(t("lvl_manual_help")).classes("text-xs text-gray-400")
                ui.label(t("manual_target_tilt_help")).classes("text-xs text-gray-400")
                ui.label(t("lvl_manual_bias_hint")).classes("text-xs text-gray-400")
                ui.label(t("lvl_manual_drag_hint_curve")).classes("text-xs text-gray-400")
                ui.label(t("manual_target_tilt_drag_hint")).classes("text-xs text-gray-400")
            lvl_manual_col.set_visibility(False)

        _initial_mode = str(get_val("mode", "BASIC") or "BASIC").upper()
        _initial_lvl_mode = normalize_lvl_mode_value(get_val("lvl_mode", LVL_MODE_AUTO), t)
        output_tilt_col = ui.column().classes("w-full gap-1")
        ctrl.register_container("output_tilt_scope", output_tilt_col)
        with output_tilt_col:
            ui.label(t("output_tilt")).classes("text-sm font-semibold mt-1")
            with ui.column().classes("w-full gap-1"):
                ctrl.register(
                    "output_tilt_source",
                    ui.radio(
                        tr_options(t, OUTPUT_TILT_SOURCE_OPTION_LABEL_KEYS),
                        value=normalize_output_tilt_source_value(
                            get_val("output_tilt_source", OUTPUT_TILT_SOURCE_OFF),
                            t,
                        ),
                    ).classes("w-full"),
                )
            ui.label(t("output_tilt_help")).classes("text-xs text-gray-400")
        output_tilt_col.set_visibility(_initial_mode == "ADVANCED" and _initial_lvl_mode == LVL_MODE_MANUAL)


def _build_target_mag_correction_section(*, t: Callable, get_val: Callable) -> None:
    from nicegui import ui

    with section_card(title=t("magnitude_correction_limits")):
        ctrl.register(
            "mag_correct",
            ui.checkbox(t("enable_corr"), value=bool(get_val("mag_correct", True))),
        )

        with ui.row().classes("w-full gap-4"):
            ctrl.register(
                "mag_c_min",
                ui.number(
                    label=t("min_freq"),
                    value=float(get_val("mag_c_min", 10.0) or 10.0),
                    format="%.1f",
                ).props("dense outlined").classes("flex-1"),
            )
            ctrl.register(
                "mag_c_max",
                ui.number(
                    label=t("max_freq"),
                    value=float(get_val("mag_c_max", 200.0) or 200.0),
                    format="%.1f",
                ).props("dense outlined").classes("flex-1"),
            )

        ctrl.register(
            "max_boost",
            ui.number(
                label=t("max_boost"),
                value=float(get_val("max_boost", 5.0) or 5.0),
                format="%.1f",
            ).props("dense outlined").classes("w-full"),
        )
