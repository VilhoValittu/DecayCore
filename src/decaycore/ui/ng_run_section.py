# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""NiceGUI Run tab: START button, progress bar, status area, results container.

Replaces PyWebIO build_run_section() + the status-area rendering in app.py.

Status updates
--------------
Instead of using run_js() DOM patches, this module polls ui_state every
500 ms via a NiceGUI timer. This is simpler and avoids all threading/session
concerns.

External access
---------------
ng_bridge.py calls set_progress_element_getter to wire up the progress bar.
ng_results_sections.py calls get_results_container() to get the container
it should render results into.
"""
from __future__ import annotations

import logging
import threading
import time

from . import ui_state
from ..resources.i8n.decaycore_i18n import t

logger = logging.getLogger("DecayCore")

_results_container_ref = None
_progress_ref = None
_progress_overlay_refs = []
_run_clock: dict = {"started_at": None, "active": False, "elapsed_s": None}
_start_button_lock = threading.Lock()
_pending_start_button_enable = None


def get_results_container():
    return _results_container_ref


def get_progress_element():
    return _progress_ref


def _queue_start_button_enable(start_btn) -> None:
    global _pending_start_button_enable
    with _start_button_lock:
        _pending_start_button_enable = start_btn


def _consume_pending_start_button_enable():
    global _pending_start_button_enable
    with _start_button_lock:
        start_btn = _pending_start_button_enable
        _pending_start_button_enable = None
    return start_btn


def _drain_pending_result_render() -> None:
    from . import ng_bridge

    pending = ng_bridge.consume_pending_render_results()
    if pending is None:
        return
    args, kwargs = pending
    try:
        from .ng_results_sections import render_results  # noqa: PLC0415

        render_results(*args, **kwargs)
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
        logger.exception("queued results render failed")


def _set_progress_overlay_text_dark(enabled: bool) -> None:
    add_class = "text-black" if enabled else "text-white"
    remove_class = "text-white" if enabled else "text-black"
    for label in list(_progress_overlay_refs):
        try:
            label.classes(add=add_class, remove=remove_class)
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
            logger.debug("Failed to update progress overlay text color", exc_info=True)


def set_progress_visual_state(*, completed: bool) -> None:
    progress = get_progress_element()
    if progress is not None:
        try:
            progress.set_text_color("light-green-4" if completed else "primary")
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
            logger.debug("Failed to update progress bar color", exc_info=True)
    _set_progress_overlay_text_dark(enabled=completed)


def _format_progress_percent(value) -> str:
    try:
        pct = int(round(max(0.0, min(1.0, float(value))) * 100.0))
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
        pct = 0
    return f"{pct}%"


def _format_elapsed_clock(value) -> str:
    try:
        elapsed = max(0.0, float(value))
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
        return ""
    total_s = int(elapsed)
    hours, rem = divmod(total_s, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def _build_measurement_status_line(*, bass_integration_enabled, value_getter, tr) -> tuple[str, str]:
    def _has_meas(file_key: str, path_key: str) -> bool:
        file_val = value_getter(file_key)
        path_val = value_getter(path_key) or ""
        return bool(file_val) or bool(str(path_val).strip())

    if bass_integration_enabled:
        l_ok = _has_meas("file_l_main", "local_path_l_main")
        r_ok = _has_meas("file_r_main", "local_path_r_main")
    else:
        l_ok = _has_meas("file_l", "local_path_l")
        r_ok = _has_meas("file_r", "local_path_r")

    mark_ok = "\u2713"
    mark_missing = "\u2013"

    parts = [
        tr("info_panel_meas"),
        f"L {mark_ok if l_ok else mark_missing}",
        f"R {mark_ok if r_ok else mark_missing}",
    ]
    if _has_meas("file_l_sub", "local_path_l_sub"):
        parts.append(f"{tr('info_panel_sub1')} \u2713")
    if _has_meas("file_r_sub", "local_path_r_sub"):
        parts.append(f"{tr('info_panel_sub2')} \u2713")

    if l_ok and r_ok:
        severity = "ok"
    elif l_ok or r_ok:
        severity = "warn"
    else:
        severity = "dim"
    return "  ".join(parts), severity


def build_global_progress_bar() -> None:
    """Render the progress bar and status area in the sticky header.

    Must be called inside the cf-top-shell column context (ng_app._build_header).
    Creates the progress bar, status labels, and the 500 ms polling timer.
    """
    global _progress_ref, _progress_overlay_refs

    from nicegui import ui
    from . import ng_bridge

    progress = ui.linear_progress(value=0.0, size="24px", show_value=False).classes("w-full")
    with progress:
        with ui.row().classes("absolute-full items-center no-wrap px-3 gap-3"):
            progress_phase_label = ui.label("").classes("text-xs text-white font-medium truncate min-w-0 grow")
            with ui.row().classes("items-center no-wrap gap-3 shrink-0"):
                progress_elapsed_label = ui.label("").classes("text-xs text-white font-medium whitespace-nowrap")
                progress_percent_label = ui.label("").classes("text-sm text-white font-medium whitespace-nowrap").bind_text_from(
                    progress,
                    "value",
                    backward=_format_progress_percent,
                )
    progress.visible = False
    _progress_ref = progress
    _progress_overlay_refs = [progress_phase_label, progress_elapsed_label, progress_percent_label]
    _set_progress_overlay_text_dark(False)

    ng_bridge.set_progress_element_getter(get_progress_element)

    with ui.column().classes("w-full gap-1"):
        info_box = ui.label("").classes("cf-status-info")
        info_box.visible = False
        auto_bar = ui.label("").classes("cf-auto-bar")
        auto_bar.visible = False
        with ui.expansion(t("run_auto_details_title")).classes("w-full text-xs") as auto_details_exp:
            auto_details_exp.visible = False
            with ui.element("div").classes("cf-auto-details-scroll"):
                auto_details_label = ui.label("").classes("whitespace-pre text-xs")

    def _refresh_status() -> None:
        snap = ui_state.get_status_snapshot()

        pending_progress = ng_bridge.consume_pending_progress()
        if pending_progress is not None:
            try:
                progress.set_value(float(pending_progress))
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
                logger.exception("progress bar queued update")

        _drain_pending_result_render()

        pending_start_btn = _consume_pending_start_button_enable()
        if pending_start_btn is not None:
            try:
                pending_start_btn.enable()
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
                logger.exception("start button queued re-enable")

        phase_txt = snap.get("status_base_message", "") or ""
        progress_phase_label.set_text(phase_txt)
        if _run_clock["active"] and _run_clock["started_at"] is not None:
            elapsed_text = _format_elapsed_clock(time.perf_counter() - float(_run_clock["started_at"]))
        else:
            elapsed_text = _format_elapsed_clock(_run_clock["elapsed_s"])
        progress_elapsed_label.set_text(elapsed_text)

        info = snap.get("status_info_text", "") or ""
        info_box.set_text(info)
        info_box.set_visibility(bool(info))

        auto_txt = snap.get("auto_selected_bar_text", "") or ""
        auto_bar.set_text(auto_txt)
        auto_bar.set_visibility(bool(auto_txt))

        details = snap.get("auto_status_detail_body", "") or ""
        auto_details_label.set_text(details)
        auto_details_exp.set_visibility(bool(details))

    ui.timer(0.5, _refresh_status)


def _info_fmt_fs(raw) -> str:
    try:
        hz = int(raw)
        return f"{hz // 1000} kHz" if hz % 1000 == 0 else f"{hz / 1000:.1f} kHz"
    except (TypeError, ValueError):
        return "\u2014"


def _info_fmt_latency(taps_raw, fs_raw, ftype_str: str) -> str:
    ftype_low = str(ftype_str).lower()
    if "min" in ftype_low or "asym" in ftype_low:
        return t("health_low_latency_mode")
    if "linear" not in ftype_low:
        return "\u2014"
    try:
        ms = (int(taps_raw) / 2.0 / int(fs_raw)) * 1000.0
        return f"~{ms:.0f} ms"
    except (TypeError, ValueError, ZeroDivisionError):
        return "\u2014"


def _info_set_measurement_line_style(line_meas, meas_severity: str) -> None:
    if str(meas_severity) == "ok":
        line_meas.classes(add="cf-info-line-ok", remove="cf-info-line-dim cf-info-line-warn")
        return
    if str(meas_severity) == "warn":
        line_meas.classes(add="cf-info-line-warn", remove="cf-info-line-dim cf-info-line-ok")
        return
    line_meas.classes(add="cf-info-line-dim", remove="cf-info-line-ok cf-info-line-warn")


def _info_render_last_run_info(line3) -> None:
    info = ui_state.get_last_run_info()
    if not info:
        line3.set_visibility(False)
        return
    score = info.get("score")
    match = info.get("match")
    conf = info.get("conf")
    parts = []
    parts.append(
        t("run_info_score").format(score=score)
        if score is not None else
        t("run_info_score_missing")
    )
    parts.append(
        t("run_info_match").format(match=match)
        if match is not None else
        t("run_info_match_missing")
    )
    if conf is not None:
        parts.append(t("run_info_conf").format(conf=conf))
    line3.set_text(" \u00b7 ".join(parts))
    line3.classes(add="cf-info-line-score", remove="cf-info-line-dim")
    line3.set_visibility(True)


def _refresh_info_panel_lines(*, line1, line2, line_meas, line3, ng_controls) -> None:
    mode = ng_controls.value("mode") or "\u2014"
    fs_raw = ng_controls.value("fs")
    taps_raw = ng_controls.value("taps")
    ftype = ng_controls.value("filter_type") or "\u2014"
    hc_mode = ng_controls.value("hc_mode") or ""

    mode_str = str(mode).strip().upper()
    ftype_str = str(ftype).strip()
    taps_str = str(int(taps_raw)) if taps_raw is not None else "\u2014"
    lat_str = _info_fmt_latency(taps_raw, fs_raw, ftype_str)
    hc_str = str(hc_mode).strip() if hc_mode else "\u2014"
    line1.set_text(f"{mode_str} \u00b7 {_info_fmt_fs(fs_raw)} \u00b7 {taps_str} taps")
    line2.set_text(f"{ftype_str} \u00b7 {lat_str} \u00b7 {hc_str}")

    meas_text, meas_severity = _build_measurement_status_line(
        bass_integration_enabled=bool(ng_controls.value("bass_integration_enable")),
        value_getter=ng_controls.value,
        tr=t,
    )
    line_meas.set_text(meas_text)
    _info_set_measurement_line_style(line_meas, str(meas_severity))
    _info_render_last_run_info(line3)


def build_info_panel() -> None:
    """Compact config/score panel for the sticky header (top-right)."""
    from nicegui import ui
    from . import ng_controls

    with ui.element("div").classes("cf-info-panel"):
        line1 = ui.label("").classes("cf-info-line-dim")
        line2 = ui.label("").classes("cf-info-line-dim")
        line_meas = ui.label("").classes("cf-info-line-dim")
        line3 = ui.label("")
        line3.set_visibility(False)

    def _refresh_info() -> None:
        _refresh_info_panel_lines(
            line1=line1,
            line2=line2,
            line_meas=line_meas,
            line3=line3,
            ng_controls=ng_controls,
        )

    ui.timer(1.0, _refresh_info)


def build_run_section(*, on_start_click) -> None:
    """Build the Run tab content. Must be called inside a ui.tab_panel context."""
    global _results_container_ref

    from nicegui import ui

    with ui.column().classes("w-full gap-3"):
        start_btn = ui.button(
            t("run_start_button"),
            on_click=lambda: _handle_start(on_start_click, start_btn, _run_clock),
        ).classes("w-full text-2xl font-bold tracking-widest py-4").props(
            'color="positive" unelevated'
        )

        _results_container_ref = ui.column().classes("w-full gap-2")


def _clear_previous_run_output() -> None:
    ui_state.set_last_run_info({})
    container = get_results_container()
    if container is None:
        return
    if bool(getattr(container, "is_deleted", False)):
        logger.debug("Previous results container has been deleted; skipping clear")
        return
    try:
        container.clear()
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
        logger.debug("Failed to clear previous run output", exc_info=True)


def _handle_start(on_start_click, start_btn, run_clock) -> None:
    """Run the DSP pipeline in a background thread so the UI stays responsive."""
    start_btn.disable()
    run_clock["started_at"] = time.perf_counter()
    run_clock["elapsed_s"] = 0.0
    run_clock["active"] = True
    if _progress_ref is not None:
        _progress_ref.set_value(0.0)
        set_progress_visual_state(completed=False)
        _progress_ref.set_visibility(True)

    _clear_previous_run_output()
    ui_state.update_status(t("stat_reading"))
    ui_state.update_status_notices(summary_text="", info_text="")
    ui_state.update_auto_selected_bar("")
    ui_state.reset_auto_status_details()

    def _run() -> None:
        try:
            on_start_click()
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
        ) as exc:
            logger.exception("DSP run failed")
            try:
                ui_state.update_status(t("stat_failed").format(error=f"{type(exc).__name__}: {exc}"))
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
                logger.exception("run failure status update")
        finally:
            try:
                run_clock["elapsed_s"] = max(0.0, float(time.perf_counter() - float(run_clock["started_at"])))
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
                logger.exception("run clock update")
            run_clock["active"] = False
            _queue_start_button_enable(start_btn)

    threading.Thread(target=_run, daemon=True).start()
