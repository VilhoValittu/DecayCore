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
import re
import threading
import time

from . import ui_state
from . import ng_run_runtime
from .ng_sections import page_shell, section_card
from .ng_timers import page_timer
from .ng_run_runtime import (
    get_progress_element,
    get_results_container,
    set_progress_visual_state,
)
from ..resources.i8n.decaycore_i18n import t

logger = logging.getLogger("DecayCore")

_run_clock: dict = {"started_at": None, "active": False, "elapsed_s": None}
_start_button_lock = threading.Lock()
_pending_start_button_enable = None
_info_panel_ref = None


def is_run_active() -> bool:
    """True while the DSP pipeline thread is running.

    Used to block maintenance actions that would delete files the running
    pipeline is still writing to.
    """
    return bool(_run_clock["active"])


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
        from .results_sections import render_results  # noqa: PLC0415

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
        sub1_ok = _has_meas("file_l_sub", "local_path_l_sub")
        sub2_ok = _has_meas("file_r_sub", "local_path_r_sub")
        bi_mode = str(value_getter("bass_integration_mode") or "direct_dac").strip().lower()
        sub2_required = bi_mode != "direct_dac"
    else:
        l_ok = _has_meas("file_l", "local_path_l")
        r_ok = _has_meas("file_r", "local_path_r")
        sub1_ok = False
        sub2_ok = False
        sub2_required = False

    mark_ok = "\u2713"
    mark_missing = "\u2013"

    parts = [
        tr("info_panel_meas"),
        f"L {mark_ok if l_ok else mark_missing}",
        f"R {mark_ok if r_ok else mark_missing}",
    ]
    if bass_integration_enabled:
        parts.append(f"{tr('info_panel_sub1')} {mark_ok if sub1_ok else mark_missing}")
        if sub2_required or sub2_ok:
            parts.append(f"{tr('info_panel_sub2')} {mark_ok if sub2_ok else mark_missing}")

    required_ready = [l_ok, r_ok]
    if bass_integration_enabled:
        required_ready.append(sub1_ok)
        if sub2_required:
            required_ready.append(sub2_ok)

    if all(required_ready):
        severity = "ok"
    elif any(required_ready):
        severity = "warn"
    else:
        severity = "dim"
    return "  ".join(parts), severity


def _should_autoscroll_auto_details(*, previous_body: str | None, current_body: str, run_active: bool) -> bool:
    if not bool(run_active):
        return False
    return bool(current_body) and str(current_body) != str(previous_body or "")


def _auto_details_autoscroll_js(element_id) -> str:
    element_key = f"c{element_id}"
    return f"""
        setTimeout(() => {{
            const sc =
                document.getElementById({element_key!r}) ||
                document.querySelector('.cf-auto-details-scroll');
            if (!sc) {{
                return;
            }}
            const previousScrollHeight = Number(sc.dataset.cfAutoDetailsScrollHeight || '0');
            const previousRemaining = previousScrollHeight > 0
                ? previousScrollHeight - sc.scrollTop - sc.clientHeight
                : 0;
            if (previousRemaining < 80) {{
                sc.scrollTop = sc.scrollHeight;
            }}
            sc.dataset.cfAutoDetailsScrollHeight = String(sc.scrollHeight);
        }}, 50);
    """


def build_global_progress_bar() -> None:
    """Render the progress bar and status area in the sticky header.

    Must be called inside the cf-top-shell column context (ng_app._build_header).
    Creates the progress bar, status labels, and the 500 ms polling timer.
    """
    from nicegui import ui
    from . import ng_bridge

    progress = ui.linear_progress(value=0.0, size="24px", show_value=False).classes("w-full")
    with progress:
        with ui.row().classes("absolute-full items-center no-wrap px-3 gap-3"):
            progress_phase_label = ui.label("").classes(
                "cf-progress-phase text-xs text-white font-medium truncate min-w-0 grow"
            )
            with ui.row().classes(
                "cf-progress-meta cf-progress-meta--running items-center no-wrap gap-3 shrink-0"
            ) as progress_meta_row:
                progress_elapsed_label = ui.label("").classes(
                    "cf-progress-meta-label text-xs text-white font-medium whitespace-nowrap"
                )
                progress_percent_label = (
                    ui.label("")
                    .classes("cf-progress-meta-label text-sm text-white font-medium whitespace-nowrap")
                    .bind_text_from(
                        progress,
                        "value",
                        backward=_format_progress_percent,
                    )
                )
    progress.visible = False
    ng_run_runtime.set_progress_elements(
        progress,
        overlay_refs=[progress_phase_label, progress_elapsed_label, progress_percent_label],
        meta_refs=[progress_meta_row],
    )

    ng_bridge.set_progress_element_getter(get_progress_element)

    with ui.column().classes("w-full gap-1"):
        info_box = ui.label("").classes("cf-status-info")
        info_box.visible = False
        auto_bar = ui.label("").classes("cf-auto-bar")
        auto_bar.visible = False
        with ui.expansion(t("run_auto_details_title")).classes("w-full text-xs") as auto_details_exp:
            auto_details_exp.visible = False
            auto_details_scroll = ui.element("div").classes("cf-auto-details-scroll")
            with auto_details_scroll:
                auto_details_label = ui.label("").classes("whitespace-pre text-xs")
    auto_details_render_state = {"body": None}

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
        previous_details = auto_details_render_state.get("body")
        details_changed = str(details) != str(previous_details or "")
        if details_changed:
            auto_details_label.set_text(details)
            auto_details_render_state["body"] = str(details)
        auto_details_exp.set_visibility(bool(details))
        if _should_autoscroll_auto_details(
            previous_body=previous_details if isinstance(previous_details, str) else None,
            current_body=str(details),
            run_active=bool(_run_clock["active"]),
        ):
            try:
                from nicegui import ui

                ui.run_javascript(_auto_details_autoscroll_js(getattr(auto_details_scroll, "id", "")))
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
                logger.debug("Failed to autoscroll auto details panel", exc_info=True)

    page_timer(0.5, _refresh_status)


def _info_fmt_fs(raw) -> str:
    try:
        hz = int(raw)
        return f"{hz // 1000} kHz" if hz % 1000 == 0 else f"{hz / 1000:.1f} kHz"
    except (TypeError, ValueError):
        return "\u2014"


def _info_fmt_grouped_int(raw) -> str:
    try:
        return f"{int(raw):,}".replace(",", "\u202f")
    except (TypeError, ValueError):
        return "\u2014"


def _info_mode_label(raw, tr) -> str:
    mode_key = str(raw or "").strip().upper()
    key = {
        "AUTO": "info_panel_mode_auto",
        "BASIC": "info_panel_mode_basic",
        "ADVANCED": "info_panel_mode_advanced",
    }.get(mode_key)
    return tr(key) if key else (mode_key or "\u2014")


def _info_filter_label(raw, tr) -> str:
    filter_key = str(raw or "").strip().lower()
    if "asym" in filter_key:
        return tr("info_panel_filter_asymmetric")
    if "linear" in filter_key:
        return tr("info_panel_filter_linear")
    if "mix" in filter_key:
        return tr("info_panel_filter_mixed")
    if "min" in filter_key:
        return tr("info_panel_filter_minimum")
    return str(raw or "\u2014")


def _info_target_label(raw, tr) -> str:
    target = str(raw or "").strip()
    harman_match = re.fullmatch(r"Harman\s*([0-9]+(?:\.[0-9]+)?)", target, re.IGNORECASE)
    if harman_match:
        amount = f"{float(harman_match.group(1)):g}"
        return tr("info_panel_harman_target").format(db=amount)
    return target or "\u2014"


def _info_target_summary(mode_raw, auto_target_mode_raw, target_raw, tr) -> str:
    app_mode = str(mode_raw or "").strip().upper()
    auto_target_mode = str(auto_target_mode_raw or "auto").strip().lower()
    if app_mode == "AUTO" and auto_target_mode == "adaptive":
        return tr("info_panel_target_adaptive")
    if app_mode == "AUTO" and auto_target_mode not in ("selected", "manual", "fixed", "user"):
        return tr("info_panel_target_auto")
    return tr("info_panel_target_value").format(target=_info_target_label(target_raw, tr))


def _info_multi_rate_label(common_enabled, ultra_high_enabled, tr) -> str | None:
    if not bool(common_enabled):
        return None
    if bool(ultra_high_enabled):
        return tr("info_panel_all_rates_ultra")
    return tr("info_panel_all_common_rates")


def _info_render_multi_rate_chip(chip, common_enabled, ultra_high_enabled, tr) -> None:
    label = _info_multi_rate_label(common_enabled, ultra_high_enabled, tr)
    chip.set_text(label or "")
    chip.set_visibility(bool(label))


def _info_set_measurement_line_style(line_meas, meas_severity: str) -> None:
    if str(meas_severity) == "ok":
        line_meas.classes(add="cf-info-line-ok", remove="cf-info-line-dim cf-info-line-warn")
        return
    if str(meas_severity) == "warn":
        line_meas.classes(add="cf-info-line-warn", remove="cf-info-line-dim cf-info-line-ok")
        return
    line_meas.classes(add="cf-info-line-dim", remove="cf-info-line-ok cf-info-line-warn")


def _info_set_readiness_style(readiness, meas_severity: str) -> None:
    if str(meas_severity) == "ok":
        readiness.classes(
            add="cf-info-readiness-ok",
            remove="cf-info-readiness-warn cf-info-readiness-dim",
        )
        return
    if str(meas_severity) == "warn":
        readiness.classes(
            add="cf-info-readiness-warn",
            remove="cf-info-readiness-ok cf-info-readiness-dim",
        )
        return
    readiness.classes(
        add="cf-info-readiness-dim",
        remove="cf-info-readiness-ok cf-info-readiness-warn",
    )


def _info_render_last_run_info(line3) -> None:
    info = ui_state.get_last_run_info()
    if not info:
        line3.set_visibility(False)
        return
    match = info.get("match")
    conf = info.get("conf")
    # The score stays in the results view and the export summary; the run header
    # reports how well the correction matched the target instead.
    parts = []
    parts.append(t("run_info_match").format(match=match) if match is not None else t("run_info_match_missing"))
    if conf is not None:
        parts.append(t("run_info_conf").format(conf=conf))
    line3.set_text(" \u00b7 ".join(parts))
    line3.classes(add="cf-info-line-score", remove="cf-info-line-dim")
    line3.set_visibility(True)


def _dock_info_panel_for_run() -> None:
    """Return the pre-run floating summary to its header slot after START."""
    panel = _info_panel_ref
    if panel is None or bool(getattr(panel, "is_deleted", False)):
        return
    panel.classes(
        add="cf-info-panel-docked cf-info-panel-run-locked",
        remove="cf-info-panel-floating",
    )


def _info_panel_float_javascript(anchor_id, panel_id) -> str:
    """Keep the panel docked at page top and float it after its header slot scrolls away."""
    anchor_key = f"c{anchor_id}"
    panel_key = f"c{panel_id}"
    return f"""
        (() => {{
            const anchor = document.getElementById({anchor_key!r});
            const panel = document.getElementById({panel_key!r});
            if (!anchor || !panel || panel.dataset.cfFloatHandlerInstalled === 'true') {{
                return;
            }}
            panel.dataset.cfFloatHandlerInstalled = 'true';
            let scheduled = false;
            const sync = () => {{
                scheduled = false;
                const locked = panel.classList.contains('cf-info-panel-run-locked');
                const anchorHasScrolledAway = anchor.getBoundingClientRect().bottom <= 104;
                const navigation = document.querySelector('.cf-tabs-shell');
                const navigationBottom = navigation
                    ? navigation.getBoundingClientRect().bottom
                    : 0;
                const floatingTop = Math.max(164, Math.ceil(navigationBottom + 16));
                panel.style.setProperty('--cf-info-panel-float-top', `${{floatingTop}}px`);
                const shouldFloat = !locked && anchorHasScrolledAway;
                panel.classList.toggle('cf-info-panel-floating', shouldFloat);
                panel.classList.toggle('cf-info-panel-docked', !shouldFloat);
            }};
            const schedule = () => {{
                if (!scheduled) {{
                    scheduled = true;
                    window.requestAnimationFrame(sync);
                }}
            }};
            window.addEventListener('scroll', schedule, {{passive: true}});
            window.addEventListener('resize', schedule, {{passive: true}});
            sync();
        }})();
    """


def _refresh_info_panel_lines(
    *,
    readiness,
    mode_chip,
    fs_chip,
    taps_chip,
    filter_chip,
    multi_rate_chip,
    target_line,
    line_meas,
    line3,
    ng_controls,
) -> None:
    mode = ng_controls.value("mode") or "\u2014"
    fs_raw = ng_controls.value("fs")
    taps_raw = ng_controls.value("taps")
    ftype = ng_controls.value("filter_type") or "\u2014"
    hc_mode = ng_controls.value("hc_mode") or ""
    auto_target_mode = ng_controls.value("auto_target_mode") or "auto"
    _info_render_multi_rate_chip(
        multi_rate_chip,
        ng_controls.value("multi_rate_opt"),
        ng_controls.value("multi_rate_ultra_high_opt"),
        t,
    )

    mode_chip.set_text(_info_mode_label(mode, t))
    fs_chip.set_text(_info_fmt_fs(fs_raw))
    taps_chip.set_text(t("info_panel_taps_value").format(taps=_info_fmt_grouped_int(taps_raw)))
    filter_chip.set_text(_info_filter_label(ftype, t))
    target_line.set_text(_info_target_summary(mode, auto_target_mode, hc_mode, t))

    meas_text, meas_severity = _build_measurement_status_line(
        bass_integration_enabled=bool(ng_controls.value("bass_integration_enable")),
        value_getter=ng_controls.value,
        tr=t,
    )
    line_meas.set_text(meas_text)
    _info_set_measurement_line_style(line_meas, str(meas_severity))
    readiness.set_text(
        {
            "ok": t("info_panel_measurements_ready"),
            "warn": t("info_panel_measurements_incomplete"),
            "dim": t("info_panel_measurements_missing"),
        }.get(str(meas_severity), t("info_panel_measurements_missing"))
    )
    _info_set_readiness_style(readiness, str(meas_severity))
    _info_render_last_run_info(line3)


def build_info_panel() -> None:
    """Build a header summary which floats only after its dock scrolls away."""
    global _info_panel_ref

    from nicegui import ui
    from . import ng_controls

    with ui.element("div").classes("cf-info-panel-slot") as info_panel_slot:
        with ui.element("div").classes("cf-info-panel cf-info-panel-docked") as info_panel:
            _info_panel_ref = info_panel
            with ui.row().classes("cf-info-panel-header"):
                ui.label(t("info_panel_title")).classes("cf-info-panel-title")
                readiness = ui.label("").classes("cf-info-readiness cf-info-readiness-dim")
            with ui.row().classes("cf-info-chip-row"):
                mode_chip = ui.label("").classes("cf-info-chip cf-info-chip-primary")
                fs_chip = ui.label("").classes("cf-info-chip")
                taps_chip = ui.label("").classes("cf-info-chip")
                filter_chip = ui.label("").classes("cf-info-chip")
                multi_rate_chip = ui.label(t("info_panel_all_common_rates")).classes("cf-info-chip cf-info-chip-option")
                multi_rate_chip.set_visibility(False)
            target_line = ui.label("").classes("cf-info-detail")
            line_meas = ui.label("").classes("cf-info-detail cf-info-line-dim")
            line3 = ui.label("")
            line3.set_visibility(False)

    def _refresh_info() -> None:
        _refresh_info_panel_lines(
            readiness=readiness,
            mode_chip=mode_chip,
            fs_chip=fs_chip,
            taps_chip=taps_chip,
            filter_chip=filter_chip,
            multi_rate_chip=multi_rate_chip,
            target_line=target_line,
            line_meas=line_meas,
            line3=line3,
            ng_controls=ng_controls,
        )

    def _install_float_behavior() -> None:
        ui.run_javascript(
            _info_panel_float_javascript(
                getattr(info_panel_slot, "id", ""),
                getattr(info_panel, "id", ""),
            )
        )

    page_timer(0.1, _install_float_behavior, once=True, immediate=False)
    page_timer(1.0, _refresh_info)


def build_run_section(*, on_start_click) -> None:
    """Build the Run tab content. Must be called inside a ui.tab_panel context."""
    from nicegui import ui

    with page_shell(title=t("tab_run"), intro=t("run_page_intro"), wide=True):
        start_btn = (
            ui.button(
                t("run_start_button"),
                on_click=lambda: _handle_start(on_start_click, start_btn, _run_clock),
            )
            .classes("w-full text-2xl font-bold tracking-widest py-4")
            .props('color="positive" unelevated')
        )

        with section_card(title=t("run_results_section_title"), hero=True):
            ng_run_runtime.set_results_container(ui.column().classes("w-full gap-4"))


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
    _dock_info_panel_for_run()
    run_clock["started_at"] = time.perf_counter()
    run_clock["elapsed_s"] = 0.0
    run_clock["active"] = True
    progress = get_progress_element()
    if progress is not None:
        progress.set_value(0.0)
        set_progress_visual_state(completed=False)
        progress.set_visibility(True)

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
