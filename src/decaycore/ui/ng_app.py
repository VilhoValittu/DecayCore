# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""NiceGUI app configuration for DecayCore.

Public API:
    configure_app(*, process_run, PROGRAM_NAME, VERSION, MAX_SAFE_BOOST)
    build_app(*, process_run, PROGRAM_NAME, VERSION, MAX_SAFE_BOOST)
    update_status(msg)
    update_status_notices(*, summary_text, info_text)
    update_auto_selected_bar(msg)
"""
from __future__ import annotations

import base64
import importlib
import importlib.resources as pkgres
import logging
import sys
from pathlib import Path

from ..config.decaycore_config import load_config, save_config
from ..features import has_measurement_module
from ..resources.i8n.decaycore_i18n import t
from . import ui_state
from .ng_theme import apply_theme

logger = logging.getLogger("DecayCore")

_PROCESS_RUN = None
PROGRAM_NAME = "DecayCore"
VERSION = ""
MAX_SAFE_BOOST = 12.0
_STATIC_ASSETS_REGISTERED = False


def _register_static_assets() -> None:
    """Expose packaged UI assets for browser metadata such as favicon."""
    global _STATIC_ASSETS_REGISTERED
    if _STATIC_ASSETS_REGISTERED:
        return

    from nicegui import app

    assets_dir = Path(__file__).with_name("assets")
    app.add_static_files("/static", assets_dir)
    app.add_static_file(local_file=assets_dir / "favicon.ico", url_path="/favicon.ico")
    _STATIC_ASSETS_REGISTERED = True


def _external_link_head_html() -> str:
    return (
        "<script>"
        "document.addEventListener('click',function(e){"
        "var a=e.target.closest('a[href]');"
        "if(a&&/^https?:\\/\\//i.test(a.getAttribute('href'))){"
        "e.preventDefault();e.stopPropagation();"
        "window.open(a.href,'_blank','noopener,noreferrer');"
        "}"
        "});"
        "</script>"
    )


def _load_user_manual_text() -> str:
    manual_path = _resolve_user_manual_path()
    if manual_path is None:
        logger.warning("User manual not found. Tried: %s", ", ".join(str(p) for p in _user_manual_path_candidates()))
        return "_User Manual not found._"
    try:
        return manual_path.read_text(encoding="utf-8")
    except OSError:
        logger.exception("Failed to read user manual: %s", manual_path)
        return "_User Manual not available._"


def _user_manual_path_candidates() -> list[Path]:
    candidates: list[Path] = []
    if hasattr(sys, "_MEIPASS"):
        try:
            candidates.append(Path(sys._MEIPASS) / "docs" / "User_Manual.md")  # type: ignore[attr-defined]
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
            logger.exception("Failed to resolve bundled user manual path")
    candidates.append(Path(__file__).resolve().parents[3] / "docs" / "User_Manual.md")
    return candidates


def _resolve_user_manual_path() -> Path | None:
    for path in _user_manual_path_candidates():
        try:
            if path.is_file():
                return path
        except OSError:
            logger.exception("Failed to inspect user manual path: %s", path)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def configure_app(*, process_run, PROGRAM_NAME: str, VERSION: str, MAX_SAFE_BOOST: float) -> None:
    """Wire runtime globals and register the NiceGUI main page."""
    g = globals()
    g["_PROCESS_RUN"] = process_run
    g["PROGRAM_NAME"] = PROGRAM_NAME
    g["VERSION"] = VERSION
    g["MAX_SAFE_BOOST"] = float(MAX_SAFE_BOOST)
    _register_static_assets()
    register_main_page()


def build_app(*, process_run, PROGRAM_NAME: str, VERSION: str, MAX_SAFE_BOOST: float):
    """Backward-compatible wrapper around the current NiceGUI app setup."""
    configure_app(
        process_run=process_run,
        PROGRAM_NAME=PROGRAM_NAME,
        VERSION=VERSION,
        MAX_SAFE_BOOST=MAX_SAFE_BOOST,
    )
    return lambda *a, **kw: None


def update_status(msg) -> None:
    ui_state.update_status(msg)


def update_status_notices(*, summary_text=None, info_text=None) -> None:
    ui_state.update_status_notices(summary_text=summary_text, info_text=info_text)


def update_auto_selected_bar(msg) -> None:
    ui_state.update_auto_selected_bar(msg)


def get_status_base_message(default: str = "DecayCore running") -> str:
    return ui_state.get_status_base_message(default=default)


def set_run_wall_clock_text(value) -> None:
    ui_state.set_run_wall_clock_text(value)


def get_run_wall_clock_text(default: str = "") -> str:
    return ui_state.get_run_wall_clock_text(default=default)


# ---------------------------------------------------------------------------
# Page registration
# ---------------------------------------------------------------------------

def register_main_page() -> None:
    from nicegui import ui
    from . import ng_controls

    @ui.page("/")
    def _page() -> None:
        ui.add_head_html('<link rel="icon" type="image/x-icon" href="/favicon.ico">')
        ng_controls.reset()
        d = load_config()
        theme_dark = bool(d.get("ui_theme_dark", True))
        dark_mode = apply_theme(dark=theme_dark)
        get_val = lambda k, def_v: d.get(k, def_v)
        measurement_available = has_measurement_module()

        with ui.column().classes("w-full gap-0 cf-brand-shell"):
            _build_brand_header(version=VERSION, dark_mode=dark_mode, initial_theme_dark=theme_dark)

        with ui.column().classes("w-full gap-0 cf-tabs-shell"):
            with ui.column().classes("w-full cf-tabs-shell-inner"):
                from .ng_run_section import build_global_progress_bar  # noqa: PLC0415

                build_global_progress_bar()
                with ui.tabs().classes("w-full") as tabs:
                    tab_files = ui.tab(t("tab_files"))
                    tab_measurement = ui.tab(t("tab_measurement")) if measurement_available else None
                    tab_basic = ui.tab(t("tab_basic"))
                    tab_target = ui.tab(t("tab_target"))
                    tab_advanced = ui.tab(t("tab_adv"))
                    tab_export = ui.tab(t("tab_window_tdc"))
                    tab_xo = ng_controls.register("tab_xo", ui.tab(t("tab_xo")))
                    tab_run = ui.tab(t("tab_run"))

        with ui.tab_panels(tabs, value=tab_files).classes("w-full cf-tab-panels-shell"):
            with ui.tab_panel(tab_files):
                from .ng_tab_files import build_files_tab  # noqa: PLC0415

                build_files_tab(t=t, get_val=get_val)

            if measurement_available and tab_measurement is not None:
                with ui.tab_panel(tab_measurement):
                    module_name = ".".join(("decaycore", "ui", "measurement_tab"))
                    build_measurement_tab = importlib.import_module(module_name).build_measurement_tab
                    build_measurement_tab(t=t, get_val=get_val)

            with ui.tab_panel(tab_basic):
                from .ng_tab_basic import build_basic_tab  # noqa: PLC0415

                build_basic_tab(t=t, get_val=get_val, max_safe_boost=float(MAX_SAFE_BOOST))

            with ui.tab_panel(tab_target):
                from .ng_tab_target import build_target_tab  # noqa: PLC0415

                build_target_tab(t=t, get_val=get_val)

            with ui.tab_panel(tab_advanced):
                from .ng_tab_advanced import build_advanced_tab  # noqa: PLC0415

                build_advanced_tab(t=t, get_val=get_val, max_safe_boost=float(MAX_SAFE_BOOST))

            with ui.tab_panel(tab_export):
                from .ng_tab_window import build_window_tab  # noqa: PLC0415

                build_window_tab(t=t, get_val=get_val)

            with ui.tab_panel(tab_xo):
                from .ng_tab_xo import build_xo_tab  # noqa: PLC0415

                build_xo_tab(t=t, get_val=get_val)

            with ui.tab_panel(tab_run):
                from .ng_run_section import build_run_section  # noqa: PLC0415

                build_run_section(on_start_click=_on_start_click)

        # Register cross-tab callbacks after all elements exist.
        from .ng_callbacks import register_callbacks  # noqa: PLC0415

        register_callbacks(t=t, get_val=get_val, max_safe_boost=float(MAX_SAFE_BOOST))

        # Install the toast drain timer so background-thread toasts reach the UI.
        from .ng_health import install_toast_timer  # noqa: PLC0415

        install_toast_timer()


# ---------------------------------------------------------------------------
# Start handler
# ---------------------------------------------------------------------------

def _on_start_click() -> None:
    """Called from ng_run_section in a background thread."""
    if not callable(_PROCESS_RUN):
        logger.warning("_on_start_click: _PROCESS_RUN not configured")
        return
    try:
        _PROCESS_RUN()
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
        logger.exception("process_run raised")


# ---------------------------------------------------------------------------
# Header rendering
# ---------------------------------------------------------------------------

def _persist_theme_preference(*, dark: bool) -> None:
    cfg = dict(load_config() or {})
    cfg["ui_theme_dark"] = bool(dark)
    save_config(cfg)


def _build_brand_header(*, version: str, dark_mode, initial_theme_dark: bool) -> None:
    from nicegui import ui

    ui.add_head_html(_external_link_head_html())
    manual_text = _load_user_manual_text()

    def _load_logo_b64(filename: str) -> str:
        try:
            with pkgres.files("decaycore.ui.assets").joinpath(filename).open("rb") as f:
                return base64.b64encode(f.read()).decode()
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

    logo_dark_b64 = _load_logo_b64("DecayCore_logo.png")
    logo_light_b64 = _load_logo_b64("DecayCore_logo_light.png")

    logo_img_style = "display:block; width:100%; height:100%; object-fit:cover; border-radius:inherit;"
    logo_dark_src = f"data:image/png;base64,{logo_dark_b64}" if logo_dark_b64 else ""
    logo_light_src = f"data:image/png;base64,{logo_light_b64}" if logo_light_b64 else ""

    with ui.column().classes("cf-brand-hero w-full"):
        with ui.row().classes("items-start gap-6 w-full justify-between no-wrap"):
            with ui.row().classes("items-center gap-5 min-w-0"):
                with ui.element("div").classes("cf-brand-logo-frame shrink-0"):
                    initial_logo_src = logo_dark_src if initial_theme_dark or not logo_light_src else logo_light_src
                    logo_el = ui.html(
                        f'<img id="cf-brand-logo" src="{initial_logo_src}" style="{logo_img_style}" />'
                        if initial_logo_src else ""
                    )
                with ui.column().classes("cf-brand-block gap-1 min-w-0"):
                    ui.label("DecayCore").classes("cf-brand-title")
                    ui.label(
                        t("brand_subtitle")
                    ).classes("cf-brand-subtitle")
                    ui.label(version).classes("cf-brand-version")
            from .ng_run_section import build_info_panel  # noqa: PLC0415

            with ui.column().classes("cf-brand-actions gap-3 shrink-0"):
                build_info_panel()
                with ui.row().classes("justify-end gap-2"):
                    _state = {"dark": bool(initial_theme_dark)}

                    def _toggle_theme(btn=None):
                        from nicegui import ui as _ui  # noqa: PLC0415
                        if _state["dark"]:
                            dark_mode.disable()
                            _ui.run_javascript("document.body.classList.add('cf-light')")
                            if logo_light_src:
                                logo_el.set_content(
                                    f'<img id="cf-brand-logo" src="{logo_light_src}" style="{logo_img_style}" />'
                                )
                            _toggle_btn.props("icon=light_mode")
                            _persist_theme_preference(dark=False)
                        else:
                            dark_mode.enable()
                            _ui.run_javascript("document.body.classList.remove('cf-light')")
                            if logo_dark_src:
                                logo_el.set_content(
                                    f'<img id="cf-brand-logo" src="{logo_dark_src}" style="{logo_img_style}" />'
                                )
                            _toggle_btn.props("icon=dark_mode")
                            _persist_theme_preference(dark=True)
                        _state["dark"] = not _state["dark"]

                    _toggle_btn = (
                        ui.button(icon="dark_mode" if initial_theme_dark else "light_mode", on_click=_toggle_theme)
                        .props("flat dense round")
                        .tooltip(t("theme_toggle_tooltip"))
                    )

                    async def _do_quit():
                        from nicegui import app as _nicegui_app
                        _nicegui_app.shutdown()

                    ui.button(icon="power_settings_new", on_click=_do_quit) \
                        .props("flat dense round color=negative") \
                        .tooltip(t("quit_btn_tooltip"))

        # About / guide (collapsed by default)
        with ui.expansion(t("about_title")).classes("w-full"):
            ui.markdown(t("about_body"))
            with ui.dialog() as manual_dlg, ui.card().classes("w-full max-w-3xl max-h-[80vh] cf-modal-card"):
                with ui.row().classes("w-full justify-between items-center shrink-0"):
                    ui.label("User Manual").classes("text-lg font-semibold")
                    ui.button(icon="close", on_click=manual_dlg.close).props("flat dense round")
                ui.markdown(manual_text).classes("w-full overflow-y-auto")
            ui.button(t("open_manual_btn"), on_click=manual_dlg.open).props("flat").classes("mt-2")
