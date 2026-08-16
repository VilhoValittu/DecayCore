# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""NiceGUI Files tab builder.

Replaces build_input_section() from layout_builders.py.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable

from .file_slot_helpers import (
    _build_upload_payload,
    _describe_local_path,
    _file_slot_input_name,
    _file_slot_scope_name,
    _format_upload_size,
    _guess_upload_format,
    _normalize_local_path_value,
)

logger = logging.getLogger("DecayCore")

from .. import ng_controls as ctrl
from ..ng_sections import page_shell, section_card
from ..ng_timers import page_timer
from ...config.legacy_keys import CAMILLAFIR_AUTO_MODE
from ...app_paths import default_measurements_dir
from ...ui_i18n import (
    DEVICE_AUDIO_FORMAT_OPTION_LABEL_KEYS,
    DEVICE_AUDIO_FORMAT_S32LE,
    FILTER_WAV_FORMAT_FLOAT32,
    FILTER_WAV_FORMAT_OPTION_LABEL_KEYS,
    LAYOUT_MONO,
    LAYOUT_OPTION_LABEL_KEYS,
    normalize_layout_value,
    tr_options,
)


from .measurement_library import (
    _MEASUREMENT_SLOT_UPLOAD_KEYS,
    _build_measurement_library_options,
    _build_measurement_library_refresh_payload,
    _build_measurement_library_slot_options,
    _build_measurement_library_state,
    _build_slot_options,
    _entry_passes_slot_filter,
    _measurement_library_refresh_payload_for_token,
    _measurement_library_status_key,
    _persist_measurement_library_dir,
    _scan_measurement_library,
    _score_measurement_candidate,
    _score_measurement_tokens,
    _suggest_measurement_library_matches,
    _suggest_measurement_library_matches_if_ready,
)


@dataclass
class _MeasurementLibraryState:
    """Mutable owner-specific state for asynchronous library refreshes."""

    entries: list[dict[str, Any]]
    options: dict[str, str]
    slot_options: dict[str, dict[str, str]]
    is_scanning: bool = False
    refresh_token: int = 0


@dataclass
class _FilesTabContext:
    """Shared state and behavior for the Files tab.

    Replaces the closure soup that used to live inside ``build_files_tab``.
    Each former nested helper is now a named method operating on ``self``
    state; ``build_files_tab`` keeps only orchestration and the widget tree.
    """

    t: Callable
    ui: Any
    mode_value: str
    bass_integration_enabled: bool
    file_holders: dict[str, Any]
    path_holders: dict[str, Any]
    slot_configs: dict[str, list[dict[str, Any]]]
    path_inputs: dict[str, list[Any]]
    library_selects: dict[str, list[Any]]
    library_state: _MeasurementLibraryState
    syncing_paths: set[str]
    measurement_library_input: Any | None = None
    measurement_library_status: Any | None = None
    measurement_library_suggest_button: Any | None = None

    # --- rendering ---------------------------------------------------------
    def refresh_target_preview(self) -> None:
        try:
            from ..ng_tab_target_parts import refresh_target_preview  # noqa: PLC0415

            refresh_target_preview()
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
            logger.exception("target preview refresh from files tab")

    def render_measurement_slots(self, upload_key: str) -> None:
        for slot_cfg in self.slot_configs.get(upload_key, []):
            self.render_file_status(
                channel_label=str(slot_cfg["channel_label"]),
                holder=self.file_holders[upload_key],
                scope_name=str(slot_cfg["scope_name"]),
                path_holder=self.path_holders[str(slot_cfg["path_key"])],
                upload_key=upload_key,
            )

    def render_uploaded_file_status(
        self,
        *,
        file_data: dict,
        upload_key: str,
        holder,
    ) -> None:
        upload_format = _guess_upload_format(file_data)
        if upload_format == "Unknown":
            upload_format = self.t("file_status_unknown")
        upload_size_bytes = int(file_data.get("size_bytes") or len(file_data.get("content", b"") or b""))
        filename = str(file_data.get("filename", "") or "")

        def _clear_uploaded_file() -> None:
            holder.set_value(None)
            self.render_measurement_slots(upload_key)
            self.refresh_target_preview()

        with self.ui.row().classes("items-center gap-2 text-xs overflow-hidden"):
            self.ui.label("●").classes("text-green-500 shrink-0")
            self.ui.label(filename).classes("truncate")
            self.ui.label(f"· {upload_format} · {_format_upload_size(upload_size_bytes)}").classes(
                "text-gray-400 shrink-0"
            )
        self.ui.button(
            self.t("file_status_clear"),
            on_click=_clear_uploaded_file,
        ).props(
            'color="secondary" flat size="xs"'
        ).classes("shrink-0")

    def render_local_path_status(self, *, local_path_info: dict) -> None:
        path_format = str(local_path_info["format"] or "Unknown")
        if path_format == "Unknown":
            path_format = self.t("file_status_unknown")
        filename = local_path_info["filename"]
        if local_path_info["exists"]:
            size_text = _format_upload_size(local_path_info["size_bytes"])
            with self.ui.row().classes("items-center gap-2 text-xs overflow-hidden"):
                self.ui.label("●").classes("text-green-500 shrink-0")
                self.ui.label(filename).classes("truncate")
                self.ui.label(f"· {path_format} · {size_text}").classes("text-gray-400 shrink-0")
                if bool(local_path_info.get("has_harmonics", False)):
                    self.ui.label("· H2–H5").classes("text-green-500 shrink-0")
                if local_path_info.get("rt60_val") is not None:
                    self.ui.label(f"· RT60 {float(local_path_info['rt60_val']):.2f}s").classes(
                        "text-green-500 shrink-0"
                    )
            return
        with self.ui.row().classes("items-center gap-2 text-xs overflow-hidden"):
            self.ui.label("⚠").classes("text-yellow-500 shrink-0")
            self.ui.label(filename).classes("truncate text-yellow-600")
            self.ui.label(f"· {self.t('file_status_path_missing')}").classes("text-yellow-500 shrink-0")

    def render_empty_file_status(self) -> None:
        self.ui.label("○  " + self.t("file_status_not_loaded")).classes("text-xs text-gray-400")

    def render_file_status(
        self,
        *,
        channel_label: str,
        holder,
        scope_name: str,
        path_holder,
        upload_key: str,
    ) -> None:
        scope = ctrl.get_container(scope_name)
        if scope is None:
            return

        file_data = holder.value if isinstance(holder.value, dict) else None
        upload_loaded = bool(file_data and file_data.get("content"))
        local_path_info = _describe_local_path(path_holder.value)
        scope.clear()
        with scope:
            with self.ui.row().classes("w-full items-center justify-between gap-2 min-h-6"):
                if upload_loaded:
                    self.render_uploaded_file_status(file_data=file_data or {}, upload_key=upload_key, holder=holder)
                elif local_path_info["entered"]:
                    self.render_local_path_status(local_path_info=local_path_info)
                else:
                    self.render_empty_file_status()

    # --- widget helpers ----------------------------------------------------
    def set_input_value(self, input_control: Any, value: Any) -> None:
        try:
            input_control.set_value(value)
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
            input_control.value = value
            input_control.update()

    def set_options(self, control: Any, options: dict[str, str]) -> None:
        try:
            control.set_options(options)
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
            control.options = options
            control.update()

    def set_enabled(self, control: Any, enabled: bool) -> None:
        if control is None:
            return
        try:
            if enabled:
                control.enable()
            else:
                control.disable()
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
            try:
                if enabled:
                    control.props(remove="disable")
                else:
                    control.props("disable")
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
                logger.debug("measurement library control enable failed", exc_info=True)

    # --- measurement library ----------------------------------------------
    def current_measurement_path_keys(self) -> list[str]:
        mode_current = str(ctrl.value("mode", self.mode_value) or self.mode_value).strip().upper()
        auto_current = bool(mode_current == "AUTO" or ctrl.value(CAMILLAFIR_AUTO_MODE, False))
        bass_integration_current = bool(ctrl.value("bass_integration_enable", self.bass_integration_enabled))
        if auto_current and bass_integration_current:
            return [
                "local_path_l_main",
                "local_path_r_main",
                "local_path_l_sub",
                "local_path_r_sub",
            ]
        return ["local_path_l", "local_path_r"]

    def set_measurement_library_scanning(self, is_scanning: bool) -> None:
        self.library_state.is_scanning = bool(is_scanning)
        self.set_enabled(self.measurement_library_suggest_button, not is_scanning)

    def update_measurement_library_status(
        self, *, dir_value: str, exists: bool, entry_count: int, is_scanning: bool
    ) -> None:
        if self.measurement_library_status is None:
            return
        status_key = _measurement_library_status_key(
            dir_value=dir_value,
            exists=exists,
            entry_count=entry_count,
            is_scanning=is_scanning,
        )
        if status_key == "measurement_library_status_found":
            self.measurement_library_status.set_text(self.t(status_key).format(count=entry_count))
            return
        self.measurement_library_status.set_text(self.t(status_key))

    def apply_measurement_library_state(self, payload: dict[str, Any]) -> None:
        entries = list(payload.get("entries") or [])
        options = dict(payload.get("options") or {})
        slot_options_raw = payload.get("slot_options") or {}
        slot_options = {str(path_key): dict(slot_options_raw.get(path_key) or {}) for path_key in self.library_selects}
        self.library_state.entries = entries
        self.library_state.options = options
        self.library_state.slot_options = slot_options

        for path_key, selects in self.library_selects.items():
            slot_opts = self.library_state.slot_options.get(path_key, self.library_state.options)
            for select in selects:
                self.set_options(select, slot_opts)
            self.sync_library_selects(path_key)

    def sync_library_selects(self, path_key: str) -> None:
        guarded = False
        if path_key not in self.syncing_paths:
            self.syncing_paths.add(path_key)
            guarded = True
        try:
            current_value = _normalize_local_path_value(self.path_holders[path_key].value)
            slot_opts = self.library_state.slot_options.get(path_key, self.library_state.options)
            desired_value = current_value if current_value in slot_opts else None
            for select in self.library_selects[path_key]:
                if getattr(select, "value", None) != desired_value:
                    self.set_input_value(select, desired_value)
        finally:
            if guarded:
                self.syncing_paths.discard(path_key)

    def sync_path_value(
        self,
        *,
        path_key: str,
        upload_key: str,
        current_input: Any,
        value: Any,
        clear_upload: bool = False,
    ) -> None:
        if path_key in self.syncing_paths:
            return
        self.syncing_paths.add(path_key)
        try:
            normalized_value = _normalize_local_path_value(value)
            holder = self.path_holders[path_key]
            if clear_upload and normalized_value:
                self.file_holders[upload_key].set_value(None)
            holder.set_value(normalized_value)
            for peer in self.path_inputs[path_key]:
                if peer is current_input:
                    continue
                if getattr(peer, "value", None) != normalized_value:
                    self.set_input_value(peer, normalized_value)
            slot_opts = self.library_state.slot_options.get(path_key, self.library_state.options)
            for peer in self.library_selects[path_key]:
                if peer is current_input:
                    continue
                desired_value = normalized_value if normalized_value in slot_opts else None
                if getattr(peer, "value", None) != desired_value:
                    self.set_input_value(peer, desired_value)
            self.render_measurement_slots(upload_key)
        finally:
            self.syncing_paths.discard(path_key)
        self.refresh_target_preview()

    async def refresh_measurement_library_async(self, token: int) -> None:
        if self.measurement_library_input is None:
            return
        path_keys = list(self.library_selects)
        dir_value = _normalize_local_path_value(self.measurement_library_input.value)
        self.set_measurement_library_scanning(True)
        self.apply_measurement_library_state(_build_measurement_library_state([], path_keys=path_keys))
        self.update_measurement_library_status(
            dir_value=dir_value,
            exists=False,
            entry_count=0,
            is_scanning=True,
        )
        try:
            payload = await asyncio.to_thread(
                _build_measurement_library_refresh_payload,
                dir_value,
                path_keys=path_keys,
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
            logger.exception("measurement library refresh failed")
            if token == int(self.library_state.refresh_token):
                self.set_measurement_library_scanning(False)
                self.apply_measurement_library_state(_build_measurement_library_state([], path_keys=path_keys))
                self.update_measurement_library_status(
                    dir_value=dir_value,
                    exists=False,
                    entry_count=0,
                    is_scanning=False,
                )
            return

        applied_payload = _measurement_library_refresh_payload_for_token(
            payload,
            token=token,
            current_token=int(self.library_state.refresh_token),
        )
        if applied_payload is None:
            return

        self.apply_measurement_library_state(applied_payload)
        self.set_measurement_library_scanning(False)
        self.update_measurement_library_status(
            dir_value=str(applied_payload.get("dir_value", "") or ""),
            exists=bool(applied_payload.get("exists", False)),
            entry_count=len(list(applied_payload.get("entries") or [])),
            is_scanning=False,
        )

    def schedule_measurement_library_refresh(self, delay_s: float = 0.35, *, force: bool = False) -> None:
        self.library_state.refresh_token += 1
        token = int(self.library_state.refresh_token)

        def _run() -> None:
            if token != int(self.library_state.refresh_token):
                return
            try:
                asyncio.create_task(self.refresh_measurement_library_async(token))
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
                logger.exception("measurement library refresh task scheduling failed")

        page_timer(0.0 if force else max(float(delay_s), 0.0), _run, once=True, immediate=False)

    def use_default_measurement_library(self) -> None:
        self.set_input_value(
            self.measurement_library_input,
            _persist_measurement_library_dir(str(default_measurements_dir())),
        )
        self.schedule_measurement_library_refresh(0.0, force=True)

    def persist_measurement_library_dir_from_ui(self, value: Any) -> None:
        if self.measurement_library_input is None:
            return
        persisted_value = _persist_measurement_library_dir(value)
        current_value = _normalize_local_path_value(self.measurement_library_input.value)
        if current_value != persisted_value:
            self.set_input_value(self.measurement_library_input, persisted_value)
            self.schedule_measurement_library_refresh(0.0, force=True)

    def apply_measurement_library_suggestions(self) -> None:
        suggestions = _suggest_measurement_library_matches_if_ready(
            list(self.library_state.entries),
            path_keys=self.current_measurement_path_keys(),
            is_scanning=bool(self.library_state.is_scanning),
        )
        for path_key, suggested_path in suggestions.items():
            upload_key = _MEASUREMENT_SLOT_UPLOAD_KEYS[path_key]
            self.sync_path_value(
                path_key=path_key,
                upload_key=upload_key,
                current_input=None,
                value=suggested_path,
                clear_upload=True,
            )

    # --- upload / slot building -------------------------------------------
    async def handle_upload(self, e, *, upload_key: str) -> None:
        self.file_holders[upload_key].set_value(
            _build_upload_payload(
                filename=e.file.name,
                content=await e.file.read(),
                mime_type=getattr(e.file, "content_type", ""),
            )
        )
        self.render_measurement_slots(upload_key)
        self.refresh_target_preview()

    def build_measurement_slot(
        self,
        *,
        upload_key: str,
        path_key: str,
        slot_variant: str,
        channel_label_key: str,
        path_label_key: str,
    ) -> None:
        channel_label = self.t(channel_label_key)
        scope_name = _file_slot_scope_name(upload_key, slot_variant)
        path_control_name = _file_slot_input_name(path_key, slot_variant)

        with self.ui.column().classes("flex-1 gap-2"):
            self.ui.label(channel_label).classes("text-sm font-medium")
            library_select = (
                self.ui.select(
                    {},
                    value=None,
                    label=self.t("measurement_library_select"),
                )
                .props("clearable")
                .classes("w-full")
            )
            status_scope = self.ui.column().classes("w-full")
            ctrl.register_container(scope_name, status_scope)
            with self.ui.expansion(self.t("file_slot_manual_expand")).classes("w-full text-xs"):

                async def _on_upload(
                    e,
                    *,
                    _upload_key=upload_key,
                ) -> None:
                    await self.handle_upload(
                        e,
                        upload_key=_upload_key,
                    )

                self.ui.upload(
                    label=channel_label,
                    on_upload=_on_upload,
                    auto_upload=True,
                ).props(
                    'accept=".txt,.wav"'
                ).classes("w-full")
                path_input = self.ui.input(
                    label=self.t(path_label_key), value=self.path_holders[path_key].value
                ).classes("w-full")
                ctrl.register(path_control_name, path_input)
        self.path_inputs[path_key].append(path_input)
        self.library_selects[path_key].append(library_select)
        self.slot_configs[upload_key].append(
            {
                "channel_label": channel_label,
                "scope_name": scope_name,
                "path_key": path_key,
            }
        )
        path_input.on_value_change(
            lambda e, _path_key=path_key, _upload_key=upload_key, _path_input=path_input: self.sync_path_value(
                path_key=_path_key,
                upload_key=_upload_key,
                current_input=_path_input,
                value=e.value,
            )
        )
        library_select.on_value_change(
            lambda e, _path_key=path_key, _upload_key=upload_key, _library_select=library_select: self.sync_path_value(
                path_key=_path_key,
                upload_key=_upload_key,
                current_input=_library_select,
                value=e.value,
                clear_upload=True,
            )
        )
        slot_opts = self.library_state.slot_options.get(path_key, self.library_state.options)
        self.set_options(library_select, slot_opts)
        self.sync_library_selects(path_key)
        self.render_measurement_slots(upload_key)


def build_files_tab(*, t: Callable, get_val: Callable) -> _FilesTabContext:
    from nicegui import ui

    mode_value = str(get_val("mode", "BASIC") or "BASIC").strip().upper()
    if bool(get_val(CAMILLAFIR_AUTO_MODE, False)):
        mode_value = "AUTO"
    bass_integration_visible = bool(mode_value == "AUTO")
    bass_integration_enabled = bool(get_val("bass_integration_enable", False))
    bass_integration_active = bool(bass_integration_visible and bass_integration_enabled)
    is_direct_dac = bass_integration_active and str(get_val("bass_integration_mode", "") or "").strip() == "direct_dac"

    # File uploads – store as {"filename": ..., "content": bytes} in holders
    file_holders = {
        "file_l": ctrl._ValueHolder(get_val("file_l", None)),
        "file_r": ctrl._ValueHolder(get_val("file_r", None)),
        "file_l_main": ctrl._ValueHolder(get_val("file_l_main", None)),
        "file_r_main": ctrl._ValueHolder(get_val("file_r_main", None)),
        "file_l_sub": ctrl._ValueHolder(get_val("file_l_sub", None)),
        "file_r_sub": ctrl._ValueHolder(get_val("file_r_sub", None)),
    }
    for key, holder in file_holders.items():
        ctrl.register(key, holder)
    path_holders = {
        "local_path_l": ctrl._ValueHolder(get_val("local_path_l", "")),
        "local_path_r": ctrl._ValueHolder(get_val("local_path_r", "")),
        "local_path_l_main": ctrl._ValueHolder(get_val("local_path_l_main", "")),
        "local_path_r_main": ctrl._ValueHolder(get_val("local_path_r_main", "")),
        "local_path_l_sub": ctrl._ValueHolder(get_val("local_path_l_sub", "")),
        "local_path_r_sub": ctrl._ValueHolder(get_val("local_path_r_sub", "")),
    }
    for key, holder in path_holders.items():
        ctrl.register(key, holder)
    ctrl.register(
        "comparison_mode",
        ctrl._ValueHolder(bool(get_val("comparison_mode", True))),
    )

    ctx = _FilesTabContext(
        t=t,
        ui=ui,
        mode_value=mode_value,
        bass_integration_enabled=bass_integration_enabled,
        file_holders=file_holders,
        path_holders=path_holders,
        slot_configs={key: [] for key in file_holders},
        path_inputs={key: [] for key in path_holders},
        library_selects={key: [] for key in path_holders},
        library_state=_MeasurementLibraryState(entries=[], options={}, slot_options={}),
        syncing_paths=set(),
    )

    for upload_key, holder in file_holders.items():
        holder.on_value_change(
            lambda _e, _upload_key=upload_key: (
                ctx.render_measurement_slots(_upload_key),
                ctx.refresh_target_preview(),
            )
        )

    default_library_dir = _normalize_local_path_value(
        get_val("measurement_library_dir", str(default_measurements_dir()))
    ) or str(default_measurements_dir())

    with page_shell(title=t("tab_files"), intro=t("files_page_intro")):
        with section_card(title=t("input_files_title"), intro=t("input_files_help")):
            with ui.column().classes("w-full gap-3"):
                ui.label(t("measurement_library_title")).classes("text-sm font-medium")
                ui.label(t("measurement_library_help")).classes("text-xs text-gray-400")
                with ui.row().classes("w-full items-end gap-3"):
                    ctx.measurement_library_input = ctrl.register(
                        "measurement_library_dir",
                        ui.input(
                            label=t("measurement_library_dir"),
                            value=default_library_dir,
                        ).classes("flex-1"),
                    )
                    ui.button(
                        t("measurement_library_use_default"),
                        on_click=lambda: ctx.use_default_measurement_library(),
                    ).props("outline")
                    ui.button(
                        t("measurement_library_refresh"),
                        on_click=lambda: ctx.schedule_measurement_library_refresh(0.0, force=True),
                    ).props("outline")
                with ui.row().classes("w-full items-center justify-between gap-3"):
                    ctx.measurement_library_status = ui.label("").classes("text-xs text-gray-500")
                    ctx.measurement_library_suggest_button = ui.button(
                        t("measurement_library_suggest"),
                        on_click=lambda: ctx.apply_measurement_library_suggestions(),
                    ).props('color="primary"')
                ctx.measurement_library_input.on_value_change(lambda _e: ctx.schedule_measurement_library_refresh(0.35))
                ctrl.on_commit(
                    "measurement_library_dir",
                    ctx.persist_measurement_library_dir_from_ui,
                )

            with ui.column().classes("w-full gap-4") as legacy_scope:
                with ui.row().classes("w-full gap-4"):
                    ctx.build_measurement_slot(
                        upload_key="file_l",
                        path_key="local_path_l",
                        slot_variant="legacy",
                        channel_label_key="upload_l",
                        path_label_key="path_l",
                    )
                    ctx.build_measurement_slot(
                        upload_key="file_r",
                        path_key="local_path_r",
                        slot_variant="legacy",
                        channel_label_key="upload_r",
                        path_label_key="path_r",
                    )
            ctrl.register_container("files_legacy_topology_scope", legacy_scope)
            legacy_scope.set_visibility(not bass_integration_active)

            with ui.column().classes("w-full gap-4") as direct_dac_scope:
                with ui.row().classes("w-full gap-4"):
                    ctx.build_measurement_slot(
                        upload_key="file_l_main",
                        path_key="local_path_l_main",
                        slot_variant="direct",
                        channel_label_key="upload_l_main",
                        path_label_key="path_l_main",
                    )
                    ctx.build_measurement_slot(
                        upload_key="file_r_main",
                        path_key="local_path_r_main",
                        slot_variant="direct",
                        channel_label_key="upload_r_main",
                        path_label_key="path_r_main",
                    )
            ctrl.register_container("files_direct_dac_topology_scope", direct_dac_scope)
            direct_dac_scope.set_visibility(bool(bass_integration_active and is_direct_dac))

            ctx.schedule_measurement_library_refresh(0.0, force=True)

        with ui.expansion(t("files_export_compact_title")).classes("w-full"):
            ui.label(t("files_export_section_intro")).classes("text-xs text-gray-400 mb-2")
            with ui.grid(columns=2).classes("w-full gap-4"):
                with ui.card().classes("w-full gap-3").props("flat bordered"):
                    ui.label(t("files_export_layout_title")).classes("text-sm font-semibold")
                    ui.label(t("files_export_layout_help")).classes("text-xs text-gray-400")
                    ctrl.register(
                        "layout",
                        ui.radio(
                            tr_options(t, LAYOUT_OPTION_LABEL_KEYS),
                            value=normalize_layout_value(get_val("layout", LAYOUT_MONO), t),
                        ).props("inline"),
                    )

                with ui.card().classes("w-full gap-3").props("flat bordered"):
                    ui.label(t("files_export_file_formats_title")).classes("text-sm font-semibold")
                    with ui.column().classes("w-full gap-1"):
                        ui.label(t("filter_wav_format")).classes("text-sm font-medium")
                        ui.label(t("filter_wav_format_help")).classes("text-xs text-gray-400")
                        ctrl.register(
                            "filter_wav_format",
                            ui.radio(
                                tr_options(t, FILTER_WAV_FORMAT_OPTION_LABEL_KEYS),
                                value=str(
                                    get_val("filter_wav_format", FILTER_WAV_FORMAT_FLOAT32) or FILTER_WAV_FORMAT_FLOAT32
                                ),
                            ).props("inline"),
                        )
                    ui.separator()
                    with ui.column().classes("w-full gap-1"):
                        ui.label(t("device_audio_format")).classes("text-sm font-medium")
                        ui.label(t("device_audio_format_help")).classes("text-xs text-gray-400")
                        ctrl.register(
                            "device_audio_format",
                            ui.radio(
                                tr_options(t, DEVICE_AUDIO_FORMAT_OPTION_LABEL_KEYS),
                                value=str(
                                    get_val("device_audio_format", DEVICE_AUDIO_FORMAT_S32LE)
                                    or DEVICE_AUDIO_FORMAT_S32LE
                                ),
                            ).props("inline"),
                        )

            with ui.card().classes("w-full gap-3 mt-4").props("flat bordered"):
                ui.label(t("files_export_rates_title")).classes("text-sm font-semibold")
                ui.label(t("files_export_rates_help")).classes("text-xs text-gray-400")
                ctrl.register(
                    "multi_rate_opt",
                    ui.checkbox(
                        t("multi_rate"),
                        value=bool(get_val("multi_rate_opt", False)),
                    ),
                )
                with ctrl.register_container("multi_rate_ultra_high_scope", ui.column().classes("w-full pl-6")):
                    ctrl.register(
                        "multi_rate_ultra_high_opt",
                        ui.checkbox(
                            t("multi_rate_ultra_high"),
                            value=bool(get_val("multi_rate_ultra_high_opt", False)),
                        ),
                    )
                with ui.expansion(t("files_export_tap_details_title")).classes("w-full"):
                    ctrl.register_container("taps_auto_info_scope_files", ui.column().classes("w-full"))

    return ctx


__all__ = [
    "_persist_measurement_library_dir",
    "_score_measurement_tokens",
    "_score_measurement_candidate",
    "_scan_measurement_library",
    "_build_measurement_library_options",
    "_entry_passes_slot_filter",
    "_build_slot_options",
    "_suggest_measurement_library_matches",
    "_build_measurement_library_slot_options",
    "_build_measurement_library_state",
    "_build_measurement_library_refresh_payload",
    "_measurement_library_refresh_payload_for_token",
    "_measurement_library_status_key",
    "_suggest_measurement_library_matches_if_ready",
    "build_files_tab",
]
