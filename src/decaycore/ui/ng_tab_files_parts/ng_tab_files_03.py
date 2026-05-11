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

import hashlib
import logging
import os
import re
from typing import Any, Callable

logger = logging.getLogger("DecayCore")

from .. import ng_controls as ctrl
from ...app_paths import default_measurements_dir
from ...io.measurements_loader import _try_load_harmonic_sidecar, _try_load_rt60_sidecar
from ...ui_i18n import LAYOUT_MONO, LAYOUT_OPTION_LABEL_KEYS, normalize_layout_value, tr_options


















_MEASUREMENT_LIBRARY_EXTENSIONS = frozenset({".txt", ".wav"})
_MEASUREMENT_SLOT_UPLOAD_KEYS = {
    "local_path_l": "file_l",
    "local_path_r": "file_r",
    "local_path_l_main": "file_l_main",
    "local_path_r_main": "file_r_main",
    "local_path_l_sub": "file_l_sub",
    "local_path_r_sub": "file_r_sub",
}


























_SLOT_FILTER_THRESHOLD = -50.0
_SUB_SLOT_KEYS = frozenset({"local_path_l_sub", "local_path_r_sub"})
_SUB_FILENAME_PREFIXES = ("sub", "lfe", "sw")

def _score_measurement_tokens(tokens: list[str], slot_key: str, *, scale: float) -> float:
    if not tokens:
        return 0.0

    first = tokens[0]
    last = tokens[-1]
    has_main = any(token in {"main", "mainonly", "mains"} for token in tokens)
    has_leftish = any(_token_is_leftish(token) for token in tokens)
    has_rightish = any(_token_is_rightish(token) for token in tokens)
    has_subish = any(_token_is_subish(token) for token in tokens)
    has_sub1ish = any(_token_is_sub1ish(token) for token in tokens)
    has_sub2ish = any(_token_is_sub2ish(token) for token in tokens)

    score = 0.0
    if slot_key in {"local_path_l", "local_path_l_main"}:
        if "left" in tokens:
            score += 180.0 * scale
        if any(token in {"fl", "frontleft", "frontl"} or _token_has_numeric_suffix(token, "fl") for token in tokens):
            score += 140.0 * scale
        if "lmain" in tokens:
            score += 120.0 * scale
        if first == "l" or last == "l":
            score += 90.0 * scale
        if slot_key == "local_path_l_main" and has_main:
            score += 45.0 * scale
        if slot_key == "local_path_l" and has_main:
            score -= 10.0 * scale
        if has_rightish:
            score -= 160.0 * scale
        if has_subish:
            score -= 150.0 * scale
        return score

    if slot_key in {"local_path_r", "local_path_r_main"}:
        if "right" in tokens:
            score += 180.0 * scale
        if any(token in {"fr", "frontright", "frontr"} or _token_has_numeric_suffix(token, "fr") for token in tokens):
            score += 140.0 * scale
        if "rmain" in tokens:
            score += 120.0 * scale
        if first == "r" or last == "r":
            score += 90.0 * scale
        if slot_key == "local_path_r_main" and has_main:
            score += 45.0 * scale
        if slot_key == "local_path_r" and has_main:
            score -= 10.0 * scale
        if has_leftish:
            score -= 160.0 * scale
        if has_subish:
            score -= 150.0 * scale
        return score

    if slot_key == "local_path_l_sub":
        if has_subish:
            score += 140.0 * scale
        if has_sub1ish:
            score += 120.0 * scale
        if "left" in tokens or first == "l" or last == "l":
            score += 35.0 * scale
        if has_sub2ish and not has_sub1ish:
            score -= 200.0 * scale
        if "right" in tokens or first == "r" or last == "r":
            score -= 30.0 * scale
        if has_main:
            score -= 110.0 * scale
        return score

    if slot_key == "local_path_r_sub":
        if has_subish:
            score += 140.0 * scale
        if has_sub2ish:
            score += 120.0 * scale
        if "right" in tokens or first == "r" or last == "r":
            score += 35.0 * scale
        if has_sub1ish and not has_sub2ish:
            score -= 200.0 * scale
        if "left" in tokens or first == "l" or last == "l":
            score -= 30.0 * scale
        if has_main:
            score -= 110.0 * scale
        return score

    return score

def _score_measurement_candidate(entry: dict[str, Any], slot_key: str) -> float:
    name_tokens = list(entry.get("name_tokens") or [])
    path_tokens = list(entry.get("path_tokens") or [])
    return _score_measurement_tokens(name_tokens, slot_key, scale=1.0) + _score_measurement_tokens(path_tokens, slot_key, scale=0.65)

def _scan_measurement_library(path_raw: Any) -> list[dict[str, Any]]:
    root = _normalize_local_path_value(path_raw)
    if not root:
        return []
    try:
        if not os.path.isdir(root):
            return []
    except OSError:
        return []

    root_abs = os.path.abspath(root)
    _POSITION_DIR_RE = re.compile(r"^position_\d+$", re.IGNORECASE)
    entries: list[dict[str, Any]] = []
    for current_root, dirs, files in os.walk(root_abs):
        dirs[:] = [d for d in dirs if not _POSITION_DIR_RE.match(d)]
        for filename in sorted(files):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in _MEASUREMENT_LIBRARY_EXTENSIONS:
                continue
            full_path = os.path.abspath(os.path.join(current_root, filename))
            try:
                stat_result = os.stat(full_path)
                mtime_ns = int(getattr(stat_result, "st_mtime_ns", int(stat_result.st_mtime * 1_000_000_000)))
            except OSError:
                mtime_ns = 0
            relative_path = os.path.relpath(full_path, root_abs).replace("\\", "/")
            entries.append(
                {
                    "path": full_path,
                    "format": ext[1:].upper(),
                    "display_label": f"{relative_path} [{ext[1:].upper()}]",
                    "mtime_ns": mtime_ns,
                    "name_tokens": _measurement_hint_tokens(os.path.splitext(filename)[0]),
                    "path_tokens": _measurement_hint_tokens(relative_path),
                }
            )
    entries.sort(key=lambda entry: (-_measurement_entry_mtime_ns(entry), str(entry.get("display_label", "")).lower()))
    return entries

def _build_measurement_library_options(entries: list[dict[str, Any]]) -> dict[str, str]:
    return {
        str(entry.get("path", "")): str(entry.get("display_label", "") or entry.get("path", ""))
        for entry in entries
        if str(entry.get("path", "")).strip()
    }

def _entry_passes_slot_filter(entry: dict[str, Any], slot_key: str) -> bool:
    """Kova whitelist-tarkistus sub-sloteille.

    Sub-sloteissa näytetään vain tiedostot joiden ensimmäinen nimitokeni
    alkaa jollakin sub-etuliitteellä (sub, lfe, sw). Muissa sloteissa
    kaikki tiedostot läpäisevät tämän tarkistuksen.
    """
    if slot_key not in _SUB_SLOT_KEYS:
        return True
    name_tokens = entry.get("name_tokens") or []
    if not name_tokens:
        return False
    return any(name_tokens[0].startswith(prefix) for prefix in _SUB_FILENAME_PREFIXES)

def _build_slot_options(entries: list[dict[str, Any]], slot_key: str) -> dict[str, str]:
    """Palauttaa vain slotin kannalta relevantit tiedostot.

    Sub-sloteissa sovelletaan ensin whitelist-ehtoa (tiedoston nimi alkaa
    sub/lfe/sw), sen jälkeen scoring-kynnnystä väärän kanavan karsimiseksi.
    Muissa sloteissa vain scoring-kynnys.
    """
    filtered = [
        entry for entry in entries
        if _entry_passes_slot_filter(entry, slot_key)
        and _score_measurement_candidate(entry, slot_key) >= _SLOT_FILTER_THRESHOLD
    ]
    return _build_measurement_library_options(filtered)

def _suggest_measurement_library_matches(
    entries: list[dict[str, Any]],
    *,
    path_keys: list[str],
    min_score: float = 80.0,
) -> dict[str, str]:
    suggestions: dict[str, str] = {}
    used_paths: set[str] = set()

    for path_key in path_keys:
        ranked = sorted(
            (
                (
                    _score_measurement_candidate(entry, path_key),
                    _measurement_entry_mtime_ns(entry),
                    str(entry.get("path", "")),
                )
                for entry in entries
                if str(entry.get("path", "")).strip()
            ),
            key=lambda item: (-float(item[0]), -int(item[1]), item[2].lower()),
        )
        for score, _mtime_ns, candidate_path in ranked:
            if score < float(min_score) or candidate_path in used_paths:
                continue
            suggestions[path_key] = candidate_path
            used_paths.add(candidate_path)
            break
    return suggestions

def build_files_tab(*, t: Callable, get_val: Callable) -> None:
    from nicegui import ui
    mode_value = str(get_val("mode", "BASIC") or "BASIC").strip().upper()
    if bool(get_val("camillafir_automatic_mode", False)):
        mode_value = "AUTO"
    bass_integration_visible = bool(mode_value == "AUTO")
    bass_integration_enabled = bool(get_val("bass_integration_enable", False))
    bass_integration_active = bool(bass_integration_visible and bass_integration_enabled)
    is_direct_dac = (
        bass_integration_active
        and str(get_val("bass_integration_mode", "") or "").strip() == "direct_dac"
    )

    ui.markdown(f"### {t('tab_files')}")
    ui.separator()
    ui.label(t("input_files_help")).classes("text-sm text-gray-400 mb-1")

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
    slot_configs: dict[str, list[dict[str, Any]]] = {key: [] for key in file_holders}
    path_inputs: dict[str, list[Any]] = {key: [] for key in path_holders}
    library_selects: dict[str, list[Any]] = {key: [] for key in path_holders}
    library_state: dict[str, Any] = {"entries": [], "options": {}, "slot_options": {}}
    syncing_paths: set[str] = set()

    def _refresh_target_preview() -> None:
        try:
            from ..ng_tab_target import refresh_target_preview  # noqa: PLC0415

            refresh_target_preview()
        except Exception:
            logger.exception("target preview refresh from files tab")

    def _render_measurement_slots(upload_key: str) -> None:
        for slot_cfg in slot_configs.get(upload_key, []):
            _render_file_status(
                channel_label=str(slot_cfg["channel_label"]),
                holder=file_holders[upload_key],
                scope_name=str(slot_cfg["scope_name"]),
                path_holder=path_holders[str(slot_cfg["path_key"])],
                upload_key=upload_key,
            )

    for upload_key, holder in file_holders.items():
        holder.on_value_change(
            lambda _e, _upload_key=upload_key: (
                _render_measurement_slots(_upload_key),
                _refresh_target_preview(),
            )
        )

    def _set_input_value(input_control: Any, value: Any) -> None:
        try:
            input_control.set_value(value)
        except Exception:
            input_control.value = value
            input_control.update()

    def _set_options(control: Any, options: dict[str, str]) -> None:
        try:
            control.set_options(options)
        except Exception:
            control.options = options
            control.update()

    def _current_measurement_path_keys() -> list[str]:
        mode_current = str(ctrl.value("mode", mode_value) or mode_value).strip().upper()
        auto_current = bool(mode_current == "AUTO" or ctrl.value("camillafir_automatic_mode", False))
        bass_integration_current = bool(ctrl.value("bass_integration_enable", bass_integration_enabled))
        if auto_current and bass_integration_current:
            return [
                "local_path_l_main",
                "local_path_r_main",
                "local_path_l_sub",
                "local_path_r_sub",
            ]
        return ["local_path_l", "local_path_r"]

    def _sync_library_selects(path_key: str) -> None:
        guarded = False
        if path_key not in syncing_paths:
            syncing_paths.add(path_key)
            guarded = True
        try:
            current_value = _normalize_local_path_value(path_holders[path_key].value)
            slot_opts = library_state["slot_options"].get(path_key, library_state["options"])
            desired_value = current_value if current_value in slot_opts else None
            for select in library_selects[path_key]:
                if getattr(select, "value", None) != desired_value:
                    _set_input_value(select, desired_value)
        finally:
            if guarded:
                syncing_paths.discard(path_key)

    def _sync_path_value(
        *,
        path_key: str,
        upload_key: str,
        current_input: Any,
        value: Any,
        clear_upload: bool = False,
    ) -> None:
        if path_key in syncing_paths:
            return
        syncing_paths.add(path_key)
        try:
            normalized_value = _normalize_local_path_value(value)
            holder = path_holders[path_key]
            if clear_upload and normalized_value:
                file_holders[upload_key].set_value(None)
            holder.set_value(normalized_value)
            for peer in path_inputs[path_key]:
                if peer is current_input:
                    continue
                if getattr(peer, "value", None) != normalized_value:
                    _set_input_value(peer, normalized_value)
            slot_opts = library_state["slot_options"].get(path_key, library_state["options"])
            for peer in library_selects[path_key]:
                if peer is current_input:
                    continue
                desired_value = normalized_value if normalized_value in slot_opts else None
                if getattr(peer, "value", None) != desired_value:
                    _set_input_value(peer, desired_value)
            _render_measurement_slots(upload_key)
        finally:
            syncing_paths.discard(path_key)
        _refresh_target_preview()

    default_library_dir = _normalize_local_path_value(
        get_val("measurement_library_dir", str(default_measurements_dir()))
    ) or str(default_measurements_dir())

    with ui.card().classes("w-full gap-2"):
        ui.label(t("measurement_library_title")).classes("text-sm font-medium")
        ui.label(t("measurement_library_help")).classes("text-xs text-gray-400")
        with ui.row().classes("w-full items-end gap-3"):
            measurement_library_input = ctrl.register(
                "measurement_library_dir",
                ui.input(
                    label=t("measurement_library_dir"),
                    value=default_library_dir,
                ).classes("flex-1"),
            )
            use_default_library_btn = ui.button(
                t("measurement_library_use_default"),
                on_click=lambda: _use_default_measurement_library(),
            ).props("outline")
            refresh_library_btn = ui.button(
                t("measurement_library_refresh"),
                on_click=lambda: _refresh_measurement_library(),
            ).props("outline")
        with ui.row().classes("w-full items-center justify-between gap-3"):
            measurement_library_status = ui.label("").classes("text-xs text-gray-500")
            suggest_library_btn = ui.button(
                t("measurement_library_suggest"),
                on_click=lambda: _apply_measurement_library_suggestions(),
            ).props('color="primary"')

    def _refresh_measurement_library() -> None:
        dir_value = _normalize_local_path_value(measurement_library_input.value)
        library_state["entries"] = _scan_measurement_library(dir_value)
        library_state["options"] = _build_measurement_library_options(library_state["entries"])
        library_state["slot_options"] = {
            path_key: _build_slot_options(library_state["entries"], path_key)
            for path_key in library_selects
        }

        for path_key, selects in library_selects.items():
            slot_opts = library_state["slot_options"].get(path_key, library_state["options"])
            for select in selects:
                _set_options(select, slot_opts)
            _sync_library_selects(path_key)

        if not dir_value:
            measurement_library_status.set_text(t("measurement_library_status_idle"))
            return
        if not os.path.isdir(dir_value):
            measurement_library_status.set_text(t("measurement_library_status_missing"))
            return
        if library_state["entries"]:
            measurement_library_status.set_text(
                t("measurement_library_status_found").format(count=len(library_state["entries"]))
            )
            return
        measurement_library_status.set_text(t("measurement_library_status_empty"))

    def _use_default_measurement_library() -> None:
        _set_input_value(measurement_library_input, str(default_measurements_dir()))
        _refresh_measurement_library()

    def _apply_measurement_library_suggestions() -> None:
        suggestions = _suggest_measurement_library_matches(
            list(library_state["entries"]),
            path_keys=_current_measurement_path_keys(),
        )
        for path_key, suggested_path in suggestions.items():
            upload_key = _MEASUREMENT_SLOT_UPLOAD_KEYS[path_key]
            _sync_path_value(
                path_key=path_key,
                upload_key=upload_key,
                current_input=None,
                value=suggested_path,
                clear_upload=True,
            )

    measurement_library_input.on_value_change(lambda _e: _refresh_measurement_library())

    def _render_file_status(
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
        if upload_loaded:
            preview_source_text = t("file_status_preview_upload")
        elif local_path_info["entered"]:
            preview_source_text = t("file_status_preview_path")
        else:
            preview_source_text = t("file_status_preview_none")
        header_loaded = bool(upload_loaded or local_path_info["exists"])

        def _clear_uploaded_file() -> None:
            holder.set_value(None)
            _render_measurement_slots(upload_key)
            _refresh_target_preview()

        scope.clear()
        with scope:
            with ui.row().classes("w-full items-center justify-between gap-2 min-h-6"):
                if upload_loaded:
                    upload_format = _guess_upload_format(file_data)
                    if upload_format == "Unknown":
                        upload_format = t("file_status_unknown")
                    upload_size_bytes = int(file_data.get("size_bytes") or len(file_data.get("content", b"") or b""))
                    filename = str(file_data.get("filename", "") or "")
                    with ui.row().classes("items-center gap-2 text-xs overflow-hidden"):
                        ui.label("●").classes("text-green-500 shrink-0")
                        ui.label(filename).classes("truncate")
                        ui.label(f"· {upload_format} · {_format_upload_size(upload_size_bytes)}").classes("text-gray-400 shrink-0")
                    ui.button(
                        t("file_status_clear"),
                        on_click=_clear_uploaded_file,
                    ).props('color="secondary" flat size="xs"').classes("shrink-0")
                elif local_path_info["entered"]:
                    path_format = str(local_path_info["format"] or "Unknown")
                    if path_format == "Unknown":
                        path_format = t("file_status_unknown")
                    filename = local_path_info["filename"]
                    if local_path_info["exists"]:
                        size_text = _format_upload_size(local_path_info["size_bytes"])
                        with ui.row().classes("items-center gap-2 text-xs overflow-hidden"):
                            ui.label("●").classes("text-green-500 shrink-0")
                            ui.label(filename).classes("truncate")
                            ui.label(f"· {path_format} · {size_text}").classes("text-gray-400 shrink-0")
                            if bool(local_path_info.get("has_harmonics", False)):
                                ui.label("· H2–H5").classes("text-green-500 shrink-0")
                            if local_path_info.get("rt60_val", None) is not None:
                                ui.label(f"· RT60 {float(local_path_info['rt60_val']):.2f}s").classes("text-green-500 shrink-0")
                    else:
                        with ui.row().classes("items-center gap-2 text-xs overflow-hidden"):
                            ui.label("⚠").classes("text-yellow-500 shrink-0")
                            ui.label(filename).classes("truncate text-yellow-600")
                            ui.label(f"· {t('file_status_path_missing')}").classes("text-yellow-500 shrink-0")
                else:
                    ui.label("○  " + t("file_status_not_loaded")).classes("text-xs text-gray-400")

    async def _handle_upload(e, *, upload_key: str) -> None:
        file_holders[upload_key].set_value(_build_upload_payload(
            filename=e.file.name,
            content=await e.file.read(),
            mime_type=getattr(e.file, "content_type", ""),
        ))
        _render_measurement_slots(upload_key)
        _refresh_target_preview()

    def _build_measurement_slot(
        *,
        upload_key: str,
        path_key: str,
        slot_variant: str,
        channel_label_key: str,
        path_label_key: str,
    ) -> None:
        channel_label = t(channel_label_key)
        scope_name = _file_slot_scope_name(upload_key, slot_variant)
        path_control_name = _file_slot_input_name(path_key, slot_variant)

        with ui.column().classes("flex-1 gap-2"):
            ui.label(channel_label).classes("text-sm font-medium")
            library_select = ui.select(
                {},
                value=None,
                label=t("measurement_library_select"),
            ).props("clearable").classes("w-full")
            status_scope = ui.column().classes("w-full")
            ctrl.register_container(scope_name, status_scope)
            with ui.expansion(t("file_slot_manual_expand")).classes("w-full text-xs"):
                async def _on_upload(
                    e,
                    *,
                    _upload_key=upload_key,
                ) -> None:
                    await _handle_upload(
                        e,
                        upload_key=_upload_key,
                    )

                ui.upload(
                    label=channel_label,
                    on_upload=_on_upload,
                    auto_upload=True,
                ).props('accept=".txt,.wav"').classes("w-full")
                path_input = ui.input(label=t(path_label_key), value=path_holders[path_key].value).classes("w-full")
                ctrl.register(path_control_name, path_input)
        path_inputs[path_key].append(path_input)
        library_selects[path_key].append(library_select)
        slot_configs[upload_key].append(
            {
                "channel_label": channel_label,
                "scope_name": scope_name,
                "path_key": path_key,
            }
        )
        path_input.on_value_change(
            lambda e, _path_key=path_key, _upload_key=upload_key, _path_input=path_input: _sync_path_value(
                path_key=_path_key,
                upload_key=_upload_key,
                current_input=_path_input,
                value=e.value,
            )
        )
        library_select.on_value_change(
            lambda e, _path_key=path_key, _upload_key=upload_key, _library_select=library_select: _sync_path_value(
                path_key=_path_key,
                upload_key=_upload_key,
                current_input=_library_select,
                value=e.value,
                clear_upload=True,
            )
        )
        slot_opts = library_state["slot_options"].get(path_key, library_state["options"])
        _set_options(library_select, slot_opts)
        _sync_library_selects(path_key)
        _render_measurement_slots(upload_key)

    with ui.column().classes("w-full gap-4") as legacy_scope:
        with ui.row().classes("w-full gap-4"):
            _build_measurement_slot(
                upload_key="file_l",
                path_key="local_path_l",
                slot_variant="legacy",
                channel_label_key="upload_l",
                path_label_key="path_l",
            )
            _build_measurement_slot(
                upload_key="file_r",
                path_key="local_path_r",
                slot_variant="legacy",
                channel_label_key="upload_r",
                path_label_key="path_r",
            )
    ctrl.register_container("files_legacy_topology_scope", legacy_scope)
    legacy_scope.set_visibility(not bass_integration_active)

    with ui.column().classes("w-full gap-4") as bi_scope:
        ui.label(t("bass_integration_requires_wav")).classes("text-xs text-gray-400")
        ui.label(t("bass_integration_wav_format")).classes("text-xs text-gray-400")
        with ui.row().classes("w-full gap-4"):
            _build_measurement_slot(
                upload_key="file_l_main",
                path_key="local_path_l_main",
                slot_variant="bi",
                channel_label_key="upload_l_main",
                path_label_key="path_l_main",
            )
            _build_measurement_slot(
                upload_key="file_r_main",
                path_key="local_path_r_main",
                slot_variant="bi",
                channel_label_key="upload_r_main",
                path_label_key="path_r_main",
            )
        with ui.row().classes("w-full gap-4"):
            _build_measurement_slot(
                upload_key="file_l_sub",
                path_key="local_path_l_sub",
                slot_variant="bi",
                channel_label_key="upload_l_sub",
                path_label_key="path_l_sub",
            )
            _build_measurement_slot(
                upload_key="file_r_sub",
                path_key="local_path_r_sub",
                slot_variant="bi",
                channel_label_key="upload_r_sub",
                path_label_key="path_r_sub",
            )
    ctrl.register_container("files_bass_integration_topology_scope", bi_scope)
    bi_scope.set_visibility(bass_integration_active and not is_direct_dac)

    with ui.column().classes("w-full gap-4") as direct_dac_scope:
        ui.label(t("bi_direct_sub_help")).classes("text-xs text-gray-400")
        with ui.row().classes("w-full gap-4"):
            _build_measurement_slot(
                upload_key="file_l_main",
                path_key="local_path_l_main",
                slot_variant="direct",
                channel_label_key="upload_l_main",
                path_label_key="path_l_main",
            )
            _build_measurement_slot(
                upload_key="file_r_main",
                path_key="local_path_r_main",
                slot_variant="direct",
                channel_label_key="upload_r_main",
                path_label_key="path_r_main",
            )
        with ui.row().classes("w-full gap-4"):
            _build_measurement_slot(
                upload_key="file_l_sub",
                path_key="local_path_l_sub",
                slot_variant="direct",
                channel_label_key="upload_l_sub",
                path_label_key="path_l_sub",
            )
            _build_measurement_slot(
                upload_key="file_r_sub",
                path_key="local_path_r_sub",
                slot_variant="direct",
                channel_label_key="upload_r_sub",
                path_label_key="path_r_sub",
            )
    ctrl.register_container("files_direct_dac_topology_scope", direct_dac_scope)
    direct_dac_scope.set_visibility(bool(bass_integration_active and is_direct_dac))

    _refresh_measurement_library()

    ui.separator()

    # Filter layout
    with ui.row().classes("w-full gap-4 items-end"):
        with ui.column().classes("gap-1"):
            ui.label(t("layout")).classes("text-sm font-medium")
            ctrl.register(
                "layout",
                ui.radio(
                    tr_options(t, LAYOUT_OPTION_LABEL_KEYS),
                    value=normalize_layout_value(get_val("layout", LAYOUT_MONO), t),
                ),
            )

    ui.separator()

    # Checkboxes
    ctrl.register(
        "multi_rate_opt",
        ui.checkbox(
            t("multi_rate"),
            value=bool(get_val("multi_rate_opt", False)),
        ),
    )
    ctrl.register(
        "comparison_mode",
        ui.checkbox(
            t("comparison_mode"),
            value=bool(get_val("comparison_mode", True)),
        ),
    )

    # Dynamic multi-rate info container (replaces taps_auto_info_scope_files)
    ctrl.register_container("taps_auto_info_scope_files", ui.column().classes("w-full"))


__all__ = ['_score_measurement_tokens', '_score_measurement_candidate', '_scan_measurement_library', '_build_measurement_library_options', '_entry_passes_slot_filter', '_build_slot_options', '_suggest_measurement_library_matches', 'build_files_tab']


def _load_sibling_symbols() -> None:
    import importlib
    package = __package__
    for module_name in ['ng_tab_files_01', 'ng_tab_files_02', 'ng_tab_files_03']:
        if module_name == __name__.rsplit('.', 1)[-1]:
            continue
        module = importlib.import_module(f"{package}.{module_name}")
        for symbol in getattr(module, "__all__", ()):
            globals().setdefault(symbol, getattr(module, symbol))


_load_sibling_symbols()
