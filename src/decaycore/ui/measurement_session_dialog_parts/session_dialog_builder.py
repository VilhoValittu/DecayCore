# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

from __future__ import annotations

import logging
from html import escape
from typing import Any, Callable

logger = logging.getLogger("DecayCore")

from ...measurement.models import MeasurementRequest, MeasurementSessionAggregate
from ..measurement_session_runner import MeasurementSessionRunner, MeasurementSessionUiState


def _format_session_progress_percent(completed_steps: int, total_steps: int) -> str:
    total = max(0, int(total_steps))
    completed = max(0, int(completed_steps))
    if total <= 0:
        return "0%"
    ratio = min(1.0, max(0.0, float(completed) / float(total)))
    return f"{ratio * 100.0:.0f}%"


def build_measurement_session_dialog(  # noqa: C901 - dialog builder intentionally coordinates many UI callbacks
    *,
    t: Callable[[str], str],
    build_request_for_role: Callable[[str], MeasurementRequest],
    apply_session_results: Callable[[MeasurementSessionAggregate], None],
    on_session_start: Callable[[str], None],
    on_session_complete: Callable[[MeasurementSessionAggregate], None],
    on_session_error: Callable[[str], None],
) -> Callable[[], None]:
    from nicegui import ui

    runner = MeasurementSessionRunner()
    state = MeasurementSessionUiState()

    def _safe_int(value: Any, default: int, *, minimum: int = 1, maximum: int | None = None) -> int:
        try:
            parsed = int(round(float(value)))
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
            parsed = int(default)
        parsed = max(int(minimum), parsed)
        if maximum is not None:
            parsed = min(int(maximum), parsed)
        return int(parsed)

    def _set_html_content(element, content: str) -> None:
        try:
            element.set_content(content)
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
                element.content = content
                element.update()
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
                logger.exception("html element content set")

    def _role_label(role: str) -> str:
        value = str(role or "").strip().lower()
        if value == "left":
            return "Left"
        if value == "right":
            return "Right"
        if value == "sub":
            return "Sub"
        if value == "sub1":
            return "Sub 1"
        if value == "sub2":
            return "Sub 2"
        return value.title() or "Channel"

    def _role_code(role: str) -> str:
        value = str(role or "").strip().lower()
        if value == "left":
            return "L"
        if value == "right":
            return "R"
        if value == "sub1":
            return "S1"
        if value == "sub2":
            return "S2"
        if value == "sub":
            return "S"
        return _role_label(value)

    def _selected_channels_from_inputs() -> list[str]:
        selected: list[str] = []
        if bool(left_checkbox.value):
            selected.append("left")
        if bool(right_checkbox.value):
            selected.append("right")
        if bool(sub1_checkbox.value):
            selected.append("sub1")
        if bool(sub2_checkbox.value):
            selected.append("sub2")
        return selected

    def _sequence_preview_text() -> str:
        position_count = _safe_int(position_input.value, 3, minimum=1, maximum=12)
        repeats = _safe_int(repeats_input.value, 5, minimum=1, maximum=12)
        channel_parts = [f"{_role_code(role)}x{repeats}" for role in _selected_channels_from_inputs()]
        if not channel_parts:
            return "Select at least one channel."
        return "\n".join(
            f"P{position_index}: {', '.join(channel_parts)}" for position_index in range(1, position_count + 1)
        )

    def _append_log(message: str) -> None:
        text = str(message or "").strip()
        if not text:
            return
        state.event_log.append(text)
        if len(state.event_log) > 80:
            state.event_log[:] = state.event_log[-80:]

    def _channel_summary_ref(role: str) -> dict[str, Any]:
        normalized_role = str(role or "").strip().lower()
        current = state.channel_summaries.get(normalized_role, None)
        if not isinstance(current, dict):
            current = {
                "kept": 0,
                "total": 0,
                "rejected": [],
            }
            state.channel_summaries[normalized_role] = current
        return current

    def _clear_pause_prompt() -> None:
        state.pause_pending = False
        state.pause_stage = ""
        state.pause_title = ""
        state.pause_help = ""
        state.next_position_index = None

    def _capture_step_index(position_index: int, role: str, take_index: int) -> int:
        channels = list(state.selected_channels or [])
        role_index = channels.index(role) if role in channels else 0
        return (
            ((int(position_index) - 1) * len(channels) * int(state.repeats_per_channel))
            + (role_index * int(state.repeats_per_channel))
            + int(take_index)
        )

    def _summary_html() -> str:
        if state.error_text:
            return f"<div class='text-red-600'><b>Error</b>: {escape(state.error_text)}</div>"

        lines = [
            "<div><b>Spatial combine mode</b>: magnitude average + phase/timing from primary position</div>",
            (
                f"<div><b>Primary position</b>: "
                f"{int(state.session_result.primary_position_index) if state.session_result is not None else 0}</div>"
            ),
        ]
        summary_channels = list(state.selected_channels or []) or list(state.channel_summaries.keys())
        for role in summary_channels:
            summary = state.channel_summaries.get(str(role), None)
            if not isinstance(summary, dict):
                continue
            kept = int(summary.get("kept", 0) or 0)
            total = int(summary.get("total", 0) or 0)
            rejected = list(summary.get("rejected", []) or [])
            lines.append(f"<div class='mt-2'><b>{_role_label(str(role))}</b>: kept {kept}/{total}</div>")
            if rejected:
                lines.append("<div>Rejected takes:</div>")
                for item in rejected:
                    lines.append(f"<div>{escape(str(item))}</div>")
        return "".join(lines)

    def _event_progress_values(event: dict[str, Any]) -> tuple[int, int, str]:
        position_index = int(event.get("position_index", 0) or 0)
        position_count = int(event.get("position_count", state.position_count) or 0)
        role = _role_label(str(event.get("role", "") or ""))
        return position_index, position_count, role

    def _event_to_log_line(event: dict[str, Any]) -> str:
        stage = str(event.get("stage", "") or "")
        position_index, position_count, role = _event_progress_values(event)
        fixed_messages = {
            "saving_session": "Saving measurement session to disk...",
            "session_complete": "Measurement session complete.",
        }
        if stage in fixed_messages:
            return fixed_messages[stage]
        if stage == "session_error":
            return str(event.get("error_text", "") or "Measurement session failed.")
        if stage == "building_final_average":
            return f"Building final spatial average - {role}"
        if stage == "analyzing_repeats":
            return f"Position {position_index}/{position_count} - {role} - analyzing repeats"
        if stage == "capture_take":
            take_index = int(event.get("take_index", 0) or 0)
            repeats = int(event.get("repeats_per_channel", state.repeats_per_channel) or 0)
            return f"Position {position_index}/{position_count} - {role} - Take {take_index}/{repeats}"
        if stage == "position_ready":
            kept = int(event.get("take_count_used", 0) or 0)
            total = int(event.get("take_count_total", 0) or 0)
            return f"Position {position_index}/{position_count} - {role} kept {kept}/{total}"
        if stage == "move_mic":
            next_position = int(event.get("next_position_index", 0) or 0)
            return f"Move microphone to position {next_position}/{position_count}"
        if stage == "prepare_subwoofer":
            next_role = _role_label(str(event.get("next_role", event.get("role", "sub1")) or "sub1"))
            return f"Position {position_index}/{position_count} - prepare {next_role}"
        if stage == "switch_subwoofer":
            next_role = _role_label(str(event.get("next_role", event.get("role", "sub2")) or "sub2"))
            return f"Position {position_index}/{position_count} - switch system to {next_role}"
        return stage.replace("_", " ").strip().title() or "Measurement event"

    def _reset_state() -> None:
        state.phase = "setup"
        state.is_running = False
        state.current_position = 0
        state.position_count = 0
        state.current_channel = ""
        state.current_take = 0
        state.repeats_per_channel = 0
        state.total_steps = 0
        state.completed_steps = 0
        state.status_text = ""
        _clear_pause_prompt()
        state.selected_channels = []
        state.channel_summaries = {}
        state.session_result = None
        state.event_log = []
        state.error_text = ""

    def _render() -> None:
        setup_col.set_visibility(state.phase == "setup")
        running_col.set_visibility(state.phase == "running")
        pause_col.set_visibility(state.phase == "pause")
        summary_col.set_visibility(state.phase == "summary")

        current_position_value = _safe_int(position_input.value, 3, minimum=1, maximum=12)
        if current_position_value != _safe_int(position_input.value, current_position_value, minimum=1, maximum=999):
            position_input.set_value(current_position_value)
        current_repeats_value = _safe_int(repeats_input.value, 5, minimum=1, maximum=12)
        if current_repeats_value != _safe_int(repeats_input.value, current_repeats_value, minimum=1, maximum=999):
            repeats_input.set_value(current_repeats_value)
        max_primary = current_position_value
        primary_value = _safe_int(primary_input.value, 1, minimum=1, maximum=max_primary)
        try:
            raw_primary = int(round(float(primary_input.value)))
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
            raw_primary = primary_value
        if primary_value != raw_primary:
            primary_input.set_value(primary_value)

        selected_input_channels = _selected_channels_from_inputs()
        setup_error_label.set_text("" if selected_input_channels else "Select at least one channel.")
        sequence_preview_label.set_text(_sequence_preview_text())

        status_label.set_text(state.status_text or "Preparing measurement session...")
        progress_text = _format_session_progress_percent(state.completed_steps, state.total_steps)
        progress_value_label.set_text(progress_text)
        progress_value = 0.0
        if state.total_steps > 0:
            progress_value = min(1.0, max(0.0, float(state.completed_steps) / float(state.total_steps)))
        try:
            progress_bar.set_value(progress_value)
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
            progress_bar.value = progress_value
            progress_bar.update()
        log_label.set_text("\n".join(state.event_log[-12:]) if state.event_log else "No events yet.")

        visible_counter_channels = (
            selected_input_channels or list(state.selected_channels or []) or list(counter_labels.keys())
        )
        for role, label in counter_labels.items():
            summary = state.channel_summaries.get(role, {"kept": 0, "total": 0})
            label.set_text(
                f"{_role_label(role)} kept {int(summary.get('kept', 0) or 0)}/{int(summary.get('total', 0) or 0)}"
            )
            show_label = bool(role in visible_counter_channels or int(summary.get("total", 0) or 0) > 0)
            label.set_visibility(show_label)

        pause_title.set_text(state.pause_title or "Continue when ready.")
        pause_help.set_text(state.pause_help)

        summary_title.set_text(
            "Measurement session complete." if not state.error_text else "Measurement session ended with an error."
        )
        _set_html_content(summary_html, _summary_html())
        summary_error_label.set_text(state.error_text)
        summary_error_label.set_visibility(bool(state.error_text))
        use_final_btn.set_visibility(state.session_result is not None and not state.error_text)

    def _apply_summary_to_state(summary: dict[str, Any] | None, *, role: str) -> None:
        if not isinstance(summary, dict):
            return
        rejected_items: list[str] = []
        for item in list(summary.get("rejected", []) or []):
            if isinstance(item, dict):
                position_index = int(item.get("position_index", 0) or 0)
                take_index = int(item.get("take_index", 0) or 0)
                reason = str(item.get("reason", "rejected") or "rejected")
                prefix = (
                    f"P{position_index} {_role_label(role)} take {take_index}"
                    if position_index > 0
                    else f"{_role_label(role)} take {take_index}"
                )
                rejected_items.append(f"{prefix}: {reason}")
                continue
            rejected_items.append(str(item))
        state.channel_summaries[str(role)] = {
            "kept": int(summary.get("kept", 0) or 0),
            "total": int(summary.get("total", 0) or 0),
            "rejected": rejected_items,
        }

    def _handle_capture_take_event(event: dict[str, Any]) -> None:
        position_index = int(event.get("position_index", 0) or 0)
        role = str(event.get("role", "") or "")
        take_index = int(event.get("take_index", 0) or 0)
        repeats = int(event.get("repeats_per_channel", state.repeats_per_channel) or 0)
        state.phase = "running"
        _clear_pause_prompt()
        state.current_position = position_index
        state.position_count = int(event.get("position_count", state.position_count) or state.position_count)
        state.current_channel = _role_label(role)
        state.current_take = take_index
        state.repeats_per_channel = repeats
        state.completed_steps = max(0, _capture_step_index(position_index, role, take_index) - 1)
        state.status_text = (
            f"Position {position_index}/{state.position_count} - {state.current_channel} - Take {take_index}/{repeats}"
        )

    def _handle_analyzing_repeats_event(event: dict[str, Any]) -> None:
        position_index = int(event.get("position_index", 0) or 0)
        role = str(event.get("role", "") or "")
        state.phase = "running"
        _clear_pause_prompt()
        state.completed_steps = max(
            state.completed_steps,
            _capture_step_index(position_index, role, int(state.repeats_per_channel or 1)),
        )
        state.status_text = (
            f"Position {position_index}/{state.position_count} - {_role_label(role)} - analyzing repeats"
        )

    def _handle_position_ready_event(event: dict[str, Any]) -> None:
        position_index = int(event.get("position_index", 0) or 0)
        role = str(event.get("role", "") or "")
        summary = _channel_summary_ref(role)
        kept = int(event.get("take_count_used", 0) or 0)
        total = int(event.get("take_count_total", 0) or 0)
        summary["kept"] = int(summary.get("kept", 0) or 0) + kept
        summary["total"] = int(summary.get("total", 0) or 0) + total
        rejected = list(summary.get("rejected", []) or [])
        reject_reasons = dict(event.get("reject_reasons", {}) or {})
        for take_index in list(event.get("rejected_take_indices", []) or []):
            reason = str(reject_reasons.get(str(int(take_index)), "rejected") or "rejected")
            rejected.append(f"P{position_index} {_role_label(role)} take {int(take_index)}: {reason}")
        summary["rejected"] = rejected

    def _handle_move_mic_event(event: dict[str, Any]) -> None:
        state.phase = "pause"
        state.pause_pending = True
        state.pause_stage = "move_mic"
        state.next_position_index = int(event.get("next_position_index", 0) or 0)
        state.position_count = int(event.get("position_count", state.position_count) or state.position_count)
        state.pause_title = f"Move microphone to position {state.next_position_index}/{state.position_count}"
        state.pause_help = "Place the microphone at the next listening position, shutdown subwoofer/s, keep the rig unchanged, and continue when ready."
        state.status_text = f"Move microphone to position {state.next_position_index}/{state.position_count}"

    def _handle_prepare_subwoofer_event(event: dict[str, Any]) -> None:
        position_index = int(event.get("position_index", 0) or 0)
        position_count = int(event.get("position_count", state.position_count) or state.position_count)
        next_role = _role_label(str(event.get("next_role", event.get("role", "sub1")) or "sub1"))
        state.phase = "pause"
        state.pause_pending = True
        state.pause_stage = "prepare_subwoofer"
        state.position_count = position_count
        state.current_position = position_index
        state.pause_title = f"Prepare {next_role} before measuring"
        state.pause_help = (
            f"Turn on {next_role}, make sure any other subwoofer under test is off, "
            "keep the microphone in place, and continue when ready."
        )
        state.status_text = f"Position {position_index}/{position_count} - prepare {next_role}"

    def _handle_switch_subwoofer_event(event: dict[str, Any]) -> None:
        position_index = int(event.get("position_index", 0) or 0)
        position_count = int(event.get("position_count", state.position_count) or state.position_count)
        next_role = _role_label(str(event.get("next_role", event.get("role", "sub2")) or "sub2"))
        previous_role = _role_label(str(event.get("previous_role", "sub1") or "sub1"))
        state.phase = "pause"
        state.pause_pending = True
        state.pause_stage = "switch_subwoofer"
        state.position_count = position_count
        state.current_position = position_index
        state.pause_title = f"Switch subwoofers before measuring {next_role}"
        state.pause_help = (
            f"Turn off {previous_role}, turn on {next_role}, keep the microphone in place, and continue when ready."
        )
        state.status_text = f"Position {position_index}/{position_count} - switch to {next_role}"

    def _handle_building_final_average_event(event: dict[str, Any]) -> None:
        state.phase = "running"
        _clear_pause_prompt()
        role = _role_label(str(event.get("role", "") or ""))
        state.completed_steps = max(state.completed_steps, state.total_steps - 2)
        state.status_text = f"Building final spatial average - {role}"

    def _handle_saving_session_event(_event: dict[str, Any]) -> None:
        state.phase = "running"
        _clear_pause_prompt()
        state.completed_steps = state.total_steps - 1
        state.status_text = "Saving measurement session to disk..."

    def _handle_session_complete_event(_event: dict[str, Any]) -> None:
        result = runner.get_result()
        state.phase = "summary"
        state.is_running = False
        _clear_pause_prompt()
        state.completed_steps = int(state.total_steps)
        state.status_text = "Measurement session complete."
        state.error_text = ""
        state.session_result = result
        if result is not None:
            for role in ("left", "right", "sub1", "sub2"):
                _apply_summary_to_state(result.summary.get(role, None), role=role)
            on_session_complete(result)

    def _handle_session_error_event(event: dict[str, Any]) -> None:
        state.phase = "summary"
        state.is_running = False
        _clear_pause_prompt()
        state.error_text = str(event.get("error_text", "") or "Measurement session failed.")
        state.status_text = "Measurement session failed."
        on_session_error(state.error_text)

    def _handle_event(event: dict[str, Any]) -> None:
        handlers = {
            "capture_take": _handle_capture_take_event,
            "analyzing_repeats": _handle_analyzing_repeats_event,
            "position_ready": _handle_position_ready_event,
            "move_mic": _handle_move_mic_event,
            "prepare_subwoofer": _handle_prepare_subwoofer_event,
            "switch_subwoofer": _handle_switch_subwoofer_event,
            "building_final_average": _handle_building_final_average_event,
            "saving_session": _handle_saving_session_event,
            "session_complete": _handle_session_complete_event,
            "session_error": _handle_session_error_event,
        }
        stage = str(event.get("stage", "") or "")
        handler = handlers.get(stage)
        if callable(handler):
            handler(event)
        _append_log(_event_to_log_line(event))

    def _start_session() -> None:
        selected_channels = _selected_channels_from_inputs()
        if not selected_channels:
            _render()
            return

        position_count = _safe_int(position_input.value, 3, minimum=1, maximum=12)
        repeats_per_channel = _safe_int(repeats_input.value, 5, minimum=1, maximum=12)
        primary_position_index = _safe_int(primary_input.value, 1, minimum=1, maximum=position_count)
        state.phase = "running"
        state.is_running = True
        state.current_position = 0
        state.position_count = int(position_count)
        state.current_channel = ""
        state.current_take = 0
        state.repeats_per_channel = int(repeats_per_channel)
        state.total_steps = int(position_count * repeats_per_channel * len(selected_channels)) + 2
        state.completed_steps = 0
        state.status_text = "Preparing measurement session..."
        _clear_pause_prompt()
        state.selected_channels = list(selected_channels)
        state.channel_summaries = {}
        state.session_result = None
        state.event_log = []
        state.error_text = ""
        _append_log("Starting guided measurement session.")

        try:
            runner.start(
                left_request=build_request_for_role("left") if "left" in selected_channels else None,
                right_request=build_request_for_role("right") if "right" in selected_channels else None,
                sub1_request=build_request_for_role("sub1") if "sub1" in selected_channels else None,
                sub2_request=build_request_for_role("sub2") if "sub2" in selected_channels else None,
                position_count=position_count,
                repeats_per_channel=repeats_per_channel,
                primary_position_index=primary_position_index,
                outlier_rejection_enabled=bool(outlier_checkbox.value),
                outlier_strictness=str(strictness_select.value or "normal"),
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
        ) as exc:
            state.phase = "summary"
            state.is_running = False
            state.error_text = str(exc)
            state.status_text = "Measurement session failed."
            on_session_error(state.error_text)
        else:
            on_session_start("Guided measurement session running.")
        _render()

    def _continue_after_pause() -> None:
        pause_stage = str(state.pause_stage or "")
        runner.continue_after_pause()
        state.phase = "running"
        _clear_pause_prompt()
        state.status_text = "Resuming measurement session..."
        if pause_stage == "prepare_subwoofer":
            _append_log("Continuing after subwoofer preparation.")
        elif pause_stage == "switch_subwoofer":
            _append_log("Continuing after subwoofer switch.")
        elif pause_stage == "move_mic":
            _append_log("Continuing after microphone move.")
        else:
            _append_log("Continuing guided measurement session.")
        _render()

    def _cancel_session() -> None:
        runner.cancel()
        state.status_text = "Cancelling measurement session..."
        _append_log("Cancelling measurement session.")
        _render()

    def _use_final_measurements() -> None:
        if state.session_result is None:
            return
        apply_session_results(state.session_result)
        ui.notify("Final measurements applied.", type="positive", position="top")
        _append_log("Applied final measurements to the project.")
        _render()

    def _close_dialog() -> None:
        dialog.close()
        if not runner.is_running():
            _reset_state()
            _render()

    def _open_dialog() -> None:
        if not runner.is_running() and state.phase != "summary":
            _reset_state()
        _render()
        dialog.open()

    with ui.dialog().props("persistent") as dialog, ui.card().classes("w-full max-w-3xl gap-4 cf-modal-card"):
        ui.label(t("session_guided_title")).classes("text-xl font-semibold")

        setup_col = ui.column().classes("w-full gap-4")
        with setup_col:
            ui.label(t("session_guided_desc")).classes("text-sm text-gray-500")
            with ui.row().classes("w-full gap-4"):
                position_input = (
                    ui.number(
                        label=t("session_num_positions"),
                        value=3,
                        min=1,
                        max=12,
                        step=1,
                        format="%.0f",
                    )
                    .props("dense outlined")
                    .classes("flex-1")
                )
                repeats_input = (
                    ui.number(
                        label=t("session_repeats"),
                        value=5,
                        min=1,
                        max=12,
                        step=1,
                        format="%.0f",
                    )
                    .props("dense outlined")
                    .classes("flex-1")
                )
                primary_input = (
                    ui.number(
                        label=t("session_primary_position"),
                        value=1,
                        min=1,
                        max=12,
                        step=1,
                        format="%.0f",
                    )
                    .props("dense outlined")
                    .classes("flex-1")
                )

            with ui.row().classes("w-full gap-4 items-center"):
                outlier_checkbox = ui.checkbox(t("session_enable_outlier"), value=True)
                strictness_select = (
                    ui.select(
                        options={
                            "safe": t("session_strictness_safe"),
                            "normal": t("session_strictness_normal"),
                            "strict": t("session_strictness_strict"),
                        },
                        value="normal",
                        label=t("session_outlier_strictness"),
                    )
                    .props("dense outlined")
                    .classes("w-40")
                )

            with ui.row().classes("w-full gap-6 items-center"):
                left_checkbox = ui.checkbox(t("session_measure_left"), value=True)
                right_checkbox = ui.checkbox(t("session_measure_right"), value=True)
                sub1_checkbox = ui.checkbox(t("session_measure_sub1"), value=False)
                sub2_checkbox = ui.checkbox(t("session_measure_sub2"), value=False)
            ui.label(t("session_sub_slots_note")).classes("text-xs text-gray-500")

            ui.label(t("session_sequence_preview")).classes("text-sm font-medium")
            sequence_preview_label = ui.label("").classes("whitespace-pre-wrap text-sm text-gray-500")
            setup_error_label = ui.label("").classes("text-sm text-red-500")

            with ui.row().classes("w-full gap-3 justify-end"):
                ui.button(t("manual_close_btn"), on_click=_close_dialog).props('flat color="secondary"')
                ui.button(t("session_start"), on_click=_start_session).props('unelevated color="positive"')

        running_col = ui.column().classes("w-full gap-4")
        with running_col:
            status_label = ui.label("").classes("text-2xl font-semibold")
            progress_bar = ui.linear_progress(value=0.0, size="20px", show_value=False).classes("w-full")
            with progress_bar:
                with ui.row().classes("absolute-full items-center justify-center px-3"):
                    progress_value_label = ui.label("0%").classes("text-xs text-white font-medium whitespace-nowrap")
            counter_labels: dict[str, Any] = {}
            with ui.row().classes("w-full gap-6 flex-wrap"):
                counter_labels["left"] = ui.label("Left kept 0/0").classes("text-sm")
                counter_labels["right"] = ui.label("Right kept 0/0").classes("text-sm")
                counter_labels["sub1"] = ui.label("Sub 1 kept 0/0").classes("text-sm")
                counter_labels["sub2"] = ui.label("Sub 2 kept 0/0").classes("text-sm")
            ui.label(t("session_live_event_log")).classes("text-sm font-medium")
            log_label = ui.label("No events yet.").classes("whitespace-pre-wrap text-sm text-gray-500")

        pause_col = ui.column().classes("w-full gap-4")
        with pause_col:
            pause_title = ui.label("").classes("text-2xl font-semibold")
            pause_help = ui.label("").classes("text-sm text-gray-500")
            with ui.row().classes("w-full gap-3 justify-end"):
                ui.button(t("session_cancel"), on_click=_cancel_session).props('flat color="negative"')
                ui.button(t("session_continue"), on_click=_continue_after_pause).props('unelevated color="positive"')

        summary_col = ui.column().classes("w-full gap-4")
        with summary_col:
            summary_title = ui.label("").classes("text-2xl font-semibold")
            summary_error_label = ui.label("").classes("text-sm text-red-500")
            summary_error_label.set_visibility(False)
            summary_html = ui.html("")
            with ui.row().classes("w-full gap-3 justify-end"):
                use_final_btn = ui.button(t("session_use_final"), on_click=_use_final_measurements).props(
                    'unelevated color="positive"'
                )
                ui.button(t("manual_close_btn"), on_click=_close_dialog).props('flat color="secondary"')

    for control in (
        position_input,
        repeats_input,
        primary_input,
        outlier_checkbox,
        strictness_select,
        left_checkbox,
        right_checkbox,
        sub1_checkbox,
        sub2_checkbox,
    ):
        control.on_value_change(lambda _e: _render())

    def _tick() -> None:
        updated = False
        for event in runner.poll_events():
            _handle_event(event)
            updated = True
        if updated:
            _render()

    ui.timer(0.2, _tick)
    _render()
    return _open_dialog


__all__ = ["_format_session_progress_percent", "build_measurement_session_dialog"]
