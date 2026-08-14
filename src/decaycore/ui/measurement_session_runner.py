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

from dataclasses import dataclass, field
import queue
import threading
from typing import Any, Callable, Literal

from ..measurement.models import MeasurementRequest, MeasurementSessionAggregate
from ..measurement.workflow_parts import run_measurement_session


@dataclass(slots=True)
class MeasurementSessionUiState:
    phase: Literal["setup", "running", "pause", "summary"] = "setup"
    is_running: bool = False
    current_position: int = 0
    position_count: int = 0
    current_channel: str = ""
    current_take: int = 0
    repeats_per_channel: int = 0
    total_steps: int = 0
    completed_steps: int = 0
    status_text: str = ""
    pause_pending: bool = False
    pause_stage: str = ""
    pause_title: str = ""
    pause_help: str = ""
    next_position_index: int | None = None
    selected_channels: list[str] = field(default_factory=list)
    channel_summaries: dict[str, dict[str, Any]] = field(default_factory=dict)
    session_result: MeasurementSessionAggregate | None = None
    event_log: list[str] = field(default_factory=list)
    error_text: str = ""


class MeasurementSessionRunner:
    def __init__(
        self,
        *,
        run_session_fn: Callable[..., MeasurementSessionAggregate] = run_measurement_session,
    ) -> None:
        self._run_session_fn = run_session_fn
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._pause_gate = threading.Event()
        self._cancel_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._result: MeasurementSessionAggregate | None = None
        self._error_text: str = ""

    def is_running(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    def get_result(self) -> MeasurementSessionAggregate | None:
        with self._lock:
            return self._result

    def get_error_text(self) -> str:
        with self._lock:
            return self._error_text

    def start(
        self,
        *,
        left_request: MeasurementRequest | None,
        right_request: MeasurementRequest | None,
        sub1_request: MeasurementRequest | None = None,
        sub2_request: MeasurementRequest | None = None,
        position_count: int,
        repeats_per_channel: int,
        primary_position_index: int,
        outlier_rejection_enabled: bool,
        outlier_strictness: str,
    ) -> None:
        if self.is_running():
            raise RuntimeError("Measurement session is already running.")

        with self._lock:
            self._result = None
            self._error_text = ""
        self._cancel_event.clear()
        self._pause_gate.set()
        self._queue = queue.Queue()

        def _progress_callback(payload: dict[str, Any]) -> None:
            if self._cancel_event.is_set():
                raise RuntimeError("Measurement session cancelled.")

            event = dict(payload or {})
            self._queue.put(event)

            if str(event.get("stage", "") or "") in {"move_mic", "prepare_subwoofer", "switch_subwoofer"}:
                self._pause_gate.clear()
                while not self._cancel_event.is_set():
                    if self._pause_gate.wait(timeout=0.1):
                        break
                if self._cancel_event.is_set():
                    raise RuntimeError("Measurement session cancelled.")

        def _run() -> None:
            try:
                result = self._run_session_fn(
                    left_request=left_request,
                    right_request=right_request,
                    sub1_request=sub1_request,
                    sub2_request=sub2_request,
                    position_count=int(position_count),
                    repeats_per_channel=int(repeats_per_channel),
                    primary_position_index=int(primary_position_index),
                    outlier_rejection_enabled=bool(outlier_rejection_enabled),
                    outlier_strictness=str(outlier_strictness),
                    progress_callback=_progress_callback,
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
                with self._lock:
                    self._result = None
                    self._error_text = str(exc or "Measurement session failed.")
                self._queue.put(
                    {
                        "stage": "session_error",
                        "error_text": str(exc or "Measurement session failed."),
                    }
                )
                return

            with self._lock:
                self._result = result
                self._error_text = ""
            self._queue.put(
                {
                    "stage": "session_complete",
                    "summary": dict(result.summary or {}),
                }
            )

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def poll_events(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        while True:
            try:
                items.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return items

    def continue_after_pause(self) -> None:
        self._pause_gate.set()

    def continue_after_move_mic(self) -> None:
        self.continue_after_pause()

    def cancel(self) -> None:
        self._cancel_event.set()
        self._pause_gate.set()
