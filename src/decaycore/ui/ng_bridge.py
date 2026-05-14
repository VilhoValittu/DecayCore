# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""NiceGUI workflow bridge helpers.

The bridge keeps workflow/process-run code decoupled from NiceGUI widgets while
letting the UI layer provide progress, notifications, target loading, and
result rendering hooks.
"""
from __future__ import annotations

import logging
import threading
import time
import typing

logger = logging.getLogger("DecayCore")

from . import ui_state
from ..workflow.bridge_types import ProcessRunCallbacks, ProcessRunUiBridge


# ---------------------------------------------------------------------------
# Progress element holder
# ---------------------------------------------------------------------------

_progress_getter: typing.Callable[[], typing.Any] | None = None
_progress_lock = threading.Lock()
_pending_progress_value: float | None = None

_render_results_lock = threading.Lock()
_pending_render_results: tuple[tuple[typing.Any, ...], dict[str, typing.Any]] | None = None


def set_progress_element_getter(fn: typing.Callable[[], typing.Any]) -> None:
    """Register a callable that returns the live NiceGUI LinearProgress element."""
    global _progress_getter
    _progress_getter = fn if callable(fn) else None


def queue_progress_value(value: float) -> None:
    """Store the latest progress value for the NiceGUI UI timer to apply."""
    global _pending_progress_value
    with _progress_lock:
        _pending_progress_value = float(value)


def consume_pending_progress() -> float | None:
    """Return and clear the latest queued progress value."""
    global _pending_progress_value
    with _progress_lock:
        value = _pending_progress_value
        _pending_progress_value = None
    return value


def queue_render_results(*args: typing.Any, **kwargs: typing.Any) -> None:
    """Queue the latest results payload for rendering from the NiceGUI UI timer."""
    global _pending_render_results
    with _render_results_lock:
        _pending_render_results = (tuple(args), dict(kwargs))


def consume_pending_render_results() -> tuple[tuple[typing.Any, ...], dict[str, typing.Any]] | None:
    """Return and clear the latest queued results payload."""
    global _pending_render_results
    with _render_results_lock:
        value = _pending_render_results
        _pending_render_results = None
    return value


def _default_ensure_progress_bar() -> None:
    queue_progress_value(0.0)


def _default_set_progress(v: float) -> None:
    queue_progress_value(float(v))


# ---------------------------------------------------------------------------
# Callbacks factory
# ---------------------------------------------------------------------------

def _default_make_callbacks(run_started_at: float) -> ProcessRunCallbacks:
    def _elapsed() -> float:
        try:
            return max(0.0, float(time.perf_counter() - run_started_at))
        except Exception:
            return 0.0

    def _status(msg: str) -> None:
        try:
            ui_state.update_status(f"{msg} | {_elapsed():.1f} s")
        except Exception:
            logger.exception("status update")

    def _set_auto_selected_bar(msg: typing.Any = "") -> None:
        try:
            ui_state.update_auto_selected_bar(msg)
        except Exception:
            logger.exception("auto selected bar update")

    return ProcessRunCallbacks(status=_status, set_auto_selected_bar=_set_auto_selected_bar)


# ---------------------------------------------------------------------------
# Lazy import to avoid loading result sections before the page is assembled.
# ---------------------------------------------------------------------------

def _lazy_render_results(*args: typing.Any, **kwargs: typing.Any) -> None:
    queue_render_results(*args, **kwargs)


# ---------------------------------------------------------------------------
# Bridge factory
# ---------------------------------------------------------------------------

def build_default_ui_bridge() -> ProcessRunUiBridge:
    from .ng_health import toast_health_gate_result  # noqa: PLC0415
    from ..ui.decaycore_export import build_export_zip, save_export_bundle  # noqa: PLC0415

    return ProcessRunUiBridge(
        ensure_progress_bar=_default_ensure_progress_bar,
        set_progress=_default_set_progress,
        toast_health_gate_result=toast_health_gate_result,
        render_results=_lazy_render_results,
        build_export_zip=build_export_zip,
        save_export_bundle=save_export_bundle,
        make_callbacks=_default_make_callbacks,
    )
