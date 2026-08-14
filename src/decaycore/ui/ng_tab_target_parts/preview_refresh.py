# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Target tab preview refresh machinery: plot mount, debounced refresh, drag relayout."""

from __future__ import annotations

from .. import ng_controls as ctrl
from ..target_preview_interaction import (
    extract_target_tilt_from_shape_relayout,
    extract_vertical_shift_from_shape_relayout,
)
from ...ui_i18n import LVL_MODE_AUTO, LVL_MODE_MANUAL, normalize_lvl_mode_value
from .preview_fig import _build_target_preview_fig
from .preview_metadata import _render_target_decay_hint, _render_target_preview_metadata
from .preview_state import STATE


def refresh_target_preview() -> None:
    """Regenerate the target curve preview plot (NiceGUI version).

    Reads values from ng_controls instead of PyWebIO pin.
    """
    preview_col = ctrl.get_container("target_preview_scope")
    if preview_col is None:
        return
    _render_target_decay_hint()
    _render_target_preview_metadata()

    fig, drag_base_points, tilt_handle_points = _build_target_preview_fig()
    STATE.drag_base_points = drag_base_points
    STATE.tilt_handle_points = tilt_handle_points
    if fig is None:
        preview_col.clear()
        STATE.plot = None
        STATE.drag_active = False
        return

    if STATE.plot is None:
        STATE.plot = _mount_target_preview_plot(preview_col, fig)
    else:
        try:
            STATE.plot.update_figure(fig)
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
            STATE.plot = _mount_target_preview_plot(preview_col, fig)
    STATE.drag_active = False


def _mount_target_preview_plot(preview_col, fig):
    from nicegui import ui  # noqa: PLC0415

    preview_col.clear()
    with preview_col:
        plot = ui.plotly(fig).classes("w-full")
        plot.on("plotly_relayout", _on_target_preview_relayout)
    return plot


def _schedule_target_preview_refresh(delay_s: float = 0.10) -> None:

    preview_col = ctrl.get_container("target_preview_scope")
    if preview_col is None:
        return

    STATE.refresh_token += 1
    token = STATE.refresh_token

    from nicegui import ui  # noqa: PLC0415

    def _run() -> None:

        if token != STATE.refresh_token:
            return
        try:
            refresh_target_preview()
        finally:
            STATE.drag_active = False

    with preview_col:
        ui.timer(delay_s, _run, once=True, immediate=False)


def _on_target_preview_relayout(e) -> None:

    app_mode = str(ctrl.value("mode", "BASIC") or "BASIC").upper()
    lvl_mode = normalize_lvl_mode_value(ctrl.value("lvl_mode", LVL_MODE_AUTO))
    if app_mode in ("BASIC", "AUTO") or lvl_mode != LVL_MODE_MANUAL:
        return

    payload = getattr(e, "args", None)
    if not isinstance(payload, dict):
        return

    new_tilt = extract_target_tilt_from_shape_relayout(
        payload,
        STATE.tilt_handle_points,
    )
    if new_tilt is not None:
        STATE.drag_active = True
        ctrl.set_value("manual_target_tilt_db_per_oct", new_tilt, emit=False)
        _schedule_target_preview_refresh()
        return

    new_db = extract_vertical_shift_from_shape_relayout(
        payload,
        STATE.drag_base_points,
    )
    if new_db is None:
        return

    STATE.drag_active = True
    ctrl.set_value("lvl_manual_db", new_db, emit=False)
    _schedule_target_preview_refresh()
