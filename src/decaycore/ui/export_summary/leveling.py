# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

import logging

logger = logging.getLogger(__name__)

_RECOVERABLE_LEVELING_EXCEPTIONS = (
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
)


def _append_leveling_numeric_line(
    lines: list[str],
    *,
    label: str,
    value,
    fmt: str,
    log_msg: str,
) -> None:
    try:
        lines.append(f"{label}: {format(float(value or 0.0), fmt)}\n")
    except _RECOVERABLE_LEVELING_EXCEPTIONS:
        logger.exception(log_msg)


def _append_leveling_window_line(lines: list[str], window_value) -> None:
    if not (isinstance(window_value, (list, tuple)) and len(window_value) >= 2):
        return
    try:
        lo = float(window_value[0])
        hi = float(window_value[1])
        lines.append(f"Window: {lo:.0f}-{hi:.0f} Hz\n")
    except _RECOVERABLE_LEVELING_EXCEPTIONS:
        logger.exception("leveling window range format")


def _append_leveling_tilt_lines(lines: list[str], tilt_value) -> None:
    if tilt_value is None:
        return
    try:
        tilt_f = float(tilt_value)
    except _RECOVERABLE_LEVELING_EXCEPTIONS:
        logger.exception("leveling tilt slope format")
        return
    lines.append(f"Tilt slope: {tilt_f:+.2f} dB/oct\n")
    if abs(tilt_f) > 1.5:
        lines.append(
            "Warning: Large broadband tilt detected. " "May indicate measurement/target mismatch or strong room tilt.\n"
        )


def _append_leveling_side(lines: list[str], side: str, state: dict | None) -> None:
    if not isinstance(state, dict):
        return
    lines.append(f"[{side}]\n")
    lines.append(f"Method: {state.get('offset_method', 'n/a')}\n")
    _append_leveling_window_line(lines, state.get("smart_scan_range", None))
    _append_leveling_numeric_line(
        lines,
        label="Offset to measurement",
        value=state.get("offset_db", 0.0),
        fmt="+.2f",
        log_msg="leveling offset_db format",
    )
    _append_leveling_numeric_line(
        lines,
        label="Effective target level",
        value=state.get("eff_target_db", 0.0),
        fmt=".2f",
        log_msg="leveling eff_target_db format",
    )
    _append_leveling_tilt_lines(lines, state.get("tilt_slope_db_per_oct", None))
    lines.append("\n")


def _append_leveling_summary(
    summary_content: str,
    l_st: dict | None,
    r_st: dict | None,
) -> str:
    lines: list[str] = [summary_content, "\n=== LEVELING ===\n"]
    _append_leveling_side(lines, "LEFT", l_st)
    _append_leveling_side(lines, "RIGHT", r_st)
    return "".join(lines)


__all__ = ["_append_leveling_summary"]
