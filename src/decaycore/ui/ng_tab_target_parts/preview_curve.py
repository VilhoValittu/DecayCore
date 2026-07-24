# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Shared target-preview curve resolution and TXT serialization."""
from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import re
from typing import Any, Callable

import numpy as np

from .. import ng_controls as ctrl
from ..target_preview_common import apply_manual_target_preview_adjustments
from ...application.house_curve_service import (
    _normalize_hc_mode_key,
    load_house_curve,
    load_target_curve,
)
from ...ui_i18n import LVL_MODE_AUTO, LVL_MODE_MANUAL, normalize_lvl_mode_value

_TARGET_PREVIEW_MIN_HZ = 10.0
_TARGET_PREVIEW_MAX_HZ = 20_000.0
_TARGET_PREVIEW_POINT_COUNT = 400
logger = logging.getLogger("DecayCore")


@dataclass(frozen=True)
class _TargetPreviewCurve:
    frequency_hz: np.ndarray
    base_magnitude_db: np.ndarray
    display_magnitude_db: np.ndarray
    mode_key: str
    mode_label: str
    is_manual_level: bool
    manual_level_db: float
    manual_tilt_db_per_oct: float


def _finite_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return float(parsed) if math.isfinite(parsed) else float(default)


def _preview_frequency_axis() -> np.ndarray:
    axis = np.logspace(
        math.log10(_TARGET_PREVIEW_MIN_HZ),
        math.log10(_TARGET_PREVIEW_MAX_HZ),
        _TARGET_PREVIEW_POINT_COUNT,
    )
    axis[0] = _TARGET_PREVIEW_MIN_HZ
    axis[-1] = _TARGET_PREVIEW_MAX_HZ
    return axis


def _load_preview_base_curve(
    *,
    frequency_hz: np.ndarray,
    mode_key: str,
    custom_file: Any,
) -> np.ndarray | None:
    source_freqs = None
    source_mags = None
    if mode_key == "Upload" and isinstance(custom_file, dict) and custom_file.get("content"):
        source_freqs, source_mags = load_target_curve(custom_file["content"])

    if source_freqs is None or source_mags is None:
        source_freqs, source_mags, _ = load_house_curve({"hc_mode": mode_key})

    if source_freqs is None or source_mags is None:
        return None

    try:
        freqs = np.asarray(source_freqs, dtype=float).reshape(-1)
        mags = np.asarray(source_mags, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    valid = np.isfinite(freqs) & np.isfinite(mags)
    freqs = freqs[valid]
    mags = mags[valid]
    if freqs.size < 2 or mags.size != freqs.size:
        return None

    order = np.argsort(freqs)
    freqs = freqs[order]
    mags = mags[order]
    unique_freqs, unique_indices = np.unique(freqs, return_index=True)
    unique_mags = mags[unique_indices]
    if unique_freqs.size < 2:
        return None
    return np.interp(
        frequency_hz,
        unique_freqs,
        unique_mags,
        left=unique_mags[0],
        right=unique_mags[-1],
    )


def _build_target_preview_curve(
    *,
    hc_mode_raw: Any,
    hc_custom_file: Any,
    app_mode_raw: Any,
    lvl_mode_raw: Any,
    lvl_manual_db_raw: Any,
    manual_target_tilt_db_per_oct_raw: Any,
) -> _TargetPreviewCurve | None:
    try:
        mode_label = str(hc_mode_raw or "Harman6")
    except (TypeError, ValueError):
        mode_label = "Harman6"
    mode_key = str(_normalize_hc_mode_key(mode_label))
    frequency_hz = _preview_frequency_axis()
    base_magnitude_db = _load_preview_base_curve(
        frequency_hz=frequency_hz,
        mode_key=mode_key,
        custom_file=hc_custom_file,
    )
    if base_magnitude_db is None:
        return None

    app_mode = str(app_mode_raw or "BASIC").strip().upper()
    lvl_mode = normalize_lvl_mode_value(lvl_mode_raw)
    if app_mode in ("BASIC", "AUTO"):
        lvl_mode = LVL_MODE_AUTO
    is_manual_level = lvl_mode == LVL_MODE_MANUAL
    manual_level_db = _finite_float(lvl_manual_db_raw, 0.0)
    manual_tilt_db_per_oct = _finite_float(manual_target_tilt_db_per_oct_raw, 0.0)
    display_magnitude_db = apply_manual_target_preview_adjustments(
        frequency_hz,
        base_magnitude_db,
        manual_level_db if is_manual_level else 0.0,
        manual_tilt_db_per_oct if is_manual_level else 0.0,
    )
    display_magnitude_db = np.asarray(display_magnitude_db, dtype=float).reshape(-1)
    if (
        display_magnitude_db.size != frequency_hz.size
        or not np.all(np.isfinite(display_magnitude_db))
    ):
        return None

    return _TargetPreviewCurve(
        frequency_hz=frequency_hz,
        base_magnitude_db=np.asarray(base_magnitude_db, dtype=float),
        display_magnitude_db=display_magnitude_db,
        mode_key=mode_key,
        mode_label=mode_label,
        is_manual_level=bool(is_manual_level),
        manual_level_db=float(manual_level_db),
        manual_tilt_db_per_oct=float(manual_tilt_db_per_oct),
    )


def _current_target_preview_curve(
    read_value: Callable[[str, Any], Any] = ctrl.value,
) -> _TargetPreviewCurve | None:
    return _build_target_preview_curve(
        hc_mode_raw=read_value("hc_mode", "Harman6"),
        hc_custom_file=read_value("hc_custom_file", None),
        app_mode_raw=read_value("mode", "BASIC"),
        lvl_mode_raw=read_value("lvl_mode", LVL_MODE_AUTO),
        lvl_manual_db_raw=read_value("lvl_manual_db", 0.0),
        manual_target_tilt_db_per_oct_raw=read_value(
            "manual_target_tilt_db_per_oct",
            0.0,
        ),
    )


def _serialize_target_curve_txt(curve: _TargetPreviewCurve) -> bytes:
    freqs = np.asarray(curve.frequency_hz, dtype=float).reshape(-1)
    mags = np.asarray(curve.display_magnitude_db, dtype=float).reshape(-1)
    valid = np.isfinite(freqs) & np.isfinite(mags) & (freqs > 0.0)
    freqs = freqs[valid]
    mags = mags[valid]
    if freqs.size < 2 or mags.size != freqs.size:
        raise ValueError("target curve has fewer than two finite points")

    lines = [
        "# DecayCore target curve",
        f"# Target: {curve.mode_key}",
        "# Frequency_Hz Magnitude_dB",
    ]
    lines.extend(f"{freq:.6f} {mag:.6f}" for freq, mag in zip(freqs, mags))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _target_curve_download_filename(mode_key: Any) -> str:
    try:
        raw = str(mode_key or "target")
    except (TypeError, ValueError):
        raw = "target"
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-")
    return f"DecayCore_target_{token or 'target'}.txt"


def _download_current_target_curve(*, t: Callable[[str], str]) -> None:
    from nicegui import ui  # noqa: PLC0415

    try:
        curve = _current_target_preview_curve()
    except (
        AttributeError,
        TypeError,
        ValueError,
        RuntimeError,
        OSError,
        OverflowError,
    ):
        logger.warning("Target curve TXT export failed", exc_info=True)
        curve = None
    if curve is None:
        ui.notify(
            t("target_curve_download_unavailable"),
            type="warning",
            position="top",
        )
        return

    try:
        payload = _serialize_target_curve_txt(curve)
    except (TypeError, ValueError, OverflowError):
        logger.warning("Target curve TXT serialization failed", exc_info=True)
        ui.notify(
            t("target_curve_download_unavailable"),
            type="warning",
            position="top",
        )
        return
    ui.download(
        payload,
        filename=_target_curve_download_filename(curve.mode_key),
    )
