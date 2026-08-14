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

from typing import Any
import logging
import math

from ...config.legacy_keys import CAMILLAFIR_AUTO_MODE
from ...config.schema import AUTO_MODE_DEFAULT_CFG_TO_UI
from ..auto_mode_policy import (
    auto_filter_type_for_key,
    auto_goal_is_flat_family,
)
from ..auto_mode_priors import get_auto_mode_filter_auto_defaults
from ...config.mode_policy import MODE_DEFAULTS
from ..value_normalization import (
    LVL_ALGO_MEDIAN,
    LVL_MODE_AUTO,
    LVL_MODE_MANUAL,
    OUTPUT_TILT_SOURCE_MANUAL_TARGET_TILT,
    OUTPUT_TILT_SOURCE_OFF,
    normalize_lvl_algo_value,
    normalize_lvl_mode_value,
    normalize_output_tilt_source_value,
)

logger = logging.getLogger("DecayCore")


_AUTO_MODE_DEFAULT_CFG_TO_UI = AUTO_MODE_DEFAULT_CFG_TO_UI


def _finite_float_or_default(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
        if math.isfinite(parsed):
            return float(parsed)
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
        logger.exception("float parse in pipeline config")
    return float(default)


def _advanced_manual_output_tilt_enabled(data: dict[str, Any]) -> bool:
    try:
        mode_u = str(data.get("mode", "BASIC") or "BASIC").strip().upper()
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
        mode_u = "BASIC"

    if mode_u != "ADVANCED":
        return False

    return normalize_lvl_mode_value(data.get("lvl_mode", LVL_MODE_AUTO)) == LVL_MODE_MANUAL


def _effective_output_tilt_source(data: dict[str, Any]) -> str:
    if _advanced_manual_output_tilt_enabled(data):
        return OUTPUT_TILT_SOURCE_MANUAL_TARGET_TILT
    return normalize_output_tilt_source_value(data.get("output_tilt_source", OUTPUT_TILT_SOURCE_OFF))


def _resolve_output_tilt_db_per_oct(data: dict[str, Any]) -> float:
    try:
        mode_u = str(data.get("mode", "BASIC") or "BASIC").strip().upper()
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
        mode_u = "BASIC"

    lvl_mode = normalize_lvl_mode_value(data.get("lvl_mode", LVL_MODE_AUTO))
    output_tilt_source = _effective_output_tilt_source(data)

    if mode_u == "AUTO":
        return 0.0
    if lvl_mode == LVL_MODE_MANUAL and output_tilt_source == OUTPUT_TILT_SOURCE_MANUAL_TARGET_TILT:
        return _finite_float_or_default(data.get("manual_target_tilt_db_per_oct", 0.0), 0.0)
    return 0.0


def _auto_mode_filter_type_or_default(value: Any) -> str:
    try:
        raw = str(value or "").strip()
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
        raw = ""
    low = raw.lower()
    if (
        "asym" in low
        or "mixed" in low
        or "minimum" in low
        or "minphase" in low
        or ("min" in low and "phase" in low)
        or "linear" in low
    ):
        return str(raw)
    return str(auto_filter_type_for_key("asym"))


def _auto_mode_min_local_correction_db(data: dict[str, Any]) -> float:
    """Keep the local-correction policy explicit through AUTO-mode setup."""
    value = _finite_float_or_default(data.get("min_boost_peak_db", 2.0), 2.0)
    return float(max(0.0, min(3.0, value)))


def _apply_auto_mode_managed_settings(data: dict[str, Any]) -> None:
    """Force AUTO mode to use program-managed settings except allowed user choices."""
    filter_type = _auto_mode_filter_type_or_default(data.get("filter_type", "Asymmetric"))

    merged_defaults = dict(MODE_DEFAULTS.get("AUTO", {}) or {})
    merged_defaults.update(dict(get_auto_mode_filter_auto_defaults(filter_type) or {}))

    forced = {
        "mode": "AUTO",
        CAMILLAFIR_AUTO_MODE: True,
        "auto_mode_workers": 0,
        "mag_correct": True,
        "gain": 0.10,
        "lvl_mode": LVL_MODE_AUTO,
        "lvl_algo": LVL_ALGO_MEDIAN,
        "lvl_manual_db": 0.0,
        "manual_target_tilt_db_per_oct": 0.0,
        "output_tilt_source": OUTPUT_TILT_SOURCE_OFF,
        "normalize_opt": False,
        "align_opt": True,
        "unsafe_raw_dsp": False,
        "stereo_link": True,
        "stereo_link_strategy": "auto",
        "exc_prot": True,
        "low_bass_cut_enable": False,
        "hpf_enable": True,
        "comparison_mode": True,
        "df_smoothing": False,
        "min_boost_peak_db": _auto_mode_min_local_correction_db(data),
        "auto_target_mode": str(data.get("auto_target_mode", "auto") or "auto"),
        "filter_type": str(filter_type),
    }

    for cfg_key, ui_key in _AUTO_MODE_DEFAULT_CFG_TO_UI.items():
        if cfg_key == "filter_type_str":
            continue
        if cfg_key in merged_defaults:
            if ui_key == "enable_afdw" and data.get("enable_afdw") is not None:
                forced[ui_key] = bool(data.get("enable_afdw", False))
                continue
            if ui_key in ("mag_c_min", "mag_c_max") and data.get(ui_key) is not None:
                forced[ui_key] = float(data[ui_key])
                continue
            forced[ui_key] = merged_defaults[cfg_key]
    forced["gain"] = 0.10

    forced["lvl_mode"] = normalize_lvl_mode_value(forced.get("lvl_mode", LVL_MODE_AUTO))
    forced["lvl_algo"] = normalize_lvl_algo_value(forced.get("lvl_algo", LVL_ALGO_MEDIAN))

    for key, value in forced.items():
        data[key] = value
    if auto_goal_is_flat_family(str(data.get("auto_goal", "balanced") or "balanced")):
        data["unsafe_raw_dsp"] = True


__all__ = [
    "_finite_float_or_default",
    "_advanced_manual_output_tilt_enabled",
    "_effective_output_tilt_source",
    "_resolve_output_tilt_db_per_oct",
    "_apply_auto_mode_managed_settings",
    "get_auto_mode_filter_auto_defaults",
]
