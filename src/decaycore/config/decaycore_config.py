# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

import json
import logging
from pathlib import Path

logger = logging.getLogger("DecayCore")

from ..app_paths import decaycore_config_path
from .legacy_keys import CAMILLAFIR_AUTO_MODE
from .schema import (
    DEVICE_AUDIO_FORMAT_OPTIONS,
    FILTER_WAV_FORMAT_OPTIONS,
    FS_OPTIONS,
    IR_EXPORT_WINDOW_MODE_OPTIONS,
    IR_EXPORT_WINDOW_SHAPE_OPTIONS,
    PLOT_SMOOTHING_LEVEL_OPTIONS,
    SLOPE_OPTIONS,
    STEREO_LINK_STRATEGY_OPTIONS,
    TAPS_OPTIONS,
    AppConfigSnapshot,
    app_config_snapshot,
    default_config_dict,
    normalize_choice_fields,
    normalize_choice_value,
    normalize_filter_type,
    persistable_config_dict,
)
from .value_normalization import (
    LAYOUT_MONO,
    LVL_ALGO_MEDIAN,
    LVL_MODE_AUTO,
    normalize_layout_value,
    normalize_lvl_algo_value,
    normalize_lvl_mode_value,
    normalize_output_tilt_source_value,
)

CONFIG_FILE: Path = decaycore_config_path()

_RECOVERABLE_CONFIG_EXCEPTIONS = (
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


def _normalize_filter_type(value) -> str:
    """Normalize persisted filter type names to the UI/program canonical labels."""
    return normalize_filter_type(value)


# Versioned migrations: each migration is registered with a version number.
# Only migrations with version > saved config's _config_version are applied.
# This ensures each migration runs exactly once, even across restarts.

_CONFIG_CURRENT_VERSION = 1  # Increment this when adding a new migration

_FS_OPTIONS = FS_OPTIONS
_TAPS_OPTIONS = TAPS_OPTIONS
_SLOPE_OPTIONS = SLOPE_OPTIONS
_PLOT_SMOOTHING_LEVEL_OPTIONS = PLOT_SMOOTHING_LEVEL_OPTIONS
_FILTER_WAV_FORMAT_OPTIONS = FILTER_WAV_FORMAT_OPTIONS
_DEVICE_AUDIO_FORMAT_OPTIONS = DEVICE_AUDIO_FORMAT_OPTIONS
_IR_EXPORT_WINDOW_MODE_OPTIONS = IR_EXPORT_WINDOW_MODE_OPTIONS
_IR_EXPORT_WINDOW_SHAPE_OPTIONS = IR_EXPORT_WINDOW_SHAPE_OPTIONS
_STEREO_LINK_STRATEGY_OPTIONS = STEREO_LINK_STRATEGY_OPTIONS


def _migration_v1_coerce_legacy_boolean_lists(saved: dict) -> None:
    """Migration 1: Convert list-valued booleans to bool type.

    Old configs stored booleans as lists; extract the first element if present.
    """
    for key in [
        "mag_correct",
        "normalize_opt",
        "align_opt",
        "multi_rate_opt",
        "multi_rate_ultra_high_opt",
        "stereo_link",
        "exc_prot",
        "hpf_enable",
        "df_smoothing",
        "bass_smooth_adaptive",
        "bass_adaptive_isolation_mode",
        "mid_refit_enable",
        "bass_boost_cap_enable",
        "bass_boost_post_restore_enable",
        "comparison_mode",
        "phase_safe_2058",
        "enable_ir_pre_energy_guard",
        "phase_tail_monotonic_enable",
        "unsafe_raw_dsp",
        "enable_channel_specific_auto_policy",
        "bass_integration_enable",
        "bass_integration_sub_polarity_invert",
        "bass_integration_alignment_auto_applied",
        "bass_integration_allpass_auto_enable",
        "bass_integration_allpass_auto_applied",
        CAMILLAFIR_AUTO_MODE,
    ]:
        if key in saved and isinstance(saved[key], list):
            # Extract first element if list is non-empty, otherwise False
            saved[key] = bool(saved[key][0]) if saved[key] else False


def _migration_v1_lvl_manual_db_shift(saved: dict) -> None:
    """Migration 1: Shift lvl_manual_db values from old 40–110 range to relative."""
    try:
        if "lvl_manual_db" in saved:
            value = float(saved.get("lvl_manual_db"))
            if 40.0 <= value <= 110.0:
                saved["lvl_manual_db"] = float(value - 75.0)
    except _RECOVERABLE_CONFIG_EXCEPTIONS:
        pass


def _parse_legacy_choice_index(value) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not float(value).is_integer():
            return None
        return int(value)
    try:
        text = str(value).strip()
    except _RECOVERABLE_CONFIG_EXCEPTIONS:
        return None
    if not text:
        return None
    if text[0] in "+-":
        digits = text[1:]
    else:
        digits = text
    if not digits.isdigit():
        return None
    try:
        return int(text)
    except _RECOVERABLE_CONFIG_EXCEPTIONS:
        return None


def _normalize_choice_value(value, *, options: tuple, default):
    return normalize_choice_value(value, options=options, default=default)


def _normalize_saved_choice_fields(saved: dict, default_conf: dict) -> None:
    normalize_choice_fields(saved, default_conf)


def _apply_config_migrations(saved: dict) -> None:
    """Apply versioned config migrations.

    Each migration is applied only if its version > saved config's _config_version.
    Migrations are defined as (_migration_vN_..., version) tuples.
    """
    current_version = int(saved.get("_config_version", 0))

    migrations = [
        (_migration_v1_coerce_legacy_boolean_lists, 1),
        (_migration_v1_lvl_manual_db_shift, 1),
    ]

    for migration_fn, version in migrations:
        if version > current_version:
            try:
                migration_fn(saved)
            except _RECOVERABLE_CONFIG_EXCEPTIONS:
                logger.exception(f"config migration {migration_fn.__name__}")

    saved["_config_version"] = _CONFIG_CURRENT_VERSION


def _normalize_saved_filter_type(saved: dict, default_conf: dict) -> None:
    try:
        saved["filter_type"] = _normalize_filter_type(saved.get("filter_type", default_conf.get("filter_type")))
    except _RECOVERABLE_CONFIG_EXCEPTIONS:
        saved["filter_type"] = str(default_conf.get("filter_type", "Asymmetric"))


def _load_and_merge_saved_config(default_conf: dict) -> bool:
    saved_mode_explicit = False
    config_path = Path(CONFIG_FILE)
    if not config_path.exists():
        return saved_mode_explicit
    try:
        with open(config_path, encoding="utf-8") as f:
            saved = json.load(f)
        if not isinstance(saved, dict):
            return saved_mode_explicit

        _apply_config_migrations(saved)
        _normalize_saved_filter_type(saved, default_conf)
        _normalize_saved_choice_fields(saved, default_conf)

        saved_mode_explicit = saved.get("mode", None) not in (None, "")
        saved["layout"] = normalize_layout_value(saved.get("layout", default_conf.get("layout")))
        saved["lvl_mode"] = normalize_lvl_mode_value(saved.get("lvl_mode", default_conf.get("lvl_mode")))
        saved["lvl_algo"] = normalize_lvl_algo_value(saved.get("lvl_algo", default_conf.get("lvl_algo")))
        default_conf.update(saved)
    except _RECOVERABLE_CONFIG_EXCEPTIONS:
        logger.exception("config load and merge")
    return saved_mode_explicit


def _load_saved_config_dict() -> dict:
    config_path = Path(CONFIG_FILE)
    if not config_path.exists():
        return {}
    try:
        with open(config_path, encoding="utf-8") as f:
            saved = json.load(f)
        if isinstance(saved, dict):
            return saved
    except _RECOVERABLE_CONFIG_EXCEPTIONS:
        logger.exception("config raw load")
    return {}


def _resolve_runtime_mode(default_conf: dict, *, saved_mode_explicit: bool) -> str:
    try:
        mode_u = str(default_conf.get("mode", "AUTO") or "AUTO").strip().upper()
    except _RECOVERABLE_CONFIG_EXCEPTIONS:
        mode_u = "AUTO"
    try:
        legacy_auto = bool(default_conf.get(CAMILLAFIR_AUTO_MODE, False))
    except _RECOVERABLE_CONFIG_EXCEPTIONS:
        legacy_auto = False
    if legacy_auto and not saved_mode_explicit:
        mode_u = "AUTO"
    if mode_u not in ("AUTO", "BASIC", "ADVANCED"):
        mode_u = "AUTO"
    return mode_u


def _make_default_config() -> dict:
    """Return pristine default config without reading any saved file."""
    return default_config_dict()


def load_config() -> dict:
    return load_config_snapshot().to_flat_dict()


def load_config_snapshot() -> AppConfigSnapshot:
    default_conf = _make_default_config()

    saved_mode_explicit = _load_and_merge_saved_config(default_conf)

    default_conf["unsafe_raw_dsp"] = False
    mode_u = _resolve_runtime_mode(default_conf, saved_mode_explicit=saved_mode_explicit)

    default_conf["mode"] = mode_u
    default_conf[CAMILLAFIR_AUTO_MODE] = bool(mode_u == "AUTO")
    if mode_u == "AUTO":
        default_conf["hpf_enable"] = True
    default_conf["layout"] = normalize_layout_value(default_conf.get("layout", LAYOUT_MONO))
    default_conf["lvl_mode"] = normalize_lvl_mode_value(default_conf.get("lvl_mode", LVL_MODE_AUTO))
    default_conf["lvl_algo"] = normalize_lvl_algo_value(default_conf.get("lvl_algo", LVL_ALGO_MEDIAN))
    default_conf["output_tilt_source"] = normalize_output_tilt_source_value(
        default_conf.get("output_tilt_source", "off")
    )

    return app_config_snapshot(default_conf)


def save_config(data: dict) -> None:
    save_config_snapshot(AppConfigSnapshot(values=dict(data or {})))


def save_config_snapshot(snapshot: AppConfigSnapshot) -> None:
    try:
        clean_data = persistable_config_dict(snapshot.to_flat_dict())
        saved_existing = _load_saved_config_dict()
        if "ui_theme_dark" not in clean_data and "ui_theme_dark" in saved_existing:
            clean_data["ui_theme_dark"] = bool(saved_existing["ui_theme_dark"])
        clean_data["filter_type"] = _normalize_filter_type(clean_data.get("filter_type", "Asymmetric"))
        _normalize_saved_choice_fields(clean_data, _make_default_config())
        clean_data["layout"] = normalize_layout_value(clean_data.get("layout", LAYOUT_MONO))
        clean_data["lvl_mode"] = normalize_lvl_mode_value(clean_data.get("lvl_mode", LVL_MODE_AUTO))
        clean_data["lvl_algo"] = normalize_lvl_algo_value(clean_data.get("lvl_algo", LVL_ALGO_MEDIAN))
        clean_data["output_tilt_source"] = normalize_output_tilt_source_value(
            clean_data.get("output_tilt_source", "off")
        )
        config_path = Path(CONFIG_FILE)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(clean_data, f, indent=4)
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
        logger.exception("config save")
