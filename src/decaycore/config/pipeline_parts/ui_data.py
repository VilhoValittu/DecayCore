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

from ..auto_mode_policy import auto_goal_is_flat_family
from ...config.legacy_keys import CAMILLAFIR_AUTO_MODE
from ...config.schema import (
    AUTO_MODE_DEFAULT_CFG_TO_UI,
    HIDDEN_CONF_DEFAULTS_ADVANCED,
    HIDDEN_CONF_DEFAULTS_BASIC_AUTO,
    LIST_BOOL_KEYS,
    UI_PIN_KEYS,
    RunConfigSnapshot,
    normalize_list_backed_booleans,
)
from ..value_normalization import (
    LAYOUT_MONO,
    LVL_ALGO_MEDIAN,
    LVL_MODE_AUTO,
    normalize_bass_integration_profile,
    normalize_layout_value,
    normalize_lvl_algo_value,
    normalize_lvl_mode_value,
    normalize_sub_combine_mode,
)
from .managed_settings import (
    _apply_auto_mode_managed_settings,
    _effective_output_tilt_source,
    _resolve_output_tilt_db_per_oct,
)
from .xo_hpf import filter_type_short

logger = logging.getLogger("DecayCore")


_AUTO_MODE_DEFAULT_CFG_TO_UI = AUTO_MODE_DEFAULT_CFG_TO_UI
_UI_PIN_KEYS = UI_PIN_KEYS
_LIST_BOOL_KEYS = LIST_BOOL_KEYS
_HIDDEN_CONF_DEFAULTS_ADVANCED = HIDDEN_CONF_DEFAULTS_ADVANCED
_HIDDEN_CONF_DEFAULTS_BASIC_AUTO = HIDDEN_CONF_DEFAULTS_BASIC_AUTO

_UI_PARSE_EXCEPTIONS = (TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError)


def _read_pin_value(pin: Any, key: str) -> Any:
    try:
        return pin[key]
    except (KeyError, TypeError):
        return None


def _safe_text(value: Any, default: str = "") -> str:
    try:
        return str(value if value is not None else default)
    except _UI_PARSE_EXCEPTIONS:
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else default
    except _UI_PARSE_EXCEPTIONS:
        return default


def _safe_positive_float(value: Any, default: float) -> float:
    parsed = _safe_float(value, default)
    return parsed if parsed > 0.0 else default


def _normalize_list_backed_booleans(data: dict[str, Any]) -> None:
    normalize_list_backed_booleans(data)


def _normalize_mode_and_auto_flags(data: dict[str, Any]) -> tuple[str, bool]:
    mode_raw = data.get("mode")
    mode_explicit = mode_raw not in (None, "")
    try:
        mode_u = str(mode_raw or "BASIC").strip().upper()
    except _UI_PARSE_EXCEPTIONS:
        mode_u = "BASIC"
    if mode_u not in ("BASIC", "ADVANCED", "AUTO"):
        mode_u = "BASIC"
        mode_explicit = False

    is_auto_mode = mode_u == "AUTO"
    if not mode_explicit and not is_auto_mode:
        try:
            is_auto_mode = bool(data.get(CAMILLAFIR_AUTO_MODE, False))
        except _UI_PARSE_EXCEPTIONS:
            is_auto_mode = False

    if is_auto_mode:
        mode_u = "AUTO"
        data["mode"] = "AUTO"
    data[CAMILLAFIR_AUTO_MODE] = bool(is_auto_mode)

    if bool(data.get("bass_integration_enable", False)):
        mode_u = "AUTO"
        is_auto_mode = True
        data["mode"] = "AUTO"
        data[CAMILLAFIR_AUTO_MODE] = True
    return mode_u, is_auto_mode


def _normalize_ui_modes(data: dict[str, Any]) -> None:
    data["layout"] = normalize_layout_value(data.get("layout", LAYOUT_MONO))
    data["lvl_mode"] = normalize_lvl_mode_value(data.get("lvl_mode", LVL_MODE_AUTO))
    data["lvl_algo"] = normalize_lvl_algo_value(data.get("lvl_algo", LVL_ALGO_MEDIAN))


def _normalize_auto_target_mode(data: dict[str, Any]) -> None:
    try:
        atm = str(data.get("auto_target_mode", "auto") or "auto").strip().lower()
    except _UI_PARSE_EXCEPTIONS:
        atm = "auto"
    if atm in ("selected", "manual", "fixed", "user"):
        atm = "selected"
    elif atm == "adaptive":
        atm = "adaptive"
    else:
        atm = "auto"
    data["auto_target_mode"] = str(atm)


def _normalize_channel_policy_defaults(data: dict[str, Any]) -> None:
    if data.get("enable_channel_specific_auto_policy") is None:
        data["enable_channel_specific_auto_policy"] = False
    if data.get("channel_specific_policy_max_hz") in (None, ""):
        data["channel_specific_policy_max_hz"] = 220.0


def _normalize_stereo_link_strategy(data: dict[str, Any]) -> None:
    try:
        sls = str(data.get("stereo_link_strategy", "") or "").strip().lower()
    except _UI_PARSE_EXCEPTIONS:
        sls = ""
    if sls == "off":
        data["stereo_link"] = False
        sls = "auto"
    elif sls not in ("shared", "hybrid", "auto"):
        sls = "auto"
    if sls == "":
        sls = "auto"
    data["stereo_link_strategy"] = sls


def _apply_hidden_conf_defaults(data: dict[str, Any], mode_u: str) -> None:
    defaults = _HIDDEN_CONF_DEFAULTS_ADVANCED if mode_u == "ADVANCED" else _HIDDEN_CONF_DEFAULTS_BASIC_AUTO
    for key, value in defaults.items():
        if data.get(key) in (None, ""):
            data[key] = value


def _collect_xo_fields(pin: Any, data: dict[str, Any]) -> None:
    for i in range(1, 6):
        for suffix in ("f", "s"):
            key = f"xo{i}_{suffix}"
            data[key] = _read_pin_value(pin, key)


def _normalize_level_controls(data: dict[str, Any]) -> None:
    data["max_cut_db"] = abs(_safe_float(data.get("max_cut_db", 15.0) or 15.0, 15.0))
    for key, default in (
        ("max_slope_db_per_oct", 24.0),
        ("max_slope_boost_db_per_oct", 0.0),
        ("max_slope_cut_db_per_oct", 0.0),
    ):
        data[key] = max(0.0, _safe_float(data.get(key, default) or default, default))
    data["lvl_manual_db"] = _safe_float(data.get("lvl_manual_db", 0.0) or 0.0, 0.0)
    data["manual_target_tilt_db_per_oct"] = _safe_float(data.get("manual_target_tilt_db_per_oct", 0.0) or 0.0, 0.0)
    data["output_tilt_source"] = _effective_output_tilt_source(data)
    data["output_tilt_db_per_oct"] = _safe_float(data.get("output_tilt_db_per_oct", 0.0) or 0.0, 0.0)
    data["output_tilt_db_per_oct"] = _resolve_output_tilt_db_per_oct(data)


def _normalize_misc_numeric_controls(data: dict[str, Any]) -> None:
    data["gain"] = max(0.0, _safe_float(data.get("gain", 0.0) or 0.0, 0.0))
    try:
        data["auto_mode_workers"] = int(float(data.get("auto_mode_workers", 0) or 0))
    except _UI_PARSE_EXCEPTIONS:
        data["auto_mode_workers"] = 0
    data["avr_crossover_hz"] = _safe_float(data.get("avr_crossover_hz", 80.0) or 80.0, 80.0)


def _normalize_bass_integration_mode_and_profile(data: dict[str, Any]) -> None:
    try:
        bi_mode = str(data.get("bass_integration_mode", "direct_dac") or "direct_dac").strip().lower()
    except _UI_PARSE_EXCEPTIONS:
        bi_mode = "direct_dac"
    if bi_mode != "direct_dac" or bool(data.get("bass_integration_enable", False)):
        bi_mode = "direct_dac"
    data["bass_integration_mode"] = bi_mode
    try:
        bi_profile = str(data.get("bass_integration_profile", "safe") or "safe").strip().lower()
    except _UI_PARSE_EXCEPTIONS:
        bi_profile = "safe"
    data["bass_integration_profile"] = normalize_bass_integration_profile(bi_profile)


def _normalize_bass_integration_alignment_fields(data: dict[str, Any], is_auto_mode: bool) -> None:
    data["bass_integration_sub_combine_mode"] = normalize_sub_combine_mode(
        data.get("bass_integration_sub_combine_mode", "average")
    )
    data["bass_integration_sub_delay_ms"] = _safe_float(data.get("bass_integration_sub_delay_ms", 0.0) or 0.0, 0.0)
    data["bass_integration_sub_array_delay_ms"] = _safe_float(
        data.get("bass_integration_sub_array_delay_ms", 0.0) or 0.0,
        0.0,
    )
    data["bass_integration_sub1_delay_ms"] = _safe_float(
        data.get("bass_integration_sub1_delay_ms", 0.0) or 0.0,
        0.0,
    )
    data["bass_integration_sub2_delay_ms"] = _safe_float(
        data.get("bass_integration_sub2_delay_ms", 0.0) or 0.0,
        0.0,
    )
    data["bass_integration_main_l_delay_ms"] = _safe_float(
        data.get("bass_integration_main_l_delay_ms", 0.0) or 0.0,
        0.0,
    )
    data["bass_integration_main_r_delay_ms"] = _safe_float(
        data.get("bass_integration_main_r_delay_ms", 0.0) or 0.0,
        0.0,
    )
    data["bass_integration_sub_polarity_invert"] = bool(data.get("bass_integration_sub_polarity_invert", False))
    data["bass_integration_sub_gain_trim_db"] = _safe_float(data.get("bass_integration_sub_gain_trim_db", 0.0) or 0.0, 0.0)
    data["bass_integration_alignment_auto_applied"] = bool(data.get("bass_integration_alignment_auto_applied", False))
    data["bass_integration_alignment_reason"] = str(data.get("bass_integration_alignment_reason", "") or "")
    data["bass_integration_allpass_auto_enable"] = bool(data.get("bass_integration_allpass_auto_enable", False))
    if bool(data.get("bass_integration_enable", False)) and is_auto_mode:
        data["bass_integration_allpass_auto_enable"] = False
    data["bass_integration_allpass_freq_hz"] = _safe_positive_float(data.get("bass_integration_allpass_freq_hz", 0.0) or 0.0, 0.0)
    data["bass_integration_allpass_q"] = _safe_positive_float(data.get("bass_integration_allpass_q", 0.707) or 0.707, 0.707)
    data["bass_integration_allpass_auto_applied"] = bool(data.get("bass_integration_allpass_auto_applied", False))
    if (
        not bool(data.get("bass_integration_enable", False))
        or str(data.get("bass_integration_mode", "") or "").strip().lower() != "direct_dac"
        or not bool(data.get("bass_integration_allpass_auto_enable", False))
    ):
        data["bass_integration_allpass_auto_applied"] = False


def _normalize_bass_integration_lpf(data: dict[str, Any]) -> None:
    main_xo_hz = _safe_float(
        data.get("sub_crossover_hz", data.get("avr_crossover_hz", 80.0))
        or data.get("avr_crossover_hz", 80.0)
        or 80.0,
        80.0,
    )
    if main_xo_hz <= 0.0:
        main_xo_hz = 80.0
    direct_sub_lpf_hz = _safe_float(data.get("direct_dac_sub_lpf_hz", main_xo_hz) or main_xo_hz, main_xo_hz)
    if direct_sub_lpf_hz <= 0.0:
        direct_sub_lpf_hz = main_xo_hz
    if data["bass_integration_mode"] == "direct_dac":
        direct_sub_lpf_hz = max(float(main_xo_hz), float(direct_sub_lpf_hz))
    data["direct_dac_sub_lpf_hz"] = float(direct_sub_lpf_hz)


def _normalize_ir_window_mode(data: dict[str, Any]) -> None:
    v_raw = data.get("ir_export_window_mode")
    if v_raw is None or (isinstance(v_raw, str) and v_raw.strip() == ""):
        v_raw = data.get("ir_window_mode", "auto")
    v = str(v_raw or "auto").strip().lower()
    v = v if v in ("auto", "off", "rew_sym", "rew_asym") else "auto"
    data["ir_export_window_mode"] = v
    data["ir_window_mode"] = v


def _normalize_ir_anchor_and_shape(data: dict[str, Any]) -> None:
    am = str(data.get("ir_anchor_mode", "min_causal") or "min_causal").strip().lower()
    if am not in ("peak", "centroid", "min_causal"):
        am = "min_causal"
    data["ir_anchor_mode"] = am
    try:
        sh = str(data.get("ir_export_window_shape", "hann") or "hann").strip().lower()
    except _UI_PARSE_EXCEPTIONS:
        sh = "hann"
    data["ir_export_window_shape"] = sh if sh in ("hann", "tukey") else "hann"
    alpha = _safe_float(data.get("ir_export_tukey_alpha", 0.25), 0.25)
    data["ir_export_tukey_alpha"] = max(0.0, min(1.0, float(alpha)))


def _force_asymmetric_window_defaults(data: dict[str, Any]) -> None:
    try:
        if filter_type_short(str(data.get("filter_type", "") or "")) == "Asymmetric":
            data["ir_export_window_mode"] = "rew_asym"
            data["ir_window_mode"] = "rew_asym"
            data["ir_export_window_shape"] = "tukey"
            data["ir_export_tukey_alpha"] = 0.25
    except (AttributeError, TypeError, ValueError):
        logger.debug("Failed to normalize asymmetric export-window defaults", exc_info=True)


def _normalize_smoothing_defaults(data: dict[str, Any]) -> None:
    if data.get("filter_smooth") is None and data.get("smoothing_level") is not None:
        data["filter_smooth"] = data.get("smoothing_level")
    if data.get("plot_smoothing_level") is None:
        data["plot_smoothing_level"] = "Psychoacoustic"

def collect_ui_data(pin) -> dict[str, Any]:
    """Funktio: collect ui data."""
    data: dict[str, Any] = {key: _read_pin_value(pin, key) for key in _UI_PIN_KEYS}

    if data.get("ir_window_right") in (None, ""):
        data["ir_window_right"] = data.get("ir_window", 500.0)
    if data.get("ir_window") in (None, ""):
        data["ir_window"] = data.get("ir_window_right", 500.0)

    _normalize_list_backed_booleans(data)
    mode_u, is_auto_mode = _normalize_mode_and_auto_flags(data)
    _normalize_channel_policy_defaults(data)
    _normalize_auto_target_mode(data)
    _normalize_ui_modes(data)

    if is_auto_mode:
        _apply_auto_mode_managed_settings(data)

    auto_prefer_bass = bool(
        is_auto_mode
        and auto_goal_is_flat_family(str(data.get("auto_goal", "balanced") or "balanced"))
    )
    if auto_prefer_bass:
        data["auto_target_mode"] = "selected"
    if mode_u in ("BASIC", "AUTO") and not auto_prefer_bass:
        data["lvl_mode"] = LVL_MODE_AUTO
        data["unsafe_raw_dsp"] = False

    _normalize_stereo_link_strategy(data)
    _apply_hidden_conf_defaults(data, mode_u)

    data["align_opt"] = True
    _collect_xo_fields(pin, data)
    _normalize_level_controls(data)
    _normalize_misc_numeric_controls(data)
    _normalize_bass_integration_mode_and_profile(data)
    _normalize_bass_integration_alignment_fields(data, is_auto_mode)
    _normalize_bass_integration_lpf(data)
    _normalize_ir_window_mode(data)
    _normalize_ir_anchor_and_shape(data)
    _force_asymmetric_window_defaults(data)
    _normalize_smoothing_defaults(data)

    logger.info(
        f"UI pins: ir_export_window_mode={data.get('ir_export_window_mode')}, "
        f"shape={data.get('ir_export_window_shape')}, alpha={data.get('ir_export_tukey_alpha')}"
    )
    return data


def collect_ui_config(pin) -> RunConfigSnapshot:
    return RunConfigSnapshot(values=collect_ui_data(pin))


def log_df_smoothing_toggle(source, logger) -> bool:
    try:
        if isinstance(source, dict):
            df_on = bool(source.get("df_smoothing"))
        else:
            df_on = bool(source["df_smoothing"])
    except (KeyError, TypeError, AttributeError):
        df_on = False
    logger.info(f"DF smoothing: {'ON' if df_on else 'OFF'}")
    return df_on


__all__ = ['collect_ui_config', 'collect_ui_data', 'log_df_smoothing_toggle']
