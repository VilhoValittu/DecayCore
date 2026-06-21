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

from typing import Any, Dict, List, Optional, Tuple
import logging
import math

from ...config.legacy_keys import is_auto_mode as _is_auto_mode_active_helper
from ...config.schema import AUTO_MODE_DEFAULT_CFG_TO_UI

logger = logging.getLogger("DecayCore")

_RECOVERABLE_XO_EXCEPTIONS = (
    AttributeError,
    TypeError,
    ValueError,
    OverflowError,
    RuntimeError,
)


_AUTO_MODE_DEFAULT_CFG_TO_UI = {
    "global_gain_db": "gain",
    "mag_c_min": "mag_c_min",
    "mag_c_max": "mag_c_max",
    "max_boost_db": "max_boost",
    "max_cut_db": "max_cut_db",
    "phase_limit": "phase_limit",
    "reg_strength": "reg_strength",
    "fdw_cycles": "fdw_cycles",
    "filter_smooth": "filter_smooth",
    "tdc_strength": "tdc_strength",
    "tdc_max_reduction_db": "tdc_max_reduction_db",
    "tdc_slope_db_per_oct": "tdc_slope_db_per_oct",
    "low_bass_cut_hz": "low_bass_cut_hz",
    "hpf_enable": "hpf_enable",
    "hpf_freq": "hpf_freq",
    "hpf_slope": "hpf_slope",
    "ir_window_ms": "ir_window",
    "ir_window_ms_left": "ir_window_left",
    "ir_window_right": "ir_window",
    "ir_window_left": "ir_window_left",
    "mixed_split_freq": "mixed_freq",
    "trans_width": "trans_width",
    "bass_first_mode_max_hz": "bass_first_mode_max_hz",
    "max_slope_db_per_oct": "max_slope_db_per_oct",
    "max_slope_boost_db_per_oct": "max_slope_boost_db_per_oct",
    "max_slope_cut_db_per_oct": "max_slope_cut_db_per_oct",
    "lvl_manual_db": "lvl_manual_db",
    "manual_target_tilt_db_per_oct": "manual_target_tilt_db_per_oct",
    "output_tilt_db_per_oct": "output_tilt_db_per_oct",
    "lvl_min": "lvl_min",
    "lvl_max": "lvl_max",
    "conf_pull_floor": "conf_pull_floor",
    "conf_pull_ceil": "conf_pull_ceil",
    "conf_pull_max_hz": "conf_pull_max_hz",
    "conf_pull_gamma_cut": "conf_pull_gamma_cut",
    "conf_pull_gamma_boost": "conf_pull_gamma_boost",
    "conf_pull_bass_boost_floor_min": "conf_pull_bass_boost_floor_min",
    "conf_pull_bass_boost_restore": "conf_pull_bass_boost_restore",
    "low_bass_cut_strength": "low_bass_cut_strength",
    "filter_type_str": "filter_type",
    "plot_smoothing_level": "plot_smoothing_level",
    "lvl_mode": "lvl_mode",
    "lvl_algo": "lvl_algo",
    "stereo_link_strategy": "stereo_link_strategy",
    "enable_mag_correction": "mag_correct",
    "unsafe_raw_dsp": "unsafe_raw_dsp",
    "exc_prot": "exc_prot",
    "enable_tdc": "enable_tdc",
    "enable_afdw": "enable_afdw",
    "df_smoothing": "df_smoothing",
    "comparison_mode": "comparison_mode",
    "bass_first_ai": "bass_first_ai",
    "phase_safe_2058": "phase_safe_2058",
    "stereo_link": "stereo_link",
    "low_bass_cut_enable": "low_bass_cut_enable",
}

_AUTO_MODE_DEFAULT_CFG_TO_UI = AUTO_MODE_DEFAULT_CFG_TO_UI


def _safe_mode_upper(data: Dict[str, Any], *, key: str, default: str) -> str:
    try:
        return str(data.get(key, default) or default).strip().upper()
    except _RECOVERABLE_XO_EXCEPTIONS:
        return str(default)


def _safe_mode_lower(data: Dict[str, Any], *, key: str, default: str = "") -> str:
    try:
        return str(data.get(key, default) or default).strip().lower()
    except _RECOVERABLE_XO_EXCEPTIONS:
        return str(default)


def _safe_positive_frequency(value: Any, *, default: float) -> float:
    try:
        freq_hz = float(value or default)
    except _RECOVERABLE_XO_EXCEPTIONS:
        freq_hz = float(default)
    if not math.isfinite(freq_hz) or freq_hz <= 0.0:
        return float(default)
    return float(freq_hz)


def _safe_positive_order(value: Any, *, default: int) -> int:
    try:
        order = int(round(float(value or default)))
    except _RECOVERABLE_XO_EXCEPTIONS:
        order = int(default)
    return max(1, int(order))


def _safe_slope_db_oct(data: Dict[str, Any], key: str, *, default: int) -> int:
    try:
        slope_db_oct = int(round(float(data.get(key, default) or float(default))))
    except _RECOVERABLE_XO_EXCEPTIONS:
        slope_db_oct = int(default)
    if slope_db_oct <= 0:
        slope_db_oct = int(default)
    return int(slope_db_oct)


def _slope_to_order(slope_db_oct: int) -> int:
    return max(1, int(round(float(slope_db_oct) / 6.0)))


def _auto_mode_active(data: Dict[str, Any]) -> bool:
    mode_u = _safe_mode_upper(data, key="mode", default="BASIC")
    return _is_auto_mode_active_helper(data, mode_u)


def _is_direct_dac_bass_integration(data: Dict[str, Any]) -> bool:
    bi_mode = _safe_mode_lower(data, key="bass_integration_mode", default="")
    return bool(data.get("bass_integration_enable", False)) and bi_mode == "direct_dac"


def _collect_xo_entries(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    xos: List[Dict[str, Any]] = []
    for i in range(1, 6):
        f_raw = data.get(f"xo{i}_f", None)
        if f_raw in (None, "", 0):
            continue
        freq_hz = _safe_positive_frequency(f_raw, default=-1.0)
        if freq_hz <= 0.0:
            continue
        slope_db_oct = _safe_slope_db_oct(data, f"xo{i}_s", default=12)
        xos.append(
            {
                "freq": float(freq_hz),
                "order": _slope_to_order(slope_db_oct),
                "slope": int(slope_db_oct),
                "idx": i,
            }
        )
    xos.sort(key=lambda d: float(d.get("freq", 0.0)))
    return xos


def _apply_auto_hpf_runtime_override(
    data: Dict[str, Any],
    hpf: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not _auto_mode_active(data):
        return hpf

    if _is_direct_dac_bass_integration(data):
        return hpf

    override = data.get("_auto_hpf_runtime_override", None)
    if not isinstance(override, dict):
        return hpf

    base_hpf = dict(hpf or {}) if isinstance(hpf, dict) else {}
    enabled = bool(override.get("enabled", False))
    auto_goal = _safe_mode_lower(data, key="auto_goal", default="").replace("_", "-")
    if auto_goal in {"flat", "prefer bass", "prefer-bass", "bass"}:
        enabled = True

    freq_fallback = base_hpf.get("freq", data.get("hpf_freq", 20.0))
    freq_hz = _safe_positive_frequency(override.get("freq", freq_fallback), default=20.0)
    order_fallback = base_hpf.get("order", _slope_to_order(_safe_slope_db_oct(data, "hpf_slope", default=24)))
    order = _safe_positive_order(override.get("order", order_fallback), default=4)

    return {
        "enabled": bool(enabled),
        "freq": float(freq_hz),
        "order": int(order),
    }

def build_xos_hpf(data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    auto_mode_active = _auto_mode_active(data)
    is_direct_dac_bi = _is_direct_dac_bass_integration(data)
    xos: List[Dict[str, Any]] = []
    if filter_type_supports_xo_phase_model(data.get("filter_type", "")):
        xos = _collect_xo_entries(data)

    hpf_enabled = bool(data.get("hpf_enable")) or bool(auto_mode_active and not is_direct_dac_bi)
    hpf_slope_db_oct = _safe_slope_db_oct(data, "hpf_slope", default=12)
    hpf = (
        {
            "enabled": bool(hpf_enabled),
            "freq": data.get("hpf_freq"),
            "order": _slope_to_order(hpf_slope_db_oct),
        }
        if bool(hpf_enabled)
        else None
    )
    hpf = _apply_auto_hpf_runtime_override(data, hpf)
    return xos, hpf

def filter_type_short(filter_type: str) -> str:
    raw = str(filter_type or "").strip()
    s = raw.lower()

    if "asym" in s:
        return "Asymmetric"
    if "mixed" in s:
        return "Mixed"
    if "minimum" in s or "minphase" in s or s == "min" or ("min" in s and "phase" in s):
        return "Minimum"
    if "linear" in s or s == "lin":
        return "Linear"

    return "Linear"

def filter_type_supports_xo_phase_model(filter_type: Any) -> bool:
    """Return True when the main-speaker XO phase model is applicable."""
    raw = str(filter_type or "").strip().lower()
    if raw == "":
        return True
    if "asym" in raw or "linear" in raw:
        return True
    if "mixed" in raw or "minimum" in raw or raw == "min":
        return False
    return True


_MULTI_RATE_BASE_TARGET_RATES = (44100, 48000, 88200, 96000, 176400, 192000)
_MULTI_RATE_ULTRA_HIGH_TARGET_RATES = (352800, 384000)


def multi_rate_target_rates(*, include_ultra_high: bool = False) -> List[int]:
    rates = list(_MULTI_RATE_BASE_TARGET_RATES)
    if include_ultra_high:
        rates.extend(_MULTI_RATE_ULTRA_HIGH_TARGET_RATES)
    return rates

def choose_target_rates(data: Dict[str, Any]) -> List[int]:
    if bool(data.get("multi_rate_opt")):
        return multi_rate_target_rates(
            include_ultra_high=bool(data.get("multi_rate_ultra_high_opt", False))
        )
    try:
        return [int(data.get("fs") or 44100)]
    except (TypeError, ValueError):
        return [44100]

def choose_dash_fs(target_rates: List[int], *, multi_rate_on: bool, forced_plot_fs_hz: int) -> int:
    if not target_rates:
        return forced_plot_fs_hz
    dash_fs = int(forced_plot_fs_hz) if multi_rate_on else int(target_rates[0])
    if multi_rate_on and dash_fs not in target_rates:
        dash_fs = int(target_rates[0])
    return dash_fs

def detect_is_wav_source(data: Dict[str, Any]) -> bool:
    try:
        local_paths = [
            str(data.get("local_path_l", "") or "").lower(),
            str(data.get("local_path_r", "") or "").lower(),
            str(data.get("local_path_l_main", "") or "").lower(),
            str(data.get("local_path_r_main", "") or "").lower(),
            str(data.get("local_path_l_sub", "") or "").lower(),
            str(data.get("local_path_r_sub", "") or "").lower(),
        ]
    except (AttributeError, TypeError, ValueError):
        local_paths = []

    try:
        upload_names = []
        for key in (
            "file_l",
            "file_r",
            "file_l_main",
            "file_r_main",
            "file_l_sub",
            "file_r_sub",
        ):
            if isinstance(data.get(key), dict):
                upload_names.append(str(data[key].get("filename", "") or "").lower())
    except (AttributeError, TypeError, ValueError, KeyError, IndexError):
        upload_names = []

    return bool(
        any(str(path or "").endswith(".wav") for path in local_paths)
        or any(str(name or "").endswith(".wav") for name in upload_names)
    )


__all__ = ['_apply_auto_hpf_runtime_override', 'build_xos_hpf', 'filter_type_short', 'filter_type_supports_xo_phase_model', 'multi_rate_target_rates', 'choose_target_rates', 'choose_dash_fs', 'detect_is_wav_source']
