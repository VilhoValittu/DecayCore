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
import numpy as np

from ...auto_mode.filter_priors import get_auto_mode_filter_auto_defaults
from ...auto_mode.shared import AUTO_MODE_GOAL_FLAT, _auto_bass_integration_profile_norm, _auto_goal_norm
from ...config.mode_policy import MODE_DEFAULTS
from ...config.models import StereoAutoPolicyConfig
from ...dsp.bass_integration import normalize_sub_combine_mode
from ...ui_i18n import (
    LAYOUT_MONO,
    LVL_ALGO_MEDIAN,
    LVL_MODE_AUTO,
    LVL_MODE_MANUAL,
    OUTPUT_TILT_SOURCE_MANUAL_TARGET_TILT,
    OUTPUT_TILT_SOURCE_OFF,
    lvl_algo_legacy_name,
    lvl_mode_legacy_name,
    normalize_layout_value,
    normalize_lvl_algo_value,
    normalize_lvl_mode_value,
    normalize_output_tilt_source_value,
)

logger = logging.getLogger("DecayCore")


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
    "conf_pull_max_hz": "conf_pull_max_hz",
    "conf_pull_gamma_cut": "conf_pull_gamma_cut",
    "conf_pull_gamma_boost": "conf_pull_gamma_boost",
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

def _apply_auto_hpf_runtime_override(
    data: Dict[str, Any],
    hpf: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    try:
        mode_u = str(data.get("mode", "BASIC") or "BASIC").strip().upper()
    except Exception:
        mode_u = "BASIC"
    auto_mode_active = bool(mode_u == "AUTO" or data.get("camillafir_automatic_mode", False))
    if not bool(auto_mode_active):
        return hpf

    try:
        bi_mode = str(data.get("bass_integration_mode", "") or "").strip().lower()
    except Exception:
        bi_mode = ""
    if bool(data.get("bass_integration_enable", False)) and bi_mode == "direct_dac":
        return hpf

    override = data.get("_auto_hpf_runtime_override", None)
    if not isinstance(override, dict):
        return hpf

    base_hpf = dict(hpf or {}) if isinstance(hpf, dict) else {}
    enabled = bool(override.get("enabled", False))
    try:
        auto_goal = str(data.get("auto_goal", "") or "").strip().lower().replace("_", "-")
    except Exception:
        auto_goal = ""
    if auto_goal in {"flat", "prefer bass", "prefer-bass", "bass"}:
        enabled = True

    try:
        freq_hz = float(
            override.get(
                "freq",
                base_hpf.get("freq", data.get("hpf_freq", 20.0)),
            )
            or 20.0
        )
    except Exception:
        freq_hz = 20.0
    if not math.isfinite(freq_hz) or freq_hz <= 0.0:
        freq_hz = 20.0

    try:
        order = int(
            round(
                float(
                    override.get(
                        "order",
                        base_hpf.get("order", round(float(data.get("hpf_slope", 24) or 24.0) / 6.0)),
                    )
                    or 4
                )
            )
        )
    except Exception:
        order = 4
    order = max(1, int(order))

    return {
        "enabled": bool(enabled),
        "freq": float(freq_hz),
        "order": int(order),
    }

def build_xos_hpf(data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    xos: List[Dict[str, Any]] = []
    try:
        mode_u = str(data.get("mode", "BASIC") or "BASIC").strip().upper()
    except Exception:
        mode_u = "BASIC"
    auto_mode_active = bool(mode_u == "AUTO" or data.get("camillafir_automatic_mode", False))
    if filter_type_supports_xo_phase_model(data.get("filter_type", "")):
        for i in range(1, 6):
            f_raw = data.get(f"xo{i}_f", None)
            if f_raw in (None, "", 0):
                continue
            try:
                f_hz = float(f_raw)
            except Exception:
                continue
            if not math.isfinite(f_hz) or f_hz <= 0:
                continue
            s_raw = data.get(f"xo{i}_s", 12)
            try:
                slope_db_oct = int(round(float(s_raw)))
            except Exception:
                slope_db_oct = 12
            if slope_db_oct <= 0:
                slope_db_oct = 12
            order = max(1, int(round(slope_db_oct / 6.0)))
            xos.append({"freq": f_hz, "order": order, "slope": slope_db_oct, "idx": i})
    xos.sort(key=lambda d: float(d.get("freq", 0.0)))

    hpf_enabled = bool(data.get("hpf_enable")) or bool(auto_mode_active)
    try:
        hpf_slope_db_oct = int(round(float(data.get("hpf_slope", 12) or 12.0)))
    except Exception:
        hpf_slope_db_oct = 12
    if hpf_slope_db_oct <= 0:
        hpf_slope_db_oct = 12
    hpf = (
        {"enabled": bool(hpf_enabled),
         "freq": data.get("hpf_freq"),
         "order": max(1, int(round(hpf_slope_db_oct / 6.0)))}
        if bool(hpf_enabled)
        else None
    )
    if (
        bool(data.get("bass_integration_enable", False))
        and str(data.get("bass_integration_mode", "") or "").strip().lower() == "direct_dac"
    ):
        try:
            direct_xo_hz = float(data.get("sub_crossover_hz", data.get("avr_crossover_hz", 80.0)) or 80.0)
        except Exception:
            direct_xo_hz = 80.0
        try:
            direct_xo_order = int(round(float(data.get("sub_crossover_slope", 24) or 24.0))) // 6
        except Exception:
            direct_xo_order = 4
        hpf = {
            "enabled": True,
            "freq": float(direct_xo_hz if math.isfinite(direct_xo_hz) and direct_xo_hz > 0.0 else 80.0),
            "order": max(1, int(direct_xo_order)),
        }
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

def choose_target_rates(data: Dict[str, Any]) -> List[int]:
    if bool(data.get("multi_rate_opt")):
        return [44100, 48000, 88200, 96000, 176400, 192000]
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
    except Exception:
        upload_names = []

    return bool(
        any(str(path or "").endswith(".wav") for path in local_paths)
        or any(str(name or "").endswith(".wav") for name in upload_names)
    )


__all__ = ['_apply_auto_hpf_runtime_override', 'build_xos_hpf', 'filter_type_short', 'filter_type_supports_xo_phase_model', 'choose_target_rates', 'choose_dash_fs', 'detect_is_wav_source']


def _load_sibling_symbols() -> None:
    import importlib
    package = __package__
    for module_name in ['managed_settings', 'ui_data', 'xo_hpf', 'filter_config']:
        if module_name == __name__.rsplit('.', 1)[-1]:
            continue
        module = importlib.import_module(f"{package}.{module_name}")
        for symbol in getattr(module, "__all__", ()):
            globals().setdefault(symbol, getattr(module, symbol))


_load_sibling_symbols()
