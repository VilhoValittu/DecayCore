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
from .managed_settings import _effective_output_tilt_source, _resolve_output_tilt_db_per_oct

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

def build_filter_config(
    *,
    FilterConfig_cls,
    fs_v: int,
    taps_v: int,
    data: Dict[str, Any],
    xos,
    hpf,
    hc_f,
    hc_m,
    pin=None,
) -> Any:
    """Rakentaa tai generoi: build filter config."""

    def _as_float(v, default=0.0) -> float:
        try:
            x = float(v)
            return x if x == x else float(default)
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
            return float(default)

    def _as_int(v, default=0) -> int:
        try:
            return int(float(v))
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
            return int(default)
    def _as_bool_default(v, default: bool) -> bool:
        """Sisainen apufunktio: as bool with None/empty fallback."""
        if v is None:
            return bool(default)
        if isinstance(v, str):
            s = v.strip().lower()
            if s == "":
                return bool(default)
            if s in ("1", "true", "yes", "on"):
                return True
            if s in ("0", "false", "no", "off"):
                return False
        if isinstance(v, (list, tuple)):
            if len(v) == 0:
                return bool(default)
            if len(v) == 1:
                return _as_bool_default(v[0], default)
        try:
            return bool(v)
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
            return bool(default)
    def _as_float_allow_zero(v, default: float) -> float:
        """Sisainen apufunktio: as float allow zero."""
        if v is None:
            return float(default)
        if isinstance(v, str) and v.strip() == "":
            return float(default)
        return _as_float(v, default)

    def _as_float_or_none(v, default: Optional[float]) -> Optional[float]:
        """Sisainen apufunktio: as float or none."""
        if v is None:
            return default
        if isinstance(v, str):
            s = v.strip()
            if s == "" or s.lower() == "none":
                return default
        try:
            x = float(v)
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
            return default
        if not math.isfinite(x):
            return default
        return float(x)

    conf_pull_floor = _as_float_allow_zero(data.get("conf_pull_floor", None), 0.05)
    conf_pull_ceil  = _as_float_allow_zero(data.get("conf_pull_ceil", None), 0.95)
    conf_pull_max_hz = _as_float_or_none(data.get("conf_pull_max_hz", None), 200.0)
    conf_pull_gamma_cut   = _as_float_allow_zero(data.get("conf_pull_gamma_cut", None), 0.55)
    conf_pull_gamma_boost = _as_float_allow_zero(data.get("conf_pull_gamma_boost", None), 0.35)

    conf_pull_conf_smooth_sigma = _as_float_allow_zero(data.get("conf_pull_conf_smooth_sigma", None), 2.0)
    conf_pull_bass_floor_hz     = _as_float_allow_zero(data.get("conf_pull_bass_floor_hz", None), 120.0)
    conf_pull_bass_floor_min    = _as_float_allow_zero(data.get("conf_pull_bass_floor_min", None), 0.25)

    low_bass_cut_strength = _as_float_allow_zero(data.get("low_bass_cut_strength", None), 0.0)
    low_bass_cut_strength = float(max(0.0, min(1.0, low_bass_cut_strength)))
    mixed_excess_phase_strength = _as_float_allow_zero(data.get("excess_phase_strength", None), 0.9)
    mixed_low_full_hz = _as_float_allow_zero(data.get("low_freq_full_correction_hz", None), 140.0)
    mixed_high_none_hz = _as_float_allow_zero(data.get("high_freq_no_correction_hz", None), 900.0)
    mixed_phase_budget_lf_deg = _as_float_allow_zero(data.get("mixed_phase_budget_lf_deg", None), 40.0)
    mixed_phase_budget_hf_deg = _as_float_allow_zero(data.get("mixed_phase_budget_hf_deg", None), 22.5)
    enable_ir_pre_energy_guard = bool(data.get("enable_ir_pre_energy_guard", True))
    pre_energy_ratio_max = _as_float_allow_zero(data.get("pre_energy_ratio_max", None), 0.25)
    pre_energy_guard_strength = _as_float_allow_zero(data.get("pre_energy_guard_strength", None), 0.8)
    mixed_max_pre_db = _as_float_allow_zero(data.get("max_pre_ringing_db", None), -35.0)
    mixed_max_excess_delay_ms = _as_float_allow_zero(data.get("max_excess_delay_ms", None), 2.5)
    gd_grad_limit_ms_per_oct = _as_float_allow_zero(data.get("gd_grad_limit_ms_per_oct", None), 30.0)
    ir_anchor_mode = str(data.get("ir_anchor_mode", "min_causal") or "min_causal").strip().lower()
    if ir_anchor_mode not in ("peak", "centroid", "min_causal"):
        ir_anchor_mode = "min_causal"
    min_causal_ms = _as_float_allow_zero(data.get("min_causal_ms", None), 80.0)
    auto_asym_left_ratio = _as_float_allow_zero(data.get("auto_asym_left_ratio", None), 0.35)
    auto_asym_left_max_ms = _as_float_allow_zero(data.get("auto_asym_left_max_ms", None), 25.0)
    mixed_kwargs = {}
    if hasattr(FilterConfig_cls, "excess_phase_strength"):
        mixed_kwargs["excess_phase_strength"] = float(max(0.0, min(1.0, mixed_excess_phase_strength)))
    if hasattr(FilterConfig_cls, "low_freq_full_correction_hz"):
        mixed_kwargs["low_freq_full_correction_hz"] = float(max(20.0, mixed_low_full_hz))
    if hasattr(FilterConfig_cls, "high_freq_no_correction_hz"):
        mixed_kwargs["high_freq_no_correction_hz"] = float(max(20.0, mixed_high_none_hz))
    if hasattr(FilterConfig_cls, "mixed_phase_budget_lf_deg"):
        mixed_kwargs["mixed_phase_budget_lf_deg"] = float(max(0.0, mixed_phase_budget_lf_deg))
    if hasattr(FilterConfig_cls, "mixed_phase_budget_hf_deg"):
        mixed_kwargs["mixed_phase_budget_hf_deg"] = float(max(0.0, mixed_phase_budget_hf_deg))
    if hasattr(FilterConfig_cls, "enable_ir_pre_energy_guard"):
        mixed_kwargs["enable_ir_pre_energy_guard"] = bool(enable_ir_pre_energy_guard)
    if hasattr(FilterConfig_cls, "pre_energy_ratio_max"):
        mixed_kwargs["pre_energy_ratio_max"] = float(max(0.0, pre_energy_ratio_max))
    if hasattr(FilterConfig_cls, "pre_energy_guard_strength"):
        mixed_kwargs["pre_energy_guard_strength"] = float(np.clip(pre_energy_guard_strength, 0.0, 1.0))
    if hasattr(FilterConfig_cls, "max_pre_ringing_db"):
        mixed_kwargs["max_pre_ringing_db"] = float(min(0.0, mixed_max_pre_db))
    if hasattr(FilterConfig_cls, "max_excess_delay_ms"):
        mixed_kwargs["max_excess_delay_ms"] = float(max(0.0, mixed_max_excess_delay_ms))
    if hasattr(FilterConfig_cls, "gd_grad_limit_ms_per_oct"):
        mixed_kwargs["gd_grad_limit_ms_per_oct"] = float(max(0.0, gd_grad_limit_ms_per_oct))
    if hasattr(FilterConfig_cls, "ir_anchor_mode"):
        mixed_kwargs["ir_anchor_mode"] = str(ir_anchor_mode)
    if hasattr(FilterConfig_cls, "min_causal_ms"):
        mixed_kwargs["min_causal_ms"] = float(max(0.0, min_causal_ms))
    if hasattr(FilterConfig_cls, "auto_asym_left_ratio"):
        mixed_kwargs["auto_asym_left_ratio"] = float(np.clip(auto_asym_left_ratio, 0.0, 1.0))
    if hasattr(FilterConfig_cls, "auto_asym_left_max_ms"):
        mixed_kwargs["auto_asym_left_max_ms"] = float(max(0.0, auto_asym_left_max_ms))
    lb_en = bool(data.get("low_bass_cut_enable", True))
    lb_raw = data.get("low_bass_cut_hz", "")
    if (not lb_en) or (lb_raw in (None, "", "None")):
        lb_hz = 0.0
    else:
        lb_hz = _as_float(lb_raw, 40.0)
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
    is_auto_mode = bool(mode_u == "AUTO" or data.get("camillafir_automatic_mode", False))
    df_smoothing = _as_bool_default(
        data.get("df_smoothing", False),
        False,
    )
    bass_smooth_adaptive = _as_bool_default(
        data.get("bass_smooth_adaptive", True),
        True,
    )
    bass_smooth_hz = _as_float_allow_zero(data.get("bass_smooth_hz", None), 200.0)
    bass_smooth_sigma_scale = _as_float_allow_zero(data.get("bass_smooth_sigma_scale", None), 1.4)
    bass_smooth_conf_floor = _as_float_allow_zero(data.get("bass_smooth_conf_floor", None), 0.3)
    mid_refit_enable = _as_bool_default(
        data.get("mid_refit_enable", True),
        True,
    )
    mid_refit_hz_lo = _as_float_allow_zero(data.get("mid_refit_hz_lo", None), 200.0)
    mid_refit_hz_hi = _as_float_allow_zero(data.get("mid_refit_hz_hi", None), 2000.0)
    mid_refit_k = _as_float_allow_zero(data.get("mid_refit_k", None), 0.45)
    mid_refit_smooth_oct = _as_float_allow_zero(data.get("mid_refit_smooth_oct", None), 0.60)
    mid_refit_conf_min_avg = _as_float_allow_zero(data.get("mid_refit_conf_min_avg", None), 0.20)
    # process_run may enforce this in `data`; keep data as highest-priority source.
    bass_adaptive_isolation_mode = _as_bool_default(
        data.get("bass_adaptive_isolation_mode", False),
        False,
    )
    bass_smooth_kwargs = {}
    if hasattr(FilterConfig_cls, "bass_smooth_adaptive"):
        bass_smooth_kwargs["bass_smooth_adaptive"] = bool(bass_smooth_adaptive)
    if hasattr(FilterConfig_cls, "bass_smooth_hz"):
        bass_smooth_kwargs["bass_smooth_hz"] = float(max(20.0, bass_smooth_hz))
    if hasattr(FilterConfig_cls, "bass_smooth_sigma_scale"):
        bass_smooth_kwargs["bass_smooth_sigma_scale"] = float(max(1.0, bass_smooth_sigma_scale))
    if hasattr(FilterConfig_cls, "bass_smooth_conf_floor"):
        bass_smooth_kwargs["bass_smooth_conf_floor"] = float(np.clip(bass_smooth_conf_floor, 0.05, 1.0))
    if hasattr(FilterConfig_cls, "bass_adaptive_isolation_mode"):
        bass_smooth_kwargs["bass_adaptive_isolation_mode"] = bool(bass_adaptive_isolation_mode)
    if hasattr(FilterConfig_cls, "mid_refit_enable"):
        bass_smooth_kwargs["mid_refit_enable"] = bool(mid_refit_enable)
    if hasattr(FilterConfig_cls, "mid_refit_hz_lo"):
        bass_smooth_kwargs["mid_refit_hz_lo"] = float(max(20.0, mid_refit_hz_lo))
    if hasattr(FilterConfig_cls, "mid_refit_hz_hi"):
        bass_smooth_kwargs["mid_refit_hz_hi"] = float(max(max(20.0, mid_refit_hz_lo) + 1.0, mid_refit_hz_hi))
    if hasattr(FilterConfig_cls, "mid_refit_k"):
        bass_smooth_kwargs["mid_refit_k"] = float(np.clip(mid_refit_k, 0.0, 1.0))
    if hasattr(FilterConfig_cls, "mid_refit_smooth_oct"):
        bass_smooth_kwargs["mid_refit_smooth_oct"] = float(np.clip(mid_refit_smooth_oct, 1.0 / 192.0, 1.0))
    if hasattr(FilterConfig_cls, "mid_refit_conf_min_avg"):
        bass_smooth_kwargs["mid_refit_conf_min_avg"] = float(np.clip(mid_refit_conf_min_avg, 0.0, 1.0))
    bass_boost_cap_enable = _as_bool_default(
        data.get("bass_boost_cap_enable", True),
        True,
    )
    bass_boost_cap_hz = _as_float_allow_zero(data.get("bass_boost_cap_hz", None), 200.0)
    bass_boost_cap_extra_db = _as_float_allow_zero(data.get("bass_boost_cap_extra_db", None), 5.0)
    bass_boost_cap_conf_min = _as_float_allow_zero(data.get("bass_boost_cap_conf_min", None), 0.55)
    bass_boost_post_restore_enable = _as_bool_default(
        data.get("bass_boost_post_restore_enable", True),
        True,
    )
    bass_boost_post_restore_strength = _as_float_allow_zero(data.get("bass_boost_post_restore_strength", None), 0.75)
    bass_boost_cap_kwargs = {}
    if hasattr(FilterConfig_cls, "bass_boost_cap_enable"):
        bass_boost_cap_kwargs["bass_boost_cap_enable"] = bool(bass_boost_cap_enable)
    if hasattr(FilterConfig_cls, "bass_boost_cap_hz"):
        bass_boost_cap_kwargs["bass_boost_cap_hz"] = float(max(20.0, bass_boost_cap_hz))
    if hasattr(FilterConfig_cls, "bass_boost_cap_extra_db"):
        bass_boost_cap_kwargs["bass_boost_cap_extra_db"] = float(max(0.0, bass_boost_cap_extra_db))
    if hasattr(FilterConfig_cls, "bass_boost_cap_conf_min"):
        bass_boost_cap_kwargs["bass_boost_cap_conf_min"] = float(np.clip(bass_boost_cap_conf_min, 0.0, 0.99))
    if hasattr(FilterConfig_cls, "bass_boost_post_restore_enable"):
        bass_boost_cap_kwargs["bass_boost_post_restore_enable"] = bool(bass_boost_post_restore_enable)
    if hasattr(FilterConfig_cls, "bass_boost_post_restore_strength"):
        bass_boost_cap_kwargs["bass_boost_post_restore_strength"] = float(np.clip(bass_boost_post_restore_strength, 0.0, 1.0))
    residual_pass_mode = str(data.get("residual_pass_mode", "modal_polish") or "modal_polish").strip().lower()
    if residual_pass_mode not in ("modal_polish", "general_fit", "off"):
        residual_pass_mode = "modal_polish"
    residual_authority_kwargs = {}
    if hasattr(FilterConfig_cls, "residual_pass_mode"):
        residual_authority_kwargs["residual_pass_mode"] = residual_pass_mode
    if hasattr(FilterConfig_cls, "residual_null_guard_enable"):
        residual_authority_kwargs["residual_null_guard_enable"] = _as_bool_default(
            data.get("residual_null_guard_enable", True),
            True,
        )
    _residual_float_defaults = {
        "residual_null_guard_strength": 1.0,
        "residual_modal_min_support": 0.45,
        "residual_boost_authority_min": 0.40,
        "residual_cut_authority_min": 0.35,
        "residual_reflection_risk_max": 0.65,
        "residual_null_risk_max_for_boost": 0.35,
        "residual_null_risk_max_for_cut": 0.75,
        "residual_max_boost_when_null_risk_db": 0.5,
        "residual_max_boost_general_db": 2.0,
        "residual_max_cut_general_db": 4.0,
        "residual_authority_smooth_oct": 1.0 / 9.0,
    }
    for _key, _default in _residual_float_defaults.items():
        if hasattr(FilterConfig_cls, _key):
            residual_authority_kwargs[_key] = float(_as_float_allow_zero(data.get(_key, None), _default))
    enable_afdw = _as_bool_default(
        data.get("enable_afdw", False),
        False,
    )
    enable_tdc = _as_bool_default(
        data.get("enable_tdc", False),
        False,
    )
    tdc_max_red = _as_float(
        data.get("tdc_max_reduction_db", 9.0),
        9.0,
    )
    tdc_slope = _as_float(
        data.get("tdc_slope_db_per_oct", 0.0),
        0.0,
    )
    filter_smooth = _as_int(
        data.get("filter_smooth", data.get("smoothing_level", 96)),
        96,
    )
    comparison_mode = bool(data.get("comparison_mode", True))
    lvl_mode = lvl_mode_legacy_name(data.get("lvl_mode", LVL_MODE_AUTO))
    if mode_u in ("BASIC", "AUTO"):
        lvl_mode = lvl_mode_legacy_name(LVL_MODE_AUTO)
    lvl_algo = lvl_algo_legacy_name(data.get("lvl_algo", LVL_ALGO_MEDIAN))
    sls = str(data.get("stereo_link_strategy", "auto") or "").strip().lower()
    if sls not in ("shared", "hybrid", "auto"):
        sls = "auto"
    bass_integration_kwargs = {}
    if hasattr(FilterConfig_cls, "bass_integration_enable"):
        bass_integration_kwargs["bass_integration_enable"] = bool(data.get("bass_integration_enable", False))
    if hasattr(FilterConfig_cls, "bass_integration_mode"):
        bass_integration_kwargs["bass_integration_mode"] = "direct_dac"
    if hasattr(FilterConfig_cls, "bass_integration_profile"):
        bass_integration_kwargs["bass_integration_profile"] = str(
            data.get("bass_integration_profile", "safe") or "safe"
        )
    if hasattr(FilterConfig_cls, "bass_integration_sub_combine_mode"):
        bass_integration_kwargs["bass_integration_sub_combine_mode"] = str(
            normalize_sub_combine_mode(data.get("bass_integration_sub_combine_mode", "average"))
        )
    if hasattr(FilterConfig_cls, "avr_crossover_hz"):
        bass_integration_kwargs["avr_crossover_hz"] = float(_as_float_allow_zero(data.get("avr_crossover_hz", None), 80.0))
    if hasattr(FilterConfig_cls, "direct_dac_sub_lpf_hz"):
        _main_xo_hz = float(
            _as_float_allow_zero(
                data.get("sub_crossover_hz", None),
                _as_float_allow_zero(data.get("avr_crossover_hz", None), 80.0),
            )
        )
        _direct_sub_lpf_hz = float(
            _as_float_allow_zero(
                data.get("direct_dac_sub_lpf_hz", None),
                _main_xo_hz,
            )
        )
        if str(data.get("bass_integration_mode", "") or "").strip().lower() == "direct_dac":
            _direct_sub_lpf_hz = max(_main_xo_hz, _direct_sub_lpf_hz)
        bass_integration_kwargs["direct_dac_sub_lpf_hz"] = float(_direct_sub_lpf_hz)
    for _side in ("l", "r"):
        _k = f"avr_crossover_hz_{_side}"
        _v = data.get(_k)
        if _v is not None and hasattr(FilterConfig_cls, _k):
            try:
                bass_integration_kwargs[_k] = float(_v)
            except (TypeError, ValueError):
                pass
    if hasattr(FilterConfig_cls, "bass_integration_guard_lo_ratio"):
        bass_integration_kwargs["bass_integration_guard_lo_ratio"] = float(
            max(0.05, _as_float_allow_zero(data.get("bass_integration_guard_lo_ratio", None), 0.60))
        )
    if hasattr(FilterConfig_cls, "bass_integration_guard_hi_ratio"):
        bass_integration_kwargs["bass_integration_guard_hi_ratio"] = float(
            max(
                float(bass_integration_kwargs.get("bass_integration_guard_lo_ratio", 0.60)) + 0.05,
                _as_float_allow_zero(data.get("bass_integration_guard_hi_ratio", None), 1.40),
            )
        )
    dataclass_fields = getattr(FilterConfig_cls, "__dataclass_fields__", {})
    stereo_auto_policy = StereoAutoPolicyConfig.from_dict(
        {
            "enable_channel_specific_auto_policy": bool(data.get("enable_channel_specific_auto_policy", False)),
            "channel_specific_policy_max_hz": _as_float_allow_zero(
                data.get("channel_specific_policy_max_hz", None),
                220.0,
            ),
        }
    )
    if mode_u == "BASIC":
        stereo_auto_policy.enable_channel_specific_auto_policy = False

    cfg = FilterConfig_cls(
        fs=int(fs_v),
        num_taps=int(taps_v),
        df_smoothing=bool(df_smoothing),
        **({"comparison_mode": bool(comparison_mode)} if hasattr(FilterConfig_cls, "comparison_mode") else {}),
        **({"auto_goal": str(data.get("auto_goal", "balanced") or "balanced")} if hasattr(FilterConfig_cls, "auto_goal") else {}),
        filter_type_str=data["filter_type"],
        mixed_split_freq=data["mixed_freq"],
        global_gain_db=0.0,
        mag_c_min=data["mag_c_min"],
        mag_c_max=data["mag_c_max"],
        max_boost_db=data["max_boost"],
        max_cut_db=data.get("max_cut_db", 30.0),
        max_slope_db_per_oct=data.get("max_slope_db_per_oct", 24.0),
        max_slope_boost_db_per_oct=data.get("max_slope_boost_db_per_oct", 0.0),
        max_slope_cut_db_per_oct=data.get("max_slope_cut_db_per_oct", 0.0),
        phase_limit=data["phase_limit"],
        phase_safe_2058=False,
        enable_mag_correction=bool(data.get("mag_correct", True)),
        unsafe_raw_dsp=bool(data.get("unsafe_raw_dsp", False)),
        lvl_mode=lvl_mode,
        reg_strength=float(data.get("reg_strength", 30.0)),
        do_normalize=bool(data["normalize_opt"]),
        exc_prot=bool(data["exc_prot"]),
        exc_freq=data["exc_freq"],
        low_bass_cut_hz=float(lb_hz),
        ir_window_ms=data.get("ir_window_right", 500.0),
        ir_window_ms_left=data.get("ir_window_left", 85.0),
        ir_export_window_mode=data.get("ir_export_window_mode", "auto"),
        enable_afdw=bool(enable_afdw),
        enable_tdc=bool(enable_tdc),
        tdc_strength=data.get("tdc_strength", 50.0),
        tdc_max_reduction_db=float(tdc_max_red),
        tdc_slope_db_per_oct=float(tdc_slope),
        plot_smoothing_level=data.get("plot_smoothing_level", "Psychoacoustic"),
        filter_smooth=int(filter_smooth),
        fdw_cycles=data["fdw_cycles"],
        lvl_manual_db=data["lvl_manual_db"],
        manual_target_tilt_db_per_oct=data["manual_target_tilt_db_per_oct"],
        **(
            {"output_tilt_source": _effective_output_tilt_source(data)}
            if hasattr(FilterConfig_cls, "output_tilt_source") else {}
        ),
        output_tilt_db_per_oct=_resolve_output_tilt_db_per_oct(data),
        lvl_min=data["lvl_min"],
        lvl_max=data["lvl_max"],
        lvl_algo=lvl_algo,
        stereo_link=bool(data.get("stereo_link", False)),
        stereo_link_strategy=str(sls),
        crossovers=xos,
        hpf_settings=hpf,
        house_freqs=hc_f,
        house_mags=hc_m,
        trans_width=float(_as_float_allow_zero(data.get("trans_width", None), 100.0)),
        bass_first_ai=bool(data.get("bass_first_ai", False)),
        bass_first_mode_max_hz=float(data.get("bass_first_mode_max_hz", 200.0) or 200.0),
        conf_pull_floor=float(_as_float_allow_zero(data.get("conf_pull_floor", None), 0.05)),
        conf_pull_ceil=float(_as_float_allow_zero(data.get("conf_pull_ceil", None), 0.95)),
        conf_pull_max_hz=_as_float_or_none(data.get("conf_pull_max_hz", None), 200.0),
        conf_pull_gamma_cut=float(_as_float_allow_zero(data.get("conf_pull_gamma_cut", None), 0.55)),
        conf_pull_gamma_boost=float(_as_float_allow_zero(data.get("conf_pull_gamma_boost", None), 0.35)),
        conf_pull_conf_smooth_sigma=float(_as_float_allow_zero(data.get("conf_pull_conf_smooth_sigma", None), 2.0)),
        conf_pull_bass_floor_hz=float(_as_float_allow_zero(data.get("conf_pull_bass_floor_hz", None), 120.0)),
        conf_pull_bass_floor_min=float(_as_float_allow_zero(data.get("conf_pull_bass_floor_min", None), 0.25)),
        conf_pull_bass_boost_floor_hz=float(_as_float_allow_zero(data.get("conf_pull_bass_boost_floor_hz", None), 200.0)),
        conf_pull_bass_boost_floor_min=float(_as_float_allow_zero(data.get("conf_pull_bass_boost_floor_min", None), 0.45)),
        conf_pull_bass_boost_restore=float(_as_float_allow_zero(data.get("conf_pull_bass_boost_restore", None), 0.55)),
        low_bass_cut_enable=bool(data.get("low_bass_cut_enable", True)),
        low_bass_cut_strength=float(max(0.0, min(1.0, _as_float_allow_zero(data.get("low_bass_cut_strength", None), 0.0)))),
        hybrid_iir_enabled=bool(data.get("hybrid_iir_enabled", False)),
        hybrid_iir_max_filters_per_channel=int(max(0, _as_float_allow_zero(data.get("hybrid_iir_max_filters_per_channel", None), 3))),
        hybrid_iir_min_freq_hz=float(max(1.0, _as_float_allow_zero(data.get("hybrid_iir_min_freq_hz", None), 20.0))),
        hybrid_iir_max_freq_hz=float(max(2.0, _as_float_allow_zero(data.get("hybrid_iir_max_freq_hz", None), 150.0))),
        hybrid_iir_min_peak_db=float(max(0.0, _as_float_allow_zero(data.get("hybrid_iir_min_peak_db", None), 4.0))),
        hybrid_iir_min_q=float(max(0.2, _as_float_allow_zero(data.get("hybrid_iir_min_q", None), 3.0))),
        hybrid_iir_max_q=float(max(0.2, _as_float_allow_zero(data.get("hybrid_iir_max_q", None), 12.0))),
        hybrid_iir_max_cut_db=float(max(0.0, _as_float_allow_zero(data.get("hybrid_iir_max_cut_db", None), 6.0))),
        hybrid_iir_min_confidence=float(max(0.0, min(1.0, _as_float_allow_zero(data.get("hybrid_iir_min_confidence", None), 0.65)))),
        hybrid_iir_min_gd_excess_ms=float(max(0.0, _as_float_allow_zero(data.get("hybrid_iir_min_gd_excess_ms", None), 15.0))),
        hybrid_iir_min_cut_priority=float(max(0.0, min(1.0, _as_float_allow_zero(data.get("hybrid_iir_min_cut_priority", None), 0.0)))),
        **({"stereo_auto_policy": stereo_auto_policy} if "stereo_auto_policy" in dataclass_fields else {}),
        **bass_smooth_kwargs,
        **bass_boost_cap_kwargs,
        **residual_authority_kwargs,
        **mixed_kwargs,
        **bass_integration_kwargs,
    )
    try:
        setattr(cfg, "auto_gain_margin_db", float(max(0.0, _as_float_allow_zero(data.get("gain", None), 0.0))))
    except (AttributeError, TypeError, ValueError):
        logger.debug("FilterConfig has no writable auto_gain_margin_db", exc_info=True)
        pass
    logger.info(f"UI raw: conf_pull_floor pin={data.get('conf_pull_floor')}, low_bass_cut_strength pin={data.get('low_bass_cut_strength')}")

    try:
        setattr(cfg, "enable_residual_pass", bool(data.get("enable_residual_pass", False)))
    except (AttributeError, TypeError, ValueError):
        logger.debug("FilterConfig has no writable enable_residual_pass", exc_info=True)
        pass

    try:
        setattr(cfg, "lvl_force_window", None)
        setattr(cfg, "lvl_force_offset_db", None)
    except (AttributeError, TypeError, ValueError):
        logger.debug("FilterConfig has no writable forced leveling fields", exc_info=True)
        pass

    _is_direct_dac = (
        bool(data.get("bass_integration_enable", False))
        and str(data.get("bass_integration_mode", "") or "").strip() == "direct_dac"
    )
    if _is_direct_dac or (is_auto_mode and bool(data.get("sub_integration_enable", False))):
        sub_xo_hz = float(data.get("sub_crossover_hz", 80.0) or 80.0)
        sub_xo_order = max(1, int(data.get("sub_crossover_slope", 24) or 24) // 6)
        sub_hpf_f = float(data.get("sub_hpf_freq", 20.0) or 20.0)
        sub_hpf_ord = max(1, int(data.get("sub_hpf_slope", 12) or 12) // 6)
        try:
            direct_sub_lpf_hz = float(data.get("direct_dac_sub_lpf_hz", sub_xo_hz) or sub_xo_hz)
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
            direct_sub_lpf_hz = sub_xo_hz
        if not math.isfinite(direct_sub_lpf_hz) or direct_sub_lpf_hz <= 0.0:
            direct_sub_lpf_hz = sub_xo_hz
        if _is_direct_dac:
            direct_sub_lpf_hz = max(float(sub_xo_hz), float(direct_sub_lpf_hz))

        if _is_direct_dac:
            main_hpf_f = 0.0
            main_hpf_order = int(sub_xo_order)
        else:
            existing_hpf_f = float((hpf or {}).get("freq", 0.0) or 0.0)
            main_hpf_f = max(existing_hpf_f, sub_xo_hz)
            main_hpf_order = sub_xo_order if main_hpf_f >= sub_xo_hz else int((hpf or {}).get("order", 2) or 2)
        cfg.hpf_settings = None if _is_direct_dac else {"enabled": True, "freq": float(main_hpf_f), "order": int(main_hpf_order)}
        cfg.sub_integration_enable = True
        cfg.sub_generate_ir = _is_direct_dac or bool(data.get("sub_generate_ir", False))
        cfg.sub_crossover_hz = float(sub_xo_hz)
        cfg.sub_crossover_order = int(sub_xo_order)
        if hasattr(cfg, "direct_dac_sub_lpf_hz"):
            cfg.direct_dac_sub_lpf_hz = float(direct_sub_lpf_hz)
        cfg.sub_hpf_freq = float(sub_hpf_f)
        cfg.sub_hpf_order = int(sub_hpf_ord)

    return cfg


__all__ = ['build_filter_config']
