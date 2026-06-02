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
import os

logger = logging.getLogger("DecayCore")

from ..app_paths import default_measurements_dir
from .legacy_keys import CAMILLAFIR_AUTO_MODE
from ..common.measurement_defaults import (
    DEFAULT_MEASUREMENT_SAMPLE_RATE,
    DEFAULT_OUTPUT_GAIN_DB,
    DEFAULT_SWEEP_END_HZ,
    DEFAULT_SWEEP_LENGTH_S,
    DEFAULT_SWEEP_START_HZ,
)
from ..ui_i18n import (
    LAYOUT_MONO,
    LVL_ALGO_MEDIAN,
    LVL_MODE_AUTO,
    normalize_layout_value,
    normalize_lvl_algo_value,
    normalize_lvl_mode_value,
    normalize_output_tilt_source_value,
)

CONFIG_FILE = "config.json"

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
    try:
        ft = str(value or "").strip()
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
        ft = ""
    ft_l = ft.lower()
    if "asym" in ft_l:
        return "Asymmetric"
    if "mixed" in ft_l:
        return "Mixed"
    if "minimum" in ft_l or "minphase" in ft_l or ft_l == "min":
        return "Minimum"
    if "linear" in ft_l:
        return "Linear"
    return "Asymmetric"


def _coerce_legacy_boolean_lists(saved: dict) -> None:
    for key in [
        "mag_correct",
        "normalize_opt",
        "align_opt",
        "multi_rate_opt",
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
            saved[key] = bool(saved[key])


def _migrate_lvl_manual_db(saved: dict) -> None:
    try:
        if "lvl_manual_db" in saved:
            value = float(saved.get("lvl_manual_db"))
            if 40.0 <= value <= 110.0:
                saved["lvl_manual_db"] = float(value - 75.0)
    except _RECOVERABLE_CONFIG_EXCEPTIONS:
        logger.exception("lvl_manual_db migration")


def _normalize_saved_filter_type(saved: dict, default_conf: dict) -> None:
    try:
        saved["filter_type"] = _normalize_filter_type(
            saved.get("filter_type", default_conf.get("filter_type"))
        )
    except _RECOVERABLE_CONFIG_EXCEPTIONS:
        saved["filter_type"] = str(default_conf.get("filter_type", "Asymmetric"))


def _load_and_merge_saved_config(default_conf: dict) -> bool:
    saved_mode_explicit = False
    if not os.path.exists(CONFIG_FILE):
        return saved_mode_explicit
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
        if not isinstance(saved, dict):
            return saved_mode_explicit

        _coerce_legacy_boolean_lists(saved)
        _migrate_lvl_manual_db(saved)
        _normalize_saved_filter_type(saved, default_conf)

        saved_mode_explicit = saved.get("mode", None) not in (None, "")
        saved["layout"] = normalize_layout_value(saved.get("layout", default_conf.get("layout")))
        saved["lvl_mode"] = normalize_lvl_mode_value(saved.get("lvl_mode", default_conf.get("lvl_mode")))
        saved["lvl_algo"] = normalize_lvl_algo_value(saved.get("lvl_algo", default_conf.get("lvl_algo")))
        default_conf.update(saved)
    except _RECOVERABLE_CONFIG_EXCEPTIONS:
        logger.exception("config load and merge")
    return saved_mode_explicit


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


def load_config() -> dict:
    default_conf = {
        "fmt": "WAV",
        "layout": LAYOUT_MONO,
        "fs": 44100,
        "taps": 65536,
        "mode": "AUTO",
        "auto_goal": "balanced",
        "auto_target_mode": "auto",
        "auto_mode_workers": 0,
        "bass_integration_enable": False,
        "bass_integration_mode": "direct_dac",
        "bass_integration_profile": "safe",
        "bass_integration_sub_combine_mode": "average",
        "avr_crossover_hz": 80.0,
        "direct_dac_sub_lpf_hz": 80.0,
        "bass_integration_sub_delay_ms": 0.0,
        "bass_integration_sub_polarity_invert": False,
        "bass_integration_sub_gain_trim_db": 0.0,
        "bass_integration_alignment_auto_applied": False,
        "bass_integration_alignment_reason": "",
        "bass_integration_allpass_auto_enable": False,
        "bass_integration_allpass_freq_hz": 0.0,
        "bass_integration_allpass_q": 0.707,
        "bass_integration_allpass_auto_applied": False,
        "auto_mode_optuna_multivariate": True,
        "auto_mode_optuna_group": False,
        "auto_mode_optuna_constant_liar": True,
        "auto_mode_optuna_persistent_study": True,
        "auto_mode_optuna_avoid_duplicates": True,
        "filter_type": "Asymmetric",
        "gain": 0.0,
        "hc_mode": "Harman6",
        "mag_correct": True,
        "unsafe_raw_dsp": False,
        CAMILLAFIR_AUTO_MODE: True,
        "plot_smoothing_level": "Psychoacoustic",
        "filter_smooth": 96,
        "bass_smooth_adaptive": True,
        "bass_smooth_hz": 200.0,
        "bass_smooth_sigma_scale": 1.4,
        "bass_smooth_conf_floor": 0.3,
        "bass_adaptive_isolation_mode": True,
        "mid_refit_enable": True,
        "mid_refit_hz_lo": 200.0,
        "mid_refit_hz_hi": 2000.0,
        "mid_refit_k": 0.45,
        "mid_refit_smooth_oct": 0.60,
        "mid_refit_conf_min_avg": 0.20,
        "bass_boost_cap_enable": True,
        "bass_boost_cap_hz": 200.0,
        "bass_boost_cap_extra_db": 5.0,
        "bass_boost_cap_conf_min": 0.55,
        "bass_boost_post_restore_enable": True,
        "bass_boost_post_restore_strength": 0.90,
        "acoustic_authority_limits_enable": True,
        "authority_boost_gamma": 1.35,
        "authority_boost_min_frac": 0.05,
        "authority_boost_min_cap_db": 0.0,
        "authority_cut_gamma": 0.75,
        "authority_cut_min_frac": 0.35,
        "authority_cut_min_cap_db": 3.0,
        "authority_caps_smooth_oct": 1.0 / 9.0,
        "residual_pass_mode": "modal_polish",
        "residual_null_guard_enable": True,
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
        "auto_voice_clarity_penalty_enable": True,
        "auto_voice_clarity_penalty_weight": 1.0,
        "auto_voice_band_lo_hz": 70.0,
        "hybrid_iir_enabled": False,
        "hybrid_iir_max_filters_per_channel": 3,
        "hybrid_iir_min_freq_hz": 20.0,
        "hybrid_iir_max_freq_hz": 150.0,
        "hybrid_iir_min_peak_db": 4.0,
        "hybrid_iir_min_q": 3.0,
        "hybrid_iir_max_q": 12.0,
        "hybrid_iir_max_cut_db": 6.0,
        "hybrid_iir_min_confidence": 0.65,
        "hybrid_iir_min_gd_excess_ms": 15.0,
        "auto_voice_band_hi_hz": 180.0,
        "fdw_cycles": 10.0,
        "mag_c_min": 10.0,
        "mag_c_max": 200.0,
        "max_boost": 5.0,
        "lvl_mode": LVL_MODE_AUTO,
        "lvl_algo": LVL_ALGO_MEDIAN,
        "lvl_manual_db": 0.0,
        "manual_target_tilt_db_per_oct": 0.0,
        "output_tilt_source": "off",
        "output_tilt_db_per_oct": 0.0,
        "lvl_min": 300.0,
        "lvl_max": 3000.0,
        "normalize_opt": False,
        "align_opt": True,
        "multi_rate_opt": False,
        "reg_strength": 30.0,
        "stereo_link": True,
        "stereo_link_strategy": "auto",
        "exc_prot": True,
        "exc_freq": 20.0,
        "low_bass_cut_hz": 40.0,
        "hpf_enable": False,
        "hpf_freq": 20.0,
        "hpf_slope": 24,
        "measurement_library_dir": str(default_measurements_dir()),
        "local_path_l": "",
        "local_path_r": "",
        "measurement_input_device": None,
        "measurement_output_device": None,
        "measurement_input_channel": 0,
        "measurement_output_channel": 0,
        "measurement_samplerate": DEFAULT_MEASUREMENT_SAMPLE_RATE,
        "measurement_sweep_start_hz": DEFAULT_SWEEP_START_HZ,
        "measurement_sweep_end_hz": DEFAULT_SWEEP_END_HZ,
        "measurement_sweep_length_s": DEFAULT_SWEEP_LENGTH_S,
        "measurement_output_gain_db": DEFAULT_OUTPUT_GAIN_DB,
        "measurement_source_path": "",
        "measurement_role": "left",
        "measurement_use_wasapi": False,
        "measurement_mic_calibration_path": "",
        "measurement_mic_calibration_label": "",
        "local_path_l_main": "",
        "local_path_r_main": "",
        "local_path_l_sub": "",
        "local_path_r_sub": "",
        "xo1_f": None,
        "xo1_s": 12,
        "xo2_f": None,
        "xo2_s": 12,
        "xo3_f": None,
        "xo3_s": 12,
        "xo4_f": None,
        "xo4_s": 12,
        "xo5_f": None,
        "xo5_s": 12,
        "mixed_freq": 180.0,
        "phase_limit": 400.0,
        "phase_safe_2058": False,
        "phase_authority_enable": True,
        "phase_authority_gamma": 1.20,
        "phase_authority_min_gain": 0.0,
        "phase_authority_soft_floor": 0.20,
        "phase_authority_smooth_oct": 1.0 / 6.0,
        "phase_authority_disable_above_hz": 1200.0,
        "phase_authority_warn_threshold": 0.35,
        "excess_phase_strength": 0.9,
        "low_freq_full_correction_hz": 140.0,
        "high_freq_no_correction_hz": 900.0,
        "phase_boundary_smooth_sigma_bins": 1.2,
        "phase_tail_monotonic_enable": True,
        "phase_tail_start_ratio": 0.72,
        "phase_tail_abs_smooth_sigma_bins": 2.5,
        "phase_tail_cosine_strength": 0.85,
        "linear_phase_blend_start_ratio": 0.65,
        "enable_ir_pre_energy_guard": True,
        "pre_energy_ratio_max": 0.25,
        "pre_energy_guard_strength": 0.8,
        "max_pre_ringing_db": -35.0,
        "max_excess_delay_ms": 2.5,
        "gd_grad_limit_ms_per_oct": 30.0,
        "ir_anchor_mode": "min_causal",
        "min_causal_ms": 80.0,
        "auto_asym_left_ratio": 0.35,
        "auto_asym_left_max_ms": 25.0,
        "ir_window_right": 500.0,
        "ir_window_left": 85.0,
        "ir_export_window_mode": "auto",
        "ir_export_window_shape": "hann",
        "ir_export_tukey_alpha": 0.25,
        "enable_tdc": True,
        "tdc_strength": 50.0,
        "enable_afdw": True,
        "max_cut_db": 30.0,
        "max_slope_db_per_oct": 24.0,
        "max_slope_boost_db_per_oct": 0.0,
        "max_slope_cut_db_per_oct": 0.0,
        "df_smoothing": False,
        "comparison_mode": True,
        "tdc_max_reduction_db": 9.0,
        "tdc_slope_db_per_oct": 6.0,
        "bass_first_ai": True,
        "bass_first_mode_max_hz": 200.0,
        "enable_channel_specific_auto_policy": False,
        "channel_specific_policy_max_hz": 220.0,
        "conf_pull_bass_boost_floor_hz": 200.0,
        "conf_pull_bass_boost_floor_min": 0.45,
        "conf_pull_bass_boost_restore": 0.55,
        "debug_stage_stats": True,
    }

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
    default_conf["output_tilt_source"] = normalize_output_tilt_source_value(default_conf.get("output_tilt_source", "off"))

    return default_conf


def save_config(data: dict) -> None:
    try:
        clean_data = {
            k: v for k, v in (data or {}).items()
            if (
                not str(k).startswith("file_")
                and not str(k).startswith("generated_measurement_")
                and str(k) != "auto_mode_compat_version"
                and str(k) != "unsafe_raw_dsp"
                and v is not None
            )
        }
        clean_data["filter_type"] = _normalize_filter_type(
            clean_data.get("filter_type", "Asymmetric")
        )
        clean_data["layout"] = normalize_layout_value(clean_data.get("layout", LAYOUT_MONO))
        clean_data["lvl_mode"] = normalize_lvl_mode_value(clean_data.get("lvl_mode", LVL_MODE_AUTO))
        clean_data["lvl_algo"] = normalize_lvl_algo_value(clean_data.get("lvl_algo", LVL_ALGO_MEDIAN))
        clean_data["output_tilt_source"] = normalize_output_tilt_source_value(clean_data.get("output_tilt_source", "off"))
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
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
