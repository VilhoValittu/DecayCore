# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Typed config schema and compatibility projections.

The application still persists and exposes the historical flat dict shape.
This module owns the field registry behind that shape so defaults, UI pins,
mode policies, and runtime-only knobs do not drift across modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Literal

from ..app_paths import default_measurements_dir
from ..common.measurement_defaults import (
    DEFAULT_MEASUREMENT_DITHER_LEVEL_DB,
    DEFAULT_MEASUREMENT_SAMPLE_RATE,
    DEFAULT_OUTPUT_GAIN_DB,
    DEFAULT_SWEEP_END_HZ,
    DEFAULT_SWEEP_LENGTH_S,
    DEFAULT_SWEEP_START_HZ,
)
from ..ui_i18n import LAYOUT_MONO, LVL_ALGO_MEDIAN, LVL_MODE_AUTO
from .legacy_keys import CAMILLAFIR_AUTO_MODE

FieldKind = Literal["bool", "int", "float", "str", "choice", "any"]
CacheRelevance = Literal["dsp", "auto", "measurement", "ui", "runtime", "none"]

FS_OPTIONS = (44100, 48000, 88200, 96000, 176400, 192000, 352800, 384000)
TAPS_OPTIONS = (
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048576,
)
SLOPE_OPTIONS = (6, 12, 18, 24, 36, 48)
PLOT_SMOOTHING_LEVEL_OPTIONS = ("Psychoacoustic", 12, 24, 48, 96)
FILTER_WAV_FORMAT_OPTIONS = ("FLOAT32", "S32_LE", "S16_LE")
DEVICE_AUDIO_FORMAT_OPTIONS = ("S32_LE", "S16_LE")
IR_EXPORT_WINDOW_MODE_OPTIONS = ("auto", "rew_asym")
IR_EXPORT_WINDOW_SHAPE_OPTIONS = ("hann", "tukey")
STEREO_LINK_STRATEGY_OPTIONS = ("off", "auto", "hybrid", "shared")

CHOICE_OPTIONS_BY_KEY: dict[str, tuple[Any, ...]] = {
    "fs": FS_OPTIONS,
    "taps": TAPS_OPTIONS,
    "hpf_slope": SLOPE_OPTIONS,
    "xo1_s": SLOPE_OPTIONS,
    "xo2_s": SLOPE_OPTIONS,
    "xo3_s": SLOPE_OPTIONS,
    "xo4_s": SLOPE_OPTIONS,
    "xo5_s": SLOPE_OPTIONS,
    "plot_smoothing_level": PLOT_SMOOTHING_LEVEL_OPTIONS,
    "filter_wav_format": FILTER_WAV_FORMAT_OPTIONS,
    "device_audio_format": DEVICE_AUDIO_FORMAT_OPTIONS,
    "ir_export_window_mode": IR_EXPORT_WINDOW_MODE_OPTIONS,
    "ir_export_window_shape": IR_EXPORT_WINDOW_SHAPE_OPTIONS,
    "stereo_link_strategy": STEREO_LINK_STRATEGY_OPTIONS,
}

DEFAULT_CONFIG_ITEMS: tuple[tuple[str, Any], ...] = (
    ("fmt", "WAV"),
    ("layout", LAYOUT_MONO),
    ("fs", 44100),
    ("taps", 65536),
    ("mode", "AUTO"),
    ("auto_goal", "balanced"),
    ("auto_target_mode", "auto"),
    ("auto_mode_workers", 0),
    ("bass_integration_enable", False),
    ("bass_integration_mode", "direct_dac"),
    ("bass_integration_profile", "safe"),
    ("bass_integration_sub_combine_mode", "average"),
    ("avr_crossover_hz", 80.0),
    ("direct_dac_sub_lpf_hz", 80.0),
    ("bass_integration_sub_delay_ms", 0.0),
    ("bass_integration_sub_array_delay_ms", 0.0),
    ("bass_integration_sub1_delay_ms", 0.0),
    ("bass_integration_sub2_delay_ms", 0.0),
    ("bass_integration_main_l_delay_ms", 0.0),
    ("bass_integration_main_r_delay_ms", 0.0),
    ("bass_integration_sub_polarity_invert", False),
    ("bass_integration_sub_gain_trim_db", 0.0),
    ("bass_integration_alignment_auto_applied", False),
    ("bass_integration_alignment_reason", ""),
    ("bass_integration_allpass_auto_enable", False),
    ("bass_integration_allpass_freq_hz", 0.0),
    ("bass_integration_allpass_q", 0.707),
    ("bass_integration_allpass_auto_applied", False),
    ("auto_mode_optuna_multivariate", True),
    ("auto_mode_optuna_group", False),
    ("auto_mode_optuna_constant_liar", True),
    ("auto_mode_optuna_persistent_study", True),
    ("auto_mode_optuna_avoid_duplicates", True),
    ("filter_type", "Mixed"),
    ("gain", 0.0),
    ("hc_mode", "Harman6"),
    ("mag_correct", True),
    ("unsafe_raw_dsp", False),
    (CAMILLAFIR_AUTO_MODE, True),
    ("plot_smoothing_level", "Psychoacoustic"),
    ("filter_smooth", 96),
    ("bass_smooth_adaptive", True),
    ("bass_smooth_hz", 200.0),
    ("bass_smooth_sigma_scale", 1.4),
    ("bass_smooth_conf_floor", 0.3),
    ("bass_adaptive_isolation_mode", True),
    ("mid_refit_enable", True),
    ("mid_refit_hz_lo", 200.0),
    ("mid_refit_hz_hi", 2000.0),
    ("mid_refit_k", 0.45),
    ("mid_refit_smooth_oct", 0.60),
    ("mid_refit_conf_min_avg", 0.20),
    ("bass_boost_cap_enable", True),
    ("bass_boost_cap_hz", 200.0),
    ("bass_boost_cap_extra_db", 5.0),
    ("bass_boost_cap_conf_min", 0.55),
    ("bass_boost_post_restore_enable", True),
    ("bass_boost_post_restore_strength", 0.90),
    ("acoustic_authority_limits_enable", True),
    ("authority_boost_gamma", 1.35),
    ("authority_boost_min_frac", 0.05),
    ("authority_boost_min_cap_db", 0.0),
    ("authority_cut_gamma", 0.75),
    ("authority_cut_min_frac", 0.35),
    ("authority_cut_min_cap_db", 3.0),
    ("authority_caps_smooth_oct", 1.0 / 9.0),
    ("residual_pass_mode", "modal_polish"),
    ("residual_null_guard_enable", True),
    ("residual_null_guard_strength", 1.0),
    ("residual_modal_min_support", 0.45),
    ("residual_boost_authority_min", 0.40),
    ("residual_cut_authority_min", 0.35),
    ("residual_reflection_risk_max", 0.65),
    ("residual_null_risk_max_for_boost", 0.35),
    ("residual_null_risk_max_for_cut", 0.75),
    ("residual_max_boost_when_null_risk_db", 0.5),
    ("residual_max_boost_general_db", 2.0),
    ("residual_max_cut_general_db", 4.0),
    ("residual_authority_smooth_oct", 1.0 / 9.0),
    ("auto_voice_clarity_penalty_enable", True),
    ("auto_voice_clarity_penalty_weight", 1.0),
    ("auto_voice_band_lo_hz", 70.0),
    ("hybrid_iir_enabled", False),
    ("hybrid_iir_max_filters_per_channel", 3),
    ("hybrid_iir_min_freq_hz", 20.0),
    ("hybrid_iir_max_freq_hz", 200.0),
    ("hybrid_iir_min_peak_db", 4.0),
    ("hybrid_iir_min_q", 3.0),
    ("hybrid_iir_max_q", 12.0),
    ("hybrid_iir_max_cut_db", 6.0),
    ("hybrid_iir_min_confidence", 0.30),
    ("hybrid_iir_min_gd_excess_ms", 10.0),
    ("auto_voice_band_hi_hz", 180.0),
    ("fdw_cycles", 10.0),
    ("mag_c_min", 10.0),
    ("mag_c_max", 200.0),
    ("max_boost", 5.0),
    ("lvl_mode", LVL_MODE_AUTO),
    ("lvl_algo", LVL_ALGO_MEDIAN),
    ("lvl_manual_db", 0.0),
    ("manual_target_tilt_db_per_oct", 0.0),
    ("output_tilt_source", "off"),
    ("output_tilt_db_per_oct", 0.0),
    ("lvl_min", 300.0),
    ("lvl_max", 3000.0),
    ("normalize_opt", False),
    ("align_opt", True),
    ("multi_rate_opt", False),
    ("multi_rate_ultra_high_opt", False),
    ("reg_strength", 30.0),
    ("stereo_link", True),
    ("stereo_link_strategy", "auto"),
    ("exc_prot", True),
    ("exc_freq", 20.0),
    ("low_bass_cut_hz", 40.0),
    ("hpf_enable", False),
    ("hpf_freq", 20.0),
    ("hpf_slope", 24),
    ("measurement_library_dir", str(default_measurements_dir())),
    ("local_path_l", ""),
    ("local_path_r", ""),
    ("measurement_input_device", None),
    ("measurement_output_device", None),
    ("measurement_input_channel", 0),
    ("measurement_output_channel", 0),
    ("measurement_samplerate", DEFAULT_MEASUREMENT_SAMPLE_RATE),
    ("measurement_sweep_start_hz", DEFAULT_SWEEP_START_HZ),
    ("measurement_sweep_end_hz", DEFAULT_SWEEP_END_HZ),
    ("measurement_sweep_length_s", DEFAULT_SWEEP_LENGTH_S),
    ("measurement_output_gain_db", DEFAULT_OUTPUT_GAIN_DB),
    ("measurement_dither_level_db", DEFAULT_MEASUREMENT_DITHER_LEVEL_DB),
    ("measurement_source_path", ""),
    ("measurement_role", "left"),
    ("measurement_use_wasapi", False),
    ("measurement_mic_calibration_path", ""),
    ("measurement_mic_calibration_label", ""),
    ("local_path_l_main", ""),
    ("local_path_r_main", ""),
    ("local_path_l_sub", ""),
    ("local_path_r_sub", ""),
    ("xo1_f", None),
    ("xo1_s", 12),
    ("xo2_f", None),
    ("xo2_s", 12),
    ("xo3_f", None),
    ("xo3_s", 12),
    ("xo4_f", None),
    ("xo4_s", 12),
    ("xo5_f", None),
    ("xo5_s", 12),
    ("mixed_freq", 180.0),
    ("phase_limit", 400.0),
    ("phase_safe_2058", False),
    ("phase_authority_enable", True),
    ("phase_authority_gamma", 1.20),
    ("phase_authority_min_gain", 0.0),
    ("phase_authority_soft_floor", 0.20),
    ("phase_authority_smooth_oct", 1.0 / 6.0),
    ("phase_authority_disable_above_hz", 1200.0),
    ("phase_authority_warn_threshold", 0.35),
    ("excess_phase_strength", 0.9),
    ("low_freq_full_correction_hz", 140.0),
    ("high_freq_no_correction_hz", 900.0),
    ("phase_boundary_smooth_sigma_bins", 1.2),
    ("phase_tail_monotonic_enable", True),
    ("phase_tail_start_ratio", 0.72),
    ("phase_tail_abs_smooth_sigma_bins", 2.5),
    ("phase_tail_cosine_strength", 0.85),
    ("linear_phase_blend_start_ratio", 0.65),
    ("enable_ir_pre_energy_guard", True),
    ("pre_energy_ratio_max", 0.25),
    ("pre_energy_guard_strength", 0.8),
    ("max_pre_ringing_db", -35.0),
    ("max_excess_delay_ms", 2.5),
    ("gd_grad_limit_ms_per_oct", 30.0),
    ("ir_anchor_mode", "min_causal"),
    ("min_causal_ms", 80.0),
    ("auto_asym_left_ratio", 0.35),
    ("auto_asym_left_max_ms", 25.0),
    ("ir_window_right", 500.0),
    ("ir_window_left", 85.0),
    ("ir_export_window_mode", "auto"),
    ("ir_export_window_shape", "hann"),
    ("ir_export_tukey_alpha", 0.25),
    ("enable_tdc", True),
    ("tdc_strength", 50.0),
    ("enable_afdw", True),
    ("max_cut_db", 30.0),
    ("max_slope_db_per_oct", 24.0),
    ("max_slope_boost_db_per_oct", 0.0),
    ("max_slope_cut_db_per_oct", 0.0),
    ("df_smoothing", False),
    ("comparison_mode", True),
    ("ui_theme_dark", True),
    ("tdc_max_reduction_db", 9.0),
    ("tdc_slope_db_per_oct", 6.0),
    ("bass_first_ai", True),
    ("bass_first_mode_max_hz", 200.0),
    ("enable_channel_specific_auto_policy", False),
    ("channel_specific_policy_max_hz", 220.0),
    ("conf_pull_bass_boost_floor_hz", 200.0),
    ("conf_pull_bass_boost_floor_min", 0.55),
    ("conf_pull_bass_boost_restore", 0.70),
    ("debug_stage_stats", True),
)

AUTO_MODE_DEFAULT_CFG_TO_UI: dict[str, str] = {
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

UI_PIN_KEYS: tuple[str, ...] = (
    "mode", "auto_goal", "auto_target_mode", "auto_mode_workers", "fs", "taps", "filter_type", "mixed_freq", "gain", "hc_mode",
    "bass_integration_enable", "bass_integration_mode", "bass_integration_profile", "bass_integration_sub_combine_mode", "avr_crossover_hz",
    "bass_integration_sub_delay_ms", "bass_integration_sub_array_delay_ms",
    "bass_integration_sub1_delay_ms", "bass_integration_sub2_delay_ms",
    "bass_integration_main_l_delay_ms", "bass_integration_main_r_delay_ms",
    "bass_integration_sub_polarity_invert", "bass_integration_sub_gain_trim_db",
    "bass_integration_alignment_auto_applied", "bass_integration_alignment_reason",
    "bass_integration_allpass_auto_enable", "bass_integration_allpass_freq_hz", "bass_integration_allpass_q", "bass_integration_allpass_auto_applied",
    "sub_crossover_hz", "sub_crossover_slope", "sub_crossover_manual_override", "direct_dac_sub_lpf_hz", "sub_hpf_freq", "sub_hpf_slope",
    "mag_c_min", "mag_c_max", "max_boost", "max_cut_db", "max_slope_db_per_oct",
    "max_slope_boost_db_per_oct", "max_slope_cut_db_per_oct", "phase_limit", "mag_correct",
    "excess_phase_strength", "low_freq_full_correction_hz", "high_freq_no_correction_hz",
    "mixed_phase_budget_lf_deg", "mixed_phase_budget_hf_deg",
    "phase_budget_mode", "linear_excess_strength",
    "phase_conf_gain_floor", "phase_conf_gain_power",
    "phase_corr_clamp_lf_deg", "phase_corr_clamp_hf_deg", "max_excess_delay_cycles",
    "enable_ir_pre_energy_guard", "pre_energy_ratio_max", "pre_energy_guard_strength",
    "max_pre_ringing_db", "max_excess_delay_ms", "gd_grad_limit_ms_per_oct",
    "ir_anchor_mode", "min_causal_ms", "auto_asym_left_ratio", "auto_asym_left_max_ms",
    "lvl_mode", "reg_strength", "normalize_opt", "align_opt",
    "stereo_link", "stereo_link_strategy", "exc_prot", "exc_freq", "low_bass_cut_hz", "low_bass_cut_enable", "hpf_enable", "hpf_freq",
    "hpf_slope", "multi_rate_opt", "multi_rate_ultra_high_opt", "ir_window", "ir_window_left", "ir_window_right", "ir_export_window_mode", "ir_window_mode",
    "ir_export_window_shape", "ir_export_tukey_alpha",
    "measurement_library_dir",
    "local_path_l", "local_path_r",
    "measurement_input_device", "measurement_output_device", "measurement_input_channel", "measurement_output_channel",
    "measurement_samplerate", "measurement_sweep_start_hz", "measurement_sweep_end_hz",
    "measurement_sweep_length_s", "measurement_output_gain_db", "measurement_dither_level_db", "measurement_source_path", "measurement_role",
    "measurement_use_wasapi",
    "measurement_mic_calibration_path", "measurement_mic_calibration_label",
    "generated_measurement_l", "generated_measurement_r",
    "local_path_l_main", "local_path_r_main", "local_path_l_sub", "local_path_r_sub",
    "fmt", "layout", "lvl_manual_db",
    "manual_target_tilt_db_per_oct", "output_tilt_source", "output_tilt_db_per_oct",
    "lvl_min", "lvl_max", "lvl_algo", "fdw_cycles",
    "trans_width", "smoothing_level", "filter_smooth", "plot_smoothing_level",
    "bass_smooth_adaptive", "bass_smooth_hz", "bass_smooth_sigma_scale", "bass_smooth_conf_floor",
    "bass_adaptive_isolation_mode",
    "bass_boost_cap_enable", "bass_boost_cap_hz", "bass_boost_cap_extra_db", "bass_boost_cap_conf_min",
    "bass_boost_post_restore_enable", "bass_boost_post_restore_strength",
    "enable_tdc", "tdc_strength", "tdc_max_reduction_db",
    "tdc_slope_db_per_oct", "enable_afdw", "df_smoothing", "comparison_mode",
    "bass_first_ai", "bass_first_mode_max_hz",
    "enable_channel_specific_auto_policy", "channel_specific_policy_max_hz",
    "hybrid_iir_enabled", "hybrid_iir_max_filters_per_channel",
    "hybrid_iir_min_freq_hz", "hybrid_iir_max_freq_hz", "hybrid_iir_min_peak_db",
    "hybrid_iir_min_q", "hybrid_iir_max_q", "hybrid_iir_max_cut_db",
    "hybrid_iir_min_confidence", "hybrid_iir_min_gd_excess_ms",
    "local_path_house",
    "conf_pull_floor", "conf_pull_ceil", "conf_pull_max_hz",
    "conf_pull_gamma_cut", "conf_pull_gamma_boost",
    "conf_pull_conf_smooth_sigma",
    "conf_pull_bass_floor_hz", "conf_pull_bass_floor_min",
    "conf_pull_bass_boost_floor_hz", "conf_pull_bass_boost_floor_min",
    "conf_pull_bass_boost_restore",
    "low_bass_cut_strength", "auto_optimize_low_bass_cut", "hc_custom_file",
    "file_l", "file_r",
    "file_l_main", "file_r_main", "file_l_sub", "file_r_sub",
    "unsafe_raw_dsp",
    CAMILLAFIR_AUTO_MODE,
)

LIST_BOOL_KEYS: tuple[str, ...] = (
    "mag_correct", "normalize_opt", "align_opt", "multi_rate_opt", "multi_rate_ultra_high_opt",
    "stereo_link", "exc_prot", "hpf_enable", "df_smoothing",
    "comparison_mode", "bass_first_ai", "phase_safe_2058",
    "enable_tdc", "enable_afdw", "low_bass_cut_enable", "auto_optimize_low_bass_cut", "enable_ir_pre_energy_guard",
    "bass_smooth_adaptive",
    "bass_adaptive_isolation_mode",
    "bass_boost_cap_enable",
    "bass_boost_post_restore_enable",
    "enable_channel_specific_auto_policy",
    "hybrid_iir_enabled",
    "unsafe_raw_dsp",
    "bass_integration_enable",
    "bass_integration_sub_polarity_invert",
    "bass_integration_alignment_auto_applied",
    "bass_integration_allpass_auto_enable",
    "bass_integration_allpass_auto_applied",
    "sub_crossover_manual_override",
    CAMILLAFIR_AUTO_MODE,
)

HIDDEN_CONF_DEFAULTS_ADVANCED = {
    "conf_pull_floor": 0.05,
    "conf_pull_ceil": 0.85,
    "conf_pull_max_hz": 180.0,
    "conf_pull_gamma_cut": 0.45,
    "conf_pull_gamma_boost": 0.35,
    "conf_pull_bass_boost_floor_min": 0.55,
    "conf_pull_bass_boost_restore": 0.70,
    "low_bass_cut_strength": 0.0,
}

HIDDEN_CONF_DEFAULTS_BASIC_AUTO = {
    "conf_pull_floor": 0.05,
    "conf_pull_ceil": 0.85,
    "conf_pull_max_hz": 200.0,
    "conf_pull_gamma_cut": 0.45,
    "conf_pull_gamma_boost": 0.35,
    "conf_pull_bass_boost_floor_min": 0.55,
    "conf_pull_bass_boost_restore": 0.70,
    "low_bass_cut_strength": 0.0,
}

REQUEST_RUNTIME_DEFAULTS = {
    "bass_adaptive_isolation_mode": False,
    "bass_smooth_sigma_scale": 1.20,
    "bass_smooth_conf_floor": 0.25,
    "bass_smooth_w_gamma": 2.40,
    "bass_smooth_w_max": 0.45,
}

MODE_DEFAULTS_BASE: dict[str, dict[str, Any]] = {
    "BASIC": {
        "filter_type_str": "Mixed Phase",
        "global_gain_db": 0.0,
        "enable_mag_correction": True,
        "unsafe_raw_dsp": False,
        "mag_c_min": 25.0,
        "mag_c_max": 250.0,
        "max_boost_db": 3.0,
        "max_cut_db": 15.0,
        "phase_safe_2058": False,
        "phase_limit": 400.0,
        "plot_smoothing_level": "Psychoacoustic",
        "filter_smooth": 96,
        "fdw_cycles": 10.0,
        "reg_strength": 30.0,
        "max_slope_db_per_oct": 12.0,
        "max_slope_boost_db_per_oct": 6.0,
        "max_slope_cut_db_per_oct": 24.0,
        "df_smoothing": True,
        "enable_tdc": True,
        "tdc_strength": 50.0,
        "tdc_max_reduction_db": 9.0,
        "tdc_slope_db_per_oct": 6.0,
        "enable_afdw": True,
        "ir_export_window_mode": "auto",
        "ir_window_right": 500.0,
        "ir_window_left": 80.0,
        "mixed_split_freq": 180.0,
        "trans_width": 100.0,
        "bass_first_ai": True,
        "bass_first_mode_max_hz": 180.0,
        "lvl_mode": "Auto",
        "lvl_algo": "Median",
        "lvl_manual_db": 0.0,
        "manual_target_tilt_db_per_oct": 0.0,
        "output_tilt_source": "off",
        "lvl_min": 500.0,
        "lvl_max": 2000.0,
        "stereo_link": True,
        "stereo_link_strategy": "auto",
        "do_normalize": False,
        "exc_prot": True,
        "low_bass_cut_hz": 50.0,
        "low_bass_cut_enable": True,
    },
    "ADVANCED": {
        "filter_type_str": "Mixed Phase",
        "global_gain_db": 0.0,
        "enable_mag_correction": True,
        "unsafe_raw_dsp": False,
        "mag_c_min": 18.0,
        "mag_c_max": 230.0,
        "max_boost_db": 5.0,
        "max_cut_db": 24.0,
        "phase_safe_2058": False,
        "phase_limit": 320.0,
        "plot_smoothing_level": "Psychoacoustic",
        "filter_smooth": 96,
        "fdw_cycles": 10.0,
        "reg_strength": 18.0,
        "max_slope_db_per_oct": 24.0,
        "max_slope_boost_db_per_oct": 36.0,
        "max_slope_cut_db_per_oct": 0.0,
        "df_smoothing": False,
        "enable_tdc": True,
        "tdc_strength": 15.0,
        "tdc_max_reduction_db": 6.0,
        "tdc_slope_db_per_oct": 12.0,
        "enable_afdw": True,
        "ir_window_right": 500.0,
        "ir_window_left": 85.0,
        "ir_export_window_mode": "auto",
        "bass_first_ai": True,
        "bass_first_mode_max_hz": 200.0,
        "lvl_mode": "Auto",
        "lvl_algo": "Median",
        "lvl_manual_db": 0.0,
        "manual_target_tilt_db_per_oct": 0.0,
        "output_tilt_source": "off",
        "lvl_min": 200.0,
        "lvl_max": 3000.0,
        "stereo_link": True,
        "stereo_link_strategy": "auto",
        "mixed_split_freq": 180.0,
        "trans_width": 100.0,
        "do_normalize": False,
        "exc_prot": True,
        "low_bass_cut_hz": 40.0,
        "low_bass_cut_enable": False,
        "comparison_mode": True,
        "bass_adaptive_isolation_mode": False,
        "conf_pull_floor": 0.05,
        "conf_pull_ceil": 0.85,
        "conf_pull_max_hz": 180.0,
        "conf_pull_gamma_cut": 0.45,
        "conf_pull_gamma_boost": 0.35,
        "low_bass_cut_strength": 0.0,
        "ir_anchor_mode": "min_causal",
    },
}

MODE_CLAMPS_BASE: dict[str, dict[str, tuple[Any, Any]]] = {
    "BASIC": {
        "max_boost_db": (0.0, 4.0),
        "max_cut_db": (0.0, 15.0),
        "filter_smooth": (1, 96),
        "reg_strength": (10.0, 60.0),
        "ir_export_window_mode": ("auto", "auto"),
        "enable_tdc": (True, True),
        "tdc_strength": (0.0, 70.0),
        "tdc_max_reduction_db": (0.0, 12.0),
        "tdc_slope_db_per_oct": (0.0, 12.0),
        "mixed_split_freq": (100.0, 200.0),
        "fdw_cycles": (10.0, 15.0),
        "mag_c_min": (18.0, 300.0),
        "mag_c_max": (18.0, 300.0),
        "phase_limit": (200.0, 450.0),
        "low_bass_cut_hz": (20.0, 100.0),
        "low_bass_cut_enable": (True, True),
        "stereo_link": (True, True),
        "unsafe_raw_dsp": (False, False),
    },
    "ADVANCED": {},
}


@dataclass(frozen=True)
class ConfigFieldSpec:
    key: str
    default: Any = None
    kind: FieldKind = "any"
    choices: tuple[Any, ...] = ()
    persist: bool = True
    ui_pin: str | None = None
    filter_attr: str | None = None
    cache_relevance: CacheRelevance = "none"


@dataclass(frozen=True)
class AppConfigSnapshot:
    values: dict[str, Any] = field(default_factory=dict)

    def to_flat_dict(self) -> dict[str, Any]:
        return dict(self.values)


@dataclass(frozen=True)
class RunConfigSnapshot:
    values: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_flat_dict(cls, data: dict[str, Any] | None) -> "RunConfigSnapshot":
        return cls(values=normalize_flat_config(data or {}, include_runtime=True))

    def to_flat_dict(self) -> dict[str, Any]:
        return dict(self.values)


@dataclass(frozen=True)
class FilterConfigProjection:
    values: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_run_config(cls, snapshot: RunConfigSnapshot | "FilterConfigProjection" | dict[str, Any]) -> "FilterConfigProjection":
        if isinstance(snapshot, FilterConfigProjection):
            return snapshot
        if isinstance(snapshot, RunConfigSnapshot):
            return cls(values=snapshot.to_flat_dict())
        return cls(values=normalize_flat_config(snapshot or {}, include_runtime=True))

    def to_legacy_dict(self) -> dict[str, Any]:
        return dict(self.values)


def _infer_kind(default: Any, choices: tuple[Any, ...]) -> FieldKind:
    if choices:
        return "choice"
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, int) and not isinstance(default, bool):
        return "int"
    if isinstance(default, float):
        return "float"
    if isinstance(default, str):
        return "str"
    return "any"


def _default_specs() -> list[ConfigFieldSpec]:
    ui_keys = set(UI_PIN_KEYS)
    specs: list[ConfigFieldSpec] = []
    for key, default in DEFAULT_CONFIG_ITEMS:
        choices = CHOICE_OPTIONS_BY_KEY.get(key, ())
        specs.append(
            ConfigFieldSpec(
                key=key,
                default=default,
                kind=_infer_kind(default, choices),
                choices=choices,
                persist=key not in REQUEST_RUNTIME_DEFAULTS,
                ui_pin=key if key in ui_keys else None,
                filter_attr=_filter_attr_for_key(key),
                cache_relevance=_cache_relevance_for_key(key),
            )
        )
    known = {spec.key for spec in specs}
    for key in UI_PIN_KEYS:
        if key not in known:
            specs.append(
                ConfigFieldSpec(
                    key=key,
                    default=None,
                    kind="any",
                    persist=not _is_runtime_only_key(key),
                    ui_pin=key,
                    filter_attr=_filter_attr_for_key(key),
                    cache_relevance=_cache_relevance_for_key(key),
                )
            )
            known.add(key)
    for key, default in REQUEST_RUNTIME_DEFAULTS.items():
        if key not in known:
            specs.append(
                ConfigFieldSpec(
                    key=key,
                    default=default,
                    kind=_infer_kind(default, ()),
                    persist=False,
                    ui_pin=None,
                    filter_attr=_filter_attr_for_key(key),
                    cache_relevance="runtime",
                )
            )
    return specs


def _filter_attr_for_key(key: str) -> str | None:
    reverse = {ui_key: cfg_key for cfg_key, ui_key in AUTO_MODE_DEFAULT_CFG_TO_UI.items()}
    return reverse.get(key, key if key in {cfg_key for cfg_key in AUTO_MODE_DEFAULT_CFG_TO_UI} else None)


def _cache_relevance_for_key(key: str) -> CacheRelevance:
    if key.startswith("measurement_") or key.startswith("local_path") or key.startswith("file_"):
        return "measurement"
    if key.startswith("auto_mode_") or key in {"auto_goal", "auto_target_mode", CAMILLAFIR_AUTO_MODE}:
        return "auto"
    if key.startswith("ui_") or key in {"layout", "fmt"}:
        return "ui"
    if key in REQUEST_RUNTIME_DEFAULTS:
        return "runtime"
    if key in UI_PIN_KEYS or key in dict(DEFAULT_CONFIG_ITEMS):
        return "dsp"
    return "none"


def _is_runtime_only_key(key: str) -> bool:
    return (
        key.startswith("file_")
        or key.startswith("generated_measurement_")
        or key in {"auto_mode_compat_version", "unsafe_raw_dsp", "_config_version"}
        or key in REQUEST_RUNTIME_DEFAULTS
    )


FIELD_SPECS: tuple[ConfigFieldSpec, ...] = tuple(_default_specs())
FIELD_SPECS_BY_KEY: dict[str, ConfigFieldSpec] = {spec.key: spec for spec in FIELD_SPECS}

MODE_DEFAULTS: dict[str, dict[str, Any]] = {
    "BASIC": dict(MODE_DEFAULTS_BASE["BASIC"]),
    "ADVANCED": dict(MODE_DEFAULTS_BASE["ADVANCED"]),
}
MODE_DEFAULTS["AUTO"] = dict(MODE_DEFAULTS["ADVANCED"])
MODE_DEFAULTS["AUTO"]["stereo_link_strategy"] = "auto"

MODE_CLAMPS: dict[str, dict[str, tuple[Any, Any]]] = {
    "BASIC": dict(MODE_CLAMPS_BASE["BASIC"]),
    "ADVANCED": dict(MODE_CLAMPS_BASE["ADVANCED"]),
}
MODE_CLAMPS["AUTO"] = dict(MODE_CLAMPS["ADVANCED"])


def default_config_dict() -> dict[str, Any]:
    return dict(DEFAULT_CONFIG_ITEMS)


def normalize_filter_type(value: Any) -> str:
    try:
        ft = str(value or "").strip()
    except (AttributeError, TypeError, ValueError, KeyError, IndexError, RuntimeError, OSError):
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
    return "Mixed"


def coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off", ""):
            return False
    if isinstance(value, (list, tuple)):
        if not value:
            return False
        return coerce_bool(value[0], default)
    try:
        return bool(value)
    except (AttributeError, TypeError, ValueError):
        return bool(default)


def normalize_list_backed_booleans(data: dict[str, Any]) -> None:
    for key in LIST_BOOL_KEYS:
        if isinstance(data.get(key, None), list):
            data[key] = coerce_bool(data[key], False)


def _coerce_by_spec(value: Any, spec: ConfigFieldSpec) -> Any:
    if value is None:
        return spec.default
    if spec.kind == "bool":
        return coerce_bool(value, bool(spec.default))
    if spec.kind == "int":
        try:
            return int(float(value))
        except (TypeError, ValueError, OverflowError):
            return int(spec.default or 0)
    if spec.kind == "float":
        try:
            return float(value)
        except (TypeError, ValueError, OverflowError):
            return float(spec.default or 0.0)
    if spec.kind == "str":
        try:
            return str(value)
        except (TypeError, ValueError):
            return str(spec.default or "")
    if spec.kind == "choice":
        return normalize_choice_value(value, options=spec.choices, default=spec.default)
    return value


def normalize_flat_config(data: dict[str, Any], *, include_runtime: bool = False) -> dict[str, Any]:
    out = default_config_dict()
    if include_runtime:
        out.update(REQUEST_RUNTIME_DEFAULTS)
    src = dict(data or {})
    normalize_list_backed_booleans(src)
    for key, value in src.items():
        spec = FIELD_SPECS_BY_KEY.get(key)
        out[key] = _coerce_by_spec(value, spec) if spec is not None else value
    if "filter_type" in out:
        out["filter_type"] = normalize_filter_type(out.get("filter_type"))
    return out


def persistable_config_dict(data: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in dict(data or {}).items():
        if value is None:
            continue
        spec = FIELD_SPECS_BY_KEY.get(str(key))
        if spec is not None and not spec.persist:
            continue
        if _is_runtime_only_key(str(key)):
            continue
        clean[key] = value
    return clean


def _parse_legacy_choice_index(value: Any) -> int | None:
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
    except (AttributeError, TypeError, ValueError):
        return None
    if not text:
        return None
    digits = text[1:] if text[0] in "+-" else text
    if not digits.isdigit():
        return None
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def normalize_choice_value(value: Any, *, options: tuple[Any, ...], default: Any) -> Any:
    if value in options:
        return value
    try:
        raw = str(value).strip()
    except (AttributeError, TypeError, ValueError):
        raw = ""
    if raw:
        for option in options:
            if raw.casefold() == str(option).strip().casefold():
                return option
    index = _parse_legacy_choice_index(value)
    if index is not None and 0 <= index < len(options):
        return options[index]
    return default


def normalize_choice_fields(data: dict[str, Any], default_conf: dict[str, Any]) -> None:
    for key, options in CHOICE_OPTIONS_BY_KEY.items():
        data[key] = normalize_choice_value(
            data.get(key, default_conf.get(key)),
            options=options,
            default=default_conf.get(key, FIELD_SPECS_BY_KEY.get(key, ConfigFieldSpec(key)).default),
        )


def app_config_snapshot(data: dict[str, Any] | None = None) -> AppConfigSnapshot:
    return AppConfigSnapshot(values=normalize_flat_config(data or {}, include_runtime=False))


def run_config_snapshot(data: dict[str, Any] | None = None) -> RunConfigSnapshot:
    return RunConfigSnapshot.from_flat_dict(data or {})


def snapshot_field_names(snapshot: AppConfigSnapshot | RunConfigSnapshot | FilterConfigProjection) -> tuple[str, ...]:
    return tuple(field_obj.name for field_obj in fields(snapshot))


__all__ = [
    "AUTO_MODE_DEFAULT_CFG_TO_UI",
    "AppConfigSnapshot",
    "CHOICE_OPTIONS_BY_KEY",
    "ConfigFieldSpec",
    "DEVICE_AUDIO_FORMAT_OPTIONS",
    "FIELD_SPECS",
    "FIELD_SPECS_BY_KEY",
    "FILTER_WAV_FORMAT_OPTIONS",
    "FS_OPTIONS",
    "FilterConfigProjection",
    "HIDDEN_CONF_DEFAULTS_ADVANCED",
    "HIDDEN_CONF_DEFAULTS_BASIC_AUTO",
    "IR_EXPORT_WINDOW_MODE_OPTIONS",
    "IR_EXPORT_WINDOW_SHAPE_OPTIONS",
    "LIST_BOOL_KEYS",
    "MODE_CLAMPS",
    "MODE_DEFAULTS",
    "PLOT_SMOOTHING_LEVEL_OPTIONS",
    "REQUEST_RUNTIME_DEFAULTS",
    "RunConfigSnapshot",
    "SLOPE_OPTIONS",
    "STEREO_LINK_STRATEGY_OPTIONS",
    "TAPS_OPTIONS",
    "UI_PIN_KEYS",
    "app_config_snapshot",
    "coerce_bool",
    "default_config_dict",
    "normalize_choice_fields",
    "normalize_choice_value",
    "normalize_filter_type",
    "normalize_flat_config",
    "normalize_list_backed_booleans",
    "persistable_config_dict",
    "run_config_snapshot",
]
