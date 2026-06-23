# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

from dataclasses import dataclass, field
from typing import Optional


def _bool_value(value, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off", ""):
            return False
    try:
        return bool(value)
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


def _float_value(value, default: float) -> float:
    try:
        parsed = float(value)
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
    if parsed != parsed:
        return float(default)
    return float(parsed)


def _float_or_none(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        parsed = float(value)
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
        return None
    if parsed != parsed:
        return None
    return float(parsed)


@dataclass
class StereoAutoPolicyConfig:
    enable_channel_specific_auto_policy: bool = True
    channel_specific_policy_max_hz: float = 220.0
    channel_specific_refine_trials: int = 32
    channel_specific_seed_from_asymmetry: bool = True
    allow_per_channel_confidence_pull: bool = True
    allow_per_channel_tdc: bool = True
    allow_per_channel_bass_first_mode_max_hz: bool = True
    allow_per_channel_low_bass_cut_strength: bool = True
    allow_per_channel_lf_phase_restraint: bool = True
    max_confidence_pull_delta: float = 0.20
    max_tdc_strength_delta: float = 20.0
    max_tdc_max_reduction_delta_db: float = 2.0
    max_bass_first_mode_max_hz_delta: float = 30.0
    max_low_bass_cut_strength_delta: float = 0.35
    max_excess_phase_strength_delta: float = 0.15
    stereo_coherence_weight: float = 0.6
    phantom_center_stability_weight: float = 0.6
    policy_divergence_weight: float = 0.25
    asymmetry_budget_weight: float = 0.5
    max_lr_predicted_delta_db_below_split: float = 4.0
    max_lr_predicted_delta_db_above_split: float = 1.5
    max_policy_divergence_score: float = 1.4
    min_worst_channel_improvement_db: float = 0.015
    shared_preference_bias: float = 0.0

    @classmethod
    def from_dict(cls, data: dict | None) -> "StereoAutoPolicyConfig":
        src = dict(data or {})
        return cls(
            enable_channel_specific_auto_policy=_bool_value(
                src.get("enable_channel_specific_auto_policy", False),
                False,
            ),
            channel_specific_policy_max_hz=max(
                20.0,
                _float_value(src.get("channel_specific_policy_max_hz", 220.0), 220.0),
            ),
            channel_specific_refine_trials=max(
                0,
                int(
                    round(
                        _float_value(src.get("channel_specific_refine_trials", 32), 32)
                    )
                ),
            ),
            channel_specific_seed_from_asymmetry=_bool_value(
                src.get("channel_specific_seed_from_asymmetry", True),
                True,
            ),
            allow_per_channel_confidence_pull=_bool_value(
                src.get("allow_per_channel_confidence_pull", True),
                True,
            ),
            allow_per_channel_tdc=_bool_value(
                src.get("allow_per_channel_tdc", True),
                True,
            ),
            allow_per_channel_bass_first_mode_max_hz=_bool_value(
                src.get("allow_per_channel_bass_first_mode_max_hz", True),
                True,
            ),
            allow_per_channel_low_bass_cut_strength=_bool_value(
                src.get("allow_per_channel_low_bass_cut_strength", True),
                True,
            ),
            allow_per_channel_lf_phase_restraint=_bool_value(
                src.get("allow_per_channel_lf_phase_restraint", False),
                False,
            ),
            max_confidence_pull_delta=max(
                0.0,
                _float_value(src.get("max_confidence_pull_delta", 0.20), 0.20),
            ),
            max_tdc_strength_delta=max(
                0.0,
                _float_value(src.get("max_tdc_strength_delta", 20.0), 20.0),
            ),
            max_tdc_max_reduction_delta_db=max(
                0.0,
                _float_value(src.get("max_tdc_max_reduction_delta_db", 2.0), 2.0),
            ),
            max_bass_first_mode_max_hz_delta=max(
                0.0,
                _float_value(src.get("max_bass_first_mode_max_hz_delta", 30.0), 30.0),
            ),
            max_low_bass_cut_strength_delta=max(
                0.0,
                _float_value(src.get("max_low_bass_cut_strength_delta", 0.35), 0.35),
            ),
            max_excess_phase_strength_delta=max(
                0.0,
                _float_value(src.get("max_excess_phase_strength_delta", 0.15), 0.15),
            ),
            stereo_coherence_weight=max(
                0.0,
                _float_value(src.get("stereo_coherence_weight", 0.6), 0.6),
            ),
            phantom_center_stability_weight=max(
                0.0,
                _float_value(src.get("phantom_center_stability_weight", 0.6), 0.6),
            ),
            policy_divergence_weight=max(
                0.0,
                _float_value(src.get("policy_divergence_weight", 0.25), 0.25),
            ),
            asymmetry_budget_weight=max(
                0.0,
                _float_value(src.get("asymmetry_budget_weight", 0.5), 0.5),
            ),
            max_lr_predicted_delta_db_below_split=max(
                0.0,
                _float_value(
                    src.get("max_lr_predicted_delta_db_below_split", 4.0), 4.0
                ),
            ),
            max_lr_predicted_delta_db_above_split=max(
                0.0,
                _float_value(
                    src.get("max_lr_predicted_delta_db_above_split", 1.5), 1.5
                ),
            ),
            max_policy_divergence_score=max(
                0.0,
                _float_value(src.get("max_policy_divergence_score", 1.4), 1.4),
            ),
            min_worst_channel_improvement_db=max(
                0.0,
                _float_value(src.get("min_worst_channel_improvement_db", 0.015), 0.015),
            ),
            shared_preference_bias=max(
                0.0,
                _float_value(src.get("shared_preference_bias", 0.0), 0.0),
            ),
        )

    def to_dict(self) -> dict[str, bool | float | int]:
        return {
            "enable_channel_specific_auto_policy": bool(
                self.enable_channel_specific_auto_policy
            ),
            "channel_specific_policy_max_hz": float(
                self.channel_specific_policy_max_hz
            ),
            "channel_specific_refine_trials": int(self.channel_specific_refine_trials),
            "channel_specific_seed_from_asymmetry": bool(
                self.channel_specific_seed_from_asymmetry
            ),
            "allow_per_channel_confidence_pull": bool(
                self.allow_per_channel_confidence_pull
            ),
            "allow_per_channel_tdc": bool(self.allow_per_channel_tdc),
            "allow_per_channel_bass_first_mode_max_hz": bool(
                self.allow_per_channel_bass_first_mode_max_hz
            ),
            "allow_per_channel_low_bass_cut_strength": bool(
                self.allow_per_channel_low_bass_cut_strength
            ),
            "allow_per_channel_lf_phase_restraint": bool(
                self.allow_per_channel_lf_phase_restraint
            ),
            "max_confidence_pull_delta": float(self.max_confidence_pull_delta),
            "max_tdc_strength_delta": float(self.max_tdc_strength_delta),
            "max_tdc_max_reduction_delta_db": float(
                self.max_tdc_max_reduction_delta_db
            ),
            "max_bass_first_mode_max_hz_delta": float(
                self.max_bass_first_mode_max_hz_delta
            ),
            "max_low_bass_cut_strength_delta": float(
                self.max_low_bass_cut_strength_delta
            ),
            "max_excess_phase_strength_delta": float(
                self.max_excess_phase_strength_delta
            ),
            "stereo_coherence_weight": float(self.stereo_coherence_weight),
            "phantom_center_stability_weight": float(
                self.phantom_center_stability_weight
            ),
            "policy_divergence_weight": float(self.policy_divergence_weight),
            "asymmetry_budget_weight": float(self.asymmetry_budget_weight),
            "max_lr_predicted_delta_db_below_split": float(
                self.max_lr_predicted_delta_db_below_split
            ),
            "max_lr_predicted_delta_db_above_split": float(
                self.max_lr_predicted_delta_db_above_split
            ),
            "max_policy_divergence_score": float(self.max_policy_divergence_score),
            "min_worst_channel_improvement_db": float(
                self.min_worst_channel_improvement_db
            ),
            "shared_preference_bias": float(self.shared_preference_bias),
        }


@dataclass
class ResolvedChannelAutoPolicy:
    conf_pull_floor: float | None = None
    tdc_strength: float | None = None
    tdc_max_reduction_db: float | None = None
    bass_first_mode_max_hz: float | None = None
    low_bass_cut_strength: float | None = None
    excess_phase_strength: float | None = None

    @classmethod
    def from_dict(cls, data: dict | None) -> "ResolvedChannelAutoPolicy":
        src = dict(data or {})
        return cls(
            conf_pull_floor=_float_or_none(src.get("conf_pull_floor")),
            tdc_strength=_float_or_none(src.get("tdc_strength")),
            tdc_max_reduction_db=_float_or_none(src.get("tdc_max_reduction_db")),
            bass_first_mode_max_hz=_float_or_none(src.get("bass_first_mode_max_hz")),
            low_bass_cut_strength=_float_or_none(src.get("low_bass_cut_strength")),
            excess_phase_strength=_float_or_none(src.get("excess_phase_strength")),
        )

    def to_dict(self) -> dict[str, float]:
        out: dict[str, float] = {}
        if self.conf_pull_floor is not None:
            out["conf_pull_floor"] = float(self.conf_pull_floor)
        if self.tdc_strength is not None:
            out["tdc_strength"] = float(self.tdc_strength)
        if self.tdc_max_reduction_db is not None:
            out["tdc_max_reduction_db"] = float(self.tdc_max_reduction_db)
        if self.bass_first_mode_max_hz is not None:
            out["bass_first_mode_max_hz"] = float(self.bass_first_mode_max_hz)
        if self.low_bass_cut_strength is not None:
            out["low_bass_cut_strength"] = float(self.low_bass_cut_strength)
        if self.excess_phase_strength is not None:
            out["excess_phase_strength"] = float(self.excess_phase_strength)
        return out

    def effective_value(
        self, key: str, shared: "ResolvedChannelAutoPolicy | None" = None
    ):
        value = getattr(self, str(key), None)
        if value is not None:
            return value
        if isinstance(shared, ResolvedChannelAutoPolicy):
            return getattr(shared, str(key), None)
        return None

    def is_empty(self) -> bool:
        return all(
            getattr(self, key) is None
            for key in (
                "conf_pull_floor",
                "tdc_strength",
                "tdc_max_reduction_db",
                "bass_first_mode_max_hz",
                "low_bass_cut_strength",
                "excess_phase_strength",
            )
        )


@dataclass
class StereoResolvedAutoPolicies:
    split_hz: float = 220.0
    shared: ResolvedChannelAutoPolicy = field(default_factory=ResolvedChannelAutoPolicy)
    left: ResolvedChannelAutoPolicy = field(default_factory=ResolvedChannelAutoPolicy)
    right: ResolvedChannelAutoPolicy = field(default_factory=ResolvedChannelAutoPolicy)

    @classmethod
    def from_dict(cls, data: dict | None) -> Optional["StereoResolvedAutoPolicies"]:
        if not isinstance(data, dict) or not data:
            return None
        return cls(
            split_hz=max(20.0, _float_value(data.get("split_hz", 220.0), 220.0)),
            shared=ResolvedChannelAutoPolicy.from_dict(data.get("shared")),
            left=ResolvedChannelAutoPolicy.from_dict(data.get("left")),
            right=ResolvedChannelAutoPolicy.from_dict(data.get("right")),
        )

    def to_dict(self) -> dict[str, float | dict[str, float]]:
        return {
            "split_hz": float(self.split_hz),
            "shared": self.shared.to_dict(),
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }

    def is_effective(self) -> bool:
        return not (self.left.is_empty() and self.right.is_empty())


@dataclass
class FilterConfig:

    fs: int = 44100
    num_taps: int = 65536
    filter_type_str: str = "Mixed Phase"
    global_gain_db: float = 0.0

    mag_c_min: float = 10.0
    mag_c_max: float = 200.0
    max_boost_db: float = 9.0
    max_cut_db: float = 15.0
    phase_limit: float = 1000.0
    phase_safe_2058: bool = False
    enable_mag_correction: bool = True
    unsafe_raw_dsp: bool = False

    plot_smoothing_level: str | int = "Psychoacoustic"
    filter_smooth: int = 96
    fdw_cycles: float = 10.0
    reg_strength: float = 30.0
    max_slope_db_per_oct: float = 24.0
    max_slope_boost_db_per_oct: float = 0.0
    max_slope_cut_db_per_oct: float = 0.0
    df_smoothing: bool = False
    bass_smooth_adaptive: bool = True
    bass_smooth_hz: float = 200.0
    bass_smooth_sigma_scale: float = 1.4
    bass_smooth_conf_floor: float = 0.3
    bass_adaptive_isolation_mode: bool = False
    mid_refit_enable: bool = True
    mid_refit_hz_lo: float = 200.0
    mid_refit_hz_hi: float = 2000.0
    mid_refit_k: float = 0.45
    mid_refit_smooth_oct: float = 0.60
    mid_refit_conf_min_avg: float = 0.20
    bass_boost_cap_enable: bool = True
    bass_boost_cap_hz: float = 200.0
    bass_boost_cap_extra_db: float = 5.0
    bass_boost_cap_conf_min: float = 0.30
    bass_boost_post_restore_enable: bool = True
    bass_boost_post_restore_strength: float = 0.90
    acoustic_authority_limits_enable: bool = True
    authority_boost_gamma: float = 1.35
    authority_boost_min_frac: float = 0.05
    authority_boost_min_cap_db: float = 0.0
    authority_cut_gamma: float = 0.75
    authority_cut_min_frac: float = 0.35
    authority_cut_min_cap_db: float = 3.0
    authority_caps_smooth_oct: float = 1.0 / 9.0
    residual_pass_mode: str = "modal_polish"
    residual_null_guard_enable: bool = True
    residual_null_guard_strength: float = 1.0
    residual_modal_min_support: float = 0.45
    residual_boost_authority_min: float = 0.40
    residual_cut_authority_min: float = 0.35
    residual_reflection_risk_max: float = 0.65
    residual_null_risk_max_for_boost: float = 0.35
    residual_null_risk_max_for_cut: float = 0.75
    residual_max_boost_when_null_risk_db: float = 0.5
    residual_max_boost_general_db: float = 2.0
    residual_max_cut_general_db: float = 4.0
    residual_authority_smooth_oct: float = 1.0 / 9.0
    auto_voice_clarity_penalty_enable: bool = True
    auto_voice_clarity_penalty_weight: float = 1.0
    auto_voice_band_lo_hz: float = 70.0
    auto_voice_band_hi_hz: float = 180.0
    hybrid_iir_enabled: bool = False
    hybrid_iir_max_filters_per_channel: int = 3
    hybrid_iir_min_freq_hz: float = 20.0
    hybrid_iir_max_freq_hz: float = 200.0
    hybrid_iir_min_peak_db: float = 4.0
    hybrid_iir_min_q: float = 3.0
    hybrid_iir_max_q: float = 12.0
    hybrid_iir_max_cut_db: float = 6.0
    hybrid_iir_min_confidence: float = 0.30
    hybrid_iir_min_gd_excess_ms: float = 10.0
    hybrid_iir_min_cut_priority: float = 0.0

    comparison_mode: bool = True
    comparison_ref_fs: int = 44100
    comparison_ref_taps: int = 65536
    auto_goal: str = "balanced"

    enable_tdc: bool = True
    tdc_strength: float = 50.0
    tdc_max_reduction_db: float = 9.0
    tdc_slope_db_per_oct: float = 6.0
    enable_afdw: bool = True

    ir_export_window_mode: str = "auto"
    ir_export_window_shape: str = "hann"
    ir_export_tukey_alpha: float = 0.25
    ir_anchor_mode: str = "min_causal"
    min_causal_ms: float = 80.0
    auto_asym_left_ratio: float = 0.35
    auto_asym_left_max_ms: float = 25.0
    ir_window_ms: float = 500.0
    ir_window_ms_left: float = 85.0
    ir_window_right: float = 500.0
    ir_window_left: float = 85.0
    mixed_split_freq: float = 180.0
    trans_width: float = 100.0
    mixed_transition_mode: str = "width_based"
    mixed_confidence_blend_enable: bool = False
    mixed_confidence_power: float = 1.5
    mixed_phase_budget_lf_deg: float = 40.0
    mixed_phase_budget_hf_deg: float = 22.5
    mixed_min_tilt_comp_enable: bool = True
    excess_phase_strength: float = 0.9
    phase_budget_mode: str = "unified"
    linear_excess_strength: float = 0.9
    phase_conf_gain_floor: float = 0.20
    phase_conf_gain_power: float = 1.0
    phase_corr_clamp_lf_deg: float = 540.0
    phase_corr_clamp_hf_deg: float = 90.0
    max_excess_delay_cycles: float = 1.0
    low_freq_full_correction_hz: float = 140.0
    high_freq_no_correction_hz: float = 900.0
    phase_boundary_smooth_sigma_bins: float = 1.2
    phase_tail_monotonic_enable: bool = True
    phase_tail_start_ratio: float = 0.72
    phase_tail_abs_smooth_sigma_bins: float = 2.5
    phase_tail_cosine_strength: float = 0.85
    linear_phase_blend_start_ratio: float = 0.65
    enable_ir_pre_energy_guard: bool = True
    pre_energy_ratio_max: float = 0.25
    pre_energy_guard_strength: float = 0.8
    max_pre_ringing_db: float = -35.0
    max_excess_delay_ms: float = 2.5
    gd_grad_limit_ms_per_oct: float = 30.0
    bass_first_ai: bool = False
    bass_first_mode_max_hz: float = 200.0
    bass_first_smooth_floor_lo: float = 0.75
    bass_first_smooth_floor_hi: float = 0.35
    bass_first_k_mode_cut: float = 0.6
    bass_first_k_mode_boost: float = 0.9
    is_wav_source: bool = False

    bass_integration_enable: bool = False
    bass_integration_mode: str = "direct_dac"
    bass_integration_profile: str = "safe"
    bass_integration_sub_combine_mode: str = "average"
    avr_crossover_hz: float = 80.0
    avr_crossover_hz_l: float | None = (
        None  # per-channel HPF override; None = use avr_crossover_hz
    )
    avr_crossover_hz_r: float | None = None
    bass_integration_guard_lo_ratio: float = 0.60
    bass_integration_guard_hi_ratio: float = 1.40

    hpf_settings: dict | None = None
    lpf_settings: dict | None = None
    crossovers: list[dict] = field(default_factory=list)
    house_freqs: list[float] | None = None
    house_mags: list[float] | None = None

    lvl_mode: str = "Auto"
    lvl_algo: str = "Median"
    lvl_manual_db: float = 0.0
    manual_target_tilt_db_per_oct: float = 0.0
    output_tilt_source: str = "off"
    output_tilt_db_per_oct: float = 0.0
    lvl_min: float = 200.0
    lvl_max: float = 3000.0
    stereo_link: bool = False
    stereo_link_strategy: str = "auto"
    lvl_force_window: tuple[float, float] | None = None
    lvl_force_offset_db: float | None = None

    do_normalize: bool = False
    exc_prot: bool = False
    exc_freq: float = 40.0

    sub_integration_enable: bool = False
    sub_generate_ir: bool = False
    sub_crossover_hz: float = 80.0
    sub_crossover_order: int = 4
    direct_dac_sub_lpf_hz: float = 80.0
    bass_integration_sub_delay_ms: float = 0.0
    bass_integration_sub_array_delay_ms: float = 0.0
    bass_integration_sub1_delay_ms: float = 0.0
    bass_integration_sub2_delay_ms: float = 0.0
    bass_integration_main_l_delay_ms: float = 0.0
    bass_integration_main_r_delay_ms: float = 0.0
    bass_integration_sub_gain_trim_db: float = 0.0
    bass_integration_sub_polarity_invert: bool = False
    sub_hpf_freq: float = 20.0
    sub_hpf_order: int = 2
    low_bass_cut_hz: float = 20.0
    room_volume_m3: float = 40.0

    conf_pull_floor: float = 0.05
    conf_pull_ceil: float = 0.85
    conf_pull_max_hz: float | None = 200.0
    conf_pull_gamma_cut: float = 0.45
    conf_pull_gamma_boost: float = 0.35

    conf_pull_conf_smooth_sigma: float = 2.0
    conf_pull_bass_floor_hz: float = 120.0
    conf_pull_bass_floor_min: float = 0.25
    conf_pull_bass_boost_floor_hz: float = 200.0
    conf_pull_bass_boost_floor_min: float = 0.55
    conf_pull_bass_boost_restore: float = 0.70

    low_bass_cut_enable: bool = True
    low_bass_cut_strength: float = 0.0
    auto_optimize_low_bass_cut: bool = True
    stereo_auto_policy: StereoAutoPolicyConfig = field(
        default_factory=StereoAutoPolicyConfig
    )
    stereo_resolved_auto_policies: StereoResolvedAutoPolicies | None = None
