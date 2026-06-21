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

import logging

import numpy as np

_logger = logging.getLogger(__name__)

from ..auto_mode_profile import profiled_section

from ...config.models import StereoAutoPolicyConfig, StereoResolvedAutoPolicies
from ...dsp.quality_metrics import (
    band_lr_mismatch_change_from_stats,
    band_lr_mismatch_rms_from_stats,
    normalized_policy_divergence_score,
    worst_channel_relief_db,
)
from .. import shared

from .metric_penalties import (
    _auto_bass_boost_metrics_from_stats, _auto_bass_integration_penalty,
    _auto_bass_preference_bonus, _auto_bass_under_target_metrics_from_stats,
    _auto_channel_overfit_metrics_from_stats, _auto_correction_sharpness_metrics_from_stats,
    _auto_dip_fill_risk_metrics_from_stats, _auto_merge_bass_boost_metrics,
    _auto_merge_bass_under_target_metrics, _auto_merge_correction_sharpness_metrics,
    _auto_merge_dip_fill_risk_metrics, _auto_merge_target_tracking_metrics,
    _auto_target_tracking_metrics_from_stats, _auto_target_tracking_penalty,
)
from .penalty_voice import (
    merge_voice_clarity_metrics,
    voice_clarity_lr_metrics_from_stats,
    voice_clarity_metrics_from_stats,
)

def score_bass_integration(result, l_st, r_st, *, base_data, net_boost_max, peak_lo, peak_hi, exc_penalty) -> dict:
    with profiled_section("bass_score.integration_penalty"):
        bass_integration_penalty, bass_feasibility_penalty, bass_dbg = _auto_bass_integration_penalty(
            result,
            base_data=base_data,
            net_boost_max_db=net_boost_max,
        )

    goal_norm = shared._auto_goal(base_data)
    with profiled_section("bass_score.target_tracking"):
        target_tracking_l = _auto_target_tracking_metrics_from_stats(l_st)
        target_tracking_r = _auto_target_tracking_metrics_from_stats(r_st)
    target_tracking_metrics = _auto_merge_target_tracking_metrics(target_tracking_l, target_tracking_r)
    target_tracking_penalty = _auto_target_tracking_penalty(target_tracking_metrics)
    with profiled_section("bass_score.bass_under_target"):
        bass_under_target_l = _auto_bass_under_target_metrics_from_stats(l_st)
        bass_under_target_r = _auto_bass_under_target_metrics_from_stats(r_st)
    bass_under_target_metrics = _auto_merge_bass_under_target_metrics(bass_under_target_l, bass_under_target_r)
    bass_under_target_penalty = 0.0
    if goal_norm == shared.AUTO_MODE_GOAL_FLAT:
        bass_under_target_penalty = float(
            np.clip(
                shared._auto_safe_float(
                    bass_under_target_metrics.get("bass_under_target_penalty", 0.0),
                    0.0,
                ),
                0.0,
                5.0,
            )
        )
        target_tracking_penalty = float(np.clip(target_tracking_penalty + bass_under_target_penalty, 0.0, 16.0))
    with profiled_section("bass_score.bass_boost"):
        bass_boost_l = _auto_bass_boost_metrics_from_stats(l_st)
        bass_boost_r = _auto_bass_boost_metrics_from_stats(r_st)
    bass_boost_metrics = _auto_merge_bass_boost_metrics(bass_boost_l, bass_boost_r)
    _lf_boost_l = shared._auto_safe_float(l_st.get("lf_boost_max_db", 0.0), 0.0)
    _lf_boost_r = shared._auto_safe_float(r_st.get("lf_boost_max_db", 0.0), 0.0)
    _effective_bass_boost = float(max(
        shared._auto_safe_float(bass_boost_metrics.get("bass_boost_20_200_db", 0.0), 0.0),
        shared._auto_safe_float(bass_boost_metrics.get("bass_boost_peak_20_200_db", 0.0), 0.0),
        float(_lf_boost_l),
        float(_lf_boost_r),
    ))
    bass_preference_bonus = _auto_bass_preference_bonus(
        bass_boost_db=_effective_bass_boost,
        target_tracking_penalty=target_tracking_penalty,
        exc_penalty=exc_penalty,
        bass_integration_penalty=bass_integration_penalty,
        bass_feasibility_penalty=bass_feasibility_penalty,
    )
    if goal_norm == shared.AUTO_MODE_GOAL_FLAT:
        bass_preference_bonus = float(np.clip(bass_preference_bonus * 2.5, 0.0, 7.5))
    with profiled_section("bass_score.sharpness"):
        sharpness_l = _auto_correction_sharpness_metrics_from_stats(l_st, lo_hz=float(peak_lo), hi_hz=float(peak_hi))
        sharpness_r = _auto_correction_sharpness_metrics_from_stats(r_st, lo_hz=float(peak_lo), hi_hz=float(peak_hi))
    sharpness_metrics = _auto_merge_correction_sharpness_metrics(sharpness_l, sharpness_r)
    correction_sharpness_penalty = float(
        np.clip(
            shared._auto_safe_float(sharpness_metrics.get("correction_sharpness_penalty", 0.0), 0.0),
            0.0,
            8.0,
        )
    )
    with profiled_section("bass_score.dip_fill"):
        dip_fill_l = _auto_dip_fill_risk_metrics_from_stats(l_st, lo_hz=float(peak_lo), hi_hz=float(peak_hi))
        dip_fill_r = _auto_dip_fill_risk_metrics_from_stats(r_st, lo_hz=float(peak_lo), hi_hz=float(peak_hi))
    dip_fill_metrics = _auto_merge_dip_fill_risk_metrics(dip_fill_l, dip_fill_r)
    dip_fill_risk_penalty = float(
        np.clip(
            shared._auto_safe_float(dip_fill_metrics.get("dip_fill_risk_penalty", 0.0), 0.0),
            0.0,
            10.0,
        )
    )
    with profiled_section("bass_score.channel_overfit"):
        channel_overfit_metrics = _auto_channel_overfit_metrics_from_stats(l_st, r_st, lo_hz=float(peak_lo), hi_hz=float(peak_hi))
    channel_overfit_penalty = float(
        np.clip(
            shared._auto_safe_float(channel_overfit_metrics.get("channel_overfit_penalty", 0.0), 0.0),
            0.0,
            8.0,
        )
    )
    voice_lo = shared._auto_safe_float((base_data or {}).get("auto_voice_band_lo_hz", 70.0), 70.0)
    voice_hi = shared._auto_safe_float((base_data or {}).get("auto_voice_band_hi_hz", 180.0), 180.0)
    if not (np.isfinite(voice_lo) and np.isfinite(voice_hi) and float(voice_hi) > float(voice_lo)):
        voice_lo = 70.0
        voice_hi = 180.0
    with profiled_section("bass_score.voice_clarity"):
        voice_l = voice_clarity_metrics_from_stats(l_st, lo_hz=float(voice_lo), hi_hz=float(voice_hi))
        voice_r = voice_clarity_metrics_from_stats(r_st, lo_hz=float(voice_lo), hi_hz=float(voice_hi))
        voice_lr = voice_clarity_lr_metrics_from_stats(l_st, r_st, lo_hz=float(voice_lo), hi_hz=float(voice_hi))
    voice_metrics = merge_voice_clarity_metrics(voice_l, voice_r, voice_lr)
    voice_enabled = bool((base_data or {}).get("auto_voice_clarity_penalty_enable", True))
    voice_weight = shared._auto_safe_float((base_data or {}).get("auto_voice_clarity_penalty_weight", 1.0), 1.0)
    voice_clarity_penalty = 0.0
    if bool(voice_enabled):
        voice_clarity_penalty = float(
            np.clip(
                shared._auto_safe_float(voice_metrics.get("voice_clarity_penalty", 0.0), 0.0)
                * max(0.0, float(voice_weight)),
                0.0,
                6.0,
            )
        )
    return {
        "bass_integration_penalty": bass_integration_penalty,
        "bass_feasibility_penalty": bass_feasibility_penalty,
        "bass_dbg": bass_dbg,
        "target_tracking_penalty": target_tracking_penalty,
        "bass_under_target_penalty": bass_under_target_penalty,
        "bass_under_target_metrics": bass_under_target_metrics,
        "bass_under_target_l": bass_under_target_l,
        "bass_under_target_r": bass_under_target_r,
        "target_tracking_metrics": target_tracking_metrics,
        "target_tracking_l": target_tracking_l,
        "target_tracking_r": target_tracking_r,
        "bass_boost_metrics": bass_boost_metrics,
        "bass_boost_l": bass_boost_l,
        "bass_boost_r": bass_boost_r,
        "bass_preference_bonus": bass_preference_bonus,
        "sharpness_metrics": sharpness_metrics,
        "correction_sharpness_penalty": correction_sharpness_penalty,
        "dip_fill_metrics": dip_fill_metrics,
        "dip_fill_risk_penalty": dip_fill_risk_penalty,
        "channel_overfit_metrics": channel_overfit_metrics,
        "channel_overfit_penalty": channel_overfit_penalty,
        "voice_metrics": voice_metrics,
        "voice_clarity_penalty": voice_clarity_penalty,
    }


def score_stereo_policy(l_st, r_st, *, base_data) -> dict:
    stereo_policy_cfg = StereoAutoPolicyConfig.from_dict(base_data or {})
    resolved_policies = StereoResolvedAutoPolicies.from_dict(
        dict(base_data or {}).get("_stereo_resolved_auto_policies")
    )
    stereo_policy_active = bool(
        isinstance(resolved_policies, StereoResolvedAutoPolicies) and resolved_policies.is_effective()
    )
    stereo_split_hz = float(
        max(
            20.0,
            shared._auto_safe_float(
                getattr(resolved_policies, "split_hz", stereo_policy_cfg.channel_specific_policy_max_hz),
                stereo_policy_cfg.channel_specific_policy_max_hz,
            ),
        )
    )
    stereo_above_hi_hz = float(
        max(
            stereo_split_hz + 20.0,
            min(
                3000.0,
                shared._auto_safe_float((base_data or {}).get("mag_c_max", 3000.0), 3000.0),
            ),
        )
    )
    stereo_lr_mismatch_below = float("nan")
    stereo_lr_mismatch_above = float("nan")
    stereo_lr_mismatch_below_delta = float("nan")
    stereo_lr_mismatch_above_delta = float("nan")
    phantom_center_change_db = float("nan")
    phantom_center_change_delta_db = float("nan")
    policy_divergence_score = 0.0
    stereo_coherence_penalty = 0.0
    phantom_center_stability_penalty = 0.0
    policy_divergence_penalty = 0.0
    asymmetry_budget_overflow_penalty = 0.0
    worst_channel_relief_bonus = 0.0
    shared_preference_bias = float(stereo_policy_cfg.shared_preference_bias) if stereo_policy_active else 0.0
    stereo_worst_channel_relief_db = float("nan")
    stereo_policy_gate_failed = False
    if stereo_policy_active:
        stereo_lr_mismatch_below = float(
            band_lr_mismatch_rms_from_stats(
                l_st,
                r_st,
                lo_hz=20.0,
                hi_hz=stereo_split_hz,
                corrected=True,
            )
        )
        stereo_lr_mismatch_above = float(
            band_lr_mismatch_rms_from_stats(
                l_st,
                r_st,
                lo_hz=stereo_split_hz,
                hi_hz=stereo_above_hi_hz,
                corrected=True,
            )
        )
        phantom_center_change_db = float(
            band_lr_mismatch_change_from_stats(
                l_st,
                r_st,
                lo_hz=max(200.0, stereo_split_hz),
                hi_hz=stereo_above_hi_hz,
            )
        )
        policy_divergence_score = float(
            normalized_policy_divergence_score(
                resolved_policies,
                max_confidence_pull_delta=stereo_policy_cfg.max_confidence_pull_delta,
                max_tdc_strength_delta=stereo_policy_cfg.max_tdc_strength_delta,
                max_tdc_max_reduction_delta_db=stereo_policy_cfg.max_tdc_max_reduction_delta_db,
                max_bass_first_mode_max_hz_delta=stereo_policy_cfg.max_bass_first_mode_max_hz_delta,
                max_low_bass_cut_strength_delta=stereo_policy_cfg.max_low_bass_cut_strength_delta,
                max_excess_phase_strength_delta=stereo_policy_cfg.max_excess_phase_strength_delta,
            )
        )
        shared_l_st = dict((base_data or {}).get("_stereo_shared_l_st", {}) or {})
        shared_r_st = dict((base_data or {}).get("_stereo_shared_r_st", {}) or {})
        shared_lr_mismatch_below = float("nan")
        shared_lr_mismatch_above = float("nan")
        shared_phantom_center_change_db = float("nan")
        if shared_l_st and shared_r_st:
            shared_lr_mismatch_below = float(
                band_lr_mismatch_rms_from_stats(
                    shared_l_st,
                    shared_r_st,
                    lo_hz=20.0,
                    hi_hz=stereo_split_hz,
                    corrected=True,
                )
            )
            shared_lr_mismatch_above = float(
                band_lr_mismatch_rms_from_stats(
                    shared_l_st,
                    shared_r_st,
                    lo_hz=stereo_split_hz,
                    hi_hz=stereo_above_hi_hz,
                    corrected=True,
                )
            )
            shared_phantom_center_change_db = float(
                band_lr_mismatch_change_from_stats(
                    shared_l_st,
                    shared_r_st,
                    lo_hz=max(200.0, stereo_split_hz),
                    hi_hz=stereo_above_hi_hz,
                )
            )
            stereo_worst_channel_relief_db = float(
                worst_channel_relief_db(
                    shared_l_st,
                    shared_r_st,
                    l_st,
                    r_st,
                    lo_hz=20.0,
                    hi_hz=stereo_split_hz,
                )
            )
        if np.isfinite(stereo_lr_mismatch_below) and np.isfinite(shared_lr_mismatch_below):
            stereo_lr_mismatch_below_delta = float(stereo_lr_mismatch_below) - float(shared_lr_mismatch_below)
        if np.isfinite(stereo_lr_mismatch_above) and np.isfinite(shared_lr_mismatch_above):
            stereo_lr_mismatch_above_delta = float(stereo_lr_mismatch_above) - float(shared_lr_mismatch_above)
        if np.isfinite(phantom_center_change_db) and np.isfinite(shared_phantom_center_change_db):
            phantom_center_change_delta_db = float(phantom_center_change_db) - float(shared_phantom_center_change_db)
        below_guard_value = (
            float(stereo_lr_mismatch_below_delta)
            if np.isfinite(stereo_lr_mismatch_below_delta)
            else float(stereo_lr_mismatch_below)
        )
        above_guard_value = (
            float(stereo_lr_mismatch_above_delta)
            if np.isfinite(stereo_lr_mismatch_above_delta)
            else float(stereo_lr_mismatch_above)
        )
        phantom_guard_value = (
            float(phantom_center_change_delta_db)
            if np.isfinite(phantom_center_change_delta_db)
            else float(phantom_center_change_db)
        )
        if np.isfinite(stereo_lr_mismatch_below):
            stereo_coherence_penalty += float(stereo_policy_cfg.stereo_coherence_weight) * max(
                0.0,
                float(below_guard_value) - float(stereo_policy_cfg.max_lr_predicted_delta_db_below_split),
            ) * 0.60
        if np.isfinite(stereo_lr_mismatch_above):
            stereo_coherence_penalty += float(stereo_policy_cfg.stereo_coherence_weight) * max(
                0.0,
                float(above_guard_value) - float(stereo_policy_cfg.max_lr_predicted_delta_db_above_split),
            ) * 2.20
        if np.isfinite(phantom_center_change_db):
            phantom_center_stability_penalty = float(
                stereo_policy_cfg.phantom_center_stability_weight
            ) * max(0.0, float(phantom_guard_value)) * 1.50
        policy_divergence_penalty = float(stereo_policy_cfg.policy_divergence_weight) * max(
            0.0,
            float(policy_divergence_score),
        )
        asymmetry_budget_overflow_penalty = float(stereo_policy_cfg.asymmetry_budget_weight) * max(
            0.0,
            float(policy_divergence_score) - float(stereo_policy_cfg.max_policy_divergence_score),
        ) * 4.0
        if np.isfinite(stereo_worst_channel_relief_db):
            worst_channel_relief_bonus = min(1.5, max(0.0, float(stereo_worst_channel_relief_db)) * 0.60)
        stereo_policy_gate_failed = bool(
            (
                np.isfinite(below_guard_value)
                and float(below_guard_value)
                > float(stereo_policy_cfg.max_lr_predicted_delta_db_below_split) + 1.0
            )
            or (
                np.isfinite(above_guard_value)
                and float(above_guard_value)
                > float(stereo_policy_cfg.max_lr_predicted_delta_db_above_split) + 0.25
            )
            or (
                np.isfinite(policy_divergence_score)
                and float(policy_divergence_score)
                > float(stereo_policy_cfg.max_policy_divergence_score) + 0.10
            )
            or (
                np.isfinite(phantom_guard_value)
                and float(phantom_guard_value) > 0.60
            )
        )
    return {
        "stereo_policy_active": stereo_policy_active,
        "stereo_policy_gate_failed": stereo_policy_gate_failed,
        "stereo_split_hz": stereo_split_hz,
        "stereo_above_hi_hz": stereo_above_hi_hz,
        "stereo_lr_mismatch_below": stereo_lr_mismatch_below,
        "stereo_lr_mismatch_above": stereo_lr_mismatch_above,
        "stereo_lr_mismatch_below_delta": stereo_lr_mismatch_below_delta,
        "stereo_lr_mismatch_above_delta": stereo_lr_mismatch_above_delta,
        "phantom_center_change_db": phantom_center_change_db,
        "phantom_center_change_delta_db": phantom_center_change_delta_db,
        "policy_divergence_score": policy_divergence_score,
        "stereo_coherence_penalty": stereo_coherence_penalty,
        "phantom_center_stability_penalty": phantom_center_stability_penalty,
        "policy_divergence_penalty": policy_divergence_penalty,
        "asymmetry_budget_overflow_penalty": asymmetry_budget_overflow_penalty,
        "worst_channel_relief_bonus": worst_channel_relief_bonus,
        "shared_preference_bias": shared_preference_bias,
        "stereo_worst_channel_relief_db": stereo_worst_channel_relief_db,
    }


__all__ = ["score_bass_integration", "score_stereo_policy"]
