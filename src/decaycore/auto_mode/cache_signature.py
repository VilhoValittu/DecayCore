# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Cache signature, version constants, seed helpers, and public re-exports."""

from __future__ import annotations

import logging
import hashlib
import json
import random

import numpy as np

from .cache_io import (
    _AUTO_CACHE_LOCK as _AUTO_CACHE_LOCK,
    _AUTO_CACHE_RUNTIME,
    _AUTO_CACHE_STATS,
    _auto_cache_guard,
    _auto_cache_load as _auto_cache_load,
    _auto_cache_save as _auto_cache_save,
    _auto_cache_stats_snapshot as _auto_cache_stats_snapshot,
)
from .cache_get_put import (
    _auto_cache_get_best as _auto_cache_get_best,
    _auto_cache_get_best_target as _auto_cache_get_best_target,
    _auto_cache_get_entry as _auto_cache_get_entry,
    _auto_cache_get_target_for_measurements as _auto_cache_get_target_for_measurements,
    _auto_cache_get_target_for_measurements_global as _auto_cache_get_target_for_measurements_global,
    _auto_cache_put_best as _auto_cache_put_best,
    _auto_cache_put_target_for_measurements as _auto_cache_put_target_for_measurements,
    _auto_cache_put_target_for_measurements_global as _auto_cache_put_target_for_measurements_global,
)
from .cache_lastused import (
    _auto_cache_get_last_used_best as _auto_cache_get_last_used_best,
    _auto_cache_put_last_used_best as _auto_cache_put_last_used_best,
)
from .cache_measurement_sig import (
    _auto_get_measurement_signature,
    _auto_measurement_metadata_identity,
    _auto_measurement_signature as _auto_measurement_signature,
    _auto_search_measurement_identity,
)
from .cache_paths import (
    _auto_cache_compat_token as _auto_cache_compat_token,
    _auto_cache_filename as _auto_cache_filename,
    _auto_cache_path,
    get_auto_mode_cache_path as get_auto_mode_cache_path,
)
from .cache_structure import (
    _auto_cache_bucket as _auto_cache_bucket,
    _auto_cache_bucket_template as _auto_cache_bucket_template,
    _auto_cache_empty as _auto_cache_empty,
    _auto_compat_version as _auto_compat_version,
)
from .cache_synth_target import (
    _SYNTH_TARGET_CACHE,
    _SYNTH_TARGET_CACHE_LOCK,
    _SYNTH_TARGET_MISS as _SYNTH_TARGET_MISS,
    _synth_target_cache_key as _synth_target_cache_key,
    get_or_build_synth_target as get_or_build_synth_target,
)
from .shared_parts import (
    AUTO_MODE_CACHE_SCHEMA_VERSION,
    AUTO_MODE_COMPAT_VERSION as AUTO_MODE_COMPAT_VERSION,
    _auto_goal,
    _auto_goal_norm as _auto_goal_norm,
    _auto_hash_array,
    _auto_safe_float,
)
from ..dsp.hybrid_iir import HYBRID_IIR_POLICY_VERSION
from ..dsp.hpf_policy import HPF_IIR_TAP_THRESHOLD, hpf_settings_should_use_iir

logger = logging.getLogger("DecayCore")

def _auto_cache_resolve_path(*, compat_version: str | None = None) -> str:
    if compat_version is None:
        return _auto_cache_path()
    try:
        return _auto_cache_path(compat_version=compat_version)
    except TypeError:
        # Tests may monkeypatch _auto_cache_path with a zero-arg lambda.
        return _auto_cache_path()


_BASS_ALLPASS_ALGO_V = 1
_BASS_INTEGRATION_COMBINE_ALGO_V = 2
_BASS_INTEGRATION_ALGO_V = 6
_DIRECT_DAC_SUB_TARGET_POLICY_V = 1
_AUTO_LF_ROLLOFF_POLICY_V = 5
_AUTO_TDC_DECAY_SCORING_ALGO_V = 2
_AUTO_BROAD_RESIDUAL_PEAK_SCORING_ALGO_V = 3
_AUTO_CORRECTION_SHARPNESS_SCORING_ALGO_V = 1
_AUTO_DIP_FILL_RISK_SCORING_ALGO_V = 1
_AUTO_CHANNEL_OVERFIT_SCORING_ALGO_V = 1
_AUTO_VOICE_CLARITY_SCORING_ALGO_V = 1
_AUTO_RESIDUAL_PEAK_WINNER_POLISH_POLICY_V = 2
_AUTO_GAIN_AUTHORITY_POLICY_V = 4
_AUTO_CONFIDENCE_MODEL_POLICY_V = 2
_AUTO_BASS_INTEGRATION_FEASIBILITY_POLICY_V = 1
_AUTO_PHASE_GD_GUARD_POLICY_V = 5
_AUTO_FINAL_IR_VALIDATION_POLICY_V = 1
_AUTO_MEASUREMENT_METADATA_IDENTITY_V = 2
_AUTO_HYBRID_IIR_POLICY_V = HYBRID_IIR_POLICY_VERSION
_AUTO_HPF_IIR_ROUTING_POLICY_V = 1


def _auto_signature(
    *,
    base_data: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    hc_mode: str | None = None,
    include_hc_mode: bool = True,
) -> str:
    h = hashlib.sha256()
    payload = _auto_signature_payload(
        base_data=base_data,
        measurements=measurements,
        fs_v=int(fs_v),
        taps_v=int(taps_v),
        xos=xos,
        hpf=hpf,
        hc_mode=hc_mode,
        include_hc_mode=bool(include_hc_mode),
    )
    try:
        h.update(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))
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
    ):
        h.update(str(sorted(payload.items())).encode("utf-8", "ignore"))
    return h.hexdigest()


def _auto_signature_payload(
    *,
    base_data: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    hc_mode: str | None = None,
    include_hc_mode: bool = True,
) -> dict:
    base_data = dict(base_data or {})
    measurements = dict(measurements or {})
    ft = str(base_data.get("filter_type", "") or "").strip().lower()
    keys = {
        "schema_version": int(AUTO_MODE_CACHE_SCHEMA_VERSION),
        "fs": int(fs_v),
        "taps": int(taps_v),
        "filter_type": ft,
        "auto_goal": str(_auto_goal(base_data)),
        "measurement_sig": str(_auto_get_measurement_signature(measurements)),
        "measurement_metadata_identity": str(_auto_measurement_metadata_identity(measurements)),
        "frequency_grid_sig": str(
            _auto_hash_array(
                np.asarray(
                    measurements.get(
                        "f_l",
                        measurements.get("f_r", []),
                    ),
                    dtype=float,
                )
            )
        ),
        "target_identity": {
            "hc_mode": str(hc_mode or base_data.get("hc_mode", "") or "").strip()
            if bool(include_hc_mode)
            else "",
            "auto_target_mode": str(base_data.get("auto_target_mode", "") or "").strip().lower(),
            "synth_tilt_frac": float(_auto_safe_float(base_data.get("synth_tilt_frac", float("nan")), float("nan"))),
        },
        "correction_band": {
            "mag_c_min": float(_auto_safe_float(base_data.get("mag_c_min", float("nan")), float("nan"))),
            "mag_c_max": float(_auto_safe_float(base_data.get("mag_c_max", 250.0), 250.0)),
            "trans_width": float(_auto_safe_float(base_data.get("trans_width", float("nan")), float("nan"))),
            "mixed_freq": float(_auto_safe_float(base_data.get("mixed_freq", float("nan")), float("nan"))),
            "low_bass_cut_hz": float(_auto_safe_float(base_data.get("low_bass_cut_hz", float("nan")), float("nan"))),
            "hpf": hpf if isinstance(hpf, dict) or hpf is None else str(hpf),
            "xos": xos if isinstance(xos, list) else [],
        },
        "smoothing": {
            "filter_smooth": float(_auto_safe_float(base_data.get("filter_smooth", float("nan")), float("nan"))),
            "fdw_cycles": float(_auto_safe_float(base_data.get("fdw_cycles", float("nan")), float("nan"))),
            "enable_afdw": bool(base_data.get("enable_afdw", True)),
            "afdw_strength": float(_auto_safe_float(base_data.get("afdw_strength", float("nan")), float("nan"))),
            "bass_smooth_w_gamma": float(_auto_safe_float(base_data.get("bass_smooth_w_gamma", float("nan")), float("nan"))),
            "bass_smooth_w_max": float(_auto_safe_float(base_data.get("bass_smooth_w_max", float("nan")), float("nan"))),
        },
        "mag_post_limits": {
            "min_boost_peak_db": float(
                _auto_safe_float(base_data.get("min_boost_peak_db", 2.0), 2.0)
            ),
            "acoustic_authority_limits_enable": bool(base_data.get("acoustic_authority_limits_enable", True)),
            "authority_boost_gamma": float(
                _auto_safe_float(base_data.get("authority_boost_gamma", 1.35), 1.35)
            ),
            "authority_boost_min_frac": float(
                _auto_safe_float(base_data.get("authority_boost_min_frac", 0.05), 0.05)
            ),
            "authority_boost_min_cap_db": float(
                _auto_safe_float(base_data.get("authority_boost_min_cap_db", 0.0), 0.0)
            ),
            "authority_cut_gamma": float(
                _auto_safe_float(base_data.get("authority_cut_gamma", 0.75), 0.75)
            ),
            "authority_cut_min_frac": float(
                _auto_safe_float(base_data.get("authority_cut_min_frac", 0.35), 0.35)
            ),
            "authority_cut_min_cap_db": float(
                _auto_safe_float(base_data.get("authority_cut_min_cap_db", 3.0), 3.0)
            ),
            "authority_caps_smooth_oct": float(
                _auto_safe_float(base_data.get("authority_caps_smooth_oct", 1.0 / 9.0), 1.0 / 9.0)
            ),
            "residual_pass_mode": str(base_data.get("residual_pass_mode", "modal_polish") or "modal_polish").strip().lower(),
            "residual_null_guard_enable": bool(base_data.get("residual_null_guard_enable", True)),
            "residual_null_guard_strength": float(
                _auto_safe_float(base_data.get("residual_null_guard_strength", 1.0), 1.0)
            ),
            "residual_modal_min_support": float(
                _auto_safe_float(base_data.get("residual_modal_min_support", 0.45), 0.45)
            ),
            "residual_boost_authority_min": float(
                _auto_safe_float(base_data.get("residual_boost_authority_min", 0.40), 0.40)
            ),
            "residual_cut_authority_min": float(
                _auto_safe_float(base_data.get("residual_cut_authority_min", 0.35), 0.35)
            ),
            "residual_reflection_risk_max": float(
                _auto_safe_float(base_data.get("residual_reflection_risk_max", 0.65), 0.65)
            ),
            "residual_null_risk_max_for_boost": float(
                _auto_safe_float(base_data.get("residual_null_risk_max_for_boost", 0.35), 0.35)
            ),
            "residual_null_risk_max_for_cut": float(
                _auto_safe_float(base_data.get("residual_null_risk_max_for_cut", 0.75), 0.75)
            ),
            "residual_max_boost_when_null_risk_db": float(
                _auto_safe_float(base_data.get("residual_max_boost_when_null_risk_db", 0.5), 0.5)
            ),
            "residual_max_boost_general_db": float(
                _auto_safe_float(base_data.get("residual_max_boost_general_db", 2.0), 2.0)
            ),
            "residual_max_cut_general_db": float(
                _auto_safe_float(base_data.get("residual_max_cut_general_db", 4.0), 4.0)
            ),
            "residual_authority_smooth_oct": float(
                _auto_safe_float(base_data.get("residual_authority_smooth_oct", 1.0 / 9.0), 1.0 / 9.0)
            ),
        },
        "signature_policy_versions": {
            "gain_authority_policy_v": int(_AUTO_GAIN_AUTHORITY_POLICY_V),
            "confidence_model_policy_v": int(_AUTO_CONFIDENCE_MODEL_POLICY_V),
            "residual_peak_scorer_v": int(_AUTO_BROAD_RESIDUAL_PEAK_SCORING_ALGO_V),
            "bass_integration_feasibility_policy_v": int(_AUTO_BASS_INTEGRATION_FEASIBILITY_POLICY_V),
            "lf_rolloff_policy_v": int(_AUTO_LF_ROLLOFF_POLICY_V),
            "phase_gd_guard_policy_v": int(_AUTO_PHASE_GD_GUARD_POLICY_V),
            "final_ir_validation_policy_v": int(_AUTO_FINAL_IR_VALIDATION_POLICY_V),
            "measurement_metadata_identity_v": int(_AUTO_MEASUREMENT_METADATA_IDENTITY_V),
            "hybrid_iir_policy_v": int(_AUTO_HYBRID_IIR_POLICY_V),
            "hpf_iir_routing_policy_v": int(_AUTO_HPF_IIR_ROUTING_POLICY_V),
        },
        "hpf_iir_routing": {
            "policy_v": int(_AUTO_HPF_IIR_ROUTING_POLICY_V),
            "tap_threshold": int(HPF_IIR_TAP_THRESHOLD),
            "enabled_for_this_run": bool(hpf_settings_should_use_iir(hpf, taps_v)),
        },
        "hybrid_iir": {
            "policy_v": int(_AUTO_HYBRID_IIR_POLICY_V),
            "enabled": bool(base_data.get("hybrid_iir_enabled", False)),
            "max_filters_per_channel": int(
                _auto_safe_float(base_data.get("hybrid_iir_max_filters_per_channel", 3), 3)
            ),
            "min_freq_hz": float(_auto_safe_float(base_data.get("hybrid_iir_min_freq_hz", 20.0), 20.0)),
            "max_freq_hz": float(_auto_safe_float(base_data.get("hybrid_iir_max_freq_hz", 200.0), 200.0)),
            "min_peak_db": float(_auto_safe_float(base_data.get("hybrid_iir_min_peak_db", 4.0), 4.0)),
            "min_q": float(_auto_safe_float(base_data.get("hybrid_iir_min_q", 3.0), 3.0)),
            "max_q": float(_auto_safe_float(base_data.get("hybrid_iir_max_q", 12.0), 12.0)),
            "max_cut_db": float(_auto_safe_float(base_data.get("hybrid_iir_max_cut_db", 6.0), 6.0)),
            "min_confidence": float(_auto_safe_float(base_data.get("hybrid_iir_min_confidence", 0.30), 0.30)),
            "min_gd_excess_ms": float(
                _auto_safe_float(base_data.get("hybrid_iir_min_gd_excess_ms", 10.0), 10.0)
            ),
            "min_cut_priority": float(
                _auto_safe_float(base_data.get("hybrid_iir_min_cut_priority", 0.0), 0.0)
            ),
        },
        "gain_authority_policy": {
            "policy_v": int(_AUTO_GAIN_AUTHORITY_POLICY_V),
            "unsafe_raw_dsp": bool(base_data.get("unsafe_raw_dsp", False)),
            "max_boost": float(_auto_safe_float(base_data.get("max_boost", float("nan")), float("nan"))),
            "max_boost_db": float(_auto_safe_float(base_data.get("max_boost_db", float("nan")), float("nan"))),
            "max_cut": float(_auto_safe_float(base_data.get("max_cut", float("nan")), float("nan"))),
            "max_cut_db": float(_auto_safe_float(base_data.get("max_cut_db", float("nan")), float("nan"))),
            "low_bass_cut_enable": bool(base_data.get("low_bass_cut_enable", True)),
            "low_bass_cut_hz": float(_auto_safe_float(base_data.get("low_bass_cut_hz", float("nan")), float("nan"))),
            "low_bass_cut_strength": float(
                _auto_safe_float(base_data.get("low_bass_cut_strength", float("nan")), float("nan"))
            ),
            "exc_prot": bool(base_data.get("exc_prot", False)),
            "exc_freq": float(_auto_safe_float(base_data.get("exc_freq", float("nan")), float("nan"))),
            "bass_boost_cap_enable": bool(base_data.get("bass_boost_cap_enable", True)),
            "bass_boost_cap_extra_db": float(
                _auto_safe_float(base_data.get("bass_boost_cap_extra_db", 5.0), 5.0)
            ),
            "bass_boost_cap_hz": float(_auto_safe_float(base_data.get("bass_boost_cap_hz", 200.0), 200.0)),
            "bass_boost_cap_conf_min": float(
                _auto_safe_float(base_data.get("bass_boost_cap_conf_min", 0.55), 0.55)
            ),
            "bass_boost_post_restore_enable": bool(base_data.get("bass_boost_post_restore_enable", True)),
            "bass_boost_post_restore_strength": float(
                _auto_safe_float(base_data.get("bass_boost_post_restore_strength", 1.05), 1.05)
            ),
            "acoustic_authority_limits_enable": bool(base_data.get("acoustic_authority_limits_enable", True)),
        },
        "confidence_model": {
            "policy_v": int(_AUTO_CONFIDENCE_MODEL_POLICY_V),
            "conf_pull_floor": float(_auto_safe_float(base_data.get("conf_pull_floor", 0.05), 0.05)),
            "conf_pull_ceil": float(_auto_safe_float(base_data.get("conf_pull_ceil", 0.85), 0.85)),
            "conf_pull_max_hz": float(_auto_safe_float(base_data.get("conf_pull_max_hz", float("nan")), float("nan"))),
            "conf_pull_gamma_cut": float(_auto_safe_float(base_data.get("conf_pull_gamma_cut", 0.45), 0.45)),
            "conf_pull_gamma_boost": float(_auto_safe_float(base_data.get("conf_pull_gamma_boost", 0.35), 0.35)),
            "conf_pull_conf_smooth_sigma": float(
                _auto_safe_float(base_data.get("conf_pull_conf_smooth_sigma", 2.0), 2.0)
            ),
            "conf_pull_bass_floor_hz": float(
                _auto_safe_float(base_data.get("conf_pull_bass_floor_hz", 120.0), 120.0)
            ),
            "conf_pull_bass_floor_min": float(
                _auto_safe_float(base_data.get("conf_pull_bass_floor_min", 0.25), 0.25)
            ),
            "conf_pull_bass_boost_floor_min": float(
                _auto_safe_float(base_data.get("conf_pull_bass_boost_floor_min", 0.55), 0.55)
            ),
            "conf_pull_bass_boost_restore": float(
                _auto_safe_float(base_data.get("conf_pull_bass_boost_restore", 0.70), 0.70)
            ),
            "bass_first_ai": bool(base_data.get("bass_first_ai", True)),
            "bass_first_mode_max_hz": float(
                _auto_safe_float(base_data.get("bass_first_mode_max_hz", float("nan")), float("nan"))
            ),
        },
        "residual_peak_scorer": {
            "scorer_v": int(_AUTO_BROAD_RESIDUAL_PEAK_SCORING_ALGO_V),
            "winner_polish_policy_v": int(_AUTO_RESIDUAL_PEAK_WINNER_POLISH_POLICY_V),
            "threshold_db": float(
                _auto_safe_float(base_data.get("auto_mode_residual_peak_threshold_db", base_data.get("residual_peak_threshold_db", 3.0)), 3.0)
            ),
            "hard_gate_db": float(
                _auto_safe_float(base_data.get("auto_mode_residual_peak_hard_gate_db", base_data.get("residual_peak_hard_gate_db", 6.0)), 6.0)
            ),
            "penalty_cap": float(
                _auto_safe_float(base_data.get("auto_mode_residual_peak_penalty_cap", base_data.get("residual_peak_penalty_cap", 20.0)), 20.0)
            ),
        },
        "bass_integration_feasibility": {
            "policy_v": int(_AUTO_BASS_INTEGRATION_FEASIBILITY_POLICY_V),
            "guard_lo_ratio": float(
                _auto_safe_float(base_data.get("bass_integration_guard_lo_ratio", 0.60), 0.60)
            ),
            "guard_hi_ratio": float(
                _auto_safe_float(base_data.get("bass_integration_guard_hi_ratio", 1.40), 1.40)
            ),
            "profile": str(base_data.get("bass_integration_profile", "") or ""),
            "sub_combine_mode": str(base_data.get("bass_integration_sub_combine_mode", "average") or "average"),
        },
        "phase_gd_guard": {
            "policy_v": int(_AUTO_PHASE_GD_GUARD_POLICY_V),
            "phase_limit": float(_auto_safe_float(base_data.get("phase_limit", float("nan")), float("nan"))),
            "mixed_freq": float(_auto_safe_float(base_data.get("mixed_freq", float("nan")), float("nan"))),
            "excess_phase_strength": float(
                _auto_safe_float(base_data.get("excess_phase_strength", float("nan")), float("nan"))
            ),
            "phase_correction_enable": bool(base_data.get("phase_correction_enable", base_data.get("enable_phase_correction", True))),
            "phase_guard_max_gd_gradient_ms_per_oct": float(
                _auto_safe_float(base_data.get("phase_guard_max_gd_gradient_ms_per_oct", 50.0), 50.0)
            ),
            "phase_guard_prering_enable": bool(base_data.get("phase_guard_prering_enable", True)),
            "phase_guard_excess_delay_enable": bool(base_data.get("phase_guard_excess_delay_enable", True)),
            "phase_budget_mode": str(base_data.get("phase_budget_mode", "unified") or "unified"),
            "linear_excess_strength": float(
                _auto_safe_float(base_data.get("linear_excess_strength", 0.9), 0.9)
            ),
            "phase_conf_gain_floor": float(
                _auto_safe_float(base_data.get("phase_conf_gain_floor", 0.20), 0.20)
            ),
            "phase_conf_gain_power": float(
                _auto_safe_float(base_data.get("phase_conf_gain_power", 1.0), 1.0)
            ),
            "phase_corr_clamp_lf_deg": float(
                _auto_safe_float(base_data.get("phase_corr_clamp_lf_deg", 540.0), 540.0)
            ),
            "phase_corr_clamp_hf_deg": float(
                _auto_safe_float(base_data.get("phase_corr_clamp_hf_deg", 90.0), 90.0)
            ),
            "max_excess_delay_cycles": float(
                _auto_safe_float(base_data.get("max_excess_delay_cycles", 1.0), 1.0)
            ),
            "phase_authority_enable": bool(base_data.get("phase_authority_enable", True)),
            "phase_authority_gamma": float(
                _auto_safe_float(base_data.get("phase_authority_gamma", 1.20), 1.20)
            ),
            "phase_authority_min_gain": float(
                _auto_safe_float(base_data.get("phase_authority_min_gain", 0.0), 0.0)
            ),
            "phase_authority_soft_floor": float(
                _auto_safe_float(base_data.get("phase_authority_soft_floor", 0.20), 0.20)
            ),
            "phase_authority_smooth_oct": float(
                _auto_safe_float(base_data.get("phase_authority_smooth_oct", 1.0 / 6.0), 1.0 / 6.0)
            ),
            "phase_authority_disable_above_hz": float(
                _auto_safe_float(base_data.get("phase_authority_disable_above_hz", 1200.0), 1200.0)
            ),
        },
        "final_ir_validation": {
            "policy_v": int(_AUTO_FINAL_IR_VALIDATION_POLICY_V),
            "enable": bool(base_data.get("final_ir_validation_enable", True)),
            "mode": str(base_data.get("final_ir_validation_mode", "warn") or "warn").strip().lower(),
            "score_weight": float(
                _auto_safe_float(base_data.get("final_ir_validation_score_weight", 1.0), 1.0)
            ),
            "candidate_count": int(
                _auto_safe_float(base_data.get("final_ir_validation_candidate_count", 3), 3)
            ),
        },
        "enable_tdc": bool(base_data.get("enable_tdc", True)),
        "bass_first_ai": bool(base_data.get("bass_first_ai", True)),
        "bass_first_mode_max_hz": float(
            _auto_safe_float(base_data.get("bass_first_mode_max_hz", float("nan")), float("nan"))
        ),
        "conf_pull_max_hz": float(_auto_safe_float(base_data.get("conf_pull_max_hz", float("nan")), float("nan"))),
        "enable_channel_specific_auto_policy": bool(base_data.get("enable_channel_specific_auto_policy", False)),
        "channel_specific_policy_max_hz": float(
            _auto_safe_float(base_data.get("channel_specific_policy_max_hz", float("nan")), float("nan"))
        ),
        "channel_specific_policy_refine_trials": int(
            _auto_safe_float(base_data.get("channel_specific_refine_trials", 32), 32)
        ),
        "min_worst_channel_improvement_db": float(
            _auto_safe_float(base_data.get("min_worst_channel_improvement_db", 0.015), 0.015)
        ),
        "mag_c_max": float(_auto_safe_float(base_data.get("mag_c_max", 250.0), 250.0)),
        "_auto_mag_c_min_hz": float(_auto_safe_float(base_data.get("_auto_mag_c_min_hz", float("nan")), float("nan"))),
        "_auto_low_bass_cut_hz": float(_auto_safe_float(base_data.get("_auto_low_bass_cut_hz", float("nan")), float("nan"))),
        "_auto_exc_freq_hz": float(_auto_safe_float(base_data.get("_auto_exc_freq_hz", float("nan")), float("nan"))),
        "_auto_lf_rolloff_policy_v": int(_AUTO_LF_ROLLOFF_POLICY_V),
        "_bass_allpass_algo_v": int(_BASS_ALLPASS_ALGO_V),
        "_bass_integration_combine_algo_v": int(_BASS_INTEGRATION_COMBINE_ALGO_V),
        "_direct_dac_sub_target_policy_v": int(_DIRECT_DAC_SUB_TARGET_POLICY_V),
        "_auto_tdc_decay_scoring_algo_v": int(_AUTO_TDC_DECAY_SCORING_ALGO_V),
        "auto_mode_scoring_policy": {
            "broad_residual_peak_scoring_algo_v": int(_AUTO_BROAD_RESIDUAL_PEAK_SCORING_ALGO_V),
            "correction_sharpness_scoring_algo_v": int(_AUTO_CORRECTION_SHARPNESS_SCORING_ALGO_V),
            "dip_fill_risk_scoring_algo_v": int(_AUTO_DIP_FILL_RISK_SCORING_ALGO_V),
            "channel_overfit_scoring_algo_v": int(_AUTO_CHANNEL_OVERFIT_SCORING_ALGO_V),
            "voice_clarity_scoring_algo_v": int(_AUTO_VOICE_CLARITY_SCORING_ALGO_V),
            "auto_voice_clarity_penalty_enable": bool(
                base_data.get("auto_voice_clarity_penalty_enable", True)
            ),
            "auto_voice_clarity_penalty_weight": float(
                _auto_safe_float(base_data.get("auto_voice_clarity_penalty_weight", 1.0), 1.0)
            ),
            "auto_voice_band_lo_hz": float(
                _auto_safe_float(base_data.get("auto_voice_band_lo_hz", 70.0), 70.0)
            ),
            "auto_voice_band_hi_hz": float(
                _auto_safe_float(base_data.get("auto_voice_band_hi_hz", 180.0), 180.0)
            ),
            "residual_peak_winner_polish_policy_v": int(_AUTO_RESIDUAL_PEAK_WINNER_POLISH_POLICY_V),
            "residual_peak_threshold_db": float(
                _auto_safe_float(base_data.get("auto_mode_residual_peak_threshold_db", base_data.get("residual_peak_threshold_db", 3.0)), 3.0)
            ),
            "residual_peak_hard_gate_db": float(
                _auto_safe_float(base_data.get("auto_mode_residual_peak_hard_gate_db", base_data.get("residual_peak_hard_gate_db", 6.0)), 6.0)
            ),
            "residual_peak_penalty_cap": float(
                _auto_safe_float(base_data.get("auto_mode_residual_peak_penalty_cap", base_data.get("residual_peak_penalty_cap", 20.0)), 20.0)
            ),
            "residual_peak_winner_polish_enabled": bool(base_data.get("residual_peak_winner_polish_enabled", True)),
            "residual_peak_winner_polish_max_variants": int(
                _auto_safe_float(base_data.get("residual_peak_winner_polish_max_variants", 8), 8)
            ),
            "residual_peak_winner_polish_min_improvement_db": float(
                _auto_safe_float(base_data.get("residual_peak_winner_polish_min_improvement_db", 0.75), 0.75)
            ),
        },
        "tdc": {
            "tdc_strength": float(_auto_safe_float(base_data.get("tdc_strength", float("nan")), float("nan"))),
            "tdc_max_reduction_db": float(
                _auto_safe_float(base_data.get("tdc_max_reduction_db", float("nan")), float("nan"))
            ),
            "tdc_slope_db_per_oct": float(
                _auto_safe_float(base_data.get("tdc_slope_db_per_oct", float("nan")), float("nan"))
            ),
            "auto_mode_tdc_decay_penalty_cap": float(
                _auto_safe_float(base_data.get("auto_mode_tdc_decay_penalty_cap", 8.0), 8.0)
            ),
            "auto_mode_tdc_decay_penalty_weight": float(
                _auto_safe_float(base_data.get("auto_mode_tdc_decay_penalty_weight", 1.0), 1.0)
            ),
            "auto_mode_tdc_extreme_peak_reduction_db": float(
                _auto_safe_float(base_data.get("auto_mode_tdc_extreme_peak_reduction_db", 10.0), 10.0)
            ),
            "auto_mode_tdc_low_need_threshold": float(
                _auto_safe_float(base_data.get("auto_mode_tdc_low_need_threshold", 0.15), 0.15)
            ),
            "auto_mode_tdc_rt60_target_low_s": float(
                _auto_safe_float(base_data.get("auto_mode_tdc_rt60_target_low_s", 0.45), 0.45)
            ),
            "auto_mode_tdc_rt60_target_upper_s": float(
                _auto_safe_float(base_data.get("auto_mode_tdc_rt60_target_upper_s", 0.38), 0.38)
            ),
            "auto_mode_tdc_rt60_low_max_hz": float(
                _auto_safe_float(base_data.get("auto_mode_tdc_rt60_low_max_hz", 160.0), 160.0)
            ),
            "auto_mode_tdc_rt60_eval_max_hz": float(
                _auto_safe_float(base_data.get("auto_mode_tdc_rt60_eval_max_hz", 300.0), 300.0)
            ),
        },
        "phase_gd": {
            "phase_limit": float(_auto_safe_float(base_data.get("phase_limit", float("nan")), float("nan"))),
            "excess_phase_strength": float(
                _auto_safe_float(base_data.get("excess_phase_strength", float("nan")), float("nan"))
            ),
            "phase_correction_enable": bool(base_data.get("phase_correction_enable", base_data.get("enable_phase_correction", True))),
            "max_slope_db_per_oct": float(_auto_safe_float(base_data.get("max_slope_db_per_oct", float("nan")), float("nan"))),
            "reg_strength": float(_auto_safe_float(base_data.get("reg_strength", float("nan")), float("nan"))),
            "phase_budget_mode": str(base_data.get("phase_budget_mode", "unified") or "unified"),
            "linear_excess_strength": float(
                _auto_safe_float(base_data.get("linear_excess_strength", 0.9), 0.9)
            ),
            "max_excess_delay_cycles": float(
                _auto_safe_float(base_data.get("max_excess_delay_cycles", 1.0), 1.0)
            ),
        },
        "bass_integration_sub_combine_mode": str(base_data.get("bass_integration_sub_combine_mode", "average") or "average"),
        "bass_integration_enabled": bool(
            base_data.get("bass_integration_enabled", base_data.get("bass_integration_enable", False))
            or measurements.get("bass_integration_enabled", False)
        ),
        "bass_integration_sub_delay_ms": float(
            _auto_safe_float(base_data.get("bass_integration_sub_delay_ms", float("nan")), float("nan"))
        ),
        "bass_integration_sub_array_delay_ms": float(
            _auto_safe_float(base_data.get("bass_integration_sub_array_delay_ms", float("nan")), float("nan"))
        ),
        "bass_integration_sub1_delay_ms": float(
            _auto_safe_float(base_data.get("bass_integration_sub1_delay_ms", float("nan")), float("nan"))
        ),
        "bass_integration_sub2_delay_ms": float(
            _auto_safe_float(base_data.get("bass_integration_sub2_delay_ms", float("nan")), float("nan"))
        ),
        "bass_integration_main_l_delay_ms": float(
            _auto_safe_float(base_data.get("bass_integration_main_l_delay_ms", float("nan")), float("nan"))
        ),
        "bass_integration_main_r_delay_ms": float(
            _auto_safe_float(base_data.get("bass_integration_main_r_delay_ms", float("nan")), float("nan"))
        ),
        "bass_integration_sub_polarity_invert": bool(base_data.get("bass_integration_sub_polarity_invert", False)),
        "bass_integration_sub_gain_trim_db": float(
            _auto_safe_float(base_data.get("bass_integration_sub_gain_trim_db", float("nan")), float("nan"))
        ),
        "bass_integration_alignment_auto_applied": bool(
            base_data.get("bass_integration_alignment_auto_applied", False)
        ),
        "bass_integration_allpass_auto_enable": bool(base_data.get("bass_integration_allpass_auto_enable", False)),
        "bass_integration_allpass_auto_applied": bool(base_data.get("bass_integration_allpass_auto_applied", False)),
        "bass_integration_allpass_freq_hz": float(
            _auto_safe_float(base_data.get("bass_integration_allpass_freq_hz", float("nan")), float("nan"))
        ),
        "bass_integration_allpass_q": float(
            _auto_safe_float(base_data.get("bass_integration_allpass_q", float("nan")), float("nan"))
        ),
        "avr_crossover_hz": float(
            _auto_safe_float(base_data.get("avr_crossover_hz", float("nan")), float("nan"))
        ),
        "sub_crossover_slope": float(
            _auto_safe_float(base_data.get("sub_crossover_slope", float("nan")), float("nan"))
        ),
        "sub_hpf_freq": float(
            _auto_safe_float(base_data.get("sub_hpf_freq", float("nan")), float("nan"))
        ),
        "sub_hpf_slope": float(
            _auto_safe_float(base_data.get("sub_hpf_slope", float("nan")), float("nan"))
        ),
        "direct_dac_sub_lpf_hz": float(
            _auto_safe_float(base_data.get("direct_dac_sub_lpf_hz", float("nan")), float("nan"))
        ),
        "bass_integration_sub_topology": str(
            dict(base_data.get("_bass_integration_meta", {}) or {}).get("sub_topology", "")
            if isinstance(base_data.get("_bass_integration_meta", {}), dict)
            else ""
        ),
        "bass_integration_algorithm_v": int(_BASS_INTEGRATION_ALGO_V),
        "bass_integration_robust_policy_v": 1,
        "bass_integration_robust_gain_offsets_db": (-1.0, 0.0, 1.0),
        "bass_integration_robust_delay_offsets_ms": (-0.5, 0.0, 0.5),
        "bass_integration_sub_lpf_policy": "optimized_lpf_gte_main_hpf",
        "bass_integration_profile": str(base_data.get("bass_integration_profile", "") or ""),
        "bass_integration_mode": "direct_dac",
        "max_boost": float(_auto_safe_float(base_data.get("max_boost", float("nan")), float("nan"))),
        "max_cut": float(_auto_safe_float(base_data.get("max_cut", float("nan")), float("nan"))),
        "max_slope_db_per_oct": float(_auto_safe_float(base_data.get("max_slope_db_per_oct", float("nan")), float("nan"))),
    }
    return dict(keys)


def _auto_seed_from_signature(
    *,
    base_data: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    hc_mode: str | None = None,
    include_hc_mode: bool = True,
) -> int:
    try:
        payload = _auto_signature_payload(
            base_data=base_data,
            measurements=measurements,
            fs_v=int(fs_v),
            taps_v=int(taps_v),
            xos=xos,
            hpf=hpf,
            hc_mode=hc_mode,
            include_hc_mode=bool(include_hc_mode),
        )
        stable_measurement_identity = _auto_search_measurement_identity(measurements or {})
        payload["measurement_sig"] = str(stable_measurement_identity)
        sig = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()
        if not sig:
            raise ValueError("empty signature")
        return int(str(sig)[:8], 16) & 0xFFFFFFFF
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
    ):
        try:
            msig = _auto_search_measurement_identity(measurements or {})
            return int(str(msig)[:8], 16) & 0xFFFFFFFF if msig else 0
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
        ):
            return 0


def _auto_apply_seed(seed: int) -> None:
    try:
        s = int(seed) & 0xFFFFFFFF
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
    ):
        s = 0
    try:
        random.seed(s)
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
    ):
        logger.exception("random seed apply")
    try:
        np.random.seed(s)
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
    ):
        logger.exception("numpy random seed apply")


@_auto_cache_guard
def clear_auto_mode_runtime_caches() -> None:
    """Clear process-local auto-mode caches without touching persisted files."""
    with _SYNTH_TARGET_CACHE_LOCK:
        _SYNTH_TARGET_CACHE.clear()
    _AUTO_CACHE_RUNTIME.clear()
    for key in tuple(_AUTO_CACHE_STATS.keys()):
        _AUTO_CACHE_STATS[key] = "" if key.startswith("last_") else 0
