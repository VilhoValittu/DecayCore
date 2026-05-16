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

import hashlib
import json
import random

import numpy as np

from .cache_get_put import (
    _auto_cache_get_best,
    _auto_cache_get_best_target,
    _auto_cache_get_entry,
    _auto_cache_get_target_for_measurements,
    _auto_cache_get_target_for_measurements_global,
    _auto_cache_put_best,
    _auto_cache_put_target_for_measurements,
    _auto_cache_put_target_for_measurements_global,
)
from .cache_io import (
    _AUTO_CACHE_LOCK,
    _AUTO_CACHE_RUNTIME,
    _AUTO_CACHE_STATS,
    _auto_cache_guard,
    _auto_cache_load,
    _auto_cache_save,
    _auto_cache_stats_snapshot,
)
from .cache_lastused import _auto_cache_get_last_used_best, _auto_cache_put_last_used_best
from .cache_measurement_sig import _auto_get_measurement_signature, _auto_measurement_signature
from .cache_paths import (
    _auto_cache_compat_token,
    _auto_cache_filename,
    _auto_cache_path,
    get_auto_mode_cache_path,
)
from .cache_structure import (
    _auto_cache_bucket,
    _auto_cache_bucket_template,
    _auto_cache_empty,
    _auto_compat_version,
)
from .cache_synth_target import (
    _SYNTH_TARGET_CACHE,
    _SYNTH_TARGET_CACHE_LOCK,
    _SYNTH_TARGET_MISS,
    _synth_target_cache_key,
    get_or_build_synth_target,
)
from .shared import (
    AUTO_MODE_CACHE_SCHEMA_VERSION,
    AUTO_MODE_COMPAT_VERSION,
    _auto_goal,
    _auto_goal_norm,
    _auto_hash_array,
    _auto_safe_float,
    logger,
)

def _auto_cache_resolve_path(*, compat_version: str | None = None) -> str:
    if compat_version is None:
        return _auto_cache_path()
    try:
        return _auto_cache_path(compat_version=compat_version)
    except TypeError:
        # Tests may monkeypatch _auto_cache_path with a zero-arg lambda.
        return _auto_cache_path()


_BASS_ALLPASS_ALGO_V = 1
_BASS_INTEGRATION_COMBINE_ALGO_V = 1
_DIRECT_DAC_SUB_TARGET_POLICY_V = 1
_AUTO_TDC_DECAY_SCORING_ALGO_V = 2
_AUTO_BROAD_RESIDUAL_PEAK_SCORING_ALGO_V = 1
_AUTO_CORRECTION_SHARPNESS_SCORING_ALGO_V = 1
_AUTO_DIP_FILL_RISK_SCORING_ALGO_V = 1
_AUTO_CHANNEL_OVERFIT_SCORING_ALGO_V = 1
_AUTO_VOICE_CLARITY_SCORING_ALGO_V = 1
_AUTO_RESIDUAL_PEAK_WINNER_POLISH_POLICY_V = 2


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
    except Exception:
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
        },
        "bass_integration_sub_combine_mode": str(base_data.get("bass_integration_sub_combine_mode", "average") or "average"),
        "bass_integration_enabled": bool(
            base_data.get("bass_integration_enabled", base_data.get("bass_integration_enable", False))
            or measurements.get("bass_integration_enabled", False)
        ),
        "bass_integration_sub_delay_ms": float(
            _auto_safe_float(base_data.get("bass_integration_sub_delay_ms", float("nan")), float("nan"))
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
        "bass_integration_profile": str(base_data.get("bass_integration_profile", "") or ""),
        "bass_integration_mode": str(base_data.get("bass_integration_mode", "") or ""),
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
        sig = _auto_signature(
            base_data=base_data,
            measurements=measurements,
            fs_v=int(fs_v),
            taps_v=int(taps_v),
            xos=xos,
            hpf=hpf,
            hc_mode=hc_mode,
            include_hc_mode=bool(include_hc_mode),
        )
        if not sig:
            raise ValueError("empty signature")
        return int(str(sig)[:8], 16) & 0xFFFFFFFF
    except Exception:
        try:
            msig = _auto_measurement_signature(measurements or {})
            return int(str(msig)[:8], 16) & 0xFFFFFFFF if msig else 0
        except Exception:
            return 0


def _auto_apply_seed(seed: int) -> None:
    try:
        s = int(seed) & 0xFFFFFFFF
    except Exception:
        s = 0
    try:
        random.seed(s)
    except Exception:
        logger.exception("random seed apply")
    try:
        np.random.seed(s)
    except Exception:
        logger.exception("numpy random seed apply")


@_auto_cache_guard
def clear_auto_mode_runtime_caches() -> None:
    """Clear process-local auto-mode caches without touching persisted files."""
    with _SYNTH_TARGET_CACHE_LOCK:
        _SYNTH_TARGET_CACHE.clear()
    _AUTO_CACHE_RUNTIME.clear()
    for key in tuple(_AUTO_CACHE_STATS.keys()):
        _AUTO_CACHE_STATS[key] = "" if key.startswith("last_") else 0
