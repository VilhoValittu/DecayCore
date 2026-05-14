# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Coarse Optuna candidate suggestion and seed projection for auto mode search."""

import logging

import numpy as np

logger = logging.getLogger(__name__)

from .shared import (
    AUTO_MODE_GOAL_FLAT,
    AUTO_MODE_MAG_C_MAX_MIN_HZ,
    AUTO_MODE_PHASE_LIMIT_MAX_HZ,
    AUTO_MODE_PHASE_LIMIT_MIN_HZ,
    AUTO_MODE_LOCAL_REFINE_SHRINK,
    _auto_is_phase_search_filter,
    _auto_goal,
    _auto_output_tilt_bounds,
    _auto_phase_limit_center,
    _auto_safe_float,
)
from .candidate_base import (
    _TDC_STRENGTH_MIN,
    _TDC_STRENGTH_MAX,
    _TDC_MAX_REDUCTION_MIN_DB,
    _TDC_MAX_REDUCTION_MAX_DB,
    _BASS_FIRST_MODE_MIN_HZ,
    _BASS_FIRST_MODE_MAX_HZ,
    _CONF_PULL_MAX_MIN_HZ,
    _CONF_PULL_MAX_MAX_HZ,
    AUTO_MODE_MAG_C_MIN_MIN_HZ,
    AUTO_MODE_MAG_C_MIN_MAX_HZ,
    AUTO_MODE_LOW_BASS_MIN_HZ,
    AUTO_MODE_LOW_BASS_MAX_HZ,
    _auto_optuna_snap_to_step,
    _auto_optuna_nearest_choice,
    _bi_search_enabled,
    _derive_adaptive_freq_bounds,
    _auto_filter_normalized_base_data,
    _mag_low_search_bounds,
    _seed_bi_optuna_params,
    _suggest_bi_optuna_params,
)


def _suggest_auto_mode_candidate_optuna(
    base_data: dict,
    trial,
    *,
    optimize_mag_low: bool = True,
) -> dict:
    base_data = _auto_filter_normalized_base_data(base_data)
    keep_tdc = bool(base_data.get("enable_tdc", True))
    keep_afdw = bool(base_data.get("enable_afdw", True))
    keep_bass_first = bool(base_data.get("bass_first_ai", True))
    prefer_bass = bool(_auto_goal(base_data) == AUTO_MODE_GOAL_FLAT)
    ft = str(base_data.get("filter_type", "") or "").strip().lower()
    is_mixed = "mixed" in ft
    is_phase_search = _auto_is_phase_search_filter(ft)
    is_synth_target = str(base_data.get("auto_target_mode", "") or "").strip().lower() == "adaptive"
    tune_mag_low = bool(optimize_mag_low)
    output_tilt_lo, output_tilt_hi = _auto_output_tilt_bounds(base_data)

    adaptive = _derive_adaptive_freq_bounds(base_data)
    _mag_c_min_lo, _mag_c_min_hi, _low_bass_lo, _low_bass_hi = _mag_low_search_bounds(base_data)
    _bass_first_hi = float(adaptive.get("bass_first_hi", _BASS_FIRST_MODE_MAX_HZ))
    _conf_pull_hi = float(adaptive.get("conf_pull_hi", _CONF_PULL_MAX_MAX_HZ))
    if adaptive and getattr(trial, "number", None) == 0:
        logger.info(
            "auto_mode adaptive bounds: mag_c_min=[%.1f,%.1f] low_bass=[%.1f,%.1f] "
            "bass_first_hi=%.1f conf_pull_hi=%.1f",
            _mag_c_min_lo, _mag_c_min_hi, _low_bass_lo, _low_bass_hi,
            _bass_first_hi, _conf_pull_hi,
        )

    mag_c_min_seed = float(
        np.clip(
            _auto_safe_float(base_data.get("mag_c_min", 25.0), 25.0),
            _mag_c_min_lo,
            _mag_c_min_hi,
        )
    )
    low_bass_cut_seed = float(
        np.clip(
            _auto_safe_float(base_data.get("low_bass_cut_hz", 40.0), 40.0),
            _low_bass_lo,
            _low_bass_hi,
        )
    )

    if bool(tune_mag_low):
        mag_c_min = _auto_optuna_snap_to_step(
            trial.suggest_float(
                "mag_c_min",
                float(AUTO_MODE_MAG_C_MIN_MIN_HZ),
                float(AUTO_MODE_MAG_C_MIN_MAX_HZ),
                log=True,
            ),
            lo=float(AUTO_MODE_MAG_C_MIN_MIN_HZ),
            hi=float(AUTO_MODE_MAG_C_MIN_MAX_HZ),
            step=0.1,
        )
        low_bass_cut_hz = _auto_optuna_snap_to_step(
            trial.suggest_float(
                "low_bass_cut_hz",
                float(AUTO_MODE_LOW_BASS_MIN_HZ),
                float(AUTO_MODE_LOW_BASS_MAX_HZ),
                log=True,
            ),
            lo=float(AUTO_MODE_LOW_BASS_MIN_HZ),
            hi=float(AUTO_MODE_LOW_BASS_MAX_HZ),
            step=0.1,
        )
    else:
        mag_c_min = float(round(mag_c_min_seed, 1))
        low_bass_cut_hz = float(round(low_bass_cut_seed, 1))

    cand = {
        "comparison_mode": True,
        "enable_tdc": bool(keep_tdc),
        "enable_afdw": bool(trial.suggest_categorical("enable_afdw", [False, True])),
        "bass_first_ai": bool(trial.suggest_categorical("bass_first_ai", [False, True])),
        "fdw_cycles": _auto_optuna_snap_to_step(
            trial.suggest_float("fdw_cycles", 5.0, 16.0, step=0.01),
            lo=5.0,
            hi=16.0,
            step=0.01,
        ),
        "tdc_strength": _auto_optuna_snap_to_step(
            trial.suggest_float("tdc_strength", _TDC_STRENGTH_MIN, _TDC_STRENGTH_MAX, step=0.1),
            lo=_TDC_STRENGTH_MIN,
            hi=_TDC_STRENGTH_MAX,
            step=0.1,
        ),
        "tdc_max_reduction_db": _auto_optuna_snap_to_step(
            trial.suggest_float("tdc_max_reduction_db", _TDC_MAX_REDUCTION_MIN_DB, _TDC_MAX_REDUCTION_MAX_DB, step=0.1),
            lo=_TDC_MAX_REDUCTION_MIN_DB,
            hi=_TDC_MAX_REDUCTION_MAX_DB,
            step=0.1,
        ),
        "tdc_slope_db_per_oct": float(trial.suggest_categorical("tdc_slope_db_per_oct", [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 24.0, 36.0])),
        "reg_strength": _auto_optuna_snap_to_step(
            trial.suggest_float("reg_strength", 15.0, 45.0, step=0.1),
            lo=15.0,
            hi=45.0,
            step=0.1,
        ),
        "max_slope_db_per_oct": float(trial.suggest_categorical("max_slope_db_per_oct", [8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0])),
        "max_slope_boost_db_per_oct": float(
            trial.suggest_categorical("max_slope_boost_db_per_oct", [0.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 24.0, 36.0])
        ),
        "max_slope_cut_db_per_oct": float(
            trial.suggest_categorical("max_slope_cut_db_per_oct", [0.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 24.0, 36.0])
        ),
        "max_boost": _auto_optuna_snap_to_step(
            trial.suggest_float("max_boost", 0.1, 12.0, step=0.01),
            lo=0.1,
            hi=12.0,
            step=0.03,
        ),
        "mag_c_min": float(mag_c_min),
        "mag_c_max": _auto_optuna_snap_to_step(
            trial.suggest_float("mag_c_max", float(AUTO_MODE_MAG_C_MAX_MIN_HZ), 400.0, step=0.1),
            lo=float(AUTO_MODE_MAG_C_MAX_MIN_HZ),
            hi=400.0,
            step=0.1,
        ),
        "trans_width": _auto_optuna_snap_to_step(
            trial.suggest_float("trans_width", 70.0, 150.0, step=0.1),
            lo=70.0,
            hi=150.0,
            step=0.1,
        ),
        "filter_smooth": 96,
        "bass_first_mode_max_hz": _auto_optuna_snap_to_step(
            trial.suggest_float("bass_first_mode_max_hz", _BASS_FIRST_MODE_MIN_HZ, _BASS_FIRST_MODE_MAX_HZ, step=0.1),
            lo=_BASS_FIRST_MODE_MIN_HZ,
            hi=_BASS_FIRST_MODE_MAX_HZ,
            step=0.1,
        ),
        "conf_pull_max_hz": _auto_optuna_snap_to_step(
            trial.suggest_float("conf_pull_max_hz", _CONF_PULL_MAX_MIN_HZ, _CONF_PULL_MAX_MAX_HZ, step=5.0),
            lo=_CONF_PULL_MAX_MIN_HZ,
            hi=_CONF_PULL_MAX_MAX_HZ,
            step=2.5,
        ),
        "low_bass_cut_hz": float(low_bass_cut_hz),
    }
    cand["output_tilt_db_per_oct"] = _auto_optuna_snap_to_step(
        trial.suggest_float(
            "output_tilt_db_per_oct",
            -2.0,
            2.0,
            step=0.05,
        ),
        lo=-2.0,
        hi=2.0,
        step=0.05,
    )
    if bool(is_synth_target):
        cand["synth_tilt_frac"] = _auto_optuna_snap_to_step(
            trial.suggest_float("synth_tilt_frac", 0.05, 0.55, step=0.01),
            lo=0.05,
            hi=0.55,
            step=0.01,
        )
    if bool(is_mixed):
        cand["mixed_freq"] = _auto_optuna_snap_to_step(
            trial.suggest_float("mixed_freq", 80.0, 320.0, step=0.1),
            lo=80.0,
            hi=320.0,
            step=0.1,
        )
    if bool(is_phase_search):
        cand["phase_limit"] = _auto_optuna_snap_to_step(
            trial.suggest_float(
                "phase_limit",
                float(AUTO_MODE_PHASE_LIMIT_MIN_HZ),
                float(AUTO_MODE_PHASE_LIMIT_MAX_HZ),
                log=True,
            ),
            lo=float(AUTO_MODE_PHASE_LIMIT_MIN_HZ),
            hi=float(AUTO_MODE_PHASE_LIMIT_MAX_HZ),
            step=0.1,
        )
    if _bi_search_enabled(base_data):
        cand.update(_suggest_bi_optuna_params(base_data, trial, coarse=True))
    return cand


def _seed_auto_mode_candidate_optuna_params(
    base_data: dict,
    preset: dict | None,
    *,
    optimize_mag_low: bool = True,
) -> dict:
    base_data = _auto_filter_normalized_base_data(base_data)
    ft = str(base_data.get("filter_type", "") or "").strip().lower()
    is_mixed = "mixed" in ft
    is_phase_search = _auto_is_phase_search_filter(ft)
    is_synth_target = str(base_data.get("auto_target_mode", "") or "").strip().lower() == "adaptive"
    output_tilt_lo, output_tilt_hi = _auto_output_tilt_bounds(base_data)
    p = dict(base_data or {})
    p.update(dict(preset or {}))

    mag_lo, mag_hi, low_lo, low_hi = _mag_low_search_bounds(base_data)
    mag_c_min = float(np.clip(_auto_safe_float(p.get("mag_c_min", 25.0), 25.0), mag_lo, mag_hi))
    low_bass_cut_hz = float(np.clip(_auto_safe_float(p.get("low_bass_cut_hz", 40.0), 40.0), low_lo, low_hi))

    out = {
        "enable_afdw": bool(p.get("enable_afdw", base_data.get("enable_afdw", True))),
        "bass_first_ai": bool(p.get("bass_first_ai", base_data.get("bass_first_ai", True))),
        "fdw_cycles": _auto_optuna_snap_to_step(_auto_safe_float(p.get("fdw_cycles", 10.0), 10.0), lo=5.0, hi=16.0, step=0.01),
        "tdc_strength": _auto_optuna_snap_to_step(
            _auto_safe_float(p.get("tdc_strength", 50.0), 50.0),
            lo=_TDC_STRENGTH_MIN,
            hi=_TDC_STRENGTH_MAX,
            step=0.1,
        ),
        "tdc_max_reduction_db": _auto_optuna_snap_to_step(
            _auto_safe_float(p.get("tdc_max_reduction_db", 9.0), 9.0),
            lo=_TDC_MAX_REDUCTION_MIN_DB,
            hi=_TDC_MAX_REDUCTION_MAX_DB,
            step=0.1,
        ),
        "tdc_slope_db_per_oct": _auto_optuna_nearest_choice(
            _auto_safe_float(p.get("tdc_slope_db_per_oct", 6.0), 6.0),
            [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 24.0, 36.0],
            default=6.0,
        ),
        "reg_strength": _auto_optuna_snap_to_step(_auto_safe_float(p.get("reg_strength", 30.0), 30.0), lo=15.0, hi=45.0, step=0.1),
        "max_slope_db_per_oct": _auto_optuna_nearest_choice(
            _auto_safe_float(p.get("max_slope_db_per_oct", 12.0), 12.0),
            [8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0],
            default=12.0,
        ),
        "max_slope_boost_db_per_oct": _auto_optuna_nearest_choice(
            _auto_safe_float(p.get("max_slope_boost_db_per_oct", 0.0), 0.0),
            [0.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 24.0, 36.0],
            default=0.0,
        ),
        "max_slope_cut_db_per_oct": _auto_optuna_nearest_choice(
            _auto_safe_float(p.get("max_slope_cut_db_per_oct", 0.0), 0.0),
            [0.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 24.0, 36.0],
            default=0.0,
        ),
        "max_boost": _auto_optuna_snap_to_step(_auto_safe_float(p.get("max_boost", 4.0), 4.0), lo=3.0, hi=12.0, step=0.01),
        "mag_c_max": _auto_optuna_snap_to_step(_auto_safe_float(p.get("mag_c_max", 220.0), 220.0), lo=170.0, hi=300.0, step=0.1),
        "trans_width": _auto_optuna_snap_to_step(_auto_safe_float(p.get("trans_width", 100.0), 100.0), lo=70.0, hi=150.0, step=0.1),
        "filter_smooth": 96,
        "bass_first_mode_max_hz": _auto_optuna_snap_to_step(
            _auto_safe_float(p.get("bass_first_mode_max_hz", 180.0), 180.0),
            lo=_BASS_FIRST_MODE_MIN_HZ,
            hi=_BASS_FIRST_MODE_MAX_HZ,
            step=0.1,
        ),
        "conf_pull_max_hz": _auto_optuna_snap_to_step(
            _auto_safe_float(p.get("conf_pull_max_hz", 200.0), 200.0),
            lo=_CONF_PULL_MAX_MIN_HZ,
            hi=_CONF_PULL_MAX_MAX_HZ,
            step=5.0,
        ),
    }
    out["output_tilt_db_per_oct"] = _auto_optuna_snap_to_step(
        _auto_safe_float(p.get("output_tilt_db_per_oct", 0.0), 0.0),
        lo=float(output_tilt_lo),
        hi=float(output_tilt_hi),
        step=0.05,
    )
    if bool(is_synth_target):
        out["synth_tilt_frac"] = _auto_optuna_snap_to_step(
            _auto_safe_float(p.get("synth_tilt_frac", 0.30), 0.30),
            lo=0.05,
            hi=0.55,
            step=0.01,
        )
    if bool(optimize_mag_low):
        out["mag_c_min"] = _auto_optuna_snap_to_step(
            float(mag_c_min),
            lo=mag_lo,
            hi=mag_hi,
            step=0.1,
        )
        out["low_bass_cut_hz"] = _auto_optuna_snap_to_step(
            float(low_bass_cut_hz),
            lo=low_lo,
            hi=low_hi,
            step=0.1,
        )
    if bool(is_mixed):
        out["mixed_freq"] = _auto_optuna_snap_to_step(
            _auto_safe_float(p.get("mixed_freq", 180.0), 180.0),
            lo=80.0,
            hi=320.0,
            step=0.1,
        )
    if bool(is_phase_search):
        out["phase_limit"] = _auto_optuna_snap_to_step(
            _auto_safe_float(p.get("phase_limit", _auto_phase_limit_center(base_data.get("phase_limit", None))), 320.0),
            lo=float(AUTO_MODE_PHASE_LIMIT_MIN_HZ),
            hi=float(AUTO_MODE_PHASE_LIMIT_MAX_HZ),
            step=0.1,
        )
    if _bi_search_enabled(base_data):
        out.update(_seed_bi_optuna_params(base_data, p))
    return dict(out)
