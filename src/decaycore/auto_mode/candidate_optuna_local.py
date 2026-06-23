# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Local and micro Optuna candidate suggestion and seeding for auto mode."""

import logging

import numpy as np

logger = logging.getLogger(__name__)

from .shared import (
    AUTO_MODE_GOAL_FLAT,
    AUTO_MODE_LOCAL_REFINE_SHRINK,
    AUTO_MODE_LOW_BASS_MAX_HZ,
    AUTO_MODE_LOW_BASS_MIN_HZ,
    AUTO_MODE_MAG_C_MIN_MAX_HZ,
    AUTO_MODE_MAG_C_MIN_MIN_HZ,
    AUTO_MODE_PHASE_LIMIT_LOCAL_SIGMA_HZ,
    AUTO_MODE_PHASE_LIMIT_MAX_HZ,
    AUTO_MODE_PHASE_LIMIT_MIN_HZ,
    _auto_is_phase_search_filter,
    _auto_goal,
    _auto_mag_c_min_center,
    _auto_output_tilt_bounds,
    _auto_phase_limit_center,
    _auto_safe_float,
    _clip,
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
    _auto_optuna_suggest_centered_unit_float,
    _auto_optuna_seed_centered_unit_float,
    _auto_optuna_choice_from_unit,
    _auto_optuna_seed_choice_unit,
    _bi_search_enabled,
    _auto_filter_normalized_base_data,
    _seed_bi_optuna_params,
    _suggest_bi_optuna_params,
)


def _suggest_auto_mode_candidate_local_optuna(
    base_data: dict,
    center: dict,
    trial,
    *,
    shrink: float = AUTO_MODE_LOCAL_REFINE_SHRINK,
    optimize_mag_low: bool = True,
) -> dict:
    base = _auto_filter_normalized_base_data(base_data)
    c = dict(base)
    c.update(dict(center or {}))
    prefer_bass = bool(_auto_goal(base) == AUTO_MODE_GOAL_FLAT)

    s = float(np.clip(_auto_safe_float(shrink, AUTO_MODE_LOCAL_REFINE_SHRINK), 0.05, 1.50))
    tune_mag_low = bool(optimize_mag_low)
    ft = str(c.get("filter_type", base.get("filter_type", "")) or "").strip().lower()
    is_mixed = "mixed" in ft
    is_phase_search = _auto_is_phase_search_filter(ft)
    output_tilt_lo, output_tilt_hi = _auto_output_tilt_bounds(base)
    output_tilt_center = float(
        np.clip(
            _auto_safe_float(c.get("output_tilt_db_per_oct", 0.0), 0.0),
            float(output_tilt_lo),
            float(output_tilt_hi),
        )
    )
    phase_center = _auto_phase_limit_center(c.get("phase_limit", base.get("phase_limit", None)))

    keep_tdc = bool(c.get("enable_tdc", True))
    keep_afdw = bool(c.get("enable_afdw", True))
    keep_bass_first = bool(c.get("bass_first_ai", True))
    mag_c_min_center = float(round(_auto_mag_c_min_center(c, default=25.0), 1))
    low_bass_cut_center = float(
        round(
            _clip(
                c.get("low_bass_cut_hz", base.get("low_bass_cut_hz", 40.0)),
                float(AUTO_MODE_LOW_BASS_MIN_HZ),
                float(AUTO_MODE_LOW_BASS_MAX_HZ),
            ),
            1,
        )
    )

    slope_choices = [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 24.0, 36.0]
    slope_center = _auto_safe_float(c.get("tdc_slope_db_per_oct", base.get("tdc_slope_db_per_oct", 6.0)), 6.0)
    max_slope_choices = [8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0]
    max_slope_center = _auto_safe_float(c.get("max_slope_db_per_oct", base.get("max_slope_db_per_oct", 12.0)), 12.0)

    if bool(tune_mag_low):
        mag_c_min_cand = round(
            _auto_optuna_suggest_centered_unit_float(
                trial,
                "mag_c_min",
                float(mag_c_min_center),
                max(0.4, 3.2 * s),
                float(AUTO_MODE_MAG_C_MIN_MIN_HZ),
                float(AUTO_MODE_MAG_C_MIN_MAX_HZ),
            ),
            1,
        )
        low_bass_cut_cand = round(
            _auto_optuna_suggest_centered_unit_float(
                trial,
                "low_bass_cut_hz",
                float(low_bass_cut_center),
                max(0.6, 4.0 * s),
                float(AUTO_MODE_LOW_BASS_MIN_HZ),
                float(AUTO_MODE_LOW_BASS_MAX_HZ),
            ),
            1,
        )
    else:
        mag_c_min_cand = float(mag_c_min_center)
        low_bass_cut_cand = float(low_bass_cut_center)

    cand = {
        "comparison_mode": True,
        "enable_tdc": bool(keep_tdc),
        "enable_afdw": bool(keep_afdw),
        "bass_first_ai": bool(keep_bass_first),
        "fdw_cycles": round(
            _auto_optuna_suggest_centered_unit_float(
                trial,
                "fdw_cycles",
                _auto_safe_float(c.get("fdw_cycles", base.get("fdw_cycles", 10.0)), 10.0),
                2.5 * s,
                8.0,
                16.0,
            ),
            2,
        ),
        "tdc_strength": round(
            _auto_optuna_suggest_centered_unit_float(
                trial,
                "tdc_strength",
                _auto_safe_float(c.get("tdc_strength", base.get("tdc_strength", 50.0)), 50.0),
                12.0 * s,
                _TDC_STRENGTH_MIN,
                _TDC_STRENGTH_MAX,
            ),
            1,
        ),
        "tdc_max_reduction_db": round(
            _auto_optuna_suggest_centered_unit_float(
                trial,
                "tdc_max_reduction_db",
                _auto_safe_float(c.get("tdc_max_reduction_db", base.get("tdc_max_reduction_db", 9.0)), 9.0),
                6.0 * s,
                _TDC_MAX_REDUCTION_MIN_DB,
                _TDC_MAX_REDUCTION_MAX_DB,
            ),
            1,
        ),
        "tdc_slope_db_per_oct": float(
            _auto_optuna_choice_from_unit(
                trial,
                "tdc_slope_db_per_oct",
                slope_choices,
                float(slope_center),
                radius=max(1, int(round(1.5 * s))),
            )
        ),
        "reg_strength": round(
            _auto_optuna_suggest_centered_unit_float(
                trial,
                "reg_strength",
                _auto_safe_float(c.get("reg_strength", base.get("reg_strength", 30.0)), 30.0),
                10.0 * s,
                15.0,
                45.0,
            ),
            1,
        ),
        "max_slope_db_per_oct": float(
            _auto_optuna_choice_from_unit(
                trial,
                "max_slope_db_per_oct",
                max_slope_choices,
                float(max_slope_center),
                radius=max(1, int(round(1.5 * s))),
            )
        ),
        "max_boost": round(
            _auto_optuna_suggest_centered_unit_float(
                trial,
                "max_boost",
                max(
                    6.0 if prefer_bass else 3.0,
                    _auto_safe_float(c.get("max_boost", base.get("max_boost", 6.0 if prefer_bass else 4.0)), 6.0 if prefer_bass else 4.0),
                ),
                1.0 * s,
                5.0 if prefer_bass else 3.0,
                8.0,
            ),
            2,
        ),
        "mag_c_min": float(mag_c_min_cand),
        "mag_c_max": round(
            _auto_optuna_suggest_centered_unit_float(
                trial,
                "mag_c_max",
                _auto_safe_float(c.get("mag_c_max", base.get("mag_c_max", 220.0)), 220.0),
                25.0 * s,
                170.0,
                300.0,
            ),
            1,
        ),
        "trans_width": round(
            _auto_optuna_suggest_centered_unit_float(
                trial,
                "trans_width",
                _auto_safe_float(c.get("trans_width", base.get("trans_width", 100.0)), 100.0),
                25.0 * s,
                70.0,
                150.0,
            ),
            1,
        ),
        "filter_smooth": 96,
        "bass_first_mode_max_hz": round(
            _auto_optuna_suggest_centered_unit_float(
                trial,
                "bass_first_mode_max_hz",
                _auto_safe_float(c.get("bass_first_mode_max_hz", base.get("bass_first_mode_max_hz", 180.0)), 180.0),
                25.0 * s,
                _BASS_FIRST_MODE_MIN_HZ,
                _BASS_FIRST_MODE_MAX_HZ,
            ),
            1,
        ),
        "conf_pull_max_hz": round(
            _auto_optuna_suggest_centered_unit_float(
                trial,
                "conf_pull_max_hz",
                _auto_safe_float(c.get("conf_pull_max_hz", base.get("conf_pull_max_hz", 200.0)), 200.0),
                30.0 * s,
                _CONF_PULL_MAX_MIN_HZ,
                _CONF_PULL_MAX_MAX_HZ,
            ),
            1,
        ),
        "low_bass_cut_hz": float(low_bass_cut_cand),
        "output_tilt_db_per_oct": round(
            _auto_optuna_suggest_centered_unit_float(
                trial,
                "output_tilt_db_per_oct",
                float(output_tilt_center),
                0.5 * s,
                float(output_tilt_lo),
                float(output_tilt_hi),
            ),
            2,
        ),
    }
    if bool(is_mixed):
        cand["mixed_freq"] = round(
            _auto_optuna_suggest_centered_unit_float(
                trial,
                "mixed_freq",
                _auto_safe_float(c.get("mixed_freq", base.get("mixed_freq", 180.0)), 180.0),
                35.0 * s,
                80.0,
                320.0,
            ),
            1,
        )
    if bool(is_phase_search):
        cand["phase_limit"] = round(
            _auto_optuna_suggest_centered_unit_float(
                trial,
                "phase_limit",
                float(phase_center),
                float(AUTO_MODE_PHASE_LIMIT_LOCAL_SIGMA_HZ) * s,
                float(AUTO_MODE_PHASE_LIMIT_MIN_HZ),
                float(AUTO_MODE_PHASE_LIMIT_MAX_HZ),
            ),
            1,
        )
    if _bi_search_enabled(base):
        cand.update(_suggest_bi_optuna_params(base, trial, coarse=False, center=c))
    return cand


def _seed_auto_mode_candidate_local_optuna_params(
    base_data: dict,
    center: dict,
    preset: dict | None,
    *,
    shrink: float = AUTO_MODE_LOCAL_REFINE_SHRINK,
    optimize_mag_low: bool = True,
) -> dict:
    base_data = _auto_filter_normalized_base_data(base_data)
    base = dict(base_data or {})
    c = dict(base)
    c.update(dict(center or {}))
    p = dict(c)
    p.update(dict(preset or {}))
    prefer_bass = bool(_auto_goal(base) == AUTO_MODE_GOAL_FLAT)
    s = float(np.clip(_auto_safe_float(shrink, AUTO_MODE_LOCAL_REFINE_SHRINK), 0.05, 1.50))
    ft = str(c.get("filter_type", base.get("filter_type", "")) or "").strip().lower()
    is_mixed = "mixed" in ft
    is_phase_search = _auto_is_phase_search_filter(ft)
    output_tilt_lo, output_tilt_hi = _auto_output_tilt_bounds(base_data)
    output_tilt_center = float(
        np.clip(
            _auto_safe_float(c.get("output_tilt_db_per_oct", 0.0), 0.0),
            float(output_tilt_lo),
            float(output_tilt_hi),
        )
    )
    phase_center = _auto_phase_limit_center(c.get("phase_limit", base.get("phase_limit")))
    mag_c_min_center = float(round(_auto_mag_c_min_center(c, default=25.0), 1))
    low_bass_cut_center = float(
        round(
            _clip(
                c.get("low_bass_cut_hz", base.get("low_bass_cut_hz", 40.0)),
                float(AUTO_MODE_LOW_BASS_MIN_HZ),
                float(AUTO_MODE_LOW_BASS_MAX_HZ),
            ),
            1,
        )
    )
    slope_choices = [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 24.0, 36.0]
    max_slope_choices = [8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0]
    slope_center = _auto_safe_float(c.get("tdc_slope_db_per_oct", base.get("tdc_slope_db_per_oct", 6.0)), 6.0)
    max_slope_center = _auto_safe_float(c.get("max_slope_db_per_oct", base.get("max_slope_db_per_oct", 12.0)), 12.0)

    def _windowed_unit(name: str, center_v: float, span: float, lo: float, hi: float, default: float) -> float:
        value = float(np.clip(_auto_safe_float(p.get(name, default), default), float(lo), float(hi)))
        return float(
            _auto_optuna_seed_centered_unit_float(
                value,
                center=float(center_v),
                span=float(span),
                lo=float(lo),
                hi=float(hi),
            )
        )

    out = {
        "fdw_cycles_u": _windowed_unit(
            "fdw_cycles",
            _auto_safe_float(c.get("fdw_cycles", base.get("fdw_cycles", 10.0)), 10.0),
            2.5 * s,
            8.0,
            16.0,
            10.0,
        ),
        "tdc_strength_u": _windowed_unit(
            "tdc_strength",
            _auto_safe_float(c.get("tdc_strength", base.get("tdc_strength", 50.0)), 50.0),
            12.0 * s,
            _TDC_STRENGTH_MIN,
            _TDC_STRENGTH_MAX,
            50.0,
        ),
        "tdc_max_reduction_db_u": _windowed_unit(
            "tdc_max_reduction_db",
            _auto_safe_float(c.get("tdc_max_reduction_db", base.get("tdc_max_reduction_db", 9.0)), 9.0),
            6.0 * s,
            _TDC_MAX_REDUCTION_MIN_DB,
            _TDC_MAX_REDUCTION_MAX_DB,
            9.0,
        ),
        "tdc_slope_db_per_oct_u": _auto_optuna_seed_choice_unit(
            _auto_safe_float(p.get("tdc_slope_db_per_oct", slope_center), slope_center),
            slope_choices,
            float(slope_center),
            radius=max(1, int(round(1.5 * s))),
            default=slope_center,
        ),
        "reg_strength_u": _windowed_unit(
            "reg_strength",
            _auto_safe_float(c.get("reg_strength", base.get("reg_strength", 30.0)), 30.0),
            10.0 * s,
            15.0,
            45.0,
            30.0,
        ),
        "max_slope_db_per_oct_u": _auto_optuna_seed_choice_unit(
            _auto_safe_float(p.get("max_slope_db_per_oct", max_slope_center), max_slope_center),
            max_slope_choices,
            float(max_slope_center),
            radius=max(1, int(round(1.5 * s))),
            default=max_slope_center,
        ),
        "max_boost_u": _windowed_unit(
            "max_boost",
            max(
                6.0 if prefer_bass else 3.0,
                _auto_safe_float(c.get("max_boost", base.get("max_boost", 6.0 if prefer_bass else 4.0)), 6.0 if prefer_bass else 4.0),
            ),
            1.0 * s,
            5.0 if prefer_bass else 3.0,
            8.0,
            6.0 if prefer_bass else 4.0,
        ),
        "mag_c_max_u": _windowed_unit(
            "mag_c_max",
            _auto_safe_float(c.get("mag_c_max", base.get("mag_c_max", 220.0)), 220.0),
            25.0 * s,
            170.0,
            300.0,
            220.0,
        ),
        "trans_width_u": _windowed_unit(
            "trans_width",
            _auto_safe_float(c.get("trans_width", base.get("trans_width", 100.0)), 100.0),
            25.0 * s,
            70.0,
            150.0,
            100.0,
        ),
        "bass_first_mode_max_hz_u": _windowed_unit(
            "bass_first_mode_max_hz",
            _auto_safe_float(c.get("bass_first_mode_max_hz", base.get("bass_first_mode_max_hz", 180.0)), 180.0),
            25.0 * s,
            _BASS_FIRST_MODE_MIN_HZ,
            _BASS_FIRST_MODE_MAX_HZ,
            180.0,
        ),
        "conf_pull_max_hz_u": _windowed_unit(
            "conf_pull_max_hz",
            _auto_safe_float(c.get("conf_pull_max_hz", base.get("conf_pull_max_hz", 200.0)), 200.0),
            30.0 * s,
            _CONF_PULL_MAX_MIN_HZ,
            _CONF_PULL_MAX_MAX_HZ,
            200.0,
        ),
        "output_tilt_db_per_oct_u": _windowed_unit(
            "output_tilt_db_per_oct",
            float(output_tilt_center),
            0.5 * s,
            float(output_tilt_lo),
            float(output_tilt_hi),
            float(output_tilt_center),
        ),
    }
    if bool(optimize_mag_low):
        out["mag_c_min_u"] = _windowed_unit(
            "mag_c_min",
            float(mag_c_min_center),
            max(0.4, 3.2 * s),
            float(AUTO_MODE_MAG_C_MIN_MIN_HZ),
            float(AUTO_MODE_MAG_C_MIN_MAX_HZ),
            float(mag_c_min_center),
        )
        out["low_bass_cut_hz_u"] = _windowed_unit(
            "low_bass_cut_hz",
            float(low_bass_cut_center),
            max(0.6, 4.0 * s),
            float(AUTO_MODE_LOW_BASS_MIN_HZ),
            float(AUTO_MODE_LOW_BASS_MAX_HZ),
            float(low_bass_cut_center),
        )
    if bool(is_mixed):
        out["mixed_freq_u"] = _windowed_unit(
            "mixed_freq",
            _auto_safe_float(c.get("mixed_freq", base.get("mixed_freq", 180.0)), 180.0),
            35.0 * s,
            80.0,
            320.0,
            180.0,
        )
    if bool(is_phase_search):
        out["phase_limit_u"] = _windowed_unit(
            "phase_limit",
            float(phase_center),
            float(AUTO_MODE_PHASE_LIMIT_LOCAL_SIGMA_HZ) * s,
            float(AUTO_MODE_PHASE_LIMIT_MIN_HZ),
            float(AUTO_MODE_PHASE_LIMIT_MAX_HZ),
            float(phase_center),
        )
    if _bi_search_enabled(base):
        out.update(_seed_bi_optuna_params(base, p))
    return dict(out)
