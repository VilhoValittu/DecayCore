# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Builtin (non-Optuna) local and micro candidate builders for auto mode."""

import logging

import numpy as np

logger = logging.getLogger(__name__)

from .shared import (
    AUTO_MODE_LOCAL_REFINE_SHRINK,
    AUTO_MODE_LOW_BASS_MAX_HZ,
    AUTO_MODE_LOW_BASS_MIN_HZ,
    AUTO_MODE_PHASE_LIMIT_LOCAL_SIGMA_HZ,
    AUTO_MODE_PHASE_LIMIT_MAX_HZ,
    AUTO_MODE_PHASE_LIMIT_MIN_HZ,
    AUTO_MODE_PHASE3_MICRO_TRIALS,
    _auto_is_phase_search_filter,
    _auto_goal,
    _auto_goal_is_flat_family,
    _auto_mag_c_min_center,
    _auto_mag_c_min_search_bounds,
    _auto_phase_limit_center,
    _auto_safe_float,
    _auto_sample_mag_low_pair,
    _clip,
    _jitter,
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
)


def _build_auto_mode_candidates_local(
    base_data: dict,
    center: dict,
    n_trials: int,
    seed: int,
    shrink: float = AUTO_MODE_LOCAL_REFINE_SHRINK,
    optimize_mag_low: bool = True,
) -> list[dict]:
    n_eff = max(1, int(n_trials))
    rng = np.random.default_rng(int(seed))
    s = float(np.clip(_auto_safe_float(shrink, AUTO_MODE_LOCAL_REFINE_SHRINK), 0.05, 1.50))
    tune_mag_low = bool(optimize_mag_low)

    base = dict(base_data or {})
    c = dict(base)
    c.update(dict(center or {}))

    ft = str(c.get("filter_type", base.get("filter_type", "")) or "").strip().lower()
    prefer_bass = bool(_auto_goal_is_flat_family(_auto_goal(base)))
    is_mixed = "mixed" in ft
    is_phase_search = _auto_is_phase_search_filter(ft)
    phase_center = _auto_phase_limit_center(c.get("phase_limit", base.get("phase_limit")))

    keep_tdc = bool(c.get("enable_tdc", True))
    keep_afdw = bool(c.get("enable_afdw", True))
    keep_bass_first = bool(c.get("bass_first_ai", True))
    mag_c_min_lo, mag_c_min_hi = _auto_mag_c_min_search_bounds(base)
    mag_c_min_center = round(
        float(
            np.clip(
                _auto_mag_c_min_center(c, default=25.0),
                float(mag_c_min_lo),
                float(mag_c_min_hi),
            )
        ),
        1,
    )
    low_bass_cut_center = round(
        _clip(
            c.get("low_bass_cut_hz", base.get("low_bass_cut_hz", 40.0)),
            float(AUTO_MODE_LOW_BASS_MIN_HZ),
            float(AUTO_MODE_LOW_BASS_MAX_HZ),
        ),
        1,
    )

    slope_choices = [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 24.0, 36.0]
    slope_center = _auto_safe_float(c.get("tdc_slope_db_per_oct", base.get("tdc_slope_db_per_oct", 6.0)), 6.0)
    slope_idx = int(min(range(len(slope_choices)), key=lambda i: abs(float(slope_choices[i]) - float(slope_center))))
    center_out = dict(c)
    center_out["comparison_mode"] = True
    center_out["enable_tdc"] = bool(keep_tdc)
    center_out["enable_afdw"] = bool(keep_afdw)
    center_out["bass_first_ai"] = bool(keep_bass_first)
    center_out["mag_c_min"] = float(mag_c_min_center)
    center_out["low_bass_cut_hz"] = float(low_bass_cut_center)
    if bool(prefer_bass):
        center_out["max_boost"] = round(
            float(max(6.0, _auto_safe_float(center_out.get("max_boost", base.get("max_boost", 5.0)), 5.0))),
            2,
        )
    if bool(is_phase_search):
        center_out["phase_limit"] = round(float(phase_center), 1)

    out: list[dict] = [center_out]
    for _ in range(max(0, n_eff - 1)):
        step = int(rng.choice(np.array([-1, 0, 1], dtype=int), p=np.array([0.20, 0.60, 0.20])))
        idx = int(np.clip(int(slope_idx + step), 0, len(slope_choices) - 1))
        if bool(tune_mag_low):
            mag_c_min_cand, low_bass_cut_cand = _auto_sample_mag_low_pair(
                rng,
                mag_center=_auto_safe_float(c.get("mag_c_min", base.get("mag_c_min", mag_c_min_center)), mag_c_min_center),
                low_center=_auto_safe_float(c.get("low_bass_cut_hz", base.get("low_bass_cut_hz", low_bass_cut_center)), low_bass_cut_center),
                mag_sigma=max(0.4, 3.2 * s),
                low_sigma=max(0.6, 4.0 * s),
                mag_lo=float(mag_c_min_lo),
                mag_hi=float(mag_c_min_hi),
            )
        else:
            mag_c_min_cand = float(mag_c_min_center)
            low_bass_cut_cand = float(low_bass_cut_center)
        cand = {
            "comparison_mode": True,
            "enable_tdc": bool(keep_tdc),
            "enable_afdw": bool(keep_afdw),
            "bass_first_ai": bool(keep_bass_first),
            "fdw_cycles": round(_jitter(rng, c.get("fdw_cycles"), 2.5 * s, 8.0, 16.0, base_data=base, key="fdw_cycles", default=10.0), 2),
            "tdc_strength": round(_jitter(rng, c.get("tdc_strength"), 12.0 * s, _TDC_STRENGTH_MIN, _TDC_STRENGTH_MAX, base_data=base, key="tdc_strength", default=50.0), 1),
            "tdc_max_reduction_db": round(_jitter(rng, c.get("tdc_max_reduction_db"), 6.0 * s, _TDC_MAX_REDUCTION_MIN_DB, _TDC_MAX_REDUCTION_MAX_DB, base_data=base, key="tdc_max_reduction_db", default=9.0), 1),
            "tdc_slope_db_per_oct": float(slope_choices[idx]),
            "reg_strength": round(_jitter(rng, c.get("reg_strength"), 10.0 * s, 15.0, 45.0, base_data=base, key="reg_strength", default=30.0), 1),
            "max_boost": round(
                _jitter(
                    rng,
                    max(6.0, _auto_safe_float(c.get("max_boost", base.get("max_boost", 5.0)), 5.0)) if prefer_bass else c.get("max_boost"),
                    1.0 * s,
                    5.0 if prefer_bass else 3.0,
                    12.0,
                    base_data=base,
                    key="max_boost",
                    default=6.0 if prefer_bass else 4.0,
                ),
                2,
            ),
            "mag_c_min": float(mag_c_min_cand),
            "mag_c_max": round(_jitter(rng, c.get("mag_c_max"), 25.0 * s, 170.0, 300.0, base_data=base, key="mag_c_max", default=220.0), 1),
            "trans_width": round(_jitter(rng, c.get("trans_width"), 25.0 * s, 70.0, 150.0, base_data=base, key="trans_width", default=100.0), 1),
            "bass_first_mode_max_hz": round(_jitter(rng, c.get("bass_first_mode_max_hz"), 25.0 * s, _BASS_FIRST_MODE_MIN_HZ, _BASS_FIRST_MODE_MAX_HZ, base_data=base, key="bass_first_mode_max_hz", default=180.0), 1),
            "conf_pull_max_hz": round(_jitter(rng, c.get("conf_pull_max_hz"), 30.0 * s, _CONF_PULL_MAX_MIN_HZ, _CONF_PULL_MAX_MAX_HZ, base_data=base, key="conf_pull_max_hz", default=200.0), 1),
            "low_bass_cut_hz": float(low_bass_cut_cand),
        }
        if is_mixed:
            cand["mixed_freq"] = round(_jitter(rng, c.get("mixed_freq"), 35.0 * s, 80.0, 320.0, base_data=base, key="mixed_freq", default=180.0), 1)
        if is_phase_search:
            cand["phase_limit"] = round(
                _jitter(
                    rng,
                    c.get("phase_limit"),
                    float(AUTO_MODE_PHASE_LIMIT_LOCAL_SIGMA_HZ) * s,
                    float(AUTO_MODE_PHASE_LIMIT_MIN_HZ),
                    float(AUTO_MODE_PHASE_LIMIT_MAX_HZ),
                    base_data=base,
                    key="phase_limit",
                    default=float(phase_center),
                ),
                1,
            )
        out.append(cand)
    return out


def _build_auto_mode_candidates_micro(
    base_data: dict,
    center: dict,
    *,
    n_trials: int = AUTO_MODE_PHASE3_MICRO_TRIALS,
    shrink: float = 1.0,
) -> list[dict]:
    n_eff = max(1, int(n_trials))
    p = dict(base_data or {})
    p.update(dict(center or {}))
    ft = str(p.get("filter_type", "") or "").strip().lower()
    is_mixed = "mixed" in ft
    is_phase_search = _auto_is_phase_search_filter(ft)

    s = float(np.clip(_auto_safe_float(shrink, 1.0), 0.25, 1.0))
    mixed_steps = [0.0, -16.0 * s, -8.0 * s, +8.0 * s, +16.0 * s]
    phase_steps = [0.0, -28.0 * s, -14.0 * s, +14.0 * s, +28.0 * s]
    tdc_steps = [0.0, -8.0 * s, -4.0 * s, +4.0 * s, +8.0 * s]
    fdw_steps = [0.0, -1.0 * s, +1.0 * s]
    reg_steps = [0.0, -6.0 * s, +6.0 * s]
    tw_steps = [0.0, -15.0 * s, +15.0 * s]
    patterns = [
        (0, 0, 0, 0, 0),
        (2, 2, 1, 1, 1),
        (3, 3, 2, 2, 2),
        (1, 1, 2, 1, 2),
        (4, 4, 1, 2, 1),
        (2, 1, 2, 2, 0),
        (3, 4, 1, 0, 2),
        (1, 3, 0, 2, 0),
        (4, 2, 2, 0, 1),
        (0, 4, 1, 2, 2),
        (0, 1, 2, 0, 1),
        (2, 0, 0, 1, 2),
        (4, 1, 0, 1, 1),
        (1, 4, 2, 1, 0),
        (2, 3, 0, 2, 1),
        (3, 2, 1, 0, 2),
        (4, 0, 2, 2, 0),
        (1, 2, 1, 2, 2),
        (2, 4, 2, 0, 0),
        (3, 1, 0, 1, 2),
        (4, 3, 2, 1, 0),
        (1, 0, 1, 2, 1),
    ]

    base_mixed = _auto_safe_float(p.get("mixed_freq", 180.0), 180.0)
    base_phase = _auto_phase_limit_center(p.get("phase_limit"))
    base_tdc = _auto_safe_float(p.get("tdc_strength", 55.0), 55.0)
    base_fdw = _auto_safe_float(p.get("fdw_cycles", 10.0), 10.0)
    base_reg = _auto_safe_float(p.get("reg_strength", 30.0), 30.0)
    base_tw = _auto_safe_float(p.get("trans_width", 100.0), 100.0)

    out: list[dict] = []
    seen = set()
    for i in range(max(1, n_eff)):
        pi = patterns[int(i % len(patterns))]
        cand = dict(center or {})
        cand["comparison_mode"] = True
        cand["tdc_strength"] = round(_clip(base_tdc + float(tdc_steps[int(pi[1])]), _TDC_STRENGTH_MIN, _TDC_STRENGTH_MAX), 1)
        cand["fdw_cycles"] = round(_clip(base_fdw + float(fdw_steps[int(pi[2])]), 6.0, 16.0), 2)
        cand["reg_strength"] = round(_clip(base_reg + float(reg_steps[int(pi[3])]), 15.0, 45.0), 1)
        cand["trans_width"] = round(_clip(base_tw + float(tw_steps[int(pi[4])]), 70.0, 150.0), 1)
        if bool(is_mixed):
            cand["mixed_freq"] = round(_clip(base_mixed + float(mixed_steps[int(pi[0])]), 80.0, 320.0), 1)
        if bool(is_phase_search):
            cand["phase_limit"] = round(
                _clip(
                    base_phase + float(phase_steps[int(pi[0])]),
                    float(AUTO_MODE_PHASE_LIMIT_MIN_HZ),
                    float(AUTO_MODE_PHASE_LIMIT_MAX_HZ),
                ),
                1,
            )

        sig = (
            float(_auto_safe_float(cand.get("mixed_freq", float("nan")), float("nan"))) if bool(is_mixed) else float("nan"),
            float(_auto_safe_float(cand.get("phase_limit", float("nan")), float("nan"))) if bool(is_phase_search) else float("nan"),
            float(_auto_safe_float(cand.get("tdc_strength", float("nan")), float("nan"))),
            float(_auto_safe_float(cand.get("fdw_cycles", float("nan")), float("nan"))),
            float(_auto_safe_float(cand.get("reg_strength", float("nan")), float("nan"))),
            float(_auto_safe_float(cand.get("trans_width", float("nan")), float("nan"))),
        )
        if sig in seen:
            continue
        seen.add(sig)
        out.append(cand)
        if len(out) >= n_eff:
            break

    if not out:
        base_c = dict(center or {})
        base_c["comparison_mode"] = True
        out = [base_c]
    return out
