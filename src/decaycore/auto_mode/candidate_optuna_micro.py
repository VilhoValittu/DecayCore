# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Micro Optuna candidate suggestion and seeding for auto mode phase 3."""

import logging

import numpy as np

logger = logging.getLogger(__name__)

from .shared_parts import (
    AUTO_MODE_PHASE_LIMIT_MAX_HZ,
    AUTO_MODE_PHASE_LIMIT_MIN_HZ,
    _auto_is_phase_search_filter,
    _auto_phase_limit_center,
    _auto_safe_float,
)
from .candidate_base import (
    _TDC_STRENGTH_MIN,
    _TDC_STRENGTH_MAX,
    _auto_optuna_suggest_centered_unit_float,
    _auto_optuna_seed_centered_unit_float,
    _bi_search_enabled,
    _auto_filter_normalized_base_data,
    _seed_bi_optuna_params,
    _suggest_bi_optuna_params,
)


def _suggest_auto_mode_candidate_micro_optuna(
    base_data: dict,
    center: dict,
    trial,
    *,
    shrink: float = 1.0,
) -> dict:
    base_data = _auto_filter_normalized_base_data(base_data)
    p = dict(base_data or {})
    p.update(dict(center or {}))
    ft = str(p.get("filter_type", "") or "").strip().lower()
    is_mixed = "mixed" in ft
    is_phase_search = _auto_is_phase_search_filter(ft)
    s = float(np.clip(_auto_safe_float(shrink, 1.0), 0.25, 1.0))

    base_mixed = _auto_safe_float(p.get("mixed_freq", 180.0), 180.0)
    base_phase = _auto_phase_limit_center(p.get("phase_limit"))
    base_tdc = _auto_safe_float(p.get("tdc_strength", 50.0), 50.0)
    base_fdw = _auto_safe_float(p.get("fdw_cycles", 10.0), 10.0)
    base_reg = _auto_safe_float(p.get("reg_strength", 30.0), 30.0)
    base_tw = _auto_safe_float(p.get("trans_width", 100.0), 100.0)

    cand = dict(center or {})
    cand["comparison_mode"] = True
    cand["tdc_strength"] = round(
        _auto_optuna_suggest_centered_unit_float(
            trial,
            "tdc_strength",
            float(base_tdc),
            8.0 * s,
            _TDC_STRENGTH_MIN,
            _TDC_STRENGTH_MAX,
        ),
        1,
    )
    cand["fdw_cycles"] = round(
        _auto_optuna_suggest_centered_unit_float(
            trial,
            "fdw_cycles",
            float(base_fdw),
            1.0 * s,
            6.0,
            16.0,
        ),
        2,
    )
    cand["reg_strength"] = round(
        _auto_optuna_suggest_centered_unit_float(
            trial,
            "reg_strength",
            float(base_reg),
            6.0 * s,
            15.0,
            45.0,
        ),
        1,
    )
    cand["trans_width"] = round(
        _auto_optuna_suggest_centered_unit_float(
            trial,
            "trans_width",
            float(base_tw),
            15.0 * s,
            70.0,
            150.0,
        ),
        1,
    )
    if bool(is_mixed):
        cand["mixed_freq"] = round(
            _auto_optuna_suggest_centered_unit_float(
                trial,
                "mixed_freq",
                float(base_mixed),
                16.0 * s,
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
                float(base_phase),
                28.0 * s,
                float(AUTO_MODE_PHASE_LIMIT_MIN_HZ),
                float(AUTO_MODE_PHASE_LIMIT_MAX_HZ),
            ),
            1,
        )
    if _bi_search_enabled(dict(base_data or {})):
        cand.update(_suggest_bi_optuna_params(dict(base_data or {}), trial, coarse=False, center=p))
    return cand


def _seed_auto_mode_candidate_micro_optuna_params(
    base_data: dict,
    center: dict,
    preset: dict | None,
    *,
    shrink: float = 1.0,
) -> dict:
    base_data = _auto_filter_normalized_base_data(base_data)
    p = dict(base_data or {})
    p.update(dict(center or {}))
    d = dict(p)
    d.update(dict(preset or {}))
    s = float(np.clip(_auto_safe_float(shrink, 1.0), 0.25, 1.0))
    ft = str(p.get("filter_type", "") or "").strip().lower()
    is_mixed = "mixed" in ft
    is_phase_search = _auto_is_phase_search_filter(ft)
    base_mixed = _auto_safe_float(p.get("mixed_freq", 180.0), 180.0)
    base_phase = _auto_phase_limit_center(p.get("phase_limit"))
    base_tdc = _auto_safe_float(p.get("tdc_strength", 50.0), 50.0)
    base_fdw = _auto_safe_float(p.get("fdw_cycles", 10.0), 10.0)
    base_reg = _auto_safe_float(p.get("reg_strength", 30.0), 30.0)
    base_tw = _auto_safe_float(p.get("trans_width", 100.0), 100.0)

    def _windowed_unit(name: str, center_v: float, span: float, lo: float, hi: float, default: float) -> float:
        value = float(np.clip(_auto_safe_float(d.get(name, default), default), float(lo), float(hi)))
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
        "tdc_strength_u": _windowed_unit("tdc_strength", float(base_tdc), 8.0 * s, _TDC_STRENGTH_MIN, _TDC_STRENGTH_MAX, float(base_tdc)),
        "fdw_cycles_u": _windowed_unit("fdw_cycles", float(base_fdw), 1.0 * s, 6.0, 16.0, float(base_fdw)),
        "reg_strength_u": _windowed_unit("reg_strength", float(base_reg), 6.0 * s, 15.0, 45.0, float(base_reg)),
        "trans_width_u": _windowed_unit("trans_width", float(base_tw), 15.0 * s, 70.0, 150.0, float(base_tw)),
    }
    if bool(is_mixed):
        out["mixed_freq_u"] = _windowed_unit("mixed_freq", float(base_mixed), 16.0 * s, 80.0, 320.0, float(base_mixed))
    if bool(is_phase_search):
        out["phase_limit_u"] = _windowed_unit(
            "phase_limit",
            float(base_phase),
            28.0 * s,
            float(AUTO_MODE_PHASE_LIMIT_MIN_HZ),
            float(AUTO_MODE_PHASE_LIMIT_MAX_HZ),
            float(base_phase),
        )
    if _bi_search_enabled(dict(base_data or {})):
        out.update(_seed_bi_optuna_params(dict(base_data or {}), d))
    return dict(out)
