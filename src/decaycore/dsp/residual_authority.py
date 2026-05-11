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

from typing import Any

import numpy as np


def _authority_array(value, freq_axis, default):
    freq = np.asarray(freq_axis, dtype=float)
    if freq.ndim != 1:
        freq = freq.reshape(-1)
    if value is None:
        return np.full_like(freq, float(default), dtype=float)
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return np.full_like(freq, float(default), dtype=float)
    if arr.shape != freq.shape:
        return np.full_like(freq, float(default), dtype=float)
    arr = np.nan_to_num(arr, nan=float(default), posinf=1.0, neginf=0.0)
    return np.clip(arr, 0.0, 1.0)


def _pick_authority_array(st: Any, *names: str):
    if not isinstance(st, dict):
        return None
    for name in names:
        if name in st:
            return st.get(name)
    return None


def _mode_value(residual_pass_mode: str) -> str:
    mode = str(residual_pass_mode or "modal_polish").strip().lower()
    if mode not in ("modal_polish", "general_fit", "off"):
        return "modal_polish"
    return mode


def build_residual_authority_caps(
    freq_axis,
    *,
    authority_null_risk=None,
    authority_boost=None,
    authority_cut=None,
    authority_modal_support=None,
    authority_reflection_risk=None,
    residual_pass_mode: str = "modal_polish",
    max_boost_db: float = 2.0,
    max_cut_db: float = 4.0,
    null_guard_strength: float = 1.0,
    modal_min_support: float = 0.45,
    boost_authority_min: float = 0.40,
    cut_authority_min: float = 0.35,
    reflection_risk_max: float = 0.65,
    null_risk_max_for_boost: float = 0.35,
    null_risk_max_for_cut: float = 0.75,
    max_boost_when_null_risk_db: float = 0.5,
) -> dict:
    freq = np.asarray(freq_axis, dtype=float)
    if freq.ndim != 1:
        freq = freq.reshape(-1)
    mode = _mode_value(residual_pass_mode)
    max_boost = float(max(0.0, np.nan_to_num(max_boost_db, nan=2.0, posinf=2.0, neginf=0.0)))
    max_cut = float(max(0.0, np.nan_to_num(max_cut_db, nan=4.0, posinf=4.0, neginf=0.0)))
    null_strength = float(np.clip(np.nan_to_num(null_guard_strength, nan=1.0), 0.0, 1.0))
    modal_min = float(np.clip(np.nan_to_num(modal_min_support, nan=0.45), 0.0, 1.0))
    boost_min = float(np.clip(np.nan_to_num(boost_authority_min, nan=0.40), 0.0, 1.0))
    cut_min = float(np.clip(np.nan_to_num(cut_authority_min, nan=0.35), 0.0, 1.0))
    refl_max = float(np.clip(np.nan_to_num(reflection_risk_max, nan=0.65), 0.0, 1.0))
    null_boost_max = float(np.clip(np.nan_to_num(null_risk_max_for_boost, nan=0.35), 0.0, 1.0))
    null_cut_max = float(np.clip(np.nan_to_num(null_risk_max_for_cut, nan=0.75), 0.0, 1.0))
    null_boost_cap = float(max(0.0, np.nan_to_num(max_boost_when_null_risk_db, nan=0.5, posinf=0.5, neginf=0.0)))

    zero = np.zeros_like(freq, dtype=float)
    false = np.zeros_like(freq, dtype=bool)
    if mode == "off":
        return {
            "residual_boost_cap_db": zero.copy(),
            "residual_cut_cap_db": zero.copy(),
            "residual_allowed": false.copy(),
            "residual_boost_allowed": false.copy(),
            "residual_cut_allowed": false.copy(),
            "residual_modal_polish_mask": false.copy(),
            "residual_block_reason": {
                "null_risk_bins": 0,
                "reflection_risk_bins": 0,
                "low_authority_bins": 0,
                "low_modal_support_bins": 0,
            },
        }

    null_risk = _authority_array(authority_null_risk, freq, 0.0)
    reflection_risk = _authority_array(authority_reflection_risk, freq, 0.0)
    boost_authority = _authority_array(authority_boost, freq, 1.0)
    cut_authority = _authority_array(authority_cut, freq, 1.0)
    modal_default = 1.0 if mode == "general_fit" else 0.0
    modal_support = _authority_array(authority_modal_support, freq, modal_default)

    reflection_ok = reflection_risk <= refl_max
    boost_allowed = (
        (boost_authority >= boost_min)
        & (null_risk <= null_boost_max)
        & reflection_ok
    )
    modal_polish_mask = (modal_support >= modal_min) & reflection_ok

    if mode == "general_fit":
        cut_allowed = (
            (cut_authority >= cut_min)
            & (null_risk <= null_cut_max)
            & reflection_ok
        )
        boost_cap = max_boost * boost_authority
        boost_cap *= 1.0 - null_strength * 0.85 * null_risk
        boost_cap *= 1.0 - 0.65 * reflection_risk
        cut_cap = max_cut * (0.35 + 0.65 * cut_authority)
        cut_cap *= 1.0 - 0.45 * reflection_risk
    else:
        cut_allowed = (
            (modal_support >= modal_min)
            & (cut_authority >= cut_min)
            & reflection_ok
            & (null_risk <= null_cut_max)
        )
        boost_cap = max_boost * boost_authority
        boost_cap *= 1.0 - 0.90 * null_risk
        boost_cap *= 1.0 - 0.75 * reflection_risk
        boost_cap *= 0.35 + 0.65 * modal_support
        cut_cap = max_cut
        cut_cap *= 0.25 + 0.75 * cut_authority
        cut_cap *= 0.35 + 0.65 * modal_support
        cut_cap *= 1.0 - 0.45 * reflection_risk

    boost_cap = np.clip(np.nan_to_num(boost_cap, nan=0.0, posinf=max_boost, neginf=0.0), 0.0, max_boost)
    cut_cap = np.clip(np.nan_to_num(cut_cap, nan=0.0, posinf=max_cut, neginf=0.0), 0.0, max_cut)
    boost_cap = np.where(null_risk > null_boost_max, np.minimum(boost_cap, null_boost_cap), boost_cap)
    boost_cap = np.where(boost_allowed, boost_cap, 0.0)
    cut_cap = np.where(cut_allowed, cut_cap, 0.0)
    allowed = boost_allowed | cut_allowed

    return {
        "residual_boost_cap_db": np.asarray(boost_cap, dtype=float),
        "residual_cut_cap_db": np.asarray(cut_cap, dtype=float),
        "residual_allowed": np.asarray(allowed, dtype=bool),
        "residual_boost_allowed": np.asarray(boost_allowed, dtype=bool),
        "residual_cut_allowed": np.asarray(cut_allowed, dtype=bool),
        "residual_modal_polish_mask": np.asarray(modal_polish_mask, dtype=bool),
        "residual_block_reason": {
            "null_risk_bins": int(np.count_nonzero(null_risk > null_boost_max)),
            "reflection_risk_bins": int(np.count_nonzero(~reflection_ok)),
            "low_authority_bins": int(
                np.count_nonzero((boost_authority < boost_min) & (cut_authority < cut_min))
            ),
            "low_modal_support_bins": int(np.count_nonzero(modal_support < modal_min)),
        },
    }


def _apply_residual_authority_caps(residual_correction_db, caps: dict) -> np.ndarray:
    correction = np.asarray(residual_correction_db, dtype=float).copy()
    boost_cap = np.asarray(caps.get("residual_boost_cap_db"), dtype=float)
    cut_cap = np.asarray(caps.get("residual_cut_cap_db"), dtype=float)
    allowed = np.asarray(caps.get("residual_allowed"), dtype=bool)
    if boost_cap.shape != correction.shape or cut_cap.shape != correction.shape or allowed.shape != correction.shape:
        return np.nan_to_num(correction, nan=0.0, posinf=0.0, neginf=0.0)
    correction = np.nan_to_num(correction, nan=0.0, posinf=0.0, neginf=0.0)
    correction = np.minimum(correction, boost_cap)
    correction = np.maximum(correction, -cut_cap)
    correction[~allowed] = 0.0
    return np.asarray(correction, dtype=float)


__all__ = [
    "_apply_residual_authority_caps",
    "_authority_array",
    "_pick_authority_array",
    "build_residual_authority_caps",
]
