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

from ..shared import (
    AUTO_MODE_ADAPTIVE_SHRINK_ENABLED,
    AUTO_MODE_ADAPTIVE_SHRINK_MAX,
    AUTO_MODE_ADAPTIVE_SHRINK_MIN,
    AUTO_MODE_GOAL_DEFAULT,
    AUTO_MODE_GOAL_FLAT,
    AUTO_MODE_GOAL_LOW_RIPPLE,
    AUTO_MODE_GOAL_ROOM_SAFE,
    AUTO_MODE_GOAL_SUBWOOFERS,
    AUTO_MODE_HYBRID_MIXED_FREQ_SOFT_DEN_HZ,
    AUTO_MODE_HYBRID_MIXED_FREQ_SOFT_MAX_HZ,
    AUTO_MODE_MAG_C_MAX_MIN_HZ,
    AUTO_MODE_MAX_AVG_SCORE_LOSS_FOR_SAFETY_OVERRIDE,
    AUTO_MODE_PHASE2_HARD_GATE_ABS_MAX_PEAK_DB,
    AUTO_MODE_PHASE2_HARD_GATE_FALLBACK_TO_RANK,
    AUTO_MODE_PHASE2_HARD_GATE_KEEP_EVENT_FRACTION,
    AUTO_MODE_PHASE2_HARD_GATE_KEEP_PEAK_FRACTION,
    AUTO_MODE_PHASE2_HARD_GATE_KEEP_RIPPLE_FRACTION,
    AUTO_MODE_PHASE2_HARD_GATE_MIN_KEEP,
    AUTO_MODE_PHASE2_PARETO_ACOUSTIC_DROP,
    AUTO_MODE_PHASE2_PARETO_BOOST_EPS,
    AUTO_MODE_PHASE2_PARETO_MODE_RIPPLE_EPS,
    AUTO_MODE_PHASE2_PARETO_PREPOST_EPS,
    AUTO_MODE_PHASE2_PARETO_RMS20_200_EPS,
    AUTO_MODE_REFINE_MODE_BOOST_GUARD_MIN_RIPPLE_GAIN_DB,
    AUTO_MODE_REFINE_TIEBREAK_ENABLE,
    AUTO_MODE_REFINE_TIEBREAK_RANK_EPS,
    AUTO_MODE_REFINE_TIEBREAK_RIPPLE_EPS,
    AUTO_MODE_REFINE_TIEBREAK_PHASE_EPS,
    AUTO_MODE_REFINE_TIEBREAK_TRACKING_EPS,
    AUTO_MODE_TARGET_BASS_FORWARD_MAX_RANK_DROP,
    AUTO_MODE_TARGET_BEST_RANK_TIE_EPS,
    MAX_SAFE_BOOST,
    _auto_goal_norm,
    _auto_safe_float,
    _m,
)

logger = logging.getLogger("DecayCore")
AUTO_MODE_PREFER_BASS_MAX_NET_BOOST_HARD_GATE_DB = 12.0

from .ranking_gates import _auto_rank_value

def _tc_score(tc: dict | None) -> float:
    return float(
        _auto_safe_float(
            (tc or {}).get(
                "preselect_score",
                (tc or {}).get("fit_rms_db", float("inf")),
            ),
            float("inf"),
        )
    )


def _auto_target_result_rank_key(item: dict | None) -> tuple:
    bm = dict((item or {}).get("best_metrics", {}) or {})
    return (
        -_auto_safe_float(bm.get("rank_score"), 0.0),
        -_auto_safe_float((item or {}).get("avg_rank_score"), 0.0),
        _auto_safe_float((item or {}).get("fit_rms_db"), 1e9),
    )


def _auto_target_result_mode_ripple(item: dict | None) -> float:
    bm = dict((item or {}).get("best_metrics", {}) or {})
    v = _auto_safe_float(bm.get("mode_ripple_db", float("nan")), float("nan"))
    if np.isfinite(v):
        return float(v)
    return float("inf")


def _auto_target_mildness_index(hc_name: str) -> int:
    info = _auto_target_ladder_info(hc_name)
    if info is not None:
        return int(info[1])
    return 10_000


def _auto_target_ladder_info(hc_name: str) -> tuple[int, int] | None:
    name = str(hc_name or "").strip()
    if not name:
        return None
    ladders = (
        ("Harman4", "Harman6", "Harman8", "Harman10", "Harman12"),
        ("BK_Light", "BK_Medium", "BK_Strong"),
    )
    for ladder_idx, ladder in enumerate(ladders):
        if name in ladder:
            return int(ladder_idx), int(ladder.index(name))
    return None


def _auto_target_strength_tie_value(hc_name: str) -> int:
    info = _auto_target_ladder_info(hc_name)
    if info is None:
        return 0
    return -int(info[1])


def _auto_target_result_tie_key(item: dict | None) -> tuple:
    it = dict(item or {})
    return (
        -_auto_safe_float(it.get("avg_rank_score"), 0.0),
        _auto_target_result_mode_ripple(it),
        _auto_safe_float(it.get("boost_penalty", 0.0), 0.0),
        _auto_safe_float(it.get("fit_rms_db"), 1e9),
        _tc_score(it),
        _auto_target_strength_tie_value(str(it.get("hc_mode", "") or "").strip()),
        str(it.get("hc_mode", "") or "").strip(),
    )


def _auto_target_bass_forward_candidates(
    pool: list[dict],
    rank_winner: dict,
    *,
    max_rank_drop: float,
) -> list[dict]:
    winner_hc = str(rank_winner.get("hc_mode", "") or "").strip()
    winner_info = _auto_target_ladder_info(winner_hc)
    if winner_info is None:
        return []
    winner_ladder, winner_idx = winner_info
    winner_rank = _auto_safe_float(
        dict(rank_winner.get("best_metrics", {}) or {}).get("rank_score"),
        float("nan"),
    )
    if not np.isfinite(winner_rank):
        return []

    stronger: list[tuple[int, dict]] = []
    for it in pool:
        hc = str(dict(it or {}).get("hc_mode", "") or "").strip()
        info = _auto_target_ladder_info(hc)
        if info is None:
            continue
        ladder_idx, target_idx = info
        if int(ladder_idx) != int(winner_ladder) or int(target_idx) <= int(winner_idx):
            continue
        rank = _auto_safe_float(
            dict(dict(it or {}).get("best_metrics", {}) or {}).get("rank_score"),
            float("nan"),
        )
        if not np.isfinite(rank):
            continue
        rank_drop = float(winner_rank) - float(rank)
        if float(rank_drop) < 0.0 or float(rank_drop) > float(max_rank_drop):
            continue
        stronger.append((int(target_idx), dict(it or {})))

    if not stronger:
        return []
    next_idx = min(int(idx) for idx, _it in stronger)
    return [dict(it) for idx, it in stronger if int(idx) == int(next_idx)]



__all__ = ["_tc_score", "_auto_target_result_rank_key", "_auto_target_result_mode_ripple", "_auto_target_mildness_index", "_auto_target_ladder_info", "_auto_target_strength_tie_value", "_auto_target_result_tie_key", "_auto_target_bass_forward_candidates"]
