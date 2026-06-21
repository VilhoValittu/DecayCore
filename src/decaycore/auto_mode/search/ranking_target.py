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
    _auto_safe_float,
)

logger = logging.getLogger("DecayCore")
AUTO_MODE_PREFER_BASS_MAX_NET_BOOST_HARD_GATE_DB = 12.0


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
