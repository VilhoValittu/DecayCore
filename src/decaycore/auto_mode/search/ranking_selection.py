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
    AUTO_MODE_GOAL_DEFAULT,
    AUTO_MODE_MAG_C_MAX_MIN_HZ,
    AUTO_MODE_PHASE2_PARETO_ACOUSTIC_DROP,
    AUTO_MODE_TARGET_BASS_FORWARD_MAX_RANK_DROP,
    AUTO_MODE_TARGET_BEST_RANK_TIE_EPS,
    _auto_goal_is_flat_family,
    _auto_safe_float,
)

logger = logging.getLogger("DecayCore")
AUTO_MODE_PREFER_BASS_MAX_NET_BOOST_HARD_GATE_DB = 12.0

from .ranking_gates import _auto_hard_gate_reasons, _auto_rank_value, filter_hard_failed_candidates
from .ranking_refine import _auto_rank_key_goal
from .ranking_phase2 import _auto_phase2_pareto_front, _auto_phase2_pick_pareto_winner
from .ranking_target import _auto_target_bass_forward_candidates, _auto_target_result_rank_key, _auto_target_result_tie_key


def _select_has_valid_auto_mag_c_max(item: dict) -> bool:
    for key in ("preset", "best_preset"):
        preset = dict(item.get(key, {}) or {})
        v = _auto_safe_float(preset.get("mag_c_max", float("nan")), float("nan"))
        if not np.isfinite(v):
            continue
        if float(v) < float(AUTO_MODE_MAG_C_MAX_MIN_HZ):
            return False
    return True


def _select_apply_mag_c_floor(pool: list[dict]) -> list[dict]:
    floor_pool = [dict(x or {}) for x in pool if _select_has_valid_auto_mag_c_max(dict(x or {}))]
    return floor_pool or list(pool)


def _select_apply_finite_rank_filter(pool: list[dict]) -> list[dict]:
    finite_pool = [
        dict(x or {})
        for x in pool
        if np.isfinite(_auto_rank_value(dict((x or {}).get("metrics", {}) or {}), default=float("nan")))
    ]
    return finite_pool or list(pool)


def _select_apply_hard_gate_filter(pool: list[dict], *, goal: str) -> list[dict]:
    hard_gate_pool, _hard_gate_diag = filter_hard_failed_candidates(pool, goal=goal)
    if hard_gate_pool:
        dropped = int(len(pool) - len(hard_gate_pool))
        if dropped > 0:
            logger.info("Auto-mode selection skipped %d hard-gated candidate(s).", int(dropped))
        return list(hard_gate_pool)
    logger.warning(
        "Auto-mode selection has no non-hard-gated candidates; falling back to ranked hard-gated pool (n=%d).",
        int(len(pool)),
    )
    return list(pool)


def _select_target_curve_winner(pool: list[dict]) -> dict:
    rank_tie_eps = float(
        max(
            0.0,
            _auto_safe_float(
                pool[0].get("_target_rank_tie_eps", AUTO_MODE_TARGET_BEST_RANK_TIE_EPS),
                AUTO_MODE_TARGET_BEST_RANK_TIE_EPS,
            ),
        )
    )
    winner = dict(sorted(pool, key=_auto_target_result_rank_key)[0])
    winner_rank = _auto_safe_float(
        dict(winner.get("best_metrics", {}) or {}).get("rank_score"),
        0.0,
    )
    winner["_auto_selection_method"] = "top3x10_trials"
    bass_forward_drop = float(
        max(
            float(rank_tie_eps),
            _auto_safe_float(
                pool[0].get(
                    "_target_bass_forward_max_rank_drop",
                    AUTO_MODE_TARGET_BASS_FORWARD_MAX_RANK_DROP,
                ),
                AUTO_MODE_TARGET_BASS_FORWARD_MAX_RANK_DROP,
            ),
        )
    )
    bass_forward = _auto_target_bass_forward_candidates(
        pool,
        winner,
        max_rank_drop=float(bass_forward_drop),
    )
    if bass_forward:
        winner = dict(sorted(bass_forward, key=_auto_target_result_rank_key)[0])
        winner["_auto_selection_method"] = "top3x10_trials_bass_forward_close_rank"
        return winner
    near_top = []
    for it in pool:
        it_rank = _auto_safe_float(
            dict(it.get("best_metrics", {}) or {}).get("rank_score"),
            0.0,
        )
        if abs(float(winner_rank) - float(it_rank)) < rank_tie_eps:
            near_top.append(dict(it))
    if len(near_top) >= 2:
        winner = dict(sorted(near_top, key=_auto_target_result_tie_key)[0])
        winner["_auto_selection_method"] = "top3x10_trials_rank_tie_composite"
    if bool(winner.get("from_cache_wildcard", False)):
        winner["_auto_selection_method"] = "trial_with_cache_wildcard"
    return winner


def _select_phase2_pareto_winner(pool: list[dict]) -> dict | None:
    acoustic_drop = float(
        max(
            0.0,
            _auto_safe_float(
                pool[0].get("_phase2_pareto_acoustic_drop", AUTO_MODE_PHASE2_PARETO_ACOUSTIC_DROP),
                AUTO_MODE_PHASE2_PARETO_ACOUSTIC_DROP,
            ),
        )
    )
    front = _auto_phase2_pareto_front(pool)
    winner = _auto_phase2_pick_pareto_winner(
        front,
        pool,
        acoustic_drop=float(acoustic_drop),
    )
    if isinstance(winner, dict):
        return dict(winner)
    return None


def _auto_select_best_scored(scored: list[dict], *, goal: str = AUTO_MODE_GOAL_DEFAULT) -> dict | None:
    pool = [dict(x or {}) for x in (scored or []) if isinstance(x, dict)]
    if not pool:
        return None

    pool = _select_apply_mag_c_floor(pool)
    pool = _select_apply_finite_rank_filter(pool)
    pool = _select_apply_hard_gate_filter(pool, goal=goal)

    select_kind = str(pool[0].get("_auto_select_kind", "rank_metrics") or "rank_metrics").strip().lower()
    if select_kind == "target_curve":
        return _select_target_curve_winner(pool)

    if select_kind == "phase2_pareto":
        winner = _select_phase2_pareto_winner(pool)
        if isinstance(winner, dict):
            return winner

    return dict(
        sorted(
            pool,
            key=lambda it: _auto_rank_key_goal(dict(it.get("metrics", {}) or {}), goal=goal),
        )[0]
    )


def _auto_reject(metrics: dict, st_l: dict | None, st_r: dict | None, goal: str) -> bool:
    if not _auto_goal_is_flat_family(goal):
        return False
    return bool(_auto_hard_gate_reasons(metrics, st_l, st_r, goal=goal))

__all__ = ["_auto_select_best_scored", "_auto_reject"]
