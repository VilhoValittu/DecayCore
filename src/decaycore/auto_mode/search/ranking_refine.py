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

from .ranking_gates import _auto_hard_gate_reasons
from .ranking_keys import (
    _auto_rank_key,
    _auto_rank_key_acoustic,
    _auto_rank_key_flat,
    _auto_rank_key_hybrid,
    _auto_rank_key_low_ripple,
    _auto_rank_key_prefer_bass,
    _auto_rank_key_room_safe,
    _auto_phase_net_for_rank,
    _auto_phase_risk_for_rank,
    _auto_target_tracking_for_pareto,
)

def _auto_build_refine_profile(
    *,
    base_data: dict,
    phase1_top: list,
) -> dict:
    mixed_vals = []
    tdc_vals = []
    for it in (phase1_top or []):
        p = dict(it.get("preset", {}) or {})
        mf = _auto_safe_float(p.get("mixed_freq", float("nan")), float("nan"))
        td = _auto_safe_float(p.get("tdc_strength", float("nan")), float("nan"))
        if np.isfinite(mf):
            mixed_vals.append(float(mf))
        if np.isfinite(td):
            tdc_vals.append(float(td))

    if not mixed_vals:
        mixed_center = 120.0
        focus_lo = float(max(20.0, float(mixed_center) - 70.0))
        focus_hi = float(min(220.0, float(mixed_center) + 50.0))
        bf_hi = float("nan")
        if bool(base_data.get("bass_first_ai", True)):
            bf_hi = _auto_safe_float(base_data.get("bass_first_mode_max_hz", 200.0), 200.0)
            if np.isfinite(bf_hi):
                focus_hi = min(float(focus_hi), float(bf_hi))
        focus_lo = float(np.clip(focus_lo, 20.0, 200.0))
        focus_hi = float(np.clip(focus_hi, 60.0, 220.0))
        if np.isfinite(bf_hi):
            focus_hi = min(float(focus_hi), float(bf_hi))
        if focus_hi <= focus_lo:
            focus_lo = float(np.clip(min(float(focus_lo), float(focus_hi) - 5.0), 20.0, 200.0))
        if focus_hi <= focus_lo:
            focus_hi = float(np.clip(float(focus_lo) + 5.0, 60.0, 220.0))
        return {
            "mixed_center": float(mixed_center),
            "mixed_span": 60.0,
            "focus_lo": float(focus_lo),
            "focus_hi": float(focus_hi),
            "tdc_lo": 45.0,
            "tdc_hi": 70.0,
        }

    mixed_center = float(np.median(mixed_vals))
    mixed_spread = float(np.std(mixed_vals)) if len(mixed_vals) > 1 else 20.0
    mixed_span = float(np.clip(mixed_spread * 1.5, 25.0, 80.0))
    focus_lo = float(max(20.0, float(mixed_center) - 70.0))
    focus_hi = float(min(220.0, float(mixed_center) + 50.0))
    bf_hi = float("nan")
    if bool(base_data.get("bass_first_ai", True)):
        bf_hi = _auto_safe_float(base_data.get("bass_first_mode_max_hz", 200.0), 200.0)
        if np.isfinite(bf_hi):
            focus_hi = min(float(focus_hi), float(bf_hi))
    focus_hi = float(np.clip(focus_hi, 60.0, 220.0))
    if np.isfinite(bf_hi):
        focus_hi = min(float(focus_hi), float(bf_hi))
    if focus_hi <= focus_lo:
        focus_lo = float(np.clip(min(float(focus_lo), float(focus_hi) - 5.0), 20.0, 200.0))
    if focus_hi <= focus_lo:
        focus_hi = float(np.clip(float(focus_lo) + 5.0, 60.0, 220.0))

    tdc_center = float(np.median(tdc_vals)) if tdc_vals else 60.0
    return {
        "mixed_center": mixed_center,
        "mixed_span": mixed_span,
        "focus_lo": float(focus_lo),
        "focus_hi": float(focus_hi),
        "tdc_lo": float(np.clip(tdc_center - 12.0, 35.0, 80.0)),
        "tdc_hi": float(np.clip(tdc_center + 12.0, 40.0, 85.0)),
    }


def _auto_goal_uses_local_refine(goal: str | None) -> bool:
    g = _auto_goal_norm(goal)
    return bool(
        g in (
            AUTO_MODE_GOAL_DEFAULT,
            AUTO_MODE_GOAL_ROOM_SAFE,
            AUTO_MODE_GOAL_LOW_RIPPLE,
            AUTO_MODE_GOAL_SUBWOOFERS,
            AUTO_MODE_GOAL_FLAT,
        )
    )


def _auto_rank_key_goal(metrics: dict, goal: str = AUTO_MODE_GOAL_DEFAULT) -> tuple:
    raw_goal = str(goal or "").strip().lower().replace("_", "-")
    if raw_goal == "acoustic":
        return _auto_rank_key_acoustic(metrics)
    if raw_goal == "hybrid":
        return _auto_rank_key_hybrid(metrics)
    if raw_goal in ("prefer bass", "prefer-bass", "bass"):
        return _auto_rank_key_prefer_bass(metrics)
    g = _auto_goal_norm(goal)
    if g == AUTO_MODE_GOAL_FLAT:
        return _auto_rank_key_flat(metrics)
    if g == AUTO_MODE_GOAL_ROOM_SAFE:
        return _auto_rank_key_room_safe(metrics)
    if g == AUTO_MODE_GOAL_LOW_RIPPLE:
        return _auto_rank_key_low_ripple(metrics)
    return _auto_rank_key(metrics)


def _auto_is_better_refine(
    new_metrics: dict,
    best_metrics: dict,
    goal: str = AUTO_MODE_GOAL_DEFAULT,
    *,
    return_reason: bool = False,
) -> bool | tuple[bool, str]:
    new_m = dict(new_metrics or {})
    best_m = dict(best_metrics or {})
    new_hard_gates = _auto_hard_gate_reasons(new_m, goal=goal)
    best_hard_gates = _auto_hard_gate_reasons(best_m, goal=goal)
    if new_hard_gates and not best_hard_gates:
        out = (False, "hard_gate:" + ",".join(new_hard_gates))
        return out if bool(return_reason) else bool(out[0])
    if best_hard_gates and not new_hard_gates:
        out = (True, "best_hard_gate:" + ",".join(best_hard_gates))
        return out if bool(return_reason) else bool(out[0])
    rank_eps = float(max(0.0, _auto_safe_float(AUTO_MODE_REFINE_TIEBREAK_RANK_EPS, 0.20)))
    ripple_eps = float(max(0.0, _auto_safe_float(AUTO_MODE_REFINE_TIEBREAK_RIPPLE_EPS, 0.02)))
    mode_guard_gain = float(
        max(0.0, _auto_safe_float(AUTO_MODE_REFINE_MODE_BOOST_GUARD_MIN_RIPPLE_GAIN_DB, 0.06))
    )
    new_rank_raw = _auto_safe_float(new_m.get("rank_score"), 0.0)
    best_rank_raw = _auto_safe_float(best_m.get("rank_score"), 0.0)
    raw_rank_diff = float(new_rank_raw - best_rank_raw)
    new_rank_ref = _auto_safe_float(new_m.get("rank_score_refine", new_rank_raw), new_rank_raw)
    best_rank_ref = _auto_safe_float(best_m.get("rank_score_refine", best_rank_raw), best_rank_raw)
    ref_rank_diff = float(new_rank_ref - best_rank_ref)
    new_mode_ripple = _auto_safe_float(new_m.get("mode_ripple_db"), float("nan"))
    best_mode_ripple = _auto_safe_float(best_m.get("mode_ripple_db"), float("nan"))
    mode_pair_ok = bool(np.isfinite(new_mode_ripple) and np.isfinite(best_mode_ripple))
    new_tracking = _auto_target_tracking_for_pareto(new_m)
    best_tracking = _auto_target_tracking_for_pareto(best_m)
    tracking_pair_ok = bool(np.isfinite(new_tracking) and np.isfinite(best_tracking))
    tracking_eps = float(max(0.0, _auto_safe_float(AUTO_MODE_REFINE_TIEBREAK_TRACKING_EPS, 0.05)))

    if ref_rank_diff > 1e-9:
        if abs(raw_rank_diff) <= rank_eps and tracking_pair_ok and float(new_tracking) > float(best_tracking) + tracking_eps:
            out = (False, "target_tracking")
        elif abs(raw_rank_diff) <= rank_eps and tracking_pair_ok and float(best_tracking) > float(new_tracking) + tracking_eps:
            out = (True, "target_tracking")
        elif abs(raw_rank_diff) <= rank_eps and mode_pair_ok:
            mode_improve = float(best_mode_ripple - new_mode_ripple)
            if mode_improve > ripple_eps:
                new_boost = _auto_safe_float(new_m.get("max_net_boost_db"), 0.0)
                best_boost = _auto_safe_float(best_m.get("max_net_boost_db"), 0.0)
                boost_rise = float(new_boost - best_boost)
                if boost_rise > 1e-6 and mode_improve <= mode_guard_gain:
                    out = (False, "mode_guard")
                else:
                    out = (True, "mode_ripple")
            else:
                out = (True, "rank_refine")
        else:
            out = (True, "rank_refine")
    elif ref_rank_diff < -1e-9:
        if abs(raw_rank_diff) <= rank_eps and mode_pair_ok and (float(new_mode_ripple - best_mode_ripple) > ripple_eps):
            out = (False, "mode_ripple")
        else:
            out = (False, "rank_refine")
    else:
        out = (False, "rank_tie")
        if bool(AUTO_MODE_REFINE_TIEBREAK_ENABLE):
            if mode_pair_ok:
                mode_improve = float(best_mode_ripple - new_mode_ripple)
                if mode_improve > ripple_eps:
                    new_boost = _auto_safe_float(new_m.get("max_net_boost_db"), 0.0)
                    best_boost = _auto_safe_float(best_m.get("max_net_boost_db"), 0.0)
                    boost_rise = float(new_boost - best_boost)
                    if boost_rise > 1e-6 and mode_improve <= mode_guard_gain:
                        out = (False, "mode_guard")
                    else:
                        out = (True, "mode_ripple")
                elif float(new_mode_ripple - best_mode_ripple) > ripple_eps:
                    out = (False, "mode_ripple")

            if out[1] == "rank_tie":
                if tracking_pair_ok:
                    if float(best_tracking - new_tracking) > tracking_eps:
                        out = (True, "target_tracking")
                    elif float(new_tracking - best_tracking) > tracking_eps:
                        out = (False, "target_tracking")

            if out[1] == "rank_tie":
                new_ripple = _auto_safe_float(new_m.get("focus_ripple_db"), float("nan"))
                best_ripple = _auto_safe_float(best_m.get("focus_ripple_db"), float("nan"))
                if np.isfinite(new_ripple) and np.isfinite(best_ripple):
                    if float(best_ripple - new_ripple) > ripple_eps:
                        out = (True, "focus_ripple")
                    elif float(new_ripple - best_ripple) > ripple_eps:
                        out = (False, "focus_ripple")

            if out[1] == "rank_tie":
                phase_eps = float(max(0.0, _auto_safe_float(AUTO_MODE_REFINE_TIEBREAK_PHASE_EPS, 0.10)))
                new_phase_net = _auto_phase_net_for_rank(new_m)
                best_phase_net = _auto_phase_net_for_rank(best_m)
                if float(new_phase_net - best_phase_net) > phase_eps:
                    out = (True, "phase_net")
                elif float(best_phase_net - new_phase_net) > phase_eps:
                    out = (False, "phase_net")

            if out[1] == "rank_tie":
                phase_risk_eps = 0.10
                new_phase_risk = _auto_phase_risk_for_rank(new_m)
                best_phase_risk = _auto_phase_risk_for_rank(best_m)
                if float(best_phase_risk - new_phase_risk) > phase_risk_eps:
                    out = (True, "phase_risk")
                elif float(new_phase_risk - best_phase_risk) > phase_risk_eps:
                    out = (False, "phase_risk")

        if out[1] == "rank_tie":
            out = (
                bool(_auto_rank_key_goal(new_m, goal) < _auto_rank_key_goal(best_m, goal)),
                "goal_key",
            )
    return out if bool(return_reason) else bool(out[0])


def _auto_adaptive_shrink_factor(
    phase1_top: list[dict],
    *,
    base_shrink: float,
    plateau_hit: bool,
) -> float:
    if not bool(AUTO_MODE_ADAPTIVE_SHRINK_ENABLED):
        return float(base_shrink)
    base = float(np.clip(_auto_safe_float(base_shrink, 0.35), 0.05, 1.0))
    if not isinstance(phase1_top, list) or len(phase1_top) < 2:
        if bool(plateau_hit):
            return float(np.clip(base * 0.85, AUTO_MODE_ADAPTIVE_SHRINK_MIN, AUTO_MODE_ADAPTIVE_SHRINK_MAX))
        return float(np.clip(base, AUTO_MODE_ADAPTIVE_SHRINK_MIN, AUTO_MODE_ADAPTIVE_SHRINK_MAX))

    mixed = []
    tdc = []
    fdw = []
    reg = []
    for it in phase1_top[:4]:
        p = dict((it or {}).get("preset", {}) or {})
        mixed.append(_auto_safe_float(p.get("mixed_freq", float("nan")), float("nan")))
        tdc.append(_auto_safe_float(p.get("tdc_strength", float("nan")), float("nan")))
        fdw.append(_auto_safe_float(p.get("fdw_cycles", float("nan")), float("nan")))
        reg.append(_auto_safe_float(p.get("reg_strength", float("nan")), float("nan")))

    def _spread(vals: list[float]) -> float:
        vv = [float(v) for v in vals if np.isfinite(v)]
        if len(vv) < 2:
            return 0.0
        vv = sorted(vv)
        return float(vv[-1] - vv[0])

    spread_score = 0.0
    spread_score += _spread(mixed) / 80.0
    spread_score += _spread(tdc) / 15.0
    spread_score += _spread(fdw) / 3.0
    spread_score += _spread(reg) / 20.0
    if spread_score <= 0.35:
        mul = 0.75
    elif spread_score <= 0.70:
        mul = 0.85
    elif spread_score <= 1.10:
        mul = 0.95
    else:
        mul = 1.05
    if bool(plateau_hit):
        mul *= 0.90
    out = float(base * mul)
    return float(np.clip(out, AUTO_MODE_ADAPTIVE_SHRINK_MIN, AUTO_MODE_ADAPTIVE_SHRINK_MAX))



__all__ = ["_auto_build_refine_profile", "_auto_goal_uses_local_refine", "_auto_rank_key_goal", "_auto_is_better_refine", "_auto_adaptive_shrink_factor"]
