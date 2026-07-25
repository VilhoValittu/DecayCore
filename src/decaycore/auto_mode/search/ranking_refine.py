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

from ..shared_parts import (
    AUTO_MODE_ADAPTIVE_SHRINK_ENABLED,
    AUTO_MODE_ADAPTIVE_SHRINK_MAX,
    AUTO_MODE_ADAPTIVE_SHRINK_MIN,
    AUTO_MODE_GOAL_ACOUSTIC,
    AUTO_MODE_GOAL_DEFAULT,
    AUTO_MODE_GOAL_FLAT,
    AUTO_MODE_GOAL_HYBRID,
    AUTO_MODE_GOAL_LOW_RIPPLE,
    AUTO_MODE_GOAL_PREFER_BASS,
    AUTO_MODE_GOAL_ROOM_SAFE,
    AUTO_MODE_GOAL_SUBWOOFERS,
    AUTO_MODE_REFINE_MODE_BOOST_GUARD_MIN_RIPPLE_GAIN_DB,
    AUTO_MODE_REFINE_TIEBREAK_ENABLE,
    AUTO_MODE_REFINE_TIEBREAK_RANK_EPS,
    AUTO_MODE_REFINE_TIEBREAK_RIPPLE_EPS,
    AUTO_MODE_REFINE_TIEBREAK_PHASE_EPS,
    AUTO_MODE_REFINE_TIEBREAK_TRACKING_EPS,
    _auto_goal_norm,
    _auto_safe_float,
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


def _refine_profile_collect_centers(phase1_top: list) -> tuple[list[float], list[float]]:
    mixed_vals: list[float] = []
    tdc_vals: list[float] = []
    for it in (phase1_top or []):
        p = dict(it.get("preset", {}) or {})
        mf = _auto_safe_float(p.get("mixed_freq", float("nan")), float("nan"))
        td = _auto_safe_float(p.get("tdc_strength", float("nan")), float("nan"))
        if np.isfinite(mf):
            mixed_vals.append(float(mf))
        if np.isfinite(td):
            tdc_vals.append(float(td))
    return list(mixed_vals), list(tdc_vals)


def _refine_profile_focus_bounds(*, mixed_center: float, base_data: dict) -> tuple[float, float]:
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
    return float(focus_lo), float(focus_hi)


def _refine_profile_default(base_data: dict) -> dict:
    mixed_center = 120.0
    focus_lo, focus_hi = _refine_profile_focus_bounds(
        mixed_center=float(mixed_center),
        base_data=base_data,
    )
    return {
        "mixed_center": float(mixed_center),
        "mixed_span": 60.0,
        "focus_lo": float(focus_lo),
        "focus_hi": float(focus_hi),
        "tdc_lo": 45.0,
        "tdc_hi": 70.0,
    }


def _refine_profile_from_mixed(*, mixed_vals: list[float], tdc_vals: list[float], base_data: dict) -> dict:
    mixed_center = float(np.median(mixed_vals))
    mixed_spread = float(np.std(mixed_vals)) if len(mixed_vals) > 1 else 20.0
    mixed_span = float(np.clip(mixed_spread * 1.5, 25.0, 80.0))
    focus_lo, focus_hi = _refine_profile_focus_bounds(
        mixed_center=float(mixed_center),
        base_data=base_data,
    )
    tdc_center = float(np.median(tdc_vals)) if tdc_vals else 60.0
    return {
        "mixed_center": mixed_center,
        "mixed_span": mixed_span,
        "focus_lo": float(focus_lo),
        "focus_hi": float(focus_hi),
        "tdc_lo": float(np.clip(tdc_center - 12.0, 35.0, 80.0)),
        "tdc_hi": float(np.clip(tdc_center + 12.0, 40.0, 85.0)),
    }


def _auto_build_refine_profile(
    *,
    base_data: dict,
    phase1_top: list,
) -> dict:
    mixed_vals, tdc_vals = _refine_profile_collect_centers(phase1_top)
    if not mixed_vals:
        return _refine_profile_default(base_data)
    return _refine_profile_from_mixed(
        mixed_vals=mixed_vals,
        tdc_vals=tdc_vals,
        base_data=base_data,
    )


def _auto_goal_uses_local_refine(goal: str | None) -> bool:
    g = _auto_goal_norm(goal)
    return bool(
        g in (
            AUTO_MODE_GOAL_DEFAULT,
            AUTO_MODE_GOAL_ROOM_SAFE,
            AUTO_MODE_GOAL_LOW_RIPPLE,
            AUTO_MODE_GOAL_SUBWOOFERS,
            AUTO_MODE_GOAL_FLAT,
            AUTO_MODE_GOAL_ACOUSTIC,
            AUTO_MODE_GOAL_HYBRID,
            AUTO_MODE_GOAL_PREFER_BASS,
        )
    )


def _auto_rank_key_goal(metrics: dict, goal: str = AUTO_MODE_GOAL_DEFAULT) -> tuple:
    g = _auto_goal_norm(goal)
    if g == AUTO_MODE_GOAL_ACOUSTIC:
        return _auto_rank_key_acoustic(metrics)
    if g == AUTO_MODE_GOAL_HYBRID:
        return _auto_rank_key_hybrid(metrics)
    if g == AUTO_MODE_GOAL_PREFER_BASS:
        return _auto_rank_key_prefer_bass(metrics)
    if g == AUTO_MODE_GOAL_FLAT:
        return _auto_rank_key_flat(metrics)
    if g == AUTO_MODE_GOAL_ROOM_SAFE:
        return _auto_rank_key_room_safe(metrics)
    if g == AUTO_MODE_GOAL_LOW_RIPPLE:
        return _auto_rank_key_low_ripple(metrics)
    return _auto_rank_key(metrics)


def _refine_mode_guard_or_accept(
    *,
    mode_improve: float,
    new_m: dict,
    best_m: dict,
    mode_guard_gain: float,
    accept_reason: str,
) -> tuple[bool, str]:
    new_boost = _auto_safe_float(new_m.get("max_net_boost_db"), 0.0)
    best_boost = _auto_safe_float(best_m.get("max_net_boost_db"), 0.0)
    boost_rise = float(new_boost - best_boost)
    if boost_rise > 1e-6 and float(mode_improve) <= float(mode_guard_gain):
        return False, "mode_guard"
    return True, accept_reason


def _refine_rank_improves_decision(
    *,
    raw_rank_diff: float,
    ref_rank_diff: float,
    rank_eps: float,
    tracking_pair_ok: bool,
    new_tracking: float,
    best_tracking: float,
    tracking_eps: float,
    mode_pair_ok: bool,
    best_mode_ripple: float,
    new_mode_ripple: float,
    ripple_eps: float,
    new_m: dict,
    best_m: dict,
    mode_guard_gain: float,
) -> tuple[bool, str]:
    if ref_rank_diff <= 1e-9:
        return False, "rank_refine"
    if abs(raw_rank_diff) <= rank_eps and tracking_pair_ok and float(new_tracking) > float(best_tracking) + tracking_eps:
        return False, "target_tracking"
    if abs(raw_rank_diff) <= rank_eps and tracking_pair_ok and float(best_tracking) > float(new_tracking) + tracking_eps:
        return True, "target_tracking"
    if abs(raw_rank_diff) <= rank_eps and mode_pair_ok:
        mode_improve = float(best_mode_ripple - new_mode_ripple)
        if mode_improve > ripple_eps:
            return _refine_mode_guard_or_accept(
                mode_improve=mode_improve,
                new_m=new_m,
                best_m=best_m,
                mode_guard_gain=mode_guard_gain,
                accept_reason="mode_ripple",
            )
    return True, "rank_refine"


def _refine_rank_worsens_decision(
    *,
    raw_rank_diff: float,
    ref_rank_diff: float,
    rank_eps: float,
    mode_pair_ok: bool,
    new_mode_ripple: float,
    best_mode_ripple: float,
    ripple_eps: float,
) -> tuple[bool, str]:
    if ref_rank_diff < -1e-9 and abs(raw_rank_diff) <= rank_eps and mode_pair_ok:
        if float(new_mode_ripple - best_mode_ripple) > ripple_eps:
            return False, "mode_ripple"
    return False, "rank_refine"


def _refine_tie_mode_step(
    out: tuple[bool, str],
    *,
    mode_pair_ok: bool,
    best_mode_ripple: float,
    new_mode_ripple: float,
    ripple_eps: float,
    new_m: dict,
    best_m: dict,
    mode_guard_gain: float,
) -> tuple[bool, str]:
    if mode_pair_ok:
        mode_improve = float(best_mode_ripple - new_mode_ripple)
        if mode_improve > ripple_eps:
            return _refine_mode_guard_or_accept(
                mode_improve=mode_improve,
                new_m=new_m,
                best_m=best_m,
                mode_guard_gain=mode_guard_gain,
                accept_reason="mode_ripple",
            )
        if float(new_mode_ripple - best_mode_ripple) > ripple_eps:
            return False, "mode_ripple"
    return out


def _refine_tie_tracking_step(
    out: tuple[bool, str],
    *,
    tracking_pair_ok: bool,
    best_tracking: float,
    new_tracking: float,
    tracking_eps: float,
) -> tuple[bool, str]:
    if out[1] == "rank_tie" and tracking_pair_ok:
        if float(best_tracking - new_tracking) > tracking_eps:
            return True, "target_tracking"
        if float(new_tracking - best_tracking) > tracking_eps:
            return False, "target_tracking"
    return out


def _refine_tie_focus_ripple_step(
    out: tuple[bool, str],
    *,
    new_m: dict,
    best_m: dict,
    ripple_eps: float,
) -> tuple[bool, str]:
    if out[1] == "rank_tie":
        new_ripple = _auto_safe_float(new_m.get("focus_ripple_db"), float("nan"))
        best_ripple = _auto_safe_float(best_m.get("focus_ripple_db"), float("nan"))
        if np.isfinite(new_ripple) and np.isfinite(best_ripple):
            if float(best_ripple - new_ripple) > ripple_eps:
                return True, "focus_ripple"
            if float(new_ripple - best_ripple) > ripple_eps:
                return False, "focus_ripple"
    return out


def _refine_tie_phase_net_step(out: tuple[bool, str], *, new_m: dict, best_m: dict) -> tuple[bool, str]:
    if out[1] == "rank_tie":
        phase_eps = float(max(0.0, _auto_safe_float(AUTO_MODE_REFINE_TIEBREAK_PHASE_EPS, 0.10)))
        new_phase_net = _auto_phase_net_for_rank(new_m)
        best_phase_net = _auto_phase_net_for_rank(best_m)
        if float(new_phase_net - best_phase_net) > phase_eps:
            return True, "phase_net"
        if float(best_phase_net - new_phase_net) > phase_eps:
            return False, "phase_net"
    return out


def _refine_tie_phase_risk_step(out: tuple[bool, str], *, new_m: dict, best_m: dict) -> tuple[bool, str]:
    if out[1] == "rank_tie":
        phase_risk_eps = 0.10
        new_phase_risk = _auto_phase_risk_for_rank(new_m)
        best_phase_risk = _auto_phase_risk_for_rank(best_m)
        if float(best_phase_risk - new_phase_risk) > phase_risk_eps:
            return True, "phase_risk"
        if float(new_phase_risk - best_phase_risk) > phase_risk_eps:
            return False, "phase_risk"
    return out


def _refine_rank_tie_tiebreak(
    *,
    out: tuple[bool, str],
    mode_pair_ok: bool,
    best_mode_ripple: float,
    new_mode_ripple: float,
    ripple_eps: float,
    new_m: dict,
    best_m: dict,
    mode_guard_gain: float,
    tracking_pair_ok: bool,
    best_tracking: float,
    new_tracking: float,
    tracking_eps: float,
) -> tuple[bool, str]:
    out = _refine_tie_mode_step(
        out,
        mode_pair_ok=mode_pair_ok,
        best_mode_ripple=best_mode_ripple,
        new_mode_ripple=new_mode_ripple,
        ripple_eps=ripple_eps,
        new_m=new_m,
        best_m=best_m,
        mode_guard_gain=mode_guard_gain,
    )
    out = _refine_tie_tracking_step(
        out,
        tracking_pair_ok=tracking_pair_ok,
        best_tracking=best_tracking,
        new_tracking=new_tracking,
        tracking_eps=tracking_eps,
    )
    out = _refine_tie_focus_ripple_step(out, new_m=new_m, best_m=best_m, ripple_eps=ripple_eps)
    out = _refine_tie_phase_net_step(out, new_m=new_m, best_m=best_m)
    out = _refine_tie_phase_risk_step(out, new_m=new_m, best_m=best_m)
    return out


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
        out = _refine_rank_improves_decision(
            raw_rank_diff=raw_rank_diff,
            ref_rank_diff=ref_rank_diff,
            rank_eps=rank_eps,
            tracking_pair_ok=tracking_pair_ok,
            new_tracking=float(new_tracking),
            best_tracking=float(best_tracking),
            tracking_eps=tracking_eps,
            mode_pair_ok=mode_pair_ok,
            best_mode_ripple=float(best_mode_ripple),
            new_mode_ripple=float(new_mode_ripple),
            ripple_eps=ripple_eps,
            new_m=new_m,
            best_m=best_m,
            mode_guard_gain=mode_guard_gain,
        )
    elif ref_rank_diff < -1e-9:
        out = _refine_rank_worsens_decision(
            raw_rank_diff=raw_rank_diff,
            ref_rank_diff=ref_rank_diff,
            rank_eps=rank_eps,
            mode_pair_ok=mode_pair_ok,
            new_mode_ripple=float(new_mode_ripple),
            best_mode_ripple=float(best_mode_ripple),
            ripple_eps=ripple_eps,
        )
    else:
        out = (False, "rank_tie")
        if bool(AUTO_MODE_REFINE_TIEBREAK_ENABLE):
            out = _refine_rank_tie_tiebreak(
                out=out,
                mode_pair_ok=mode_pair_ok,
                best_mode_ripple=float(best_mode_ripple),
                new_mode_ripple=float(new_mode_ripple),
                ripple_eps=ripple_eps,
                new_m=new_m,
                best_m=best_m,
                mode_guard_gain=mode_guard_gain,
                tracking_pair_ok=tracking_pair_ok,
                best_tracking=float(best_tracking),
                new_tracking=float(new_tracking),
                tracking_eps=tracking_eps,
            )

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
        return _auto_clip_adaptive_shrink(base * (0.85 if bool(plateau_hit) else 1.0))
    spread_score = _auto_adaptive_spread_score(phase1_top)
    mul = _auto_adaptive_shrink_multiplier(spread_score)
    if bool(plateau_hit):
        mul *= 0.90
    return _auto_clip_adaptive_shrink(float(base * mul))


def _auto_clip_adaptive_shrink(value: float) -> float:
    return float(np.clip(float(value), AUTO_MODE_ADAPTIVE_SHRINK_MIN, AUTO_MODE_ADAPTIVE_SHRINK_MAX))


def _auto_adaptive_spread_score(phase1_top: list[dict]) -> float:
    mixed: list[float] = []
    tdc: list[float] = []
    fdw: list[float] = []
    reg: list[float] = []
    for item in phase1_top[:4]:
        preset = dict((item or {}).get("preset", {}) or {})
        mixed.append(_auto_safe_float(preset.get("mixed_freq", float("nan")), float("nan")))
        tdc.append(_auto_safe_float(preset.get("tdc_strength", float("nan")), float("nan")))
        fdw.append(_auto_safe_float(preset.get("fdw_cycles", float("nan")), float("nan")))
        reg.append(_auto_safe_float(preset.get("reg_strength", float("nan")), float("nan")))
    return float(
        (_auto_spread(mixed) / 80.0)
        + (_auto_spread(tdc) / 15.0)
        + (_auto_spread(fdw) / 3.0)
        + (_auto_spread(reg) / 20.0)
    )


def _auto_spread(values: list[float]) -> float:
    finite_values = [float(v) for v in values if np.isfinite(v)]
    if len(finite_values) < 2:
        return 0.0
    finite_values = sorted(finite_values)
    return float(finite_values[-1] - finite_values[0])


def _auto_adaptive_shrink_multiplier(spread_score: float) -> float:
    if spread_score <= 0.35:
        return 0.75
    if spread_score <= 0.70:
        return 0.85
    if spread_score <= 1.10:
        return 0.95
    return 1.05



__all__ = ["_auto_build_refine_profile", "_auto_goal_uses_local_refine", "_auto_rank_key_goal", "_auto_is_better_refine", "_auto_adaptive_shrink_factor"]
