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

def _auto_build_winner_explanation(
    best_metrics: dict | None,
    prev_metrics: dict | None = None,
    *,
    phase_label: str | None = None,
    target_name: str | None = None,
) -> dict:
    phase_txt = str(phase_label or "").strip() or None
    target_txt = str(target_name or "").strip() or None
    fallback = {
        "summary": "Winner explanation unavailable.",
        "reasons": [],
        "deltas": {},
        "phase_label": phase_txt,
        "target_name": target_txt,
    }
    if not isinstance(best_metrics, dict) or not best_metrics:
        return dict(fallback)

    best = dict(best_metrics or {})
    prev = dict(prev_metrics or {}) if isinstance(prev_metrics, dict) else {}

    def _metric(src: dict | None, *keys: str) -> float:
        if not isinstance(src, dict):
            return float("nan")
        for key in keys:
            v = _auto_safe_float(src.get(key, float("nan")), float("nan"))
            if np.isfinite(v):
                return float(v)
        return float("nan")

    def _fmt(val: float, *, unit: str = "", decimals: int = 2) -> str:
        if not np.isfinite(val):
            return "n/a"
        return f"{float(val):.{int(decimals)}f}{unit}"

    def _fmt_signed(val: float, *, unit: str = "", decimals: int = 2) -> str:
        if not np.isfinite(val):
            return "n/a"
        return f"{float(val):+.{int(decimals)}f}{unit}"

    deltas: dict[str, float] = {}
    delta_specs = (
        ("avg_score_delta", ("avg_score",)),
        ("rank_score_delta", ("rank_score",)),
        ("mode_ripple_delta", ("mode_ripple_db",)),
        ("residual_peak_delta", ("worst_residual_peak_db",)),
        ("boost_delta", ("max_net_boost_db",)),
        ("event_penalty_delta", ("event_penalty",)),
    )
    for out_key, keys in delta_specs:
        best_v = _metric(best, *keys)
        prev_v = _metric(prev, *keys)
        if np.isfinite(best_v) and np.isfinite(prev_v):
            deltas[str(out_key)] = float(best_v - prev_v)

    reasons: list[str] = []
    summary_bits: list[str] = []

    def _push(summary_bit: str | None, reason: str | None) -> None:
        sb = str(summary_bit or "").strip()
        rs = str(reason or "").strip()
        if sb:
            summary_bits.append(sb)
        if rs:
            reasons.append(rs)

    rank_score = _metric(best, "rank_score")
    rank_delta = deltas.get("rank_score_delta", float("nan"))
    if np.isfinite(rank_score):
        if np.isfinite(rank_delta) and float(rank_delta) > 0.0:
            _push(
                "improved rank score",
                f"Improved rank score to {_fmt(rank_score)} ({_fmt_signed(rank_delta)}).",
            )
        else:
            _push("rank score " + _fmt(rank_score), f"Rank score {_fmt(rank_score)}.")

    avg_score = _metric(best, "avg_score")
    avg_delta = deltas.get("avg_score_delta", float("nan"))
    if np.isfinite(avg_score):
        if np.isfinite(avg_delta) and float(avg_delta) > 0.0:
            _push(
                "improved average score",
                f"Improved average score to {_fmt(avg_score)} ({_fmt_signed(avg_delta)}).",
            )
        else:
            _push("average score " + _fmt(avg_score), f"Average score {_fmt(avg_score)}.")

    mode_ripple = _metric(best, "mode_ripple_db")
    mode_delta = deltas.get("mode_ripple_delta", float("nan"))
    if np.isfinite(mode_ripple):
        if np.isfinite(mode_delta) and float(mode_delta) < 0.0:
            _push(
                "reduced mode ripple",
                f"Reduced mode ripple to {_fmt(mode_ripple, unit=' dB')} ({_fmt_signed(mode_delta, unit=' dB')}).",
            )
        else:
            _push(
                "mode ripple " + _fmt(mode_ripple, unit=" dB"),
                f"Mode ripple {_fmt(mode_ripple, unit=' dB')}.",
            )

    residual_peak = _metric(best, "worst_residual_peak_db")
    residual_peak_hz = _metric(best, "worst_residual_peak_hz")
    residual_peak_delta = deltas.get("residual_peak_delta", float("nan"))
    if np.isfinite(residual_peak):
        hz_txt = f" @ {_fmt(residual_peak_hz, unit=' Hz', decimals=1)}" if np.isfinite(residual_peak_hz) else ""
        if np.isfinite(residual_peak_delta) and float(residual_peak_delta) < -0.05:
            _push(
                "reduced worst residual peak",
                f"Reduced worst residual peak to {_fmt(residual_peak, unit=' dB')}{hz_txt}.",
            )
        elif float(residual_peak) > 1.8:
            _push(
                "residual peaks remained elevated",
                f"Residual peaks remained elevated: {_fmt(residual_peak, unit=' dB')}{hz_txt}.",
            )
        else:
            _push(
                "residual peak controlled",
                f"Worst residual peak {_fmt(residual_peak, unit=' dB')}{hz_txt}.",
            )

    net_boost = _metric(best, "max_net_boost_db")
    boost_delta = deltas.get("boost_delta", float("nan"))
    if np.isfinite(net_boost):
        if np.isfinite(boost_delta) and float(boost_delta) < 0.0:
            _push(
                "reduced net boost",
                f"Reduced net boost to {_fmt(net_boost, unit=' dB')} ({_fmt_signed(boost_delta, unit=' dB')}).",
            )
        elif float(net_boost) <= float(MAX_SAFE_BOOST) * 0.5:
            _push(
                "kept net boost controlled",
                f"Kept net boost controlled at {_fmt(net_boost, unit=' dB')}.",
            )
        else:
            _push(
                "net boost " + _fmt(net_boost, unit=" dB"),
                f"Net boost {_fmt(net_boost, unit=' dB')}.",
            )

    event_penalty = _metric(best, "event_penalty")
    event_delta = deltas.get("event_penalty_delta", float("nan"))
    if np.isfinite(event_penalty):
        if np.isfinite(event_delta) and float(event_delta) < 0.0:
            _push(
                "reduced event penalty",
                f"Reduced event penalty to {_fmt(event_penalty)} ({_fmt_signed(event_delta)}).",
            )
        elif abs(float(event_penalty)) <= 1e-9:
            _push("avoided event penalty", "Avoided event penalty.")
        else:
            _push("event penalty " + _fmt(event_penalty), f"Event penalty {_fmt(event_penalty)}.")

    focus_rms = _metric(best, "focus_rms_db")
    if np.isfinite(focus_rms):
        _push("focus RMS " + _fmt(focus_rms, unit=" dB"), f"Focus RMS {_fmt(focus_rms, unit=' dB')}.")

    fit_rms = _metric(best, "fit_rms_db", "mode_fit_rms_db")
    if np.isfinite(fit_rms):
        _push("target-fit RMS " + _fmt(fit_rms, unit=" dB"), f"Target-fit RMS {_fmt(fit_rms, unit=' dB')}.")

    rms_20_200 = _metric(best, "rms_20_200", "realized_rms_20_200_db")
    if np.isfinite(rms_20_200):
        _push(
            "LF RMS 20-200 Hz " + _fmt(rms_20_200, unit=" dB"),
            f"LF RMS (20-200 Hz) {_fmt(rms_20_200, unit=' dB')}.",
        )

    phase_benefit = _metric(best, "phase_benefit_bonus")
    phase_risk = _metric(best, "phase_risk_penalty")
    if np.isfinite(phase_benefit) and float(phase_benefit) > 0.5:
        _push(
            f"phase benefit {_fmt(phase_benefit)}",
            f"Phase benefit bonus {_fmt(phase_benefit)} (positive phase correction contribution).",
        )
    if np.isfinite(phase_risk) and float(phase_risk) > 0.5:
        _push(
            f"phase risk {_fmt(phase_risk)}",
            f"Phase risk penalty {_fmt(phase_risk)} (phase correction risk).",
        )

    lr_delta = _metric(best, "lr_delta_score", "lr_delta_penalty")
    if np.isfinite(lr_delta) and abs(float(lr_delta)) > 0.5:
        _push(
            f"L/R delta {_fmt(lr_delta)}",
            f"L/R channel delta score {_fmt(lr_delta)}.",
        )

    if not reasons:
        return dict(fallback)

    summary_top = list(summary_bits[:3])
    if not summary_top:
        summary = "Winner selected based on available auto-mode metrics."
    elif len(summary_top) == 1:
        summary = f"Won on {summary_top[0]}."
    elif len(summary_top) == 2:
        summary = f"Won on {summary_top[0]} and {summary_top[1]}."
    else:
        summary = f"Won on {summary_top[0]}, {summary_top[1]}, and {summary_top[2]}."

    return {
        "summary": str(summary),
        "reasons": list(reasons),
        "deltas": dict(deltas),
        "phase_label": phase_txt,
        "target_name": target_txt,
    }



__all__ = ["_auto_build_winner_explanation"]
