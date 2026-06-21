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
    MAX_SAFE_BOOST,
    _auto_safe_float,
)

logger = logging.getLogger("DecayCore")
AUTO_MODE_PREFER_BASS_MAX_NET_BOOST_HARD_GATE_DB = 12.0


def _auto_winner_metric(src: dict | None, *keys: str) -> float:
    if not isinstance(src, dict):
        return float("nan")
    for key in keys:
        value = _auto_safe_float(src.get(key, float("nan")), float("nan"))
        if np.isfinite(value):
            return float(value)
    return float("nan")


def _auto_winner_fmt(val: float, *, unit: str = "", decimals: int = 2) -> str:
    if not np.isfinite(val):
        return "n/a"
    return f"{float(val):.{int(decimals)}f}{unit}"


def _auto_winner_fmt_signed(val: float, *, unit: str = "", decimals: int = 2) -> str:
    if not np.isfinite(val):
        return "n/a"
    return f"{float(val):+.{int(decimals)}f}{unit}"


def _auto_winner_collect_deltas(best: dict, prev: dict) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for out_key, keys in (
        ("avg_score_delta", ("avg_score",)),
        ("rank_score_delta", ("rank_score",)),
        ("mode_ripple_delta", ("mode_ripple_db",)),
        ("residual_peak_delta", ("worst_residual_peak_db",)),
        ("boost_delta", ("max_net_boost_db",)),
        ("event_penalty_delta", ("event_penalty",)),
    ):
        best_v = _auto_winner_metric(best, *keys)
        prev_v = _auto_winner_metric(prev, *keys)
        if np.isfinite(best_v) and np.isfinite(prev_v):
            deltas[str(out_key)] = float(best_v - prev_v)
    return deltas


def _auto_winner_push(summary_bits: list[str], reasons: list[str], summary_bit: str | None, reason: str | None) -> None:
    sb = str(summary_bit or "").strip()
    rs = str(reason or "").strip()
    if sb:
        summary_bits.append(sb)
    if rs:
        reasons.append(rs)


def _auto_winner_add_rank_and_average(
    best: dict,
    *,
    deltas: dict[str, float],
    summary_bits: list[str],
    reasons: list[str],
) -> None:
    rank_score = _auto_winner_metric(best, "rank_score")
    rank_delta = deltas.get("rank_score_delta", float("nan"))
    if np.isfinite(rank_score):
        if np.isfinite(rank_delta) and float(rank_delta) > 0.0:
            _auto_winner_push(
                summary_bits,
                reasons,
                "improved rank score",
                f"Improved rank score to {_auto_winner_fmt(rank_score)} ({_auto_winner_fmt_signed(rank_delta)}).",
            )
        else:
            _auto_winner_push(summary_bits, reasons, "rank score " + _auto_winner_fmt(rank_score), f"Rank score {_auto_winner_fmt(rank_score)}.")

    avg_score = _auto_winner_metric(best, "avg_score")
    avg_delta = deltas.get("avg_score_delta", float("nan"))
    if np.isfinite(avg_score):
        if np.isfinite(avg_delta) and float(avg_delta) > 0.0:
            _auto_winner_push(
                summary_bits,
                reasons,
                "improved average score",
                f"Improved average score to {_auto_winner_fmt(avg_score)} ({_auto_winner_fmt_signed(avg_delta)}).",
            )
        else:
            _auto_winner_push(summary_bits, reasons, "average score " + _auto_winner_fmt(avg_score), f"Average score {_auto_winner_fmt(avg_score)}.")

    mode_ripple = _auto_winner_metric(best, "mode_ripple_db")
    mode_delta = deltas.get("mode_ripple_delta", float("nan"))
    if np.isfinite(mode_ripple):
        if np.isfinite(mode_delta) and float(mode_delta) < 0.0:
            _auto_winner_push(
                summary_bits,
                reasons,
                "reduced mode ripple",
                f"Reduced mode ripple to {_auto_winner_fmt(mode_ripple, unit=' dB')} ({_auto_winner_fmt_signed(mode_delta, unit=' dB')}).",
            )
        else:
            _auto_winner_push(summary_bits, reasons, "mode ripple " + _auto_winner_fmt(mode_ripple, unit=" dB"), f"Mode ripple {_auto_winner_fmt(mode_ripple, unit=' dB')}.")


def _auto_winner_add_residual_boost_and_penalty(
    best: dict,
    *,
    deltas: dict[str, float],
    summary_bits: list[str],
    reasons: list[str],
) -> None:
    residual_peak = _auto_winner_metric(best, "worst_residual_peak_db")
    residual_peak_hz = _auto_winner_metric(best, "worst_residual_peak_hz")
    residual_peak_delta = deltas.get("residual_peak_delta", float("nan"))
    if np.isfinite(residual_peak):
        hz_txt = f" @ {_auto_winner_fmt(residual_peak_hz, unit=' Hz', decimals=1)}" if np.isfinite(residual_peak_hz) else ""
        if np.isfinite(residual_peak_delta) and float(residual_peak_delta) < -0.05:
            _auto_winner_push(
                summary_bits,
                reasons,
                "reduced worst residual peak",
                f"Reduced worst residual peak to {_auto_winner_fmt(residual_peak, unit=' dB')}{hz_txt}.",
            )
        elif float(residual_peak) > 1.8:
            _auto_winner_push(
                summary_bits,
                reasons,
                "residual peaks remained elevated",
                f"Residual peaks remained elevated: {_auto_winner_fmt(residual_peak, unit=' dB')}{hz_txt}.",
            )
        else:
            _auto_winner_push(
                summary_bits,
                reasons,
                "residual peak controlled",
                f"Worst residual peak {_auto_winner_fmt(residual_peak, unit=' dB')}{hz_txt}.",
            )

    net_boost = _auto_winner_metric(best, "max_net_boost_db")
    boost_delta = deltas.get("boost_delta", float("nan"))
    if np.isfinite(net_boost):
        if np.isfinite(boost_delta) and float(boost_delta) < 0.0:
            _auto_winner_push(
                summary_bits,
                reasons,
                "reduced net boost",
                f"Reduced net boost to {_auto_winner_fmt(net_boost, unit=' dB')} ({_auto_winner_fmt_signed(boost_delta, unit=' dB')}).",
            )
        elif float(net_boost) <= float(MAX_SAFE_BOOST) * 0.5:
            _auto_winner_push(
                summary_bits,
                reasons,
                "kept net boost controlled",
                f"Kept net boost controlled at {_auto_winner_fmt(net_boost, unit=' dB')}.",
            )
        else:
            _auto_winner_push(summary_bits, reasons, "net boost " + _auto_winner_fmt(net_boost, unit=" dB"), f"Net boost {_auto_winner_fmt(net_boost, unit=' dB')}.")

    event_penalty = _auto_winner_metric(best, "event_penalty")
    event_delta = deltas.get("event_penalty_delta", float("nan"))
    if np.isfinite(event_penalty):
        if np.isfinite(event_delta) and float(event_delta) < 0.0:
            _auto_winner_push(
                summary_bits,
                reasons,
                "reduced event penalty",
                f"Reduced event penalty to {_auto_winner_fmt(event_penalty)} ({_auto_winner_fmt_signed(event_delta)}).",
            )
        elif abs(float(event_penalty)) <= 1e-9:
            _auto_winner_push(summary_bits, reasons, "avoided event penalty", "Avoided event penalty.")
        else:
            _auto_winner_push(summary_bits, reasons, "event penalty " + _auto_winner_fmt(event_penalty), f"Event penalty {_auto_winner_fmt(event_penalty)}.")


def _auto_winner_add_fit_and_phase_signals(best: dict, *, summary_bits: list[str], reasons: list[str]) -> None:
    focus_rms = _auto_winner_metric(best, "focus_rms_db")
    if np.isfinite(focus_rms):
        _auto_winner_push(summary_bits, reasons, "focus RMS " + _auto_winner_fmt(focus_rms, unit=" dB"), f"Focus RMS {_auto_winner_fmt(focus_rms, unit=' dB')}.")

    fit_rms = _auto_winner_metric(best, "fit_rms_db", "mode_fit_rms_db")
    if np.isfinite(fit_rms):
        _auto_winner_push(summary_bits, reasons, "target-fit RMS " + _auto_winner_fmt(fit_rms, unit=" dB"), f"Target-fit RMS {_auto_winner_fmt(fit_rms, unit=' dB')}.")

    rms_20_200 = _auto_winner_metric(best, "rms_20_200", "realized_rms_20_200_db")
    if np.isfinite(rms_20_200):
        _auto_winner_push(summary_bits, reasons, "LF RMS 20-200 Hz " + _auto_winner_fmt(rms_20_200, unit=" dB"), f"LF RMS (20-200 Hz) {_auto_winner_fmt(rms_20_200, unit=' dB')}.")

    phase_benefit = _auto_winner_metric(best, "phase_benefit_bonus")
    phase_risk = _auto_winner_metric(best, "phase_risk_penalty")
    if np.isfinite(phase_benefit) and float(phase_benefit) > 0.5:
        _auto_winner_push(
            summary_bits,
            reasons,
            "phase benefit " + _auto_winner_fmt(phase_benefit),
            f"Phase benefit bonus {_auto_winner_fmt(phase_benefit)} (positive phase correction contribution).",
        )
    if np.isfinite(phase_risk) and float(phase_risk) > 0.5:
        _auto_winner_push(
            summary_bits,
            reasons,
            "phase risk " + _auto_winner_fmt(phase_risk),
            f"Phase risk penalty {_auto_winner_fmt(phase_risk)} (phase correction risk).",
        )

    lr_delta = _auto_winner_metric(best, "lr_delta_score", "lr_delta_penalty")
    if np.isfinite(lr_delta) and abs(float(lr_delta)) > 0.5:
        _auto_winner_push(summary_bits, reasons, f"L/R delta {_auto_winner_fmt(lr_delta)}", f"L/R channel delta score {_auto_winner_fmt(lr_delta)}.")


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
    deltas = _auto_winner_collect_deltas(best, prev)
    reasons: list[str] = []
    summary_bits: list[str] = []
    _auto_winner_add_rank_and_average(best, deltas=deltas, summary_bits=summary_bits, reasons=reasons)
    _auto_winner_add_residual_boost_and_penalty(best, deltas=deltas, summary_bits=summary_bits, reasons=reasons)
    _auto_winner_add_fit_and_phase_signals(best, summary_bits=summary_bits, reasons=reasons)

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
