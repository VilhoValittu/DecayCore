# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Finalize-stage cache helpers and formatting utilities for automatic mode."""

from __future__ import annotations

import logging

import numpy as np

from ..rank_score import official_rank_score
from ..scoring_ranking import (
    _auto_mode_ripple_for_pareto,
    _auto_prepost_for_pareto,
)
from ..shared import _auto_safe_float, _m

logger = logging.getLogger("DecayCore")

_LOW_BASS_CUT_WINNER_POLISH_STEP_HZ = 2.0
_LOW_BASS_CUT_WINNER_POLISH_MAX_DELTA_HZ = 8.0

def _fmt_status_metric(value: float, *, decimals: int = 3, unit: str = "") -> str:
    if np.isfinite(value):
        return f"{float(value):.{int(decimals)}f}{unit}"
    return "n/a"

def _build_phase2_pareto_status(
    *,
    rank_best_metrics: dict | None,
    winner_metrics: dict | None,
) -> str:
    rank_best = dict(rank_best_metrics or {})
    winner = dict(winner_metrics or {})
    return (
        "DecayCore automatic mode: phase 2 pareto comparison "
        f"(rank_best {_fmt_status_metric(official_rank_score(rank_best), decimals=3)} -> "
        f"pareto {_fmt_status_metric(official_rank_score(winner), decimals=3)}, "
        f"avg {_fmt_status_metric(_m(rank_best, 'avg_score', float('nan')), decimals=3)} -> "
        f"{_fmt_status_metric(_m(winner, 'avg_score', float('nan')), decimals=3)}, "
        f"prepost {_fmt_status_metric(_auto_prepost_for_pareto(rank_best), decimals=4)} -> "
        f"{_fmt_status_metric(_auto_prepost_for_pareto(winner), decimals=4)}, "
        f"mode_ripple {_fmt_status_metric(_auto_mode_ripple_for_pareto(rank_best), decimals=3, unit=' dB')} -> "
        f"{_fmt_status_metric(_auto_mode_ripple_for_pareto(winner), decimals=3, unit=' dB')})"
    )

def _build_modal_intelligence_debug(best_metrics: dict | None, polish_meta: dict | None) -> dict:
    metrics = dict(best_metrics or {})
    polish = dict(polish_meta or {})
    modal_events = [dict(event) for event in list(metrics.get("modal_events", []) or []) if isinstance(event, dict)]
    residual_hz = metrics.get("residual_peak_modal_dominant_freq_hz")
    tdc_reductions = [
        dict(item) for item in list(metrics.get("tdc_modal_reductions", []) or []) if isinstance(item, dict)
    ]
    residual_polish = dict(polish.get("residual_peak_winner_polish", {}) or {})
    strongest: list[dict] = []
    for event in modal_events[:5]:
        freq = _auto_safe_float(event.get("freq_hz"), float("nan"))
        used_by: list[str] = []
        if np.isfinite(freq) and residual_hz is not None:
            res_f = _auto_safe_float(residual_hz, float("nan"))
            if np.isfinite(res_f) and res_f > 0.0 and abs(float(np.log2(float(freq) / float(res_f)))) <= 0.12:
                used_by.append("residual_peak")
        for reduction in tdc_reductions:
            red_f = _auto_safe_float(reduction.get("freq_hz"), float("nan"))
            if np.isfinite(freq) and np.isfinite(red_f) and red_f > 0.0:
                if abs(float(np.log2(float(freq) / float(red_f)))) <= 0.12:
                    used_by.append("tdc")
                    break
        if bool(residual_polish.get("applicable", False)) and _auto_safe_float(
            residual_polish.get("modal_priority", 0.0),
            0.0,
        ) >= 0.25:
            polish_f = _auto_safe_float(residual_polish.get("modal_dominant_freq_hz"), float("nan"))
            if np.isfinite(freq) and np.isfinite(polish_f) and polish_f > 0.0:
                if abs(float(np.log2(float(freq) / float(polish_f)))) <= 0.12:
                    used_by.append("winner_polish")
        strongest.append(
            {
                "freq_hz": float(freq) if np.isfinite(freq) else float("nan"),
                "severity": float(_auto_safe_float(event.get("severity", 0.0), 0.0)),
                "confidence": float(_auto_safe_float(event.get("confidence", 0.0), 0.0)),
                "peak_db": float(_auto_safe_float(event.get("peak_db", 0.0), 0.0)),
                "gd_excess_ms": float(_auto_safe_float(event.get("gd_excess_ms", 0.0), 0.0)),
                "safe_cut_db": float(_auto_safe_float(event.get("safe_cut_db", 0.0), 0.0)),
                "used_by": sorted(set(used_by)),
                "kind": str(event.get("kind", "unknown") or "unknown"),
            }
        )
    return {
        "enabled": True,
        "event_count": int(metrics.get("modal_mode_count", 0) or 0),
        "strongest_events": strongest,
        "residual_peak": {
            "support": float(_auto_safe_float(metrics.get("residual_peak_modal_support", 0.0), 0.0)),
            "priority": float(_auto_safe_float(metrics.get("residual_peak_modal_priority", 0.0), 0.0)),
            "penalty": float(_auto_safe_float(metrics.get("residual_peak_modal_penalty", 0.0), 0.0)),
            "dominant_freq_hz": residual_hz,
        },
        "tdc": {
            "event_count": int(metrics.get("tdc_modal_event_count", 0) or 0),
            "reductions": tdc_reductions[:5],
        },
        "winner_polish": {
            "modal_priority": float(_auto_safe_float(residual_polish.get("modal_priority", 0.0), 0.0)),
            "modal_support": float(_auto_safe_float(residual_polish.get("modal_support", 0.0), 0.0)),
            "applied": bool(residual_polish.get("applied", False)),
        },
    }

def _stereo_refine_materialize_base_data(
    base_data: dict | None,
    stereo_refine_meta: dict | None,
) -> dict:
    out = dict(base_data or {})
    meta = dict(stereo_refine_meta or {})
    shared_l_st = dict(meta.get("_shared_l_st", {}) or {})
    shared_r_st = dict(meta.get("_shared_r_st", {}) or {})
    if shared_l_st and shared_r_st:
        out["_stereo_shared_l_st"] = dict(shared_l_st)
        out["_stereo_shared_r_st"] = dict(shared_r_st)
    return out

def _public_stereo_policy_refine_meta(stereo_refine_meta: dict | None) -> dict:
    out = dict(stereo_refine_meta or {})
    for key in list(out.keys()):
        if str(key).startswith("_"):
            out.pop(key, None)
    return out

def _cache_refine_winner_phase_label(seed_source: str | None) -> str:
    seed_source_s = str(seed_source or "")
    if seed_source_s in (
        "cached_target_seed",
        "cache_signature_target_seed",
        "cache_measurement_target_seed",
        "cache_optuna_target_seed",
    ):
        return "cached target seed + micro refine"
    return (
        "target preselect + micro refine"
        if seed_source_s == "target_preselect"
        else "exact cache hit + micro refine"
    )

def _cache_refine_winner_summary(seed_source: str | None, *, improved_any: bool) -> str:
    seed_source_s = str(seed_source or "")
    if seed_source_s in (
        "cached_target_seed",
        "cache_signature_target_seed",
        "cache_measurement_target_seed",
        "cache_optuna_target_seed",
    ):
        return (
            "Loaded cached target seed and ran micro-refine trials."
            if bool(improved_any)
            else "Loaded cached target seed and verified it with micro-refine trials."
        )
    if seed_source_s == "target_preselect":
        return (
            "Loaded target preselect seed and ran micro-refine trials."
            if bool(improved_any)
            else "Loaded target preselect seed and verified it with micro-refine trials."
        )
    return (
        "Loaded exact cached preset and ran extra cache-refine micro-trials."
        if bool(improved_any)
        else "Loaded exact cached preset and verified it with cache-refine micro-trials."
    )

def _override_candidates(search_state) -> list[dict]:
    out: list[dict] = []
    for item in list(getattr(search_state, "phase2_pool", []) or []) + list(getattr(search_state, "scored", []) or []):
        if isinstance(item, dict):
            out.append(dict(item or {}))
    best_metrics = dict(getattr(search_state, "best_metrics", {}) or {})
    best_preset = dict(getattr(search_state, "best_preset", {}) or {})
    if best_metrics or best_preset:
        out.append({"metrics": dict(best_metrics), "preset": dict(best_preset)})
    return out


__all__ = [
    '_fmt_status_metric',
    '_build_phase2_pareto_status',
    '_build_modal_intelligence_debug',
    '_stereo_refine_materialize_base_data',
    '_public_stereo_policy_refine_meta',
    '_cache_refine_winner_phase_label',
    '_cache_refine_winner_summary',
    '_override_candidates',
    '_LOW_BASS_CUT_WINNER_POLISH_STEP_HZ',
    '_LOW_BASS_CUT_WINNER_POLISH_MAX_DELTA_HZ',
]

