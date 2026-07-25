# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Cross-iteration residual+phase polish for auto-mode finalize.

Runs a second round of residual_peak + phase_limit polish only when the
first residual_peak polish improved the worst peak by more than
_CROSS_POLISH_MIN_IMPROVEMENT_DB.  The second round is cheap: it reuses
the same variant generators and early-exits immediately if there is no
further improvement available.
"""

from __future__ import annotations

import logging

import numpy as np

from .auto_mode_profile import profiled_section
from .shared_parts import _auto_safe_float
from .winner_polish import apply_phase_limit_winner_polish, apply_residual_peak_winner_polish

logger = logging.getLogger("DecayCore")

_CROSS_POLISH_MIN_IMPROVEMENT_DB = 0.3

__all__ = ["_apply_cross_residual_phase_polish"]


def _apply_cross_residual_phase_polish(
    *,
    runtime,
    search_state,
    search_base_data: dict,
    cfg,
    goal: str,
    filter_key: str,
    status_cb,
    winner_target_name: str | None,
    _cache_ready_preset,
    _materialize_preset_result,
    _set_search_winner_if_improved,
    _auto_is_better_refine,
    first_residual_meta: dict,
) -> dict:
    """Second round of residual+phase polish if first round improved enough.

    Returns a meta dict with round2 results (empty if skipped).
    """
    cross_meta: dict = {
        "triggered": False,
        "trigger_improvement_db": float("nan"),
        "residual_peak_round2": {},
        "phase_limit_round2": {},
    }

    peak_before = _auto_safe_float(
        first_residual_meta.get("worst_peak_before_db", float("nan")), float("nan")
    )
    peak_after = _auto_safe_float(
        first_residual_meta.get("worst_peak_after_db", float("nan")), float("nan")
    )
    if not (np.isfinite(peak_before) and np.isfinite(peak_after)):
        return cross_meta

    improvement_db = float(peak_before) - float(peak_after)
    cross_meta["trigger_improvement_db"] = float(improvement_db)
    if improvement_db < float(_CROSS_POLISH_MIN_IMPROVEMENT_DB):
        return cross_meta

    cross_meta["triggered"] = True
    logger.info(
        "Automatic mode: cross-polish triggered (residual improved %.2f dB > %.2f dB threshold).",
        float(improvement_db),
        float(_CROSS_POLISH_MIN_IMPROVEMENT_DB),
    )

    # Round 2 — phase limit (TDC change from residual round may have shifted phase risk)
    with profiled_section("finalize.phase_limit_winner_polish_round2"):
        pl2_preset, pl2_metrics, pl2_improved, pl2_meta = apply_phase_limit_winner_polish(
            best_preset=search_state.best_preset,
            best_metrics=search_state.best_metrics,
            base_data_ref=search_base_data,
            phase_label="phase_limit winner polish round2",
            goal=goal,
            filter_key=filter_key,
            enabled=bool(getattr(runtime, "phase_limit_winner_polish_enabled", True)),
            offsets_hz=tuple(getattr(runtime, "phase_limit_winner_polish_offsets_hz", ())),
            status_cb=status_cb,
            materialize_preset_result=_materialize_preset_result,
            cache_ready_preset=_cache_ready_preset,
            auto_is_better_refine=_auto_is_better_refine,
        )
    _set_search_winner_if_improved(
        search_state=search_state,
        improved=bool(pl2_improved),
        metrics=pl2_metrics,
        preset=pl2_preset,
        phase_label="phase_limit winner polish round2",
        target_name=winner_target_name,
    )
    cross_meta["phase_limit_round2"] = dict(pl2_meta or {})

    # Round 2 — residual peak (re-evaluate after phase changes)
    with profiled_section("finalize.residual_peak_winner_polish_round2"):
        rp2_preset, rp2_metrics, rp2_improved, rp2_meta = apply_residual_peak_winner_polish(
            best_preset=search_state.best_preset,
            best_metrics=search_state.best_metrics,
            base_data_ref=search_base_data,
            phase_label="residual_peak winner polish round2",
            goal=goal,
            enabled=bool(
                getattr(cfg, "residual_peak_winner_polish_enabled",
                        getattr(runtime, "residual_peak_winner_polish_enabled", True))
            ),
            max_variants=int(
                getattr(cfg, "residual_peak_winner_polish_max_variants",
                        getattr(runtime, "residual_peak_winner_polish_max_variants", 8))
            ),
            min_improvement_db=float(
                getattr(cfg, "residual_peak_winner_polish_min_improvement_db",
                        getattr(runtime, "residual_peak_winner_polish_min_improvement_db", 0.75))
            ),
            status_cb=status_cb,
            materialize_preset_result=_materialize_preset_result,
            cache_ready_preset=_cache_ready_preset,
            auto_is_better_refine=_auto_is_better_refine,
        )
    _set_search_winner_if_improved(
        search_state=search_state,
        improved=bool(rp2_improved),
        metrics=rp2_metrics,
        preset=rp2_preset,
        phase_label="residual_peak winner polish round2",
        target_name=winner_target_name,
    )
    cross_meta["residual_peak_round2"] = dict(rp2_meta or {})

    return cross_meta
