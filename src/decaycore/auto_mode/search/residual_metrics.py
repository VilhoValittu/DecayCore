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

import inspect
import logging
import math
import sys

import numpy as np

_logger = logging.getLogger(__name__)

from ...common.acoustic_stats import calc_acoustic_score, calc_ai_summary_from_stats
from ...config.models import StereoAutoPolicyConfig, StereoResolvedAutoPolicies
from ...dsp.quality_metrics import (
    band_lr_mismatch_change_from_stats,
    band_lr_mismatch_rms_from_stats,
    normalized_policy_divergence_score,
    worst_channel_relief_db,
)
from ...dsp.modal_analysis import ModalAnalysisResult, RoomModeEvent, detect_room_modes
from ...dsp.smoothing import smooth_gain_fractional_octave
from ...dsp.target_match import target_match_from_stats
from .. import shared
from ..rank_score import (
    OFFICIAL_RANK_SCORE_CONTEXT,
    attach_official_rank_score,
    calibrated_auto_quality,
    compute_rank_score_components,
)
from ..runtime_context import (
    _auto_collect_reflections,
    _auto_event_penalty_weighted,
    _auto_event_severity,
    _auto_get_top_modes_hz,
    _auto_get_worst_mode_hz,
    _auto_mode_band,
    _auto_pick_metric,
)

from .residual_analysis import compute_broad_residual_peak_metrics

BROAD_RESIDUAL_PEAK_SCORING_VERSION = 1

def _auto_residual_peak_metrics_from_stats(
    st: dict | None,
    *,
    lo_hz: float,
    hi_hz: float,
    baseline_smooth_oct: float = 2.0,
    detect_smooth_oct: float = 6.0,
    min_prom_db: float = 0.75,
    threshold_db: float | None = None,
    hard_gate_db: float | None = None,
    conf_floor: float = 0.25,
    top_n: int = 3,
) -> dict:
    return compute_broad_residual_peak_metrics(
        st,
        lo_hz=lo_hz,
        hi_hz=hi_hz,
        baseline_smooth_oct=baseline_smooth_oct,
        detect_smooth_oct=detect_smooth_oct,
        min_prom_db=min_prom_db,
        threshold_db=threshold_db,
        hard_gate_db=hard_gate_db,
        conf_floor=conf_floor,
        top_n=top_n,
    )


def _auto_merge_residual_peak_metrics(left_metrics, right_metrics) -> dict:
    peaks: list[dict] = []
    count = 0
    for channel, src in (("L", left_metrics), ("R", right_metrics)):
        m = dict(src or {}) if isinstance(src, dict) else {}
        count += int(max(0, shared._auto_safe_float(m.get("residual_peak_count", 0), 0.0)))
        for peak in list(m.get("residual_peak_candidates", []) or []):
            if not isinstance(peak, dict):
                continue
            p = dict(peak)
            p["channel"] = str(channel)
            peaks.append(p)

    peaks = sorted(
        peaks,
        key=lambda p: (
            -float(shared._auto_safe_float(p.get("residual_db"), 0.0)),
            -float(shared._auto_safe_float(p.get("severity", p.get("weighted_peak_db")), 0.0)),
            -float(shared._auto_safe_float(p.get("prominence_db"), 0.0)),
        ),
    )
    if peaks:
        top_vals = [
            float(shared._auto_safe_float(p.get("severity", p.get("weighted_peak_db")), 0.0))
            for p in peaks[:3]
        ]
        worst = dict(peaks[0])
        total_area = float(
            np.sum(
                np.asarray(
                    [max(0.0, shared._auto_safe_float(p.get("area_db_oct", 0.0), 0.0)) for p in peaks],
                    dtype=float,
                )
            )
        )
        return {
            "worst_residual_peak_db": float(shared._auto_safe_float(worst.get("severity", worst.get("weighted_peak_db")), 0.0)),
            "worst_residual_peak_hz": float(shared._auto_safe_float(worst.get("freq_hz"), float("nan"))),
            "worst_residual_peak_raw_db": float(shared._auto_safe_float(worst.get("residual_db"), float("nan"))),
            "worst_residual_peak_width_hz": float(shared._auto_safe_float(worst.get("width_hz"), float("nan"))),
            "worst_residual_peak_width_oct": float(shared._auto_safe_float(worst.get("width_oct"), float("nan"))),
            "residual_peak_area_db_oct": float(total_area),
            "residual_peak_severity": float(shared._auto_safe_float(worst.get("severity", worst.get("weighted_peak_db")), 0.0)),
            "residual_peak_threshold_db": float(shared._auto_safe_float(worst.get("threshold_db"), shared.AUTO_MODE_RESIDUAL_PEAK_THRESHOLD_DB)),
            "residual_peak_hard_gate_db": float(shared._auto_safe_float(worst.get("hard_gate_db"), shared.AUTO_MODE_RESIDUAL_PEAK_HARD_GATE_DB)),
            "top3_residual_peak_mean_db": float(np.mean(np.asarray(top_vals, dtype=float))) if top_vals else 0.0,
            "residual_peak_count": int(count),
            "residual_peak_candidates": [dict(p) for p in peaks[:6]],
            "broad_residual_peak_scoring_version": int(BROAD_RESIDUAL_PEAK_SCORING_VERSION),
        }

    vals = []
    top_vals = []
    for src in (left_metrics, right_metrics):
        m = dict(src or {}) if isinstance(src, dict) else {}
        w = shared._auto_safe_float(m.get("worst_residual_peak_db"), float("nan"))
        t = shared._auto_safe_float(m.get("top3_residual_peak_mean_db"), float("nan"))
        if np.isfinite(w):
            vals.append(float(w))
        if np.isfinite(t):
            top_vals.append(float(t))
    return {
        "worst_residual_peak_db": float(max(vals)) if vals else float("nan"),
        "worst_residual_peak_hz": float("nan"),
        "worst_residual_peak_raw_db": float("nan"),
        "worst_residual_peak_width_hz": float("nan"),
        "worst_residual_peak_width_oct": float("nan"),
        "residual_peak_area_db_oct": 0.0,
        "residual_peak_severity": float(max(vals)) if vals else float("nan"),
        "residual_peak_threshold_db": float(shared.AUTO_MODE_RESIDUAL_PEAK_THRESHOLD_DB),
        "residual_peak_hard_gate_db": float(shared.AUTO_MODE_RESIDUAL_PEAK_HARD_GATE_DB),
        "top3_residual_peak_mean_db": float(np.mean(np.asarray(top_vals, dtype=float))) if top_vals else float("nan"),
        "residual_peak_count": int(count),
        "residual_peak_candidates": [],
        "broad_residual_peak_scoring_version": int(BROAD_RESIDUAL_PEAK_SCORING_VERSION),
    }



__all__ = ["_auto_residual_peak_metrics_from_stats", "_auto_merge_residual_peak_metrics"]
