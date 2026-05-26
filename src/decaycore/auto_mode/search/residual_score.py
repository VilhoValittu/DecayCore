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

from ..auto_mode_profile import profiled_section
from .modal_intelligence import (
    _attach_modal_support_to_residual_metrics,
    _auto_merge_modal_intelligence_metrics,
    _modal_residual_fallback_metrics,
    compute_modal_intelligence_metrics,
)
from .residual_metrics import _auto_merge_residual_peak_metrics, _auto_residual_peak_metrics_from_stats

def score_residual_peaks(l_st, r_st, *, base_data) -> dict:
    bd_for_peaks = dict(base_data or {})
    peak_lo = max(
        20.0,
        shared._auto_safe_float(
            bd_for_peaks.get("mag_c_min", bd_for_peaks.get("mag_c_min_hz", 20.0)),
            20.0,
        ),
    )
    peak_hi_raw = shared._auto_safe_float(
        bd_for_peaks.get("mag_c_max", bd_for_peaks.get("mag_c_max_hz", shared.AUTO_MODE_RESIDUAL_PEAK_MAX_HZ)),
        shared.AUTO_MODE_RESIDUAL_PEAK_MAX_HZ,
    )
    peak_hi = min(
        float(peak_hi_raw) if np.isfinite(peak_hi_raw) else float(shared.AUTO_MODE_RESIDUAL_PEAK_MAX_HZ),
        float(shared.AUTO_MODE_RESIDUAL_PEAK_MAX_HZ),
    )
    if float(peak_hi) <= float(peak_lo):
        residual_peak_metrics = {
            "worst_residual_peak_db": float("nan"),
            "worst_residual_peak_hz": float("nan"),
            "worst_residual_peak_raw_db": float("nan"),
            "worst_residual_peak_width_hz": float("nan"),
            "worst_residual_peak_width_oct": float("nan"),
            "residual_peak_severity": float("nan"),
            "residual_peak_threshold_db": float(shared.AUTO_MODE_RESIDUAL_PEAK_THRESHOLD_DB),
            "residual_peak_hard_gate_db": float(shared.AUTO_MODE_RESIDUAL_PEAK_HARD_GATE_DB),
            "top3_residual_peak_mean_db": float("nan"),
            "residual_peak_count": 0,
            "residual_peak_candidates": [],
        }
        modal_intelligence_metrics = _auto_merge_modal_intelligence_metrics({}, {})
    else:
        residual_peak_threshold_db = max(
            0.0,
            shared._auto_safe_float(
                bd_for_peaks.get("auto_mode_residual_peak_threshold_db", bd_for_peaks.get("residual_peak_threshold_db", shared.AUTO_MODE_RESIDUAL_PEAK_THRESHOLD_DB)),
                shared.AUTO_MODE_RESIDUAL_PEAK_THRESHOLD_DB,
            ),
        )
        residual_peak_hard_gate_db = max(
            0.0,
            shared._auto_safe_float(
                bd_for_peaks.get("auto_mode_residual_peak_hard_gate_db", bd_for_peaks.get("residual_peak_hard_gate_db", shared.AUTO_MODE_RESIDUAL_PEAK_HARD_GATE_DB)),
                shared.AUTO_MODE_RESIDUAL_PEAK_HARD_GATE_DB,
            ),
        )
        with profiled_section("residual_score.peak_metrics"):
            l_residual_peak_metrics = _auto_residual_peak_metrics_from_stats(
                l_st,
                lo_hz=float(peak_lo),
                hi_hz=float(peak_hi),
                threshold_db=float(residual_peak_threshold_db),
                hard_gate_db=float(residual_peak_hard_gate_db),
            )
            r_residual_peak_metrics = _auto_residual_peak_metrics_from_stats(
                r_st,
                lo_hz=float(peak_lo),
                hi_hz=float(peak_hi),
                threshold_db=float(residual_peak_threshold_db),
                hard_gate_db=float(residual_peak_hard_gate_db),
            )
        residual_peak_metrics = _auto_merge_residual_peak_metrics(
            l_residual_peak_metrics,
            r_residual_peak_metrics,
        )
        with profiled_section("residual_score.modal_intelligence"):
            modal_intelligence_metrics = _auto_merge_modal_intelligence_metrics(
                compute_modal_intelligence_metrics(l_st, lo_hz=float(peak_lo), hi_hz=float(peak_hi)),
                compute_modal_intelligence_metrics(r_st, lo_hz=float(peak_lo), hi_hz=float(peak_hi)),
            )
        residual_peak_metrics = _attach_modal_support_to_residual_metrics(
            residual_peak_metrics,
            modal_intelligence_metrics,
        )
    worst_residual_peak_db = shared._auto_safe_float(
        residual_peak_metrics.get("worst_residual_peak_db", float("nan")),
        float("nan"),
    )
    worst_residual_peak_raw_db = shared._auto_safe_float(
        residual_peak_metrics.get("worst_residual_peak_raw_db", float("nan")),
        float("nan"),
    )
    residual_peak_severity = shared._auto_safe_float(
        residual_peak_metrics.get("residual_peak_severity", worst_residual_peak_db),
        float("nan"),
    )
    residual_peak_threshold_db = shared._auto_safe_float(
        residual_peak_metrics.get("residual_peak_threshold_db", shared.AUTO_MODE_RESIDUAL_PEAK_THRESHOLD_DB),
        shared.AUTO_MODE_RESIDUAL_PEAK_THRESHOLD_DB,
    )
    residual_peak_hard_gate_db = shared._auto_safe_float(
        residual_peak_metrics.get("residual_peak_hard_gate_db", shared.AUTO_MODE_RESIDUAL_PEAK_HARD_GATE_DB),
        shared.AUTO_MODE_RESIDUAL_PEAK_HARD_GATE_DB,
    )
    residual_peak_penalty_cap = max(
        0.0,
        shared._auto_safe_float(
            bd_for_peaks.get(
                "auto_mode_residual_peak_penalty_cap",
                bd_for_peaks.get("residual_peak_penalty_cap", shared.AUTO_MODE_RESIDUAL_PEAK_PENALTY_CAP),
            ),
            shared.AUTO_MODE_RESIDUAL_PEAK_PENALTY_CAP,
        ),
    )
    top3_residual_peak_mean_db = shared._auto_safe_float(
        residual_peak_metrics.get("top3_residual_peak_mean_db", float("nan")),
        float("nan"),
    )
    residual_modal_priority = float(
        np.clip(
            shared._auto_safe_float(residual_peak_metrics.get("residual_peak_modal_priority", 0.0), 0.0),
            0.0,
            1.0,
        )
    )
    modal_fallback = _modal_residual_fallback_metrics(
        residual_peak_metrics,
        modal_intelligence_metrics,
        lo_hz=float(peak_lo),
        hi_hz=float(peak_hi),
        threshold_db=float(residual_peak_threshold_db),
    )
    modal_fallback_penalty = float(
        np.clip(
            shared._auto_safe_float(modal_fallback.get("modal_residual_fallback_penalty", 0.0), 0.0),
            0.0,
            3.0,
        )
    )
    residual_modal_penalty = 0.0
    residual_peak_penalty = 0.0
    if np.isfinite(worst_residual_peak_raw_db) or np.isfinite(residual_peak_severity) or np.isfinite(top3_residual_peak_mean_db):
        threshold_eff = max(0.0, float(residual_peak_threshold_db))
        residual_peak_penalty = 0.0
        if np.isfinite(worst_residual_peak_raw_db):
            residual_peak_penalty += 1.20 * max(0.0, float(worst_residual_peak_raw_db) - threshold_eff)
        if np.isfinite(residual_peak_severity):
            residual_peak_penalty += 0.85 * max(0.0, float(residual_peak_severity))
        residual_peak_penalty += (
            0.35 * max(0.0, float(top3_residual_peak_mean_db))
            if np.isfinite(top3_residual_peak_mean_db)
            else 0.0
        )
        if residual_modal_priority > 0.0:
            residual_modal_penalty = float(
                min(
                    1.25,
                    residual_modal_priority
                    * (
                        0.40
                        + 0.20 * max(0.0, float(worst_residual_peak_raw_db) - threshold_eff)
                        if np.isfinite(worst_residual_peak_raw_db)
                        else 0.40
                    ),
                )
            )
            residual_peak_penalty += float(residual_modal_penalty)
        residual_peak_penalty = float(
            np.clip(
                float(residual_peak_penalty) + float(modal_fallback_penalty),
                0.0,
                float(residual_peak_penalty_cap),
            )
        )
    elif modal_fallback_penalty > 0.0:
        residual_peak_penalty = float(
            np.clip(
                float(modal_fallback_penalty),
                0.0,
                float(residual_peak_penalty_cap),
            )
        )
    residual_gate_value_db = float("nan")
    residual_gate_source = ""
    if np.isfinite(worst_residual_peak_raw_db):
        residual_gate_value_db = float(max(0.0, worst_residual_peak_raw_db))
        residual_gate_source = "raw_db"
    elif np.isfinite(worst_residual_peak_db):
        residual_gate_value_db = float(max(0.0, worst_residual_peak_db))
        residual_gate_source = "severity"
    return {
        "peak_lo": peak_lo,
        "peak_hi": peak_hi,
        "residual_peak_metrics": residual_peak_metrics,
        "worst_residual_peak_db": worst_residual_peak_db,
        "worst_residual_peak_raw_db": worst_residual_peak_raw_db,
        "residual_peak_severity": residual_peak_severity,
        "residual_peak_threshold_db": residual_peak_threshold_db,
        "residual_peak_hard_gate_db": residual_peak_hard_gate_db,
        "residual_peak_penalty_cap": residual_peak_penalty_cap,
        "top3_residual_peak_mean_db": top3_residual_peak_mean_db,
        "residual_peak_modal_support": float(
            shared._auto_safe_float(residual_peak_metrics.get("residual_peak_modal_support", 0.0), 0.0)
        ),
        "residual_peak_modal_max_severity": float(
            shared._auto_safe_float(residual_peak_metrics.get("residual_peak_modal_max_severity", 0.0), 0.0)
        ),
        "residual_peak_modal_confidence": float(
            shared._auto_safe_float(residual_peak_metrics.get("residual_peak_modal_confidence", 0.0), 0.0)
        ),
        "residual_peak_modal_priority": float(residual_modal_priority),
        "residual_peak_modal_dominant_freq_hz": residual_peak_metrics.get("residual_peak_modal_dominant_freq_hz"),
        "residual_peak_modal_event_count": int(residual_peak_metrics.get("residual_peak_modal_event_count", 0) or 0),
        "residual_peak_modal_penalty": float(residual_modal_penalty),
        "modal_residual_fallback_used": bool(modal_fallback.get("modal_residual_fallback_used", False)),
        "modal_residual_fallback_kind": str(modal_fallback.get("modal_residual_fallback_kind", "") or ""),
        "modal_residual_fallback_hz": float(
            shared._auto_safe_float(modal_fallback.get("modal_residual_fallback_hz", float("nan")), float("nan"))
        ),
        "modal_residual_fallback_peak_db": float(
            shared._auto_safe_float(modal_fallback.get("modal_residual_fallback_peak_db", float("nan")), float("nan"))
        ),
        "modal_residual_fallback_penalty": float(modal_fallback_penalty),
        "residual_peak_gate_value_db": float(residual_gate_value_db),
        "residual_peak_gate_source": str(residual_gate_source),
        "residual_peak_penalty": residual_peak_penalty,
        "modal_intelligence_metrics": modal_intelligence_metrics,
    }



__all__ = ["score_residual_peaks"]
