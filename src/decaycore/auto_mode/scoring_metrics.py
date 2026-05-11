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

from ..common.acoustic_stats import calc_ai_summary_from_stats
from .search.metric_penalties import (
    _auto_bass_integration_penalty, _auto_channel_overfit_metrics_from_stats,
    _auto_correction_sharpness_metrics_from_stats, _auto_dip_fill_risk_metrics_from_stats,
    _auto_dsp_quality_penalty, _auto_exc_penalty_bins_from_dbg, _auto_exc_zero_penalty_freq_hz_from_stats,
    _auto_excursion_penalty, _auto_focus_ripple_from_stats, _auto_harmonic_boost_penalty,
    _auto_harmonic_local_boost_penalty, _auto_rt60_policy_penalty, _auto_tdc_decay_tradeoff,
    _auto_bass_boost_metrics_from_stats, _auto_target_tracking_metrics_from_stats,
    _auto_target_tracking_penalty,
)
from .search.residual_peaks import (
    _auto_merge_residual_peak_metrics, _auto_residual_peak_metrics_from_stats,
    compute_broad_residual_peak_metrics, compute_modal_intelligence_metrics,
)
from .search.component_scores import score_residual_peaks
from .search.score_result import _auto_score_result

__all__ = [
    "calc_ai_summary_from_stats", "_auto_dsp_quality_penalty", "_auto_exc_penalty_bins_from_dbg",
    "_auto_exc_zero_penalty_freq_hz_from_stats", "_auto_excursion_penalty", "_auto_bass_integration_penalty",
    "_auto_focus_ripple_from_stats", "compute_broad_residual_peak_metrics", "compute_modal_intelligence_metrics",
    "_auto_residual_peak_metrics_from_stats", "_auto_merge_residual_peak_metrics",
    "_auto_correction_sharpness_metrics_from_stats", "_auto_dip_fill_risk_metrics_from_stats",
    "_auto_channel_overfit_metrics_from_stats", "_auto_harmonic_boost_penalty", "_auto_rt60_policy_penalty",
    "_auto_tdc_decay_tradeoff", "_auto_harmonic_local_boost_penalty", "score_residual_peaks", "_auto_score_result",
    "_auto_bass_boost_metrics_from_stats", "_auto_target_tracking_metrics_from_stats", "_auto_target_tracking_penalty",
]
