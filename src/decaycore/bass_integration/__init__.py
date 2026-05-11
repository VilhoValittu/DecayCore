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

from ._constants import (
    AVR_CROSSOVER_CANDIDATES,
    AVR_LFE_MAIN_ALIGNMENT_COARSE_DELAYS_MS,
    AVR_LFE_MAIN_ALIGNMENT_COARSE_GAINS_DB,
    AVR_LFE_MAIN_ALIGNMENT_REFINE_DELTA_DB,
    AVR_LFE_MAIN_ALIGNMENT_REFINE_DELTA_MS,
    BASS_INTEGRATION_FEASIBILITY_OBJECTIVE_PENALTY,
    BASS_INTEGRATION_FEASIBILITY_THRESHOLDS,
    COMBINED_SUB_ALIGNMENT_MAX_LAG_MS,
    DIRECT_DAC_ALIGNMENT_DELAY_CANDIDATES_MS,
    DIRECT_DAC_ALIGNMENT_GAIN_CANDIDATES_DB,
    DIRECT_DAC_ALIGNMENT_MIN_IMPROVEMENT_SCORE,
    DIRECT_DAC_ALLPASS_FREQ_MULTIPLIERS,
    DIRECT_DAC_ALLPASS_MIN_CANCEL_IMPROVEMENT,
    DIRECT_DAC_ALLPASS_MIN_GD_IMPROVEMENT_MS,
    DIRECT_DAC_ALLPASS_MIN_IMPROVEMENT_SCORE,
    DIRECT_DAC_ALLPASS_MIN_RIPPLE_IMPROVEMENT_DB,
    DIRECT_DAC_ALLPASS_Q_CANDIDATES,
    DIRECT_DAC_ALLPASS_REFINE_FREQ_FACTORS,
    DIRECT_DAC_ALLPASS_REFINE_Q_FACTORS,
    DIRECT_DAC_CROSSOVER_STEP_HZ,
    DIRECT_DAC_OVERLAP_RATIOS,
    GD_CONTINUITY_GUARD_HI_RATIO,
    GD_CONTINUITY_GUARD_LO_RATIO,
    MIN_DIRECT_DAC_OVERLAP_RATIO,
)
from ._channel_metrics import (
    compute_cancellation_metrics,
    compute_overlap_metrics,
    compute_sub_dominance_metrics,
)
from ._diagnostics import (
    compute_bass_integration_diagnostics,
    compute_bass_integration_metric_payload,
)
from ._final_metrics import compute_final_bass_integration_metrics
from ._gd_feasibility import (
    compute_direct_dac_bass_integration_analysis,
    compute_direct_dac_bass_integration_diagnostics,
    compute_xo_gd_continuity,
    _main_guard_band_drop_db,
)
from ._recommend_alignment import recommend_direct_dac_alignment
from ._recommend_allpass import recommend_direct_dac_allpass
from ._recommend_crossover import (
    recommend_avr_crossover,
    recommend_direct_dac_crossover,
)
from ._recommend_prepare_avr import recommend_avr_lfe_main_prepare
from ._recommend_prepare_optuna import recommend_direct_dac_prepare_optuna
from ._sub_combine import (
    avg_complex_responses,
    build_bundle_combined_sub_transfer,
    build_combined_sub_transfer,
    sum_complex_responses,
    sum_complex_responses_aligned,
    _xcorr_lag_from_spectra,
)
from ._utils import normalize_sub_combine_mode
from ..bass_cache import _BUTTER_RESPONSE_CACHE, clear_bass_integration_caches

# Private helpers re-exported for callers that import them from this package
from ._filters import (
    _apply_allpass_to_transfer,
    _apply_branch_filters,
    _apply_delay_to_transfer,
    _apply_gain_trim_to_transfer,
    _apply_polarity_to_transfer,
    _get_filtered_branches,
)
from ._bundles import _build_direct_dac_trial_bundle, _build_avr_lfe_main_trial_bundle

__all__ = [
    "AVR_CROSSOVER_CANDIDATES",
    "DIRECT_DAC_CROSSOVER_STEP_HZ",
    "DIRECT_DAC_ALLPASS_FREQ_MULTIPLIERS",
    "DIRECT_DAC_ALLPASS_Q_CANDIDATES",
    "DIRECT_DAC_ALLPASS_REFINE_FREQ_FACTORS",
    "DIRECT_DAC_ALLPASS_REFINE_Q_FACTORS",
    "DIRECT_DAC_ALLPASS_MIN_IMPROVEMENT_SCORE",
    "DIRECT_DAC_ALLPASS_MIN_CANCEL_IMPROVEMENT",
    "DIRECT_DAC_ALLPASS_MIN_RIPPLE_IMPROVEMENT_DB",
    "DIRECT_DAC_ALLPASS_MIN_GD_IMPROVEMENT_MS",
    "MIN_DIRECT_DAC_OVERLAP_RATIO",
    "DIRECT_DAC_OVERLAP_RATIOS",
    "BASS_INTEGRATION_FEASIBILITY_THRESHOLDS",
    "BASS_INTEGRATION_FEASIBILITY_OBJECTIVE_PENALTY",
    "COMBINED_SUB_ALIGNMENT_MAX_LAG_MS",
    "DIRECT_DAC_ALIGNMENT_DELAY_CANDIDATES_MS",
    "DIRECT_DAC_ALIGNMENT_GAIN_CANDIDATES_DB",
    "DIRECT_DAC_ALIGNMENT_MIN_IMPROVEMENT_SCORE",
    "AVR_LFE_MAIN_ALIGNMENT_COARSE_DELAYS_MS",
    "AVR_LFE_MAIN_ALIGNMENT_COARSE_GAINS_DB",
    "AVR_LFE_MAIN_ALIGNMENT_REFINE_DELTA_MS",
    "AVR_LFE_MAIN_ALIGNMENT_REFINE_DELTA_DB",
    "GD_CONTINUITY_GUARD_LO_RATIO",
    "GD_CONTINUITY_GUARD_HI_RATIO",
    "normalize_sub_combine_mode",
    "build_combined_sub_transfer",
    "build_bundle_combined_sub_transfer",
    "sum_complex_responses",
    "avg_complex_responses",
    "sum_complex_responses_aligned",
    "compute_overlap_metrics",
    "compute_cancellation_metrics",
    "compute_sub_dominance_metrics",
    "compute_xo_gd_continuity",
    "compute_direct_dac_bass_integration_analysis",
    "compute_direct_dac_bass_integration_diagnostics",
    "compute_bass_integration_diagnostics",
    "compute_bass_integration_metric_payload",
    "compute_final_bass_integration_metrics",
    "recommend_direct_dac_alignment",
    "recommend_avr_crossover",
    "recommend_direct_dac_crossover",
    "recommend_direct_dac_allpass",
    "recommend_avr_lfe_main_prepare",
    "recommend_direct_dac_prepare_optuna",
]
