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

AVR_CROSSOVER_CANDIDATES: tuple[float, ...] = (40.0, 60.0, 70.0, 80.0, 90.0, 110.0, 120.0, 150.0, 180.0)
DIRECT_DAC_CROSSOVER_STEP_HZ = 0.5
DIRECT_DAC_ALLPASS_FREQ_MULTIPLIERS: tuple[float, ...] = (0.55, 0.70, 0.85, 1.00, 1.15, 1.35, 1.60)
DIRECT_DAC_ALLPASS_Q_CANDIDATES: tuple[float, ...] = (0.45, 0.60, 0.80, 1.00, 1.30, 1.70, 2.20)
DIRECT_DAC_ALLPASS_REFINE_FREQ_FACTORS: tuple[float, ...] = (0.88, 0.94, 0.98, 1.00, 1.02, 1.06, 1.12)
DIRECT_DAC_ALLPASS_REFINE_Q_FACTORS: tuple[float, ...] = (0.78, 0.90, 0.97, 1.00, 1.03, 1.10, 1.22)
DIRECT_DAC_ALLPASS_MIN_IMPROVEMENT_SCORE = 0.08
DIRECT_DAC_ALLPASS_MIN_CANCEL_IMPROVEMENT = 0.010
DIRECT_DAC_ALLPASS_MIN_RIPPLE_IMPROVEMENT_DB = 0.12
DIRECT_DAC_ALLPASS_MIN_GD_IMPROVEMENT_MS = 0.04
MIN_DIRECT_DAC_OVERLAP_RATIO: float = 1.1
DIRECT_DAC_OVERLAP_RATIOS: tuple[float, ...] = (1.1, 1.25, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0)
BASS_INTEGRATION_FEASIBILITY_THRESHOLDS: dict[str, dict[str, float]] = {
    "good": {
        "overlap_ripple_db": 8.0,
        "sub_dominance_db": 8.0,
        "xo_gd_rms_mismatch_ms": 12.0,
        "overlap_ripple_delta_db": 4.0,
        "sub_dominance_delta_db": 4.0,
        "xo_gd_mismatch_delta_ms": 10.0,
    },
    "marginal": {
        "overlap_ripple_db": 12.0,
        "sub_dominance_db": 12.0,
        "xo_gd_rms_mismatch_ms": 20.0,
        "overlap_ripple_delta_db": 8.0,
        "sub_dominance_delta_db": 8.0,
        "xo_gd_mismatch_delta_ms": 18.0,
    },
}
BASS_INTEGRATION_FEASIBILITY_OBJECTIVE_PENALTY = {
    "good": 0.0,
    "marginal": 0.9,
    "infeasible": 2.8,
}
COMBINED_SUB_ALIGNMENT_MAX_LAG_MS = 50.0
DIRECT_DAC_ALIGNMENT_DELAY_CANDIDATES_MS: tuple[float, ...] = (
    -50.0,
    -40.0,
    -30.0,
    -20.0,
    -15.0,
    -12.0,
    -10.0,
    -8.0,
    -6.0,
    -4.0,
    -3.0,
    -2.0,
    -1.0,
    0.0,
    1.0,
    2.0,
    3.0,
    4.0,
    6.0,
    8.0,
    10.0,
    12.0,
    15.0,
    20.0,
    30.0,
    40.0,
    50.0,
)
DIRECT_DAC_ALIGNMENT_GAIN_CANDIDATES_DB: tuple[float, ...] = (
    -15.0,
    -12.0,
    -9.0,
    -6.0,
    -4.0,
    -3.0,
    -2.0,
    -1.0,
    0.0,
    1.0,
    2.0,
    3.0,
    4.0,
    6.0,
    9.0,
    12.0,
    15.0,
)
DIRECT_DAC_ALIGNMENT_MIN_IMPROVEMENT_SCORE = 0.05
AVR_LFE_MAIN_ALIGNMENT_COARSE_DELAYS_MS: tuple[float, ...] = (-20.0, -10.0, -4.0, 0.0, 4.0, 10.0, 20.0)
AVR_LFE_MAIN_ALIGNMENT_COARSE_GAINS_DB: tuple[float, ...] = (-6.0, -3.0, 0.0, 3.0)
AVR_LFE_MAIN_ALIGNMENT_REFINE_DELTA_MS: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)
AVR_LFE_MAIN_ALIGNMENT_REFINE_DELTA_DB: tuple[float, ...] = (-1.0, 0.0, 1.0)
GD_CONTINUITY_GUARD_LO_RATIO = 0.70
GD_CONTINUITY_GUARD_HI_RATIO = 1.30
