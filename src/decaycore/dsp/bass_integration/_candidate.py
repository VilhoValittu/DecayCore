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

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DirectDacCandidate:
    main_hpf_hz: float
    main_hpf_order: int
    sub_hpf_hz: float
    sub_hpf_order: int
    sub_lpf_hz: float
    sub_lpf_order: int
    sub_delay_ms: float = 0.0
    sub_array_delay_ms: float = 0.0
    sub1_delay_ms: float = 0.0
    sub2_delay_ms: float = 0.0
    main_l_delay_ms: float = 0.0
    main_r_delay_ms: float = 0.0
    sub_gain_trim_db: float = 0.0
    sub_polarity_invert: bool = False
    sub_allpass_enabled: bool = False
    sub_allpass_freq_hz: float = 0.0
    sub_allpass_q: float = 0.707
    topology: str = "single_sub_bus"
    source: str = "unknown"


@dataclass(frozen=True)
class DirectDacChannelMetrics:
    channel: str
    score: float
    cancellation_risk: float
    null_depth_db: float
    null_width_hz: float
    overlap_ripple_db: float
    sub_dominance_db: float
    xo_gd_rms_mismatch_ms: float
    xo_gd_max_mismatch_ms: float
    predicted_sum_flatness_db: float
    predicted_sum_dip_depth_db: float
    predicted_sum_peak_excess_db: float
    feasible: bool
    reject_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class DirectDacCandidateMetrics:
    candidate: DirectDacCandidate
    score: float
    objective: float
    applied: bool
    feasible: bool
    reject_reasons: tuple[str, ...]
    left: DirectDacChannelMetrics
    right: DirectDacChannelMetrics
    dominant_channel: str
    summary: dict[str, Any] = field(default_factory=dict)
