# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

from .phase_computation import (
    _merge_minphase_and_excess,
    _phase_region_profiles,
    _phase_confidence_profile,
    _max_abs_gd_gradient_ms_per_oct,
    _gd_grad_metrics,
    _gd_grad_limiter,
    _weighted_mean,
    _weighted_share,
)
from .phase_finalization import (
    _store_phase_profile_metrics,
    _has_active_theoretical_phase_model,
    _pre_ringing_band_protection_floor,
    _apply_phase_model,
)
from .phase_windowing import (
    _PhaseComponents,
    _unwrap_phases,
    _compute_excess_phase,
    _apply_mixed_excess_mask,
    _linear_excess_weight,
    _smooth_linear_boundary,
    _enforce_linear_tail_decay,
    _linear_to_minphase_blend_mask,
)

__all__ = [
    "_PhaseComponents",
    "_apply_mixed_excess_mask",
    "_apply_phase_model",
    "_compute_excess_phase",
    "_enforce_linear_tail_decay",
    "_gd_grad_limiter",
    "_gd_grad_metrics",
    "_has_active_theoretical_phase_model",
    "_linear_excess_weight",
    "_linear_to_minphase_blend_mask",
    "_max_abs_gd_gradient_ms_per_oct",
    "_merge_minphase_and_excess",
    "_phase_confidence_profile",
    "_phase_region_profiles",
    "_pre_ringing_band_protection_floor",
    "_smooth_linear_boundary",
    "_store_phase_profile_metrics",
    "_unwrap_phases",
    "_weighted_mean",
    "_weighted_share",
]
