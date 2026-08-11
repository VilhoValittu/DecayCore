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

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..phase_ir_phase_models import (
    apply_mixed_excess_mask as _apply_mixed_excess_mask_impl,
)
from ..phase_ir_phase_models import (
    enforce_linear_tail_decay as _enforce_linear_tail_decay_impl,
)
from ..phase_ir_phase_models import (
    linear_excess_weight as _linear_excess_weight_impl,
)
from ..phase_ir_phase_models import (
    linear_to_minphase_blend_mask as _linear_to_minphase_blend_mask_impl,
)
from ..phase_ir_phase_models import (
    smooth_linear_boundary as _smooth_linear_boundary_impl,
)

@dataclass
class _PhaseComponents:
    raw_u: np.ndarray
    ref_u: np.ndarray
    excess_u: np.ndarray
    min_phase: np.ndarray
    theo_xo: np.ndarray
    conf_mask: np.ndarray | None
    total_mag: np.ndarray
    n_fft: int
    is_mixed: bool
    mixed_split_hz: float
    mixed_transition_hz: float
    use_bassfirst: bool
    afdw_on: bool
    logger: Any
    limit_gd_gradient_ms_per_oct_fn: Any
    static_profiles: dict[str, Any] | None = None
    low_phase: np.ndarray | None = None
    extra_phase: np.ndarray | None = None
    phase_mask: np.ndarray | None = None

def _unwrap_phases(raw_phase, min_phase) -> tuple[np.ndarray, np.ndarray]:
    raw_u = np.unwrap(np.asarray(raw_phase, dtype=float))
    min_u = np.unwrap(np.asarray(min_phase, dtype=float))
    return raw_u, min_u

def _compute_excess_phase(raw_phase, ref_phase) -> np.ndarray:
    raw = np.asarray(raw_phase, dtype=float)
    ref = np.asarray(ref_phase, dtype=float)
    # Keep excess phase branch-stable: if raw/ref unwrap to different 2*pi branches,
    # direct subtraction can inject large artificial zig-zag into mixed-phase correction.
    # Unwrap the principal delta afterwards so smooth multi-rotation excess is still
    # available to the conservative phase guards.
    return np.unwrap(np.angle(np.exp(1j * (raw - ref))))

def _apply_mixed_excess_mask(freq_axis, excess, cfg, st) -> np.ndarray:
    return _apply_mixed_excess_mask_impl(freq_axis, excess, cfg, st)

def _linear_excess_weight(freq_axis: np.ndarray, phase_lim_hz: float) -> np.ndarray:
    return _linear_excess_weight_impl(freq_axis, phase_lim_hz)

def _smooth_linear_boundary(
    freq_axis: np.ndarray, extra_phase: np.ndarray, phase_lim_hz: float, cfg, st
) -> np.ndarray:
    return _smooth_linear_boundary_impl(freq_axis, extra_phase, phase_lim_hz, cfg, st)

def _enforce_linear_tail_decay(
    freq_axis: np.ndarray, extra_phase: np.ndarray, phase_lim_hz: float, cfg, st
) -> np.ndarray:
    return _enforce_linear_tail_decay_impl(
        freq_axis, extra_phase, phase_lim_hz, cfg, st
    )

def _linear_to_minphase_blend_mask(
    freq_axis: np.ndarray, phase_lim_hz: float, cfg, st
) -> np.ndarray:
    return _linear_to_minphase_blend_mask_impl(freq_axis, phase_lim_hz, cfg, st)


__all__ = [
    '_PhaseComponents',
    '_unwrap_phases',
    '_compute_excess_phase',
    '_apply_mixed_excess_mask',
    '_linear_excess_weight',
    '_smooth_linear_boundary',
    '_enforce_linear_tail_decay',
    '_linear_to_minphase_blend_mask',
]
