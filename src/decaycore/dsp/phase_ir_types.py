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
from typing import Any, Callable

import numpy as np


@dataclass
class PhaseIRInputs:
    """Phase/IR-vaiheen syotepaketti siistimpaa funktiorajaa varten."""

    cfg: Any
    freq_axis: np.ndarray
    n_fft: int
    gain_db: np.ndarray
    p_rad_interp: np.ndarray
    conf_mask: np.ndarray | None
    m_anal: np.ndarray
    calc_offset_db: float
    target_mags: np.ndarray
    st: Any
    mask_c: np.ndarray
    base_sigma: Any
    filter_smooth: Any
    df_mode: Any
    raw_g: Any
    final_g: Any
    use_bassfirst: bool
    afdw_on: bool
    logger: Any
    apply_hpf_to_mags_fn: Callable[..., np.ndarray]
    apply_lpf_to_mags_fn: Callable[..., np.ndarray]
    limit_gd_gradient_ms_per_oct_fn: Callable[..., np.ndarray]
    cfg_float_allow_zero_fn: Callable[[Any, str, float], float]


@dataclass
class PhaseIROutputs:
    """Phase/IR-vaiheen typed tuotospaketti."""

    impulse: np.ndarray
    gain_db: np.ndarray
    auto_global_gain_db: float
    gain_margin_db: float
    auto_headroom_db: float
    current_peak_gain: float
    final_gain_total: np.ndarray
    residual_telemetry: "ResidualTelemetry | None" = None


@dataclass
class ResidualTelemetry:
    """Typed residual-pass telemetria stats-sovitusta varten."""

    residual_pass_enabled: bool
    residual_strength: float = 0.0
    residual_smoothing_mult: float = 0.0
    residual_conf_power: float = 0.0
    residual_max_delta_db: float = 0.0
    residual_band_min_hz: float = 0.0
    residual_band_max_hz: float = 0.0
    residual_band_edge_hz: float = 0.0
    residual_band_bins: int = 0
    residual_lf_guard_bins: int = 0
    residual_lf_boost_blocked_bins: int = 0
    residual_right_transition_fade_skipped: bool = False
    residual_pass_mode: str = "modal_polish"
    residual_null_guard_enabled: bool = True
    residual_authority_boost_cap_mean_db: float = 0.0
    residual_authority_boost_cap_min_db: float = 0.0
    residual_authority_boost_cap_max_db: float = 0.0
    residual_authority_cut_cap_mean_db: float = 0.0
    residual_authority_cut_cap_min_db: float = 0.0
    residual_authority_cut_cap_max_db: float = 0.0
    residual_blocked_null_risk_bins: int = 0
    residual_blocked_reflection_risk_bins: int = 0
    residual_blocked_low_authority_bins: int = 0
    residual_blocked_low_modal_support_bins: int = 0
    residual_modal_polish_bins: int = 0
    residual_requested_boost_max_db: float = 0.0
    residual_applied_boost_max_db: float = 0.0
    residual_requested_cut_max_db: float = 0.0
    residual_applied_cut_max_db: float = 0.0
