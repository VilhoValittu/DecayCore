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
import numpy as np


@dataclass
class DspContext:
    n_fft: int
    freq_axis: np.ndarray
    gain_db: np.ndarray
    target_mags: np.ndarray
    st: dict = field(default_factory=dict)


@dataclass
class PreprocessResult:
    ctx: DspContext
    f_in: np.ndarray
    m_in: np.ndarray
    p_in: np.ndarray
    m_smooth_std: np.ndarray
    p_smooth: np.ndarray
    m_interp: np.ndarray
    p_rad_raw: np.ndarray
    p_rad_interp: np.ndarray
    delay_slope: float
    m_plot_db: np.ndarray | None
    complex_meas: np.ndarray
    m_anal: np.ndarray
    p_anal_rad: np.ndarray
    complex_anal: np.ndarray
    conf_mask: np.ndarray
    reflections: list
    cmp: dict | None
    analysis_mode: str
    is_psy: bool
