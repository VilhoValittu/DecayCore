# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Winner-polish helpers for auto-mode finalize orchestration."""

from .polish_excess_phase import apply_excess_phase_strength_winner_polish
from .polish_hpf import apply_hpf_winner_polish
from .polish_low_bass import apply_low_bass_cut_winner_polish
from .polish_mag_c_min import apply_mag_c_min_winner_polish
from .polish_phase_limits import apply_phase_limit_winner_polish
from .polish_residual_peak import apply_residual_peak_winner_polish
from .polish_tdc_strength import apply_tdc_strength_winner_polish

__all__ = [
    "apply_excess_phase_strength_winner_polish",
    "apply_hpf_winner_polish",
    "apply_low_bass_cut_winner_polish",
    "apply_mag_c_min_winner_polish",
    "apply_phase_limit_winner_polish",
    "apply_residual_peak_winner_polish",
    "apply_tdc_strength_winner_polish",
]
