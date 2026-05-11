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

from scipy.fft import next_fast_len as _scipy_next_fast_len

from .dsp_config import CfgReader, coerce_range2


def choose_fft_len(n: int) -> int:
    """Return the next FFT-efficient length >= n (scipy next_fast_len)."""
    return int(_scipy_next_fast_len(max(1, int(n))))


def cfg_float_allow_zero(cfg, key: str, default: float) -> float:
    """Funktio: cfg float allow zero."""
    return CfgReader(cfg).float_allow_zero(key, default)


def safe_range(x, default_min=200.0, default_max=3000.0):
    return coerce_range2(x, default_min, default_max)
