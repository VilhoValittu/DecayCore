# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Shared FIR length calculations without UI dependencies."""

from __future__ import annotations

import scipy.fft

_COMMON_RATE_MULTIPLIERS = {
    44100: 1,
    48000: 1,
    88200: 2,
    96000: 2,
    176400: 4,
    192000: 4,
    352800: 8,
    384000: 8,
}


def scale_taps_with_fs(
    fs: int,
    base_taps: int = 65536,
    base_fs: int = 44100,
) -> int:
    """Scale FIR taps across common rate families, retaining a fast FFT length."""
    try:
        fs_i = int(float(fs))
        base_taps_i = int(float(base_taps))
        base_fs_i = int(float(base_fs))
        if fs_i <= 0 or base_taps_i <= 0 or base_fs_i <= 0:
            return int(base_taps_i if base_taps_i > 0 else 65536)

        fs_multiplier = _COMMON_RATE_MULTIPLIERS.get(fs_i)
        base_multiplier = _COMMON_RATE_MULTIPLIERS.get(base_fs_i)
        if fs_multiplier is not None and base_multiplier is not None:
            ratio = float(fs_multiplier) / float(base_multiplier)
        else:
            ratio = float(fs_i) / float(base_fs_i)
        scaled = int(round(float(base_taps_i) * ratio))
        return int(scipy.fft.next_fast_len(max(1, scaled), real=True))
    except (
        AttributeError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        RuntimeError,
        OSError,
        ImportError,
        ModuleNotFoundError,
        NameError,
    ):
        return int(base_taps)


__all__ = ["scale_taps_with_fs"]
