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

_MULTI_RATE_SEARCH_FS_HZ = 44100
_MULTI_RATE_SEARCH_TAPS = 65536


def resolve_auto_search_reference(
    *,
    data: dict,
    target_rates: list[int],
    taps_base: int,
) -> tuple[int, int]:
    """Return the DSP grid used by automatic target and phase searches."""
    if bool(data.get("multi_rate_opt", False)):
        return _MULTI_RATE_SEARCH_FS_HZ, _MULTI_RATE_SEARCH_TAPS

    search_fs = int(target_rates[0]) if target_rates else int(data.get("fs", 44100) or 44100)
    return search_fs, int(taps_base)


__all__ = ["resolve_auto_search_reference"]
