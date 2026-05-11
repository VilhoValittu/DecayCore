# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Compatibility shim — policy logic lives in config.mode_policy."""
from ..config.mode_policy import (  # noqa: F401
    MODE_CLAMPS,
    MODE_DEFAULTS,
    _apply_clamps,
    _apply_defaults,
    _clamp_float,
    apply_mode_to_cfg,
)
