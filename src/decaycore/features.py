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

import importlib.util


def has_measurement_module() -> bool:
    """Return True when the private measurement feature package is installed."""
    try:
        return importlib.util.find_spec("decaycore.measurement") is not None
    except (ImportError, AttributeError, ValueError):
        return False


__all__ = ["has_measurement_module"]
