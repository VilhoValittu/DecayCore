# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Backward-compatible facade for stable UI/config values and normalization."""

from __future__ import annotations

from .config import value_normalization as _value_normalization

for _name in _value_normalization.__all__:
    globals()[_name] = getattr(_value_normalization, _name)

__all__ = list(_value_normalization.__all__)
