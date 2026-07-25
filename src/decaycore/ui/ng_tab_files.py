# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Backward-compatible facade for :mod:`.ng_tab_files_parts`."""

from __future__ import annotations

from . import ng_tab_files_parts as _impl_package


for _name in getattr(_impl_package, "__all__", dir(_impl_package)):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_impl_package, _name)

__all__ = sorted(_name for _name in globals() if not _name.startswith("__"))
