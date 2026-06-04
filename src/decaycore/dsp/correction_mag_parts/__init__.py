# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Magnitude correction pipeline submodules.

Split into:
- .correction_mag_01: Bass smoothing and mid refit logic
- .correction_mag_02: Core pipeline orchestration

Uses dynamic imports with explicit __all__ for IDE support.
"""

import importlib as _importlib

_MODULE_NAMES = ['bass_smoothing', 'mag_pipeline']
for _module_name in _MODULE_NAMES:
    _module = _importlib.import_module(f"{__name__}.{_module_name}")
    for _symbol in dir(_module):
        if not _symbol.startswith("__"):
            globals().setdefault(_symbol, getattr(_module, _symbol))

__all__ = sorted(_symbol for _symbol in globals() if not _symbol.startswith("__"))
