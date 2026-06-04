# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Post-limit magnitude response processing and constraints.

Split into:
- .pipeline: Main orchestration and soft clipping pipeline
- .clamps: Hard clamps and soft curve limiting
- .authority: Authority caps and override logic
- .low_frequency: Low-frequency guard masks and limits
- .metrics: Telemetry and diagnostic metrics

Uses dynamic imports with explicit __all__ for IDE support.
"""

import importlib as _importlib

_MODULE_NAMES = ['pipeline', 'clamps', 'authority', 'low_frequency', 'metrics']
for _module_name in _MODULE_NAMES:
    _module = _importlib.import_module(f"{__name__}.{_module_name}")
    for _symbol in dir(_module):
        if not _symbol.startswith("__"):
            globals().setdefault(_symbol, getattr(_module, _symbol))

__all__ = sorted(_symbol for _symbol in globals() if not _symbol.startswith("__"))
