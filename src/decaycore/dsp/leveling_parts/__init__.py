# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Equalization leveling (loudness matching by octave band).

Split into:
- .api: Main public entry points
- .window_scoring: Candidate window evaluation with metrics
- .tilt_helpers: Tilt and target curve utilities
- .state_cache: Configuration and state caching

Uses dynamic imports with explicit __all__ for IDE support.
"""

import importlib as _importlib

_MODULE_NAMES = ['tilt_helpers', 'window_scoring', 'api', 'state_cache']
for _module_name in _MODULE_NAMES:
    _module = _importlib.import_module(f"{__name__}.{_module_name}")
    for _symbol in dir(_module):
        if not _symbol.startswith("__"):
            globals().setdefault(_symbol, getattr(_module, _symbol))

__all__ = sorted(_symbol for _symbol in globals() if not _symbol.startswith("__"))
