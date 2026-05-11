# DecayCore
# Copyright (c) 2026 Vilho Valittu
# All rights reserved.
#
# Legacy import shim for the former ``camillafir`` package name.

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_decaycore = importlib.import_module("decaycore")
_shim_path = Path(__file__).resolve().parent

for _name, _value in vars(_decaycore).items():
    if _name not in {"__name__", "__package__", "__path__", "__spec__"}:
        globals()[_name] = _value

__path__ = [str(_shim_path), *list(getattr(_decaycore, "__path__", []))]
__all__ = list(getattr(_decaycore, "__all__", []))
sys.modules.setdefault("camillafir", sys.modules[__name__])
