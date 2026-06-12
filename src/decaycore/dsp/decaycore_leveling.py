# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Backward-compatible facade for :mod:`.leveling_parts`."""

from __future__ import annotations

import functools as _functools
import types as _types

from . import leveling_parts as _impl_package


_SYNC_EXCLUDE = frozenset(
    {"_functools", "_types", "_impl_package", "_sync_impl_globals", "_wrap_impl_function", "_SYNC_EXCLUDE", "_SYNC_NAMES", "_SYNC_NAMES_LEN"}
)
_SYNC_NAMES: list = []
_SYNC_NAMES_LEN = -1


def _sync_impl_globals(_func):
    # Kuuma polku: synkataan facade-globaalit impl-funktioon joka kutsulla,
    # jotta monkeypatch facadessa nakyy implille. Nimilista lasketaan vain
    # kun globaalien maara muuttuu; arvot haetaan silti joka kerralla.
    global _SYNC_NAMES_LEN
    _g = globals()
    if len(_g) != _SYNC_NAMES_LEN:
        _SYNC_NAMES[:] = [_name for _name in _g if not _name.startswith("__") and _name not in _SYNC_EXCLUDE]
        _SYNC_NAMES_LEN = len(_g)
    _fg = _func.__globals__
    for _name in _SYNC_NAMES:
        _fg[_name] = _g[_name]


def _wrap_impl_function(_func):
    @_functools.wraps(_func)
    def _wrapped(*args, **kwargs):
        _sync_impl_globals(_func)
        return _func(*args, **kwargs)

    return _wrapped


for _name in getattr(_impl_package, "__all__", dir(_impl_package)):
    if _name.startswith("__"):
        continue
    _value = getattr(_impl_package, _name)
    if isinstance(_value, _types.FunctionType) and _value.__module__.startswith(_impl_package.__name__ + "."):
        _value = _wrap_impl_function(_value)
    globals()[_name] = _value

__all__ = sorted(_name for _name in globals() if not _name.startswith("__"))
