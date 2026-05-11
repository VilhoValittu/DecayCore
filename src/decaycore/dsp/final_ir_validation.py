# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Backward-compatible facade for :mod:`.final_ir_validation_parts`."""

from __future__ import annotations

import functools as _functools
import types as _types

from . import final_ir_validation_parts as _impl_package


def _sync_impl_globals(_func):
    _func.__globals__.update(
        {
            _name: _value
            for _name, _value in globals().items()
            if not _name.startswith("__") and _name not in {"_functools", "_types", "_impl_package", "_sync_impl_globals", "_wrap_impl_function"}
        }
    )


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
