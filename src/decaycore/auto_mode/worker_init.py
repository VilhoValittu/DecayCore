# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Worker-process initializer for ProcessPoolExecutor trial workers."""

from __future__ import annotations


def _auto_worker_init() -> None:
    """Warm up Numba JIT functions in a fresh worker process.

    Called once per worker process at startup. Eliminates JIT cold-start
    latency from the first trial executed in each worker.
    """
    import numpy as np
    try:
        from numba.core.errors import NumbaError as _NumbaError
    except (
        AttributeError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        RuntimeError,
        ArithmeticError,
        ZeroDivisionError,
        OSError,
        ImportError,
        ModuleNotFoundError,
        NameError,
    ):
        _NumbaError = RuntimeError

    try:
        from ..dsp.smoothing import _smooth_mag_core

        _f = np.linspace(20.0, 20000.0, 64, dtype=np.float64)
        _m = np.zeros(64, dtype=np.float64)
        _w = np.hanning(8)
        _smooth_mag_core(_f, _m, _f, _w, 4)
    except (

        AttributeError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        RuntimeError,
        ArithmeticError,
        ZeroDivisionError,
        OSError,
        ImportError,
        ModuleNotFoundError,
        NameError,
        _NumbaError,
    ):
        pass

    try:
        from ..dsp.limits import _slope_passes, _slope_passes_asym

        _m2 = np.zeros(64, dtype=np.float64)
        _slope_passes(_m2, 6.0, 6.0)
        _slope_passes_asym(_m2, 6.0, 12.0)
    except (

        AttributeError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        RuntimeError,
        ArithmeticError,
        ZeroDivisionError,
        OSError,
        ImportError,
        ModuleNotFoundError,
        NameError,
        _NumbaError,
    ):
        pass

    try:
        from ..dsp.dsp_ops import _gradient1d, _gd_smooth_loop

        _f2 = np.linspace(20.0, 20000.0, 64, dtype=np.float64)
        _g = np.zeros(64, dtype=np.float64)
        _gradient1d(_f2, _g)
    except (

        AttributeError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        RuntimeError,
        ArithmeticError,
        ZeroDivisionError,
        OSError,
        ImportError,
        ModuleNotFoundError,
        NameError,
        _NumbaError,
    ):
        pass
