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
    """Pre-import the DSP hot-path modules in a fresh worker process.

    Called once per worker process at startup. The DSP kernels are native
    Rust (with pure-Python fallbacks), so no JIT warm-up is required; importing
    the modules here just moves their one-time import cost out of the first
    trial executed in each worker.
    """
    try:
        from ..dsp import dsp_ops, limits, smoothing  # noqa: F401
    except (
        AttributeError,
        TypeError,
        ValueError,
        RuntimeError,
        OSError,
        ImportError,
        ModuleNotFoundError,
    ):
        pass
