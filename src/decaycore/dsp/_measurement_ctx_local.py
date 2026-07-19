# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Thread-local measurement side context for the DSP pipeline.

The context is set by engine_run before each generate_filter call and consumed
inside correction_baseline and mag_post_limits. Uses the same thread-local
pattern as _pruning.py so that parallel Optuna trials each see their own
context independently.

Usage:
    set_measurement_ctx(ctx)  # before generate_filter
    ...                       # generate_filter runs DSP pipeline
    clear_measurement_ctx()   # in finally block
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

from .correction_types import MeasurementSideContext

_local = threading.local()


def set_measurement_ctx(ctx: MeasurementSideContext | None) -> None:
    _local.ctx = ctx


def get_measurement_ctx() -> MeasurementSideContext | None:
    return getattr(_local, "ctx", None)


def clear_measurement_ctx() -> None:
    _local.ctx = None


@contextmanager
def measurement_ctx_scope(ctx: MeasurementSideContext | None) -> Iterator[None]:
    """Temporarily install one channel's measurement context and restore the caller state."""
    previous = get_measurement_ctx()
    set_measurement_ctx(ctx)
    try:
        yield
    finally:
        set_measurement_ctx(previous)
