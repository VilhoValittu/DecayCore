# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

from __future__ import annotations

import time

from ..application.run_request import RunRequest
from ..application.runtime_cache import reset_runtime_caches
from .pipeline_flow import _run_pipeline
from .process_run_support import ProcessRunSupport
from .run_finalize import _finalize_run_outputs
from .run_prepare_parts import (
    _prepare_target_curve_and_run_context,
    _prepare_ui_and_measurements,
)


def _run_auto_mode_seed_phases(*args, **kwargs):
    from .auto_flow_parts import _run_auto_mode_seed_phases as implementation

    return implementation(*args, **kwargs)


def _run_auto_mode_search_if_needed(*args, **kwargs):
    from .auto_flow_parts import _run_auto_mode_search_if_needed as implementation

    return implementation(*args, **kwargs)


def run_process_flow(*, request: RunRequest, support: ProcessRunSupport):
    run_started_at = time.perf_counter()
    request.run_started_at = run_started_at
    callbacks = support.ui_bridge.make_callbacks(run_started_at)

    try:
        ctx = _prepare_ui_and_measurements(
            request=request,
            callbacks=callbacks,
            support=support,
        )
        if ctx is None:
            return

        if bool(getattr(ctx, "auto_mode_enabled", False)):
            _run_auto_mode_seed_phases(
                ctx,
                callbacks=callbacks,
                support=support,
            )
        _prepare_target_curve_and_run_context(
            ctx,
            support=support,
            callbacks=callbacks,
        )
        if bool(getattr(ctx, "auto_mode_enabled", False)):
            _run_auto_mode_search_if_needed(
                ctx,
                callbacks=callbacks,
                support=support,
            )
        if not _run_pipeline(
            ctx,
            callbacks=callbacks,
            support=support,
        ):
            return
        _finalize_run_outputs(
            ctx,
            callbacks=callbacks,
            support=support,
        )
    finally:
        # Large FFT/preprocess caches are useful only during this run. Leaving
        # them alive until the next Run click makes completed ultra-high-rate
        # jobs retain hundreds of MiB (or more) while the UI is idle.
        reset_runtime_caches()


__all__ = ["ProcessRunSupport", "run_process_flow"]
