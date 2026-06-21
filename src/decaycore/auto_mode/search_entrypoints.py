# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Automatic-mode search entrypoints."""

from __future__ import annotations

import logging

from . import api as auto_api
from . import (
    orchestrator_finalize as orchestrator_finalize,
    orchestrator_refine as orchestrator_refine,
)
logger = logging.getLogger("DecayCore")


def _build_auto_mode_orchestrator_runtime():
    from .search_v2.legacy_adapter import build_auto_mode_orchestrator_runtime

    return build_auto_mode_orchestrator_runtime()


def _record_auto_search_fallback(*args, **kwargs):
    from .search_v2.legacy_adapter import record_auto_search_fallback

    return record_auto_search_fallback(*args, **kwargs)


def _attach_auto_search_fallbacks(*args, **kwargs):
    from .search_v2.legacy_adapter import attach_auto_search_fallbacks

    return attach_auto_search_fallbacks(*args, **kwargs)


def _run_auto_search_v2(**kwargs):
    from .search_v2.runner import run_auto_search_v2

    return run_auto_search_v2(**kwargs)


def _run_auto_mode_search(
    *,
    base_data: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    hc_f,
    hc_m,
    pin_obj=None,
    status_cb,
    n_trials: int = auto_api.AUTO_MODE_TRIALS,
) -> dict | None:
    return _run_auto_search_v2(
        base_data=base_data,
        measurements=measurements,
        fs_v=int(fs_v),
        taps_v=int(taps_v),
        xos=list(xos or []),
        hpf=hpf,
        hc_f=hc_f,
        hc_m=hc_m,
        pin_obj=pin_obj,
        status_cb=status_cb,
        n_trials=int(n_trials),
    )
