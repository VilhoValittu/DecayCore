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
from . import orchestrator_finalize, orchestrator_refine
from .search_v2.legacy_adapter import (
    attach_auto_search_fallbacks,
    build_auto_mode_orchestrator_runtime,
    record_auto_search_fallback,
)
from .search_v2.runner import run_auto_search_v2 as _run_auto_search_v2

logger = logging.getLogger("DecayCore")

_build_auto_mode_orchestrator_runtime = build_auto_mode_orchestrator_runtime
_record_auto_search_fallback = record_auto_search_fallback
_attach_auto_search_fallbacks = attach_auto_search_fallbacks


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
