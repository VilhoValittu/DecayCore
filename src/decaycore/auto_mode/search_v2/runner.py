# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Auto Search v2 runner."""

from __future__ import annotations

from .executor import execute_auto_search_plan
from .finalize import attach_search_v2_debug
from .input_model import build_auto_search_input
from .logging import log_plan_decision
from .plan import AutoSearchPlan
from .planner import determine_auto_search_plan
from .signature import compute_auto_search_signature_object
from ..shared import _auto_filter_cache_key, _auto_goal
from ..cache_signature import _auto_compat_version


def run_auto_search_v2(
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
    status_cb=None,
    n_trials: int,
) -> dict | None:
    context = {
        "fs_v": int(fs_v),
        "taps_v": int(taps_v),
        "xos": list(xos or []),
        "hpf": hpf,
        "hc_mode": str(dict(base_data or {}).get("hc_mode", "") or "").strip() or None,
    }
    search_input = build_auto_search_input(base_data, measurements, context)
    signature_obj = compute_auto_search_signature_object(search_input)
    signature = str(signature_obj.signature)
    decision = determine_auto_search_plan(
        search_input,
        dict(base_data or {}),
        options={
            "signature": signature,
            "signature_object": signature_obj,
            "compat_version": _auto_compat_version(dict(base_data or {})),
            "filter_key": _auto_filter_cache_key(dict(base_data or {})),
            "goal": _auto_goal(dict(base_data or {})),
        },
    )
    if isinstance(decision.cache_decision_report, dict):
        report = dict(decision.cache_decision_report or {})
        report["identity_base_data"] = search_input.base_data()
        object.__setattr__(decision, "cache_decision_report", report)
    seed_source = str(getattr(decision, "seed_source", "") or "").strip() or None
    if seed_source is None and decision.plan == AutoSearchPlan.CACHE_MICRO_REFINE:
        seed_source = "exact_cache"
    elif seed_source is None and decision.plan == AutoSearchPlan.LAST_BEST_MICRO_REFINE:
        seed_source = "last_best"
    log_plan_decision(decision, seed_source=seed_source)
    result = execute_auto_search_plan(
        decision=decision,
        base_data=dict(base_data or {}),
        measurements=dict(measurements or {}),
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
    return attach_search_v2_debug(result, decision=decision)
