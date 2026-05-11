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

import sys
from typing import Any

import numpy as np

from ...io.measurement_bundle import BassIntegrationBundle
from ._final_metrics import _final_metric_snapshot
from ._recommend_alignment import recommend_direct_dac_alignment
from ._recommend_allpass import recommend_direct_dac_allpass
from ._recommend_crossover import recommend_direct_dac_crossover
from ._utils import _safe_float, normalize_sub_combine_mode


def _get_pkg():
    """Return the bass_integration package module for patchable attribute lookup."""
    return sys.modules[__name__.rsplit(".", 1)[0]]


def _recommend_direct_dac_prepare_builtin_core(
    bundle: BassIntegrationBundle,
    *,
    profile: str,
    main_hpf_order: int,
    sub_lpf_order: int,
    sub_hpf_hz: float,
    sub_hpf_order: int,
    sub_combine_mode: str = "average",
    allpass_auto_enable: bool = False,
    callbacks=None,
) -> dict[str, Any]:
    """Fallback: run existing staged alignment + crossover and return unified result dict."""
    combine_mode_norm = normalize_sub_combine_mode(sub_combine_mode)

    def _cb_status(msg: str) -> None:
        if callbacks is not None:
            try:
                callbacks.status(msg)
            except Exception:
                pass

    # Baseline
    baseline_metrics = _get_pkg().compute_final_bass_integration_metrics(
        bundle,
        80.0,
        profile,
        mode="direct_dac",
        main_hpf_order=int(main_hpf_order),
        sub_lpf_order=int(sub_lpf_order),
        sub_hpf_hz=float(sub_hpf_hz),
        sub_hpf_order=int(sub_hpf_order),
        sub_combine_mode=combine_mode_norm,
    )
    baseline_snap = _final_metric_snapshot(baseline_metrics)

    # Alignment
    align_result: dict[str, Any] = {}
    try:
        align_result = dict(
            recommend_direct_dac_alignment(
                bundle,
                fc_hz=80.0,
                profile=profile,
                main_hpf_order=int(main_hpf_order),
                sub_lpf_order=int(sub_lpf_order),
                sub_hpf_hz=float(sub_hpf_hz),
                sub_hpf_order=int(sub_hpf_order),
                sub_combine_mode=combine_mode_norm,
            )
        )
    except Exception:
        pass
    best_delay = float(align_result.get("sub_delay_ms", 0.0) or 0.0)
    best_polarity = bool(align_result.get("sub_polarity_invert", False))
    best_gain = float(align_result.get("sub_gain_trim_db", 0.0) or 0.0)
    applied = bool(align_result.get("applied", False))

    # Crossover
    best_fc = 80.0
    best_sub_lpf = 80.0
    try:
        xo_result = dict(
            recommend_direct_dac_crossover(
                bundle,
                profile=profile,
                main_hpf_order=int(main_hpf_order),
                sub_lpf_order=int(sub_lpf_order),
                sub_hpf_hz=float(sub_hpf_hz),
                sub_hpf_order=int(sub_hpf_order),
                sub_combine_mode=combine_mode_norm,
                sub_delay_ms=best_delay,
                sub_polarity_invert=best_polarity,
                sub_gain_trim_db=best_gain,
            )
        )
        _rec_hz = _safe_float(xo_result.get("recommended_hz", 80.0), 80.0)
        _rec_lpf = _safe_float(xo_result.get("recommended_sub_lpf_hz", _rec_hz), _rec_hz)
        if np.isfinite(_rec_hz) and _rec_hz > 0.0:
            best_fc = float(_rec_hz)
            best_sub_lpf = float(_rec_lpf) if np.isfinite(_rec_lpf) and _rec_lpf >= best_fc else best_fc
    except Exception:
        pass

    optimized_metrics = _get_pkg().compute_final_bass_integration_metrics(
        bundle,
        best_fc,
        profile,
        mode="direct_dac",
        main_hpf_order=int(main_hpf_order),
        sub_lpf_order=int(sub_lpf_order),
        sub_hpf_hz=float(sub_hpf_hz),
        sub_hpf_order=int(sub_hpf_order),
        sub_combine_mode=combine_mode_norm,
        sub_delay_ms=best_delay,
        sub_polarity_invert=best_polarity,
        sub_gain_trim_db=best_gain,
        sub_lpf_hz=best_sub_lpf,
    )
    optimized_snap = _final_metric_snapshot(optimized_metrics)
    improvement_score = float(align_result.get("improvement_score", 0.0) or 0.0)

    # Allpass post-pass
    allpass_enabled = False
    allpass_freq_hz = 0.0
    allpass_q = 0.707
    allpass_reason = "Auto allpass optimization is OFF."
    if allpass_auto_enable:
        _cb_status("DecayCore automatic mode: bass integration allpass scan")
        try:
            _ap = dict(
                recommend_direct_dac_allpass(
                    bundle,
                    fc_hz=best_fc,
                    profile=profile,
                    main_hpf_order=int(main_hpf_order),
                    sub_lpf_order=int(sub_lpf_order),
                    sub_hpf_hz=float(sub_hpf_hz),
                    sub_hpf_order=int(sub_hpf_order),
                    sub_combine_mode=combine_mode_norm,
                    sub_delay_ms=best_delay,
                    sub_polarity_invert=best_polarity,
                    sub_gain_trim_db=best_gain,
                    sub_lpf_hz=best_sub_lpf,
                )
            )
            allpass_enabled = bool(_ap.get("enabled", False))
            allpass_freq_hz = float(_ap.get("freq_hz", 0.0) or 0.0)
            allpass_q = float(_ap.get("q", 0.707) or 0.707)
            allpass_reason = str(_ap.get("reason", "") or "")
        except Exception:
            allpass_reason = "Allpass post-pass failed."

    return {
        "applied": bool(applied),
        "backend": "builtin",
        "sub_delay_ms": float(best_delay if applied else 0.0),
        "sub_polarity_invert": bool(best_polarity if applied else False),
        "sub_gain_trim_db": float(best_gain if applied else 0.0),
        "recommended_hz": float(best_fc),
        "recommended_sub_lpf_hz": float(best_sub_lpf),
        "allpass_enabled": bool(allpass_enabled),
        "allpass_freq_hz": float(allpass_freq_hz),
        "allpass_q": float(allpass_q),
        "baseline": dict(baseline_snap),
        "optimized": dict(optimized_snap),
        "improvement_score": float(improvement_score),
        "reason": str(align_result.get("reason", "") or "Builtin fallback."),
        "study_trials": 0,
        "allpass_reason": str(allpass_reason),
    }
