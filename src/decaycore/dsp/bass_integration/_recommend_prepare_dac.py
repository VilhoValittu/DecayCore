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

from typing import Any

import numpy as np

from ...io.measurement_bundle import BassIntegrationBundle
from ._final_metrics import _final_metric_snapshot
from ._recommend_alignment import recommend_direct_dac_alignment
from ._recommend_allpass import recommend_direct_dac_allpass
from ._recommend_crossover import recommend_direct_dac_crossover
from ._utils import _LOG, _get_pkg, _safe_float, _status_callback, normalize_sub_combine_mode


def _direct_dac_prepare_allpass_postpass(
    bundle: BassIntegrationBundle,
    *,
    enabled: bool,
    callbacks,
    fc_hz: float,
    profile: str,
    main_hpf_order: int,
    sub_lpf_order: int,
    sub_hpf_hz: float,
    sub_hpf_order: int,
    sub_combine_mode: str,
    sub_delay_ms: float,
    sub_polarity_invert: bool,
    sub_gain_trim_db: float,
    sub_lpf_hz: float,
) -> dict[str, Any]:
    result = {
        "allpass_enabled": False,
        "allpass_freq_hz": 0.0,
        "allpass_q": 0.707,
        "allpass_reason": "Auto allpass optimization is OFF.",
    }
    if not bool(enabled):
        return result
    _status_callback(callbacks, "DecayCore automatic mode: bass integration allpass scan")
    try:
        ap_result = dict(
            recommend_direct_dac_allpass(
                bundle,
                fc_hz=float(fc_hz),
                profile=profile,
                main_hpf_order=int(main_hpf_order),
                sub_lpf_order=int(sub_lpf_order),
                sub_hpf_hz=float(sub_hpf_hz),
                sub_hpf_order=int(sub_hpf_order),
                sub_combine_mode=sub_combine_mode,
                sub_delay_ms=float(sub_delay_ms),
                sub_polarity_invert=bool(sub_polarity_invert),
                sub_gain_trim_db=float(sub_gain_trim_db),
                sub_lpf_hz=float(sub_lpf_hz),
            )
        )
    except (

        AttributeError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        RuntimeError,
        OSError,
        ImportError,
        ModuleNotFoundError,
        NameError,
    ):
        _LOG.debug("Direct-DAC allpass post-pass failed", exc_info=True)
        result["allpass_reason"] = "Allpass post-pass failed."
        return result

    result.update(
        {
            "allpass_enabled": bool(ap_result.get("enabled", False)),
            "allpass_freq_hz": float(ap_result.get("freq_hz", 0.0) or 0.0),
            "allpass_q": float(ap_result.get("q", 0.707) or 0.707),
            "allpass_reason": str(ap_result.get("reason", "") or ""),
        }
    )
    return result


def _direct_dac_prepare_result(
    *,
    applied: bool,
    backend: str,
    sub_delay_ms: float,
    sub_polarity_invert: bool,
    sub_gain_trim_db: float,
    recommended_hz: float,
    recommended_sub_lpf_hz: float,
    allpass: dict[str, Any],
    baseline: dict[str, Any],
    optimized: dict[str, Any],
    improvement_score: float,
    reason: str,
    study_trials: int,
    candidate_score: float | None = None,
    reject_reasons: list[str] | None = None,
    worst_channel: str | None = None,
    sub_array_delay_ms: float | None = None,
    main_l_delay_ms: float | None = None,
    main_r_delay_ms: float | None = None,
) -> dict[str, Any]:
    selected_sub_delay = float(sub_delay_ms if applied else 0.0)
    selected_sub_array_delay = float(selected_sub_delay if sub_array_delay_ms is None else sub_array_delay_ms)
    main_delay_default = float(max(0.0, -selected_sub_delay))
    selected_main_l_delay = float(main_delay_default if main_l_delay_ms is None else main_l_delay_ms)
    selected_main_r_delay = float(main_delay_default if main_r_delay_ms is None else main_r_delay_ms)
    result = {
        "applied": bool(applied),
        "backend": str(backend),
        "sub_delay_ms": float(selected_sub_delay),
        "sub_array_delay_ms": float(selected_sub_array_delay if applied else 0.0),
        "main_l_delay_ms": float(selected_main_l_delay if applied else 0.0),
        "main_r_delay_ms": float(selected_main_r_delay if applied else 0.0),
        "sub_polarity_invert": bool(sub_polarity_invert if applied else False),
        "sub_gain_trim_db": float(sub_gain_trim_db if applied else 0.0),
        "recommended_hz": float(recommended_hz),
        "recommended_sub_lpf_hz": float(recommended_sub_lpf_hz),
        "allpass_enabled": bool(allpass.get("allpass_enabled", False)),
        "allpass_freq_hz": float(allpass.get("allpass_freq_hz", 0.0) or 0.0),
        "allpass_q": float(allpass.get("allpass_q", 0.707) or 0.707),
        "baseline": dict(baseline),
        "optimized": dict(optimized),
        "improvement_score": float(improvement_score) if np.isfinite(float(improvement_score)) else 0.0,
        "reason": str(reason),
        "study_trials": int(study_trials),
        "allpass_reason": str(allpass.get("allpass_reason", "") or ""),
    }
    if candidate_score is not None:
        result["candidate_score"] = float(candidate_score) if np.isfinite(float(candidate_score)) else float("inf")
    if reject_reasons is not None:
        result["reject_reasons"] = list(reject_reasons)
    if worst_channel is not None:
        result["worst_channel"] = str(worst_channel)
    return result


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
    except (

        AttributeError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        RuntimeError,
        OSError,
        ImportError,
        ModuleNotFoundError,
        NameError,
    ):
        _LOG.debug("Direct-DAC builtin alignment recommendation failed; keeping baseline alignment", exc_info=True)
    best_delay = float(align_result.get("sub_delay_ms", 0.0) or 0.0)
    best_sub_array_delay = float(align_result.get("sub_array_delay_ms", best_delay) or 0.0)
    best_main_l_delay = float(align_result.get("main_l_delay_ms", max(0.0, -best_delay)) or 0.0)
    best_main_r_delay = float(align_result.get("main_r_delay_ms", max(0.0, -best_delay)) or 0.0)
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
    except (

        AttributeError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        RuntimeError,
        OSError,
        ImportError,
        ModuleNotFoundError,
        NameError,
    ):
        _LOG.debug("Direct-DAC builtin crossover recommendation failed; keeping 80 Hz default", exc_info=True)

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

    allpass = _direct_dac_prepare_allpass_postpass(
        bundle,
        enabled=allpass_auto_enable,
        callbacks=callbacks,
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

    return _direct_dac_prepare_result(
        applied=applied,
        backend="builtin",
        sub_delay_ms=best_delay,
        sub_array_delay_ms=best_sub_array_delay,
        main_l_delay_ms=best_main_l_delay,
        main_r_delay_ms=best_main_r_delay,
        sub_polarity_invert=best_polarity,
        sub_gain_trim_db=best_gain,
        recommended_hz=best_fc,
        recommended_sub_lpf_hz=best_sub_lpf,
        allpass=allpass,
        baseline=baseline_snap,
        optimized=optimized_snap,
        improvement_score=improvement_score,
        reason=str(align_result.get("reason", "") or "Builtin fallback."),
        study_trials=0,
    )
