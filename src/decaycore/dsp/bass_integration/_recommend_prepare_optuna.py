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

from ...auto_mode.shared import _auto_bass_integration_profile_weights
from ...io.measurement_bundle import BassIntegrationBundle
from ._constants import (
    AVR_CROSSOVER_CANDIDATES,
    DIRECT_DAC_ALIGNMENT_DELAY_CANDIDATES_MS,
    DIRECT_DAC_ALIGNMENT_GAIN_CANDIDATES_DB,
    DIRECT_DAC_OVERLAP_RATIOS,
    MIN_DIRECT_DAC_OVERLAP_RATIO,
)
from ._final_metrics import _final_metric_snapshot
from ._recommend_alignment import recommend_direct_dac_alignment
from ._recommend_allpass import recommend_direct_dac_allpass
from ._recommend_crossover import recommend_direct_dac_crossover
from ._recommend_prepare_dac import _recommend_direct_dac_prepare_builtin_core
from ._utils import _safe_float, normalize_sub_combine_mode


def _get_pkg():
    """Return the bass_integration package module for patchable attribute lookup."""
    return sys.modules[__name__.rsplit(".", 1)[0]]


def recommend_direct_dac_prepare_optuna(
    bundle: BassIntegrationBundle,
    *,
    profile: str,
    main_hpf_order: int,
    sub_lpf_order: int,
    sub_hpf_hz: float,
    sub_hpf_order: int,
    sub_combine_mode: str = "average",
    allpass_auto_enable: bool = False,
    trials: int = 48,
    startup_trials: int = 12,
    local_trials: int = 12,
    callbacks=None,
) -> dict[str, Any]:
    """Unified Optuna-based Direct-DAC bass integration prepare.

    Replaces the three separate staged passes (alignment, crossover, allpass)
    with one joint search over fc, overlap_ratio, delay, polarity, and gain.
    Falls back to builtin staged approach if optuna is unavailable.
    """
    _FC_LO = float(AVR_CROSSOVER_CANDIDATES[0])
    _FC_HI = float(AVR_CROSSOVER_CANDIDATES[-1])
    _DELAY_LO = float(min(DIRECT_DAC_ALIGNMENT_DELAY_CANDIDATES_MS))
    _DELAY_HI = float(max(DIRECT_DAC_ALIGNMENT_DELAY_CANDIDATES_MS))
    _GAIN_LO = float(min(DIRECT_DAC_ALIGNMENT_GAIN_CANDIDATES_DB))
    _GAIN_HI = float(max(DIRECT_DAC_ALIGNMENT_GAIN_CANDIDATES_DB))

    combine_mode_norm = normalize_sub_combine_mode(sub_combine_mode)
    hpf_order_i = max(1, int(main_hpf_order))
    lpf_order_i = max(1, int(sub_lpf_order))
    sub_hpf_hz_f = max(0.0, float(sub_hpf_hz))
    sub_hpf_order_i = max(1, int(sub_hpf_order))

    def _cb_status(msg: str) -> None:
        if callbacks is not None:
            try:
                callbacks.status(msg)
            except Exception:
                pass

    weights = _auto_bass_integration_profile_weights(profile)
    w_main_act = float(weights.get("main_activity", 6.0))

    # --- Memoization helpers ---
    def _quantize_fc(fc: float) -> float:
        return round(float(fc), 0)  # 1.0 Hz resolution

    def _quantize_delay(delay_ms: float) -> float:
        return round(float(delay_ms), 1)  # 0.1 ms resolution

    def _quantize_gain(gain_db: float) -> float:
        return round(float(gain_db), 1)  # 0.1 dB resolution

    _eval_cache: dict = {}
    _eval_total = [0]
    _eval_hits = [0]

    def _eval(fc: float, overlap_ratio: float, delay_ms: float, polarity: bool, gain_db: float) -> tuple[float, dict]:
        fc_q = _quantize_fc(fc)
        delay_q = _quantize_delay(delay_ms)
        gain_q = _quantize_gain(gain_db)
        cache_key = (fc_q, float(overlap_ratio), delay_q, bool(polarity), gain_q,
                     str(profile), hpf_order_i, lpf_order_i, sub_hpf_hz_f, sub_hpf_order_i,
                     combine_mode_norm)
        _eval_total[0] += 1
        cached = _eval_cache.get(cache_key)
        if cached is not None:
            _eval_hits[0] += 1
            return cached

        sub_lpf = float(fc_q * overlap_ratio)
        m = _get_pkg().compute_final_bass_integration_metrics(
            bundle,
            fc_q,
            profile,
            mode="direct_dac",
            main_hpf_order=hpf_order_i,
            sub_lpf_order=lpf_order_i,
            sub_hpf_hz=sub_hpf_hz_f,
            sub_hpf_order=sub_hpf_order_i,
            sub_combine_mode=combine_mode_norm,
            sub_delay_ms=float(delay_q),
            sub_polarity_invert=bool(polarity),
            sub_gain_trim_db=float(gain_q),
            sub_lpf_hz=sub_lpf,
        )
        obj = _safe_float(m.get("objective", float("nan")), float("nan"))
        if np.isfinite(obj):
            _pkg = _get_pkg()
            l_drop = _pkg._main_guard_band_drop_db(bundle.l_main, fc_q)
            r_drop = _pkg._main_guard_band_drop_db(bundle.r_main, fc_q)
            drop_vals = [v for v in (l_drop, r_drop) if np.isfinite(v)]
            if drop_vals:
                avg_drop = float(np.mean(np.asarray(drop_vals, dtype=float)))
                obj -= w_main_act * max(0.0, avg_drop) / 12.0
        result = (obj, m)
        _eval_cache[cache_key] = result
        return result

    # Baseline at canonical defaults
    baseline_obj, baseline_metrics = _eval(80.0, MIN_DIRECT_DAC_OVERLAP_RATIO, 0.0, False, 0.0)
    baseline_snap = _final_metric_snapshot(baseline_metrics)

    try:
        import optuna as _optuna  # type: ignore
        _optuna.logging.set_verbosity(_optuna.logging.WARNING)
    except Exception:
        return _recommend_direct_dac_prepare_builtin_core(
            bundle,
            profile=profile,
            main_hpf_order=hpf_order_i,
            sub_lpf_order=lpf_order_i,
            sub_hpf_hz=sub_hpf_hz_f,
            sub_hpf_order=sub_hpf_order_i,
            sub_combine_mode=combine_mode_norm,
            allpass_auto_enable=allpass_auto_enable,
            callbacks=callbacks,
        )

    _cb_status("DecayCore automatic mode: bass integration optuna init")

    # --- Seed candidates from builtins ---
    seeds: list[dict] = [
        {"fc_hz": 80.0, "overlap_ratio": MIN_DIRECT_DAC_OVERLAP_RATIO, "sub_delay_ms": 0.0, "sub_polarity_invert": False, "sub_gain_trim_db": 0.0},
    ]
    _align_delay = 0.0
    _align_polarity = False
    _align_gain = 0.0

    try:
        _ar = recommend_direct_dac_alignment(
            bundle,
            fc_hz=80.0,
            profile=profile,
            main_hpf_order=hpf_order_i,
            sub_lpf_order=lpf_order_i,
            sub_hpf_hz=sub_hpf_hz_f,
            sub_hpf_order=sub_hpf_order_i,
            sub_combine_mode=combine_mode_norm,
        )
        _align_delay = float(_ar.get("sub_delay_ms", 0.0) or 0.0)
        _align_polarity = bool(_ar.get("sub_polarity_invert", False))
        _align_gain = float(_ar.get("sub_gain_trim_db", 0.0) or 0.0)
        seeds.append({
            "fc_hz": 80.0, "overlap_ratio": MIN_DIRECT_DAC_OVERLAP_RATIO,
            "sub_delay_ms": _align_delay, "sub_polarity_invert": _align_polarity, "sub_gain_trim_db": _align_gain,
        })
    except Exception:
        pass

    try:
        _xr = recommend_direct_dac_crossover(
            bundle,
            profile=profile,
            main_hpf_order=hpf_order_i,
            sub_lpf_order=lpf_order_i,
            sub_hpf_hz=sub_hpf_hz_f,
            sub_hpf_order=sub_hpf_order_i,
            sub_combine_mode=combine_mode_norm,
        )
        _xo_hz = float(np.clip(_safe_float(_xr.get("recommended_hz", 80.0), 80.0), _FC_LO, _FC_HI))
        _xo_lpf = float(_safe_float(_xr.get("recommended_sub_lpf_hz", _xo_hz), _xo_hz))
        _xo_ratio_raw = _xo_lpf / max(_xo_hz, 1.0)
        _xo_ratio = min(DIRECT_DAC_OVERLAP_RATIOS, key=lambda r: abs(r - _xo_ratio_raw))
        seeds.append({
            "fc_hz": _xo_hz, "overlap_ratio": _xo_ratio,
            "sub_delay_ms": 0.0, "sub_polarity_invert": False, "sub_gain_trim_db": 0.0,
        })
        seeds.append({
            "fc_hz": _xo_hz, "overlap_ratio": _xo_ratio,
            "sub_delay_ms": _align_delay, "sub_polarity_invert": _align_polarity, "sub_gain_trim_db": _align_gain,
        })
    except Exception:
        pass

    # --- Global Optuna search ---
    sampler = _optuna.samplers.TPESampler(n_startup_trials=int(startup_trials), seed=42)
    study = _optuna.create_study(direction="maximize", sampler=sampler)

    for _s in seeds:
        try:
            study.enqueue_trial({
                "fc_hz": float(np.clip(_s["fc_hz"], _FC_LO, _FC_HI)),
                "overlap_ratio": float(_s["overlap_ratio"]),
                "sub_delay_ms": float(np.clip(_s["sub_delay_ms"], _DELAY_LO, _DELAY_HI)),
                "sub_polarity_invert": bool(_s["sub_polarity_invert"]),
                "sub_gain_trim_db": float(np.clip(_s["sub_gain_trim_db"], _GAIN_LO, _GAIN_HI)),
            })
        except Exception:
            pass

    _best_global = [float("-inf")]
    _trial_n = [0]

    def _objective(trial) -> float:
        fc = float(trial.suggest_float("fc_hz", _FC_LO, _FC_HI))
        overlap_ratio = float(trial.suggest_categorical("overlap_ratio", list(DIRECT_DAC_OVERLAP_RATIOS)))
        delay_ms = float(trial.suggest_float("sub_delay_ms", _DELAY_LO, _DELAY_HI))
        polarity = bool(trial.suggest_categorical("sub_polarity_invert", [False, True]))
        gain_db = float(trial.suggest_float("sub_gain_trim_db", _GAIN_LO, _GAIN_HI))
        obj, _ = _eval(fc, overlap_ratio, delay_ms, polarity, gain_db)
        if not np.isfinite(obj):
            return float("-inf")
        _trial_n[0] += 1
        improved = obj > _best_global[0] + 0.005
        if improved:
            _best_global[0] = obj
        if _trial_n[0] % 4 == 0 or improved:
            _cb_status(
                f"DecayCore automatic mode: bass integration optuna search "
                f"(trial {_trial_n[0]}/{trials})"
            )
        return float(obj)

    try:
        study.optimize(_objective, n_trials=int(trials))
    except Exception:
        pass

    # Extract global best params
    best_fc = 80.0
    best_ratio = 1.0
    best_delay = 0.0
    best_polarity = False
    best_gain = 0.0
    try:
        _gb = study.best_trial
        best_fc = float(np.clip(_gb.params.get("fc_hz", 80.0), _FC_LO, _FC_HI))
        best_ratio = float(_gb.params.get("overlap_ratio", MIN_DIRECT_DAC_OVERLAP_RATIO))
        best_delay = float(np.clip(_gb.params.get("sub_delay_ms", 0.0), _DELAY_LO, _DELAY_HI))
        best_polarity = bool(_gb.params.get("sub_polarity_invert", False))
        best_gain = float(np.clip(_gb.params.get("sub_gain_trim_db", 0.0), _GAIN_LO, _GAIN_HI))
    except Exception:
        pass

    # --- Local refine around best ---
    _cb_status("DecayCore automatic mode: bass integration optuna local refine")

    local_fc_lo = float(np.clip(best_fc - 10.0, _FC_LO, _FC_HI))
    local_fc_hi = float(np.clip(best_fc + 10.0, _FC_LO, _FC_HI))
    local_delay_lo = float(np.clip(best_delay - 2.0, _DELAY_LO, _DELAY_HI))
    local_delay_hi = float(np.clip(best_delay + 2.0, _DELAY_LO, _DELAY_HI))
    local_gain_lo = float(np.clip(best_gain - 2.0, _GAIN_LO, _GAIN_HI))
    local_gain_hi = float(np.clip(best_gain + 2.0, _GAIN_LO, _GAIN_HI))

    # Polarity: lock unless competing polarities are near-tie
    _polarity_tie = False
    try:
        _opp_obj, _ = _eval(best_fc, best_ratio, best_delay, not best_polarity, best_gain)
        _cur_obj, _ = _eval(best_fc, best_ratio, best_delay, best_polarity, best_gain)
        _polarity_tie = (
            np.isfinite(_opp_obj) and np.isfinite(_cur_obj)
            and abs(_opp_obj - _cur_obj) < 0.05
        )
    except Exception:
        pass
    local_polarity_choices = [False, True] if _polarity_tie else [best_polarity]

    # Overlap ratio: allow ±1 step around best
    _ratio_list = list(DIRECT_DAC_OVERLAP_RATIOS)
    try:
        _ratio_idx = _ratio_list.index(best_ratio)
    except ValueError:
        _ratio_idx = 0
    local_ratio_choices = list(dict.fromkeys([
        _ratio_list[max(0, _ratio_idx - 1)],
        best_ratio,
        _ratio_list[min(len(_ratio_list) - 1, _ratio_idx + 1)],
    ]))

    local_sampler = _optuna.samplers.TPESampler(
        n_startup_trials=max(3, int(local_trials) // 3), seed=43
    )
    local_study = _optuna.create_study(direction="maximize", sampler=local_sampler)
    try:
        local_study.enqueue_trial({
            "fc_hz": best_fc,
            "overlap_ratio": best_ratio,
            "sub_delay_ms": best_delay,
            "sub_polarity_invert": best_polarity,
            "sub_gain_trim_db": best_gain,
        })
    except Exception:
        pass

    def _local_objective(trial) -> float:
        fc = float(trial.suggest_float("fc_hz", local_fc_lo, local_fc_hi))
        overlap_ratio = float(trial.suggest_categorical("overlap_ratio", local_ratio_choices))
        delay_ms = float(trial.suggest_float("sub_delay_ms", local_delay_lo, local_delay_hi))
        polarity = bool(trial.suggest_categorical("sub_polarity_invert", local_polarity_choices))
        gain_db = float(trial.suggest_float("sub_gain_trim_db", local_gain_lo, local_gain_hi))
        obj, _ = _eval(fc, overlap_ratio, delay_ms, polarity, gain_db)
        return float(obj) if np.isfinite(obj) else float("-inf")

    try:
        local_study.optimize(_local_objective, n_trials=int(local_trials))
    except Exception:
        pass

    # Pick best of global + local
    try:
        _lb = local_study.best_trial
        _gb_val = study.best_value if study.trials else float("-inf")
        if _lb.value is not None and np.isfinite(_lb.value) and _lb.value > (_gb_val or float("-inf")):
            best_fc = float(np.clip(_lb.params.get("fc_hz", best_fc), _FC_LO, _FC_HI))
            best_ratio = float(_lb.params.get("overlap_ratio", best_ratio))
            best_delay = float(np.clip(_lb.params.get("sub_delay_ms", best_delay), _DELAY_LO, _DELAY_HI))
            best_polarity = bool(_lb.params.get("sub_polarity_invert", best_polarity))
            best_gain = float(np.clip(_lb.params.get("sub_gain_trim_db", best_gain), _GAIN_LO, _GAIN_HI))
    except Exception:
        pass

    # Final evaluation of chosen params
    best_sub_lpf = float(best_fc * best_ratio)
    _, optimized_metrics = _eval(best_fc, best_ratio, best_delay, best_polarity, best_gain)
    optimized_obj = _safe_float(optimized_metrics.get("objective", float("nan")), float("nan"))
    optimized_snap = _final_metric_snapshot(optimized_metrics)

    improvement_score = (
        float(optimized_obj - baseline_obj)
        if np.isfinite(optimized_obj) and np.isfinite(baseline_obj)
        else float("nan")
    )

    # Acceptance guards: reject if not meaningfully better or if quality worsens
    _opt_cancel = _safe_float(optimized_snap.get("cancellation_risk", float("nan")), float("nan"))
    _base_cancel = _safe_float(baseline_snap.get("cancellation_risk", float("nan")), float("nan"))
    _opt_ripple = _safe_float(optimized_snap.get("overlap_ripple_db", float("nan")), float("nan"))
    _base_ripple = _safe_float(baseline_snap.get("overlap_ripple_db", float("nan")), float("nan"))

    cancel_worsened = (
        np.isfinite(_opt_cancel) and np.isfinite(_base_cancel)
        and _opt_cancel > _base_cancel + 0.15
    )
    ripple_worsened = (
        np.isfinite(_opt_ripple) and np.isfinite(_base_ripple)
        and _opt_ripple > _base_ripple + 1.5
    )
    applied = bool(
        np.isfinite(improvement_score)
        and improvement_score > 0.03
        and not cancel_worsened
        and not ripple_worsened
    )

    if not applied:
        reason = "Optuna search found no meaningful improvement over baseline."
        best_fc = 80.0
        best_ratio = 1.0
        best_delay = 0.0
        best_polarity = False
        best_gain = 0.0
        best_sub_lpf = 80.0
        optimized_snap = dict(baseline_snap)
        improvement_score = 0.0
    else:
        reason = "Optuna unified search applied."

    # --- Allpass post-pass (separate, lightweight) ---
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
                    main_hpf_order=hpf_order_i,
                    sub_lpf_order=lpf_order_i,
                    sub_hpf_hz=sub_hpf_hz_f,
                    sub_hpf_order=sub_hpf_order_i,
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

    _study_trials = len(getattr(study, "trials", []))
    try:
        _study_trials += len(local_study.trials)
    except Exception:
        pass

    # Memoization telemetry
    _unique_evals = len(_eval_cache)
    _hit_ratio = float(_eval_hits[0]) / max(1, _eval_total[0])
    import logging as _logging
    _logging.getLogger("DecayCore.dsp").info(
        f"Bass-integration Optuna: trials={_study_trials}, "
        f"evals={_eval_total[0]}, unique={_unique_evals}, "
        f"cache_hits={_eval_hits[0]} ({_hit_ratio:.0%})"
    )

    _cb_status("DecayCore automatic mode: bass integration diagnostics refresh")

    return {
        "applied": bool(applied),
        "backend": "optuna",
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
        "improvement_score": float(improvement_score) if np.isfinite(improvement_score) else 0.0,
        "reason": str(reason),
        "study_trials": int(_study_trials),
        "allpass_reason": str(allpass_reason),
    }
