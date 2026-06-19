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
from ._recommend_crossover import recommend_direct_dac_crossover
from ._recommend_prepare_dac import (
    _direct_dac_prepare_allpass_postpass,
    _direct_dac_prepare_result,
    _recommend_direct_dac_prepare_builtin_core,
)
from ._sub_combine import build_bundle_combined_sub_transfer
from ._utils import (
    _LOG,
    _get_pkg,
    _safe_float,
    _status_callback,
    normalize_sub_combine_mode,
)
from .direct_dac import DirectDacBassIntegrationResult, run_direct_dac_bass_integration

_RECOVERABLE_PREPARE_EXCEPTIONS = (
    AttributeError,
    KeyError,
    TypeError,
    ValueError,
    RuntimeError,
    FloatingPointError,
)
_RECOVERABLE_OPTUNA_EXCEPTIONS = _RECOVERABLE_PREPARE_EXCEPTIONS + (
    ImportError,
    ModuleNotFoundError,
    OSError,
)


def _nearest_overlap_ratio(*, fc_hz: float, sub_lpf_hz: float) -> float:
    fc = max(float(fc_hz), 1.0)
    ratio_raw = max(MIN_DIRECT_DAC_OVERLAP_RATIO, float(sub_lpf_hz) / fc)
    return float(
        min(DIRECT_DAC_OVERLAP_RATIOS, key=lambda ratio: abs(float(ratio) - ratio_raw))
    )


def _seed_candidate_from_vectorized_result(
    result: DirectDacBassIntegrationResult,
) -> dict[str, float | bool] | None:
    fc_hz = _safe_float(result.main_hpf_hz, float("nan"))
    sub_lpf_hz = _safe_float(result.sub_lpf_hz, float("nan"))
    if not (
        np.isfinite(fc_hz)
        and fc_hz > 0.0
        and np.isfinite(sub_lpf_hz)
        and sub_lpf_hz >= fc_hz
    ):
        return None
    return {
        "fc_hz": float(fc_hz),
        "overlap_ratio": _nearest_overlap_ratio(
            fc_hz=float(fc_hz), sub_lpf_hz=float(sub_lpf_hz)
        ),
        "sub_delay_ms": float(_safe_float(result.sub_delay_ms, 0.0)),
        "sub_polarity_invert": bool(result.sub_polarity_invert),
        "sub_gain_trim_db": float(_safe_float(result.sub_gain_db, 0.0)),
    }


def _direct_dac_fast_seed_candidates(
    bundle: BassIntegrationBundle,
    *,
    combine_mode_norm: str,
) -> list[dict[str, float | bool]]:
    left_sub, _ = build_bundle_combined_sub_transfer(
        bundle,
        channel="l",
        mode=combine_mode_norm,
        label="L Direct-DAC fast-seed sub",
    )
    right_sub, _ = build_bundle_combined_sub_transfer(
        bundle,
        channel="r",
        mode=combine_mode_norm,
        label="R Direct-DAC fast-seed sub",
    )
    raw_results: list[DirectDacBassIntegrationResult] = [
        run_direct_dac_bass_integration(
            bundle.l_main.complex_spec, left_sub.complex_spec, bundle.l_main.freqs_hz
        ),
        run_direct_dac_bass_integration(
            bundle.r_main.complex_spec, right_sub.complex_spec, bundle.r_main.freqs_hz
        ),
    ]
    seeds = [
        seed
        for result in raw_results
        for seed in (_seed_candidate_from_vectorized_result(result),)
        if seed is not None
    ]
    if len(seeds) >= 2:
        avg_fc = float(
            np.mean(np.asarray([seed["fc_hz"] for seed in seeds], dtype=float))
        )
        avg_ratio = float(
            np.mean(np.asarray([seed["overlap_ratio"] for seed in seeds], dtype=float))
        )
        avg_delay = float(
            np.mean(np.asarray([seed["sub_delay_ms"] for seed in seeds], dtype=float))
        )
        avg_gain = float(
            np.mean(
                np.asarray([seed["sub_gain_trim_db"] for seed in seeds], dtype=float)
            )
        )
        polarities = {bool(seed["sub_polarity_invert"]) for seed in seeds}
        avg_candidate = {
            "fc_hz": float(
                np.clip(
                    avg_fc,
                    float(AVR_CROSSOVER_CANDIDATES[0]),
                    float(AVR_CROSSOVER_CANDIDATES[-1]),
                )
            ),
            "overlap_ratio": float(
                min(
                    DIRECT_DAC_OVERLAP_RATIOS,
                    key=lambda ratio: abs(float(ratio) - avg_ratio),
                )
            ),
            "sub_delay_ms": float(avg_delay),
            "sub_polarity_invert": (
                bool(next(iter(polarities))) if len(polarities) == 1 else False
            ),
            "sub_gain_trim_db": float(avg_gain),
        }
        seeds.append(avg_candidate)
    return seeds


def _direct_dac_build_seed_candidates(
    bundle: BassIntegrationBundle,
    *,
    profile: str,
    hpf_order_i: int,
    lpf_order_i: int,
    sub_hpf_hz_f: float,
    sub_hpf_order_i: int,
    combine_mode_norm: str,
    callbacks=None,
) -> tuple[list[dict], float, dict]:
    seeds: list[dict] = [
        {
            "fc_hz": 80.0,
            "overlap_ratio": MIN_DIRECT_DAC_OVERLAP_RATIO,
            "sub_delay_ms": 0.0,
            "sub_polarity_invert": False,
            "sub_gain_trim_db": 0.0,
        },
    ]
    baseline_obj = float("-inf")
    baseline_metrics: dict = {}

    try:
        baseline_obj, baseline_metrics = _direct_dac_eval_candidate(
            bundle,
            80.0,
            MIN_DIRECT_DAC_OVERLAP_RATIO,
            0.0,
            False,
            0.0,
            profile=profile,
            hpf_order_i=hpf_order_i,
            lpf_order_i=lpf_order_i,
            sub_hpf_hz_f=sub_hpf_hz_f,
            sub_hpf_order_i=sub_hpf_order_i,
            combine_mode_norm=combine_mode_norm,
        )
    except _RECOVERABLE_PREPARE_EXCEPTIONS:
        baseline_obj, baseline_metrics = float("-inf"), {}

    try:
        fast_seeds = _direct_dac_fast_seed_candidates(
            bundle,
            combine_mode_norm=combine_mode_norm,
        )
    except _RECOVERABLE_PREPARE_EXCEPTIONS:
        fast_seeds = []
        _LOG.debug(
            "Direct-DAC Optuna vectorized seed build failed; falling back to legacy seed scans",
            exc_info=True,
        )

    if fast_seeds:
        ranked_fast_seeds: list[tuple[float, dict[str, float | bool]]] = []
        for seed in fast_seeds:
            try:
                obj, _metrics = _direct_dac_eval_candidate(
                    bundle,
                    float(seed["fc_hz"]),
                    float(seed["overlap_ratio"]),
                    float(seed["sub_delay_ms"]),
                    bool(seed["sub_polarity_invert"]),
                    float(seed["sub_gain_trim_db"]),
                    profile=profile,
                    hpf_order_i=hpf_order_i,
                    lpf_order_i=lpf_order_i,
                    sub_hpf_hz_f=sub_hpf_hz_f,
                    sub_hpf_order_i=sub_hpf_order_i,
                    combine_mode_norm=combine_mode_norm,
                )
            except _RECOVERABLE_PREPARE_EXCEPTIONS:
                _LOG.debug(
                    "Direct-DAC Optuna fast seed evaluation failed; skipping candidate",
                    exc_info=True,
                )
                continue
            if np.isfinite(obj):
                ranked_fast_seeds.append((float(obj), dict(seed)))
        ranked_fast_seeds.sort(key=lambda item: item[0], reverse=True)
        seen_seed_keys: set[tuple[float, float, float, bool, float]] = {
            (
                round(float(seed["fc_hz"]), 4),
                round(float(seed["overlap_ratio"]), 4),
                round(float(seed["sub_delay_ms"]), 4),
                bool(seed["sub_polarity_invert"]),
                round(float(seed["sub_gain_trim_db"]), 4),
            )
            for seed in seeds
        }
        for _obj, seed in ranked_fast_seeds[:3]:
            seed_key = (
                round(float(seed["fc_hz"]), 4),
                round(float(seed["overlap_ratio"]), 4),
                round(float(seed["sub_delay_ms"]), 4),
                bool(seed["sub_polarity_invert"]),
                round(float(seed["sub_gain_trim_db"]), 4),
            )
            if seed_key in seen_seed_keys:
                continue
            seen_seed_keys.add(seed_key)
            seeds.append(seed)
        if len(seeds) > 1:
            return seeds, float(baseline_obj), dict(baseline_metrics or {})

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
        align_delay = float(_ar.get("sub_delay_ms", 0.0) or 0.0)
        align_polarity = bool(_ar.get("sub_polarity_invert", False))
        align_gain = float(_ar.get("sub_gain_trim_db", 0.0) or 0.0)
        seeds.append(
            {
                "fc_hz": 80.0,
                "overlap_ratio": MIN_DIRECT_DAC_OVERLAP_RATIO,
                "sub_delay_ms": align_delay,
                "sub_polarity_invert": align_polarity,
                "sub_gain_trim_db": align_gain,
            }
        )
    except _RECOVERABLE_PREPARE_EXCEPTIONS:
        _LOG.debug(
            "Direct-DAC Optuna alignment seed failed; continuing with baseline seed",
            exc_info=True,
        )

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
        xo_hz = float(
            np.clip(
                _safe_float(_xr.get("recommended_hz", 80.0), 80.0),
                float(AVR_CROSSOVER_CANDIDATES[0]),
                float(AVR_CROSSOVER_CANDIDATES[-1]),
            )
        )
        xo_lpf = float(_safe_float(_xr.get("recommended_sub_lpf_hz", xo_hz), xo_hz))
        xo_ratio_raw = xo_lpf / max(xo_hz, 1.0)
        xo_ratio = min(DIRECT_DAC_OVERLAP_RATIOS, key=lambda r: abs(r - xo_ratio_raw))
        seeds.append(
            {
                "fc_hz": xo_hz,
                "overlap_ratio": xo_ratio,
                "sub_delay_ms": 0.0,
                "sub_polarity_invert": False,
                "sub_gain_trim_db": 0.0,
            }
        )
        seeds.append(
            {
                "fc_hz": xo_hz,
                "overlap_ratio": xo_ratio,
                "sub_delay_ms": (
                    float(_ar.get("sub_delay_ms", 0.0) or 0.0)
                    if "_ar" in locals()
                    else 0.0
                ),
                "sub_polarity_invert": (
                    bool(_ar.get("sub_polarity_invert", False))
                    if "_ar" in locals()
                    else False
                ),
                "sub_gain_trim_db": (
                    float(_ar.get("sub_gain_trim_db", 0.0) or 0.0)
                    if "_ar" in locals()
                    else 0.0
                ),
            }
        )
    except _RECOVERABLE_PREPARE_EXCEPTIONS:
        _LOG.debug(
            "Direct-DAC Optuna crossover seed failed; continuing without crossover seed",
            exc_info=True,
        )

    return seeds, float(baseline_obj), dict(baseline_metrics or {})


def _direct_dac_eval_candidate(
    bundle: BassIntegrationBundle,
    fc: float,
    overlap_ratio: float,
    delay_ms: float,
    polarity: bool,
    gain_db: float,
    *,
    profile: str,
    hpf_order_i: int,
    lpf_order_i: int,
    sub_hpf_hz_f: float,
    sub_hpf_order_i: int,
    combine_mode_norm: str,
    weights: dict[str, float] | None = None,
    cache: dict | None = None,
) -> tuple[float, dict]:
    fc_q = round(float(fc), 0)
    delay_q = round(float(delay_ms), 1)
    gain_q = round(float(gain_db), 1)
    cache_key = (
        fc_q,
        float(overlap_ratio),
        delay_q,
        bool(polarity),
        gain_q,
        hpf_order_i,
        lpf_order_i,
        sub_hpf_hz_f,
        sub_hpf_order_i,
        str(profile),
        combine_mode_norm,
    )
    if cache is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

    sub_lpf = float(fc_q * overlap_ratio)
    metrics = _get_pkg().compute_final_bass_integration_metrics(
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
    obj = _safe_float(metrics.get("objective", float("nan")), float("nan"))
    if np.isfinite(obj):
        main_act_weight = float((weights or {}).get("main_activity", 6.0))
        l_drop = _get_pkg()._main_guard_band_drop_db(bundle.l_main, fc_q)
        r_drop = _get_pkg()._main_guard_band_drop_db(bundle.r_main, fc_q)
        drop_vals = [v for v in (l_drop, r_drop) if np.isfinite(v)]
        if drop_vals:
            avg_drop = float(np.mean(np.asarray(drop_vals, dtype=float)))
            obj -= main_act_weight * max(0.0, avg_drop) / 12.0
    result = (obj, metrics)
    if cache is not None:
        cache[cache_key] = result
    return result


def _direct_dac_enqueue_seed_trials(
    study,
    *,
    seeds: list[dict],
    fc_lo: float,
    fc_hi: float,
    delay_lo: float,
    delay_hi: float,
    gain_lo: float,
    gain_hi: float,
) -> None:
    for seed in seeds:
        try:
            study.enqueue_trial(
                {
                    "fc_hz": float(np.clip(seed["fc_hz"], fc_lo, fc_hi)),
                    "overlap_ratio": float(seed["overlap_ratio"]),
                    "sub_delay_ms": float(
                        np.clip(seed["sub_delay_ms"], delay_lo, delay_hi)
                    ),
                    "sub_polarity_invert": bool(seed["sub_polarity_invert"]),
                    "sub_gain_trim_db": float(
                        np.clip(seed["sub_gain_trim_db"], gain_lo, gain_hi)
                    ),
                }
            )
        except _RECOVERABLE_OPTUNA_EXCEPTIONS:
            _LOG.debug(
                "Direct-DAC Optuna seed enqueue failed; skipping seed", exc_info=True
            )


def _direct_dac_global_search_objective(
    trial,
    *,
    eval_fn,
    callbacks=None,
    trials: int,
    best_global: list[float],
    trial_n: list[int],
    fc_lo: float,
    fc_hi: float,
    delay_lo: float,
    delay_hi: float,
    gain_lo: float,
    gain_hi: float,
) -> float:
    fc = float(trial.suggest_float("fc_hz", fc_lo, fc_hi))
    overlap_ratio = float(
        trial.suggest_categorical("overlap_ratio", list(DIRECT_DAC_OVERLAP_RATIOS))
    )
    delay_ms = float(trial.suggest_float("sub_delay_ms", delay_lo, delay_hi))
    polarity = bool(trial.suggest_categorical("sub_polarity_invert", [False, True]))
    gain_db = float(trial.suggest_float("sub_gain_trim_db", gain_lo, gain_hi))
    obj, _ = eval_fn(fc, overlap_ratio, delay_ms, polarity, gain_db)
    if not np.isfinite(obj):
        return float("-inf")
    trial_n[0] += 1
    improved = obj > best_global[0] + 0.005
    if improved:
        best_global[0] = obj
    if trial_n[0] % 4 == 0 or improved:
        _status_callback(
            callbacks,
            f"DecayCore automatic mode: bass integration optuna search "
            f"(trial {trial_n[0]}/{trials})",
        )
    return float(obj)


def _direct_dac_extract_best_params(
    study,
    *,
    fc_lo: float,
    fc_hi: float,
    delay_lo: float,
    delay_hi: float,
    gain_lo: float,
    gain_hi: float,
) -> tuple[float, float, float, bool, float]:
    best_fc = 80.0
    best_ratio = MIN_DIRECT_DAC_OVERLAP_RATIO
    best_delay = 0.0
    best_polarity = False
    best_gain = 0.0
    try:
        _gb = study.best_trial
        best_fc = float(np.clip(_gb.params.get("fc_hz", 80.0), fc_lo, fc_hi))
        best_ratio = float(
            _gb.params.get("overlap_ratio", MIN_DIRECT_DAC_OVERLAP_RATIO)
        )
        best_delay = float(
            np.clip(_gb.params.get("sub_delay_ms", 0.0), delay_lo, delay_hi)
        )
        best_polarity = bool(_gb.params.get("sub_polarity_invert", False))
        best_gain = float(
            np.clip(_gb.params.get("sub_gain_trim_db", 0.0), gain_lo, gain_hi)
        )
    except _RECOVERABLE_OPTUNA_EXCEPTIONS:
        _LOG.debug(
            "Direct-DAC Optuna global best extraction failed; using default candidate",
            exc_info=True,
        )
    return (
        float(best_fc),
        float(best_ratio),
        float(best_delay),
        bool(best_polarity),
        float(best_gain),
    )


def _direct_dac_run_global_search(
    *,
    bundle: BassIntegrationBundle,
    profile: str,
    hpf_order_i: int,
    lpf_order_i: int,
    sub_hpf_hz_f: float,
    sub_hpf_order_i: int,
    combine_mode_norm: str,
    weights: dict[str, float],
    seeds: list[dict],
    callbacks=None,
    trials: int,
    startup_trials: int,
) -> tuple[Any, float, float, float, bool, float, Any]:
    try:
        import optuna as _optuna  # type: ignore

        _optuna.logging.set_verbosity(_optuna.logging.WARNING)
    except _RECOVERABLE_OPTUNA_EXCEPTIONS:
        _LOG.debug(
            "Optuna unavailable for Direct-DAC prepare; falling back to builtin",
            exc_info=True,
        )
        return None, float("nan"), 80.0, MIN_DIRECT_DAC_OVERLAP_RATIO, False, 0.0, None

    sampler = _optuna.samplers.TPESampler(n_startup_trials=int(startup_trials), seed=42)
    study = _optuna.create_study(direction="maximize", sampler=sampler)
    eval_cache: dict = {}
    eval_total = [0]
    eval_hits = [0]

    def _eval(
        fc: float, overlap_ratio: float, delay_ms: float, polarity: bool, gain_db: float
    ) -> tuple[float, dict]:
        eval_total[0] += 1
        cache_key = (
            round(float(fc), 0),
            float(overlap_ratio),
            round(float(delay_ms), 1),
            bool(polarity),
            round(float(gain_db), 1),
            hpf_order_i,
            lpf_order_i,
            sub_hpf_hz_f,
            sub_hpf_order_i,
            str(profile),
            combine_mode_norm,
        )
        cached = eval_cache.get(cache_key)
        if cached is not None:
            eval_hits[0] += 1
            return cached
        result = _direct_dac_eval_candidate(
            bundle,
            fc,
            overlap_ratio,
            delay_ms,
            polarity,
            gain_db,
            profile=profile,
            hpf_order_i=hpf_order_i,
            lpf_order_i=lpf_order_i,
            sub_hpf_hz_f=sub_hpf_hz_f,
            sub_hpf_order_i=sub_hpf_order_i,
            combine_mode_norm=combine_mode_norm,
            weights=weights,
        )
        eval_cache[cache_key] = result
        return result

    _direct_dac_enqueue_seed_trials(
        study,
        seeds=seeds,
        fc_lo=float(AVR_CROSSOVER_CANDIDATES[0]),
        fc_hi=float(AVR_CROSSOVER_CANDIDATES[-1]),
        delay_lo=float(min(DIRECT_DAC_ALIGNMENT_DELAY_CANDIDATES_MS)),
        delay_hi=float(max(DIRECT_DAC_ALIGNMENT_DELAY_CANDIDATES_MS)),
        gain_lo=float(min(DIRECT_DAC_ALIGNMENT_GAIN_CANDIDATES_DB)),
        gain_hi=float(max(DIRECT_DAC_ALIGNMENT_GAIN_CANDIDATES_DB)),
    )

    _best_global = [float("-inf")]
    _trial_n = [0]
    _objective = lambda trial: _direct_dac_global_search_objective(
        trial,
        eval_fn=_eval,
        callbacks=callbacks,
        trials=trials,
        best_global=_best_global,
        trial_n=_trial_n,
        fc_lo=float(AVR_CROSSOVER_CANDIDATES[0]),
        fc_hi=float(AVR_CROSSOVER_CANDIDATES[-1]),
        delay_lo=float(min(DIRECT_DAC_ALIGNMENT_DELAY_CANDIDATES_MS)),
        delay_hi=float(max(DIRECT_DAC_ALIGNMENT_DELAY_CANDIDATES_MS)),
        gain_lo=float(min(DIRECT_DAC_ALIGNMENT_GAIN_CANDIDATES_DB)),
        gain_hi=float(max(DIRECT_DAC_ALIGNMENT_GAIN_CANDIDATES_DB)),
    )

    try:
        study.optimize(_objective, n_trials=int(trials))
    except _RECOVERABLE_OPTUNA_EXCEPTIONS:
        _LOG.debug(
            "Direct-DAC Optuna global search failed; using best available seed/default",
            exc_info=True,
        )

    best_fc, best_ratio, best_delay, best_polarity, best_gain = (
        _direct_dac_extract_best_params(
            study,
            fc_lo=float(AVR_CROSSOVER_CANDIDATES[0]),
            fc_hi=float(AVR_CROSSOVER_CANDIDATES[-1]),
            delay_lo=float(min(DIRECT_DAC_ALIGNMENT_DELAY_CANDIDATES_MS)),
            delay_hi=float(max(DIRECT_DAC_ALIGNMENT_DELAY_CANDIDATES_MS)),
            gain_lo=float(min(DIRECT_DAC_ALIGNMENT_GAIN_CANDIDATES_DB)),
            gain_hi=float(max(DIRECT_DAC_ALIGNMENT_GAIN_CANDIDATES_DB)),
        )
    )

    setattr(study, "_decaycore_eval_total", eval_total[0])
    setattr(study, "_decaycore_eval_hits", eval_hits[0])
    setattr(study, "_decaycore_eval_cache", eval_cache)
    return (
        study,
        float(best_fc),
        float(best_ratio),
        float(best_delay),
        bool(best_polarity),
        float(best_gain),
        _eval,
    )


def _direct_dac_run_local_refine(
    *,
    study,
    eval_fn,
    best_fc: float,
    best_ratio: float,
    best_delay: float,
    best_polarity: bool,
    best_gain: float,
    callbacks=None,
    local_trials: int,
) -> tuple[Any, float, float, float, bool, float]:
    try:
        import optuna as _optuna  # type: ignore
    except _RECOVERABLE_OPTUNA_EXCEPTIONS:
        return None, best_fc, best_ratio, best_delay, best_polarity, best_gain

    _status_callback(
        callbacks, "DecayCore automatic mode: bass integration optuna local refine"
    )

    local_fc_lo = float(
        np.clip(
            best_fc - 10.0,
            float(AVR_CROSSOVER_CANDIDATES[0]),
            float(AVR_CROSSOVER_CANDIDATES[-1]),
        )
    )
    local_fc_hi = float(
        np.clip(
            best_fc + 10.0,
            float(AVR_CROSSOVER_CANDIDATES[0]),
            float(AVR_CROSSOVER_CANDIDATES[-1]),
        )
    )
    local_delay_lo = float(
        np.clip(
            best_delay - 2.0,
            float(min(DIRECT_DAC_ALIGNMENT_DELAY_CANDIDATES_MS)),
            float(max(DIRECT_DAC_ALIGNMENT_DELAY_CANDIDATES_MS)),
        )
    )
    local_delay_hi = float(
        np.clip(
            best_delay + 2.0,
            float(min(DIRECT_DAC_ALIGNMENT_DELAY_CANDIDATES_MS)),
            float(max(DIRECT_DAC_ALIGNMENT_DELAY_CANDIDATES_MS)),
        )
    )
    local_gain_lo = float(
        np.clip(
            best_gain - 2.0,
            float(min(DIRECT_DAC_ALIGNMENT_GAIN_CANDIDATES_DB)),
            float(max(DIRECT_DAC_ALIGNMENT_GAIN_CANDIDATES_DB)),
        )
    )
    local_gain_hi = float(
        np.clip(
            best_gain + 2.0,
            float(min(DIRECT_DAC_ALIGNMENT_GAIN_CANDIDATES_DB)),
            float(max(DIRECT_DAC_ALIGNMENT_GAIN_CANDIDATES_DB)),
        )
    )

    _polarity_tie = False
    try:
        _opp_obj, _ = eval_fn(
            best_fc, best_ratio, best_delay, not best_polarity, best_gain
        )
        _cur_obj, _ = eval_fn(best_fc, best_ratio, best_delay, best_polarity, best_gain)
        _polarity_tie = (
            np.isfinite(_opp_obj)
            and np.isfinite(_cur_obj)
            and abs(_opp_obj - _cur_obj) < 0.05
        )
    except _RECOVERABLE_PREPARE_EXCEPTIONS:
        _LOG.debug(
            "Direct-DAC Optuna polarity tie check failed; locking current polarity",
            exc_info=True,
        )
    local_polarity_choices = [False, True] if _polarity_tie else [best_polarity]

    _ratio_list = list(DIRECT_DAC_OVERLAP_RATIOS)
    try:
        _ratio_idx = _ratio_list.index(best_ratio)
    except ValueError:
        _ratio_idx = 0
    local_ratio_choices = list(
        dict.fromkeys(
            [
                _ratio_list[max(0, _ratio_idx - 1)],
                best_ratio,
                _ratio_list[min(len(_ratio_list) - 1, _ratio_idx + 1)],
            ]
        )
    )

    local_sampler = _optuna.samplers.TPESampler(
        n_startup_trials=max(3, int(local_trials) // 3), seed=43
    )
    local_study = _optuna.create_study(direction="maximize", sampler=local_sampler)
    try:
        local_study.enqueue_trial(
            {
                "fc_hz": best_fc,
                "overlap_ratio": best_ratio,
                "sub_delay_ms": best_delay,
                "sub_polarity_invert": best_polarity,
                "sub_gain_trim_db": best_gain,
            }
        )
    except _RECOVERABLE_OPTUNA_EXCEPTIONS:
        _LOG.debug("Direct-DAC Optuna local seed enqueue failed", exc_info=True)

    def _local_objective(trial) -> float:
        fc = float(trial.suggest_float("fc_hz", local_fc_lo, local_fc_hi))
        overlap_ratio = float(
            trial.suggest_categorical("overlap_ratio", local_ratio_choices)
        )
        delay_ms = float(
            trial.suggest_float("sub_delay_ms", local_delay_lo, local_delay_hi)
        )
        polarity = bool(
            trial.suggest_categorical("sub_polarity_invert", local_polarity_choices)
        )
        gain_db = float(
            trial.suggest_float("sub_gain_trim_db", local_gain_lo, local_gain_hi)
        )
        obj, _ = eval_fn(fc, overlap_ratio, delay_ms, polarity, gain_db)
        return float(obj) if np.isfinite(obj) else float("-inf")

    try:
        local_study.optimize(_local_objective, n_trials=int(local_trials))
    except _RECOVERABLE_OPTUNA_EXCEPTIONS:
        _LOG.debug(
            "Direct-DAC Optuna local refine failed; keeping global best", exc_info=True
        )

    try:
        _lb = local_study.best_trial
        _gb_val = study.best_value if study.trials else float("-inf")
        if (
            _lb.value is not None
            and np.isfinite(_lb.value)
            and _lb.value > (_gb_val or float("-inf"))
        ):
            best_fc = float(
                np.clip(
                    _lb.params.get("fc_hz", best_fc),
                    float(AVR_CROSSOVER_CANDIDATES[0]),
                    float(AVR_CROSSOVER_CANDIDATES[-1]),
                )
            )
            best_ratio = float(_lb.params.get("overlap_ratio", best_ratio))
            best_delay = float(
                np.clip(
                    _lb.params.get("sub_delay_ms", best_delay),
                    float(min(DIRECT_DAC_ALIGNMENT_DELAY_CANDIDATES_MS)),
                    float(max(DIRECT_DAC_ALIGNMENT_DELAY_CANDIDATES_MS)),
                )
            )
            best_polarity = bool(_lb.params.get("sub_polarity_invert", best_polarity))
            best_gain = float(
                np.clip(
                    _lb.params.get("sub_gain_trim_db", best_gain),
                    float(min(DIRECT_DAC_ALIGNMENT_GAIN_CANDIDATES_DB)),
                    float(max(DIRECT_DAC_ALIGNMENT_GAIN_CANDIDATES_DB)),
                )
            )
    except _RECOVERABLE_OPTUNA_EXCEPTIONS:
        _LOG.debug(
            "Direct-DAC Optuna local best extraction failed; keeping global best",
            exc_info=True,
        )

    return local_study, best_fc, best_ratio, best_delay, best_polarity, best_gain


def _direct_dac_finalize_prepare_result(
    *,
    bundle: BassIntegrationBundle,
    profile: str,
    hpf_order_i: int,
    lpf_order_i: int,
    sub_hpf_hz_f: float,
    sub_hpf_order_i: int,
    combine_mode_norm: str,
    allpass_auto_enable: bool,
    callbacks=None,
    baseline_obj: float,
    baseline_snap: dict,
    study,
    local_study,
    eval_fn,
    best_fc: float,
    best_ratio: float,
    best_delay: float,
    best_polarity: bool,
    best_gain: float,
    weights: dict[str, float],
) -> dict[str, Any]:
    best_sub_lpf = float(best_fc * best_ratio)
    _, optimized_metrics = eval_fn(
        best_fc, best_ratio, best_delay, best_polarity, best_gain
    )
    optimized_obj = _safe_float(
        optimized_metrics.get("objective", float("nan")), float("nan")
    )
    optimized_snap = _final_metric_snapshot(optimized_metrics)
    candidate_score = _safe_float(
        optimized_metrics.get("bass_direct_dac_candidate_score", -optimized_obj),
        -optimized_obj,
    )
    reject_reasons = list(
        optimized_metrics.get("bass_direct_dac_reject_reasons", []) or []
    )
    worst_channel = str(
        optimized_metrics.get(
            "bass_direct_dac_worst_channel",
            optimized_metrics.get("bass_dominant_channel", "unknown"),
        )
        or "unknown"
    )

    improvement_score = (
        float(optimized_obj - baseline_obj)
        if np.isfinite(optimized_obj) and np.isfinite(baseline_obj)
        else float("nan")
    )

    _opt_cancel = _safe_float(
        optimized_snap.get("cancellation_risk", float("nan")), float("nan")
    )
    _base_cancel = _safe_float(
        baseline_snap.get("cancellation_risk", float("nan")), float("nan")
    )
    _opt_ripple = _safe_float(
        optimized_snap.get("overlap_ripple_db", float("nan")), float("nan")
    )
    _base_ripple = _safe_float(
        baseline_snap.get("overlap_ripple_db", float("nan")), float("nan")
    )

    cancel_worsened = (
        np.isfinite(_opt_cancel)
        and np.isfinite(_base_cancel)
        and _opt_cancel > _base_cancel + 0.15
    )
    ripple_worsened = (
        np.isfinite(_opt_ripple)
        and np.isfinite(_base_ripple)
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

    allpass = _direct_dac_prepare_allpass_postpass(
        bundle,
        enabled=allpass_auto_enable,
        callbacks=callbacks,
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

    _study_trials = len(getattr(study, "trials", []))
    try:
        _study_trials += len(local_study.trials)
    except _RECOVERABLE_OPTUNA_EXCEPTIONS:
        _LOG.debug("Direct-DAC Optuna local trial count unavailable", exc_info=True)

    _unique_evals = len(getattr(study, "_decaycore_eval_cache", {}) or {})
    _eval_total = int(getattr(study, "_decaycore_eval_total", 0) or 0)
    _eval_hits = int(getattr(study, "_decaycore_eval_hits", 0) or 0)
    _hit_ratio = float(_eval_hits) / max(1, _eval_total)
    _LOG.info(
        f"Bass-integration Optuna: trials={_study_trials}, "
        f"evals={_eval_total}, unique={_unique_evals}, "
        f"cache_hits={_eval_hits} ({_hit_ratio:.0%})"
    )

    _status_callback(
        callbacks, "DecayCore automatic mode: bass integration diagnostics refresh"
    )

    return _direct_dac_prepare_result(
        applied=applied,
        backend="optuna",
        sub_delay_ms=best_delay,
        sub_polarity_invert=best_polarity,
        sub_gain_trim_db=best_gain,
        recommended_hz=best_fc,
        recommended_sub_lpf_hz=best_sub_lpf,
        allpass=allpass,
        baseline=baseline_snap,
        optimized=optimized_snap,
        improvement_score=improvement_score,
        reason=reason,
        study_trials=_study_trials,
        candidate_score=candidate_score,
        reject_reasons=reject_reasons,
        worst_channel=worst_channel,
    )


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
    trials: int = 2048,
    startup_trials: int = 12,
    local_trials: int = 12,
    callbacks=None,
) -> dict[str, Any]:
    """Unified Optuna-based Direct-DAC bass integration prepare.

    Replaces the three separate staged passes (alignment, crossover, allpass)
    with one joint search over fc, overlap_ratio, delay, polarity, and gain.
    Falls back to builtin staged approach if optuna is unavailable.
    """
    combine_mode_norm = normalize_sub_combine_mode(sub_combine_mode)
    hpf_order_i = max(1, int(main_hpf_order))
    lpf_order_i = max(1, int(sub_lpf_order))
    sub_hpf_hz_f = max(0.0, float(sub_hpf_hz))
    sub_hpf_order_i = max(1, int(sub_hpf_order))

    weights = _auto_bass_integration_profile_weights(profile)
    seeds, baseline_obj, baseline_metrics = _direct_dac_build_seed_candidates(
        bundle,
        profile=profile,
        hpf_order_i=hpf_order_i,
        lpf_order_i=lpf_order_i,
        sub_hpf_hz_f=sub_hpf_hz_f,
        sub_hpf_order_i=sub_hpf_order_i,
        combine_mode_norm=combine_mode_norm,
        callbacks=callbacks,
    )
    baseline_snap = _final_metric_snapshot(baseline_metrics)

    study, best_fc, best_ratio, best_delay, best_polarity, best_gain, eval_fn = (
        _direct_dac_run_global_search(
            bundle=bundle,
            profile=profile,
            hpf_order_i=hpf_order_i,
            lpf_order_i=lpf_order_i,
            sub_hpf_hz_f=sub_hpf_hz_f,
            sub_hpf_order_i=sub_hpf_order_i,
            combine_mode_norm=combine_mode_norm,
            weights=weights,
            seeds=seeds,
            callbacks=callbacks,
            trials=trials,
            startup_trials=startup_trials,
        )
    )
    if study is None:
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

    local_study, best_fc, best_ratio, best_delay, best_polarity, best_gain = (
        _direct_dac_run_local_refine(
            study=study,
            eval_fn=eval_fn,
            best_fc=best_fc,
            best_ratio=best_ratio,
            best_delay=best_delay,
            best_polarity=best_polarity,
            best_gain=best_gain,
            callbacks=callbacks,
            local_trials=local_trials,
        )
    )

    return _direct_dac_finalize_prepare_result(
        bundle=bundle,
        profile=profile,
        hpf_order_i=hpf_order_i,
        lpf_order_i=lpf_order_i,
        sub_hpf_hz_f=sub_hpf_hz_f,
        sub_hpf_order_i=sub_hpf_order_i,
        combine_mode_norm=combine_mode_norm,
        allpass_auto_enable=allpass_auto_enable,
        callbacks=callbacks,
        baseline_obj=baseline_obj,
        baseline_snap=baseline_snap,
        study=study,
        local_study=local_study,
        eval_fn=eval_fn,
        best_fc=best_fc,
        best_ratio=best_ratio,
        best_delay=best_delay,
        best_polarity=best_polarity,
        best_gain=best_gain,
        weights=weights,
    )
