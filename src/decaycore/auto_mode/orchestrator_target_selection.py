# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Final selection coordination for target-curve selection orchestration."""

from __future__ import annotations

import logging

from .cache_signature import (
    _auto_apply_seed,
    _auto_compat_version,
    get_or_build_synth_target,
)
from .cache_measurement_sig import _auto_search_measurement_identity, _auto_target_study_sig
from .runtime_context import coerce_orchestrator_runtime
from .rank_score import official_rank_score
from .scoring_ranking import (
    _auto_select_best_scored,
    _auto_target_result_mode_ripple,
    _auto_target_result_rank_key,
)
from .shared_parts import (
    AutoModeConfig,
    AUTO_MODE_CACHE_SCHEMA_VERSION,
    AUTO_MODE_SYNTH_TARGET_BASS_COMP_FRAC,
    AUTO_MODE_SYNTH_TARGET_BASS_COMP_REF_DB,
    AUTO_MODE_SYNTH_TARGET_HF_COMP_FRAC,
    AUTO_MODE_SYNTH_TARGET_NAME,
    AUTO_MODE_SYNTH_TARGET_SMOOTH_OCT,
    AUTO_MODE_SYNTH_TARGET_TILT_COMP_FRAC,
    AUTO_MODE_TARGET_BEST_RANK_TIE_EPS,
    AUTO_MODE_TARGET_TOP_N,
    AUTO_MODE_TARGET_TRIALS_PER_CURVE,
    AUTO_MODE_TARGET_USE_OPTUNA,
    _auto_filter_cache_key,
    _auto_goal,
    _auto_goal_basis_text,
    _auto_metric_text,
    _auto_optimizer_backend,
    _auto_safe_float,
)
from ..dsp.target_synthesis import (
    adaptive_target_diagnostics,
    synthesize_target_from_measurements,
)
from .orchestrator_target_types import (
    _TargetSelectionContext,
    _TargetSelectionOutcome,
    _TargetSelectionSetup,
    _TargetShortlistState,
)
from .orchestrator_target_cache import (
    _resolve_cached_target_state,
    _try_exact_cached_target_result,
    _try_measurement_global_target_result,
)
from .orchestrator_target_shortlist import (
    _apply_target_shortlist_modifiers,
    _evaluate_target_shortlist_core,
    _load_quick_target_selection,
)

logger = logging.getLogger("DecayCore")


def _prepare_target_selection_setup(
    *,
    base_data: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    runtime=None,
) -> _TargetSelectionSetup:
    runtime_obj = coerce_orchestrator_runtime(runtime)
    goal = _auto_goal(base_data)
    cfg = AutoModeConfig.from_base_data(base_data)
    compat_version = _auto_compat_version(base_data)
    filter_key = _auto_filter_cache_key(base_data)
    rank_basis = _auto_goal_basis_text(goal)
    optimizer_backend = _auto_optimizer_backend(
        base_data,
        default_optuna_enabled=bool(cfg.optuna_pilot_enabled and AUTO_MODE_TARGET_USE_OPTUNA),
    )
    optuna_mod = (
        runtime_obj.auto_import_optuna() if str(optimizer_backend) == "optuna" else None
    )
    if str(optimizer_backend) == "optuna" and optuna_mod is None:
        logger.warning(
            "Automatic mode target select: optuna backend requested but unavailable; "
            "falling back to builtin sampler."
        )
        optimizer_backend = "builtin"
    logger.info(
        "Automatic mode target select: goal=%s, basis=%s",
        str(goal),
        str(rank_basis),
    )
    measurement_identity = _auto_search_measurement_identity(measurements)
    target_study_sig = _auto_target_study_sig(measurement_identity, goal, filter_key,)
    seed_target = int(str(target_study_sig)[:16], 16) % (2**31 - 1)
    _auto_apply_seed(seed_target)
    logger.info(
        "Automatic mode target select: seed=%d cache_schema=%d",
        int(seed_target),
        int(AUTO_MODE_CACHE_SCHEMA_VERSION),
    )
    return _TargetSelectionSetup(
        runtime=runtime_obj,
        cfg=cfg,
        goal=str(goal),
        compat_version=str(compat_version),
        filter_key=str(filter_key),
        rank_basis=str(rank_basis),
        optimizer_backend=str(optimizer_backend),
        optuna_mod=optuna_mod,
        seed_target=int(seed_target),
        target_study_sig=str(target_study_sig),
    )


def _try_adaptive_target_fast_path(
    *,
    base_data: dict,
    measurements: dict,
    rank_basis: str,
    goal: str,
    status_cb,
) -> dict | None:
    atm = str(base_data.get("auto_target_mode", "auto") or "auto").strip().lower()
    if atm != "adaptive":
        return None
    try:
        synth = get_or_build_synth_target(
            measurements,
            tilt_comp_frac=float(AUTO_MODE_SYNTH_TARGET_TILT_COMP_FRAC),
            bass_comp_frac=float(AUTO_MODE_SYNTH_TARGET_BASS_COMP_FRAC),
            bass_comp_ref_db=float(AUTO_MODE_SYNTH_TARGET_BASS_COMP_REF_DB),
            hf_comp_frac=float(AUTO_MODE_SYNTH_TARGET_HF_COMP_FRAC),
            smooth_oct=float(AUTO_MODE_SYNTH_TARGET_SMOOTH_OCT),
            synth_fn=synthesize_target_from_measurements,
        )
        if synth is not None:
            synth_f, synth_m = synth
            diagnostics = adaptive_target_diagnostics(
                measurements.get("f_l"),
                measurements.get("m_l"),
                measurements.get("f_r"),
                measurements.get("m_r"),
                target_f=synth_f,
                target_m=synth_m,
                smooth_oct=float(AUTO_MODE_SYNTH_TARGET_SMOOTH_OCT),
                measurements=measurements,
            )
            logger.info(
                "Automatic mode target select: adaptive fast path used "
                "(bass_residual=%.3f dB, channel_disagreement=%.3f dB, "
                "channel_confidence=%.3f, rt60_confidence=%.3f, "
                "target_delta=[%.3f, %.3f] dB%s)",
                _auto_safe_float(diagnostics.get("bass_residual_db"), 0.0),
                _auto_safe_float(diagnostics.get("channel_disagreement_db"), 0.0),
                _auto_safe_float(diagnostics.get("channel_confidence"), 0.0),
                _auto_safe_float(diagnostics.get("rt60_confidence"), 0.0),
                _auto_safe_float(diagnostics.get("target_delta_min_db"), 0.0),
                _auto_safe_float(diagnostics.get("target_delta_max_db"), 0.0),
                (
                    f", fallback={diagnostics.get('fallback_reason')}"
                    if diagnostics.get("fallback_reason")
                    else ""
                ),
            )
            if callable(status_cb):
                if diagnostics.get("fallback_reason"):
                    status_cb(
                        "DecayCore automatic mode: adaptive target used conservative "
                        f"fallback ({diagnostics.get('fallback_reason')})"
                    )
                else:
                    status_cb(
                        "DecayCore automatic mode: adaptive target synthesized "
                        "from confidence-weighted bass evidence"
                    )
            return {
                "selected_hc_mode": str(AUTO_MODE_SYNTH_TARGET_NAME),
                "fit_rms_db": _auto_safe_float(
                    diagnostics.get("fit_rms_db"),
                    0.0,
                ),
                "offset_db": 0.0,
                "selection_method": "adaptive_guarded",
                "selection_basis": str(rank_basis),
                "auto_goal": str(goal),
                "top_n": 0,
                "trials_per_curve": 0,
                "candidates": [],
                "evaluated": [],
                "best_preset": {},
                "adaptive_target_diagnostics": dict(diagnostics),
                "_synth_hc_f": synth_f,
                "_synth_hc_m": synth_m,
            }
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
    ) as exc:
        logger.warning("Adaptive target synthesis failed: %s: %s", type(exc).__name__, exc)
    return None


def _finalize_target_selection_result(
    *,
    setup: _TargetSelectionSetup,
    shortlist_state: _TargetShortlistState,
    evaluated: list[dict],
    status_cb,
) -> dict:
    if not evaluated:
        quick_out = dict(shortlist_state.quick or {})
        quick_out.setdefault("selection_method", "fit_rms")
        quick_out["selection_basis"] = str(setup.rank_basis)
        quick_out["auto_goal"] = str(setup.goal)
        return quick_out
    evaluated_sorted = sorted(evaluated, key=_auto_target_result_rank_key)
    rank_tie_eps = float(max(0.0, _auto_safe_float(AUTO_MODE_TARGET_BEST_RANK_TIE_EPS, 0.05)))
    target_scored = [
        {
            **dict(it or {}),
            "_auto_select_kind": "target_curve",
            "_target_rank_tie_eps": float(rank_tie_eps),
        }
        for it in evaluated_sorted
    ]
    winner = dict(_auto_select_best_scored(target_scored) or evaluated_sorted[0])
    selection_method = str(winner.pop("_auto_selection_method", "top3x10_trials") or "top3x10_trials")
    if selection_method == "top3x10_trials_rank_tie_composite":
        old_winner = dict(evaluated_sorted[0])
        logger.info(
            "Automatic mode target select: rank tie-break by avg/mode/boost (eps=%.3f) %s -> %s, avg_rank=%.3f -> %.3f, mode_ripple=%.4f -> %.4f, boost_penalty=%.3f -> %.3f",
            float(rank_tie_eps),
            str(old_winner.get("hc_mode", "n/a")),
            str(winner.get("hc_mode", "n/a")),
            _auto_safe_float(old_winner.get("avg_rank_score"), 0.0),
            _auto_safe_float(winner.get("avg_rank_score"), 0.0),
            _auto_target_result_mode_ripple(old_winner),
            _auto_target_result_mode_ripple(winner),
            _auto_safe_float(old_winner.get("boost_penalty", 0.0), 0.0),
            _auto_safe_float(winner.get("boost_penalty", 0.0), 0.0),
        )
    if bool(shortlist_state.cache_wildcard_participated) and bool(winner.get("from_cache_wildcard", False)):
        selection_method = "trial_with_cache_wildcard"
    winner_mode_ripple = _auto_target_result_mode_ripple(winner)
    logger.info(
        "Automatic mode target select: goal=%s, basis=%s, winner=%s, %s, avg_rank=%.3f, mode_ripple=%.4f, pre=%.3f, boost=%.3f, asym=%.3f, method=%s",
        str(setup.goal),
        str(setup.rank_basis),
        str(winner.get("hc_mode", "n/a")),
        _auto_metric_text(dict(winner.get("best_metrics", {}) or {}), setup.goal),
        _auto_safe_float(winner.get("avg_rank_score", 0.0), 0.0),
        float(winner_mode_ripple),
        _auto_safe_float(winner.get("preselect_score", winner.get("fit_rms_db", 1e9)), 1e9),
        _auto_safe_float(winner.get("boost_penalty", 0.0), 0.0),
        _auto_safe_float(winner.get("asym_penalty_db", 0.0), 0.0),
        str(selection_method),
    )
    if callable(status_cb):
        winner_best_metrics = dict(winner.get("best_metrics", {}) or {})
        status_cb(
            "DecayCore automatic mode: target finalize "
            f"(winner {winner.get('hc_mode', 'n/a')!s}, method {selection_method}, "
            f"rank {official_rank_score(winner_best_metrics):.3f}, "
            f"avg {_auto_safe_float(winner_best_metrics.get('avg_score', 0.0), 0.0):.3f}, "
            f"pre {_auto_safe_float(winner.get('preselect_score', winner.get('fit_rms_db', 1e9)), 1e9):.3f}, "
            f"fit {_auto_safe_float(winner.get('fit_rms_db', 0.0), 0.0):.3f} dB)"
        )
    return {
        "selected_hc_mode": str(winner.get("hc_mode", shortlist_state.quick.get("selected_hc_mode", "Harman6"))),
        "fit_rms_db": float(winner.get("fit_rms_db", shortlist_state.quick.get("fit_rms_db", 0.0))),
        "offset_db": float(winner.get("offset_db", shortlist_state.quick.get("offset_db", 0.0))),
        "selection_method": str(selection_method),
        "selection_basis": str(setup.rank_basis),
        "auto_goal": str(setup.goal),
        "top_n": int(len(shortlist_state.shortlisted)),
        "trials_per_curve": int(shortlist_state.trials_eff),
        "candidates": list(shortlist_state.shortlisted),
        "evaluated": list(evaluated_sorted),
        "best_preset": dict(winner.get("best_preset", {}) or {}),
        "best_metrics": dict(winner.get("best_metrics", {}) or {}),
    }


def _resolve_target_cache_seed(
    *,
    base_data: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    pin_obj,
    status_cb=None,
    top_n: int = AUTO_MODE_TARGET_TOP_N,
    trials_per_curve: int = AUTO_MODE_TARGET_TRIALS_PER_CURVE,
    runtime=None,
) -> _TargetSelectionContext:
    return _TargetSelectionContext(
        params={
            "base_data": dict(base_data or {}),
            "measurements": dict(measurements or {}),
            "fs_v": int(fs_v),
            "taps_v": int(taps_v),
            "xos": list(xos or []),
            "hpf": hpf,
            "pin_obj": pin_obj,
            "status_cb": status_cb,
            "top_n": int(top_n),
            "trials_per_curve": int(trials_per_curve),
            "runtime": runtime,
        }
    )


def _build_target_candidate_list(
    *,
    context: _TargetSelectionContext,
) -> _TargetSelectionContext:
    params = dict(context.params or {})
    setup = _prepare_target_selection_setup(
        base_data=dict(params.get("base_data", {}) or {}),
        measurements=dict(params.get("measurements", {}) or {}),
        fs_v=int(params.get("fs_v", 0) or 0),
        taps_v=int(params.get("taps_v", 0) or 0),
        xos=list(params.get("xos", []) or []),
        hpf=params.get("hpf"),
        runtime=params.get("runtime"),
    )
    params["_target_setup"] = setup
    adaptive_result = _try_adaptive_target_fast_path(
        base_data=dict(params.get("base_data", {}) or {}),
        measurements=dict(params.get("measurements", {}) or {}),
        rank_basis=setup.rank_basis,
        goal=setup.goal,
        status_cb=params.get("status_cb"),
    )
    if isinstance(adaptive_result, dict):
        params["_target_final_result"] = dict(adaptive_result)
        return _TargetSelectionContext(params=params)
    cache_state = _resolve_cached_target_state(
        setup=setup,
        base_data=dict(params.get("base_data", {}) or {}),
        measurements=dict(params.get("measurements", {}) or {}),
        fs_v=int(params.get("fs_v", 0) or 0),
        taps_v=int(params.get("taps_v", 0) or 0),
        xos=list(params.get("xos", []) or []),
        hpf=params.get("hpf"),
        status_cb=params.get("status_cb"),
    )
    params["_target_cache_state"] = cache_state
    cached_result = _try_measurement_global_target_result(
        setup=setup,
        cache_state=cache_state,
        base_data=dict(params.get("base_data", {}) or {}),
        measurements=dict(params.get("measurements", {}) or {}),
        status_cb=params.get("status_cb"),
    )
    if isinstance(cached_result, dict):
        params["_target_final_result"] = dict(cached_result)
        return _TargetSelectionContext(params=params)
    cached_result = _try_exact_cached_target_result(
        setup=setup,
        cache_state=cache_state,
        base_data=dict(params.get("base_data", {}) or {}),
        measurements=dict(params.get("measurements", {}) or {}),
        status_cb=params.get("status_cb"),
    )
    if isinstance(cached_result, dict):
        params["_target_final_result"] = dict(cached_result)
        return _TargetSelectionContext(params=params)
    shortlist_state = _load_quick_target_selection(
        setup=setup,
        cache_state=cache_state,
        base_data=dict(params.get("base_data", {}) or {}),
        measurements=dict(params.get("measurements", {}) or {}),
        top_n=int(params.get("top_n", AUTO_MODE_TARGET_TOP_N) or AUTO_MODE_TARGET_TOP_N),
        trials_per_curve=int(
            params.get("trials_per_curve", AUTO_MODE_TARGET_TRIALS_PER_CURVE)
            or AUTO_MODE_TARGET_TRIALS_PER_CURVE
        ),
        status_cb=params.get("status_cb"),
    )
    if isinstance(shortlist_state, dict):
        params["_target_final_result"] = dict(shortlist_state)
        return _TargetSelectionContext(params=params)
    if shortlist_state is None:
        params["_target_final_result"] = None
        return _TargetSelectionContext(params=params)
    shortlist_state = _apply_target_shortlist_modifiers(
        setup=setup,
        cache_state=cache_state,
        shortlist_state=shortlist_state,
        base_data=dict(params.get("base_data", {}) or {}),
        measurements=dict(params.get("measurements", {}) or {}),
        status_cb=params.get("status_cb"),
    )
    if isinstance(shortlist_state, dict):
        params["_target_final_result"] = dict(shortlist_state)
        return _TargetSelectionContext(params=params)
    if shortlist_state is None:
        params["_target_final_result"] = None
        return _TargetSelectionContext(params=params)
    params["_target_shortlist_state"] = shortlist_state
    return _TargetSelectionContext(params=params)


def _run_target_shortlist_trials(
    *,
    context: _TargetSelectionContext,
) -> _TargetSelectionOutcome:
    params = dict(context.params or {})
    result = params.get("_target_final_result")
    if "_target_final_result" not in params:
        setup = params.get("_target_setup")
        shortlist_state = params.get("_target_shortlist_state")
        evaluated = _evaluate_target_shortlist_core(
            setup=setup,
            shortlist_state=shortlist_state,
            base_data=dict(params.get("base_data", {}) or {}),
            measurements=dict(params.get("measurements", {}) or {}),
            fs_v=int(params.get("fs_v", 0) or 0),
            taps_v=int(params.get("taps_v", 0) or 0),
            xos=list(params.get("xos", []) or []),
            hpf=params.get("hpf"),
            pin_obj=params.get("pin_obj"),
            status_cb=params.get("status_cb"),
        )
        result = _finalize_target_selection_result(
            setup=setup,
            shortlist_state=shortlist_state,
            evaluated=list(evaluated or []),
            status_cb=params.get("status_cb"),
        )
    result_dict = dict(result or {}) if isinstance(result, dict) else None
    return _TargetSelectionOutcome(
        result=result_dict,
        candidates=list((result_dict or {}).get("candidates", []) or []),
        evaluated=list((result_dict or {}).get("evaluated", []) or []),
    )


def _choose_target_winner(
    *,
    outcome: _TargetSelectionOutcome,
) -> _TargetSelectionOutcome:
    result_dict = dict(outcome.result or {}) if isinstance(outcome.result, dict) else None
    winner = None
    if isinstance(result_dict, dict):
        winner = {
            "hc_mode": str(result_dict.get("selected_hc_mode", "") or "").strip(),
            "best_preset": dict(result_dict.get("best_preset", {}) or {}),
        }
    return _TargetSelectionOutcome(
        result=result_dict,
        candidates=list(outcome.candidates or []),
        evaluated=list(outcome.evaluated or []),
        winner=winner,
    )


def _finalize_target_selection_metadata(
    *,
    outcome: _TargetSelectionOutcome,
) -> dict | None:
    if not isinstance(outcome.result, dict):
        return None
    return dict(outcome.result or {})


def _select_target_curve_with_trials_impl(
    *,
    base_data: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    pin_obj,
    status_cb=None,
    top_n: int = AUTO_MODE_TARGET_TOP_N,
    trials_per_curve: int = AUTO_MODE_TARGET_TRIALS_PER_CURVE,
    runtime=None,
) -> dict | None:
    context = _resolve_target_cache_seed(
        base_data=base_data,
        measurements=measurements,
        fs_v=int(fs_v),
        taps_v=int(taps_v),
        xos=xos,
        hpf=hpf,
        pin_obj=pin_obj,
        status_cb=status_cb,
        top_n=int(top_n),
        trials_per_curve=int(trials_per_curve),
        runtime=runtime,
    )
    shortlisted = _build_target_candidate_list(context=context)
    evaluated = _run_target_shortlist_trials(context=shortlisted)
    winner = _choose_target_winner(outcome=evaluated)
    return _finalize_target_selection_metadata(outcome=winner)
