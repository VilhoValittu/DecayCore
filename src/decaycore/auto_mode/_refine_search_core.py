# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Search-refine phase1/2/3 core logic."""

from __future__ import annotations

import logging

import numpy as np

from .candidate_generation import (
    _build_auto_mode_candidates,
    _build_auto_mode_candidates_local,
    _seed_auto_mode_candidate_local_optuna_params,
    _seed_auto_mode_candidate_micro_optuna_params,
    _seed_auto_mode_candidate_optuna_params,
    _suggest_auto_mode_candidate_local_optuna,
    _suggest_auto_mode_candidate_micro_optuna,
)
from .rank_score import official_rank_score
from .refine_eval import RefineEvalContext, run_candidate_phase
from .scoring_ranking import (
    _auto_adaptive_shrink_factor,
    _auto_build_refine_profile,
    _auto_goal_uses_local_refine,
    _auto_rank_key,
    _auto_rank_value,
    _auto_select_best_scored,
)
from .shared import AUTO_MODE_ADAPTIVE_SHRINK_MIN, _auto_safe_float
from ._refine_types import _SearchPhase1State, _SearchPhase2State

logger = logging.getLogger("DecayCore")


def _build_search_refine_eval_context(
    *,
    search_base_data: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    hc_f,
    hc_m,
    pin_obj,
    status_cb,
    cfg,
    goal: str,
    filter_key: str,
    optimizer_backend: str,
    optuna_mod,
    seed: int,
    optuna_search_sig: str,
    status_prefix: str,
    winner_target_name: str | None,
    search_state,
    runtime,
) -> RefineEvalContext:
    return RefineEvalContext(
        search_base_data=dict(search_base_data or {}),
        measurements=dict(measurements or {}),
        fs_v=int(fs_v),
        taps_v=int(taps_v),
        xos=list(xos or []),
        hpf=hpf,
        hc_f=hc_f,
        hc_m=hc_m,
        pin_obj=pin_obj,
        cfg=cfg,
        goal=str(goal),
        filter_key=str(filter_key),
        optimizer_backend=str(optimizer_backend),
        optuna_mod=optuna_mod,
        seed=int(seed),
        optuna_search_sig=str(optuna_search_sig),
        status_cb=status_cb,
        status_prefix=str(status_prefix),
        winner_target_name=winner_target_name,
        search_state=search_state,
        runtime=runtime,
    )


def _run_search_refine_phase1_core(
    *,
    search_base_data: dict,
    candidates: list[dict],
    prior_seed_preset: dict | None,
    use_optuna_trials: bool,
    cfg,
    filter_key: str,
    seed: int,
    n_trials_eff: int,
    runtime,
    status_cb,
    ctx: RefineEvalContext,
) -> _SearchPhase1State:
    phase1_seed_presets = []
    if isinstance(prior_seed_preset, dict) and prior_seed_preset:
        phase1_seed_presets.append(dict(prior_seed_preset))
        logger.info(
            "Automatic mode: loaded built-in prior seed preset for %s filter.",
            str(filter_key),
        )
    phase1_seed_presets.extend(
        _build_auto_mode_candidates(
            search_base_data,
            n_trials=1,
            seed=int(seed),
        )
    )
    phase1_stats = run_candidate_phase(
        ctx,
        candidates,
        phase_label="phase 1/2",
        phase_kind="phase1",
        plateau_after_no_improve=int(cfg.phase1_plateau_rounds),
        use_refine_tiebreak=False,
        n_total_override=int(n_trials_eff),
        seed_presets=list(phase1_seed_presets or []),
        optuna_builder=(
            (lambda tr, _base=dict(search_base_data): runtime.suggest_auto_mode_candidate_optuna(_base, tr))
            if bool(use_optuna_trials)
            else None
        ),
        seed_to_params=(
            (lambda preset, _base=dict(search_base_data): _seed_auto_mode_candidate_optuna_params(_base, preset))
            if bool(use_optuna_trials)
            else None
        ),
        study_scope="phase1",
    )
    phase1_entries = [
        dict(it)
        for it in list(ctx.search_state.scored)
        if str(dict(it.get("metrics", {}) or {}).get("phase", "")) == "phase 1/2"
    ]
    phase1_top = sorted(
        phase1_entries,
        key=lambda x: _auto_rank_key(x.get("metrics", {})),
    )[: int(max(1, cfg.local_refine_top_k))]
    if phase1_top:
        phase1_top_best = dict(_auto_select_best_scored(phase1_top, goal=ctx.goal) or phase1_top[0])
        p1m = dict(phase1_top_best.get("metrics", {}) or {})
        p1p = dict(phase1_top_best.get("preset", {}) or {})
        p1_mixed = _auto_safe_float(p1p.get("mixed_freq", search_base_data.get("mixed_freq", float("nan"))), float("nan"))
        p1_phase = _auto_safe_float(p1p.get("phase_limit", search_base_data.get("phase_limit", float("nan"))), float("nan"))
        p1_tdc = _auto_safe_float(p1p.get("tdc_strength", search_base_data.get("tdc_strength", float("nan"))), float("nan"))
        p1_mode = _auto_safe_float(p1m.get("mode_ripple_db"), float("nan"))
        p1_boost = _auto_safe_float(p1m.get("max_net_boost_db"), float("nan"))
        p1_mode_txt = f"{p1_mode:.3f} dB" if np.isfinite(p1_mode) else "n/a"
        p1_boost_txt = f"{p1_boost:.2f} dB" if np.isfinite(p1_boost) else "n/a"
        if str(ctx.filter_key) == "mixed":
            p1_detail = f"mixed_freq={p1_mixed:.1f} Hz, tdc={p1_tdc:.1f}"
        elif str(ctx.filter_key) in ("linear", "asym"):
            p1_detail = (
                f"phase_limit={p1_phase:.1f} Hz, tdc={p1_tdc:.1f}"
                if np.isfinite(p1_phase)
                else f"phase_limit=n/a, tdc={p1_tdc:.1f}"
            )
        else:
            p1_detail = f"tdc={p1_tdc:.1f}"
        phase1_optuna_tel = dict(phase1_stats.get("optuna_telemetry", {}) or {})
        p1_optuna_txt = runtime.auto_optuna_telemetry_text(phase1_optuna_tel)
        p1_status_suffix = f", {p1_optuna_txt}" if p1_optuna_txt else ""
        logger.info(
            "Automatic mode Phase1 done: avg_score=%.3f, %s%s",
            _auto_safe_float(p1m.get("avg_score"), 0.0),
            str(p1_detail),
            str(p1_status_suffix),
        )
        if callable(status_cb):
            status_cb(
                "DecayCore automatic mode: Phase1 done "
                f"rank={official_rank_score(p1m):.3f}, "
                f"avg_score={_auto_safe_float(p1m.get('avg_score'), 0.0):.3f}, "
                f"mode_ripple={p1_mode_txt}, "
                f"boost={p1_boost_txt}, "
                f"{p1_detail}{p1_status_suffix}"
            )
    phase1_best_item = _auto_select_best_scored(phase1_top, goal=ctx.goal) if phase1_top else None
    return _SearchPhase1State(
        ctx=ctx,
        phase1_ok=int(phase1_stats.get("ok", 0) or 0),
        phase1_tried=int(phase1_stats.get("tried", 0) or 0),
        phase1_plateau_hit=bool(phase1_stats.get("plateau_hit", False)),
        phase1_optuna_tel=dict(phase1_stats.get("optuna_telemetry", {}) or {}),
        phase1_top=list(phase1_top or []),
        phase1_best_metrics=(dict((phase1_best_item or {}).get("metrics", {}) or {}) if phase1_best_item else None),
        phase1_best_preset=(dict((phase1_best_item or {}).get("preset", {}) or {}) if phase1_best_item else None),
    )


def _maybe_revalidate_seed_winner(
    *,
    cfg,
    search_state,
    phase1: _SearchPhase1State,
    winner_target_name: str | None,
) -> None:
    # If a cached seed set search_state.best_metrics to a high score but phase 1 found
    # nothing close to it, the seed score is likely stale (from a different run context).
    # A stale seed causes Optuna's constraint baseline to be unreachably high, resulting
    # in feas=0/N for all phase 2 and micro refine trials. Fix: when the gap between
    # the seed score and phase1 best exceeds the threshold, swap the winner to phase1 best
    # so that phase 2 and micro refine get a realistic constraint baseline.
    if not isinstance(phase1.phase1_best_metrics, dict):
        return
    if search_state.best_metrics is None:
        return
    seed_rank = _auto_rank_value(search_state.best_metrics)
    phase1_rank = _auto_rank_value(phase1.phase1_best_metrics)
    revalidate_threshold = float(_auto_safe_float(getattr(cfg, "seed_revalidate_rank_gap", 3.0), 3.0))
    if seed_rank - phase1_rank > revalidate_threshold:
        from .search_state import _auto_set_search_winner
        prev_best = dict(search_state.best_metrics)
        _auto_set_search_winner(
            search_state,
            phase1.phase1_best_metrics,
            phase1.phase1_best_preset or {},
            prev_metrics=prev_best,
            phase_label="seed revalidation (phase1 fallback)",
            target_name=winner_target_name,
        )
        logger.info(
            "Automatic mode: seed winner revalidated — seed rank %.3f far above phase1 %.3f "
            "(gap %.3f > threshold %.3f), switching to phase1 best for phase2 baseline",
            float(seed_rank),
            float(phase1_rank),
            float(seed_rank - phase1_rank),
            float(revalidate_threshold),
        )


def _run_search_refine_phase2_local_core(
    *,
    search_base_data: dict,
    cfg,
    goal: str,
    filter_key: str,
    seed: int,
    winner_target_name: str | None,
    use_optuna_trials: bool,
    status_cb,
    search_state,
    runtime,
    phase1: _SearchPhase1State,
) -> _SearchPhase2State:
    phase2 = _SearchPhase2State()
    if not (bool(cfg.local_refine_enabled) and phase1.phase1_top and _auto_goal_uses_local_refine(goal)):
        return phase2
    _maybe_revalidate_seed_winner(
        cfg=cfg,
        search_state=search_state,
        phase1=phase1,
        winner_target_name=winner_target_name,
    )
    ref_profile = _auto_build_refine_profile(
        base_data=search_base_data,
        phase1_top=phase1.phase1_top,
    )
    phase2.phase2_focus_lo = float(_auto_safe_float(ref_profile.get("focus_lo", float("nan")), float("nan")))
    phase2.phase2_focus_hi = float(_auto_safe_float(ref_profile.get("focus_hi", float("nan")), float("nan")))
    for ci, item in enumerate(phase1.phase1_top, start=1):
        center = dict(item.get("preset", {}) or {})
        c_mixed = _auto_safe_float(center.get("mixed_freq", search_base_data.get("mixed_freq", float("nan"))), float("nan"))
        c_phase = _auto_safe_float(center.get("phase_limit", search_base_data.get("phase_limit", float("nan"))), float("nan"))
        local_detail = None
        if str(filter_key) == "mixed":
            local_detail = f"mixed_freq={c_mixed:.1f} Hz"
        elif str(filter_key) in ("linear", "asym"):
            local_detail = (
                f"phase refine phase_limit={c_phase:.1f} Hz"
                if np.isfinite(c_phase)
                else "phase refine phase_limit=n/a"
            )
        if local_detail is not None:
            logger.info("Automatic mode Local refine: center #%d %s", int(ci), str(local_detail))
            if callable(status_cb):
                status_cb(
                    "DecayCore automatic mode: Local refine "
                    f"center #{ci} {local_detail}"
                )
        local_seed = int(seed + 7919 + ci * 100003)
        local_shrink = float(
            _auto_adaptive_shrink_factor(
                phase1.phase1_top,
                base_shrink=float(cfg.local_refine_shrink),
                plateau_hit=bool(phase1.phase1_plateau_hit),
            )
        )
        local_candidates = []
        local_seed_presets = _build_auto_mode_candidates_local(
            search_base_data,
            center,
            1,
            int(local_seed),
            shrink=float(local_shrink),
        )
        if not bool(use_optuna_trials):
            local_candidates = _build_auto_mode_candidates_local(
                search_base_data,
                center,
                int(cfg.local_refine_trials_per_top),
                int(local_seed),
                shrink=float(local_shrink),
            )
        before = dict(search_state.best_metrics or {})
        stats = run_candidate_phase(
            phase1.ctx,
            local_candidates,
            phase_label=f"phase 2/2 local center#{ci}",
            phase_kind="local",
            plateau_after_no_improve=0,
            use_refine_tiebreak=True,
            focus_lo_hz=float(phase2.phase2_focus_lo) if np.isfinite(phase2.phase2_focus_lo) else None,
            focus_hi_hz=float(phase2.phase2_focus_hi) if np.isfinite(phase2.phase2_focus_hi) else None,
            n_total_override=int(cfg.local_refine_trials_per_top),
            seed_presets=list(local_seed_presets or []),
            optuna_builder=(
                (
                    lambda tr,
                    _base=dict(search_base_data),
                    _center=dict(center),
                    _shrink=float(local_shrink): _suggest_auto_mode_candidate_local_optuna(
                        _base,
                        _center,
                        tr,
                        shrink=float(_shrink),
                    )
                )
                if bool(use_optuna_trials)
                else None
            ),
            seed_to_params=(
                (
                    lambda preset,
                    _base=dict(search_base_data),
                    _center=dict(center),
                    _shrink=float(local_shrink): _seed_auto_mode_candidate_local_optuna_params(
                        _base,
                        _center,
                        preset,
                        shrink=float(_shrink),
                    )
                )
                if bool(use_optuna_trials)
                else None
            ),
            study_scope=runtime.auto_optuna_scope_with_context(
                f"phase2-local-center-{int(ci)}-u1",
                center=dict(center or {}),
                shrink=float(local_shrink),
                extra={
                    "filter_key": str(filter_key),
                    "target": str(winner_target_name or ""),
                },
            ),
        )
        phase2.phase2_ok += int(stats.get("ok", 0) or 0)
        phase2.phase2_tried += int(stats.get("tried", 0) or 0)
        local_tel = dict(stats.get("optuna_telemetry", {}) or {})
        if local_tel:
            phase2.phase2_local_optuna_tels.append(
                {
                    "center_index": int(ci),
                    "phase_label": f"phase 2/2 local center#{ci}",
                    "telemetry": dict(local_tel),
                }
            )
        local_tel_txt = runtime.auto_optuna_telemetry_text(local_tel)
        local_rescue_suffix = ", zero-feasible fallback used" if bool(stats.get("optuna_zero_feasible_fallback_used", False)) else ""
        local_fallback_txt = runtime.auto_optuna_fallback_summary_text(local_tel) if bool(stats.get("optuna_zero_feasible_fallback_used", False)) else ""
        local_best_metrics = dict(search_state.best_metrics or {})
        local_rank_txt = runtime.auto_optuna_fmt_value(official_rank_score(local_best_metrics), 3)
        local_avg_txt = runtime.auto_optuna_fmt_value(local_best_metrics.get("avg_score"), 3)
        logger.info(
            "Automatic mode Local refine summary: center #%d, current_best_rank=%s, avg=%s%s%s",
            int(ci),
            str(local_rank_txt),
            str(local_avg_txt),
            "" if not local_tel_txt else f", {local_tel_txt}",
            str(local_rescue_suffix),
        )
        if callable(status_cb):
            status_cb(
                "DecayCore automatic mode: Local refine summary "
                f"center #{int(ci)}, current_best_rank={local_rank_txt}, avg_score={local_avg_txt}"
                f"{'' if not local_tel_txt else f', {local_tel_txt}'}"
                f"{local_rescue_suffix}"
            )
        if local_fallback_txt:
            logger.info(
                "Automatic mode Local refine fallback detail: center #%d, %s",
                int(ci),
                str(local_fallback_txt),
            )
            if callable(status_cb):
                status_cb(
                    "DecayCore automatic mode: Local refine fallback "
                    f"center #{int(ci)}, {local_fallback_txt}"
                )
        if bool(stats.get("improved_any", False)):
            logger.info(
                "Automatic mode Local refine winner improved: avg_score %.3f -> %.3f, rank_score %.3f -> %.3f",
                _auto_safe_float(before.get("avg_score"), 0.0),
                _auto_safe_float((search_state.best_metrics or {}).get("avg_score"), 0.0),
                _auto_safe_float(before.get("rank_score"), 0.0),
                _auto_safe_float((search_state.best_metrics or {}).get("rank_score"), 0.0),
            )
    return phase2


def _carry_forward_phase1_best_core(
    *,
    cfg,
    winner_target_name: str | None,
    search_state,
    phase1: _SearchPhase1State,
) -> None:
    if not (bool(cfg.local_refine_keep_best_phase1) and isinstance(phase1.phase1_best_metrics, dict)):
        return
    if search_state.best_metrics is not None and _auto_rank_key(search_state.best_metrics) <= _auto_rank_key(phase1.phase1_best_metrics):
        return
    from .search_state import _auto_set_search_winner

    prev_best = dict(search_state.best_metrics or {})
    _auto_set_search_winner(
        search_state,
        phase1.phase1_best_metrics,
        phase1.phase1_best_preset or {},
        prev_metrics=prev_best,
        phase_label="phase 1 carry-forward",
        target_name=winner_target_name,
    )


def _run_search_refine_micro_core(
    *,
    search_base_data: dict,
    cfg,
    goal: str,
    filter_key: str,
    winner_target_name: str | None,
    use_optuna_trials: bool,
    status_cb,
    search_state,
    runtime,
    phase1: _SearchPhase1State,
    phase2: _SearchPhase2State,
) -> _SearchPhase2State:
    if not (
        bool(cfg.phase3_micro_enabled)
        and _auto_goal_uses_local_refine(goal)
        and isinstance(search_state.best_preset, dict)
        and bool(search_state.best_preset)
    ):
        return phase2
    micro_shrink = float(
        _auto_adaptive_shrink_factor(
            phase1.phase1_top,
            base_shrink=float(cfg.adaptive_shrink_max),
            plateau_hit=bool(phase1.phase1_plateau_hit),
        )
    )
    micro_shrink = float(np.clip(micro_shrink * 0.70, AUTO_MODE_ADAPTIVE_SHRINK_MIN, 1.0))
    micro_center = dict(search_state.best_preset or {})
    micro_candidates = []
    micro_seed_presets = runtime.build_auto_mode_candidates_micro(
        search_base_data,
        dict(micro_center),
        n_trials=1,
        shrink=float(micro_shrink),
    )
    if not bool(use_optuna_trials):
        micro_candidates = runtime.build_auto_mode_candidates_micro(
            search_base_data,
            dict(micro_center),
            n_trials=int(cfg.phase3_micro_trials),
            shrink=float(micro_shrink),
        )
    logger.info(
        "Micro refine size: %d%s",
        int(cfg.phase3_micro_trials),
        " (optuna)" if bool(use_optuna_trials) else "",
    )
    if callable(status_cb):
        status_cb(
            f"DecayCore automatic mode: micro refine {int(cfg.phase3_micro_trials)} trials around current best"
        )
    before_micro = dict(search_state.best_metrics or {})
    micro_stats = run_candidate_phase(
        phase1.ctx,
        micro_candidates,
        phase_label="micro refine",
        phase_kind="micro",
        plateau_after_no_improve=0,
        use_refine_tiebreak=True,
        focus_lo_hz=float(phase2.phase2_focus_lo) if np.isfinite(phase2.phase2_focus_lo) else None,
        focus_hi_hz=float(phase2.phase2_focus_hi) if np.isfinite(phase2.phase2_focus_hi) else None,
        n_total_override=int(cfg.phase3_micro_trials),
        seed_presets=list(micro_seed_presets or []),
        optuna_builder=(
            (
                lambda tr,
                _base=dict(search_base_data),
                _center=dict(micro_center),
                _shrink=float(micro_shrink): _suggest_auto_mode_candidate_micro_optuna(
                    _base,
                    _center,
                    tr,
                    shrink=float(_shrink),
                )
            )
            if bool(use_optuna_trials)
            else None
        ),
        seed_to_params=(
            (
                lambda preset,
                _base=dict(search_base_data),
                _center=dict(micro_center),
                _shrink=float(micro_shrink): _seed_auto_mode_candidate_micro_optuna_params(
                    _base,
                    _center,
                    preset,
                    shrink=float(_shrink),
                )
            )
            if bool(use_optuna_trials)
            else None
        ),
        study_scope=runtime.auto_optuna_scope_with_context(
            "phase3-micro-u1",
            center=dict(micro_center or {}),
            shrink=float(micro_shrink),
            extra={
                "filter_key": str(filter_key),
                "target": str(winner_target_name or ""),
            },
        ),
    )
    phase2.phase2_ok += int(micro_stats.get("ok", 0) or 0)
    phase2.phase2_tried += int(micro_stats.get("tried", 0) or 0)
    phase2.phase3_micro_optuna_tel = dict(micro_stats.get("optuna_telemetry", {}) or {})
    if bool(micro_stats.get("improved_any", False)):
        logger.info(
            "Automatic mode micro refine improved: avg_score %.3f -> %.3f, rank_score %.3f -> %.3f",
            _auto_safe_float(before_micro.get("avg_score"), 0.0),
            _auto_safe_float((search_state.best_metrics or {}).get("avg_score"), 0.0),
            _auto_safe_float(before_micro.get("rank_score"), 0.0),
            _auto_safe_float((search_state.best_metrics or {}).get("rank_score"), 0.0),
        )
    micro_tel_txt = runtime.auto_optuna_telemetry_text(phase2.phase3_micro_optuna_tel)
    micro_rescue_suffix = ", zero-feasible fallback used" if bool(micro_stats.get("optuna_zero_feasible_fallback_used", False)) else ""
    micro_fallback_txt = runtime.auto_optuna_fallback_summary_text(phase2.phase3_micro_optuna_tel) if bool(micro_stats.get("optuna_zero_feasible_fallback_used", False)) else ""
    micro_best_metrics = dict(search_state.best_metrics or {})
    micro_rank_txt = runtime.auto_optuna_fmt_value(official_rank_score(micro_best_metrics), 3)
    micro_avg_txt = runtime.auto_optuna_fmt_value(micro_best_metrics.get("avg_score"), 3)
    logger.info(
        "Automatic mode micro refine summary: current_best_rank=%s, avg=%s%s%s",
        str(micro_rank_txt),
        str(micro_avg_txt),
        "" if not micro_tel_txt else f", {micro_tel_txt}",
        str(micro_rescue_suffix),
    )
    if callable(status_cb):
        status_cb(
            "DecayCore automatic mode: micro refine summary "
            f"current_best_rank={micro_rank_txt}, avg_score={micro_avg_txt}"
            f"{'' if not micro_tel_txt else f', {micro_tel_txt}'}"
            f"{micro_rescue_suffix}"
        )
    if micro_fallback_txt:
        logger.info("Automatic mode micro refine fallback detail: %s", str(micro_fallback_txt))
        if callable(status_cb):
            status_cb(
                "DecayCore automatic mode: micro refine fallback "
                f"{micro_fallback_txt}"
            )
    return phase2


