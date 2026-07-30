# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Finalize-stage run orchestration for automatic mode.

Contains:
- _run_phase2_pareto_finalize
- finalize_search_result  (public API)

_finalize_cached_result lives in orchestrator_finalize_cached.py.
"""

from __future__ import annotations

import logging

import numpy as np

from .runtime_context import coerce_orchestrator_runtime
from .scoring_ranking import (
    _auto_mode_ripple_for_pareto,
    _auto_phase2_hard_gate_pool,
    _auto_phase2_pareto_front,
    _auto_prepost_for_pareto,
    _auto_prepost_lr_for_pareto,
    _auto_rank_key,
    _auto_select_best_scored,
    _auto_target_tracking_for_pareto,
)
from .shared_parts import _auto_safe_float, _m

from .orchestrator_finalize_cache_parts import (
    _build_phase2_pareto_status,
    _stereo_refine_materialize_base_data,
    _resolve_winner_auto_exc_hz,
)
from .orchestrator_finalize_result import (
    _materialize_final_search_winner,
    _run_p6_final_validation,
    _build_final_search_result,
)
from .orchestrator_finalize_polish import (
    _apply_search_winner_polish_pipeline,
    _save_final_search_cache,
)
from .orchestrator_finalize_cached import _finalize_cached_result

logger = logging.getLogger("DecayCore")

__all__ = [
    "finalize_search_result",
    "_run_phase2_pareto_finalize",
    "_finalize_cached_result",
]


def _build_phase2_pool_with_phase1_injection(*, search_state, cfg) -> list[dict]:
    phase2_pool_raw = [dict(it or {}) for it in (search_state.phase2_pool or []) if isinstance(it, dict)]
    phase1_top_for_pool = sorted(
        [
            dict(it)
            for it in list(search_state.scored or [])
            if str(dict(it.get("metrics", {}) or {}).get("phase", "")) == "phase 1/2"
        ],
        key=lambda x: _auto_rank_key(x.get("metrics", {})),
    )[: int(max(1, getattr(cfg, "local_refine_top_k", 3)))]
    if phase1_top_for_pool:
        existing_preset_sigs = {str(sorted((dict(it.get("preset", {}) or {})).items())) for it in phase2_pool_raw}
        p1_injected = 0
        for p1_item in phase1_top_for_pool:
            p1_sig = str(sorted((dict(p1_item.get("preset", {}) or {})).items()))
            if p1_sig in existing_preset_sigs:
                continue
            phase2_pool_raw.append(dict(p1_item))
            existing_preset_sigs.add(p1_sig)
            p1_injected += 1
        if p1_injected:
            logger.info(
                "Phase2 pool: injected %d phase1 top candidates (total pool size now %d)",
                int(p1_injected),
                int(len(phase2_pool_raw)),
            )
    return list(phase2_pool_raw)


def _build_phase2_rank_window_kept(*, phase2_pool_raw: list[dict], cfg) -> list[dict]:
    phase2_rank_vals = [_m(dict(it.get("metrics", {}) or {}), "rank_score", float("nan")) for it in phase2_pool_raw]
    phase2_rank_vals = [float(v) for v in phase2_rank_vals if np.isfinite(v)]
    phase2_best_rank = max(phase2_rank_vals) if phase2_rank_vals else float("nan")
    rank_win = float(max(0.0, _auto_safe_float(cfg.phase2_pareto_rank_window, 2.0)))
    phase2_kept = []
    for it in phase2_pool_raw:
        r = _m(dict(it.get("metrics", {}) or {}), "rank_score", float("nan"))
        if np.isfinite(phase2_best_rank):
            if np.isfinite(r) and float(r) >= float(phase2_best_rank) - float(rank_win):
                phase2_kept.append(dict(it))
        else:
            phase2_kept.append(dict(it))
    phase2_kept = sorted(
        phase2_kept,
        key=lambda it: (
            -_m(dict(it.get("metrics", {}) or {}), "rank_score", float("-inf")),
            _auto_rank_key(dict(it.get("metrics", {}) or {})),
        ),
    )[: int(max(1, cfg.phase2_pareto_pool_max))]
    logger.info(
        "Phase2 pool size: %d (kept %d)",
        int(len(phase2_pool_raw)),
        int(len(phase2_kept)),
    )
    return list(phase2_kept)


def _apply_phase2_hard_gate_if_enabled(*, phase2_kept: list[dict], cfg) -> list[dict]:
    if not (bool(cfg.phase2_hard_gate_enabled) and len(phase2_kept) >= int(max(3, cfg.phase2_hard_gate_min_keep))):
        return list(phase2_kept)
    pre_n = int(len(phase2_kept))
    phase2_kept, ev_thr, rp_thr, pk_thr = _auto_phase2_hard_gate_pool(
        phase2_kept,
        min_keep=int(cfg.phase2_hard_gate_min_keep),
        keep_event_fraction=float(cfg.phase2_hard_gate_keep_event_fraction),
        keep_ripple_fraction=float(cfg.phase2_hard_gate_keep_ripple_fraction),
        keep_peak_fraction=float(cfg.phase2_hard_gate_keep_peak_fraction),
        abs_max_peak_db=float(cfg.phase2_hard_gate_abs_max_peak_db),
        fallback_to_rank=bool(cfg.phase2_hard_gate_fallback_to_rank),
    )
    post_n = int(len(phase2_kept))
    logger.info(
        "Phase2 hard-gate: kept %d/%d (event<=%.3f, ripple<=%.3f, residual_peak<=%.3f)",
        int(post_n),
        int(pre_n),
        float(ev_thr) if np.isfinite(ev_thr) else float("nan"),
        float(rp_thr) if np.isfinite(rp_thr) else float("nan"),
        float(pk_thr) if np.isfinite(pk_thr) else float("nan"),
    )
    if any(bool((it or {}).get("unsafe_fallback", False)) for it in phase2_kept):
        _unsafe = next(it for it in phase2_kept if bool((it or {}).get("unsafe_fallback", False)))
        logger.error(
            "Phase2 hard-gate: ALL candidates failed the absolute residual peak gate "
            "(threshold=%.2f dB, policy=%s). Returning unsafe fallback with residual_peak=%.2f dB. "
            "Consider relaxing constraints or re-measuring.",
            float((_unsafe or {}).get("unsafe_fallback_abs_max_peak_db", float("nan"))),
            str((_unsafe or {}).get("unsafe_fallback_abs_gate_policy", "phase2_absolute_default")),
            float((_unsafe or {}).get("unsafe_fallback_residual_peak_db", float("nan"))),
        )
    return list(phase2_kept)


def _resolve_pareto_winner(*, phase2_kept: list[dict], cfg, goal: str) -> tuple[dict, dict | None]:
    front = _auto_phase2_pareto_front(phase2_kept)
    logger.info("Pareto front size: %d", int(len(front)))
    rank_best = dict(_auto_select_best_scored(phase2_kept, goal=goal) or phase2_kept[0])
    pareto_pool = [
        {
            **dict(it or {}),
            "_auto_select_kind": "phase2_pareto",
            "_phase2_pareto_acoustic_drop": float(_auto_safe_float(cfg.phase2_pareto_acoustic_drop, 0.35)),
        }
        for it in phase2_kept
    ]
    pareto_winner = _auto_select_best_scored(pareto_pool, goal=goal)
    if not isinstance(pareto_winner, dict):
        return rank_best, None
    return rank_best, dict(pareto_winner or {})


def _log_pareto_winner_comparison(*, rank_best: dict, pareto_winner: dict) -> tuple[dict, dict]:
    w_metrics = dict(pareto_winner.get("metrics", {}) or {})
    w_mode_ripple = _auto_mode_ripple_for_pareto(w_metrics)
    w_target_tracking = _auto_target_tracking_for_pareto(w_metrics)
    w_pre_l, w_pre_r, w_prepost = _auto_prepost_lr_for_pareto(w_metrics)
    w_boost = _m(w_metrics, "max_net_boost_db", float("nan"))
    logger.info(
        "Pareto objectives include prepost: L=%.4f R=%.4f -> max=%.4f",
        float(w_pre_l) if np.isfinite(w_pre_l) else float("nan"),
        float(w_pre_r) if np.isfinite(w_pre_r) else float("nan"),
        float(w_prepost) if np.isfinite(w_prepost) else float("nan"),
    )
    logger.info(
        "Pareto winner: avg=%.3f, prepost=%.4f, mode_ripple=%.3f, target_tracking=%.3f, net_boost=%.3f",
        _m(w_metrics, "avg_score", 0.0),
        w_prepost if np.isfinite(w_prepost) else float("nan"),
        w_mode_ripple if np.isfinite(w_mode_ripple) else float("nan"),
        w_target_tracking if np.isfinite(w_target_tracking) else float("nan"),
        w_boost if np.isfinite(w_boost) else float("nan"),
    )
    logger.info(
        "Pareto anti-overfit score terms: avg=%.3f, broad_peak_pen=%.3f, sharpness_pen=%.3f, dip_fill_pen=%.3f, lr_overfit_pen=%.3f, rank=%.3f",
        _m(w_metrics, "avg_score", 0.0),
        _m(w_metrics, "residual_peak_penalty", 0.0),
        _m(w_metrics, "correction_sharpness_penalty", 0.0),
        _m(w_metrics, "dip_fill_risk_penalty", 0.0),
        _m(w_metrics, "channel_overfit_penalty", 0.0),
        _m(w_metrics, "rank_score", 0.0),
    )
    rb_metrics = dict(rank_best.get("metrics", {}) or {})
    rb_prepost = _auto_prepost_for_pareto(rb_metrics)
    rb_mode_ripple = _auto_mode_ripple_for_pareto(rb_metrics)
    logger.info(
        "Best-by-rank would have been: avg=%.3f, prepost=%.4f, mode_ripple=%.3f, target_tracking=%.3f, net_boost=%.3f",
        _m(rb_metrics, "avg_score", 0.0),
        rb_prepost if np.isfinite(rb_prepost) else float("nan"),
        rb_mode_ripple,
        _auto_target_tracking_for_pareto(rb_metrics),
        _m(rb_metrics, "max_net_boost_db", float("nan")),
    )
    return rb_metrics, w_metrics


def _emit_pareto_status_if_changed(*, status_cb, rb_metrics: dict, w_metrics: dict) -> None:
    if not callable(status_cb):
        return
    prev_rank = _m(rb_metrics, "rank_score", float("nan"))
    next_rank = _m(w_metrics, "rank_score", float("nan"))
    prev_avg = _m(rb_metrics, "avg_score", float("nan"))
    next_avg = _m(w_metrics, "avg_score", float("nan"))
    winner_changed = False
    if np.isfinite(prev_rank) and np.isfinite(next_rank) and abs(float(prev_rank) - float(next_rank)) > 1e-9:
        winner_changed = True
    if np.isfinite(prev_avg) and np.isfinite(next_avg) and abs(float(prev_avg) - float(next_avg)) > 1e-9:
        winner_changed = True
    if winner_changed:
        status_cb(
            _build_phase2_pareto_status(
                rank_best_metrics=rb_metrics,
                winner_metrics=w_metrics,
            )
        )


def _run_phase2_pareto_finalize(
    *,
    search_state,
    cfg,
    goal: str,
    status_cb,
) -> None:
    phase2_pool_raw = _build_phase2_pool_with_phase1_injection(search_state=search_state, cfg=cfg)
    if not phase2_pool_raw:
        return

    phase2_kept = _build_phase2_rank_window_kept(phase2_pool_raw=phase2_pool_raw, cfg=cfg)
    phase2_kept = _apply_phase2_hard_gate_if_enabled(phase2_kept=phase2_kept, cfg=cfg)

    pareto_min_n = int(max(1, cfg.phase2_pareto_pool_min))
    if len(phase2_kept) < pareto_min_n:
        logger.info(
            "Pareto front skipped: phase2 kept pool too small (%d < %d)",
            int(len(phase2_kept)),
            int(pareto_min_n),
        )
        return

    rank_best, pareto_winner = _resolve_pareto_winner(
        phase2_kept=phase2_kept,
        cfg=cfg,
        goal=goal,
    )
    if not isinstance(pareto_winner, dict):
        return

    rb_metrics, w_metrics = _log_pareto_winner_comparison(
        rank_best=rank_best,
        pareto_winner=pareto_winner,
    )
    _emit_pareto_status_if_changed(
        status_cb=status_cb,
        rb_metrics=rb_metrics,
        w_metrics=w_metrics,
    )


def finalize_search_result(
    *,
    search_base_data: dict,
    cache_base_data: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    hc_f,
    hc_m,
    pin_obj,
    cfg,
    goal: str,
    rank_basis: str,
    filter_key: str,
    compat_version: str,
    optimizer_backend: str,
    status_cb,
    optuna_mod,
    optuna_search_sig: str,
    seed: int,
    search_state,
    winner_target_name: str | None,
    phase1_ok: int,
    phase2_ok: int,
    phase1_tried: int,
    phase2_tried: int,
    phase1_plateau_hit: bool,
    phase2_plateau_hit: bool,
    phase1_optuna_tel: dict,
    phase2_local_optuna_tels: list,
    phase3_micro_optuna_tel: dict,
    phase2_rollup_tel: dict,
    _cache_ready_preset,
    _materialize_preset_result,
    _maybe_apply_residual_tiebreak,
    cache_refine_result: dict | None = None,
    runtime=None,
    canonical_signature: str | None = None,
) -> dict | None:
    runtime = coerce_orchestrator_runtime(runtime)
    if isinstance(cache_refine_result, dict):
        return _finalize_cached_result(
            runtime=runtime,
            search_base_data=search_base_data,
            cache_base_data=cache_base_data,
            measurements=measurements,
            fs_v=int(fs_v),
            taps_v=int(taps_v),
            xos=xos,
            hpf=hpf,
            cfg=cfg,
            goal=goal,
            rank_basis=rank_basis,
            filter_key=filter_key,
            compat_version=compat_version,
            optimizer_backend=optimizer_backend,
            status_cb=status_cb,
            optuna_mod=optuna_mod,
            optuna_search_sig=optuna_search_sig,
            seed=int(seed),
            _cache_ready_preset=_cache_ready_preset,
            _materialize_preset_result=_materialize_preset_result,
            _maybe_apply_residual_tiebreak=_maybe_apply_residual_tiebreak,
            cache_refine_result=dict(cache_refine_result or {}),
            canonical_signature=canonical_signature,
        )
    if search_state is None:
        return None

    logger.info("Automatic mode Phase 4: Pareto finalize, winner polish, final validation")
    _run_phase2_pareto_finalize(
        search_state=search_state,
        cfg=cfg,
        goal=goal,
        status_cb=status_cb,
    )

    if search_state.best_metrics is None or not isinstance(search_state.best_preset, dict):
        return None

    polish_meta, residual_peak_safety_override_meta = _apply_search_winner_polish_pipeline(
        runtime=runtime,
        search_state=search_state,
        search_base_data=search_base_data,
        cfg=cfg,
        goal=goal,
        filter_key=filter_key,
        status_cb=status_cb,
        winner_target_name=winner_target_name,
        _cache_ready_preset=_cache_ready_preset,
        _materialize_preset_result=_materialize_preset_result,
        _maybe_apply_residual_tiebreak=_maybe_apply_residual_tiebreak,
    )
    search_materialize_base_data = _stereo_refine_materialize_base_data(
        search_base_data,
        polish_meta.get("_stereo_refine_meta", {}),
    )

    materialized_ok, residual_peak_safety_override_meta, materialized_best_preset, final_best_preset = (
        _materialize_final_search_winner(
            search_state=search_state,
            search_materialize_base_data=search_materialize_base_data,
            cfg=cfg,
            goal=goal,
            winner_target_name=winner_target_name,
            residual_peak_safety_override_meta=residual_peak_safety_override_meta,
            _materialize_preset_result=_materialize_preset_result,
        )
    )
    if not bool(materialized_ok):
        return None

    _run_p6_final_validation(
        search_state,
        cfg,
        _materialize_preset_result=_materialize_preset_result,
        measurements=measurements,
        materialize_base_data=search_materialize_base_data,
    )
    if bool(dict(search_state.best_metrics or {}).get("final_ir_validation_reranked", False)):
        materialized_best_preset = dict(search_state.best_preset or {})
        final_best_preset = dict(search_state.best_preset or {})

    top = sorted(
        search_state.scored,
        key=lambda x: _auto_rank_key(x.get("metrics", {})),
    )[:5]
    logger.info(
        "Automatic mode search result: goal=%s, basis=%s, rank=%.3f",
        str(goal),
        str(rank_basis),
        _auto_safe_float(search_state.best_metrics.get("rank_score"), 0.0),
    )

    best_auto_exc_hz = _resolve_winner_auto_exc_hz(
        optimizer_backend=str(optimizer_backend or ""),
        materialized_best_preset=materialized_best_preset,
        search_base_data=search_base_data,
        best_metrics=search_state.best_metrics,
    )
    cached_best_preset = _cache_ready_preset(
        final_best_preset,
        best_metrics=search_state.best_metrics,
    )

    _save_final_search_cache(
        cache_base_data=cache_base_data,
        measurements=measurements,
        fs_v=int(fs_v),
        taps_v=int(taps_v),
        xos=xos,
        hpf=hpf,
        cfg=cfg,
        goal=goal,
        filter_key=filter_key,
        compat_version=compat_version,
        cached_best_preset=dict(cached_best_preset or {}),
        best_metrics=dict(search_state.best_metrics or {}),
        materialized_best_preset=dict(materialized_best_preset or {}),
        canonical_signature=canonical_signature,
    )

    logger.info("Automatic mode Phase 4 complete: final winner materialized and cached")
    return _build_final_search_result(
        search_state=search_state,
        cached_best_preset=dict(cached_best_preset or {}),
        materialized_best_preset=dict(materialized_best_preset or {}),
        polish_meta=polish_meta,
        residual_peak_safety_override_meta=residual_peak_safety_override_meta,
        optimizer_backend=optimizer_backend,
        best_auto_exc_hz=best_auto_exc_hz,
        goal=goal,
        rank_basis=rank_basis,
        top=top,
        phase1_ok=int(phase1_ok),
        phase2_ok=int(phase2_ok),
        phase1_tried=int(phase1_tried),
        phase2_tried=int(phase2_tried),
        phase1_plateau_hit=bool(phase1_plateau_hit),
        phase2_plateau_hit=bool(phase2_plateau_hit),
        phase1_optuna_tel=phase1_optuna_tel,
        phase2_local_optuna_tels=phase2_local_optuna_tels,
        phase3_micro_optuna_tel=phase3_micro_optuna_tel,
        phase2_rollup_tel=phase2_rollup_tel,
        fs_v=int(fs_v),
        taps_v=int(taps_v),
    )
