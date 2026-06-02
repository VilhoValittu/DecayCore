# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Pure helpers for auto-mode refine candidate evaluation."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..compute_context import AutoRunComputeContext
from ..rank_score import official_rank_score
from ..search_v2.candidates import deduplicate_presets
from ..search_state import _AutoModePhaseState, _AutoModeSearchState, _auto_set_search_winner
from ..scoring_ranking import _auto_is_better_refine, _auto_rank_key
from ...dsp.bass_integration import compute_bass_integration_metric_payload
from ..shared import (
    AUTO_MODE_BASS_INTEGRATION_GUARD_HI_RATIO,
    AUTO_MODE_BASS_INTEGRATION_GUARD_LO_RATIO,
    AUTO_MODE_REFINE_TIEBREAK_RANK_EPS,
    AUTO_MODE_REFINE_MODE_SOFT_K,
    MAX_SAFE_BOOST,
    _auto_bass_integration_profile_norm,
    _auto_metric_text,
    _auto_phase_limit_clip,
    _auto_safe_float,
    _auto_trial_chunk_size,
    _auto_trial_workers,
    _clip,
)

logger = logging.getLogger("DecayCore")
__all__ = [
    "RefineEvalContext",
    "apply_refine_mode_soft_penalty",
    "build_phase2_rollup_telemetry",
    "evaluate_search_candidate",
    "normalize_trial_preset",
    "run_candidate_phase",
]
















__all__ = [
    "apply_refine_mode_soft_penalty",
    "RefineEvalContext",
    "build_phase2_rollup_telemetry",
    "evaluate_search_candidate",
    "normalize_trial_preset",
    "run_candidate_phase",
]

@dataclass
class RefineEvalContext:
    search_base_data: dict
    measurements: dict
    fs_v: int
    taps_v: int
    xos: list
    hpf: dict | None
    hc_f: Any
    hc_m: Any
    pin_obj: Any
    cfg: Any
    goal: str
    filter_key: str
    optimizer_backend: str
    optuna_mod: Any
    seed: int
    optuna_search_sig: str
    status_cb: Callable[[str], None] | None
    status_prefix: str
    winner_target_name: str | None
    search_state: _AutoModeSearchState
    runtime: Any
    compute_ctx: AutoRunComputeContext | None = None

    def __post_init__(self) -> None:
        if self.cfg is None:
            raise ValueError("RefineEvalContext: cfg must not be None")
        if self.runtime is None:
            raise ValueError("RefineEvalContext: runtime must not be None")
        if self.search_state is None:
            raise ValueError("RefineEvalContext: search_state must not be None")

def apply_refine_mode_soft_penalty(
    metrics: dict | None,
    *,
    refine_mode_soft_k,
) -> dict:
    out = dict(metrics or {})
    soft_k = float(
        max(
            0.0,
            _auto_safe_float(refine_mode_soft_k, AUTO_MODE_REFINE_MODE_SOFT_K),
        )
    )
    mode_ripple = _auto_safe_float(out.get("mode_ripple_db"), float("nan"))
    rank_base = _auto_safe_float(out.get("rank_score"), 0.0)
    soft_pen = (
        float(soft_k) * max(0.0, float(mode_ripple))
        if np.isfinite(mode_ripple)
        else 0.0
    )
    out["mode_ripple_soft_penalty"] = float(soft_pen)
    out["rank_score_refine"] = float(rank_base - float(soft_pen))
    return out

def normalize_trial_preset(
    preset: dict | None,
    *,
    trial_data: dict | None,
    base_data: dict | None,
    filter_key: str | None,
) -> dict:
    trial_preset = dict(preset or {})
    trial_data = dict(trial_data or {})
    base_data = dict(base_data or {})
    filter_name = str(filter_key or "")
    if filter_name == "mixed":
        trial_preset["mixed_freq"] = round(
            _clip(
                trial_data.get("mixed_freq", base_data.get("mixed_freq", 180.0)),
                80.0,
                320.0,
            ),
            1,
        )
    elif filter_name in ("linear", "asym"):
        trial_preset["phase_limit"] = round(
            float(
                _auto_phase_limit_clip(
                    trial_data.get("phase_limit", base_data.get("phase_limit", 400.0)),
                    default=400.0,
                )
            ),
            1,
        )
    return dict(trial_preset)

def build_phase2_rollup_telemetry(
    *,
    phase2_local_optuna_tels: list | None,
    phase3_micro_optuna_tel: dict | None,
    telemetry_rollup_fn,
) -> dict:
    phase2_roll_items = [
        dict((it or {}).get("telemetry", {}) or {})
        for it in list(phase2_local_optuna_tels or [])
        if isinstance(it, dict)
    ]
    if isinstance(phase3_micro_optuna_tel, dict) and phase3_micro_optuna_tel:
        phase2_roll_items.append(dict(phase3_micro_optuna_tel))
    return dict(telemetry_rollup_fn(phase2_roll_items) or {})


def _refine_eval_prepare_trial_data(ctx: RefineEvalContext, preset: dict) -> dict:
    trial_data = dict(ctx.search_base_data or {})
    trial_data.update(dict(preset or {}))
    if str(ctx.filter_key) in ("linear", "asym"):
        trial_data["phase_limit"] = round(
            float(
                _auto_phase_limit_clip(
                    trial_data.get("phase_limit", ctx.search_base_data.get("phase_limit", 400.0)),
                    default=400.0,
                )
            ),
            1,
        )
    trial_data["comparison_mode"] = True
    return trial_data


def _refine_eval_apply_trial_tuning(cfg_trial, trial_data: dict) -> None:
    try:
        setattr(cfg_trial, "bass_smooth_w_gamma", float(trial_data.get("bass_smooth_w_gamma", 2.40)))
        setattr(cfg_trial, "bass_smooth_w_max", float(trial_data.get("bass_smooth_w_max", 0.45)))
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
        pass


def _refine_eval_apply_bass_integration_metrics(result, trial_data: dict, trial_measurements: dict) -> None:
    try:
        if bool(trial_measurements.get("bass_integration_enabled", False)):
            bundle = trial_measurements.get("bass_integration_bundle", None)
            if bundle is not None:
                fc_hz = _auto_safe_float(
                    trial_data.get("avr_crossover_hz", trial_measurements.get("avr_crossover_hz", 80.0)),
                    80.0,
                )
                profile = _auto_bass_integration_profile_norm(
                    trial_data.get(
                        "bass_integration_profile",
                        trial_measurements.get("bass_integration_profile", "safe"),
                    )
                )
                bi_mode = "direct_dac"
                xo_order = max(1, int(round(_auto_safe_float(trial_data.get("sub_crossover_slope", 24.0), 24.0))) // 6)
                sub_hpf_hz = float(_auto_safe_float(trial_data.get("sub_hpf_freq", 20.0), 20.0))
                sub_hpf_order = max(1, int(round(_auto_safe_float(trial_data.get("sub_hpf_slope", 12.0), 12.0))) // 6)
                sub_allpass_freq_hz = None
                sub_allpass_q = None
                if bi_mode == "direct_dac" and bool(trial_data.get("bass_integration_allpass_auto_applied", False)):
                    sub_allpass_freq_hz = _auto_safe_float(
                        trial_data.get("bass_integration_allpass_freq_hz", 0.0),
                        0.0,
                    )
                    sub_allpass_q = _auto_safe_float(
                        trial_data.get("bass_integration_allpass_q", 0.707),
                        0.707,
                    )
                _r_slpf: float | None = float(_auto_safe_float(trial_data.get("direct_dac_sub_lpf_hz", fc_hz), fc_hz))
                if not (_r_slpf and _r_slpf >= fc_hz):
                    _r_slpf = None
                metrics_update = compute_bass_integration_metric_payload(
                    bundle,
                    fc_hz,
                    profile,
                    mode=bi_mode,
                    main_hpf_order=int(xo_order),
                    sub_lpf_order=int(xo_order),
                    sub_hpf_hz=float(sub_hpf_hz),
                    sub_hpf_order=int(sub_hpf_order),
                    sub_combine_mode=str(trial_data.get("bass_integration_sub_combine_mode", "average") or "average"),
                    sub_delay_ms=_auto_safe_float(trial_data.get("bass_integration_sub_delay_ms", 0.0), 0.0),
                    sub_polarity_invert=bool(trial_data.get("bass_integration_sub_polarity_invert", False)),
                    sub_gain_trim_db=_auto_safe_float(trial_data.get("bass_integration_sub_gain_trim_db", 0.0), 0.0),
                    sub_lpf_hz=_r_slpf,
                    sub_allpass_freq_hz=sub_allpass_freq_hz,
                    sub_allpass_q=sub_allpass_q,
                    guard_lo_ratio=_auto_safe_float(
                        trial_data.get("bass_integration_guard_lo_ratio", AUTO_MODE_BASS_INTEGRATION_GUARD_LO_RATIO),
                        AUTO_MODE_BASS_INTEGRATION_GUARD_LO_RATIO,
                    ),
                    guard_hi_ratio=_auto_safe_float(
                        trial_data.get("bass_integration_guard_hi_ratio", AUTO_MODE_BASS_INTEGRATION_GUARD_HI_RATIO),
                        AUTO_MODE_BASS_INTEGRATION_GUARD_HI_RATIO,
                    ),
                )
                metrics_obj = getattr(result, "metrics", None)
                if isinstance(metrics_obj, dict):
                    metrics_obj.update(dict(metrics_update or {}))
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
    ) as exc:
        logger.debug("Bass integration trial metrics failed: %s: %s", type(exc).__name__, exc)


def _refine_eval_trial_metrics(
    ctx: RefineEvalContext,
    *,
    result,
    trial_data: dict,
    use_refine_tiebreak: bool,
    focus_lo_hz: float | None,
    focus_hi_hz: float | None,
) -> dict:
    metrics = ctx.runtime.auto_score_result(
        result,
        auto_exc_freq_hz=_auto_safe_float(
            trial_data.get("_auto_exc_freq_hz", float("nan")),
            float("nan"),
        ),
        focus_lo_hz=focus_lo_hz if bool(use_refine_tiebreak) else None,
        focus_hi_hz=focus_hi_hz if bool(use_refine_tiebreak) else None,
        base_data=trial_data,
    )
    if bool(use_refine_tiebreak):
        metrics = apply_refine_mode_soft_penalty(
            metrics,
            refine_mode_soft_k=getattr(
                ctx.cfg,
                "refine_mode_soft_k",
                AUTO_MODE_REFINE_MODE_SOFT_K,
            ),
        )
    return dict(metrics or {})


def _refine_eval_run_trial(
    ctx: RefineEvalContext,
    preset: dict,
    trial_data: dict,
    *,
    use_refine_tiebreak: bool,
    focus_lo_hz: float | None,
    focus_hi_hz: float | None,
) -> dict:
    cfg_trial = ctx.runtime.build_config(
        trial_data,
        fs_v=int(ctx.fs_v),
        taps_v=int(ctx.taps_v),
        xos=ctx.xos,
        hpf=ctx.hpf,
        hc_f=ctx.hc_f,
        hc_m=ctx.hc_m,
        max_safe_boost=float(MAX_SAFE_BOOST),
    )
    _refine_eval_apply_trial_tuning(cfg_trial, trial_data)
    trial_measurements = dict(ctx.measurements or {})
    trial_measurements["ui_data"] = trial_data
    result = ctx.runtime.run_pipeline(
        cfg_trial,
        trial_measurements,
        include_response_arrays=False,
    )
    _refine_eval_apply_bass_integration_metrics(result, trial_data, trial_measurements)
    metrics = _refine_eval_trial_metrics(
        ctx,
        result=result,
        trial_data=trial_data,
        use_refine_tiebreak=bool(use_refine_tiebreak),
        focus_lo_hz=focus_lo_hz,
        focus_hi_hz=focus_hi_hz,
    )
    trial_preset = normalize_trial_preset(
        preset,
        trial_data=trial_data,
        base_data=ctx.search_base_data,
        filter_key=ctx.filter_key,
    )
    return {
        "ok": True,
        "metrics": dict(metrics or {}),
        "trial_preset": dict(trial_preset),
    }

def evaluate_search_candidate(
    ctx: RefineEvalContext,
    *,
    preset: dict,
    use_refine_tiebreak: bool,
    focus_lo_hz: float | None,
    focus_hi_hz: float | None,
) -> dict:
    if ctx.compute_ctx is not None:
        ctx.compute_ctx.begin_trial(0)
    trial_data = _refine_eval_prepare_trial_data(ctx, preset)
    return _refine_eval_run_trial(
        ctx,
        preset,
        trial_data,
        use_refine_tiebreak=bool(use_refine_tiebreak),
        focus_lo_hz=focus_lo_hz,
        focus_hi_hz=focus_hi_hz,
    )


def _consume_phase_result_is_better(
    ctx: RefineEvalContext,
    metrics: dict,
    *,
    use_refine_tiebreak: bool,
) -> tuple[bool, str]:
    if bool(use_refine_tiebreak):
        return _auto_is_better_refine(
            metrics,
            ctx.search_state.best_metrics or {},
            ctx.goal,
            return_reason=True,
        )
    return bool(_auto_rank_key(metrics) < _auto_rank_key(ctx.search_state.best_metrics or {})), "rank"


def _consume_phase_result_handle_ok(
    ctx: RefineEvalContext,
    phase_state: _AutoModePhaseState,
    *,
    phase_label: str,
    idx: int,
    n_total: int,
    out: dict,
    use_refine_tiebreak: bool,
) -> tuple[bool, float, float, str]:
    metrics = dict(out.get("metrics", {}) or {})
    candidate_rank = _auto_safe_float(official_rank_score(metrics), float("nan"))
    candidate_refine_rank = _auto_safe_float(metrics.get("rank_score_refine"), float("nan"))
    metrics["trial"] = int(len(ctx.search_state.scored) + 1)
    metrics["phase"] = str(phase_label)
    trial_preset = dict(out.get("trial_preset", {}) or {})
    ctx.search_state.scored.append({"metrics": dict(metrics), "preset": dict(trial_preset)})
    if bool(use_refine_tiebreak):
        ctx.search_state.phase2_pool.append(
            {
                "preset": dict(trial_preset),
                "metrics": dict(metrics or {}),
            }
        )
    phase_state.ok_n += 1

    if ctx.search_state.best_metrics is None:
        better = True
        refine_reason = "rank"
    else:
        better, refine_reason = _consume_phase_result_is_better(
            ctx,
            metrics,
            use_refine_tiebreak=bool(use_refine_tiebreak),
        )

    if better:
        prev_best = dict(ctx.search_state.best_metrics or {})
        _auto_set_search_winner(
            ctx.search_state,
            metrics,
            trial_preset,
            prev_metrics=prev_best,
            phase_label=phase_label,
            target_name=ctx.winner_target_name,
        )
        phase_state.improved_any = True
        if bool(use_refine_tiebreak) and str(refine_reason) == "mode_ripple":
            rank_prev = _auto_safe_float(prev_best.get("rank_score"), 0.0)
            rank_new = _auto_safe_float(metrics.get("rank_score"), 0.0)
            rank_eps = float(
                max(
                    0.0,
                    _auto_safe_float(
                        ctx.cfg.refine_tiebreak_rank_eps,
                        AUTO_MODE_REFINE_TIEBREAK_RANK_EPS,
                    ),
                )
            )
            if abs(float(rank_new) - float(rank_prev)) <= float(rank_eps):
                mode_hz_i = int(round(_auto_safe_float(metrics.get("mode_hz"), 0.0)))
                band_lo_i = int(round(_auto_safe_float(metrics.get("mode_band_lo"), 0.0)))
                band_hi_i = int(round(_auto_safe_float(metrics.get("mode_band_hi"), 0.0)))
                ripple_prev = _auto_safe_float(prev_best.get("mode_ripple_db"), float("nan"))
                ripple_new = _auto_safe_float(metrics.get("mode_ripple_db"), float("nan"))
                logger.info(
                    "REFINE MODE-TIE: rank within eps (A=%.3f, B=%.3f), mode=%dHz band=%d-%dHz ripple %.3f -> %.3f",
                    float(rank_prev),
                    float(rank_new),
                    int(mode_hz_i),
                    int(band_lo_i),
                    int(band_hi_i),
                    float(ripple_prev if np.isfinite(ripple_prev) else 0.0),
                    float(ripple_new if np.isfinite(ripple_new) else 0.0),
                )
    return bool(better), float(candidate_rank), float(candidate_refine_rank), str(refine_reason)


def _consume_phase_result_emit_status(
    ctx: RefineEvalContext,
    phase_state: _AutoModePhaseState,
    *,
    phase_label: str,
    idx: int,
    n_total: int,
    candidate_rank: float,
    candidate_refine_rank: float,
    use_refine_tiebreak: bool,
    improved: bool,
    refine_reason: str,
) -> None:
    if not callable(ctx.status_cb):
        return
    rank_now = _auto_safe_float(official_rank_score(ctx.search_state.best_metrics or {}), 0.0)
    if bool(improved):
        avg_now = _auto_safe_float((ctx.search_state.best_metrics or {}).get("avg_score"), 0.0)
        mode_now = _auto_safe_float((ctx.search_state.best_metrics or {}).get("mode_ripple_db"), float("nan"))
        boost_now = _auto_safe_float((ctx.search_state.best_metrics or {}).get("max_net_boost_db"), float("nan"))
        baseline_note = (
            ", baseline initialized"
            if int(phase_state.ok_n) == 1 and int(phase_state.tried_n) == 1
            else ""
        )
        ctx.status_cb(
            f"{ctx.status_prefix}: {phase_label} best improved trial {idx}/{n_total} "
            f"(goal {ctx.goal}, rank {rank_now:.3f}, avg {avg_now:.3f}, "
            f"mode {'n/a' if not np.isfinite(mode_now) else f'{mode_now:.3f} dB'}, "
            f"boost {'n/a' if not np.isfinite(boost_now) else f'{boost_now:.2f} dB'}, "
            f"ok {int(phase_state.ok_n)}/{int(phase_state.tried_n)}{baseline_note})"
        )
        return
    if ctx.search_state.best_metrics is None:
        return
    cand_txt = "n/a" if not np.isfinite(candidate_rank) else f"{float(candidate_rank):.3f}"
    refine_txt = ""
    if bool(use_refine_tiebreak):
        refine_rank_txt = (
            "n/a"
            if not np.isfinite(candidate_refine_rank)
            else f"{float(candidate_refine_rank):.3f}"
        )
        refine_txt = f", decision {refine_reason}, refine rank {refine_rank_txt}"
    ctx.status_cb(
        f"{ctx.status_prefix}: {phase_label} {idx}/{n_total} "
        f"(best rank {rank_now:.3f}, candidate rank {cand_txt}, "
        f"ok {int(phase_state.ok_n)}/{int(phase_state.tried_n)}{refine_txt})"
    )


def _consume_phase_result_handle_plateau(
    ctx: RefineEvalContext,
    phase_state: _AutoModePhaseState,
    *,
    phase_label: str,
    idx: int,
    n_total: int,
    plateau_after_no_improve: int,
    plateau_min_trials: int,
) -> bool:
    if int(plateau_after_no_improve) <= 0:
        return False
    if phase_state.no_improve_streak < int(plateau_after_no_improve):
        return False
    if int(phase_state.tried_n) < int(max(0, plateau_min_trials)):
        return False
    phase_state.plateau_hit = True
    best_now = (
        "n/a"
        if not ctx.search_state.best_metrics
        else _auto_metric_text(ctx.search_state.best_metrics, ctx.goal)
    )
    move_txt = "plateau -> phase 2" if "1/2" in str(phase_label) else "plateau -> stop"
    logger.info(
        "Automatic mode %s: no-improve plateau detected (%d rounds), %s.",
        str(phase_label),
        int(plateau_after_no_improve),
        str(move_txt),
    )
    if callable(ctx.status_cb):
        ctx.status_cb(
            f"{ctx.status_prefix}: {phase_label} {idx}/{n_total} "
            f"(best {best_now}, {move_txt})"
        )
    return True


def _consume_phase_result(
    ctx: RefineEvalContext,
    phase_state: _AutoModePhaseState,
    *,
    idx: int,
    n_total: int,
    out: dict,
    phase_label: str,
    plateau_after_no_improve: int,
    plateau_min_trials: int = 0,
    use_refine_tiebreak: bool = False,
) -> bool:
    phase_state.tried_n += 1
    improved = False
    candidate_rank = float("nan")
    candidate_refine_rank = float("nan")
    refine_reason = "rank"
    if bool(out.get("ok", False)):
        improved, candidate_rank, candidate_refine_rank, refine_reason = _consume_phase_result_handle_ok(
            ctx,
            phase_state,
            phase_label=phase_label,
            idx=int(idx),
            n_total=int(n_total),
            out=dict(out or {}),
            use_refine_tiebreak=bool(use_refine_tiebreak),
        )
    else:
        err_txt = str(out.get("error", "unknown error") or "unknown error")
        log_fn = logger.info if bool(out.get("pruned", False)) else logger.warning
        log_fn(
            "Automatic mode trial %d/%d failed (%s): %s",
            int(idx),
            int(n_total),
            str(phase_label),
            str(err_txt),
        )
    _consume_phase_result_emit_status(
        ctx,
        phase_state,
        phase_label=phase_label,
        idx=int(idx),
        n_total=int(n_total),
        candidate_rank=float(candidate_rank),
        candidate_refine_rank=float(candidate_refine_rank),
        use_refine_tiebreak=bool(use_refine_tiebreak),
        improved=bool(improved),
        refine_reason=str(refine_reason),
    )
    if improved:
        phase_state.no_improve_streak = 0
    else:
        phase_state.no_improve_streak += 1
    return _consume_phase_result_handle_plateau(
        ctx,
        phase_state,
        phase_label=phase_label,
        idx=int(idx),
        n_total=int(n_total),
        plateau_after_no_improve=int(plateau_after_no_improve),
        plateau_min_trials=int(plateau_min_trials),
    )


def _run_candidate_phase_optuna(
    ctx: RefineEvalContext,
    phase_state: _AutoModePhaseState,
    *,
    phase_label: str,
    phase_kind: str | None,
    n_total: int,
    seed_presets: list[dict] | None,
    optuna_builder,
    seed_to_params,
    study_scope: str | None,
    optuna_base_data_override: dict | None,
    use_refine_tiebreak: bool,
    focus_lo_hz: float | None,
    focus_hi_hz: float | None,
) -> dict:
    def _eval_one(idx: int, preset: dict) -> dict:
        out = evaluate_search_candidate(
            ctx,
            preset=dict(preset or {}),
            use_refine_tiebreak=bool(use_refine_tiebreak),
            focus_lo_hz=focus_lo_hz,
            focus_hi_hz=focus_hi_hz,
        )
        return {"idx": int(idx), **dict(out or {})}

    def _consume_one(idx: int, out: dict) -> bool:
        return _consume_phase_result(
            ctx,
            phase_state,
            idx=int(idx),
            n_total=int(n_total),
            out=dict(out or {}),
            phase_label=phase_label,
            plateau_after_no_improve=0,
            plateau_min_trials=0,
            use_refine_tiebreak=bool(use_refine_tiebreak),
        )

    raw_scope = str(study_scope or phase_label)
    optuna_base_data = dict(optuna_base_data_override or ctx.search_base_data or {})
    scope_eff = ctx.runtime.auto_optuna_effective_scope(
        optuna_base_data,
        raw_scope,
        phase_kind=phase_kind,
    )
    study_name = ctx.runtime.auto_optuna_study_name(
        study_sig=ctx.optuna_search_sig,
        scope=scope_eff,
    )
    phase_seed = int(ctx.seed + sum(ord(ch) for ch in str(phase_label)) * 31)
    workers = int(_auto_trial_workers(ctx.search_base_data, n_total))
    phase_tel = dict(
        ctx.runtime.auto_run_optuna_eval_loop(
            optuna_mod=ctx.optuna_mod,
            cfg=ctx.cfg,
            n_total=int(n_total),
            seed=int(phase_seed),
            base_data=dict(optuna_base_data),
            seed_presets=list(seed_presets or []),
            build_preset=optuna_builder,
            eval_one=_eval_one,
            consume_one=_consume_one,
            objective_value=lambda out, _goal=str((ctx.search_base_data or {}).get("auto_goal", "") or ""): ctx.runtime.auto_optuna_objective_value(
                dict((out or {}).get("metrics", {}) or {}),
                use_refine_tiebreak=bool(use_refine_tiebreak),
                goal=_goal,
            ),
            workers=int(workers),
            seed_to_params=seed_to_params,
            study_name=study_name,
            study_scope=raw_scope,
            phase_label=str(phase_label),
            phase_kind=phase_kind,
        )
        or {}
    )
    if ctx.runtime.auto_optuna_needs_zero_feasible_rescue(
        base_data=optuna_base_data,
        phase_kind=phase_kind,
        telemetry=phase_tel,
    ):
        logger.info(
            "Automatic mode Optuna rescue fallback: phase=%s scope=%s rerunning without constraints",
            str(phase_kind or ""),
            str(raw_scope),
        )
        rescue_base_data = ctx.runtime.auto_optuna_base_data_without_constraints(
            optuna_base_data
        )
        rescue_scope = f"{str(raw_scope)}-zf0"
        rescue_scope_eff = ctx.runtime.auto_optuna_effective_scope(
            rescue_base_data,
            rescue_scope,
            phase_kind=phase_kind,
        )
        rescue_tel = dict(
            ctx.runtime.auto_run_optuna_eval_loop(
                optuna_mod=ctx.optuna_mod,
                cfg=ctx.cfg,
                n_total=int(n_total),
                seed=int(phase_seed + 999983),
                base_data=dict(rescue_base_data or {}),
                seed_presets=list(seed_presets or []),
                build_preset=optuna_builder,
                eval_one=_eval_one,
                consume_one=_consume_one,
                objective_value=lambda out, _goal=str((ctx.search_base_data or {}).get("auto_goal", "") or ""): ctx.runtime.auto_optuna_objective_value(
                    dict((out or {}).get("metrics", {}) or {}),
                    use_refine_tiebreak=bool(use_refine_tiebreak),
                    goal=_goal,
                ),
                workers=int(workers),
                seed_to_params=seed_to_params,
                study_name=ctx.runtime.auto_optuna_study_name(
                    study_sig=ctx.optuna_search_sig,
                    scope=rescue_scope_eff,
                ),
                study_scope=rescue_scope,
                phase_label=f"{str(phase_label)} rescue",
                phase_kind=phase_kind,
            )
            or {}
        )
        logger.info(
            "Automatic mode Optuna rescue result [%s]: run=%d ok=%d startup=%d model=%d best=%s",
            str(phase_label),
            int((rescue_tel or {}).get("run_trials", 0) or 0),
            int((rescue_tel or {}).get("complete_trials", 0) or 0),
            int((rescue_tel or {}).get("startup_complete", 0) or 0),
            int((rescue_tel or {}).get("model_complete", 0) or 0),
            "n/a"
            if (rescue_tel or {}).get("best_raw_value", None) is None
            else f"{float(rescue_tel['best_raw_value']):.3f}",
        )
        phase_tel = {
            **dict(phase_tel or {}),
            "zero_feasible_fallback_used": True,
            "fallback_reason": "zero_feasible",
            "fallback_telemetry": dict(rescue_tel or {}),
        }
    else:
        phase_tel = {
            **dict(phase_tel or {}),
            "zero_feasible_fallback_used": False,
        }
    return {
        "ok": int(phase_state.ok_n),
        "tried": int(phase_state.tried_n),
        "plateau_hit": bool(phase_state.plateau_hit),
        "improved_any": bool(phase_state.improved_any),
        "optuna_telemetry": dict(phase_tel or {}),
        "optuna_zero_feasible_fallback_used": bool(
            phase_tel.get("zero_feasible_fallback_used", False)
        ),
        "optuna_zero_feasible_fallback_telemetry": dict(
            phase_tel.get("fallback_telemetry", {}) or {}
        ),
    }


def _run_candidate_phase_local(
    ctx: RefineEvalContext,
    phase_state: _AutoModePhaseState,
    *,
    eval_candidates: list[dict],
    phase_label: str,
    n_total: int,
    use_refine_tiebreak: bool,
    focus_lo_hz: float | None,
    focus_hi_hz: float | None,
    plateau_after_no_improve: int,
    plateau_min_trials: int,
    seed_presets: list[dict] | None,
) -> dict:
    workers = int(_auto_trial_workers(ctx.search_base_data, n_total))
    if workers > 1:
        logger.info(
            "Automatic mode %s: parallel trial run enabled (%d workers)",
            str(phase_label),
            int(workers),
        )
    idx_presets = list(
        enumerate(
            list(
                deduplicate_presets([*list(seed_presets or []), *list(eval_candidates or [])])
            ),
            start=1,
        )
    )
    if workers <= 1 or n_total <= 1:
        return _run_candidate_phase_local_serial(
            ctx,
            phase_state,
            idx_presets=idx_presets,
            phase_label=phase_label,
            n_total=int(n_total),
            use_refine_tiebreak=bool(use_refine_tiebreak),
            focus_lo_hz=focus_lo_hz,
            focus_hi_hz=focus_hi_hz,
            plateau_after_no_improve=int(plateau_after_no_improve),
            plateau_min_trials=int(plateau_min_trials),
        )
    return _run_candidate_phase_local_parallel(
        ctx,
        phase_state,
        idx_presets=idx_presets,
        phase_label=phase_label,
        n_total=int(n_total),
        workers=int(workers),
        use_refine_tiebreak=bool(use_refine_tiebreak),
        focus_lo_hz=focus_lo_hz,
        focus_hi_hz=focus_hi_hz,
        plateau_after_no_improve=int(plateau_after_no_improve),
        plateau_min_trials=int(plateau_min_trials),
    )


def _run_candidate_phase_local_serial(
    ctx: RefineEvalContext,
    phase_state: _AutoModePhaseState,
    *,
    idx_presets: list[tuple[int, dict]],
    phase_label: str,
    n_total: int,
    use_refine_tiebreak: bool,
    focus_lo_hz: float | None,
    focus_hi_hz: float | None,
    plateau_after_no_improve: int,
    plateau_min_trials: int,
) -> dict:
    def _eval_one(idx: int, preset: dict) -> dict:
        out = evaluate_search_candidate(
            ctx,
            preset=dict(preset or {}),
            use_refine_tiebreak=bool(use_refine_tiebreak),
            focus_lo_hz=focus_lo_hz,
            focus_hi_hz=focus_hi_hz,
        )
        return {"idx": int(idx), **dict(out or {})}

    def _consume_one(idx: int, out: dict) -> bool:
        return _consume_phase_result(
            ctx,
            phase_state,
            idx=int(idx),
            n_total=int(n_total),
            out=dict(out or {}),
            phase_label=phase_label,
            plateau_after_no_improve=int(plateau_after_no_improve),
            plateau_min_trials=int(plateau_min_trials),
            use_refine_tiebreak=bool(use_refine_tiebreak),
        )

    for idx, preset in idx_presets:
        try:
            out = _eval_one(int(idx), dict(preset or {}))
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
        ) as exc:
            out = {"idx": int(idx), "ok": False, "error": f"{type(exc).__name__}: {exc}"}
        if _consume_one(int(idx), out):
            break
    return {
        "ok": int(phase_state.ok_n),
        "tried": int(phase_state.tried_n),
        "plateau_hit": bool(phase_state.plateau_hit),
        "improved_any": bool(phase_state.improved_any),
        "optuna_telemetry": {},
        "optuna_zero_feasible_fallback_used": False,
        "optuna_zero_feasible_fallback_telemetry": {},
    }


def _run_candidate_phase_local_parallel(
    ctx: RefineEvalContext,
    phase_state: _AutoModePhaseState,
    *,
    idx_presets: list[tuple[int, dict]],
    phase_label: str,
    n_total: int,
    workers: int,
    use_refine_tiebreak: bool,
    focus_lo_hz: float | None,
    focus_hi_hz: float | None,
    plateau_after_no_improve: int,
    plateau_min_trials: int,
) -> dict:
    def _eval_one(idx: int, preset: dict) -> dict:
        out = evaluate_search_candidate(
            ctx,
            preset=dict(preset or {}),
            use_refine_tiebreak=bool(use_refine_tiebreak),
            focus_lo_hz=focus_lo_hz,
            focus_hi_hz=focus_hi_hz,
        )
        return {"idx": int(idx), **dict(out or {})}

    def _consume_one(idx: int, out: dict) -> bool:
        return _consume_phase_result(
            ctx,
            phase_state,
            idx=int(idx),
            n_total=int(n_total),
            out=dict(out or {}),
            phase_label=phase_label,
            plateau_after_no_improve=int(plateau_after_no_improve),
            plateau_min_trials=int(plateau_min_trials),
            use_refine_tiebreak=bool(use_refine_tiebreak),
        )

    chunk_size = int(_auto_trial_chunk_size(workers))
    stop_now = False
    with ThreadPoolExecutor(max_workers=int(workers)) as executor:
        for c0 in range(0, int(len(idx_presets)), int(chunk_size)):
            chunk = idx_presets[c0 : c0 + int(chunk_size)]
            future_map = {
                executor.submit(_eval_one, int(idx), dict(preset or {})): int(idx)
                for idx, preset in chunk
            }
            chunk_out: dict[int, dict] = {}
            for future in as_completed(list(future_map.keys())):
                idx = int(future_map.get(future, 0))
                try:
                    out = future.result()
                    if not isinstance(out, dict):
                        out = {"idx": int(idx), "ok": False, "error": "invalid worker result"}
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
                ) as exc:
                    out = {"idx": int(idx), "ok": False, "error": f"{type(exc).__name__}: {exc}"}
                chunk_out[int(idx)] = dict(out)
            for idx, _preset in chunk:
                out = dict(
                    chunk_out.get(
                        int(idx),
                        {"idx": int(idx), "ok": False, "error": "missing worker result"},
                    )
                    or {}
                )
                if _consume_one(int(idx), out):
                    stop_now = True
                    break
            if stop_now:
                break
    return {
        "ok": int(phase_state.ok_n),
        "tried": int(phase_state.tried_n),
        "plateau_hit": bool(phase_state.plateau_hit),
        "improved_any": bool(phase_state.improved_any),
        "optuna_telemetry": {},
        "optuna_zero_feasible_fallback_used": False,
        "optuna_zero_feasible_fallback_telemetry": {},
    }


def run_candidate_phase(
    ctx: RefineEvalContext,
    cands: list[dict],
    *,
    phase_label: str,
    phase_kind: str | None = None,
    plateau_after_no_improve: int = 0,
    plateau_min_trials: int = 0,
    use_refine_tiebreak: bool = False,
    focus_lo_hz: float | None = None,
    focus_hi_hz: float | None = None,
    n_total_override: int | None = None,
    seed_presets: list[dict] | None = None,
    optuna_builder=None,
    seed_to_params=None,
    study_scope: str | None = None,
    optuna_base_data_override: dict | None = None,
) -> dict:
    phase_state = _AutoModePhaseState()
    use_optuna_phase = bool(
        str(ctx.optimizer_backend) == "optuna"
        and ctx.runtime.auto_optuna_module_ready(ctx.optuna_mod)
        and callable(optuna_builder)
    )
    eval_candidates = list(cands or [])
    if not use_optuna_phase:
        eval_candidates = deduplicate_presets([*list(seed_presets or []), *eval_candidates])
        n_total = int(len(eval_candidates))
    else:
        n_total = int(n_total_override) if n_total_override is not None else int(len(cands))
    if n_total <= 0:
        return {
            "ok": 0,
            "tried": 0,
            "plateau_hit": False,
            "improved_any": False,
            "optuna_telemetry": {},
            "optuna_zero_feasible_fallback_used": False,
            "optuna_zero_feasible_fallback_telemetry": {},
        }
    if bool(use_optuna_phase):
        return _run_candidate_phase_optuna(
            ctx,
            phase_state,
            phase_label=phase_label,
            phase_kind=phase_kind,
            n_total=int(n_total),
            seed_presets=list(seed_presets or []),
            optuna_builder=optuna_builder,
            seed_to_params=seed_to_params,
            study_scope=study_scope,
            optuna_base_data_override=optuna_base_data_override,
            use_refine_tiebreak=bool(use_refine_tiebreak),
            focus_lo_hz=focus_lo_hz,
            focus_hi_hz=focus_hi_hz,
        )
    return _run_candidate_phase_local(
        ctx,
        phase_state,
        eval_candidates=list(eval_candidates or []),
        phase_label=phase_label,
        n_total=int(n_total),
        use_refine_tiebreak=bool(use_refine_tiebreak),
        focus_lo_hz=focus_lo_hz,
        focus_hi_hz=focus_hi_hz,
        plateau_after_no_improve=int(plateau_after_no_improve),
        plateau_min_trials=int(plateau_min_trials),
        seed_presets=list(seed_presets or []),
    )


__all__ = ['RefineEvalContext', 'apply_refine_mode_soft_penalty', 'normalize_trial_preset', 'build_phase2_rollup_telemetry', 'evaluate_search_candidate', '_consume_phase_result', 'run_candidate_phase']


def _link_sibling_exports() -> None:
    import importlib
    package = __package__
    for module_name in ['refine_eval_01']:
        if module_name == __name__.rsplit('.', 1)[-1]:
            continue
        module = importlib.import_module(f"{package}.{module_name}")
        for symbol in getattr(module, "__all__", ()):
            globals().setdefault(symbol, getattr(module, symbol))


_link_sibling_exports()
