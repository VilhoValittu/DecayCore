# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Phase1 trial execution and setup helpers for target-curve selection orchestration."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from .candidate_generation import (
    _build_auto_mode_candidates_local,
    _seed_auto_mode_candidate_local_optuna_params,
    _seed_auto_mode_candidate_optuna_params,
    _suggest_auto_mode_candidate_local_optuna,
)
from .rank_score import official_rank_score
from .scoring_ranking import (
    _auto_adaptive_shrink_factor,
    _auto_goal_uses_local_refine,
    _auto_rank_key,
    _auto_select_best_scored,
)
from .shared_parts import (
    AUTO_MODE_LOCAL_REFINE_ENABLED,
    AUTO_MODE_LOCAL_REFINE_TOP_K,
    AUTO_MODE_LOCAL_REFINE_TRIALS_PER_TOP,
    AUTO_MODE_LOCAL_REFINEMENT_SHRINK,
    AUTO_MODE_TARGET_LOCAL_REFINE_ENABLED,
    _auto_phase_limit_clip,
    _auto_safe_float,
    _clip,
)
from .orchestrator_target_types import (
    _TargetTrialAccumulator,
    _TargetTrialSetup,
    _target_trial_issue_label,
    _target_trial_log_method,
)
from .orchestrator_target_trials import _run_target_trials, _prepare_target_trial_setup

logger = logging.getLogger("DecayCore")


def _target_local_candidate_clip(
    cand_in: dict,
    *,
    ref_profile: dict,
    base_tc: dict,
    filter_key: str,
) -> dict:
    cand = dict(cand_in or {})
    mf = _auto_safe_float(cand.get("mixed_freq"), float("nan"))
    if np.isfinite(mf):
        cand["mixed_freq"] = _clip(
            mf,
            ref_profile["mixed_center"] - ref_profile["mixed_span"],
            ref_profile["mixed_center"] + ref_profile["mixed_span"],
        )
    td = _auto_safe_float(cand.get("tdc_strength"), float("nan"))
    if np.isfinite(td):
        cand["tdc_strength"] = _clip(
            td,
            ref_profile["tdc_lo"],
            ref_profile["tdc_hi"],
        )
    if str(filter_key) in ("linear", "asym"):
        cand["phase_limit"] = round(
            float(
                _auto_phase_limit_clip(
                    cand.get("phase_limit", base_tc.get("phase_limit", 400.0)),
                    default=400.0,
                )
            ),
            1,
        )
    return dict(cand)


def _run_target_phase1_trials(
    *,
    runtime,
    cfg,
    optimizer_backend: str,
    optuna_mod,
    target_study_sig: str,
    seed_target: int,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    pin_obj,
    filter_key: str,
    shortlisted: list[dict],
    status_cb,
    f6_txt: str,
    goal: str,
    tc: dict,
    t_idx: int,
    emit_status: bool,
    setup: _TargetTrialSetup,
) -> _TargetTrialAccumulator:
    accumulator = _TargetTrialAccumulator(trials_total_count=int(setup.phase1_trial_total))
    cb = status_cb if bool(emit_status) else None
    phase1_out = _run_target_trials(
        runtime=runtime,
        cfg=cfg,
        optimizer_backend=optimizer_backend,
        optuna_mod=optuna_mod,
        target_study_sig=target_study_sig,
        seed_target=seed_target,
        cands=list(setup.candidates or []),
        base_tc=dict(setup.base_tc or {}),
        measurements=measurements,
        fs_v=int(fs_v),
        taps_v=int(taps_v),
        xos=xos,
        hpf=hpf,
        hc_f_arr=setup.hc_f,
        hc_m_arr=setup.hc_m,
        pin_obj=pin_obj,
        filter_key=filter_key,
        phase_tag="phase1",
        target_name=setup.hc_name,
        phase_kind="target",
        n_total_override=int(setup.phase1_trial_total),
        seed_presets=list(setup.phase1_seed_presets or []),
        optuna_builder=(
            (
                lambda tr, _base_tc=dict(setup.base_tc): runtime.suggest_auto_mode_candidate_optuna(
                    _base_tc,
                    tr,
                    optimize_mag_low=bool(_base_tc.get("auto_optimize_low_bass_cut", True)),
                )
            )
            if bool(setup.use_optuna_curve_trials)
            else None
        ),
        seed_to_params=(
            (
                lambda preset, _base_tc=dict(setup.base_tc): _seed_auto_mode_candidate_optuna_params(
                    _base_tc,
                    preset,
                    optimize_mag_low=bool(_base_tc.get("auto_optimize_low_bass_cut", True)),
                )
            )
            if bool(setup.use_optuna_curve_trials)
            else None
        ),
    )
    for c_idx, out in enumerate(phase1_out, start=1):
        improved = False
        if bool(out.get("ok", False)):
            met = dict(out.get("metrics", {}) or {})
            trial_preset = dict(out.get("preset", {}) or {})
            accumulator.ok_n += 1
            accumulator.rank_sum += _auto_safe_float(met.get("rank_score"), 0.0)
            accumulator.avg_score_sum += _auto_safe_float(met.get("avg_score"), 0.0)
            accumulator.phase1_scored.append({"metrics": dict(met), "preset": dict(trial_preset)})
            accumulator.curve_scored.append({"metrics": dict(met), "preset": dict(trial_preset)})
            if accumulator.best_metrics is None or _auto_rank_key(met) < _auto_rank_key(accumulator.best_metrics):
                accumulator.best_metrics = dict(met)
                accumulator.best_preset = dict(trial_preset)
                improved = True
        else:
            _target_trial_log_method(out=out)(
                "Automatic mode target trial %s: target=%s %d/%d (%s)",
                _target_trial_issue_label(out=out),
                str(setup.hc_name),
                int(c_idx),
                int(setup.phase1_trial_total),
                str(out.get("error", "unknown error") or "unknown error"),
            )
        if callable(cb):
            rank_now = official_rank_score(accumulator.best_metrics or {})
            if bool(improved):
                avg_now = _auto_safe_float((accumulator.best_metrics or {}).get("avg_score"), 0.0)
                cb(
                    "DecayCore automatic mode: target trials best improved "
                    f"(target {t_idx}/{len(shortlisted)} {setup.hc_name}, "
                    f"trial {c_idx}/{int(setup.phase1_trial_total)}{f6_txt}, goal {goal}, "
                    f"rank {rank_now:.3f}, avg {avg_now:.3f}, "
                    f"fit {_auto_safe_float(tc.get('fit_rms_db', 0.0), 0.0):.3f}, "
                    f"pre {_auto_safe_float(tc.get('preselect_score', tc.get('fit_rms_db', 0.0)), 0.0):.3f})"
                )
            elif accumulator.best_metrics is not None:
                cb(
                    f"DecayCore automatic mode: target trials "
                    f"(target {t_idx}/{len(shortlisted)} {setup.hc_name}, "
                    f"trial {c_idx}/{int(setup.phase1_trial_total)}, rank {rank_now:.3f})"
                )
    return accumulator


def _run_target_local_refine_trials(
    *,
    runtime,
    cfg,
    optimizer_backend: str,
    optuna_mod,
    target_study_sig: str,
    seed_target: int,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    pin_obj,
    filter_key: str,
    goal: str,
    status_cb,
    setup: _TargetTrialSetup,
    accumulator: _TargetTrialAccumulator,
) -> _TargetTrialAccumulator:
    if not (
        accumulator.phase1_scored
        and bool(AUTO_MODE_LOCAL_REFINE_ENABLED)
        and bool(AUTO_MODE_TARGET_LOCAL_REFINE_ENABLED)
        and _auto_goal_uses_local_refine(goal)
    ):
        return accumulator
    from .scoring_ranking import _auto_build_refine_profile

    top_list = sorted(
        list(accumulator.phase1_scored),
        key=lambda it: _auto_rank_key(dict(it.get("metrics", {}) or {})),
    )[: int(max(1, AUTO_MODE_LOCAL_REFINE_TOP_K))]
    ref_profile = _auto_build_refine_profile(
        base_data=setup.base_tc,
        phase1_top=top_list,
    )
    phase1_best = dict(_auto_select_best_scored(top_list, goal=goal) or top_list[0])
    p1m = dict(phase1_best.get("metrics", {}) or {})
    p1p = dict(phase1_best.get("preset", {}) or {})
    p1_mixed = _auto_safe_float(
        p1p.get("mixed_freq", setup.base_tc.get("mixed_freq", float("nan"))),
        float("nan"),
    )
    p1_phase = _auto_safe_float(
        p1p.get("phase_limit", setup.base_tc.get("phase_limit", float("nan"))),
        float("nan"),
    )
    p1_tdc = _auto_safe_float(
        p1p.get("tdc_strength", setup.base_tc.get("tdc_strength", float("nan"))),
        float("nan"),
    )
    p1_mode = _auto_safe_float(p1m.get("mode_ripple_db"), float("nan"))
    p1_boost = _auto_safe_float(p1m.get("max_net_boost_db"), float("nan"))
    p1_mode_txt = f"{p1_mode:.3f} dB" if np.isfinite(p1_mode) else "n/a"
    p1_boost_txt = f"{p1_boost:.2f} dB" if np.isfinite(p1_boost) else "n/a"
    if str(filter_key) == "mixed":
        p1_detail = f"mixed_freq={p1_mixed:.1f} Hz, tdc={p1_tdc:.1f}"
    elif str(filter_key) in ("linear", "asym"):
        p1_detail = (
            f"phase_limit={p1_phase:.1f} Hz, tdc={p1_tdc:.1f}"
            if np.isfinite(p1_phase)
            else f"phase_limit=n/a, tdc={p1_tdc:.1f}"
        )
    else:
        p1_detail = f"tdc={p1_tdc:.1f}"
    logger.info(
        "Automatic mode target Phase1 done: target=%s, avg_score=%.3f, %s",
        str(setup.hc_name),
        _auto_safe_float(p1m.get("avg_score"), 0.0),
        str(p1_detail),
    )
    if callable(status_cb):
        status_cb(
            "DecayCore automatic mode: Phase1 done "
            f"target={setup.hc_name}, rank={official_rank_score(p1m):.3f}, "
            f"avg_score={_auto_safe_float(p1m.get('avg_score'), 0.0):.3f}, "
            f"mode_ripple={p1_mode_txt}, boost={p1_boost_txt}, {p1_detail}"
        )

    local_shrink = float(
        _auto_adaptive_shrink_factor(
            top_list,
            base_shrink=float(AUTO_MODE_LOCAL_REFINEMENT_SHRINK),
            plateau_hit=False,
        )
    )
    local_trial_total = int(AUTO_MODE_LOCAL_REFINE_TRIALS_PER_TOP)
    anchor_configs = _build_target_local_refine_anchor_configs(
        top_list=top_list,
        setup=setup,
        filter_key=str(filter_key),
        local_shrink=float(local_shrink),
        local_trial_total=int(local_trial_total),
        ref_profile=ref_profile,
        status_cb=status_cb,
        accumulator=accumulator,
        runtime=runtime,
    )

    def _run_one_anchor(acfg: dict) -> tuple[int, list[dict]]:
        return (
            int(acfg["li"]),
            _run_target_trials(
                runtime=runtime,
                cfg=cfg,
                optimizer_backend=optimizer_backend,
                optuna_mod=optuna_mod,
                target_study_sig=target_study_sig,
                seed_target=seed_target,
                cands=list(acfg["local_candidates"]),
                base_tc=setup.base_tc,
                measurements=measurements,
                fs_v=int(fs_v),
                taps_v=int(taps_v),
                xos=xos,
                hpf=hpf,
                hc_f_arr=setup.hc_f,
                hc_m_arr=setup.hc_m,
                pin_obj=pin_obj,
                filter_key=filter_key,
                phase_tag=str(acfg["phase_tag"]),
                target_name=setup.hc_name,
                phase_kind="local",
                n_total_override=int(local_trial_total),
                seed_presets=list(acfg["local_seed_presets"]),
                optuna_builder=acfg["optuna_builder"],
                seed_to_params=acfg["seed_to_params"],
            ),
        )

    # Run all anchors in parallel — they are independent (different Optuna studies,
    # different candidate pools, results only compared after all complete).
    anchor_results: dict[int, list[dict]] = {}
    n_anchors = int(len(anchor_configs))
    if n_anchors <= 1:
        for acfg in anchor_configs:
            li, out = _run_one_anchor(acfg)
            anchor_results[li] = out
    else:
        logger.info(
            "Automatic mode target Local refine: running %d anchors in parallel for target=%s",
            n_anchors,
            str(setup.hc_name),
        )
        with ThreadPoolExecutor(max_workers=n_anchors) as executor:
            future_to_li = {executor.submit(_run_one_anchor, acfg): acfg["li"] for acfg in anchor_configs}
            for future in as_completed(future_to_li):
                try:
                    li, out = future.result()
                    anchor_results[li] = out
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
                    li = future_to_li[future]
                    logger.warning(
                        "Automatic mode target Local refine anchor %d failed: %s: %s",
                        int(li), type(exc).__name__, exc,
                    )
                    anchor_results[li] = []

    _apply_target_local_refine_results(
        anchor_results=anchor_results,
        accumulator=accumulator,
        setup=setup,
        local_trial_total=int(local_trial_total),
    )
    return accumulator


def _build_target_local_refine_anchor_configs(
    *,
    top_list: list[dict],
    setup: _TargetTrialSetup,
    filter_key: str,
    local_shrink: float,
    local_trial_total: int,
    ref_profile: dict,
    status_cb,
    accumulator: _TargetTrialAccumulator,
    runtime,
) -> list[dict]:
    anchor_configs: list[dict] = []
    for li, item in enumerate(top_list, start=1):
        center = dict(item.get("preset", {}) or {})
        c_mixed = _auto_safe_float(
            center.get("mixed_freq", setup.base_tc.get("mixed_freq", float("nan"))),
            float("nan"),
        )
        c_phase = _auto_safe_float(
            center.get("phase_limit", setup.base_tc.get("phase_limit", float("nan"))),
            float("nan"),
        )
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
            logger.info(
                "Automatic mode target Local refine: target=%s, center #%d, %s",
                str(setup.hc_name),
                int(li),
                str(local_detail),
            )
            if callable(status_cb):
                status_cb(
                    f"DecayCore automatic mode: Local refine target={setup.hc_name} "
                    f"center #{li} {local_detail}"
                )
        seed_li = int(setup.seed_tc + li * 100003)
        local_seed_presets = [
            _target_local_candidate_clip(
                c,
                ref_profile=ref_profile,
                base_tc=setup.base_tc,
                filter_key=filter_key,
            )
            for c in _build_auto_mode_candidates_local(
                setup.base_tc,
                center,
                1,
                seed_li,
                shrink=float(local_shrink),
                optimize_mag_low=bool(setup.base_tc.get("auto_optimize_low_bass_cut", True)),
            )
        ]
        local_candidates = []
        if not bool(setup.use_optuna_curve_trials):
            local_candidates = [
                _target_local_candidate_clip(
                    c,
                    ref_profile=ref_profile,
                    base_tc=setup.base_tc,
                    filter_key=filter_key,
                )
                for c in _build_auto_mode_candidates_local(
                    setup.base_tc,
                    center,
                    int(local_trial_total),
                    seed_li,
                    shrink=float(local_shrink),
                    optimize_mag_low=bool(setup.base_tc.get("auto_optimize_low_bass_cut", True)),
                )
            ]
        phase_tag = runtime.auto_optuna_scope_with_context(
            f"local_center_{li}_u1",
            center=dict(center or {}),
            shrink=float(local_shrink),
            extra={
                "filter_key": str(filter_key),
                "target_name": str(setup.hc_name),
            },
        )
        optuna_builder = None
        seed_to_params = None
        if bool(setup.use_optuna_curve_trials):
            _captured_base_tc = dict(setup.base_tc)
            _captured_center = dict(center)
            _captured_shrink = float(local_shrink)
            optuna_builder = (
                lambda tr,
                _b=_captured_base_tc,
                _c=_captured_center,
                _s=_captured_shrink: _target_local_candidate_clip(
                    _suggest_auto_mode_candidate_local_optuna(
                        _b, _c, tr, shrink=_s,
                        optimize_mag_low=bool(_b.get("auto_optimize_low_bass_cut", True)),
                    ),
                    ref_profile=ref_profile,
                    base_tc=setup.base_tc,
                    filter_key=filter_key,
                )
            )
            seed_to_params = (
                lambda preset,
                _b=_captured_base_tc,
                _c=_captured_center,
                _s=_captured_shrink: _seed_auto_mode_candidate_local_optuna_params(
                    _b, _c, preset, shrink=_s,
                    optimize_mag_low=bool(_b.get("auto_optimize_low_bass_cut", True)),
                )
            )
        anchor_configs.append({
            "li": int(li),
            "center": dict(center),
            "local_seed_presets": list(local_seed_presets),
            "local_candidates": list(local_candidates),
            "phase_tag": str(phase_tag),
            "optuna_builder": optuna_builder,
            "seed_to_params": seed_to_params,
        })
        accumulator.trials_total_count += int(local_trial_total)
    return anchor_configs


def _apply_target_local_refine_results(
    *,
    anchor_results: dict[int, list[dict]],
    accumulator: _TargetTrialAccumulator,
    setup: _TargetTrialSetup,
    local_trial_total: int,
) -> None:
    for li in sorted(anchor_results.keys()):
        local_out = anchor_results[li]
        for lc_idx, out in enumerate(local_out, start=1):
            if bool(out.get("ok", False)):
                met = dict(out.get("metrics", {}) or {})
                trial_preset = dict(out.get("preset", {}) or {})
                accumulator.ok_n += 1
                accumulator.rank_sum += _auto_safe_float(met.get("rank_score"), 0.0)
                accumulator.avg_score_sum += _auto_safe_float(met.get("avg_score"), 0.0)
                accumulator.curve_scored.append({"metrics": dict(met), "preset": dict(trial_preset)})
                if accumulator.best_metrics is None or _auto_rank_key(met) < _auto_rank_key(accumulator.best_metrics):
                    prev = dict(accumulator.best_metrics or {})
                    accumulator.best_metrics = dict(met)
                    accumulator.best_preset = dict(trial_preset)
                    logger.info(
                        "Automatic mode target Local refine winner improved: target=%s, avg_score=%.3f -> %.3f, rank_score=%.3f -> %.3f",
                        str(setup.hc_name),
                        _auto_safe_float(prev.get("avg_score"), 0.0),
                        _auto_safe_float(met.get("avg_score"), 0.0),
                        _auto_safe_float(prev.get("rank_score"), 0.0),
                        _auto_safe_float(met.get("rank_score"), 0.0),
                    )
            else:
                _target_trial_log_method(out=out)(
                    "Automatic mode target local trial %s: target=%s center=%d %d/%d (%s)",
                    _target_trial_issue_label(out=out),
                    str(setup.hc_name),
                    int(li),
                    int(lc_idx),
                    int(local_trial_total),
                    str(out.get("error", "unknown error") or "unknown error"),
                )


def _build_target_eval_core_result(
    *,
    tc: dict,
    accumulator: _TargetTrialAccumulator,
    setup: _TargetTrialSetup,
    goal: str,
) -> dict | None:
    if accumulator.ok_n <= 0 or not isinstance(accumulator.best_metrics, dict):
        return None
    final_best = _auto_select_best_scored(accumulator.curve_scored, goal=goal)
    best_metrics = dict(accumulator.best_metrics or {})
    best_preset = dict(accumulator.best_preset or {})
    if isinstance(final_best, dict):
        best_metrics = dict(final_best.get("metrics", {}) or {})
        best_preset = dict(final_best.get("preset", {}) or {})
    return {
        "hc_mode": str(setup.hc_name),
        "fit_rms_db": _auto_safe_float(tc.get("fit_rms_db"), 0.0),
        "offset_db": _auto_safe_float(tc.get("offset_db"), 0.0),
        "preselect_score": _auto_safe_float(
            tc.get("preselect_score", tc.get("fit_rms_db", float("inf"))),
            float("inf"),
        ),
        "boost_penalty": _auto_safe_float(tc.get("boost_penalty", 0.0), 0.0),
        "asym_penalty_db": _auto_safe_float(tc.get("asym_penalty_db", 0.0), 0.0),
        "mode_fit_rms_db": _auto_safe_float(tc.get("mode_fit_rms_db", 0.0), 0.0),
        "from_cache_wildcard": bool(tc.get("from_cache_wildcard", False)),
        "trials_total": int(accumulator.trials_total_count),
        "trials_ok": int(accumulator.ok_n),
        "avg_rank_score": float(accumulator.rank_sum / max(1, accumulator.ok_n)),
        "avg_avg_score": float(accumulator.avg_score_sum / max(1, accumulator.ok_n)),
        "best_metrics": best_metrics,
        "best_preset": best_preset,
    }


def _run_target_eval_trials_core(
    *,
    runtime,
    cfg,
    base_data: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    pin_obj,
    goal: str,
    filter_key: str,
    optimizer_backend: str,
    optuna_mod,
    seed_target: int,
    target_study_sig: str,
    trials_eff: int,
    shortlisted: list[dict],
    status_cb,
    f6_txt: str,
    tc: dict,
    t_idx: int,
    emit_status: bool,
    curve_inner_workers: int | None,
) -> dict | None:
    hc_name = str(tc.get("hc_mode", "") or "").strip()
    if not hc_name:
        return None
    setup = _prepare_target_trial_setup(
        runtime=runtime,
        base_data=base_data,
        measurements=measurements,
        fs_v=int(fs_v),
        taps_v=int(taps_v),
        xos=xos,
        hpf=hpf,
        filter_key=str(filter_key),
        optimizer_backend=str(optimizer_backend),
        optuna_mod=optuna_mod,
        trials_eff=int(trials_eff),
        tc=dict(tc or {}),
        hc_name=str(hc_name),
        curve_inner_workers=curve_inner_workers,
        emit_status=bool(emit_status),
    )
    if setup is None:
        return None
    accumulator = _run_target_phase1_trials(
        runtime=runtime,
        cfg=cfg,
        optimizer_backend=str(optimizer_backend),
        optuna_mod=optuna_mod,
        target_study_sig=str(target_study_sig),
        seed_target=int(seed_target),
        measurements=measurements,
        fs_v=int(fs_v),
        taps_v=int(taps_v),
        xos=xos,
        hpf=hpf,
        pin_obj=pin_obj,
        filter_key=str(filter_key),
        shortlisted=list(shortlisted or []),
        status_cb=status_cb,
        f6_txt=str(f6_txt),
        goal=str(goal),
        tc=dict(tc or {}),
        t_idx=int(t_idx),
        emit_status=bool(emit_status),
        setup=setup,
    )
    accumulator = _run_target_local_refine_trials(
        runtime=runtime,
        cfg=cfg,
        optimizer_backend=str(optimizer_backend),
        optuna_mod=optuna_mod,
        target_study_sig=str(target_study_sig),
        seed_target=int(seed_target),
        measurements=measurements,
        fs_v=int(fs_v),
        taps_v=int(taps_v),
        xos=xos,
        hpf=hpf,
        pin_obj=pin_obj,
        filter_key=str(filter_key),
        goal=str(goal),
        status_cb=status_cb,
        setup=setup,
        accumulator=accumulator,
    )
    return _build_target_eval_core_result(
        tc=dict(tc or {}),
        accumulator=accumulator,
        setup=setup,
        goal=str(goal),
    )
