# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Shortlist selection helpers for target-curve selection orchestration."""

from __future__ import annotations

import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from .worker_init import _auto_worker_init

import sys as _sys
_USE_PROCESS_POOL = _sys.platform != "win32"

# Fork-inherited shared state for curve-parallel ProcessPool.
# Set in _run_shortlist_curve_parallel before pool creation.
_PROC_CURVE_ARGS: dict | None = None

import numpy as np

from .rank_score import official_rank_score
from .scoring_ranking import (
    _auto_goal_uses_local_refine,
    _auto_select_best_scored,
    _tc_score,
)
from .shared import (
    AUTO_MODE_LOCAL_REFINE_ENABLED,
    AUTO_MODE_LOCAL_REFINE_TOP_K,
    AUTO_MODE_LOCAL_REFINE_TRIALS_PER_TOP,
    AUTO_MODE_TARGET_BEST_RANK_TIE_EPS,
    AUTO_MODE_TARGET_CACHE_AS_WILDCARD,
    AUTO_MODE_TARGET_MILDER_MAX_ASYM_ADD,
    AUTO_MODE_TARGET_MILDER_MAX_DIFFICULTY_ADD,
    AUTO_MODE_TARGET_MILDER_MAX_FIT_RMS_ADD_DB,
    AUTO_MODE_TARGET_PREFER_MILDER_STEP,
    AUTO_MODE_TARGET_LOCAL_REFINE_ENABLED,
    AUTO_MODE_TARGET_TOP_N,
    AUTO_MODE_TARGET_TOP_N_SPREAD_DB,
    AUTO_MODE_TARGET_TRIALS_PER_CURVE,
    _auto_safe_bool,
    _auto_safe_float,
)
from .target_preselection import (
    _auto_target_adaptive_shortlist,
    _auto_target_insert_cached_wildcard,
    _auto_target_one_step_milder,
)
from .orchestrator_target_types import (
    _TargetCacheState,
    _TargetSelectionSetup,
    _TargetShortlistState,
)
from .orchestrator_target_cache import (
    _cache_target_valid,
    _auto_target_mode_locks_hc,
    _fallback_to_cached_target,
)
from .orchestrator_target_trials import (
    _evaluate_target_curve,
    _materialize_target_candidate,
    _run_target_eval_trials,
    _summarize_target_eval,
    _build_target_eval_result,
)

logger = logging.getLogger("DecayCore")


def _target_local_refine_trial_load(*, goal: str) -> int:
    if not (
        bool(AUTO_MODE_LOCAL_REFINE_ENABLED)
        and bool(AUTO_MODE_TARGET_LOCAL_REFINE_ENABLED)
        and _auto_goal_uses_local_refine(goal)
    ):
        return 0
    return int(AUTO_MODE_LOCAL_REFINE_TOP_K) * int(AUTO_MODE_LOCAL_REFINE_TRIALS_PER_TOP)


def _target_curve_worker_plan(*, shortlist_state: _TargetShortlistState, runtime, base_data: dict, goal: str) -> tuple[int, int]:
    local_refine_load = _target_local_refine_trial_load(goal=goal)
    per_curve_load = int(shortlist_state.trials_eff) + int(local_refine_load)
    total_target_trial_load = int(max(1, len(shortlist_state.shortlisted) * max(1, per_curve_load)))
    curve_budget = int(runtime.auto_trial_workers(base_data, total_target_trial_load))
    curve_workers = int(max(1, min(len(shortlist_state.shortlisted), curve_budget)))
    curve_inner_workers = int(max(1, curve_budget // max(1, curve_workers)))
    return int(curve_workers), int(curve_inner_workers)


def _target_curve_progress_status(
    *,
    status_cb,
    done_n: int,
    total_n: int,
    shortlist_state: _TargetShortlistState,
    setup: _TargetSelectionSetup,
    select_f6_txt: str,
    best_done_item: dict,
    hc_name: str,
) -> None:
    if not callable(status_cb):
        return
    bm_now = dict(best_done_item.get("best_metrics", {}) or {})
    status_cb(
        "DecayCore automatic mode: selecting target curve "
        f"(best improved {int(done_n)}/{int(total_n)}, "
        f"leader {str(best_done_item.get('hc_mode', 'n/a') or 'n/a')}, "
        f"tested {str(hc_name or 'n/a')}, "
        f"{int(shortlist_state.trials_eff)} trials/curve{select_f6_txt}, goal {setup.goal}, "
        f"rank {official_rank_score(bm_now):.3f}, "
        f"avg {_auto_safe_float(bm_now.get('avg_score'), 0.0):.3f})"
    )


def _target_curve_submit_parallel_job(
    *,
    executor,
    tc: dict,
    t_idx: int,
    setup: _TargetSelectionSetup,
    base_data: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    pin_obj,
    shortlist_state: _TargetShortlistState,
    status_cb,
    curve_inner_workers: int,
):
    if _USE_PROCESS_POOL:
        from .orchestrator_target_trials import _evaluate_target_curve_proc

        return executor.submit(
            _evaluate_target_curve_proc,
            tc=dict(tc or {}),
            t_idx=int(t_idx),
        )
    return executor.submit(
        _evaluate_target_curve,
        runtime=setup.runtime,
        cfg=setup.cfg,
        base_data=base_data,
        measurements=measurements,
        fs_v=int(fs_v),
        taps_v=int(taps_v),
        xos=xos,
        hpf=hpf,
        pin_obj=pin_obj,
        goal=setup.goal,
        filter_key=setup.filter_key,
        optimizer_backend=setup.optimizer_backend,
        optuna_mod=setup.optuna_mod,
        seed_target=int(setup.seed_target),
        target_study_sig=setup.target_study_sig,
        trials_eff=int(shortlist_state.trials_eff),
        shortlisted=shortlist_state.shortlisted,
        status_cb=status_cb,
        f6_txt=shortlist_state.f6_txt,
        tc=dict(tc or {}),
        t_idx=int(t_idx),
        emit_status=False,
        curve_inner_workers=int(curve_inner_workers),
    )


def _run_shortlist_curve_parallel(
    *,
    setup: _TargetSelectionSetup,
    shortlist_state: _TargetShortlistState,
    base_data: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    pin_obj,
    status_cb,
    select_f6_txt: str,
    curve_workers: int,
    curve_inner_workers: int,
) -> list[dict]:
    evaluated: list[dict] = []
    _ExecutorCls = ProcessPoolExecutor if _USE_PROCESS_POOL else ThreadPoolExecutor
    if _USE_PROCESS_POOL:
        _ctx = multiprocessing.get_context("fork")
        _executor_kwargs = {"mp_context": _ctx, "initializer": _auto_worker_init}
    else:
        _executor_kwargs = {}
    if _USE_PROCESS_POOL:
        global _PROC_CURVE_ARGS
        _PROC_CURVE_ARGS = {
            "runtime": setup.runtime,
            "cfg": setup.cfg,
            "base_data": base_data,
            "measurements": measurements,
            "fs_v": int(fs_v),
            "taps_v": int(taps_v),
            "xos": xos,
            "hpf": hpf,
            "goal": setup.goal,
            "filter_key": setup.filter_key,
            "optimizer_backend": setup.optimizer_backend,
            "optuna_mod": setup.optuna_mod,
            "seed_target": int(setup.seed_target),
            "target_study_sig": setup.target_study_sig,
            "trials_eff": int(shortlist_state.trials_eff),
            "shortlisted": shortlist_state.shortlisted,
            "f6_txt": shortlist_state.f6_txt,
            "curve_inner_workers": int(curve_inner_workers),
        }
    with _ExecutorCls(max_workers=int(curve_workers), **_executor_kwargs) as executor:
        future_map = {}
        best_done_item = None
        done_n = 0
        for t_idx, tc in enumerate(shortlist_state.shortlisted, start=1):
            future = _target_curve_submit_parallel_job(
                executor=executor,
                tc=dict(tc or {}),
                t_idx=int(t_idx),
                setup=setup,
                base_data=base_data,
                measurements=measurements,
                fs_v=int(fs_v),
                taps_v=int(taps_v),
                xos=xos,
                hpf=hpf,
                pin_obj=pin_obj,
                shortlist_state=shortlist_state,
                status_cb=status_cb,
                curve_inner_workers=int(curve_inner_workers),
            )
            future_map[future] = (
                int(t_idx),
                str((tc or {}).get("hc_mode", "") or "").strip(),
            )
        for future in as_completed(list(future_map.keys())):
            _t_idx, hc_name = future_map.get(future, (0, "n/a"))
            done_n += 1
            improved = False
            try:
                item = future.result()
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
                logger.warning(
                    "Automatic mode target curve failed: target=%s (%s)",
                    str(hc_name),
                    f"{type(exc).__name__}: {exc}",
                )
                item = None
            if isinstance(item, dict):
                item_d = dict(item)
                evaluated.append(dict(item_d))
                evaluated_scored = [
                    {
                        **dict(it or {}),
                        "_auto_select_kind": "target_curve",
                        "_target_rank_tie_eps": float(
                            max(
                                0.0,
                                _auto_safe_float(AUTO_MODE_TARGET_BEST_RANK_TIE_EPS, 0.05),
                            )
                        ),
                    }
                    for it in evaluated
                ]
                selected_done_item = _auto_select_best_scored(evaluated_scored)
                if isinstance(selected_done_item, dict) and (
                    not isinstance(best_done_item, dict)
                    or str(selected_done_item.get("hc_mode", "") or "").strip()
                    != str(best_done_item.get("hc_mode", "") or "").strip()
                ):
                    best_done_item = dict(selected_done_item)
                    improved = True
            if bool(improved) and isinstance(best_done_item, dict):
                _target_curve_progress_status(
                    status_cb=status_cb,
                    done_n=int(done_n),
                    total_n=len(shortlist_state.shortlisted),
                    shortlist_state=shortlist_state,
                    setup=setup,
                    select_f6_txt=select_f6_txt,
                    best_done_item=dict(best_done_item or {}),
                    hc_name=str(hc_name),
                )
    return list(evaluated)


def _run_shortlist_curve_serial(
    *,
    setup: _TargetSelectionSetup,
    shortlist_state: _TargetShortlistState,
    base_data: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    pin_obj,
    status_cb,
    select_f6_txt: str,
) -> list[dict]:
    evaluated: list[dict] = []
    for t_idx, tc in enumerate(shortlist_state.shortlisted, start=1):
        hc_name = str((tc or {}).get("hc_mode", "") or "").strip() or "n/a"
        if callable(status_cb):
            status_cb(
                "DecayCore automatic mode: selecting target curve "
                f"(testing {hc_name} {t_idx}/{len(shortlist_state.shortlisted)}, "
                f"{int(shortlist_state.trials_eff)} trials/curve{select_f6_txt}, goal {setup.goal})"
            )
        item = _evaluate_target_curve(
            runtime=setup.runtime,
            cfg=setup.cfg,
            base_data=base_data,
            measurements=measurements,
            fs_v=int(fs_v),
            taps_v=int(taps_v),
            xos=xos,
            hpf=hpf,
            pin_obj=pin_obj,
            goal=setup.goal,
            filter_key=setup.filter_key,
            optimizer_backend=setup.optimizer_backend,
            optuna_mod=setup.optuna_mod,
            seed_target=int(setup.seed_target),
            target_study_sig=setup.target_study_sig,
            trials_eff=int(shortlist_state.trials_eff),
            shortlisted=shortlist_state.shortlisted,
            status_cb=status_cb,
            f6_txt=shortlist_state.f6_txt,
            tc=dict(tc or {}),
            t_idx=int(t_idx),
            emit_status=True,
            curve_inner_workers=None,
        )
        if isinstance(item, dict):
            evaluated.append(dict(item))
    return list(evaluated)


def _apply_cached_wildcard_to_shortlist(
    *,
    setup: _TargetSelectionSetup,
    cache_state: _TargetCacheState,
    shortlist_state: _TargetShortlistState,
    shortlisted: list[dict],
    base_data: dict,
    measurements: dict,
    status_cb,
) -> tuple[list[dict], dict | None]:
    if not (
        _auto_safe_bool(AUTO_MODE_TARGET_CACHE_AS_WILDCARD, True)
        and _cache_target_valid(setup.runtime, cache_state.cached_target_hc)
    ):
        return list(shortlisted), None
    shortlisted, cache_meta = _auto_target_insert_cached_wildcard(
        shortlisted,
        shortlist_state.quick_candidates,
        cached_hc_mode=str(cache_state.cached_target_hc),
    )
    shortlist_state.cache_wildcard_participated = bool(
        cache_meta.get("inserted", False)
        or cache_meta.get("already_present", False)
    )
    if bool(cache_meta.get("inserted", False)):
        logger.info(
            "Automatic mode target shortlist: inserted cache wildcard target=%s",
            str(cache_meta.get("hc_mode", cache_state.cached_target_hc)),
        )
        if callable(status_cb):
            status_cb(
                "DecayCore automatic mode: target shortlist cache wildcard inserted "
                f"({str(cache_meta.get('hc_mode', cache_state.cached_target_hc))})"
            )
        return list(shortlisted), None
    if str(cache_meta.get("reason", "")) == "already_shortlisted" and str(
        cache_state.cached_target_source or ""
    ).strip() in (
        "cache_measurement",
        "cache_signature",
        "cache_optuna_target",
    ) and _cache_target_valid(setup.runtime, cache_state.cached_target_hc):
        can_bypass_shortlist = _auto_target_mode_locks_hc(base_data)
        if can_bypass_shortlist:
            logger.info(
                "Automatic mode target shortlist: already_shortlisted bypass, "
                "returning cached target=%s source=%s",
                str(cache_state.cached_target_hc),
                str(cache_state.cached_target_source),
            )
            if callable(status_cb):
                status_cb(
                    "DecayCore automatic mode: target loaded directly from cache "
                    f"(already_shortlisted -> {str(cache_meta.get('hc_mode', cache_state.cached_target_hc))}, "
                    "skipping shortlist trials)"
                )
            return list(shortlisted), _fallback_to_cached_target(
                setup=setup,
                cache_state=cache_state,
                base_data=base_data,
                measurements=measurements,
            )
    logger.info(
        "Automatic mode target shortlist: cache wildcard skipped target=%s (%s)",
        str(cache_meta.get("hc_mode", cache_state.cached_target_hc)),
        str(cache_meta.get("reason", "unknown")),
    )
    if callable(status_cb):
        status_cb(
            "DecayCore automatic mode: target shortlist cache wildcard "
            f"{str(cache_meta.get('reason', 'skipped'))}"
        )
    return list(shortlisted), None


def _try_include_milder_target(
    *,
    shortlist_state: _TargetShortlistState,
    shortlisted: list[dict],
    base_data: dict,
    status_cb,
) -> list[dict]:
    prefer_milder = _auto_safe_bool(
        base_data.get(
            "auto_target_prefer_milder_step",
            AUTO_MODE_TARGET_PREFER_MILDER_STEP,
        ),
        AUTO_MODE_TARGET_PREFER_MILDER_STEP,
    )
    if not (prefer_milder and shortlisted):
        return list(shortlisted)
    leader = dict(shortlisted[0] or {})
    lead_hc = str(leader.get("hc_mode", "") or "").strip()
    lead_milder = _auto_target_one_step_milder(lead_hc)
    if not lead_milder:
        logger.info(
            "Automatic mode target shortlist: no one-step milder target for %s",
            str(lead_hc),
        )
        return list(shortlisted)
    milder_tc = None
    for tc in shortlist_state.quick_candidates:
        if str(tc.get("hc_mode", "") or "").strip() == str(lead_milder):
            milder_tc = dict(tc)
            break
    if not isinstance(milder_tc, dict):
        logger.info(
            "Automatic mode target shortlist: milder target for %s not found in quick candidates",
            str(lead_hc),
        )
        return list(shortlisted)
    already = {
        str(tc.get("hc_mode", "") or "").strip()
        for tc in shortlisted
        if isinstance(tc, dict)
    }
    leader_fit = _auto_safe_float(leader.get("fit_rms_db", float("inf")), float("inf"))
    milder_fit = _auto_safe_float(milder_tc.get("fit_rms_db", float("inf")), float("inf"))
    leader_pre = _tc_score(leader)
    milder_pre = _tc_score(milder_tc)
    leader_asym = _auto_safe_float(leader.get("asym_penalty_db", 0.0), 0.0)
    milder_asym = _auto_safe_float(milder_tc.get("asym_penalty_db", 0.0), 0.0)
    leader_boost = _auto_safe_float(leader.get("boost_penalty", 0.0), 0.0)
    milder_boost = _auto_safe_float(milder_tc.get("boost_penalty", 0.0), 0.0)
    cond_not_dup = str(lead_milder) not in already
    cond_fit = bool(
        float(milder_fit)
        <= float(leader_fit) + float(AUTO_MODE_TARGET_MILDER_MAX_FIT_RMS_ADD_DB)
    )
    cond_diff = bool(
        float(milder_pre)
        <= float(leader_pre) + float(AUTO_MODE_TARGET_MILDER_MAX_DIFFICULTY_ADD)
    )
    cond_asym = bool(
        float(milder_asym)
        <= float(leader_asym) + float(AUTO_MODE_TARGET_MILDER_MAX_ASYM_ADD)
    )
    if cond_not_dup and cond_fit and cond_diff and cond_asym:
        shortlisted = list(shortlisted) + [dict(milder_tc)]
        logger.info(
            "Automatic mode target shortlist: included milder target %s -> %s (fit %.3f->%.3f, pre %.3f->%.3f, boost %.3f->%.3f, asym %.3f->%.3f)",
            str(lead_hc),
            str(lead_milder),
            float(leader_fit),
            float(milder_fit),
            float(leader_pre),
            float(milder_pre),
            float(leader_boost),
            float(milder_boost),
            float(leader_asym),
            float(milder_asym),
        )
        if callable(status_cb):
            status_cb(
                "DecayCore automatic mode: target shortlist milder included "
                f"({str(lead_hc)} -> {str(lead_milder)})"
            )
        return list(shortlisted)
    logger.info(
        "Automatic mode target shortlist: skipped milder target %s -> %s (not_dup=%s, fit_ok=%s, pre_ok=%s, asym_ok=%s, boost %.3f->%.3f)",
        str(lead_hc),
        str(lead_milder),
        str(cond_not_dup),
        str(cond_fit),
        str(cond_diff),
        str(cond_asym),
        float(leader_boost),
        float(milder_boost),
    )
    if callable(status_cb):
        status_cb(
            "DecayCore automatic mode: target shortlist milder skipped "
            f"({str(lead_hc)} -> {str(lead_milder)}, "
            f"not_dup={str(cond_not_dup)}, fit_ok={str(cond_fit)}, "
            f"pre_ok={str(cond_diff)}, asym_ok={str(cond_asym)})"
        )
    return list(shortlisted)


def _dedup_shortlist_by_hc(shortlisted: list[dict]) -> list[dict]:
    dedup_shortlisted = []
    seen_hc = set()
    for tc in shortlisted:
        if not isinstance(tc, dict):
            continue
        hc = str(tc.get("hc_mode", "") or "").strip()
        if not hc or hc in seen_hc:
            continue
        seen_hc.add(hc)
        dedup_shortlisted.append(dict(tc))
    return list(dedup_shortlisted)


def _populate_shortlist_defaults(shortlisted: list[dict]) -> None:
    for tc in shortlisted:
        tc.setdefault("preselect_score", _tc_score(tc))
        tc.setdefault("boost_penalty", _auto_safe_float(tc.get("boost_penalty", 0.0), 0.0))
        tc.setdefault("asym_penalty_db", _auto_safe_float(tc.get("asym_penalty_db", 0.0), 0.0))


def _load_quick_target_selection(
    *,
    setup: _TargetSelectionSetup,
    cache_state: _TargetCacheState,
    base_data: dict,
    measurements: dict,
    top_n: int,
    trials_per_curve: int,
    status_cb,
) -> _TargetShortlistState | dict | None:
    f6_hz = _auto_safe_float(
        base_data.get("_auto_mag_c_min_hz", base_data.get("mag_c_min", float("nan"))),
        float("nan"),
    )
    f6_txt = f" (-6 dB {f6_hz:.1f} Hz)" if np.isfinite(f6_hz) else ""
    quick = setup.runtime.auto_select_builtin_target_curve(
        base_data,
        f_l=measurements.get("f_l"),
        m_l=measurements.get("m_l"),
        f_r=measurements.get("f_r"),
        m_r=measurements.get("m_r"),
    )
    if not isinstance(quick, dict):
        fallback = _fallback_to_cached_target(
            setup=setup,
            cache_state=cache_state,
            base_data=base_data,
            measurements=measurements,
        )
        if isinstance(fallback, dict):
            logger.info(
                "Automatic mode target select: quick preselect unavailable, fallback to cached target=%s",
                str(cache_state.cached_target_hc),
            )
            return dict(fallback)
        return None
    quick_candidates = list(
        quick.get("candidates_all", quick.get("candidates", []))
        or quick.get("candidates", [])
        or []
    )
    quick_candidates = [dict(tc or {}) for tc in quick_candidates if isinstance(tc, dict)]
    if not quick_candidates:
        fallback = _fallback_to_cached_target(
            setup=setup,
            cache_state=cache_state,
            base_data=base_data,
            measurements=measurements,
        )
        if isinstance(fallback, dict):
            logger.info(
                "Automatic mode target select: no quick candidates, fallback to cached target=%s",
                str(cache_state.cached_target_hc),
            )
            return dict(fallback)
        return None
    quick_candidates = sorted(
        quick_candidates,
        key=lambda tc: (
            _tc_score(tc),
            _auto_safe_float(tc.get("fit_rms_db", float("inf")), float("inf")),
            str(tc.get("hc_mode", "") or "").strip(),
        ),
    )
    quick_rows = [
        f"{str(tc.get('hc_mode', 'n/a'))}: "
        f"fit={_auto_safe_float(tc.get('fit_rms_db', float('nan')), float('nan')):.3f} dB, "
        f"pre={_tc_score(tc):.3f}, "
        f"boost={_auto_safe_float(tc.get('boost_penalty', 0.0), 0.0):.3f}, "
        f"asym={_auto_safe_float(tc.get('asym_penalty_db', 0.0), 0.0):.3f}"
        for tc in quick_candidates
    ]
    logger.info(
        "Automatic mode target preselect candidates:\n%s",
        "\n".join(quick_rows),
    )
    if callable(status_cb):
        top3_txt = ", ".join(
            [
                (
                    f"{str(tc.get('hc_mode', 'n/a') or 'n/a')}"
                    f"(pre={_tc_score(tc):.3f}, fit={_auto_safe_float(tc.get('fit_rms_db', 0.0), 0.0):.3f}, "
                    f"boost={_auto_safe_float(tc.get('boost_penalty', 0.0), 0.0):.3f}, "
                    f"asym={_auto_safe_float(tc.get('asym_penalty_db', 0.0), 0.0):.3f})"
                )
                for tc in quick_candidates[:3]
            ]
        )
        status_cb(
            "DecayCore automatic mode: target preselect top-3 "
            f"(goal {setup.goal}) {top3_txt}"
        )
    shortlisted, shortlist_meta = _auto_target_adaptive_shortlist(
        quick_candidates,
        top_n=int(top_n),
    )
    if not shortlisted:
        fallback = _fallback_to_cached_target(
            setup=setup,
            cache_state=cache_state,
            base_data=base_data,
            measurements=measurements,
        )
        if isinstance(fallback, dict):
            return dict(fallback)
        return None
    logger.info(
        "Automatic mode target shortlist: n=%d/%d (base_top_n=%d, spread_based_n=%d, spread=%.3f dB, best_score=%.3f)",
        int(shortlist_meta.get("shortlist_n", len(shortlisted))),
        int(shortlist_meta.get("candidate_total", len(quick_candidates))),
        int(shortlist_meta.get("top_n_eff", max(1, int(top_n)))),
        int(shortlist_meta.get("spread_based_n", len(shortlisted))),
        float(
            _auto_safe_float(
                shortlist_meta.get("spread_db", AUTO_MODE_TARGET_TOP_N_SPREAD_DB),
                AUTO_MODE_TARGET_TOP_N_SPREAD_DB,
            )
        ),
        float(
            _auto_safe_float(
                shortlist_meta.get("best_score", _tc_score(shortlisted[0])),
                _tc_score(shortlisted[0]),
            )
        ),
    )
    if callable(status_cb):
        status_cb(
            "DecayCore automatic mode: target shortlist "
            f"(selected {int(shortlist_meta.get('shortlist_n', len(shortlisted)))}/"
            f"{int(shortlist_meta.get('candidate_total', len(quick_candidates)))} "
            f"by spread {float(_auto_safe_float(shortlist_meta.get('spread_db', AUTO_MODE_TARGET_TOP_N_SPREAD_DB), AUTO_MODE_TARGET_TOP_N_SPREAD_DB)):.2f} dB)"
        )
    return _TargetShortlistState(
        quick=dict(quick or {}),
        quick_candidates=list(quick_candidates or []),
        shortlisted=list(shortlisted or []),
        trials_eff=max(1, int(trials_per_curve)),
        f6_hz=float(f6_hz),
        f6_txt=str(f6_txt),
    )


def _apply_target_shortlist_modifiers(
    *,
    setup: _TargetSelectionSetup,
    cache_state: _TargetCacheState,
    shortlist_state: _TargetShortlistState,
    base_data: dict,
    measurements: dict,
    status_cb,
) -> _TargetShortlistState | dict | None:
    shortlisted = list(shortlist_state.shortlisted or [])
    shortlisted, bypass = _apply_cached_wildcard_to_shortlist(
        setup=setup,
        cache_state=cache_state,
        shortlist_state=shortlist_state,
        shortlisted=shortlisted,
        base_data=base_data,
        measurements=measurements,
        status_cb=status_cb,
    )
    if isinstance(bypass, dict):
        return dict(bypass)
    shortlisted = _try_include_milder_target(
        shortlist_state=shortlist_state,
        shortlisted=shortlisted,
        base_data=base_data,
        status_cb=status_cb,
    )
    shortlist_state.shortlisted = _dedup_shortlist_by_hc(shortlisted)
    if not shortlist_state.shortlisted:
        fallback = _fallback_to_cached_target(
            setup=setup,
            cache_state=cache_state,
            base_data=base_data,
            measurements=measurements,
        )
        if isinstance(fallback, dict):
            return dict(fallback)
        return None
    _populate_shortlist_defaults(shortlist_state.shortlisted)
    return shortlist_state


def _evaluate_target_shortlist_core(
    *,
    setup: _TargetSelectionSetup,
    shortlist_state: _TargetShortlistState,
    base_data: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    pin_obj,
    status_cb,
) -> list[dict]:
    curve_workers, curve_inner_workers = _target_curve_worker_plan(
        shortlist_state=shortlist_state,
        runtime=setup.runtime,
        base_data=base_data,
        goal=setup.goal,
    )
    select_f6_txt = f", -6 dB point {shortlist_state.f6_hz:.1f} Hz" if np.isfinite(shortlist_state.f6_hz) else ""
    if curve_workers > 1:
        logger.info(
            "Automatic mode target select: curve-parallel enabled (curves=%d, workers=%d, inner_workers=%d)",
            int(len(shortlist_state.shortlisted)),
            int(curve_workers),
            int(curve_inner_workers),
        )
        return _run_shortlist_curve_parallel(
            setup=setup,
            shortlist_state=shortlist_state,
            base_data=base_data,
            measurements=measurements,
            fs_v=int(fs_v),
            taps_v=int(taps_v),
            xos=xos,
            hpf=hpf,
            pin_obj=pin_obj,
            status_cb=status_cb,
            select_f6_txt=select_f6_txt,
            curve_workers=int(curve_workers),
            curve_inner_workers=int(curve_inner_workers),
        )
    return _run_shortlist_curve_serial(
        setup=setup,
        shortlist_state=shortlist_state,
        base_data=base_data,
        measurements=measurements,
        fs_v=int(fs_v),
        taps_v=int(taps_v),
        xos=xos,
        hpf=hpf,
        pin_obj=pin_obj,
        status_cb=status_cb,
        select_f6_txt=select_f6_txt,
    )


