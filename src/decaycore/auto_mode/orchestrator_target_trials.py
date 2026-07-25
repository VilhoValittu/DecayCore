# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Trial execution helpers for target-curve selection orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from .worker_init import _auto_worker_init

import sys as _sys
_USE_PROCESS_POOL = _sys.platform != "win32"

# Fork-inherited runtime — set in _run_target_trials before pool creation.
# Workers read this global directly, bypassing pickle.
_PROC_RUNTIME = None
# Fork-inherited per-run compute context — set alongside _PROC_RUNTIME.
_PROC_COMPUTE_CTX = None

import numpy as np

from .cache_signature import (
    _auto_apply_seed,
    _auto_compat_version,
    _auto_seed_from_signature,
)
from .compute_context import AutoRunComputeContext
from .cache_measurement_sig import _auto_get_measurement_signature, _auto_search_measurement_identity
from .candidate_generation import _build_auto_mode_candidates
from .search_v2.candidates import deduplicate_presets
from .shared_parts import (
    MAX_SAFE_BOOST,
    _auto_phase_limit_center,
    _auto_phase_limit_clip,
    _auto_safe_float,
    _auto_trial_chunk_size,
    _clip,
)
from .orchestrator_target_types import (
    _TargetEvalMaterialization,
    _TargetEvalSummary,
    _TargetTrialSetup,
)

logger = logging.getLogger("DecayCore")


@dataclass
class _TargetTrialDispatch:
    runtime: object
    cfg: object
    optimizer_backend: str
    optuna_mod: object
    target_study_sig: str
    seed_target: int
    base_tc: dict
    measurements: dict
    fs_v: int
    taps_v: int
    xos: list
    hpf: dict | None
    hc_f_arr: object
    hc_m_arr: object
    pin_obj: object
    filter_key: str
    phase_tag: str
    target_name: str
    phase_kind: str | None
    n_total_override: int | None
    seed_presets: list[dict]
    optuna_builder: object
    seed_to_params: object
    compute_ctx: AutoRunComputeContext
    use_optuna_trials: bool
    eval_candidates: list[dict]
    n_total: int
    workers: int


def _target_eval_one(
    *,
    runtime,
    preset: dict,
    base_tc: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    hc_f_arr,
    hc_m_arr,
    pin_obj,
    filter_key: str,
    _compute_ctx: AutoRunComputeContext | None = None,
    _trial_idx: int = 0,
) -> dict:
    if _compute_ctx is not None:
        _compute_ctx.begin_trial(int(_trial_idx))
    trial_data = dict(base_tc)
    trial_data.update(dict(preset or {}))
    if str(filter_key) in ("linear", "asym"):
        trial_data["phase_limit"] = round(
            float(
                _auto_phase_limit_clip(
                    trial_data.get("phase_limit", base_tc.get("phase_limit", 400.0)),
                    default=400.0,
                )
            ),
            1,
        )
    trial_data["comparison_mode"] = True
    trial_measurements = dict(measurements or {})
    trial_measurements["ui_data"] = trial_data

    cfg = runtime.build_config(
        trial_data,
        fs_v=int(fs_v),
        taps_v=int(taps_v),
        xos=xos,
        hpf=hpf,
        hc_f=hc_f_arr,
        hc_m=hc_m_arr,
        max_safe_boost=float(MAX_SAFE_BOOST),
    )
    try:
        cfg.bass_smooth_w_gamma = float(trial_data.get("bass_smooth_w_gamma", 2.4))
        cfg.bass_smooth_w_max = float(trial_data.get("bass_smooth_w_max", 0.45))
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
    ):
        # Optional trial-only smoothing knobs may be absent on older config objects.
        pass

    result = runtime.run_pipeline(
        cfg,
        trial_measurements,
        include_response_arrays=False,
    )
    metrics = runtime.auto_score_result(
        result,
        auto_exc_freq_hz=_auto_safe_float(
            trial_data.get("_auto_exc_freq_hz", float("nan")),
            float("nan"),
        ),
        base_data=trial_data,
    )
    trial_preset = dict(preset or {})
    if str(filter_key) == "mixed":
        trial_preset["mixed_freq"] = round(
            _clip(
                trial_data.get("mixed_freq", base_tc.get("mixed_freq", 180.0)),
                80.0,
                320.0,
            ),
            1,
        )
    elif str(filter_key) in ("linear", "asym"):
        trial_preset["phase_limit"] = round(
            float(
                _auto_phase_limit_clip(
                    trial_data.get("phase_limit", base_tc.get("phase_limit", 400.0)),
                    default=400.0,
                )
            ),
            1,
        )
    return {
        "ok": True,
        "metrics": dict(metrics or {}),
        "preset": dict(trial_preset),
    }


def _target_eval_one_proc(
    *,
    preset: dict,
    base_tc: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf,
    hc_f_arr,
    hc_m_arr,
    filter_key: str,
    _trial_idx: int = 0,
) -> dict:
    """ProcessPool wrapper: uses fork-inherited _PROC_RUNTIME instead of pickled runtime."""
    return _target_eval_one(
        runtime=_PROC_RUNTIME,
        preset=preset,
        base_tc=base_tc,
        measurements=measurements,
        fs_v=int(fs_v),
        taps_v=int(taps_v),
        xos=xos,
        hpf=hpf,
        hc_f_arr=hc_f_arr,
        hc_m_arr=hc_m_arr,
        pin_obj=None,
        filter_key=filter_key,
        _compute_ctx=_PROC_COMPUTE_CTX,
        _trial_idx=int(_trial_idx),
    )


def _prepare_target_trial_dispatch(
    *,
    runtime,
    cfg,
    optimizer_backend: str,
    optuna_mod,
    target_study_sig: str,
    seed_target: int,
    cands: list[dict],
    base_tc: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    hc_f_arr,
    hc_m_arr,
    pin_obj,
    filter_key: str,
    phase_tag: str,
    target_name: str,
    phase_kind: str | None,
    n_total_override: int | None,
    seed_presets: list[dict] | None,
    optuna_builder,
    seed_to_params,
) -> _TargetTrialDispatch:
    compute_ctx = AutoRunComputeContext(
        fs=int(fs_v),
        taps=int(taps_v),
        measurement_signature=_auto_get_measurement_signature(measurements),
        filter_key=str(filter_key or ""),
        compat_version=str(_auto_compat_version(base_tc)),
    )
    use_optuna_trials = bool(
        str(optimizer_backend) == "optuna"
        and runtime.auto_optuna_module_ready(optuna_mod)
        and callable(optuna_builder)
    )
    seed_presets_eff = list(seed_presets or [])
    eval_candidates = list(cands or [])
    if not bool(use_optuna_trials):
        eval_candidates = deduplicate_presets([*seed_presets_eff, *eval_candidates])
    n_total = int(n_total_override) if n_total_override is not None else int(len(eval_candidates))
    workers = int(runtime.auto_trial_workers(base_tc, n_total)) if n_total > 0 else 0
    if workers > 1:
        logger.info(
            "Automatic mode target trials: target=%s, phase=%s, parallel workers=%d",
            str(target_name),
            str(phase_tag),
            int(workers),
        )
    return _TargetTrialDispatch(
        runtime=runtime,
        cfg=cfg,
        optimizer_backend=str(optimizer_backend),
        optuna_mod=optuna_mod,
        target_study_sig=str(target_study_sig),
        seed_target=int(seed_target),
        base_tc=dict(base_tc or {}),
        measurements=dict(measurements or {}),
        fs_v=int(fs_v),
        taps_v=int(taps_v),
        xos=list(xos or []),
        hpf=hpf,
        hc_f_arr=hc_f_arr,
        hc_m_arr=hc_m_arr,
        pin_obj=pin_obj,
        filter_key=str(filter_key),
        phase_tag=str(phase_tag),
        target_name=str(target_name),
        phase_kind=phase_kind,
        n_total_override=n_total_override,
        seed_presets=seed_presets_eff,
        optuna_builder=optuna_builder,
        seed_to_params=seed_to_params,
        compute_ctx=compute_ctx,
        use_optuna_trials=bool(use_optuna_trials),
        eval_candidates=list(eval_candidates or []),
        n_total=int(n_total),
        workers=int(workers),
    )


def _dispatch_target_trial_eval(*, dispatch: _TargetTrialDispatch, idx: int, preset: dict) -> dict:
    return _target_eval_one(
        runtime=dispatch.runtime,
        preset=dict(preset or {}),
        base_tc=dispatch.base_tc,
        measurements=dispatch.measurements,
        fs_v=int(dispatch.fs_v),
        taps_v=int(dispatch.taps_v),
        xos=dispatch.xos,
        hpf=dispatch.hpf,
        hc_f_arr=dispatch.hc_f_arr,
        hc_m_arr=dispatch.hc_m_arr,
        pin_obj=dispatch.pin_obj,
        filter_key=dispatch.filter_key,
        _compute_ctx=dispatch.compute_ctx,
        _trial_idx=int(idx),
    )


def _run_target_trials_optuna(*, dispatch: _TargetTrialDispatch) -> list[dict]:
    out_by_idx: dict[int, dict] = {}
    base_tc_optuna = dict(dispatch.base_tc or {})
    base_tc_optuna["_optuna_measurement_sig"] = _auto_search_measurement_identity(dispatch.measurements)
    base_tc_optuna["_optuna_journal_kind"] = "target"
    base_tc_optuna["_optuna_filter_key"] = str(dispatch.filter_key or "")
    raw_scope = f"target-{dispatch.target_name!s}-{dispatch.phase_tag!s}"
    scope_eff = dispatch.runtime.auto_optuna_effective_scope(
        base_tc_optuna,
        raw_scope,
        phase_kind=dispatch.phase_kind,
    )
    study_name = dispatch.runtime.auto_optuna_study_name(
        study_sig=dispatch.target_study_sig,
        scope=scope_eff,
    )

    def _eval_one(idx: int, preset: dict) -> dict:
        out = _dispatch_target_trial_eval(
            dispatch=dispatch,
            idx=int(idx),
            preset=dict(preset or {}),
        )
        out = dict(out or {})
        out["idx"] = int(idx)
        return out

    def _consume_one(idx: int, out: dict) -> bool:
        out_by_idx[int(idx)] = dict(out or {})
        return False

    dispatch.runtime.auto_run_optuna_eval_loop(
        optuna_mod=dispatch.optuna_mod,
        cfg=dispatch.cfg,
        n_total=int(dispatch.n_total),
        seed=int(
            dispatch.seed_target
            + sum(ord(ch) for ch in str(dispatch.target_name)) * 31
            + sum(ord(ch) for ch in str(dispatch.phase_tag)) * 17
        ),
        base_data=base_tc_optuna,
        seed_presets=list(dispatch.seed_presets or []),
        build_preset=dispatch.optuna_builder,
        eval_one=_eval_one,
        consume_one=_consume_one,
        objective_value=lambda out, _goal=str((dispatch.base_tc or {}).get("auto_goal", "") or ""): dispatch.runtime.auto_optuna_objective_value(
            dict((out or {}).get("metrics", {}) or {}),
            use_refine_tiebreak=False,
            goal=_goal,
        ),
        workers=int(dispatch.workers),
        seed_to_params=dispatch.seed_to_params,
        study_name=study_name,
        study_scope=raw_scope,
        phase_label=f"target {dispatch.target_name!s} {dispatch.phase_tag!s}",
        phase_kind=dispatch.phase_kind,
        study_user_attrs={
            "decaycore_kind": "target_search",
            "decaycore_target_name": str(dispatch.target_name),
            "decaycore_target_study_sig": str(dispatch.target_study_sig),
            "decaycore_target_cache_version": 5,
            "decaycore_filter_key": str(dispatch.filter_key),
        },
    )
    return [
        dict(
            out_by_idx.get(
                int(idx),
                {"idx": int(idx), "ok": False, "error": "missing worker result"},
            )
            or {}
        )
        for idx in range(1, int(dispatch.n_total) + 1)
    ]


def _run_target_trials_serial(
    *,
    dispatch: _TargetTrialDispatch,
    idx_presets: list[tuple[int, dict]],
) -> dict[int, dict]:
    out_by_idx: dict[int, dict] = {}
    for idx, preset in idx_presets:
        try:
            out = _dispatch_target_trial_eval(
                dispatch=dispatch,
                idx=int(idx),
                preset=dict(preset or {}),
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
        ) as exc:
            out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        out_by_idx[int(idx)] = dict(out or {})
    return out_by_idx


def _run_target_trials_parallel(
    *,
    dispatch: _TargetTrialDispatch,
    idx_presets: list[tuple[int, dict]],
) -> dict[int, dict]:
    out_by_idx: dict[int, dict] = {}
    chunk_size = int(_auto_trial_chunk_size(dispatch.workers))
    executor_cls = ProcessPoolExecutor if _USE_PROCESS_POOL else ThreadPoolExecutor
    if _USE_PROCESS_POOL:
        mp_ctx = multiprocessing.get_context("fork")
        executor_kwargs = {"mp_context": mp_ctx, "initializer": _auto_worker_init}
    else:
        executor_kwargs = {}
    if _USE_PROCESS_POOL:
        global _PROC_RUNTIME
        _PROC_RUNTIME = dispatch.runtime
        global _PROC_COMPUTE_CTX
        _PROC_COMPUTE_CTX = dispatch.compute_ctx
    with executor_cls(max_workers=int(dispatch.workers), **executor_kwargs) as executor:
        for c0 in range(0, int(len(idx_presets)), int(chunk_size)):
            chunk = idx_presets[c0 : c0 + int(chunk_size)]
            if _USE_PROCESS_POOL:
                future_map = {
                    executor.submit(
                        _target_eval_one_proc,
                        preset=dict(preset or {}),
                        base_tc=dispatch.base_tc,
                        measurements=dispatch.measurements,
                        fs_v=int(dispatch.fs_v),
                        taps_v=int(dispatch.taps_v),
                        xos=dispatch.xos,
                        hpf=dispatch.hpf,
                        hc_f_arr=dispatch.hc_f_arr,
                        hc_m_arr=dispatch.hc_m_arr,
                        filter_key=dispatch.filter_key,
                        _trial_idx=int(idx),
                    ): int(idx)
                    for idx, preset in chunk
                }
            else:
                # ThreadPool: ctx is shared across threads; Python GIL makes dict
                # operations safe, so occasional duplicate computation is the only risk.
                future_map = {
                    executor.submit(
                        _dispatch_target_trial_eval,
                        dispatch=dispatch,
                        idx=int(idx),
                        preset=dict(preset or {}),
                    ): int(idx)
                    for idx, preset in chunk
                }
            for future in as_completed(list(future_map.keys())):
                idx = int(future_map.get(future, 0))
                try:
                    out = future.result()
                    if not isinstance(out, dict):
                        out = {"ok": False, "error": "invalid worker result"}
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
                    out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                out_by_idx[int(idx)] = dict(out or {})
    return out_by_idx


def _collect_target_trial_results(
    *,
    idx_presets: list[tuple[int, dict]],
    out_by_idx: dict[int, dict],
) -> list[dict]:
    return [
        dict(
            out_by_idx.get(
                int(idx),
                {"ok": False, "error": "missing worker result"},
            )
            or {}
        )
        for idx, _preset in idx_presets
    ]


def _run_target_trials(
    *,
    runtime,
    cfg,
    optimizer_backend: str,
    optuna_mod,
    target_study_sig: str,
    seed_target: int,
    cands: list[dict],
    base_tc: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    hc_f_arr,
    hc_m_arr,
    pin_obj,
    filter_key: str,
    phase_tag: str,
    target_name: str,
    phase_kind: str | None = None,
    n_total_override: int | None = None,
    seed_presets: list[dict] | None = None,
    optuna_builder=None,
    seed_to_params=None,
) -> list[dict]:
    dispatch = _prepare_target_trial_dispatch(
        runtime=runtime,
        cfg=cfg,
        optimizer_backend=optimizer_backend,
        optuna_mod=optuna_mod,
        target_study_sig=target_study_sig,
        seed_target=int(seed_target),
        cands=list(cands or []),
        base_tc=dict(base_tc or {}),
        measurements=dict(measurements or {}),
        fs_v=int(fs_v),
        taps_v=int(taps_v),
        xos=list(xos or []),
        hpf=hpf,
        hc_f_arr=hc_f_arr,
        hc_m_arr=hc_m_arr,
        pin_obj=pin_obj,
        filter_key=str(filter_key),
        phase_tag=str(phase_tag),
        target_name=str(target_name),
        phase_kind=phase_kind,
        n_total_override=n_total_override,
        seed_presets=seed_presets,
        optuna_builder=optuna_builder,
        seed_to_params=seed_to_params,
    )
    if dispatch.n_total <= 0:
        return []
    if bool(dispatch.use_optuna_trials):
        return _run_target_trials_optuna(dispatch=dispatch)

    idx_presets = list(enumerate(list(dispatch.eval_candidates or []), start=1))
    if dispatch.workers <= 1 or dispatch.n_total <= 1:
        out_by_idx = _run_target_trials_serial(
            dispatch=dispatch,
            idx_presets=idx_presets,
        )
    else:
        out_by_idx = _run_target_trials_parallel(
            dispatch=dispatch,
            idx_presets=idx_presets,
        )
    logger.debug(
        "Target trials compute context stats (target=%s phase=%s):\n%s",
        str(dispatch.target_name),
        str(dispatch.phase_tag),
        dispatch.compute_ctx.cache_stats_text(),
    )
    return _collect_target_trial_results(
        idx_presets=idx_presets,
        out_by_idx=out_by_idx,
    )


def _materialize_target_candidate(*, tc: dict) -> _TargetEvalMaterialization:
    tc_dict = dict(tc or {})
    return _TargetEvalMaterialization(
        tc=tc_dict,
        hc_name=str(tc_dict.get("hc_mode", "") or "").strip(),
    )


def _run_target_eval_trials(
    *,
    materialized: _TargetEvalMaterialization,
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
    t_idx: int,
    emit_status: bool,
    curve_inner_workers: int | None,
) -> dict | None:
    if not bool(materialized.hc_name):
        return None
    from .orchestrator_target_phase1 import _run_target_eval_trials_core
    return _run_target_eval_trials_core(
        runtime=runtime,
        cfg=cfg,
        base_data=base_data,
        measurements=measurements,
        fs_v=int(fs_v),
        taps_v=int(taps_v),
        xos=xos,
        hpf=hpf,
        pin_obj=pin_obj,
        goal=goal,
        filter_key=filter_key,
        optimizer_backend=optimizer_backend,
        optuna_mod=optuna_mod,
        seed_target=int(seed_target),
        target_study_sig=str(target_study_sig),
        trials_eff=int(trials_eff),
        shortlisted=list(shortlisted or []),
        status_cb=status_cb,
        f6_txt=str(f6_txt),
        tc=dict(materialized.tc or {}),
        t_idx=int(t_idx),
        emit_status=bool(emit_status),
        curve_inner_workers=curve_inner_workers,
    )


def _summarize_target_eval(*, trial_result: dict | None) -> _TargetEvalSummary:
    if not isinstance(trial_result, dict):
        return _TargetEvalSummary()
    item = dict(trial_result or {})
    return _TargetEvalSummary(
        item=item,
        best_metrics=dict(item.get("best_metrics", {}) or {}),
        best_preset=dict(item.get("best_preset", {}) or {}),
    )


def _build_target_eval_result(*, summary: _TargetEvalSummary) -> dict | None:
    if not isinstance(summary.item, dict):
        return None
    return dict(summary.item)


def _load_target_curve_arrays(
    *,
    runtime,
    tc: dict,
    hc_name: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        if "_synth_hc_f" in tc and "_synth_hc_m" in tc:
            hc_f = np.asarray(tc["_synth_hc_f"], dtype=float)
            hc_m = np.asarray(tc["_synth_hc_m"], dtype=float)
        else:
            hc_f_raw, hc_m_raw = runtime.get_house_curve_by_name(hc_name)
            hc_f = np.asarray(hc_f_raw, dtype=float)
            hc_m = np.asarray(hc_m_raw, dtype=float)
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
    ):
        return None
    if hc_f.size < 4 or hc_m.size != hc_f.size:
        return None
    return hc_f, hc_m


def _prepare_target_trial_setup(
    *,
    runtime,
    base_data: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    filter_key: str,
    optimizer_backend: str,
    optuna_mod,
    trials_eff: int,
    tc: dict,
    hc_name: str,
    curve_inner_workers: int | None,
    emit_status: bool,
) -> _TargetTrialSetup | None:
    curve_arrays = _load_target_curve_arrays(
        runtime=runtime,
        tc=dict(tc or {}),
        hc_name=str(hc_name),
    )
    if curve_arrays is None:
        return None
    hc_f, hc_m = curve_arrays
    seed_tc = int(
        (
            _auto_seed_from_signature(
                base_data=base_data,
                measurements=measurements,
                fs_v=int(fs_v),
                taps_v=int(taps_v),
                xos=xos,
                hpf=hpf,
                hc_mode=str(hc_name),
                include_hc_mode=True,
            )
            + sum(ord(ch) for ch in hc_name) * 13
        )
        & 0xFFFFFFFF
    )
    if bool(emit_status):
        _auto_apply_seed(seed_tc)
    base_tc = dict(base_data or {})
    base_tc["hc_mode"] = str(hc_name)
    if isinstance(curve_inner_workers, int) and int(curve_inner_workers) > 0:
        base_tc["auto_mode_workers"] = int(curve_inner_workers)
    if str(filter_key) in ("linear", "asym"):
        base_tc["phase_limit"] = round(
            float(_auto_phase_limit_center(base_tc.get("phase_limit"))),
            1,
        )
    use_optuna_curve_trials = bool(
        str(optimizer_backend) == "optuna"
        and runtime.auto_optuna_module_ready(optuna_mod)
    )
    phase1_seed_presets = _build_auto_mode_candidates(
        base_tc,
        n_trials=1,
        seed=seed_tc,
        optimize_mag_low=bool(base_tc.get("auto_optimize_low_bass_cut", True)),
    )
    candidates = []
    if not bool(use_optuna_curve_trials):
        candidates = _build_auto_mode_candidates(
            base_tc,
            n_trials=int(trials_eff),
            seed=seed_tc,
            optimize_mag_low=bool(base_tc.get("auto_optimize_low_bass_cut", True)),
        )
    phase1_trial_total = int(
        max(1, int(trials_eff) if bool(use_optuna_curve_trials) else len(candidates))
    )
    return _TargetTrialSetup(
        hc_name=str(hc_name),
        hc_f=hc_f,
        hc_m=hc_m,
        seed_tc=int(seed_tc),
        base_tc=base_tc,
        use_optuna_curve_trials=bool(use_optuna_curve_trials),
        candidates=list(candidates or []),
        phase1_seed_presets=list(phase1_seed_presets or []),
        phase1_trial_total=int(phase1_trial_total),
    )


def _evaluate_target_curve(
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
    materialized = _materialize_target_candidate(tc=dict(tc or {}))
    trial_result = _run_target_eval_trials(
        materialized=materialized,
        runtime=runtime,
        cfg=cfg,
        base_data=base_data,
        measurements=measurements,
        fs_v=int(fs_v),
        taps_v=int(taps_v),
        xos=xos,
        hpf=hpf,
        pin_obj=pin_obj,
        goal=goal,
        filter_key=filter_key,
        optimizer_backend=optimizer_backend,
        optuna_mod=optuna_mod,
        seed_target=int(seed_target),
        target_study_sig=str(target_study_sig),
        trials_eff=int(trials_eff),
        shortlisted=list(shortlisted or []),
        status_cb=status_cb,
        f6_txt=str(f6_txt),
        t_idx=int(t_idx),
        emit_status=bool(emit_status),
        curve_inner_workers=curve_inner_workers,
    )
    summary = _summarize_target_eval(trial_result=trial_result)
    return _build_target_eval_result(summary=summary)


def _evaluate_target_curve_proc(*, tc: dict, t_idx: int) -> dict | None:
    """ProcessPool wrapper: reads shared state from fork-inherited _PROC_CURVE_ARGS."""
    from .orchestrator_target_shortlist import _PROC_CURVE_ARGS

    a = _PROC_CURVE_ARGS
    return _evaluate_target_curve(
        runtime=a["runtime"],
        cfg=a["cfg"],
        base_data=a["base_data"],
        measurements=a["measurements"],
        fs_v=a["fs_v"],
        taps_v=a["taps_v"],
        xos=a["xos"],
        hpf=a["hpf"],
        pin_obj=None,
        goal=a["goal"],
        filter_key=a["filter_key"],
        optimizer_backend=a["optimizer_backend"],
        optuna_mod=a["optuna_mod"],
        seed_target=a["seed_target"],
        target_study_sig=a["target_study_sig"],
        trials_eff=a["trials_eff"],
        shortlisted=a["shortlisted"],
        status_cb=None,
        f6_txt=a["f6_txt"],
        tc=dict(tc or {}),
        t_idx=int(t_idx),
        emit_status=False,
        curve_inner_workers=a["curve_inner_workers"],
    )
