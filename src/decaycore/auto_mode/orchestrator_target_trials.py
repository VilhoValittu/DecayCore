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

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from .cache_signature import (
    _auto_apply_seed,
    _auto_seed_from_signature,
)
from .cache_measurement_sig import _auto_get_measurement_signature
from .candidate_generation import _build_auto_mode_candidates
from .search_v2.candidates import deduplicate_presets
from .shared import (
    MAX_SAFE_BOOST,
    AUTO_MODE_TARGET_TRIALS_PER_CURVE,
    _auto_filter_cache_key,
    _auto_phase_limit_center,
    _auto_phase_limit_clip,
    _auto_safe_float,
    _auto_trial_chunk_size,
    _clip,
    _auto_builtin_target_name,
)
from .orchestrator_target_types import (
    _TargetEvalMaterialization,
    _TargetEvalSummary,
    _TargetTrialSetup,
)

logger = logging.getLogger("DecayCore")


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
) -> dict:
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
        setattr(
            cfg,
            "bass_smooth_w_gamma",
            float(trial_data.get("bass_smooth_w_gamma", 2.40)),
        )
        setattr(
            cfg,
            "bass_smooth_w_max",
            float(trial_data.get("bass_smooth_w_max", 0.45)),
        )
    except Exception:
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
    use_optuna_trials = bool(
        str(optimizer_backend) == "optuna"
        and runtime.auto_optuna_module_ready(optuna_mod)
        and callable(optuna_builder)
    )
    eval_candidates = list(cands or [])
    if not bool(use_optuna_trials):
        eval_candidates = deduplicate_presets([*list(seed_presets or []), *eval_candidates])
    n_total = int(n_total_override) if n_total_override is not None else int(len(eval_candidates))
    if n_total <= 0:
        return []
    workers = int(runtime.auto_trial_workers(base_tc, n_total))
    if workers > 1:
        logger.info(
            "Automatic mode target trials: target=%s, phase=%s, parallel workers=%d",
            str(target_name),
            str(phase_tag),
            int(workers),
        )

    if bool(use_optuna_trials):
        out_by_idx: dict[int, dict] = {}
        base_tc_optuna = dict(base_tc or {})
        base_tc_optuna["_optuna_measurement_sig"] = _auto_get_measurement_signature(measurements)
        base_tc_optuna["_optuna_journal_kind"] = "target"
        base_tc_optuna["_optuna_filter_key"] = ""
        raw_scope = f"target-{str(target_name)}-{str(phase_tag)}"
        scope_eff = runtime.auto_optuna_effective_scope(
            base_tc_optuna,
            raw_scope,
            phase_kind=phase_kind,
        )
        study_name = runtime.auto_optuna_study_name(
            study_sig=target_study_sig,
            scope=scope_eff,
        )

        def _eval_one(idx: int, preset: dict) -> dict:
            out = _target_eval_one(
                runtime=runtime,
                preset=dict(preset or {}),
                base_tc=base_tc,
                measurements=measurements,
                fs_v=int(fs_v),
                taps_v=int(taps_v),
                xos=xos,
                hpf=hpf,
                hc_f_arr=hc_f_arr,
                hc_m_arr=hc_m_arr,
                pin_obj=pin_obj,
                filter_key=filter_key,
            )
            out = dict(out or {})
            out["idx"] = int(idx)
            return out

        def _consume_one(idx: int, out: dict) -> bool:
            out_by_idx[int(idx)] = dict(out or {})
            return False

        runtime.auto_run_optuna_eval_loop(
            optuna_mod=optuna_mod,
            cfg=cfg,
            n_total=int(n_total),
            seed=int(
                seed_target
                + sum(ord(ch) for ch in str(target_name)) * 31
                + sum(ord(ch) for ch in str(phase_tag)) * 17
            ),
            base_data=base_tc_optuna,
            seed_presets=list(seed_presets or []),
            build_preset=optuna_builder,
            eval_one=_eval_one,
            consume_one=_consume_one,
            objective_value=lambda out, _goal=str((base_tc or {}).get("auto_goal", "") or ""): runtime.auto_optuna_objective_value(
                dict((out or {}).get("metrics", {}) or {}),
                use_refine_tiebreak=False,
                goal=_goal,
            ),
            workers=int(workers),
            seed_to_params=seed_to_params,
            study_name=study_name,
            study_scope=raw_scope,
            phase_label=f"target {str(target_name)} {str(phase_tag)}",
            phase_kind=phase_kind,
            study_user_attrs={
                "decaycore_kind": "target_search",
                "decaycore_target_name": str(target_name),
                "decaycore_target_study_sig": str(target_study_sig),
                "decaycore_target_cache_version": 2,
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
            for idx in range(1, int(n_total) + 1)
        ]

    idx_presets = list(enumerate(list(eval_candidates or []), start=1))
    out_by_idx: dict[int, dict] = {}
    if workers <= 1 or n_total <= 1:
        for idx, preset in idx_presets:
            try:
                out = _target_eval_one(
                    runtime=runtime,
                    preset=dict(preset or {}),
                    base_tc=base_tc,
                    measurements=measurements,
                    fs_v=int(fs_v),
                    taps_v=int(taps_v),
                    xos=xos,
                    hpf=hpf,
                    hc_f_arr=hc_f_arr,
                    hc_m_arr=hc_m_arr,
                    pin_obj=pin_obj,
                    filter_key=filter_key,
                )
            except Exception as exc:
                out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            out_by_idx[int(idx)] = dict(out or {})
    else:
        chunk_size = int(_auto_trial_chunk_size(workers))
        with ThreadPoolExecutor(max_workers=int(workers)) as executor:
            for c0 in range(0, int(len(idx_presets)), int(chunk_size)):
                chunk = idx_presets[c0 : c0 + int(chunk_size)]
                future_map = {
                    executor.submit(
                        _target_eval_one,
                        runtime=runtime,
                        preset=dict(preset or {}),
                        base_tc=base_tc,
                        measurements=measurements,
                        fs_v=int(fs_v),
                        taps_v=int(taps_v),
                        xos=xos,
                        hpf=hpf,
                        hc_f_arr=hc_f_arr,
                        hc_m_arr=hc_m_arr,
                        pin_obj=pin_obj,
                        filter_key=filter_key,
                    ): int(idx)
                    for idx, preset in chunk
                }
                for future in as_completed(list(future_map.keys())):
                    idx = int(future_map.get(future, 0))
                    try:
                        out = future.result()
                        if not isinstance(out, dict):
                            out = {"ok": False, "error": "invalid worker result"}
                    except Exception as exc:
                        out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                    out_by_idx[int(idx)] = dict(out or {})

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
    except Exception:
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
            float(_auto_phase_limit_center(base_tc.get("phase_limit", None))),
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
