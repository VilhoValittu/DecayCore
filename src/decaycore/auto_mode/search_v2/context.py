# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Execution context dataclass and builder for auto search v2."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from .. import api as auto_api
from ..auto_mode_profile import AutoModeProfiler, auto_mode_profile_enabled
from ..cache_measurement_sig import _auto_get_measurement_signature, _auto_optuna_stable_study_sig
from ..cache_signature import _auto_seed_from_signature
from ..materialize import AutoModeMaterializeContext, build_materialize_helpers
from .runtime import _runtime
from .seeds import _apply_explicit_seed, _apply_legacy_opportunistic_seeds

logger = logging.getLogger("DecayCore")


@contextmanager
def _nullctx():
    yield


@dataclass
class AutoSearchExecutionContext:
    base_data: dict
    measurements: dict
    fs_v: int
    taps_v: int
    xos: list
    hpf: dict | None
    hc_f: object
    hc_m: object
    pin_obj: object
    status_cb: object
    cache_base_data: dict
    search_base_data: dict
    cfg: object
    profiler: AutoModeProfiler | None
    n_trials_eff: int
    compat_version: str
    goal: str
    filter_key: str
    rank_basis: str
    optimizer_backend: str
    optuna_mod: object
    runtime: dict
    cache_ready_preset: object
    materialize_preset_result: object
    maybe_apply_residual_tiebreak: object
    seed: int
    optuna_search_sig: str
    prior_seed_preset: dict
    use_optuna_trials: bool
    candidates: list
    status_prefix: str
    winner_target_name: str | None
    search_state: object


def _prepare_base_data(base_data: dict, measurements: dict) -> tuple[dict, dict]:
    cache_base_data = dict(base_data or {})
    search_base_data = dict(base_data or {})
    filter_key = auto_api._auto_filter_cache_key(search_base_data)
    filter_type = auto_api._auto_filter_type_for_key(filter_key)
    cache_base_data["filter_type"] = str(filter_type)
    search_base_data["filter_type"] = str(filter_type)
    for key in (
        "_auto_exc_freq_hz",
        "_auto_exc_seed_freq_hz",
        "_auto_mag_c_min_hz",
        "_auto_low_bass_cut_hz",
    ):
        cache_base_data.pop(key, None)

    for key in (
        "harmonic_freq_hz_l",
        "harmonic_magnitudes_db_l",
        "harmonic_freq_hz_r",
        "harmonic_magnitudes_db_r",
        "measured_rt60_l",
        "measured_rt60_bands_l",
        "rt60_summary_l",
        "measured_rt60_r",
        "measured_rt60_bands_r",
        "rt60_summary_r",
        "harmonic_risk_summary_l",
        "harmonic_risk_summary_r",
    ):
        value = dict(measurements or {}).get(key)
        if value is not None:
            search_base_data[key] = value
    return cache_base_data, search_base_data


def _finish_search_setup(context: AutoSearchExecutionContext) -> None:
    search_base_data = context.search_base_data
    if str(context.filter_key) in ("linear", "asym"):
        prev_phase_limit = auto_api._auto_safe_float(search_base_data.get("phase_limit", float("nan")), float("nan"))
        clamped_phase_limit = round(
            float(auto_api._auto_phase_limit_center(search_base_data.get("phase_limit", None))),
            1,
        )
        search_base_data["phase_limit"] = float(clamped_phase_limit)
        if np.isfinite(prev_phase_limit) and abs(float(prev_phase_limit) - float(clamped_phase_limit)) > 1e-9:
            logger.info(
                "Automatic mode: clamped phase_limit seed "
                f"{float(prev_phase_limit):.1f} -> {float(clamped_phase_limit):.1f} Hz "
                f"for {str(context.filter_key)} filter"
            )

    context.use_optuna_trials = bool(
        str(context.optimizer_backend) == "optuna" and auto_api._auto_optuna_module_ready(context.optuna_mod)
    )
    context.candidates = []
    if not bool(context.use_optuna_trials):
        context.candidates = auto_api._build_auto_mode_candidates(
            search_base_data,
            n_trials=int(context.n_trials_eff),
            seed=context.seed,
        )
    elif int(context.n_trials_eff) > 0:
        logger.info(
            "Automatic mode optimizer backend: optuna "
            f"(trials={int(context.n_trials_eff)}, "
            f"startup={int(auto_api._auto_optuna_startup_for_phase_kind(context.cfg, phase_kind='phase1', total=int(context.n_trials_eff)))})"
        )

    try:
        target_label = str(search_base_data.get("hc_mode", "") or "").strip()
    except Exception:
        target_label = ""
    context.winner_target_name = str(target_label or "").strip() or None
    if not target_label:
        target_label = "n/a"
    f6_hz = auto_api._auto_safe_float(
        search_base_data.get("_auto_mag_c_min_hz", search_base_data.get("mag_c_min", float("nan"))),
        float("nan"),
    )
    low_bass_hz = auto_api._auto_safe_float(
        search_base_data.get("_auto_low_bass_cut_hz", search_base_data.get("low_bass_cut_hz", float("nan"))),
        float("nan"),
    )
    exc_hz = auto_api._auto_safe_float(
        search_base_data.get("_auto_exc_freq_hz", search_base_data.get("exc_freq", float("nan"))),
        float("nan"),
    )
    hpf_enabled = bool(search_base_data.get("hpf_enable", False))
    hpf_freq = auto_api._auto_safe_float(search_base_data.get("hpf_freq", float("nan")), float("nan"))
    hpf_slope = auto_api._auto_safe_float(search_base_data.get("hpf_slope", float("nan")), float("nan"))
    hpf_meta = dict(search_base_data.get("_auto_hpf_meta", {}) or {})
    if isinstance(hpf_meta, dict):
        hpf_meta_enabled = bool(hpf_meta.get("applied", hpf_meta.get("enabled", False)))
        if not hpf_enabled:
            hpf_enabled = bool(hpf_meta_enabled)
        if not np.isfinite(hpf_freq):
            hpf_freq = auto_api._auto_safe_float(hpf_meta.get("freq", float("nan")), float("nan"))
        if not np.isfinite(hpf_slope):
            hpf_slope = auto_api._auto_safe_float(hpf_meta.get("slope_db_oct", float("nan")), float("nan"))
    if isinstance(context.hpf, dict):
        if not hpf_enabled:
            hpf_enabled = bool(context.hpf.get("enabled", False))
        if not np.isfinite(hpf_freq):
            hpf_freq = auto_api._auto_safe_float(context.hpf.get("freq", float("nan")), float("nan"))
        if not np.isfinite(hpf_slope):
            hpf_order = auto_api._auto_safe_float(context.hpf.get("order", float("nan")), float("nan"))
            if np.isfinite(hpf_order) and float(hpf_order) > 0.0:
                hpf_slope = float(6.0 * float(hpf_order))

    low_txt = f"low-cut {low_bass_hz:.1f} Hz" if np.isfinite(low_bass_hz) else "low-cut n/a"
    exc_txt = f"exc seed {exc_hz:.1f} Hz" if np.isfinite(exc_hz) else "exc seed n/a"
    if bool(hpf_enabled) and np.isfinite(hpf_freq):
        if np.isfinite(hpf_slope):
            hpf_txt = f"hpf {hpf_freq:.1f} Hz/{int(round(hpf_slope))} dB/oct"
        else:
            hpf_txt = f"hpf {hpf_freq:.1f} Hz"
    else:
        hpf_txt = "hpf off"

    if np.isfinite(f6_hz):
        context.status_prefix = (
            f"DecayCore automatic mode [{target_label}] "
            f"(-6 dB {f6_hz:.1f} Hz, {low_txt}, {exc_txt}, {hpf_txt})"
        )
    else:
        context.status_prefix = f"DecayCore automatic mode [{target_label}] ({low_txt}, {exc_txt}, {hpf_txt})"
    refine_trial_hint = int(context.cfg.refine_trial_hint(context.goal))
    logger.info(
        "Automatic mode search: "
        f"goal={context.goal}, basis={context.rank_basis}, target={target_label}, "
        f"trials={int(context.n_trials_eff)}+{int(refine_trial_hint)}, "
        f"cache_schema={int(auto_api.AUTO_MODE_CACHE_SCHEMA_VERSION)}"
    )

    context.search_state = auto_api._AutoModeSearchState()
    try:
        seed_metrics = dict(search_base_data.get("_auto_target_seed_metrics", {}) or {})
        seed_preset = dict(search_base_data.get("_auto_target_seed_preset", {}) or {})
        if seed_metrics and seed_preset:
            auto_api._auto_set_search_winner(
                context.search_state,
                seed_metrics,
                seed_preset,
                phase_label="cache_seed",
                target_name=context.winner_target_name,
            )
            logger.info(
                "Automatic mode: baseline seeded from cache (rank %.3f).",
                auto_api._auto_safe_float(seed_metrics.get("rank_score"), 0.0),
            )
    except Exception:
        logger.exception("cache seed baseline")


def build_execution_context(
    *,
    base_data: dict,
    measurements: dict,
    fs_v: int,
    taps_v: int,
    xos: list,
    hpf: dict | None,
    hc_f,
    hc_m,
    pin_obj,
    status_cb,
    n_trials: int,
    allow_legacy_cache_seeds: bool = False,
) -> AutoSearchExecutionContext:
    cache_base_data, search_base_data = _prepare_base_data(base_data, measurements)
    cfg = auto_api.AutoModeConfig.from_base_data(search_base_data)
    prof = AutoModeProfiler() if auto_mode_profile_enabled(search_base_data) else None
    n_trials_eff = int(max(1, auto_api._auto_safe_int(n_trials, cfg.trials)))
    compat_version = auto_api._auto_compat_version(search_base_data)
    goal = auto_api._auto_goal(search_base_data)
    filter_key = auto_api._auto_filter_cache_key(search_base_data)
    measurement_sig = _auto_get_measurement_signature(measurements)
    logger.info(
        "Automatic mode filter selected: %s (filter_key=%s)",
        str(search_base_data.get("filter_type", "") or ""),
        str(filter_key),
    )
    cache_base_data["_optuna_filter_key"] = str(filter_key)
    search_base_data["_optuna_filter_key"] = str(filter_key)
    cache_base_data["_optuna_measurement_sig"] = str(measurement_sig)
    search_base_data["_optuna_measurement_sig"] = str(measurement_sig)
    cache_base_data["_optuna_journal_kind"] = "filter"
    search_base_data["_optuna_journal_kind"] = "filter"
    rank_basis = auto_api._auto_goal_basis_text(goal)
    optimizer_backend = auto_api._auto_optimizer_backend(
        search_base_data,
        default_optuna_enabled=bool(cfg.optuna_pilot_enabled),
    )
    optuna_mod = auto_api._auto_import_optuna() if str(optimizer_backend) == "optuna" else None
    if str(optimizer_backend) == "optuna" and optuna_mod is None:
        logger.warning(
            "Automatic mode: optuna backend requested but unavailable; "
            "falling back to builtin sampler."
        )
        optimizer_backend = "builtin"
    rt = _runtime(prof)
    exact_cached_metrics = {}
    (
        cache_ready_preset,
        materialize_preset_result,
        _preset_signature_ignoring_residual,
        maybe_apply_residual_tiebreak,
    ) = build_materialize_helpers(
        AutoModeMaterializeContext(
            cfg=cfg,
            cache_base_data=cache_base_data,
            measurements=measurements,
            fs_v=int(fs_v),
            taps_v=int(taps_v),
            xos=xos,
            hpf=hpf,
            hc_f=hc_f,
            hc_m=hc_m,
            pin_obj=pin_obj,
            filter_key=str(filter_key),
            max_safe_boost=float(auto_api.MAX_SAFE_BOOST),
            goal=str(goal),
            status_cb=status_cb,
            exact_cached_metrics_getter=lambda: exact_cached_metrics,
            auto_score_result_fn=rt["auto_score_result"],
            auto_optuna_jsonable_fn=auto_api._auto_optuna_jsonable,
            auto_rank_key_fn=auto_api._auto_rank_key,
            auto_is_better_refine_fn=auto_api._auto_is_better_refine,
            build_config_fn=rt["build_config"],
            run_pipeline_fn=rt["run_pipeline"],
            summarize_run_fn=rt["summarize_run"],
            preset_transient_keys=tuple(auto_api.AUTO_MODE_PRESET_TRANSIENT_KEYS),
            residual_tiebreak_enabled=bool(auto_api.AUTO_MODE_RESIDUAL_TIEBREAK_ENABLED),
            residual_top_k=int(auto_api.AUTO_MODE_RESIDUAL_TIEBREAK_TOP_K),
            residual_rank_eps=float(auto_api.AUTO_MODE_RESIDUAL_TIEBREAK_RANK_EPS),
        )
    )
    seed = _auto_seed_from_signature(
        base_data=cache_base_data,
        measurements=measurements,
        fs_v=fs_v,
        taps_v=taps_v,
        xos=xos,
        hpf=hpf,
        hc_mode=str(cache_base_data.get("hc_mode", "") or "").strip() or None,
        include_hc_mode=True,
    )
    optuna_search_sig = _auto_optuna_stable_study_sig(
        measurement_identity=str(measurement_sig),
        filter_key=str(filter_key),
    )
    context = AutoSearchExecutionContext(
        base_data=dict(base_data or {}),
        measurements=dict(measurements or {}),
        fs_v=int(fs_v),
        taps_v=int(taps_v),
        xos=list(xos or []),
        hpf=hpf,
        hc_f=hc_f,
        hc_m=hc_m,
        pin_obj=pin_obj,
        status_cb=status_cb,
        cache_base_data=cache_base_data,
        search_base_data=search_base_data,
        cfg=cfg,
        profiler=prof,
        n_trials_eff=int(n_trials_eff),
        compat_version=str(compat_version),
        goal=str(goal),
        filter_key=str(filter_key),
        rank_basis=str(rank_basis),
        optimizer_backend=str(optimizer_backend),
        optuna_mod=optuna_mod,
        runtime=rt,
        cache_ready_preset=cache_ready_preset,
        materialize_preset_result=materialize_preset_result,
        maybe_apply_residual_tiebreak=maybe_apply_residual_tiebreak,
        seed=int(seed),
        optuna_search_sig=str(optuna_search_sig),
        prior_seed_preset=_apply_explicit_seed(
            search_base_data=search_base_data,
            cache_base_data=cache_base_data,
            measurements=measurements,
        ),
        use_optuna_trials=False,
        candidates=[],
        status_prefix="",
        winner_target_name=None,
        search_state=None,
    )
    if bool(allow_legacy_cache_seeds):
        _apply_legacy_opportunistic_seeds(context)
    _finish_search_setup(context)
    return context
