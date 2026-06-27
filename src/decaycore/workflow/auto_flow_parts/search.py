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

import logging
import typing

import numpy as np

from ...auto_mode.api import (
    AUTO_MODE_GOAL_FLAT,
    AUTO_MODE_COMPAT_VERSION,
    AUTO_MODE_EXC_MAX_HZ,
    AUTO_MODE_EXC_MIN_HZ,
    AUTO_MODE_LOCAL_REFINE_ENABLED,
    AUTO_MODE_LOCAL_REFINE_TOP_K,
    AUTO_MODE_LOCAL_REFINE_TRIALS_PER_TOP,
    AUTO_MODE_PHASE3_MICRO_TRIALS,
    AUTO_MODE_REFINE_TRIALS,
    AUTO_MODE_TRIALS,
    _auto_safe_float,
    _run_auto_mode_search,
)
from ...auto_mode.rank_score import attach_official_rank_score, official_rank_score
from ...application.run_contracts import apply_auto_mode_result
from ...ui.decaycore_utils import scale_taps_with_fs
from ..bridge_types import ProcessRunCallbacks

from .progress import (
    _get_auto_status_callback,
    _set_auto_progress,
)
from .status_text import (
    _build_auto_finalize_status,
    _build_auto_selected_text,
)

if typing.TYPE_CHECKING:
    from ..process_run_flow import ProcessRunSupport

logger = logging.getLogger("DecayCore")

_AUTO_PROGRESS_INIT = 0.06
_AUTO_PROGRESS_TARGET_MODE = 0.12
_AUTO_PROGRESS_TARGET_PRESELECT = 0.15
_AUTO_PROGRESS_TARGET_TRIALS_START = 0.21
_AUTO_PROGRESS_TARGET_TRIALS_END = 0.33
_AUTO_PROGRESS_PRESET_SEARCH_START = 0.36
_AUTO_PROGRESS_PHASE1_START = 0.39
_AUTO_PROGRESS_PHASE1_END = 0.61
_AUTO_PROGRESS_PHASE2_START = 0.61
_AUTO_PROGRESS_PHASE2_END = 0.82
_AUTO_PROGRESS_PHASE3_START = 0.82
_AUTO_PROGRESS_PHASE3_END = 0.85
_AUTO_PROGRESS_FINALIZE = 0.88


def _auto_mode_apply_best_preset(
    ctx: dict,
    *,
    data: dict,
    measurements: dict,
    auto_res: dict,
    auto_goal: str,
    support: ProcessRunSupport,
) -> tuple[dict, dict, dict, dict, str]:
    best_preset = dict(auto_res.get("best_preset", {}) or {})
    best_applied_preset = dict(auto_res.get("best_applied_preset", best_preset) or {})
    best_metrics = attach_official_rank_score(auto_res.get("best_metrics", {}))
    winner_meta = dict(auto_res.get("winner", {}) or {})
    optimizer_backend = str(auto_res.get("optimizer_backend", "builtin") or "builtin")
    if best_applied_preset:
        resolved_config = ctx.get("resolved_config")
        if resolved_config is not None:
            resolved_config = apply_auto_mode_result(
                resolved_config,
                auto_res,
                version=str(support.version),
                auto_mode_compat_version=AUTO_MODE_COMPAT_VERSION,
            )
            ctx["resolved_config"] = resolved_config
            ctx["resolved_data"] = resolved_config.resolved_data
            ctx["data"] = resolved_config.resolved_data
            ctx["auto_applied_preset"] = dict(resolved_config.auto_applied_preset)
            data = resolved_config.resolved_data
            if str(auto_goal) == AUTO_MODE_GOAL_FLAT:
                data["unsafe_raw_dsp"] = True
            measurements = resolved_config.measurements
            ctx["measurements"] = measurements
        else:
            data = dict(data)
            data.update(best_applied_preset)
            if str(auto_goal) == AUTO_MODE_GOAL_FLAT:
                data["unsafe_raw_dsp"] = True
            data["program_version"] = support.version
            data["auto_mode_compat_version"] = AUTO_MODE_COMPAT_VERSION
            ctx["resolved_data"] = data
            ctx["data"] = data
            ctx["auto_applied_preset"] = dict(best_applied_preset)
            measurements["ui_data"] = data
        try:
            from ...auto_mode.filter_priors import update_auto_mode_filter_priors_from_winner
            update_auto_mode_filter_priors_from_winner(
                data.get("filter_type"),
                best_preset,
                data,
                measurements=measurements,
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
            NameError,
        ) as _upd_exc:
            logger.warning(f"Filter priors update skipped: {type(_upd_exc).__name__}: {_upd_exc}")
    reported_best_auto_exc_hz = _auto_safe_float(
        auto_res.get(
            "best_auto_exc_freq_hz",
            best_metrics.get("auto_exc_zero_penalty_hz", float("nan")),
        ),
        float("nan"),
    )
    exc_seed_hz = _auto_safe_float(
        data.get(
            "_auto_exc_seed_freq_hz",
            data.get("_auto_exc_freq_hz", data.get("exc_freq", float("nan"))),
        ),
        float("nan"),
    )
    best_auto_exc_hz = float(exc_seed_hz) if np.isfinite(exc_seed_hz) else float(reported_best_auto_exc_hz)
    if np.isfinite(best_auto_exc_hz):
        best_auto_exc_hz = float(np.clip(float(best_auto_exc_hz), float(AUTO_MODE_EXC_MIN_HZ), float(AUTO_MODE_EXC_MAX_HZ)))
        data["exc_freq"] = float(round(best_auto_exc_hz, 1))
        data["_auto_exc_freq_hz"] = float(round(best_auto_exc_hz, 1))
        if np.isfinite(exc_seed_hz):
            data["_auto_exc_seed_freq_hz"] = float(round(exc_seed_hz, 1))
            logger.info(
                "DecayCore automatic mode: excursion protection kept at seed "
                f"{float(data['_auto_exc_freq_hz']):.1f} Hz"
            )
        elif str(optimizer_backend) == "optuna":
            logger.info(
                "DecayCore automatic mode: excursion protection kept at winner preset "
                f"{float(data['_auto_exc_freq_hz']):.1f} Hz for Optuna search"
            )
        else:
            logger.info(
                "DecayCore automatic mode: excursion protection kept at winner preset "
                f"{float(data['_auto_exc_freq_hz']):.1f} Hz"
            )
    return data, measurements, best_preset, best_metrics, winner_meta, optimizer_backend


def _auto_mode_store_search_meta(
    *,
    auto_res: dict,
    data: dict,
    measurements: dict,
    best_preset: dict,
    best_metrics: dict,
    winner_meta: dict,
    optimizer_backend: str,
    auto_search_fs: int,
    auto_search_taps: int,
    status_cb,
    set_auto_selected_bar,
) -> None:
    sel_goal = str(auto_res.get("auto_goal", data.get("auto_goal", "balanced")) or "balanced")
    sel_basis = str(auto_res.get("selection_basis", "rank_score") or "rank_score")
    data["_auto_mode_meta"] = {
        "enabled": True,
        "auto_goal": str(sel_goal),
        "selection_basis": str(sel_basis),
        "optimizer_backend": str(optimizer_backend),
        "trials_total": int(auto_res.get("trials_total", AUTO_MODE_TRIALS)),
        "trials_ok": int(auto_res.get("trials_ok", 0)),
        "trials_phase1_total": int(auto_res.get("trials_phase1_total", AUTO_MODE_TRIALS)),
        "trials_phase1_ok": int(auto_res.get("trials_phase1_ok", 0)),
        "phase1_plateau_hit": bool(auto_res.get("phase1_plateau_hit", False)),
        "trials_phase2_total": int(auto_res.get("trials_phase2_total", AUTO_MODE_REFINE_TRIALS)),
        "trials_phase2_ok": int(auto_res.get("trials_phase2_ok", 0)),
        "phase2_plateau_hit": bool(auto_res.get("phase2_plateau_hit", False)),
        "trials_phase3_total": int(auto_res.get("trials_phase3_total", 0)),
        "trials_phase3_ok": int(auto_res.get("trials_phase3_ok", 0)),
        "phase4_finalize": bool(auto_res.get("phase4_finalize", False)),
        "phase4_steps": dict(auto_res.get("phase4_steps", {}) or {}),
        "search_plan_enabled_phases": list(auto_res.get("enabled_phases", []) or []),
        "search_plan_skipped_phases": list(auto_res.get("skipped_phases", []) or []),
        "search_fs": int(auto_res.get("search_fs", auto_search_fs)),
        "search_taps": int(auto_res.get("search_taps", auto_search_taps)),
        "auto_exc_seed_freq_hz": (
            float(data.get("_auto_exc_seed_freq_hz"))
            if np.isfinite(_auto_safe_float(data.get("_auto_exc_seed_freq_hz", float("nan")), float("nan")))
            else float("nan")
        ),
        "best_auto_exc_freq_hz": (
            float(data.get("_auto_exc_freq_hz"))
            if np.isfinite(_auto_safe_float(data.get("_auto_exc_freq_hz", float("nan")), float("nan")))
            else float("nan")
        ),
        "best_metrics": best_metrics,
        "best_preset": best_preset,
        "phase_limit_winner_polish": dict(auto_res.get("phase_limit_winner_polish", {}) or {}),
        "mag_c_min_winner_polish": dict(auto_res.get("mag_c_min_winner_polish", {}) or {}),
        "low_bass_cut_winner_polish": dict(auto_res.get("low_bass_cut_winner_polish", {}) or {}),
        "hpf_winner_polish": dict(auto_res.get("hpf_winner_polish", {}) or {}),
        "excess_phase_strength_winner_polish": dict(auto_res.get("excess_phase_strength_winner_polish", {}) or {}),
        "residual_peak_winner_polish": dict(auto_res.get("residual_peak_winner_polish", {}) or {}),
        "stereo_policy_refine": dict(auto_res.get("stereo_policy_refine", {}) or {}),
        "winner": {
            "rank_score_official": float(
                _auto_safe_float(
                    winner_meta.get("rank_score_official", best_metrics.get("rank_score_official", float("nan"))),
                    float("nan"),
                )
            ),
            "rank_score_components": dict(
                winner_meta.get("rank_score_components", best_metrics.get("rank_score_components", {}) or {})
            ),
        },
        "top": list(auto_res.get("top", []) or []),
        "winner_explanation": dict(auto_res.get("winner_explanation", {}) or {}),
        "audit_trail": dict(auto_res.get("audit_trail", {}) or {}),
        "residual_peak_safety_override": dict(auto_res.get("residual_peak_safety_override", {}) or {}),
    }
    best_rank_official = official_rank_score(best_metrics)
    status_cb(
        _build_auto_finalize_status(
            best_metrics,
            winner_explanation=dict(auto_res.get("winner_explanation", {}) or {}),
        )
    )
    set_auto_selected_bar(_build_auto_selected_text(data))
    _exc_pen_log_txt = (
        ""
        if str(optimizer_backend) == "optuna"
        else f"exc_pen={_auto_safe_float(best_metrics.get('exc_penalty'), 0.0):.3f}, "
    )
    _bass_boost_log = _auto_safe_float(
        best_metrics.get("bass_boost_20_200_db", best_metrics.get("max_net_boost_db")),
        0.0,
    )
    logger.info(
        "Automatic mode best: "
        f"goal={sel_goal}, basis={sel_basis}, "
        f"rank={best_rank_official:.3f}/100, "
        f"avg={_auto_safe_float(best_metrics.get('avg_score'), 0.0):.3f}, "
        f"dsp_pen={_auto_safe_float(best_metrics.get('dsp_penalty'), 0.0):.3f}, "
        f"{_exc_pen_log_txt}"
        f"boost={_bass_boost_log:.2f} dB, "
        f"net_boost={_auto_safe_float(best_metrics.get('max_net_boost_db'), 0.0):.2f} dB, "
        f"events={int(best_metrics.get('events_total', 0) or 0)}, "
        f"event_sev={_auto_safe_float(best_metrics.get('events_severity'), 0.0):.2f}"
    )

def _run_auto_mode_search_if_needed(
    ctx: dict,
    *,
    callbacks: ProcessRunCallbacks,
    support: ProcessRunSupport,
):
    if not bool(ctx["auto_mode_enabled"]):
        return

    data = ctx.get("resolved_data", ctx["data"])
    ctx["resolved_data"] = data
    ctx["data"] = data
    measurements = ctx["measurements"]
    target_rates = ctx["target_rates"]
    xos = ctx["xos"]
    hpf = ctx["hpf"]
    hc_f = ctx["hc_f"]
    hc_m = ctx["hc_m"]
    taps_base = int(ctx["taps_base"])
    auto_goal = str(ctx["auto_goal"])
    if str(auto_goal) == AUTO_MODE_GOAL_FLAT:
        data["unsafe_raw_dsp"] = True
    auto_basis = str(ctx["auto_basis"])
    auto_status = _get_auto_status_callback(ctx, callbacks=callbacks, support=support)

    try:
        data["comparison_mode"] = True
        auto_search_fs = int(target_rates[0]) if target_rates else int(data.get("fs", 44100) or 44100)
        if bool(data.get("multi_rate_opt", False)):
            auto_search_taps = int(scale_taps_with_fs(auto_search_fs, base_taps=taps_base))
        else:
            auto_search_taps = int(taps_base)
        _set_auto_progress(ctx, support=support, value=_AUTO_PROGRESS_PRESET_SEARCH_START)
        f6_hz = _auto_safe_float(
            data.get("_auto_mag_c_min_hz", data.get("mag_c_min", float("nan"))),
            float("nan"),
        )
        f6_txt = f", -6 dB point {f6_hz:.1f} Hz" if np.isfinite(f6_hz) else ""
        phase2_hint = int(AUTO_MODE_REFINE_TRIALS)
        if bool(AUTO_MODE_LOCAL_REFINE_ENABLED) and str(auto_goal) in ("balanced", "room-safe", "low-ripple", "subwoofers"):
            phase2_hint = int(AUTO_MODE_LOCAL_REFINE_TOP_K * AUTO_MODE_LOCAL_REFINE_TRIALS_PER_TOP)
        n_trials_v = max(1, int(data.get("auto_mode_trials") or AUTO_MODE_TRIALS))
        auto_status(
            f"DecayCore automatic mode: phase search init "
            f"(phase1 {n_trials_v} + phase2 {phase2_hint} + "
            f"micro {AUTO_MODE_PHASE3_MICRO_TRIALS} trials @ {auto_search_fs} Hz{f6_txt}, "
            f"goal {auto_goal}, basis {auto_basis}, target {data.get('hc_mode', 'n/a') or 'n/a'!s})"
        )
        auto_res = _run_auto_mode_search(
            base_data=dict(data),
            measurements=measurements,
            fs_v=int(auto_search_fs),
            taps_v=int(auto_search_taps),
            xos=xos,
            hpf=hpf,
            hc_f=hc_f,
            hc_m=hc_m,
            status_cb=auto_status,
            n_trials=n_trials_v,
        )
        if isinstance(auto_res, dict):
            data, measurements, best_preset, best_metrics, winner_meta, optimizer_backend = _auto_mode_apply_best_preset(
                ctx,
                data=data,
                measurements=measurements,
                auto_res=auto_res,
                auto_goal=auto_goal,
                support=support,
            )
            _auto_mode_store_search_meta(
                auto_res=auto_res,
                data=data,
                measurements=measurements,
                best_preset=best_preset,
                best_metrics=best_metrics,
                winner_meta=winner_meta,
                optimizer_backend=optimizer_backend,
                auto_search_fs=auto_search_fs,
                auto_search_taps=auto_search_taps,
                status_cb=auto_status,
                set_auto_selected_bar=callbacks.set_auto_selected_bar,
            )
        else:
            logger.warning("Automatic mode could not produce a valid best preset; using current settings.")
        _set_auto_progress(ctx, support=support, value=_AUTO_PROGRESS_FINALIZE)
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
        logger.exception("Automatic mode failed: %s: %s", type(exc).__name__, exc)
        auto_status(f"Automatic mode failed: {type(exc).__name__}: {exc}")
        _set_auto_progress(ctx, support=support, value=_AUTO_PROGRESS_FINALIZE)


__all__ = [
    '_run_auto_mode_search_if_needed',
    '_run_auto_mode_search',
]
