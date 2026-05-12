# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Finalize-stage result building and final search winner materialization for automatic mode."""

from __future__ import annotations

import logging

import numpy as np

from .auto_mode_profile import profiled_section
from .cache_signature import _auto_cache_stats_snapshot
from .rank_score import attach_official_rank_score
from .scoring_ranking import _auto_rank_key
from .shared import AUTO_MODE_CACHE_SCHEMA_VERSION, _auto_safe_float, _m
from .orchestrator_finalize_cache import (
    _apply_residual_peak_safety_override,
    _build_modal_intelligence_debug,
    _override_candidates,
    _resolve_winner_auto_exc_hz,
    _stereo_refine_materialize_base_data,
)

logger = logging.getLogger("DecayCore")


def _materialize_final_search_winner(
    *,
    search_state,
    search_materialize_base_data: dict,
    cfg,
    goal: str,
    winner_target_name: str | None,
    residual_peak_safety_override_meta: dict,
    _materialize_preset_result,
) -> tuple[bool, dict, dict, dict]:
    final_best_preset = dict(search_state.best_preset or {})
    try:
        with profiled_section("finalize.materialize_best"):
            best_result, best_metrics_recalc, best_data = _materialize_preset_result(
                final_best_preset,
                include_response_arrays=True,
                summarize=True,
                base_data_override=search_materialize_base_data,
            )
        search_state.best_result = best_result
        search_state.best_metrics = dict(best_metrics_recalc or {})
        search_state.best_preset = dict(best_data or final_best_preset or {})
        residual_peak_safety_override_meta_final = _apply_residual_peak_safety_override(
            search_state=search_state,
            cfg=cfg,
            goal=goal,
            winner_target_name=winner_target_name,
            phase_label="final materialization residual_peak safety override",
            candidate_items=_override_candidates(search_state),
        )
        if bool(residual_peak_safety_override_meta_final.get("applied", False)):
            final_best_preset = dict(search_state.best_preset or {})
            best_result, best_metrics_recalc, best_data = _materialize_preset_result(
                final_best_preset,
                include_response_arrays=True,
                summarize=True,
                base_data_override=search_materialize_base_data,
            )
            search_state.best_result = best_result
            search_state.best_metrics = dict(best_metrics_recalc or search_state.best_metrics or {})
            search_state.best_preset = dict(best_data or final_best_preset or {})
            residual_peak_safety_override_meta = dict(residual_peak_safety_override_meta_final)
    except Exception as exc:
        # Materialization can fail on late-stage result packaging; keep the last known winner if available.
        logger.warning(
            "Automatic mode final materialization failed: %s",
            f"{type(exc).__name__}: {exc}",
        )
        if search_state.best_result is None:
            return False, dict(residual_peak_safety_override_meta or {}), {}, {}

    materialized_best_preset = dict(search_state.best_preset or {})
    return (
        True,
        dict(residual_peak_safety_override_meta or {}),
        materialized_best_preset,
        final_best_preset,
    )


def _p6_extract_fir_and_stats(result) -> tuple:
    """Extract (fir_l, fir_r, st_l, st_r) from a FilterResult safely."""
    try:
        fir_l = getattr(result, "l_ir", None)
        fir_r = getattr(result, "r_ir", None)
        st_l = dict(getattr(result, "l_st", None) or {})
        st_r = dict(getattr(result, "r_st", None) or {})
        return fir_l, fir_r, st_l, st_r
    except Exception:
        return None, None, {}, {}


def _p6_stat_arr(st: dict, key: str):
    """Safely extract a numpy array from a stats dict list value."""
    import numpy as _np
    try:
        v = st.get(key)
        if v is None:
            return None
        arr = _np.asarray(v, dtype=float).reshape(-1)
        return arr if arr.size > 0 and _np.any(_np.isfinite(arr)) else None
    except Exception:
        return None


def _run_p6_final_validation(
    search_state,
    cfg,
    *,
    _materialize_preset_result,
) -> None:
    """Run P6 final IR validation; attach stats to search_state.best_metrics. No-op on errors."""
    try:
        from ..dsp.final_ir_validation import validate_final_fir_against_ir, final_ir_validation_to_stats
        from ..dsp.dsp_config import CfgReader
    except Exception:
        return

    try:
        cr = CfgReader(cfg)
        if not cr.bool("final_ir_validation_enable", True):
            return

        mode = cr.enum_string("final_ir_validation_mode", "warn")
        n_check = max(1, int(round(float(cr.float_allow_zero("final_ir_validation_candidate_count", 3)))))
        score_weight = cr.float("final_ir_validation_score_weight", 1.0)

        ranked = sorted(
            list(search_state.scored or []),
            key=lambda x: _auto_rank_key(dict((x or {}).get("metrics", {}) or {})),
        )

        results: list[tuple[int, object, object]] = []
        for i, cand in enumerate(ranked[:n_check]):
            try:
                if i == 0 and search_state.best_result is not None:
                    result_obj = search_state.best_result
                else:
                    result_obj, _, _ = _materialize_preset_result(
                        dict(cand.get("preset", {}) or {}),
                        include_response_arrays=True,
                        summarize=False,
                    )
                fir_l, fir_r, st_l, st_r = _p6_extract_fir_and_stats(result_obj)
                import numpy as _np
                fs = int(getattr(cfg, "fs", 48000) or 48000)
                vr = validate_final_fir_against_ir(
                    sample_rate=fs,
                    fir_l=fir_l,
                    fir_r=fir_r,
                    freq_axis=_p6_stat_arr(st_l, "freq_axis"),
                    target_mag_db=_p6_stat_arr(st_l, "target_mags"),
                    predicted_mag_db_l=_p6_stat_arr(st_l, "predicted_filter_mags"),
                    predicted_mag_db_r=_p6_stat_arr(st_r, "predicted_filter_mags"),
                    measured_mag_db_l=_p6_stat_arr(st_l, "measured_mags"),
                    measured_mag_db_r=_p6_stat_arr(st_r, "measured_mags"),
                    ir_anchor_mode=str(st_l.get("ir_anchor_mode", "") or ""),
                    authority_voice_risk=_p6_stat_arr(st_l, "authority_voice_risk"),
                    authority_modal_support=_p6_stat_arr(st_l, "authority_modal_support"),
                    authority_null_risk=_p6_stat_arr(st_l, "authority_null_risk"),
                    authority_reflection_risk=_p6_stat_arr(st_l, "authority_reflection_risk"),
                    config=cfg,
                )
                results.append((i, cand, vr, result_obj))
            except Exception as exc:
                logger.debug("P6 validation failed for candidate %d: %s: %s", i, type(exc).__name__, exc)

        if not results:
            return

        winner_idx, winner_cand, winner_vr, winner_result_obj = results[0]

        if mode == "reject":
            non_rejected = [(idx, c, r, ro) for idx, c, r, ro in results if r.severity != "reject"]
            if non_rejected:
                new_idx, new_cand, winner_vr, winner_result_obj = non_rejected[0]
                if new_idx != 0:
                    logger.info(
                        "Final IR validation: candidate #1 rejected (severity=%s), selected candidate #%d.",
                        str(results[0][2].severity),
                        int(new_idx) + 1,
                    )
                    search_state.best_result = winner_result_obj
                    search_state.best_preset = dict((new_cand or {}).get("preset", {}) or {})
                    search_state.best_metrics = dict(
                        attach_official_rank_score(dict((new_cand or {}).get("metrics", {}) or {}))
                    )
            else:
                # All rejected — keep the first (least-bad)
                logger.warning(
                    "Final IR validation: all %d checked candidate(s) rejected; keeping least-bad.",
                    len(results),
                )

        vr_stats = final_ir_validation_to_stats(winner_vr)
        if isinstance(search_state.best_metrics, dict):
            search_state.best_metrics.update(vr_stats)
            if winner_vr.score_penalty > 0.0:
                existing_penalty = float(
                    search_state.best_metrics.get("final_ir_validation_score_penalty", 0.0) or 0.0
                )
                search_state.best_metrics["final_ir_validation_score_penalty"] = float(
                    existing_penalty * float(score_weight)
                )

        logger.info(
            "Final IR validation: severity=%s penalty=%.2f pre=%.1fdB gd=%.0fms "
            "voice=%.1fdB stereo=%.1fdB bass=%.1fdB reasons=%s",
            str(winner_vr.severity),
            float(winner_vr.score_penalty),
            float(winner_vr.pre_energy_ratio_db) if _np.isfinite(winner_vr.pre_energy_ratio_db) else float("nan"),
            float(winner_vr.gd_peak_ms) if _np.isfinite(winner_vr.gd_peak_ms) else float("nan"),
            float(winner_vr.voice_band_peak_excess_db) if _np.isfinite(winner_vr.voice_band_peak_excess_db) else float("nan"),
            float(winner_vr.stereo_delta_peak_db) if _np.isfinite(winner_vr.stereo_delta_peak_db) else float("nan"),
            float(winner_vr.bass_residual_peak_db) if _np.isfinite(winner_vr.bass_residual_peak_db) else float("nan"),
            ",".join(winner_vr.reasons) or "none",
        )
    except Exception as exc:
        logger.debug("P6 final IR validation raised: %s: %s", type(exc).__name__, exc)


def _build_top_score_breakdowns(top: list[dict] | None) -> list[dict]:
    top3_breakdowns = []
    for item in list(top or [])[:3]:
        metrics = dict((item or {}).get("metrics", {}) or {})
        breakdown = dict(metrics.get("rank_score_breakdown", {}) or {})
        if breakdown:
            top3_breakdowns.append(breakdown)
    return list(top3_breakdowns)


def _build_final_search_result(
    *,
    search_state,
    cached_best_preset: dict,
    materialized_best_preset: dict,
    polish_meta: dict,
    residual_peak_safety_override_meta: dict,
    optimizer_backend: str,
    best_auto_exc_hz: float,
    goal: str,
    rank_basis: str,
    top: list,
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
    fs_v: int,
    taps_v: int,
) -> dict:
    best_metrics = dict(search_state.best_metrics or {})
    winner_rank = float(
        _auto_safe_float(
            best_metrics.get("rank_score_official", best_metrics.get("rank_score", float("nan"))),
            float("nan"),
        )
    )
    winner_components = dict(best_metrics.get("rank_score_components", {}) or {})
    return {
        "best_result": search_state.best_result,
        "best_metrics": dict(best_metrics),
        "best_preset": dict(cached_best_preset or {}),
        "best_applied_preset": dict(materialized_best_preset or {}),
        "winner": {
            "rank_score_official": float(winner_rank) if np.isfinite(winner_rank) else float("nan"),
            "rank_score_components": dict(winner_components),
            "rank_score_breakdown": dict(best_metrics.get("rank_score_breakdown", {}) or {}),
        },
        "auto_mode_debug": {
            "cache_schema_version": int(AUTO_MODE_CACHE_SCHEMA_VERSION),
            "cache_stats": _auto_cache_stats_snapshot(),
            "winning_score_breakdown": dict(best_metrics.get("rank_score_breakdown", {}) or {}),
            "top3_score_breakdowns": _build_top_score_breakdowns(top),
            "residual_peak_safety_override": dict(residual_peak_safety_override_meta or {}),
            "modal_intelligence": _build_modal_intelligence_debug(best_metrics, polish_meta),
        },
        "residual_peak_safety_override": dict(residual_peak_safety_override_meta or {}),
        "winner_explanation": dict(search_state.winner_explanation or {}),
        "phase_limit_winner_polish": dict(polish_meta.get("phase_limit_winner_polish", {}) or {}),
        "mag_c_min_winner_polish": dict(polish_meta.get("mag_c_min_winner_polish", {}) or {}),
        "low_bass_cut_winner_polish": dict(polish_meta.get("low_bass_cut_winner_polish", {}) or {}),
        "hpf_winner_polish": dict(polish_meta.get("hpf_winner_polish", {}) or {}),
        "excess_phase_strength_winner_polish": dict(polish_meta.get("excess_phase_strength_winner_polish", {}) or {}),
        "residual_peak_winner_polish": dict(polish_meta.get("residual_peak_winner_polish", {}) or {}),
        "tdc_strength_winner_polish": dict(polish_meta.get("tdc_strength_winner_polish", {}) or {}),
        "stereo_policy_refine": dict(polish_meta.get("stereo_policy_refine", {}) or {}),
        "optimizer_backend": str(optimizer_backend or "builtin"),
        "best_auto_exc_freq_hz": float(best_auto_exc_hz) if np.isfinite(best_auto_exc_hz) else float("nan"),
        "auto_goal": str(goal),
        "selection_basis": str(rank_basis),
        "top": top,
        "trials_total": int(phase1_tried + phase2_tried),
        "trials_ok": int(len(search_state.scored)),
        "trials_phase1_total": int(phase1_tried),
        "trials_phase1_ok": int(phase1_ok),
        "trials_phase2_total": int(phase2_tried),
        "trials_phase2_ok": int(phase2_ok),
        "trials_phase3_total": int(dict(phase3_micro_optuna_tel or {}).get("n_total", 0) or 0),
        "trials_phase3_ok": int(dict(phase3_micro_optuna_tel or {}).get("ok", 0) or 0),
        "phase4_finalize": True,
        "phase4_steps": {
            "pareto_finalize": True,
            "winner_polish": True,
            "final_validation": True,
            "cache_save": True,
        },
        "optuna_phase1_telemetry": dict(phase1_optuna_tel or {}),
        "optuna_phase2_local_telemetry": list(phase2_local_optuna_tels or []),
        "optuna_phase3_micro_telemetry": dict(phase3_micro_optuna_tel or {}),
        "optuna_phase2_rollup_telemetry": dict(phase2_rollup_tel or {}),
        "phase1_plateau_hit": bool(phase1_plateau_hit),
        "phase2_plateau_hit": bool(phase2_plateau_hit),
        "search_fs": int(fs_v),
        "search_taps": int(taps_v),
    }

