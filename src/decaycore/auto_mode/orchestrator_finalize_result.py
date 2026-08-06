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
import math

import numpy as np
import scipy.signal

from .audit_trail import build_auto_mode_audit_trail
from .auto_mode_profile import profiled_section
from .cache_signature import _auto_cache_stats_snapshot
from .rank_score import attach_official_rank_score
from .scoring_ranking import _auto_rank_key
from .shared_parts import AUTO_MODE_CACHE_SCHEMA_VERSION, _auto_safe_float
from .orchestrator_finalize_cache_parts import (
    _apply_residual_peak_safety_override,
    _build_modal_intelligence_debug,
    _override_candidates,
)
from ..features import PACKAGED_AUTO_ENGINE_POLICY_VERSION

logger = logging.getLogger("DecayCore")

_RECOVERABLE_P6_EXCEPTIONS = (
    AttributeError,
    TypeError,
    ValueError,
    KeyError,
    IndexError,
    RuntimeError,
    OSError,
    ImportError,
    ModuleNotFoundError,
)


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


def _p6_analysis_mode(st: dict) -> str:
    return str(st.get("analysis_mode", "native") or "native").strip().lower()


def _p6_active_stat_arr(st: dict, key: str):
    if _p6_analysis_mode(st) == "comparison":
        cmp_key = {
            "freq_axis": "cmp_freq_axis",
            "target_mags": "cmp_target_mags",
            "measured_mags": "cmp_measured_mags",
        }.get(key)
        if cmp_key:
            arr = _p6_stat_arr(st, cmp_key)
            if arr is not None:
                return arr
    return _p6_stat_arr(st, key)


def _p6_filter_mag_arr(st: dict) -> tuple[object | None, str]:
    """Pick the magnitude curve that represents the final FIR realization for P6."""
    if _p6_analysis_mode(st) == "comparison":
        realized = _p6_stat_arr(st, "cmp_realized_filter_mags")
        if realized is not None:
            return realized, "cmp_realized_filter_mags"

        filter_source = str(st.get("cmp_filter_mags_source", "") or "").strip().lower()
        filter_mags = _p6_stat_arr(st, "cmp_filter_mags")
        if filter_mags is not None and filter_source == "ir_fft_final":
            return filter_mags, "cmp_filter_mags:ir_fft_final"

        predicted = _p6_stat_arr(st, "cmp_predicted_filter_mags")
        if predicted is not None:
            return predicted, "cmp_predicted_filter_mags"

        if filter_mags is not None:
            return filter_mags, "cmp_filter_mags:legacy"

    realized = _p6_stat_arr(st, "realized_filter_mags")
    if realized is not None:
        return realized, "realized_filter_mags"

    filter_source = str(st.get("filter_mags_source", "") or "").strip().lower()
    filter_mags = _p6_stat_arr(st, "filter_mags")
    if filter_mags is not None and filter_source == "ir_fft_final":
        return filter_mags, "filter_mags:ir_fft_final"

    predicted = _p6_stat_arr(st, "predicted_filter_mags")
    if predicted is not None:
        return predicted, "predicted_filter_mags"

    if filter_mags is not None:
        return filter_mags, "filter_mags:legacy"

    return None, "missing"

def _p6_import_validation_dependencies():
    try:
        from ..dsp.final_ir_validation_parts import validate_final_fir_against_ir, final_ir_validation_to_stats
        from ..dsp.dsp_config import CfgReader
    except _RECOVERABLE_P6_EXCEPTIONS:
        return None
    return validate_final_fir_against_ir, final_ir_validation_to_stats, CfgReader

def _p6_validation_candidates(search_state, n_check: int) -> list[dict]:
    ranked = sorted(
        list(search_state.scored or []),
        key=lambda x: _auto_rank_key(dict((x or {}).get("metrics", {}) or {})),
    )
    best_preset = dict(search_state.best_preset or {})
    best_match = next(
        (
            item
            for item in ranked
            if all(repr(best_preset.get(key)) == repr(value) for key, value in dict(item.get("preset", {}) or {}).items())
        ),
        None,
    )
    if best_match is None:
        best_match = {
            "preset": best_preset,
            "metrics": dict(search_state.best_metrics or {}),
            "_p6_current_winner": True,
        }
    ordered = [best_match, *(item for item in ranked if item is not best_match)]
    return list(ordered[:n_check])


def _p6_prepare_measured_ir(
    raw_ir,
    source_fs,
    target_fs: int,
) -> np.ndarray | None:
    """Validate and, when needed, resample a measurement IR for P6."""
    try:
        arr = np.asarray(raw_ir, dtype=float).reshape(-1)
        src_fs = int(source_fs)
        dst_fs = int(target_fs)
    except (TypeError, ValueError, OverflowError):
        return None
    if arr.size < 4 or src_fs <= 0 or dst_fs <= 0 or not np.any(np.isfinite(arr)):
        return None
    arr = np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    if src_fs == dst_fs:
        return arr
    divisor = math.gcd(src_fs, dst_fs)
    try:
        resampled = scipy.signal.resample_poly(
            arr,
            up=dst_fs // divisor,
            down=src_fs // divisor,
        )
    except (TypeError, ValueError, OverflowError, MemoryError):
        return None
    resampled = np.asarray(resampled, dtype=float).reshape(-1)
    if resampled.size < 4 or not np.all(np.isfinite(resampled)):
        return None
    return resampled


def _p6_prepare_measured_irs(measurements: dict | None, target_fs: int) -> tuple:
    data = dict(measurements or {})
    raw_l = data.get("raw_ir_l")
    raw_r = data.get("raw_ir_r")
    ir_l = _p6_prepare_measured_ir(raw_l, data.get("raw_ir_fs_l", target_fs), target_fs)
    ir_r = _p6_prepare_measured_ir(raw_r, data.get("raw_ir_fs_r", target_fs), target_fs)
    if ir_l is not None and (raw_r is None or ir_r is not None):
        source = "measured_ir_convolution"
    elif ir_l is not None or ir_r is not None:
        source = "mixed_measured_ir_fir_only"
    else:
        source = "fir_only"
    return ir_l, ir_r, source

def _p6_materialize_candidate_result(
    search_state,
    cand: dict,
    index: int,
    _materialize_preset_result,
    materialize_base_data: dict | None,
):
    if index == 0 and search_state.best_result is not None:
        return search_state.best_result, dict(search_state.best_preset or {})
    result_obj, _, materialized_preset = _materialize_preset_result(
        dict(cand.get("preset", {}) or {}),
        include_response_arrays=True,
        summarize=False,
        base_data_override=materialize_base_data,
    )
    return result_obj, dict(materialized_preset or cand.get("preset", {}) or {})

def _p6_validate_candidate(
    *,
    search_state,
    cfg,
    cand: dict,
    index: int,
    validate_final_fir_against_ir,
    _materialize_preset_result,
    measured_ir_l,
    measured_ir_r,
    analysis_source: str,
    materialize_base_data: dict | None,
):
    try:
        result_obj, materialized_preset = _p6_materialize_candidate_result(
            search_state,
            cand,
            index,
            _materialize_preset_result,
            materialize_base_data,
        )
        fir_l, fir_r, st_l, st_r = _p6_extract_fir_and_stats(result_obj)
        fs = int(getattr(cfg, "fs", 48000) or 48000)
        mag_l, mag_source_l = _p6_filter_mag_arr(st_l)
        mag_r, mag_source_r = _p6_filter_mag_arr(st_r)
        validation_result = validate_final_fir_against_ir(
            sample_rate=fs,
            measured_ir_l=measured_ir_l,
            measured_ir_r=measured_ir_r,
            fir_l=fir_l,
            fir_r=fir_r,
            freq_axis=_p6_active_stat_arr(st_l, "freq_axis"),
            target_mag_db=_p6_active_stat_arr(st_l, "target_mags"),
            target_mag_db_r=_p6_active_stat_arr(st_r, "target_mags"),
            predicted_mag_db_l=mag_l,
            predicted_mag_db_r=mag_r,
            measured_mag_db_l=_p6_active_stat_arr(st_l, "measured_mags"),
            measured_mag_db_r=_p6_active_stat_arr(st_r, "measured_mags"),
            ir_anchor_mode=str(st_l.get("ir_anchor_mode", "") or ""),
            filter_type=str(
                st_l.get(
                    "filter_type",
                    st_l.get("filter_type_str", getattr(cfg, "filter_type_str", "")),
                )
                or ""
            ),
            authority_voice_risk=_p6_stat_arr(st_l, "authority_voice_risk"),
            authority_modal_support=_p6_stat_arr(st_l, "authority_modal_support"),
            authority_null_risk=_p6_stat_arr(st_l, "authority_null_risk"),
            authority_reflection_risk=_p6_stat_arr(st_l, "authority_reflection_risk"),
            config=cfg,
        )
    except _RECOVERABLE_P6_EXCEPTIONS as exc:
        logger.debug("P6 validation failed for candidate %d: %s: %s", index, type(exc).__name__, exc)
        return None
    metrics = dict((cand or {}).get("metrics", {}) or {})
    if index == 0:
        metrics.update(dict(search_state.best_metrics or {}))
    return {
        "index": int(index),
        "candidate": cand,
        "validation": validation_result,
        "result": result_obj,
        "materialized_preset": materialized_preset,
        "mag_source_l": str(mag_source_l),
        "mag_source_r": str(mag_source_r),
        "analysis_source": str(analysis_source),
        "metrics": metrics,
    }

def _p6_collect_validation_results(
    *,
    search_state,
    cfg,
    candidates: list[dict],
    validate_final_fir_against_ir,
    _materialize_preset_result,
    measured_ir_l,
    measured_ir_r,
    analysis_source: str,
    materialize_base_data: dict | None,
) -> list[dict]:
    results: list[dict] = []
    for index, cand in enumerate(candidates):
        row = _p6_validate_candidate(
            search_state=search_state,
            cfg=cfg,
            cand=cand,
            index=index,
            validate_final_fir_against_ir=validate_final_fir_against_ir,
            _materialize_preset_result=_materialize_preset_result,
            measured_ir_l=measured_ir_l,
            measured_ir_r=measured_ir_r,
            analysis_source=analysis_source,
            materialize_base_data=materialize_base_data,
        )
        if row is not None:
            results.append(row)
    return results

def _p6_result_is_usable(row: dict) -> bool:
    validation = row.get("validation")
    reasons = tuple(getattr(validation, "reasons", ()) or ())
    penalty = _auto_safe_float(getattr(validation, "score_penalty", float("nan")), float("nan"))
    metrics = dict(row.get("metrics", {}) or {})
    rank_score = _auto_safe_float(
        metrics.get("rank_score", metrics.get("rank_score_official", 0.0)),
        0.0,
    )
    return (
        validation is not None
        and "missing_final_ir_validation_inputs" not in reasons
        and np.isfinite(penalty)
        and np.isfinite(rank_score)
    )


def _p6_metrics_hard_failed(metrics: dict) -> bool:
    failures = list(metrics.get("hard_gate_failures", metrics.get("hard_gate_reasons", [])) or [])
    return bool(metrics.get("hard_gate_failed", False) or failures)


def _p6_pick_winner_result(*, results: list[dict], mode: str, score_weight: float) -> dict:
    usable = [dict(row) for row in results if _p6_result_is_usable(row)]
    for row in usable:
        metrics = dict(row.get("metrics", {}) or {})
        base_rank = _auto_safe_float(
            metrics.get("rank_score", metrics.get("rank_score_official", 0.0)),
            0.0,
        )
        raw_penalty = float(row["validation"].score_penalty)
        weighted_penalty = float(score_weight) * raw_penalty
        row["base_rank_score"] = base_rank
        row["raw_penalty"] = raw_penalty
        row["weighted_penalty"] = weighted_penalty
        row["adjusted_rank_score"] = base_rank - weighted_penalty

    safe_rows = [row for row in usable if not _p6_metrics_hard_failed(row["metrics"])]
    pool = safe_rows or usable
    if mode == "reject":
        accepted = [row for row in pool if str(row["validation"].severity) != "reject"]
        if accepted:
            pool = accepted
        else:
            logger.warning(
                "Final IR validation: all %d eligible candidate(s) rejected; selecting least total risk.",
                len(pool),
            )
    return max(pool, key=lambda row: (float(row["adjusted_rank_score"]), -int(row["index"])))

def _p6_apply_winner_validation_stats(
    *,
    search_state,
    winner_vr,
    mode: str,
    winner_row: dict,
    candidate_count: int,
    winner_mag_source_l: str,
    winner_mag_source_r: str,
    final_ir_validation_to_stats,
) -> None:
    vr_stats = final_ir_validation_to_stats(winner_vr)
    vr_stats["final_ir_validation_mode"] = str(mode)
    vr_stats["final_ir_validation_filter_mag_source_l"] = str(winner_mag_source_l)
    vr_stats["final_ir_validation_filter_mag_source_r"] = str(winner_mag_source_r)
    vr_stats.update(
        {
            "final_ir_validation_raw_score_penalty": float(winner_row["raw_penalty"]),
            "final_ir_validation_weighted_penalty": float(winner_row["weighted_penalty"]),
            # Preserve the existing field as the selection penalty after weighting.
            "final_ir_validation_score_penalty": float(winner_row["weighted_penalty"]),
            "final_ir_validation_base_rank_score": float(winner_row["base_rank_score"]),
            "final_ir_validation_adjusted_rank_score": float(winner_row["adjusted_rank_score"]),
            "final_ir_validation_candidate_index": int(winner_row["index"]) + 1,
            "final_ir_validation_candidate_count": int(candidate_count),
            "final_ir_validation_reranked": bool(int(winner_row["index"]) != 0),
            "final_ir_validation_analysis_source": str(winner_row["analysis_source"]),
            "final_ir_validation_fallback_reason": "",
        }
    )
    if isinstance(search_state.best_metrics, dict):
        search_state.best_metrics.update(vr_stats)


def _p6_record_fallback(search_state, *, reason: str, candidate_count: int, analysis_source: str) -> None:
    logger.warning("Final IR validation fallback: %s; preserving the original winner.", str(reason))
    if not isinstance(search_state.best_metrics, dict):
        return
    base_rank = _auto_safe_float(search_state.best_metrics.get("rank_score"), float("nan"))
    search_state.best_metrics.update(
        {
            "final_ir_validation_candidate_index": 1,
            "final_ir_validation_candidate_count": int(candidate_count),
            "final_ir_validation_reranked": False,
            "final_ir_validation_analysis_source": str(analysis_source),
            "final_ir_validation_fallback_reason": str(reason),
            "final_ir_validation_base_rank_score": float(base_rank),
            "final_ir_validation_adjusted_rank_score": float(base_rank),
            "final_ir_validation_raw_score_penalty": 0.0,
            "final_ir_validation_weighted_penalty": 0.0,
        }
    )

def _p6_log_winner_validation(winner_vr, winner_mag_source_l: str, winner_mag_source_r: str) -> None:
    import numpy as _np

    logger.info(
        "Final IR validation: severity=%s penalty=%.2f pre=%.1fdB gd=%.0fms "
        "voice=%.1fdB stereo=%.1fdB bass=%.1fdB mag_source_l=%s mag_source_r=%s reasons=%s",
        str(winner_vr.severity),
        float(winner_vr.score_penalty),
        float(winner_vr.pre_energy_ratio_db) if _np.isfinite(winner_vr.pre_energy_ratio_db) else float("nan"),
        float(winner_vr.gd_peak_ms) if _np.isfinite(winner_vr.gd_peak_ms) else float("nan"),
        float(winner_vr.voice_band_peak_excess_db) if _np.isfinite(winner_vr.voice_band_peak_excess_db) else float("nan"),
        float(winner_vr.stereo_delta_peak_db) if _np.isfinite(winner_vr.stereo_delta_peak_db) else float("nan"),
        float(winner_vr.bass_residual_peak_db) if _np.isfinite(winner_vr.bass_residual_peak_db) else float("nan"),
        str(winner_mag_source_l),
        str(winner_mag_source_r),
        ",".join(winner_vr.reasons) or "none",
    )


def _run_p6_final_validation(
    search_state,
    cfg,
    *,
    _materialize_preset_result,
    measurements: dict | None = None,
    materialize_base_data: dict | None = None,
) -> None:
    """Run P6 final IR validation; attach stats to search_state.best_metrics. No-op on errors."""
    imports = _p6_import_validation_dependencies()
    if imports is None:
        if bool(getattr(cfg, "final_ir_validation_enable", True)):
            fs = int(getattr(cfg, "fs", 48000) or 48000)
            _, _, analysis_source = _p6_prepare_measured_irs(measurements, fs)
            _p6_record_fallback(
                search_state,
                reason="validation_dependencies_unavailable",
                candidate_count=0,
                analysis_source=analysis_source,
            )
        return
    validate_final_fir_against_ir, final_ir_validation_to_stats, CfgReader = imports

    candidate_count = 0
    analysis_source = "fir_only"
    try:
        cr = CfgReader(cfg)
        if not cr.bool("final_ir_validation_enable", True):
            return

        mode_raw = cr.enum_string("final_ir_validation_mode", "warn").strip().lower()
        mode = mode_raw if mode_raw in ("warn", "reject") else "warn"
        n_check = int(np.clip(round(cr.float_allow_zero("final_ir_validation_candidate_count", 3)), 1, 5))
        score_weight = max(0.0, cr.float_allow_zero("final_ir_validation_score_weight", 1.0))
        candidates = _p6_validation_candidates(search_state, int(n_check))
        candidate_count = len(candidates)
        fs = int(getattr(cfg, "fs", 48000) or 48000)
        measured_ir_l, measured_ir_r, analysis_source = _p6_prepare_measured_irs(measurements, fs)
        results = _p6_collect_validation_results(
            search_state=search_state,
            cfg=cfg,
            candidates=candidates,
            validate_final_fir_against_ir=validate_final_fir_against_ir,
            _materialize_preset_result=_materialize_preset_result,
            measured_ir_l=measured_ir_l,
            measured_ir_r=measured_ir_r,
            analysis_source=analysis_source,
            materialize_base_data=materialize_base_data,
        )
        if not results:
            _p6_record_fallback(
                search_state,
                reason="no_candidates_validated",
                candidate_count=len(candidates),
                analysis_source=analysis_source,
            )
            return
        original_row = next((row for row in results if int(row["index"]) == 0), None)
        if original_row is None or not _p6_result_is_usable(original_row):
            reason = "original_candidate_validation_failed"
            if original_row is not None:
                reason = "original_candidate_missing_validation_inputs"
            _p6_record_fallback(
                search_state,
                reason=reason,
                candidate_count=len(candidates),
                analysis_source=analysis_source,
            )
            return

        winner_row = _p6_pick_winner_result(
            results=results,
            mode=str(mode),
            score_weight=float(score_weight),
        )
        winner_idx = int(winner_row["index"])
        winner_cand = winner_row["candidate"]
        winner_vr = winner_row["validation"]
        if winner_idx != 0:
            search_state.best_result = winner_row["result"]
            search_state.best_preset = dict(winner_row["materialized_preset"] or {})
            search_state.best_metrics = dict(
                attach_official_rank_score(dict((winner_cand or {}).get("metrics", {}) or {}))
            )
        _p6_apply_winner_validation_stats(
            search_state=search_state,
            winner_vr=winner_vr,
            mode=str(mode),
            winner_row=winner_row,
            candidate_count=len(candidates),
            winner_mag_source_l=str(winner_row["mag_source_l"]),
            winner_mag_source_r=str(winner_row["mag_source_r"]),
            final_ir_validation_to_stats=final_ir_validation_to_stats,
        )
        search_state.final_ir_validation_result = winner_vr
        logger.info(
            "Final IR validation ranking: selected candidate #%d/%d base=%.3f penalty=%.3f adjusted=%.3f reranked=%s source=%s",
            winner_idx + 1,
            len(candidates),
            float(winner_row["base_rank_score"]),
            float(winner_row["weighted_penalty"]),
            float(winner_row["adjusted_rank_score"]),
            bool(winner_idx != 0),
            str(winner_row["analysis_source"]),
        )
        _p6_log_winner_validation(
            winner_vr=winner_vr,
            winner_mag_source_l=str(winner_row["mag_source_l"]),
            winner_mag_source_r=str(winner_row["mag_source_r"]),
        )
    except _RECOVERABLE_P6_EXCEPTIONS as exc:
        logger.debug("P6 final IR validation raised: %s: %s", type(exc).__name__, exc)
        _p6_record_fallback(
            search_state,
            reason=f"validation_exception:{type(exc).__name__}",
            candidate_count=candidate_count,
            analysis_source=analysis_source,
        )


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
    phase4_steps = {
        "pareto_finalize": True,
        "winner_polish": True,
        "final_validation": True,
        "cache_save": True,
    }
    cache_info = {
        "cache_schema_version": int(AUTO_MODE_CACHE_SCHEMA_VERSION),
        "cache_stats": _auto_cache_stats_snapshot(),
    }
    audit_trail = build_auto_mode_audit_trail(
        best_metrics=best_metrics,
        best_preset=materialized_best_preset or cached_best_preset,
        winner_explanation=dict(search_state.winner_explanation or {}),
        residual_peak_safety_override_meta=dict(residual_peak_safety_override_meta or {}),
        optimizer_backend=str(optimizer_backend or "builtin"),
        goal=str(goal),
        selection_basis=str(rank_basis),
        target_name=str(dict(search_state.winner_explanation or {}).get("target_name", "") or ""),
        top=list(top or []),
        cache_info=cache_info,
        polish_meta=polish_meta,
        phase1_ok=int(phase1_ok),
        phase2_ok=int(phase2_ok),
        phase1_tried=int(phase1_tried),
        phase2_tried=int(phase2_tried),
        phase1_plateau_hit=bool(phase1_plateau_hit),
        phase2_plateau_hit=bool(phase2_plateau_hit),
        phase3_total=int(dict(phase3_micro_optuna_tel or {}).get("n_total", 0) or 0),
        phase3_ok=int(dict(phase3_micro_optuna_tel or {}).get("ok", 0) or 0),
        phase4_steps=phase4_steps,
        fs_v=int(fs_v),
        taps_v=int(taps_v),
        trials_total=int(phase1_tried + phase2_tried),
        trials_ok=int(len(search_state.scored)),
        source="search",
    )
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
            "auto_engine_policy_version": int(PACKAGED_AUTO_ENGINE_POLICY_VERSION),
            "cache_schema_version": int(AUTO_MODE_CACHE_SCHEMA_VERSION),
            "cache_stats": dict(cache_info.get("cache_stats", {}) or {}),
            "winning_score_breakdown": dict(best_metrics.get("rank_score_breakdown", {}) or {}),
            "top3_score_breakdowns": _build_top_score_breakdowns(top),
            "residual_peak_safety_override": dict(residual_peak_safety_override_meta or {}),
            "modal_intelligence": _build_modal_intelligence_debug(best_metrics, polish_meta),
        },
        "audit_trail": dict(audit_trail),
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
        "phase4_steps": dict(phase4_steps),
        "optuna_phase1_telemetry": dict(phase1_optuna_tel or {}),
        "optuna_phase2_local_telemetry": list(phase2_local_optuna_tels or []),
        "optuna_phase3_micro_telemetry": dict(phase3_micro_optuna_tel or {}),
        "optuna_phase2_rollup_telemetry": dict(phase2_rollup_tel or {}),
        "phase1_plateau_hit": bool(phase1_plateau_hit),
        "phase2_plateau_hit": bool(phase2_plateau_hit),
        "search_fs": int(fs_v),
        "search_taps": int(taps_v),
    }
