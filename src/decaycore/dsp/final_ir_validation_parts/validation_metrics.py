# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""P6 — Final FIR impulse-domain validation before accepting auto-mode winner."""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.signal

from ..dsp_config import CfgReader

from .validation_checks import (
    _fir_to_mag_db,
    _gd_metrics_from_fir,
    _magnitude_metrics,
    _skip_pre_ringing,
    _stereo_metrics,
    _system_gd_improvement_metrics,
    _temporal_energy_metrics,
)
from .validation_setup import (
    FinalIRValidationResult,
    _bump_severity,
    _safe_arr,
)

_EPS = 1e-12
_SEVERITY_ORDER: dict[str, int] = {"ok": 0, "warn": 1, "reject": 2}


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Per-channel temporal energy metrics
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# FIR spectrum computation (cached for both GD and magnitude metrics)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Group delay from FIR
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Magnitude metrics
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Stereo metrics
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Safe missing result
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _safe_missing_result() -> FinalIRValidationResult:
    nan = float("nan")
    return FinalIRValidationResult(
        valid=True,
        severity="ok",
        score_penalty=0.0,
        mag_rms_db=nan,
        mag_peak_db=nan,
        pre_energy_ratio_db=nan,
        post_energy_ratio_db=nan,
        early_energy_ratio_db=nan,
        gd_peak_ms=nan,
        gd_rms_ms=nan,
        voice_band_peak_excess_db=nan,
        voice_band_energy_excess_db=nan,
        stereo_delta_rms_db=nan,
        stereo_delta_peak_db=nan,
        bass_residual_peak_db=nan,
        reasons=("missing_final_ir_validation_inputs",),
        metrics={},
    )


def _reject_non_finite_result(reason: str) -> FinalIRValidationResult:
    """Hard-reject result for a FIR that was supplied but contains NaN/Inf.

    Distinct from `_safe_missing_result`: missing inputs are a benign "nothing
    to validate" case, but a supplied FIR that is non-finite is a corrupted
    filter and must never be reported as valid/ok.
    """
    nan = float("nan")
    return FinalIRValidationResult(
        valid=False,
        severity="reject",
        score_penalty=5.0,
        mag_rms_db=nan,
        mag_peak_db=nan,
        pre_energy_ratio_db=nan,
        post_energy_ratio_db=nan,
        early_energy_ratio_db=nan,
        gd_peak_ms=nan,
        gd_rms_ms=nan,
        voice_band_peak_excess_db=nan,
        voice_band_energy_excess_db=nan,
        stereo_delta_rms_db=nan,
        stereo_delta_peak_db=nan,
        bass_residual_peak_db=nan,
        reasons=(reason,),
        metrics={},
    )


def _final_ir_validation_thresholds(cr: CfgReader) -> dict[str, float]:
    return {
        "warn_pre": cr.float("final_ir_validation_warn_pre_energy_db", -24.0),
        "reject_pre": cr.float("final_ir_validation_reject_pre_energy_db", -18.0),
        "warn_gd": cr.float("final_ir_validation_warn_gd_peak_ms", 45.0),
        "reject_gd": cr.float("final_ir_validation_reject_gd_peak_ms", 80.0),
        "warn_gd_worsening": cr.float("final_ir_validation_warn_gd_worsening_frac", 0.10),
        "reject_gd_worsening": cr.float("final_ir_validation_reject_gd_worsening_frac", 0.35),
        "warn_voice": cr.float("final_ir_validation_warn_voice_peak_db", 6.5),
        "reject_voice": cr.float("final_ir_validation_reject_voice_peak_db", 8.0),
        "warn_stereo": cr.float("final_ir_validation_warn_stereo_delta_db", 6.5),
        "reject_stereo": cr.float("final_ir_validation_reject_stereo_delta_db", 9.5),
        "warn_bass": cr.float("final_ir_validation_warn_bass_residual_peak_db", 4.0),
        "reject_bass": cr.float("final_ir_validation_reject_bass_residual_peak_db", 10.0),
        "pre_window_ms": cr.float("final_ir_validation_pre_window_ms", 25.0),
        "post_window_ms": cr.float("final_ir_validation_post_window_ms", 250.0),
        "early_window_ms": cr.float("final_ir_validation_early_window_ms", 20.0),
        "mag_lo": cr.float("final_ir_validation_mag_lo_hz", 20.0),
        "mag_hi": cr.float("final_ir_validation_mag_hi_hz", 300.0),
        "voice_lo": cr.float("final_ir_validation_voice_lo_hz", 70.0),
        "voice_hi": cr.float("final_ir_validation_voice_hi_hz", 180.0),
        "stereo_lo": cr.float("final_ir_validation_stereo_lo_hz", 80.0),
        "stereo_hi": cr.float("final_ir_validation_stereo_hi_hz", 250.0),
    }


def _final_ir_validation_analysis_irs(
    *,
    fir_l_arr: np.ndarray,
    fir_r_arr: np.ndarray | None,
    measured_ir_l: np.ndarray | None,
    measured_ir_r: np.ndarray | None,
):
    has_stereo = fir_r_arr is not None and fir_r_arr.size >= 4
    if measured_ir_l is not None:
        mir_l = _safe_arr(measured_ir_l)
        if mir_l is not None and mir_l.size >= 4:
            analysis_l = scipy.signal.fftconvolve(mir_l, fir_l_arr, mode="full")
        else:
            analysis_l = fir_l_arr
    else:
        analysis_l = fir_l_arr

    if has_stereo:
        if measured_ir_r is not None:
            mir_r = _safe_arr(measured_ir_r)
            if mir_r is not None and mir_r.size >= 4:
                analysis_r = scipy.signal.fftconvolve(mir_r, fir_r_arr, mode="full")
            else:
                analysis_r = fir_r_arr
        else:
            analysis_r = fir_r_arr
    else:
        analysis_r = None

    return analysis_l, analysis_r, has_stereo


def _final_ir_validation_temporal_metrics(
    *,
    analysis_l: np.ndarray,
    analysis_r: np.ndarray | None,
    fs: int,
    pre_window_ms: float,
    early_window_ms: float,
    post_window_ms: float,
    skip_pre: bool,
):
    temp_l = _temporal_energy_metrics(
        analysis_l,
        fs,
        pre_window_ms=pre_window_ms,
        early_window_ms=early_window_ms,
        post_window_ms=post_window_ms,
    )
    if analysis_r is not None:
        temp_r = _temporal_energy_metrics(
            analysis_r,
            fs,
            pre_window_ms=pre_window_ms,
            early_window_ms=early_window_ms,
            post_window_ms=post_window_ms,
        )
        post_db = 0.5 * (temp_l["post_energy_ratio_db"] + temp_r["post_energy_ratio_db"])
        early_db = 0.5 * (temp_l["early_energy_ratio_db"] + temp_r["early_energy_ratio_db"])
    else:
        temp_r = None
        post_db = temp_l["post_energy_ratio_db"]
        early_db = temp_l["early_energy_ratio_db"]

    if skip_pre:
        pre_db = float("nan")
    elif analysis_r is not None:
        pre_db = 0.5 * (temp_l["pre_energy_ratio_db"] + temp_r["pre_energy_ratio_db"])
    else:
        pre_db = temp_l["pre_energy_ratio_db"]

    return temp_l, temp_r, pre_db, post_db, early_db


def _final_ir_validation_mag_and_stereo_metrics(
    *,
    freq_arr: np.ndarray | None,
    fir_l_arr: np.ndarray,
    fir_r_arr: np.ndarray | None,
    fs: int,
    target_mag_db: np.ndarray | None,
    target_mag_db_r: np.ndarray | None,
    predicted_mag_db_l: np.ndarray | None,
    predicted_mag_db_r: np.ndarray | None,
    measured_mag_db_l: np.ndarray | None,
    measured_mag_db_r: np.ndarray | None,
    authority_null_risk: np.ndarray | None,
    mag_lo: float,
    mag_hi: float,
    voice_lo: float,
    voice_hi: float,
    stereo_lo: float,
    stereo_hi: float,
):
    if freq_arr is None or freq_arr.size < 4:
        mag_metrics = {
            "mag_rms_db": float("nan"),
            "mag_peak_db": float("nan"),
            "bass_residual_peak_db": float("nan"),
            "voice_band_peak_excess_db": float("nan"),
            "voice_band_energy_excess_db": float("nan"),
        }
        stereo = {"stereo_delta_rms_db": 0.0, "stereo_delta_peak_db": 0.0}
        return mag_metrics, stereo

    tgt_arr = _safe_arr(target_mag_db, n_ref=freq_arr.size)
    null_arr = _safe_arr(authority_null_risk, n_ref=freq_arr.size)

    pred_l = _safe_arr(predicted_mag_db_l, n_ref=freq_arr.size)
    mag_db_l = pred_l if pred_l is not None else _fir_to_mag_db(fir_l_arr, fs, freq_arr)
    meas_l = _safe_arr(measured_mag_db_l, n_ref=freq_arr.size)
    mag_metrics = _magnitude_metrics(
        freq_arr,
        mag_db_l,
        tgt_arr,
        measured_mag_db=meas_l,
        lo_hz=mag_lo,
        hi_hz=mag_hi,
        voice_lo_hz=voice_lo,
        voice_hi_hz=voice_hi,
        authority_null_risk=null_arr,
    )

    if fir_r_arr is not None:
        pred_r = _safe_arr(predicted_mag_db_r, n_ref=freq_arr.size)
        mag_db_r = pred_r if pred_r is not None else _fir_to_mag_db(fir_r_arr, fs, freq_arr)
        meas_r = _safe_arr(measured_mag_db_r, n_ref=freq_arr.size)
        if meas_l is not None and meas_r is not None:
            stereo_l = meas_l + mag_db_l
            stereo_r = meas_r + mag_db_r
        else:
            stereo_l = mag_db_l
            stereo_r = mag_db_r
        tgt_r = _safe_arr(target_mag_db_r, n_ref=freq_arr.size)
        if tgt_arr is not None and tgt_r is not None:
            stereo_l = stereo_l - tgt_arr
            stereo_r = stereo_r - tgt_r
        stereo = _stereo_metrics(stereo_l, stereo_r, freq_arr, lo_hz=stereo_lo, hi_hz=stereo_hi)
    else:
        stereo = {"stereo_delta_rms_db": 0.0, "stereo_delta_peak_db": 0.0}

    return mag_metrics, stereo


def _final_ir_validation_severity_and_penalty(
    *,
    pre_db: float,
    gd_peak: float,
    gd_improvement_frac: float,
    voice_peak: float,
    stereo_peak: float,
    bass_residual_peak: float,
    thresholds: dict[str, float],
):
    severity = "ok"
    reasons: list[str] = []
    checks = (
        ("pre_energy", pre_db, "warn_pre", "reject_pre"),
        ("gd_peak", gd_peak, "warn_gd", "reject_gd"),
        ("voice_peak", voice_peak, "warn_voice", "reject_voice"),
        ("stereo_delta", stereo_peak, "warn_stereo", "reject_stereo"),
        ("bass_residual", bass_residual_peak, "warn_bass", "reject_bass"),
    )
    for stem, value, warn_key, reject_key in checks:
        warn_thr = thresholds[warn_key]
        reject_thr = thresholds[reject_key]
        if np.isfinite(value) and value > warn_thr:
            severity = _bump_severity(severity, "warn")
            reasons.append(f"{stem}_warn")
        if np.isfinite(value) and value > reject_thr:
            severity = _bump_severity(severity, "reject")
            reasons.append(f"{stem}_reject")

    penalty = 0.0
    penalty_specs = (
        (pre_db, thresholds["warn_pre"], 12.0),
        (gd_peak, thresholds["warn_gd"], 80.0),
        (voice_peak, thresholds["warn_voice"], 6.0),
        (stereo_peak, thresholds["warn_stereo"], 6.0),
        (bass_residual_peak, thresholds["warn_bass"], 8.0),
    )
    for value, warn_thr, scale in penalty_specs:
        if np.isfinite(value):
            penalty += max(0.0, value - warn_thr) / scale
    if np.isfinite(gd_improvement_frac):
        worsening = max(0.0, -float(gd_improvement_frac))
        if worsening > float(thresholds["warn_gd_worsening"]):
            severity = _bump_severity(severity, "warn")
            reasons.append("gd_worsening_warn")
        if worsening > float(thresholds["reject_gd_worsening"]):
            severity = _bump_severity(severity, "reject")
            reasons.append("gd_worsening_reject")
        penalty += worsening
    return severity, min(penalty, 5.0), reasons


def validate_final_fir_against_ir(
    *,
    sample_rate: int,
    measured_ir_l: np.ndarray | None = None,
    measured_ir_r: np.ndarray | None = None,
    fir_l: np.ndarray | None = None,
    fir_r: np.ndarray | None = None,
    freq_axis: np.ndarray | None = None,
    target_mag_db: np.ndarray | None = None,
    target_mag_db_r: np.ndarray | None = None,
    predicted_mag_db_l: np.ndarray | None = None,
    predicted_mag_db_r: np.ndarray | None = None,
    measured_mag_db_l: np.ndarray | None = None,
    measured_mag_db_r: np.ndarray | None = None,
    ir_anchor_mode: str | None = None,
    filter_type: str | None = None,
    authority_voice_risk: np.ndarray | None = None,
    authority_modal_support: np.ndarray | None = None,
    authority_null_risk: np.ndarray | None = None,
    authority_reflection_risk: np.ndarray | None = None,
    config: Any | None = None,
) -> FinalIRValidationResult:
    fs = int(sample_rate) if sample_rate else 0
    if fs <= 0 or fir_l is None:
        return _safe_missing_result()

    fir_l_arr = _safe_arr(fir_l)
    if fir_l_arr is None:
        # fir_l is known non-None here (checked above) but resolved to
        # entirely non-finite/unusable content — a corrupted filter, not
        # merely "missing" data. Reject instead of silently reporting ok.
        return _reject_non_finite_result("fir_l_non_finite_or_empty")
    if fir_l_arr.size < 4:
        return _safe_missing_result()
    if not np.all(np.isfinite(fir_l_arr)):
        return _reject_non_finite_result("fir_l_non_finite")

    fir_r_arr = _safe_arr(fir_r)
    if fir_r is not None and (fir_r_arr is None or not np.all(np.isfinite(fir_r_arr))):
        return _reject_non_finite_result("fir_r_non_finite")

    cr = CfgReader(config)
    thresholds = _final_ir_validation_thresholds(cr)
    analysis_l, analysis_r, has_stereo = _final_ir_validation_analysis_irs(
        fir_l_arr=fir_l_arr,
        fir_r_arr=fir_r_arr,
        measured_ir_l=measured_ir_l,
        measured_ir_r=measured_ir_r,
    )

    skip_pre = _skip_pre_ringing(fir_l_arr, ir_anchor_mode, filter_type)

    temp_l, temp_r, pre_db, post_db, early_db = _final_ir_validation_temporal_metrics(
        analysis_l=analysis_l,
        analysis_r=analysis_r,
        fs=fs,
        pre_window_ms=thresholds["pre_window_ms"],
        early_window_ms=thresholds["early_window_ms"],
        post_window_ms=thresholds["post_window_ms"],
        skip_pre=skip_pre,
    )
    gd_peak, gd_rms, gd_ok = _gd_metrics_from_fir(fir_l_arr, fs)
    system_gd_l = _system_gd_improvement_metrics(
        _safe_arr(measured_ir_l),
        analysis_l if measured_ir_l is not None else None,
        fs,
    )
    if analysis_r is not None and measured_ir_r is not None:
        system_gd_r = _system_gd_improvement_metrics(
            _safe_arr(measured_ir_r),
            analysis_r,
            fs,
        )
    else:
        system_gd_r = {}
    improvement_values = [
        float(v)
        for v in (
            system_gd_l.get("gd_improvement_frac", float("nan")),
            system_gd_r.get("gd_improvement_frac", float("nan")),
        )
        if np.isfinite(float(v))
    ]
    gd_improvement_frac = (
        float(np.mean(np.asarray(improvement_values, dtype=float))) if improvement_values else float("nan")
    )
    freq_arr = _safe_arr(freq_axis)
    mag_metrics, stereo = _final_ir_validation_mag_and_stereo_metrics(
        freq_arr=freq_arr,
        fir_l_arr=fir_l_arr,
        fir_r_arr=fir_r_arr if has_stereo else None,
        fs=fs,
        target_mag_db=target_mag_db,
        target_mag_db_r=target_mag_db_r,
        predicted_mag_db_l=predicted_mag_db_l,
        predicted_mag_db_r=predicted_mag_db_r,
        measured_mag_db_l=measured_mag_db_l,
        measured_mag_db_r=measured_mag_db_r,
        authority_null_risk=authority_null_risk,
        mag_lo=thresholds["mag_lo"],
        mag_hi=thresholds["mag_hi"],
        voice_lo=thresholds["voice_lo"],
        voice_hi=thresholds["voice_hi"],
        stereo_lo=thresholds["stereo_lo"],
        stereo_hi=thresholds["stereo_hi"],
    )

    bass_residual_peak = mag_metrics["bass_residual_peak_db"]
    voice_peak = mag_metrics["voice_band_peak_excess_db"]
    stereo_peak = stereo["stereo_delta_peak_db"]
    stereo_rms = stereo["stereo_delta_rms_db"]

    severity, penalty, reasons = _final_ir_validation_severity_and_penalty(
        pre_db=pre_db,
        gd_peak=gd_peak,
        gd_improvement_frac=gd_improvement_frac,
        voice_peak=voice_peak,
        stereo_peak=stereo_peak,
        bass_residual_peak=bass_residual_peak,
        thresholds=thresholds,
    )
    if not gd_ok:
        reasons.append("gd_metric_unavailable")
    if freq_arr is None or freq_arr.size < 4:
        reasons.append("no_freq_axis_for_mag_metrics")

    all_metrics: dict[str, float] = {
        "pre_energy_ratio_db": pre_db,
        "post_energy_ratio_db": post_db,
        "early_energy_ratio_db": early_db,
        "gd_peak_ms": gd_peak,
        "gd_rms_ms": gd_rms,
        "gd_before_peak_ms": float(system_gd_l.get("gd_before_peak_ms", float("nan"))),
        "gd_after_peak_ms": float(system_gd_l.get("gd_after_peak_ms", float("nan"))),
        "gd_before_rms_ms": float(system_gd_l.get("gd_before_rms_ms", float("nan"))),
        "gd_after_rms_ms": float(system_gd_l.get("gd_after_rms_ms", float("nan"))),
        "gd_improvement_frac": float(gd_improvement_frac),
        **mag_metrics,
        **stereo,
    }

    return FinalIRValidationResult(
        valid=severity != "reject",
        severity=severity,
        score_penalty=float(penalty),
        mag_rms_db=mag_metrics["mag_rms_db"],
        mag_peak_db=mag_metrics["mag_peak_db"],
        pre_energy_ratio_db=pre_db,
        post_energy_ratio_db=post_db,
        early_energy_ratio_db=early_db,
        gd_peak_ms=gd_peak,
        gd_rms_ms=gd_rms,
        voice_band_peak_excess_db=voice_peak,
        voice_band_energy_excess_db=mag_metrics["voice_band_energy_excess_db"],
        stereo_delta_rms_db=stereo_rms,
        stereo_delta_peak_db=stereo_peak,
        bass_residual_peak_db=bass_residual_peak,
        reasons=tuple(reasons),
        metrics=all_metrics,
    )


def final_ir_validation_to_stats(result: FinalIRValidationResult) -> dict[str, Any]:
    def _f(v: float) -> float:
        return float(v) if np.isfinite(float(v)) else float("nan")

    return {
        "final_ir_validation_valid": bool(result.valid),
        "final_ir_validation_severity": str(result.severity),
        "final_ir_validation_score_penalty": _f(result.score_penalty),
        "final_ir_validation_mag_rms_db": _f(result.mag_rms_db),
        "final_ir_validation_mag_peak_db": _f(result.mag_peak_db),
        "final_ir_validation_pre_energy_ratio_db": _f(result.pre_energy_ratio_db),
        "final_ir_validation_post_energy_ratio_db": _f(result.post_energy_ratio_db),
        "final_ir_validation_early_energy_ratio_db": _f(result.early_energy_ratio_db),
        "final_ir_validation_gd_peak_ms": _f(result.gd_peak_ms),
        "final_ir_validation_gd_rms_ms": _f(result.gd_rms_ms),
        "final_ir_validation_gd_before_peak_ms": _f(result.metrics.get("gd_before_peak_ms", float("nan"))),
        "final_ir_validation_gd_after_peak_ms": _f(result.metrics.get("gd_after_peak_ms", float("nan"))),
        "final_ir_validation_gd_before_rms_ms": _f(result.metrics.get("gd_before_rms_ms", float("nan"))),
        "final_ir_validation_gd_after_rms_ms": _f(result.metrics.get("gd_after_rms_ms", float("nan"))),
        "final_ir_validation_gd_improvement_frac": _f(result.metrics.get("gd_improvement_frac", float("nan"))),
        "final_ir_validation_voice_band_peak_excess_db": _f(result.voice_band_peak_excess_db),
        "final_ir_validation_voice_band_energy_excess_db": _f(result.voice_band_energy_excess_db),
        "final_ir_validation_stereo_delta_rms_db": _f(result.stereo_delta_rms_db),
        "final_ir_validation_stereo_delta_peak_db": _f(result.stereo_delta_peak_db),
        "final_ir_validation_bass_residual_peak_db": _f(result.bass_residual_peak_db),
        "final_ir_validation_reasons": list(result.reasons),
    }


__all__ = ["_safe_missing_result", "validate_final_fir_against_ir", "final_ir_validation_to_stats"]
