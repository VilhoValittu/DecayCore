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

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.signal

from ..dsp_config import CfgReader

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

def validate_final_fir_against_ir(
    *,
    sample_rate: int,
    measured_ir_l: np.ndarray | None = None,
    measured_ir_r: np.ndarray | None = None,
    fir_l: np.ndarray | None = None,
    fir_r: np.ndarray | None = None,
    freq_axis: np.ndarray | None = None,
    target_mag_db: np.ndarray | None = None,
    predicted_mag_db_l: np.ndarray | None = None,
    predicted_mag_db_r: np.ndarray | None = None,
    measured_mag_db_l: np.ndarray | None = None,
    measured_mag_db_r: np.ndarray | None = None,
    ir_anchor_mode: str | None = None,
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
    if fir_l_arr is None or fir_l_arr.size < 4:
        return _safe_missing_result()

    fir_r_arr = _safe_arr(fir_r)
    has_stereo = fir_r_arr is not None and fir_r_arr.size >= 4

    cr = CfgReader(config)

    # Config thresholds
    warn_pre = cr.float("final_ir_validation_warn_pre_energy_db", -24.0)
    reject_pre = cr.float("final_ir_validation_reject_pre_energy_db", -18.0)
    warn_gd = cr.float("final_ir_validation_warn_gd_peak_ms", 45.0)
    reject_gd = cr.float("final_ir_validation_reject_gd_peak_ms", 80.0)
    warn_voice = cr.float("final_ir_validation_warn_voice_peak_db", 3.0)
    reject_voice = cr.float("final_ir_validation_reject_voice_peak_db", 5.0)
    warn_stereo = cr.float("final_ir_validation_warn_stereo_delta_db", 3.0)
    reject_stereo = cr.float("final_ir_validation_reject_stereo_delta_db", 5.0)
    warn_bass = cr.float("final_ir_validation_warn_bass_residual_peak_db", 4.0)
    reject_bass = cr.float("final_ir_validation_reject_bass_residual_peak_db", 7.0)
    pre_window_ms = cr.float("final_ir_validation_pre_window_ms", 25.0)
    post_window_ms = cr.float("final_ir_validation_post_window_ms", 250.0)
    early_window_ms = cr.float("final_ir_validation_early_window_ms", 20.0)
    mag_lo = cr.float("final_ir_validation_mag_lo_hz", 20.0)
    mag_hi = cr.float("final_ir_validation_mag_hi_hz", 300.0)
    voice_lo = cr.float("final_ir_validation_voice_lo_hz", 70.0)
    voice_hi = cr.float("final_ir_validation_voice_hi_hz", 180.0)

    # Determine analysis IRs
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

    skip_pre = _skip_pre_ringing(fir_l_arr, ir_anchor_mode)
    min_phase = _is_minimum_phase(ir_anchor_mode)

    # Temporal metrics per channel, then average
    temp_l = _temporal_energy_metrics(
        analysis_l, fs,
        pre_window_ms=pre_window_ms,
        early_window_ms=early_window_ms,
        post_window_ms=post_window_ms,
    )
    if analysis_r is not None:
        temp_r = _temporal_energy_metrics(
            analysis_r, fs,
            pre_window_ms=pre_window_ms,
            early_window_ms=early_window_ms,
            post_window_ms=post_window_ms,
        )
        post_db = 0.5 * (temp_l["post_energy_ratio_db"] + temp_r["post_energy_ratio_db"])
        early_db = 0.5 * (temp_l["early_energy_ratio_db"] + temp_r["early_energy_ratio_db"])
    else:
        post_db = temp_l["post_energy_ratio_db"]
        early_db = temp_l["early_energy_ratio_db"]

    # Pre-ringing: NaN for linear-phase and minimum-phase filters (not a meaningful metric)
    if skip_pre:
        pre_db = float("nan")
    elif analysis_r is not None:
        pre_db = 0.5 * (temp_l["pre_energy_ratio_db"] + temp_r["pre_energy_ratio_db"])
    else:
        pre_db = temp_l["pre_energy_ratio_db"]

    # Group delay (from FIR, not analysis IR); skip for minimum-phase (expected large GD is causal)
    reasons: list[str] = []
    if min_phase:
        gd_peak, gd_rms, gd_ok = float("nan"), float("nan"), True
    else:
        gd_peak, gd_rms, gd_ok = _gd_metrics_from_fir(fir_l_arr, fs)
    if not gd_ok:
        reasons.append("gd_metric_unavailable")

    # Magnitude metrics
    freq_arr = _safe_arr(freq_axis)
    tgt_arr = _safe_arr(target_mag_db, n_ref=freq_arr.size if freq_arr is not None else None)
    null_arr = _safe_arr(authority_null_risk, n_ref=freq_arr.size if freq_arr is not None else None)

    if freq_arr is not None and freq_arr.size >= 4:
        pred_l = _safe_arr(predicted_mag_db_l, n_ref=freq_arr.size)
        if pred_l is not None:
            mag_db_l = pred_l
        else:
            mag_db_l = _fir_to_mag_db(fir_l_arr, fs, freq_arr)

        meas_l = _safe_arr(measured_mag_db_l, n_ref=freq_arr.size)
        mag_metrics = _magnitude_metrics(
            freq_arr, mag_db_l, tgt_arr,
            measured_mag_db=meas_l,
            lo_hz=mag_lo, hi_hz=mag_hi,
            voice_lo_hz=voice_lo, voice_hi_hz=voice_hi,
            authority_null_risk=null_arr,
        )

        # Stereo magnitude metrics
        if has_stereo:
            pred_r = _safe_arr(predicted_mag_db_r, n_ref=freq_arr.size)
            if pred_r is not None:
                mag_db_r = pred_r
            else:
                mag_db_r = _fir_to_mag_db(fir_r_arr, fs, freq_arr)
            stereo = _stereo_metrics(mag_db_l, mag_db_r, freq_arr, lo_hz=mag_lo, hi_hz=mag_hi)
        else:
            stereo = {"stereo_delta_rms_db": 0.0, "stereo_delta_peak_db": 0.0}
    else:
        mag_metrics = {
            "mag_rms_db": float("nan"),
            "mag_peak_db": float("nan"),
            "bass_residual_peak_db": float("nan"),
            "voice_band_peak_excess_db": float("nan"),
            "voice_band_energy_excess_db": float("nan"),
        }
        stereo = {"stereo_delta_rms_db": 0.0, "stereo_delta_peak_db": 0.0}
        reasons.append("no_freq_axis_for_mag_metrics")

    bass_residual_peak = mag_metrics["bass_residual_peak_db"]
    voice_peak = mag_metrics["voice_band_peak_excess_db"]
    stereo_peak = stereo["stereo_delta_peak_db"]
    stereo_rms = stereo["stereo_delta_rms_db"]

    # ---- Severity decision ----
    severity = "ok"

    if np.isfinite(pre_db) and pre_db > warn_pre:
        severity = _bump_severity(severity, "warn")
        reasons.append("pre_energy_warn")
    if np.isfinite(pre_db) and pre_db > reject_pre:
        severity = _bump_severity(severity, "reject")
        reasons.append("pre_energy_reject")

    if np.isfinite(gd_peak) and gd_peak > warn_gd:
        severity = _bump_severity(severity, "warn")
        reasons.append("gd_peak_warn")
    if np.isfinite(gd_peak) and gd_peak > reject_gd:
        severity = _bump_severity(severity, "reject")
        reasons.append("gd_peak_reject")

    if np.isfinite(voice_peak) and voice_peak > warn_voice:
        severity = _bump_severity(severity, "warn")
        reasons.append("voice_peak_warn")
    if np.isfinite(voice_peak) and voice_peak > reject_voice:
        severity = _bump_severity(severity, "reject")
        reasons.append("voice_peak_reject")

    if np.isfinite(stereo_peak) and stereo_peak > warn_stereo:
        severity = _bump_severity(severity, "warn")
        reasons.append("stereo_delta_warn")
    if np.isfinite(stereo_peak) and stereo_peak > reject_stereo:
        severity = _bump_severity(severity, "reject")
        reasons.append("stereo_delta_reject")

    if np.isfinite(bass_residual_peak) and bass_residual_peak > warn_bass:
        severity = _bump_severity(severity, "warn")
        reasons.append("bass_residual_warn")
    if np.isfinite(bass_residual_peak) and bass_residual_peak > reject_bass:
        severity = _bump_severity(severity, "reject")
        reasons.append("bass_residual_reject")

    # ---- Penalty ----
    penalty = 0.0
    if np.isfinite(pre_db):
        penalty += max(0.0, pre_db - warn_pre) / 12.0
    if np.isfinite(gd_peak):
        penalty += max(0.0, gd_peak - warn_gd) / 80.0
    if np.isfinite(voice_peak):
        penalty += max(0.0, voice_peak - warn_voice) / 6.0
    if np.isfinite(stereo_peak):
        penalty += max(0.0, stereo_peak - warn_stereo) / 6.0
    if np.isfinite(bass_residual_peak):
        penalty += max(0.0, bass_residual_peak - warn_bass) / 8.0
    penalty = min(penalty, 5.0)

    all_metrics: dict[str, float] = {
        "pre_energy_ratio_db": pre_db,
        "post_energy_ratio_db": post_db,
        "early_energy_ratio_db": early_db,
        "gd_peak_ms": gd_peak,
        "gd_rms_ms": gd_rms,
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
        "final_ir_validation_voice_band_peak_excess_db": _f(result.voice_band_peak_excess_db),
        "final_ir_validation_voice_band_energy_excess_db": _f(result.voice_band_energy_excess_db),
        "final_ir_validation_stereo_delta_rms_db": _f(result.stereo_delta_rms_db),
        "final_ir_validation_stereo_delta_peak_db": _f(result.stereo_delta_peak_db),
        "final_ir_validation_bass_residual_peak_db": _f(result.bass_residual_peak_db),
        "final_ir_validation_reasons": list(result.reasons),
    }


__all__ = ['_safe_missing_result', 'validate_final_fir_against_ir', 'final_ir_validation_to_stats']


def _load_sibling_symbols() -> None:
    import importlib
    package = __package__
    for module_name in ['final_ir_validation_01', 'final_ir_validation_02', 'final_ir_validation_03']:
        if module_name == __name__.rsplit('.', 1)[-1]:
            continue
        module = importlib.import_module(f"{package}.{module_name}")
        for symbol in getattr(module, "__all__", ()):
            globals().setdefault(symbol, getattr(module, symbol))


_load_sibling_symbols()
