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


import numpy as np
import scipy.signal

from .validation_setup import (
    _freq_mask,
    _main_peak_index,
    _next_pow2,
    _safe_energy,
    _safe_rms,
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


def _temporal_energy_metrics(
    ir: np.ndarray,
    fs: int,
    *,
    pre_window_ms: float,
    early_window_ms: float,
    post_window_ms: float,
) -> dict[str, float]:
    n = ir.size
    peak_i = _main_peak_index(ir)
    pre_samp = int(round(float(pre_window_ms) * fs / 1000.0))
    early_samp = int(round(float(early_window_ms) * fs / 1000.0))
    post_samp = int(round(float(post_window_ms) * fs / 1000.0))

    pre_start = max(0, peak_i - pre_samp)
    pre_end = peak_i
    early_end = min(n, peak_i + early_samp)
    post_end = min(n, peak_i + post_samp)

    pre_e = _safe_energy(ir[pre_start:pre_end])
    early_e = _safe_energy(ir[peak_i:early_end])
    late_e = _safe_energy(ir[early_end:post_end])
    total_e = _safe_energy(ir[pre_start:post_end])

    ref = max(early_e, _EPS)
    pre_ratio_db = float(10.0 * np.log10(max(pre_e, _EPS) / ref))
    post_ratio_db = float(10.0 * np.log10(max(late_e, _EPS) / ref))
    early_ratio_db = float(10.0 * np.log10(max(early_e, _EPS) / max(total_e, _EPS)))

    return {
        "pre_energy_ratio_db": pre_ratio_db,
        "post_energy_ratio_db": post_ratio_db,
        "early_energy_ratio_db": early_ratio_db,
    }


def _fir_spectrum(fir: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute FFT spectrum and frequency axis for FIR. Returns (spectrum, freqs)."""
    n = _next_pow2(max(fir.size, 16384))
    spectrum = np.fft.rfft(fir, n=n)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return spectrum, freqs


def _gd_metrics_from_fir(
    fir: np.ndarray,
    fs: int,
    *,
    lo_hz: float = 20.0,
    hi_hz: float = 600.0,
) -> tuple[float, float, bool]:
    """Return delay-compensated GD spread metrics for an FIR or system IR."""
    try:
        spectrum, freqs = _fir_spectrum(fir, fs)
        phase = np.unwrap(np.angle(spectrum))
        df = freqs[1] - freqs[0] if freqs.size > 1 else 1.0
        omega = 2.0 * np.pi * freqs
        domega = 2.0 * np.pi * df
        if domega < _EPS:
            return 0.0, 0.0, False
        gd_sec = -np.gradient(phase, omega)
        gd_ms = gd_sec * 1000.0
        mask = _freq_mask(freqs, lo_hz, hi_hz)
        if not np.any(mask):
            return 0.0, 0.0, False
        gd_band = gd_ms[mask]
        if not np.any(np.isfinite(gd_band)):
            return 0.0, 0.0, False
        gd_finite = gd_band[np.isfinite(gd_band)]
        center = float(np.median(gd_finite))
        gd_residual = gd_finite - center
        gd_peak = float(np.percentile(np.abs(gd_residual), 95))
        gd_rms = _safe_rms(gd_residual)
        return gd_peak, gd_rms, True
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
        return 0.0, 0.0, False


def _fir_to_mag_db(
    fir: np.ndarray,
    fs: int,
    freq_axis: np.ndarray,
) -> np.ndarray:
    """Return magnitude response of FIR interpolated to freq_axis, normalized."""
    spectrum, freqs = _fir_spectrum(fir, fs)
    mag_db = 20.0 * np.log10(np.maximum(np.abs(spectrum), _EPS))
    mag_interp = np.interp(freq_axis, freqs, mag_db, left=mag_db[0], right=mag_db[-1])
    norm_mask = _freq_mask(freq_axis, 300.0, 3000.0)
    if np.any(norm_mask):
        mag_interp -= float(np.median(mag_interp[norm_mask]))
    return mag_interp


def _skip_pre_ringing(
    fir: np.ndarray,
    ir_anchor_mode: str | None,
    filter_type: str | None = None,
) -> bool:
    """Return True when pre-ringing check is not meaningful for this filter type."""
    del fir, ir_anchor_mode
    return _is_minimum_phase(filter_type)


def _is_minimum_phase(filter_type: str | None) -> bool:
    mode = str(filter_type or "").strip().lower()
    return "minimum" in mode or mode in {"min", "minimum phase"}


def _system_gd_improvement_metrics(
    measured_ir: np.ndarray | None,
    corrected_ir: np.ndarray | None,
    fs: int,
) -> dict[str, float]:
    out = {
        "gd_before_peak_ms": float("nan"),
        "gd_after_peak_ms": float("nan"),
        "gd_before_rms_ms": float("nan"),
        "gd_after_rms_ms": float("nan"),
        "gd_improvement_frac": float("nan"),
    }
    if measured_ir is None or corrected_ir is None:
        return out
    measured = np.asarray(measured_ir, dtype=float).reshape(-1)
    corrected = np.asarray(corrected_ir, dtype=float).reshape(-1)
    if measured.size < 16 or corrected.size < 16:
        return out
    before_peak, before_rms, before_ok = _gd_metrics_from_fir(measured, fs)
    after_peak, after_rms, after_ok = _gd_metrics_from_fir(corrected, fs)
    if not (before_ok and after_ok):
        return out
    components: list[tuple[float, float]] = []
    if np.isfinite(before_peak) and before_peak > 0.20 and np.isfinite(after_peak):
        components.append((0.55, float(np.clip((before_peak - after_peak) / before_peak, -1.0, 1.0))))
    if np.isfinite(before_rms) and before_rms > 0.10 and np.isfinite(after_rms):
        components.append((0.45, float(np.clip((before_rms - after_rms) / before_rms, -1.0, 1.0))))
    if components:
        denom = float(sum(weight for weight, _ in components))
        improvement = float(
            np.clip(
                sum(weight * value for weight, value in components) / max(denom, 1e-12),
                -1.0,
                1.0,
            )
        )
    else:
        improvement = float("nan")
    return {
        "gd_before_peak_ms": float(before_peak),
        "gd_after_peak_ms": float(after_peak),
        "gd_before_rms_ms": float(before_rms),
        "gd_after_rms_ms": float(after_rms),
        "gd_improvement_frac": float(improvement),
    }


def _magnitude_metrics(
    freq_axis: np.ndarray,
    mag_db: np.ndarray,
    target_mag_db: np.ndarray | None,
    *,
    measured_mag_db: np.ndarray | None = None,
    lo_hz: float,
    hi_hz: float,
    voice_lo_hz: float,
    voice_hi_hz: float,
    authority_null_risk: np.ndarray | None = None,
) -> dict[str, float]:
    # Build corrected response: measured room response + predicted filter gain
    if measured_mag_db is not None and measured_mag_db.size == mag_db.size:
        corrected = measured_mag_db + mag_db
    else:
        corrected = mag_db

    if target_mag_db is not None and target_mag_db.size == corrected.size:
        err = corrected - target_mag_db
    else:
        # Compare against smoothed local baseline
        try:
            smoothed = scipy.signal.savgol_filter(corrected, window_length=min(51, corrected.size | 1), polyorder=2)
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
            smoothed = corrected * 0.0
        err = corrected - smoothed

    band_mask = _freq_mask(freq_axis, lo_hz, hi_hz)
    voice_mask = _freq_mask(freq_axis, voice_lo_hz, voice_hi_hz)

    # Guard null-risk dips from counting as positive residual
    if authority_null_risk is not None and authority_null_risk.size == corrected.size:
        null_guard = np.asarray(authority_null_risk, dtype=float)
        null_suppress = np.clip(null_guard, 0.0, 1.0)
        positive_err = err * (1.0 - null_suppress * 0.8)
    else:
        positive_err = err

    band_err = err[band_mask] if np.any(band_mask) else np.array([0.0])
    voice_err = positive_err[voice_mask] if np.any(voice_mask) else np.array([0.0])
    bass_err = positive_err[band_mask] if np.any(band_mask) else np.array([0.0])

    mag_rms = float(np.sqrt(np.mean(band_err**2))) if band_err.size > 0 else 0.0
    mag_peak = float(np.max(np.abs(band_err))) if band_err.size > 0 else 0.0
    bass_residual_peak = float(np.max(np.maximum(bass_err, 0.0))) if bass_err.size > 0 else 0.0
    voice_peak = float(np.max(np.maximum(voice_err, 0.0))) if voice_err.size > 0 else 0.0
    voice_energy = float(np.mean(np.maximum(voice_err, 0.0))) if voice_err.size > 0 else 0.0

    return {
        "mag_rms_db": mag_rms,
        "mag_peak_db": mag_peak,
        "bass_residual_peak_db": bass_residual_peak,
        "voice_band_peak_excess_db": voice_peak,
        "voice_band_energy_excess_db": voice_energy,
    }


def _stereo_metrics(
    mag_l: np.ndarray | None,
    mag_r: np.ndarray | None,
    freq_axis: np.ndarray,
    *,
    lo_hz: float,
    hi_hz: float,
) -> dict[str, float]:
    if mag_l is None or mag_r is None or mag_l.size != mag_r.size:
        return {"stereo_delta_rms_db": 0.0, "stereo_delta_peak_db": 0.0}
    mask = _freq_mask(freq_axis, lo_hz, hi_hz)
    if not np.any(mask):
        return {"stereo_delta_rms_db": 0.0, "stereo_delta_peak_db": 0.0}
    delta = (mag_l - mag_r)[mask]
    return {
        "stereo_delta_rms_db": _safe_rms(delta),
        "stereo_delta_peak_db": float(np.max(np.abs(delta))) if delta.size > 0 else 0.0,
    }


__all__ = [
    "_temporal_energy_metrics",
    "_fir_spectrum",
    "_gd_metrics_from_fir",
    "_fir_to_mag_db",
    "_skip_pre_ringing",
    "_is_minimum_phase",
    "_magnitude_metrics",
    "_stereo_metrics",
]
