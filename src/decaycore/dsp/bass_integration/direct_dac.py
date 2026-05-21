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

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._filters import _butterworth_complex_response

_OVERLAP_HZ = 20.0
_SUB_HPF_HZ = 20.0
_XO_ORDER = 4
_SUB_HPF_ORDER = 2


@dataclass(frozen=True)
class DirectDacBassIntegrationResult:
    """Compatibility result for legacy array-level Direct-DAC helper output."""

    enabled: bool
    main_hpf_hz: float
    sub_hpf_hz: float
    sub_lpf_hz: float
    sub_overlap_hz: float
    sub_delay_ms: float
    sub_gain_db: float
    sub_polarity_invert: bool
    phase_error_deg: float
    gd_mismatch_ms: float
    cancellation_risk: str
    cancellation_score: float
    magnitude_ripple_db: float
    score: float
    rejected: bool = False
    reject_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "main_hpf_hz": float(self.main_hpf_hz),
            "sub_hpf_hz": float(self.sub_hpf_hz),
            "sub_lpf_hz": float(self.sub_lpf_hz),
            "sub_overlap_hz": float(self.sub_overlap_hz),
            "sub_delay_ms": float(self.sub_delay_ms),
            "sub_gain_db": float(self.sub_gain_db),
            "sub_polarity_invert": bool(self.sub_polarity_invert),
            "phase_error_deg": float(self.phase_error_deg),
            "gd_mismatch_ms": float(self.gd_mismatch_ms),
            "cancellation_risk": str(self.cancellation_risk),
            "cancellation_score": float(self.cancellation_score),
            "magnitude_ripple_db": float(self.magnitude_ripple_db),
            "score": float(self.score),
            "rejected": bool(self.rejected),
            "reject_reason": str(self.reject_reason),
        }


def _as_complex(values: Any) -> np.ndarray:
    return np.asarray(values if values is not None else [], dtype=np.complex128).reshape(-1)


def _as_freqs(values: Any) -> np.ndarray:
    return np.asarray(values if values is not None else [], dtype=float).reshape(-1)


def _finite_band(freqs: np.ndarray, lo_hz: float, hi_hz: float) -> np.ndarray:
    return (
        np.isfinite(freqs)
        & (freqs >= float(lo_hz))
        & (freqs <= float(hi_hz))
    )


def _phase_error_deg(main: np.ndarray, sub: np.ndarray) -> float:
    delta = np.angle(main * np.conj(sub))
    if delta.size == 0:
        return float("inf")
    return float(np.rad2deg(np.sqrt(np.mean(np.square(delta)))))


def _group_delay_ms(freqs: np.ndarray, response: np.ndarray) -> np.ndarray:
    if freqs.size < 3 or response.size != freqs.size:
        return np.full(freqs.shape, np.nan, dtype=float)
    phase = np.unwrap(np.angle(response))
    omega = 2.0 * np.pi * freqs
    try:
        gd_s = -np.gradient(phase, omega)
    except Exception:
        return np.full(freqs.shape, np.nan, dtype=float)
    return np.asarray(gd_s * 1000.0, dtype=float)


def _gd_mismatch_ms(freqs: np.ndarray, main: np.ndarray, sub: np.ndarray) -> float:
    gd_main = _group_delay_ms(freqs, main)
    gd_sub = _group_delay_ms(freqs, sub)
    diff = gd_main - gd_sub
    finite = np.isfinite(diff)
    if not bool(np.any(finite)):
        return float("inf")
    return float(np.sqrt(np.mean(np.square(diff[finite]))))


def _mag_db(response: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(response), 1e-12))


def _risk_label(cancellation_score: float) -> str:
    if not np.isfinite(cancellation_score):
        return "high"
    if cancellation_score >= 9.0:
        return "high"
    if cancellation_score >= 4.5:
        return "medium"
    return "low"


def _apply_candidate(
    *,
    freqs: np.ndarray,
    main_corrected_complex: np.ndarray,
    sub_corrected_complex: np.ndarray,
    xo_hz: float,
    sub_delay_ms: float,
    sub_gain_db: float,
    sub_polarity_invert: bool,
) -> tuple[np.ndarray, np.ndarray]:
    main = np.asarray(main_corrected_complex, dtype=np.complex128).copy()
    sub = np.asarray(sub_corrected_complex, dtype=np.complex128).copy()
    main *= _butterworth_complex_response(freqs, float(xo_hz), _XO_ORDER, btype="high")
    sub *= _butterworth_complex_response(freqs, _SUB_HPF_HZ, _SUB_HPF_ORDER, btype="high")
    sub *= _butterworth_complex_response(freqs, float(xo_hz) + _OVERLAP_HZ, _XO_ORDER, btype="low")
    if bool(sub_polarity_invert):
        sub *= -1.0
    sub *= float(np.power(10.0, float(sub_gain_db) / 20.0))
    sub *= np.exp(-1j * 2.0 * np.pi * freqs * (float(sub_delay_ms) / 1000.0))
    return main, sub


def score_direct_dac_bass_integration(
    main_corrected_complex: Any,
    sub_corrected_complex: Any,
    freqs: Any,
    xo_hz: float,
    sub_delay_ms: float,
    sub_gain_db: float,
    sub_polarity_invert: bool = False,
) -> dict[str, Any]:
    """Score raw array branches for backward-compatible tests and external callers.

    Production Direct-DAC orchestration uses the canonical evaluator path:
    evaluate_direct_dac_candidate() through compute_final_bass_integration_metrics(..., mode="direct_dac").
    """
    freqs_arr = _as_freqs(freqs)
    main_arr = _as_complex(main_corrected_complex)
    sub_arr = _as_complex(sub_corrected_complex)
    if freqs_arr.size == 0 or main_arr.size != freqs_arr.size or sub_arr.size != freqs_arr.size:
        return {
            "score": float("inf"),
            "rejected": True,
            "reject_reason": "invalid_input_shape",
            "phase_error_deg": float("inf"),
            "gd_mismatch_ms": float("inf"),
            "cancellation_score": float("inf"),
            "magnitude_ripple_db": float("inf"),
            "cancellation_risk": "high",
        }
    xo = float(xo_hz)
    sub_lpf = xo + _OVERLAP_HZ
    if (not np.isfinite(xo)) or xo <= 0.0 or sub_lpf < xo + _OVERLAP_HZ:
        return {
            "score": float("inf"),
            "rejected": True,
            "reject_reason": "invalid_crossover",
            "phase_error_deg": float("inf"),
            "gd_mismatch_ms": float("inf"),
            "cancellation_score": float("inf"),
            "magnitude_ripple_db": float("inf"),
            "cancellation_risk": "high",
        }

    main, sub = _apply_candidate(
        freqs=freqs_arr,
        main_corrected_complex=main_arr,
        sub_corrected_complex=sub_arr,
        xo_hz=xo,
        sub_delay_ms=float(sub_delay_ms),
        sub_gain_db=float(sub_gain_db),
        sub_polarity_invert=bool(sub_polarity_invert),
    )
    band = _finite_band(freqs_arr, max(10.0, xo * 0.6), min(float(np.nanmax(freqs_arr)), xo * 1.6))
    band &= np.isfinite(main.real) & np.isfinite(main.imag) & np.isfinite(sub.real) & np.isfinite(sub.imag)
    if int(np.count_nonzero(band)) < 4:
        return {
            "score": float("inf"),
            "rejected": True,
            "reject_reason": "insufficient_xo_band",
            "phase_error_deg": float("inf"),
            "gd_mismatch_ms": float("inf"),
            "cancellation_score": float("inf"),
            "magnitude_ripple_db": float("inf"),
            "cancellation_risk": "high",
        }

    main_b = main[band]
    sub_b = sub[band]
    freqs_b = freqs_arr[band]
    summed = main_b + sub_b
    sum_mag = _mag_db(summed)
    branch_max_mag = np.maximum(_mag_db(main_b), _mag_db(sub_b))
    cancellation = np.maximum(0.0, branch_max_mag - sum_mag)
    # mean+1.5σ is more robust than 95th percentile: a single narrow notch no longer
    # dominates the score when the rest of the band integrates cleanly.
    cancellation_score = float(np.mean(cancellation) + 1.5 * np.std(cancellation))
    ripple = float(np.percentile(sum_mag, 95.0) - np.percentile(sum_mag, 5.0))
    phase_err = _phase_error_deg(main_b, sub_b)
    gd_err = _gd_mismatch_ms(freqs_b, main_b, sub_b)

    phase_error_score = float(max(0.0, phase_err) / 90.0)
    gd_error_score = float(max(0.0, gd_err) / 8.0)
    cancellation_norm = float(max(0.0, cancellation_score) / 12.0)
    ripple_score = float(max(0.0, ripple) / 12.0)
    gain_penalty = float(abs(float(sub_gain_db)) / 12.0)
    delay_penalty = float(abs(float(sub_delay_ms)) / 15.0)
    xo_penalty = float(abs(float(xo) - 80.0) / 40.0)
    polarity_tie_penalty = 1.0 if bool(sub_polarity_invert) else 0.0
    score = (
        4.0 * phase_error_score
        + 3.0 * gd_error_score
        + 2.0 * cancellation_norm
        + 1.0 * ripple_score
        + 0.75 * xo_penalty
        + 0.25 * gain_penalty
        + 0.15 * delay_penalty
        + 0.05 * polarity_tie_penalty
    )

    rejected = False
    reject_reason = ""
    if cancellation_score >= 18.0:
        rejected = True
        reject_reason = "deep_cancellation_near_xo"
    elif gd_err >= 30.0:
        rejected = True
        reject_reason = "wild_gd_mismatch_near_xo"
    elif not np.isfinite(score):
        rejected = True
        reject_reason = "non_finite_score"

    if rejected:
        score += 1_000.0

    return {
        "score": float(score),
        "rejected": bool(rejected),
        "reject_reason": str(reject_reason),
        "phase_error_deg": float(phase_err),
        "gd_mismatch_ms": float(gd_err),
        "cancellation_score": float(cancellation_score),
        "magnitude_ripple_db": float(ripple),
        "cancellation_risk": _risk_label(cancellation_score),
        "phase_error_score": float(phase_error_score),
        "gd_error_score": float(gd_error_score),
        "magnitude_ripple_score": float(ripple_score),
        "gain_penalty": float(gain_penalty),
        "delay_penalty": float(delay_penalty),
        "xo_penalty": float(xo_penalty),
        "polarity_tie_penalty": float(polarity_tie_penalty),
    }


def _candidate_values(center: float, lo: float, hi: float, offsets: tuple[float, ...], step: float) -> list[float]:
    vals: set[float] = set()
    for off in offsets:
        vals.add(round(float(np.clip(float(center) + float(off), lo, hi)) / step) * step)
    return sorted(float(np.clip(v, lo, hi)) for v in vals)


def _select_best(
    rows: list[tuple[dict[str, Any], float, float, float, bool]]
) -> tuple[dict[str, Any], float, float, float, bool]:
    return min(
        rows,
        key=lambda row: (
            bool(row[0].get("rejected", False)),
            float(row[0].get("score", float("inf"))),
            float(row[0].get("cancellation_score", float("inf"))),
            float(row[0].get("phase_error_deg", float("inf"))),
            float(row[0].get("gd_mismatch_ms", float("inf"))),
            abs(float(row[3])),
            abs(float(row[2])),
            bool(row[4]),
        ),
    )


def _score_grid_for_xo(
    *,
    main_arr: np.ndarray,
    sub_arr: np.ndarray,
    freqs_arr: np.ndarray,
    xo_hz: float,
    delays_ms: np.ndarray,
    gains_db: np.ndarray,
    polarities: tuple[bool, ...],
) -> tuple[dict[str, Any], float, float, float, bool] | None:
    xo = float(xo_hz)
    if (not np.isfinite(xo)) or xo <= 0.0:
        return None
    main = main_arr * _butterworth_complex_response(freqs_arr, xo, _XO_ORDER, btype="high")
    sub = sub_arr * _butterworth_complex_response(freqs_arr, _SUB_HPF_HZ, _SUB_HPF_ORDER, btype="high")
    sub *= _butterworth_complex_response(freqs_arr, xo + _OVERLAP_HZ, _XO_ORDER, btype="low")

    band = _finite_band(freqs_arr, max(10.0, xo * 0.6), min(float(np.nanmax(freqs_arr)), xo * 1.6))
    band &= np.isfinite(main.real) & np.isfinite(main.imag) & np.isfinite(sub.real) & np.isfinite(sub.imag)
    if int(np.count_nonzero(band)) < 4:
        return None

    freqs_b = freqs_arr[band]
    main_b = main[band]
    sub_b = sub[band]
    delays = np.asarray(delays_ms, dtype=float).reshape(-1)
    gains = np.asarray(gains_db, dtype=float).reshape(-1)
    if delays.size == 0 or gains.size == 0:
        return None

    gd_main = _group_delay_ms(freqs_b, main_b)
    gd_sub = _group_delay_ms(freqs_b, sub_b)
    gd_base_diff = gd_main - gd_sub
    gd_finite = np.isfinite(gd_base_diff)
    if not bool(np.any(gd_finite)):
        return None
    gd_err = np.sqrt(np.mean(np.square(gd_base_diff[gd_finite][None, :] - delays[:, None]), axis=1))

    delay_phase = np.exp(-1j * 2.0 * np.pi * freqs_b[None, :] * (delays[:, None] / 1000.0))
    delayed_sub = sub_b[None, :] * delay_phase
    gain_linear = np.power(10.0, gains / 20.0)
    rows: list[tuple[dict[str, Any], float, float, float, bool]] = []

    for polarity in tuple(bool(p) for p in polarities):
        pol = -1.0 if polarity else 1.0
        sub_pd = delayed_sub * pol
        phase_delta = np.angle(main_b[None, :] * np.conj(sub_pd))
        phase_err = np.rad2deg(np.sqrt(np.mean(np.square(phase_delta), axis=1)))
        sub_grid = sub_pd[:, None, :] * gain_linear[None, :, None]
        summed = main_b[None, None, :] + sub_grid
        sum_mag = _mag_db(summed)
        branch_max_mag = np.maximum(_mag_db(main_b)[None, None, :], _mag_db(sub_grid))
        cancellation = np.maximum(0.0, branch_max_mag - sum_mag)
        cancellation_score = np.percentile(cancellation, 95.0, axis=2)
        ripple = np.percentile(sum_mag, 95.0, axis=2) - np.percentile(sum_mag, 5.0, axis=2)

        phase_error_score = np.maximum(0.0, phase_err)[:, None] / 90.0
        gd_error_score = np.maximum(0.0, gd_err)[:, None] / 8.0
        cancellation_norm = np.maximum(0.0, cancellation_score) / 12.0
        ripple_score = np.maximum(0.0, ripple) / 12.0
        gain_penalty = np.abs(gains)[None, :] / 12.0
        delay_penalty = np.abs(delays)[:, None] / 15.0
        xo_penalty = abs(xo - 80.0) / 40.0
        polarity_tie_penalty = 1.0 if polarity else 0.0
        score = (
            4.0 * phase_error_score
            + 3.0 * gd_error_score
            + 2.0 * cancellation_norm
            + 1.0 * ripple_score
            + 0.75 * xo_penalty
            + 0.25 * gain_penalty
            + 0.15 * delay_penalty
            + 0.05 * polarity_tie_penalty
        )
        rejected = (cancellation_score >= 18.0) | (gd_err[:, None] >= 30.0) | (~np.isfinite(score))
        score = np.where(rejected, score + 1_000.0, score)
        finite_score = np.isfinite(score)
        if not bool(np.any(finite_score)):
            continue

        eligible_score = np.where(finite_score, score, np.inf)
        best_flat = int(np.argmin(eligible_score))
        delay_idx, gain_idx = np.unravel_index(best_flat, eligible_score.shape)
        rejected_best = bool(rejected[delay_idx, gain_idx])
        reject_reason = ""
        if rejected_best:
            if float(cancellation_score[delay_idx, gain_idx]) >= 18.0:
                reject_reason = "deep_cancellation_near_xo"
            elif float(gd_err[delay_idx]) >= 30.0:
                reject_reason = "wild_gd_mismatch_near_xo"
            else:
                reject_reason = "non_finite_score"
        row = {
            "score": float(score[delay_idx, gain_idx]),
            "rejected": rejected_best,
            "reject_reason": reject_reason,
            "phase_error_deg": float(phase_err[delay_idx]),
            "gd_mismatch_ms": float(gd_err[delay_idx]),
            "cancellation_score": float(cancellation_score[delay_idx, gain_idx]),
            "magnitude_ripple_db": float(ripple[delay_idx, gain_idx]),
            "cancellation_risk": _risk_label(float(cancellation_score[delay_idx, gain_idx])),
            "phase_error_score": float(phase_error_score[delay_idx, 0]),
            "gd_error_score": float(gd_error_score[delay_idx, 0]),
            "magnitude_ripple_score": float(ripple_score[delay_idx, gain_idx]),
            "gain_penalty": float(gain_penalty[0, gain_idx]),
            "delay_penalty": float(delay_penalty[delay_idx, 0]),
            "xo_penalty": float(xo_penalty),
            "polarity_tie_penalty": float(polarity_tie_penalty),
        }
        rows.append((row, xo, float(delays[delay_idx]), float(gains[gain_idx]), polarity))

    if not rows:
        return None
    return _select_best(rows)


def run_direct_dac_bass_integration(
    main_corrected_complex: Any,
    sub_corrected_complex: Any,
    freqs: Any,
) -> DirectDacBassIntegrationResult:
    """Run the legacy array-level grid optimizer for compatibility callers."""
    freqs_arr = _as_freqs(freqs)
    main_arr = _as_complex(main_corrected_complex)
    sub_arr = _as_complex(sub_corrected_complex)

    def _eval_grid(
        xos: list[float],
        delays: list[float],
        gains: list[float],
        polarities: tuple[bool, ...],
    ) -> list[tuple[dict[str, Any], float, float, float, bool]]:
        if freqs_arr.size == 0 or main_arr.size != freqs_arr.size or sub_arr.size != freqs_arr.size:
            return []
        rows: list[tuple[dict[str, Any], float, float, float, bool]] = []
        delays_arr = np.asarray(delays, dtype=float)
        gains_arr = np.asarray(gains, dtype=float)
        for xo in xos:
            row = _score_grid_for_xo(
                main_arr=main_arr,
                sub_arr=sub_arr,
                freqs_arr=freqs_arr,
                xo_hz=float(xo),
                delays_ms=delays_arr,
                gains_db=gains_arr,
                polarities=polarities,
            )
            if row is not None:
                rows.append(row)
        return rows

    coarse_rows = _eval_grid(
        [float(v) for v in np.arange(50.0, 120.0 + 0.1, 5.0)],
        [float(v) for v in np.arange(-15.0, 15.0 + 0.1, 1.0)],
        [float(v) for v in np.arange(-12.0, 12.0 + 0.1, 1.0)],
        (False, True),
    )
    if not coarse_rows:
        best_score = score_direct_dac_bass_integration(main_corrected_complex, sub_corrected_complex, freqs, 80.0, 0.0, 0.0, False)
        best_xo, best_delay, best_gain, best_polarity = 80.0, 0.0, 0.0, False
    else:
        best_score, best_xo, best_delay, best_gain, best_polarity = _select_best(coarse_rows)

    refine_rows = _eval_grid(
        _candidate_values(best_xo, 50.0, 120.0, tuple(float(v) for v in np.arange(-5.0, 5.0 + 0.1, 1.0)), 1.0),
        _candidate_values(best_delay, -15.0, 15.0, tuple(float(v) for v in np.arange(-1.0, 1.0 + 0.01, 0.1)), 0.1),
        _candidate_values(best_gain, -12.0, 12.0, tuple(float(v) for v in np.arange(-1.0, 1.0 + 0.01, 0.25)), 0.25),
        (False, True),
    )
    if refine_rows:
        best_score, best_xo, best_delay, best_gain, best_polarity = _select_best(refine_rows)

    return DirectDacBassIntegrationResult(
        enabled=True,
        main_hpf_hz=float(best_xo),
        sub_hpf_hz=float(_SUB_HPF_HZ),
        sub_lpf_hz=float(best_xo + _OVERLAP_HZ),
        sub_overlap_hz=float(_OVERLAP_HZ),
        sub_delay_ms=float(best_delay),
        sub_gain_db=float(best_gain),
        sub_polarity_invert=bool(best_polarity),
        phase_error_deg=float(best_score.get("phase_error_deg", float("nan"))),
        gd_mismatch_ms=float(best_score.get("gd_mismatch_ms", float("nan"))),
        cancellation_risk=str(best_score.get("cancellation_risk", "high") or "high"),
        cancellation_score=float(best_score.get("cancellation_score", float("nan"))),
        magnitude_ripple_db=float(best_score.get("magnitude_ripple_db", float("nan"))),
        score=float(best_score.get("score", float("inf"))),
        rejected=bool(best_score.get("rejected", False)),
        reject_reason=str(best_score.get("reject_reason", "") or ""),
    )


__all__ = [
    "DirectDacBassIntegrationResult",
    "run_direct_dac_bass_integration",
    "score_direct_dac_bass_integration",
]
