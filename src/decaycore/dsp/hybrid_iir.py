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

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .modal_analysis import RoomModeEvent

HYBRID_IIR_POLICY_VERSION = 1


@dataclass(frozen=True)
class HybridIIRPolicy:
    enabled: bool = False
    max_filters_per_channel: int = 3
    min_freq_hz: float = 20.0
    max_freq_hz: float = 150.0
    min_peak_db: float = 4.0
    min_q: float = 3.0
    max_q: float = 12.0
    max_cut_db: float = 6.0
    min_confidence: float = 0.65
    min_gd_excess_ms: float = 15.0
    min_cut_priority: float = 0.0
    max_voice_clarity_risk: float = 0.45

    @classmethod
    def from_config(cls, cfg: Any) -> "HybridIIRPolicy":
        return cls(
            enabled=bool(getattr(cfg, "hybrid_iir_enabled", False)),
            max_filters_per_channel=_safe_int(getattr(cfg, "hybrid_iir_max_filters_per_channel", 3), 3),
            min_freq_hz=_safe_float(getattr(cfg, "hybrid_iir_min_freq_hz", 20.0), 20.0),
            max_freq_hz=_safe_float(getattr(cfg, "hybrid_iir_max_freq_hz", 150.0), 150.0),
            min_peak_db=_safe_float(getattr(cfg, "hybrid_iir_min_peak_db", 4.0), 4.0),
            min_q=_safe_float(getattr(cfg, "hybrid_iir_min_q", 3.0), 3.0),
            max_q=_safe_float(getattr(cfg, "hybrid_iir_max_q", 12.0), 12.0),
            max_cut_db=_safe_float(getattr(cfg, "hybrid_iir_max_cut_db", 6.0), 6.0),
            min_confidence=_safe_float(getattr(cfg, "hybrid_iir_min_confidence", 0.65), 0.65),
            min_gd_excess_ms=_safe_float(getattr(cfg, "hybrid_iir_min_gd_excess_ms", 15.0), 15.0),
            min_cut_priority=_safe_float(getattr(cfg, "hybrid_iir_min_cut_priority", 0.0), 0.0),
        ).normalized()

    def normalized(self) -> "HybridIIRPolicy":
        min_f = max(1.0, float(self.min_freq_hz))
        max_f = max(min_f + 1.0, float(self.max_freq_hz))
        min_q = max(0.2, float(self.min_q))
        max_q = max(min_q, float(self.max_q))
        return HybridIIRPolicy(
            enabled=bool(self.enabled),
            max_filters_per_channel=max(0, int(self.max_filters_per_channel)),
            min_freq_hz=min_f,
            max_freq_hz=max_f,
            min_peak_db=max(0.0, float(self.min_peak_db)),
            min_q=min_q,
            max_q=max_q,
            max_cut_db=max(0.0, float(self.max_cut_db)),
            min_confidence=float(np.clip(self.min_confidence, 0.0, 1.0)),
            min_gd_excess_ms=max(0.0, float(self.min_gd_excess_ms)),
            min_cut_priority=float(np.clip(self.min_cut_priority, 0.0, 1.0)),
            max_voice_clarity_risk=float(np.clip(self.max_voice_clarity_risk, 0.0, 1.0)),
        )

    def to_signature_dict(self) -> dict[str, Any]:
        p = self.normalized()
        return {
            "policy_v": int(HYBRID_IIR_POLICY_VERSION),
            "enabled": bool(p.enabled),
            "max_filters_per_channel": int(p.max_filters_per_channel),
            "min_freq_hz": float(p.min_freq_hz),
            "max_freq_hz": float(p.max_freq_hz),
            "min_peak_db": float(p.min_peak_db),
            "min_q": float(p.min_q),
            "max_q": float(p.max_q),
            "max_cut_db": float(p.max_cut_db),
            "min_confidence": float(p.min_confidence),
            "min_gd_excess_ms": float(p.min_gd_excess_ms),
            "min_cut_priority": float(p.min_cut_priority),
            "max_voice_clarity_risk": float(p.max_voice_clarity_risk),
        }


@dataclass(frozen=True)
class HybridBiquad:
    freq_hz: float
    q: float
    gain_db: float
    source_peak_db: float
    safe_cut_db: float
    confidence: float
    gd_excess_ms: float
    cut_priority: float

    def to_camilladsp(self) -> dict[str, float | str]:
        return {
            "type": "Peaking",
            "freq": float(self.freq_hz),
            "q": float(self.q),
            "gain": float(self.gain_db),
            "confidence": float(self.confidence),
            "safe_cut_db": float(self.safe_cut_db),
        }


@dataclass(frozen=True)
class HybridIIRResult:
    enabled: bool
    biquads: tuple[HybridBiquad, ...] = ()
    rejected: tuple[dict[str, Any], ...] = ()
    response: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=complex))
    mag_db: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    phase_rad: np.ndarray = field(default_factory=lambda: np.asarray([], dtype=float))
    policy_version: int = HYBRID_IIR_POLICY_VERSION
    mode: str = "magnitude_preconditioning_only"

    def to_stats(self) -> dict[str, Any]:
        return {
            "hybrid_iir_enabled": bool(self.enabled),
            "hybrid_iir_policy_version": int(self.policy_version),
            "hybrid_iir_mode": str(self.mode),
            "hybrid_iir_biquads": [b.to_camilladsp() for b in self.biquads],
            "hybrid_iir_rejected": [dict(item) for item in self.rejected],
            "hybrid_iir_filter_count": int(len(self.biquads)),
            "hybrid_iir_mag_db": np.asarray(self.mag_db, dtype=float).tolist(),
            "hybrid_iir_phase_rad": np.asarray(self.phase_rad, dtype=float).tolist(),
        }


def design_hybrid_iir(
    events: Sequence[RoomModeEvent],
    freq_axis: np.ndarray,
    fs: int | float,
    policy: HybridIIRPolicy,
) -> HybridIIRResult:
    policy = policy.normalized()
    freq = np.asarray(freq_axis, dtype=float).reshape(-1)
    if not policy.enabled or policy.max_filters_per_channel <= 0 or freq.size == 0:
        return HybridIIRResult(enabled=bool(policy.enabled))

    selected: list[HybridBiquad] = []
    rejected: list[dict[str, Any]] = []
    for event in sorted(tuple(events or ()), key=lambda ev: (-_event_float(ev, "cut_priority"), -_event_float(ev, "peak_db"))):
        biquad, reason = _candidate_to_biquad(event, policy)
        if biquad is None:
            rejected.append(_rejection(event, reason))
            continue
        selected.append(biquad)
        if len(selected) >= int(policy.max_filters_per_channel):
            break

    response = np.ones(freq.size, dtype=complex)
    for biquad in selected:
        response *= peaking_eq_response(freq, float(fs), biquad.freq_hz, biquad.q, biquad.gain_db)
    mag_db = 20.0 * np.log10(np.maximum(np.abs(response), 1e-12))
    phase_rad = np.unwrap(np.angle(response)) if response.size else np.asarray([], dtype=float)
    return HybridIIRResult(
        enabled=True,
        biquads=tuple(selected),
        rejected=tuple(rejected),
        response=response,
        mag_db=np.nan_to_num(mag_db, nan=0.0, posinf=0.0, neginf=0.0),
        phase_rad=np.nan_to_num(phase_rad, nan=0.0, posinf=0.0, neginf=0.0),
    )


def peaking_eq_response(freq_axis: np.ndarray, fs: int | float, freq_hz: float, q: float, gain_db: float) -> np.ndarray:
    b0, b1, b2, a0, a1, a2 = peaking_eq_coefficients(float(fs), float(freq_hz), float(q), float(gain_db))
    freq = np.asarray(freq_axis, dtype=float).reshape(-1)
    omega = 2.0 * np.pi * np.clip(freq, 0.0, max(1.0, float(fs) / 2.0)) / max(float(fs), 1.0)
    z1 = np.exp(-1j * omega)
    z2 = np.exp(-2j * omega)
    den = a0 + a1 * z1 + a2 * z2
    num = b0 + b1 * z1 + b2 * z2
    with np.errstate(divide="ignore", invalid="ignore"):
        h = num / np.where(np.abs(den) > 1e-18, den, 1.0)
    return np.nan_to_num(h, nan=1.0, posinf=1.0, neginf=1.0)


def peaking_eq_coefficients(fs: float, freq_hz: float, q: float, gain_db: float) -> tuple[float, float, float, float, float, float]:
    fs = max(1.0, float(fs))
    f0 = float(np.clip(freq_hz, 1e-6, fs * 0.499))
    q = max(1e-6, float(q))
    gain_db = min(0.0, float(gain_db))
    w0 = 2.0 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2.0 * q)
    a_amp = 10.0 ** (gain_db / 40.0)
    b0 = 1.0 + alpha * a_amp
    b1 = -2.0 * np.cos(w0)
    b2 = 1.0 - alpha * a_amp
    a0 = 1.0 + alpha / max(a_amp, 1e-12)
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha / max(a_amp, 1e-12)
    return (float(b0 / a0), float(b1 / a0), float(b2 / a0), 1.0, float(a1 / a0), float(a2 / a0))


def _candidate_to_biquad(event: RoomModeEvent, policy: HybridIIRPolicy) -> tuple[HybridBiquad | None, str]:
    freq = _event_float(event, "freq_hz")
    peak = _event_float(event, "peak_db")
    confidence = _event_float(event, "confidence")
    gd_excess = _event_float(event, "gd_excess_ms")
    cut_priority = _event_float(event, "cut_priority", _event_float(event, "correction_priority"))
    voice_risk = _event_float(event, "voice_clarity_risk")
    kind = str(getattr(event, "kind", "") or "").strip().lower()
    if not (policy.min_freq_hz <= freq <= policy.max_freq_hz):
        return None, "outside_frequency_range"
    if peak < policy.min_peak_db:
        return None, "peak_below_threshold"
    if confidence < policy.min_confidence:
        return None, "confidence_below_threshold"
    if gd_excess < policy.min_gd_excess_ms:
        return None, "gd_evidence_below_threshold"
    if voice_risk > policy.max_voice_clarity_risk:
        return None, "voice_clarity_risk"
    if kind != "room_mode" and cut_priority < policy.min_cut_priority:
        return None, "modal_priority_below_threshold"

    q_raw = _event_float(event, "q_estimate")
    if not np.isfinite(q_raw) or q_raw <= 0.0:
        width_hz = _event_float(event, "width_hz")
        q_raw = freq / max(width_hz, 1e-9)
    if q_raw < policy.min_q:
        return None, "q_below_threshold"
    q = float(np.clip(q_raw, policy.min_q, policy.max_q))

    safe_cut = min(_event_float(event, "safe_cut_db"), policy.max_cut_db, peak)
    if safe_cut <= 0.0:
        return None, "safe_cut_unavailable"
    gain_db = -float(safe_cut)
    return (
        HybridBiquad(
            freq_hz=float(freq),
            q=float(q),
            gain_db=float(gain_db),
            source_peak_db=float(peak),
            safe_cut_db=float(safe_cut),
            confidence=float(confidence),
            gd_excess_ms=float(gd_excess),
            cut_priority=float(cut_priority),
        ),
        "",
    )


def _rejection(event: RoomModeEvent, reason: str) -> dict[str, Any]:
    return {
        "freq_hz": _event_float(event, "freq_hz"),
        "peak_db": _event_float(event, "peak_db"),
        "q_estimate": _event_float(event, "q_estimate"),
        "confidence": _event_float(event, "confidence"),
        "gd_excess_ms": _event_float(event, "gd_excess_ms"),
        "reason": str(reason),
    }


def _event_float(event: Any, key: str, default: float = 0.0) -> float:
    try:
        value = getattr(event, key)
    except Exception:
        value = default
    return _safe_float(value, default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        out = float(default)
    return out if np.isfinite(out) else float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return int(default)
