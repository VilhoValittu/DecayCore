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
from typing import Any, Sequence

import numpy as np

from .modal_analysis import RoomModeEvent, detect_room_modes


@dataclass(frozen=True)
class AcousticAuthority:
    freq_axis: np.ndarray

    cut_authority: np.ndarray
    boost_authority: np.ndarray
    phase_authority: np.ndarray

    modal_support: np.ndarray
    null_risk: np.ndarray
    reflection_risk: np.ndarray
    repeatability: np.ndarray
    minphase_likelihood: np.ndarray
    voice_risk: np.ndarray

    version: int = 1


def _as_float_array(value) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=float)
    try:
        return np.asarray(value, dtype=float).reshape(-1)
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
    ):
        return np.asarray([], dtype=float)


def _clip01(value) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(arr, 0.0, 1.0)


def _align_to_freq_axis(value, freq_axis, default=0.0) -> np.ndarray:
    f = _as_float_array(freq_axis)
    n = int(f.size)
    if n <= 0:
        return np.asarray([], dtype=float)

    default_arr = _as_float_array(default)
    if default_arr.size == n:
        fallback = np.asarray(default_arr, dtype=float).copy()
    elif default_arr.size == 1:
        fallback = np.full(n, float(default_arr[0]), dtype=float)
    else:
        fallback = np.zeros(n, dtype=float)

    arr = _as_float_array(value)
    if arr.size == n:
        return np.nan_to_num(np.asarray(arr, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == 1:
        return np.full(n, float(arr[0]), dtype=float)
    return np.nan_to_num(fallback, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_log2_freq(freq_axis) -> np.ndarray:
    f = _as_float_array(freq_axis)
    return np.log2(np.maximum(np.nan_to_num(f, nan=1e-9, posinf=1e-9, neginf=1e-9), 1e-9))


def _smooth_log_box(freq_axis, values, width_oct) -> np.ndarray:
    f = _as_float_array(freq_axis)
    v = _as_float_array(values)
    if f.size < 3 or v.size != f.size:
        return np.asarray(v, dtype=float).copy()
    width = float(np.clip(float(width_oct), 1.0 / 192.0, 2.0))
    half = 0.5 * width
    log_f = _safe_log2_freq(f)
    order = np.argsort(log_f)
    x = np.asarray(log_f[order], dtype=float)
    y = np.nan_to_num(np.asarray(v[order], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    finite = np.isfinite(v[order]).astype(float)
    sum_y = np.r_[0.0, np.cumsum(y * finite)]
    sum_w = np.r_[0.0, np.cumsum(finite)]
    lefts = np.searchsorted(x, x - half, side="left")
    rights = np.searchsorted(x, x + half, side="right")
    counts = sum_w[rights] - sum_w[lefts]
    sums = sum_y[rights] - sum_y[lefts]
    out = np.empty_like(v, dtype=float)
    out[order] = np.where(counts > 0.0, sums / np.maximum(counts, 1e-12), y)
    return out


def _event_value(event: RoomModeEvent | dict, key: str, default: Any = 0.0) -> Any:
    if isinstance(event, dict):
        return event.get(key, default)
    return getattr(event, key, default)


def _safe_float(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _log_gaussian(freq_axis: np.ndarray, center_hz: float, width_oct: float) -> np.ndarray:
    center = _safe_float(center_hz, 0.0)
    if center <= 0.0:
        return np.zeros_like(freq_axis, dtype=float)
    width = float(max(1.0 / 192.0, _safe_float(width_oct, 1.0 / 12.0)))
    distance = (_safe_log2_freq(freq_axis) - float(np.log2(max(center, 1e-9)))) / max(width, 1e-9)
    return np.exp(-0.5 * distance * distance)


def _voice_weight(freq_hz: np.ndarray) -> np.ndarray:
    f = np.asarray(freq_hz, dtype=float)
    out = np.zeros_like(f, dtype=float)
    out = np.where((f >= 55.0) & (f < 80.0), (f - 55.0) / 25.0, out)
    out = np.where((f >= 80.0) & (f <= 160.0), 1.0, out)
    out = np.where((f > 160.0) & (f <= 220.0), 1.0 - (f - 160.0) / 60.0, out)
    return _clip01(out)


def _lr_mismatch_curve(freq_axis, left_mag_db, right_mag_db) -> np.ndarray:
    f = _as_float_array(freq_axis)
    left = _as_float_array(left_mag_db)
    right = _as_float_array(right_mag_db)
    if f.size == 0 or left.size != f.size or right.size != f.size:
        return np.zeros_like(f, dtype=float)
    mismatch = _clip01(np.abs(left - right) / 12.0)
    return _clip01(_smooth_log_box(f, mismatch, 1.0 / 12.0))


def _gd_spike_risk_curve(freq_axis, group_delay_ms) -> np.ndarray:
    f = _as_float_array(freq_axis)
    gd = _as_float_array(group_delay_ms)
    if f.size == 0 or gd.size != f.size:
        return np.zeros_like(f, dtype=float)
    smooth = _smooth_log_box(f, gd, 0.25)
    spike = _clip01((np.abs(gd - smooth) - 5.0) / 20.0)
    return _clip01(_smooth_log_box(f, spike, 1.0 / 18.0))


def _modal_support_curve(
    freq_axis: np.ndarray,
    events: Sequence[RoomModeEvent | dict],
) -> np.ndarray:
    support = np.zeros_like(freq_axis, dtype=float)
    for event in tuple(events or ()):
        freq_hz = _safe_float(_event_value(event, "freq_hz", _event_value(event, "freq", 0.0)), 0.0)
        width_oct = _safe_float(
            _event_value(event, "safe_width_oct", 0.0),
            0.0,
        ) or _safe_float(_event_value(event, "width_oct", 0.0), 0.0) or (1.0 / 12.0)
        width_oct = float(max(1.0 / 48.0, width_oct))
        severity = float(np.clip(_safe_float(_event_value(event, "severity", 0.0), 0.0), 0.0, 1.0))
        correction = float(np.clip(_safe_float(_event_value(event, "correction_priority", 0.0), 0.0), 0.0, 1.0))
        decay = float(np.clip(_safe_float(_event_value(event, "decay_severity", 0.0), 0.0), 0.0, 1.0))
        confidence = float(np.clip(_safe_float(_event_value(event, "confidence", 0.0), 0.0), 0.0, 1.0))
        strength = float(np.clip(0.45 * severity + 0.30 * correction + 0.15 * decay + 0.10 * confidence, 0.0, 1.0))
        kind = str(_event_value(event, "kind", "unknown") or "unknown")
        kind_factor = {
            "room_mode": 1.00,
            "broad_buildup": 0.70,
            "uncertain": 0.35,
            "local_comb": 0.10,
        }.get(kind, 0.25)
        support = np.maximum(support, float(kind_factor * strength) * _log_gaussian(freq_axis, freq_hz, width_oct))
    return _clip01(support)


def _reflection_risk_curve(freq_axis, reflection_nodes, confidence, modal_support) -> np.ndarray:
    f = _as_float_array(freq_axis)
    risk = np.zeros_like(f, dtype=float)
    for node in list(reflection_nodes or []):
        if not isinstance(node, dict):
            continue
        freq = _safe_float(node.get("freq"), 0.0)
        if freq <= 0.0:
            continue
        strength = float(np.clip(_safe_float(node.get("gd_error"), 0.0) / 20.0, 0.0, 1.0))
        if str(node.get("type", "") or "").lower() == "reflection":
            factor = 1.0
        else:
            idx = int(np.argmin(np.abs(f - freq))) if f.size else 0
            local_modal = float(modal_support[idx]) if f.size else 0.0
            factor = 0.45 if local_modal > 0.25 else 0.70
        risk = np.maximum(risk, float(strength * factor) * _log_gaussian(f, freq, 1.0 / 9.0))
    risk = np.maximum(risk, (1.0 - confidence) * 0.75)
    risk *= 1.0 - 0.45 * modal_support
    return _clip01(risk)


def _phase_band_mask(freq_axis, phase_limit_hz: float) -> np.ndarray:
    f = _as_float_array(freq_axis)
    limit = max(1e-9, _safe_float(phase_limit_hz, 600.0))
    mask = np.ones_like(f, dtype=float)
    mask = np.where(f <= limit, 1.0, mask)
    mask = np.where(f >= 2.0 * limit, 0.0, mask)
    fade = 1.0 - np.log2(np.maximum(f, 1e-9) / limit)
    mask = np.where((f > limit) & (f < 2.0 * limit), fade, mask)
    return _clip01(mask)


def _band_values(values: np.ndarray, freq_axis: np.ndarray, lo_hz: float, hi_hz: float) -> np.ndarray:
    mask = (freq_axis >= float(lo_hz)) & (freq_axis <= float(hi_hz)) & np.isfinite(values)
    return np.asarray(values[mask], dtype=float)


def _mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else 0.0


def _peak(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.max(finite)) if finite.size else 0.0


def build_acoustic_authority_map(
    freq_axis,
    measured_mag_db,
    *,
    target_mag_db=None,
    corrected_mag_db=None,
    confidence_mask=None,
    group_delay_ms=None,
    reflection_nodes=None,
    modal_events=None,
    left_mag_db=None,
    right_mag_db=None,
    repeatability_curve=None,
    mag_c_min: float = 20.0,
    mag_c_max: float = 300.0,
    phase_limit_hz: float = 600.0,
) -> AcousticAuthority:
    freq = _as_float_array(freq_axis)
    measured = _align_to_freq_axis(measured_mag_db, freq, 0.0)
    target = _align_to_freq_axis(target_mag_db, freq, 0.0)
    corrected = _align_to_freq_axis(corrected_mag_db, freq, 0.0)
    confidence = _clip01(_align_to_freq_axis(confidence_mask, freq, 1.0))
    repeatability = _clip01(_align_to_freq_axis(repeatability_curve, freq, confidence))
    gd_ms = _align_to_freq_axis(group_delay_ms, freq, 0.0)
    lr_mismatch = _lr_mismatch_curve(freq, left_mag_db, right_mag_db)
    gd_spike_risk = _gd_spike_risk_curve(freq, gd_ms)

    events = tuple(modal_events or ())
    if not events:
        modal_result = detect_room_modes(
            freq,
            measured,
            target_mag_db=target,
            corrected_mag_db=corrected,
            group_delay_ms=gd_ms,
            confidence_mask=confidence,
            left_mag_db=left_mag_db,
            right_mag_db=right_mag_db,
            lo_hz=float(mag_c_min),
            hi_hz=float(mag_c_max),
        )
        events = tuple(modal_result.events)
    modal_support = _modal_support_curve(freq, events)

    baseline = _smooth_log_box(freq, measured, 0.45)
    dip_db = baseline - measured
    null_risk = _clip01((dip_db - 6.0) / 12.0)
    null_risk = _clip01(_smooth_log_box(freq, null_risk, 1.0 / 18.0))
    null_risk = np.maximum(null_risk, 0.45 * (1.0 - confidence))
    null_risk = np.maximum(null_risk, 0.35 * lr_mismatch)
    null_risk *= 1.0 - 0.55 * modal_support
    null_risk = _clip01(null_risk)

    reflection_risk = _reflection_risk_curve(freq, reflection_nodes, confidence, modal_support)
    voice_weight = _voice_weight(freq)
    voice_peak = _clip01((measured - baseline - 1.5) / 6.0)
    voice_risk = _clip01(voice_weight * np.maximum(modal_support, voice_peak))

    minphase_likelihood = np.asarray(confidence, dtype=float).copy()
    minphase_likelihood *= 1.0 - 0.75 * null_risk
    minphase_likelihood *= 1.0 - 0.55 * reflection_risk
    if np.any(lr_mismatch > 0.0):
        minphase_likelihood *= 1.0 - 0.35 * lr_mismatch
    if np.any(gd_spike_risk > 0.0):
        minphase_likelihood *= 1.0 - 0.35 * gd_spike_risk * (1.0 - modal_support)
    minphase_likelihood = np.maximum(minphase_likelihood, 0.25 * modal_support)
    minphase_likelihood = _clip01(minphase_likelihood)

    cut_authority = 0.45 * confidence + 0.35 * modal_support + 0.20 * repeatability
    cut_authority *= 1.0 - 0.35 * reflection_risk
    cut_authority = _clip01(cut_authority)

    boost_authority = (
        confidence
        * repeatability
        * minphase_likelihood
        * (1.0 - null_risk)
        * (1.0 - 0.60 * reflection_risk)
        * (1.0 - 0.25 * voice_risk)
    )
    boost_authority = _clip01(boost_authority)

    phase_authority = 0.55 * confidence + 0.30 * modal_support + 0.15 * repeatability
    phase_authority *= 1.0 - 0.50 * reflection_risk
    phase_authority *= _phase_band_mask(freq, phase_limit_hz)
    phase_authority = _clip01(phase_authority)

    return AcousticAuthority(
        freq_axis=np.asarray(freq, dtype=float),
        cut_authority=cut_authority,
        boost_authority=boost_authority,
        phase_authority=phase_authority,
        modal_support=modal_support,
        null_risk=null_risk,
        reflection_risk=reflection_risk,
        repeatability=repeatability,
        minphase_likelihood=minphase_likelihood,
        voice_risk=voice_risk,
    )


def acoustic_authority_to_stats(authority: AcousticAuthority, *, include_arrays: bool = True) -> dict:
    freq = _as_float_array(authority.freq_axis)
    cut = _clip01(authority.cut_authority)
    boost = _clip01(authority.boost_authority)
    phase = _clip01(authority.phase_authority)
    modal = _clip01(authority.modal_support)
    null = _clip01(authority.null_risk)
    reflection = _clip01(authority.reflection_risk)
    repeatability = _clip01(authority.repeatability)
    minphase = _clip01(authority.minphase_likelihood)
    voice = _clip01(authority.voice_risk)

    stats = {
        "acoustic_authority_version": int(authority.version),
        "authority_cut_mean": _mean(cut),
        "authority_boost_mean": _mean(boost),
        "authority_phase_mean": _mean(phase),
        "authority_null_risk_peak": _peak(null),
        "authority_reflection_risk_peak": _peak(reflection),
        "authority_modal_support_peak": _peak(modal),
        "authority_voice_risk_peak": _peak(voice),
        "authority_cut_mean_20_300": _mean(_band_values(cut, freq, 20.0, 300.0)),
        "authority_boost_mean_20_300": _mean(_band_values(boost, freq, 20.0, 300.0)),
        "authority_phase_mean_20_600": _mean(_band_values(phase, freq, 20.0, 600.0)),
        "authority_null_risk_peak_20_300": _peak(_band_values(null, freq, 20.0, 300.0)),
        "authority_voice_risk_peak_70_180": _peak(_band_values(voice, freq, 70.0, 180.0)),
        "authority_modal_support_peak_20_300": _peak(_band_values(modal, freq, 20.0, 300.0)),
    }
    if include_arrays:
        stats.update(
            {
                "authority_cut": cut.tolist(),
                "authority_boost": boost.tolist(),
                "authority_phase": phase.tolist(),
                "authority_modal_support": modal.tolist(),
                "authority_null_risk": null.tolist(),
                "authority_reflection_risk": reflection.tolist(),
                "authority_repeatability": repeatability.tolist(),
                "authority_minphase_likelihood": minphase.tolist(),
                "authority_voice_risk": voice.tolist(),
            }
        )
    return stats


__all__ = [
    "AcousticAuthority",
    "build_acoustic_authority_map",
    "acoustic_authority_to_stats",
]
