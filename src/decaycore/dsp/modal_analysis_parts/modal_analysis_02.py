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
from typing import Sequence

import numpy as np
































__all__ = ["RoomModeEvent", "ModalAnalysisResult", "detect_room_modes", "modal_support_for_band"]

def _modal_candidate_geometry(
    freq: np.ndarray,
    excess: np.ndarray,
    candidate_idxs: np.ndarray,
    min_peak_db: float,
) -> dict[str, np.ndarray]:
    idxs = np.asarray(candidate_idxs, dtype=int).reshape(-1)
    if idxs.size == 0:
        empty_i = np.asarray([], dtype=int)
        empty_f = np.asarray([], dtype=float)
        return {
            "idx": empty_i,
            "left": empty_i,
            "right": empty_i,
            "peak_db": empty_f,
            "low_f": empty_f,
            "high_f": empty_f,
            "width_hz": empty_f,
            "width_oct": empty_f,
            "area_db_oct": empty_f,
            "q_estimate": empty_f,
        }
    bounds = np.asarray([_width_bounds(excess, int(idx), min_peak_db * 0.50) for idx in idxs], dtype=int)
    left = bounds[:, 0]
    right = bounds[:, 1]
    low_f = np.asarray(freq[left], dtype=float)
    high_f = np.asarray(freq[right], dtype=float)
    width_hz = np.maximum(0.0, high_f - low_f)
    width_oct = np.where(
        (low_f > 0.0) & (high_f > low_f),
        np.log2(high_f / low_f),
        0.0,
    )
    peak_db = np.maximum(0.0, np.asarray(excess[idxs], dtype=float))
    log_f = np.log2(np.maximum(np.asarray(freq, dtype=float), 1e-9))
    area_y = np.maximum(0.0, np.asarray(excess, dtype=float))
    if log_f.size >= 2:
        contrib = 0.5 * (area_y[:-1] + area_y[1:]) * np.maximum(0.0, log_f[1:] - log_f[:-1])
        prefix = np.r_[0.0, np.cumsum(np.nan_to_num(contrib, nan=0.0, posinf=0.0, neginf=0.0))]
        area_db_oct = np.maximum(0.0, prefix[right] - prefix[left])
    else:
        area_db_oct = np.zeros(idxs.shape, dtype=float)
    q_estimate = np.clip(np.asarray(freq[idxs], dtype=float) / np.maximum(width_hz, 1e-9), 0.0, 100.0)
    return {
        "idx": idxs,
        "left": left,
        "right": right,
        "peak_db": peak_db,
        "low_f": low_f,
        "high_f": high_f,
        "width_hz": width_hz,
        "width_oct": width_oct,
        "area_db_oct": area_db_oct,
        "q_estimate": q_estimate,
    }

def _voice_weight(freq_hz: float) -> float:
    f = float(freq_hz)
    if 80.0 <= f <= 160.0:
        return 1.0
    if 55.0 <= f < 80.0:
        return float(np.clip((f - 55.0) / 25.0, 0.0, 1.0))
    if 160.0 < f <= 220.0:
        return float(np.clip(1.0 - (f - 160.0) / 60.0, 0.0, 1.0))
    return 0.0

def modal_support_for_band(
    modal_events: Sequence[RoomModeEvent],
    f_min: float,
    f_max: float,
) -> dict:
    """Return conservative modal support for a frequency band."""
    try:
        lo = float(f_min)
        hi = float(f_max)
    except (TypeError, ValueError, OverflowError):
        lo = float("nan")
        hi = float("nan")
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo <= 0.0 or hi <= lo:
        return {
            "support": 0.0,
            "max_severity": 0.0,
            "dominant_freq_hz": None,
            "event_count": 0,
            "confidence": 0.0,
            "safe_cut_db": 0.0,
            "cut_priority": 0.0,
            "used_by": [],
        }

    center = float(np.sqrt(lo * hi))
    band_half_oct = max(0.02, 0.5 * float(np.log2(hi / lo)))
    matched: list[tuple[float, RoomModeEvent]] = []
    for event in tuple(modal_events or ()):
        try:
            freq = float(event.freq_hz)
        except (TypeError, ValueError, OverflowError):
            continue
        if not np.isfinite(freq) or freq <= 0.0:
            continue
        event_half_oct = max(0.03, 0.5 * float(max(0.0, event.safe_width_oct or event.width_oct)))
        distance_oct = abs(float(np.log2(freq / center)))
        overlap = max(0.0, 1.0 - distance_oct / max(1e-9, band_half_oct + event_half_oct))
        if overlap <= 0.0:
            continue
        priority = max(
            float(getattr(event, "cut_priority", 0.0) or 0.0),
            float(getattr(event, "correction_priority", 0.0) or 0.0),
        )
        confidence = float(np.clip(float(getattr(event, "confidence", 0.0) or 0.0), 0.0, 1.0))
        severity = float(np.clip(float(getattr(event, "severity", 0.0) or 0.0), 0.0, 1.0))
        score = float(overlap * max(priority, severity * confidence))
        matched.append((score, event))

    if not matched:
        return {
            "support": 0.0,
            "max_severity": 0.0,
            "dominant_freq_hz": None,
            "event_count": 0,
            "confidence": 0.0,
            "safe_cut_db": 0.0,
            "cut_priority": 0.0,
            "used_by": [],
        }

    matched = sorted(matched, key=lambda item: -float(item[0]))
    dominant = matched[0][1]
    support = float(np.clip(max(score for score, _event in matched), 0.0, 1.0))
    return {
        "support": float(support),
        "max_severity": float(
            np.clip(max(float(getattr(event, "severity", 0.0) or 0.0) for _score, event in matched), 0.0, 1.0)
        ),
        "dominant_freq_hz": float(dominant.freq_hz),
        "event_count": int(len(matched)),
        "confidence": float(np.clip(float(getattr(dominant, "confidence", 0.0) or 0.0), 0.0, 1.0)),
        "safe_cut_db": float(max(0.0, float(getattr(dominant, "safe_cut_db", 0.0) or 0.0))),
        "cut_priority": float(
            np.clip(
                max(
                    float(getattr(dominant, "cut_priority", 0.0) or 0.0),
                    float(getattr(dominant, "correction_priority", 0.0) or 0.0),
                ),
                0.0,
                1.0,
            )
        ),
        "used_by": [],
    }

def _lr_consistency_at(
    freq: np.ndarray,
    left_mag: np.ndarray,
    right_mag: np.ndarray,
    center_hz: float,
    peak_db: float,
    width_oct: float,
) -> float:
    if left_mag.size != freq.size or right_mag.size != freq.size:
        return 0.0
    half = float(np.clip(max(width_oct, 1.0 / 36.0) * 0.65, 1.0 / 72.0, 1.0 / 4.0))
    mask = np.abs(np.log2(np.maximum(freq, 1e-9) / max(float(center_hz), 1e-9))) <= half
    if int(np.count_nonzero(mask)) < 2:
        return 0.0
    left_excess = _smooth_log_box(freq[mask], left_mag[mask], 1.0 / 8.0) - _smooth_log_box(freq[mask], left_mag[mask], 1.0)
    right_excess = _smooth_log_box(freq[mask], right_mag[mask], 1.0 / 8.0) - _smooth_log_box(freq[mask], right_mag[mask], 1.0)
    left_peak = float(np.nanmax(left_excess)) if left_excess.size else 0.0
    right_peak = float(np.nanmax(right_excess)) if right_excess.size else 0.0
    floor = max(0.75, 0.35 * float(peak_db))
    if left_peak >= floor and right_peak >= floor:
        balance = min(left_peak, right_peak) / max(max(left_peak, right_peak), 1e-9)
        return float(np.clip(0.65 + 0.35 * balance, 0.0, 1.0))
    if left_peak >= floor or right_peak >= floor:
        return 0.5
    return 0.0

def _decay_severity_at(rt60_by_band, center_hz: float) -> float:
    if rt60_by_band is None:
        return 0.0
    pairs: list[tuple[float, float]] = []
    try:
        if isinstance(rt60_by_band, dict):
            iterator = rt60_by_band.items()
        else:
            iterator = rt60_by_band
        for item in iterator:
            if isinstance(item, dict):
                freq = item.get("freq_hz", item.get("freq", item.get("hz")))
                value = item.get("rt60_s", item.get("rt60", item.get("decay_s")))
            else:
                freq, value = item
            f = float(freq)
            v = float(value)
            if np.isfinite(f) and np.isfinite(v) and f > 0.0 and 0.05 <= v <= 5.0:
                pairs.append((float(f), float(v)))
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
        return 0.0
    if not pairs:
        return 0.0
    pairs = sorted(pairs, key=lambda item: float(item[0]))
    freq = np.asarray([p[0] for p in pairs], dtype=float)
    rt60 = np.asarray([p[1] for p in pairs], dtype=float)
    try:
        x = float(np.log2(max(float(center_hz), 1e-9)))
        rt = float(np.interp(x, np.log2(freq), rt60))
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
        return 0.0
    target = 0.45 if float(center_hz) <= 120.0 else 0.38
    return float(np.clip((rt - target) / 0.55, 0.0, 1.0))

def _classify_event(
    *,
    peak_db: float,
    width_oct: float,
    confidence: float,
    gd_excess_ms: float,
    area_db_oct: float,
    lr_consistency: float,
    min_width_oct: float,
) -> tuple[str, tuple[str, ...]]:
    reasons: list[str] = []
    if peak_db >= 1.5:
        reasons.append("magnitude_peak")
    if gd_excess_ms >= 12.0:
        reasons.append("gd_excess")
    if area_db_oct >= 0.12:
        reasons.append("modal_area")
    if lr_consistency >= 0.65:
        reasons.append("lr_consistent")

    if width_oct < max(1.0 / 72.0, min_width_oct * 0.75) and gd_excess_ms < 8.0 and lr_consistency < 0.65:
        return "local_comb", tuple(reasons or ["narrow_low_support"])
    if width_oct > 1.0 / 3.0:
        return "broad_buildup", tuple(reasons or ["broad_buildup"])
    support = gd_excess_ms >= 10.0 or area_db_oct >= 0.16 or lr_consistency >= 0.65
    if peak_db >= 2.0 and min_width_oct <= width_oct <= 1.0 / 3.0 and confidence >= 0.35 and support:
        return "room_mode", tuple(reasons)
    return "uncertain", tuple(reasons or ["limited_support"])


def _build_room_mode_event(
    *,
    freq: np.ndarray,
    conf: np.ndarray,
    gd_ms: np.ndarray,
    left_mag: np.ndarray,
    right_mag: np.ndarray,
    rt60_by_band,
    geom: dict,
    pos: int,
    idx: int,
    min_peak_db: float,
    min_width_oct: float,
    max_width: float,
) -> RoomModeEvent | None:
    peak_db = float(geom["peak_db"][pos])
    left_idx = int(geom["left"][pos])
    right_idx = int(geom["right"][pos])
    low_f = float(geom["low_f"][pos])
    high_f = float(geom["high_f"][pos])
    if not (np.isfinite(low_f) and np.isfinite(high_f)) or high_f <= low_f:
        return None
    width_hz = float(geom["width_hz"][pos])
    width_oct = float(geom["width_oct"][pos])
    if width_oct > max_width or width_oct < min_width_oct * 0.35:
        return None

    seg = slice(left_idx, right_idx + 1)
    area_db_oct = float(geom["area_db_oct"][pos])
    q_estimate = float(geom["q_estimate"][pos])
    conf_mean = float(np.clip(np.nanmean(conf[seg]) if conf[seg].size else conf[idx], 0.0, 1.0))

    gd_excess = 0.0
    if gd_ms.size == freq.size:
        gd_vals = np.maximum(0.0, gd_ms[seg])
        gd_vals = gd_vals[np.isfinite(gd_vals)]
        if gd_vals.size:
            gd_excess = float(np.percentile(gd_vals, 90.0))
    gd_excess = float(np.clip(gd_excess, 0.0, 250.0))
    gd_decay_severity = float(np.clip(gd_excess / 60.0, 0.0, 1.0))
    rt60_decay_severity = _decay_severity_at(rt60_by_band, float(freq[idx]))
    decay_severity = float(max(gd_decay_severity, rt60_decay_severity))
    lr_consistency = _lr_consistency_at(freq, left_mag, right_mag, float(freq[idx]), peak_db, width_oct)
    kind, reasons = _classify_event(
        peak_db=peak_db,
        width_oct=width_oct,
        confidence=conf_mean,
        gd_excess_ms=gd_excess,
        area_db_oct=area_db_oct,
        lr_consistency=lr_consistency,
        min_width_oct=min_width_oct,
    )
    if _voice_weight(float(freq[idx])) > 0.0 and "voice_band" not in reasons:
        reasons = tuple([*reasons, "voice_band"])

    normalized_peak = float(np.clip(peak_db / 8.0, 0.0, 1.0))
    normalized_area = float(np.clip(area_db_oct / 2.0, 0.0, 1.0))
    normalized_gd = float(np.clip(gd_excess / 60.0, 0.0, 1.0))
    normalized_decay = float(np.clip(decay_severity, 0.0, 1.0))
    voice = _voice_weight(float(freq[idx]))
    severity = float(
        np.clip(
            0.42 * normalized_peak
            + 0.18 * normalized_area
            + 0.18 * normalized_gd
            + 0.12 * normalized_decay
            + 0.10 * voice,
            0.0,
            1.0,
        )
    )
    kind_factor = {
        "room_mode": 1.00,
        "broad_buildup": 0.70,
        "uncertain": 0.35,
        "local_comb": 0.10,
    }.get(kind, 0.20)
    lr_factor = 1.0 if left_mag.size != freq.size else float(0.65 + 0.35 * lr_consistency)
    correction_priority = float(np.clip(severity * conf_mean * kind_factor * lr_factor, 0.0, 1.0))
    cut_priority = float(np.clip(correction_priority * (0.65 + 0.35 * normalized_gd), 0.0, 1.0))
    safe_cut_db = float(np.clip(peak_db * (0.30 + 0.40 * correction_priority), 0.0, min(6.0, peak_db)))
    safe_width_oct = float(np.clip(width_oct * 1.25, 1.0 / 36.0, 1.0 / 2.0))
    voice_clarity_risk = float(np.clip(severity * conf_mean * voice, 0.0, 1.0))
    return RoomModeEvent(
        freq_hz=float(freq[idx]),
        peak_db=float(peak_db),
        width_hz=float(width_hz),
        width_oct=float(width_oct),
        q_estimate=float(q_estimate),
        area_db_oct=float(max(0.0, area_db_oct)),
        gd_excess_ms=float(gd_excess),
        decay_severity=float(decay_severity),
        confidence=float(conf_mean),
        lr_consistency=float(lr_consistency),
        severity=float(severity),
        correction_priority=float(correction_priority),
        cut_priority=float(cut_priority),
        safe_cut_db=float(safe_cut_db),
        safe_width_oct=float(safe_width_oct),
        voice_clarity_risk=float(voice_clarity_risk),
        kind=str(kind),
        reasons=tuple(str(r) for r in reasons),
    )


def _finalize_room_mode_events(events: list[RoomModeEvent]) -> ModalAnalysisResult:
    if not events:
        return _empty_result()
    events = sorted(events, key=lambda ev: (-float(ev.severity), -float(ev.peak_db), float(ev.freq_hz)))
    deduped: list[RoomModeEvent] = []
    for event in events:
        if any(abs(np.log2(max(event.freq_hz, 1e-9) / max(kept.freq_hz, 1e-9))) < 0.08 for kept in deduped):
            continue
        deduped.append(event)
    events_tuple = tuple(deduped)
    modal_area = float(sum(max(0.0, ev.area_db_oct) for ev in events_tuple))
    voice_risk = float(
        np.clip(
            sum(ev.severity * ev.confidence * _voice_weight(ev.freq_hz) for ev in events_tuple),
            0.0,
            1.0,
        )
    )
    worst = events_tuple[0]
    return ModalAnalysisResult(
        events=events_tuple,
        worst_mode_hz=float(worst.freq_hz),
        worst_mode_severity=float(worst.severity),
        mode_count=int(len(events_tuple)),
        modal_area_db_oct=float(modal_area),
        voice_band_modal_risk=float(voice_risk),
    )

def detect_room_modes(
    freq_axis,
    measured_mag_db,
    target_mag_db=None,
    corrected_mag_db=None,
    group_delay_ms=None,
    confidence_mask=None,
    left_mag_db=None,
    right_mag_db=None,
    rt60_by_band=None,
    *,
    lo_hz: float = 20.0,
    hi_hz: float = 300.0,
    min_peak_db: float = 1.5,
    min_width_oct: float = 1.0 / 48.0,
    max_width_oct: float = 1.0 / 2.0,
    baseline_smooth_oct: float = 2.0,
    detect_smooth_oct: float = 8.0,
) -> ModalAnalysisResult:
    try:
        lo = float(lo_hz)
        hi = float(hi_hz)
        if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
            return _empty_result()
        prepared = _prepare_arrays(
            freq_axis,
            measured_mag_db,
            target_mag_db,
            corrected_mag_db,
            group_delay_ms,
            confidence_mask,
            left_mag_db,
            right_mag_db,
            lo,
            hi,
        )
        if prepared is None:
            return _empty_result()
        freq, analysis_db, conf, gd_ms, left_mag, right_mag = prepared

        baseline_width = float(np.clip(baseline_smooth_oct, 1.0 / 3.0, 3.0))
        detect_width = float(np.clip(1.0 / max(float(detect_smooth_oct), 1.0), 1.0 / 96.0, 1.0 / 3.0))
        baseline = _smooth_log_box(freq, analysis_db, baseline_width)
        fine = _smooth_log_box(freq, analysis_db, detect_width)
        excess = np.nan_to_num(fine - baseline, nan=0.0, posinf=0.0, neginf=0.0)

        if excess.size < 8 or float(np.nanmax(excess)) < float(min_peak_db):
            return _empty_result()

        local = np.zeros(excess.size, dtype=bool)
        local[1:-1] = (excess[1:-1] >= excess[:-2]) & (excess[1:-1] >= excess[2:])
        candidate_idxs = np.flatnonzero(local & (excess >= float(min_peak_db)) & (conf >= 0.20))
        if candidate_idxs.size == 0:
            return _empty_result()

        events: list[RoomModeEvent] = []
        min_width = float(max(1.0 / 192.0, min_width_oct))
        max_width = float(max(max_width_oct, min_width))
        geom = _modal_candidate_geometry(freq, excess, candidate_idxs, float(min_peak_db))
        for pos, raw_idx in enumerate(geom["idx"]):
            idx = int(raw_idx)
            event = _build_room_mode_event(
                freq=freq,
                conf=conf,
                gd_ms=gd_ms,
                left_mag=left_mag,
                right_mag=right_mag,
                rt60_by_band=rt60_by_band,
                geom=geom,
                pos=pos,
                idx=idx,
                min_peak_db=float(min_peak_db),
                min_width_oct=min_width,
                max_width=max_width,
            )
            if event is not None:
                events.append(event)

        return _finalize_room_mode_events(events)
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
        return _empty_result()


__all__ = ['_modal_candidate_geometry', '_voice_weight', 'modal_support_for_band', '_lr_consistency_at', '_decay_severity_at', '_classify_event', 'detect_room_modes']


def _link_sibling_exports() -> None:
    import importlib
    package = __package__
    for module_name in ['modal_analysis_01', 'modal_analysis_02']:
        if module_name == __name__.rsplit('.', 1)[-1]:
            continue
        module = importlib.import_module(f"{package}.{module_name}")
        for symbol in getattr(module, "__all__", ()):
            globals().setdefault(symbol, getattr(module, symbol))


_link_sibling_exports()
