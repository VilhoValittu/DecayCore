# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

import logging
from typing import Any

import numpy as np

from .limits import limit_slope_per_octave

logger = logging.getLogger(__name__)

# Severity thresholds: resonance vs. reflection events
_RES_ONSET_MS_LO: float = 18.0
_RES_ONSET_MS_HI: float = 22.0
_RES_RATIO_ONSET_LO: float = 0.10
_RES_RATIO_ONSET_HI: float = 0.14
_RES_RATIO_SPAN: float = 0.55
_REFL_ONSET_MS: float = 32.0
_REFL_RATIO_ONSET: float = 0.18
_REFL_TYPE_WEIGHT: float = 0.55
_REFL_RATIO_SPAN: float = 0.75
_ABS_SPAN_BASE: float = 80.0
_ABS_SPAN_MULT: float = 4.0


def _tdc_safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _tdc_event_get(event: Any, key: str, default: Any = None) -> Any:
    if isinstance(event, dict):
        return event.get(key, default)
    return getattr(event, key, default)


def _clip_field(event: Any, key: str, lo: float, hi: float, default: float = 0.0) -> float:
    return float(np.clip(_tdc_safe_float(_tdc_event_get(event, key, default), default), lo, hi))


def _normalize_modal_event(event: Any) -> dict[str, Any] | None:
    freq_hz = _tdc_safe_float(_tdc_event_get(event, "freq_hz", _tdc_event_get(event, "freq", np.nan)), np.nan)
    if not np.isfinite(freq_hz) or freq_hz <= 0.0:
        return None

    confidence = _clip_field(event, "confidence", 0.0, 1.0, 1.0)
    severity = _tdc_safe_float(_tdc_event_get(event, "severity", 0.0), 0.0)
    severity = max(severity, _tdc_safe_float(_tdc_event_get(event, "correction_priority", 0.0), 0.0))
    severity = max(severity, _tdc_safe_float(_tdc_event_get(event, "cut_priority", 0.0), 0.0))
    severity = float(np.clip(severity, 0.0, 1.0))

    gd_excess_ms = _clip_field(event, "gd_excess_ms", 0.0, 300.0)
    decay_severity = _clip_field(event, "decay_severity", 0.0, 1.0)
    safe_cut_db = _clip_field(event, "safe_cut_db", 0.0, 12.0)
    safe_width_oct = _clip_field(event, "safe_width_oct", 0.0, 1.0)
    voice_clarity_risk = _clip_field(event, "voice_clarity_risk", 0.0, 1.0)
    kind = str(_tdc_event_get(event, "kind", "") or "").strip().lower()

    decay_evidence = float(np.clip(max(decay_severity, gd_excess_ms / 80.0), 0.0, 1.0))
    modal_support = float(np.clip(severity * confidence * (0.55 + 0.45 * decay_evidence), 0.0, 1.0))
    if 80.0 <= freq_hz <= 160.0 and voice_clarity_risk > 0.0:
        modal_support = float(np.clip(max(modal_support, 0.45 * voice_clarity_risk * confidence), 0.0, 1.0))

    is_uncertain = bool(
        confidence < 0.35
        or kind in {"uncertain", "local_comb"}
        or (safe_width_oct > 0.0 and safe_width_oct < 1.0 / 48.0 and decay_evidence < 0.35)
    )

    return {
        "freq_hz": float(freq_hz),
        "severity": float(severity),
        "confidence": float(confidence),
        "safe_cut_db": float(safe_cut_db),
        "safe_width_oct": float(safe_width_oct),
        "voice_clarity_risk": float(voice_clarity_risk),
        "gd_excess_ms": float(gd_excess_ms),
        "decay_severity": float(decay_severity),
        "decay_evidence": float(decay_evidence),
        "modal_support": float(modal_support),
        "kind": str(kind or "unknown"),
        "is_uncertain": bool(is_uncertain),
    }


def _modal_event_match(
    freq_hz: float,
    modal_events: list[dict[str, Any]],
    *,
    max_distance_oct: float = 0.18,
) -> dict[str, Any] | None:
    if not modal_events or not np.isfinite(freq_hz) or freq_hz <= 0.0:
        return None
    event_freqs = np.array([e["freq_hz"] for e in modal_events], dtype=float)
    match_widths = np.array(
        [max(float(max_distance_oct), 0.60 * float(e.get("safe_width_oct", 0.0) or 0.0)) for e in modal_events],
        dtype=float,
    )
    dist_oct = np.abs(np.log2(np.clip(event_freqs, 1e-9, None) / max(freq_hz, 1e-9)))
    in_range = dist_oct <= match_widths
    if not np.any(in_range):
        return None
    distance_score = 1.0 - np.minimum(1.0, dist_oct / np.maximum(match_widths, 1e-9))
    modal_supports = np.array([float(e.get("modal_support", 0.0) or 0.0) for e in modal_events], dtype=float)
    scores = 0.65 * modal_supports + 0.35 * distance_score
    scores[~in_range] = -1.0
    best_idx = int(np.argmax(scores))
    return modal_events[best_idx] if scores[best_idx] >= 0.0 else None


def _has_rt60_data(rt60_info: Any) -> bool:
    """True when rt60_info is not completely absent (None). Empty dict/scalar uses fallback."""
    return rt60_info is not None


def _rt60_at(rt60_info: Any, freq_hz: float) -> float:
    default = 0.4
    try:
        if isinstance(rt60_info, (int, float)):
            v = float(rt60_info)
            return v if np.isfinite(v) and v > 0.1 else default
        if isinstance(rt60_info, dict) and rt60_info:
            c = np.array(sorted(rt60_info.keys()), dtype=float)
            r = np.array([rt60_info[k] for k in c], dtype=float)
            mask = np.isfinite(c) & np.isfinite(r) & (c > 0) & (r > 0.05) & (r < 5.0)
            if np.count_nonzero(mask) < 2:
                vv = float(np.median(r[mask])) if np.count_nonzero(mask) else 0.0
                return vv if vv > 0.1 else default
            c, r = c[mask], r[mask]
            x = np.log10(np.clip(freq_hz, c.min(), c.max()))
            return float(np.interp(x, np.log10(c), r))
    except (TypeError, ValueError, FloatingPointError, OverflowError):
        return default
    return default


def _compute_reflection_severity(
    f_res: float,
    error_ms: float,
    ref_rt60: float,
    *,
    is_resonance: bool,
) -> float:
    excess_ratio = error_ms / (ref_rt60 * 1000.0 + 1e-12)
    if not np.isfinite(excess_ratio):
        return 0.0
    if is_resonance:
        onset_ms = _RES_ONSET_MS_LO if f_res <= 120.0 else _RES_ONSET_MS_HI
        ratio_onset = _RES_RATIO_ONSET_LO if f_res <= 120.0 else _RES_RATIO_ONSET_HI
        type_weight = 1.0
        ratio_span = _RES_RATIO_SPAN
    else:
        onset_ms = _REFL_ONSET_MS
        ratio_onset = _REFL_RATIO_ONSET
        type_weight = _REFL_TYPE_WEIGHT
        ratio_span = _REFL_RATIO_SPAN
    abs_span = max(_ABS_SPAN_BASE, _ABS_SPAN_MULT * onset_ms)
    abs_severity = max(0.0, (error_ms - onset_ms) / abs_span)
    ratio_severity = max(0.0, (excess_ratio - ratio_onset) / ratio_span)
    severity = float(type_weight * max(abs_severity, ratio_severity))
    if not np.isfinite(severity) or severity <= 0.0:
        return 0.0
    return float(np.clip(severity, 0.0, 2.2))


def _apply_modal_adjustment(
    reduction_db: float,
    width_oct: float,
    f_res: float,
    modal_event: dict[str, Any],
) -> tuple[float, float, bool]:
    modal_support = float(modal_event.get("modal_support", 0.0) or 0.0)
    decay_evidence = float(modal_event.get("decay_evidence", 0.0) or 0.0)
    voice_risk = float(modal_event.get("voice_clarity_risk", 0.0) or 0.0)
    safe_cut_db = float(modal_event.get("safe_cut_db", 0.0) or 0.0)
    safe_width_oct = float(modal_event.get("safe_width_oct", 0.0) or 0.0)
    if modal_event.get("is_uncertain"):
        modal_mult = float(np.clip(0.55 + 0.35 * modal_support, 0.45, 0.90))
    else:
        modal_mult = 1.0 + 0.22 * modal_support * decay_evidence
        if 80.0 <= f_res <= 160.0 and voice_risk > 0.0:
            modal_mult += 0.06 * min(voice_risk, decay_evidence if decay_evidence > 0.0 else 0.5)
        modal_mult = float(np.clip(modal_mult, 0.85, 1.24))
    reduction_db *= modal_mult
    if safe_cut_db > 0.0:
        reduction_db = min(reduction_db, max(0.75, safe_cut_db * (1.0 + 0.10 * decay_evidence)))
    if safe_width_oct > 0.0:
        width_oct = float(np.clip(0.70 * width_oct + 0.30 * safe_width_oct, 0.06, 0.24))
    voice_used = bool(80.0 <= f_res <= 160.0 and voice_risk > 0.0)
    return reduction_db, width_oct, voice_used


def _gaussian_kernel(f_axis: np.ndarray, f_res: float, width_oct: float) -> np.ndarray:
    kernel = np.zeros_like(f_axis, dtype=float)
    positive = f_axis > 0.0
    if np.any(positive):
        dist_oct = np.abs(np.log2(np.clip(f_axis[positive], 1e-9, None) / f_res))
        kernel[positive] = np.exp(-0.5 * (dist_oct / width_oct) ** 2)
    return kernel


def _apply_tdc_slope_limit(
    freq_axis: Any,
    tdc_reduction_db: np.ndarray,
    max_red: float,
    max_slope_db_per_oct: float,
) -> np.ndarray:
    tdc_reduction_db = np.minimum(tdc_reduction_db, max_red)
    try:
        if max_slope_db_per_oct and float(max_slope_db_per_oct) > 0:
            tdc_reduction_db = limit_slope_per_octave(
                freq_axis,
                tdc_reduction_db,
                max_db_per_oct=float(max_slope_db_per_oct),
            )
    except (TypeError, ValueError, FloatingPointError, IndexError):
        logger.debug("TDC slope limiting failed; continuing without it.", exc_info=True)
    return np.clip(tdc_reduction_db, 0.0, max_red)


def _compute_tdc_reduction_stats(freq_axis: Any, tdc_reduction_db: np.ndarray) -> dict:
    try:
        f = np.asarray(freq_axis, dtype=float).reshape(-1)
        r = np.asarray(tdc_reduction_db, dtype=float).reshape(-1)
        valid = np.isfinite(f) & np.isfinite(r)
        if f.size != r.size:
            valid = np.zeros(0, dtype=bool)
    except (TypeError, ValueError):
        f = np.zeros(0, dtype=float)
        r = np.zeros(0, dtype=float)
        valid = np.zeros(0, dtype=bool)

    peak_db = peak_hz = rms_db = area_db_hz = band_lo = band_hi = 0.0
    if valid.size and np.any(valid):
        rv = np.clip(r[valid], 0.0, None)
        fv = f[valid]
        if rv.size:
            peak_idx = int(np.argmax(rv))
            peak_db = float(rv[peak_idx])
            peak_hz = float(fv[peak_idx]) if peak_db > 1e-9 else 0.0
            rms_db = float(np.sqrt(np.mean(rv * rv))) if peak_db > 1e-9 else 0.0
            if rv.size > 1:
                df = np.diff(fv)
                seg = 0.5 * (rv[:-1] + rv[1:]) * df
                area_db_hz = float(np.sum(seg[np.isfinite(seg)]))
            active = rv > 0.10
            if np.any(active):
                band_lo = float(np.min(fv[active]))
                band_hi = float(np.max(fv[active]))
    return {
        "peak_db": peak_db, "peak_hz": peak_hz, "rms_db": rms_db,
        "area_db_hz": area_db_hz, "band_lo": band_lo, "band_hi": band_hi,
    }


def _tdc_normalized_modal_events(modal_events, tdc_hi_hz: float) -> list[dict[str, Any]]:
    return [
        n
        for me in (modal_events or [])
        if (n := _normalize_modal_event(me)) is not None and n["freq_hz"] <= tdc_hi_hz
    ]


def _tdc_event_rt60(rt60_info, f_res: float) -> float | None:
    if not _has_rt60_data(rt60_info):
        logger.info(
            "TDC: skipping event at %.1f Hz — no reliable RT60 data available",
            f_res,
        )
        return None
    ref_rt60 = _rt60_at(rt60_info, f_res)
    if not (np.isfinite(ref_rt60) and ref_rt60 > 0.0):
        logger.warning(
            "TDC: rt60_at(%.1f Hz) returned invalid/missing value (%.3g); "
            "falling back to 0.4 s — TDC severity may be unreliable",
            f_res,
            ref_rt60,
        )
        ref_rt60 = 0.4
    return float(ref_rt60)


def _tdc_reflection_event_result(
    rev,
    rt60_info,
    normalized_modal_events: list[dict[str, Any]],
    *,
    tdc_hi_hz: float,
    strength: float,
) -> tuple[str, dict[str, Any] | None]:
    try:
        f_res = float(rev.get("freq", np.nan))
        error_ms = float(rev.get("gd_error", rev.get("error_ms", np.nan)))
    except (TypeError, ValueError):
        return "ignore", None
    if not (np.isfinite(f_res) and np.isfinite(error_ms)):
        return "ignore", None
    if f_res <= 0.0 or error_ms <= 0.0:
        return "ignore", None
    ref_rt60 = _tdc_event_rt60(rt60_info, f_res)
    if ref_rt60 is None:
        return "ignore", None

    node_type = str(rev.get("type", "") or "").strip().lower()
    is_resonance = bool(node_type == "resonance" or (not node_type and f_res <= 200.0))
    if f_res > tdc_hi_hz or (node_type == "reflection" and f_res > 200.0):
        return "skip_high", None

    severity = _compute_reflection_severity(f_res, error_ms, ref_rt60, is_resonance=is_resonance)
    if severity <= 0.0:
        return "ignore", None

    dynamic_mult = float(np.clip(severity * strength, 0.0, 2.0))
    reduction_db = dynamic_mult * 3.5
    if not np.isfinite(reduction_db) or reduction_db <= 0.0:
        return "ignore", None

    width_oct = float(
        np.clip(
            0.18 - 0.04 * min(severity, 2.0) + (0.04 if not is_resonance else 0.0),
            0.07,
            0.24,
        )
    )

    modal_event = _modal_event_match(f_res, normalized_modal_events)
    voice_used = False
    modal_support = 0.0
    if modal_event is not None:
        reduction_db, width_oct, voice_used = _apply_modal_adjustment(reduction_db, width_oct, f_res, modal_event)
        modal_support = float(modal_event.get("modal_support", 0.0) if modal_event else 0.0)

    return "used", {
        "freq_hz": float(f_res),
        "error_ms": float(error_ms),
        "node_type": str(node_type or ("resonance" if is_resonance else "reflection")),
        "severity": float(severity),
        "reduction_db": float(reduction_db),
        "width_oct": float(width_oct),
        "modal_support": float(modal_support),
        "modal_used": bool(modal_event is not None),
        "modal_voice_used": bool(voice_used),
    }


def _tdc_apply_reflection_events(
    reflections,
    rt60_info,
    normalized_modal_events: list[dict[str, Any]],
    *,
    f_axis: np.ndarray,
    strength: float,
    tdc_hi_hz: float,
) -> tuple[np.ndarray, int, int, int, int, int, dict[str, Any] | None]:
    tdc_reduction_db = np.zeros_like(f_axis, dtype=float)
    events_seen = events_used = skipped_high = 0
    modal_events_used = modal_voice_events_used = 0
    strongest_event: dict | None = None
    strongest_reduction = 0.0

    for rev in reflections or []:
        if not isinstance(rev, dict):
            continue
        events_seen += 1
        status, event = _tdc_reflection_event_result(
            rev,
            rt60_info,
            normalized_modal_events,
            tdc_hi_hz=tdc_hi_hz,
            strength=strength,
        )
        if status == "skip_high":
            skipped_high += 1
            continue
        if status != "used" or event is None:
            continue

        f_res = float(event["freq_hz"])
        reduction_db = float(event["reduction_db"])
        width_oct = float(event["width_oct"])
        if bool(event["modal_used"]):
            modal_events_used += 1
        if bool(event["modal_voice_used"]):
            modal_voice_events_used += 1
        tdc_reduction_db += _gaussian_kernel(f_axis, f_res, width_oct) * reduction_db
        events_used += 1

        if reduction_db > strongest_reduction:
            strongest_reduction = float(reduction_db)
            strongest_event = {
                "tdc_strongest_event_freq_hz": float(event["freq_hz"]),
                "tdc_strongest_event_gd_error_ms": float(event["error_ms"]),
                "tdc_strongest_event_type": str(event["node_type"]),
                "tdc_strongest_event_severity": float(event["severity"]),
                "tdc_strongest_event_reduction_db": float(reduction_db),
                "tdc_strongest_event_width_oct": float(width_oct),
                "tdc_strongest_event_modal_support": float(event["modal_support"]),
            }

    return (
        tdc_reduction_db,
        events_seen,
        events_used,
        skipped_high,
        modal_events_used,
        modal_voice_events_used,
        strongest_event,
    )


def _tdc_apply_reflection_events(
    reflections,
    rt60_info,
    normalized_modal_events: list[dict[str, Any]],
    *,
    f_axis: np.ndarray,
    strength: float,
    tdc_hi_hz: float,
) -> tuple[np.ndarray, int, int, int, int, int, dict[str, Any] | None]:
    tdc_reduction_db = np.zeros_like(f_axis, dtype=float)
    events_seen = events_used = skipped_high = 0
    modal_events_used = modal_voice_events_used = 0
    strongest_event: dict | None = None
    strongest_reduction = 0.0

    for rev in reflections or []:
        if not isinstance(rev, dict):
            continue
        events_seen += 1
        status, event = _tdc_reflection_event_result(
            rev,
            rt60_info,
            normalized_modal_events,
            tdc_hi_hz=tdc_hi_hz,
            strength=strength,
        )
        if status == "skip_high":
            skipped_high += 1
            continue
        if status != "used" or event is None:
            continue

        f_res = float(event["freq_hz"])
        reduction_db = float(event["reduction_db"])
        width_oct = float(event["width_oct"])
        if bool(event["modal_used"]):
            modal_events_used += 1
        if bool(event["modal_voice_used"]):
            modal_voice_events_used += 1
        tdc_reduction_db += _gaussian_kernel(f_axis, f_res, width_oct) * reduction_db
        events_used += 1

        if reduction_db > strongest_reduction:
            strongest_reduction = float(reduction_db)
            strongest_event = {
                "tdc_strongest_event_freq_hz": float(event["freq_hz"]),
                "tdc_strongest_event_gd_error_ms": float(event["error_ms"]),
                "tdc_strongest_event_type": str(event["node_type"]),
                "tdc_strongest_event_severity": float(event["severity"]),
                "tdc_strongest_event_reduction_db": float(reduction_db),
                "tdc_strongest_event_width_oct": float(width_oct),
                "tdc_strongest_event_modal_support": float(event["modal_support"]),
            }

    return (
        tdc_reduction_db,
        events_seen,
        events_used,
        skipped_high,
        modal_events_used,
        modal_voice_events_used,
        strongest_event,
    )


def apply_smart_tdc(
    freq_axis,
    target_mags,
    reflections,
    rt60_info,
    base_strength=0.5,
    max_total_reduction_db: float = 9.0,
    max_slope_db_per_oct: float = 0.0,
    *,
    max_tdc_hz: float = 300.0,
    modal_events=None,
    telemetry: dict | None = None,
):
    """Soveltaa tai paivittaa: apply smart tdc."""
    adjusted_target = np.asarray(target_mags, dtype=float).copy()
    tdc_reduction_db = np.zeros_like(adjusted_target, dtype=float)

    strength = float(np.clip(_tdc_safe_float(base_strength), 0.0, 3.0))
    max_red = _tdc_safe_float(max_total_reduction_db, 9.0)
    if not np.isfinite(max_red):
        max_red = 9.0
    tdc_hi_hz = float(np.clip(_tdc_safe_float(max_tdc_hz, 300.0), 80.0, 500.0))

    def _emit_telemetry(
        *, events_seen=0, events_used=0, skipped_high=0, strongest=None,
        reason="", modal_events_seen=0, modal_events_used=0, modal_voice_events_used=0,
    ) -> None:
        if telemetry is None:
            return
        stats = _compute_tdc_reduction_stats(freq_axis, tdc_reduction_db)
        telemetry.update({
            "tdc_enabled": bool(strength > 0.0 and max_red > 0.0),
            "tdc_applied": bool(stats["peak_db"] > 1e-9),
            "tdc_events_seen": int(events_seen),
            "tdc_events_used": int(events_used),
            "tdc_events_skipped_high": int(skipped_high),
            "tdc_peak_reduction_db": stats["peak_db"],
            "tdc_peak_reduction_hz": stats["peak_hz"],
            "tdc_reduction_rms_db": stats["rms_db"],
            "tdc_reduction_area_db_hz": stats["area_db_hz"],
            "tdc_reduction_band_low_hz": stats["band_lo"],
            "tdc_reduction_band_high_hz": stats["band_hi"],
            "tdc_max_tdc_hz": float(tdc_hi_hz),
            "tdc_skip_reason": str(reason),
            "tdc_modal_events_seen": int(modal_events_seen),
            "tdc_modal_events_used": int(modal_events_used),
            "tdc_modal_voice_events_used": int(modal_voice_events_used),
        })
        if isinstance(strongest, dict):
            telemetry.update(strongest)

    if strength <= 0.0 or max_red <= 0.0:
        _emit_telemetry(reason="disabled")
        return adjusted_target

    try:
        f_axis = np.asarray(freq_axis, dtype=float).reshape(-1)
        if f_axis.size != adjusted_target.size:
            _emit_telemetry(reason="invalid_axis")
            return adjusted_target
    except (TypeError, ValueError):
        _emit_telemetry(reason="invalid_axis")
        return adjusted_target

    normalized_modal_events = _tdc_normalized_modal_events(modal_events, tdc_hi_hz)

    (
        tdc_reduction_db,
        events_seen,
        events_used,
        skipped_high,
        modal_events_used,
        modal_voice_events_used,
        strongest_event,
    ) = _tdc_apply_reflection_events(
        reflections,
        rt60_info,
        normalized_modal_events,
        f_axis=f_axis,
        strength=strength,
        tdc_hi_hz=tdc_hi_hz,
    )

    tdc_reduction_db = _apply_tdc_slope_limit(freq_axis, tdc_reduction_db, max_red, max_slope_db_per_oct)
    adjusted_target -= tdc_reduction_db
    _emit_telemetry(
        events_seen=events_seen,
        events_used=events_used,
        skipped_high=skipped_high,
        strongest=strongest_event,
        reason="" if events_used > 0 else "no_matching_events",
        modal_events_seen=len(normalized_modal_events),
        modal_events_used=modal_events_used,
        modal_voice_events_used=modal_voice_events_used,
    )
    return adjusted_target
