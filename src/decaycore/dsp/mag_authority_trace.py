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

from collections import Counter
from typing import Any

import numpy as np

MAG_AUTHORITY_TRACE_VERSION = 1

REASON_LOW_BASS_CUTS_ONLY = "low_bass_cuts_only"
REASON_LOW_BASS_FLOOR_REAPPLIED = "low_bass_floor_reapplied"
REASON_EXCURSION_FULL_BLOCK = "excursion_full_block"
REASON_EXCURSION_SOFT_CAP = "excursion_soft_cap"
REASON_USER_BOOST_CAP = "user_boost_cap"
REASON_USER_CUT_CAP = "user_cut_cap"
REASON_ACOUSTIC_AUTHORITY_BOOST_CAP = "acoustic_authority_boost_cap"
REASON_ACOUSTIC_AUTHORITY_CUT_CAP = "acoustic_authority_cut_cap"
REASON_SOFTCLIP_BOOST = "softclip_boost"
REASON_SOFTCLIP_CUT = "softclip_cut"
REASON_HARDCLAMP_BOOST = "hardclamp_boost"
REASON_HARDCLAMP_CUT = "hardclamp_cut"
REASON_SLOPE_LIMIT = "slope_limit"
REASON_CONFIDENCE_PULL = "confidence_pull"
REASON_REGULARIZATION_SMOOTH = "regularization_smooth"
REASON_WAV_TRANSITION_SMOOTH = "wav_transition_smooth"
REASON_BASS_BOOST_RESTORE = "bass_boost_restore"
REASON_TRANSITION_FADE = "transition_fade"

BOOST_LIMIT_REASONS = {
    REASON_LOW_BASS_CUTS_ONLY,
    REASON_EXCURSION_FULL_BLOCK,
    REASON_EXCURSION_SOFT_CAP,
    REASON_USER_BOOST_CAP,
    REASON_ACOUSTIC_AUTHORITY_BOOST_CAP,
    REASON_SOFTCLIP_BOOST,
    REASON_HARDCLAMP_BOOST,
}
CUT_LIMIT_REASONS = {
    REASON_LOW_BASS_FLOOR_REAPPLIED,
    REASON_USER_CUT_CAP,
    REASON_ACOUSTIC_AUTHORITY_CUT_CAP,
    REASON_SOFTCLIP_CUT,
    REASON_HARDCLAMP_CUT,
}


def _finite_array(value: Any) -> np.ndarray:
    """Palauttaa taatusti finite-arvoisen taulukon; tulos on vain-luku-kayttoon."""
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.zeros(0, dtype=float)
    # nan_to_num tekee useita läpikäyntejä ja väliallokaatioita; data on
    # lähes aina jo finite, joten tarkistus ensin on selvästi halvempi.
    # Kopio jätetään pois: build_mag_authority_stage vain lukee tuloksen.
    if np.isfinite(arr).all():
        return arr
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(out) if np.isfinite(out) else float(default)


def _compact_reason_codes(reason_codes: Any, *, active: bool) -> list[str]:
    if not active:
        return []
    if isinstance(reason_codes, str):
        raw = [reason_codes]
    else:
        try:
            raw = list(reason_codes or [])
        except TypeError:
            raw = []
    out: list[str] = []
    for item in raw:
        code = str(item or "").strip()
        if code and code not in out:
            out.append(code)
    return out


def build_mag_authority_stage(
    stage: str,
    before_db: Any,
    after_db: Any,
    freq_axis: Any,
    mask: Any,
    *,
    reason_codes: Any = (),
    restored_allowed_correction: bool = False,
    changed_eps_db: float = 1e-9,
) -> dict[str, Any]:
    identical = after_db is before_db
    before = _finite_array(before_db)
    after = before if identical else _finite_array(after_db)
    freq = _finite_array(freq_axis)
    try:
        m = np.asarray(mask, dtype=bool).reshape(-1)
    except (TypeError, ValueError):
        m = np.zeros(0, dtype=bool)

    n = int(min(before.size, after.size, freq.size, m.size))
    if n <= 0:
        active_reasons: list[str] = []
        return {
            "stage": str(stage),
            "reason_codes": active_reasons,
            "changed_bins": 0,
            "boost_peak_before_db": 0.0,
            "boost_peak_after_db": 0.0,
            "cut_peak_before_db": 0.0,
            "cut_peak_after_db": 0.0,
            "max_delta_db": 0.0,
            "rms_delta_db": 0.0,
            "bass_changed_bins_20_200": 0,
            "bass_max_delta_db_20_200": 0.0,
            "authority_only_reduced": False,
            "restored_allowed_correction": bool(restored_allowed_correction),
        }

    before = before[:n]
    after = after[:n] if not identical else before
    freq = freq[:n]
    m = m[:n]
    # _finite_array takaa finite-arvot, joten isfinite(before/after/freq)-maskit
    # olisivat kaikkialla True: valid == m, ja fallback == all-True.
    valid = m
    if not np.any(valid):
        valid = np.ones(n, dtype=bool)
    delta = after - before
    abs_delta = np.abs(delta)
    d = delta[valid]
    changed = valid & (abs_delta > float(changed_eps_db))
    changed_bins = int(np.count_nonzero(changed))
    active_reasons = _compact_reason_codes(reason_codes, active=changed_bins > 0)

    before_v = before[valid]
    after_v = after[valid] if not identical else before_v
    d_v = d[np.isfinite(d)]
    bass = changed & (freq >= 20.0) & (freq <= 200.0)
    bass_delta = abs_delta[bass]
    authority_only_reduced = False
    if changed_bins > 0:
        dd = delta[changed]
        authority_only_reduced = bool(np.all(dd <= float(changed_eps_db)))

    return {
        "stage": str(stage),
        "reason_codes": active_reasons,
        "changed_bins": int(changed_bins),
        "boost_peak_before_db": _finite_float(np.max(before_v) if before_v.size else 0.0),
        "boost_peak_after_db": _finite_float(np.max(after_v) if after_v.size else 0.0),
        "cut_peak_before_db": _finite_float(np.min(before_v) if before_v.size else 0.0),
        "cut_peak_after_db": _finite_float(np.min(after_v) if after_v.size else 0.0),
        "max_delta_db": _finite_float(np.max(np.abs(d_v)) if d_v.size else 0.0),
        "rms_delta_db": _finite_float(np.sqrt(np.mean(d_v * d_v)) if d_v.size else 0.0),
        "bass_changed_bins_20_200": int(np.count_nonzero(bass)),
        "bass_max_delta_db_20_200": _finite_float(np.max(bass_delta) if bass_delta.size else 0.0),
        "authority_only_reduced": bool(authority_only_reduced),
        "restored_allowed_correction": bool(restored_allowed_correction),
    }


def append_mag_authority_stage(
    trace: list[dict[str, Any]] | None,
    stage: str,
    before_db: Any,
    after_db: Any,
    freq_axis: Any,
    mask: Any,
    *,
    reason_codes: Any = (),
    restored_allowed_correction: bool = False,
) -> None:
    if trace is None:
        # Presolve-ajo: jälki ei päädy mihinkään (st hylätään), joten
        # vaihekohtaista analyysia ei rakenneta lainkaan.
        return
    trace.append(
        build_mag_authority_stage(
            stage,
            before_db,
            after_db,
            freq_axis,
            mask,
            reason_codes=reason_codes,
            restored_allowed_correction=restored_allowed_correction,
        )
    )


def summarize_mag_authority_trace(trace: Any) -> dict[str, Any]:
    stages = [dict(item) for item in trace if isinstance(item, dict)] if isinstance(trace, list) else []
    reasons: list[str] = []
    boost_limited_bins = 0
    cut_limited_bins = 0
    max_delta = 0.0
    bass_changed = 0
    for stage in stages:
        stage_reasons = [str(r) for r in stage.get("reason_codes", []) if str(r)]
        changed_bins = int(_finite_float(stage.get("changed_bins", 0), 0.0))
        reasons.extend(stage_reasons)
        if any(r in BOOST_LIMIT_REASONS for r in stage_reasons):
            boost_limited_bins += changed_bins
        if any(r in CUT_LIMIT_REASONS for r in stage_reasons):
            cut_limited_bins += changed_bins
        max_delta = max(max_delta, abs(_finite_float(stage.get("max_delta_db", 0.0), 0.0)))
        bass_changed += int(_finite_float(stage.get("bass_changed_bins_20_200", 0), 0.0))
    counts = Counter(reasons)
    top_reason = counts.most_common(1)[0][0] if counts else ""
    return {
        "mag_authority_trace_version": int(MAG_AUTHORITY_TRACE_VERSION),
        "mag_authority_trace_stage_count": int(len(stages)),
        "mag_authority_trace_active_reasons": sorted(counts),
        "mag_authority_trace_top_reason": str(top_reason),
        "mag_authority_trace_boost_limited_bins": int(boost_limited_bins),
        "mag_authority_trace_cut_limited_bins": int(cut_limited_bins),
        "mag_authority_trace_max_delta_db": float(max_delta),
        "mag_authority_trace_bass_changed_bins_20_200": int(bass_changed),
    }


__all__ = [
    "MAG_AUTHORITY_TRACE_VERSION",
    "REASON_ACOUSTIC_AUTHORITY_BOOST_CAP",
    "REASON_ACOUSTIC_AUTHORITY_CUT_CAP",
    "REASON_BASS_BOOST_RESTORE",
    "REASON_CONFIDENCE_PULL",
    "REASON_EXCURSION_FULL_BLOCK",
    "REASON_EXCURSION_SOFT_CAP",
    "REASON_HARDCLAMP_BOOST",
    "REASON_HARDCLAMP_CUT",
    "REASON_LOW_BASS_CUTS_ONLY",
    "REASON_LOW_BASS_FLOOR_REAPPLIED",
    "REASON_REGULARIZATION_SMOOTH",
    "REASON_SLOPE_LIMIT",
    "REASON_SOFTCLIP_BOOST",
    "REASON_SOFTCLIP_CUT",
    "REASON_TRANSITION_FADE",
    "REASON_USER_BOOST_CAP",
    "REASON_USER_CUT_CAP",
    "REASON_WAV_TRANSITION_SMOOTH",
    "append_mag_authority_stage",
    "build_mag_authority_stage",
    "summarize_mag_authority_trace",
]
