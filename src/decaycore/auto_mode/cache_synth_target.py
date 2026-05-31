# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Adaptive target synthesis cache."""

from __future__ import annotations

from threading import RLock

from .cache_measurement_sig import _auto_get_measurement_signature

# Sentinel: distinguishes "not in cache" from "cached None (synthesis failed)"
_SYNTH_TARGET_MISS = object()
_SYNTH_TARGET_CACHE: dict = {}
_SYNTH_TARGET_CACHE_LOCK = RLock()


def _synth_target_cache_key(
    measurement_sig: str,
    *,
    tilt_comp_frac: float,
    bass_comp_frac: float,
    bass_comp_ref_db: float,
    hf_comp_frac: float,
    smooth_oct: float,
) -> tuple:
    return (
        str(measurement_sig),
        round(float(tilt_comp_frac), 5),
        round(float(bass_comp_frac), 5),
        round(float(bass_comp_ref_db), 4),
        round(float(hf_comp_frac), 5),
        round(float(smooth_oct), 6),
    )


def get_or_build_synth_target(
    measurements: dict,
    *,
    tilt_comp_frac: float,
    bass_comp_frac: float,
    bass_comp_ref_db: float,
    hf_comp_frac: float,
    smooth_oct: float,
    synth_fn,
):
    """Return cached adaptive target synthesis result, building it once per unique input.

    Returns the same value as synth_fn(...): a (hc_f, hc_m) tuple or None on failure.
    synth_fn must accept (f_l, m_l, f_r, m_r, *, bass_comp_frac, bass_comp_ref_db,
    tilt_comp_frac, hf_comp_frac, smooth_oct, measurements=None).
    """
    msig = _auto_get_measurement_signature(measurements)
    key = _synth_target_cache_key(
        msig,
        tilt_comp_frac=float(tilt_comp_frac),
        bass_comp_frac=float(bass_comp_frac),
        bass_comp_ref_db=float(bass_comp_ref_db),
        hf_comp_frac=float(hf_comp_frac),
        smooth_oct=float(smooth_oct),
    )
    with _SYNTH_TARGET_CACHE_LOCK:
        hit = _SYNTH_TARGET_CACHE.get(key, _SYNTH_TARGET_MISS)
        if hit is not _SYNTH_TARGET_MISS:
            return hit
    try:
        result = synth_fn(
            measurements.get("f_l"),
            measurements.get("m_l"),
            measurements.get("f_r"),
            measurements.get("m_r"),
            bass_comp_frac=float(bass_comp_frac),
            bass_comp_ref_db=float(bass_comp_ref_db),
            tilt_comp_frac=float(tilt_comp_frac),
            hf_comp_frac=float(hf_comp_frac),
            smooth_oct=float(smooth_oct),
            measurements=measurements,
        )
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
        result = None
    with _SYNTH_TARGET_CACHE_LOCK:
        hit = _SYNTH_TARGET_CACHE.get(key, _SYNTH_TARGET_MISS)
        if hit is not _SYNTH_TARGET_MISS:
            return hit
        if len(_SYNTH_TARGET_CACHE) >= 128:
            _SYNTH_TARGET_CACHE.clear()
        _SYNTH_TARGET_CACHE[key] = result
        return result
