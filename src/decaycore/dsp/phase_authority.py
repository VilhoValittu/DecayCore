# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""P4 — Phase authority gating.

Builds a per-frequency gain curve [0..1] that gates mixed-phase / excess-phase
correction based on the acoustic authority map (P1).  Low-authority regions
(reflections, nulls, comb filtering, noisy HF) receive reduced or zero phase
correction; high-authority regions pass through unchanged.

Only affects explicit phase/excess-phase correction.
Magnitude correction and minimum-phase reconstruction are unaffected.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .dsp_telemetry import safe_put_many
from .smoothing import smooth_gain_fractional_octave


def build_phase_authority_gain(
    freq_axis,
    authority_phase=None,
    authority_modal_support=None,
    authority_reflection_risk=None,
    authority_null_risk=None,
    *,
    phase_limit_hz: float = 600.0,
    disable_above_hz: float = 1200.0,
    gamma: float = 1.20,
    min_gain: float = 0.0,
    soft_floor: float = 0.20,
    smooth_oct: float | None = 1.0 / 6.0,
) -> np.ndarray:
    """Return per-frequency phase authority gain, shape matching freq_axis.

    Values are clipped to [0.0, 1.0].  When authority arrays are absent the
    function returns ones, preserving pre-P4 behaviour exactly.

    Args:
        freq_axis: 1-D frequency axis (Hz).
        authority_phase: P1 phase-authority array [0..1], same length as freq_axis.
        authority_modal_support: P1 modal-support array [0..1].
        authority_reflection_risk: P1 reflection-risk array [0..1].
        authority_null_risk: P1 null-risk array [0..1].
        phase_limit_hz: Frequency below which gain is 1.0 (full correction allowed).
        disable_above_hz: Frequency above which gain fades to 0.0 regardless of authority.
        gamma: Exponent — higher value makes low authority suppress more aggressively.
        min_gain: Floor gain when authority is zero.
        soft_floor: Authority value below which the region is considered unsafe
            unless rescued by modal support.  Not directly used in math but
            documents the intent of gamma / min_gain defaults.
        smooth_oct: Fractional-octave Gaussian smoothing applied to the final
            gain curve.  None disables smoothing.

    Returns:
        phase_gain: np.ndarray, same shape as freq_axis, values in [0.0, 1.0].

    """
    f = np.asarray(freq_axis, dtype=float)

    # --- authority baseline ------------------------------------------------
    # When authority is absent use full authority (ones) so that only the
    # frequency fade applies — this preserves old behaviour below phase_limit_hz
    # while still silencing phase correction above disable_above_hz.
    if authority_phase is None:
        a = np.ones_like(f)
    else:
        try:
            a = np.asarray(authority_phase, dtype=float)
            if a.shape != f.shape:
                a = np.interp(f, np.linspace(f[0], f[-1], len(a)), a)
        except (TypeError, ValueError, IndexError):
            a = np.ones_like(f)

        a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
        a = np.clip(a, 0.0, 1.0)

    # --- risk reductions ---------------------------------------------------
    refl = _safe_interp(f, authority_reflection_risk)
    null = _safe_interp(f, authority_null_risk)
    modal = _safe_interp(f, authority_modal_support)

    if refl is not None:
        a = a * (1.0 - 0.65 * refl)
    if null is not None:
        a = a * (1.0 - 0.45 * null)

    # modal support can partially rescue low-authority regions
    if modal is not None:
        a = np.maximum(a, 0.35 * modal)

    # --- gamma shaping -----------------------------------------------------
    phase_gain = min_gain + (1.0 - min_gain) * np.power(np.clip(a, 0.0, 1.0), gamma)

    # --- frequency fade ----------------------------------------------------
    fade_range = max(disable_above_hz - phase_limit_hz, 1.0)
    fade = (disable_above_hz - f) / fade_range
    fade = np.clip(fade, 0.0, 1.0)
    band = np.where(f <= phase_limit_hz, 1.0, fade)
    phase_gain = phase_gain * band

    # --- optional smoothing ------------------------------------------------
    if smooth_oct is not None and smooth_oct > 0.0 and len(f) > 1:
        phase_gain = _smooth_gain(f, phase_gain, smooth_oct)

    # Fractional-octave smoothing must not leak phase authority back above the
    # explicit correction ceiling.
    phase_gain = np.where(f >= float(disable_above_hz), 0.0, phase_gain)

    return np.clip(phase_gain, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Integration helpers (used by phase_ir_phase.py)
# ---------------------------------------------------------------------------

def apply_phase_authority_gating(
    f: np.ndarray,
    extra_phase: np.ndarray,
    cfg: Any,
    st: Any,
    logger: Any,
) -> np.ndarray:
    """Gate extra_phase by P4 acoustic authority and store telemetry.

    When phase_authority_enable is False the input is returned unchanged.
    When authority arrays are missing from st, gain = 1.0 everywhere.
    """
    enabled = bool(getattr(cfg, "phase_authority_enable", True))
    if enabled:
        phase_gain = build_phase_authority_gain(
            f,
            authority_phase=_st_array(st, "authority_phase"),
            authority_modal_support=_st_array(st, "authority_modal_support"),
            authority_reflection_risk=_st_array(st, "authority_reflection_risk"),
            authority_null_risk=_st_array(st, "authority_null_risk"),
            phase_limit_hz=float(
                getattr(cfg, "phase_c_max", None)
                or getattr(cfg, "phase_limit", 600.0)
                or 600.0
            ),
            disable_above_hz=float(getattr(cfg, "phase_authority_disable_above_hz", 1200.0) or 1200.0),
            gamma=float(getattr(cfg, "phase_authority_gamma", 1.20) or 1.20),
            min_gain=float(getattr(cfg, "phase_authority_min_gain", 0.0)),
            soft_floor=float(getattr(cfg, "phase_authority_soft_floor", 0.20) or 0.20),
            smooth_oct=float(getattr(cfg, "phase_authority_smooth_oct", 1.0 / 6.0) or 1.0 / 6.0),
        )
        result = extra_phase * phase_gain
    else:
        phase_gain = np.ones_like(f)
        result = extra_phase

    _store_phase_authority_telemetry(st, phase_gain, f, enabled=enabled, logger=logger)
    return result


def _st_array(st: Any, key: str) -> np.ndarray | None:
    """Read a list-serialised array from stats dict, return ndarray or None."""
    if not isinstance(st, dict):
        return None
    val = st.get(key)
    if val is None:
        return None
    try:
        return np.asarray(val, dtype=float)
    except (TypeError, ValueError):
        return None


def _store_phase_authority_telemetry(
    st: Any,
    phase_gain: np.ndarray,
    f: np.ndarray,
    *,
    enabled: bool,
    logger: Any,
) -> None:
    try:
        mask_lf = (f >= 20.0) & (f <= 600.0)
        mask_mid = (f > 600.0) & (f <= 1200.0)
        gain_lf = phase_gain[mask_lf] if mask_lf.any() else np.array([1.0])
        gain_mid = phase_gain[mask_mid] if mask_mid.any() else np.array([1.0])
        suppressed = int(np.sum(phase_gain < 0.05))
        low_bins = int(np.sum(mask_lf & (phase_gain < 0.35)))
        safe_put_many(
            st,
            {
                "phase_authority_enabled": bool(enabled),
                "phase_authority_gain_mean_20_600": float(gain_lf.mean()),
                "phase_authority_gain_min_20_600": float(gain_lf.min()),
                "phase_authority_gain_mean_600_1200": float(gain_mid.mean()),
                "phase_authority_low_bins_20_600": low_bins,
                "phase_authority_suppressed_bins": suppressed,
                "phase_authority_gain": phase_gain.tolist(),
            },
        )
        if logger is not None:
            state = "ON" if enabled else "OFF"
            logger.info(
                f"Phase authority: {state} "
                f"mean20_600={gain_lf.mean():.2f} "
                f"min20_600={gain_lf.min():.2f} "
                f"suppressed_bins={suppressed}"
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
    ):
        pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_interp(f: np.ndarray, arr) -> np.ndarray | None:
    """Return arr resampled to f's length, or None if arr is absent/broken."""
    if arr is None:
        return None
    try:
        a = np.asarray(arr, dtype=float)
        if a.shape == f.shape:
            return np.nan_to_num(np.clip(a, 0.0, 1.0), nan=0.0)
        if len(a) > 0:
            x_src = np.linspace(f[0], f[-1], len(a))
            out = np.interp(f, x_src, a)
            return np.nan_to_num(np.clip(out, 0.0, 1.0), nan=0.0)
        return None
    except (TypeError, ValueError, IndexError):
        return None


def _smooth_gain(f: np.ndarray, gain: np.ndarray, smooth_oct: float) -> np.ndarray:
    """Smooth a phase-authority curve by a true fractional-octave width."""
    try:
        width_oct = float(smooth_oct)
        if not np.isfinite(width_oct) or width_oct <= 0.0:
            return gain
        filter_smooth = 1.0 / width_oct
        return np.asarray(
            smooth_gain_fractional_octave(f, gain, filter_smooth),
            dtype=float,
        )
    except (TypeError, ValueError, ZeroDivisionError, FloatingPointError):
        return gain
