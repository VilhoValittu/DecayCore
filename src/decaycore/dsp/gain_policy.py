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
from typing import Any, Callable

import numpy as np

from .dsp_config import CfgReader


@dataclass(frozen=True)
class GainPolicy:
    max_cut_db: float
    max_boost_db: float
    low_cut_enable: bool
    low_cut_hz: float
    low_cut_strength: float
    exc_prot: bool
    exc_freq: float
    exc_soft_hz: float


def resolve_gain_policy(
    cfg: Any,
    *,
    cfg_float_allow_zero_fn: Callable[[Any, str, float], float] | None = None,
) -> GainPolicy:
    reader = CfgReader(cfg)
    max_cut_db = abs(reader.float("max_cut_db", 15.0))
    max_boost_db = reader.float_allow_zero("max_boost_db", 0.0)

    low_cut_enable = reader.bool("low_bass_cut_enable", True)
    low_cut_hz = reader.float_allow_zero("low_bass_cut_hz", 0.0)
    low_cut_strength = reader.float_allow_zero("low_bass_cut_strength", 0.0)
    if not np.isfinite(low_cut_strength):
        low_cut_strength = 0.0
    low_cut_strength = float(np.clip(low_cut_strength, 0.0, 1.0))

    exc_prot = reader.bool("exc_prot", False)
    exc_freq = reader.float_allow_zero("exc_freq", 0.0)
    if not np.isfinite(exc_freq) or exc_freq <= 0.0:
        exc_freq = 0.0
    exc_soft_hz = float(exc_freq * 1.41) if exc_freq > 0.0 else 0.0
    if exc_freq <= 0.0:
        # An invalid/zero excursion frequency makes the feature meaningless;
        # keep exc_prot consistent with exc_freq/exc_soft_hz instead of
        # leaving it "on" with a degenerate (zero-width) protection band.
        exc_prot = False

    return GainPolicy(
        max_cut_db=float(max_cut_db),
        max_boost_db=float(max_boost_db),
        low_cut_enable=bool(low_cut_enable),
        low_cut_hz=float(low_cut_hz) if np.isfinite(low_cut_hz) else 0.0,
        low_cut_strength=float(low_cut_strength),
        exc_prot=bool(exc_prot),
        exc_freq=float(exc_freq),
        exc_soft_hz=float(exc_soft_hz),
    )


def build_low_frequency_guard_mask(
    freq_axis: np.ndarray,
    policy: GainPolicy,
    *,
    include_low_cut: bool = True,
    include_exc_soft: bool = True,
) -> np.ndarray:
    f = np.asarray(freq_axis, dtype=float)
    guard_mask = np.zeros_like(f, dtype=bool)

    if include_low_cut and policy.low_cut_enable and np.isfinite(policy.low_cut_hz) and policy.low_cut_hz > 0.0:
        guard_mask |= (f > 0.0) & (f <= float(policy.low_cut_hz))
    if include_exc_soft and policy.exc_prot and np.isfinite(policy.exc_soft_hz) and policy.exc_soft_hz > 0.0:
        guard_mask |= (f > 0.0) & (f <= float(policy.exc_soft_hz))

    return guard_mask


def _resolve_gain_cap_array(
    cap_values: np.ndarray | None,
    *,
    fallback: np.ndarray,
    clip_hi: float,
) -> np.ndarray | None:
    try:
        if cap_values is None:
            return None
        cap = np.asarray(cap_values, dtype=float)
    except (TypeError, ValueError):
        return None
    if cap.shape != fallback.shape or not np.all(np.isfinite(cap)):
        return None
    return np.clip(cap, 0.0, float(clip_hi))


def _resolve_gain_mask(mask: np.ndarray | None, *, fallback: np.ndarray) -> np.ndarray:
    try:
        if mask is None:
            return np.ones_like(fallback, dtype=bool)
        mm = np.asarray(mask, dtype=bool)
        if mm.shape == fallback.shape:
            return mm
    except (TypeError, ValueError):
        pass
    return np.ones_like(fallback, dtype=bool)


def _apply_boost_cap(
    out: np.ndarray,
    *,
    boost_cap: np.ndarray | None,
    mask: np.ndarray,
    policy: GainPolicy,
) -> None:
    if boost_cap is None:
        out[:] = np.minimum(out, float(policy.max_boost_db))
        return
    if np.any(mask):
        out[mask] = np.minimum(out[mask], boost_cap[mask])
    if np.any(~mask):
        out[~mask] = np.minimum(out[~mask], float(policy.max_boost_db))


def _apply_cut_cap(
    out: np.ndarray,
    *,
    cut_cap: np.ndarray | None,
    mask: np.ndarray,
    policy: GainPolicy,
) -> None:
    if cut_cap is None:
        out[:] = np.maximum(out, -float(policy.max_cut_db))
        return
    if np.any(mask):
        out[mask] = np.maximum(out[mask], -cut_cap[mask])
    if np.any(~mask):
        out[~mask] = np.maximum(out[~mask], -float(policy.max_cut_db))


def clamp_gain_curve(
    curve_db: np.ndarray,
    *,
    policy: GainPolicy,
    boost_cap_db: np.ndarray | None = None,
    cut_cap_db: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    out = np.asarray(curve_db, dtype=float).copy()
    boost_cap = _resolve_gain_cap_array(
        boost_cap_db,
        fallback=out,
        clip_hi=float(np.inf),
    )
    cut_cap = _resolve_gain_cap_array(
        cut_cap_db,
        fallback=out,
        clip_hi=float(policy.max_cut_db),
    )
    m = _resolve_gain_mask(mask, fallback=out)
    _apply_boost_cap(out, boost_cap=boost_cap, mask=m, policy=policy)
    _apply_cut_cap(out, cut_cap=cut_cap, mask=m, policy=policy)
    return out


def apply_cuts_only_guard(
    curve_db: np.ndarray,
    *,
    mask: np.ndarray,
    guard_mask: np.ndarray,
    floor_ref: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    out = np.asarray(curve_db, dtype=float).copy()
    m = np.asarray(mask, dtype=bool)
    g = np.asarray(guard_mask, dtype=bool)
    if out.shape != m.shape or out.shape != g.shape:
        return out, {
            "guard_bins": 0,
            "boost_clamped_bins": 0,
            "floor_reapplied_bins": 0,
        }

    apply_mask = m & g
    if not np.any(apply_mask):
        return out, {
            "guard_bins": 0,
            "boost_clamped_bins": 0,
            "floor_reapplied_bins": 0,
        }

    before = out.copy()
    out[apply_mask] = np.minimum(out[apply_mask], 0.0)

    floor_reapplied_bins = 0
    try:
        if isinstance(floor_ref, np.ndarray) and floor_ref.shape == out.shape:
            floor_vals = np.asarray(floor_ref[apply_mask], dtype=float)
            valid_floor = np.isfinite(floor_vals)
            if np.any(valid_floor):
                cur = np.asarray(out[apply_mask], dtype=float)
                cur_before = cur.copy()
                cur[valid_floor] = np.minimum(cur[valid_floor], floor_vals[valid_floor])
                out[apply_mask] = cur
                floor_reapplied_bins = int(
                    np.count_nonzero(cur_before[valid_floor] > (cur[valid_floor] + 1e-9))
                )
    except (TypeError, ValueError):
        floor_reapplied_bins = 0

    boost_clamped_bins = int(
        np.count_nonzero((before[apply_mask] > 1e-9) & (out[apply_mask] <= 1e-9))
    )
    return out, {
        "guard_bins": int(np.count_nonzero(apply_mask)),
        "boost_clamped_bins": int(boost_clamped_bins),
        "floor_reapplied_bins": int(floor_reapplied_bins),
    }
