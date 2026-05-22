# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Per-run compute context for automatic mode.

One instance lives for the duration of a single auto mode run. It must not be
shared across runs or stored globally. All caches are keyed so that different
sample rates, tap counts, filter types, measurement signatures, or smoothing
settings produce distinct entries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class AutoRunComputeContext:
    fs: int
    taps: int
    measurement_signature: str
    filter_key: str
    compat_version: str

    freq_grids: dict[tuple, np.ndarray] = field(default_factory=dict)
    interpolations: dict[tuple, np.ndarray] = field(default_factory=dict)
    targets: dict[tuple, np.ndarray] = field(default_factory=dict)
    metric_cache: dict[tuple, dict[str, Any]] = field(default_factory=dict)
    trial_cache: dict[tuple, Any] = field(default_factory=dict)
    debug_stats: dict[str, int] = field(default_factory=dict)

    def bump(self, key: str) -> None:
        self.debug_stats[key] = self.debug_stats.get(key, 0) + 1

    def begin_trial(self, trial_number: int) -> None:
        self.trial_cache.clear()
        self.trial_cache[("trial_number",)] = int(trial_number)
        self.bump("trials_started")

    def cache_stats_text(self) -> str:
        s = self.debug_stats
        lines = [
            f"  freq grid:     {s.get('freq_grid_hits', 0)} hits / {s.get('freq_grid_misses', 0)} misses",
            f"  interp:        {s.get('interp_hits', 0)} hits / {s.get('interp_misses', 0)} misses",
            f"  meas_fixed:    {s.get('meas_fixed_hits', 0)} hits / {s.get('meas_fixed_misses', 0)} misses",
            f"  target:        {s.get('target_hits', 0)} hits / {s.get('target_misses', 0)} misses",
            f"  metric:        {s.get('metric_hits', 0)} hits / {s.get('metric_misses', 0)} misses",
            f"  trials:        {s.get('trials_started', 0)}",
        ]
        return "\n".join(lines)


def get_rfft_freq_grid(ctx: AutoRunComputeContext, n_fft: int) -> np.ndarray:
    """Return rfftfreq grid for (ctx.fs, n_fft), computing once per run."""
    key = ("rfft", ctx.fs, n_fft)
    cached = ctx.freq_grids.get(key)
    if cached is not None:
        ctx.bump("freq_grid_hits")
        return cached
    grid = np.fft.rfftfreq(n_fft, 1.0 / ctx.fs)
    ctx.freq_grids[key] = grid
    ctx.bump("freq_grid_misses")
    return grid


def cached_interp(
    ctx: AutoRunComputeContext,
    name: str,
    x_new: np.ndarray,
    x_old: np.ndarray,
    y_old: np.ndarray,
    left: float | None = None,
    right: float | None = None,
) -> np.ndarray:
    """Return np.interp result, cached by object identity + shape.

    Safe because the ctx is per-run and input arrays are stable within a run.
    Do not use across runs or when input arrays may be reallocated.
    """
    key = (
        "interp",
        name,
        id(x_new),
        id(x_old),
        id(y_old),
        x_new.shape,
        x_old.shape,
        y_old.shape,
        left,
        right,
    )
    cached = ctx.interpolations.get(key)
    if cached is not None:
        ctx.bump("interp_hits")
        return cached
    result = np.interp(x_new, x_old, y_old, left=left, right=right)
    ctx.interpolations[key] = result
    ctx.bump("interp_misses")
    return result


def get_cached_target(
    ctx: AutoRunComputeContext,
    *,
    target_key: tuple,
    build_fn,
) -> np.ndarray:
    """Return cached target array, building it once for a given key.

    target_key must include all parameters that affect the target output:
    mode, preset name, fs, taps, freq-grid key, measurement_signature,
    filter_key, correction range, leveling mode, HPF/XO metadata, compat_version.
    """
    cached = ctx.targets.get(target_key)
    if cached is not None:
        ctx.bump("target_hits")
        return cached
    target = build_fn()
    ctx.targets[target_key] = target
    ctx.bump("target_misses")
    return target


def get_cached_metric(
    ctx: AutoRunComputeContext,
    *,
    metric_name: str,
    key: tuple,
    compute_fn,
) -> dict:
    """Return cached metric dict, computing it once for a given key.

    key must include all inputs that affect the metric: channel, fs, taps,
    filter type, target id, response array object id and shape, scoring version.
    Do not use keys so broad that they could match metrics from different filters
    or targets.
    """
    full_key = ("metric", metric_name, key)
    cached = ctx.metric_cache.get(full_key)
    if cached is not None:
        ctx.bump(f"{metric_name}_hits")
        ctx.bump("metric_hits")
        return cached
    result = compute_fn()
    ctx.metric_cache[full_key] = result
    ctx.bump(f"{metric_name}_misses")
    ctx.bump("metric_misses")
    return result
