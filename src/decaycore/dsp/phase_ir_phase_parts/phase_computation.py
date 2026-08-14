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

from typing import Any

import numpy as np

from ..phase_ir_phase_gradient import (
    gd_grad_limiter as _gd_grad_limiter_impl,
)
from ..phase_ir_phase_gradient import (
    gd_grad_metrics as _gd_grad_metrics_impl,
)
from ..phase_ir_phase_gradient import (
    max_abs_gd_gradient_ms_per_oct as _max_abs_gd_gradient_ms_per_oct_impl,
)
from ..phase_ir_phase_models import (
    merge_minphase_and_excess as _merge_minphase_and_excess_impl,
)
from ..phase_ir_phase_models import (
    phase_confidence_profile as _phase_confidence_profile_impl,
)
from ..phase_ir_phase_models import (
    phase_region_profiles as _phase_region_profiles_impl,
)


def _merge_minphase_and_excess(min_u, excess_masked) -> np.ndarray:
    return _merge_minphase_and_excess_impl(min_u, excess_masked)


def _phase_region_profiles(freq_axis: np.ndarray, phase_lim_hz: float, cfg) -> dict[str, Any]:
    return _phase_region_profiles_impl(freq_axis, phase_lim_hz, cfg)


def _phase_confidence_profile(
    freq_axis: np.ndarray,
    confidence_mask,
    phase_lim_hz: float,
    cfg,
    *,
    bassfirst: bool = False,
    afdw_on: bool = False,
) -> np.ndarray:
    return _phase_confidence_profile_impl(
        freq_axis,
        confidence_mask,
        phase_lim_hz,
        cfg,
        bassfirst=bassfirst,
        afdw_on=afdw_on,
    )


def _max_abs_gd_gradient_ms_per_oct(
    freq_axis: np.ndarray,
    phase_rad: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> tuple[float, float | None]:
    return _max_abs_gd_gradient_ms_per_oct_impl(freq_axis, phase_rad, mask=mask)


def _gd_grad_metrics(
    freq_axis: np.ndarray,
    phase_rad: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    return _gd_grad_metrics_impl(freq_axis, phase_rad, mask=mask)


def _gd_grad_limiter(
    ir,
    cfg,
    st,
    *,
    freq_axis=None,
    phase_mask=None,
    use_bassfirst=False,
    afdw_on=False,
    limiter_fn=None,
) -> tuple[np.ndarray, dict[str, Any]]:
    return _gd_grad_limiter_impl(
        ir,
        cfg,
        st,
        freq_axis=freq_axis,
        phase_mask=phase_mask,
        limiter_fn=limiter_fn,
    )


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if v.size == 0 or w.size != v.size:
        return float("nan")
    sel = np.isfinite(v) & np.isfinite(w) & (w > 1e-12)
    if int(np.count_nonzero(sel)) <= 0:
        return float("nan")
    vv = np.asarray(v[sel], dtype=float)
    ww = np.asarray(w[sel], dtype=float)
    denom = float(np.sum(ww))
    if not np.isfinite(denom) or denom <= 1e-12:
        return float("nan")
    return float(np.sum(vv * ww) / denom)


def _weighted_share(values: np.ndarray, weights: np.ndarray, total_weights: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    tw = np.asarray(total_weights, dtype=float)
    if v.size == 0 or w.size != v.size or tw.size != v.size:
        return float("nan")
    num_sel = np.isfinite(v) & np.isfinite(w) & (w > 1e-12)
    den_sel = np.isfinite(v) & np.isfinite(tw) & (tw > 1e-12)
    if int(np.count_nonzero(num_sel)) <= 0 or int(np.count_nonzero(den_sel)) <= 0:
        return float("nan")
    num = float(np.sum(np.asarray(v[num_sel], dtype=float) * np.asarray(w[num_sel], dtype=float)))
    den = float(np.sum(np.asarray(v[den_sel], dtype=float) * np.asarray(tw[den_sel], dtype=float)))
    if not np.isfinite(den) or den <= 1e-12:
        return float("nan")
    return float(np.clip(num / den, 0.0, 1.0))


__all__ = [
    "_merge_minphase_and_excess",
    "_phase_region_profiles",
    "_phase_confidence_profile",
    "_max_abs_gd_gradient_ms_per_oct",
    "_gd_grad_metrics",
    "_gd_grad_limiter",
    "_weighted_mean",
    "_weighted_share",
]
