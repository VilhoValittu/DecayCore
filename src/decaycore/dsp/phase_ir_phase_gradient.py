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

import logging
from typing import Any

import numpy as np

_logger = logging.getLogger(__name__)


def gd_grad_metrics(freq_axis: np.ndarray, phase_rad: np.ndarray, *, mask: np.ndarray | None = None) -> dict[str, Any]:
    out = {
        "max_ms_per_oct": 0.0,
        "at_hz": None,
        "used_x_axis": "log2(f)",
        "df_min": None,
        "df_max": None,
        "phase_wrapped": False,
        "units_note": "gd_ms=-dphi/d(2*pi*f)*1e3; gd_grad=np.gradient(gd_ms, log2(f))",
    }
    try:
        f = np.asarray(freq_axis, dtype=float)
        p_raw = np.asarray(phase_rad, dtype=float)
        if f.size < 16 or p_raw.size != f.size:
            return out
        sel = np.isfinite(f) & np.isfinite(p_raw) & (f > 0.0)
        if mask is not None:
            sel &= np.asarray(mask, dtype=bool)
        if np.count_nonzero(sel) < 4:
            return out
        ff = f[sel]
        pp = np.unwrap(p_raw[sel])
        if ff.size < 4 or not np.all(np.diff(ff) > 0.0):
            return out
        dff = np.diff(ff)
        dff = dff[np.isfinite(dff) & (dff > 0.0)]
        if dff.size:
            out["df_min"] = float(np.min(dff))
            out["df_max"] = float(np.max(dff))
        gd_ms = np.nan_to_num((-np.gradient(pp, 2.0 * np.pi * ff)) * 1000.0, nan=0.0, posinf=0.0, neginf=0.0)
        gd_grad = np.nan_to_num(np.gradient(gd_ms, np.log2(np.maximum(ff, 1e-9))), nan=0.0, posinf=0.0, neginf=0.0)
        if gd_grad.size == 0:
            return out
        idx = int(np.argmax(np.abs(gd_grad)))
        out["max_ms_per_oct"] = float(np.max(np.abs(gd_grad)))
        out["at_hz"] = float(ff[idx]) if ff.size else None
        return out
    except (TypeError, ValueError, FloatingPointError, IndexError):
        return out


def max_abs_gd_gradient_ms_per_oct(freq_axis: np.ndarray, phase_rad: np.ndarray, *, mask: np.ndarray | None = None) -> tuple[float, float | None]:
    m = gd_grad_metrics(freq_axis, phase_rad, mask=mask)
    return float(m["max_ms_per_oct"]), m["at_hz"]


def _gd_zone_limit_ms_per_oct(base_limit: float, peak_hz: float | None) -> float:
    """Adjust GD gradient limit by frequency zone of worst gradient.

    Bass modes (<120 Hz) need more room (looser limit); HF ripple (>500 Hz)
    should be tighter. Returns base_limit when peak_hz is unknown.
    """
    if peak_hz is None or not (np.isfinite(peak_hz) and peak_hz > 0.0):
        return float(base_limit)
    if peak_hz < 120.0:
        return float(base_limit * (50.0 / 30.0))
    if peak_hz > 500.0:
        return float(base_limit * (15.0 / 30.0))
    if peak_hz <= 300.0:
        t = (peak_hz - 120.0) / 180.0
        return float(base_limit * ((50.0 + t * (30.0 - 50.0)) / 30.0))
    t = (peak_hz - 300.0) / 200.0
    return float(base_limit * ((30.0 + t * (15.0 - 30.0)) / 30.0))


def _gd_grad_limiter_info(cfg, lim_cfg: float) -> dict[str, Any]:
    info = {
        "enabled": False,
        "applied": False,
        "limit_ms_per_oct": None,
        "reason": "limit<=0",
        "max_grad_before_ms_per_oct": None,
        "max_grad_after_ms_per_oct": None,
        "max_grad_before_hz": None,
        "max_grad_after_hz": None,
        "used_x_axis": "log2(f)",
        "df_min": None,
        "df_max": None,
        "phase_wrapped": False,
        "units_note": "gd_ms=-dphi/d(2*pi*f)*1e3; gd_grad=np.gradient(gd_ms, log2(f))",
        "limit_input": None,
    }
    info["limit_input"] = float(lim_cfg)
    info["limit_ms_per_oct"] = float(lim_cfg) if lim_cfg > 0.0 else 0.0
    info["enabled"] = bool(lim_cfg > 0.0)
    info["reason"] = "applied" if info["enabled"] else "limit<=0"
    return info


def _gd_grad_limiter_measure(freq_axis, phase, phase_mask, info: dict[str, Any], *, before: bool) -> None:
    try:
        metrics = gd_grad_metrics(np.asarray(freq_axis, dtype=float), phase, mask=phase_mask)
    except (TypeError, ValueError, FloatingPointError, IndexError):
        return
    prefix = "before" if bool(before) else "after"
    info[f"max_grad_{prefix}_ms_per_oct"] = float(metrics.get("max_ms_per_oct", 0.0) or 0.0)
    info[f"max_grad_{prefix}_hz"] = metrics.get("at_hz", None)
    if bool(before):
        info["used_x_axis"] = str(metrics.get("used_x_axis", "log2(f)") or "log2(f)")
        info["df_min"] = metrics.get("df_min", None)
        info["df_max"] = metrics.get("df_max", None)
        info["phase_wrapped"] = bool(metrics.get("phase_wrapped", False))
        info["units_note"] = str(metrics.get("units_note", info["units_note"]) or info["units_note"])


def _gd_grad_limiter_apply(
    freq_axis,
    in_phase: np.ndarray,
    phase_mask,
    limiter_fn,
    cfg,
    lim_cfg: float,
    info: dict[str, Any],
) -> np.ndarray:
    if not (info["enabled"] and limiter_fn is not None and freq_axis is not None):
        return in_phase
    try:
        f_arr = np.asarray(freq_axis, dtype=float)
        valid_f = np.isfinite(f_arr) & (f_arr > 0.0)
        if phase_mask is not None:
            valid_f &= np.asarray(phase_mask, dtype=bool)
        if np.count_nonzero(valid_f) >= 4:
            f_lo = float(np.min(f_arr[valid_f]))
            f_hi = float(np.max(f_arr[valid_f]))
        else:
            base = f_arr[np.isfinite(f_arr) & (f_arr > 0.0)]
            f_lo = float(np.min(base))
            f_hi = float(np.max(base))
        try:
            gd_sigma = float(getattr(cfg, "gd_grad_smooth_sigma", 0.8) or 0.8)
        except (AttributeError, TypeError, ValueError):
            gd_sigma = 0.8
        gd_sigma = gd_sigma if np.isfinite(gd_sigma) else 0.8
        lim_effective = _gd_zone_limit_ms_per_oct(lim_cfg, info.get("max_grad_before_hz"))
        info["limit_ms_per_oct"] = float(lim_effective)
        out = limiter_fn(
            f_arr,
            in_phase,
            mask=phase_mask,
            max_grad_ms_per_oct=float(lim_effective),
            f_min=float(f_lo),
            f_max=float(f_hi),
            grad_smooth_sigma=float(max(0.0, gd_sigma)),
            soft_limit=True,
        )
        info["applied"] = True
        return out
    except (TypeError, ValueError, FloatingPointError, IndexError):
        info["enabled"] = False
        info["reason"] = "missing data"
        return in_phase


def _gd_grad_limiter_revert_if_needed(in_phase: np.ndarray, out: np.ndarray, info: dict[str, Any]) -> np.ndarray:
    try:
        gb = info["max_grad_before_ms_per_oct"]
        ga = info["max_grad_after_ms_per_oct"]
        if info.get("applied", False) and gb is not None and ga is not None and np.isfinite(float(gb)) and np.isfinite(float(ga)) and (float(ga) > (float(gb) * 1.001)):
            info["applied"] = False
            info["enabled"] = False
            info["reason"] = "reverted_non_monotonic"
            info["reverted_non_monotonic"] = True
            info["max_grad_after_ms_per_oct"] = float(gb)
            info["max_grad_after_hz"] = info["max_grad_before_hz"]
            _logger.warning(
                "gd_grad_limiter: reverted — GD gradient increased after limiting "
                "(before=%.2f ms/oct → after=%.2f ms/oct); phase correction discarded",
                float(gb),
                float(ga),
            )
            return in_phase.copy()
    except (TypeError, ValueError, FloatingPointError):
        pass
    return out


def gd_grad_limiter(ir, cfg, st, *, freq_axis=None, phase_mask=None, limiter_fn=None) -> tuple[np.ndarray, dict[str, Any]]:
    in_phase = np.asarray(ir, dtype=float).copy()
    out = in_phase.copy()
    try:
        lim_cfg = float(getattr(cfg, "gd_grad_limit_ms_per_oct", getattr(cfg, "gd_limiter_limit_ms_per_oct", 30.0)) or 0.0)
    except (AttributeError, TypeError, ValueError):
        lim_cfg = 0.0
    lim_cfg = float(max(0.0, lim_cfg if np.isfinite(lim_cfg) else 0.0))
    info = _gd_grad_limiter_info(cfg, lim_cfg)
    if freq_axis is not None:
        _gd_grad_limiter_measure(freq_axis, in_phase, phase_mask, info, before=True)
    out = _gd_grad_limiter_apply(freq_axis, in_phase, phase_mask, limiter_fn, cfg, lim_cfg, info)
    if info["enabled"] and limiter_fn is None and freq_axis is not None:
        if info["max_grad_before_ms_per_oct"] is not None and info["limit_ms_per_oct"] is not None:
            if float(info["max_grad_before_ms_per_oct"]) > float(info["limit_ms_per_oct"]):
                _logger.warning(
                    "gd_grad_limiter: limiter_fn missing but GD gradient=%.2f ms/oct exceeds limit=%.2f ms/oct",
                    float(info["max_grad_before_ms_per_oct"]),
                    float(info["limit_ms_per_oct"]),
                )
        info["enabled"] = False
        info["reason"] = "missing data"
    if freq_axis is not None:
        _gd_grad_limiter_measure(freq_axis, out, phase_mask, info, before=False)
    elif info["max_grad_before_ms_per_oct"] is not None:
        info["max_grad_after_ms_per_oct"] = info["max_grad_before_ms_per_oct"]
        info["max_grad_after_hz"] = info["max_grad_before_hz"]
    out = _gd_grad_limiter_revert_if_needed(in_phase, out, info)
    try:
        if isinstance(st, dict):
            st["gd_limiter_enabled"] = bool(info["enabled"])
            st["gd_limiter_limit_ms_per_oct"] = info["limit_ms_per_oct"]
            st["gd_limiter_reason"] = str(info["reason"])
            st["gd_limiter_applied"] = bool(info["applied"])
            st["gd_limiter_max_grad_before_ms_per_oct"] = info["max_grad_before_ms_per_oct"]
            st["gd_limiter_max_grad_after_ms_per_oct"] = info["max_grad_after_ms_per_oct"]
            st["gd_limiter_max_grad_ms_per_oct"] = info["max_grad_after_ms_per_oct"]
            st["gd_limiter_max_grad_before_hz"] = info["max_grad_before_hz"]
            st["gd_limiter_max_grad_after_hz"] = info["max_grad_after_hz"]
            st["gd_limiter_max_grad_hz"] = info["max_grad_after_hz"]
            st["gd_grad_limiter_enabled"] = bool(info["enabled"])
            st["gd_grad_limit_ms_per_oct"] = info["limit_ms_per_oct"]
            st["gd_grad_limiter_reason"] = str(info["reason"])
            st["gd_grad_limiter_applied"] = bool(info["applied"])
            st["gd_grad_limiter_max_grad_before_ms_per_oct"] = info["max_grad_before_ms_per_oct"]
            st["gd_grad_limiter_max_grad_after_ms_per_oct"] = info["max_grad_after_ms_per_oct"]
            st["gd_grad_limiter_max_grad_ms_per_oct"] = info["max_grad_after_ms_per_oct"]
            st["gd_grad_limiter_max_grad_before_hz"] = info["max_grad_before_hz"]
            st["gd_grad_limiter_max_grad_after_hz"] = info["max_grad_after_hz"]
            st["gd_grad_limiter_max_grad_hz"] = info["max_grad_after_hz"]
            st["gd_grad_limiter_limit_ms_per_oct"] = float(info["limit_ms_per_oct"] or 0.0)
            st["gd_grad_limiter_before_max_ms_per_oct"] = None if info["max_grad_before_ms_per_oct"] is None else float(info["max_grad_before_ms_per_oct"])
            st["gd_grad_limiter_after_max_ms_per_oct"] = None if info["max_grad_after_ms_per_oct"] is None else float(info["max_grad_after_ms_per_oct"])
            st["gd_grad_limiter_peak_hz"] = None if info["max_grad_after_hz"] is None else float(info["max_grad_after_hz"])
            st["gd_grad_before_max_ms_per_oct"] = None if info["max_grad_before_ms_per_oct"] is None else float(info["max_grad_before_ms_per_oct"])
            st["gd_grad_before_at_hz"] = None if info["max_grad_before_hz"] is None else float(info["max_grad_before_hz"])
            st["gd_grad_after_max_ms_per_oct"] = None if info["max_grad_after_ms_per_oct"] is None else float(info["max_grad_after_ms_per_oct"])
            st["gd_grad_after_at_hz"] = None if info["max_grad_after_hz"] is None else float(info["max_grad_after_hz"])
            st["gd_grad_used_x_axis"] = str(info.get("used_x_axis", "log2(f)") or "log2(f)")
            st["gd_grad_df_min"] = None if info.get("df_min", None) is None else float(info["df_min"])
            st["gd_grad_df_max"] = None if info.get("df_max", None) is None else float(info["df_max"])
            st["gd_grad_phase_wrapped"] = bool(info.get("phase_wrapped", False))
            st["gd_grad_units_note"] = str(info.get("units_note", ""))
            st["gd_grad_limit_input"] = None if info.get("limit_input", None) is None else float(info["limit_input"])
            st["gd_grad_limiter_reverted_non_monotonic"] = bool(info.get("reverted_non_monotonic", False))
    except (TypeError, ValueError):
        pass
    return out, info
