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

import numpy as np
import scipy.ndimage

logger = logging.getLogger("DecayCore")

from .phase_ir_utils import _mixed_excess_weight, _smoothstep01


def _cfg_value(cfg, key: str, default):
    try:
        if isinstance(cfg, dict):
            return cfg.get(key, default)
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
        logger.exception("cfg dict value read")
    try:
        return getattr(cfg, key, default)
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
        logger.exception("cfg attr value read")
        return default


def _collect_phase_anchor_hz(cfg) -> tuple[float, ...]:
    anchors: list[float] = []
    try:
        for xo in list(_cfg_value(cfg, "crossovers", []) or []):
            try:
                fc = float(xo.get("freq", 0.0) or 0.0)
            except (AttributeError, TypeError, ValueError):
                continue
            if np.isfinite(fc) and fc > 0.0:
                anchors.append(float(fc))
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
        logger.exception("crossover anchor collect")
    try:
        hs = _cfg_value(cfg, "hpf_settings", None)
        if isinstance(hs, dict) and bool(hs.get("enabled", False)):
            fc = float(hs.get("freq", 0.0) or 0.0)
            if np.isfinite(fc) and fc > 0.0:
                anchors.append(float(fc))
    except (TypeError, ValueError):
        pass

    out: list[float] = []
    for fc in sorted(anchors):
        if out and abs(float(fc) - float(out[-1])) < 1e-6:
            continue
        out.append(float(fc))
    return tuple(out)


def phase_region_profiles(freq_axis: np.ndarray, phase_lim_hz: float, cfg) -> dict[str, np.ndarray | tuple[float, ...]]:
    f = np.asarray(freq_axis, dtype=float)
    zeros = np.zeros_like(f, dtype=float)
    if f.size == 0:
        return {
            "lf": zeros,
            "xo": zeros,
            "hf": zeros,
            "audible": zeros,
            "anchors_hz": tuple(),
        }

    try:
        f_lim = float(phase_lim_hz)
    except (TypeError, ValueError, OverflowError):
        f_lim = 0.0
    if not np.isfinite(f_lim) or f_lim <= 20.0:
        f_lim = 20.0

    lf_full = float(np.clip(0.40 * f_lim, 45.0, 110.0))
    lf_fade = float(np.clip(max(lf_full + 35.0, 0.78 * f_lim), lf_full + 1.0, max(f_lim, lf_full + 1.0)))
    lf = _mixed_excess_weight(f, lf_full, lf_fade)

    xo = np.zeros_like(f, dtype=float)
    anchors_hz = _collect_phase_anchor_hz(cfg)
    for fc in anchors_hz:
        if not (np.isfinite(fc) and fc > 0.0):
            continue
        log_dist = np.abs(np.log2(np.maximum(f, 1e-9) / float(fc)))
        width_oct = 0.45 if float(fc) >= 120.0 else 0.38
        xo = np.maximum(xo, np.exp(-0.5 * (log_dist / max(width_oct, 1e-6)) ** 2.0))

    hf_start = float(np.clip(max(110.0, min(0.55 * f_lim, f_lim - 30.0)), 90.0, max(f_lim - 1.0, 90.0)))
    hf_end = float(max(hf_start + 1.0, f_lim))
    hf_x = np.clip((f - hf_start) / max(hf_end - hf_start, 1e-9), 0.0, 1.0)
    hf = _smoothstep01(hf_x)
    hf = np.where(f <= hf_start, 0.0, hf)
    hf = np.where(f >= hf_end, 1.0, hf)

    audible = np.clip(0.70 * lf + 0.55 * xo + 0.20 * (1.0 - hf), 0.0, 1.0)
    return {
        "lf": np.asarray(lf, dtype=float),
        "xo": np.asarray(xo, dtype=float),
        "hf": np.asarray(hf, dtype=float),
        "audible": np.asarray(audible, dtype=float),
        "anchors_hz": tuple(float(v) for v in anchors_hz),
    }


def phase_confidence_profile(
    freq_axis: np.ndarray,
    confidence_mask,
    phase_lim_hz: float,
    cfg,
    *,
    bassfirst: bool = False,
    afdw_on: bool = False,
) -> np.ndarray:
    f = np.asarray(freq_axis, dtype=float)
    if f.size == 0:
        return np.asarray([], dtype=float)
    try:
        conf = np.asarray(confidence_mask, dtype=float).reshape(-1)
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
        conf = np.asarray([], dtype=float)
    if conf.size != f.size:
        conf = np.ones_like(f, dtype=float)
    conf = np.clip(conf, 0.0, 1.0)
    if conf.size >= 8:
        conf = scipy.ndimage.gaussian_filter1d(conf, sigma=2.0, mode="nearest")
    conf = np.clip(conf, 0.0, 1.0)

    regions = phase_region_profiles(f, phase_lim_hz, cfg)
    lf = np.asarray(regions["lf"], dtype=float)
    xo = np.asarray(regions["xo"], dtype=float)
    hf = np.asarray(regions["hf"], dtype=float)

    lf_floor = 0.62 + (0.08 if bool(bassfirst) else 0.0) + (0.04 if bool(afdw_on) else 0.0)
    lf_floor = float(np.clip(lf_floor, 0.45, 0.82))
    xo_lift = 0.16 + (0.03 if bool(afdw_on) else 0.0)

    out = np.maximum(conf, lf_floor * lf)
    out = np.clip(out + xo_lift * xo * (0.55 + 0.45 * np.maximum(conf, lf)), 0.0, 1.0)
    out = np.clip(out * (1.0 - 0.55 * hf) + 0.10 * lf, 0.05, 1.0)
    if out.size >= 8:
        out = scipy.ndimage.gaussian_filter1d(out, sigma=1.0, mode="nearest")
    return np.clip(np.asarray(out, dtype=float), 0.05, 1.0)


def apply_mixed_excess_mask(freq_axis, excess, cfg, st) -> np.ndarray:
    f = np.asarray(freq_axis, dtype=float)
    x = np.asarray(excess, dtype=float)
    phase_lim_hz = float(getattr(cfg, "phase_limit", 1000.0) or 1000.0)
    full_hz = float(getattr(cfg, "low_freq_full_correction_hz", getattr(cfg, "mixed_split_freq", 300.0)) or 300.0)
    none_hz = float(getattr(cfg, "high_freq_no_correction_hz", phase_lim_hz) or phase_lim_hz)
    if phase_lim_hz > 0.0:
        none_hz = min(none_hz, phase_lim_hz)
    if none_hz <= (full_hz + 1.0):
        none_hz = full_hz + 1.0
    w = _mixed_excess_weight(f, full_hz, none_hz)
    if phase_lim_hz > 0.0:
        w *= ((f > 0) & (f <= phase_lim_hz)).astype(float)
    strength = float(np.clip(float(getattr(cfg, "excess_phase_strength", 0.9) or 0.0), 0.0, 1.0))
    w *= strength
    try:
        if isinstance(st, dict):
            st["mixed_phase_strength"] = float(strength)
            st["mixed_phase_full_correction_hz"] = float(full_hz)
            st["mixed_phase_no_correction_hz"] = float(none_hz)
    except (TypeError, ValueError):
        pass
    return x * w


def linear_excess_weight(freq_axis: np.ndarray, phase_lim_hz: float) -> np.ndarray:
    f = np.asarray(freq_axis, dtype=float)
    w = np.zeros_like(f, dtype=float)
    if f.size == 0:
        return w
    try:
        f_lim = float(phase_lim_hz)
    except (TypeError, ValueError, OverflowError):
        f_lim = 0.0
    if (not np.isfinite(f_lim)) or (f_lim <= 20.0):
        return w
    f0 = 20.0
    f2 = float(max(f0 + 1.0, f_lim))
    f1_hi = max(81.0, 0.88 * f2)
    f1 = float(np.clip(0.55 * f2, 80.0, f1_hi))
    if f2 <= (f1 + 1.0):
        f2 = f1 + 1.0
    w0 = 0.30
    w1 = 0.16
    band = np.isfinite(f) & (f > 0.0) & (f <= f2)
    if not np.any(band):
        return w
    ff = f[band]
    ww = np.empty_like(ff, dtype=float)
    seg1 = ff <= f1
    if np.any(seg1):
        ww[seg1] = w0 + (w1 - w0) * _smoothstep01((ff[seg1] - f0) / (f1 - f0))
    seg2 = ~seg1
    if np.any(seg2):
        ww[seg2] = w1 * (1.0 - _smoothstep01((ff[seg2] - f1) / (f2 - f1)))
    w[band] = np.clip(ww, 0.0, 1.0)
    return w


def smooth_linear_boundary(freq_axis: np.ndarray, extra_phase: np.ndarray, phase_lim_hz: float, cfg, st) -> np.ndarray:
    f = np.asarray(freq_axis, dtype=float)
    x = np.asarray(extra_phase, dtype=float)
    if f.size < 16 or x.size != f.size:
        return x
    try:
        f_lim = float(phase_lim_hz)
    except (TypeError, ValueError, OverflowError):
        f_lim = 0.0
    if (not np.isfinite(f_lim)) or (f_lim <= 30.0):
        return x
    try:
        sigma_bins = float(getattr(cfg, "phase_boundary_smooth_sigma_bins", 1.2) or 1.2)
    except (AttributeError, TypeError, ValueError):
        sigma_bins = 1.2
    sigma_bins = float(np.clip(sigma_bins if np.isfinite(sigma_bins) else 1.2, 0.0, 6.0))
    if sigma_bins <= 1e-6:
        return x
    f_start = float(max(30.0, 0.70 * f_lim))
    f_end = float(f_lim)
    if f_end <= (f_start + 1.0):
        return x
    y = scipy.ndimage.gaussian_filter1d(x, sigma=sigma_bins, mode="nearest")
    out = (1.0 - _smoothstep01(np.clip((f - f_start) / (f_end - f_start + 1e-12), 0.0, 1.0))) * x + _smoothstep01(np.clip((f - f_start) / (f_end - f_start + 1e-12), 0.0, 1.0)) * y
    try:
        if isinstance(st, dict):
            st["phase_boundary_smooth_enabled"] = True
            st["phase_boundary_smooth_sigma_bins"] = float(sigma_bins)
            st["phase_boundary_smooth_start_hz"] = float(f_start)
            st["phase_boundary_smooth_end_hz"] = float(f_end)
    except (TypeError, ValueError):
        pass
    return out


def _phase_tail_limit_hz(phase_lim_hz: float) -> float:
    try:
        f_lim = float(phase_lim_hz)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if (not np.isfinite(f_lim)) or (f_lim <= 30.0):
        return 0.0
    return float(f_lim)


def _phase_tail_is_enabled(cfg) -> bool:
    try:
        return bool(getattr(cfg, "phase_tail_monotonic_enable", True))
    except (AttributeError, TypeError, ValueError):
        return True


def _phase_tail_start_ratio(cfg) -> float:
    try:
        value = float(getattr(cfg, "phase_tail_start_ratio", 0.72) or 0.72)
    except (AttributeError, TypeError, ValueError):
        value = 0.72
    return float(np.clip(value if np.isfinite(value) else 0.72, 0.50, 0.92))


def _phase_tail_sigma_abs(cfg) -> float:
    try:
        value = float(getattr(cfg, "phase_tail_abs_smooth_sigma_bins", 2.5) or 2.5)
    except (AttributeError, TypeError, ValueError):
        value = 2.5
    return float(np.clip(value if np.isfinite(value) else 2.5, 0.0, 8.0))


def _phase_tail_cosine_strength(cfg) -> float:
    try:
        value = float(getattr(cfg, "phase_tail_cosine_strength", 0.85) or 0.85)
    except (AttributeError, TypeError, ValueError):
        value = 0.85
    return float(np.clip(value if np.isfinite(value) else 0.85, 0.0, 1.0))


def _phase_tail_profile(x_tail: np.ndarray, *, sigma_abs: float, cosine_strength: float) -> tuple[np.ndarray, float]:
    abs_tail = np.abs(np.asarray(x_tail, dtype=float))
    if sigma_abs > 1e-9:
        abs_tail = scipy.ndimage.gaussian_filter1d(abs_tail, sigma=float(sigma_abs), mode="nearest")
    mono = np.minimum.accumulate(abs_tail)
    if mono.size >= 2 and cosine_strength > 1e-6:
        t = np.linspace(0.0, 1.0, mono.size, endpoint=True, dtype=float)
        cos_env = float(max(mono[0], 0.0)) * (0.5 + 0.5 * np.cos(np.pi * t))
        mono = np.maximum(
            (1.0 - float(cosine_strength)) * mono + float(cosine_strength) * np.minimum(mono, cos_env),
            0.0,
        )
    head_n = int(max(3, mono.size // 6))
    sign0 = float(np.sign(np.median(x_tail[:head_n]))) or float(np.sign(x_tail[0])) or 1.0
    fade = np.clip(
        0.5 + 0.5 * np.cos(np.pi * np.linspace(0.0, 1.0, mono.size, endpoint=True, dtype=float)),
        0.0,
        1.0,
    )
    return np.asarray(mono * fade, dtype=float), float(sign0)


def _phase_tail_write_stats(st, *, f_start: float, f_lim: float, sigma_abs: float, f_start_ratio: float, cosine_strength: float) -> None:
    try:
        if isinstance(st, dict):
            st["phase_tail_monotonic_enabled"] = True
            st["phase_tail_monotonic_start_hz"] = float(f_start)
            st["phase_tail_monotonic_end_hz"] = float(f_lim)
            st["phase_tail_monotonic_sigma_abs_bins"] = float(sigma_abs)
            st["phase_tail_monotonic_start_ratio"] = float(f_start_ratio)
            st["phase_tail_cosine_strength"] = float(cosine_strength)
    except (TypeError, ValueError):
        pass


def enforce_linear_tail_decay(freq_axis: np.ndarray, extra_phase: np.ndarray, phase_lim_hz: float, cfg, st) -> np.ndarray:
    f = np.asarray(freq_axis, dtype=float)
    x = np.asarray(extra_phase, dtype=float)
    if f.size < 16 or x.size != f.size:
        return x
    f_lim = _phase_tail_limit_hz(phase_lim_hz)
    if f_lim <= 0.0:
        return x
    if not _phase_tail_is_enabled(cfg):
        return x
    f_start_ratio = _phase_tail_start_ratio(cfg)
    f_start = float(max(30.0, f_start_ratio * f_lim))
    idx = np.flatnonzero(np.isfinite(f) & (f >= f_start) & (f <= f_lim))
    if idx.size < 8:
        return x
    sigma_abs = _phase_tail_sigma_abs(cfg)
    cosine_strength = _phase_tail_cosine_strength(cfg)
    out = x.copy()
    mono, sign0 = _phase_tail_profile(
        np.asarray(out[idx], dtype=float),
        sigma_abs=float(sigma_abs),
        cosine_strength=float(cosine_strength),
    )
    out[idx] = float(sign0) * np.asarray(mono, dtype=float)
    _phase_tail_write_stats(
        st,
        f_start=float(f_start),
        f_lim=float(f_lim),
        sigma_abs=float(sigma_abs),
        f_start_ratio=float(f_start_ratio),
        cosine_strength=float(cosine_strength),
    )
    return out


def linear_to_minphase_blend_mask(freq_axis: np.ndarray, phase_lim_hz: float, cfg, st) -> np.ndarray:
    f = np.asarray(freq_axis, dtype=float)
    m = np.zeros_like(f, dtype=float)
    if f.size == 0:
        return m
    try:
        f_end = float(phase_lim_hz)
    except (TypeError, ValueError, OverflowError):
        f_end = 0.0
    if (not np.isfinite(f_end)) or (f_end <= 20.0):
        return m
    try:
        start_ratio = float(getattr(cfg, "linear_phase_blend_start_ratio", 0.65) or 0.65)
    except (AttributeError, TypeError, ValueError):
        start_ratio = 0.65
    start_ratio = float(np.clip(start_ratio if np.isfinite(start_ratio) else 0.65, 0.25, 0.95))
    f_start = float(max(20.0, start_ratio * f_end))
    if f_end <= (f_start + 1.0):
        return np.where(f >= f_end, 1.0, 0.0).astype(float)
    x = np.clip((f - f_start) / (f_end - f_start + 1e-12), 0.0, 1.0)
    m = _smoothstep01(x)
    m = np.where(f <= f_start, 0.0, m)
    m = np.where(f >= f_end, 1.0, m)
    try:
        if isinstance(st, dict):
            st["linear_phase_blend_start_hz"] = float(f_start)
            st["linear_phase_blend_end_hz"] = float(f_end)
            st["linear_phase_blend_start_ratio"] = float(start_ratio)
    except (TypeError, ValueError):
        pass
    return m


def merge_minphase_and_excess(min_u, excess_masked) -> np.ndarray:
    return np.asarray(min_u, dtype=float) + np.asarray(excess_masked, dtype=float)
