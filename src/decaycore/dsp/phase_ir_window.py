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

import math

import numpy as np

from .phase_ir_utils import _ms_value, _resolve_ir_window_mode


def _phase_ir_window_tukey_alpha(cfg, st, win_shape: str) -> tuple[str, float]:
    try:
        tukey_alpha = float(getattr(cfg, "ir_export_tukey_alpha", 0.25))
    except (AttributeError, TypeError, ValueError):
        tukey_alpha = 0.25
    if not np.isfinite(tukey_alpha):
        tukey_alpha = 0.25
    tukey_alpha = float(np.clip(tukey_alpha, 0.0, 1.0))
    if win_shape not in ("hann", "tukey"):
        win_shape = "hann"
    try:
        pre_ring_db = float((st or {}).get("ir_pre_ringing_db", float("nan")))
        if np.isfinite(pre_ring_db) and win_shape == "tukey":
            alpha_scale = float(np.clip(1.0 + 0.10 * (-45.0 - pre_ring_db), 0.7, 1.8))
            tukey_alpha = float(np.clip(tukey_alpha * alpha_scale, 0.05, 0.95))
    except (TypeError, ValueError, FloatingPointError, KeyError):
        pass
    return win_shape, tukey_alpha


def _phase_ir_window_auto_asym_params(cfg, st) -> tuple[float, float]:
    try:
        auto_asym_left_ratio = float(getattr(cfg, "auto_asym_left_ratio", 0.35) or 0.35)
    except (AttributeError, TypeError, ValueError):
        auto_asym_left_ratio = 0.35
    if not np.isfinite(auto_asym_left_ratio):
        auto_asym_left_ratio = 0.35
    auto_asym_left_ratio = float(np.clip(auto_asym_left_ratio, 0.0, 1.0))
    try:
        pre_ring_db = float((st or {}).get("ir_pre_ringing_db", float("nan")))
        if np.isfinite(pre_ring_db):
            ratio_scale = float(np.clip(1.0 + 0.06 * (-42.0 - pre_ring_db), 0.6, 1.4))
            auto_asym_left_ratio = float(np.clip(auto_asym_left_ratio * ratio_scale, 0.15, 0.60))
    except (TypeError, ValueError, FloatingPointError, KeyError):
        pass
    try:
        auto_asym_left_max_ms = float(getattr(cfg, "auto_asym_left_max_ms", 25.0) or 25.0)
    except (AttributeError, TypeError, ValueError):
        auto_asym_left_max_ms = 25.0
    if not np.isfinite(auto_asym_left_max_ms):
        auto_asym_left_max_ms = 25.0
    auto_asym_left_max_ms = float(max(0.0, auto_asym_left_max_ms))
    return auto_asym_left_ratio, auto_asym_left_max_ms


def _phase_ir_window_edge_taper(L: int, *, alpha: float, side: str, win_shape: str) -> np.ndarray:
    L = int(L)
    if L <= 0:
        return np.zeros(0, dtype=float)
    if win_shape == "hann" or alpha >= 0.999:
        if side == "left":
            return (np.sin(np.linspace(0, np.pi / 2, L + 1))[:-1] ** 2).astype(float)
        return (np.cos(np.linspace(0, np.pi / 2, L + 1))[1:] ** 2).astype(float)
    a = float(np.clip(alpha, 0.0, 1.0))
    taper_len = int(max(1, round(a * L)))
    plateau_len = int(max(0, L - taper_len))
    k = (np.arange(L)[::-1] if side == "left" else np.arange(L)).astype(float)
    w = np.ones(L, dtype=float)
    if plateau_len > 0:
        w[k >= plateau_len] = 0.0
        w[k < plateau_len] = 1.0
    else:
        w[:] = 0.0
    kk = np.clip(k - plateau_len, 0.0, float(taper_len))
    x = kk / float(max(taper_len, 1))
    taper = 0.5 * (1.0 + np.cos(np.pi * x))
    mask = k >= plateau_len
    w[mask] = taper[mask]
    return np.clip(w, 0.0, 1.0)


def _phase_ir_window_apply_auto_asym_cap(
    *,
    win_mode: str,
    requested_win_mode: str,
    cfg,
    left_ms: float,
    right_ms: float,
    s_left: int,
    s_right: int,
    auto_asym_left_ratio: float,
    auto_asym_left_max_ms: float,
    logger=None,
) -> tuple[int, int, bool, float]:
    auto_asym_left_cap_applied = False
    auto_asym_left_ms_after = float(left_ms)
    if win_mode == "auto":
        if not ("Asym" in cfg.filter_type_str or "Min" in cfg.filter_type_str):
            radius = s_right
            s_left = radius
            s_right = radius
        elif str(requested_win_mode).strip().lower() == "auto":
            left_cap_ms = min(
                float(left_ms),
                float(auto_asym_left_ratio * max(float(right_ms), 0.0)),
                float(auto_asym_left_max_ms),
            )
            if left_cap_ms < float(left_ms):
                s_left = int(left_cap_ms * cfg.fs / 1000.0)
                auto_asym_left_cap_applied = True
                auto_asym_left_ms_after = float(left_cap_ms)
                if logger is not None:
                    logger.info(
                        "IR export auto-asym left cap applied: "
                        f"{left_ms:.2f} ms -> {left_cap_ms:.2f} ms "
                        f"(ratio={auto_asym_left_ratio:.3f}, max={auto_asym_left_max_ms:.2f} ms)"
                    )
    return s_left, s_right, auto_asym_left_cap_applied, auto_asym_left_ms_after


def _phase_ir_window_build_window(
    *,
    n: int,
    peak_idx: int,
    win_mode: str,
    s_left: int,
    s_right: int,
    win_shape: str,
    tukey_alpha: float,
) -> tuple[np.ndarray, int, int]:
    if win_mode == "off":
        window = np.ones(n, dtype=float)
        return window, 0, 0

    window = np.zeros(n, dtype=float)
    if win_mode == "rew_sym":
        radius = max(s_left, s_right)
        s_left = radius
        s_right = radius
    if s_left > 0:
        win_rise = _phase_ir_window_edge_taper(s_left, alpha=tukey_alpha, side="left", win_shape=win_shape)
        start_idx = peak_idx - s_left
        if start_idx >= 0:
            window[start_idx:peak_idx] = win_rise
        else:
            window[0:peak_idx] = win_rise[-start_idx:]
    if s_right > 0:
        win_fall = _phase_ir_window_edge_taper(s_right, alpha=tukey_alpha, side="right", win_shape=win_shape)
        end_idx = peak_idx + 1 + s_right
        if end_idx <= n:
            window[peak_idx + 1:end_idx] = win_fall
        else:
            avail = n - (peak_idx + 1)
            if avail > 0:
                window[peak_idx + 1:n] = win_fall[:avail]
    window[peak_idx] = 1.0
    return window, s_left, s_right


def _phase_ir_window_apply_min_tail(
    impulse: np.ndarray,
    cfg,
    st,
    *,
    win_mode: str,
    is_min_filter: bool,
    logger=None,
) -> np.ndarray:
    n = len(impulse)
    if not (is_min_filter and win_mode == "off" and n > 16):
        return impulse
    try:
        taper_ms = float(getattr(cfg, "min_off_tail_taper_ms", 2.0) or 2.0)
    except (AttributeError, TypeError, ValueError):
        taper_ms = 2.0
    if not np.isfinite(taper_ms):
        taper_ms = 2.0
    taper_ms = float(np.clip(taper_ms, 0.0, 20.0))
    taper_n = int(round((taper_ms / 1000.0) * float(cfg.fs)))
    taper_n = int(np.clip(taper_n, 8, max(8, n // 8)))
    if taper_n < n:
        t = (np.cos(np.linspace(0.0, np.pi / 2.0, taper_n, endpoint=True)) ** 2).astype(float)
        impulse[-taper_n:] *= t
        try:
            if isinstance(st, dict):
                st["min_off_tail_taper_ms"] = float(taper_ms)
                st["min_off_tail_taper_samples"] = int(taper_n)
        except (TypeError, ValueError):
            pass
        if logger is not None:
            logger.info(f"Minimum OFF anti-wrap taper applied: {taper_ms:.2f} ms ({taper_n} samples)")
    return impulse


def _phase_ir_window_remove_dc(impulse: np.ndarray, window: np.ndarray, *, win_mode: str) -> np.ndarray:
    if win_mode == "off":
        return impulse
    try:
        w_sum = float(np.sum(window))
        if w_sum > 0.0:
            dc = float(np.sum(impulse) / w_sum)
            impulse = impulse - (dc * window)
    except (TypeError, ValueError, FloatingPointError):
        pass
    return impulse


def _phase_ir_window_store_stats(
    *,
    st,
    win_mode: str,
    win_shape: str,
    s_left: int,
    s_right: int,
    auto_asym_left_cap_applied: bool,
    auto_asym_left_ms_before: float,
    auto_asym_left_ms_after: float,
    auto_asym_left_ratio: float,
    auto_asym_left_max_ms: float,
) -> None:
    try:
        if isinstance(st, dict):
            st["ir_window_mode"] = str(win_mode)
            st["ir_window_shape"] = str(win_shape)
            st["ir_window_left_samples"] = int(s_left)
            st["ir_window_right_samples"] = int(s_right)
            st["ir_window_auto_asym_left_cap_applied"] = bool(auto_asym_left_cap_applied)
            st["ir_window_auto_asym_left_ms_before"] = float(auto_asym_left_ms_before)
            st["ir_window_auto_asym_left_ms_after"] = float(auto_asym_left_ms_after)
            st["ir_window_auto_asym_left_ratio"] = float(auto_asym_left_ratio)
            st["ir_window_auto_asym_left_max_ms"] = float(auto_asym_left_max_ms)
    except (TypeError, ValueError):
        pass


def _apply_ir_window(ir, cfg, st, logger=None) -> np.ndarray:
    impulse = np.asarray(ir, dtype=float).copy()
    requested_win_mode, win_mode, is_min_filter = _resolve_ir_window_mode(cfg, logger=logger)
    left_ms = _ms_value(cfg, "ir_window_ms_left", "ir_window_left", 0.0)
    right_ms = _ms_value(
        cfg,
        "ir_window_ms_right",
        "ir_window_right",
        _ms_value(cfg, "ir_window_ms", "ir_window", 0.0),
    )

    win_shape = str(getattr(cfg, "ir_export_window_shape", "hann") or "hann").strip().lower()
    win_shape, tukey_alpha = _phase_ir_window_tukey_alpha(cfg, st, win_shape)

    n = len(impulse)
    if n == 0:
        return impulse
    peak_idx = int(np.argmax(np.abs(impulse)))
    s_left = int(left_ms * cfg.fs / 1000.0)
    s_right = int(right_ms * cfg.fs / 1000.0)
    auto_asym_left_cap_applied = False
    auto_asym_left_ms_before = float(left_ms)
    auto_asym_left_ms_after = float(left_ms)
    auto_asym_left_ratio, auto_asym_left_max_ms = _phase_ir_window_auto_asym_params(cfg, st)
    if logger is not None:
        logger.info(
            f"IR export: mode={win_mode}, peak={peak_idx}, user_s_left={s_left}, user_s_right={s_right}"
        )
        logger.info(f"IR export: shape={win_shape}, tukey_alpha={tukey_alpha:.2f}")

    s_left, s_right, auto_asym_left_cap_applied, auto_asym_left_ms_after = _phase_ir_window_apply_auto_asym_cap(
        win_mode=win_mode,
        requested_win_mode=requested_win_mode,
        cfg=cfg,
        left_ms=left_ms,
        right_ms=right_ms,
        s_left=s_left,
        s_right=s_right,
        auto_asym_left_ratio=auto_asym_left_ratio,
        auto_asym_left_max_ms=auto_asym_left_max_ms,
        logger=logger,
    )
    window, s_left, s_right = _phase_ir_window_build_window(
        n=n,
        peak_idx=peak_idx,
        win_mode=win_mode,
        s_left=s_left,
        s_right=s_right,
        win_shape=win_shape,
        tukey_alpha=tukey_alpha,
    )

    impulse_before = impulse.copy()
    impulse *= window

    impulse = _phase_ir_window_apply_min_tail(
        impulse,
        cfg,
        st,
        win_mode=win_mode,
        is_min_filter=is_min_filter,
        logger=logger,
    )

    if win_shape == "tukey" and tukey_alpha > 0.0 and win_mode != "off":
        try:
            d = float(np.max(np.abs(impulse - impulse_before)))
        except (TypeError, ValueError, FloatingPointError):
            d = 0.0
        if (not math.isfinite(d)) or (d <= 0.0):
            if logger is not None:
                logger.info(
                    "IR export window: Tukey had no measurable effect on impulse "
                    f"(mode={win_mode}, alpha={tukey_alpha:.3f}). Continuing."
                )

    impulse = _phase_ir_window_remove_dc(impulse, window, win_mode=win_mode)
    _phase_ir_window_store_stats(
        st=st,
        win_mode=win_mode,
        win_shape=win_shape,
        s_left=s_left,
        s_right=s_right,
        auto_asym_left_cap_applied=auto_asym_left_cap_applied,
        auto_asym_left_ms_before=auto_asym_left_ms_before,
        auto_asym_left_ms_after=auto_asym_left_ms_after,
        auto_asym_left_ratio=auto_asym_left_ratio,
        auto_asym_left_max_ms=auto_asym_left_max_ms,
    )
    return impulse


def _apply_fdw_if_enabled(ir_win, cfg, st) -> np.ndarray:
    out = np.asarray(ir_win, dtype=float)
    try:
        if isinstance(st, dict):
            st["phase_ir_fdw_applied"] = False
            st["phase_ir_fdw_reason"] = "afdw_is_magnitude_stage"
    except (TypeError, ValueError):
        pass
    return out
