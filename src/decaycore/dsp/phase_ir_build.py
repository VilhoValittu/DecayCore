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

import numpy as np
import scipy.fft

from .phase import calculate_minimum_phase, combine_mixed_phase
from .phase_ir_align import _compute_alignment_target, _shift_ir
from .phase_ir_guards import _pre_energy_guard, _tdc_postprocess
from .phase_ir_ir import _phase_batch_to_ir, _phase_to_ir, _normalize_ir
from .phase_ir_metrics import _summarize_ir_metrics
from .phase_ir_phase_parts import _PhaseComponents, _apply_phase_model, _compute_excess_phase, _unwrap_phases
from .phase_ir_realized import compute_realized_phase_gd_metrics
from .phase_ir_utils import _ms_value, _resolve_ir_anchor_mode, _resolve_ir_window_mode
from .phase_ir_window import _apply_fdw_if_enabled, _apply_ir_window


def _write_midband_realized_level_match_state(
    st: dict,
    *,
    enabled: bool,
    applied: bool,
    reason: str,
    f_lo: float,
    f_hi: float,
    delta_raw,
    delta_apply,
    delta_after,
    scale,
) -> None:
    try:
        if isinstance(st, dict):
            st["ir_realized_level_match_enabled"] = bool(enabled)
            st["ir_realized_level_match_applied"] = bool(applied)
            st["ir_realized_level_match_reason"] = str(reason)
            st["ir_realized_level_match_mid_lo_hz"] = float(f_lo)
            st["ir_realized_level_match_mid_hi_hz"] = float(f_hi)
            st["ir_realized_level_match_delta_db_raw"] = (
                float(delta_raw) if delta_raw is not None and np.isfinite(float(delta_raw)) else None
            )
            st["ir_realized_level_match_delta_db_applied"] = (
                float(delta_apply) if delta_apply is not None and np.isfinite(float(delta_apply)) else 0.0
            )
            st["ir_realized_level_match_delta_db_after"] = (
                float(delta_after) if delta_after is not None and np.isfinite(float(delta_after)) else None
            )
            st["ir_realized_level_match_scale"] = (
                float(scale) if scale is not None and np.isfinite(float(scale)) else 1.0
            )
    except (TypeError, ValueError):
        pass


def _resolve_midband_realized_level_match_cfg(cfg) -> tuple[bool, float, float, float, float]:
    try:
        enabled = bool(getattr(cfg, "ir_realized_level_match_enable", True))
    except (AttributeError, TypeError, ValueError):
        enabled = True
    try:
        f_lo = float(getattr(cfg, "ir_realized_level_match_mid_lo_hz", 200.0) or 200.0)
    except (AttributeError, TypeError, ValueError):
        f_lo = 200.0
    try:
        f_hi = float(getattr(cfg, "ir_realized_level_match_mid_hi_hz", 2000.0) or 2000.0)
    except (AttributeError, TypeError, ValueError):
        f_hi = 2000.0
    try:
        min_abs_db = float(getattr(cfg, "ir_realized_level_match_min_abs_db", 0.25) or 0.25)
    except (AttributeError, TypeError, ValueError):
        min_abs_db = 0.25
    try:
        max_abs_db = float(getattr(cfg, "ir_realized_level_match_max_abs_db", 3.0) or 3.0)
    except (AttributeError, TypeError, ValueError):
        max_abs_db = 3.0

    if not np.isfinite(f_lo):
        f_lo = 200.0
    if not np.isfinite(f_hi) or f_hi <= f_lo:
        f_hi = max(f_lo + 1.0, 2000.0)
    if not np.isfinite(min_abs_db) or min_abs_db < 0.0:
        min_abs_db = 0.25
    if not np.isfinite(max_abs_db) or max_abs_db <= 0.0:
        max_abs_db = 3.0
    return enabled, float(f_lo), float(f_hi), float(min_abs_db), float(max_abs_db)


def _select_phase_ir_impulse(
    raw_imp: np.ndarray,
    cfg,
    *,
    n_fft: int,
    left_ms: float,
    want_left_anchor: bool,
) -> np.ndarray:
    if "Asym" in cfg.filter_type_str:
        shift = min(int(left_ms * cfg.fs / 1000.0), int(n_fft * 0.4))
        return np.roll(raw_imp, shift)
    if "Min" in cfg.filter_type_str:
        return raw_imp
    if "Linear" in cfg.filter_type_str:
        if want_left_anchor:
            shift = min(int(left_ms * cfg.fs / 1000.0), int(n_fft * 0.4))
            return np.roll(raw_imp, shift)
        return np.roll(raw_imp, n_fft // 2)
    return np.roll(raw_imp, n_fft // 2)


def _build_phase_ir_raw_impulse(
    *,
    total_mag: np.ndarray,
    min_p: np.ndarray,
    final_phase: np.ndarray,
    cfg,
    n_fft: int,
    is_mixed: bool,
    mixed_split_hz: float,
    mixed_transition_hz: float,
    low_phase: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if is_mixed and low_phase is not None:
        ir_lin, ir_min = _phase_batch_to_ir(
            total_mag,
            np.vstack((low_phase, min_p)),
            n=n_fft,
        )
        raw_imp = combine_mixed_phase(
            ir_lin,
            ir_min,
            fs=float(cfg.fs),
            split_freq=mixed_split_hz,
            transition_hz=mixed_transition_hz,
        )
        final_phase = np.angle(scipy.fft.rfft(raw_imp))
    else:
        raw_imp = _phase_to_ir(total_mag, final_phase, n=n_fft)
    return raw_imp, final_phase


def _apply_phase_ir_pre_window_alignment(
    *,
    raw_imp: np.ndarray,
    cfg,
    st: dict,
    logger,
    n_fft: int,
) -> tuple[np.ndarray, str, str, bool]:
    anchor_mode = _resolve_ir_anchor_mode(cfg)
    requested_win_mode, win_mode, is_min_filter = _resolve_ir_window_mode(cfg, logger=logger)
    left_ms = _ms_value(cfg, "ir_window_ms_left", "ir_window_left", 0.0)
    want_left_anchor = str(requested_win_mode or "").strip().lower() == "rew_asym"
    impulse = _select_phase_ir_impulse(
        raw_imp,
        cfg,
        n_fft=n_fft,
        left_ms=left_ms,
        want_left_anchor=want_left_anchor,
    )

    try:
        if isinstance(st, dict):
            st["ir_anchor_mode"] = str(anchor_mode)
    except (TypeError, ValueError):
        pass
    logger.info(f"IR init: type={cfg.filter_type_str}, peak0={int(np.argmax(np.abs(impulse)))}, n={n_fft}")

    pre_target = _compute_alignment_target(impulse, cfg, st, stage="pre_window")
    impulse, pre_info = _shift_ir(
        impulse,
        pre_target,
        True,
        st=st,
        key_prefix="pre_align",
        anchor_mode=anchor_mode,
    )
    if is_min_filter and win_mode == "off" and bool(pre_info.get("applied", False)):
        try:
            if isinstance(st, dict):
                st["min_off_peak_shift_samples"] = int(pre_info.get("shift_samples", 0))
                st["min_off_peak_before_samples"] = int(pre_info.get("peak_before_samples", 0))
        except (TypeError, ValueError):
            pass
        logger.info(
            "Minimum OFF causal shift applied: "
            f"peak {int(pre_info.get('peak_before_samples', 0))} -> 0 "
            f"(shift={int(pre_info.get('shift_samples', 0))})"
        )
    return impulse, anchor_mode, win_mode, is_min_filter


def _apply_phase_ir_post_window_alignment(
    *,
    impulse: np.ndarray,
    cfg,
    st: dict,
    logger,
    anchor_mode: str,
    is_mixed: bool,
) -> tuple[np.ndarray, dict]:
    impulse = _apply_ir_window(impulse, cfg, st, logger=logger)
    impulse = _apply_fdw_if_enabled(impulse, cfg, st)

    post_target = _compute_alignment_target(impulse, cfg, st, stage="post_window")
    impulse, post_info = _shift_ir(
        impulse,
        post_target,
        True,
        st=st,
        key_prefix="post_align",
        anchor_mode=anchor_mode,
    )
    try:
        if isinstance(st, dict):
            split_idx = int(post_info.get("anchor_after_samples", post_info.get("peak_after_samples", 0)))
            st["ir_energy_split_samples"] = int(np.clip(split_idx, 0, max(int(impulse.size) - 1, 0)))
            st["ir_energy_split_source"] = "post_align_anchor"
            if anchor_mode == "min_causal":
                st["min_causal_shift"] = int(post_info.get("shift_samples", 0))
                st["min_causal_peak_after_samples"] = int(post_info.get("peak_after_samples", 0))
                st["min_causal_anchor_after_samples"] = int(post_info.get("anchor_after_samples", 0))
    except (TypeError, ValueError):
        pass
    if is_mixed and impulse.size > 0 and anchor_mode != "min_causal":
        mixed_forced_peak_ms = 90.0
        try:
            if isinstance(st, dict):
                st["mixed_forced_peak_ms"] = float(mixed_forced_peak_ms)
                st["mixed_forced_peak_samples"] = int(post_info.get("desired_samples", 0))
                st["mixed_forced_shift_samples"] = int(post_info.get("shift_samples", 0))
        except (TypeError, ValueError):
            pass
        logger.info(
            f"Mixed forced peak shift applied: peak {int(post_info.get('peak_before_samples', 0))} "
            f"-> {int(post_info.get('desired_samples', 0))} "
            f"({mixed_forced_peak_ms:.1f} ms, shift={int(post_info.get('shift_samples', 0))})"
        )
    return impulse, post_info


def _apply_phase_ir_alignment_and_window(
    *,
    raw_imp: np.ndarray,
    cfg,
    st: dict,
    logger,
    n_fft: int,
    is_mixed: bool,
) -> tuple[np.ndarray, str, bool, dict]:
    impulse, anchor_mode, win_mode, is_min_filter = _apply_phase_ir_pre_window_alignment(
        raw_imp=raw_imp,
        cfg=cfg,
        st=st,
        logger=logger,
        n_fft=n_fft,
    )
    impulse, post_info = _apply_phase_ir_post_window_alignment(
        impulse=impulse,
        cfg=cfg,
        st=st,
        logger=logger,
        anchor_mode=anchor_mode,
        is_mixed=is_mixed,
    )
    return impulse, win_mode, is_min_filter, post_info


def _apply_midband_realized_level_match(
    impulse: np.ndarray,
    total_mag: np.ndarray,
    cfg,
    st: dict,
) -> np.ndarray:
    """Match realized IR level to total_mag in 200-2000 Hz with a global scale."""
    ir = np.asarray(impulse, dtype=float).copy()
    mag_target = np.asarray(total_mag, dtype=float).reshape(-1)
    enabled, f_lo, f_hi, min_abs_db, max_abs_db = _resolve_midband_realized_level_match_cfg(cfg)

    if not enabled:
        _write_midband_realized_level_match_state(
            st,
            enabled=enabled,
            applied=False,
            reason="disabled",
            f_lo=f_lo,
            f_hi=f_hi,
            delta_raw=None,
            delta_apply=0.0,
            delta_after=None,
            scale=1.0,
        )
        return ir

    fs = float(getattr(cfg, "fs", 0.0) or 0.0)
    if ir.size < 8 or mag_target.size < 8 or fs <= 0.0:
        _write_midband_realized_level_match_state(
            st,
            enabled=enabled,
            applied=False,
            reason="invalid_input",
            f_lo=f_lo,
            f_hi=f_hi,
            delta_raw=None,
            delta_apply=0.0,
            delta_after=None,
            scale=1.0,
        )
        return ir

    n = int(ir.size)
    h2 = scipy.fft.rfft(ir, n=n)
    mag2 = np.abs(h2)
    n2 = int(min(mag2.size, mag_target.size))
    if n2 < 8:
        _write_midband_realized_level_match_state(
            st,
            enabled=enabled,
            applied=False,
            reason="insufficient_bins",
            f_lo=f_lo,
            f_hi=f_hi,
            delta_raw=None,
            delta_apply=0.0,
            delta_after=None,
            scale=1.0,
        )
        return ir

    freq = scipy.fft.rfftfreq(n, d=1.0 / fs)[:n2]
    mag2_db = 20.0 * np.log10(np.maximum(np.asarray(mag2[:n2], dtype=float), 1e-12))
    target_db = 20.0 * np.log10(np.maximum(np.asarray(mag_target[:n2], dtype=float), 1e-12))

    valid = np.isfinite(freq) & np.isfinite(mag2_db) & np.isfinite(target_db) & (freq > 0.0)
    mid = valid & (freq >= float(f_lo)) & (freq <= float(f_hi))
    if int(np.count_nonzero(mid)) < 8:
        _write_midband_realized_level_match_state(
            st,
            enabled=enabled,
            applied=False,
            reason="mid_band_sparse",
            f_lo=f_lo,
            f_hi=f_hi,
            delta_raw=None,
            delta_apply=0.0,
            delta_after=None,
            scale=1.0,
        )
        return ir

    delta_db_raw = float(np.median(mag2_db[mid] - target_db[mid]))
    if not np.isfinite(delta_db_raw):
        _write_midband_realized_level_match_state(
            st,
            enabled=enabled,
            applied=False,
            reason="delta_invalid",
            f_lo=f_lo,
            f_hi=f_hi,
            delta_raw=None,
            delta_apply=0.0,
            delta_after=None,
            scale=1.0,
        )
        return ir

    delta_db_apply = float(np.clip(delta_db_raw, -max_abs_db, max_abs_db))
    if abs(delta_db_apply) <= float(min_abs_db):
        _write_midband_realized_level_match_state(
            st,
            enabled=enabled,
            applied=False,
            reason="below_threshold",
            f_lo=f_lo,
            f_hi=f_hi,
            delta_raw=delta_db_raw,
            delta_apply=0.0,
            delta_after=delta_db_raw,
            scale=1.0,
        )
        return ir

    scale = float(10.0 ** (-delta_db_apply / 20.0))
    ir *= scale

    try:
        # FFT is linear: scaling ir by `scale` shifts mag by -delta_db_apply
        delta_after = float(delta_db_raw - delta_db_apply)
    except (TypeError, ValueError, FloatingPointError):
        delta_after = None

    _write_midband_realized_level_match_state(
        st,
        enabled=enabled,
        applied=True,
        reason="applied",
        f_lo=f_lo,
        f_hi=f_hi,
        delta_raw=delta_db_raw,
        delta_apply=delta_db_apply,
        delta_after=delta_after,
        scale=scale,
    )
    return ir


def build_phase_and_ir(
    *,
    cfg,
    freq_axis: np.ndarray,
    n_fft: int,
    gain_db: np.ndarray,
    p_rad_interp: np.ndarray,
    conf_mask,
    st: dict,
    mask_c: np.ndarray,
    use_bassfirst: bool,
    afdw_on: bool,
    logger,
    theo_xo: np.ndarray,
    auto_global_gain_db: float,
    auto_headroom_db: float,
    final_gain_total: np.ndarray,
    limit_gd_gradient_ms_per_oct_fn,
    phase_feedback_static: dict | None = None,
) -> dict:
    """Contract:
      - This stage builds phase + IR only.
      - It must not modify gain_db/autogain values.
      - Allowed side effects are st window/align/guard/metric indicators.

    Returns dict with:
      impulse: np.ndarray
      total_mag: np.ndarray
      min_phase: np.ndarray
      final_phase: np.ndarray
      mixed_split_hz: float
      mixed_transition_hz: float
    Side effects: st updates exactly like before (mixed_blend keys etc.)
    """
    static = phase_feedback_static if isinstance(phase_feedback_static, dict) else {}
    required_static = {
        "total_mag",
        "min_phase",
        "raw_unwrapped",
        "reference_unwrapped",
        "excess_unwrapped",
        "is_mixed",
        "mixed_split_hz",
        "mixed_transition_hz",
    }
    if required_static.issubset(static):
        total_mag = np.asarray(static["total_mag"], dtype=float)
        min_p = np.asarray(static["min_phase"], dtype=float)
        raw_u = np.asarray(static["raw_unwrapped"], dtype=float)
        ref_u = np.asarray(static["reference_unwrapped"], dtype=float)
        excess_u = np.asarray(static["excess_unwrapped"], dtype=float)
        is_mixed = bool(static["is_mixed"])
        mixed_split_hz = float(static["mixed_split_hz"])
        mixed_transition_hz = float(static["mixed_transition_hz"])
    else:
        total_mag = 10 ** (final_gain_total / 20.0)
        min_p = calculate_minimum_phase(total_mag, max_phase_deg=None)
        is_mixed = "Mixed" in cfg.filter_type_str
        mixed_split_hz = float(
            np.clip(
                float(getattr(cfg, "mixed_split_freq", 300.0) or 300.0),
                20.0,
                float(cfg.fs) * 0.49,
            )
        )
        mixed_transition_hz = float(getattr(cfg, "trans_width", mixed_split_hz) or mixed_split_hz)
        if not np.isfinite(mixed_transition_hz) or mixed_transition_hz < 0.0:
            mixed_transition_hz = mixed_split_hz
        raw_u, ref_u = _unwrap_phases(p_rad_interp, theo_xo)
        excess_u = _compute_excess_phase(raw_u, ref_u)
        static.update(
            {
                "total_mag": np.asarray(total_mag, dtype=float),
                "min_phase": np.asarray(min_p, dtype=float),
                "raw_unwrapped": np.asarray(raw_u, dtype=float),
                "reference_unwrapped": np.asarray(ref_u, dtype=float),
                "excess_unwrapped": np.asarray(excess_u, dtype=float),
                "is_mixed": bool(is_mixed),
                "mixed_split_hz": float(mixed_split_hz),
                "mixed_transition_hz": float(mixed_transition_hz),
                "profiles": {},
            }
        )

    phase_components = _PhaseComponents(
        raw_u=raw_u,
        ref_u=ref_u,
        excess_u=excess_u,
        min_phase=min_p,
        theo_xo=theo_xo,
        conf_mask=conf_mask,
        total_mag=total_mag,
        n_fft=n_fft,
        is_mixed=is_mixed,
        mixed_split_hz=mixed_split_hz,
        mixed_transition_hz=mixed_transition_hz,
        use_bassfirst=use_bassfirst,
        afdw_on=afdw_on,
        logger=logger,
        limit_gd_gradient_ms_per_oct_fn=limit_gd_gradient_ms_per_oct_fn,
        static_profiles=static.setdefault("profiles", {}),
    )
    final_phase = _apply_phase_model(freq_axis, cfg, st, phase_components)
    low_phase = phase_components.low_phase
    if is_mixed and low_phase is not None:
        try:
            if isinstance(st, dict):
                st["mixed_blend_split_hz"] = float(mixed_split_hz)
                st["mixed_blend_transition_hz"] = float(mixed_transition_hz)
        except (TypeError, ValueError):
            pass
    raw_imp, final_phase = _build_phase_ir_raw_impulse(
        total_mag=total_mag,
        min_p=min_p,
        final_phase=final_phase,
        cfg=cfg,
        n_fft=n_fft,
        is_mixed=is_mixed,
        mixed_split_hz=mixed_split_hz,
        mixed_transition_hz=mixed_transition_hz,
        low_phase=low_phase,
    )

    raw_imp = _normalize_ir(raw_imp, cfg)
    impulse, win_mode, is_min_filter, post_info = _apply_phase_ir_alignment_and_window(
        raw_imp=raw_imp,
        cfg=cfg,
        st=st,
        logger=logger,
        n_fft=n_fft,
        is_mixed=is_mixed,
    )

    impulse, _ = _pre_energy_guard(impulse, cfg, st)
    impulse = _tdc_postprocess(impulse, cfg, st)
    impulse = _apply_midband_realized_level_match(impulse, total_mag, cfg, st)
    _summarize_ir_metrics(impulse, cfg, st)
    try:
        realized_metrics = compute_realized_phase_gd_metrics(
            freq_axis=np.asarray(freq_axis, dtype=float),
            measured_phase_rad=np.asarray(p_rad_interp, dtype=float),
            impulse=np.asarray(impulse, dtype=float),
            fs=float(cfg.fs),
            phase_limit_hz=float(getattr(cfg, "phase_limit", 0.0) or 0.0),
            confidence_mask=conf_mask,
        )
        if isinstance(st, dict):
            st.update(realized_metrics)
    except (AttributeError, TypeError, ValueError, FloatingPointError, IndexError):
        pass

    return {
        "impulse": np.asarray(impulse, dtype=float),
        "total_mag": np.asarray(total_mag, dtype=float),
        "min_phase": np.asarray(min_p, dtype=float),
        "final_phase": np.asarray(final_phase, dtype=float),
        "mixed_split_hz": float(mixed_split_hz),
        "mixed_transition_hz": float(mixed_transition_hz),
        "phase_feedback_static": static,
    }
