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

from .dsp_telemetry import apply_residual_telemetry
from .phase_ir_autogain import compute_auto_gain_and_headroom
from .phase_ir_build import build_phase_and_ir
from .phase_ir_contracts import (
    require_allowed_keys,
    require_scalar_unchanged,
    require_unchanged,
)
from .phase_ir_phase import _compute_excess_phase
from .phase_ir_residual import apply_residual_pass_if_enabled
from .phase_ir_theoretical import compute_theoretical_phase_and_store_stats
from .phase_ir_types import PhaseIRInputs, PhaseIROutputs

__all__ = ("run_phase_ir_stage", "_compute_excess_phase")

_THEORETICAL_ALLOWED_KEYS = frozenset(
    {
        "theo_xo",
        "theo_on_raw",
        "theo_off_raw",
        "xo_txt",
        "hpf_txt",
        "hpf_freq",
        "hpf_slope",
        "p_rad_interp",
    }
)

_AUTOGAIN_ALLOWED_KEYS = frozenset(
    {
        "current_peak_gain",
        "gain_margin_db",
        "auto_global_gain_db",
        "auto_headroom_db",
        "final_gain_total",
    }
)

_BUILD_ALLOWED_KEYS = frozenset(
    {
        "impulse",
        "total_mag",
        "min_phase",
        "final_phase",
        "mixed_split_hz",
        "mixed_transition_hz",
    }
)


def _apply_hpf_magnitude_to_gain(*, cfg, freq_axis: np.ndarray, gain_db: np.ndarray, apply_hpf_to_mags, logger) -> np.ndarray:
    hs = cfg.hpf_settings
    if not (isinstance(hs, dict) and hs.get("enabled")):
        return gain_db
    hpf_f = float(hs.get("freq", 0.0) or 0.0)
    hpf_order = int(hs.get("order", 0) or 0)
    if not (hpf_f > 0 and hpf_order > 0):
        return gain_db
    hpf_db = apply_hpf_to_mags(freq_axis, np.zeros_like(freq_axis), hpf_f, hpf_order)
    out = np.asarray(gain_db, dtype=float) + np.asarray(hpf_db, dtype=float)
    hpf_guard_mask = np.isfinite(freq_axis) & (freq_axis <= float(hpf_f)) & (out > hpf_db)
    hpf_guard_bins = int(np.count_nonzero(hpf_guard_mask))
    if hpf_guard_bins > 0:
        out[hpf_guard_mask] = np.asarray(hpf_db, dtype=float)[hpf_guard_mask]
    try:
        logger.info(
            "HPF magnitude applied to FIR: "
            f"fc={hpf_f:.1f} Hz, "
            f"order={hpf_order} "
            f"({hpf_order * 6:.0f} dB/oct)"
            + (f", clamped {hpf_guard_bins} sub-HPF boost bins" if hpf_guard_bins > 0 else "")
        )
    except (AttributeError, TypeError, ValueError):
        pass
    return np.asarray(out, dtype=float)


def _apply_lpf_magnitude_to_gain(*, cfg, freq_axis: np.ndarray, gain_db: np.ndarray, apply_lpf_to_mags, logger) -> np.ndarray:
    ls = cfg.lpf_settings
    if not (isinstance(ls, dict) and ls.get("enabled")):
        return gain_db
    lpf_f = float(ls.get("freq", 0.0) or 0.0)
    lpf_order = int(ls.get("order", 0) or 0)
    if not (lpf_f > 0 and lpf_order > 0):
        return gain_db
    lpf_db = apply_lpf_to_mags(freq_axis, np.zeros_like(freq_axis), lpf_f, lpf_order)
    out = np.asarray(gain_db, dtype=float) + np.asarray(lpf_db, dtype=float)
    try:
        logger.info(
            f"LPF magnitude applied to FIR: fc={lpf_f:.1f} Hz, "
            f"order={lpf_order} ({lpf_order * 6:.0f} dB/oct)"
        )
    except (AttributeError, TypeError, ValueError):
        pass
    return np.asarray(out, dtype=float)


def _apply_phase_clamp_mag_feedback(
    *,
    cfg,
    mask_c: np.ndarray,
    gain_db: np.ndarray,
    p_rad_interp: np.ndarray,
    theo_xo: np.ndarray,
    logger,
) -> np.ndarray:
    phase_clamp_mag_feedback = float(getattr(cfg, "phase_clamp_mag_feedback", 0.5))
    if not (phase_clamp_mag_feedback > 0.0 and np.any(mask_c)):
        return np.asarray(gain_db, dtype=float)
    out = np.asarray(gain_db, dtype=float)
    try:
        excess_approx = np.abs(np.asarray(p_rad_interp, dtype=float) - np.asarray(theo_xo, dtype=float))
        try:
            lim_max_deg = float(getattr(cfg, "mixed_phase_budget_lf_deg", 60.0) or 60.0)
        except (AttributeError, TypeError, ValueError):
            lim_max_deg = 45.0
        lim_rad = float(np.deg2rad(lim_max_deg))
        pred_frac = np.where(
            excess_approx > lim_rad,
            np.clip(1.0 - lim_rad / np.maximum(excess_approx, 1e-9), 0.0, 1.0),
            0.0,
        )
        boost_mask = np.asarray(mask_c, dtype=bool) & (out > 0.0)
        if np.any(boost_mask):
            strength = float(np.clip(phase_clamp_mag_feedback, 0.0, 1.0))
            out[boost_mask] -= strength * pred_frac[boost_mask] * out[boost_mask]
            softened = int(np.sum(pred_frac[boost_mask] > 0.05))
            logger.info(
                f"PhaseClampMagFeedback: gain softened at {softened} bins "
                f"(strength={strength:.2f}, phase_limit≈{lim_max_deg:.1f} deg)"
            )
    except (TypeError, ValueError, FloatingPointError, AttributeError, IndexError):
        pass
    return np.asarray(out, dtype=float)


def run_phase_ir_stage(
    *,
    cfg,
    freq_axis,
    n_fft,
    gain_db,
    p_rad_interp,
    conf_mask,
    m_anal,
    calc_offset_db,
    target_mags,
    st,
    mask_c,
    base_sigma,
    _filter_smooth,
    df_mode,
    raw_g,
    final_g,
    use_bassfirst,
    afdw_on,
    logger,
    apply_hpf_to_mags_fn,
    apply_lpf_to_mags_fn,
    limit_gd_gradient_ms_per_oct_fn,
    cfg_float_allow_zero_fn,
):
    inputs = PhaseIRInputs(
        cfg=cfg,
        freq_axis=freq_axis,
        n_fft=n_fft,
        gain_db=gain_db,
        p_rad_interp=p_rad_interp,
        conf_mask=conf_mask,
        m_anal=m_anal,
        calc_offset_db=calc_offset_db,
        target_mags=target_mags,
        st=st,
        mask_c=mask_c,
        base_sigma=base_sigma,
        filter_smooth=_filter_smooth,
        df_mode=df_mode,
        raw_g=raw_g,
        final_g=final_g,
        use_bassfirst=use_bassfirst,
        afdw_on=afdw_on,
        logger=logger,
        apply_hpf_to_mags_fn=apply_hpf_to_mags_fn,
        apply_lpf_to_mags_fn=apply_lpf_to_mags_fn,
        limit_gd_gradient_ms_per_oct_fn=limit_gd_gradient_ms_per_oct_fn,
        cfg_float_allow_zero_fn=cfg_float_allow_zero_fn,
    )
    outputs = _run_phase_ir_stage(inputs)
    apply_residual_telemetry(st=inputs.st, telemetry=outputs.residual_telemetry)
    return outputs


def _run_phase_ir_stage(inputs: PhaseIRInputs) -> PhaseIROutputs:
    cfg = inputs.cfg
    freq_axis = np.asarray(inputs.freq_axis, dtype=float)
    n_fft = int(inputs.n_fft)
    gain_db = np.asarray(inputs.gain_db, dtype=float).copy()
    p_rad_interp = np.asarray(inputs.p_rad_interp, dtype=float).copy()
    conf_mask = inputs.conf_mask
    m_anal = np.asarray(inputs.m_anal, dtype=float)
    calc_offset_db = float(inputs.calc_offset_db)
    target_mags = np.asarray(inputs.target_mags, dtype=float)
    st = inputs.st
    mask_c = np.asarray(inputs.mask_c, dtype=bool)
    base_sigma = inputs.base_sigma
    _filter_smooth = inputs.filter_smooth
    df_mode = inputs.df_mode
    raw_g = inputs.raw_g
    final_g = inputs.final_g
    use_bassfirst = bool(inputs.use_bassfirst)
    afdw_on = bool(inputs.afdw_on)
    logger = inputs.logger
    apply_hpf_to_mags = inputs.apply_hpf_to_mags_fn
    apply_lpf_to_mags = inputs.apply_lpf_to_mags_fn
    _limit_gd_gradient_ms_per_oct = inputs.limit_gd_gradient_ms_per_oct_fn
    _cfg_float_allow_zero = inputs.cfg_float_allow_zero_fn

    gain_db_before_theoretical = gain_db.copy()
    p_rad_input_for_theoretical = p_rad_interp.copy()
    theo = compute_theoretical_phase_and_store_stats(
        cfg=cfg,
        freq_axis=freq_axis,
        p_rad_interp=p_rad_input_for_theoretical,
        st=st,
        logger=logger,
    )
    require_allowed_keys("phase_ir_theoretical", theo, _THEORETICAL_ALLOWED_KEYS)
    require_unchanged(
        "phase_ir_theoretical",
        "gain_db",
        gain_db_before_theoretical,
        gain_db,
    )
    require_unchanged(
        "phase_ir_theoretical",
        "p_rad_interp input",
        p_rad_interp,
        p_rad_input_for_theoretical,
    )

    theo_xo = np.asarray(theo["theo_xo"], dtype=float)
    p_rad_interp = np.asarray(theo.get("p_rad_interp", p_rad_interp), dtype=float)

    gain_db = _apply_hpf_magnitude_to_gain(
        cfg=cfg,
        freq_axis=freq_axis,
        gain_db=gain_db,
        apply_hpf_to_mags=apply_hpf_to_mags,
        logger=logger,
    )
    gain_db = _apply_lpf_magnitude_to_gain(
        cfg=cfg,
        freq_axis=freq_axis,
        gain_db=gain_db,
        apply_lpf_to_mags=apply_lpf_to_mags,
        logger=logger,
    )

    gain_db_before_residual = gain_db.copy()
    freq_axis_before_residual = freq_axis.copy()
    p_rad_before_residual = p_rad_interp.copy()
    m_anal_before_residual = m_anal.copy()
    target_mags_before_residual = target_mags.copy()
    mask_c_before_residual = mask_c.copy()
    gain_db, residual_telemetry = apply_residual_pass_if_enabled(
        cfg=cfg,
        freq_axis=freq_axis,
        gain_db=gain_db,
        conf_mask=conf_mask,
        m_anal=m_anal,
        calc_offset_db=calc_offset_db,
        target_mags=target_mags,
        st=st,
        mask_c=mask_c,
        base_sigma=base_sigma,
        filter_smooth=_filter_smooth,
        df_mode=df_mode,
        raw_g=raw_g,
        final_g=final_g,
        logger=logger,
        cfg_float_allow_zero_fn=_cfg_float_allow_zero,
    )
    gain_db = np.asarray(gain_db, dtype=float)
    if gain_db.shape != gain_db_before_residual.shape:
        raise RuntimeError(
            "Phase-IR contract breach in phase_ir_residual: "
            "gain_db shape changed unexpectedly."
        )
    require_unchanged(
        "phase_ir_residual",
        "freq_axis",
        freq_axis_before_residual,
        freq_axis,
    )
    require_unchanged(
        "phase_ir_residual",
        "p_rad_interp",
        p_rad_before_residual,
        p_rad_interp,
    )
    require_unchanged(
        "phase_ir_residual",
        "m_anal",
        m_anal_before_residual,
        m_anal,
    )
    require_unchanged(
        "phase_ir_residual",
        "target_mags",
        target_mags_before_residual,
        target_mags,
    )
    require_unchanged(
        "phase_ir_residual",
        "mask_c",
        mask_c_before_residual,
        mask_c,
    )

    gain_db = _apply_phase_clamp_mag_feedback(
        cfg=cfg,
        mask_c=mask_c,
        gain_db=gain_db,
        p_rad_interp=p_rad_interp,
        theo_xo=theo_xo,
        logger=logger,
    )

    gain_db_before_autogain = gain_db.copy()
    ag = compute_auto_gain_and_headroom(
        cfg=cfg,
        gain_db=gain_db,
        mask_c=mask_c,
        logger=logger,
    )
    require_allowed_keys("phase_ir_autogain", ag, _AUTOGAIN_ALLOWED_KEYS)
    require_unchanged(
        "phase_ir_autogain",
        "gain_db",
        gain_db_before_autogain,
        gain_db,
    )

    current_peak_gain = float(ag["current_peak_gain"])
    gain_margin_db = float(ag["gain_margin_db"])
    auto_global_gain_db = float(ag["auto_global_gain_db"])
    auto_headroom_db = float(ag["auto_headroom_db"])
    final_gain_total = np.asarray(ag["final_gain_total"], dtype=float)
    expected_final_gain_total = gain_db + auto_global_gain_db + auto_headroom_db
    require_unchanged(
        "phase_ir_autogain",
        "final_gain_total formula",
        expected_final_gain_total,
        final_gain_total,
    )

    gain_db_before_build = gain_db.copy()
    p_rad_before_build = p_rad_interp.copy()
    final_gain_total_before_build = final_gain_total.copy()
    auto_global_gain_db_before_build = float(auto_global_gain_db)
    auto_headroom_db_before_build = float(auto_headroom_db)
    built = build_phase_and_ir(
        cfg=cfg,
        freq_axis=freq_axis,
        n_fft=n_fft,
        gain_db=gain_db,
        p_rad_interp=p_rad_interp,
        conf_mask=conf_mask,
        st=st,
        mask_c=mask_c,
        use_bassfirst=use_bassfirst,
        afdw_on=afdw_on,
        logger=logger,
        theo_xo=theo_xo,
        auto_global_gain_db=auto_global_gain_db,
        auto_headroom_db=auto_headroom_db,
        final_gain_total=final_gain_total,
        limit_gd_gradient_ms_per_oct_fn=_limit_gd_gradient_ms_per_oct,
    )
    require_allowed_keys("phase_ir_build", built, _BUILD_ALLOWED_KEYS)
    require_unchanged(
        "phase_ir_build",
        "gain_db",
        gain_db_before_build,
        gain_db,
    )
    require_unchanged(
        "phase_ir_build",
        "p_rad_interp",
        p_rad_before_build,
        p_rad_interp,
    )
    require_unchanged(
        "phase_ir_build",
        "final_gain_total",
        final_gain_total_before_build,
        final_gain_total,
    )
    require_scalar_unchanged(
        "phase_ir_build",
        "auto_global_gain_db",
        auto_global_gain_db_before_build,
        auto_global_gain_db,
    )
    require_scalar_unchanged(
        "phase_ir_build",
        "auto_headroom_db",
        auto_headroom_db_before_build,
        auto_headroom_db,
    )

    return PhaseIROutputs(
        impulse=np.asarray(built["impulse"], dtype=float),
        gain_db=np.asarray(gain_db, dtype=float),
        auto_global_gain_db=float(auto_global_gain_db),
        gain_margin_db=float(gain_margin_db),
        auto_headroom_db=float(auto_headroom_db),
        current_peak_gain=float(current_peak_gain),
        final_gain_total=np.asarray(final_gain_total, dtype=float),
        residual_telemetry=residual_telemetry,
    )
