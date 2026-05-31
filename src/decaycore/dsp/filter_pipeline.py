# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

import logging
from dataclasses import replace

import numpy as np

from decaycore.auto_mode.auto_mode_profile import profiled_section
from decaycore.config.models import FilterConfig

from ._pruning import get_pruning_hook as _get_pruning_hook
from .decaycore_leveling import StereoLinkContext
from .dsp_correction import run_correction_stage
from .dsp_ops import (
    _limit_gd_gradient_ms_per_oct,
    _stage_probe,
    apply_confidence_weighted_target_pull,
    apply_hpf_to_mags,
    apply_lpf_to_mags,
    interpolate_response,
)
from .dsp_phase_ir import run_phase_ir_stage
from .dsp_preprocess import run_preprocess
from .dsp_utils import cfg_float_allow_zero as _cfg_float_allow_zero
from .hybrid_iir import HybridIIRPolicy, design_hybrid_iir
from .modal_analysis import detect_room_modes
from .phase_ir_autogain import compute_auto_gain_and_headroom
from .phase_ir_residual import apply_residual_pass_if_enabled

logger = logging.getLogger("DecayCore.dsp")


def _array_from_stats(st: dict, *keys: str) -> np.ndarray:
    for key in keys:
        try:
            arr = np.asarray(st.get(key, []), dtype=float).reshape(-1)
        except (TypeError, ValueError):
            arr = np.asarray([], dtype=float)
        if arr.size:
            return arr
    return np.asarray([], dtype=float)


def _apply_hybrid_iir_preconditioning(
    *,
    cfg: FilterConfig,
    freq_axis: np.ndarray,
    gain_db: np.ndarray,
    m_anal: np.ndarray,
    calc_offset_db: float,
    target_mags: np.ndarray,
    conf_mask: np.ndarray,
    phase_rad: np.ndarray | None = None,
    st: dict,
) -> tuple[np.ndarray, dict]:
    policy = HybridIIRPolicy.from_config(cfg)
    if not policy.enabled:
        return gain_db, {
            "hybrid_iir_enabled": False,
            "hybrid_iir_policy_version": int(policy.to_signature_dict()["policy_v"]),
        }
    gd = _array_from_stats(st, "group_delay_ms", "gd_ms", "gd_curve_ms")
    gd_source = "stats"
    if gd.size != np.asarray(freq_axis).size:
        gd = _group_delay_excess_from_phase(freq_axis, phase_rad)
        gd_source = "phase"
    has_gd = bool(gd.size == np.asarray(freq_axis).size)
    if not has_gd:
        policy = replace(policy, min_gd_excess_ms=0.0)
        gd_source = "unavailable"
    try:
        modal = detect_room_modes(
            freq_axis,
            np.asarray(m_anal, dtype=float) - float(calc_offset_db),
            target_mag_db=target_mags,
            corrected_mag_db=None,
            group_delay_ms=gd if has_gd else None,
            confidence_mask=conf_mask,
            lo_hz=float(policy.min_freq_hz),
            hi_hz=float(policy.max_freq_hz),
            min_peak_db=float(policy.min_peak_db),
        )
        result = design_hybrid_iir(
            modal.events,
            np.asarray(freq_axis, dtype=float),
            int(getattr(cfg, "fs", 0) or 0),
            policy,
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
    ) as exc:
        logger.warning("Hybrid IIR design failed; continuing FIR-only", exc_info=True)
        return gain_db, {
            "hybrid_iir_enabled": True,
            "hybrid_iir_error": f"{type(exc).__name__}: {exc}",
            "hybrid_iir_policy_version": int(policy.to_signature_dict()["policy_v"]),
        }

    stats = result.to_stats()
    stats["hybrid_iir_modal_event_count"] = int(getattr(modal, "mode_count", 0) or 0)
    stats["hybrid_iir_gd_source"] = str(gd_source)
    stats["hybrid_iir_min_gd_excess_ms_effective"] = float(policy.min_gd_excess_ms)
    if result.biquads and result.mag_db.size == np.asarray(gain_db).size:
        iir_mag = np.asarray(result.mag_db, dtype=float)
        adjusted_gain = np.asarray(gain_db, dtype=float) - iir_mag
        stats["hybrid_iir_preconditioned_mags"] = (
            np.asarray(m_anal, dtype=float) - float(calc_offset_db) + iir_mag
        ).tolist()
        stats["hybrid_iir_fir_gain_before_db"] = np.asarray(gain_db, dtype=float).tolist()
        stats["hybrid_iir_fir_gain_after_db"] = np.asarray(adjusted_gain, dtype=float).tolist()
        logger.debug("Hybrid IIR enabled: selected %d modal cuts", len(result.biquads))
        return adjusted_gain, stats
    logger.debug("Hybrid IIR enabled: no eligible modal cuts selected")
    return gain_db, stats


def _group_delay_excess_from_phase(freq_axis: np.ndarray, phase_rad: np.ndarray | None) -> np.ndarray:
    try:
        freq = np.asarray(freq_axis, dtype=float).reshape(-1)
        phase = np.asarray(phase_rad, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.asarray([], dtype=float)
    if freq.size < 8 or phase.size != freq.size:
        return np.asarray([], dtype=float)
    valid = np.isfinite(freq) & np.isfinite(phase) & (freq > 0.0)
    if int(np.count_nonzero(valid)) < 8:
        return np.asarray([], dtype=float)
    phase_u = np.unwrap(np.nan_to_num(phase, nan=0.0, posinf=0.0, neginf=0.0))
    omega = 2.0 * np.pi * np.maximum(freq, 1e-9)
    try:
        gd_ms = -np.gradient(phase_u, omega) * 1000.0
    except (TypeError, ValueError, FloatingPointError):
        return np.asarray([], dtype=float)
    gd_ms = np.nan_to_num(gd_ms, nan=0.0, posinf=0.0, neginf=0.0)
    band = valid & (freq >= 20.0) & (freq <= 300.0)
    if int(np.count_nonzero(band)) < 8:
        band = valid
    baseline = float(np.nanmedian(gd_ms[band])) if int(np.count_nonzero(band)) else 0.0
    return np.maximum(0.0, gd_ms - baseline)


def _run_generate_filter_pre_correction(
    freqs,
    meas_mags,
    raw_phases,
    cfg: FilterConfig,
    *,
    stereo_link_ctx: StereoLinkContext | None = None,
    presolve_mode: bool = False,
) -> dict:
    with profiled_section("generate_filter.preprocess"):
        prep = run_preprocess(
            freqs,
            meas_mags,
            raw_phases,
            cfg,
            stereo_link_ctx=stereo_link_ctx,
            presolve_mode=bool(presolve_mode),
        )
    f_in = prep.f_in
    m_in = prep.m_in
    n_fft = prep.ctx.n_fft
    freq_axis = prep.ctx.freq_axis
    gain_db = prep.ctx.gain_db
    st = prep.ctx.st
    p_rad_interp = prep.p_rad_interp
    delay_slope = prep.delay_slope
    m_plot_db = prep.m_plot_db
    complex_meas = prep.complex_meas
    m_anal = prep.m_anal
    conf_mask = prep.conf_mask
    reflections = prep.reflections
    cmp = prep.cmp
    analysis_mode = prep.analysis_mode
    is_psy = prep.is_psy
    with profiled_section("generate_filter.correction"):
        corr = run_correction_stage(
            cfg=cfg,
            freq_axis=freq_axis,
            f_in=f_in,
            m_in=m_in,
            reflections=reflections,
            st=st,
            m_anal=m_anal,
            m_plot_db=m_plot_db,
            is_psy=is_psy,
            cmp=cmp,
            analysis_mode=analysis_mode,
            gain_db=gain_db,
            conf_mask=conf_mask,
            complex_meas=complex_meas,
            logger=logger,
            interpolate_response_fn=interpolate_response,
            apply_confidence_weighted_target_pull_fn=apply_confidence_weighted_target_pull,
            stage_probe_fn=_stage_probe,
            cfg_float_allow_zero_fn=_cfg_float_allow_zero,
            stereo_link_ctx=stereo_link_ctx,
            presolve_mode=bool(presolve_mode),
        )

    _pruning_hook = _get_pruning_hook()
    if callable(_pruning_hook):
        try:
            _g = np.asarray(getattr(corr, "final_g", []), dtype=float)
            _g_fin = _g[np.isfinite(_g)]
            _p90 = float(np.percentile(np.abs(_g_fin), 90)) if _g_fin.size > 0 else 0.0
            _clip_pen = (
                float(getattr(corr, "over_boost", 0.0) or 0.0) * 5.0
                + float(getattr(corr, "over_cut", 0.0) or 0.0) * 2.0
            )
            _pruning_hook(-(_p90 + _clip_pen))
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
        ) as exc:
            if (
                type(exc).__name__ == "TrialPruned"
                and str(getattr(type(exc), "__module__", "")).startswith("optuna")
            ):
                raise
            logger.exception("pruning hook call")

    return {
        "cfg": cfg,
        "n_fft": n_fft,
        "freq_axis": freq_axis,
        "st": st,
        "reflections": reflections,
        "target_mags": corr.target_mags,
        "m_anal": m_anal,
        "conf_mask": conf_mask,
        "cmp": corr.cmp,
        "analysis_mode": corr.analysis_mode,
        "delay_slope": delay_slope,
        "gain_db": corr.gain_db,
        "mask_c": corr.mask_c,
        "base_sigma": corr.base_sigma,
        "filter_smooth": corr.filter_smooth,
        "df_mode": corr.df_mode,
        "raw_g": corr.raw_g,
        "final_g": corr.final_g,
        "p_rad_interp": p_rad_interp,
        "current_rt60": corr.current_rt60,
        "rt60_bands": corr.rt60_bands,
        "band_avg": corr.band_avg,
        "target_level_db": corr.target_level_db,
        "calc_offset_db": corr.calc_offset_db,
        "meas_level_db_window": corr.meas_level_db_window,
        "target_level_db_window": corr.target_level_db_window,
        "offset_method": corr.offset_method,
        "s_min": corr.s_min,
        "s_max": corr.s_max,
        "target_shift_db": corr.target_shift_db,
        "afdw_on": corr.afdw_on,
        "stage_probes": corr.stage_probes,
        "use_bassfirst": corr.use_bassfirst,
        "bf_room_mode": corr.bf_room_mode,
        "bf_rel": corr.bf_rel,
        "bf_conf_for_smoothing": corr.bf_conf_for_smoothing,
        "boost_peak_db": corr.boost_peak_db,
        "cut_peak_db": corr.cut_peak_db,
        "n_boost": corr.n_boost,
        "boost_cand_peak": corr.boost_cand_peak,
        "boost_cand_min_hz": corr.boost_cand_min_hz,
        "n_boost_cand": corr.n_boost_cand,
        "n_boost_cand_low": corr.n_boost_cand_low,
        "n_boost_cand_exc": corr.n_boost_cand_exc,
        "softclip_boost_bins": corr.softclip_boost_bins,
        "softclip_cut_bins": corr.softclip_cut_bins,
        "over_boost": corr.over_boost,
        "over_cut": corr.over_cut,
        "hardclamp_boost_bins": corr.hardclamp_boost_bins,
        "hardclamp_cut_bins": corr.hardclamp_cut_bins,
        "hard_over_boost": corr.hard_over_boost,
        "hard_over_cut": corr.hard_over_cut,
    }


def _run_generate_filter_stereo_link_presolve(
    freqs,
    meas_mags,
    raw_phases,
    cfg: FilterConfig,
    *,
    stereo_link_ctx: StereoLinkContext | None = None,
) -> dict:
    with profiled_section("generate_filter.stereo_link_presolve"):
        state = _run_generate_filter_pre_correction(
            freqs,
            meas_mags,
            raw_phases,
            cfg,
            stereo_link_ctx=stereo_link_ctx,
            presolve_mode=True,
        )

        freq_axis = np.asarray(state["freq_axis"], dtype=float)
        gain_db = np.asarray(state["gain_db"], dtype=float).copy()
        hs = getattr(cfg, "hpf_settings", None)
        if isinstance(hs, dict) and hs.get("enabled"):
            hpf_f = float(hs.get("freq", 0.0) or 0.0)
            hpf_order = int(hs.get("order", 0) or 0)
            if hpf_f > 0 and hpf_order > 0:
                hpf_db = apply_hpf_to_mags(freq_axis, np.zeros_like(freq_axis), hpf_f, hpf_order)
                gain_db = gain_db + hpf_db
                hpf_guard_mask = np.isfinite(freq_axis) & (freq_axis <= float(hpf_f)) & (gain_db > hpf_db)
                if int(np.count_nonzero(hpf_guard_mask)) > 0:
                    gain_db[hpf_guard_mask] = hpf_db[hpf_guard_mask]

        gain_db, _residual_telemetry = apply_residual_pass_if_enabled(
            cfg=cfg,
            freq_axis=freq_axis,
            gain_db=gain_db,
            conf_mask=state["conf_mask"],
            m_anal=np.asarray(state["m_anal"], dtype=float),
            calc_offset_db=float(state["calc_offset_db"]),
            target_mags=np.asarray(state["target_mags"], dtype=float),
            st=state["st"],
            mask_c=np.asarray(state["mask_c"], dtype=bool),
            base_sigma=state["base_sigma"],
            filter_smooth=state["filter_smooth"],
            df_mode=bool(state["df_mode"]),
            raw_g=state["raw_g"],
            final_g=state["final_g"],
            logger=logger,
            cfg_float_allow_zero_fn=_cfg_float_allow_zero,
        )
        ag = compute_auto_gain_and_headroom(
            cfg=cfg,
            gain_db=gain_db,
            mask_c=np.asarray(state["mask_c"], dtype=bool),
            logger=logger,
        )

    return {
        "freq_axis": state["freq_axis"],
        "target_mags": state["target_mags"],
        "m_anal": state["m_anal"],
        "analysis_mode": state["analysis_mode"],
        "target_level_db": state["target_level_db"],
        "calc_offset_db": state["calc_offset_db"],
        "meas_level_db_window": state["meas_level_db_window"],
        "target_level_db_window": state["target_level_db_window"],
        "offset_method": state["offset_method"],
        "s_min": state["s_min"],
        "s_max": state["s_max"],
        "target_shift_db": state["target_shift_db"],
        "current_peak_gain": float(ag["current_peak_gain"]),
        "gain_margin_db": float(ag["gain_margin_db"]),
        "auto_global_gain_db": float(ag["auto_global_gain_db"]),
        "auto_headroom_db": float(ag["auto_headroom_db"]),
    }


def _run_generate_filter_pipeline(
    freqs,
    meas_mags,
    raw_phases,
    cfg: FilterConfig,
    *,
    stereo_link_ctx: StereoLinkContext | None = None,
) -> dict:
    state = _run_generate_filter_pre_correction(
        freqs,
        meas_mags,
        raw_phases,
        cfg,
        stereo_link_ctx=stereo_link_ctx,
        presolve_mode=False,
    )
    n_fft = state["n_fft"]
    freq_axis = state["freq_axis"]
    gain_db = state["gain_db"]
    p_rad_interp = state["p_rad_interp"]
    conf_mask = state["conf_mask"]
    m_anal = state["m_anal"]
    calc_offset_db = state["calc_offset_db"]
    target_mags = state["target_mags"]
    st = state["st"]
    mask_c = state["mask_c"]
    base_sigma = state["base_sigma"]
    _filter_smooth = state["filter_smooth"]
    df_mode = state["df_mode"]
    raw_g = state["raw_g"]
    final_g = state["final_g"]
    use_bassfirst = state["use_bassfirst"]
    afdw_on = state["afdw_on"]

    gain_db, hybrid_iir_stats = _apply_hybrid_iir_preconditioning(
        cfg=cfg,
        freq_axis=freq_axis,
        gain_db=np.asarray(gain_db, dtype=float),
        m_anal=np.asarray(m_anal, dtype=float),
        calc_offset_db=float(calc_offset_db),
        target_mags=np.asarray(target_mags, dtype=float),
        conf_mask=np.asarray(conf_mask, dtype=float),
        phase_rad=np.asarray(p_rad_interp, dtype=float),
        st=st,
    )
    if isinstance(st, dict) and hybrid_iir_stats:
        st.update(hybrid_iir_stats)

    _output_tilt = float(getattr(cfg, "output_tilt_db_per_oct", 0.0) or 0.0)
    if _output_tilt != 0.0:
        _safe_f = np.maximum(freq_axis, 1.0)
        gain_db = gain_db + _output_tilt * np.log2(1000.0 / _safe_f)

    with profiled_section("generate_filter.phase_ir"):
        phase_ir = run_phase_ir_stage(
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
            _filter_smooth=_filter_smooth,
            df_mode=df_mode,
            raw_g=raw_g,
            final_g=final_g,
            use_bassfirst=use_bassfirst,
            afdw_on=afdw_on,
            logger=logger,
            apply_hpf_to_mags_fn=apply_hpf_to_mags,
            apply_lpf_to_mags_fn=apply_lpf_to_mags,
            limit_gd_gradient_ms_per_oct_fn=_limit_gd_gradient_ms_per_oct,
            cfg_float_allow_zero_fn=_cfg_float_allow_zero,
        )

    impulse = phase_ir.impulse
    gain_db = phase_ir.gain_db
    auto_global_gain_db = phase_ir.auto_global_gain_db
    gain_margin_db = phase_ir.gain_margin_db
    auto_headroom_db = phase_ir.auto_headroom_db
    current_peak_gain = phase_ir.current_peak_gain
    final_gain_total = phase_ir.final_gain_total

    return {
        "cfg": cfg,
        "freq_axis": freq_axis,
        "st": state["st"],
        "reflections": state["reflections"],
        "target_mags": target_mags,
        "m_anal": m_anal,
        "conf_mask": conf_mask,
        "cmp": state["cmp"],
        "analysis_mode": state["analysis_mode"],
        "delay_slope": state["delay_slope"],
        "current_rt60": state["current_rt60"],
        "rt60_bands": state["rt60_bands"],
        "band_avg": state["band_avg"],
        "target_level_db": state["target_level_db"],
        "calc_offset_db": calc_offset_db,
        "meas_level_db_window": state["meas_level_db_window"],
        "target_level_db_window": state["target_level_db_window"],
        "offset_method": state["offset_method"],
        "s_min": state["s_min"],
        "s_max": state["s_max"],
        "target_shift_db": state["target_shift_db"],
        "gain_db": gain_db,
        "hybrid_iir_stats": hybrid_iir_stats,
        "afdw_on": state["afdw_on"],
        "mask_c": mask_c,
        "stage_probes": state["stage_probes"],
        "use_bassfirst": state["use_bassfirst"],
        "bf_room_mode": state["bf_room_mode"],
        "bf_rel": state["bf_rel"],
        "bf_conf_for_smoothing": state["bf_conf_for_smoothing"],
        "boost_peak_db": state["boost_peak_db"],
        "cut_peak_db": state["cut_peak_db"],
        "n_boost": state["n_boost"],
        "boost_cand_peak": state["boost_cand_peak"],
        "boost_cand_min_hz": state["boost_cand_min_hz"],
        "n_boost_cand": state["n_boost_cand"],
        "n_boost_cand_low": state["n_boost_cand_low"],
        "n_boost_cand_exc": state["n_boost_cand_exc"],
        "softclip_boost_bins": state["softclip_boost_bins"],
        "softclip_cut_bins": state["softclip_cut_bins"],
        "over_boost": state["over_boost"],
        "over_cut": state["over_cut"],
        "hardclamp_boost_bins": state["hardclamp_boost_bins"],
        "hardclamp_cut_bins": state["hardclamp_cut_bins"],
        "hard_over_boost": state["hard_over_boost"],
        "hard_over_cut": state["hard_over_cut"],
        "impulse": impulse,
        "auto_global_gain_db": auto_global_gain_db,
        "gain_margin_db": gain_margin_db,
        "auto_headroom_db": auto_headroom_db,
        "current_peak_gain": current_peak_gain,
        "final_gain_total": final_gain_total,
    }
