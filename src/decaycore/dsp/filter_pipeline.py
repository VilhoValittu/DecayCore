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
from .dsp_stats import (
    apply_afdw_stats,
    apply_boost_blocked_reason,
    apply_clamp_stats,
    apply_lf_guard_stats,
    apply_measured_mag_stats,
    arr_if_valid_for_stats,
    safe_stage_probes,
    safe_stats_update,
)
from .dsp_utils import cfg_float_allow_zero as _cfg_float_allow_zero
from .phase_ir_autogain import compute_auto_gain_and_headroom
from .phase_ir_residual import apply_residual_pass_if_enabled

logger = logging.getLogger("DecayCore.dsp")


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
    target_mags = prep.ctx.target_mags
    m_interp = prep.m_interp
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
        except Exception as exc:
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
                hpf_guard_mask = np.isfinite(freq_axis) & (freq_axis <= float(hpf_f)) & (gain_db > 0.0)
                if int(np.count_nonzero(hpf_guard_mask)) > 0:
                    gain_db[hpf_guard_mask] = 0.0

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
