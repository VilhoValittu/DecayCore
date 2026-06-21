# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

import copy as copy
import logging

import numpy as np
logger = logging.getLogger("DecayCore.dsp")
from decaycore.auto_mode.auto_mode_profile import profiled_section
from decaycore.config.models import FilterConfig
from ..acoustic_authority import acoustic_authority_to_stats, build_acoustic_authority_map
from ..dsp_correction import run_correction_stage as run_correction_stage
from ..dsp_ops import (
    _limit_gd_gradient_ms_per_oct as _limit_gd_gradient_ms_per_oct,
    _stage_probe as _stage_probe,
    apply_confidence_weighted_target_pull as apply_confidence_weighted_target_pull,
    apply_hpf_to_mags as apply_hpf_to_mags,
    interpolate_response,
)
from ..dsp_phase_ir import run_phase_ir_stage as run_phase_ir_stage
from ..dsp_preprocess import run_preprocess as run_preprocess
from ..dsp_stats import (
    apply_afdw_stats,
    apply_boost_blocked_reason,
    apply_clamp_stats,
    apply_lf_guard_stats,
    apply_measured_mag_stats,
    arr_if_valid_for_stats as arr_if_valid_for_stats,
    safe_stage_probes,
    safe_stats_update,
)
from ..decaycore_leveling import StereoLinkContext, find_shared_stereo_level_window as find_shared_stereo_level_window
from ..dsp_utils import cfg_float_allow_zero as _cfg_float_allow_zero
from ..filter_pipeline import (
    _run_generate_filter_pipeline,
    _run_generate_filter_stereo_link_presolve,
)
from ..filter_result import _assemble_generate_filter_result

def _run_generate_filter_stereo_link_presolve_stats(
    freqs,
    meas_mags,
    raw_phases,
    cfg: FilterConfig,
) -> dict:
    presolve = _run_generate_filter_stereo_link_presolve(freqs, meas_mags, raw_phases, cfg)
    freq_axis = np.asarray(presolve["freq_axis"], dtype=float)
    target_mags = np.asarray(presolve["target_mags"], dtype=float)
    m_anal = np.asarray(presolve["m_anal"], dtype=float)
    calc_offset_db = float(presolve["calc_offset_db"])
    return {
        "analysis_mode": str(presolve["analysis_mode"]),
        "freq_axis": freq_axis,
        "target_mags": target_mags,
        "measured_mags": (m_anal - calc_offset_db),
        "smart_scan_range": (float(presolve["s_min"]), float(presolve["s_max"])),
        "eff_target_db": float(presolve["target_level_db"]),
        "offset_db": float(calc_offset_db),
        "meas_level_db_window": float(presolve["meas_level_db_window"]),
        "target_level_db_window": float(presolve["target_level_db_window"]),
        "offset_method": str(presolve["offset_method"]),
        "target_shift_db": float(presolve["target_shift_db"]),
        "tilt_slope_db_per_oct": (
            float(getattr(cfg, "_lvl_tilt_slope_db_per_oct"))
            if getattr(cfg, "_lvl_tilt_slope_db_per_oct", None) is not None
            else None
        ),
        "gain_margin_db": float(presolve["gain_margin_db"]),
        "auto_global_gain_db": float(presolve["auto_global_gain_db"]),
        "auto_headroom_db": float(presolve["auto_headroom_db"]),
        "peak_gain_db": float(presolve["current_peak_gain"]),
    }

def _normalize_impulse_if_requested(impulse: np.ndarray, cfg: FilterConfig) -> tuple[float, float]:
    max_peak = np.max(np.abs(impulse))
    if not (cfg.do_normalize and max_peak > 0):
        return float(max_peak), 0.0
    norm_scale = float(0.89 / max_peak)
    impulse *= norm_scale
    try:
        return float(max_peak), float(20.0 * np.log10(max(norm_scale, 1e-12)))
    except (TypeError, ValueError, FloatingPointError):
        return float(max_peak), 0.0


def _add_response_payload_fields(
    stats: dict,
    *,
    include_response_arrays: bool,
    m_anal: np.ndarray,
    gain_db: np.ndarray,
    mask_c: np.ndarray,
) -> None:
    if not bool(include_response_arrays):
        return
    stats["measured_mags_raw"] = np.asarray(m_anal, dtype=float).tolist()
    stats["fir_only_predicted_filter_mags"] = np.asarray(gain_db, dtype=float).tolist()
    stats["mag_mask"] = np.asarray(mask_c, dtype=float).tolist()


def _authority_array_mode(include_response_arrays: bool) -> bool | str:
    return True if bool(include_response_arrays) else "scoring"


def _strip_ui_authority_arrays_for_score_only(stats: dict, *, include_response_arrays: bool) -> None:
    if bool(include_response_arrays):
        return
    for key in (
        "authority_cut",
        "authority_boost",
        "authority_phase",
        "authority_reflection_risk",
        "authority_repeatability",
        "authority_minphase_likelihood",
    ):
        stats.pop(key, None)


def generate_filter(  # noqa: C901 - single-channel pipeline keeps policy, limits, and stats in one place
    freqs,
    meas_mags,
    raw_phases,
    cfg: FilterConfig,
    *,
    stereo_link_ctx: StereoLinkContext | None = None,
    include_response_arrays: bool = True,
):
    with profiled_section("generate_filter.pipeline"):
        pipeline = _run_generate_filter_pipeline(
            freqs,
            meas_mags,
            raw_phases,
            cfg,
            stereo_link_ctx=stereo_link_ctx,
        )
    freq_axis = pipeline["freq_axis"]
    st = pipeline["st"]
    reflections = pipeline["reflections"]
    target_mags = pipeline["target_mags"]
    m_anal = pipeline["m_anal"]
    conf_mask = pipeline["conf_mask"]
    cmp = pipeline["cmp"]
    analysis_mode = pipeline["analysis_mode"]
    delay_slope = pipeline["delay_slope"]
    current_rt60 = pipeline["current_rt60"]
    rt60_bands = pipeline["rt60_bands"]
    band_avg = pipeline["band_avg"]
    target_level_db = pipeline["target_level_db"]
    calc_offset_db = pipeline["calc_offset_db"]
    meas_level_db_window = pipeline["meas_level_db_window"]
    target_level_db_window = pipeline["target_level_db_window"]
    offset_method = pipeline["offset_method"]
    s_min = pipeline["s_min"]
    s_max = pipeline["s_max"]
    target_shift_db = pipeline["target_shift_db"]
    gain_db = pipeline["gain_db"]
    hybrid_iir_stats = dict(pipeline.get("hybrid_iir_stats", {}) or {})
    hybrid_iir_mag_db = np.asarray(hybrid_iir_stats.get("hybrid_iir_mag_db", []), dtype=float).reshape(-1)
    hybrid_iir_active = bool(hybrid_iir_stats.get("hybrid_iir_enabled", False)) and hybrid_iir_mag_db.size == np.asarray(gain_db).size
    combined_gain_db = (
        np.asarray(gain_db, dtype=float) + hybrid_iir_mag_db
        if hybrid_iir_active
        else np.asarray(gain_db, dtype=float)
    )
    afdw_on = pipeline["afdw_on"]
    mask_c = pipeline["mask_c"]
    stage_probes = pipeline["stage_probes"]
    use_bassfirst = pipeline["use_bassfirst"]
    bf_room_mode = pipeline["bf_room_mode"]
    bf_rel = pipeline["bf_rel"]
    bf_conf_for_smoothing = pipeline["bf_conf_for_smoothing"]
    boost_peak_db = pipeline["boost_peak_db"]
    cut_peak_db = pipeline["cut_peak_db"]
    n_boost = pipeline["n_boost"]
    boost_cand_peak = pipeline["boost_cand_peak"]
    boost_cand_min_hz = pipeline["boost_cand_min_hz"]
    n_boost_cand = pipeline["n_boost_cand"]
    n_boost_cand_low = pipeline["n_boost_cand_low"]
    n_boost_cand_exc = pipeline["n_boost_cand_exc"]
    softclip_boost_bins = pipeline["softclip_boost_bins"]
    softclip_cut_bins = pipeline["softclip_cut_bins"]
    over_boost = pipeline["over_boost"]
    over_cut = pipeline["over_cut"]
    hardclamp_boost_bins = pipeline["hardclamp_boost_bins"]
    hardclamp_cut_bins = pipeline["hardclamp_cut_bins"]
    hard_over_boost = pipeline["hard_over_boost"]
    hard_over_cut = pipeline["hard_over_cut"]
    impulse = pipeline["impulse"]
    auto_global_gain_db = pipeline["auto_global_gain_db"]
    gain_margin_db = pipeline["gain_margin_db"]
    auto_headroom_db = pipeline["auto_headroom_db"]
    current_peak_gain = pipeline["current_peak_gain"]
    final_gain_total = pipeline["final_gain_total"]

    max_peak, normalize_gain_db_applied = _normalize_impulse_if_requested(impulse, cfg)

    stats = {

        'analysis_mode': analysis_mode,
        'freq_axis': freq_axis.tolist(),
        'mag_c_min': float(getattr(cfg, 'mag_c_min', 0.0) or 0.0),
        'mag_c_max': float(getattr(cfg, 'mag_c_max', 0.0) or 0.0),
        'target_mags': target_mags.tolist(),
        'measured_mags': (m_anal - calc_offset_db).tolist(),
        'predicted_filter_mags': combined_gain_db.tolist(),
        'predicted_filter_mags_source': "hybrid_iir_plus_mag_post_limits_pre_ir" if hybrid_iir_active else "mag_post_limits_pre_ir",
        'filter_mags': combined_gain_db.tolist(),
        'filter_mags_source': "hybrid_iir_plus_mag_post_limits_pre_ir" if hybrid_iir_active else "mag_post_limits_pre_ir",
        'confidence_mask': conf_mask.tolist(),
        'afdw_active': bool(afdw_on),
        'reflections': reflections,
        'smart_scan_range': [float(s_min), float(s_max)],
        'eff_target_db': float(target_level_db),
        'offset_db': float(calc_offset_db),
        'meas_level_db_window': float(meas_level_db_window),
        'target_level_db_window': float(target_level_db_window),
        'offset_method': str(offset_method),
            'tilt_slope_db_per_oct': (
                float(getattr(cfg, "_lvl_tilt_slope_db_per_oct"))
                if getattr(cfg, "_lvl_tilt_slope_db_per_oct", None) is not None
                else None
            ),
        'rt60_val': float(current_rt60),
        'rt60_band_avg': float(band_avg),
        'rt60_bands': rt60_bands,
        'avg_confidence': float(np.mean(conf_mask)*100),
        'delay_samples': float((delay_slope * cfg.fs) / (2 * np.pi)) if 'delay_slope' in locals() else 0.0,
        'peak_before_norm': float(20*np.log10(max_peak + 1e-12)),
        'do_normalize': bool(getattr(cfg, 'do_normalize', False)),
        'gain_margin_db': float(gain_margin_db),
        'auto_global_gain_db': float(auto_global_gain_db),
        'auto_headroom_db': float(auto_headroom_db),
        'normalize_gain_db_applied': float(normalize_gain_db_applied),
        'peak_gain_db': float(current_peak_gain),
        'final_max_db': float(np.max(final_gain_total)),

        'max_boost_db': float(getattr(cfg, 'max_boost_db', 0.0) or 0.0),
        'max_boost_db_effective': float(getattr(cfg, 'max_boost_db', 0.0) or 0.0),
        'max_boost_db_user': float(getattr(cfg, 'max_boost_db_user', getattr(cfg, 'max_boost_db', 0.0)) or 0.0),
        'max_safe_boost_db': float(getattr(cfg, 'max_safe_boost_db', 0.0) or 0.0),
        'max_cut_db': float(abs(float(getattr(cfg, 'max_cut_db', 15.0) or 15.0))),
        'low_bass_cut_hz': _cfg_float_allow_zero(cfg, "low_bass_cut_hz", 40.0),
        'exc_prot': bool(getattr(cfg, 'exc_prot', False)),
        'exc_freq': float(getattr(cfg, 'exc_freq', 0.0) or 0.0),
        'max_slope_db_per_oct': float(getattr(cfg, 'max_slope_db_per_oct', 0.0) or 0.0),
        'max_slope_boost_db_per_oct': float(getattr(cfg, 'max_slope_boost_db_per_oct', 0.0) or 0.0),
        'max_slope_cut_db_per_oct': float(getattr(cfg, 'max_slope_cut_db_per_oct', 0.0) or 0.0),


        'boost_peak_db': float(boost_peak_db or 0.0),
        'cut_peak_db': float(cut_peak_db or 0.0),
        'boost_bins': int(n_boost or 0),
        'boost_candidate_peak_db': float(boost_cand_peak or 0.0),
        'boost_candidate_min_hz': float(boost_cand_min_hz if boost_cand_min_hz is not None else float("nan")),
        'boost_candidate_bins': int(n_boost_cand or 0),
        'boost_candidate_bins_lowbass': int(n_boost_cand_low or 0),
        'boost_candidate_bins_excprot': int(n_boost_cand_exc or 0),


        'bass_first_ai': bool(use_bassfirst),

        'bass_first_mode_peak_hz': (
            float(freq_axis[int(np.argmax(np.asarray(bf_room_mode)))])
            if (
                bool(use_bassfirst)
                and (bf_room_mode is not None)
                and len(np.asarray(bf_room_mode)) > 0
            )
            else None
        ),

        'bass_first_mode_peak_score': (
            float(np.max(np.asarray(bf_room_mode)))
            if (
                bool(use_bassfirst)
                and (bf_room_mode is not None)
                and len(np.asarray(bf_room_mode)) > 0
            )
            else None
        ),

        'bass_first_conf_floor_applied': (
            bool(
                bool(use_bassfirst)
                and (bf_conf_for_smoothing is not None)
                and (bf_rel is not None)
                and np.any(
                    np.asarray(bf_conf_for_smoothing) > (np.asarray(bf_rel) + 1e-6)
                )
            )
            if (
                bf_conf_for_smoothing is not None
                and bf_rel is not None
            )
            else False
        ),


        'bass_first_rel_mean_20_200': (
            float(np.mean(np.asarray(bf_rel)[(freq_axis >= 20.0) & (freq_axis <= 200.0)]))
            if (
                bool(use_bassfirst)
                and (bf_rel is not None)
                and np.any((freq_axis >= 20.0) & (freq_axis <= 200.0))
            )
            else None
        ),

        'bass_first_rel_min_20_200': (
            float(np.min(np.asarray(bf_rel)[(freq_axis >= 20.0) & (freq_axis <= 200.0)]))
            if (
                bool(use_bassfirst)
                and (bf_rel is not None)
                and np.any((freq_axis >= 20.0) & (freq_axis <= 200.0))
            )
            else None
        ),

        'bass_first_conf_eff_mean_20_200': (
            float(np.mean(np.asarray(bf_conf_for_smoothing)[(freq_axis >= 20.0) & (freq_axis <= 200.0)]))
            if (
                bool(use_bassfirst)
                and (bf_conf_for_smoothing is not None)
                and np.any((freq_axis >= 20.0) & (freq_axis <= 200.0))
            )
            else None
        ),

        'bass_first_conf_eff_min_20_200': (
            float(np.min(np.asarray(bf_conf_for_smoothing)[(freq_axis >= 20.0) & (freq_axis <= 200.0)]))
            if (
                bool(use_bassfirst)
                and (bf_conf_for_smoothing is not None)
                and np.any((freq_axis >= 20.0) & (freq_axis <= 200.0))
            )
            else None
        ),

        'bass_first_roommode_max_20_200': (
            float(np.max(np.asarray(bf_room_mode)[(freq_axis >= 20.0) & (freq_axis <= 200.0)]))
            if (
                bool(use_bassfirst)
                and (bf_room_mode is not None)
                and np.any((freq_axis >= 20.0) & (freq_axis <= 200.0))
            )
            else None
        ),

    }

    try:
        cfg_house_freqs = np.asarray(getattr(cfg, "house_freqs", []), dtype=float).reshape(-1)
        cfg_house_mags = np.asarray(getattr(cfg, "house_mags", []), dtype=float).reshape(-1)
        if cfg_house_freqs.size >= 2 and cfg_house_mags.size == cfg_house_freqs.size:
            stats["selected_target_mags"] = interpolate_response(
                cfg_house_freqs,
                cfg_house_mags,
                np.asarray(freq_axis, dtype=float),
            ).tolist()
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
        logger.debug("selected target trace export skipped", exc_info=True)

    _add_response_payload_fields(
        stats,
        include_response_arrays=bool(include_response_arrays),
        m_anal=m_anal,
        gain_db=gain_db,
        mask_c=mask_c,
    )

    safe_stats_update(stats, st)
    if hybrid_iir_stats:
        safe_stats_update(stats, hybrid_iir_stats)
    # Keep target on a shared absolute reference level for scoring/quality.
    # Preserve measured arrays from `st` when available to keep UI view
    # behavior stable; only fill missing measured fields.
    apply_measured_mag_stats(
        stats,
        target_mags=target_mags,
        freq_axis=freq_axis,
        m_anal=m_anal,
        calc_offset_db=float(calc_offset_db),
        include_raw=bool(include_response_arrays),
    )
    apply_afdw_stats(
        stats,
        afdw_on=bool(afdw_on),
        afdw_bw_oct=pipeline.get("afdw_bw_oct"),
        afdw_bw_min_oct=pipeline.get("afdw_bw_min_oct"),
        afdw_bw_mean_oct=pipeline.get("afdw_bw_mean_oct"),
        afdw_bw_max_oct=pipeline.get("afdw_bw_max_oct"),
        afdw_bw_min_hz=pipeline.get("afdw_bw_min_hz"),
        afdw_bw_max_hz=pipeline.get("afdw_bw_max_hz"),
    )
    stats["stage_probes"] = safe_stage_probes(stage_probes)
    apply_lf_guard_stats(stats, cfg=cfg, freq_axis=freq_axis, gain_db=gain_db)
    apply_clamp_stats(
        stats,
        softclip_boost_bins=int(softclip_boost_bins),
        softclip_cut_bins=int(softclip_cut_bins),
        over_boost=float(over_boost),
        over_cut=float(over_cut),
        hardclamp_boost_bins=int(hardclamp_boost_bins),
        hardclamp_cut_bins=int(hardclamp_cut_bins),
        hard_over_boost=float(hard_over_boost),
        hard_over_cut=float(hard_over_cut),
    )
    apply_boost_blocked_reason(stats, cfg=cfg)

    try:
        authority_gd_ms = stats.get("group_delay_ms", None)
        if authority_gd_ms is None:
            authority_gd_ms = stats.get("gd_ms", None)
        authority = build_acoustic_authority_map(
            freq_axis,
            m_anal - calc_offset_db,
            target_mag_db=target_mags,
            corrected_mag_db=gain_db,
            confidence_mask=conf_mask,
            group_delay_ms=authority_gd_ms,
            reflection_nodes=reflections,
            rt60_by_band=rt60_bands,
            mag_c_min=float(getattr(cfg, "mag_c_min", 20.0) or 20.0),
            mag_c_max=float(getattr(cfg, "mag_c_max", 300.0) or 300.0),
            phase_limit_hz=float(getattr(cfg, "phase_c_max", 600.0) or 600.0),
        )
        stats.update(acoustic_authority_to_stats(authority, include_arrays=_authority_array_mode(bool(include_response_arrays))))
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
        logger.warning("acoustic authority map failed", exc_info=True)
        stats["acoustic_authority_error"] = str(exc)

    if isinstance(cmp, dict) and cmp:
        stats.update(cmp)
        if stats.get('analysis_mode') != "comparison":
            stats['analysis_mode'] = "native"
    try:
        cmp_g_pred = np.asarray(stats.get("cmp_predicted_filter_mags", []), dtype=float).reshape(-1)
        cmp_g_cur = np.asarray(stats.get("cmp_filter_mags", []), dtype=float).reshape(-1)
        if cmp_g_pred.size < 8 and cmp_g_cur.size >= 8:
            stats["cmp_predicted_filter_mags"] = cmp_g_cur.tolist()
            stats["cmp_predicted_filter_mags_source"] = str(
                stats.get("cmp_filter_mags_source", "mag_post_limits_pre_ir") or "mag_post_limits_pre_ir"
            )
    except (TypeError, ValueError):
        pass
    try:
        cmp_m_raw = np.asarray(stats.get("cmp_measured_mags_raw", []), dtype=float).reshape(-1)
        cmp_m_cur = np.asarray(stats.get("cmp_measured_mags", []), dtype=float).reshape(-1)
        if cmp_m_raw.size < 8 and cmp_m_cur.size >= 8:
            cmp_off = float(stats.get("cmp_offset_db", 0.0) or 0.0)
            stats["cmp_measured_mags_raw"] = (cmp_m_cur + cmp_off).tolist()
    except (TypeError, ValueError):
        pass

    # Canonical filter magnitude for reporting: always derive from final IR.
    # This keeps DSP Quality Report aligned with the exported/used filter.
    try:
        with profiled_section("generate_filter.ir_realization_stats"):
            ir = np.asarray(impulse, dtype=float).flatten()
            fs_i = int(getattr(cfg, "fs", 0) or 0)
            f_native = np.asarray(stats.get("freq_axis", freq_axis), dtype=float).reshape(-1)
            f_q = np.asarray([], dtype=float)
            if ir.size >= 8 and fs_i > 0 and f_native.size >= 4:
                h = np.fft.rfft(ir)
                f_fft = np.fft.rfftfreq(ir.size, d=1.0 / float(fs_i))
                g_db = 20.0 * np.log10(np.maximum(np.abs(h), 1e-12))
                f_q = np.clip(f_native, float(np.min(f_fft)), float(np.max(f_fft)))
                g_pred_native = np.asarray(stats.get("predicted_filter_mags", []), dtype=float).reshape(-1)
                if g_pred_native.size < 8:
                    g_cur_native = np.asarray(stats.get("filter_mags", []), dtype=float).reshape(-1)
                    if g_cur_native.size >= 8:
                        stats["predicted_filter_mags"] = g_cur_native.tolist()
                        stats["predicted_filter_mags_source"] = str(
                            stats.get("filter_mags_source", "mag_post_limits_pre_ir") or "mag_post_limits_pre_ir"
                        )

                g_real_native_fir = np.interp(f_q, f_fft, g_db)
                if hybrid_iir_active:
                    iir_native = np.asarray(hybrid_iir_mag_db, dtype=float).reshape(-1)
                    iir_eval = iir_native[: g_real_native_fir.size]
                    g_real_native = g_real_native_fir + iir_eval
                    if bool(include_response_arrays):
                        stats["fir_only_realized_filter_mags"] = g_real_native_fir.tolist()
                    stats["fir_only_filter_mags"] = g_real_native_fir.tolist()
                else:
                    g_real_native = g_real_native_fir
                if bool(include_response_arrays):
                    stats["realized_filter_mags"] = g_real_native.tolist()
                    stats["realized_filter_mags_source"] = "hybrid_iir_plus_ir_fft_final" if hybrid_iir_active else "ir_fft_final"
                stats["filter_mags"] = g_real_native.tolist()
                stats["filter_mags_source"] = "hybrid_iir_plus_ir_fft_final" if hybrid_iir_active else "ir_fft_final"

                f_cmp = np.asarray(stats.get("cmp_freq_axis", []), dtype=float).reshape(-1)
                if f_cmp.size >= 4:
                    f_cmp_q = np.clip(f_cmp, float(np.min(f_fft)), float(np.max(f_fft)))
                    g_pred_cmp = np.asarray(stats.get("cmp_predicted_filter_mags", []), dtype=float).reshape(-1)
                    if g_pred_cmp.size < 4:
                        g_cur_cmp = np.asarray(stats.get("cmp_filter_mags", []), dtype=float).reshape(-1)
                        if g_cur_cmp.size >= 4:
                            stats["cmp_predicted_filter_mags"] = g_cur_cmp.tolist()
                            stats["cmp_predicted_filter_mags_source"] = str(
                                stats.get("cmp_filter_mags_source", "mag_post_limits_pre_ir") or "mag_post_limits_pre_ir"
                            )
                    g_real_cmp = np.interp(f_cmp_q, f_fft, g_db)
                    if bool(include_response_arrays):
                        stats["cmp_realized_filter_mags"] = g_real_cmp.tolist()
                        stats["cmp_realized_filter_mags_source"] = "ir_fft_final"
                    stats["cmp_filter_mags"] = g_real_cmp.tolist()
                    stats["cmp_filter_mags_source"] = "ir_fft_final"

            # Realization delta diagnostics:
            # compare post-limits mag gain_db against final IR-derived filter_mags.
            try:
                g_post = np.asarray(gain_db, dtype=float).reshape(-1)
                g_ir = np.asarray(stats.get("filter_mags", []), dtype=float).reshape(-1)
                n = int(min(f_q.size, g_post.size, g_ir.size))
                if n >= 8:
                    f_eval = np.asarray(f_q[:n], dtype=float)
                    d_eval = np.asarray(g_ir[:n], dtype=float) - np.asarray(g_post[:n], dtype=float)
                    valid = np.isfinite(f_eval) & np.isfinite(d_eval) & (f_eval > 0.0)

                    def _band_delta_on(d_arr: np.ndarray, lo_hz: float, hi_hz: float):
                        m = valid & (f_eval >= float(lo_hz)) & (f_eval <= float(hi_hz))
                        if int(np.count_nonzero(m)) < 8:
                            return None, None, None
                        dv = np.asarray(d_arr[m], dtype=float)
                        fv = f_eval[m]
                        idx = int(np.argmax(np.abs(dv)))
                        return float(np.sqrt(np.mean(dv * dv))), float(np.abs(dv[idx])), float(fv[idx])

                    rms_b, max_b, hz_b = _band_delta_on(d_eval, 20.0, 200.0)
                    stats["post_to_ir_delta_rms_20_200_db"] = rms_b
                    stats["post_to_ir_delta_max_20_200_db"] = max_b
                    stats["post_to_ir_delta_max_hz_20_200"] = hz_b

                    m20 = valid & (f_eval >= 20.0) & (f_eval <= 200.0)
                    if int(np.count_nonzero(m20)) >= 8:
                        off20 = float(np.median(np.asarray(d_eval[m20], dtype=float)))
                        d_shape_20 = np.asarray(d_eval, dtype=float) - float(off20)
                        srms20, smax20, shz20 = _band_delta_on(d_shape_20, 20.0, 200.0)
                        stats["post_to_ir_delta_offset_20_200_db"] = float(off20)
                        stats["post_to_ir_shape_delta_rms_20_200_db"] = srms20
                        stats["post_to_ir_shape_delta_max_20_200_db"] = smax20
                        stats["post_to_ir_shape_delta_max_hz_20_200"] = shz20

                        # Same diagnostic, but baseline includes gain staging
                        # (auto gain/headroom + possible final normalize scale).
                        g_stage = (
                            np.asarray(g_post, dtype=float)
                            + float(auto_global_gain_db)
                            + float(auto_headroom_db)
                            + float(normalize_gain_db_applied)
                        )
                        d_stage = np.asarray(g_ir[:n], dtype=float) - np.asarray(g_stage[:n], dtype=float)
                        srms_abs, smax_abs, shz_abs = _band_delta_on(d_stage, 20.0, 200.0)
                        stats["post_to_ir_staged_delta_rms_20_200_db"] = srms_abs
                        stats["post_to_ir_staged_delta_max_20_200_db"] = smax_abs
                        stats["post_to_ir_staged_delta_max_hz_20_200"] = shz_abs

                        off_stage = float(np.median(np.asarray(d_stage[m20], dtype=float)))
                        d_stage_shape = np.asarray(d_stage, dtype=float) - float(off_stage)
                        srms_shape, smax_shape, shz_shape = _band_delta_on(d_stage_shape, 20.0, 200.0)
                        stats["post_to_ir_staged_delta_offset_20_200_db"] = float(off_stage)
                        stats["post_to_ir_staged_shape_delta_rms_20_200_db"] = srms_shape
                        stats["post_to_ir_staged_shape_delta_max_20_200_db"] = smax_shape
                        stats["post_to_ir_staged_shape_delta_max_hz_20_200"] = shz_shape

                    cmin = float(stats.get("mag_c_min", getattr(cfg, "mag_c_min", 20.0)) or 20.0)
                    cmax = float(stats.get("mag_c_max", getattr(cfg, "mag_c_max", 20000.0)) or 20000.0)
                    if (not np.isfinite(cmin)) or (cmin < 0.0):
                        cmin = 20.0
                    if (not np.isfinite(cmax)) or (cmax <= cmin):
                        cmax = max(200.0, cmin + 1.0)
                    rms_c, max_c, hz_c = _band_delta_on(d_eval, cmin, cmax)
                    stats["post_to_ir_delta_rms_magc_db"] = rms_c
                    stats["post_to_ir_delta_max_magc_db"] = max_c
                    stats["post_to_ir_delta_max_hz_magc"] = hz_c
                    m_c = valid & (f_eval >= float(cmin)) & (f_eval <= float(cmax))
                    if int(np.count_nonzero(m_c)) >= 8:
                        off_c = float(np.median(np.asarray(d_eval[m_c], dtype=float)))
                        d_shape_c = np.asarray(d_eval, dtype=float) - float(off_c)
                        srms_c, smax_c, shz_c = _band_delta_on(d_shape_c, cmin, cmax)
                        stats["post_to_ir_delta_offset_magc_db"] = float(off_c)
                        stats["post_to_ir_shape_delta_rms_magc_db"] = srms_c
                        stats["post_to_ir_shape_delta_max_magc_db"] = smax_c
                        stats["post_to_ir_shape_delta_max_hz_magc"] = shz_c
            except (TypeError, ValueError, FloatingPointError, IndexError):
                pass
    except (TypeError, ValueError, FloatingPointError, IndexError, KeyError):
        pass

    _strip_ui_authority_arrays_for_score_only(stats, include_response_arrays=bool(include_response_arrays))
    return _assemble_generate_filter_result(impulse, stats)


__all__ = [
    '_run_generate_filter_stereo_link_presolve_stats',
    '_normalize_impulse_if_requested',
    'generate_filter',
    '_limit_gd_gradient_ms_per_oct',
    'apply_confidence_weighted_target_pull',
]

