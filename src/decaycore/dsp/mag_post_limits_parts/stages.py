# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Named magnitude post-limit stages used by the pipeline orchestrator."""

from __future__ import annotations

import numpy as np
import scipy.ndimage

from ..decaycore_analysis import _sigma_bins_from_hz
from ..dsp_config import CfgReader
from ..dsp_telemetry import safe_put_many
from ..gain_policy import apply_cuts_only_guard, build_low_frequency_guard_mask
from ..mag_authority_trace import (
    REASON_ACOUSTIC_AUTHORITY_BOOST_CAP,
    REASON_ACOUSTIC_AUTHORITY_CUT_CAP,
    REASON_BASS_BOOST_RESTORE,
    REASON_CONFIDENCE_PULL,
    REASON_EXCURSION_FULL_BLOCK,
    REASON_EXCURSION_SOFT_CAP,
    REASON_HARDCLAMP_BOOST,
    REASON_HARDCLAMP_CUT,
    REASON_LOW_BASS_CUTS_ONLY,
    REASON_LOW_BASS_FLOOR_REAPPLIED,
    REASON_MIN_BOOST_PEAK,
    REASON_REGULARIZATION_SMOOTH,
    REASON_SLOPE_LIMIT,
    REASON_SOFTCLIP_BOOST,
    REASON_SOFTCLIP_CUT,
    REASON_TRANSITION_FADE,
    REASON_USER_BOOST_CAP,
    REASON_USER_CUT_CAP,
    REASON_WAV_TRANSITION_SMOOTH,
    append_mag_authority_stage,
)
from ..mag_limits import (
    _apply_hard_boost_cut_clamp,
    _apply_slope_limits,
    _blend_masked_fractional_octave,
)
from ..mag_postprocess import apply_bass_boost_post_restore, apply_confpull_post_slope
from ..mag_telemetry import (
    _log_stage_stats,
)
from ..phase_ir_utils import _cosine_fade_out_01
from ..smoothing import smooth_gain_fractional_octave

from .clamps import (
    _apply_hard_clamps,
    _apply_soft_clamps,
)


def _suppress_subthreshold_local_detail_lobes(
    gain_db: np.ndarray,
    freq_axis: np.ndarray,
    mask_c: np.ndarray,
    *,
    min_peak_db: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Suppress inaudible local filter detail above 100 Hz without flattening broad response."""
    gain = np.asarray(gain_db, dtype=float)
    freq = np.asarray(freq_axis, dtype=float)
    mask = np.asarray(mask_c, dtype=bool)
    if (
        gain.shape != mask.shape
        or freq.shape != gain.shape
        or not np.all(np.isfinite(freq))
        or not np.all(np.diff(freq) > 0.0)
        or not np.isfinite(min_peak_db)
        or min_peak_db <= 0.0
    ):
        return gain, {
            "threshold_db": float(max(min_peak_db, 0.0)) if np.isfinite(min_peak_db) else 0.0,
            "suppressed_lobes": 0,
            "suppressed_bins": 0,
            "suppressed_peak_max_db": 0.0,
        }

    if np.ptp(gain[mask]) <= 1e-9:
        return gain, {
            "threshold_db": float(min_peak_db),
            "suppressed_lobes": 0,
            "suppressed_bins": 0,
            "suppressed_peak_max_db": 0.0,
        }

    baseline = smooth_gain_fractional_octave(freq, gain, 6.0)
    detail = gain - baseline
    active = mask & (freq >= 80.0) & np.isfinite(detail) & (np.abs(detail) > 1e-9)
    if not np.any(active):
        return gain, {
            "threshold_db": float(min_peak_db),
            "suppressed_lobes": 0,
            "suppressed_bins": 0,
            "suppressed_peak_max_db": 0.0,
        }

    sign = np.sign(detail)
    same_lobe_as_previous = np.concatenate(
        (np.array([False]), active[:-1] & (sign[:-1] == sign[1:])),
    )
    starts = active & ~same_lobe_as_previous
    lobe_ids = np.cumsum(starts, dtype=int) - 1
    active_ids = lobe_ids[active]
    peak_by_lobe = np.zeros(int(np.max(active_ids)) + 1, dtype=float)
    np.maximum.at(peak_by_lobe, active_ids, np.abs(detail[active]))
    suppressed_lobes = peak_by_lobe < float(min_peak_db)
    suppress = active & suppressed_lobes[lobe_ids]

    if not np.any(suppress):
        return gain, {
            "threshold_db": float(min_peak_db),
            "suppressed_lobes": 0,
            "suppressed_bins": 0,
            "suppressed_peak_max_db": 0.0,
        }

    fade = np.clip((freq - 80.0) / 20.0, 0.0, 1.0)
    fade = fade * fade * (3.0 - 2.0 * fade)
    out = gain.copy()
    out[suppress] = gain[suppress] + fade[suppress] * (baseline[suppress] - gain[suppress])
    return out, {
        "threshold_db": float(min_peak_db),
        "suppressed_lobes": int(np.count_nonzero(suppressed_lobes)),
        "suppressed_bins": int(np.count_nonzero(suppress)),
        "suppressed_peak_max_db": float(np.max(peak_by_lobe[suppressed_lobes])),
    }


def _run_min_boost_peak_stage(
    gain_db: np.ndarray,
    freq_axis: np.ndarray,
    mask_c: np.ndarray,
    cfg_reader: CfgReader,
    st: dict,
    mag_authority_trace,
) -> np.ndarray:
    threshold_db = cfg_reader.float_allow_zero("min_boost_peak_db", 0.5)
    if not np.isfinite(threshold_db):
        threshold_db = 0.5
    threshold_db = float(np.clip(threshold_db, 0.0, 3.0))
    before = np.asarray(gain_db, dtype=float).copy()
    out, info = _suppress_subthreshold_local_detail_lobes(
        before,
        freq_axis,
        mask_c,
        min_peak_db=threshold_db,
    )
    safe_put_many(
        st,
        {
            "min_boost_peak_db": float(info["threshold_db"]),
            "min_boost_peak_suppressed_lobes": int(info["suppressed_lobes"]),
            "min_boost_peak_suppressed_bins": int(info["suppressed_bins"]),
            "min_boost_peak_suppressed_max_db": float(info["suppressed_peak_max_db"]),
        },
    )
    append_mag_authority_stage(
        mag_authority_trace,
        "after_min_boost_peak",
        before,
        out,
        freq_axis,
        mask_c,
        reason_codes=[REASON_MIN_BOOST_PEAK],
    )
    return out


def _run_softclip_stage(
    gain_apply,
    gain_db,
    freq_axis,
    mask_c,
    cfg,
    max_cut_db,
    max_boost_db_base,
    boost_cap_db,
    cut_cap_db,
    stage_probes,
    stage_probe_fn,
    logger,
    authority_boost_reduced_bins,
    authority_cut_reduced_bins,
    mag_authority_trace,
):
    _pre_softclip = np.zeros_like(gain_db, dtype=float)
    _pre_softclip[mask_c] = gain_apply[mask_c]
    tmp, over_boost, over_cut, softclip_boost_bins, softclip_cut_bins = _apply_soft_clamps(
        cfg=cfg,
        freq_axis=freq_axis,
        mask_c=mask_c,
        gain_apply=gain_apply,
        gain_db=gain_db,
        max_cut_db=max_cut_db,
        max_boost_db_base=max_boost_db_base,
        boost_cap_db=boost_cap_db,
        cut_cap_db=cut_cap_db,
        stage_probes=stage_probes,
        stage_probe_fn=stage_probe_fn,
        logger=logger,
    )
    _softclip_reasons = []
    try:
        _softclip_changed = mask_c & (
            np.abs(np.asarray(tmp, dtype=float) - np.asarray(_pre_softclip, dtype=float)) > 1e-9
        )
        _softclip_reduced_boost = bool(
            np.any(_softclip_changed & (_pre_softclip > 0.0) & (np.asarray(tmp, dtype=float) < _pre_softclip))
        )
        _softclip_limited_cut = bool(
            np.any(_softclip_changed & (_pre_softclip < 0.0) & (np.asarray(tmp, dtype=float) > _pre_softclip))
        )
    except (TypeError, ValueError, FloatingPointError):
        _softclip_reduced_boost = False
        _softclip_limited_cut = False
    if int(softclip_boost_bins) > 0 or _softclip_reduced_boost:
        _softclip_reasons.extend([REASON_SOFTCLIP_BOOST, REASON_USER_BOOST_CAP])
        if authority_boost_reduced_bins > 0:
            _softclip_reasons.append(REASON_ACOUSTIC_AUTHORITY_BOOST_CAP)
    if int(softclip_cut_bins) > 0 or _softclip_limited_cut:
        _softclip_reasons.extend([REASON_SOFTCLIP_CUT, REASON_USER_CUT_CAP])
        if authority_cut_reduced_bins > 0:
            _softclip_reasons.append(REASON_ACOUSTIC_AUTHORITY_CUT_CAP)
    append_mag_authority_stage(
        mag_authority_trace,
        "after_softclip",
        _pre_softclip,
        tmp,
        freq_axis,
        mask_c,
        reason_codes=_softclip_reasons,
    )
    gain_db[mask_c] = tmp[mask_c]
    return gain_db, over_boost, over_cut, softclip_boost_bins, softclip_cut_bins


def _run_mid_refit_stage(
    gain_db,
    freq_axis,
    mask_c,
    m_anal,
    target_mags,
    calc_offset_db,
    conf_mask,
    cfg,
    st,
    logger,
    debug_stage_stats,
    apply_mid_refit_pre_slope,
    mag_authority_trace,
):
    try:
        _pre_mid_refit = np.asarray(gain_db, dtype=float).copy()
        gain_db = apply_mid_refit_pre_slope(
            gain_db,
            freq_axis,
            mask_c,
            m_anal=m_anal,
            target_mags=target_mags,
            calc_offset_db=calc_offset_db,
            conf_mask=conf_mask,
            cfg=cfg,
            st=st,
            logger=logger,
        )
        _log_stage_stats(
            "gain_db_post_mid_refit",
            gain_db,
            mask_c,
            ref=_pre_mid_refit,
            logger=logger,
            enabled=debug_stage_stats,
        )
        append_mag_authority_stage(
            mag_authority_trace,
            "after_mid_refit",
            _pre_mid_refit,
            gain_db,
            freq_axis,
            mask_c,
        )
    except (TypeError, ValueError, FloatingPointError):
        pass
    return gain_db


def _run_slope_confpull_stage(
    gain_db,
    freq_axis,
    mask_c,
    cfg,
    st,
    raw_safe_ref,
    conf_mask,
    logger,
    debug_stage_stats,
    apply_confidence_weighted_target_pull,
    mag_authority_trace,
):
    _pre_slope = np.asarray(gain_db, dtype=float).copy()
    gain_db, slope_info = _apply_slope_limits(gain_db, freq_axis, cfg, st, mask_c)
    max_slope = float(slope_info["max_slope"])
    max_slope_boost = float(slope_info["max_slope_boost"])
    max_slope_cut = float(slope_info["max_slope_cut"])
    _slope_reasons = []
    if max_slope > 0 or max_slope_boost > 0 or max_slope_cut > 0:
        _slope_reasons.append(REASON_SLOPE_LIMIT)
    append_mag_authority_stage(
        mag_authority_trace,
        "after_slope",
        _pre_slope,
        gain_db,
        freq_axis,
        mask_c,
        reason_codes=_slope_reasons,
    )
    if max_slope > 0 or max_slope_boost > 0 or max_slope_cut > 0:
        _log_stage_stats("gain_db_post_slope", gain_db, mask_c, logger=logger, enabled=debug_stage_stats)
        try:
            _pre = gain_db.copy()
            gain_db = apply_confpull_post_slope(
                gain_db,
                mask_c,
                raw_safe_ref,
                cfg=cfg,
                st=st,
                conf_mask=conf_mask,
                freq_axis=freq_axis,
                logger=logger,
                apply_confidence_weighted_target_pull=apply_confidence_weighted_target_pull,
            )
            _log_stage_stats(
                "gain_db_post_confpull", gain_db, mask_c, ref=_pre, logger=logger, enabled=debug_stage_stats
            )
            append_mag_authority_stage(
                mag_authority_trace,
                "after_confpull",
                _pre,
                gain_db,
                freq_axis,
                mask_c,
                reason_codes=[REASON_CONFIDENCE_PULL],
            )
        except (TypeError, ValueError, FloatingPointError):
            pass
        try:
            logger.info(
                "Slope limit: "
                f"boost={float(max_slope_boost):.1f} dB/oct | "
                f"cut={float(max_slope_cut):.1f} dB/oct "
                f"(legacy max_slope_db_per_oct={float(max_slope):.1f})"
            )
        except (AttributeError, TypeError, ValueError):
            pass
    else:
        append_mag_authority_stage(
            mag_authority_trace,
            "after_confpull",
            gain_db,
            gain_db,
            freq_axis,
            mask_c,
        )
    return gain_db


def _run_regularization_stage(
    gain_db,
    freq_axis,
    mask_c,
    cfg_reader,
    filter_smooth,
    logger,
    debug_stage_stats,
    mag_authority_trace,
):
    try:
        _pre_regularization = np.asarray(gain_db, dtype=float).copy()
        if np.any(mask_c):
            mix = float(np.clip(float(cfg_reader.float("reg_strength", 30.0)) / 100.0, 0.0, 1.0))
            if mix > 0.0:
                _pre = gain_db.copy()
                gain_db = _blend_masked_fractional_octave(
                    gain_db,
                    freq_axis,
                    mask_c,
                    smooth_value=filter_smooth,
                    mix=mix,
                )
                _log_stage_stats(
                    "gain_db_post_filter_smooth", gain_db, mask_c, ref=_pre, logger=logger, enabled=debug_stage_stats
                )
        append_mag_authority_stage(
            mag_authority_trace,
            "after_regularization_smooth",
            _pre_regularization,
            gain_db,
            freq_axis,
            mask_c,
            reason_codes=[REASON_REGULARIZATION_SMOOTH],
        )
    except (TypeError, ValueError, FloatingPointError):
        pass
    return gain_db


def _run_excursion_protection_stage(
    gain_db,
    freq_axis,
    mask_c,
    gain_policy,
    max_boost_db_base,
    logger,
    mag_authority_trace,
):
    _pre_excursion = np.asarray(gain_db, dtype=float).copy() if gain_policy.exc_prot else gain_db
    if gain_policy.exc_prot:
        f_start = float(gain_policy.exc_freq)
        f_end = float(gain_policy.exc_soft_hz)
        prot_mask = freq_axis < f_start
        gain_db[prot_mask] = np.minimum(gain_db[prot_mask], 0.0)
        trans_mask = (freq_axis >= f_start) & (freq_axis <= f_end)
        if f_end > f_start and np.any(trans_mask):
            fade = (freq_axis[trans_mask] - f_start) / (f_end - f_start)
            allowed_boost = fade * float(max_boost_db_base)
            gain_db[trans_mask] = np.minimum(gain_db[trans_mask], allowed_boost)
        logger.info(f"Exc Prot: Full protection < {f_start}Hz, Soft fade up to {f_end:.1f}Hz.")
    append_mag_authority_stage(
        mag_authority_trace,
        "after_excursion_protection",
        _pre_excursion,
        gain_db,
        freq_axis,
        mask_c,
        reason_codes=([REASON_EXCURSION_FULL_BLOCK, REASON_EXCURSION_SOFT_CAP] if bool(gain_policy.exc_prot) else []),
    )
    return gain_db


def _run_wav_transition_smooth_stage(
    gain_db,
    freq_axis,
    mask_c,
    cfg_reader,
    st,
    logger,
    debug_stage_stats,
    mag_authority_trace,
):
    try:
        _pre_wav_transition = gain_db
        if cfg_reader.bool("is_wav_source", False) and np.any(mask_c):
            cmin = cfg_reader.float_allow_zero("mag_c_min", 0.0)
            cmax = cfg_reader.float_allow_zero("mag_c_max", 0.0)
            tw = cfg_reader.float_allow_zero("trans_width", 0.0)
            if np.isfinite(cmin) and np.isfinite(cmax) and np.isfinite(tw) and (cmax > cmin):
                if tw <= 0.0:
                    tw = max(50.0, 0.4 * cmax)
                f_lo = max(cmin, cmax - max(30.0, 0.35 * tw))
                f_hi = min(float(np.max(freq_axis)), cmax + max(45.0, 0.55 * tw))
                zone = (freq_axis >= f_lo) & (freq_axis <= f_hi)
                if int(np.count_nonzero(zone)) >= 8:
                    g0 = np.asarray(gain_db, dtype=float).copy()
                    _pre_wav_transition = g0
                    g_sm = smooth_gain_fractional_octave(freq_axis, g0, 24.0)
                    span = max(1e-9, float(f_hi - f_lo))
                    x = np.clip((freq_axis - f_lo) / span, 0.0, 1.0)
                    ramp = 0.5 - 0.5 * np.cos(np.pi * x)
                    sigma_hz = max(20.0, 0.20 * max(tw, 1.0))
                    focus = np.exp(-0.5 * ((freq_axis - cmax) / sigma_hz) ** 2)
                    w = np.zeros_like(g0, dtype=float)
                    w[zone] = ramp[zone] * focus[zone]
                    mix = 0.55
                    gain_db = g0 + (g_sm - g0) * (mix * w)
                    _log_stage_stats(
                        "gain_db_post_wav_transition_smooth",
                        gain_db,
                        mask_c,
                        ref=_pre_wav_transition,
                        logger=logger,
                        enabled=debug_stage_stats,
                    )
                    if isinstance(st, dict):
                        st["wav_transition_smoothing"] = True
                        st["wav_transition_smoothing_zone_hz"] = [float(f_lo), float(f_hi)]
        append_mag_authority_stage(
            mag_authority_trace,
            "after_wav_transition_smooth",
            _pre_wav_transition,
            gain_db,
            freq_axis,
            mask_c,
            reason_codes=[REASON_WAV_TRANSITION_SMOOTH],
        )
    except (TypeError, ValueError, FloatingPointError):
        pass
    return gain_db


def _run_hardclamp_stage(
    gain_db,
    freq_axis,
    mask_c,
    cfg,
    st,
    logger,
    boost_cap_db,
    cut_cap_db,
    max_cut_db,
    max_boost_db_base,
    filter_smooth,
    debug_stage_stats,
    stage_probes,
    stage_probe_fn,
    authority_boost_reduced_bins,
    authority_cut_reduced_bins,
    mag_authority_trace,
):
    _pre_hardclamp = np.asarray(gain_db, dtype=float).copy()
    (
        gain_db,
        hardclamp_boost_bins,
        hardclamp_cut_bins,
        hard_over_boost,
        hard_over_cut,
        clamp_dominance_level,
    ) = _apply_hard_clamps(
        cfg=cfg,
        freq_axis=freq_axis,
        mask_c=mask_c,
        gain_db=gain_db,
        boost_cap_db=boost_cap_db,
        cut_cap_db=cut_cap_db,
        max_cut_db=max_cut_db,
        max_boost_db_base=max_boost_db_base,
        filter_smooth=filter_smooth,
        debug_stage_stats=debug_stage_stats,
        stage_probes=stage_probes,
        stage_probe_fn=stage_probe_fn,
        logger=logger,
        st=st,
    )
    _hardclamp_reasons = []
    if int(hardclamp_boost_bins) > 0:
        _hardclamp_reasons.extend([REASON_HARDCLAMP_BOOST, REASON_USER_BOOST_CAP])
        if authority_boost_reduced_bins > 0:
            _hardclamp_reasons.append(REASON_ACOUSTIC_AUTHORITY_BOOST_CAP)
    if int(hardclamp_cut_bins) > 0:
        _hardclamp_reasons.extend([REASON_HARDCLAMP_CUT, REASON_USER_CUT_CAP])
        if authority_cut_reduced_bins > 0:
            _hardclamp_reasons.append(REASON_ACOUSTIC_AUTHORITY_CUT_CAP)
    append_mag_authority_stage(
        mag_authority_trace,
        "after_hardclamp",
        _pre_hardclamp,
        gain_db,
        freq_axis,
        mask_c,
        reason_codes=_hardclamp_reasons,
    )
    return gain_db, hardclamp_boost_bins, hardclamp_cut_bins, hard_over_boost, hard_over_cut, clamp_dominance_level


def _run_wav_final_polish_stage(
    gain_db,
    freq_axis,
    mask_c,
    cfg_reader,
    st,
    cfg,
    max_cut_db,
    boost_cap_db,
    cut_cap_db,
    logger,
    debug_stage_stats,
    mag_authority_trace,
):
    try:
        _pre_wav_final = gain_db
        if cfg_reader.bool("is_wav_source", False) and np.any(mask_c):
            cmin = cfg_reader.float_allow_zero("mag_c_min", 0.0)
            cmax = cfg_reader.float_allow_zero("mag_c_max", 0.0)
            tw = cfg_reader.float_allow_zero("trans_width", 0.0)
            if np.isfinite(cmin) and np.isfinite(cmax) and (cmax > cmin):
                if not np.isfinite(tw) or tw <= 0.0:
                    tw = max(50.0, 0.4 * cmax)
                f_lo = max(cmin, cmax - 0.95 * tw)
                f_hi = min(float(np.max(freq_axis)), cmax + 1.45 * tw)
                zone = (freq_axis >= f_lo) & (freq_axis <= f_hi)
                if int(np.count_nonzero(zone)) >= 8:
                    g0 = np.asarray(gain_db, dtype=float).copy()
                    _pre_wav_final = g0
                    sigma_bins = _sigma_bins_from_hz(freq_axis, sigma_hz=8.0, fallback_bins=12.0)
                    g_sm = scipy.ndimage.gaussian_filter1d(g0, sigma=float(max(2.0, sigma_bins)))
                    x = np.zeros_like(g0, dtype=float)
                    span = max(1e-9, float(f_hi - f_lo))
                    x[zone] = np.clip((freq_axis[zone] - f_lo) / span, 0.0, 1.0)
                    w = np.zeros_like(g0, dtype=float)
                    w[zone] = 0.5 - 0.5 * np.cos(np.pi * x[zone])
                    mix = 0.95
                    gain_db = g0 + (g_sm - g0) * (mix * w)
                    gain_db = _apply_hard_boost_cut_clamp(
                        gain_db,
                        cfg,
                        max_cut_db,
                        boost_cap_db=boost_cap_db,
                        cut_cap_db=cut_cap_db,
                        mask=mask_c,
                    )
                    _log_stage_stats(
                        "gain_db_post_wav_final_ripple_polish",
                        gain_db,
                        mask_c,
                        ref=_pre_wav_final,
                        logger=logger,
                        enabled=debug_stage_stats,
                    )
                    if isinstance(st, dict):
                        st["wav_final_ripple_polish"] = True
                        st["wav_final_ripple_polish_zone_hz"] = [float(f_lo), float(f_hi)]
        append_mag_authority_stage(
            mag_authority_trace,
            "after_wav_final_ripple_polish",
            _pre_wav_final,
            gain_db,
            freq_axis,
            mask_c,
            reason_codes=[REASON_WAV_TRANSITION_SMOOTH],
        )
    except (TypeError, ValueError, FloatingPointError):
        pass
    return gain_db


def _run_bass_boost_restore_stage(
    gain_db,
    freq_axis,
    mask_c,
    bass_boost_post_restore_enable,
    bass_boost_post_restore_strength,
    low_hz,
    bass_boost_cap_hz,
    gain_apply,
    boost_cap_db,
    cut_cap_db,
    cfg,
    max_cut_db,
    st,
    logger,
    debug_stage_stats,
    mag_authority_trace,
):
    try:
        _pre_post_restore = gain_db
        if bool(bass_boost_post_restore_enable) and np.any(mask_c):
            _pre_post_restore = np.asarray(gain_db, dtype=float).copy()
            restore_lo = float(max(20.0, float(low_hz) + 1e-6))
            restore_hi = float(max(restore_lo, bass_boost_cap_hz))
            restore_src = np.asarray(gain_apply, dtype=float)
            boost_restore_target = np.minimum(restore_src, np.asarray(boost_cap_db, dtype=float))
            boost_restore_target = np.maximum(boost_restore_target, -np.asarray(cut_cap_db, dtype=float))
            tgt_restore = np.where(restore_src > 0.0, boost_restore_target, _pre_post_restore)
            gain_db, _restore_meta = apply_bass_boost_post_restore(
                gain_db,
                tgt_restore,
                boost_cap_db,
                freq_axis,
                mask_c,
                hz_lo=restore_lo,
                hz_hi=restore_hi,
                strength=float(bass_boost_post_restore_strength),
            )
            gain_db = _apply_hard_boost_cut_clamp(
                gain_db,
                cfg,
                max_cut_db,
                boost_cap_db=boost_cap_db,
                cut_cap_db=cut_cap_db,
                mask=mask_c,
            )
            if isinstance(st, dict):
                safe_put_many(
                    st,
                    {
                        "bass_boost_post_restore_applied": bool(_restore_meta.get("enabled", False)),
                        "bass_boost_post_restore_bins": int(_restore_meta.get("bins", 0) or 0),
                        "bass_boost_post_restore_delta_rms_20_200": float(
                            _restore_meta.get("delta_rms_20_200", 0.0) or 0.0
                        ),
                        "bass_boost_post_restore_delta_max_20_200": float(
                            _restore_meta.get("delta_max_20_200", 0.0) or 0.0
                        ),
                    },
                )
            _log_stage_stats(
                "gain_db_post_bass_boost_restore",
                gain_db,
                mask_c,
                ref=_pre_post_restore,
                logger=logger,
                enabled=debug_stage_stats,
            )
        append_mag_authority_stage(
            mag_authority_trace,
            "after_bass_boost_restore",
            _pre_post_restore,
            gain_db,
            freq_axis,
            mask_c,
            reason_codes=[REASON_BASS_BOOST_RESTORE],
            restored_allowed_correction=True,
        )
    except (TypeError, ValueError, FloatingPointError):
        pass
    return gain_db


def _run_lowbass_hard_reapply_stage(
    gain_db,
    freq_axis,
    mask_c,
    low_cut_enable,
    low_hz,
    gain_policy,
    low_cut_floor_ref,
    low_cut_strength,
    st,
    logger,
    mag_authority_trace,
):
    try:
        _pre_lowbass_reapply = gain_db
        lf_guard_meta = {}
        if bool(low_cut_enable) and np.isfinite(float(low_hz)) and float(low_hz) > 0.0:
            low_mask_final = build_low_frequency_guard_mask(
                freq_axis,
                gain_policy,
                include_low_cut=True,
                include_exc_soft=False,
            )
            if np.any(mask_c & low_mask_final):
                _pre_lowbass_reapply = np.asarray(gain_db, dtype=float).copy()
                gain_db, lf_guard_meta = apply_cuts_only_guard(
                    gain_db,
                    mask=mask_c,
                    guard_mask=low_mask_final,
                    floor_ref=(low_cut_floor_ref if low_cut_strength > 0.0 else None),
                )
                lf_boost_clamped_bins = int(lf_guard_meta.get("boost_clamped_bins", 0) or 0)
                lf_floor_reapplied_bins = int(lf_guard_meta.get("floor_reapplied_bins", 0) or 0)
                if lf_boost_clamped_bins > 0 or lf_floor_reapplied_bins > 0:
                    logger.info(
                        "Low-bass hard lock reapply: "
                        f"clamped {lf_boost_clamped_bins} boost bins, "
                        f"reapplied cut floor on {lf_floor_reapplied_bins} bins <= {float(low_hz):.1f} Hz"
                    )
                try:
                    if isinstance(st, dict):
                        safe_put_many(
                            st,
                            {
                                "low_bass_hard_reapply_hz": float(low_hz),
                                "low_bass_hard_reapply_bins": int(lf_guard_meta.get("guard_bins", 0) or 0),
                                "low_bass_hard_reapply_clamped_bins": int(lf_boost_clamped_bins),
                                "low_bass_hard_reapply_floor_bins": int(lf_floor_reapplied_bins),
                            },
                        )
                except (TypeError, ValueError):
                    pass
        _lf_reapply_reasons = []
        if int(lf_guard_meta.get("boost_clamped_bins", 0) or 0) > 0:
            _lf_reapply_reasons.append(REASON_LOW_BASS_CUTS_ONLY)
        if int(lf_guard_meta.get("floor_reapplied_bins", 0) or 0) > 0:
            _lf_reapply_reasons.append(REASON_LOW_BASS_FLOOR_REAPPLIED)
        append_mag_authority_stage(
            mag_authority_trace,
            "after_lowbass_hard_reapply",
            _pre_lowbass_reapply,
            gain_db,
            freq_axis,
            mask_c,
            reason_codes=_lf_reapply_reasons,
        )
    except (TypeError, ValueError, FloatingPointError):
        pass
    return gain_db


def _run_transition_fade_stage(  # noqa: C901 - transition fade keeps the band-edge math and telemetry together
    gain_db,
    freq_axis,
    mask_c,
    cfg_reader,
    st,
    mag_authority_trace,
):
    try:
        mag_c_min_fade = cfg_reader.float_allow_zero("mag_c_min", 0.0)
    except (TypeError, ValueError):
        mag_c_min_fade = 0.0
    try:
        mag_c_max_fade = cfg_reader.float_allow_zero("mag_c_max", 0.0)
    except (TypeError, ValueError):
        mag_c_max_fade = 0.0
    tw_raw = cfg_reader.raw("trans_width", 100.0)
    try:
        trans_width_fade = 100.0 if tw_raw is None else float(tw_raw)
    except (TypeError, ValueError):
        trans_width_fade = 100.0
    if not np.isfinite(trans_width_fade):
        trans_width_fade = 100.0

    band_width = max(mag_c_max_fade - mag_c_min_fade, 0.0)
    if band_width > 0.0:
        # Fade covers at least 25 % of the correction band so that a narrow
        # trans_width doesn't leave the upper 75 % fully unconstrained when
        # mag_c_max is very high (e.g. 20 000 Hz).
        min_fade_width = max(trans_width_fade, 0.25 * band_width)
        f_start = max(mag_c_max_fade - min_fade_width, mag_c_min_fade)
    else:
        f_start = max(mag_c_max_fade - trans_width_fade, mag_c_min_fade)
    f_mask = (freq_axis > f_start) & (freq_axis <= mag_c_max_fade)
    fade_len = mag_c_max_fade - f_start
    _pre_transition_fade = np.asarray(gain_db, dtype=float).copy()
    if np.any(f_mask) and fade_len > 0:
        x = (freq_axis[f_mask] - f_start) / fade_len
        w = _cosine_fade_out_01(x)
        gain_db[f_mask] *= w
        try:
            if isinstance(st, dict):
                st["mag_transition_fade_applied"] = True
                try:
                    band = (freq_axis >= f_start) & (freq_axis <= mag_c_max_fade)
                    if np.count_nonzero(band) >= 8:
                        g = np.asarray(gain_db, dtype=float)
                        f = np.asarray(freq_axis, dtype=float)
                        x_band = np.log2(np.maximum(f[band], 1e-9))
                        dg = np.diff(g[band])
                        dx = np.diff(x_band)
                        slope = dg / (dx + 1e-30)
                        st["mag_transition_slope_abs_max_db_per_oct"] = float(np.max(np.abs(slope)))
                except (TypeError, ValueError, FloatingPointError, IndexError):
                    pass
        except (TypeError, ValueError):
            pass
    append_mag_authority_stage(
        mag_authority_trace,
        "after_transition_fade",
        _pre_transition_fade,
        gain_db,
        freq_axis,
        mask_c,
        reason_codes=[REASON_TRANSITION_FADE],
    )
    return gain_db



