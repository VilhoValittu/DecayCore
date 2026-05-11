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
import scipy.ndimage

from ..decaycore_analysis import _sigma_bins_from_hz
from ..correction_types import _MagPostProcessInputs, _MagPostProcessOutputs
from .._measurement_ctx_local import get_measurement_ctx
from ..dsp_config import CfgReader
from ..dsp_telemetry import safe_put_many
from ..gain_policy import apply_cuts_only_guard, build_low_frequency_guard_mask, resolve_gain_policy
from ..mag_limits import (
    _apply_hard_boost_cut_clamp,
    _apply_max_boost_cut,
    _apply_slope_limits,
    _blend_masked_fractional_octave,
)
from ..mag_postprocess import apply_bass_boost_post_restore, apply_confpull_post_slope
from ..mag_telemetry import (
    _band_delta_metrics,
    _band_error_rms,
    _log_stage_stats,
    _record_stage_probe,
    _summarize_correction_metrics,
)
from ..phase_ir_utils import _cosine_fade_out_01
from ..smoothing import smooth_gain_fractional_octave

def apply_post_limits_and_metrics(
    inputs: _MagPostProcessInputs,
    *,
    apply_mid_refit_pre_slope,
) -> _MagPostProcessOutputs:
    """Ajaa mag-korjauksen loppuvaiheen: low-bass policy, clamping, slope/fade ja metriikat."""

    cfg = inputs.cfg
    cfg_reader = CfgReader(cfg)
    freq_axis = inputs.freq_axis
    st = inputs.st
    logger = inputs.logger
    _stage_probe = inputs.stage_probe
    _cfg_float_allow_zero = inputs.cfg_float_allow_zero
    mask_c = inputs.mask_c
    gain_db = inputs.gain_db
    gain_apply = inputs.gain_apply
    raw_g = inputs.raw_g
    final_g = inputs.final_g
    pre_bass_adapt_g = inputs.pre_bass_adapt_g
    raw_safe_ref = inputs.raw_safe_ref
    conf_mask = inputs.conf_mask
    _filter_smooth = inputs.filter_smooth
    debug_stage_stats = inputs.debug_stage_stats
    stage_probes = dict(inputs.stage_probes)
    apply_confidence_weighted_target_pull = inputs.apply_confidence_weighted_target_pull

    boost_peak_db = 0.0
    cut_peak_db = 0.0
    n_boost = 0
    boost_cand_peak = 0.0
    boost_cand_min_hz = float("nan")
    n_boost_cand = 0
    n_boost_cand_low = 0
    n_boost_cand_exc = 0
    softclip_boost_bins = 0
    softclip_cut_bins = 0
    over_boost = 0.0
    over_cut = 0.0
    hardclamp_boost_bins = 0
    hardclamp_cut_bins = 0
    hard_over_boost = 0.0
    hard_over_cut = 0.0
    clamp_dominance_level = "NONE"

    gain_policy = resolve_gain_policy(cfg, cfg_float_allow_zero_fn=_cfg_float_allow_zero)
    low_cut_enable = bool(gain_policy.low_cut_enable)
    low_hz = float(gain_policy.low_cut_hz)
    low_cut_strength = float(gain_policy.low_cut_strength)
    low_cut_floor_ref = _apply_low_frequency_policy(
        freq_axis=freq_axis,
        mask_c=mask_c,
        gain_apply=gain_apply,
        raw_g=raw_g,
        final_g=final_g,
        gain_db=gain_db,
        gain_policy=gain_policy,
        stage_probes=stage_probes,
        stage_probe_fn=_stage_probe,
        cfg=cfg,
        logger=logger,
    )
    try:
        logger.info(
            f"CFG CHECK: conf_pull_floor={cfg_reader.float('conf_pull_floor', 0.05)}, "
            f"gamma_cut={cfg_reader.float('conf_pull_gamma_cut', 0.55)}, "
            f"low_bass_cut_strength={cfg_reader.float_allow_zero('low_bass_cut_strength', 0.0)}"
        )
    except (AttributeError, TypeError, ValueError, FloatingPointError, IndexError):
        pass

    caps = _prepare_boost_caps(
        cfg=cfg,
        cfg_reader=cfg_reader,
        st=st,
        logger=logger,
        freq_axis=freq_axis,
        mask_c=mask_c,
        conf_mask=conf_mask,
        gain_db=gain_db,
        gain_policy=gain_policy,
    )
    max_cut_db = float(caps["max_cut_db"])
    max_boost_db_base = float(caps["max_boost_db_base"])
    boost_cap_db = np.asarray(caps["boost_cap_db"], dtype=float)
    bass_boost_cap_hz = float(caps["bass_boost_cap_hz"])
    bass_boost_post_restore_enable = bool(caps["bass_boost_post_restore_enable"])
    bass_boost_post_restore_strength = float(caps["bass_boost_post_restore_strength"])
    authority_caps = _apply_acoustic_authority_caps(
        cfg=cfg,
        cfg_reader=cfg_reader,
        st=st,
        logger=logger,
        freq_axis=freq_axis,
        mask_c=mask_c,
        boost_cap_db=boost_cap_db,
        max_cut_db=max_cut_db,
    )
    boost_cap_db = np.asarray(authority_caps["boost_cap_db"], dtype=float)
    cut_cap_db = np.asarray(authority_caps["cut_cap_db"], dtype=float)

    (
        boost_cand_peak,
        boost_cand_min_hz,
        n_boost_cand,
        n_boost_cand_low,
        n_boost_cand_exc,
    ) = _apply_candidate_metrics(
        gain_apply=gain_apply,
        gain_db=gain_db,
        freq_axis=freq_axis,
        mask_c=mask_c,
        gain_policy=gain_policy,
    )

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
        stage_probe_fn=_stage_probe,
        logger=logger,
    )

    gain_db[mask_c] = tmp[mask_c]
    try:
        _pre_mid_refit = np.asarray(gain_db, dtype=float).copy()
        gain_db = apply_mid_refit_pre_slope(
            gain_db,
            freq_axis,
            mask_c,
            m_anal=inputs.m_anal,
            target_mags=inputs.target_mags,
            calc_offset_db=inputs.calc_offset_db,
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
    except (TypeError, ValueError, FloatingPointError):
        pass
    gain_db, slope_info = _apply_slope_limits(gain_db, freq_axis, cfg, st, mask_c)
    max_slope = float(slope_info["max_slope"])
    max_slope_boost = float(slope_info["max_slope_boost"])
    max_slope_cut = float(slope_info["max_slope_cut"])
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
            _log_stage_stats("gain_db_post_confpull", gain_db, mask_c, ref=_pre, logger=logger, enabled=debug_stage_stats)
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
    try:
        if np.any(mask_c):
            mix = float(np.clip(float(cfg_reader.float("reg_strength", 30.0)) / 100.0, 0.0, 1.0))
            if mix > 0.0:
                _pre = gain_db.copy()
                gain_db = _blend_masked_fractional_octave(
                    gain_db,
                    freq_axis,
                    mask_c,
                    smooth_value=_filter_smooth,
                    mix=mix,
                )
                _log_stage_stats("gain_db_post_filter_smooth", gain_db, mask_c, ref=_pre, logger=logger, enabled=debug_stage_stats)
    except (TypeError, ValueError, FloatingPointError):
        pass

    if gain_policy.exc_prot:
        f_start = float(gain_policy.exc_freq)
        f_end = float(gain_policy.exc_soft_hz)
        prot_mask = freq_axis < f_start
        gain_db[prot_mask] = np.minimum(gain_db[prot_mask], 0.0)
        trans_mask = (freq_axis >= f_start) & (freq_axis <= f_end)
        if np.any(trans_mask):
            fade = (freq_axis[trans_mask] - f_start) / (f_end - f_start)
            allowed_boost = fade * float(max_boost_db_base)
            gain_db[trans_mask] = np.minimum(gain_db[trans_mask], allowed_boost)
        logger.info(f"Exc Prot: Full protection < {f_start}Hz, Soft fade up to {f_end:.1f}Hz.")

    try:
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
                    _pre = gain_db.copy()
                    g0 = np.asarray(gain_db, dtype=float).copy()
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
                    _log_stage_stats("gain_db_post_wav_transition_smooth", gain_db, mask_c, ref=_pre, logger=logger, enabled=debug_stage_stats)
                    if isinstance(st, dict):
                        st["wav_transition_smoothing"] = True
                        st["wav_transition_smoothing_zone_hz"] = [float(f_lo), float(f_hi)]
    except (TypeError, ValueError, FloatingPointError):
        pass
    try:
        if "after_slope" not in stage_probes:
            _record_stage_probe(stage_probes, "after_slope", _stage_probe, freq_axis, gain_db, mask_c, cfg, logger)
    except (TypeError, ValueError, FloatingPointError, IndexError):
        pass

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
        filter_smooth=_filter_smooth,
        debug_stage_stats=debug_stage_stats,
        stage_probes=stage_probes,
        stage_probe_fn=_stage_probe,
        logger=logger,
        st=st,
    )
    try:
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
                    _pre = gain_db.copy()
                    g0 = np.asarray(gain_db, dtype=float).copy()
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
                    _log_stage_stats("gain_db_post_wav_final_ripple_polish", gain_db, mask_c, ref=_pre, logger=logger, enabled=debug_stage_stats)
                    if isinstance(st, dict):
                        st["wav_final_ripple_polish"] = True
                        st["wav_final_ripple_polish_zone_hz"] = [float(f_lo), float(f_hi)]
    except (TypeError, ValueError, FloatingPointError):
        pass

    # Ensure bass boost changes can propagate to final filter (post-limits domain),
    # instead of being fully neutralized by conf-pull/slope interactions.
    try:
        if bool(bass_boost_post_restore_enable) and np.any(mask_c):
            restore_lo = float(max(20.0, float(low_hz) + 1e-6))
            restore_hi = float(max(restore_lo, bass_boost_cap_hz))
            tgt_restore = np.asarray(gain_apply, dtype=float).copy()
            tgt_restore = np.minimum(tgt_restore, np.asarray(boost_cap_db, dtype=float))
            tgt_restore = np.maximum(tgt_restore, -np.asarray(cut_cap_db, dtype=float))
            _pre_post_restore = np.asarray(gain_db, dtype=float).copy()
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
    except (TypeError, ValueError, FloatingPointError):
        pass

    # Final hard reapply: low-bass cuts-only policy must survive all smoothing/clamps.
    try:
        if bool(low_cut_enable) and np.isfinite(float(low_hz)) and float(low_hz) > 0.0:
            low_mask_final = build_low_frequency_guard_mask(
                freq_axis,
                gain_policy,
                include_low_cut=True,
                include_exc_soft=False,
            )
            if np.any(mask_c & low_mask_final):
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
    except (TypeError, ValueError, FloatingPointError):
        pass

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

    f_start = max(mag_c_max_fade - trans_width_fade, mag_c_min_fade)
    f_mask = (freq_axis > f_start) & (freq_axis <= mag_c_max_fade)
    fade_len = mag_c_max_fade - f_start
    if np.any(f_mask) and fade_len > 0:
        x = (freq_axis[f_mask] - f_start) / fade_len
        w = _cosine_fade_out_01(x)
        gain_db[f_mask] *= w
        try:
            if isinstance(st, dict):
                st["mag_transition_fade_applied"] = True
                try:
                    band = (freq_axis >= (mag_c_max_fade - trans_width_fade)) & (freq_axis <= mag_c_max_fade)
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
    _record_stage_probe(stage_probes, "after_fade", _stage_probe, freq_axis, gain_db, mask_c, cfg, logger)
    _store_realized_pre_ir_metrics(
        st=st,
        cfg_reader=cfg_reader,
        freq_axis=freq_axis,
        mask_c=mask_c,
        gain_db=gain_db,
        pre_bass_adapt_g=pre_bass_adapt_g,
        m_anal=inputs.m_anal,
        target_mags=inputs.target_mags,
        calc_offset_db=inputs.calc_offset_db,
    )

    try:
        summary = _summarize_correction_metrics(
            gain_db,
            freq_axis,
            cfg,
            st,
            mask_c,
            logger,
            boost_cand_peak=boost_cand_peak,
            n_boost_cand=n_boost_cand,
            n_boost_cand_low=n_boost_cand_low,
            n_boost_cand_exc=n_boost_cand_exc,
        )
        boost_peak_db = float(summary["boost_peak_db"])
        cut_peak_db = float(summary["cut_peak_db"])
        n_boost = int(summary["n_boost"])
    except (TypeError, ValueError, KeyError):
        boost_peak_db, cut_peak_db, n_boost = 0.0, 0.0, 0

    return _MagPostProcessOutputs(
        gain_db=np.asarray(gain_db, dtype=float),
        stage_probes=dict(stage_probes),
        boost_peak_db=float(boost_peak_db),
        cut_peak_db=float(cut_peak_db),
        n_boost=int(n_boost),
        boost_cand_peak=float(boost_cand_peak),
        boost_cand_min_hz=float(boost_cand_min_hz),
        n_boost_cand=int(n_boost_cand),
        n_boost_cand_low=int(n_boost_cand_low),
        n_boost_cand_exc=int(n_boost_cand_exc),
        softclip_boost_bins=int(softclip_boost_bins),
        softclip_cut_bins=int(softclip_cut_bins),
        over_boost=float(over_boost),
        over_cut=float(over_cut),
        hardclamp_boost_bins=int(hardclamp_boost_bins),
        hardclamp_cut_bins=int(hardclamp_cut_bins),
        hard_over_boost=float(hard_over_boost),
        hard_over_cut=float(hard_over_cut),
        clamp_dominance_level=str(clamp_dominance_level),
    )


__all__ = ['apply_post_limits_and_metrics']


def _load_sibling_symbols() -> None:
    import importlib
    package = __package__
    for module_name in ['low_frequency', 'authority', 'clamps', 'metrics', 'pipeline']:
        if module_name == __name__.rsplit('.', 1)[-1]:
            continue
        module = importlib.import_module(f"{package}.{module_name}")
        for symbol in getattr(module, "__all__", ()):
            globals().setdefault(symbol, getattr(module, symbol))


_load_sibling_symbols()
