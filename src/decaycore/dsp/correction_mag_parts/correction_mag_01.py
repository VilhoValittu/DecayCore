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

from typing import Any, Callable

import numpy as np
import scipy.ndimage

from decaycore.auto_mode.auto_mode_profile import profiled_section

from .. import bassfirst as bf
from ..decaycore_analysis import _sigma_bins_from_hz
from ..correction_types import (
    _MagAdaptiveStageOutputs,
    _MagCoreOutputs,
    _MagCorrectionContext,
    _MagPipelineInputs,
    _MagPostProcessInputs,
    _MagPostProcessOutputs,
    _MagRawStageOutputs,
)
from ..dsp_config import CfgReader
from ..dsp_telemetry import safe_put_many
from ..gain_policy import apply_cuts_only_guard, build_low_frequency_guard_mask, resolve_gain_policy
from ..mag_limits import (
    _apply_hard_boost_cut_clamp,
    _apply_max_boost_cut,
    _apply_slope_limits,
    _blend_masked_fractional_octave,
)
from ..mag_post_limits import apply_post_limits_and_metrics as _apply_post_limits_and_metrics_impl
from ..mag_postprocess import apply_bass_boost_post_restore, apply_confpull_post_slope
from ..mag_shape import (
    _apply_confidence_logic,
    _apply_regularization,
    _compute_error_db,
    _error_to_correction_mag,
    _resolve_filter_smooth,
    _select_active_band,
)
from ..mag_stage import (
    run_mag_bassfirst_afdw_conf_stage,
    run_mag_core_stage,
    run_mag_raw_stage,
)
from ..mag_telemetry import (
    _log_stage_stats,
    _record_stage_probe,
    _summarize_correction_metrics,
)
from ..phase_ir_utils import _cosine_fade_out_01
from ..smoothing import (
    AFDW_BW_MAX_OCT,
    AFDW_BW_MIN_OCT,
    apply_adaptive_fdw,
    psycho_smooth_safe_gain,
    smooth_gain_fractional_octave,
)

def _apply_peak_priority_error_shaping(
    err_db: np.ndarray,
    freq_axis: np.ndarray,
    cfg: Any,
    st: Any,
    *,
    mask_c: np.ndarray,
    logger: Any,
) -> np.ndarray:
    """
   Peak-priority error formulation:
      - Keep CUT requests (negative error) intact.
      - Leave BOOST requests unchanged up to max_boost_db.
      - Soft-limit only the portion above max_boost_db so deep dips don't
        dominate the solution when max_boost is small anyway.
    """
    cfg_reader = CfgReader(cfg)
    enable = cfg_reader.bool("peak_priority_enable", True)
    if not enable:
        return err_db

    strength = cfg_reader.float_allow_zero("peak_priority_strength", 0.1) #vaikuttaa boostin voimakkuuteen, mitä isompi, sitä miedompi boost
    if not np.isfinite(strength):
        strength = 0.0
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0:
        return err_db

    max_boost = cfg_reader.float_allow_zero("max_boost_db", 0.0)
    if not np.isfinite(max_boost) or max_boost <= 0.0:
        return err_db

    e = np.asarray(err_db, dtype=float).copy()
    m = np.asarray(mask_c, dtype=bool)
    pos = m & (e > 0.0)
    if not np.any(pos):
        return e

    e = _apply_peak_priority_pos_soft_cap(
        e,
        pos_mask=pos,
        max_boost=float(max_boost),
        strength=float(strength),
    )

    # Optional (small) cut emphasis if you want even more peak-first behavior.
    cut_emph = cfg_reader.float_allow_zero("peak_priority_cut_emphasis", 0.0)
    e = _apply_peak_priority_optional_cut_emphasis(
        e,
        band_mask=m,
        cut_emphasis=cut_emph,
        strength=float(strength),
    )

    # Stats (safe)
    _record_peak_priority_stats(st, err_db=err_db, shaped_error=e, pos_mask=pos, strength=float(strength), max_boost=float(max_boost))

    _log_peak_priority_summary(logger, err_db=err_db, shaped_error=e, pos_mask=pos, strength=float(strength), max_boost=float(max_boost))

    return e


def _apply_peak_priority_pos_soft_cap(
    error_db: np.ndarray,
    *,
    pos_mask: np.ndarray,
    max_boost: float,
    strength: float,
) -> np.ndarray:
    cap = float(max_boost)
    pos_values = error_db[pos_mask]
    over = np.maximum(0.0, pos_values - cap)
    saturated = np.where(
        pos_values <= cap,
        pos_values,
        cap + cap * np.tanh(over / (cap + 1e-12)),
    )
    out = np.asarray(error_db, dtype=float).copy()
    out[pos_mask] = (1.0 - float(strength)) * pos_values + float(strength) * saturated
    return out


def _apply_peak_priority_optional_cut_emphasis(
    error_db: np.ndarray,
    *,
    band_mask: np.ndarray,
    cut_emphasis: float,
    strength: float,
) -> np.ndarray:
    if not np.isfinite(cut_emphasis) or cut_emphasis <= 0.0:
        return error_db
    clipped = float(np.clip(cut_emphasis, 0.0, 2.0))
    neg = np.asarray(band_mask, dtype=bool) & (np.asarray(error_db, dtype=float) < 0.0)
    if not np.any(neg):
        return error_db
    out = np.asarray(error_db, dtype=float).copy()
    out[neg] *= (1.0 + clipped * float(strength))
    return out


def _record_peak_priority_stats(
    stats: Any,
    *,
    err_db: np.ndarray,
    shaped_error: np.ndarray,
    pos_mask: np.ndarray,
    strength: float,
    max_boost: float,
) -> None:
    try:
        if not isinstance(stats, dict):
            return
        stats["peak_priority_enable"] = True
        stats["peak_priority_strength"] = float(strength)
        stats["peak_priority_max_boost_db"] = float(max_boost)
        stats["peak_priority_pos_err_peak_before_db"] = float(np.max(np.asarray(err_db, dtype=float)[pos_mask]))
        stats["peak_priority_pos_err_peak_after_db"] = float(np.max(np.asarray(shaped_error, dtype=float)[pos_mask]))
    except (TypeError, ValueError):
        return


def _log_peak_priority_summary(
    logger: Any,
    *,
    err_db: np.ndarray,
    shaped_error: np.ndarray,
    pos_mask: np.ndarray,
    strength: float,
    max_boost: float,
) -> None:
    try:
        logger.info(
            "PeakPriority: "
            f"enable=ON, strength={strength:.2f}, max_boost={max_boost:.2f} dB, "
            f"pos_err_peak: {float(np.max(np.asarray(err_db, dtype=float)[pos_mask])):.2f} -> "
            f"{float(np.max(np.asarray(shaped_error, dtype=float)[pos_mask])):.2f} dB"
        )
    except (AttributeError, TypeError, ValueError):
        return


def _set_bass_adaptive_smoothing_state(
    st: Any,
    *,
    enabled: bool,
    avg_w_20_200: float = 0.0,
    sigma_scale: float | None = None,
    conf_floor: float | None = None,
    w_gamma: float | None = None,
    w_max: float | None = None,
) -> None:
    try:
        if not isinstance(st, dict):
            return
        st["bass_adaptive_smoothing_enabled"] = bool(enabled)
        st["bass_adaptive_smoothing_avg_w_20_200"] = float(avg_w_20_200)
        if sigma_scale is not None:
            st["bass_adaptive_smoothing_sigma_scale"] = float(sigma_scale)
        if conf_floor is not None:
            st["bass_adaptive_smoothing_conf_floor"] = float(conf_floor)
        if w_gamma is not None:
            st["bass_adaptive_smoothing_w_gamma"] = float(w_gamma)
        if w_max is not None:
            st["bass_adaptive_smoothing_w_max"] = float(w_max)
    except (TypeError, ValueError):
        pass


def _resolve_bass_smoothing_params(cfg_reader: CfgReader) -> tuple[float, float, float, float, float]:
    f_bass = cfg_reader.float("bass_smooth_hz", 200.0)
    if not np.isfinite(f_bass) or f_bass <= 0.0:
        f_bass = 200.0

    conf_floor = cfg_reader.float("bass_smooth_conf_floor", 0.3)
    if not np.isfinite(conf_floor) or conf_floor <= 0.0:
        conf_floor = 0.3

    sigma_scale = cfg_reader.float("bass_smooth_sigma_scale", 1.4)
    if (not np.isfinite(sigma_scale)) or sigma_scale < 1.0:
        sigma_scale = 1.0

    w_gamma = cfg_reader.float("bass_smooth_w_gamma", 2.0)
    if (not np.isfinite(w_gamma)) or w_gamma <= 0.0:
        w_gamma = 2.0

    w_max = cfg_reader.float("bass_smooth_w_max", 0.55)
    if not np.isfinite(w_max):
        w_max = 0.55
    w_max = float(np.clip(w_max, 0.0, 1.0))
    return float(f_bass), float(conf_floor), float(sigma_scale), float(w_gamma), float(w_max)


def _build_bass_smoothing_weights(conf_arr: np.ndarray, conf_floor: float, w_gamma: float, w_max: float) -> np.ndarray:
    c = np.asarray(conf_arr, dtype=float)
    c = np.nan_to_num(c, nan=1.0, posinf=1.0, neginf=0.0)
    c = np.clip(c, 0.0, 1.0)
    w = np.clip((conf_floor - c) / max(conf_floor, 1e-9), 0.0, 1.0)
    w = np.power(w, float(w_gamma))
    w = np.minimum(w, float(w_max))
    return np.clip(w, 0.0, 1.0)


def _blend_bass_smoothing(base: np.ndarray, more: np.ndarray, bass_mask: np.ndarray, weights: np.ndarray) -> np.ndarray:
    out = np.asarray(base, dtype=float).copy()
    w_eff = np.zeros_like(weights, dtype=float)
    w_eff[bass_mask] = np.asarray(weights[bass_mask], dtype=float)
    out[bass_mask] = (out[bass_mask] * (1.0 - w_eff[bass_mask])) + (np.asarray(more, dtype=float)[bass_mask] * w_eff[bass_mask])
    return out


def _apply_smoothing(
    err_db: np.ndarray,
    freq_axis: np.ndarray,
    cfg: Any,
    st: Any,
    filter_smooth: float,
    df_mode: bool,
    conf_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Soveltaa nykyisen valinnan mukaista smoothingia (DF tai fractional octave)."""
    cfg_reader = CfgReader(cfg)
    if df_mode:
        base_sigma = 60 // (filter_smooth / 12 if filter_smooth > 0 else 1)
        df_ref = 44100.0 / 65536.0
        sigma_hz = float(base_sigma) * df_ref
        sigma_bins = _sigma_bins_from_hz(freq_axis, sigma_hz=sigma_hz, fallback_bins=max(2.0, float(base_sigma)))
        smoothed_base = scipy.ndimage.gaussian_filter1d(err_db, sigma=float(sigma_bins))
    else:
        smoothed_base = smooth_gain_fractional_octave(freq_axis, err_db, filter_smooth)

    if not cfg_reader.bool("bass_smooth_adaptive", False):
        _set_bass_adaptive_smoothing_state(st, enabled=False)
        return smoothed_base

    try:
        conf_arr = np.asarray(conf_mask, dtype=float)
    except (TypeError, ValueError):
        conf_arr = None
    if conf_arr is None or conf_arr.shape != np.asarray(freq_axis, dtype=float).shape:
        _set_bass_adaptive_smoothing_state(st, enabled=False)
        return smoothed_base

    f_bass, conf_floor, sigma_scale, w_gamma, w_max = _resolve_bass_smoothing_params(cfg_reader)
    _set_bass_adaptive_smoothing_state(
        st,
        enabled=False,
        sigma_scale=sigma_scale,
        conf_floor=conf_floor,
        w_gamma=w_gamma,
        w_max=w_max,
    )

    f = np.asarray(freq_axis, dtype=float)
    w = _build_bass_smoothing_weights(conf_arr, conf_floor, w_gamma, w_max)
    bass = (f > 0.0) & (f <= float(f_bass))
    if not np.any(bass):
        _set_bass_adaptive_smoothing_state(
            st,
            enabled=False,
            sigma_scale=sigma_scale,
            conf_floor=conf_floor,
            w_gamma=w_gamma,
            w_max=w_max,
        )
        return smoothed_base

    if df_mode:
        sigma_bins2 = float(max(1.0, float(sigma_bins) * float(sigma_scale)))
        smoothed_more = scipy.ndimage.gaussian_filter1d(err_db, sigma=sigma_bins2)
    else:
        smooth2 = float(max(1.0, float(filter_smooth) / float(sigma_scale)))
        smoothed_more = smooth_gain_fractional_octave(freq_axis, err_db, smooth2)

    out = _blend_bass_smoothing(smoothed_base, smoothed_more, bass, w)
    w_eff = np.zeros_like(w, dtype=float)
    w_eff[bass] = np.asarray(w[bass], dtype=float)

    try:
        if isinstance(st, dict):
            _set_bass_adaptive_smoothing_state(
                st,
                enabled=True,
                avg_w_20_200=float(np.mean(w_eff[(f >= 20.0) & (f <= 200.0)]))
                if np.any((f >= 20.0) & (f <= 200.0))
                else 0.0,
                sigma_scale=sigma_scale,
                conf_floor=conf_floor,
                w_gamma=w_gamma,
                w_max=w_max,
            )
    except (TypeError, ValueError):
        pass
    return out


def _confidence_adaptive_bass_smoothing_mark_disabled(st: Any, stage_key: str) -> None:
    try:
        if not isinstance(st, dict):
            return
        safe_put_many(
            st,
            {
                "bass_adaptive_smoothing_enabled": False,
                "bass_adaptive_smoothing_avg_w_20_200": 0.0,
                "bass_adaptive_smoothing_delta_rms_db_20_200": 0.0,
                "bass_adaptive_smoothing_delta_max_db_20_200": 0.0,
                "bass_adaptive_smoothing_delta_max_hz_20_200": None,
                "bass_adaptive_smoothing_delta_rms_db_20_200_stage_local": 0.0,
                "bass_adaptive_smoothing_delta_max_db_20_200_stage_local": 0.0,
                "bass_adaptive_smoothing_delta_max_hz_20_200_stage_local": None,
                f"bass_adaptive_smoothing_{stage_key}_enabled": False,
                f"bass_adaptive_smoothing_{stage_key}_avg_w_20_200": 0.0,
                f"bass_adaptive_smoothing_{stage_key}_delta_rms_db_20_200": 0.0,
                f"bass_adaptive_smoothing_{stage_key}_delta_max_db_20_200": 0.0,
                f"bass_adaptive_smoothing_{stage_key}_delta_max_hz_20_200": None,
            },
        )
    except (TypeError, ValueError):
        pass


def _confidence_adaptive_bass_smoothing_stats(
    base: np.ndarray,
    out: np.ndarray,
    f: np.ndarray,
    w_eff: np.ndarray,
) -> tuple[float, float, float | None, float]:
    b20 = (f >= 20.0) & (f <= 200.0)
    delta = out - base
    if not np.any(b20):
        return 0.0, 0.0, None, 0.0
    d_rms = float(np.sqrt(np.mean(delta[b20] * delta[b20])))
    ad = np.abs(delta[b20])
    d_max = float(np.max(ad))
    fb = np.asarray(f[b20], dtype=float)
    d_hz = float(fb[int(np.argmax(ad))]) if (fb.size == ad.size and ad.size) else None
    w_avg = float(np.mean(w_eff[b20]))
    return d_rms, d_max, d_hz, w_avg


def _confidence_adaptive_bass_smoothing_run(
    base: np.ndarray,
    f: np.ndarray,
    conf_arr: np.ndarray,
    *,
    cfg_reader: CfgReader,
) -> tuple[np.ndarray, float, float, float | None, float, float] | None:
    f_bass, conf_floor, sigma_scale, w_gamma, w_max = _resolve_bass_smoothing_params(cfg_reader)
    w = _build_bass_smoothing_weights(conf_arr, conf_floor, w_gamma, w_max)
    bass = (f > 0.0) & (f <= float(f_bass))
    if not np.any(bass):
        return None

    sigma_hz = max(3.0, float(f_bass) / 24.0) * float(sigma_scale)
    sigma_bins = _sigma_bins_from_hz(
        f,
        sigma_hz=float(sigma_hz),
        fallback_bins=max(3.0, 6.0 * float(sigma_scale)),
    )
    sigma_bins = float(max(1.0, sigma_bins))
    sm_more = scipy.ndimage.gaussian_filter1d(base, sigma=sigma_bins)
    out = _blend_bass_smoothing(base, sm_more, bass, w)
    w_eff = np.zeros_like(w, dtype=float)
    w_eff[bass] = np.asarray(w[bass], dtype=float)
    d_rms, d_max, d_hz, w_avg = _confidence_adaptive_bass_smoothing_stats(base, out, f, w_eff)

    if d_rms < 5e-4 and w_avg > 0.10:
        sigma_bins2 = float(max(1.0, sigma_bins * 2.0))
        sm_more2 = scipy.ndimage.gaussian_filter1d(base, sigma=sigma_bins2)
        out2 = _blend_bass_smoothing(base, sm_more2, bass, w)
        d_rms2, d_max2, d_hz2, w_avg2 = _confidence_adaptive_bass_smoothing_stats(base, out2, f, w_eff)
        if d_rms2 > d_rms:
            out, d_rms, d_max, d_hz, w_avg, sigma_bins = out2, d_rms2, d_max2, d_hz2, w_avg2, sigma_bins2

    return out, d_rms, d_max, d_hz, w_avg, sigma_bins


def _apply_confidence_adaptive_bass_smoothing(
    curve_db: np.ndarray,
    freq_axis: np.ndarray,
    cfg: Any,
    st: Any,
    conf_mask: np.ndarray | None,
    *,
    stage_tag: str = "core",
) -> np.ndarray:
    """Lisasmoottaa bassoa confidence-maskin mukaan myohaisessa vaiheessa."""
    base = np.asarray(curve_db, dtype=float)
    f = np.asarray(freq_axis, dtype=float)
    stage_key = str(stage_tag or "core").strip().lower()
    cfg_reader = CfgReader(cfg)
    if not cfg_reader.bool("bass_smooth_adaptive", False):
        _confidence_adaptive_bass_smoothing_mark_disabled(st, stage_key)
        return base

    try:
        c = np.asarray(conf_mask, dtype=float)
    except (TypeError, ValueError):
        c = None
    if c is None or c.shape != f.shape:
        _confidence_adaptive_bass_smoothing_mark_disabled(st, stage_key)
        return base

    result = _confidence_adaptive_bass_smoothing_run(base, f, c, cfg_reader=cfg_reader)
    if result is None:
        _confidence_adaptive_bass_smoothing_mark_disabled(st, stage_key)
        return base
    out, d_rms, d_max, d_hz, w_avg, sigma_bins = result

    try:
        if isinstance(st, dict):
            safe_put_many(
                st,
                {
                    "bass_adaptive_smoothing_enabled": True,
                    "bass_adaptive_smoothing_avg_w_20_200": float(w_avg),
                    "bass_adaptive_smoothing_delta_rms_db_20_200": float(d_rms),
                    "bass_adaptive_smoothing_delta_max_db_20_200": float(d_max),
                    "bass_adaptive_smoothing_delta_max_hz_20_200": float(d_hz) if d_hz is not None else None,
                    "bass_adaptive_smoothing_delta_rms_db_20_200_stage_local": float(d_rms),
                    "bass_adaptive_smoothing_delta_max_db_20_200_stage_local": float(d_max),
                    "bass_adaptive_smoothing_delta_max_hz_20_200_stage_local": float(d_hz) if d_hz is not None else None,
                    "bass_adaptive_smoothing_sigma_bins": float(sigma_bins),
                    f"bass_adaptive_smoothing_{stage_key}_enabled": True,
                    f"bass_adaptive_smoothing_{stage_key}_avg_w_20_200": float(w_avg),
                    f"bass_adaptive_smoothing_{stage_key}_delta_rms_db_20_200": float(d_rms),
                    f"bass_adaptive_smoothing_{stage_key}_delta_max_db_20_200": float(d_max),
                    f"bass_adaptive_smoothing_{stage_key}_delta_max_hz_20_200": float(d_hz) if d_hz is not None else None,
                },
            )
    except (TypeError, ValueError):
        pass
    return out

def _select_bass_adaptive_conf_mask(
    conf_mask: np.ndarray | None,
    bf_conf_for_smoothing: np.ndarray | None,
    *,
    use_bassfirst: bool,
) -> tuple[np.ndarray | None, str]:
    """Valitsee bass-adaptive smoothingille conf-maskin ilman bassfirst-floor lockia."""
    try:
        c_raw = np.asarray(conf_mask, dtype=float) if conf_mask is not None else None
    except (TypeError, ValueError):
        c_raw = None
    try:
        c_bf = np.asarray(bf_conf_for_smoothing, dtype=float) if bf_conf_for_smoothing is not None else None
    except (TypeError, ValueError):
        c_bf = None

    if c_raw is not None:
        c_raw = np.clip(np.nan_to_num(c_raw, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        return c_raw, "raw_conf"

    if bool(use_bassfirst) and c_bf is not None:
        c_bf = np.clip(np.nan_to_num(c_bf, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        return c_bf, "bassfirst_fallback"

    if c_bf is not None:
        c_bf = np.clip(np.nan_to_num(c_bf, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        return c_bf, "bassfirst_fallback"

    return None, "missing"


def _mid_refit_write_stats(
    st: Any,
    *,
    enabled: bool,
    k_used: float,
    before_rms: float | None,
    after_rms: float | None,
    delta_rms: float | None,
    conf_avg: float | None,
    reason: str,
) -> None:
    try:
        if not isinstance(st, dict):
            return
        before_val = float(before_rms) if before_rms is not None and np.isfinite(float(before_rms)) else None
        after_val = float(after_rms) if after_rms is not None and np.isfinite(float(after_rms)) else None
        delta_val = float(delta_rms) if delta_rms is not None and np.isfinite(float(delta_rms)) else None
        conf_val = float(conf_avg) if conf_avg is not None and np.isfinite(float(conf_avg)) else None
        safe_put_many(
            st,
            {
                "mid_refit_enabled": bool(enabled),
                "mid_refit_k": float(k_used),
                "mid_refit_err_rms_before": before_val,
                "mid_refit_err_rms_after": after_val,
                "mid_refit_delta_rms": delta_val,
                "mid_refit_err_rms_before_stage_local": before_val,
                "mid_refit_err_rms_after_stage_local": after_val,
                "mid_refit_delta_rms_stage_local": delta_val,
                "mid_refit_conf_avg_200_2000": conf_val,
                "mid_refit_reason": str(reason),
            },
        )
    except (TypeError, ValueError):
        pass


def _mid_refit_resolve_params(cfg_reader: CfgReader) -> tuple[float, float, float, float, float]:
    mid_lo = cfg_reader.float("mid_refit_hz_lo", 200.0)
    mid_hi = cfg_reader.float("mid_refit_hz_hi", 2000.0)
    if (not np.isfinite(mid_lo)) or mid_lo < 1.0:
        mid_lo = 200.0
    if (not np.isfinite(mid_hi)) or mid_hi <= (mid_lo + 1.0):
        mid_hi = 2000.0
        if mid_hi <= (mid_lo + 1.0):
            mid_hi = mid_lo + 1.0

    k_refit = cfg_reader.float("mid_refit_k", 0.45)
    if not np.isfinite(k_refit):
        k_refit = 0.45
    k_refit = float(np.clip(k_refit, 0.0, 1.0))

    smooth_oct = cfg_reader.float("mid_refit_smooth_oct", 0.6)
    if (not np.isfinite(smooth_oct)) or smooth_oct <= 0.0:
        smooth_oct = 0.6
    smooth_oct = float(np.clip(smooth_oct, 1.0 / 192.0, 1.0))
    smooth_value = float(np.clip(1.0 / smooth_oct, 1.0, 192.0))

    conf_min_avg = cfg_reader.float("mid_refit_conf_min_avg", 0.2)
    if not np.isfinite(conf_min_avg):
        conf_min_avg = 0.2
    conf_min_avg = float(np.clip(conf_min_avg, 0.0, 1.0))
    return float(mid_lo), float(mid_hi), float(k_refit), float(smooth_value), float(conf_min_avg)


def _mid_refit_prepare_arrays(
    gain_db: np.ndarray,
    freq_axis: np.ndarray,
    mask_c: np.ndarray,
    m_anal: np.ndarray,
    target_mags: np.ndarray,
    *,
    mid_lo: float,
    mid_hi: float,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int] | None, str]:
    f = np.asarray(freq_axis, dtype=float).reshape(-1)
    g = np.asarray(gain_db, dtype=float).reshape(-1)
    m = np.asarray(mask_c, dtype=bool).reshape(-1)
    meas = np.asarray(m_anal, dtype=float).reshape(-1)
    tgt = np.asarray(target_mags, dtype=float).reshape(-1)
    n = int(min(f.size, g.size, m.size, meas.size, tgt.size))
    if n < 32:
        return None, "insufficient_data"

    f = f[:n]
    g = g[:n]
    m = m[:n]
    meas = meas[:n]
    tgt = tgt[:n]
    valid = np.isfinite(f) & np.isfinite(g) & np.isfinite(m) & np.isfinite(meas) & np.isfinite(tgt)
    mid = valid & m & (f >= float(mid_lo)) & (f <= float(mid_hi))
    if int(np.count_nonzero(mid)) < 8:
        return None, "no_mid_bins"
    return (f, g, m, meas, tgt, mid, n), "ok"


def _mid_refit_measure_conf_avg(conf_mask: np.ndarray | None, n: int, mid: np.ndarray) -> float | None:
    try:
        if conf_mask is None:
            return None
        c = np.asarray(conf_mask, dtype=float).reshape(-1)
        if c.size < n:
            return None
        c = np.clip(np.nan_to_num(c[:n], nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
        return float(np.mean(c[mid])) if np.any(mid) else None
    except (TypeError, ValueError, FloatingPointError, IndexError):
        return None


def _mid_refit_apply(
    out: np.ndarray,
    *,
    f: np.ndarray,
    g: np.ndarray,
    mid: np.ndarray,
    meas: np.ndarray,
    tgt: np.ndarray,
    calc_offset_db: float,
    conf_mask: np.ndarray | None,
    conf_min_avg: float,
    k_refit: float,
    smooth_value: float,
    mid_lo: float,
    mid_hi: float,
    n: int,
    st: Any,
    logger: Any,
) -> np.ndarray:
    conf_avg = _mid_refit_measure_conf_avg(conf_mask, n, mid)
    if conf_avg is not None and conf_avg < float(conf_min_avg):
        err_before = tgt - ((meas - float(calc_offset_db)) + g)
        rms_before = float(np.sqrt(np.mean(err_before[mid] * err_before[mid])))
        _mid_refit_write_stats(
            st,
            enabled=False,
            k_used=0.0,
            before_rms=rms_before,
            after_rms=rms_before,
            delta_rms=0.0,
            conf_avg=conf_avg,
            reason="low_mid_conf",
        )
        return out

    err_before = tgt - ((meas - float(calc_offset_db)) + g)
    rms_before = float(np.sqrt(np.mean(err_before[mid] * err_before[mid])))
    err_s = smooth_gain_fractional_octave(f, err_before, smooth_value)

    g_refit = g.copy()
    g_refit[mid] = g_refit[mid] + float(k_refit) * np.asarray(err_s[mid], dtype=float)
    err_after = tgt - ((meas - float(calc_offset_db)) + g_refit)
    rms_after = float(np.sqrt(np.mean(err_after[mid] * err_after[mid])))
    delta_rms = float(rms_before - rms_after)

    if delta_rms <= 0.0:
        _mid_refit_write_stats(
            st,
            enabled=False,
            k_used=0.0,
            before_rms=rms_before,
            after_rms=rms_before,
            delta_rms=0.0,
            conf_avg=conf_avg,
            reason="no_improvement",
        )
        return out

    out[:n] = g_refit
    _mid_refit_write_stats(
        st,
        enabled=True,
        k_used=float(k_refit),
        before_rms=rms_before,
        after_rms=rms_after,
        delta_rms=delta_rms,
        conf_avg=conf_avg,
        reason="applied",
    )
    try:
        conf_txt = "n/a" if conf_avg is None else f"{float(conf_avg):.3f}"
        logger.info(
            "MidRefit: "
            f"k={float(k_refit):.2f}, smooth_oct={float(1.0 / smooth_value):.2f}, "
            f"conf_avg={conf_txt}, "
            f"rms_before={float(rms_before):.3f} dB, rms_after={float(rms_after):.3f} dB, "
            f"delta={float(delta_rms):.3f} dB, band={float(mid_lo):.0f}-{float(mid_hi):.0f} Hz"
        )
    except (AttributeError, TypeError, ValueError):
        pass
    return out


def _apply_mid_refit_pre_slope(
    gain_db: np.ndarray,
    freq_axis: np.ndarray,
    mask_c: np.ndarray,
    *,
    m_anal: np.ndarray,
    target_mags: np.ndarray,
    calc_offset_db: float,
    conf_mask: np.ndarray | None,
    cfg: Any,
    st: Any,
    logger: Any,
) -> np.ndarray:
    """Kevyt mid-band match-pass ennen slope-rajoja."""
    out = np.asarray(gain_db, dtype=float).copy()
    cfg_reader = CfgReader(cfg)
    if not cfg_reader.bool("enable_mag_correction", False):
        _mid_refit_write_stats(
            st,
            enabled=False,
            k_used=0.0,
            before_rms=None,
            after_rms=None,
            delta_rms=None,
            conf_avg=None,
            reason="mag_off",
        )
        return out

    refit_on = cfg_reader.bool("mid_refit_enable", True)
    if not refit_on:
        _mid_refit_write_stats(
            st,
            enabled=False,
            k_used=0.0,
            before_rms=None,
            after_rms=None,
            delta_rms=None,
            conf_avg=None,
            reason="disabled",
        )
        return out

    mid_lo, mid_hi, k_refit, smooth_value, conf_min_avg = _mid_refit_resolve_params(cfg_reader)
    if k_refit <= 0.0:
        _mid_refit_write_stats(
            st,
            enabled=False,
            k_used=0.0,
            before_rms=None,
            after_rms=None,
            delta_rms=None,
            conf_avg=None,
            reason="k<=0",
        )
        return out

    prepared, prep_reason = _mid_refit_prepare_arrays(
        out,
        freq_axis,
        mask_c,
        m_anal,
        target_mags,
        mid_lo=mid_lo,
        mid_hi=mid_hi,
    )
    if prepared is None:
        _mid_refit_write_stats(
            st,
            enabled=False,
            k_used=0.0,
            before_rms=None,
            after_rms=None,
            delta_rms=None,
            conf_avg=None,
            reason=prep_reason,
        )
        return out

    f, g, m, meas, tgt, mid, n = prepared
    return _mid_refit_apply(
        out,
        f=f,
        g=g,
        mid=mid,
        meas=meas,
        tgt=tgt,
        calc_offset_db=calc_offset_db,
        conf_mask=conf_mask,
        conf_min_avg=conf_min_avg,
        k_refit=k_refit,
        smooth_value=smooth_value,
        mid_lo=mid_lo,
        mid_hi=mid_hi,
        n=n,
        st=st,
        logger=logger,
    )

def _apply_bass_boost_post_restore(
    gain_db: np.ndarray,
    target_db: np.ndarray,
    boost_cap_db: np.ndarray,
    freq_axis: np.ndarray,
    mask_c: np.ndarray,
    *,
    hz_lo: float = 20.0,
    hz_hi: float = 200.0,
    strength: float = 0.6,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    return apply_bass_boost_post_restore(
        gain_db,
        target_db,
        boost_cap_db,
        freq_axis,
        mask_c,
        hz_lo=hz_lo,
        hz_hi=hz_hi,
        strength=strength,
    )

def _apply_confpull_post_slope(
    gain_db_in: np.ndarray,
    mask_c_in: np.ndarray,
    measured_ref_db: np.ndarray | None,
    *,
    cfg: Any,
    st: Any,
    conf_mask: np.ndarray,
    freq_axis: np.ndarray,
    logger: Any,
    apply_confidence_weighted_target_pull: Callable[..., Any],
) -> np.ndarray:
    return apply_confpull_post_slope(
        gain_db_in,
        mask_c_in,
        measured_ref_db,
        cfg=cfg,
        st=st,
        conf_mask=conf_mask,
        freq_axis=freq_axis,
        logger=logger,
        apply_confidence_weighted_target_pull=apply_confidence_weighted_target_pull,
    )

def _apply_post_limits_and_metrics(inputs: _MagPostProcessInputs) -> _MagPostProcessOutputs:
    return _apply_post_limits_and_metrics_impl(
        inputs,
        apply_mid_refit_pre_slope=_apply_mid_refit_pre_slope,
    )


__all__ = ['_apply_peak_priority_error_shaping', '_apply_smoothing', '_apply_confidence_adaptive_bass_smoothing', '_select_bass_adaptive_conf_mask', '_apply_mid_refit_pre_slope', '_apply_bass_boost_post_restore', '_apply_confpull_post_slope', '_apply_post_limits_and_metrics']


def _link_sibling_exports() -> None:
    import importlib
    package = __package__
    for module_name in ['correction_mag_01', 'correction_mag_02']:
        if module_name == __name__.rsplit('.', 1)[-1]:
            continue
        module = importlib.import_module(f"{package}.{module_name}")
        for symbol in getattr(module, "__all__", ()):
            globals().setdefault(symbol, getattr(module, symbol))


_link_sibling_exports()
