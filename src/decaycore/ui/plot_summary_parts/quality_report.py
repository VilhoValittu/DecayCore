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
import scipy.ndimage

logger = logging.getLogger("DecayCore")

_RECOVERABLE_QUALITY_EXCEPTIONS = (
    AttributeError,
    TypeError,
    ValueError,
    KeyError,
    IndexError,
    RuntimeError,
    OverflowError,
    FloatingPointError,
)
_RECOVERABLE_FLOAT_PARSE_EXCEPTIONS = (TypeError, ValueError, OverflowError)

from ...common.acoustic_stats import _clamp, calc_acoustic_score, calc_ai_summary_from_stats
from ...dsp.phase_ir_metrics import format_pre_energy_status
from ...dsp.quality_metrics import _mag_error_db, _rms
from ...dsp.target_match import target_match_from_stats as _target_match_from_stats_ssot
















_calc_acoustic_score = calc_acoustic_score


def format_band_rt60_summary(bands, *, picks=None):
    if not bands:
        return "-"
    pick_list = [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0] if picks is None else list(picks)
    normalized = []
    if isinstance(bands, dict):
        items = list(bands.items())
    else:
        items = list(bands or [])
    for key, value in items:
        try:
            freq = float(key)
            rt60 = float(value)
        except _RECOVERABLE_FLOAT_PARSE_EXCEPTIONS:
            continue
        if np.isfinite(freq) and np.isfinite(rt60):
            normalized.append((float(freq), float(rt60)))
    if not normalized:
        return "-"
    seen = set()
    out = []
    for pick in pick_list:
        freq, rt60 = min(normalized, key=lambda item: abs(float(item[0]) - float(pick)))
        label = f"{float(freq):.0f}Hz"
        if label in seen:
            continue
        seen.add(label)
        out.append(f"{label}:{float(rt60):.2f}s")
    return " | ".join(out) if out else "-"

def _float_allow_zero(v, default: float) -> float:
    """Sisainen apufunktio: float allow zero."""
    if v is None:
        return float(default)
    if isinstance(v, str) and v.strip() == "":
        return float(default)
    try:
        return float(v)
    except _RECOVERABLE_QUALITY_EXCEPTIONS:
        return float(default)

def _calc_target_match(stats):
    """Sisainen apufunktio: calc target match."""
    return _target_match_from_stats_ssot(
        stats or {},
        include_filter=True,
        use_confidence=True,
        use_smart_scan_range=True,
    )

def calc_target_match_from_stats(stats: dict):
    """Laskee: calc target match from stats."""
    try:
        return _calc_target_match(stats or {})
    except _RECOVERABLE_QUALITY_EXCEPTIONS:
        return None, None


def _quality_report_safe_float(v):
    if v is None:
        return None
    try:
        x = float(v)
        if np.isfinite(x):
            return float(x)
    except (TypeError, ValueError):
        logger.exception("safe float parse in quality report")
    return None


def _quality_report_as_arr(v):
    try:
        a = np.asarray(v, dtype=float).reshape(-1)
    except _RECOVERABLE_QUALITY_EXCEPTIONS:
        return np.asarray([], dtype=float)
    if a.size == 0:
        return np.asarray([], dtype=float)
    return a


def _quality_report_gd_grad_max_value(st):
    for k in (
        "gd_grad_limiter_after_max_ms_per_oct",
        "gd_grad_limiter_before_max_ms_per_oct",
        "gd_limiter_max_grad_ms_per_oct",
        "gd_grad_limiter_max_grad_ms_per_oct",
        "gd_limiter_max_grad_after_ms_per_oct",
        "gd_grad_limiter_max_grad_after_ms_per_oct",
        "gd_limiter_max_grad_before_ms_per_oct",
        "gd_grad_limiter_max_grad_before_ms_per_oct",
    ):
        v = _quality_report_safe_float(st.get(k, None))
        if v is not None:
            return v
    return None


def _quality_report_gd_grad_max_hz(st):
    for k in (
        "gd_grad_limiter_peak_hz",
        "gd_limiter_max_grad_hz",
        "gd_grad_limiter_max_grad_hz",
        "gd_limiter_max_grad_after_hz",
        "gd_grad_limiter_max_grad_after_hz",
        "gd_limiter_max_grad_before_hz",
        "gd_grad_limiter_max_grad_before_hz",
    ):
        v = _quality_report_safe_float(st.get(k, None))
        if v is not None:
            return v
    return None


def _quality_report_gd_max_value(st):
    for k in (
        "gd_max_ms",
        "max_gd_ms",
        "mixed_excess_delay_before_ms",
        "xo_diff_raw_max_gd_ms",
        "hpf_diff_raw_max_gd_ms",
    ):
        v = _quality_report_safe_float(st.get(k, None))
        if v is not None:
            return abs(v)
    refs = st.get("reflections", []) or []
    vals = []
    for ref in refs:
        if isinstance(ref, dict):
            g = _quality_report_safe_float(ref.get("gd_error", None))
            if g is not None:
                vals.append(abs(g))
    if vals:
        return float(max(vals))
    return None


def _quality_report_pre_ringing_db(st):
    if bool(st.get("pre_energy_metric_suspect", False)):
        return None
    for k in (
        "ir_pre_ringing_db",
        "mixed_pre_ringing_after_db",
        "ir_pre_energy_guard_after_db",
        "mixed_pre_ringing_before_db",
        "ir_pre_energy_guard_before_db",
    ):
        v = _quality_report_safe_float(st.get(k, None))
        if v is not None:
            return v
    return None


def _quality_report_pre_post_ratio(st, pre_db):
    if bool(st.get("pre_energy_metric_suspect", False)):
        return None
    for k in (
        "ir_pre_post_ratio",
        "ir_pre_energy_guard_after_ratio",
        "ir_pre_energy_guard_before_ratio",
    ):
        v = _quality_report_safe_float(st.get(k, None))
        if v is not None and v >= 0.0:
            return v
    if pre_db is None:
        return None
    try:
        r = float(10.0 ** (float(pre_db) / 10.0))
        return r if np.isfinite(r) else None
    except _RECOVERABLE_QUALITY_EXCEPTIONS:
        return None


def _quality_report_pre_metric_info(st, *, debug_report=False):
    valid = bool(st.get("pre_energy_metric_valid", not bool(st.get("pre_energy_metric_suspect", False))))
    suspect = bool(not valid)
    reason_code = str(st.get("pre_energy_metric_reason_code", "") or "").strip()
    note = "ok" if not suspect else format_pre_energy_status(reason_code, debug=debug_report)
    return suspect, note


def _quality_report_active_axes_repair_arrays(m_raw, m_corr, g_pred, g_real, g_legacy, offset_db, g_legacy_src):
    if m_raw.size < 8 and m_corr.size >= 8:
        m_raw = np.asarray(m_corr, dtype=float) + float(offset_db)
    if m_corr.size < 8 and m_raw.size >= 8:
        m_corr = np.asarray(m_raw, dtype=float) - float(offset_db)
    if g_pred.size < 8 and g_legacy.size >= 8 and g_legacy_src != "ir_fft_final":
        g_pred = g_legacy
    if g_real.size < 8 and g_legacy.size >= 8 and g_legacy_src == "ir_fft_final":
        g_real = g_legacy
    if g_pred.size < 8 and g_legacy.size >= 8:
        g_pred = g_legacy
    if g_real.size < 8 and g_legacy.size >= 8:
        g_real = g_legacy
    return m_corr, g_pred, g_real


def _quality_report_active_axes_from_prefix(st, *, prefix: str):
    pref = str(prefix or "")
    f = _quality_report_as_arr(st.get(f"{pref}freq_axis", None))
    t = _quality_report_as_arr(st.get(f"{pref}target_mags", None))
    mm_key = "cmp_mag_mask" if pref else "mag_mask"
    mm = _quality_report_as_arr(st.get(mm_key, st.get("mask_c", None) if not pref else None))
    offset_db = _quality_report_safe_float(st.get(f"{pref}offset_db", 0.0))
    if offset_db is None:
        offset_db = 0.0
    m_raw = _quality_report_as_arr(st.get(f"{pref}measured_mags_raw", None))
    m_corr = _quality_report_as_arr(st.get(f"{pref}measured_mags", None))
    g_pred = _quality_report_as_arr(st.get(f"{pref}predicted_filter_mags", None))
    g_real = _quality_report_as_arr(st.get(f"{pref}realized_filter_mags", None))
    g_legacy = _quality_report_as_arr(st.get(f"{pref}filter_mags", None))
    g_legacy_src = str(st.get(f"{pref}filter_mags_source", "") or "").strip().lower()
    m_corr, g_pred, g_real = _quality_report_active_axes_repair_arrays(
        m_raw,
        m_corr,
        g_pred,
        g_real,
        g_legacy,
        float(offset_db),
        g_legacy_src,
    )
    return f, m_corr, t, g_pred, g_real, mm, float(offset_db)


def _quality_report_active_axes(st):
    mode = str(st.get("analysis_mode", "native") or "native").strip().lower()
    if mode == "comparison":
        return _quality_report_active_axes_from_prefix(st, prefix="cmp_")
    return _quality_report_active_axes_from_prefix(st, prefix="")


def _quality_report_phase_limit_hz(st, settings):
    for k in ("phase_limit_hz", "phase_limit"):
        v = _quality_report_safe_float(st.get(k, None))
        if v is not None and v > 0.0:
            return float(v)
    v = _quality_report_safe_float(settings.get("phase_limit", None))
    if v is not None and v > 0.0:
        return float(v)
    return None


def _quality_report_phase_boundary_peak_mdb(freqs, filt_db, valid_mask, phase_lim_hz):
    if phase_lim_hz is None:
        return None, None
    f = np.asarray(freqs, dtype=float)
    g = np.asarray(filt_db, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool)
    if f.size < 32 or g.size != f.size or valid.size != f.size:
        return None, None
    lim = float(phase_lim_hz)
    lo = max(20.0, 0.75 * lim)
    hi = min(float(np.max(f[np.isfinite(f)])) if np.any(np.isfinite(f)) else lim, 1.20 * lim)
    if hi <= (lo + 1.0):
        return None, None
    sel = valid & (f >= lo) & (f <= hi)
    if np.count_nonzero(sel) < 12:
        return None, None
    fv = f[sel]
    gv = g[sel]
    sigma = float(np.clip(float(gv.size) / 18.0, 1.0, 14.0))
    try:
        trend = scipy.ndimage.gaussian_filter1d(gv, sigma=sigma, mode="nearest")
        resid = gv - trend
        idx = int(np.argmax(np.abs(resid)))
        return float(np.abs(resid[idx]) * 1000.0), float(fv[idx])
    except _RECOVERABLE_QUALITY_EXCEPTIONS:
        return None, None


def _quality_report_band_stats(freqs, err, valid, *, lo_hz: float, hi_hz: float):
    m_band = valid & (freqs >= float(lo_hz)) & (freqs <= float(hi_hz))
    if np.count_nonzero(m_band) < 8:
        return None, None, None
    ev_loc = np.asarray(err[m_band], dtype=float)
    fv_loc = np.asarray(freqs[m_band], dtype=float)
    rms_loc = _rms(ev_loc)
    if not np.isfinite(rms_loc):
        rms_loc = None
    max_loc = None
    hz_loc = None
    if ev_loc.size:
        idx_loc = int(np.argmax(np.abs(ev_loc)))
        max_loc = float(np.abs(ev_loc[idx_loc]))
        if fv_loc.size > idx_loc:
            hz_loc = float(fv_loc[idx_loc])
    return rms_loc, max_loc, hz_loc


def _quality_report_compute_error_bundle(
    freqs,
    measured_aligned,
    target_db,
    gain_db,
    mag_mask,
    *,
    include_ripple=False,
    cmin=20.0,
    cmax=20000.0,
):
    res = {
        "rms_magc": None,
        "max_magc": None,
        "max_hz_magc": None,
        "rms_20_200": None,
        "max_20_200": None,
        "rms_200_2000": None,
        "ripple_rms": None,
        "valid": None,
    }
    n_loc = int(min(freqs.size, measured_aligned.size, target_db.size, gain_db.size))
    if n_loc < 8:
        return res
    f_loc = np.asarray(freqs[:n_loc], dtype=float)
    m_loc = np.asarray(measured_aligned[:n_loc], dtype=float)
    t_loc = np.asarray(target_db[:n_loc], dtype=float)
    g_loc = np.asarray(gain_db[:n_loc], dtype=float)
    if mag_mask.size >= n_loc:
        mm_loc = np.asarray(mag_mask[:n_loc], dtype=float)
    else:
        mm_loc = np.asarray([], dtype=float)

    valid = (
        np.isfinite(f_loc)
        & np.isfinite(m_loc)
        & np.isfinite(t_loc)
        & np.isfinite(g_loc)
        & (f_loc > 0.0)
    )
    err = _mag_error_db(t_loc, m_loc, g_loc)

    rms_20_200, max_20_200, _ = _quality_report_band_stats(f_loc, err, valid, lo_hz=20.0, hi_hz=200.0)
    rms_200_2000, _, _ = _quality_report_band_stats(f_loc, err, valid, lo_hz=200.0, hi_hz=2000.0)
    res["rms_20_200"] = rms_20_200
    res["max_20_200"] = max_20_200
    res["rms_200_2000"] = rms_200_2000

    if mm_loc.size == n_loc:
        band = np.asarray(mm_loc, dtype=float) > 0.5
    else:
        band = (f_loc >= float(cmin)) & (f_loc <= float(cmax))
    mask = valid & band
    if np.count_nonzero(mask) < 8:
        mask = valid & (f_loc >= float(cmin)) & (f_loc <= float(cmax))
    if np.count_nonzero(mask) >= 8:
        ev = np.asarray(err[mask], dtype=float)
        fv = np.asarray(f_loc[mask], dtype=float)
        rms_magc = _rms(ev)
        res["rms_magc"] = rms_magc if np.isfinite(rms_magc) else None
        if ev.size:
            idx = int(np.argmax(np.abs(ev)))
            res["max_magc"] = float(np.abs(ev[idx]))
            res["max_hz_magc"] = float(fv[idx]) if fv.size else None
        if include_ripple:
            try:
                sigma = max(1.0, float(ev.size) / 64.0)
                ev_sm = scipy.ndimage.gaussian_filter1d(ev, sigma=sigma)
                rp = ev - ev_sm
                rp_rms = _rms(rp)
                res["ripple_rms"] = rp_rms if np.isfinite(rp_rms) else None
            except _RECOVERABLE_QUALITY_EXCEPTIONS:
                res["ripple_rms"] = None

    res["valid"] = valid
    return res


def _quality_report_collect(st, settings, *, debug_report=False):
    out = {
        "pred_mag_error_rms": None,
        "pred_mag_error_max": None,
        "pred_mag_error_max_hz": None,
        "pred_mag_error_rms_20_200": None,
        "pred_mag_error_max_20_200": None,
        "pred_mag_error_rms_200_2000": None,
        "real_mag_error_rms": None,
        "real_mag_error_max": None,
        "real_mag_error_max_hz": None,
        "real_mag_error_rms_20_200": None,
        "real_mag_error_max_20_200": None,
        "real_mag_error_rms_200_2000": None,
        "mid_refit_enabled": bool(st.get("mid_refit_enabled", False)),
        "mid_refit_reason": str(st.get("mid_refit_reason", "") or ""),
        "mid_refit_k": _quality_report_safe_float(st.get("mid_refit_k", None)),
        "mid_refit_err_rms_before": _quality_report_safe_float(st.get("mid_refit_err_rms_before", None)),
        "mid_refit_err_rms_after": _quality_report_safe_float(st.get("mid_refit_err_rms_after", None)),
        "mid_refit_delta_rms": _quality_report_safe_float(st.get("mid_refit_delta_rms", None)),
        "mid_refit_conf_avg_200_2000": _quality_report_safe_float(st.get("mid_refit_conf_avg_200_2000", None)),
        "ripple_rms": None,
        "gd_max": _quality_report_gd_max_value(st),
        "gd_abs_max_20_500_ms": _quality_report_safe_float(st.get("gd_abs_max_20_500_ms", None)),
        "gd_grad_max": _quality_report_gd_grad_max_value(st),
        "gd_grad_max_hz": _quality_report_gd_grad_max_hz(st),
        "phase_boundary_peak_mdb": None,
        "phase_boundary_peak_hz": None,
        "bass_adaptive_enabled": bool(st.get("bass_adaptive_smoothing_enabled", False)),
        "bass_adaptive_conf_source": str(st.get("bass_adaptive_conf_source", "") or ""),
        "bass_adaptive_isolation_mode": bool(st.get("bass_adaptive_isolation_mode", False)),
        "bass_adaptive_sigma_scale": _quality_report_safe_float(st.get("bass_adaptive_smoothing_sigma_scale", None)),
        "bass_adaptive_conf_floor": _quality_report_safe_float(st.get("bass_adaptive_smoothing_conf_floor", None)),
        "bass_adaptive_w_gamma": _quality_report_safe_float(st.get("bass_adaptive_smoothing_w_gamma", None)),
        "bass_adaptive_w_max": _quality_report_safe_float(st.get("bass_adaptive_smoothing_w_max", None)),
        "bass_adaptive_avg_w": _quality_report_safe_float(st.get("bass_adaptive_smoothing_avg_w_20_200", None)),
        "bass_adaptive_delta_rms_20_200": _quality_report_safe_float(st.get("bass_adaptive_smoothing_delta_rms_db_20_200", None)),
        "bass_adaptive_delta_max_20_200": _quality_report_safe_float(st.get("bass_adaptive_smoothing_delta_max_db_20_200", None)),
        "bass_adaptive_delta_max_hz_20_200": _quality_report_safe_float(st.get("bass_adaptive_smoothing_delta_max_hz_20_200", None)),
        "bass_adaptive_delta_basis": str(st.get("bass_adaptive_smoothing_delta_basis", "") or ""),
        "bass_adaptive_effectiveness_pct": None,
        "post_to_ir_delta_rms_20_200": _quality_report_safe_float(st.get("post_to_ir_delta_rms_20_200_db", None)),
        "post_to_ir_delta_max_20_200": _quality_report_safe_float(st.get("post_to_ir_delta_max_20_200_db", None)),
        "post_to_ir_delta_max_hz_20_200": _quality_report_safe_float(st.get("post_to_ir_delta_max_hz_20_200", None)),
        "post_to_ir_delta_offset_20_200": _quality_report_safe_float(st.get("post_to_ir_delta_offset_20_200_db", None)),
        "post_to_ir_shape_delta_rms_20_200": _quality_report_safe_float(st.get("post_to_ir_shape_delta_rms_20_200_db", None)),
        "post_to_ir_shape_delta_max_20_200": _quality_report_safe_float(st.get("post_to_ir_shape_delta_max_20_200_db", None)),
        "post_to_ir_shape_delta_max_hz_20_200": _quality_report_safe_float(st.get("post_to_ir_shape_delta_max_hz_20_200", None)),
        "post_to_ir_staged_delta_rms_20_200": _quality_report_safe_float(st.get("post_to_ir_staged_delta_rms_20_200_db", None)),
        "post_to_ir_staged_delta_max_20_200": _quality_report_safe_float(st.get("post_to_ir_staged_delta_max_20_200_db", None)),
        "post_to_ir_staged_delta_max_hz_20_200": _quality_report_safe_float(st.get("post_to_ir_staged_delta_max_hz_20_200", None)),
        "post_to_ir_staged_delta_offset_20_200": _quality_report_safe_float(st.get("post_to_ir_staged_delta_offset_20_200_db", None)),
        "post_to_ir_staged_shape_delta_rms_20_200": _quality_report_safe_float(st.get("post_to_ir_staged_shape_delta_rms_20_200_db", None)),
        "post_to_ir_staged_shape_delta_max_20_200": _quality_report_safe_float(st.get("post_to_ir_staged_shape_delta_max_20_200_db", None)),
        "post_to_ir_staged_shape_delta_max_hz_20_200": _quality_report_safe_float(st.get("post_to_ir_staged_shape_delta_max_hz_20_200", None)),
        "ir_realized_level_match_enabled": bool(st.get("ir_realized_level_match_enabled", False)),
        "ir_realized_level_match_applied": bool(st.get("ir_realized_level_match_applied", False)),
        "ir_realized_level_match_reason": str(st.get("ir_realized_level_match_reason", "") or ""),
        "ir_realized_level_match_mid_lo_hz": _quality_report_safe_float(st.get("ir_realized_level_match_mid_lo_hz", None)),
        "ir_realized_level_match_mid_hi_hz": _quality_report_safe_float(st.get("ir_realized_level_match_mid_hi_hz", None)),
        "ir_realized_level_match_delta_db_raw": _quality_report_safe_float(st.get("ir_realized_level_match_delta_db_raw", None)),
        "ir_realized_level_match_delta_db_applied": _quality_report_safe_float(st.get("ir_realized_level_match_delta_db_applied", None)),
        "ir_realized_level_match_delta_db_after": _quality_report_safe_float(st.get("ir_realized_level_match_delta_db_after", None)),
        "ir_realized_level_match_scale": _quality_report_safe_float(st.get("ir_realized_level_match_scale", None)),
        "post_to_ir_delta_rms_magc": _quality_report_safe_float(st.get("post_to_ir_delta_rms_magc_db", None)),
        "post_to_ir_delta_max_magc": _quality_report_safe_float(st.get("post_to_ir_delta_max_magc_db", None)),
        "post_to_ir_delta_max_hz_magc": _quality_report_safe_float(st.get("post_to_ir_delta_max_hz_magc", None)),
        "bass_boost_cap_enabled": bool(st.get("bass_boost_cap_enabled", False)),
        "bass_boost_cap_avg_extra_db_20_200": _quality_report_safe_float(st.get("bass_boost_cap_avg_extra_db_20_200", None)),
        "bass_boost_cap_max_extra_db_20_200": _quality_report_safe_float(st.get("bass_boost_cap_max_extra_db_20_200", None)),
        "bass_boost_post_restore_applied": bool(st.get("bass_boost_post_restore_applied", False)),
        "bass_boost_post_restore_strength": _quality_report_safe_float(st.get("bass_boost_post_restore_strength", None)),
        "bass_boost_post_restore_bins": _quality_report_safe_float(st.get("bass_boost_post_restore_bins", None)),
        "bass_boost_post_restore_delta_rms_20_200": _quality_report_safe_float(st.get("bass_boost_post_restore_delta_rms_20_200", None)),
        "bass_boost_post_restore_delta_max_20_200": _quality_report_safe_float(st.get("bass_boost_post_restore_delta_max_20_200", None)),
        "conf_pull_bass_boost_floor_hz": _quality_report_safe_float(st.get("conf_pull_post_bass_boost_floor_hz", None)),
        "conf_pull_bass_boost_floor_min": _quality_report_safe_float(st.get("conf_pull_post_bass_boost_floor_min", None)),
        "conf_pull_bass_boost_restore": _quality_report_safe_float(st.get("conf_pull_post_bass_boost_restore", None)),
        "conf_pull_bass_boost_restore_mean_eff": _quality_report_safe_float(st.get("conf_pull_post_bass_boost_restore_mean_eff", None)),
        "pre_ringing_db": None,
        "ir_pre_post_ratio": None,
        "pre_metric_suspect": False,
        "pre_metric_note": "",
    }
    f, m_corr, t, g_pred, g_real, mm, offset_db = _quality_report_active_axes(st)
    if min(f.size, m_corr.size, t.size, g_pred.size) < 8:
        pre_db = _quality_report_pre_ringing_db(st)
        out["pre_ringing_db"] = pre_db
        out["ir_pre_post_ratio"] = _quality_report_pre_post_ratio(st, pre_db)
        out["pre_metric_suspect"], out["pre_metric_note"] = _quality_report_pre_metric_info(st, debug_report=debug_report)
        return out

    cmin = _quality_report_safe_float(st.get("mag_c_min", settings.get("mag_c_min", 20.0)))
    cmax = _quality_report_safe_float(st.get("mag_c_max", settings.get("mag_c_max", 20000.0)))
    if cmin is None:
        cmin = 20.0
    if cmax is None or cmax <= cmin:
        cmax = float(np.max(f)) if f.size else (cmin + 1.0)

    pred = _quality_report_compute_error_bundle(f, m_corr, t, g_pred, mm, include_ripple=True, cmin=cmin, cmax=cmax)
    out["pred_mag_error_rms"] = pred["rms_magc"]
    out["pred_mag_error_max"] = pred["max_magc"]
    out["pred_mag_error_max_hz"] = pred["max_hz_magc"]
    out["pred_mag_error_rms_20_200"] = pred["rms_20_200"]
    out["pred_mag_error_max_20_200"] = pred["max_20_200"]
    out["pred_mag_error_rms_200_2000"] = pred["rms_200_2000"]
    out["ripple_rms"] = pred["ripple_rms"]

    real = _quality_report_compute_error_bundle(f, m_corr, t, g_real, mm, include_ripple=False, cmin=cmin, cmax=cmax)
    out["real_mag_error_rms"] = real["rms_magc"]
    out["real_mag_error_max"] = real["max_magc"]
    out["real_mag_error_max_hz"] = real["max_hz_magc"]
    out["real_mag_error_rms_20_200"] = real["rms_20_200"]
    out["real_mag_error_max_20_200"] = real["max_20_200"]
    out["real_mag_error_rms_200_2000"] = real["rms_200_2000"]

    n_phase = int(min(f.size, g_real.size))
    if n_phase < 8:
        n_phase = int(min(f.size, g_pred.size))
        g_phase = np.asarray(g_pred[:n_phase], dtype=float)
    else:
        g_phase = np.asarray(g_real[:n_phase], dtype=float)
    f_phase = np.asarray(f[:n_phase], dtype=float)
    valid_phase = np.isfinite(f_phase) & np.isfinite(g_phase) & (f_phase > 0.0)
    phase_lim = _quality_report_phase_limit_hz(st, settings)
    pb_mdb, pb_hz = _quality_report_phase_boundary_peak_mdb(f_phase, g_phase, valid_phase, phase_lim)
    out["phase_boundary_peak_mdb"] = pb_mdb
    out["phase_boundary_peak_hz"] = pb_hz

    pre_db = _quality_report_pre_ringing_db(st)
    out["pre_ringing_db"] = pre_db
    out["ir_pre_post_ratio"] = _quality_report_pre_post_ratio(st, pre_db)
    out["pre_metric_suspect"], out["pre_metric_note"] = _quality_report_pre_metric_info(st, debug_report=debug_report)
    try:
        dmax = _quality_report_safe_float(out.get("bass_adaptive_delta_max_20_200"))
        emax = _quality_report_safe_float(out.get("pred_mag_error_max_20_200"))
        if dmax is not None and emax is not None and emax > 1e-9:
            out["bass_adaptive_effectiveness_pct"] = float((dmax / emax) * 100.0)
    except _RECOVERABLE_QUALITY_EXCEPTIONS:
        logger.exception("bass adaptive effectiveness compute")
    return out


def _quality_report_fmt(v, unit="", prec=2):
    if v is None:
        return "n/a"
    try:
        x = float(v)
        if not np.isfinite(x):
            return "n/a"
        return f"{x:.{int(prec)}f}{unit}"
    except _RECOVERABLE_FLOAT_PARSE_EXCEPTIONS:
        return "n/a"


def _quality_report_fmt_ratio(v):
    if v is None:
        return "n/a"
    try:
        x = float(v)
        if not np.isfinite(x):
            return "n/a"
        return f"{x:.4g}"
    except _RECOVERABLE_FLOAT_PARSE_EXCEPTIONS:
        return "n/a"


def _quality_report_fmt_onoff(v):
    return "ON" if bool(v) else "OFF"


def _quality_report_fmt_src(v):
    s = str(v or "").strip()
    return s or "n/a"


def _quality_report_build_lines(lq, rq, *, debug_report: bool):
    lines = [
        "",
        "--- DSP Quality Report ---",
        f"Predicted mag error RMS within mag_c band:      L {_quality_report_fmt(lq['pred_mag_error_rms'], ' dB')} | R {_quality_report_fmt(rq['pred_mag_error_rms'], ' dB')}",
        f"Predicted mag error max within mag_c band:      L {_quality_report_fmt(lq['pred_mag_error_max'], ' dB')} | R {_quality_report_fmt(rq['pred_mag_error_max'], ' dB')}",
        f"Predicted mag error max within mag_c band @ Hz: L {_quality_report_fmt(lq['pred_mag_error_max_hz'], '', 1)} | R {_quality_report_fmt(rq['pred_mag_error_max_hz'], '', 1)}",
        f"Predicted mag error RMS @ 20-200 Hz:            L {_quality_report_fmt(lq['pred_mag_error_rms_20_200'], ' dB')} | R {_quality_report_fmt(rq['pred_mag_error_rms_20_200'], ' dB')}",
        f"Predicted mag error max @ 20-200 Hz:            L {_quality_report_fmt(lq['pred_mag_error_max_20_200'], ' dB')} | R {_quality_report_fmt(rq['pred_mag_error_max_20_200'], ' dB')}",
        f"Predicted mag error RMS @ 200-2000 Hz:          L {_quality_report_fmt(lq['pred_mag_error_rms_200_2000'], ' dB')} | R {_quality_report_fmt(rq['pred_mag_error_rms_200_2000'], ' dB')}",
        f"Realized mag error RMS within mag_c band:       L {_quality_report_fmt(lq['real_mag_error_rms'], ' dB')} | R {_quality_report_fmt(rq['real_mag_error_rms'], ' dB')}",
        f"Realized mag error max within mag_c band:       L {_quality_report_fmt(lq['real_mag_error_max'], ' dB')} | R {_quality_report_fmt(rq['real_mag_error_max'], ' dB')}",
        f"Realized mag error max within mag_c band @ Hz:  L {_quality_report_fmt(lq['real_mag_error_max_hz'], '', 1)} | R {_quality_report_fmt(rq['real_mag_error_max_hz'], '', 1)}",
        f"Realized mag error RMS @ 20-200 Hz:             L {_quality_report_fmt(lq['real_mag_error_rms_20_200'], ' dB')} | R {_quality_report_fmt(rq['real_mag_error_rms_20_200'], ' dB')}",
        f"Realized mag error RMS @ 200-2000 Hz:           L {_quality_report_fmt(lq['real_mag_error_rms_200_2000'], ' dB')} | R {_quality_report_fmt(rq['real_mag_error_rms_200_2000'], ' dB')}",
        f"Ripple RMS:              L {_quality_report_fmt(lq['ripple_rms'], ' dB')} | R {_quality_report_fmt(rq['ripple_rms'], ' dB')}",
        f"GD max (ms):             L {_quality_report_fmt(lq['gd_max'], '', 2)} | R {_quality_report_fmt(rq['gd_max'], '', 2)}",
        f"GD abs spread 20-500 Hz (ms): L {_quality_report_fmt(lq['gd_abs_max_20_500_ms'], '', 2)} | R {_quality_report_fmt(rq['gd_abs_max_20_500_ms'], '', 2)}",
        f"GD gradient max (ms/oct): L {_quality_report_fmt(lq['gd_grad_max'], '', 2)} | R {_quality_report_fmt(rq['gd_grad_max'], '', 2)}",
        f"GD gradient max @ Hz:    L {_quality_report_fmt(lq['gd_grad_max_hz'], '', 1)} | R {_quality_report_fmt(rq['gd_grad_max_hz'], '', 1)}",
        f"Phase boundary peak (mdB): L {_quality_report_fmt(lq['phase_boundary_peak_mdb'], '', 2)} | R {_quality_report_fmt(rq['phase_boundary_peak_mdb'], '', 2)}",
        f"Phase boundary peak @ Hz:  L {_quality_report_fmt(lq['phase_boundary_peak_hz'], '', 1)} | R {_quality_report_fmt(rq['phase_boundary_peak_hz'], '', 1)}",
        f"Pre-ringing dB:          L {_quality_report_fmt(lq['pre_ringing_db'], ' dB')} | R {_quality_report_fmt(rq['pre_ringing_db'], ' dB')}",
        f"IR pre/post energy ratio: L {_quality_report_fmt_ratio(lq['ir_pre_post_ratio'])} | R {_quality_report_fmt_ratio(rq['ir_pre_post_ratio'])}",
    ]
    if debug_report:
        lines += [
            f"Mid re-fit 200-2000: L {_quality_report_fmt_onoff(lq['mid_refit_enabled'])} (k {_quality_report_fmt(lq['mid_refit_k'], '', 2)}, conf {_quality_report_fmt(lq['mid_refit_conf_avg_200_2000'], '', 2)}) | R {_quality_report_fmt_onoff(rq['mid_refit_enabled'])} (k {_quality_report_fmt(rq['mid_refit_k'], '', 2)}, conf {_quality_report_fmt(rq['mid_refit_conf_avg_200_2000'], '', 2)})",
            f"Mid re-fit err RMS 200-2000 (dB): L {_quality_report_fmt(lq['mid_refit_err_rms_before'], '', 2)} -> {_quality_report_fmt(lq['mid_refit_err_rms_after'], '', 2)} (delta {_quality_report_fmt(lq['mid_refit_delta_rms'], '', 2)}) | R {_quality_report_fmt(rq['mid_refit_err_rms_before'], '', 2)} -> {_quality_report_fmt(rq['mid_refit_err_rms_after'], '', 2)} (delta {_quality_report_fmt(rq['mid_refit_delta_rms'], '', 2)})",
            f"Mid re-fit reason: L {_quality_report_fmt_src(lq['mid_refit_reason'])} | R {_quality_report_fmt_src(rq['mid_refit_reason'])}",
            f"Bass adaptive smoothing:  L {_quality_report_fmt_onoff(lq['bass_adaptive_enabled'])} | R {_quality_report_fmt_onoff(rq['bass_adaptive_enabled'])}",
            f"Bass adaptive conf source: L {_quality_report_fmt_src(lq['bass_adaptive_conf_source'])} | R {_quality_report_fmt_src(rq['bass_adaptive_conf_source'])}",
            f"Bass adaptive isolation mode: L {_quality_report_fmt_onoff(lq['bass_adaptive_isolation_mode'])} | R {_quality_report_fmt_onoff(rq['bass_adaptive_isolation_mode'])}",
            f"Bass adaptive params: L sigma {_quality_report_fmt(lq['bass_adaptive_sigma_scale'], '', 2)} / conf {_quality_report_fmt(lq['bass_adaptive_conf_floor'], '', 2)} / gamma {_quality_report_fmt(lq['bass_adaptive_w_gamma'], '', 2)} / wmax {_quality_report_fmt(lq['bass_adaptive_w_max'], '', 2)} | R sigma {_quality_report_fmt(rq['bass_adaptive_sigma_scale'], '', 2)} / conf {_quality_report_fmt(rq['bass_adaptive_conf_floor'], '', 2)} / gamma {_quality_report_fmt(rq['bass_adaptive_w_gamma'], '', 2)} / wmax {_quality_report_fmt(rq['bass_adaptive_w_max'], '', 2)}",
            f"Bass adaptive avg w 20-200: L {_quality_report_fmt(lq['bass_adaptive_avg_w'], '', 3)} | R {_quality_report_fmt(rq['bass_adaptive_avg_w'], '', 3)}",
            f"Bass adaptive delta RMS 20-200 (dB): L {_quality_report_fmt(lq['bass_adaptive_delta_rms_20_200'], '', 2)} | R {_quality_report_fmt(rq['bass_adaptive_delta_rms_20_200'], '', 2)}",
            f"Bass adaptive delta max 20-200 (dB): L {_quality_report_fmt(lq['bass_adaptive_delta_max_20_200'], '', 2)} | R {_quality_report_fmt(rq['bass_adaptive_delta_max_20_200'], '', 2)}",
            f"Bass adaptive delta max @ Hz (20-200): L {_quality_report_fmt(lq['bass_adaptive_delta_max_hz_20_200'], '', 1)} | R {_quality_report_fmt(rq['bass_adaptive_delta_max_hz_20_200'], '', 1)}",
            f"Bass adaptive effectiveness (% of max err 20-200): L {_quality_report_fmt(lq['bass_adaptive_effectiveness_pct'], '%', 1)} | R {_quality_report_fmt(rq['bass_adaptive_effectiveness_pct'], '%', 1)}",
            f"Bass adaptive delta basis: L {_quality_report_fmt_src(lq['bass_adaptive_delta_basis'])} | R {_quality_report_fmt_src(rq['bass_adaptive_delta_basis'])}",
            f"Post->IR delta RMS 20-200 (dB): L {_quality_report_fmt(lq['post_to_ir_delta_rms_20_200'], '', 2)} | R {_quality_report_fmt(rq['post_to_ir_delta_rms_20_200'], '', 2)}",
            f"Post->IR delta max @ Hz (20-200): L {_quality_report_fmt(lq['post_to_ir_delta_max_20_200'], '', 2)} @ {_quality_report_fmt(lq['post_to_ir_delta_max_hz_20_200'], '', 1)} | R {_quality_report_fmt(rq['post_to_ir_delta_max_20_200'], '', 2)} @ {_quality_report_fmt(rq['post_to_ir_delta_max_hz_20_200'], '', 1)}",
            f"Post->IR offset 20-200 (dB): L {_quality_report_fmt(lq['post_to_ir_delta_offset_20_200'], '', 2)} | R {_quality_report_fmt(rq['post_to_ir_delta_offset_20_200'], '', 2)}",
            f"Post->IR shape delta RMS/max 20-200 (dB): L {_quality_report_fmt(lq['post_to_ir_shape_delta_rms_20_200'], '', 2)} / {_quality_report_fmt(lq['post_to_ir_shape_delta_max_20_200'], '', 2)} @ {_quality_report_fmt(lq['post_to_ir_shape_delta_max_hz_20_200'], '', 1)} | R {_quality_report_fmt(rq['post_to_ir_shape_delta_rms_20_200'], '', 2)} / {_quality_report_fmt(rq['post_to_ir_shape_delta_max_20_200'], '', 2)} @ {_quality_report_fmt(rq['post_to_ir_shape_delta_max_hz_20_200'], '', 1)}",
            f"Post+staging->IR delta RMS 20-200 (dB): L {_quality_report_fmt(lq['post_to_ir_staged_delta_rms_20_200'], '', 2)} | R {_quality_report_fmt(rq['post_to_ir_staged_delta_rms_20_200'], '', 2)}",
            f"Post+staging->IR delta max @ Hz (20-200): L {_quality_report_fmt(lq['post_to_ir_staged_delta_max_20_200'], '', 2)} @ {_quality_report_fmt(lq['post_to_ir_staged_delta_max_hz_20_200'], '', 1)} | R {_quality_report_fmt(rq['post_to_ir_staged_delta_max_20_200'], '', 2)} @ {_quality_report_fmt(rq['post_to_ir_staged_delta_max_hz_20_200'], '', 1)}",
            f"Post+staging->IR offset 20-200 (dB): L {_quality_report_fmt(lq['post_to_ir_staged_delta_offset_20_200'], '', 2)} | R {_quality_report_fmt(rq['post_to_ir_staged_delta_offset_20_200'], '', 2)}",
            f"Post+staging->IR shape delta RMS/max 20-200 (dB): L {_quality_report_fmt(lq['post_to_ir_staged_shape_delta_rms_20_200'], '', 2)} / {_quality_report_fmt(lq['post_to_ir_staged_shape_delta_max_20_200'], '', 2)} @ {_quality_report_fmt(lq['post_to_ir_staged_shape_delta_max_hz_20_200'], '', 1)} | R {_quality_report_fmt(rq['post_to_ir_staged_shape_delta_rms_20_200'], '', 2)} / {_quality_report_fmt(rq['post_to_ir_staged_shape_delta_max_20_200'], '', 2)} @ {_quality_report_fmt(rq['post_to_ir_staged_shape_delta_max_hz_20_200'], '', 1)}",
            f"IR mid-band level match: L {_quality_report_fmt_onoff(lq['ir_realized_level_match_applied'])} ({_quality_report_fmt_src(lq['ir_realized_level_match_reason'])}) | R {_quality_report_fmt_onoff(rq['ir_realized_level_match_applied'])} ({_quality_report_fmt_src(rq['ir_realized_level_match_reason'])})",
            f"IR mid-band level match delta dB: L raw {_quality_report_fmt(lq['ir_realized_level_match_delta_db_raw'], '', 2)} / applied {_quality_report_fmt(lq['ir_realized_level_match_delta_db_applied'], '', 2)} / after {_quality_report_fmt(lq['ir_realized_level_match_delta_db_after'], '', 2)} | R raw {_quality_report_fmt(rq['ir_realized_level_match_delta_db_raw'], '', 2)} / applied {_quality_report_fmt(rq['ir_realized_level_match_delta_db_applied'], '', 2)} / after {_quality_report_fmt(rq['ir_realized_level_match_delta_db_after'], '', 2)}",
            f"IR mid-band level match scale/band: L x{_quality_report_fmt(lq['ir_realized_level_match_scale'], '', 4)} @ {_quality_report_fmt(lq['ir_realized_level_match_mid_lo_hz'], '', 0)}-{_quality_report_fmt(lq['ir_realized_level_match_mid_hi_hz'], '', 0)} Hz | R x{_quality_report_fmt(rq['ir_realized_level_match_scale'], '', 4)} @ {_quality_report_fmt(rq['ir_realized_level_match_mid_lo_hz'], '', 0)}-{_quality_report_fmt(rq['ir_realized_level_match_mid_hi_hz'], '', 0)} Hz",
            f"Bass boost cap: L {_quality_report_fmt_onoff(lq['bass_boost_cap_enabled'])} | R {_quality_report_fmt_onoff(rq['bass_boost_cap_enabled'])}",
            f"Bass boost cap extra 20-200 (dB): L avg {_quality_report_fmt(lq['bass_boost_cap_avg_extra_db_20_200'], '', 2)} / max {_quality_report_fmt(lq['bass_boost_cap_max_extra_db_20_200'], '', 2)} | R avg {_quality_report_fmt(rq['bass_boost_cap_avg_extra_db_20_200'], '', 2)} / max {_quality_report_fmt(rq['bass_boost_cap_max_extra_db_20_200'], '', 2)}",
            f"Bass boost post-restore: L {_quality_report_fmt_onoff(lq['bass_boost_post_restore_applied'])} (set {_quality_report_fmt(lq['bass_boost_post_restore_strength'], '', 2)}, bins {_quality_report_fmt(lq['bass_boost_post_restore_bins'], '', 0)}) | R {_quality_report_fmt_onoff(rq['bass_boost_post_restore_applied'])} (set {_quality_report_fmt(rq['bass_boost_post_restore_strength'], '', 2)}, bins {_quality_report_fmt(rq['bass_boost_post_restore_bins'], '', 0)})",
            f"Bass boost post-restore delta 20-200 (dB): L rms {_quality_report_fmt(lq['bass_boost_post_restore_delta_rms_20_200'], '', 2)} / max {_quality_report_fmt(lq['bass_boost_post_restore_delta_max_20_200'], '', 2)} | R rms {_quality_report_fmt(rq['bass_boost_post_restore_delta_rms_20_200'], '', 2)} / max {_quality_report_fmt(rq['bass_boost_post_restore_delta_max_20_200'], '', 2)}",
            f"Conf-pull bass boost floor: L {_quality_report_fmt(lq['conf_pull_bass_boost_floor_min'], '', 2)} <= {_quality_report_fmt(lq['conf_pull_bass_boost_floor_hz'], '', 0)} Hz | R {_quality_report_fmt(rq['conf_pull_bass_boost_floor_min'], '', 2)} <= {_quality_report_fmt(rq['conf_pull_bass_boost_floor_hz'], '', 0)} Hz",
            f"Conf-pull bass boost restore: L set {_quality_report_fmt(lq['conf_pull_bass_boost_restore'], '', 2)} / eff {_quality_report_fmt(lq['conf_pull_bass_boost_restore_mean_eff'], '', 2)} | R set {_quality_report_fmt(rq['conf_pull_bass_boost_restore'], '', 2)} / eff {_quality_report_fmt(rq['conf_pull_bass_boost_restore_mean_eff'], '', 2)}",
        ]
    if bool(lq.get("pre_metric_suspect", False)) or bool(rq.get("pre_metric_suspect", False)):
        l_note = str(lq.get("pre_metric_note", "") or "suspect").strip()
        r_note = str(rq.get("pre_metric_note", "") or "suspect").strip()
        lines.append(f"Pre-energy metric sanity: L {l_note} | R {r_note}")
    return lines


def format_dsp_quality_report_block(settings, l_stats, r_stats):
    """Rakenna Summaryyn DSP Quality Report -blokki (L/R)."""
    settings = settings or {}
    l_stats = l_stats or {}
    r_stats = r_stats or {}
    debug_report = bool(settings.get("quality_report_debug", False))
    lq = _quality_report_collect(l_stats, settings, debug_report=debug_report)
    rq = _quality_report_collect(r_stats, settings, debug_report=debug_report)
    return _quality_report_build_lines(lq, rq, debug_report=debug_report)


__all__ = [
    'format_band_rt60_summary',
    '_float_allow_zero',
    '_calc_target_match',
    'calc_target_match_from_stats',
    'format_dsp_quality_report_block',
    '_clamp',
    'calc_acoustic_score',
    '_calc_acoustic_score',
    'calc_ai_summary_from_stats',
]
