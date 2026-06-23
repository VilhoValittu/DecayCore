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

from ...common.house_curves import get_house_curve_by_name

logger = logging.getLogger("DecayCore")
from ...dsp.smoothing import smooth_meas_freq_dep
from ...dsp.target_synthesis import synthesize_target_from_measurements
from ..cache_signature import get_or_build_synth_target
from ..shared import (
    AUTO_MODE_BUILTIN_TARGETS,
    AUTO_MODE_SYNTH_TARGET_ENABLED,
    AUTO_MODE_SYNTH_TARGET_NAME,
    AUTO_MODE_SYNTH_TARGET_BASS_COMP_FRAC,
    AUTO_MODE_SYNTH_TARGET_BASS_COMP_REF_DB,
    AUTO_MODE_SYNTH_TARGET_HF_COMP_FRAC,
    AUTO_MODE_TARGET_PRESELECT_ASYM_W,
    AUTO_MODE_TARGET_PRESELECT_BASS_SHAPE_W,
    AUTO_MODE_TARGET_PRESELECT_BOOST_W,
    AUTO_MODE_TARGET_PRESELECT_MAX_BASS_BOOST_REF_DB,
    AUTO_MODE_TARGET_PRESELECT_MODE_BAND_MAX_HZ,
    AUTO_MODE_TARGET_PRESELECT_MODE_BAND_MIN_HZ,
    AUTO_MODE_TARGET_PRESELECT_MODE_W,
    AUTO_MODE_TARGET_PRESELECT_SLOPE_W,
    AUTO_MODE_TARGET_PRESELECT_TREBLE_SHAPE_W,
    AUTO_MODE_TARGET_TOP_N,
    AUTO_MODE_TARGET_TOP_N_MAX,
    AUTO_MODE_TARGET_TOP_N_MIN,
    AUTO_MODE_TARGET_TOP_N_SPREAD_DB,
    _auto_builtin_target_name,
    _auto_safe_float,
    _auto_safe_int,
)

def _auto_target_one_step_milder(hc_name: str) -> str | None:
    name = str(hc_name or "").strip()
    if not name:
        return None
    ladders = (
        ("Harman4", "Harman6", "Harman8", "Harman10", "Harman12"),
        ("BK_Light", "BK_Medium", "BK_Strong"),
    )
    for ladder in ladders:
        if name not in ladder:
            continue
        idx = int(ladder.index(name))
        if idx <= 0:
            return None
        return str(ladder[idx - 1])
    return None


def _auto_target_preselect_safe_mask(raw_mask, *, n: int, fallback=None, min_pts: int = 8):
    try:
        mk = np.asarray(raw_mask, dtype=bool).reshape(-1)
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError, NameError):
        mk = np.asarray([], dtype=bool)
    if mk.size != n:
        if fallback is not None:
            mk = np.asarray(fallback, dtype=bool).reshape(-1)
        else:
            mk = np.ones(n, dtype=bool)
    if int(np.count_nonzero(mk)) < int(max(1, min_pts)):
        if fallback is not None:
            mk = np.asarray(fallback, dtype=bool).reshape(-1)
        if mk.size != n or int(np.count_nonzero(mk)) < int(max(1, min_pts)):
            mk = np.ones(n, dtype=bool)
    return np.asarray(mk, dtype=bool)


def _auto_target_preselect_safe_median(v, default=0.0) -> float:
    vv = np.asarray(v, dtype=float).reshape(-1)
    vv = vv[np.isfinite(vv)]
    if vv.size <= 0:
        return float(default)
    return float(np.median(vv))


def _auto_target_preselect_safe_rms(v, default=float("nan")) -> float:
    vv = np.asarray(v, dtype=float).reshape(-1)
    vv = vv[np.isfinite(vv)]
    if vv.size <= 0:
        return float(default)
    return float(np.sqrt(np.mean(np.square(vv))))


def _auto_target_preselect_safe_pos_mean(v, default=0.0) -> float:
    vv = np.asarray(v, dtype=float).reshape(-1)
    vv = vv[np.isfinite(vv)]
    if vv.size <= 0:
        return float(default)
    return float(np.mean(np.maximum(vv, 0.0)))


def _auto_target_preselect_weighted_band_median_rms(ff, err_v, bands, *, min_pts: int = 6, default=0.0) -> float:
    ev = np.asarray(err_v, dtype=float).reshape(-1)
    ff = np.asarray(ff, dtype=float).reshape(-1)
    if ev.size != ff.size:
        return float(default)
    acc = 0.0
    w_sum = 0.0
    for lo_hz, hi_hz, weight in list(bands or []):
        try:
            lo = float(lo_hz)
            hi = float(hi_hz)
            w = float(weight)
        except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError, NameError):
            continue
        if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo or w <= 0.0:
            continue
        m_band = np.isfinite(ev) & (ff >= lo) & (ff < hi)
        if int(np.count_nonzero(m_band)) < int(max(1, min_pts)):
            continue
        band_med = _auto_target_preselect_safe_median(ev[m_band], 0.0)
        acc += float(w) * float(band_med) * float(band_med)
        w_sum += float(w)
    if w_sum <= 1e-9:
        return float(default)
    return float(np.sqrt(acc / w_sum))


def _auto_target_preselect_sort_xy(f, y):
    idx = np.argsort(f)
    ff = np.asarray(f[idx], dtype=float)
    yy = np.asarray(y[idx], dtype=float)
    m = np.isfinite(ff) & np.isfinite(yy) & (ff > 0.0)
    return ff[m], yy[m]


def _auto_target_preselect_common_ranges(data: dict) -> tuple[float, float, float, float, float, float]:
    try:
        lvl_min = float(data.get("lvl_min", 500.0) or 500.0)
        lvl_max = float(data.get("lvl_max", 2000.0) or 2000.0)
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError, NameError):
        lvl_min, lvl_max = 500.0, 2000.0
    if not np.isfinite(lvl_min) or not np.isfinite(lvl_max) or lvl_min <= 0.0 or lvl_max <= lvl_min:
        lvl_min, lvl_max = 500.0, 2000.0

    try:
        mag_lo = float(
            data.get(
                "_auto_mag_c_min_hz",
                data.get("mag_c_min", 20.0),
            )
            or 20.0
        )
        mag_hi = float(data.get("mag_c_max", 250.0) or 250.0)
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError, NameError):
        mag_lo, mag_hi = 20.0, 250.0
    if not np.isfinite(mag_lo) or not np.isfinite(mag_hi) or mag_lo <= 0.0 or mag_hi <= mag_lo:
        mag_lo, mag_hi = 20.0, 250.0

    mode_lo = float(_auto_safe_float(AUTO_MODE_TARGET_PRESELECT_MODE_BAND_MIN_HZ, 25.0))
    mode_hi = float(_auto_safe_float(AUTO_MODE_TARGET_PRESELECT_MODE_BAND_MAX_HZ, 160.0))
    if (not np.isfinite(mode_lo)) or (not np.isfinite(mode_hi)) or mode_hi <= mode_lo:
        mode_lo, mode_hi = 25.0, 160.0
    return float(lvl_min), float(lvl_max), float(mag_lo), float(mag_hi), float(mode_lo), float(mode_hi)


def _auto_target_preselect_prepare_grid_and_masks(
    data: dict,
    *,
    fl,
    ml,
    fr,
    mr,
    hf,
    hm,
):
    lvl_min, lvl_max, mag_lo, mag_hi, mode_lo, mode_hi = _auto_target_preselect_common_ranges(data)
    try:
        f_lo = max(20.0, float(np.min(fl)), float(np.min(fr)), float(np.min(hf)))
        f_hi = min(20000.0, float(np.max(fl)), float(np.max(fr)), float(np.max(hf)))
        if not np.isfinite(f_lo) or not np.isfinite(f_hi) or f_hi <= f_lo * 1.15:
            return None
        fg = np.logspace(np.log10(f_lo), np.log10(f_hi), 320)
        ml_g = np.interp(fg, fl, ml)
        mr_g = np.interp(fg, fr, mr)
        try:
            ml_g = smooth_meas_freq_dep(ml_g, fg)
        except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError, NameError):
            logger.exception("left smoothing in target preselection score")
        try:
            mr_g = smooth_meas_freq_dep(mr_g, fg)
        except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError, NameError):
            logger.exception("right smoothing in target preselection score")
        t_g = np.interp(fg, hf, hm)
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError, NameError):
        return None

    lvl_mask = (fg >= lvl_min) & (fg <= lvl_max)
    if int(np.count_nonzero(lvl_mask)) < 16:
        lvl_mask = (fg >= 300.0) & (fg <= 3000.0)
    if int(np.count_nonzero(lvl_mask)) < 16:
        lvl_mask = np.ones_like(fg, dtype=bool)

    corr_mask = (fg >= mag_lo) & (fg <= mag_hi)
    if int(np.count_nonzero(corr_mask)) < 16:
        corr_mask = np.ones_like(fg, dtype=bool)

    mode_mask = (fg >= mode_lo) & (fg <= mode_hi) & corr_mask
    if int(np.count_nonzero(mode_mask)) < 8:
        mode_mask = (fg >= mode_lo) & (fg <= mode_hi)

    return fg, ml_g, mr_g, t_g, lvl_mask, corr_mask, mode_mask


def _auto_target_preselect_score_curve(
    data: dict,
    *,
    fl,
    ml,
    fr,
    mr,
    hf,
    hm,
) -> dict | None:
    try:
        fl = np.asarray(fl, dtype=float).reshape(-1)
        ml = np.asarray(ml, dtype=float).reshape(-1)
        fr = np.asarray(fr, dtype=float).reshape(-1)
        mr = np.asarray(mr, dtype=float).reshape(-1)
        hf = np.asarray(hf, dtype=float).reshape(-1)
        hm = np.asarray(hm, dtype=float).reshape(-1)
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError, NameError):
        return None
    fl, ml = _auto_target_preselect_sort_xy(fl, ml)
    fr, mr = _auto_target_preselect_sort_xy(fr, mr)
    hf, hm = _auto_target_preselect_sort_xy(hf, hm)
    if fl.size < 32 or fr.size < 32 or hf.size < 4 or hm.size != hf.size:
        return None
    prepared = _auto_target_preselect_prepare_grid_and_masks(
        data,
        fl=fl,
        ml=ml,
        fr=fr,
        mr=mr,
        hf=hf,
        hm=hm,
    )
    if prepared is None:
        return None
    fg, ml_g, mr_g, t_g, lvl_mask, corr_mask, mode_mask = prepared
    tc = _auto_target_preselect_score(
        fg=fg,
        ml_g=ml_g,
        mr_g=mr_g,
        t_g=t_g,
        lvl_mask=lvl_mask,
        corr_mask=corr_mask,
        mode_mask=mode_mask,
    )
    if not isinstance(tc, dict) or not tc:
        return None
    tc["_preselect_fg"] = fg
    tc["_preselect_tg"] = t_g
    return dict(tc)

def _auto_target_slope_estimate(f_hz, mag_db, *, mask=None) -> float:
    try:
        ff = np.asarray(f_hz, dtype=float).reshape(-1)
        mm = np.asarray(mag_db, dtype=float).reshape(-1)
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError, NameError):
        return float("nan")
    if ff.size <= 6 or mm.size != ff.size:
        return float("nan")
    m = np.isfinite(ff) & np.isfinite(mm) & (ff > 0.0)
    try:
        if mask is not None:
            mk = np.asarray(mask, dtype=bool).reshape(-1)
            if mk.size == ff.size:
                m &= mk
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError, NameError):
        logger.exception("mask apply in slope estimate")
    if int(np.count_nonzero(m)) < 6:
        return float("nan")
    x = np.log10(ff[m])
    y = mm[m]
    if x.size < 6:
        return float("nan")
    x_span = float(np.max(x) - np.min(x))
    if (not np.isfinite(x_span)) or x_span <= 1e-6:
        return float("nan")
    try:
        p = np.polyfit(x, y, 1)
        slope_db_per_dec = float(p[0])
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError, NameError):
        return float("nan")
    return float(slope_db_per_dec * np.log10(2.0))

def _auto_target_preselect_score(
    *,
    fg,
    ml_g,
    mr_g,
    t_g,
    lvl_mask,
    corr_mask,
    mode_mask,
) -> dict:
    ff = np.asarray(fg, dtype=float).reshape(-1)
    ml_arr = np.asarray(ml_g, dtype=float).reshape(-1)
    mr_arr = np.asarray(mr_g, dtype=float).reshape(-1)
    tg_arr = np.asarray(t_g, dtype=float).reshape(-1)
    n = int(ff.size)
    if n <= 0 or ml_arr.size != n or mr_arr.size != n or tg_arr.size != n:
        return {}

    lvl_m = _auto_target_preselect_safe_mask(lvl_mask, n=n, fallback=np.ones(n, dtype=bool), min_pts=8)
    corr_m = _auto_target_preselect_safe_mask(corr_mask, n=n, fallback=np.ones(n, dtype=bool), min_pts=8)
    mode_m = _auto_target_preselect_safe_mask(mode_mask, n=n, fallback=corr_m, min_pts=6)

    off_l = _auto_target_preselect_safe_median(ml_arr[lvl_m] - tg_arr[lvl_m], 0.0)
    off_r = _auto_target_preselect_safe_median(mr_arr[lvl_m] - tg_arr[lvl_m], 0.0)
    off = 0.5 * (float(off_l) + float(off_r))

    err_l = ml_arr - (tg_arr + float(off_l))
    err_r = mr_arr - (tg_arr + float(off_r))
    err_avg = 0.5 * (err_l + err_r)

    fit_l = _auto_target_preselect_safe_rms(err_l[corr_m], float("nan"))
    fit_r = _auto_target_preselect_safe_rms(err_r[corr_m], float("nan"))
    if not np.isfinite(fit_l):
        fit_l = _auto_target_preselect_safe_rms(err_l, 0.0)
    if not np.isfinite(fit_r):
        fit_r = _auto_target_preselect_safe_rms(err_r, 0.0)
    fit_rms_db = 0.5 * (float(fit_l) + float(fit_r))
    asym_penalty_db = abs(float(fit_l) - float(fit_r))

    needed_l = (tg_arr + float(off_l)) - ml_arr
    needed_r = (tg_arr + float(off_r)) - mr_arr
    total_boost = 0.5 * (
        _auto_target_preselect_safe_pos_mean(needed_l[corr_m], 0.0)
        + _auto_target_preselect_safe_pos_mean(needed_r[corr_m], 0.0)
    )
    mode_boost = 0.5 * (
        _auto_target_preselect_safe_pos_mean(needed_l[mode_m], 0.0)
        + _auto_target_preselect_safe_pos_mean(needed_r[mode_m], 0.0)
    )
    boost_raw = 0.65 * float(total_boost) + 0.35 * float(mode_boost)
    boost_ref = float(max(0.1, _auto_safe_float(AUTO_MODE_TARGET_PRESELECT_MAX_BASS_BOOST_REF_DB, 8.0)))
    boost_penalty = float(boost_ref * np.tanh(float(boost_raw) / float(boost_ref)))

    slope_meas = _auto_target_slope_estimate(ff, 0.5 * (ml_arr + mr_arr), mask=corr_m)
    slope_target = _auto_target_slope_estimate(ff, tg_arr, mask=corr_m)
    slope_penalty = 0.0
    if np.isfinite(slope_meas) and np.isfinite(slope_target):
        slope_penalty = abs(float(slope_meas) - float(slope_target))

    mode_fit_rms_db = _auto_target_preselect_safe_rms(err_avg[mode_m], float("nan"))
    if not np.isfinite(mode_fit_rms_db):
        mode_fit_rms_db = _auto_target_preselect_safe_rms(err_avg[corr_m], 0.0)

    bass_shape_penalty = _auto_target_preselect_weighted_band_median_rms(
        ff,
        err_avg,
        (
            (20.0, 35.0, 1.35),
            (35.0, 55.0, 1.30),
            (55.0, 85.0, 1.15),
            (85.0, 130.0, 0.95),
            (130.0, 220.0, 0.75),
        ),
        min_pts=5,
        default=0.0,
    )
    treble_shape_penalty = _auto_target_preselect_weighted_band_median_rms(
        ff,
        err_avg,
        (
            (2200.0, 4000.0, 0.80),
            (4000.0, 8000.0, 1.00),
            (8000.0, 12000.0, 1.15),
            (12000.0, 18000.0, 1.25),
        ),
        min_pts=4,
        default=0.0,
    )

    preselect_score = (
        float(fit_rms_db)
        + float(_auto_safe_float(AUTO_MODE_TARGET_PRESELECT_BOOST_W, 0.22)) * float(boost_penalty)
        + float(_auto_safe_float(AUTO_MODE_TARGET_PRESELECT_SLOPE_W, 0.18)) * float(slope_penalty)
        + float(_auto_safe_float(AUTO_MODE_TARGET_PRESELECT_ASYM_W, 0.30)) * float(asym_penalty_db)
        + float(_auto_safe_float(AUTO_MODE_TARGET_PRESELECT_MODE_W, 0.16)) * float(mode_fit_rms_db)
        + float(_auto_safe_float(AUTO_MODE_TARGET_PRESELECT_BASS_SHAPE_W, 0.34))
        * float(bass_shape_penalty)
        + float(_auto_safe_float(AUTO_MODE_TARGET_PRESELECT_TREBLE_SHAPE_W, 0.28))
        * float(treble_shape_penalty)
    )
    if not np.isfinite(preselect_score):
        preselect_score = 1e9

    return {
        "fit_rms_db": float(fit_rms_db),
        "fit_rms_l_db": float(fit_l),
        "fit_rms_r_db": float(fit_r),
        "offset_db": float(off),
        "offset_l_db": float(off_l),
        "offset_r_db": float(off_r),
        "asym_penalty_db": float(asym_penalty_db),
        "boost_penalty": float(boost_penalty),
        "slope_penalty": float(slope_penalty),
        "mode_fit_rms_db": float(mode_fit_rms_db),
        "bass_shape_penalty": float(bass_shape_penalty),
        "treble_shape_penalty": float(treble_shape_penalty),
        "preselect_score": float(preselect_score),
    }

def _auto_target_adaptive_shortlist(quick_candidates: list[dict], *, top_n: int) -> tuple[list[dict], dict]:
    cands = [dict(tc or {}) for tc in list(quick_candidates or []) if isinstance(tc, dict)]
    if not cands:
        return [], {}

    def _score(tc: dict) -> float:
        return float(
            _auto_safe_float(
                tc.get("preselect_score", tc.get("fit_rms_db", float("inf"))),
                float("inf"),
            )
        )

    cands = sorted(
        cands,
        key=lambda tc: (
            _score(tc),
            _auto_safe_float(tc.get("fit_rms_db", float("inf")), float("inf")),
            str(tc.get("hc_mode", "") or "").strip(),
        ),
    )
    top_n_eff = int(max(1, _auto_safe_int(top_n, AUTO_MODE_TARGET_TOP_N)))
    n_min = int(max(1, _auto_safe_int(AUTO_MODE_TARGET_TOP_N_MIN, 3)))
    n_max = int(max(n_min, _auto_safe_int(AUTO_MODE_TARGET_TOP_N_MAX, 6)))
    spread_db = float(max(0.0, _auto_safe_float(AUTO_MODE_TARGET_TOP_N_SPREAD_DB, 0.35)))
    best_score = _score(cands[0])
    spread_based_n = int(
        sum(1 for tc in cands if _score(tc) <= (float(best_score) + float(spread_db)))
    )
    shortlist_n = int(max(top_n_eff, spread_based_n))
    shortlist_n = int(min(shortlist_n, n_max))
    shortlist_n = int(max(shortlist_n, n_min))
    shortlist_n = int(min(shortlist_n, len(cands)))
    return list(cands[:shortlist_n]), {
        "best_score": float(best_score),
        "spread_db": float(spread_db),
        "spread_based_n": int(spread_based_n),
        "top_n_eff": int(top_n_eff),
        "shortlist_n": int(shortlist_n),
        "candidate_total": int(len(cands)),
    }

def _auto_target_insert_cached_wildcard(
    shortlisted: list[dict],
    quick_candidates: list[dict],
    *,
    cached_hc_mode: str | None,
) -> tuple[list[dict], dict]:
    out = [dict(tc or {}) for tc in list(shortlisted or []) if isinstance(tc, dict)]
    cached_name = _auto_builtin_target_name(cached_hc_mode)
    if not cached_name:
        return out, {"inserted": False, "reason": "no_valid_cache_target"}

    for tc in out:
        hc = str(tc.get("hc_mode", "") or "").strip()
        if hc == str(cached_name):
            tc["from_cache_wildcard"] = True
            return out, {
                "inserted": False,
                "already_present": True,
                "hc_mode": str(cached_name),
                "reason": "already_shortlisted",
            }

    matched = None
    for tc in list(quick_candidates or []):
        if not isinstance(tc, dict):
            continue
        hc = str(tc.get("hc_mode", "") or "").strip()
        if hc == str(cached_name):
            matched = dict(tc)
            break
    if not isinstance(matched, dict):
        return out, {
            "inserted": False,
            "already_present": False,
            "hc_mode": str(cached_name),
            "reason": "not_in_quick_candidates",
        }

    matched["from_cache_wildcard"] = True
    out.append(dict(matched))
    return out, {
        "inserted": True,
        "already_present": False,
        "hc_mode": str(cached_name),
        "reason": "inserted",
    }

_SYNTH_TILT_VARIANTS = (0.15, 0.30, 0.45)


def _auto_build_synth_target_candidate(
    data: dict,
    *,
    f_l,
    m_l,
    f_r,
    m_r,
    measurements: dict | None = None,
) -> dict | None:
    """Build and score a synthesized adaptive target candidate from measurements.

    Tries three tilt values and returns the best scoring variant.
    """
    try:
        fl = np.asarray(f_l, dtype=float).reshape(-1)
        ml = np.asarray(m_l, dtype=float).reshape(-1)
        fr = np.asarray(f_r, dtype=float).reshape(-1)
        mr = np.asarray(m_r, dtype=float).reshape(-1)
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError, NameError):
        return None
    if fl.size < 32 and fr.size >= 32:
        fl, ml = np.asarray(fr, dtype=float).copy(), np.asarray(mr, dtype=float).copy()
    if fr.size < 32 and fl.size >= 32:
        fr, mr = np.asarray(fl, dtype=float).copy(), np.asarray(ml, dtype=float).copy()

    best_tc = None
    best_score = float("inf")
    for tilt_val in _SYNTH_TILT_VARIANTS:
        try:
            _ms = dict(measurements or {})
            _ms.update({"f_l": f_l, "m_l": m_l, "f_r": f_r, "m_r": m_r})
            synth = get_or_build_synth_target(
                _ms,
                tilt_comp_frac=float(tilt_val),
                bass_comp_frac=float(AUTO_MODE_SYNTH_TARGET_BASS_COMP_FRAC),
                bass_comp_ref_db=float(AUTO_MODE_SYNTH_TARGET_BASS_COMP_REF_DB),
                hf_comp_frac=float(AUTO_MODE_SYNTH_TARGET_HF_COMP_FRAC),
                smooth_oct=float(AUTO_MODE_SYNTH_TARGET_SMOOTH_OCT),
                synth_fn=synthesize_target_from_measurements,
            )
            if synth is None:
                continue
            hf, hm = synth
        except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError, NameError):
            continue
        try:
            tc = _auto_target_preselect_score_curve(
                data,
                fl=fl,
                ml=ml,
                fr=fr,
                mr=mr,
                hf=hf,
                hm=hm,
            )
            if not isinstance(tc, dict) or not tc:
                continue
            score = float(tc.get("preselect_score", float("inf")))
            if score < best_score:
                best_score = score
                best_tc = dict(tc)
                best_tc["_synth_tilt_frac_used"] = float(tilt_val)
                best_tc["_synth_hc_f"] = np.asarray(hf, dtype=float).reshape(-1)
                best_tc["_synth_hc_m"] = np.asarray(hm, dtype=float).reshape(-1)
        except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError, NameError):
            continue

    if best_tc is None:
        return None
    best_tc["hc_mode"] = str(AUTO_MODE_SYNTH_TARGET_NAME)
    return dict(best_tc)


def _auto_select_builtin_target_curve_scores(
    data: dict,
    *,
    fl,
    ml,
    fr,
    mr,
    measurements: dict | None = None,
) -> list[dict]:
    scored = []
    for hc_name in AUTO_MODE_BUILTIN_TARGETS:
        try:
            hf, hm = get_house_curve_by_name(hc_name)
            tc = _auto_target_preselect_score_curve(
                data,
                fl=fl,
                ml=ml,
                fr=fr,
                mr=mr,
                hf=hf,
                hm=hm,
            )
            if not isinstance(tc, dict) or not tc:
                continue
            tc["hc_mode"] = str(hc_name)
            scored.append(dict(tc))
        except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError, NameError):
            continue

    if bool(AUTO_MODE_SYNTH_TARGET_ENABLED):
        try:
            synth_tc = _auto_build_synth_target_candidate(
                data,
                f_l=fl,
                m_l=ml,
                f_r=fr,
                m_r=mr,
                measurements=measurements,
            )
            if isinstance(synth_tc, dict) and synth_tc:
                scored.append(dict(synth_tc))
        except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError, NameError):
            logger.exception("synth target candidate build")
    return scored

def _auto_select_builtin_target_curve(
    data: dict,
    *,
    f_l,
    m_l,
    f_r,
    m_r,
    measurements: dict | None = None,
) -> dict | None:
    try:
        fl = np.asarray(f_l, dtype=float).reshape(-1)
        ml = np.asarray(m_l, dtype=float).reshape(-1)
        fr = np.asarray(f_r, dtype=float).reshape(-1)
        mr = np.asarray(m_r, dtype=float).reshape(-1)
    except (RuntimeError, OSError, ImportError, TypeError, ValueError, AttributeError, KeyError, IndexError, OverflowError, FloatingPointError, NameError):
        return None
    if fl.size < 32 and fr.size >= 32:
        fl, ml = np.asarray(fr, dtype=float).copy(), np.asarray(mr, dtype=float).copy()
    if fr.size < 32 and fl.size >= 32:
        fr, mr = np.asarray(fl, dtype=float).copy(), np.asarray(ml, dtype=float).copy()
    scored = _auto_select_builtin_target_curve_scores(
        data,
        fl=fl,
        ml=ml,
        fr=fr,
        mr=mr,
        measurements=measurements,
    )
    if not scored:
        return None
    scored = sorted(
        scored,
        key=lambda d: (
            _auto_safe_float(
                d.get("preselect_score", d.get("fit_rms_db", float("inf"))),
                float("inf"),
            ),
            _auto_safe_float(d.get("fit_rms_db", float("inf")), float("inf")),
            str(d.get("hc_mode", "") or "").strip(),
        ),
    )
    best = scored[0]
    return {
        "selected_hc_mode": str(best.get("hc_mode", "Harman6")),
        "fit_rms_db": float(best.get("fit_rms_db", 0.0)),
        "offset_db": float(best.get("offset_db", 0.0)),
        "candidates": list(scored[:5]),
        "candidates_all": list(scored),
    }


__all__ = [
    '_auto_target_one_step_milder',
    '_auto_target_slope_estimate',
    '_auto_target_preselect_score',
    '_auto_target_adaptive_shortlist',
    '_auto_target_insert_cached_wildcard',
    '_auto_build_synth_target_candidate',
    '_auto_select_builtin_target_curve',
]

