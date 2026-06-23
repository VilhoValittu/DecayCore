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

import logging

import numpy as np

_logger = logging.getLogger(__name__)

from .. import shared

from .metrics_common import _auto_stats_pick_arr

_RECOVERABLE_TARGET_PENALTY_EXCEPTIONS = (
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
)


def _target_penalty_pick_arr(
    st: dict,
    *,
    mode: str,
    base_key: str,
    fallback_keys: tuple[str, ...] = (),
    max_n: int | None = None,
) -> np.ndarray:
    keys: list[str] = []
    if str(mode) == "comparison":
        keys.append(f"cmp_{base_key!s}")
        keys.extend([f"cmp_{k!s}" for k in fallback_keys])
    keys.append(str(base_key))
    keys.extend([str(k) for k in fallback_keys])
    for key in keys:
        try:
            raw = st.get(key, [])
            if max_n is not None:
                raw = raw[:max_n]
            arr = np.asarray(raw, dtype=float).reshape(-1)
        except _RECOVERABLE_TARGET_PENALTY_EXCEPTIONS:
            arr = np.asarray([], dtype=float)
        if arr.size:
            return arr
    return np.asarray([], dtype=float)


def _bass_boost_guard_hi(st: dict) -> float:
    guard_hi = 20.0
    low_cut = shared._auto_safe_float(st.get("low_bass_cut_hz", float("nan")), float("nan"))
    if np.isfinite(low_cut) and float(low_cut) > 0.0:
        guard_hi = max(float(guard_hi), float(low_cut))
    exc_freq = shared._auto_safe_float(st.get("exc_freq", float("nan")), float("nan"))
    if bool(st.get("exc_prot", False)) and np.isfinite(exc_freq) and float(exc_freq) > 0.0:
        guard_hi = max(float(guard_hi), float(exc_freq) * 1.41)
    return float(guard_hi)


def _bass_boost_metrics_from_arrays(f: np.ndarray, filt: np.ndarray, *, guard_hi: float) -> tuple[float, float]:
    mask = np.isfinite(f) & np.isfinite(filt) & (f >= float(guard_hi)) & (f <= 200.0)
    if int(np.count_nonzero(mask)) < 4:
        return 0.0, 0.0
    positive = np.maximum(np.asarray(filt[mask], dtype=float), 0.0)
    finite = positive[np.isfinite(positive)]
    if finite.size == 0:
        return 0.0, 0.0
    boosted = finite[finite > 0.05]
    if boosted.size == 0:
        return 0.0, 0.0
    return float(np.percentile(boosted, 70.0)), float(np.max(boosted))


def _target_penalty_mode(st: dict) -> str:
    return str(st.get("analysis_mode", "native") or "native").strip().lower()


def _target_penalty_pick_arr_for_mode(
    st: dict,
    *,
    mode: str,
    base_key: str,
    fallback_keys: tuple[str, ...] = (),
    max_n: int | None = None,
) -> np.ndarray:
    return _target_penalty_pick_arr(
        st,
        mode=mode,
        base_key=base_key,
        fallback_keys=fallback_keys,
        max_n=max_n,
    )


def _target_penalty_band_limit_from_freq(f_full: np.ndarray, *, hi_hz: float, scale: float) -> int | None:
    if f_full.size < 8:
        return None
    n_band = int(np.searchsorted(f_full, float(hi_hz) * float(scale), side="right"))
    if n_band < 8:
        return None
    return int(n_band)


def _focus_ripple_primary_rms(
    *,
    f: np.ndarray,
    measured: np.ndarray,
    target: np.ndarray,
    realized: np.ndarray,
    confidence: np.ndarray,
    lo_hz: float,
    hi_hz: float,
) -> float | None:
    n = int(min(f.size, measured.size, target.size))
    if n < 8:
        return None
    f_use = np.asarray(f[:n], dtype=float)
    pred = np.asarray(measured[:n], dtype=float)
    if realized.size >= n:
        pred = pred + np.asarray(realized[:n], dtype=float)
    err = pred - np.asarray(target[:n], dtype=float)
    mask = np.isfinite(f_use) & np.isfinite(err) & (f_use >= float(lo_hz)) & (f_use <= float(hi_hz))
    if int(np.count_nonzero(mask)) < 8:
        return None
    err_use = np.asarray(err[mask], dtype=float)
    if confidence.size >= n:
        w = np.clip(np.asarray(confidence[:n], dtype=float)[mask], 0.0, 1.0)
        w = np.maximum(w, 0.05)
        w_sum = float(np.sum(w))
        if np.isfinite(w_sum) and w_sum > 1e-12:
            return float(np.sqrt(np.sum(w * err_use * err_use) / w_sum))
    return float(np.sqrt(np.mean(err_use * err_use)))


def _focus_ripple_fallback_rms(
    *,
    f: np.ndarray,
    pred_filter: np.ndarray,
    realized_filter: np.ndarray,
    lo_hz: float,
    hi_hz: float,
) -> float | None:
    n = int(min(f.size, pred_filter.size, realized_filter.size))
    if n < 8:
        return None
    f_use = np.asarray(f[:n], dtype=float)
    diff = np.asarray(realized_filter[:n], dtype=float) - np.asarray(pred_filter[:n], dtype=float)
    mask = np.isfinite(f_use) & np.isfinite(diff) & (f_use >= float(lo_hz)) & (f_use <= float(hi_hz))
    if int(np.count_nonzero(mask)) < 8:
        return None
    diff_use = np.asarray(diff[mask], dtype=float)
    offset = float(np.median(diff_use))
    diff_shape = np.asarray(diff_use, dtype=float) - float(offset)
    return float(np.sqrt(np.mean(diff_shape * diff_shape)))


def _auto_focus_ripple_from_stats(
    st: dict | None,
    *,
    focus_lo_hz: float,
    focus_hi_hz: float,
) -> float | None:
    st = dict(st or {})
    lo = shared._auto_safe_float(focus_lo_hz, float("nan"))
    hi = shared._auto_safe_float(focus_hi_hz, float("nan"))
    if not (np.isfinite(lo) and np.isfinite(hi)) or float(hi) <= float(lo):
        return None

    mode = _target_penalty_mode(st)

    # Get freq first to compute band limit before creating other large arrays.
    # focus_hi_hz is the highest frequency of interest; guard at 2.5× to be safe.
    f_full = _target_penalty_pick_arr_for_mode(st, mode=mode, base_key="freq_axis")
    n_lim = _target_penalty_band_limit_from_freq(f_full, hi_hz=float(hi), scale=2.5)

    f = f_full[:n_lim] if n_lim else f_full
    measured = _target_penalty_pick_arr_for_mode(st, mode=mode, base_key="measured_mags", max_n=n_lim)
    target = _target_penalty_pick_arr_for_mode(st, mode=mode, base_key="target_mags", max_n=n_lim)
    realized = _target_penalty_pick_arr_for_mode(
        st,
        mode=mode,
        base_key="realized_filter_mags",
        fallback_keys=("filter_mags",),
        max_n=n_lim,
    )
    confidence = _target_penalty_pick_arr_for_mode(st, mode=mode, base_key="confidence_mask", max_n=n_lim)

    primary = _focus_ripple_primary_rms(
        f=f,
        measured=measured,
        target=target,
        realized=realized,
        confidence=confidence,
        lo_hz=float(lo),
        hi_hz=float(hi),
    )
    if primary is not None:
        return float(primary)

    # Fallback: if corrected-response data is incomplete, fall back to filter-realization delta.
    predicted_filter = _target_penalty_pick_arr_for_mode(
        st,
        mode=mode,
        base_key="predicted_filter_mags",
        max_n=n_lim,
    )
    fallback = _focus_ripple_fallback_rms(
        f=f,
        pred_filter=predicted_filter,
        realized_filter=realized,
        lo_hz=float(lo),
        hi_hz=float(hi),
    )
    return float(fallback) if fallback is not None else None


def _target_tracking_output_defaults() -> dict:
    return {
        "target_tracking_rms_20_200_db": float("nan"),
        "target_tracking_max_20_200_db": float("nan"),
        "target_tracking_rms_100_500_db": float("nan"),
        "target_tracking_max_100_500_db": float("nan"),
    }


def _target_tracking_prepare_arrays(st: dict, *, mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    f_full = _target_penalty_pick_arr_for_mode(st, mode=mode, base_key="freq_axis")
    n_lim = _target_penalty_band_limit_from_freq(f_full, hi_hz=8000.0, scale=1.0)
    f = f_full[:n_lim] if n_lim else f_full
    measured = _target_penalty_pick_arr_for_mode(st, mode=mode, base_key="measured_mags", max_n=n_lim)
    target = _target_penalty_pick_arr_for_mode(st, mode=mode, base_key="target_mags", max_n=n_lim)
    filt = _target_penalty_pick_arr_for_mode(
        st,
        mode=mode,
        base_key="filter_mags",
        fallback_keys=("realized_filter_mags",),
        max_n=n_lim,
    )
    conf = _target_penalty_pick_arr_for_mode(st, mode=mode, base_key="confidence_mask", max_n=n_lim)
    return f, measured, target, filt, conf


def _target_tracking_confidence(conf: np.ndarray, *, n: int) -> np.ndarray | None:
    if conf.size < n:
        return None
    conf_use = np.asarray(conf[:n], dtype=float)
    finite_conf = conf_use[np.isfinite(conf_use)]
    if finite_conf.size and float(np.nanmax(finite_conf)) > 1.5:
        conf_use = conf_use / 100.0
    return np.clip(conf_use, 0.0, 1.0)


def _target_tracking_apply_band_metrics(
    out: dict,
    *,
    f: np.ndarray,
    err: np.ndarray,
    conf_use: np.ndarray | None,
) -> None:
    for lo, hi, suffix in (
        (20.0, 200.0, "20_200"),
        (100.0, 500.0, "100_500"),
        (500.0, 8000.0, "500_8000"),
    ):
        mask = np.isfinite(f) & np.isfinite(err) & (f >= float(lo)) & (f <= float(hi))
        if int(np.count_nonzero(mask)) < 4:
            continue
        err_band = np.asarray(err[mask], dtype=float)
        if conf_use is None:
            rms = float(np.sqrt(np.mean(err_band * err_band)))
        else:
            w = np.maximum(np.asarray(conf_use[mask], dtype=float), 0.05)
            w_sum = float(np.sum(w))
            if np.isfinite(w_sum) and w_sum > 1e-12:
                rms = float(np.sqrt(np.sum(w * err_band * err_band) / w_sum))
            else:
                rms = float(np.sqrt(np.mean(err_band * err_band)))
        max_err = float(np.max(np.abs(err_band)))
        out[f"target_tracking_rms_{suffix}_db"] = float(rms)
        out[f"target_tracking_max_{suffix}_db"] = float(max_err)


def _auto_target_tracking_metrics_from_stats(st: dict | None) -> dict:
    st = dict(st or {})
    out = _target_tracking_output_defaults()
    mode = _target_penalty_mode(st)
    f, measured, target, filt, conf = _target_tracking_prepare_arrays(st, mode=mode)
    n = int(min(f.size, measured.size, target.size, filt.size))
    if n < 8:
        return dict(out)

    f = np.asarray(f[:n], dtype=float)
    err = (
        np.asarray(measured[:n], dtype=float)
        + np.asarray(filt[:n], dtype=float)
        - np.asarray(target[:n], dtype=float)
    )
    conf_use = _target_tracking_confidence(conf, n=n)
    _target_tracking_apply_band_metrics(out, f=f, err=err, conf_use=conf_use)
    return dict(out)


def _auto_merge_target_tracking_metrics(l_metrics: dict | None, r_metrics: dict | None) -> dict:
    out: dict[str, float] = {}
    for key in (
        "target_tracking_rms_20_200_db",
        "target_tracking_max_20_200_db",
        "target_tracking_rms_100_500_db",
        "target_tracking_max_100_500_db",
    ):
        vals = []
        for metrics in (l_metrics, r_metrics):
            v = shared._auto_safe_float((metrics or {}).get(key, float("nan")), float("nan"))
            if np.isfinite(v):
                vals.append(float(v))
        out[key] = float(max(vals)) if vals else float("nan")
    return out


def _auto_target_tracking_penalty(metrics: dict | None) -> float:
    m = dict(metrics or {})
    rms_20_200 = shared._auto_safe_float(m.get("target_tracking_rms_20_200_db", float("nan")), float("nan"))
    max_20_200 = shared._auto_safe_float(m.get("target_tracking_max_20_200_db", float("nan")), float("nan"))
    rms_100_500 = shared._auto_safe_float(m.get("target_tracking_rms_100_500_db", float("nan")), float("nan"))
    max_100_500 = shared._auto_safe_float(m.get("target_tracking_max_100_500_db", float("nan")), float("nan"))
    rms_500_8000 = shared._auto_safe_float(m.get("target_tracking_rms_500_8000_db", float("nan")), float("nan"))
    max_500_8000 = shared._auto_safe_float(m.get("target_tracking_max_500_8000_db", float("nan")), float("nan"))

    penalty = 0.0
    if np.isfinite(rms_20_200):
        penalty += 3.00 * max(0.0, float(rms_20_200) - 1.20)
    if np.isfinite(max_20_200):
        penalty += 1.00 * max(0.0, float(max_20_200) - 3.50)
    if np.isfinite(rms_100_500):
        penalty += 3.00 * max(0.0, float(rms_100_500) - 1.00)
    if np.isfinite(max_100_500):
        penalty += 0.90 * max(0.0, float(max_100_500) - 3.50)
    if np.isfinite(rms_500_8000):
        penalty += 1.00 * max(0.0, float(rms_500_8000) - 2.00)
    if np.isfinite(max_500_8000):
        penalty += 0.40 * max(0.0, float(max_500_8000) - 4.50)
    return float(np.clip(float(penalty), 0.0, 12.0))


def _auto_bass_under_target_metrics_from_stats(st: dict | None) -> dict:
    st = dict(st or {})
    out = {
        "bass_under_target_rms_20_200_db": float("nan"),
        "bass_under_target_max_20_200_db": float("nan"),
        "bass_under_target_penalty": 0.0,
    }

    # Get freq_axis first to determine the band limit, then restrict all other arrays.
    f_full = _auto_stats_pick_arr(st, "freq_axis")
    _n_band = int(np.searchsorted(f_full, 500.0, side="right")) if f_full.size >= 8 else f_full.size
    _n_lim: int | None = _n_band if _n_band >= 8 else None

    f = f_full[:_n_lim] if _n_lim else f_full
    measured = _auto_stats_pick_arr(st, "measured_mags", _max_n=_n_lim)
    target = _auto_stats_pick_arr(st, "target_mags", _max_n=_n_lim)
    filt = _auto_stats_pick_arr(st, "realized_filter_mags", "filter_mags", _max_n=_n_lim)
    conf = _auto_stats_pick_arr(st, "confidence_mask", _max_n=_n_lim)
    n = int(min(f.size, measured.size, target.size, filt.size))
    if n < 8:
        return dict(out)

    f = np.asarray(f[:n], dtype=float)
    err = (
        np.asarray(measured[:n], dtype=float)
        + np.asarray(filt[:n], dtype=float)
        - np.asarray(target[:n], dtype=float)
    )
    mask = np.isfinite(f) & np.isfinite(err) & (f >= 20.0) & (f <= 200.0)
    if int(np.count_nonzero(mask)) < 4:
        return dict(out)

    under = np.maximum(0.0, -np.asarray(err[mask], dtype=float) - 0.20)
    if conf.size >= n:
        conf_use = np.asarray(conf[:n], dtype=float)
        finite_conf = conf_use[np.isfinite(conf_use)]
        if finite_conf.size and float(np.nanmax(finite_conf)) > 1.5:
            conf_use = conf_use / 100.0
        w = np.maximum(np.clip(np.asarray(conf_use[mask], dtype=float), 0.0, 1.0), 0.35)
        w_sum = float(np.sum(w))
        rms = float(np.sqrt(np.sum(w * under * under) / w_sum)) if w_sum > 1e-12 else float(np.sqrt(np.mean(under * under)))
    else:
        rms = float(np.sqrt(np.mean(under * under)))
    max_under = float(np.max(under)) if under.size else 0.0
    penalty = float((2.2 * max(0.0, rms - 0.10)) + (0.65 * max(0.0, max_under - 0.45)))
    out["bass_under_target_rms_20_200_db"] = float(rms)
    out["bass_under_target_max_20_200_db"] = float(max_under)
    out["bass_under_target_penalty"] = float(np.clip(penalty, 0.0, 5.0))
    return dict(out)


def _auto_merge_bass_under_target_metrics(l_metrics: dict | None, r_metrics: dict | None) -> dict:
    out: dict[str, float] = {}
    for key in (
        "bass_under_target_rms_20_200_db",
        "bass_under_target_max_20_200_db",
        "bass_under_target_penalty",
    ):
        vals = []
        for metrics in (l_metrics, r_metrics):
            v = shared._auto_safe_float((metrics or {}).get(key, float("nan")), float("nan"))
            if np.isfinite(v):
                vals.append(float(v))
        out[key] = float(max(vals)) if vals else float("nan")
    return out


def _auto_bass_boost_metrics_from_stats(st: dict | None) -> dict:
    st = dict(st or {})
    mode = str(st.get("analysis_mode", "native") or "native").strip().lower()
    f_full = _target_penalty_pick_arr(st, mode=mode, base_key="freq_axis")
    _n_band = int(np.searchsorted(f_full, 500.0, side="right")) if f_full.size >= 4 else f_full.size
    _n_lim: int | None = _n_band if _n_band >= 4 else None

    f = f_full[:_n_lim] if _n_lim else f_full
    filt = _target_penalty_pick_arr(
        st,
        mode=mode,
        base_key="filter_mags",
        fallback_keys=("realized_filter_mags",),
        max_n=_n_lim,
    )
    n = int(min(f.size, filt.size))
    out = {
        "bass_boost_20_200_db": 0.0,
        "bass_boost_peak_20_200_db": 0.0,
    }
    if n < 4:
        return dict(out)

    f = np.asarray(f[:n], dtype=float)
    filt = np.asarray(filt[:n], dtype=float)
    guard_hi = _bass_boost_guard_hi(st)
    boost_avg, boost_peak = _bass_boost_metrics_from_arrays(f, filt, guard_hi=float(guard_hi))
    out["bass_boost_20_200_db"] = float(boost_avg)
    out["bass_boost_peak_20_200_db"] = float(boost_peak)
    return dict(out)


def _auto_merge_bass_boost_metrics(l_metrics: dict | None, r_metrics: dict | None) -> dict:
    vals = []
    peaks = []
    for metrics in (l_metrics, r_metrics):
        v = shared._auto_safe_float((metrics or {}).get("bass_boost_20_200_db", float("nan")), float("nan"))
        p = shared._auto_safe_float((metrics or {}).get("bass_boost_peak_20_200_db", float("nan")), float("nan"))
        if np.isfinite(v):
            vals.append(float(max(0.0, v)))
        if np.isfinite(p):
            peaks.append(float(max(0.0, p)))
    shared_bass = float(min(vals)) if len(vals) >= 2 else (float(vals[0]) if vals else 0.0)
    return {
        "bass_boost_20_200_db": float(shared_bass),
        "bass_boost_20_200_mean_db": float(np.mean(vals)) if vals else 0.0,
        "bass_boost_20_200_max_db": float(max(vals)) if vals else 0.0,
        "bass_boost_peak_20_200_db": float(max(peaks)) if peaks else 0.0,
    }


def _auto_bass_preference_bonus(
    *,
    bass_boost_db: float,
    target_tracking_penalty: float,
    exc_penalty: float,
    bass_integration_penalty: float,
    bass_feasibility_penalty: float,
) -> float:
    bass = shared._auto_safe_float(bass_boost_db, 0.0)
    if not np.isfinite(bass) or float(bass) <= 0.5:
        return 0.0
    norm = float(np.clip((float(bass) - 0.5) / 4.5, 0.0, 1.0))
    raw = 3.0 * float(norm)
    safety_scale = 1.0 / (
        1.0
        + 0.10 * max(0.0, shared._auto_safe_float(exc_penalty, 0.0))
        + 0.15 * max(0.0, shared._auto_safe_float(bass_integration_penalty, 0.0))
        + 0.25 * max(0.0, shared._auto_safe_float(bass_feasibility_penalty, 0.0))
        + 0.03 * max(0.0, shared._auto_safe_float(target_tracking_penalty, 0.0))
    )
    return float(np.clip(float(raw) * float(safety_scale), 0.0, 3.0))



__all__ = ["_auto_focus_ripple_from_stats", "_auto_target_tracking_metrics_from_stats", "_auto_merge_target_tracking_metrics", "_auto_target_tracking_penalty", "_auto_bass_under_target_metrics_from_stats", "_auto_merge_bass_under_target_metrics", "_auto_bass_boost_metrics_from_stats", "_auto_merge_bass_boost_metrics", "_auto_bass_preference_bonus"]
