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

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy

import numpy as np

from .decaycore_analysis import analyze_acoustic_confidence
from .decaycore_leveling import compute_leveling
from .phase import remove_time_of_flight
from .smoothing import apply_adaptive_fdw, apply_smoothing_std, psychoacoustic_smoothing
from .dsp_types import DspContext, PreprocessResult

_PREPROCESS_CACHE: dict = {}


def _cache_hash_array(values, *, decimals: int = 6) -> str:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return "empty"
    rounded = np.round(arr, int(decimals))
    rounded = np.nan_to_num(rounded, nan=1.0e300, posinf=1.0e301, neginf=-1.0e301)
    return hashlib.blake2b(np.ascontiguousarray(rounded, dtype=np.float64).view(np.uint8), digest_size=16).hexdigest()


def _stable_jsonable(value):
    if dataclasses.is_dataclass(value):
        return _stable_jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _stable_jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_stable_jsonable(v) for v in value]
    if isinstance(value, (bool, str)) or value is None:
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        v = float(value)
        return round(v, 8) if np.isfinite(v) else str(v)
    return str(value)


def _stable_object_hash(value) -> str:
    if value is None:
        return "none"
    try:
        public_data = value
        if not dataclasses.is_dataclass(value) and not isinstance(value, Mapping) and hasattr(value, "__dict__"):
            public_data = {
                k: v
                for k, v in vars(value).items()
                if not str(k).startswith("_")
            }
        payload = json.dumps(_stable_jsonable(public_data), sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError, AttributeError):
        payload = type(value).__qualname__
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


def _preprocess_cache_key(freqs, meas_mags, raw_phases, cfg, stereo_link_ctx, presolve_mode) -> tuple | None:
    n = len(freqs)
    if n < 2:
        return None
    try:
        afdw_on = bool(getattr(cfg, "enable_afdw", False))
        fdw_cyc = round(float(getattr(cfg, "fdw_cycles", 15.0)), 2) if afdw_on else 0.0
        return (
            n,
            round(float(freqs[0]), 4),
            round(float(freqs[-1]), 4),
            _cache_hash_array(freqs),
            _cache_hash_array(meas_mags),
            _cache_hash_array(raw_phases),
            int(cfg.num_taps), int(cfg.fs),
            str(getattr(cfg, "plot_smoothing_level", "")),
            bool(getattr(cfg, "comparison_mode", False)),
            int(getattr(cfg, "comparison_ref_fs", 44100) or 44100),
            int(getattr(cfg, "comparison_ref_taps", 65536) or 65536),
            afdw_on, fdw_cyc,
            _stable_object_hash(stereo_link_ctx),
            bool(presolve_mode),
        )
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None


def clear_preprocess_cache() -> None:
    _PREPROCESS_CACHE.clear()


def _copy_array(values):
    return np.array(values, copy=True)


def clone_preprocess_result(result: PreprocessResult) -> PreprocessResult:
    ctx = DspContext(
        n_fft=int(result.ctx.n_fft),
        freq_axis=_copy_array(result.ctx.freq_axis),
        gain_db=_copy_array(result.ctx.gain_db),
        target_mags=_copy_array(result.ctx.target_mags),
        st=deepcopy(dict(result.ctx.st or {})),
    )
    return PreprocessResult(
        ctx=ctx,
        f_in=_copy_array(result.f_in),
        m_in=_copy_array(result.m_in),
        p_in=_copy_array(result.p_in),
        m_smooth_std=_copy_array(result.m_smooth_std),
        p_smooth=_copy_array(result.p_smooth),
        m_interp=_copy_array(result.m_interp),
        p_rad_raw=_copy_array(result.p_rad_raw),
        p_rad_interp=_copy_array(result.p_rad_interp),
        delay_slope=float(result.delay_slope),
        m_plot_db=None if result.m_plot_db is None else _copy_array(result.m_plot_db),
        complex_meas=_copy_array(result.complex_meas),
        m_anal=_copy_array(result.m_anal),
        p_anal_rad=_copy_array(result.p_anal_rad),
        complex_anal=_copy_array(result.complex_anal),
        conf_mask=_copy_array(result.conf_mask),
        reflections=deepcopy(list(result.reflections or [])),
        cmp=None if result.cmp is None else deepcopy(dict(result.cmp)),
        analysis_mode=str(result.analysis_mode),
        is_psy=bool(result.is_psy),
    )


def analysis_smoothing_lf_to_hf(
    freqs,
    mags,
    *,
    low_bw=1 / 3.0,
    high_bw=1 / 1.0,
    f_lo=230.0,
    f_hi=500.0,
):
    f = np.asarray(freqs, dtype=float)
    m = np.asarray(mags, dtype=float)
    if f.size < 8 or m.size != f.size:
        return np.copy(m)

    dummy = np.zeros_like(m)
    try:
        m_low, _ = apply_smoothing_std(f, m, dummy, float(low_bw))
        m_high, _ = apply_smoothing_std(f, m, dummy, float(high_bw))
    except (TypeError, ValueError, FloatingPointError, IndexError):
        return np.copy(m)

    ff = np.maximum(f, 1.0)
    lo = float(max(f_lo, 1.0))
    hi = float(max(f_hi, lo * 1.01))
    w = (np.log10(ff) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
    w = np.clip(w, 0.0, 1.0)

    return (1.0 - w) * m_low + w * m_high


def run_preprocess(freqs, meas_mags, raw_phases, cfg, *, stereo_link_ctx=None, presolve_mode: bool = False) -> PreprocessResult:
    _cache_key = _preprocess_cache_key(freqs, meas_mags, raw_phases, cfg, stereo_link_ctx, presolve_mode)
    if _cache_key is not None:
        _hit = _PREPROCESS_CACHE.get(_cache_key)
        if _hit is not None:
            return clone_preprocess_result(_hit)

    min_len = min(len(freqs), len(meas_mags), len(raw_phases))
    f_in = np.asarray(freqs[:min_len], dtype=float)
    m_in = np.asarray(meas_mags[:min_len], dtype=float)
    p_in = np.asarray(raw_phases[:min_len], dtype=float)

    if f_in.size > 1:
        order = np.argsort(f_in, kind="mergesort")
        f_in = f_in[order]
        m_in = m_in[order]
        p_in = p_in[order]
        uniq_mask = np.concatenate(([True], np.diff(f_in) > 0.0))
        f_in = f_in[uniq_mask]
        m_in = m_in[uniq_mask]
        p_in = p_in[uniq_mask]

    # Preserve the requested FIR length; bumping even tap counts to n+1 creates
    # prime-sized FFTs (for example 65536 -> 65537), which is dramatically slower.
    n_fft = int(cfg.num_taps)
    freq_axis = np.fft.rfftfreq(n_fft, d=1.0 / float(cfg.fs))

    gain_db = np.zeros_like(freq_axis, dtype=float)
    st: dict = {}
    target_mags = np.zeros_like(freq_axis, dtype=float)

    is_psy = (not bool(presolve_mode)) and ("psy" in str(cfg.plot_smoothing_level).lower())

    m_smooth_std = analysis_smoothing_lf_to_hf(
        f_in, m_in, low_bw=1 / 3.0, high_bw=1 / 1.0, f_lo=230.0, f_hi=400.0
    )

    p_smooth_oct = 1 / 12.0 if cfg.fs > 96000 else 1 / 24.0
    p_smooth, _ = apply_smoothing_std(f_in, p_in, np.zeros_like(p_in), p_smooth_oct)

    m_interp = np.interp(freq_axis, f_in, m_in)
    p_rad_raw = np.deg2rad(np.interp(freq_axis, f_in, p_in))
    p_rad_interp, delay_slope = remove_time_of_flight(freq_axis, p_rad_raw)

    m_plot_db = None
    if is_psy:
        try:
            m_plot_db = psychoacoustic_smoothing(freq_axis, m_interp)
        except (TypeError, ValueError, FloatingPointError):
            m_plot_db = None

    complex_meas = 10 ** (m_interp / 20.0) * np.exp(1j * p_rad_interp)

    m_anal = np.interp(freq_axis, f_in, m_smooth_std)
    p_anal_rad = np.deg2rad(np.interp(freq_axis, f_in, p_smooth))
    p_anal_rad, _ = remove_time_of_flight(freq_axis, p_anal_rad)
    complex_anal = 10 ** (m_anal / 20.0) * np.exp(1j * p_anal_rad)

    conf_mask, reflections, _ = analyze_acoustic_confidence(freq_axis, complex_anal, cfg.fs)

    cmp = None
    analysis_mode = "native"
    try:
        if (not bool(presolve_mode)) and bool(getattr(cfg, "comparison_mode", False)):
            ref_fs = int(getattr(cfg, "comparison_ref_fs", 44100) or 44100)
            ref_taps = int(getattr(cfg, "comparison_ref_taps", 65536) or 65536)
            ref_nfft = int(ref_taps)
            freq_cmp_full = np.fft.rfftfreq(ref_nfft, d=1.0 / float(ref_fs))

            fmax = float(freq_axis[-1]) if freq_axis.size else 0.0
            if fmax > 0:
                freq_cmp = freq_cmp_full[freq_cmp_full <= fmax]
            else:
                freq_cmp = freq_cmp_full

            m_cmp_raw = np.interp(freq_cmp, freq_axis, m_anal)
            p_cmp_rad = np.interp(freq_cmp, freq_axis, p_anal_rad)
            complex_cmp = 10 ** (m_cmp_raw / 20.0) * np.exp(1j * p_cmp_rad)

            conf_cmp, refl_cmp, _ = analyze_acoustic_confidence(freq_cmp, complex_cmp, ref_fs)

            target_cmp = np.interp(freq_cmp, freq_axis, target_mags)
            (
                target_level_db_cmp,
                calc_offset_db_cmp,
                meas_level_db_window_cmp,
                target_level_db_window_cmp,
                offset_method_cmp,
                s_min_cmp,
                s_max_cmp,
            ) = compute_leveling(cfg, freq_cmp, m_cmp_raw, target_cmp, stereo_link_ctx=stereo_link_ctx)

            filt_cmp = np.interp(freq_cmp, freq_axis, gain_db)

            cmp = {
                "cmp_ref_fs": float(ref_fs),
                "cmp_ref_taps": float(ref_taps),
                "cmp_freq_axis": freq_cmp.tolist(),
                "cmp_target_mags": target_cmp.tolist(),
                "cmp_measured_mags": (m_cmp_raw - calc_offset_db_cmp).tolist(),
                "cmp_filter_mags": filt_cmp.tolist(),
                "cmp_confidence_mask": conf_cmp.tolist(),
                "cmp_reflections": refl_cmp,
                "cmp_smart_scan_range": [float(s_min_cmp), float(s_max_cmp)],
                "cmp_eff_target_db": float(target_level_db_cmp),
                "cmp_offset_db": float(calc_offset_db_cmp),
                "cmp_meas_level_db_window": float(meas_level_db_window_cmp),
                "cmp_target_level_db_window": float(target_level_db_window_cmp),
                "cmp_offset_method": str(offset_method_cmp),
                "cmp_avg_confidence": float(np.mean(conf_cmp) * 100.0),
            }
            if (
                isinstance(cmp.get("cmp_freq_axis", None), list)
                and isinstance(cmp.get("cmp_measured_mags", None), list)
                and isinstance(cmp.get("cmp_target_mags", None), list)
                and isinstance(cmp.get("cmp_filter_mags", None), list)
                and isinstance(cmp.get("cmp_confidence_mask", None), list)
                and len(cmp["cmp_freq_axis"]) > 16
                and len(cmp["cmp_freq_axis"]) == len(cmp["cmp_measured_mags"])
                and len(cmp["cmp_freq_axis"]) == len(cmp["cmp_target_mags"])
                and len(cmp["cmp_freq_axis"]) == len(cmp["cmp_filter_mags"])
                and len(cmp["cmp_freq_axis"]) == len(cmp["cmp_confidence_mask"])
            ):
                analysis_mode = "comparison"
    except (AttributeError, TypeError, ValueError, FloatingPointError, IndexError, KeyError):
        cmp = None
        analysis_mode = "native"

    if getattr(cfg, "enable_afdw", False):
        base = float(getattr(cfg, "fdw_cycles", 15.0))
        min_c = max(3.0, base / 3.0)
        m_anal = apply_adaptive_fdw(
            freq_axis,
            m_anal,
            conf_mask,
            base_cycles=base,
            min_cycles=min_c,
        )

    ctx = DspContext(
        n_fft=n_fft,
        freq_axis=freq_axis,
        gain_db=gain_db,
        target_mags=target_mags,
        st=st,
    )

    _result = PreprocessResult(
        ctx=ctx,
        f_in=f_in,
        m_in=m_in,
        p_in=p_in,
        m_smooth_std=m_smooth_std,
        p_smooth=p_smooth,
        m_interp=m_interp,
        p_rad_raw=p_rad_raw,
        p_rad_interp=p_rad_interp,
        delay_slope=float(delay_slope),
        m_plot_db=None if m_plot_db is None else np.asarray(m_plot_db, dtype=float),
        complex_meas=complex_meas,
        m_anal=m_anal,
        p_anal_rad=p_anal_rad,
        complex_anal=complex_anal,
        conf_mask=conf_mask,
        reflections=reflections,
        cmp=cmp,
        analysis_mode=analysis_mode,
        is_psy=is_psy,
    )
    if _cache_key is not None:
        if len(_PREPROCESS_CACHE) >= 16:
            _PREPROCESS_CACHE.clear()
        _PREPROCESS_CACHE[_cache_key] = clone_preprocess_result(_result)
    return _result
