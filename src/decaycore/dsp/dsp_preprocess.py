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

import hashlib

import numpy as np

from .cache_utils import BoundedLruCache
from .decaycore_analysis import analyze_acoustic_confidence
from .leveling_parts import compute_leveling
from .phase import remove_time_of_flight
from .smoothing import apply_adaptive_fdw, apply_smoothing_std, psychoacoustic_smoothing
from .dsp_types import DspContext, PreprocessResult

# Caches the measurement-fixed portion of run_preprocess (everything before
# AFDW). Its key fingerprints the input contents, so recycled NumPy object
# identities cannot reuse a previous measurement's result. Trials that differ
# only in fdw_cycles or enable_afdw still hit this cache.
_MEAS_FIXED_CACHE_MAX_BYTES = 128 * 1024 * 1024
_MEAS_FIXED_CACHE = BoundedLruCache(8, max_bytes=_MEAS_FIXED_CACHE_MAX_BYTES)
_MEAS_FIXED_STATS: dict = {"hits": 0, "misses": 0}


def _cache_hash_array(values, *, decimals: int = 6) -> str:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return "empty"
    rounded = np.round(arr, int(decimals))
    rounded = np.nan_to_num(rounded, nan=1.0e300, posinf=1.0e301, neginf=-1.0e301)
    return hashlib.blake2b(np.ascontiguousarray(rounded, dtype=np.float64).view(np.uint8), digest_size=16).hexdigest()


def clear_preprocess_cache() -> None:
    _MEAS_FIXED_CACHE.clear()


def clear_meas_fixed_cache() -> None:
    _MEAS_FIXED_CACHE.clear()


def get_meas_fixed_cache_stats() -> dict:
    return dict(_MEAS_FIXED_STATS)


def _meas_fixed_cache_key(
    freqs,
    meas_mags,
    raw_phases,
    n_fft: int,
    fs: float,
) -> tuple | None:
    # Keys only the presolve_mode-independent core (FFT axis, smoothing,
    # confidence, reflections). The presolve_mode/comparison/plot-dependent
    # extras (m_plot_db, cmp) are layered on top per call, so they are not
    # part of this key — this lets the presolve pass and the real pipeline
    # pass share the heavy core for the same measurement arrays.
    try:
        n = len(freqs)
        if n < 2:
            return None
        return (
            n_fft,
            int(float(fs)),
            _cache_hash_array(freqs),
            _cache_hash_array(meas_mags),
            _cache_hash_array(raw_phases),
        )
    except (TypeError, ValueError, IndexError, OverflowError):
        return None


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


def _meas_fixed_unpack(core: dict, extras: dict) -> dict:
    # The cached core is immutable process-local data. Return inexpensive array
    # copies so correction stages and callers receive independent per-run state
    # without recursively cloning large telemetry/comparison dictionaries.
    return {
        "f_in": np.array(core["f_in"], copy=True),
        "m_in": np.array(core["m_in"], copy=True),
        "p_in": np.array(core["p_in"], copy=True),
        "freq_axis": np.array(core["freq_axis"], copy=True),
        "m_smooth_std": np.array(core["m_smooth_std"], copy=True),
        "p_smooth": np.array(core["p_smooth"], copy=True),
        "m_interp": np.array(core["m_interp"], copy=True),
        "p_rad_raw": np.array(core["p_rad_raw"], copy=True),
        "p_rad_interp": np.array(core["p_rad_interp"], copy=True),
        "delay_slope": core["delay_slope"],
        "m_plot_db": extras["m_plot_db"],
        "complex_meas": np.array(core["complex_meas"], copy=True),
        "m_anal": np.array(core["m_anal_base"], copy=True),
        "p_anal_rad": np.array(core["p_anal_rad"], copy=True),
        "complex_anal": np.array(core["complex_anal"], copy=True),
        "conf_mask": np.array(core["conf_mask"], copy=True),
        "reflections": [dict(node) for node in core["reflections"]],
        "cmp": extras["cmp"],
        "analysis_mode": extras["analysis_mode"],
        "is_psy": extras["is_psy"],
    }


def _prepare_preprocess_inputs(freqs, meas_mags, raw_phases) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    min_len = min(len(freqs), len(meas_mags), len(raw_phases))
    f_in = np.asarray(freqs[:min_len], dtype=float)
    m_in = np.asarray(meas_mags[:min_len], dtype=float)
    p_in = np.asarray(raw_phases[:min_len], dtype=float)
    if f_in.size <= 1:
        return f_in, m_in, p_in
    order = np.argsort(f_in, kind="mergesort")
    f_in = f_in[order]
    m_in = m_in[order]
    p_in = p_in[order]
    uniq_mask = np.concatenate(([True], np.diff(f_in) > 0.0))
    return f_in[uniq_mask], m_in[uniq_mask], p_in[uniq_mask]


def _compute_comparison_payload(
    *,
    cfg,
    presolve_mode: bool,
    stereo_link_ctx,
    freq_axis: np.ndarray,
    m_anal: np.ndarray,
    p_anal_rad: np.ndarray,
) -> tuple[dict | None, str]:
    cmp = None
    analysis_mode = "native"
    if bool(presolve_mode) or not bool(getattr(cfg, "comparison_mode", False)):
        return cmp, analysis_mode
    try:
        ref_fs = int(getattr(cfg, "comparison_ref_fs", 44100) or 44100)
        ref_taps = int(getattr(cfg, "comparison_ref_taps", 65536) or 65536)
        ref_nfft = int(ref_taps)
        freq_cmp_full = np.fft.rfftfreq(ref_nfft, d=1.0 / float(ref_fs))
        fmax = float(freq_axis[-1]) if freq_axis.size else 0.0
        freq_cmp = freq_cmp_full[freq_cmp_full <= fmax] if fmax > 0 else freq_cmp_full
        m_cmp_raw = np.interp(freq_cmp, freq_axis, m_anal)
        p_cmp_rad = np.interp(freq_cmp, freq_axis, p_anal_rad)
        complex_cmp = 10 ** (m_cmp_raw / 20.0) * np.exp(1j * p_cmp_rad)
        conf_cmp, refl_cmp, _ = analyze_acoustic_confidence(freq_cmp, complex_cmp, ref_fs)
        target_cmp = np.zeros_like(freq_cmp, dtype=float)
        (
            target_level_db_cmp,
            calc_offset_db_cmp,
            meas_level_db_window_cmp,
            target_level_db_window_cmp,
            offset_method_cmp,
            s_min_cmp,
            s_max_cmp,
        ) = compute_leveling(cfg, freq_cmp, m_cmp_raw, target_cmp, stereo_link_ctx=stereo_link_ctx)
        cmp = {
            "cmp_ref_fs": float(ref_fs),
            "cmp_ref_taps": float(ref_taps),
            "cmp_freq_axis": freq_cmp.tolist(),
            "cmp_target_mags": target_cmp.tolist(),
            "cmp_measured_mags": (m_cmp_raw - calc_offset_db_cmp).tolist(),
            "cmp_filter_mags": np.zeros_like(freq_cmp, dtype=float).tolist(),
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
        if _comparison_payload_valid(cmp):
            analysis_mode = "comparison"
    except (AttributeError, TypeError, ValueError, FloatingPointError, IndexError, KeyError):
        cmp = None
        analysis_mode = "native"
    return cmp, analysis_mode


def _comparison_payload_valid(cmp: dict | None) -> bool:
    if not isinstance(cmp, dict):
        return False
    keys = ("cmp_freq_axis", "cmp_measured_mags", "cmp_target_mags", "cmp_filter_mags", "cmp_confidence_mask")
    if any(not isinstance(cmp.get(k), list) for k in keys):
        return False
    n = len(cmp["cmp_freq_axis"])
    if n <= 16:
        return False
    return all(len(cmp[k]) == n for k in keys)


def _store_meas_fixed_entry(cache_key: tuple | None, core: dict) -> None:
    _MEAS_FIXED_STATS["misses"] = _MEAS_FIXED_STATS.get("misses", 0) + 1
    if cache_key is None:
        return
    _MEAS_FIXED_CACHE.put(cache_key, core)


def _compute_meas_fixed_core(
    freqs,
    meas_mags,
    raw_phases,
    cfg,
    *,
    n_fft: int,
) -> dict:
    # presolve_mode-independent heavy core: rFFT axis, magnitude/phase
    # smoothing, interpolation, confidence/reflections. Shared between the
    # stereo presolve pass and the real pipeline pass via _MEAS_FIXED_CACHE.
    f_in, m_in, p_in = _prepare_preprocess_inputs(freqs, meas_mags, raw_phases)
    freq_axis = np.fft.rfftfreq(n_fft, d=1.0 / float(cfg.fs))
    m_smooth_std = analysis_smoothing_lf_to_hf(f_in, m_in, low_bw=1 / 3.0, high_bw=1 / 1.0, f_lo=230.0, f_hi=400.0)
    p_smooth_oct = 1 / 12.0 if cfg.fs > 96000 else 1 / 24.0
    p_smooth, _ = apply_smoothing_std(f_in, p_in, np.zeros_like(p_in), p_smooth_oct)
    m_interp = np.interp(freq_axis, f_in, m_in)
    p_rad_raw = np.deg2rad(np.interp(freq_axis, f_in, p_in))
    p_rad_interp, delay_slope = remove_time_of_flight(freq_axis, p_rad_raw)
    complex_meas = 10 ** (m_interp / 20.0) * np.exp(1j * p_rad_interp)
    m_anal = np.interp(freq_axis, f_in, m_smooth_std)
    p_anal_rad = np.deg2rad(np.interp(freq_axis, f_in, p_smooth))
    p_anal_rad, _ = remove_time_of_flight(freq_axis, p_anal_rad)
    complex_anal = 10 ** (m_anal / 20.0) * np.exp(1j * p_anal_rad)
    conf_mask, reflections, _ = analyze_acoustic_confidence(freq_axis, complex_anal, cfg.fs)
    return {
        "f_in": f_in,
        "m_in": m_in,
        "p_in": p_in,
        "freq_axis": freq_axis,
        "m_smooth_std": m_smooth_std,
        "p_smooth": p_smooth,
        "m_interp": m_interp,
        "p_rad_raw": p_rad_raw,
        "p_rad_interp": p_rad_interp,
        "delay_slope": float(delay_slope),
        "complex_meas": complex_meas,
        "m_anal_base": m_anal,
        "p_anal_rad": p_anal_rad,
        "complex_anal": complex_anal,
        "conf_mask": conf_mask,
        "reflections": reflections,
    }


def _compute_meas_fixed_extras(
    core: dict,
    cfg,
    *,
    presolve_mode: bool,
    stereo_link_ctx,
) -> dict:
    # presolve_mode-dependent extras layered on top of the shared core:
    # the psychoacoustic plot curve and the comparison payload. Cheap no-ops
    # in the common path (presolve, or non-comparison non-psy configs).
    is_psy = (not bool(presolve_mode)) and ("psy" in str(cfg.plot_smoothing_level).lower())
    m_plot_db = None
    if is_psy:
        try:
            m_plot_db = psychoacoustic_smoothing(core["freq_axis"], core["m_interp"])
        except (TypeError, ValueError, FloatingPointError):
            m_plot_db = None
    cmp, analysis_mode = _compute_comparison_payload(
        cfg=cfg,
        presolve_mode=bool(presolve_mode),
        stereo_link_ctx=stereo_link_ctx,
        freq_axis=core["freq_axis"],
        m_anal=core["m_anal_base"],
        p_anal_rad=core["p_anal_rad"],
    )
    return {
        "m_plot_db": m_plot_db,
        "cmp": cmp,
        "analysis_mode": analysis_mode,
        "is_psy": is_psy,
    }


def _run_preprocess_meas_fixed(
    freqs,
    meas_mags,
    raw_phases,
    cfg,
    *,
    n_fft: int,
    presolve_mode: bool,
    stereo_link_ctx,
) -> dict:
    mfk = _meas_fixed_cache_key(freqs, meas_mags, raw_phases, n_fft, float(cfg.fs))
    core = _MEAS_FIXED_CACHE.get(mfk) if mfk is not None else None
    if core is not None:
        _MEAS_FIXED_STATS["hits"] = _MEAS_FIXED_STATS.get("hits", 0) + 1
    else:
        core = _compute_meas_fixed_core(freqs, meas_mags, raw_phases, cfg, n_fft=n_fft)
        _store_meas_fixed_entry(mfk, core)
    extras = _compute_meas_fixed_extras(
        core,
        cfg,
        presolve_mode=bool(presolve_mode),
        stereo_link_ctx=stereo_link_ctx,
    )
    return _meas_fixed_unpack(core, extras)


def _run_preprocess_apply_afdw(cfg, meas_data: dict) -> None:
    if not getattr(cfg, "enable_afdw", False):
        return
    base = float(getattr(cfg, "fdw_cycles", 15.0))
    min_c = max(3.0, base / 3.0)
    meas_data["m_anal"] = apply_adaptive_fdw(
        meas_data["freq_axis"],
        meas_data["m_anal"],
        meas_data["conf_mask"],
        base_cycles=base,
        min_cycles=min_c,
    )


def _run_preprocess_build_result(meas_data: dict, *, n_fft: int) -> PreprocessResult:
    gain_db = np.zeros_like(meas_data["freq_axis"], dtype=float)
    st: dict = {}
    target_mags = np.zeros_like(meas_data["freq_axis"], dtype=float)
    ctx = DspContext(
        n_fft=n_fft,
        freq_axis=meas_data["freq_axis"],
        gain_db=gain_db,
        target_mags=target_mags,
        st=st,
    )
    return PreprocessResult(
        ctx=ctx,
        f_in=meas_data["f_in"],
        m_in=meas_data["m_in"],
        p_in=meas_data["p_in"],
        m_smooth_std=meas_data["m_smooth_std"],
        p_smooth=meas_data["p_smooth"],
        m_interp=meas_data["m_interp"],
        p_rad_raw=meas_data["p_rad_raw"],
        p_rad_interp=meas_data["p_rad_interp"],
        delay_slope=float(meas_data["delay_slope"]),
        m_plot_db=None if meas_data["m_plot_db"] is None else np.asarray(meas_data["m_plot_db"], dtype=float),
        complex_meas=meas_data["complex_meas"],
        m_anal=meas_data["m_anal"],
        p_anal_rad=meas_data["p_anal_rad"],
        complex_anal=meas_data["complex_anal"],
        conf_mask=meas_data["conf_mask"],
        reflections=meas_data["reflections"],
        cmp=meas_data["cmp"],
        analysis_mode=meas_data["analysis_mode"],
        is_psy=meas_data["is_psy"],
    )


def run_preprocess(
    freqs, meas_mags, raw_phases, cfg, *, stereo_link_ctx=None, presolve_mode: bool = False
) -> PreprocessResult:
    n_fft = int(cfg.num_taps)
    meas_data = _run_preprocess_meas_fixed(
        freqs,
        meas_mags,
        raw_phases,
        cfg,
        n_fft=n_fft,
        presolve_mode=bool(presolve_mode),
        stereo_link_ctx=stereo_link_ctx,
    )
    _run_preprocess_apply_afdw(cfg, meas_data)
    return _run_preprocess_build_result(meas_data, n_fft=n_fft)
