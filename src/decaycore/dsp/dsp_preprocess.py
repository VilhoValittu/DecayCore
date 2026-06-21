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

from .cache_utils import BoundedLruCache
from .decaycore_analysis import analyze_acoustic_confidence
from .decaycore_leveling import compute_leveling
from .phase import remove_time_of_flight
from .smoothing import apply_adaptive_fdw, apply_smoothing_std, psychoacoustic_smoothing
from .dsp_types import DspContext, PreprocessResult

_PREPROCESS_CACHE = BoundedLruCache(16)

# Caches the measurement-fixed portion of run_preprocess (everything before
# AFDW). Key is based on object identity of the input arrays plus a lightweight
# fingerprint so different arrays with the same id after GC are detected.
# Trials that differ only in fdw_cycles or enable_afdw still hit this cache.
_MEAS_FIXED_CACHE = BoundedLruCache(8)
_MEAS_FIXED_STATS: dict = {"hits": 0, "misses": 0}


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
            id(freqs),
            id(meas_mags),
            id(raw_phases),
            n_fft,
            int(float(fs)),
            n,
            round(float(freqs[0]), 4),
            round(float(freqs[-1]), 4),
        )
    except (TypeError, ValueError, IndexError, OverflowError):
        return None


def _copy_array(values):
    return np.array(values, copy=True)


def _fast_deepcopy(obj):
    # copy.deepcopy on per-objekti-overheadiltaan kallis per-trial-kuumassa
    # polussa; telemetria/cmp-rakenteet ovat dict/list/ndarray/skalaari-
    # puita, jotka kopioituvat suoraan. Tuntemattomat tyypit menevat
    # edelleen deepcopyn kautta.
    if obj is None or isinstance(obj, (bool, int, float, complex, str, bytes)):
        return obj
    if isinstance(obj, np.ndarray):
        return obj.copy()
    if isinstance(obj, dict):
        return {key: _fast_deepcopy(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_fast_deepcopy(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_fast_deepcopy(value) for value in obj)
    return deepcopy(obj)


def clone_preprocess_result(result: PreprocessResult) -> PreprocessResult:
    ctx = DspContext(
        n_fft=int(result.ctx.n_fft),
        freq_axis=_copy_array(result.ctx.freq_axis),
        gain_db=_copy_array(result.ctx.gain_db),
        target_mags=_copy_array(result.ctx.target_mags),
        st=_fast_deepcopy(dict(result.ctx.st or {})),
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
        reflections=_fast_deepcopy(list(result.reflections or [])),
        cmp=None if result.cmp is None else _fast_deepcopy(dict(result.cmp)),
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


def _run_preprocess_cached_result(
    freqs,
    meas_mags,
    raw_phases,
    cfg,
    stereo_link_ctx,
    presolve_mode: bool,
) -> tuple[tuple | None, PreprocessResult | None]:
    cache_key = _preprocess_cache_key(freqs, meas_mags, raw_phases, cfg, stereo_link_ctx, presolve_mode)
    if cache_key is None:
        return None, None
    cached = _PREPROCESS_CACHE.get(cache_key)
    if cached is None:
        return cache_key, None
    return cache_key, clone_preprocess_result(cached)


def _meas_fixed_unpack(core: dict, extras: dict) -> dict:
    return {
        "f_in": core["f_in"],
        "m_in": core["m_in"],
        "p_in": core["p_in"],
        "freq_axis": core["freq_axis"],
        "m_smooth_std": core["m_smooth_std"],
        "p_smooth": core["p_smooth"],
        "m_interp": core["m_interp"],
        "p_rad_raw": core["p_rad_raw"],
        "p_rad_interp": core["p_rad_interp"],
        "delay_slope": core["delay_slope"],
        "m_plot_db": extras["m_plot_db"],
        "complex_meas": core["complex_meas"],
        "m_anal": np.copy(core["m_anal_base"]),
        "p_anal_rad": core["p_anal_rad"],
        "complex_anal": core["complex_anal"],
        "conf_mask": core["conf_mask"],
        "reflections": core["reflections"],
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
        target_level_db_cmp, calc_offset_db_cmp, meas_level_db_window_cmp, target_level_db_window_cmp, offset_method_cmp, s_min_cmp, s_max_cmp = (
            compute_leveling(cfg, freq_cmp, m_cmp_raw, target_cmp, stereo_link_ctx=stereo_link_ctx)
        )
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
    m_smooth_std = analysis_smoothing_lf_to_hf(
        f_in, m_in, low_bw=1 / 3.0, high_bw=1 / 1.0, f_lo=230.0, f_hi=400.0
    )
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


def _store_preprocess_cache_result(cache_key: tuple | None, result: PreprocessResult) -> None:
    if cache_key is None:
        return
    _PREPROCESS_CACHE.put(cache_key, clone_preprocess_result(result))


def run_preprocess(freqs, meas_mags, raw_phases, cfg, *, stereo_link_ctx=None, presolve_mode: bool = False) -> PreprocessResult:
    cache_key, cached_result = _run_preprocess_cached_result(
        freqs,
        meas_mags,
        raw_phases,
        cfg,
        stereo_link_ctx,
        bool(presolve_mode),
    )
    if cached_result is not None:
        return cached_result
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
    result = _run_preprocess_build_result(meas_data, n_fft=n_fft)
    _store_preprocess_cache_result(cache_key, result)
    return result
