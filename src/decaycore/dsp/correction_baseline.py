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
import threading
from concurrent.futures import Future
from typing import Any, Callable

import numpy as np

from decaycore.auto_mode.auto_mode_profile import profiled_section
from decaycore.common.measurement_features import estimate_schroeder_hz, median_rt60_mid_band

from ._measurement_ctx_local import get_measurement_ctx
from .cache_utils import BoundedLruCache
from .correction_types import (
    BaselineComparisonTelemetry,
    BaselineNativeTelemetry,
    MeasurementSideContext,
    _BaselineContext,
)
from .decaycore_analysis import calculate_rt60, calculate_rt60_bands
from .leveling_parts import compute_leveling
from .dsp_config import CfgReader
from .limits import build_slope_limit_envelope
from .mag_shape import _active_correction_band_min
from .phase import get_min_phase_impulse
from .tdc import apply_smart_tdc

# ---- RT60 computation cache ------------------------------------------------
# This cache contains only magnitude-derived fallback estimates. Measured RT60
# metadata remains source-of-truth data and never enters this cache.
_RT60_CACHE_MAX_ITEMS = 32
_RT60_CACHE_KEY_VERSION = b"decaycore-rt60-fallback-v1"
_RT60_CACHE_MISS = object()
_Rt60CacheValue = tuple[float, dict, float]
_RT60_CACHE = BoundedLruCache(_RT60_CACHE_MAX_ITEMS)
_RT60_SINGLEFLIGHT_LOCK = threading.Lock()
_RT60_INFLIGHT: dict[tuple[int, str], Future[_Rt60CacheValue]] = {}
_RT60_CACHE_GENERATION = 0


def _rt60_cache_key(f_in: np.ndarray, m_in: np.ndarray, fs: int, num_taps: int | None = None) -> str:
    """Build an exact, framed key for the fixed fallback RT60 pipeline.

    ``num_taps`` remains in the internal compatibility signature but is not
    acoustically relevant: fallback RT60 reconstruction uses a fixed grid and
    FFT length independent of the exported FIR length.
    """
    _ = num_taps
    h = hashlib.blake2b(digest_size=16)
    h.update(_RT60_CACHE_KEY_VERSION)
    for label, values in ((b"freq", f_in), (b"mag", m_in)):
        arr = np.ascontiguousarray(np.asarray(values, dtype="<f8").reshape(-1))
        h.update(label)
        h.update(int(arr.size).to_bytes(8, "little", signed=False))
        h.update(memoryview(arr).cast("B"))
    h.update(b"fs")
    h.update(int(fs).to_bytes(8, "little", signed=True))
    return h.hexdigest()


def _clear_rt60_cache() -> None:
    """Explicitly clear fallback RT60 results without racing in-flight work."""
    global _RT60_CACHE_GENERATION
    with _RT60_SINGLEFLIGHT_LOCK:
        _RT60_CACHE_GENERATION += 1
        _RT60_CACHE.clear()


def _copy_rt60_cache_value(value: _Rt60CacheValue) -> _Rt60CacheValue:
    current_rt60, rt60_bands, band_avg = value
    return float(current_rt60), dict(rt60_bands), float(band_avg)


def _rt60_cache_get_or_compute(
    key: str,
    compute: Callable[[], _Rt60CacheValue],
) -> _Rt60CacheValue:
    """Return one cached result per key while allowing different keys in parallel."""
    cached = _RT60_CACHE.get(key, _RT60_CACHE_MISS)
    if cached is not _RT60_CACHE_MISS:
        return _copy_rt60_cache_value(cached)

    with _RT60_SINGLEFLIGHT_LOCK:
        cached = _RT60_CACHE.get(key, _RT60_CACHE_MISS)
        if cached is not _RT60_CACHE_MISS:
            return _copy_rt60_cache_value(cached)

        generation = int(_RT60_CACHE_GENERATION)
        flight_key = (generation, key)
        future = _RT60_INFLIGHT.get(flight_key)
        is_leader = future is None
        if future is None:
            future = Future()
            _RT60_INFLIGHT[flight_key] = future

    if not is_leader:
        return _copy_rt60_cache_value(future.result())

    try:
        value = _copy_rt60_cache_value(compute())
    except BaseException as exc:
        with _RT60_SINGLEFLIGHT_LOCK:
            if _RT60_INFLIGHT.get(flight_key) is future:
                _RT60_INFLIGHT.pop(flight_key, None)
        future.set_exception(exc)
        raise

    with _RT60_SINGLEFLIGHT_LOCK:
        if generation == _RT60_CACHE_GENERATION:
            _RT60_CACHE.put(key, value)
        if _RT60_INFLIGHT.get(flight_key) is future:
            _RT60_INFLIGHT.pop(flight_key, None)
    future.set_result(value)
    return _copy_rt60_cache_value(value)


def _smooth_with_edge_padding(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    k = np.asarray(kernel, dtype=float).reshape(-1)
    if arr.size == 0 or k.size == 0:
        return arr.copy()
    half = int(k.size // 2)
    if half <= 0:
        return arr.copy()
    padded = np.pad(arr, (half, half), mode="edge")
    return np.convolve(padded, k, mode="valid")


def apply_null_guard_target(
    freq_axis: np.ndarray,
    target_mags: np.ndarray,
    measured_mags: np.ndarray,
    *,
    mag_c_min: float,
    mag_c_max: float,
    guard_max_hz: float | None = None,
    enable: bool = True,
    depth_db: float = 12.0,  # min dip depth to start guarding
    max_blend: float = 0.85,  # max blend toward measured at dip center
    max_total_relax_db: float = 12.0,  # cap how much target can be relaxed downward vs original
    smooth_oct: float = 0.18,  # gaussian smoothing width (in octaves) for the mask
) -> np.ndarray:
    """Makes target more realistic by relaxing deep, narrow cancellation nulls.
    We detect dips relative to a smoothed "envelope" of the measured response,
    then blend target toward measured at those frequencies.
    """
    if not enable:
        return target_mags
    f = np.asarray(freq_axis, dtype=float)
    t = np.asarray(target_mags, dtype=float)
    m = np.asarray(measured_mags, dtype=float)
    if f.size != t.size or f.size != m.size or f.size < 32:
        return t

    # active correction band
    try:
        fmin = float(mag_c_min or 0.0)
        fmax = float(mag_c_max or 0.0)
    except (TypeError, ValueError, OverflowError):
        fmin, fmax = 0.0, 0.0
    try:
        guard_hi = float(guard_max_hz) if guard_max_hz is not None else float("nan")
    except (TypeError, ValueError, OverflowError):
        guard_hi = float("nan")
    if np.isfinite(guard_hi) and guard_hi > 0.0:
        fmax = min(float(fmax), float(guard_hi))
    if not (np.isfinite(fmin) and np.isfinite(fmax) and fmax > fmin and fmax > 0):
        # if no band set, do nothing (safer default)
        return t

    band = (f >= fmin) & (f <= fmax)
    if np.count_nonzero(band) < 16:
        return t

    # log-frequency axis for smoothing
    lf = np.log2(np.clip(f, 1e-3, None))
    # estimate median log-step
    dlf = np.diff(lf[band])
    dlf = dlf[np.isfinite(dlf) & (dlf > 0)]
    if dlf.size < 8:
        return t
    step = float(np.median(dlf))

    # build a smooth "envelope" of measured response in-band
    # (gaussian smoothing in log-frequency domain)
    sigma_bins = max(1.0, float(smooth_oct) / max(step, 1e-6))
    env = m.copy()
    # simple gaussian smoothing without scipy: do it via convolution
    # kernel length ~ 6*sigma
    half = int(np.ceil(3.0 * sigma_bins))
    x = np.arange(-half, half + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma_bins) ** 2)
    k /= np.sum(k)
    env_band = _smooth_with_edge_padding(env[band], k)
    env[band] = env_band

    # dip depth relative to envelope
    dip = env - m  # positive where measured is below envelope
    dip = np.where(band, dip, 0.0)

    # soft mask based on depth
    try:
        d0 = float(depth_db)
    except (TypeError, ValueError, OverflowError):
        d0 = 12.0
    d0 = float(np.clip(d0, 3.0, 30.0))

    # weight grows from 0 at depth=d0 to 1 at ~2*d0
    w = np.clip((dip - d0) / max(d0, 1e-6), 0.0, 1.0)
    recovery_mask = _null_guard_local_recovery_mask(
        f[band],
        m[band],
        depth_db=d0,
        side_oct=max(0.35, float(smooth_oct) * 2.5),
    )
    w_band_base = w[band]
    if recovery_mask.shape == w_band_base.shape:
        w_band_base = w_band_base * recovery_mask.astype(float)
        w_new = np.zeros_like(w)
        w_new[band] = w_band_base
        w = w_new
    # sharpen slightly so only true nulls get high weight
    w = w**1.4

    # smooth the mask too (prevents zipper artifacts)
    w_band = _smooth_with_edge_padding(w[band], k)
    w2 = np.zeros_like(w)
    w2[band] = np.clip(w_band, 0.0, 1.0)

    try:
        mb = float(max_blend)
    except (TypeError, ValueError, OverflowError):
        mb = 0.85
    mb = float(np.clip(mb, 0.0, 1.0))
    w2 *= mb

    # blend target toward measured (only where mask > 0)
    t0 = t.copy()
    t_new = (1.0 - w2) * t + w2 * m

    # cap total downward relaxation vs original target (avoid over-darkening)
    try:
        cap = float(max_total_relax_db)
    except (TypeError, ValueError, OverflowError):
        cap = 12.0
    cap = float(np.clip(cap, 0.0, 30.0))
    # do not allow target to drop more than 'cap' dB below original target at any bin
    t_new = np.maximum(t_new, t0 - cap)

    return t_new


def _null_guard_local_recovery_mask(
    freqs_hz: np.ndarray,
    measured_mags: np.ndarray,
    *,
    depth_db: float,
    side_oct: float = 0.45,
) -> np.ndarray:
    """Return bins that look like local dips instead of one-sided LF rolloff."""
    f = np.asarray(freqs_hz, dtype=float).reshape(-1)
    m = np.asarray(measured_mags, dtype=float).reshape(-1)
    if f.size != m.size or f.size < 8:
        return np.zeros_like(f, dtype=bool)
    valid = np.isfinite(f) & np.isfinite(m) & (f > 0.0)
    if int(np.count_nonzero(valid)) < 8:
        return np.zeros_like(f, dtype=bool)

    lf = np.log2(np.clip(f, 1e-9, None))
    try:
        side = float(side_oct)
    except (TypeError, ValueError, OverflowError):
        side = 0.45
    if not np.isfinite(side) or side <= 0.0:
        side = 0.45
    side = float(np.clip(side, 0.25, 0.75))
    min_recovery_db = float(max(3.0, 0.35 * float(depth_db)))

    out = np.zeros_like(f, dtype=bool)
    idx = np.arange(f.size)
    lo = np.searchsorted(lf, lf - side, side="left")
    hi = np.searchsorted(lf, lf + side, side="right")
    eligible = valid & (lo < idx) & (hi > idx + 1)
    if not bool(np.any(eligible)):
        return out
    # Ei-finite naapurit -inf:na: tyhja/kokonaan ei-finite puoli tuottaa
    # -inf-maksimin -> recovery ei-finite -> False, kuten per-bin-silmukassa.
    m_neg = np.where(np.isfinite(m), m, -np.inf)
    lo_e = lo[eligible]
    hi_e = hi[eligible]
    i_e = idx[eligible]
    left_max = _range_max_query(m_neg, lo_e, i_e)
    right_max = _range_max_query(m_neg, i_e + 1, hi_e)
    m_e = m[eligible]
    recovery = np.minimum(left_max - m_e, right_max - m_e)
    out[eligible] = np.isfinite(recovery) & (recovery >= min_recovery_db)
    return out


def _range_max_query(values: np.ndarray, starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    """Vektoroitu range-max: max(values[starts[j]:stops[j]]) jokaiselle kyselylle.

    Vaatii 0 <= starts[j] < stops[j] <= values.size. Sparse-table-RMQ; max on
    assosiatiivinen ja jarjestysriippumaton, joten tulos on bitti-identtinen
    viipaleittaisen np.max:n kanssa.
    """
    n = int(values.size)
    lengths = (stops - starts).astype(np.int64)
    max_len = int(np.max(lengths))
    k_max = int(np.frexp(float(max_len))[1]) - 1
    levels = np.full((k_max + 1, n), -np.inf, dtype=float)
    levels[0, :] = values
    width = 1
    for k in range(1, k_max + 1):
        count = n - 2 * width + 1
        if count > 0:
            levels[k, :count] = np.maximum(levels[k - 1, :count], levels[k - 1, width : width + count])
        width *= 2
    k_q = (np.frexp(lengths.astype(float))[1] - 1).astype(np.int64)
    pow2 = np.left_shift(np.int64(1), k_q)
    return np.maximum(levels[k_q, starts], levels[k_q, stops - pow2])


def _resample_or_interpolate_to_axis(
    meas_freq: np.ndarray,
    meas_values: np.ndarray,
    axis: np.ndarray,
) -> np.ndarray:
    """Yhtenainen interpolointikaare, jotta akselisiirrot ovat yhdessa paikassa."""
    return np.interp(axis, meas_freq, meas_values)


def _build_target_curve(
    freq_axis: np.ndarray,
    cfg: Any,
    interpolate_response: Callable[..., np.ndarray],
) -> np.ndarray:
    """Rakentaa tavoitekayran (house curve) nykyisen ehdollisen logiikan mukaan."""
    if (
        cfg.house_freqs is not None
        and cfg.house_mags is not None
        and len(cfg.house_freqs) >= 2
        and len(cfg.house_mags) >= 2
    ):
        return interpolate_response(cfg.house_freqs, cfg.house_mags, freq_axis)
    return np.zeros_like(freq_axis, dtype=float)


def _apply_target_preview_adjustments(target_mag_db: np.ndarray, cfg: Any) -> np.ndarray:
    """Paikka UI-preview-saadolle; nyt identiteetti, jotta kaytos sailyy ennallaan."""
    return target_mag_db


def _leveling_uses_forced_offset(cfg: Any, stereo_link_ctx: Any | None = None) -> bool:
    forced_offset = None
    try:
        forced_offset = getattr(cfg, "lvl_force_offset_db", None)
    except (AttributeError, TypeError, ValueError):
        forced_offset = None
    if stereo_link_ctx is not None:
        try:
            ctx_forced = getattr(stereo_link_ctx, "forced_offset_db", None)
            if ctx_forced is not None:
                forced_offset = ctx_forced
        except (AttributeError, TypeError, ValueError):
            pass
    if forced_offset is None:
        return False
    try:
        return bool(np.isfinite(float(forced_offset)))
    except (TypeError, ValueError, OverflowError):
        return False


def _apply_uniform_target_shift_leveling(
    *,
    cfg: Any,
    stereo_link_ctx: Any | None,
    target_mags: np.ndarray,
    target_shift_db: float,
    calc_offset_db: float,
    target_level_db_window: float,
) -> tuple[np.ndarray, float, float]:
    shift_db = float(target_shift_db)
    shifted_target = np.asarray(target_mags, dtype=float) + shift_db
    updated_offset_db = float(calc_offset_db)
    if not _leveling_uses_forced_offset(cfg, stereo_link_ctx=stereo_link_ctx):
        updated_offset_db -= shift_db
    updated_target_level_db_window = float(target_level_db_window) + shift_db
    return shifted_target, float(updated_offset_db), float(updated_target_level_db_window)


def _prepare_correction_baseline(  # noqa: C901 - baseline assembly intentionally keeps all safety gates together
    *,
    cfg,
    freq_axis,
    f_in,
    m_in,
    reflections,
    st,
    m_anal,
    m_plot_db,
    is_psy,
    cmp,
    analysis_mode,
    gain_db,
    logger,
    interpolate_response,
    _cfg_float_allow_zero,
    stereo_link_ctx=None,
    presolve_mode: bool = False,
) -> _BaselineContext:
    cfg_reader = CfgReader(cfg)
    # m_interp_for_rt lasketaan aina: halpa (np.interp) ja käytössä myöhemmin
    # funktiossa, joten se ei voi olla else-haaran sisällä.
    m_interp_for_rt = _resample_or_interpolate_to_axis(f_in, m_in, freq_axis)

    # Prefer measured RT60 from the measurement-side context (set by engine_run).
    # Fall back to estimating from the IR when no measured data is available.
    _mctx: MeasurementSideContext | None = get_measurement_ctx()
    _measured_rt60_val: float | None = None
    _measured_rt60_bands: dict | None = None
    if _mctx is not None:
        if _mctx.measured_rt60 is not None:
            try:
                v = float(_mctx.measured_rt60)
                if v > 0.0 and np.isfinite(v):
                    _measured_rt60_val = v
            except (TypeError, ValueError):
                pass
        if isinstance(_mctx.measured_rt60_bands, dict) and _mctx.measured_rt60_bands:
            _measured_rt60_bands = dict(_mctx.measured_rt60_bands)

    if _measured_rt60_val is not None:
        # Measured RT60 is the source of truth; skip expensive estimation.
        current_rt60 = float(_measured_rt60_val)
        if _measured_rt60_bands:
            rt60_bands = {
                k: v
                for k, v in _measured_rt60_bands.items()
                if isinstance(k, (int, float)) and float(k) > 0.0 and isinstance(v, (int, float)) and float(v) > 0.0
            }
        else:
            rt60_bands = {}
        band_avg = median_rt60_mid_band(rt60_bands)
    else:
        _rt60_key = _rt60_cache_key(f_in, m_in, int(cfg.fs))

        def _compute_fallback_rt60() -> _Rt60CacheValue:
            with profiled_section("generate_filter.correction.baseline.rt60"):
                m_rt_lin = _resample_or_interpolate_to_axis(
                    freq_axis,
                    m_interp_for_rt,
                    np.linspace(0, cfg.fs / 2, 65537),
                )
                rt_ir = get_min_phase_impulse(m_rt_lin, 131072)
                fallback_rt60 = calculate_rt60(rt_ir, cfg.fs)
                fallback_bands = calculate_rt60_bands(
                    rt_ir,
                    cfg.fs,
                    f_min=31.5,
                    f_max=8000.0,
                    order=4,
                )
                fallback_band_avg = median_rt60_mid_band(fallback_bands)
            return float(fallback_rt60), dict(fallback_bands), float(fallback_band_avg)

        current_rt60, rt60_bands, band_avg = _rt60_cache_get_or_compute(
            _rt60_key,
            _compute_fallback_rt60,
        )

    if isinstance(st, dict):
        st["dsp_rt60_val_used"] = float(current_rt60)
        st["dsp_rt60_source"] = "measurement_context" if _measured_rt60_val is not None else "estimated_fallback"
        if _mctx is not None and str(getattr(_mctx, "side", "") or "").strip():
            st["measurement_context_side"] = str(_mctx.side).strip().lower()

    schroeder_hz = estimate_schroeder_hz(
        current_rt60,
        room_volume_m3=getattr(cfg, "room_volume_m3", 40.0),
    )
    if schroeder_hz is not None and isinstance(st, dict):
        st["schroeder_hz_estimate"] = float(schroeder_hz)

    # B) Target ja mahdollinen preview-saato (nyt no-op)
    with profiled_section("generate_filter.correction.baseline.target_curve"):
        target_mags = _build_target_curve(freq_axis, cfg, interpolate_response)
        target_mags = _apply_target_preview_adjustments(target_mags, cfg)

    tdc_telemetry: dict[str, Any] = {}
    if cfg_reader.bool("enable_tdc", False):
        with profiled_section("generate_filter.correction.baseline.tdc"):
            rt60_for_tdc = rt60_bands or current_rt60
            tdc_strength = cfg_reader.float_allow_zero("tdc_strength", 50.0)
            tdc_max_red = cfg_reader.float_allow_zero("tdc_max_reduction_db", 9.0)
            tdc_slope = cfg_reader.float_allow_zero("tdc_slope_db_per_oct", 0.0)
            if tdc_strength < 0:
                tdc_strength = 0.0
            if tdc_strength > 100:
                tdc_strength = 100.0
            if tdc_max_red < 0:
                tdc_max_red = 0.0
            if tdc_max_red > 24:
                tdc_max_red = 24.0
            if tdc_slope < 0:
                tdc_slope = 0.0
            if tdc_slope > 24:
                tdc_slope = 24.0
            tdc_band_max = cfg_reader.float_allow_zero("mag_c_max", 300.0)
            if not np.isfinite(tdc_band_max) or tdc_band_max <= 0.0:
                tdc_band_max = 300.0
            tdc_max_hz = float(min(300.0, max(80.0, tdc_band_max)))
            # TDC targets modal decay; above the room's Schroeder frequency the field
            # is statistical and decay-driven cuts lose their physical basis. Narrow
            # the band to the per-room estimate — never widen past the 300 Hz cap.
            if schroeder_hz is not None and np.isfinite(float(schroeder_hz)):
                tdc_max_hz = float(min(tdc_max_hz, max(80.0, float(schroeder_hz))))
            target_mags = apply_smart_tdc(
                freq_axis,
                target_mags,
                reflections,
                rt60_for_tdc,
                tdc_strength / 100.0,
                max_total_reduction_db=tdc_max_red,
                max_slope_db_per_oct=tdc_slope,
                max_tdc_hz=tdc_max_hz,
                telemetry=tdc_telemetry,
            )
    else:
        tdc_telemetry.update(
            {
                "tdc_enabled": False,
                "tdc_applied": False,
                "tdc_events_seen": 0,
                "tdc_events_used": 0,
                "tdc_events_skipped_high": 0,
                "tdc_peak_reduction_db": 0.0,
                "tdc_peak_reduction_hz": 0.0,
                "tdc_reduction_rms_db": 0.0,
                "tdc_reduction_area_db_hz": 0.0,
                "tdc_reduction_band_low_hz": 0.0,
                "tdc_reduction_band_high_hz": 0.0,
                "tdc_max_tdc_hz": 0.0,
                "tdc_skip_reason": "disabled",
            }
        )
    if isinstance(st, dict):
        st.update(tdc_telemetry)
    # ---- Null-guard: make target realistic for deep/narrow cancellations ----
    ng_enable = cfg_reader.bool("enable_null_guard", True)
    if ng_enable:
        with profiled_section("generate_filter.correction.baseline.null_guard"):
            mag_c_min = _active_correction_band_min(cfg)
            mag_c_max = cfg_reader.float_allow_zero("mag_c_max", 0.0)
            null_guard_max_hz = float(
                np.clip(
                    float(schroeder_hz) if schroeder_hz is not None and np.isfinite(float(schroeder_hz)) else 300.0,
                    120.0,
                    500.0,
                )
            )
            if isinstance(st, dict):
                st["null_guard_max_hz"] = float(null_guard_max_hz)
            # use analysis magnitude (m_anal) already on freq_axis scale
            target_mags = apply_null_guard_target(
                np.asarray(freq_axis, dtype=float),
                np.asarray(target_mags, dtype=float),
                np.asarray(m_anal, dtype=float),
                mag_c_min=mag_c_min,
                mag_c_max=mag_c_max,
                guard_max_hz=null_guard_max_hz,
                enable=True,
                depth_db=cfg_reader.float("null_guard_depth_db", 12.0),
                max_blend=cfg_reader.float("null_guard_max_blend", 0.85),
                max_total_relax_db=cfg_reader.float("null_guard_max_relax_db", 12.0),
                smooth_oct=cfg_reader.float("null_guard_smooth_oct", 0.18),
            )
    hpf_f = 0.0
    hpf_order = 0
    if cfg.hpf_settings and cfg.hpf_settings.get("enabled"):
        hpf_f = float(cfg.hpf_settings.get("freq", 0.0) or 0.0)
        hpf_order = int(cfg.hpf_settings.get("order", 0) or 0)
    target_level_db = 0.0
    calc_offset_db = 0.0
    meas_level_db_window = 0.0
    target_level_db_window = 0.0
    offset_method = "Init"
    shared_target_shift_db = None
    if stereo_link_ctx is not None:
        try:
            v = getattr(stereo_link_ctx, "shared_target_shift_db", None)
            if v is not None:
                v = float(v)
                if np.isfinite(v):
                    shared_target_shift_db = float(v)
        except (AttributeError, TypeError, ValueError):
            shared_target_shift_db = None
    s_min = cfg_reader.float("lvl_min", 500.0)
    s_max = cfg_reader.float("lvl_max", 2000.0)
    try:
        with profiled_section("generate_filter.correction.baseline.leveling"):
            (
                target_level_db,
                calc_offset_db,
                meas_level_db_window,
                target_level_db_window,
                offset_method,
                s_min,
                s_max,
            ) = compute_leveling(
                cfg,
                np.asarray(freq_axis, dtype=float),
                np.asarray(m_anal, dtype=float),
                np.asarray(target_mags, dtype=float),
                stereo_link_ctx=stereo_link_ctx,
            )
    except (AttributeError, TypeError, ValueError, FloatingPointError, IndexError):
        pass
    target_shift_db = 0.0
    try:
        with profiled_section("generate_filter.correction.baseline.target_shift"):
            f = np.asarray(freq_axis, dtype=float)
            t = np.asarray(target_mags, dtype=float)
            if f.size == t.size and f.size > 16:
                if shared_target_shift_db is not None:
                    target_shift_db = float(shared_target_shift_db)
                    (
                        target_mags,
                        calc_offset_db,
                        target_level_db_window,
                    ) = _apply_uniform_target_shift_leveling(
                        cfg=cfg,
                        stereo_link_ctx=stereo_link_ctx,
                        target_mags=t,
                        target_shift_db=target_shift_db,
                        calc_offset_db=calc_offset_db,
                        target_level_db_window=target_level_db_window,
                    )
                elif np.isfinite(float(s_min)) and np.isfinite(float(s_max)):
                    mask_lvl = (f >= float(s_min)) & (f <= float(s_max))
                    if int(np.count_nonzero(mask_lvl)) > 10:
                        tgt_win_mean = float(np.mean(t[mask_lvl]))
                        if np.isfinite(tgt_win_mean) and np.isfinite(float(target_level_db)):
                            target_shift_db = float(target_level_db) - tgt_win_mean
                            (
                                target_mags,
                                calc_offset_db,
                                target_level_db_window,
                            ) = _apply_uniform_target_shift_leveling(
                                cfg=cfg,
                                stereo_link_ctx=stereo_link_ctx,
                                target_mags=t,
                                target_shift_db=target_shift_db,
                                calc_offset_db=calc_offset_db,
                                target_level_db_window=target_level_db_window,
                            )
    except (AttributeError, TypeError, ValueError, FloatingPointError, IndexError):
        target_shift_db = 0.0
    native_telemetry: BaselineNativeTelemetry | None = None
    if not bool(presolve_mode):
        try:
            with profiled_section("generate_filter.correction.baseline.telemetry_native"):
                m_src = np.asarray(m_anal, dtype=float)
                if is_psy and (m_plot_db is not None):
                    mp = np.asarray(m_plot_db, dtype=float)
                    if mp.size == m_src.size:
                        m_src = mp
                measured_aligned = m_src - float(calc_offset_db)
                target_aligned = np.asarray(target_mags, dtype=float) - float(target_level_db)
                env_lo = env_hi = None
                env_pivot = None
                try:
                    max_slope = cfg_reader.float_allow_zero("max_slope_db_per_oct", 0.0)
                    max_slope_boost = cfg_reader.float_allow_zero("max_slope_boost_db_per_oct", 0.0)
                    max_slope_cut = cfg_reader.float_allow_zero("max_slope_cut_db_per_oct", 0.0)
                    if max_slope_boost <= 0.0:
                        max_slope_boost = max_slope
                    if max_slope_cut <= 0.0:
                        max_slope_cut = max_slope
                    env_lo, env_hi, env_pivot = build_slope_limit_envelope(
                        np.asarray(freq_axis, dtype=float),
                        target_aligned,
                        mag_c_min=cfg_reader.float_allow_zero("mag_c_min", 0.0),
                        mag_c_max=cfg_reader.float_allow_zero("mag_c_max", 0.0),
                        max_slope_boost_db_per_oct=float(max_slope_boost),
                        max_slope_cut_db_per_oct=float(max_slope_cut),
                    )
                except (AttributeError, TypeError, ValueError, FloatingPointError, IndexError):
                    env_lo = env_hi = None
                    env_pivot = None
                native_telemetry = BaselineNativeTelemetry(
                    analysis_mode="native",
                    freq_axis=np.asarray(freq_axis, dtype=float),
                    measured_mags=np.asarray(measured_aligned, dtype=float),
                    target_mags=np.asarray(target_aligned, dtype=float),
                    target_env_lo=None if env_lo is None else np.asarray(env_lo, dtype=float),
                    target_env_hi=None if env_hi is None else np.asarray(env_hi, dtype=float),
                    target_env_pivot_hz=float(env_pivot) if env_pivot is not None else None,
                    target_shift_db=float(target_shift_db),
                    eff_target_db=float(target_level_db),
                    target_level_db_window=float(target_level_db_window),
                    meas_level_db_window=float(meas_level_db_window),
                    offset_db=float(calc_offset_db),
                    offset_method=str(offset_method),
                    smart_scan_range=(float(s_min), float(s_max)),
                )
        except (AttributeError, TypeError, ValueError, FloatingPointError, IndexError):
            native_telemetry = None

    comparison_telemetry: BaselineComparisonTelemetry | None = None
    if not bool(presolve_mode):
        try:
            with profiled_section("generate_filter.correction.baseline.telemetry_comparison"):
                if isinstance(cmp, dict) and analysis_mode == "comparison":
                    freq_cmp = np.asarray(cmp.get("cmp_freq_axis", []) or [], dtype=float)
                    if freq_cmp.size > 8:
                        m_cmp_raw = np.interp(freq_cmp, freq_axis, m_anal)
                        target_cmp = np.interp(freq_cmp, freq_axis, target_mags)
                        filt_cmp = np.interp(freq_cmp, freq_axis, gain_db)
                        (
                            target_level_db_cmp,
                            calc_offset_db_cmp,
                            meas_level_db_window_cmp,
                            target_level_db_window_cmp,
                            offset_method_cmp,
                            s_min_cmp,
                            s_max_cmp,
                        ) = compute_leveling(cfg, freq_cmp, m_cmp_raw, target_cmp, stereo_link_ctx=stereo_link_ctx)
                        comparison_telemetry = BaselineComparisonTelemetry(
                            analysis_mode="comparison",
                            target_mags=np.asarray(target_cmp, dtype=float),
                            measured_mags=np.asarray(m_cmp_raw - calc_offset_db_cmp, dtype=float),
                            filter_mags=np.asarray(filt_cmp, dtype=float),
                            eff_target_db=float(target_level_db_cmp),
                            offset_db=float(calc_offset_db_cmp),
                            smart_scan_range=(float(s_min_cmp), float(s_max_cmp)),
                            meas_level_db_window=float(meas_level_db_window_cmp),
                            target_level_db_window=float(target_level_db_window_cmp),
                            offset_method=str(offset_method_cmp),
                            target_shift_db=float(target_shift_db),
                        )
        except (AttributeError, TypeError, ValueError, FloatingPointError, IndexError):
            comparison_telemetry = None

    return _BaselineContext(
        m_interp=np.asarray(m_interp_for_rt, dtype=float),
        current_rt60=float(current_rt60),
        rt60_bands=dict(rt60_bands) if isinstance(rt60_bands, dict) else {},
        band_avg=float(band_avg),
        target_mags=np.asarray(target_mags, dtype=float),
        hpf_f=float(hpf_f),
        hpf_order=int(hpf_order),
        target_level_db=float(target_level_db),
        calc_offset_db=float(calc_offset_db),
        meas_level_db_window=float(meas_level_db_window),
        target_level_db_window=float(target_level_db_window),
        offset_method=str(offset_method),
        s_min=float(s_min),
        s_max=float(s_max),
        target_shift_db=float(target_shift_db),
        cmp=cmp,
        analysis_mode=str(analysis_mode),
        gain_db=np.asarray(gain_db, dtype=float),
        native_telemetry=native_telemetry,
        comparison_telemetry=comparison_telemetry,
    )
