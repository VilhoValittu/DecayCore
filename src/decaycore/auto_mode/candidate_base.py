# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Base candidate generation: constants, shared helpers, builtin/optuna coarse builders."""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Static choice arrays for rng.choice calls in _build_auto_mode_candidates.
# Defined once at module level to avoid recreating them on every loop iteration.
_TDC_SLOPE_CHOICES = np.array([3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 18.0, 24.0])
_MAX_SLOPE_CHOICES = np.array([8.0, 10.0, 12.0, 14.0, 16.0,])
_TDC_STRENGTH_MIN = 5.0
_TDC_STRENGTH_MAX = 75.0
_TDC_MAX_REDUCTION_MIN_DB = 1.0
_TDC_MAX_REDUCTION_MAX_DB = 18.0
_PHASE1_SAFE_MAX_BOOST_DB = 8.0
_PHASE1_FULL_MAX_BOOST_DB = 12.0
_PHASE1_PHASE_LIMIT_PRIMARY_LO_HZ = 180.0
_PHASE1_PHASE_LIMIT_PRIMARY_HI_HZ = 420.0
_PHASE1_PHASE_LIMIT_FULL_RANGE_FRAC = 0.10
_BASS_FIRST_MODE_MIN_HZ = 120.0
_BASS_FIRST_MODE_MAX_HZ = 220.0
_CONF_PULL_MAX_MIN_HZ = 80.0
_CONF_PULL_MAX_MAX_HZ = 220.0

from decaycore.common.measurement_features import estimate_schroeder_hz

from .shared import (
    AUTO_MODE_LOW_BASS_MAX_HZ,
    AUTO_MODE_LOW_BASS_MIN_HZ,
    AUTO_MODE_MAG_C_MAX_MIN_HZ,
    AUTO_MODE_MAG_C_MIN_MAX_HZ,
    AUTO_MODE_MAG_C_MIN_MIN_HZ,
    AUTO_MODE_OPTUNA_PILOT_STARTUP_TRIALS,
    AUTO_MODE_PHASE_LIMIT_EXPLORE_GLOBAL_SIGMA_HZ,
    AUTO_MODE_PHASE_LIMIT_MAX_HZ,
    AUTO_MODE_PHASE_LIMIT_MIN_HZ,
    AUTO_MODE_PHASE_LIMIT_PRIOR_CENTER_HZ,
    _auto_is_phase_search_filter,
    _auto_goal,
    _auto_goal_is_flat_family,
    _auto_filter_cache_key,
    _auto_filter_type_for_key,
    _auto_mag_c_min_center,
    _auto_optuna_sampler_kwargs,
    _auto_phase_limit_center,
    _auto_safe_float,
    _auto_sample_mag_low_pair,
    _clip,
)


# ---------------------------------------------------------------------------
# Optuna snap / unit helpers
# ---------------------------------------------------------------------------

def _auto_filter_normalized_base_data(base_data: dict | None) -> dict:
    out = dict(base_data or {})
    fk = str(_auto_filter_cache_key(out))
    out["filter_type"] = str(_auto_filter_type_for_key(fk))
    out["_auto_filter_key"] = str(fk)
    return out

def _auto_optuna_snap_to_step(value: float, *, lo: float, hi: float, step: float) -> float:
    clipped = float(np.clip(_auto_safe_float(value, lo), float(lo), float(hi)))
    step_f = float(max(step, 1e-9))
    snapped = float(lo) + round((clipped - float(lo)) / step_f) * step_f
    decimals_txt = f"{step_f:.10f}".rstrip("0")
    decimals = len(decimals_txt.split(".", 1)[1]) if "." in decimals_txt else 0
    return float(np.clip(round(snapped, decimals), float(lo), float(hi)))


def _auto_candidate_bool_choice(rng, preferred: bool) -> bool:
    pref = bool(preferred)
    return pref if float(rng.random()) < 0.65 else (not pref)


def _auto_phase1_conservative_goal(base_data: dict | None) -> bool:
    raw = str(dict(base_data or {}).get("auto_goal", "") or "").strip().lower()
    if not raw:
        raw = "balanced"
    return raw in {"balanced", "room-safe", "room safe", "acoustic", "hybrid"}


def _auto_phase1_bool_search_enabled(base_data: dict | None) -> bool:
    raw = str(dict(base_data or {}).get("auto_goal", "") or "").strip().lower()
    return raw not in {"balanced", "room-safe", "room safe"}


def _auto_phase1_max_boost_hi(base_data: dict | None) -> float:
    return (
        float(_PHASE1_SAFE_MAX_BOOST_DB)
        if _auto_phase1_conservative_goal(base_data)
        else float(_PHASE1_FULL_MAX_BOOST_DB)
    )


def _auto_phase1_phase_limit_center(base_data: dict | None) -> float:
    data = dict(base_data or {})
    raw = data.get("phase_limit")
    center = _auto_safe_float(raw, float("nan"))
    if not np.isfinite(center):
        center = float(AUTO_MODE_PHASE_LIMIT_PRIOR_CENTER_HZ)
    return float(np.clip(center, float(AUTO_MODE_PHASE_LIMIT_MIN_HZ), float(AUTO_MODE_PHASE_LIMIT_MAX_HZ)))


def _auto_phase1_phase_limit_primary_bounds(base_data: dict | None) -> tuple[float, float]:
    center = _auto_phase1_phase_limit_center(base_data)
    lo = min(float(_PHASE1_PHASE_LIMIT_PRIMARY_LO_HZ), float(center))
    hi = max(float(_PHASE1_PHASE_LIMIT_PRIMARY_HI_HZ), float(center))
    return (
        float(np.clip(lo, float(AUTO_MODE_PHASE_LIMIT_MIN_HZ), float(AUTO_MODE_PHASE_LIMIT_MAX_HZ))),
        float(np.clip(hi, float(AUTO_MODE_PHASE_LIMIT_MIN_HZ), float(AUTO_MODE_PHASE_LIMIT_MAX_HZ))),
    )


def _auto_phase1_sample_phase_limit(base_data: dict | None, rng) -> float:
    center = _auto_phase1_phase_limit_center(base_data)
    full_frac = float(np.clip(_PHASE1_PHASE_LIMIT_FULL_RANGE_FRAC, 0.0, 1.0))
    if float(rng.random()) < full_frac:
        return float(rng.uniform(float(AUTO_MODE_PHASE_LIMIT_MIN_HZ), float(AUTO_MODE_PHASE_LIMIT_MAX_HZ)))
    lo, hi = _auto_phase1_phase_limit_primary_bounds(base_data)
    if float(rng.random()) < 0.65:
        draw = float(rng.normal(loc=float(center), scale=float(AUTO_MODE_PHASE_LIMIT_EXPLORE_GLOBAL_SIGMA_HZ)))
    else:
        draw = float(rng.uniform(float(lo), float(hi)))
    return float(_clip(draw, float(lo), float(hi)))


def _auto_optuna_window(center: float, span: float, lo: float, hi: float) -> tuple[float, float]:
    lo_eff = float(max(float(lo), float(center) - float(max(0.0, span))))
    hi_eff = float(min(float(hi), float(center) + float(max(0.0, span))))
    if hi_eff < lo_eff:
        lo_eff, hi_eff = hi_eff, lo_eff
    if abs(float(hi_eff) - float(lo_eff)) <= 1e-9:
        center_clip = float(_clip(center, lo, hi))
        return float(center_clip), float(center_clip)
    return float(lo_eff), float(hi_eff)


def _auto_optuna_project_to_unit(value: float, *, lo_eff: float, hi_eff: float) -> float:
    lo_f = float(lo_eff)
    hi_f = float(hi_eff)
    if abs(float(hi_f) - float(lo_f)) <= 1e-9:
        return 0.5
    v = float(np.clip(_auto_safe_float(value, lo_f), lo_f, hi_f))
    return float(np.clip((v - lo_f) / max(1e-12, hi_f - lo_f), 0.0, 1.0))


def _auto_optuna_suggest_centered_unit_float(trial, name: str, center: float, span: float, lo: float, hi: float) -> float:
    lo_eff, hi_eff = _auto_optuna_window(center, span, lo, hi)
    if abs(float(hi_eff) - float(lo_eff)) <= 1e-9:
        return float(lo_eff)
    u = float(trial.suggest_float(f"{name!s}_u", 0.0, 1.0))
    return float(lo_eff + u * (hi_eff - lo_eff))


def _auto_optuna_seed_centered_unit_float(
    value: float,
    *,
    center: float,
    span: float,
    lo: float,
    hi: float,
) -> float:
    lo_eff, hi_eff = _auto_optuna_window(center, span, lo, hi)
    return float(_auto_optuna_project_to_unit(value, lo_eff=float(lo_eff), hi_eff=float(hi_eff)))


def _auto_optuna_choice_from_unit(trial, name: str, choices: list[float], center: float, *, radius: int) -> float:
    band = _auto_optuna_choice_band(choices, float(center), radius=int(radius))
    if not band:
        return float(center)
    if len(band) == 1:
        return float(band[0])
    u = float(trial.suggest_float(f"{name!s}_u", 0.0, 1.0))
    idx = int(np.clip(round(u * float(len(band) - 1)), 0, len(band) - 1))
    return float(band[idx])


def _auto_optuna_seed_choice_unit(value: float, choices: list[float], center: float, *, radius: int, default: float) -> float:
    band = _auto_optuna_choice_band(choices, float(center), radius=int(radius))
    if not band:
        return 0.5
    pick = _auto_optuna_nearest_choice(value, band, default=float(default))
    if len(band) <= 1:
        return 0.5
    idx = int(min(range(len(band)), key=lambda i: abs(float(band[i]) - float(pick))))
    return float(idx / float(len(band) - 1))


def _auto_optuna_choice_band(choices: list[float], center: float, *, radius: int) -> list[float]:
    if not choices:
        return []
    idx = int(min(range(len(choices)), key=lambda i: abs(float(choices[i]) - float(center))))
    r = int(max(0, radius))
    lo = int(max(0, idx - r))
    hi = int(min(len(choices), idx + r + 1))
    band = [float(v) for v in list(choices[lo:hi])]
    return band or [float(choices[idx])]


def _auto_optuna_nearest_choice(value: float, choices: list[float], *, default: float) -> float:
    vals = [float(v) for v in list(choices or [])]
    if not vals:
        return float(default)
    x = _auto_safe_float(value, default)
    return float(min(vals, key=lambda item: abs(float(item) - float(x))))


# ---------------------------------------------------------------------------
# Adaptive frequency bounds
# ---------------------------------------------------------------------------

def _derive_adaptive_freq_bounds(base_data: dict) -> dict:
    # Derive tighter frequency parameter bounds from detected room harmonics.
    # Returns a bounds dict with adaptive lo/hi values; empty dict means fall back to constants.
    # Minimum window guard: skip adaptive bound if it would collapse below _MIN_ADAPTIVE_WINDOW Hz.
    _MIN_ADAPTIVE_WINDOW = 5.0
    _raw_l = base_data.get("harmonic_freq_hz_l")
    _raw_r = base_data.get("harmonic_freq_hz_r")
    freq_l = list(_raw_l) if _raw_l is not None and len(_raw_l) > 0 else []
    freq_r = list(_raw_r) if _raw_r is not None and len(_raw_r) > 0 else []
    all_freqs = sorted(
        float(f) for f in (*freq_l, *freq_r)
        if isinstance(f, (int, float)) and np.isfinite(float(f)) and 10.0 < float(f) < 350.0
    )
    if len(all_freqs) < 2:
        return {}

    modal_floor = all_freqs[0]
    modal_ceiling = all_freqs[-1]

    bounds: dict = {}

    mag_c_min_lo = max(float(AUTO_MODE_MAG_C_MIN_MIN_HZ), modal_floor * 0.5)
    mag_c_min_hi = min(float(AUTO_MODE_MAG_C_MIN_MAX_HZ), modal_ceiling * 0.7)
    if mag_c_min_hi - mag_c_min_lo >= _MIN_ADAPTIVE_WINDOW:
        bounds["mag_c_min_lo"] = round(mag_c_min_lo, 2)
        bounds["mag_c_min_hi"] = round(mag_c_min_hi, 2)

    low_bass_lo = max(float(AUTO_MODE_LOW_BASS_MIN_HZ), modal_floor * 0.4)
    low_bass_hi = min(float(AUTO_MODE_LOW_BASS_MAX_HZ), modal_floor * 1.2)
    if low_bass_hi - low_bass_lo >= _MIN_ADAPTIVE_WINDOW:
        bounds["low_bass_lo"] = round(low_bass_lo, 2)
        bounds["low_bass_hi"] = round(low_bass_hi, 2)

    bass_first_hi = min(float(_BASS_FIRST_MODE_MAX_HZ), modal_ceiling * 1.3)
    if bass_first_hi - float(_BASS_FIRST_MODE_MIN_HZ) >= _MIN_ADAPTIVE_WINDOW:
        bounds["bass_first_hi"] = round(bass_first_hi, 2)

    conf_pull_hi = min(float(_CONF_PULL_MAX_MAX_HZ), modal_ceiling * 1.8)
    if conf_pull_hi - float(_CONF_PULL_MAX_MIN_HZ) >= _MIN_ADAPTIVE_WINDOW:
        bounds["conf_pull_hi"] = round(conf_pull_hi, 2)

    # Schroeder-based mag_c_max upper bound: the Schroeder frequency separates
    # the modal region (correctable with FIR) from the diffuse field (unreliable).
    # Allow mag_c_max search up to 1.4× f_Schroeder so the optimizer explores
    # the full modal band, but doesn't waste trials far above it.
    rt60_vals = [
        v for v in (
            base_data.get("measured_rt60_l"),
            base_data.get("measured_rt60_r"),
        )
        if v is not None
    ]
    if rt60_vals:
        rt60_repr = float(np.median(np.asarray(rt60_vals, dtype=float)))
        fs_hz = estimate_schroeder_hz(rt60_repr)
        if fs_hz is not None:
            mag_c_max_hi = float(np.clip(fs_hz * 1.4, float(AUTO_MODE_MAG_C_MAX_MIN_HZ) + 10.0, 400.0))
            bounds["mag_c_max_hi"] = round(mag_c_max_hi, 1)
            bounds["schroeder_hz_estimate"] = round(float(fs_hz), 1)

    return bounds


def _mag_low_search_bounds(base_data: dict) -> tuple[float, float, float, float]:
    adaptive = _derive_adaptive_freq_bounds(base_data)
    mag_lo = float(adaptive.get("mag_c_min_lo", AUTO_MODE_MAG_C_MIN_MIN_HZ))
    mag_hi = float(adaptive.get("mag_c_min_hi", AUTO_MODE_MAG_C_MIN_MAX_HZ))
    low_lo = float(adaptive.get("low_bass_lo", AUTO_MODE_LOW_BASS_MIN_HZ))
    low_hi = float(adaptive.get("low_bass_hi", AUTO_MODE_LOW_BASS_MAX_HZ))
    return mag_lo, mag_hi, low_lo, low_hi


# ---------------------------------------------------------------------------
# Bass Integration helpers
# ---------------------------------------------------------------------------

def _bi_search_enabled(base_data: dict) -> bool:
    return bool((base_data or {}).get("bass_integration_enabled", False))


def _build_bi_random_params(base_data: dict, rng) -> dict:
    """Generate randomized Bass Integration search parameters for builtin candidates."""
    xo_seed = float(np.clip(_auto_safe_float(base_data.get("avr_crossover_hz", 80.0), 80.0), 50.0, 160.0))
    xo = round(float(np.clip(rng.normal(loc=xo_seed, scale=18.0), 50.0, 160.0)), 1)
    delay_seed = float(np.clip(_auto_safe_float(base_data.get("bass_integration_sub_delay_ms", 0.0), 0.0), -40.0, 40.0))
    delay = round(float(np.clip(rng.normal(loc=delay_seed, scale=6.0), delay_seed - 10.0, delay_seed + 10.0)), 2)
    polarity = bool(rng.random() < 0.5)
    gain_seed = float(np.clip(_auto_safe_float(base_data.get("bass_integration_sub_gain_trim_db", 0.0), 0.0), -9.0, 3.0))
    gain = round(float(np.clip(rng.normal(loc=gain_seed, scale=2.5), -9.0, 3.0)), 2)
    out = {
        "avr_crossover_hz": float(xo),
        "bass_integration_sub_delay_ms": float(delay),
        "bass_integration_sub_polarity_invert": bool(polarity),
        "bass_integration_sub_gain_trim_db": float(gain),
    }
    lpf_min = round(float(xo), 1)
    lpf_seed = float(np.clip(_auto_safe_float(base_data.get("direct_dac_sub_lpf_hz", lpf_min), lpf_min), lpf_min, 200.0))
    lpf = round(float(np.clip(rng.normal(loc=lpf_seed, scale=12.0), lpf_min, 200.0)), 1)
    out["direct_dac_sub_lpf_hz"] = float(lpf)
    if bool(base_data.get("bass_integration_allpass_auto_applied", False)):
        ap_freq_seed = float(np.clip(_auto_safe_float(base_data.get("bass_integration_allpass_freq_hz", 80.0), 80.0), 40.0, 200.0))
        ap_freq = round(float(np.clip(rng.normal(loc=ap_freq_seed, scale=15.0), 40.0, 200.0)), 1)
        ap_q_seed = float(np.clip(_auto_safe_float(base_data.get("bass_integration_allpass_q", 0.707), 0.707), 0.5, 2.0))
        ap_q = round(float(np.clip(rng.normal(loc=ap_q_seed, scale=0.2), 0.5, 2.0)), 3)
        out["bass_integration_allpass_freq_hz"] = float(ap_freq)
        out["bass_integration_allpass_q"] = float(ap_q)
    return out


def _suggest_bi_optuna_params(base_data: dict, trial, *, coarse: bool = True, center: dict | None = None) -> dict:
    """Suggest Bass Integration parameters via Optuna trial."""
    c = dict(base_data or {})
    if center:
        c.update(dict(center))
    xo_seed = float(np.clip(_auto_safe_float(c.get("avr_crossover_hz", 80.0), 80.0), 50.0, 160.0))
    delay_seed = float(np.clip(_auto_safe_float(c.get("bass_integration_sub_delay_ms", 0.0), 0.0), -40.0, 40.0))
    gain_seed = float(np.clip(_auto_safe_float(c.get("bass_integration_sub_gain_trim_db", 0.0), 0.0), -9.0, 3.0))
    if coarse:
        xo_lo, xo_hi = 50.0, 160.0
        delay_span = 10.0
        gain_span = 6.0
    else:
        xo_lo = float(np.clip(xo_seed - 15.0, 50.0, 160.0))
        xo_hi = float(np.clip(xo_seed + 15.0, 50.0, 160.0))
        delay_span = 3.0
        gain_span = 3.0
    xo = round(float(trial.suggest_float("bi_avr_crossover_hz", float(xo_lo), float(xo_hi), step=1.0)), 1)
    delay_lo = float(np.clip(delay_seed - delay_span, -40.0, 40.0))
    delay_hi = float(np.clip(delay_seed + delay_span, -40.0, 40.0))
    if delay_hi <= delay_lo:
        delay_hi = delay_lo + 0.1
    delay = round(float(trial.suggest_float("bi_sub_delay_ms", float(delay_lo), float(delay_hi), step=0.1)), 2)
    polarity = bool(trial.suggest_categorical("bi_sub_polarity", [False, True]))
    gain_lo = float(np.clip(gain_seed - gain_span, -9.0, 3.0))
    gain_hi = float(np.clip(gain_seed + gain_span, -9.0, 3.0))
    if gain_hi <= gain_lo:
        gain_hi = gain_lo + 0.1
    gain = round(float(trial.suggest_float("bi_sub_gain_trim_db", float(gain_lo), float(gain_hi), step=0.1)), 2)
    out = {
        "avr_crossover_hz": float(xo),
        "bass_integration_sub_delay_ms": float(delay),
        "bass_integration_sub_polarity_invert": bool(polarity),
        "bass_integration_sub_gain_trim_db": float(gain),
    }
    lpf_min = round(float(xo), 1)
    lpf_seed = float(np.clip(_auto_safe_float(c.get("direct_dac_sub_lpf_hz", lpf_min), lpf_min), lpf_min, 200.0))
    lpf_lo = lpf_min
    lpf_hi = float(np.clip(lpf_seed + (20.0 if coarse else 8.0), lpf_min, 200.0))
    if lpf_hi <= lpf_lo:
        lpf_hi = lpf_lo + 0.1
    lpf = round(float(trial.suggest_float("bi_direct_dac_sub_lpf_hz", float(lpf_lo), float(lpf_hi), step=1.0)), 1)
    out["direct_dac_sub_lpf_hz"] = float(max(lpf_min, float(lpf)))
    if bool(c.get("bass_integration_allpass_auto_applied", False)):
        ap_seed = float(np.clip(_auto_safe_float(c.get("bass_integration_allpass_freq_hz", 80.0), 80.0), 40.0, 200.0))
        ap_span = 20.0 if coarse else 8.0
        ap_lo = float(np.clip(ap_seed - ap_span, 40.0, 200.0))
        ap_hi = float(np.clip(ap_seed + ap_span, 40.0, 200.0))
        if ap_hi <= ap_lo:
            ap_hi = ap_lo + 0.1
        ap_freq = round(float(trial.suggest_float("bi_allpass_freq_hz", float(ap_lo), float(ap_hi), step=1.0)), 1)
        q_seed = float(np.clip(_auto_safe_float(c.get("bass_integration_allpass_q", 0.707), 0.707), 0.5, 2.0))
        q_span = 0.3 if coarse else 0.15
        q_lo = float(np.clip(q_seed - q_span, 0.5, 2.0))
        q_hi = float(np.clip(q_seed + q_span, 0.5, 2.0))
        if q_hi <= q_lo:
            q_hi = q_lo + 0.01
        ap_q = round(float(trial.suggest_float("bi_allpass_q", float(q_lo), float(q_hi), step=0.01)), 3)
        out["bass_integration_allpass_freq_hz"] = float(ap_freq)
        out["bass_integration_allpass_q"] = float(ap_q)
    return out


def _seed_bi_optuna_params(base_data: dict, p: dict) -> dict:
    """Project Bass Integration preset values to Optuna unit params for seeding."""
    c = dict(base_data or {})
    c.update(dict(p or {}))
    xo_seed = float(np.clip(_auto_safe_float(c.get("avr_crossover_hz", 80.0), 80.0), 50.0, 160.0))
    delay_seed = float(np.clip(_auto_safe_float(c.get("bass_integration_sub_delay_ms", 0.0), 0.0), -40.0, 40.0))
    gain_seed = float(np.clip(_auto_safe_float(c.get("bass_integration_sub_gain_trim_db", 0.0), 0.0), -9.0, 3.0))
    out = {
        "bi_avr_crossover_hz": float(xo_seed),
        "bi_sub_delay_ms": float(delay_seed),
        "bi_sub_polarity": bool(c.get("bass_integration_sub_polarity_invert", False)),
        "bi_sub_gain_trim_db": float(gain_seed),
    }
    lpf_seed = float(np.clip(_auto_safe_float(c.get("direct_dac_sub_lpf_hz", xo_seed), xo_seed), xo_seed, 200.0))
    out["bi_direct_dac_sub_lpf_hz"] = float(lpf_seed)
    if bool(c.get("bass_integration_allpass_auto_applied", False)):
        out["bi_allpass_freq_hz"] = float(np.clip(_auto_safe_float(c.get("bass_integration_allpass_freq_hz", 80.0), 80.0), 40.0, 200.0))
        out["bi_allpass_q"] = float(np.clip(_auto_safe_float(c.get("bass_integration_allpass_q", 0.707), 0.707), 0.5, 2.0))
    return out


# ---------------------------------------------------------------------------
# Builtin candidate builder
# ---------------------------------------------------------------------------

def _build_auto_mode_candidates(
    base_data: dict,
    *,
    n_trials: int,
    seed: int,
    optimize_mag_low: bool = True,
) -> list[dict]:
    rng = np.random.default_rng(int(seed))
    logger.info("candidate_generation: seed=%d, n_trials=%d", int(seed), int(n_trials))
    n_eff = max(1, int(n_trials))
    tune_mag_low = bool(optimize_mag_low)

    keep_tdc = bool(base_data.get("enable_tdc", True))
    keep_afdw = bool(base_data.get("enable_afdw", True))
    keep_bass_first = bool(base_data.get("bass_first_ai", True))
    prefer_bass = bool(_auto_goal_is_flat_family(_auto_goal(base_data)))
    bool_search = bool(_auto_phase1_bool_search_enabled(base_data))
    max_boost_hi = float(_auto_phase1_max_boost_hi(base_data))
    ft = str(base_data.get("filter_type", "") or "").strip().lower()
    is_mixed = "mixed" in ft
    is_phase_search = _auto_is_phase_search_filter(ft)
    mixed_center = _auto_safe_float(base_data.get("mixed_freq", 180.0), 180.0)
    if not np.isfinite(mixed_center) or mixed_center <= 0.0:
        mixed_center = 180.0
    phase_center = _auto_phase_limit_center(base_data.get("phase_limit"))

    adaptive = _derive_adaptive_freq_bounds(base_data)
    _r_mag_c_min_lo = float(adaptive.get("mag_c_min_lo", AUTO_MODE_MAG_C_MIN_MIN_HZ))
    _r_mag_c_min_hi = float(adaptive.get("mag_c_min_hi", AUTO_MODE_MAG_C_MIN_MAX_HZ))
    _r_low_bass_lo = float(adaptive.get("low_bass_lo", AUTO_MODE_LOW_BASS_MIN_HZ))
    _r_low_bass_hi = float(adaptive.get("low_bass_hi", AUTO_MODE_LOW_BASS_MAX_HZ))
    _r_bass_first_hi = float(adaptive.get("bass_first_hi", _BASS_FIRST_MODE_MAX_HZ))
    _r_conf_pull_hi = float(adaptive.get("conf_pull_hi", _CONF_PULL_MAX_MAX_HZ))

    mag_c_min_seed = float(_auto_mag_c_min_center(base_data, default=25.0))
    low_bass_cut_seed = float(
        np.clip(
            _auto_safe_float(base_data.get("low_bass_cut_hz", 40.0), 40.0),
            _r_low_bass_lo,
            _r_low_bass_hi,
        )
    )

    out_seed = {}
    if bool(keep_tdc):
        out_seed["tdc_strength"] = round(
            float(
                _clip(
                    _auto_safe_float(base_data.get("tdc_strength", 50.0), 50.0),
                    _TDC_STRENGTH_MIN,
                    _TDC_STRENGTH_MAX,
                )
            ),
            1,
        )
    out_seed["enable_tdc"] = bool(keep_tdc)
    out_seed["enable_afdw"] = bool(keep_afdw)
    out_seed["bass_first_ai"] = bool(keep_bass_first)
    if bool(prefer_bass):
        out_seed["max_boost"] = round(
            float(np.clip(max(6.0, _auto_safe_float(base_data.get("max_boost", 5.0), 5.0)), 0.1, max_boost_hi)),
            2,
        )
    if bool(is_phase_search):
        out_seed["phase_limit"] = round(float(phase_center), 1)
    out: list[dict] = [out_seed]
    for _ in range(max(0, n_eff - 1)):
        if bool(tune_mag_low):
            mag_c_min_cand, low_bass_cut_cand = _auto_sample_mag_low_pair(
                rng,
                mag_center=float(mag_c_min_seed),
                low_center=float(low_bass_cut_seed),
                mag_sigma=2.6,
                low_sigma=3.2,
            )
        else:
            mag_c_min_cand = float(round(mag_c_min_seed, 1))
            low_bass_cut_cand = float(round(low_bass_cut_seed, 1))
        cand = {
            "comparison_mode": True,
            "enable_tdc": bool(keep_tdc),
            "enable_afdw": _auto_candidate_bool_choice(rng, keep_afdw) if bool(bool_search) else bool(keep_afdw),
            "bass_first_ai": _auto_candidate_bool_choice(rng, keep_bass_first) if bool(bool_search) else bool(keep_bass_first),
            "fdw_cycles": round(float(rng.uniform(5.0, 16.0)), 2),
            "tdc_strength": round(float(rng.uniform(_TDC_STRENGTH_MIN, _TDC_STRENGTH_MAX)), 1),
            "tdc_max_reduction_db": round(float(rng.uniform(_TDC_MAX_REDUCTION_MIN_DB, _TDC_MAX_REDUCTION_MAX_DB)), 1),
            "tdc_slope_db_per_oct": float(rng.choice(_TDC_SLOPE_CHOICES)),
            "reg_strength": round(float(rng.uniform(15.0, 45.0)), 1),
            "max_slope_db_per_oct": float(rng.choice(_MAX_SLOPE_CHOICES)),
            "max_boost": round(float(rng.uniform(5.0 if prefer_bass else 3.0, max_boost_hi)), 2),
            "mag_c_min": float(mag_c_min_cand),
            "mag_c_max": round(float(rng.uniform(170.0, 300.0)), 1),
            "trans_width": round(float(rng.uniform(70.0, 150.0)), 1),
            "filter_smooth": 96,
            "bass_first_mode_max_hz": round(float(rng.uniform(_BASS_FIRST_MODE_MIN_HZ, _r_bass_first_hi)), 1),
            "conf_pull_max_hz": round(float(rng.uniform(_CONF_PULL_MAX_MIN_HZ, _r_conf_pull_hi)), 1),
            "low_bass_cut_hz": float(low_bass_cut_cand),
        }
        if is_mixed:
            cand["mixed_freq"] = round(float(np.clip(rng.normal(loc=mixed_center, scale=35.0), 80.0, 320.0)), 1)
        if is_phase_search:
            cand["phase_limit"] = round(
                float(_auto_phase1_sample_phase_limit(base_data, rng)),
                1,
            )
        if _bi_search_enabled(base_data):
            cand.update(_build_bi_random_params(base_data, rng))
        out.append(cand)
    return out


# ---------------------------------------------------------------------------
# Optuna coarse study builder
# ---------------------------------------------------------------------------

def _build_auto_mode_candidates_optuna(
    base_data: dict,
    *,
    n_trials: int,
    seed: int,
    startup_trials: int = AUTO_MODE_OPTUNA_PILOT_STARTUP_TRIALS,
    optimize_mag_low: bool = True,
) -> list[dict] | None:
    base_data = _auto_filter_normalized_base_data(base_data)
    try:
        import optuna  # type: ignore
    except (

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
    ):
        return None

    n_eff = max(1, int(n_trials))
    startup = int(max(1, min(int(startup_trials), int(n_eff))))
    logger.info("candidate_generation (optuna): seed=%d, n_trials=%d, startup_trials=%d", int(seed), n_eff, startup)
    sampler = optuna.samplers.TPESampler(
        seed=int(seed),
        n_startup_trials=int(startup),
        **_auto_optuna_sampler_kwargs(base_data, workers=1),
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)

    keep_tdc = bool(base_data.get("enable_tdc", True))
    keep_afdw = bool(base_data.get("enable_afdw", True))
    keep_bass_first = bool(base_data.get("bass_first_ai", True))
    ft = str(base_data.get("filter_type", "") or "").strip().lower()
    is_phase_search = _auto_is_phase_search_filter(ft)

    out_seed = {}
    if bool(keep_tdc):
        out_seed["tdc_strength"] = round(
            float(
                _clip(
                    _auto_safe_float(base_data.get("tdc_strength", 50.0), 50.0),
                    _TDC_STRENGTH_MIN,
                    _TDC_STRENGTH_MAX,
                )
            ),
            1,
        )
    out_seed["enable_tdc"] = bool(keep_tdc)
    out_seed["enable_afdw"] = bool(keep_afdw)
    out_seed["bass_first_ai"] = bool(keep_bass_first)
    if bool(is_phase_search):
        out_seed["phase_limit"] = round(float(_auto_phase_limit_center(base_data.get("phase_limit", None))), 1)

    out: list[dict] = [dict(out_seed)]

    # Import here to avoid circular dependency: candidate_optuna_coarse imports from candidate_base
    from .candidate_optuna_coarse import _suggest_auto_mode_candidate_optuna

    for _ in range(max(0, n_eff - 1)):
        tr = study.ask()
        cand = _suggest_auto_mode_candidate_optuna(
            base_data,
            tr,
            optimize_mag_low=bool(optimize_mag_low),
        )

        # Use a constant score so TPE sampler stays in exploration mode rather
        # than over-fitting toward default parameter values. Actual acoustic
        # evaluation happens downstream; Optuna here only provides parameter
        # diversity via its sampler.
        study.tell(tr, 0.0)

        out.append(cand)

    return out
