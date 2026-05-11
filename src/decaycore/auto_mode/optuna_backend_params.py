# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Optuna backend — parameter handling, distributions, and trial building."""

from __future__ import annotations

import hashlib
import json
import logging

import numpy as np

from .cache_signature import _auto_compat_version
from .shared import (
    AUTO_MODE_LOW_BASS_MAX_HZ,
    AUTO_MODE_LOW_BASS_MIN_HZ,
    AUTO_MODE_MAG_C_MAX_MIN_HZ,
    AUTO_MODE_MAG_C_MIN_MAX_HZ,
    AUTO_MODE_MAG_C_MIN_MIN_HZ,
    AUTO_MODE_OPTUNA_USER_ATTR_OUT,
    AUTO_MODE_PHASE_LIMIT_MAX_HZ,
    AUTO_MODE_PHASE_LIMIT_MIN_HZ,
    _auto_output_tilt_bounds,
    _auto_safe_float,
)
from .optuna_backend_storage import (
    _OPTUNA_CROSS_STUDY_BEST_PARAMS,
    _auto_optuna_create_storage,
    _auto_optuna_note_trial_scan,
)

logger = logging.getLogger("DecayCore")


def _auto_optuna_cross_study_best_params(
    optuna_mod,
    *,
    base_data: dict | None,
    scope: str,
    current_study_name: str,
    top_n: int = 8,
) -> list[dict]:
    """Return top param dicts from sibling studies (same scope, different sig)."""
    cache_key = _auto_optuna_param_signature(
        {
            "compat": _auto_compat_version(base_data),
            "scope": str(scope or ""),
            "current": str(current_study_name or ""),
            "top_n": int(top_n),
        }
    )
    cached = _OPTUNA_CROSS_STUDY_BEST_PARAMS.get(cache_key)
    if isinstance(cached, list):
        return [dict(p) for p in cached if isinstance(p, dict)]
    storage = _auto_optuna_create_storage(optuna_mod, base_data=base_data)
    if storage is None:
        return []
    get_summaries = getattr(optuna_mod, "get_all_study_summaries", None)
    if not callable(get_summaries):
        return []
    try:
        summaries = get_summaries(storage=storage)
    except Exception:
        return []
    seen_sigs: set[str] = set()
    results: list[tuple[float, dict]] = []
    sibling_summaries = []
    for summary in list(summaries or []):
        sname = str(getattr(summary, "study_name", "") or "")
        if sname == current_study_name:
            continue
        if not scope or scope not in sname:
            continue
        sibling_summaries.append(summary)

    def _summary_value(summary) -> float:
        try:
            best_trial = getattr(summary, "best_trial", None)
            val = getattr(best_trial, "value", None)
            return float(val) if val is not None and np.isfinite(float(val)) else float("-inf")
        except Exception:
            return float("-inf")

    sibling_summaries.sort(key=_summary_value, reverse=True)
    max_sibling_loads = int(max(int(top_n) * 4, int(top_n), 1))
    for summary in sibling_summaries[:max_sibling_loads]:
        best_trial = getattr(summary, "best_trial", None)
        best_val = getattr(best_trial, "value", None)
        best_params = {}
        try:
            best_params = dict(getattr(best_trial, "params", {}) or {})
            best_val_f = float(best_val)
        except Exception:
            best_params = {}
            best_val_f = float("nan")
        if best_params and np.isfinite(best_val_f):
            sig = _auto_optuna_param_signature(best_params)
            if sig and sig not in seen_sigs:
                seen_sigs.add(sig)
                results.append((float(best_val_f), dict(best_params)))
                if len(results) >= int(top_n):
                    break
            continue

        sname = str(getattr(summary, "study_name", "") or "")
        try:
            s = optuna_mod.load_study(study_name=sname, storage=storage)
            trials = list(s.get_trials(deepcopy=False) or [])
        except Exception:
            continue
        _auto_optuna_note_trial_scan(len(trials))
        for tr in trials:
            val = getattr(tr, "value", None)
            try:
                vf = float(val)
            except Exception:
                vf = float("nan")
            if not np.isfinite(vf):
                continue
            try:
                params = dict(getattr(tr, "params", {}) or {})
            except Exception:
                params = {}
            if not params:
                continue
            sig = _auto_optuna_param_signature(params)
            if not sig or sig in seen_sigs:
                continue
            seen_sigs.add(sig)
            results.append((vf, params))
    results.sort(key=lambda kv: kv[0], reverse=True)
    out = [p for _, p in results[: int(top_n)]]
    _OPTUNA_CROSS_STUDY_BEST_PARAMS[cache_key] = [dict(p) for p in out]
    return out

def _auto_optuna_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _auto_optuna_jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_auto_optuna_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        if np.isnan(v):
            return "nan"
        if np.isposinf(v):
            return "inf"
        if np.isneginf(v):
            return "-inf"
        return round(v, 6)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    return str(value)

def _auto_optuna_scope_context_hash(
    *,
    center: dict | None = None,
    shrink: float | None = None,
    extra: dict | None = None,
) -> str:
    payload = {
        "center": _auto_optuna_jsonable(dict(center or {})),
        "shrink": None if shrink is None else round(float(shrink), 6),
        "extra": _auto_optuna_jsonable(dict(extra or {})),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8", "ignore")).hexdigest()[:12]

def _auto_optuna_scope_with_context(
    scope_base: str,
    *,
    center: dict | None = None,
    shrink: float | None = None,
    extra: dict | None = None,
) -> str:
    ctx = _auto_optuna_scope_context_hash(center=center, shrink=shrink, extra=extra)
    return f"{str(scope_base)}-{ctx}"

def _auto_optuna_param_signature(params: dict | None) -> str:
    if not isinstance(params, dict) or not params:
        return ""
    try:
        payload = json.dumps(_auto_optuna_jsonable(params), sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        payload = str(params)
    return hashlib.sha1(payload.encode("utf-8", "ignore")).hexdigest()

def _auto_optuna_trial_params(
    *,
    trial_obj,
    preset: dict | None,
    seed_to_params=None,
) -> dict:
    if callable(seed_to_params):
        try:
            params = dict(seed_to_params(dict(preset or {})) or {})
        except Exception:
            # Seed adapters are pluggable; keep the study alive even if one returns malformed data.
            logger.debug("Optuna seed_to_params adapter failed", exc_info=True)
            params = {}
        if params:
            return params
    try:
        params = dict(getattr(trial_obj, "params", {}) or {})
    except (TypeError, ValueError, AttributeError):
        params = {}
    if params:
        return params
    return dict(preset or {})

def _auto_optuna_trial_payload_preset(user_attrs: dict | None) -> dict:
    payload = dict((user_attrs or {}).get(AUTO_MODE_OPTUNA_USER_ATTR_OUT, {}) or {})
    preset = payload.get("trial_preset", None)
    if not isinstance(preset, dict) or not preset:
        preset = payload.get("preset", None)
    return dict(preset or {})

def _auto_optuna_tdc_min(base_data: dict | None) -> float:
    _ = base_data
    return 20.0


def _auto_optuna_adaptive_freq_bounds(base_data: dict | None) -> tuple[float, float, float, float]:
    data = dict(base_data or {}) if isinstance(base_data, dict) else {}
    min_window = 5.0
    raw_l = data.get("harmonic_freq_hz_l")
    raw_r = data.get("harmonic_freq_hz_r")
    try:
        freq_l = list(raw_l) if raw_l is not None and len(raw_l) > 0 else []
    except Exception:
        freq_l = []
    try:
        freq_r = list(raw_r) if raw_r is not None and len(raw_r) > 0 else []
    except Exception:
        freq_r = []
    all_freqs = sorted(
        float(f)
        for f in (*freq_l, *freq_r)
        if isinstance(f, (int, float)) and np.isfinite(float(f)) and 10.0 < float(f) < 350.0
    )
    mag_lo = float(AUTO_MODE_MAG_C_MIN_MIN_HZ)
    mag_hi = float(AUTO_MODE_MAG_C_MIN_MAX_HZ)
    low_lo = float(AUTO_MODE_LOW_BASS_MIN_HZ)
    low_hi = float(AUTO_MODE_LOW_BASS_MAX_HZ)
    if len(all_freqs) < 2:
        return mag_lo, mag_hi, low_lo, low_hi

    modal_floor = float(all_freqs[0])
    modal_ceiling = float(all_freqs[-1])
    cand_mag_lo = max(mag_lo, modal_floor * 0.5)
    cand_mag_hi = min(mag_hi, modal_ceiling * 0.7)
    if cand_mag_hi - cand_mag_lo >= min_window:
        mag_lo, mag_hi = round(cand_mag_lo, 2), round(cand_mag_hi, 2)
    cand_low_lo = max(low_lo, modal_floor * 0.4)
    cand_low_hi = min(low_hi, modal_floor * 1.2)
    if cand_low_hi - cand_low_lo >= min_window:
        low_lo, low_hi = round(cand_low_lo, 2), round(cand_low_hi, 2)
    return float(mag_lo), float(mag_hi), float(low_lo), float(low_hi)


def _auto_optuna_sanitize_enqueued_params(params: dict | None, *, base_data: dict | None) -> dict:
    out = dict(params or {})
    if not out:
        return out
    mag_lo, mag_hi, low_lo, low_hi = _auto_optuna_adaptive_freq_bounds(base_data)
    for key, lo, hi in (
        ("mag_c_min", mag_lo, mag_hi),
        ("low_bass_cut_hz", low_lo, low_hi),
    ):
        if key not in out:
            continue
        value = _auto_safe_float(out.get(key), float(lo))
        out[key] = float(np.clip(value, float(lo), float(hi)))
    return out


def _auto_optuna_trial_distributions(optuna_mod, *, params: dict | None, base_data: dict | None) -> dict | None:
    params_in = dict(params or {})
    if not params_in:
        return None
    dist_mod = getattr(optuna_mod, "distributions", None)
    float_dist = getattr(dist_mod, "FloatDistribution", None)
    cat_dist = getattr(dist_mod, "CategoricalDistribution", None)
    if not callable(float_dist) or not callable(cat_dist):
        return None

    tdc_min = float(_auto_optuna_tdc_min(base_data))
    categorical_choices = {
        "enable_afdw": [False, True],
        "bass_first_ai": [False, True],
        "tdc_slope_db_per_oct": [3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 24.0, 36.0],
        "max_slope_db_per_oct": [8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0],
        "max_slope_boost_db_per_oct": [0.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 24.0, 36.0],
        "max_slope_cut_db_per_oct": [0.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 24.0, 36.0],
        "filter_smooth": [96],

    }
    output_tilt_lo, output_tilt_hi = _auto_output_tilt_bounds(base_data)
    mag_lo, mag_hi, low_lo, low_hi = _auto_optuna_adaptive_freq_bounds(base_data)
    float_ranges = {
        "fdw_cycles": (5.0, 16.0, 0.01),
        "tdc_strength": (tdc_min, 75.0, 0.1),
        "tdc_max_reduction_db": (0.0, 18.0, 0.1),
        "reg_strength": (15.0, 45.0, 0.1),
        "max_boost": (3.0, 12.0, 0.01),
        "mag_c_min": (float(mag_lo), float(mag_hi), 0.1),
        "mag_c_max": (float(AUTO_MODE_MAG_C_MAX_MIN_HZ), 400.0, 0.1),
        "trans_width": (70.0, 150.0, 0.1),
        "bass_first_mode_max_hz": (120.0, 220.0, 0.1),
        "conf_pull_max_hz": (80.0, 220.0, 5.0),
        "low_bass_cut_hz": (float(low_lo), float(low_hi), 0.1),
        "mixed_freq": (80.0, 320.0, 0.1),
        "phase_limit": (float(AUTO_MODE_PHASE_LIMIT_MIN_HZ), float(AUTO_MODE_PHASE_LIMIT_MAX_HZ), 0.1),
        "output_tilt_db_per_oct": (float(output_tilt_lo), float(output_tilt_hi), 0.05),
        "synth_tilt_frac": (0.05, 0.55, 0.01),
    }
    # Parameters sampled on a log scale in _suggest_auto_mode_candidate_optuna;
    # distributions must match or enqueue_trial / add_trial will fail.
    log_params = {"mag_c_min", "low_bass_cut_hz", "phase_limit"}

    out: dict[str, object] = {}
    for name in list(params_in.keys()):
        key = str(name)
        if key.endswith("_u"):
            try:
                out[key] = float_dist(0.0, 1.0)
            except Exception:
                return None
            continue
        if key in categorical_choices:
            try:
                out[key] = cat_dist(list(categorical_choices[key]))
            except Exception:
                return None
            continue
        if key in float_ranges:
            lo, hi, _step = float_ranges[key]
            try:
                if key in log_params:
                    out[key] = float_dist(float(lo), float(hi), log=True)
                else:
                    out[key] = float_dist(float(lo), float(hi), step=float(_step))
            except Exception:
                return None
            continue
        return None
    return dict(out)

def _auto_optuna_build_completed_trial(
    optuna_mod,
    *,
    params: dict | None,
    value: float,
    user_attrs: dict | None,
    base_data: dict | None,
    system_attrs: dict | None = None,
):
    create_trial = getattr(getattr(optuna_mod, "trial", None), "create_trial", None)
    if not callable(create_trial):
        create_trial = getattr(optuna_mod, "create_trial", None)
    if not callable(create_trial):
        return None
    params = _auto_optuna_sanitize_enqueued_params(dict(params or {}), base_data=base_data)
    distributions = _auto_optuna_trial_distributions(optuna_mod, params=params, base_data=base_data)
    if not distributions:
        return None

    trial_kwargs = {
        "params": dict(params or {}),
        "distributions": dict(distributions),
        "value": float(value),
        "user_attrs": dict(user_attrs or {}),
    }
    if system_attrs:
        trial_kwargs["system_attrs"] = dict(system_attrs)
    try:
        return create_trial(**trial_kwargs)
    except TypeError:
        trial_kwargs.pop("system_attrs", None)
    except Exception:
        return None

    try:
        return create_trial(**trial_kwargs)
    except TypeError:
        pass
    except Exception:
        return None

    trial_state = getattr(getattr(optuna_mod, "trial", None), "TrialState", None)
    complete_state = getattr(trial_state, "COMPLETE", None) if trial_state is not None else None
    if complete_state is None:
        return None
    try:
        return create_trial(state=complete_state, **trial_kwargs)
    except Exception:
        return None
