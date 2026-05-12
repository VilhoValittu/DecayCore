# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

import hashlib
import logging
import os
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("DecayCore")
MAX_SAFE_BOOST = 12.0
AUTO_MODE_TRIALS = 32
AUTO_MODE_REFINE_TRIALS = 16
AUTO_MODE_GOAL_DEFAULT = "balanced"
AUTO_MODE_GOAL_ROOM_SAFE = "room-safe"
AUTO_MODE_GOAL_LOW_RIPPLE = "low-ripple"
AUTO_MODE_GOAL_FLAT = "flat"
AUTO_MODE_GOAL_SUBWOOFERS = "subwoofers"
AUTO_MODE_GOAL_ACOUSTIC = "acoustic"
AUTO_MODE_GOAL_HYBRID = "hybrid"
AUTO_MODE_SUBWOOFERS_LEVEL_MIN_HZ = 20.0
AUTO_MODE_SUBWOOFERS_LEVEL_MAX_HZ = 200.0
AUTO_MODE_CACHE_ENABLED = True
AUTO_MODE_CACHE_MAX_ITEMS = 64
# Bump only when AUTO mode cache/Optuna persistence compatibility changes.
AUTO_MODE_CACHE_SCHEMA_VERSION = 9
AUTO_MODE_COMPAT_VERSION = "am20"
AUTO_MODE_CACHE_FILENAME = "decaycore_auto_mode_cache.json"
AUTO_MODE_CACHE_FILTER_KEYS = ("linear", "mixed", "minimum", "asym")
AUTO_MODE_PHASE1_PLATEAU_ROUNDS = 5
AUTO_MODE_PHASE2_PLATEAU_ROUNDS = 8
AUTO_MODE_TARGET_TOP_N = 3
AUTO_MODE_TARGET_TRIALS_PER_CURVE = 1
AUTO_MODE_TARGET_PRESELECT_SMOOTH_OCT = 1.00
AUTO_MODE_TARGET_TOP_N_MIN = 3
AUTO_MODE_TARGET_TOP_N_MAX = 6
AUTO_MODE_TARGET_TOP_N_SPREAD_DB = 0.35
AUTO_MODE_TARGET_PRESELECT_BOOST_W = 0.12
AUTO_MODE_TARGET_PRESELECT_SLOPE_W = 0.12
AUTO_MODE_TARGET_PRESELECT_ASYM_W = 0.18
AUTO_MODE_TARGET_PRESELECT_MODE_W = 0.10
AUTO_MODE_TARGET_PRESELECT_BASS_SHAPE_W = 0.22
AUTO_MODE_TARGET_PRESELECT_TREBLE_SHAPE_W = 0.20
AUTO_MODE_TARGET_PRESELECT_MAX_BASS_BOOST_REF_DB = 8.0
AUTO_MODE_TARGET_PRESELECT_MODE_BAND_MIN_HZ = 25.0
AUTO_MODE_TARGET_PRESELECT_MODE_BAND_MAX_HZ = 160.0
AUTO_MODE_TARGET_CACHE_AS_WILDCARD = True
AUTO_MODE_TARGET_PREFER_MILDER_STEP = False
AUTO_MODE_TARGET_MILDER_MAX_RANK_DROP = 1.50
AUTO_MODE_TARGET_MILDER_MAX_FIT_RMS_ADD_DB = 0.25
AUTO_MODE_TARGET_MILDER_MAX_DIFFICULTY_ADD = 0.20
AUTO_MODE_TARGET_MILDER_MAX_ASYM_ADD = 0.15
AUTO_MODE_TARGET_MILDER_MAX_AVG_SCORE_DROP_ACOUSTIC = 0.8
AUTO_MODE_TARGET_BEST_RANK_TIE_EPS = 0.05
AUTO_MODE_TARGET_BASS_FORWARD_MAX_RANK_DROP = 1.25
AUTO_MODE_SYNTH_TARGET_ENABLED = True
AUTO_MODE_SYNTH_TARGET_NAME = "Adaptive"
AUTO_MODE_SYNTH_TARGET_BASS_COMP_FRAC = 0.50  # basson määrä adaptiivisessa käyrässä
AUTO_MODE_SYNTH_TARGET_BASS_COMP_REF_DB = 8.0
AUTO_MODE_SYNTH_TARGET_TILT_COMP_FRAC = 0.30  # legacy / fallback default
AUTO_MODE_SYNTH_TARGET_TILT_COMP_FRAC_DEFAULT = (
    0.30  # used when preset omits synth_tilt_frac
)
AUTO_MODE_SYNTH_TARGET_HF_COMP_FRAC = 0.50
AUTO_MODE_SYNTH_TARGET_SMOOTH_OCT = 1.0 / 3.0
AUTO_MODE_LOCAL_REFINE_ENABLED = True
AUTO_MODE_LOCAL_REFINE_TOP_K = 1
AUTO_MODE_LOCAL_REFINE_TRIALS_PER_TOP = 2
AUTO_MODE_LOCAL_REFINE_SHRINK = 0.35
AUTO_MODE_LOCAL_REFINE_KEEP_BEST_PHASE1 = True
AUTO_MODE_LOCAL_REFINEMENT_TOP_N = 3
AUTO_MODE_LOCAL_REFINEMENT_PER_ANCHOR = 16
AUTO_MODE_LOCAL_REFINEMENT_SHRINK = 0.35
AUTO_MODE_REFINE_TIEBREAK_ENABLE = True
AUTO_MODE_REFINE_TIEBREAK_RANK_EPS = 0.20
AUTO_MODE_REFINE_TIEBREAK_RIPPLE_EPS = 0.02
AUTO_MODE_REFINE_TIEBREAK_PHASE_EPS = 0.10
AUTO_MODE_REFINE_TIEBREAK_TRACKING_EPS = 0.05
AUTO_MODE_REFINE_MODE_SOFT_K = 0.25
AUTO_MODE_REFINE_MODE_BOOST_GUARD_MIN_RIPPLE_GAIN_DB = 0.06
AUTO_MODE_PARALLEL_ENABLED = True
AUTO_MODE_PARALLEL_MIN_TRIALS = 4
AUTO_MODE_PARALLEL_MAX_WORKERS = 0
AUTO_MODE_PARALLEL_BATCH_MULTIPLIER = 2
AUTO_MODE_OPTUNA_PILOT_ENABLED = True
AUTO_MODE_OPTUNA_PILOT_MIN_TRIALS = 24
AUTO_MODE_OPTUNA_PILOT_STARTUP_TRIALS = 16
AUTO_MODE_OPTUNA_STARTUP_PHASE1 = 16
AUTO_MODE_OPTUNA_STARTUP_TARGET = 4
AUTO_MODE_OPTUNA_STARTUP_LOCAL = 4
AUTO_MODE_OPTUNA_STARTUP_MICRO = 4
AUTO_MODE_OPTUNA_CONSTRAINTS_ENABLED = True
AUTO_MODE_OPTUNA_CONSTRAINTS_REFINE_ONLY = True
AUTO_MODE_OPTUNA_CONSTRAINTS_MAX_MODE_RIPPLE_DB = 0.20
AUTO_MODE_OPTUNA_CONSTRAINTS_MAX_EVENTS_SEVERITY = 1.10
AUTO_MODE_OPTUNA_CONSTRAINTS_MAX_NET_BOOST_DB = 12.0
AUTO_MODE_OPTUNA_CONSTRAINTS_USE_EVENTS_IN_REFINE = False
AUTO_MODE_OPTUNA_TELEMETRY = True
AUTO_MODE_OPTUNA_TELEMETRY_LOG_SUMMARY = True
AUTO_MODE_OPTUNA_CONSTRAINTS_ZERO_FEASIBLE_FALLBACK = True
AUTO_MODE_OPTUNA_MULTIVARIATE = True
AUTO_MODE_OPTUNA_GROUP = False
AUTO_MODE_OPTUNA_CONSTANT_LIAR = True
AUTO_MODE_OPTUNA_PERSISTENT_STUDY = True
AUTO_MODE_OPTUNA_AVOID_DUPLICATES = True
AUTO_MODE_OPTUNA_CROSS_STUDY_SEEDS = True
AUTO_MODE_OPTUNA_CROSS_STUDY_SEEDS_TOP_N = 8
AUTO_MODE_OPTUNA_PRUNING_ENABLED = True
AUTO_MODE_OPTUNA_PRUNING_N_STARTUP = 8
AUTO_MODE_OPTUNA_STORAGE_FILENAME = "decaycore_auto_mode_optuna_journal.log"
AUTO_MODE_OPTUNA_DUPLICATE_MAX_ATTEMPTS = 24
AUTO_MODE_OPTUNA_USER_ATTR_OUT = "camillafir_out"
AUTO_MODE_DUAL_MODE_ENABLED = True
AUTO_MODE_DUAL_MODE_TOP_N = 2
AUTO_MODE_MODE_RIPPLE_OK_DB = 0.050
AUTO_MODE_MODE_RIPPLE_PENALTY_W = 18.0
AUTO_MODE_MODE_RIPPLE_SECONDARY_W = 0.85
AUTO_MODE_PHASE2_PARETO_POOL_MIN = 6
AUTO_MODE_PHASE2_PARETO_POOL_MAX = 15
AUTO_MODE_PHASE2_PARETO_RANK_WINDOW = 2.0
AUTO_MODE_PHASE2_PARETO_ACOUSTIC_DROP = 0.35
AUTO_MODE_PHASE2_PARETO_PREPOST_EPS = 0.002
AUTO_MODE_PHASE2_PARETO_MODE_RIPPLE_EPS = 0.005
AUTO_MODE_PHASE2_PARETO_RMS20_200_EPS = 0.003
AUTO_MODE_PHASE2_PARETO_BOOST_EPS = 0.02
AUTO_MODE_PHASE2_PARETO_IACC_EPS = 0.015
AUTO_MODE_RESIDUAL_PEAK_MAX_HZ = 400.0
AUTO_MODE_RESIDUAL_PEAK_THRESHOLD_DB = 3.0
AUTO_MODE_RESIDUAL_PEAK_HARD_GATE_DB = 6.0
AUTO_MODE_RESIDUAL_PEAK_PENALTY_CAP = 20.0
AUTO_MODE_MAX_AVG_SCORE_LOSS_FOR_SAFETY_OVERRIDE = 0.10
AUTO_MODE_TDC_DECAY_PENALTY_CAP = 8.0
AUTO_MODE_TDC_DECAY_PENALTY_WEIGHT = 1.0
AUTO_MODE_TDC_EXTREME_PEAK_REDUCTION_DB = 10.0
AUTO_MODE_TDC_LOW_NEED_THRESHOLD = 0.15
AUTO_MODE_TDC_RT60_TARGET_LOW_S = 0.45
AUTO_MODE_TDC_RT60_TARGET_UPPER_S = 0.38
AUTO_MODE_TDC_RT60_LOW_MAX_HZ = 160.0
AUTO_MODE_TDC_RT60_EVAL_MAX_HZ = 300.0
AUTO_MODE_RESIDUAL_PEAK_WINNER_POLISH_ENABLED = True
AUTO_MODE_RESIDUAL_PEAK_WINNER_POLISH_MAX_VARIANTS = 8
AUTO_MODE_RESIDUAL_PEAK_WINNER_POLISH_MIN_IMPROVEMENT_DB = 0.75
AUTO_MODE_PHASE2_HARD_GATE_ENABLED = True
AUTO_MODE_PHASE2_HARD_GATE_MIN_KEEP = 8
AUTO_MODE_PHASE2_HARD_GATE_KEEP_EVENT_FRACTION = 0.70
AUTO_MODE_PHASE2_HARD_GATE_KEEP_RIPPLE_FRACTION = 0.75
AUTO_MODE_PHASE2_HARD_GATE_KEEP_PEAK_FRACTION = 0.70
AUTO_MODE_PHASE2_HARD_GATE_ABS_MAX_PEAK_DB = AUTO_MODE_RESIDUAL_PEAK_HARD_GATE_DB
AUTO_MODE_PHASE2_HARD_GATE_FALLBACK_TO_RANK = True
AUTO_MODE_ADAPTIVE_SHRINK_ENABLED = True
AUTO_MODE_ADAPTIVE_SHRINK_MIN = 0.20
AUTO_MODE_ADAPTIVE_SHRINK_MAX = 0.55
AUTO_MODE_PHASE3_MICRO_ENABLED = True
AUTO_MODE_PHASE3_MICRO_TRIALS = 8
AUTO_MODE_HYBRID_MIXED_FREQ_SOFT_MAX_HZ = 110.0
AUTO_MODE_HYBRID_MIXED_FREQ_SOFT_DEN_HZ = 40.0
AUTO_MODE_HYBRID_LOCAL_TOP_N = 2
AUTO_MODE_HYBRID_LOCAL_PER_ANCHOR = 10
# Rank score calibration (UI/report scale only).
AUTO_MODE_RANK_SCORE_GAIN = 0.95
AUTO_MODE_RANK_SCORE_BIAS = 3.0
AUTO_MODE_EVENT_PEN_CONF_GATE_ENABLE = True
AUTO_MODE_EVENT_PEN_CONF_GATE_MIN_SCALE = 0.30
AUTO_MODE_EVENT_PEN_CONF_GATE_FULL_CONF = 0.80
AUTO_MODE_EVENT_PEN_BASE_PER_EVENT = 0.35
AUTO_MODE_EVENT_PEN_DT_WEIGHT = 0.015
AUTO_MODE_EVENT_PEN_DT_POWER = 2.0
AUTO_MODE_EVENT_PEN_DT_REF_MS = 100.0
AUTO_MODE_CORRECTION_RANGE_SCORING = True
AUTO_MODE_MAG_C_MIN_MIN_HZ = 15.0
AUTO_MODE_MAG_C_MIN_MAX_HZ = 70.0
AUTO_MODE_MAG_C_MAX_MIN_HZ = 170.0
AUTO_MODE_MAG_C_MIN_REF_MIN_HZ = 80.0
AUTO_MODE_MAG_C_MIN_REF_MAX_HZ = 200.0
AUTO_MODE_MAG_C_MIN_SEARCH_MAX_HZ = 80.0
AUTO_MODE_MAG_C_MIN_SMOOTH_OCT = 1.0
AUTO_MODE_LOW_BASS_FROM_F6_ADD_HZ = 2.0
AUTO_MODE_LOW_BASS_MIN_HZ = 18.0
AUTO_MODE_LOW_BASS_MAX_HZ = 55.0
AUTO_MODE_EXC_FROM_F6_ADD_HZ = 8.0
AUTO_MODE_EXC_MIN_HZ = 20.0
AUTO_MODE_EXC_MAX_HZ = 80.0
AUTO_MODE_PHASE_LIMIT_MIN_HZ = 100.0
AUTO_MODE_PHASE_LIMIT_MAX_HZ = 500.0
AUTO_MODE_PHASE_LIMIT_SIGMA_HZ = 20.0
AUTO_MODE_PHASE_LIMIT_LOCAL_SIGMA_HZ = 30.0
AUTO_MODE_PHASE_LIMIT_DEFAULT_HZ = 380.0
AUTO_MODE_PHASE_LIMIT_PRIOR_CENTER_HZ = 300.0
AUTO_MODE_PHASE_LIMIT_PRIOR_TOL_HZ = 90.0
AUTO_MODE_PHASE_LIMIT_PRIOR_SPAN_HZ = 70.0
AUTO_MODE_PHASE_LIMIT_PRIOR_WEIGHT = 0.0
AUTO_MODE_PHASE_LIMIT_PRIOR_MAX_PEN = 4.0
AUTO_MODE_PHASE_LIMIT_EXPLORE_GLOBAL_FRAC = 0.35
AUTO_MODE_PHASE_LIMIT_EXPLORE_UNIFORM_FRAC = 0.20
AUTO_MODE_PHASE_LIMIT_EXPLORE_GLOBAL_SIGMA_HZ = 90.0
AUTO_MODE_HPF_MIN_HZ = 14.0
AUTO_MODE_HPF_MAX_HZ = 140.0
AUTO_MODE_HPF_REF_MIN_HZ = 90.0
AUTO_MODE_HPF_REF_MAX_HZ = 260.0
AUTO_MODE_HPF_SEARCH_MAX_HZ = 180.0
AUTO_MODE_HPF_SMOOTH_OCT = 1.00
AUTO_MODE_HPF_AUTO_ENABLE_MIN_CONF = 0.45
AUTO_MODE_HPF_ALLOWED_SLOPES_DB_OCT = (6, 12, 18, 24, 30, 36, 42, 48, 54)
AUTO_MODE_HPF_WINNER_POLISH_ENABLED = True
AUTO_MODE_BASS_INTEGRATION_GUARD_LO_RATIO = 0.60
AUTO_MODE_BASS_INTEGRATION_GUARD_HI_RATIO = 1.40
AUTO_MODE_BASS_INTEGRATION_PROFILE_WEIGHTS = {
    "safe": {
        "cancellation": 6.0,
        "overlap_ripple": 1.4,
        "anti_null_boost": 1.0,
        "sub_dominance": 0.7,
        "xo_gd_continuity": 0.6,
        "main_activity": 4.5,
    },
    "normal": {
        "cancellation": 4.2,
        "overlap_ripple": 0.9,
        "anti_null_boost": 0.65,
        "sub_dominance": 0.45,
        "xo_gd_continuity": 0.45,
        "main_activity": 3.0,
    },
    "assertive": {
        "cancellation": 3.0,
        "overlap_ripple": 0.7,
        "anti_null_boost": 0.4,
        "sub_dominance": 0.25,
        "xo_gd_continuity": 0.3,
        "main_activity": 1.5,
    },
}
AUTO_MODE_BUILTIN_TARGETS = (
    "Harman6",
    "Harman8",
    "Harman4",
    "Harman10",
    "Harman12",
    "Studio",
    "Nearfield",
    "HiFi",
    "Speech",
    "Toole",
    "BK_Light",
    "BK_Medium",
    "BK_Strong",
    "Flat",
    "Cinema",
)
AUTO_MODE_BUILTIN_TARGET_LOOKUP = {
    str(name).strip().lower(): str(name).strip() for name in AUTO_MODE_BUILTIN_TARGETS
}

def _auto_filter_cache_key(
    base_data: dict | None = None, *, filter_type: str | None = None
) -> str:
    ft = (
        str(
            filter_type
            if filter_type is not None
            else (base_data or {}).get("filter_type", "") or ""
        )
        .strip()
        .lower()
    )
    if ft in AUTO_MODE_CACHE_FILTER_KEYS:
        return str(ft)
    if "asym" in ft:
        return "asym"
    if "mixed" in ft:
        return "mixed"
    if "minimum" in ft or "minphase" in ft or ("min" in ft and "phase" in ft):
        return "minimum"
    if "linear" in ft:
        return "linear"
    return "mixed"


def _auto_filter_type_for_key(filter_key: str | None) -> str:
    fk = str(_auto_filter_cache_key(filter_type=str(filter_key or "")))
    if fk == "asym":
        return "Asymmetric"
    if fk == "linear":
        return "Linear Phase"
    if fk == "minimum":
        return "Minimum Phase"
    return "Mixed"

def _auto_hash_array(a: np.ndarray, *, decimals: int = 4, max_len: int = 1200) -> str:
    try:
        x = np.asarray(a, dtype=float).reshape(-1)
    except Exception:
        return ""
    if x.size <= 0:
        return ""
    m = np.isfinite(x)
    x = x[m]
    if x.size <= 0:
        return ""
    if x.size > int(max_len):
        idx = np.linspace(0, x.size - 1, int(max_len)).astype(int)
        x = x[idx]
    x = np.round(x, int(decimals))
    b = x.astype(np.float32).tobytes()
    return hashlib.sha256(b).hexdigest()

def _auto_hash_array_full(a: np.ndarray) -> str:
    try:
        raw = np.asarray(a)
    except Exception:
        return ""
    if raw.size <= 0:
        return ""
    try:
        x = raw.reshape(-1)
        mask = np.isfinite(x)
        x = x[mask]
    except Exception:
        return ""
    if x.size <= 0:
        return ""
    h = hashlib.sha256()
    if np.iscomplexobj(x):
        canonical = np.ascontiguousarray(x.astype(np.dtype("<c16"), copy=False))
        dtype_name = "complex128"
    else:
        try:
            canonical = np.ascontiguousarray(x.astype(np.dtype("<f8"), copy=False))
        except Exception:
            return ""
        dtype_name = "float64"
    h.update(str(dtype_name).encode("ascii", "ignore"))
    h.update(str(tuple(raw.shape)).encode("ascii", "ignore"))
    h.update(str(int(canonical.size)).encode("ascii", "ignore"))
    h.update(canonical.tobytes())
    return h.hexdigest()


__all__ = ['_auto_filter_cache_key', '_auto_hash_array', '_auto_hash_array_full']


def _load_sibling_symbols() -> None:
    import importlib
    package = __package__
    for module_name in ['cache_hash', 'goal_profile', 'safe_values', 'backend', 'config', 'phase_sampling']:
        if module_name == __name__.rsplit('.', 1)[-1]:
            continue
        module = importlib.import_module(f"{package}.{module_name}")
        for symbol in getattr(module, "__all__", ()):
            globals().setdefault(symbol, getattr(module, symbol))


_load_sibling_symbols()
