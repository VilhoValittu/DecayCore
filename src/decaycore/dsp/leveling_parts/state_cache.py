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
import copy
from dataclasses import dataclass
import hashlib
import threading
import numpy as np


__all__ = [
    "StereoLinkContext",
    "find_stable_level_window",
    "find_shared_stereo_level_window",
    "compute_leveling",
]











_LEVELING_CACHE: dict[str, tuple[tuple[float, float, float, float, str, float, float], dict[str, object]]] = {}
_LEVELING_CACHE_LOCK = threading.Lock()
_LEVELING_CACHE_MAX = 128
_LEVEL_WINDOW_CACHE: dict[str, tuple[float, float]] = {}
_LEVEL_WINDOW_CACHE_LOCK = threading.Lock()
_LEVEL_WINDOW_CACHE_MAX = 256
_LEVELING_CACHE_ATTRS = (
    "_lvl_last_error",
    "_lvl_tilt_slope_db_per_oct",
    "_lvl_perceptual_enabled",
    "_lvl_perceptual_strength",
    "_lvl_perceptual_band_hz",
    "_lvl_perceptual_error_rms",
    "_lvl_window_debug",
)

@dataclass(frozen=True)
class StereoLinkContext:


    forced_window_hz: tuple[float, float] | None = None
    forced_offset_db: float | None = None
    shared_target_level_db: float | None = None
    shared_target_shift_db: float | None = None

def _to_float(x, default: float) -> float:
    """Sisainen apufunktio: to float."""
    try:
        v = float(x)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    if not np.isfinite(v):
        return float(default)
    return float(v)

def _to_bool(x, default: bool) -> bool:
    """Sisainen apufunktio: to bool."""
    try:
        if isinstance(x, (bool, np.bool_)):
            return bool(x)
        if x is None:
            return bool(default)
        if isinstance(x, str):
            s = x.strip().lower()
            if s in {"1", "true", "yes", "on", "y"}:
                return True
            if s in {"0", "false", "no", "off", "n", ""}:
                return False
            return bool(default)
        if isinstance(x, (int, float, np.integer, np.floating)):
            v = float(x)
            if not np.isfinite(v):
                return bool(default)
            return bool(v != 0.0)
        return bool(x)
    except (TypeError, ValueError, OverflowError):
        return bool(default)

def _remember_leveling_error(cfg, stage: str, exc: Exception | None = None) -> None:
    try:
        if exc is None:
            msg = str(stage)
        else:
            msg = f"{stage}:{type(exc).__name__}"
        setattr(cfg, "_lvl_last_error", msg)
    except (AttributeError, TypeError, ValueError):
        return

def _safe_setattr(cfg, name: str, value) -> None:
    try:
        setattr(cfg, name, value)
    except (AttributeError, TypeError, ValueError):
        return

def _normalize_optional_float(value):
    if value is None:
        return None
    try:
        v = _to_float(value, 0.0)
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(v):
        return None
    return float(v)

def _normalize_optional_window(value):
    if value is None:
        return None
    try:
        lo, hi = value
    except (TypeError, ValueError):
        return ("invalid",)
    return (
        _normalize_optional_float(lo),
        _normalize_optional_float(hi),
    )

def _normalize_hpf_freq(cfg):
    hpf_settings = getattr(cfg, "hpf_settings", None)
    if not hpf_settings:
        return None
    try:
        return _normalize_optional_float(hpf_settings.get("freq", 0.0))
    except (AttributeError, TypeError, ValueError):
        return ("invalid",)

def _normalize_level_window_params(
    *,
    f_min,
    f_max,
    window_size_octaves,
    hpf_freq,
    tilt_comp,
    tilt_max_db_per_oct,
    perceptual_weighting,
    perceptual_strength,
    perceptual_min_hz,
    perceptual_max_hz,
    perceptual_tie_only,
):
    tilt_comp = _to_bool(tilt_comp, True)
    perceptual_weighting = _to_bool(perceptual_weighting, False)
    perceptual_tie_only = _to_bool(perceptual_tie_only, False) if perceptual_weighting else None
    return (
        _normalize_optional_float(f_min),
        _normalize_optional_float(f_max),
        _normalize_optional_float(window_size_octaves),
        _normalize_optional_float(hpf_freq),
        bool(tilt_comp),
        _normalize_optional_float(tilt_max_db_per_oct) if tilt_comp else None,
        bool(perceptual_weighting),
        _normalize_optional_float(perceptual_strength) if (perceptual_weighting and not perceptual_tie_only) else None,
        _normalize_optional_float(perceptual_min_hz) if perceptual_weighting else None,
        _normalize_optional_float(perceptual_max_hz) if perceptual_weighting else None,
        perceptual_tie_only,
    )

def _hash_leveling_array(h: "hashlib._Hash", value) -> None:
    arr = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    h.update(repr(arr.shape).encode("ascii"))
    h.update(arr.tobytes())

def _stable_level_window_cache_key(
    freq_axis: np.ndarray,
    magnitudes: np.ndarray,
    target_mags: np.ndarray,
    *,
    f_min,
    f_max,
    window_size_octaves,
    hpf_freq,
    tilt_comp,
    tilt_max_db_per_oct,
    perceptual_weighting,
    perceptual_strength,
    perceptual_min_hz,
    perceptual_max_hz,
    perceptual_tie_only,
) -> str:
    h = hashlib.md5()
    h.update(b"stable-window")
    _hash_leveling_array(h, freq_axis)
    _hash_leveling_array(h, magnitudes)
    _hash_leveling_array(h, target_mags)
    h.update(
        repr(
            _normalize_level_window_params(
                f_min=f_min,
                f_max=f_max,
                window_size_octaves=window_size_octaves,
                hpf_freq=hpf_freq,
                tilt_comp=tilt_comp,
                tilt_max_db_per_oct=tilt_max_db_per_oct,
                perceptual_weighting=perceptual_weighting,
                perceptual_strength=perceptual_strength,
                perceptual_min_hz=perceptual_min_hz,
                perceptual_max_hz=perceptual_max_hz,
                perceptual_tie_only=perceptual_tie_only,
            )
        ).encode("utf-8")
    )
    return h.hexdigest()

def _leveling_cache_key(
    cfg,
    freq_axis: np.ndarray,
    m_anal: np.ndarray,
    target_mags: np.ndarray,
    *,
    stereo_link_ctx: StereoLinkContext | None = None,
) -> str:
    h = hashlib.md5()
    _hash_leveling_array(h, freq_axis)
    _hash_leveling_array(h, m_anal)
    _hash_leveling_array(h, target_mags)
    payload = (
        str(getattr(cfg, "auto_goal", "balanced") or "balanced").strip().lower(),
        _normalize_optional_float(getattr(cfg, "lvl_manual_db", 0.0)),
        _normalize_optional_float(getattr(cfg, "lvl_min", 500.0)),
        _normalize_optional_float(getattr(cfg, "lvl_max", 2000.0)),
        str(getattr(cfg, "lvl_mode", "Auto")),
        _to_bool(getattr(cfg, "lvl_tilt_comp", True), True),
        _normalize_optional_float(getattr(cfg, "lvl_tilt_max_db_per_oct", 2.0)),
        _to_bool(getattr(cfg, "lvl_perceptual_weighting", False), False),
        _normalize_optional_float(getattr(cfg, "lvl_perceptual_strength", 0.12)),
        _normalize_optional_float(getattr(cfg, "lvl_perceptual_min_hz", 250.0)),
        _normalize_optional_float(getattr(cfg, "lvl_perceptual_max_hz", 4000.0)),
        _to_bool(getattr(cfg, "lvl_perceptual_tie_only", True), True),
        _normalize_optional_window(getattr(cfg, "lvl_force_window", None)),
        _normalize_optional_float(getattr(cfg, "lvl_force_offset_db", None)),
        _normalize_hpf_freq(cfg),
        None if stereo_link_ctx is None else _normalize_optional_window(getattr(stereo_link_ctx, "forced_window_hz", None)),
        None if stereo_link_ctx is None else _normalize_optional_float(getattr(stereo_link_ctx, "forced_offset_db", None)),
        None if stereo_link_ctx is None else _normalize_optional_float(getattr(stereo_link_ctx, "shared_target_level_db", None)),
        None if stereo_link_ctx is None else _normalize_optional_float(getattr(stereo_link_ctx, "shared_target_shift_db", None)),
    )
    h.update(repr(payload).encode("utf-8"))
    return h.hexdigest()

def _capture_leveling_state(cfg) -> dict[str, object]:
    state: dict[str, object] = {}
    for name in _LEVELING_CACHE_ATTRS:
        try:
            state[name] = copy.deepcopy(getattr(cfg, name, None))
        except (AttributeError, TypeError, ValueError):
            state[name] = None
    return state

def _restore_leveling_state(cfg, state: dict[str, object]) -> None:
    for name in _LEVELING_CACHE_ATTRS:
        _safe_setattr(cfg, name, copy.deepcopy(state.get(name)))

def _clear_leveling_cache() -> None:
    """Tyhjentaa leveling-valimuistin. Vain testeille."""
    with _LEVELING_CACHE_LOCK:
        _LEVELING_CACHE.clear()

def _clear_level_window_cache() -> None:
    """Tyhjentaa level-window-valimuistin. Vain testeille."""
    with _LEVEL_WINDOW_CACHE_LOCK:
        _LEVEL_WINDOW_CACHE.clear()


__all__ = ['StereoLinkContext', '_to_float', '_to_bool', '_remember_leveling_error', '_safe_setattr', '_normalize_optional_float', '_normalize_optional_window', '_normalize_hpf_freq', '_normalize_level_window_params', '_hash_leveling_array', '_stable_level_window_cache_key', '_leveling_cache_key', '_capture_leveling_state', '_restore_leveling_state', '_clear_leveling_cache', '_clear_level_window_cache']


def _link_sibling_exports() -> None:
    import importlib
    package = __package__
    for module_name in ['state_cache', 'tilt_helpers', 'window_scoring', 'api']:
        if module_name == __name__.rsplit('.', 1)[-1]:
            continue
        module = importlib.import_module(f"{package}.{module_name}")
        for symbol in getattr(module, "__all__", ()):
            globals().setdefault(symbol, getattr(module, symbol))


_link_sibling_exports()
