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

import json
import math
from pathlib import Path

from ..app_paths import decaycore_data_dir
from .auto_mode_policy import auto_filter_cache_key

_BOOL_KEYS = {
    "bass_first_ai",
    "comparison_mode",
    "enable_afdw",
    "enable_tdc",
}
_INT_KEYS: set[str] = set()


def normalize_prior_scalar(key: str, value):
    if key in _BOOL_KEYS:
        return bool(value)
    if key in _INT_KEYS:
        try:
            return int(round(float(value)))
        except (TypeError, ValueError):
            return value
    if isinstance(value, float):
        try:
            if float(value).is_integer():
                return int(round(float(value)))
        except (TypeError, ValueError):
            return value
    return value


def normalize_prior_mapping(payload: dict) -> dict:
    return {str(key): normalize_prior_scalar(str(key), value) for key, value in dict(payload or {}).items()}


def normalize_adaptive_curve(payload: dict | None) -> dict:
    """Return a JSON-safe adaptive target curve or an empty mapping."""
    payload_d = dict(payload or {})
    try:
        frequency_hz = [float(value) for value in payload_d.get("frequency_hz", [])]
        magnitude_db = [float(value) for value in payload_d.get("magnitude_db", [])]
    except (TypeError, ValueError, OverflowError):
        return {}
    if len(frequency_hz) < 4 or len(frequency_hz) != len(magnitude_db):
        return {}
    if not all(math.isfinite(value) for value in (*frequency_hz, *magnitude_db)):
        return {}
    if any(right <= left for left, right in zip(frequency_hz, frequency_hz[1:])):
        return {}
    out = {
        "frequency_hz": frequency_hz,
        "magnitude_db": magnitude_db,
    }
    cache_identity = dict(payload_d.get("cache_identity", {}) or {})
    try:
        normalized_identity = {
            "algorithm_version": int(cache_identity["algorithm_version"]),
            "measurement_signature": str(cache_identity["measurement_signature"]),
            "tilt_comp_frac": float(cache_identity["tilt_comp_frac"]),
            "bass_comp_frac": float(cache_identity["bass_comp_frac"]),
            "bass_comp_ref_db": float(cache_identity["bass_comp_ref_db"]),
            "hf_comp_frac": float(cache_identity["hf_comp_frac"]),
            "smooth_oct": float(cache_identity["smooth_oct"]),
        }
    except (KeyError, TypeError, ValueError, OverflowError):
        normalized_identity = {}
    if (
        normalized_identity
        and all(
            math.isfinite(value)
            for key, value in normalized_identity.items()
            if key not in {"algorithm_version", "measurement_signature"}
        )
        and normalized_identity["measurement_signature"]
    ):
        out["cache_identity"] = normalized_identity
    return out


def normalize_filter_prior_entry(payload: dict) -> dict:
    payload_d = dict(payload or {})
    entry = {
        "filter_type": str(payload_d.get("filter_type", "") or ""),
        "auto_defaults": normalize_prior_mapping(payload_d.get("auto_defaults", {}) or {}),
        "seed_preset": normalize_prior_mapping(payload_d.get("seed_preset", {}) or {}),
    }
    adaptive_curve = normalize_adaptive_curve(payload_d.get("adaptive_curve", {}) or {})
    if adaptive_curve:
        entry["adaptive_curve"] = adaptive_curve
    return entry


def normalize_filter_priors(payload: dict) -> dict:
    return {
        str(filter_key): normalize_filter_prior_entry(dict(entry or {}))
        for filter_key, entry in dict(payload or {}).items()
    }


def merge_filter_prior_entry(base: dict | None, overlay: dict | None) -> dict:
    base_d = normalize_filter_prior_entry(dict(base or {}))
    overlay_d = normalize_filter_prior_entry(dict(overlay or {}))
    entry = {
        "filter_type": str(overlay_d.get("filter_type") or base_d.get("filter_type") or ""),
        "auto_defaults": {
            **dict(base_d.get("auto_defaults", {}) or {}),
            **dict(overlay_d.get("auto_defaults", {}) or {}),
        },
        "seed_preset": {
            **dict(base_d.get("seed_preset", {}) or {}),
            **dict(overlay_d.get("seed_preset", {}) or {}),
        },
    }
    adaptive_curve = dict(overlay_d.get("adaptive_curve", {}) or {}) or dict(base_d.get("adaptive_curve", {}) or {})
    if adaptive_curve:
        entry["adaptive_curve"] = adaptive_curve
    return entry


def merge_filter_priors(base: dict | None, overlay: dict | None) -> dict:
    keys = set(dict(base or {})) | set(dict(overlay or {}))
    return {
        str(key): merge_filter_prior_entry(
            dict(base or {}).get(key, {}),
            dict(overlay or {}).get(key, {}),
        )
        for key in sorted(str(item) for item in keys)
    }


def read_prior_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (TypeError, ValueError, OSError):
        return {}


def factory_filter_priors_path() -> Path:
    return Path(__file__).resolve().parents[1] / "resources" / "auto_mode_filter_priors.json"


def user_filter_priors_path() -> Path:
    return decaycore_data_dir() / "auto_mode_filter_priors.json"


def get_auto_mode_filter_auto_defaults(filter_type: str | None) -> dict:
    """Read non-measurement AUTO defaults needed while building run config."""
    factory = read_prior_json(factory_filter_priors_path())
    user = read_prior_json(user_filter_priors_path())
    filters = merge_filter_priors(
        normalize_filter_priors(factory.get("filters", {}) or {}),
        normalize_filter_priors(user.get("filters", {}) or {}),
    )
    filter_key = auto_filter_cache_key(filter_type=filter_type)
    return dict(dict(filters.get(filter_key, {}) or {}).get("auto_defaults", {}) or {})


__all__ = [
    "factory_filter_priors_path",
    "get_auto_mode_filter_auto_defaults",
    "merge_filter_prior_entry",
    "merge_filter_priors",
    "normalize_adaptive_curve",
    "normalize_filter_prior_entry",
    "normalize_filter_priors",
    "normalize_prior_mapping",
    "normalize_prior_scalar",
    "read_prior_json",
    "user_filter_priors_path",
]
