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
from pathlib import Path

from ..app_paths import decaycore_data_dir
from .shared import _auto_filter_cache_key


_BOOL_KEYS = {
    "bass_first_ai",
    "comparison_mode",
    "enable_afdw",
    "enable_tdc",
}

_INT_KEYS: set = set()

_PRIORS_CACHE: dict | None = None
_PRIOR_LOOKUP_CACHE: dict[tuple[str, str], dict] = {}


def clear_auto_mode_filter_priors_cache() -> None:
    """Clear the cached auto-mode filter priors so the JSON is re-read."""
    global _PRIORS_CACHE
    _PRIORS_CACHE = None
    _PRIOR_LOOKUP_CACHE.clear()


def _factory_priors_path() -> Path:
    return Path(__file__).resolve().parents[1] / "resources" / "auto_mode_filter_priors.json"


def _user_priors_path() -> Path:
    return decaycore_data_dir() / "auto_mode_filter_priors.json"


def _priors_path() -> Path:
    """Return the bundled factory prior path.

    Kept as a compatibility hook for older tests/callers that monkeypatch the
    original single-path helper.
    """
    return _factory_priors_path()


def _measurement_signature(
    measurements: dict | None = None,
    measurement_sig: str | None = None,
) -> str:
    sig = str(measurement_sig or "").strip()
    if sig:
        return sig
    if not isinstance(measurements, dict):
        return ""
    try:
        from .cache_measurement_sig import _auto_search_measurement_identity
        return str(_auto_search_measurement_identity(measurements)).strip()
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
        return ""


def _normalize_scalar(key: str, value):
    if key in _BOOL_KEYS:
        return bool(value)
    if key in _INT_KEYS:
        try:
            return int(round(float(value)))
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
            return value
    if isinstance(value, float):
        try:
            if float(value).is_integer():
                return int(round(float(value)))
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
            return value
    return value


def _normalize_mapping(payload: dict) -> dict:
    out = {}
    for key, value in dict(payload or {}).items():
        out[str(key)] = _normalize_scalar(str(key), value)
    return out


def _normalize_filter_entry(payload: dict) -> dict:
    payload_d = dict(payload or {})
    return {
        "filter_type": str(payload_d.get("filter_type", "") or ""),
        "auto_defaults": _normalize_mapping(payload_d.get("auto_defaults", {}) or {}),
        "seed_preset": _normalize_mapping(payload_d.get("seed_preset", {}) or {}),
    }


def _normalize_filters(payload: dict) -> dict:
    filters = {}
    for filter_key, entry in dict(payload or {}).items():
        filters[str(filter_key)] = _normalize_filter_entry(dict(entry or {}))
    return filters


def _merge_filter_entry(base: dict | None, overlay: dict | None) -> dict:
    base_d = _normalize_filter_entry(dict(base or {}))
    overlay_d = _normalize_filter_entry(dict(overlay or {}))
    return {
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


def _merge_filters(base: dict | None, overlay: dict | None) -> dict:
    out = {}
    keys = set(dict(base or {}).keys()) | set(dict(overlay or {}).keys())
    for key in sorted(str(k) for k in keys):
        out[str(key)] = _merge_filter_entry(
            dict(base or {}).get(key, {}),
            dict(overlay or {}).get(key, {}),
        )
    return out


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
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
        return {}


def load_auto_mode_filter_priors(*, force_reload: bool = False) -> dict:
    global _PRIORS_CACHE
    if _PRIORS_CACHE is not None and not bool(force_reload):
        return dict(_PRIORS_CACHE)
    if bool(force_reload):
        _PRIOR_LOOKUP_CACHE.clear()

    factory = _read_json(_priors_path())
    user = _read_json(_user_priors_path())

    filters = _merge_filters(
        _normalize_filters(factory.get("filters", {}) or {}),
        _normalize_filters(user.get("filters", {}) or {}),
    )

    measurements = {}
    for msig, payload in dict(user.get("measurements", {}) or {}).items():
        payload_d = dict(payload or {})
        measurements[str(msig)] = {
            "source": str(payload_d.get("source", "") or ""),
            "updated_at": str(payload_d.get("updated_at", "") or ""),
            "filters": _normalize_filters(payload_d.get("filters", {}) or {}),
        }

    _PRIORS_CACHE = {
        "version": max(
            int(dict(factory or {}).get("version", 1) or 1),
            int(dict(user or {}).get("version", 1) or 1),
        ),
        "source": str(dict(user or {}).get("source", dict(factory or {}).get("source", "")) or ""),
        "filters": filters,
        "measurements": measurements,
    }
    return dict(_PRIORS_CACHE)


def get_auto_mode_filter_prior(
    filter_type: str | None,
    *,
    measurements: dict | None = None,
    measurement_sig: str | None = None,
) -> dict:
    filter_key = str(_auto_filter_cache_key(filter_type=str(filter_type or "")))
    msig = _measurement_signature(measurements, measurement_sig)
    cache_key = (str(filter_key), str(msig))
    cached = _PRIOR_LOOKUP_CACHE.get(cache_key)
    if isinstance(cached, dict):
        return dict(cached)
    priors = load_auto_mode_filter_priors()
    base = dict(dict(priors.get("filters", {}) or {}).get(filter_key, {}) or {})
    if msig:
        scoped = dict(
            dict(
                dict(priors.get("measurements", {}) or {})
                .get(msig, {})
            ).get("filters", {}) or {}
        ).get(filter_key, {})
        if isinstance(scoped, dict) and scoped:
            base = _merge_filter_entry(base, scoped)
    _PRIOR_LOOKUP_CACHE[cache_key] = dict(base)
    return dict(base)


def get_auto_mode_filter_auto_defaults(
    filter_type: str | None,
    *,
    measurements: dict | None = None,
    measurement_sig: str | None = None,
) -> dict:
    return dict(
        get_auto_mode_filter_prior(
            filter_type,
            measurements=measurements,
            measurement_sig=measurement_sig,
        ).get("auto_defaults", {}) or {}
    )


def get_auto_mode_filter_seed_preset(
    filter_type: str | None,
    *,
    measurements: dict | None = None,
    measurement_sig: str | None = None,
) -> dict:
    return dict(
        get_auto_mode_filter_prior(
            filter_type,
            measurements=measurements,
            measurement_sig=measurement_sig,
        ).get("seed_preset", {}) or {}
    )


_PRIOR_SKIP_KEYS: frozenset[str] = frozenset({"filter_type_str"})

_DEFAULTS_KEY_TO_DATA_KEY: dict[str, str] = {
    "max_boost_db": "max_boost",
    "mixed_split_freq": "mixed_freq",
}

_KNOWN_SEED_KEYS: tuple[str, ...] = (
    "fdw_cycles",
    "tdc_strength",
    "tdc_max_reduction_db",
    "tdc_slope_db_per_oct",
    "reg_strength",
    "max_boost",
    "mag_c_min",
    "low_bass_cut_hz",
    "mag_c_max",
    "trans_width",
    "filter_smooth",
    "bass_first_mode_max_hz",
    "conf_pull_max_hz",
    "phase_limit",
    "mixed_freq",
    "hpf_enable",
    "hpf_freq",
    "hpf_slope",
    "bass_first_ai",
    "comparison_mode",
    "enable_afdw",
    "enable_tdc",
    "max_slope_db_per_oct",
    "max_slope_boost_db_per_oct",
    "max_slope_cut_db_per_oct",
)

_KNOWN_AUTO_DEFAULT_KEYS: tuple[str, ...] = (
    "filter_type_str",
    "fdw_cycles",
    "tdc_strength",
    "tdc_max_reduction_db",
    "tdc_slope_db_per_oct",
    "reg_strength",
    "max_boost_db",
    "mag_c_min",
    "low_bass_cut_hz",
    "mag_c_max",
    "trans_width",
    "filter_smooth",
    "bass_first_mode_max_hz",
    "conf_pull_max_hz",
    "phase_limit",
    "mixed_split_freq",
    "hpf_enable",
    "hpf_freq",
    "hpf_slope",
    "bass_first_ai",
    "comparison_mode",
    "enable_afdw",
    "enable_tdc",
    "max_slope_db_per_oct",
    "max_slope_boost_db_per_oct",
    "max_slope_cut_db_per_oct",
)

_KNOWN_FILTER_TYPES: dict[str, str] = {
    "asym": "Asymmetric (low-latency)",
    "linear": "Linear Phase",
    "minimum": "Minimum Phase",
    "mixed": "Mixed Phase",
}


def _value_from_winner_or_data(
    key: str,
    best_preset: dict,
    data: dict,
    *,
    is_default_key: bool = False,
):
    data_key = _DEFAULTS_KEY_TO_DATA_KEY.get(key, key) if bool(is_default_key) else key
    if data_key in best_preset:
        return best_preset[data_key], True
    if data_key in data:
        return data[data_key], True
    if key in data:
        return data[key], True
    return None, False


def _seed_template_from_known_keys(best_preset: dict, data: dict) -> dict:
    seed = {}
    for key in _KNOWN_SEED_KEYS:
        value, found = _value_from_winner_or_data(key, best_preset, data)
        if found:
            seed[key] = value
    return _normalize_mapping(seed)


def _defaults_template_from_known_keys(filter_type: str | None, best_preset: dict, data: dict) -> dict:
    defaults = {}
    for key in _KNOWN_AUTO_DEFAULT_KEYS:
        if key == "filter_type_str":
            defaults[key] = str(filter_type or data.get("filter_type", "") or "")
            continue
        value, found = _value_from_winner_or_data(
            key,
            best_preset,
            data,
            is_default_key=True,
        )
        if found:
            defaults[key] = value
    return _normalize_mapping(defaults)


def _update_entry_from_winner(
    entry: dict,
    *,
    filter_type: str | None,
    best_preset: dict,
    data: dict,
) -> tuple[dict, int, int]:
    entry = dict(entry or {})
    seed = dict(entry.get("seed_preset", {}) or {})
    defaults = dict(entry.get("auto_defaults", {}) or {})
    if not seed:
        seed = _seed_template_from_known_keys(best_preset, data)
    if not defaults:
        defaults = _defaults_template_from_known_keys(filter_type, best_preset, data)

    updated_seed = _update_prior_mapping(seed, best_preset=best_preset, data=data, map_default_key=False)
    updated_defaults = _update_prior_mapping(defaults, best_preset=best_preset, data=data, map_default_key=True)

    entry["seed_preset"] = seed
    entry["auto_defaults"] = defaults
    if not str(entry.get("filter_type", "") or "").strip():
        entry["filter_type"] = str(filter_type or "")
    return entry, len(updated_seed), len(updated_defaults)


def _update_prior_mapping(
    payload: dict,
    *,
    best_preset: dict,
    data: dict,
    map_default_key: bool,
) -> list[str]:
    updated_keys: list[str] = []
    for key in list(payload.keys()):
        if key in _PRIOR_SKIP_KEYS:
            continue
        data_key = _DEFAULTS_KEY_TO_DATA_KEY.get(key, key) if bool(map_default_key) else key
        if data_key in best_preset:
            payload[key] = best_preset[data_key]
            updated_keys.append(key)
            continue
        if data_key in data:
            payload[key] = data[data_key]
            updated_keys.append(key)
    return updated_keys


def _entry_from_known_keys(filter_type: str | None, best_preset: dict, data: dict) -> dict:
    return {
        "filter_type": str(filter_type or data.get("filter_type", "") or ""),
        "auto_defaults": _defaults_template_from_known_keys(filter_type, best_preset, data),
        "seed_preset": _seed_template_from_known_keys(best_preset, data),
    }


def update_auto_mode_filter_priors_from_winner(
    filter_type: str | None,
    best_preset: dict,
    data: dict,
    *,
    measurements: dict | None = None,
    measurement_sig: str | None = None,
) -> None:
    """Write winner parameters into the measurement-scoped user prior JSON.

    Iterates the existing JSON keys as the update template and resolves each value
    from best_preset (optimizer keys) or data (full merged state). Only updates keys
    already tracked in the JSON — does not add new keys.
    """
    import logging
    logger = logging.getLogger(__name__)
    try:
        filter_key = str(_auto_filter_cache_key(filter_type=str(filter_type or "")))
        msig = _measurement_signature(measurements, measurement_sig)
        if not msig:
            logger.warning("Filter priors update skipped: missing measurement signature")
            return

        from datetime import datetime
        now = datetime.now()
        source = f"auto_run {filter_key} {now.strftime('%Y%m%d_%H%M%S')}"
        path = _user_priors_path()
        raw = _read_json(path)
        if not isinstance(raw, dict):
            raw = {}
        raw["version"] = 3
        raw["source"] = "auto_run"
        measurement_map = dict(raw.get("measurements", {}) or {})
        measurement_entry = dict(measurement_map.get(msig, {}) or {})
        filters = _normalize_filters(measurement_entry.get("filters", {}) or {})
        prior_snapshot = load_auto_mode_filter_priors()
        global_filters = _normalize_filters(prior_snapshot.get("filters", {}) or {})
        for known_filter_key, known_entry in global_filters.items():
            filters[str(known_filter_key)] = _merge_filter_entry(
                dict(known_entry or {}),
                dict(filters.get(str(known_filter_key), {}) or {}),
            )
        template = get_auto_mode_filter_prior(
            filter_key,
            measurements=measurements,
            measurement_sig=msig,
        )
        entry, updated_seed_count, updated_defaults_count = _update_entry_from_winner(
            dict(template or {}),
            filter_type=filter_type,
            best_preset=best_preset,
            data=data,
        )
        filters[filter_key] = entry
        measurement_entry["source"] = source
        measurement_entry["updated_at"] = now.isoformat(timespec="seconds")
        measurement_entry["filters"] = filters
        measurement_map[msig] = measurement_entry
        raw["measurements"] = measurement_map

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(raw, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        clear_auto_mode_filter_priors_cache()
        logger.info(
            f"Updated measurement-scoped auto-mode filter priors for {filter_key} "
            f"({updated_seed_count} seed keys, {updated_defaults_count} defaults keys; "
            f"{len(filters)} filter types stored)"
        )
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
    ) as exc:
        logging.getLogger(__name__).warning(
            f"Failed to update auto-mode filter priors: {type(exc).__name__}: {exc}"
        )
