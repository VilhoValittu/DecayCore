# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Typed config schema and compatibility projections.

The application still persists and exposes the historical flat dict shape.
This module owns the field registry behind that shape so defaults, UI pins,
mode policies, and runtime-only knobs do not drift across modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Literal

from .legacy_keys import CAMILLAFIR_AUTO_MODE

FieldKind = Literal["bool", "int", "float", "str", "choice", "any"]
CacheRelevance = Literal["dsp", "auto", "measurement", "ui", "runtime", "none"]

from .schema_defaults import (
    AUTO_MODE_DEFAULT_CFG_TO_UI,
    CHOICE_OPTIONS_BY_KEY,
    DEFAULT_CONFIG_ITEMS,
    DEVICE_AUDIO_FORMAT_OPTIONS,
    FILTER_WAV_FORMAT_OPTIONS,
    FS_OPTIONS,
    HIDDEN_CONF_DEFAULTS_ADVANCED,
    HIDDEN_CONF_DEFAULTS_BASIC_AUTO,
    IR_EXPORT_WINDOW_MODE_OPTIONS,
    IR_EXPORT_WINDOW_SHAPE_OPTIONS,
    LIST_BOOL_KEYS,
    MODE_CLAMPS_BASE,
    MODE_DEFAULTS_BASE,
    PLOT_SMOOTHING_LEVEL_OPTIONS,
    REQUEST_RUNTIME_DEFAULTS,
    SLOPE_OPTIONS,
    STEREO_LINK_STRATEGY_OPTIONS,
    TAPS_OPTIONS,
    UI_PIN_KEYS,
)


@dataclass(frozen=True)
class ConfigFieldSpec:
    key: str
    default: Any = None
    kind: FieldKind = "any"
    choices: tuple[Any, ...] = ()
    persist: bool = True
    ui_pin: str | None = None
    filter_attr: str | None = None
    cache_relevance: CacheRelevance = "none"


@dataclass(frozen=True)
class AppConfigSnapshot:
    values: dict[str, Any] = field(default_factory=dict)

    def to_flat_dict(self) -> dict[str, Any]:
        return dict(self.values)


@dataclass(frozen=True)
class RunConfigSnapshot:
    values: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_flat_dict(cls, data: dict[str, Any] | None) -> RunConfigSnapshot:
        return cls(values=normalize_flat_config(data or {}, include_runtime=True))

    def to_flat_dict(self) -> dict[str, Any]:
        return dict(self.values)


@dataclass(frozen=True)
class FilterConfigProjection:
    values: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_run_config(
        cls, snapshot: RunConfigSnapshot | FilterConfigProjection | dict[str, Any]
    ) -> FilterConfigProjection:
        if isinstance(snapshot, FilterConfigProjection):
            return snapshot
        if isinstance(snapshot, RunConfigSnapshot):
            return cls(values=snapshot.to_flat_dict())
        return cls(values=normalize_flat_config(snapshot or {}, include_runtime=True))

    def to_legacy_dict(self) -> dict[str, Any]:
        return dict(self.values)


def _infer_kind(default: Any, choices: tuple[Any, ...]) -> FieldKind:
    if choices:
        return "choice"
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, int) and not isinstance(default, bool):
        return "int"
    if isinstance(default, float):
        return "float"
    if isinstance(default, str):
        return "str"
    return "any"


def _default_specs() -> list[ConfigFieldSpec]:
    ui_keys = set(UI_PIN_KEYS)
    specs: list[ConfigFieldSpec] = []
    for key, default in DEFAULT_CONFIG_ITEMS:
        choices = CHOICE_OPTIONS_BY_KEY.get(key, ())
        specs.append(
            ConfigFieldSpec(
                key=key,
                default=default,
                kind=_infer_kind(default, choices),
                choices=choices,
                persist=key not in REQUEST_RUNTIME_DEFAULTS,
                ui_pin=key if key in ui_keys else None,
                filter_attr=_filter_attr_for_key(key),
                cache_relevance=_cache_relevance_for_key(key),
            )
        )
    known = {spec.key for spec in specs}
    for key in UI_PIN_KEYS:
        if key not in known:
            specs.append(
                ConfigFieldSpec(
                    key=key,
                    default=None,
                    kind="any",
                    persist=not _is_runtime_only_key(key),
                    ui_pin=key,
                    filter_attr=_filter_attr_for_key(key),
                    cache_relevance=_cache_relevance_for_key(key),
                )
            )
            known.add(key)
    for key, default in REQUEST_RUNTIME_DEFAULTS.items():
        if key not in known:
            specs.append(
                ConfigFieldSpec(
                    key=key,
                    default=default,
                    kind=_infer_kind(default, ()),
                    persist=False,
                    ui_pin=None,
                    filter_attr=_filter_attr_for_key(key),
                    cache_relevance="runtime",
                )
            )
    return specs


def _filter_attr_for_key(key: str) -> str | None:
    reverse = {
        ui_key: cfg_key for cfg_key, ui_key in AUTO_MODE_DEFAULT_CFG_TO_UI.items()
    }
    return reverse.get(
        key,
        key if key in {cfg_key for cfg_key in AUTO_MODE_DEFAULT_CFG_TO_UI} else None,
    )


def _cache_relevance_for_key(key: str) -> CacheRelevance:
    if (
        key.startswith("measurement_")
        or key.startswith("local_path")
        or key.startswith("file_")
    ):
        return "measurement"
    if key.startswith("auto_mode_") or key in {
        "auto_goal",
        "auto_target_mode",
        CAMILLAFIR_AUTO_MODE,
    }:
        return "auto"
    if key.startswith("ui_") or key in {"layout", "fmt"}:
        return "ui"
    if key in REQUEST_RUNTIME_DEFAULTS:
        return "runtime"
    if key in UI_PIN_KEYS or key in dict(DEFAULT_CONFIG_ITEMS):
        return "dsp"
    return "none"


def _is_runtime_only_key(key: str) -> bool:
    return (
        key.startswith("file_")
        or key.startswith("generated_measurement_")
        or key in {"auto_mode_compat_version", "unsafe_raw_dsp", "_config_version"}
        or key in REQUEST_RUNTIME_DEFAULTS
    )


FIELD_SPECS: tuple[ConfigFieldSpec, ...] = tuple(_default_specs())
FIELD_SPECS_BY_KEY: dict[str, ConfigFieldSpec] = {
    spec.key: spec for spec in FIELD_SPECS
}

MODE_DEFAULTS: dict[str, dict[str, Any]] = {
    "BASIC": dict(MODE_DEFAULTS_BASE["BASIC"]),
    "ADVANCED": dict(MODE_DEFAULTS_BASE["ADVANCED"]),
}
MODE_DEFAULTS["AUTO"] = dict(MODE_DEFAULTS["ADVANCED"])
MODE_DEFAULTS["AUTO"]["stereo_link_strategy"] = "auto"

MODE_CLAMPS: dict[str, dict[str, tuple[Any, Any]]] = {
    "BASIC": dict(MODE_CLAMPS_BASE["BASIC"]),
    "ADVANCED": dict(MODE_CLAMPS_BASE["ADVANCED"]),
}
MODE_CLAMPS["AUTO"] = dict(MODE_CLAMPS["ADVANCED"])


def default_config_dict() -> dict[str, Any]:
    return dict(DEFAULT_CONFIG_ITEMS)


def normalize_filter_type(value: Any) -> str:
    try:
        ft = str(value or "").strip()
    except (
        AttributeError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        RuntimeError,
        OSError,
    ):
        ft = ""
    ft_l = ft.lower()
    if "asym" in ft_l:
        return "Asymmetric"
    if "mixed" in ft_l:
        return "Mixed"
    if "minimum" in ft_l or "minphase" in ft_l or ft_l == "min":
        return "Minimum"
    if "linear" in ft_l:
        return "Linear"
    return "Asymmetric"


def coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off", ""):
            return False
    if isinstance(value, (list, tuple)):
        if not value:
            return False
        return coerce_bool(value[0], default)
    try:
        return bool(value)
    except (AttributeError, TypeError, ValueError):
        return bool(default)


def normalize_list_backed_booleans(data: dict[str, Any]) -> None:
    for key in LIST_BOOL_KEYS:
        if isinstance(data.get(key), list):
            data[key] = coerce_bool(data[key], False)


def _coerce_by_spec(value: Any, spec: ConfigFieldSpec) -> Any:
    if value is None:
        return spec.default
    if spec.kind == "bool":
        return coerce_bool(value, bool(spec.default))
    if spec.kind == "int":
        try:
            return int(float(value))
        except (TypeError, ValueError, OverflowError):
            return int(spec.default or 0)
    if spec.kind == "float":
        try:
            return float(value)
        except (TypeError, ValueError, OverflowError):
            return float(spec.default or 0.0)
    if spec.kind == "str":
        try:
            return str(value)
        except (TypeError, ValueError):
            return str(spec.default or "")
    if spec.kind == "choice":
        return normalize_choice_value(value, options=spec.choices, default=spec.default)
    return value


def normalize_flat_config(
    data: dict[str, Any], *, include_runtime: bool = False
) -> dict[str, Any]:
    out = default_config_dict()
    if include_runtime:
        out.update(REQUEST_RUNTIME_DEFAULTS)
    src = dict(data or {})
    normalize_list_backed_booleans(src)
    for key, value in src.items():
        spec = FIELD_SPECS_BY_KEY.get(key)
        out[key] = _coerce_by_spec(value, spec) if spec is not None else value
    if "filter_type" in out:
        out["filter_type"] = normalize_filter_type(out.get("filter_type"))
    return out


def persistable_config_dict(data: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in dict(data or {}).items():
        if value is None:
            continue
        spec = FIELD_SPECS_BY_KEY.get(str(key))
        if spec is not None and not spec.persist:
            continue
        if _is_runtime_only_key(str(key)):
            continue
        clean[key] = value
    return clean


def _parse_legacy_choice_index(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not float(value).is_integer():
            return None
        return int(value)
    try:
        text = str(value).strip()
    except (AttributeError, TypeError, ValueError):
        return None
    if not text:
        return None
    digits = text[1:] if text[0] in "+-" else text
    if not digits.isdigit():
        return None
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def normalize_choice_value(
    value: Any, *, options: tuple[Any, ...], default: Any
) -> Any:
    if value in options:
        return value
    try:
        raw = str(value).strip()
    except (AttributeError, TypeError, ValueError):
        raw = ""
    if raw:
        for option in options:
            if raw.casefold() == str(option).strip().casefold():
                return option
    index = _parse_legacy_choice_index(value)
    if index is not None and 0 <= index < len(options):
        return options[index]
    return default


def normalize_choice_fields(data: dict[str, Any], default_conf: dict[str, Any]) -> None:
    for key, options in CHOICE_OPTIONS_BY_KEY.items():
        data[key] = normalize_choice_value(
            data.get(key, default_conf.get(key)),
            options=options,
            default=default_conf.get(
                key, FIELD_SPECS_BY_KEY.get(key, ConfigFieldSpec(key)).default
            ),
        )


def app_config_snapshot(data: dict[str, Any] | None = None) -> AppConfigSnapshot:
    return AppConfigSnapshot(
        values=normalize_flat_config(data or {}, include_runtime=False)
    )


def run_config_snapshot(data: dict[str, Any] | None = None) -> RunConfigSnapshot:
    return RunConfigSnapshot.from_flat_dict(data or {})


def snapshot_field_names(
    snapshot: AppConfigSnapshot | RunConfigSnapshot | FilterConfigProjection,
) -> tuple[str, ...]:
    return tuple(field_obj.name for field_obj in fields(snapshot))


__all__ = [
    "AUTO_MODE_DEFAULT_CFG_TO_UI",
    "AppConfigSnapshot",
    "CHOICE_OPTIONS_BY_KEY",
    "ConfigFieldSpec",
    "DEVICE_AUDIO_FORMAT_OPTIONS",
    "FIELD_SPECS",
    "FIELD_SPECS_BY_KEY",
    "FILTER_WAV_FORMAT_OPTIONS",
    "FS_OPTIONS",
    "FilterConfigProjection",
    "HIDDEN_CONF_DEFAULTS_ADVANCED",
    "HIDDEN_CONF_DEFAULTS_BASIC_AUTO",
    "IR_EXPORT_WINDOW_MODE_OPTIONS",
    "IR_EXPORT_WINDOW_SHAPE_OPTIONS",
    "LIST_BOOL_KEYS",
    "MODE_CLAMPS",
    "MODE_DEFAULTS",
    "PLOT_SMOOTHING_LEVEL_OPTIONS",
    "REQUEST_RUNTIME_DEFAULTS",
    "RunConfigSnapshot",
    "SLOPE_OPTIONS",
    "STEREO_LINK_STRATEGY_OPTIONS",
    "TAPS_OPTIONS",
    "UI_PIN_KEYS",
    "app_config_snapshot",
    "coerce_bool",
    "default_config_dict",
    "normalize_choice_fields",
    "normalize_choice_value",
    "normalize_filter_type",
    "normalize_flat_config",
    "normalize_list_backed_booleans",
    "persistable_config_dict",
    "run_config_snapshot",
]
