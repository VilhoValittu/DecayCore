# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Canonical immutable Auto Search v2 input model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ..cache_measurement_sig import _auto_measurement_metadata_identity, _auto_measurement_signature
from ..shared import _auto_filter_cache_key, _auto_filter_type_for_key, _auto_hash_array

_WINNER_ONLY_KEYS = frozenset(
    {
        "_auto_exc_freq_hz",
        "_auto_exc_seed_freq_hz",
        "_auto_hpf_meta",
        "_auto_low_bass_cut_hz",
        "_auto_mag_c_min_hz",
        "_auto_measurement_signature",
        "_auto_target_curve_meta",
        "_auto_target_seed_metrics",
        "_auto_target_seed_preset",
        "_auto_target_seed_source",
        "auto_applied_preset",
        "best_applied_preset",
        "best_metrics",
        "best_preset",
        "winner",
    }
)

_AUTO_PREFIX_DENY = ("_auto_",)


@dataclass(frozen=True)
class AutoSearchInput:
    """Canonical user/measurement identity for automatic search decisions."""

    base_data_items: tuple[tuple[str, Any], ...]
    measurement_identity: str
    measurement_metadata_identity: str
    frequency_grid_identity: str
    fs_v: int
    taps_v: int
    xos_items: tuple[Any, ...]
    hpf_items: tuple[tuple[str, Any], ...] | None
    hc_mode: str | None
    include_hc_mode: bool = True

    def base_data(self) -> dict:
        return dict(self.base_data_items)

    def xos(self) -> list:
        return [dict(item) if isinstance(item, tuple) else item for item in self.xos_items]

    def hpf(self) -> dict | None:
        return dict(self.hpf_items) if self.hpf_items is not None else None

    def summary(self) -> dict:
        base = self.base_data()
        return {
            "measurement_identity": str(self.measurement_identity),
            "measurement_metadata_identity": str(self.measurement_metadata_identity),
            "fs": int(self.fs_v),
            "taps": int(self.taps_v),
            "filter_type": str(base.get("filter_type", "") or ""),
            "auto_goal": str(base.get("auto_goal", "") or ""),
            "hc_mode": str(self.hc_mode or base.get("hc_mode", "") or ""),
            "auto_target_mode": str(base.get("auto_target_mode", "") or ""),
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, MappingProxyType):
        return _jsonable(dict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return tuple(_jsonable(v) for v in value)
    try:
        json.dumps(value, sort_keys=True, default=str)
        return value
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
        return str(value)


def _items(value: dict | None) -> tuple[tuple[str, Any], ...]:
    return tuple((str(k), _jsonable(v)) for k, v in sorted(dict(value or {}).items(), key=lambda kv: str(kv[0])))


def _matching_auto_applied_keys(raw: dict) -> set[str]:
    applied: dict[str, Any] = {}
    explicit = raw.get("auto_applied_preset")
    if isinstance(explicit, dict):
        applied.update(explicit)
    meta = raw.get("_auto_mode_meta")
    if isinstance(meta, dict):
        for key in ("best_preset", "best_applied_preset"):
            value = meta.get(key)
            if isinstance(value, dict):
                applied.update(value)
    out: set[str] = set()
    for key, value in applied.items():
        key_s = str(key)
        if key_s not in raw:
            continue
        if _jsonable(raw.get(key_s)) == _jsonable(value):
            out.add(key_s)
    return out


def _identity_base_data(raw: dict) -> dict:
    explicit = raw.get("_auto_search_identity_base_data")
    if isinstance(explicit, dict):
        return dict(explicit)
    meta = raw.get("_auto_mode_meta")
    if isinstance(meta, dict):
        search_v2 = meta.get("auto_search_v2")
        if isinstance(search_v2, dict) and isinstance(search_v2.get("identity_base_data"), dict):
            return dict(search_v2.get("identity_base_data") or {})
    return {}


def canonicalize_auto_search_base_data(raw_data: dict | None) -> dict:
    """Return raw user input stripped of winner/runtime fields."""

    raw = dict(raw_data or {})
    auto_applied_keys = _matching_auto_applied_keys(raw)
    out: dict[str, Any] = _identity_base_data(raw)
    for key, value in raw.items():
        key_s = str(key)
        if key_s in _WINNER_ONLY_KEYS:
            continue
        if key_s in auto_applied_keys:
            continue
        if key_s.startswith(_AUTO_PREFIX_DENY) and key_s not in {
            "_auto_user_phase_limit",
            "_auto_user_mixed_freq",
        }:
            continue
        if key_s in {"synthetic_target", "synthetic_target_f", "synthetic_target_m"}:
            continue
        out[key_s] = value
    if "filter_type" in out:
        out["filter_type"] = _auto_filter_type_for_key(_auto_filter_cache_key(out))
    return out


def build_auto_search_input(raw_data, measurements, context: dict | None = None) -> AutoSearchInput:
    ctx = dict(context or {})
    base = canonicalize_auto_search_base_data(dict(raw_data or {}))
    measurement_dict = dict(measurements or {})
    measurement_identity = str(_auto_measurement_signature(measurement_dict))
    measurement_metadata_identity = str(_auto_measurement_metadata_identity(measurement_dict))
    try:
        import numpy as np

        frequency_grid_identity = str(
            _auto_hash_array(
                np.asarray(
                    measurement_dict.get(
                        "f_l",
                        measurement_dict.get("f_r", []),
                    ),
                    dtype=float,
                )
            )
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
    ):
        frequency_grid_identity = ""
    hpf = ctx.get("hpf")
    hc_mode = str(ctx.get("hc_mode", base.get("hc_mode", "")) or "").strip() or None
    xos_raw = ctx.get("xos", [])
    xos_items = tuple(_jsonable(item) for item in list(xos_raw or []))
    return AutoSearchInput(
        base_data_items=_items(base),
        measurement_identity=str(measurement_identity),
        measurement_metadata_identity=str(measurement_metadata_identity),
        frequency_grid_identity=str(frequency_grid_identity),
        fs_v=int(ctx.get("fs_v", base.get("fs", 44100)) or 44100),
        taps_v=int(ctx.get("taps_v", base.get("taps", 0)) or 0),
        xos_items=xos_items,
        hpf_items=_items(hpf) if isinstance(hpf, dict) else None,
        hc_mode=hc_mode,
        include_hc_mode=bool(ctx.get("include_hc_mode", True)),
    )
