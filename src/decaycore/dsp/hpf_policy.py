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

from typing import Any

HPF_IIR_TAP_THRESHOLD = 65536


def _positive_int(value: Any, default: int = 0) -> int:
    try:
        out = int(value or default)
    except (TypeError, ValueError, OverflowError):
        return int(default)
    return int(out)


def _positive_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value or default)
    except (TypeError, ValueError, OverflowError):
        return float(default)
    return float(out)


def hpf_settings_are_active(hpf_settings: Any) -> bool:
    if not isinstance(hpf_settings, dict) or not bool(hpf_settings.get("enabled", False)):
        return False
    return _positive_float(hpf_settings.get("freq", 0.0)) > 0.0 and _positive_int(hpf_settings.get("order", 0)) > 0


def hpf_settings_should_use_iir(hpf_settings: Any, num_taps: Any) -> bool:
    taps = _positive_int(num_taps)
    return bool(hpf_settings_are_active(hpf_settings) and taps > 0 and taps < int(HPF_IIR_TAP_THRESHOLD))


def filter_config_should_use_iir_hpf(cfg: Any) -> bool:
    return hpf_settings_should_use_iir(
        getattr(cfg, "hpf_settings", None),
        getattr(cfg, "num_taps", 0),
    )


__all__ = [
    "HPF_IIR_TAP_THRESHOLD",
    "filter_config_should_use_iir_hpf",
    "hpf_settings_are_active",
    "hpf_settings_should_use_iir",
]
