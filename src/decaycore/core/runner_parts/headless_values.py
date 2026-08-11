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

import math
from typing import Any

from ...auto_mode.rank_score import attach_official_rank_score, official_rank_score
from ...config.legacy_keys import CAMILLAFIR_AUTO_MODE


def _f(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
        if math.isfinite(out):
            return out
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
        return default
    return default


def _safe_filename_token(value: Any, default: str = "v0") -> str:
    raw = str(value or "").strip()
    if not raw:
        return default
    out = "".join(
        ch if ch.isalnum() or ch in "._-" else "-" for ch in raw
    ).strip(".-_")
    return out or default


def _headless_winner_rank_score(data: dict | None) -> float:
    try:
        if not bool((data or {}).get(CAMILLAFIR_AUTO_MODE, False)):
            return float("nan")
        auto_meta = dict((data or {}).get("_auto_mode_meta", {}) or {})
        best_metrics = attach_official_rank_score(auto_meta.get("best_metrics", {}))
        return float(official_rank_score(best_metrics))
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
        return float("nan")


__all__ = [
    "_f",
    "_headless_winner_rank_score",
    "_safe_filename_token",
]
