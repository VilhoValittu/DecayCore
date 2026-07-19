# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Shared mutable state for the Target tab preview."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class _TargetPreviewState:
    refresh_token: int = 0
    drag_base_points: list[tuple[float, float]] = field(default_factory=list)
    tilt_handle_points: list[tuple[float, float]] = field(default_factory=list)
    drag_active: bool = False
    plot: Any = None
    translate: Callable[[str], str] | None = None

    def reset(self) -> None:
        self.drag_base_points = []
        self.tilt_handle_points = []
        self.drag_active = False
        self.plot = None


STATE = _TargetPreviewState()
