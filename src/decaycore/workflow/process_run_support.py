# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Dependencies supplied to the process-run workflow."""

from __future__ import annotations

import typing
from dataclasses import dataclass

from .bridge_types import ProcessRunUiBridge


@dataclass(frozen=True)
class ProcessRunSupport:
    version: str
    max_safe_boost: float
    force_single_plot_fs_hz: int
    auto_target_mode_norm: typing.Callable[[typing.Any], str]
    auto_target_selection_method_text: typing.Callable[[typing.Any], str]
    pick_target_curve_label: typing.Callable[[dict], str]
    slugify_filename_token: typing.Callable[..., str]
    has_uploaded_target_file: typing.Callable[[dict], bool]
    ui_bridge: ProcessRunUiBridge


__all__ = ["ProcessRunSupport"]
