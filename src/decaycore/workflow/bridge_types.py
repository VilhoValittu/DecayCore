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

import typing
from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessRunCallbacks:
    status: typing.Callable[[str], None]
    set_auto_selected_bar: typing.Callable[[typing.Any], None]


@dataclass(frozen=True)
class ProcessRunUiBridge:
    ensure_progress_bar: typing.Callable[[], None]
    set_progress: typing.Callable[[float], None]
    toast_health_gate_result: typing.Callable[..., bool]
    render_results: typing.Callable[..., None]
    build_export_zip: typing.Callable[..., tuple]
    save_export_bundle: typing.Callable[..., tuple]
    make_callbacks: typing.Callable[[float], ProcessRunCallbacks]


__all__ = ["ProcessRunCallbacks", "ProcessRunUiBridge"]
