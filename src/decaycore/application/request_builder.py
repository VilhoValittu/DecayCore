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

from ..config.decaycore_pipeline import collect_ui_data

from .run_request import RunRequest


def build_run_request_from_pin(
    pin_obj: Any,
    *,
    version: str,
    auto_mode_compat_version: str,
) -> RunRequest:
    data = collect_ui_data(pin_obj)
    data["bass_adaptive_isolation_mode"] = False
    data["bass_smooth_sigma_scale"] = 1.20
    data["bass_smooth_conf_floor"] = 0.25
    data["bass_smooth_w_gamma"] = 2.40
    data["bass_smooth_w_max"] = 0.45
    data["program_version"] = str(version)
    data["auto_mode_compat_version"] = str(auto_mode_compat_version)
    return RunRequest(raw_ui_data=dict(data))
