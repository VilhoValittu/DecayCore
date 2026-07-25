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

from ..config.pipeline_parts import collect_ui_config
from ..config.schema import REQUEST_RUNTIME_DEFAULTS

from .run_request import RunRequest


def build_run_request_from_pin(
    pin_obj: Any,
    *,
    version: str,
    auto_mode_compat_version: str,
) -> RunRequest:
    data = collect_ui_config(pin_obj).to_flat_dict()
    data.update(REQUEST_RUNTIME_DEFAULTS)
    data["program_version"] = str(version)
    data["auto_mode_compat_version"] = str(auto_mode_compat_version)
    return RunRequest(raw_ui_data=dict(data))
