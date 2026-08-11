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

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .process_run_support import ProcessRunSupport

__all__ = ["ProcessRunSupport", "run_process_flow"]


def __getattr__(name: str) -> Any:
    """Load run orchestration only when its public facade is requested.

    Importing a lightweight workflow submodule must not pull packaged-only
    Automatic mode into the public source application's startup path.
    """
    if name == "ProcessRunSupport":
        from .process_run_support import ProcessRunSupport

        return ProcessRunSupport
    if name == "run_process_flow":
        from .process_run_flow import run_process_flow

        return run_process_flow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
