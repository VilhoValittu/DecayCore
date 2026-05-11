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

import threading
from typing import Any


_LOCK = threading.Lock()
_STATE: dict[str, Any] = {
    "busy": False,
    "status_text": "",
    "error_text": "",
    "bundle": None,
    "nonce": 0,
    "timing_ref_latency_ms": None,
}


def _bump() -> None:
    _STATE["nonce"] = int(_STATE.get("nonce", 0) or 0) + 1


def set_busy(status_text: str) -> None:
    with _LOCK:
        _STATE["busy"] = True
        _STATE["status_text"] = str(status_text or "")
        _STATE["error_text"] = ""
        _bump()


def set_result(bundle, status_text: str) -> None:
    with _LOCK:
        _STATE["busy"] = False
        _STATE["bundle"] = bundle
        _STATE["status_text"] = str(status_text or "")
        _STATE["error_text"] = ""
        _bump()


def set_status(status_text: str) -> None:
    with _LOCK:
        _STATE["busy"] = False
        _STATE["status_text"] = str(status_text or "")
        _STATE["error_text"] = ""
        _bump()


def set_error(error_text: str) -> None:
    with _LOCK:
        _STATE["busy"] = False
        _STATE["error_text"] = str(error_text or "")
        _STATE["status_text"] = ""
        _bump()


def set_timing_reference(latency_ms: float) -> None:
    with _LOCK:
        _STATE["timing_ref_latency_ms"] = float(latency_ms)
        _bump()


def clear_timing_reference() -> None:
    with _LOCK:
        _STATE["timing_ref_latency_ms"] = None
        _bump()


def get_snapshot() -> dict[str, Any]:
    with _LOCK:
        return dict(_STATE)
