"""Shared phase 3 status helpers for automatic-mode orchestration."""

from __future__ import annotations


PHASE3_SKIP_STATUS = "DecayCore automatic mode: phase 3 skipped"


def emit_phase3_skip_notice(status_cb) -> str:
    msg = str(PHASE3_SKIP_STATUS)
    if callable(status_cb):
        status_cb(msg)
    return msg


__all__ = ["PHASE3_SKIP_STATUS", "emit_phase3_skip_notice"]
