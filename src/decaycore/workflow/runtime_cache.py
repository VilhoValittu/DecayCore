"""Backward-compatible facade for application-level runtime cache resets."""

from __future__ import annotations

from ..application.runtime_cache import reset_runtime_caches

__all__ = ["reset_runtime_caches"]
