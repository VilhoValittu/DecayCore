# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""Lightweight profiling helpers shared by orchestration and DSP layers."""

from __future__ import annotations

import contextvars
import os
import threading
import time
from contextlib import contextmanager


def auto_mode_profile_enabled(base_data: dict | None) -> bool:
    """Return whether automatic-mode profiling is enabled for this run."""
    env = os.environ.get("DECAYCORE_AUTO_PROFILE", os.environ.get("CAMILLAFIR_AUTO_PROFILE", "")).strip()
    if env in ("1", "true", "yes"):
        return True
    return bool((base_data or {}).get("auto_mode_profile", False))


class AutoModeProfiler:
    """Accumulate wall-clock timings for named sections, thread-safely."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sections: dict[str, list] = {}
        self._started_at = time.perf_counter()

    @contextmanager
    def section(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            with self._lock:
                if name not in self._sections:
                    self._sections[name] = [0.0, 0]
                self._sections[name][0] += elapsed
                self._sections[name][1] += 1

    def wrap(self, fn, name: str):
        """Return a wrapped callable that times each call under *name*."""
        profiler = self

        def _timed(*args, **kwargs):
            with active_profiler_scope(profiler):
                with profiler.section(name):
                    return fn(*args, **kwargs)

        return _timed

    def log_summary(self, logger=None, *, label: str = "auto-mode") -> None:
        with self._lock:
            sections = dict(self._sections)
        if not sections:
            return

        # Section totals may overlap through nesting and are accumulated across
        # worker threads. Only elapsed time since profiler creation is wall time.
        wall_s = max(0.0, time.perf_counter() - self._started_at)
        accumulated_s = sum(float(values[0]) for values in sections.values())
        lines = [
            f"[PROFILE] [{label}] wall={wall_s:.3f}s "
            f"accumulated_sections={accumulated_s:.3f}s"
        ]
        for name, (s, n) in sorted(sections.items(), key=lambda x: -x[1][0]):
            pct = 100.0 * s / wall_s if wall_s > 0 else 0.0
            per = s / n if n > 0 else 0.0
            if n == 1:
                lines.append(f"  {name}: {s:.3f}s ({pct:.1f}% wall)")
            else:
                lines.append(f"  {name}: {s:.3f}s ({pct:.1f}% wall) n={n} avg={per*1000:.1f}ms")
        print("\n".join(lines), flush=True)


_ACTIVE_PROFILER: contextvars.ContextVar[AutoModeProfiler | None] = contextvars.ContextVar(
    "camillafir_auto_mode_profiler",
    default=None,
)


def current_active_profiler() -> AutoModeProfiler | None:
    return _ACTIVE_PROFILER.get()


@contextmanager
def active_profiler_scope(profiler: AutoModeProfiler | None):
    if profiler is None:
        yield
        return
    token = _ACTIVE_PROFILER.set(profiler)
    try:
        yield
    finally:
        _ACTIVE_PROFILER.reset(token)


@contextmanager
def profiled_section(name: str):
    profiler = current_active_profiler()
    if profiler is None:
        yield
        return
    with profiler.section(name):
        yield


__all__ = [
    "AutoModeProfiler",
    "active_profiler_scope",
    "auto_mode_profile_enabled",
    "current_active_profiler",
    "profiled_section",
]
