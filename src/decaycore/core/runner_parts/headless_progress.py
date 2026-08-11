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

import io
import logging
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .headless_values import (
    _headless_winner_rank_score,
    _safe_filename_token,
)

logger = logging.getLogger("DecayCore")

























































@dataclass
class ProgressSink:
    messages: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    progress_value: float = 0.0

    def info(self, msg: str) -> None:
        self.messages.append(str(msg))
        logger.info(str(msg))

    def warning(self, msg: str) -> None:
        self.warnings.append(str(msg))
        logger.warning(str(msg))

    def error(self, msg: str) -> None:
        self.errors.append(str(msg))
        logger.error(str(msg))

    def progress(self, value: float, msg: str = "") -> None:
        try:
            self.progress_value = float(np.clip(float(value), 0.0, 1.0))
        except (

            AttributeError,
            TypeError,
            ValueError,
            KeyError,
            IndexError,
            RuntimeError,
            OSError,
            ImportError,
            ModuleNotFoundError,
            NameError,
        ):
            self.progress_value = 0.0
        if msg:
            self.info(msg)

class ConsoleProgressSink(ProgressSink):
    pass

class _HeadlessCallbacks:
    def __init__(self, sink: ProgressSink, started_at: float) -> None:
        self._sink = sink
        self._started_at = float(started_at)

    def status(self, msg: str) -> None:
        elapsed = max(0.0, time.perf_counter() - self._started_at)
        self._sink.info(f"{msg} | {elapsed:.1f} s")

    def set_auto_selected_bar(self, msg: Any = "") -> None:
        if str(msg or "").strip():
            self._sink.info(str(msg))

class _HeadlessBridge:
    def __init__(self, output_dir: Path, sink: ProgressSink, *, no_plots: bool = True) -> None:
        self.output_dir = Path(output_dir)
        self.sink = sink
        self.no_plots = bool(no_plots)

    def ensure_progress_bar(self) -> None:
        self.sink.progress(0.0)

    def set_progress(self, value: float) -> None:
        self.sink.progress(value)

    def toast_health_gate_result(self, hr: Any, mode: str) -> bool:
        blocked = False
        try:
            blocked = bool(getattr(hr, "blocked", False) or getattr(hr, "should_block", False))
        except (

            AttributeError,
            TypeError,
            ValueError,
            KeyError,
            IndexError,
            RuntimeError,
            OSError,
            ImportError,
            ModuleNotFoundError,
            NameError,
        ):
            blocked = False
        if blocked:
            self.sink.error(f"Health gate blocked {mode} run: {hr}")
        return bool(blocked)

    def render_results(self, *args: Any, **kwargs: Any) -> None:
        return None

    def build_export_zip(self, **kwargs: Any) -> tuple[io.BytesIO, dict]:
        from .headless_export_bundle import _build_headless_export_zip

        return _build_headless_export_zip(**kwargs)

    def save_export_bundle(self, zip_buffer: io.BytesIO, **kwargs: Any) -> tuple[str, str, str]:
        from .headless_export_bundle import _save_headless_export_bundle

        return _save_headless_export_bundle(zip_buffer, output_dir=self.output_dir, **kwargs)

    def make_callbacks(self, run_started_at: float) -> _HeadlessCallbacks:
        return _HeadlessCallbacks(self.sink, run_started_at)

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def _git_commit() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[3]),
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        if proc.returncode == 0:
            out = proc.stdout.strip()
            return out or None
    except (

        AttributeError,
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        RuntimeError,
        OSError,
        ImportError,
        ModuleNotFoundError,
        NameError,
    ):
        return None
    return None

__all__ = [
    'ProgressSink',
    'ConsoleProgressSink',
    '_HeadlessCallbacks',
    '_HeadlessBridge',
    '_utc_now',
    '_git_commit',
    '_safe_filename_token',
    '_headless_winner_rank_score',
]
