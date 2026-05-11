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
import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
import time
import traceback
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ...application.run_request import RunRequest
from ...auto_mode.api import AUTO_MODE_COMPAT_VERSION
from ...auto_mode.rank_score import attach_official_rank_score, official_rank_score
from ...config.decaycore_config import load_config
from ...version import VERSION
from ...workflow.auto_flow import _run_auto_mode_search_if_needed, _run_auto_mode_seed_phases
from ...workflow.pipeline_flow import _run_pipeline
from ...workflow.process_run_flow import ProcessRunSupport
from ...workflow.process_support import (
    auto_target_mode_norm,
    auto_target_selection_method_text,
    has_uploaded_target_file,
    pick_target_curve_label,
    slugify_filename_token,
)
from ...workflow.run_finalize import _finalize_run_outputs
from ...workflow.run_prepare import _prepare_target_curve_and_run_context, _prepare_ui_and_measurements

logger = logging.getLogger("DecayCore")
























































__all__ = [
    "ConsoleProgressSink",
    "ProgressSink",
    "prepare_headless_config",
    "run_batch",
]

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
        except Exception:
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
        except Exception:
            blocked = False
        if blocked:
            self.sink.error(f"Health gate blocked {mode} run: {hr}")
        return bool(blocked)

    def render_results(self, *args: Any, **kwargs: Any) -> None:
        return None

    def build_export_zip(self, **kwargs: Any) -> tuple[io.BytesIO, dict, dict]:
        return _build_headless_export_zip(**kwargs)

    def save_export_bundle(self, zip_buffer: io.BytesIO, **kwargs: Any) -> tuple[str, str, str]:
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
    except Exception:
        return None
    return None

def _safe_filename_token(value: Any, default: str = "v0") -> str:
    raw = str(value or "").strip()
    if not raw:
        return default
    out = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in raw).strip(".-_")
    return out or default

def _headless_winner_rank_score(data: dict | None) -> float:
    try:
        if not bool((data or {}).get("camillafir_automatic_mode", False)):
            return float("nan")
        auto_meta = dict((data or {}).get("_auto_mode_meta", {}) or {})
        best_metrics = attach_official_rank_score(auto_meta.get("best_metrics", {}))
        return float(official_rank_score(best_metrics))
    except Exception:
        return float("nan")


__all__ = ['ProgressSink', 'ConsoleProgressSink', '_HeadlessCallbacks', '_HeadlessBridge', '_utc_now', '_git_commit', '_safe_filename_token', '_headless_winner_rank_score']


def _load_sibling_symbols() -> None:
    import importlib
    package = __package__
    for module_name in ['runner_01', 'runner_02', 'runner_03', 'runner_04']:
        if module_name == __name__.rsplit('.', 1)[-1]:
            continue
        module = importlib.import_module(f"{package}.{module_name}")
        for symbol in getattr(module, "__all__", ()):
            globals().setdefault(symbol, getattr(module, symbol))


_load_sibling_symbols()
