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

def _copy_export_artifacts(saved_filters_dir: str | None, output_dir: Path) -> None:
    if not saved_filters_dir:
        return
    src = Path(saved_filters_dir)
    if not src.exists() or src.resolve() == output_dir.resolve():
        return
    for item in src.iterdir():
        dest = output_dir / item.name
        if item.is_file() and not dest.exists():
            shutil.copy2(item, dest)

def run_batch(config: dict, output_dir: Path, headless: bool = True) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sink = config.get("_progress_sink") if isinstance(config.get("_progress_sink"), ProgressSink) else ConsoleProgressSink()
    started_at = _utc_now()
    t0 = time.perf_counter()
    ctx: dict | None = None
    data = dict(config)
    metadata = dict(config.get("_metadata", {}) or {})
    rt60 = dict(config.get("_rt60", {}) or {})
    harmonics = dict(config.get("_harmonics", {}) or {})
    status = "failed"
    errors: list[str] = []
    saved_filters_dir: str | None = None

    try:
        required = ["local_path_l", "local_path_r"]
        if bool(data.get("bass_integration_enable", False)):
            required = ["local_path_l_main", "local_path_r_main", "local_path_l_sub", "local_path_r_sub"]
        missing = [key for key in required if not str(data.get(key, "") or "").strip() or not Path(str(data.get(key))).exists()]
        if missing:
            sink.warning("Missing input files for DSP run: " + ", ".join(missing))
            status = "partial"
            return _build_metrics(status, data, None, metadata, rt60, harmonics, started_at=started_at, finished_at=_utc_now(), runtime_s=time.perf_counter() - t0, warnings=sink.warnings, errors=errors)

        bridge = _HeadlessBridge(output_dir, sink, no_plots=bool(data.get("_no_plots", True)))
        support = ProcessRunSupport(
            version=str(VERSION),
            max_safe_boost=12.0,
            force_single_plot_fs_hz=int(data.get("fs", 48000) or 48000),
            auto_target_mode_norm=auto_target_mode_norm,
            auto_target_selection_method_text=auto_target_selection_method_text,
            pick_target_curve_label=pick_target_curve_label,
            slugify_filename_token=slugify_filename_token,
            has_uploaded_target_file=has_uploaded_target_file,
            ui_bridge=bridge,
        )
        request = RunRequest(raw_ui_data=data, run_started_at=t0)
        callbacks = bridge.make_callbacks(t0)
        ctx = _prepare_ui_and_measurements(request=request, callbacks=callbacks, support=support)
        if ctx is None:
            raise RuntimeError("measurement loading or health gate failed")
        _run_auto_mode_seed_phases(ctx, callbacks=callbacks, support=support)
        _prepare_target_curve_and_run_context(ctx, support=support, callbacks=callbacks)
        _run_auto_mode_search_if_needed(ctx, callbacks=callbacks, support=support)
        if not _run_pipeline(ctx, callbacks=callbacks, support=support):
            status = "partial"
            raise RuntimeError("filter generation produced no results")
        _finalize_run_outputs(ctx, callbacks=callbacks, support=support)
        data = dict(ctx.get("data", data) or {})
        saved_filters_dir = str(ctx.get("saved_filters_dir", "") or "")
        status = "success"
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
        logger.exception("Headless batch run failed")
        if status != "partial":
            status = "failed"
    finally:
        finished_at = _utc_now()
        metrics_doc = _build_metrics(
            status,
            data,
            ctx,
            metadata,
            rt60,
            harmonics,
            started_at=started_at,
            finished_at=finished_at,
            runtime_s=time.perf_counter() - t0,
            warnings=sink.warnings,
            errors=errors + sink.errors,
        )
        _write_outputs(output_dir, metrics_doc)
        _copy_export_artifacts(saved_filters_dir, output_dir)
    return metrics_doc

def prepare_headless_config(config_path: Path, output_dir: Path, *, no_plots: bool = True) -> dict:
    config_path = Path(config_path).resolve()
    raw = _read_json(config_path)
    data = _normalize_headless_config(raw, config_dir=config_path.parent, output_dir=output_dir)
    metadata, rt60, harmonics = _load_optional_metadata(config_path.parent, raw)
    data["_config_path"] = str(config_path)
    data["_metadata"] = metadata
    data["_rt60"] = rt60
    data["_harmonics"] = harmonics
    data["_no_plots"] = bool(no_plots)
    return data


__all__ = ['_copy_export_artifacts', 'run_batch', 'prepare_headless_config']


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
