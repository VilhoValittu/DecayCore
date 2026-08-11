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

import logging
import time
import typing

from ..application.run_contracts import DSPRunInput, RunContext, build_dsp_measurements
from ..common.filter_lengths import scale_taps_with_fs
from ..engine import build_config, run_pipeline, summarize_run
from ..resources.i8n.decaycore_i18n import t
from .bridge_types import ProcessRunCallbacks

if typing.TYPE_CHECKING:
    from .process_run_support import ProcessRunSupport

logger = logging.getLogger("DecayCore")

_PIPELINE_PROGRESS_START = 0.20
_PIPELINE_PROGRESS_START_AUTO = 0.88
_PIPELINE_PROGRESS_END = 0.96


def _run_pipeline(ctx: RunContext, *, callbacks: ProcessRunCallbacks, support: ProcessRunSupport) -> bool:
    resolved = ctx.require_resolved_config()
    data = resolved.resolved_data
    measurements = resolved.measurements
    target_rates = resolved.target_rates
    xos = resolved.xos
    hpf = resolved.hpf
    hc_f = resolved.hc_f
    hc_m = resolved.hc_m
    taps_base = int(ctx.taps_base)
    dash_fs = int(resolved.dash_fs)
    results_by_fs = ctx.results_by_fs
    perf_stats = ctx.perf_stats
    per_fs_stats = ctx.per_fs_stats
    pipeline_start = (
        _PIPELINE_PROGRESS_START_AUTO
        if ctx.auto_mode_enabled
        else _PIPELINE_PROGRESS_START
    )
    pipeline_span = float(max(0.0, _PIPELINE_PROGRESS_END - pipeline_start))

    for index, fs_v in enumerate(target_rates):
        if bool(data.get("multi_rate_opt", False)):
            taps_v = scale_taps_with_fs(fs_v, base_taps=taps_base)
            logger.info(f"Auto taps: {int(fs_v)} Hz -> {int(taps_v)} taps (ref 44100 Hz -> {taps_base} taps)")
        else:
            taps_v = taps_base

        callbacks.status(f"{t('stat_calc')} {fs_v}Hz...")
        support.ui_bridge.set_progress(
            float(pipeline_start) + float(pipeline_span) * (float(index) / float(max(1, len(target_rates))))
        )
        data["enable_residual_pass"] = bool(data.get("enable_residual_pass", False))

        cfg = build_config(
            data,
            fs_v=int(fs_v),
            taps_v=int(taps_v),
            xos=xos,
            hpf=hpf,
            hc_f=hc_f,
            hc_m=hc_m,
            max_safe_boost=float(support.max_safe_boost),
        )
        try:
            cfg.bass_smooth_w_gamma = float(data.get("bass_smooth_w_gamma", 2.4))
            cfg.bass_smooth_w_max = float(data.get("bass_smooth_w_max", 0.45))
        except Exception:
            logger.exception("bass smooth weight attr set")

        callbacks.status(f"{t('stat_calc')} {int(fs_v)} Hz")
        dsp_input = DSPRunInput(
            config=cfg,
            measurements=build_dsp_measurements(data, measurements),
            fs_v=int(fs_v),
            taps_v=int(taps_v),
        )
        dsp_started_at = time.perf_counter()
        result = run_pipeline(dsp_input.config, dsp_input.measurements)
        result.metrics["summary"] = summarize_run(result)
        dsp_elapsed = max(0.0, float(time.perf_counter() - dsp_started_at))

        perf_stats["dsp_s"] += dsp_elapsed
        fs_key = int(fs_v)
        slot = per_fs_stats.setdefault(fs_key, {})
        slot["dsp_s"] = float(slot.get("dsp_s", 0.0)) + dsp_elapsed
        results_by_fs.append(result)

        if int(fs_v) == dash_fs:
            ctx.l_st_f = result.l_st
            ctx.r_st_f = result.r_st
            ctx.l_imp_f = result.l_ir
            ctx.r_imp_f = result.r_ir
            ctx.sub_ir_f = result.sub_ir
            ctx.sub_st_f = result.sub_st
            ctx.sub_meas_f = {
                k: result.measurements[k]
                for k in ("f_sub", "m_sub", "p_sub")
                if k in result.measurements
            }

    if not results_by_fs:
        return False
    return True
