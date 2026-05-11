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

from typing import Any, Callable

import numpy as np
import scipy.ndimage

from decaycore.auto_mode.auto_mode_profile import profiled_section

from .. import bassfirst as bf
from ..decaycore_analysis import _sigma_bins_from_hz
from ..correction_types import (
    _MagAdaptiveStageOutputs,
    _MagCoreOutputs,
    _MagCorrectionContext,
    _MagPipelineInputs,
    _MagPostProcessInputs,
    _MagPostProcessOutputs,
    _MagRawStageOutputs,
)
from ..dsp_config import CfgReader
from ..dsp_telemetry import safe_put_many
from ..gain_policy import apply_cuts_only_guard, build_low_frequency_guard_mask, resolve_gain_policy
from ..mag_limits import (
    _apply_hard_boost_cut_clamp,
    _apply_max_boost_cut,
    _apply_slope_limits,
    _blend_masked_fractional_octave,
)
from ..mag_post_limits import apply_post_limits_and_metrics as _apply_post_limits_and_metrics_impl
from ..mag_postprocess import apply_bass_boost_post_restore, apply_confpull_post_slope
from ..mag_shape import (
    _apply_confidence_logic,
    _apply_regularization,
    _compute_error_db,
    _error_to_correction_mag,
    _resolve_filter_smooth,
    _select_active_band,
)
from ..mag_stage import (
    run_mag_bassfirst_afdw_conf_stage,
    run_mag_core_stage,
    run_mag_raw_stage,
)
from ..mag_telemetry import (
    _log_stage_stats,
    _record_stage_probe,
    _summarize_correction_metrics,
)
from ..phase_ir_utils import _cosine_fade_out_01
from ..smoothing import (
    AFDW_BW_MAX_OCT,
    AFDW_BW_MIN_OCT,
    apply_adaptive_fdw,
    psycho_smooth_safe_gain,
    smooth_gain_fractional_octave,
)

def _run_mag_raw_stage(inputs: _MagPipelineInputs) -> _MagRawStageOutputs:
    return run_mag_raw_stage(
        inputs,
        apply_peak_priority_error_shaping=_apply_peak_priority_error_shaping,
        apply_smoothing=_apply_smoothing,
    )

def _run_mag_bassfirst_afdw_conf_stage(
    inputs: _MagPipelineInputs,
    raw_stage: _MagRawStageOutputs,
) -> _MagAdaptiveStageOutputs:
    return run_mag_bassfirst_afdw_conf_stage(
        inputs,
        raw_stage,
        select_bass_adaptive_conf_mask=_select_bass_adaptive_conf_mask,
        apply_confidence_adaptive_bass_smoothing=_apply_confidence_adaptive_bass_smoothing,
    )

def _run_mag_core_stage(inputs: _MagPipelineInputs) -> _MagCoreOutputs:
    return run_mag_core_stage(
        inputs,
        run_mag_raw_stage_fn=_run_mag_raw_stage,
        run_mag_bassfirst_afdw_conf_stage_fn=_run_mag_bassfirst_afdw_conf_stage,
    )

def _run_mag_correction_pipeline(inputs: _MagPipelineInputs) -> _MagCorrectionContext:
    cfg = inputs.cfg
    freq_axis = inputs.freq_axis
    st = inputs.st
    conf_mask = inputs.conf_mask
    logger = inputs.logger
    _stage_probe = inputs.stage_probe
    _cfg_float_allow_zero = inputs.cfg_float_allow_zero
    apply_confidence_weighted_target_pull = inputs.apply_confidence_weighted_target_pull

    with profiled_section("generate_filter.correction.mag_core"):
        core = _run_mag_core_stage(inputs)

    afdw_on = core.afdw_on
    base_sigma = core.base_sigma
    _filter_smooth = core.filter_smooth
    df_mode = core.df_mode
    raw_g = core.raw_g
    final_g = core.final_g
    mask_c = core.mask_c
    stage_probes = core.stage_probes
    use_bassfirst = core.use_bassfirst
    bf_room_mode = core.bf_room_mode
    bf_rel = core.bf_rel
    bf_conf_for_smoothing = core.bf_conf_for_smoothing
    gain_db = core.gain_db

    boost_peak_db = 0.0
    cut_peak_db = 0.0
    n_boost = 0
    boost_cand_peak = 0.0
    boost_cand_min_hz = float("nan")
    n_boost_cand = 0
    n_boost_cand_low = 0
    n_boost_cand_exc = 0
    softclip_boost_bins = 0
    softclip_cut_bins = 0
    over_boost = 0.0
    over_cut = 0.0
    hardclamp_boost_bins = 0
    hardclamp_cut_bins = 0
    hard_over_boost = 0.0
    hard_over_cut = 0.0
    clamp_dominance_level = "NONE"

    if core.mag_enabled:
        with profiled_section("generate_filter.correction.mag_post"):
            post = _apply_post_limits_and_metrics(
                _MagPostProcessInputs(
                    cfg=cfg,
                    freq_axis=freq_axis,
                    st=st,
                    logger=logger,
                    stage_probe=_stage_probe,
                    cfg_float_allow_zero=_cfg_float_allow_zero,
                    mask_c=mask_c,
                    gain_db=gain_db,
                    gain_apply=core.gain_apply,
                    raw_g=raw_g,
                    final_g=final_g,
                    pre_bass_adapt_g=(
                        None
                        if core.pre_bass_adapt_g is None
                        else np.asarray(core.pre_bass_adapt_g, dtype=float)
                    ),
                    raw_safe_ref=core.raw_safe_ref,
                    conf_mask=conf_mask,
                    filter_smooth=_filter_smooth,
                    debug_stage_stats=core.debug_stage_stats,
                    stage_probes=stage_probes,
                    apply_confidence_weighted_target_pull=apply_confidence_weighted_target_pull,
                    m_anal=np.asarray(inputs.m_anal, dtype=float),
                    target_mags=np.asarray(inputs.target_mags, dtype=float),
                    calc_offset_db=float(inputs.calc_offset_db),
                )
            )
        gain_db = post.gain_db
        stage_probes = post.stage_probes
        boost_peak_db = post.boost_peak_db
        cut_peak_db = post.cut_peak_db
        n_boost = post.n_boost
        boost_cand_peak = post.boost_cand_peak
        boost_cand_min_hz = post.boost_cand_min_hz
        n_boost_cand = post.n_boost_cand
        n_boost_cand_low = post.n_boost_cand_low
        n_boost_cand_exc = post.n_boost_cand_exc
        softclip_boost_bins = post.softclip_boost_bins
        softclip_cut_bins = post.softclip_cut_bins
        over_boost = post.over_boost
        over_cut = post.over_cut
        hardclamp_boost_bins = post.hardclamp_boost_bins
        hardclamp_cut_bins = post.hardclamp_cut_bins
        hard_over_boost = post.hard_over_boost
        hard_over_cut = post.hard_over_cut
        clamp_dominance_level = post.clamp_dominance_level

    return _MagCorrectionContext(
        afdw_on=bool(afdw_on),
        base_sigma=base_sigma,
        filter_smooth=_filter_smooth,
        df_mode=df_mode,
        raw_g=raw_g,
        final_g=final_g,
        mask_c=np.asarray(mask_c, dtype=bool),
        stage_probes=dict(stage_probes),
        use_bassfirst=bool(use_bassfirst),
        bf_room_mode=bf_room_mode,
        bf_rel=bf_rel,
        bf_conf_for_smoothing=bf_conf_for_smoothing,
        boost_peak_db=float(boost_peak_db),
        cut_peak_db=float(cut_peak_db),
        n_boost=int(n_boost),
        boost_cand_peak=float(boost_cand_peak),
        boost_cand_min_hz=float(boost_cand_min_hz),
        n_boost_cand=int(n_boost_cand),
        n_boost_cand_low=int(n_boost_cand_low),
        n_boost_cand_exc=int(n_boost_cand_exc),
        softclip_boost_bins=int(softclip_boost_bins),
        softclip_cut_bins=int(softclip_cut_bins),
        over_boost=float(over_boost),
        over_cut=float(over_cut),
        hardclamp_boost_bins=int(hardclamp_boost_bins),
        hardclamp_cut_bins=int(hardclamp_cut_bins),
        hard_over_boost=float(hard_over_boost),
        hard_over_cut=float(hard_over_cut),
        clamp_dominance_level=str(clamp_dominance_level),
        gain_db=np.asarray(gain_db, dtype=float),
    )


__all__ = ['_run_mag_raw_stage', '_run_mag_bassfirst_afdw_conf_stage', '_run_mag_core_stage', '_run_mag_correction_pipeline']


def _load_sibling_symbols() -> None:
    import importlib
    package = __package__
    for module_name in ['correction_mag_01', 'correction_mag_02']:
        if module_name == __name__.rsplit('.', 1)[-1]:
            continue
        module = importlib.import_module(f"{package}.{module_name}")
        for symbol in getattr(module, "__all__", ()):
            globals().setdefault(symbol, getattr(module, symbol))


_load_sibling_symbols()
