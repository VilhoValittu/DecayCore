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

import numpy as np

from ..correction_types import _MagPostProcessInputs, _MagPostProcessOutputs
from ..dsp_config import CfgReader
from ..dsp_telemetry import safe_put_many
from ..gain_policy import resolve_gain_policy
from ..mag_authority_trace import (
    MAG_AUTHORITY_TRACE_VERSION,
    REASON_LOW_BASS_CUTS_ONLY,
    REASON_LOW_BASS_FLOOR_REAPPLIED,
    append_mag_authority_stage,
    summarize_mag_authority_trace,
)
from ..mag_telemetry import (
    _record_stage_probe,
    _summarize_correction_metrics,
)

from .authority import (
    _apply_acoustic_authority_caps,
    _apply_candidate_metrics,
)
from .low_frequency import (
    _apply_low_frequency_policy,
    _prepare_boost_caps,
)
from .metrics import _store_realized_pre_ir_metrics


from .stages import (
    _run_bass_boost_restore_stage,
    _run_excursion_protection_stage,
    _run_hardclamp_stage,
    _run_lowbass_hard_reapply_stage,
    _run_mid_refit_stage,
    _run_min_boost_peak_stage,
    _run_regularization_stage,
    _run_slope_confpull_stage,
    _run_softclip_stage,
    _run_transition_fade_stage,
    _run_wav_final_polish_stage,
    _run_wav_transition_smooth_stage,
    _suppress_subthreshold_local_detail_lobes,  # noqa: F401 - compatibility alias
)


def apply_post_limits_and_metrics(
    inputs: _MagPostProcessInputs,
    *,
    apply_mid_refit_pre_slope,
) -> _MagPostProcessOutputs:
    """Ajaa mag-korjauksen loppuvaiheen: low-bass policy, clamping, slope/fade ja metriikat."""
    cfg = inputs.cfg
    cfg_reader = CfgReader(cfg)
    freq_axis = inputs.freq_axis
    st = inputs.st
    logger = inputs.logger
    _stage_probe = inputs.stage_probe
    _cfg_float_allow_zero = inputs.cfg_float_allow_zero
    mask_c = inputs.mask_c
    gain_db = inputs.gain_db
    gain_apply = inputs.gain_apply
    raw_g = inputs.raw_g
    final_g = inputs.final_g
    pre_bass_adapt_g = inputs.pre_bass_adapt_g
    raw_safe_ref = inputs.raw_safe_ref
    conf_mask = inputs.conf_mask
    _filter_smooth = inputs.filter_smooth
    debug_stage_stats = inputs.debug_stage_stats
    stage_probes = dict(inputs.stage_probes)
    # Presolve hylkaa st:n eika jalki paady mihinkaan; None poistaa
    # vaihekohtaisen analyysin rakentamisen (append on silloin no-op).
    mag_authority_trace: list[dict[str, object]] | None = None if bool(inputs.presolve_mode) else []
    apply_confidence_weighted_target_pull = inputs.apply_confidence_weighted_target_pull

    append_mag_authority_stage(
        mag_authority_trace,
        "post_input",
        gain_apply,
        gain_apply,
        freq_axis,
        mask_c,
    )

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

    gain_policy = resolve_gain_policy(cfg, cfg_float_allow_zero_fn=_cfg_float_allow_zero)
    low_cut_enable = bool(gain_policy.low_cut_enable)
    low_hz = float(gain_policy.low_cut_hz)
    low_cut_strength = float(gain_policy.low_cut_strength)
    _pre_lowbass_policy = np.asarray(gain_apply, dtype=float).copy()
    low_cut_floor_ref = _apply_low_frequency_policy(
        freq_axis=freq_axis,
        mask_c=mask_c,
        gain_apply=gain_apply,
        raw_g=raw_g,
        final_g=final_g,
        gain_db=gain_db,
        gain_policy=gain_policy,
        stage_probes=stage_probes,
        stage_probe_fn=_stage_probe,
        cfg=cfg,
        logger=logger,
    )
    _lowbass_reasons = []
    if bool(low_cut_enable):
        _lowbass_reasons.append(REASON_LOW_BASS_CUTS_ONLY)
        if low_cut_floor_ref is not None:
            _lowbass_reasons.append(REASON_LOW_BASS_FLOOR_REAPPLIED)
    append_mag_authority_stage(
        mag_authority_trace,
        "after_lowbass_policy",
        _pre_lowbass_policy,
        gain_apply,
        freq_axis,
        mask_c,
        reason_codes=_lowbass_reasons,
    )
    try:
        logger.info(
            f"CFG CHECK: conf_pull_floor={cfg_reader.float('conf_pull_floor', 0.05)}, "
            f"gamma_cut={cfg_reader.float('conf_pull_gamma_cut', 0.45)}, "
            f"low_bass_cut_strength={cfg_reader.float_allow_zero('low_bass_cut_strength', 0.0)}"
        )
    except (AttributeError, TypeError, ValueError, FloatingPointError, IndexError):
        pass

    caps = _prepare_boost_caps(
        cfg=cfg,
        cfg_reader=cfg_reader,
        st=st,
        logger=logger,
        freq_axis=freq_axis,
        mask_c=mask_c,
        conf_mask=conf_mask,
        gain_db=gain_db,
        gain_policy=gain_policy,
    )
    max_cut_db = float(caps["max_cut_db"])
    max_boost_db_base = float(caps["max_boost_db_base"])
    boost_cap_db = np.asarray(caps["boost_cap_db"], dtype=float)
    bass_boost_cap_hz = float(caps["bass_boost_cap_hz"])
    bass_boost_post_restore_enable = bool(caps["bass_boost_post_restore_enable"])
    bass_boost_post_restore_strength = float(caps["bass_boost_post_restore_strength"])
    authority_caps = _apply_acoustic_authority_caps(
        cfg=cfg,
        cfg_reader=cfg_reader,
        st=st,
        logger=logger,
        freq_axis=freq_axis,
        mask_c=mask_c,
        boost_cap_db=boost_cap_db,
        max_cut_db=max_cut_db,
    )
    boost_cap_db = np.asarray(authority_caps["boost_cap_db"], dtype=float)
    cut_cap_db = np.asarray(authority_caps["cut_cap_db"], dtype=float)
    authority_boost_reduced_bins = int(authority_caps.get("boost_reduced_bins", 0) or 0)
    authority_cut_reduced_bins = int(authority_caps.get("cut_reduced_bins", 0) or 0)

    (
        boost_cand_peak,
        boost_cand_min_hz,
        n_boost_cand,
        n_boost_cand_low,
        n_boost_cand_exc,
    ) = _apply_candidate_metrics(
        gain_apply=gain_apply,
        gain_db=gain_db,
        freq_axis=freq_axis,
        mask_c=mask_c,
        gain_policy=gain_policy,
    )

    gain_db, over_boost, over_cut, softclip_boost_bins, softclip_cut_bins = _run_softclip_stage(
        gain_apply=gain_apply,
        gain_db=gain_db,
        freq_axis=freq_axis,
        mask_c=mask_c,
        cfg=cfg,
        max_cut_db=max_cut_db,
        max_boost_db_base=max_boost_db_base,
        boost_cap_db=boost_cap_db,
        cut_cap_db=cut_cap_db,
        stage_probes=stage_probes,
        stage_probe_fn=_stage_probe,
        logger=logger,
        authority_boost_reduced_bins=authority_boost_reduced_bins,
        authority_cut_reduced_bins=authority_cut_reduced_bins,
        mag_authority_trace=mag_authority_trace,
    )

    gain_db = _run_mid_refit_stage(
        gain_db=gain_db,
        freq_axis=freq_axis,
        mask_c=mask_c,
        m_anal=inputs.m_anal,
        target_mags=inputs.target_mags,
        calc_offset_db=inputs.calc_offset_db,
        conf_mask=conf_mask,
        cfg=cfg,
        st=st,
        logger=logger,
        debug_stage_stats=debug_stage_stats,
        apply_mid_refit_pre_slope=apply_mid_refit_pre_slope,
        mag_authority_trace=mag_authority_trace,
    )

    gain_db = _run_slope_confpull_stage(
        gain_db=gain_db,
        freq_axis=freq_axis,
        mask_c=mask_c,
        cfg=cfg,
        st=st,
        raw_safe_ref=raw_safe_ref,
        conf_mask=conf_mask,
        logger=logger,
        debug_stage_stats=debug_stage_stats,
        apply_confidence_weighted_target_pull=apply_confidence_weighted_target_pull,
        mag_authority_trace=mag_authority_trace,
    )

    gain_db = _run_regularization_stage(
        gain_db=gain_db,
        freq_axis=freq_axis,
        mask_c=mask_c,
        cfg_reader=cfg_reader,
        filter_smooth=_filter_smooth,
        logger=logger,
        debug_stage_stats=debug_stage_stats,
        mag_authority_trace=mag_authority_trace,
    )

    gain_db = _run_excursion_protection_stage(
        gain_db=gain_db,
        freq_axis=freq_axis,
        mask_c=mask_c,
        gain_policy=gain_policy,
        max_boost_db_base=max_boost_db_base,
        logger=logger,
        mag_authority_trace=mag_authority_trace,
    )

    gain_db = _run_wav_transition_smooth_stage(
        gain_db=gain_db,
        freq_axis=freq_axis,
        mask_c=mask_c,
        cfg_reader=cfg_reader,
        st=st,
        logger=logger,
        debug_stage_stats=debug_stage_stats,
        mag_authority_trace=mag_authority_trace,
    )

    try:
        if "after_slope" not in stage_probes:
            _record_stage_probe(stage_probes, "after_slope", _stage_probe, freq_axis, gain_db, mask_c, cfg, logger)
    except (TypeError, ValueError, FloatingPointError, IndexError):
        pass

    gain_db, hardclamp_boost_bins, hardclamp_cut_bins, hard_over_boost, hard_over_cut, clamp_dominance_level = (
        _run_hardclamp_stage(
            gain_db=gain_db,
            freq_axis=freq_axis,
            mask_c=mask_c,
            cfg=cfg,
            st=st,
            logger=logger,
            boost_cap_db=boost_cap_db,
            cut_cap_db=cut_cap_db,
            max_cut_db=max_cut_db,
            max_boost_db_base=max_boost_db_base,
            filter_smooth=_filter_smooth,
            debug_stage_stats=debug_stage_stats,
            stage_probes=stage_probes,
            stage_probe_fn=_stage_probe,
            authority_boost_reduced_bins=authority_boost_reduced_bins,
            authority_cut_reduced_bins=authority_cut_reduced_bins,
            mag_authority_trace=mag_authority_trace,
        )
    )

    gain_db = _run_wav_final_polish_stage(
        gain_db=gain_db,
        freq_axis=freq_axis,
        mask_c=mask_c,
        cfg_reader=cfg_reader,
        st=st,
        cfg=cfg,
        max_cut_db=max_cut_db,
        boost_cap_db=boost_cap_db,
        cut_cap_db=cut_cap_db,
        logger=logger,
        debug_stage_stats=debug_stage_stats,
        mag_authority_trace=mag_authority_trace,
    )

    gain_db = _run_bass_boost_restore_stage(
        gain_db=gain_db,
        freq_axis=freq_axis,
        mask_c=mask_c,
        bass_boost_post_restore_enable=bass_boost_post_restore_enable,
        bass_boost_post_restore_strength=bass_boost_post_restore_strength,
        low_hz=low_hz,
        bass_boost_cap_hz=bass_boost_cap_hz,
        gain_apply=gain_apply,
        boost_cap_db=boost_cap_db,
        cut_cap_db=cut_cap_db,
        cfg=cfg,
        max_cut_db=max_cut_db,
        st=st,
        logger=logger,
        debug_stage_stats=debug_stage_stats,
        mag_authority_trace=mag_authority_trace,
    )

    gain_db = _run_lowbass_hard_reapply_stage(
        gain_db=gain_db,
        freq_axis=freq_axis,
        mask_c=mask_c,
        low_cut_enable=low_cut_enable,
        low_hz=low_hz,
        gain_policy=gain_policy,
        low_cut_floor_ref=low_cut_floor_ref,
        low_cut_strength=low_cut_strength,
        st=st,
        logger=logger,
        mag_authority_trace=mag_authority_trace,
    )

    gain_db = _run_min_boost_peak_stage(
        gain_db=gain_db,
        freq_axis=freq_axis,
        mask_c=mask_c,
        cfg_reader=cfg_reader,
        st=st,
        mag_authority_trace=mag_authority_trace,
    )

    gain_db = _run_transition_fade_stage(
        gain_db=gain_db,
        freq_axis=freq_axis,
        mask_c=mask_c,
        cfg_reader=cfg_reader,
        st=st,
        mag_authority_trace=mag_authority_trace,
    )

    append_mag_authority_stage(
        mag_authority_trace,
        "final",
        gain_db,
        gain_db,
        freq_axis,
        mask_c,
    )
    try:
        if mag_authority_trace is not None and isinstance(st, dict):
            trace_summary = summarize_mag_authority_trace(mag_authority_trace)
            safe_put_many(
                st,
                {
                    "mag_authority_trace": list(mag_authority_trace),
                    "mag_authority_trace_version": int(MAG_AUTHORITY_TRACE_VERSION),
                    **trace_summary,
                },
            )
    except (TypeError, ValueError):
        pass
    _record_stage_probe(stage_probes, "after_fade", _stage_probe, freq_axis, gain_db, mask_c, cfg, logger)
    _store_realized_pre_ir_metrics(
        st=st,
        cfg_reader=cfg_reader,
        freq_axis=freq_axis,
        mask_c=mask_c,
        gain_db=gain_db,
        pre_bass_adapt_g=pre_bass_adapt_g,
        m_anal=inputs.m_anal,
        target_mags=inputs.target_mags,
        calc_offset_db=inputs.calc_offset_db,
    )

    try:
        summary = _summarize_correction_metrics(
            gain_db,
            freq_axis,
            cfg,
            st,
            mask_c,
            logger,
            boost_cand_peak=boost_cand_peak,
            n_boost_cand=n_boost_cand,
            n_boost_cand_low=n_boost_cand_low,
            n_boost_cand_exc=n_boost_cand_exc,
        )
        boost_peak_db = float(summary["boost_peak_db"])
        cut_peak_db = float(summary["cut_peak_db"])
        n_boost = int(summary["n_boost"])
    except (TypeError, ValueError, KeyError):
        boost_peak_db, cut_peak_db, n_boost = 0.0, 0.0, 0

    return _MagPostProcessOutputs(
        gain_db=np.asarray(gain_db, dtype=float),
        stage_probes=dict(stage_probes),
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
    )


__all__ = ["apply_post_limits_and_metrics"]
