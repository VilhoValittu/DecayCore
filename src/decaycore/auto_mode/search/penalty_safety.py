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

import inspect
import logging
import math
import sys

import numpy as np

_logger = logging.getLogger(__name__)

from ...common.acoustic_stats import calc_acoustic_score, calc_ai_summary_from_stats
from ...config.models import StereoAutoPolicyConfig, StereoResolvedAutoPolicies
from ...dsp.quality_metrics import (
    band_lr_mismatch_change_from_stats,
    band_lr_mismatch_rms_from_stats,
    normalized_policy_divergence_score,
    worst_channel_relief_db,
)
from ...dsp.modal_analysis import ModalAnalysisResult, RoomModeEvent, detect_room_modes
from ...dsp.smoothing import smooth_gain_fractional_octave
from ...dsp.target_match import target_match_from_stats
from .. import shared
from ..rank_score import (
    OFFICIAL_RANK_SCORE_CONTEXT,
    attach_official_rank_score,
    calibrated_auto_quality,
    compute_rank_score_components,
)
from ..runtime_context import (
    _auto_collect_reflections,
    _auto_event_penalty_weighted,
    _auto_event_severity,
    _auto_get_top_modes_hz,
    _auto_get_worst_mode_hz,
    _auto_mode_band,
    _auto_pick_metric,
)

CORRECTION_SHARPNESS_SCORING_VERSION = 1
DIP_FILL_RISK_SCORING_VERSION = 1
CHANNEL_OVERFIT_SCORING_VERSION = 1

def _auto_tdc_overreach_penalty(st: dict | None) -> tuple[float, dict]:
    st = st or {}
    dbg = {}
    if not bool(st.get("tdc_applied", False)):
        return 0.0, {
            "tdc_applied": False,
            "tdc_peak_reduction_db": 0.0,
            "tdc_band_width_oct": 0.0,
            "tdc_reduction_area_db_hz": 0.0,
        }

    peak_db = shared._auto_safe_float(st.get("tdc_peak_reduction_db", 0.0), 0.0)
    band_lo = shared._auto_safe_float(st.get("tdc_reduction_band_low_hz", 0.0), 0.0)
    band_hi = shared._auto_safe_float(st.get("tdc_reduction_band_high_hz", 0.0), 0.0)
    area_db_hz = shared._auto_safe_float(st.get("tdc_reduction_area_db_hz", 0.0), 0.0)
    events_used = shared._auto_safe_float(st.get("tdc_events_used", 0.0), 0.0)

    band_width_oct = 0.0
    if np.isfinite(band_lo) and np.isfinite(band_hi) and band_lo > 0.0 and band_hi > band_lo:
        band_width_oct = float(np.log2(float(band_hi) / float(band_lo)))

    penalty = 0.0
    if np.isfinite(peak_db):
        penalty += 0.20 * max(0.0, float(peak_db) - 7.0)
    if np.isfinite(band_width_oct):
        penalty += 0.35 * max(0.0, float(band_width_oct) - 0.85)
    if np.isfinite(area_db_hz):
        penalty += 0.002 * max(0.0, float(area_db_hz) - 250.0)
    if np.isfinite(events_used) and events_used <= 0.0 and np.isfinite(peak_db) and peak_db > 0.0:
        penalty += 1.0

    dbg["tdc_applied"] = True
    dbg["tdc_peak_reduction_db"] = float(peak_db) if np.isfinite(peak_db) else float("nan")
    dbg["tdc_band_width_oct"] = float(band_width_oct) if np.isfinite(band_width_oct) else float("nan")
    dbg["tdc_reduction_area_db_hz"] = float(area_db_hz) if np.isfinite(area_db_hz) else float("nan")
    dbg["tdc_events_used"] = float(events_used) if np.isfinite(events_used) else float("nan")
    return float(max(0.0, penalty)), dbg


def _auto_dsp_quality_penalty(st: dict | None) -> tuple[float, float, dict]:
    st = st or {}
    penalty = 0.0
    dbg = {}

    real_rms = _auto_pick_metric(
        st,
        (
            "real_mag_error_rms",
        ),
        abs_value=True,
        nonneg=True,
    )
    if real_rms is not None:
        penalty += 6.0 * max(0.0, float(real_rms) - 0.90)
    dbg["real_rms"] = real_rms

    ripple_rms = _auto_pick_metric(
        st,
        (
            "ripple_rms",
        ),
        abs_value=True,
        nonneg=True,
    )
    if ripple_rms is not None:
        penalty += 4.0 * max(0.0, float(ripple_rms) - 0.50)
    dbg["ripple_rms"] = ripple_rms

    gd_grad_max = _auto_pick_metric(
        st,
        (
            "gd_grad_limiter_after_max_ms_per_oct",
            "gd_grad_limiter_before_max_ms_per_oct",
            "gd_limiter_max_grad_ms_per_oct",
            "gd_grad_limiter_max_grad_ms_per_oct",
            "gd_limiter_max_grad_after_ms_per_oct",
            "gd_grad_limiter_max_grad_after_ms_per_oct",
            "gd_limiter_max_grad_before_ms_per_oct",
            "gd_grad_limiter_max_grad_before_ms_per_oct",
        ),
        abs_value=True,
        nonneg=True,
    )
    if gd_grad_max is not None:
        penalty += 1.5 * max(0.0, float(gd_grad_max) - 8.0)
    dbg["gd_grad_max"] = gd_grad_max

    gd_abs_max = _auto_pick_metric(
        st,
        ("gd_abs_max_20_500_ms",),
        abs_value=True,
        nonneg=True,
    )
    if gd_abs_max is not None:
        penalty += 0.12 * max(0.0, float(gd_abs_max) - 20.0)
    dbg["gd_abs_max_20_500_ms"] = gd_abs_max

    pre_ringing_db = None if bool(st.get("pre_energy_metric_suspect", False)) else _auto_pick_metric(
        st,
        (
            "ir_pre_ringing_db",
            "mixed_pre_ringing_after_db",
            "ir_pre_energy_guard_after_db",
            "mixed_pre_ringing_before_db",
            "ir_pre_energy_guard_before_db",
        ),
    )
    if pre_ringing_db is not None:
        penalty += 0.70 * max(0.0, float(pre_ringing_db) + 45.0)
    dbg["pre_ringing_db"] = pre_ringing_db

    gd_rms_var = _auto_pick_metric(
        st,
        ("gd_rms_variance_ms",),
        abs_value=True,
        nonneg=True,
    )
    if gd_rms_var is not None:
        penalty += 0.08 * max(0.0, float(gd_rms_var) - 5.0)
    dbg["gd_rms_variance_ms"] = gd_rms_var

    pre_post_ratio = None if bool(st.get("pre_energy_metric_suspect", False)) else _auto_pick_metric(
        st,
        (
            "ir_pre_post_ratio",
            "ir_pre_energy_guard_after_ratio",
            "ir_pre_energy_guard_before_ratio",
        ),
        nonneg=True,
    )
    if pre_post_ratio is not None:
        penalty += 30.0 * max(0.0, float(pre_post_ratio) - 0.015)
    dbg["ir_pre_post_ratio"] = pre_post_ratio

    # Bass pre-ringing guard: penalise when bass-band phase correction had to be
    # heavily attenuated (scale_bass < 0.60) to prevent introducing pre-ringing.
    # Threshold matches the warning level in phase_ir_phase_03 guard logic.
    bass_prering_scale = None if bool(st.get("pre_energy_metric_suspect", False)) else _auto_pick_metric(
        st, ("mixed_pre_ringing_scale_bass",), nonneg=True
    )
    bass_prering_raw = 0.0
    if bass_prering_scale is not None and np.isfinite(float(bass_prering_scale)):
        bass_prering_raw = 8.0 * max(0.0, 0.60 - float(bass_prering_scale))
        penalty += bass_prering_raw
    dbg["mixed_pre_ringing_scale_bass"] = bass_prering_scale

    phase_boundary_mdb = _auto_pick_metric(
        st,
        (
            "phase_boundary_peak_mdb",
            "phase_corr_boundary_peak_mdb",
        ),
        abs_value=True,
        nonneg=True,
    )
    if phase_boundary_mdb is not None:
        penalty += 0.015 * max(0.0, float(phase_boundary_mdb) - 120.0)
    dbg["phase_boundary_peak_mdb"] = phase_boundary_mdb

    tdc_penalty, tdc_dbg = _auto_tdc_overreach_penalty(st)
    penalty += float(tdc_penalty)
    dbg["tdc_overreach_penalty"] = float(tdc_penalty)
    dbg["tdc"] = dict(tdc_dbg)

    return float(max(0.0, penalty)), float(bass_prering_raw), dbg


def _auto_phase_quality(st: dict | None) -> tuple[float, float, dict]:
    st = st or {}
    dbg = {}

    def _metric01(key: str) -> float:
        v = shared._auto_safe_float(st.get(key, float("nan")), float("nan"))
        if np.isfinite(v):
            return float(np.clip(v, 0.0, 1.0))
        return float("nan")

    useful_lf = _metric01("phase_useful_lf_score")
    useful_xo = _metric01("phase_useful_xo_score")
    useful_aud = _metric01("phase_useful_audible_score")
    risk_hf = _metric01("phase_risk_hf_score")
    risk_spiky = _metric01("phase_risk_spiky_score")
    risk_clamp = _metric01("phase_risk_clamp_score")
    conf_mean = _metric01("phase_confidence_mean")
    conf_lf = _metric01("phase_confidence_lf_mean")
    conf_xo = _metric01("phase_confidence_xo_mean")
    guard_scale = shared._auto_safe_float(st.get("phase_guard_scale_total", 1.0), 1.0)
    if not np.isfinite(guard_scale):
        guard_scale = 1.0
    guard_scale = float(np.clip(guard_scale, 0.0, 1.0))

    conf_anchor_vals = [float(v) for v in (conf_lf, conf_xo, conf_mean) if np.isfinite(v)]
    conf_anchor = float(np.mean(np.asarray(conf_anchor_vals, dtype=float))) if conf_anchor_vals else float("nan")

    benefit = 0.0
    if np.isfinite(useful_lf):
        benefit += 1.65 * float(useful_lf)
    if np.isfinite(useful_xo):
        benefit += 1.15 * float(useful_xo)
    if np.isfinite(useful_aud):
        benefit += 0.55 * float(useful_aud)
    if np.isfinite(conf_anchor):
        benefit *= 0.65 + 0.35 * float(np.clip(conf_anchor, 0.0, 1.0))

    risk = 0.0
    if np.isfinite(risk_hf):
        risk += 1.60 * float(risk_hf)
    if np.isfinite(risk_spiky):
        risk += 1.35 * float(risk_spiky)
    if np.isfinite(risk_clamp):
        risk += 0.70 * float(risk_clamp)
    risk += 0.85 * max(0.0, 1.0 - float(guard_scale))

    pre_ringing_db = None if bool(st.get("pre_energy_metric_suspect", False)) else _auto_pick_metric(
        st,
        (
            "ir_pre_ringing_db",
            "mixed_pre_ringing_after_db",
            "mixed_pre_ringing_before_db",
        ),
    )
    if pre_ringing_db is not None:
        risk += 0.20 * max(0.0, float(pre_ringing_db) + 42.0)

    gd_grad_max = _auto_pick_metric(
        st,
        (
            "gd_grad_limiter_after_max_ms_per_oct",
            "gd_grad_limiter_before_max_ms_per_oct",
            "gd_limiter_max_grad_ms_per_oct",
            "gd_grad_limiter_max_grad_ms_per_oct",
            "gd_limiter_max_grad_after_ms_per_oct",
            "gd_grad_limiter_max_grad_after_ms_per_oct",
            "gd_limiter_max_grad_before_ms_per_oct",
            "gd_grad_limiter_max_grad_before_ms_per_oct",
        ),
        abs_value=True,
        nonneg=True,
    )
    if gd_grad_max is not None:
        risk += 0.010 * max(0.0, float(gd_grad_max) - 8.0)

    gd_abs_max = _auto_pick_metric(
        st,
        ("gd_abs_max_20_500_ms",),
        abs_value=True,
        nonneg=True,
    )
    if gd_abs_max is not None:
        risk += 0.020 * max(0.0, float(gd_abs_max) - 20.0)

    hf_share = _auto_pick_metric(
        st,
        ("phase_corr_hf_share",),
        abs_value=True,
        nonneg=True,
    )
    if hf_share is not None:
        risk += 0.40 * max(0.0, float(hf_share) - 0.35)

    dbg["useful_lf"] = float(useful_lf) if np.isfinite(useful_lf) else float("nan")
    dbg["useful_xo"] = float(useful_xo) if np.isfinite(useful_xo) else float("nan")
    dbg["useful_audible"] = float(useful_aud) if np.isfinite(useful_aud) else float("nan")
    dbg["risk_hf"] = float(risk_hf) if np.isfinite(risk_hf) else float("nan")
    dbg["risk_spiky"] = float(risk_spiky) if np.isfinite(risk_spiky) else float("nan")
    dbg["risk_clamp"] = float(risk_clamp) if np.isfinite(risk_clamp) else float("nan")
    dbg["conf_anchor"] = float(conf_anchor) if np.isfinite(conf_anchor) else float("nan")
    dbg["guard_scale"] = float(guard_scale)
    dbg["pre_ringing_db"] = pre_ringing_db
    dbg["gd_grad_max"] = gd_grad_max
    dbg["gd_abs_max"] = gd_abs_max
    dbg["benefit"] = float(np.clip(benefit, 0.0, 3.5))
    dbg["risk"] = float(np.clip(risk, 0.0, 4.5))
    return float(dbg["benefit"]), float(dbg["risk"]), dbg


def _auto_excursion_penalty(st: dict | None) -> tuple[float, dict]:
    st = st or {}
    penalty = 0.0
    dbg = {}

    exc_raw = st.get("exc_prot", None)
    exc_known = exc_raw is not None
    exc_on = bool(exc_raw) if exc_known else None
    exc_freq = shared._auto_safe_float(st.get("exc_freq", 0.0), 0.0)

    try:
        exc_bins = int(float(st.get("boost_candidate_bins_excprot", 0) or 0))
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
        exc_bins = 0
    lf_boost_max = shared._auto_safe_float(st.get("lf_boost_max_db", 0.0), 0.0)
    pen_exc_off = 0.0
    pen_exc_invalid = 0.0
    pen_bins = 0.0
    pen_lf = 0.0

    if exc_known and (exc_on is False):
        pen_exc_off = 2.0
        penalty += float(pen_exc_off)
    if exc_known and (exc_on is True) and (not np.isfinite(exc_freq) or exc_freq <= 0.0):
        pen_exc_invalid = 0.8
        penalty += float(pen_exc_invalid)
    if exc_bins > 0:
        pen_bins = float(min(2.5, 0.10 * float(exc_bins)))
        penalty += float(pen_bins)

    pen_lf = float(min(12.0, 1.25 * max(0.0, float(lf_boost_max) - 1.5)))
    penalty += float(pen_lf)
    penalty = min(16.0, float(penalty))

    dbg["exc_known"] = bool(exc_known)
    dbg["exc_on"] = exc_on
    dbg["exc_freq"] = float(exc_freq)
    dbg["exc_bins"] = int(exc_bins)
    dbg["lf_boost_max_db"] = float(lf_boost_max)
    dbg["pen_exc_off"] = float(pen_exc_off)
    dbg["pen_exc_invalid"] = float(pen_exc_invalid)
    dbg["pen_bins"] = float(pen_bins)
    dbg["pen_lf"] = float(pen_lf)
    dbg["pen_total_pre_cap"] = float(pen_exc_off + pen_exc_invalid + pen_bins + pen_lf)
    dbg["pen_total"] = float(penalty)
    return float(max(0.0, penalty)), dbg



__all__ = ["_auto_tdc_overreach_penalty", "_auto_dsp_quality_penalty", "_auto_phase_quality", "_auto_excursion_penalty"]
