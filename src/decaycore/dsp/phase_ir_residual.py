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
import numpy as np
import scipy.ndimage

_logger = logging.getLogger(__name__)

if __package__ in (None, ""):
    import pathlib
    import sys

    _src_root = pathlib.Path(__file__).resolve().parents[2]
    _src_root_s = str(_src_root)
    if _src_root_s not in sys.path:
        sys.path.insert(0, _src_root_s)
    from decaycore.dsp.decaycore_analysis import _sigma_bins_from_hz
    from decaycore.dsp.dsp_config import CfgReader
    from decaycore.dsp.gain_policy import build_low_frequency_guard_mask, clamp_gain_curve, resolve_gain_policy
    from decaycore.dsp.phase_ir_types import ResidualTelemetry
    from decaycore.dsp.phase_ir_utils import _cosine_fade_out_01, _smoothstep01
    from decaycore.dsp.residual_authority import (
        _apply_residual_authority_caps,
        _pick_authority_array,
        build_residual_authority_caps,
    )
    from decaycore.dsp.smoothing import smooth_gain_fractional_octave, smooth_meas_freq_dep
else:
    from .decaycore_analysis import _sigma_bins_from_hz
    from .dsp_config import CfgReader
    from .gain_policy import build_low_frequency_guard_mask, clamp_gain_curve, resolve_gain_policy
    from .phase_ir_types import ResidualTelemetry
    from .phase_ir_utils import _cosine_fade_out_01, _smoothstep01
    from .residual_authority import (
        _apply_residual_authority_caps,
        _pick_authority_array,
        build_residual_authority_caps,
    )
    from .smoothing import smooth_gain_fractional_octave, smooth_meas_freq_dep


def _residual_pass_mode(cfg_reader: CfgReader) -> str:
    mode = cfg_reader.enum_string("residual_pass_mode", "modal_polish").strip().lower()
    if mode not in ("modal_polish", "general_fit", "off"):
        return "modal_polish"
    return mode


def _finite_stat(values, *, op: str) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0
    if op == "min":
        return float(np.min(finite))
    if op == "max":
        return float(np.max(finite))
    return float(np.mean(finite))


def _smooth_authority_array(value, freq_axis: np.ndarray, smooth_oct: float, *, preserve_peaks: bool = False):
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return value
    if arr.shape != freq_axis.shape:
        return value
    width = float(np.clip(smooth_oct, 1.0 / 192.0, 1.0))
    try:
        smoothed = smooth_gain_fractional_octave(freq_axis, arr, 1.0 / width)
    except (TypeError, ValueError, FloatingPointError, IndexError):
        return value
    if bool(preserve_peaks):
        smoothed = np.maximum(np.asarray(arr, dtype=float), np.asarray(smoothed, dtype=float))
    return np.clip(np.nan_to_num(smoothed, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def _full_residual_authority_caps(freq_axis: np.ndarray, *, max_boost_db: float, max_cut_db: float) -> dict:
    boost = np.full_like(freq_axis, float(max(0.0, max_boost_db)), dtype=float)
    cut = np.full_like(freq_axis, float(max(0.0, max_cut_db)), dtype=float)
    allowed = np.ones_like(freq_axis, dtype=bool)
    return {
        "residual_boost_cap_db": boost,
        "residual_cut_cap_db": cut,
        "residual_allowed": allowed,
        "residual_boost_allowed": allowed.copy(),
        "residual_cut_allowed": allowed.copy(),
        "residual_modal_polish_mask": np.zeros_like(freq_axis, dtype=bool),
        "residual_block_reason": {
            "null_risk_bins": 0,
            "reflection_risk_bins": 0,
            "low_authority_bins": 0,
            "low_modal_support_bins": 0,
        },
    }


def _residual_authority_arrays(st, freq_axis: np.ndarray, smooth_oct: float) -> tuple[np.ndarray | None, ...]:
    return (
        _smooth_authority_array(_pick_authority_array(st, "authority_null_risk", "null_risk"), freq_axis, smooth_oct, preserve_peaks=True),
        _smooth_authority_array(_pick_authority_array(st, "authority_boost", "boost_authority"), freq_axis, smooth_oct),
        _smooth_authority_array(_pick_authority_array(st, "authority_cut", "cut_authority"), freq_axis, smooth_oct),
        _smooth_authority_array(_pick_authority_array(st, "authority_modal_support", "modal_support"), freq_axis, smooth_oct),
        _smooth_authority_array(_pick_authority_array(st, "authority_decay_need", "decay_need"), freq_axis, smooth_oct),
        _smooth_authority_array(_pick_authority_array(st, "authority_reflection_risk", "reflection_risk"), freq_axis, smooth_oct, preserve_peaks=True),
    )


def _residual_build_authority_caps(
    *,
    freq_axis: np.ndarray,
    st,
    cfg_reader: CfgReader,
    residual_mode: str,
    null_guard_enabled: bool,
    authority_max_boost_db: float,
    authority_max_cut_db: float,
) -> dict:
    if residual_mode == "off":
        return build_residual_authority_caps(freq_axis, residual_pass_mode="off")
    if not bool(null_guard_enabled):
        return _full_residual_authority_caps(
            freq_axis,
            max_boost_db=authority_max_boost_db,
            max_cut_db=authority_max_cut_db,
        )
    smooth_oct = cfg_reader.float_allow_zero("residual_authority_smooth_oct", 1.0 / 9.0)
    if (not np.isfinite(smooth_oct)) or smooth_oct <= 0.0:
        smooth_oct = 1.0 / 9.0
    (
        authority_null_risk,
        authority_boost,
        authority_cut,
        authority_modal_support,
        authority_decay_need,
        authority_reflection_risk,
    ) = _residual_authority_arrays(
        st,
        freq_axis,
        smooth_oct,
    )
    return build_residual_authority_caps(
        freq_axis,
        authority_null_risk=authority_null_risk,
        authority_boost=authority_boost,
        authority_cut=authority_cut,
        authority_modal_support=authority_modal_support,
        authority_decay_need=authority_decay_need,
        authority_reflection_risk=authority_reflection_risk,
        residual_pass_mode=residual_mode,
        max_boost_db=authority_max_boost_db,
        max_cut_db=authority_max_cut_db,
        null_guard_strength=float(
            np.clip(cfg_reader.float_allow_zero("residual_null_guard_strength", 1.0), 0.0, 1.0)
        ),
        modal_min_support=float(
            np.clip(cfg_reader.float_allow_zero("residual_modal_min_support", 0.45), 0.0, 1.0)
        ),
        boost_authority_min=float(
            np.clip(cfg_reader.float_allow_zero("residual_boost_authority_min", 0.40), 0.0, 1.0)
        ),
        cut_authority_min=float(
            np.clip(cfg_reader.float_allow_zero("residual_cut_authority_min", 0.35), 0.0, 1.0)
        ),
        reflection_risk_max=float(
            np.clip(cfg_reader.float_allow_zero("residual_reflection_risk_max", 0.65), 0.0, 1.0)
        ),
        null_risk_max_for_boost=float(
            np.clip(cfg_reader.float_allow_zero("residual_null_risk_max_for_boost", 0.35), 0.0, 1.0)
        ),
        null_risk_max_for_cut=float(
            np.clip(cfg_reader.float_allow_zero("residual_null_risk_max_for_cut", 0.75), 0.0, 1.0)
        ),
        max_boost_when_null_risk_db=float(
            max(0.0, cfg_reader.float_allow_zero("residual_max_boost_when_null_risk_db", 0.5))
        ),
    )


def _residual_apply_delta_and_limits(
    *,
    gain_db: np.ndarray,
    gain_before: np.ndarray,
    residual_delta: np.ndarray,
    freq_axis: np.ndarray,
    mask_c: np.ndarray,
    gain_policy,
) -> tuple[np.ndarray, np.ndarray, int, float, float]:
    low_guard_mask = build_low_frequency_guard_mask(freq_axis, gain_policy)
    low_guard_apply_mask = mask_c & low_guard_mask
    blocked_boost_bins = int(np.count_nonzero((residual_delta > 1e-9) & low_guard_apply_mask))
    if np.any(low_guard_apply_mask):
        residual_delta[low_guard_apply_mask] = np.minimum(residual_delta[low_guard_apply_mask], 0.0)
    apply_mask = mask_c & (np.isfinite(residual_delta))
    applied_boost_max_db = float(np.max(np.maximum(residual_delta[apply_mask], 0.0))) if np.any(apply_mask) else 0.0
    applied_cut_max_db = float(np.max(np.maximum(-residual_delta[apply_mask], 0.0))) if np.any(apply_mask) else 0.0
    if np.any(apply_mask):
        gain_db[apply_mask] = gain_before[apply_mask] + residual_delta[apply_mask]
    if np.any(low_guard_apply_mask):
        gain_db[low_guard_apply_mask] = np.minimum(
            gain_db[low_guard_apply_mask],
            gain_before[low_guard_apply_mask],
        )
    if np.any(mask_c):
        gain_db = clamp_gain_curve(gain_db, policy=gain_policy, mask=mask_c)
    return gain_db, low_guard_apply_mask, blocked_boost_bins, applied_boost_max_db, applied_cut_max_db


def _residual_band_weight(
    *,
    freq_axis: np.ndarray,
    cfg,
    mask_c: np.ndarray,
    skip_right_transition_fade: bool = False,
) -> tuple[np.ndarray, float, float, float]:
    cfg_reader = CfgReader(cfg)
    f = np.asarray(freq_axis, dtype=float)
    m = np.asarray(mask_c, dtype=bool)
    w = np.zeros_like(f, dtype=float)
    if f.size == 0 or m.shape != f.shape:
        return w, 0.0, 0.0, 0.0

    f_min = cfg_reader.float_allow_zero("mag_c_min", 0.0)
    f_max = cfg_reader.float_allow_zero("mag_c_max", 0.0)

    if (not np.isfinite(f_min)) or (not np.isfinite(f_max)) or (f_max <= f_min):
        idx = np.where(m)[0]
        if idx.size < 2:
            return w, 0.0, 0.0, 0.0
        f_min = float(f[int(idx[0])])
        f_max = float(f[int(idx[-1])])

    span = float(f_max - f_min)
    if span <= 1e-9:
        return w, float(f_min), float(f_max), 0.0

    edge_hz = cfg_reader.float_allow_zero("residual_band_edge_hz", 0.0)
    if (not np.isfinite(edge_hz)) or edge_hz <= 0.0:
        edge_hz = cfg_reader.float_allow_zero("trans_width", 0.0)
    if (not np.isfinite(edge_hz)) or edge_hz <= 0.0:
        edge_hz = 0.08 * span
    edge_hz = float(np.clip(edge_hz, 1e-6, 0.49 * span))

    in_band = (f >= f_min) & (f <= f_max) & m
    w[in_band] = 1.0

    left = (f >= f_min) & (f < (f_min + edge_hz)) & m
    if np.any(left):
        w[left] = _smoothstep01((f[left] - f_min) / edge_hz)

    right = (f > (f_max - edge_hz)) & (f <= f_max) & m
    if (not bool(skip_right_transition_fade)) and np.any(right):
        x_right = (f[right] - (f_max - edge_hz)) / edge_hz
        w[right] = _cosine_fade_out_01(x_right)

    return np.clip(w, 0.0, 1.0), float(f_min), float(f_max), float(edge_hz)


def _residual_process_enabled_pass(
    *,
    cfg,
    cfg_reader: CfgReader,
    freq_axis: np.ndarray,
    gain_db: np.ndarray,
    conf_mask,
    m_anal: np.ndarray,
    calc_offset_db: float,
    target_mags: np.ndarray,
    st: dict,
    mask_c: np.ndarray,
    base_sigma,
    filter_smooth,
    df_mode: bool,
    logger,
    cfg_float_allow_zero_fn,
) -> tuple[np.ndarray, ResidualTelemetry]:
    residual_mode, freq_axis, gain_db, mask_c, resid, k, strength, mult, _base_sigma, _df_mode, gain_policy = _residual_prepare_inputs(
        cfg,
        freq_axis,
        gain_db,
        mask_c,
        m_anal,
        calc_offset_db,
        target_mags,
        base_sigma,
        filter_smooth,
        df_mode,
        conf_mask,
        cfg_reader,
    )
    if residual_mode == "off":
        return gain_db, ResidualTelemetry(
            residual_pass_enabled=False,
            residual_pass_mode=residual_mode,
            residual_null_guard_enabled=bool(cfg_reader.bool("residual_null_guard_enable", True)),
        )

    (
        residual_delta,
        authority_caps,
        _band_w,
        apply_mask,
        _,
        band_min_hz,
        band_max_hz,
        band_edge_hz,
        already,
        requested_boost_max_db,
        requested_cut_max_db,
        max_delta_db,
        authority_max_boost_db,
        authority_max_cut_db,
        null_guard_enabled,
        _strength,
    ) = _residual_build_delta_and_authority(
        cfg=cfg,
        cfg_reader=cfg_reader,
        freq_axis=freq_axis,
        resid=resid,
        strength=strength,
        mult=mult,
        base_sigma=_base_sigma,
        df_mode=_df_mode,
        mask_c=mask_c,
        st=st,
        gain_policy=gain_policy,
        residual_mode=residual_mode,
    )

    residual_delta = _apply_residual_authority_caps(residual_delta, authority_caps)
    gain_before = gain_db.copy()
    gain_db, low_guard_apply_mask, blocked_boost_bins, applied_boost_max_db, applied_cut_max_db = _residual_apply_delta_and_limits(
        gain_db=gain_db,
        gain_before=gain_before,
        residual_delta=residual_delta,
        freq_axis=freq_axis,
        mask_c=mask_c,
        gain_policy=gain_policy,
    )

    boost_cap = np.asarray(authority_caps.get("residual_boost_cap_db"), dtype=float)
    cut_cap = np.asarray(authority_caps.get("residual_cut_cap_db"), dtype=float)
    block_reason = dict(authority_caps.get("residual_block_reason", {}) or {})
    modal_bins = int(np.count_nonzero(np.asarray(authority_caps.get("residual_modal_polish_mask"), dtype=bool)))
    try:
        if logger is not None:
            logger.info(
                "Residual authority: mode=%s boost_max=%.2fdB cut_max=%.2fdB "
                "blocked_null=%d blocked_reflection=%d modal_bins=%d",
                residual_mode,
                float(_finite_stat(boost_cap, op="max")),
                float(_finite_stat(cut_cap, op="max")),
                int(block_reason.get("null_risk_bins", 0)),
                int(block_reason.get("reflection_risk_bins", 0)),
                int(modal_bins),
            )
    except (AttributeError, TypeError, ValueError):
        pass

    residual_telemetry = ResidualTelemetry(
        residual_pass_enabled=True,
        residual_pass_mode=str(residual_mode),
        residual_null_guard_enabled=bool(null_guard_enabled),
        residual_strength=float(strength),
        residual_smoothing_mult=float(mult),
        residual_conf_power=float(k),
        residual_max_delta_db=float(max_delta_db),
        residual_band_min_hz=float(band_min_hz),
        residual_band_max_hz=float(band_max_hz),
        residual_band_edge_hz=float(band_edge_hz),
        residual_band_bins=int(np.count_nonzero(apply_mask)),
        residual_lf_guard_bins=int(np.count_nonzero(low_guard_apply_mask)),
        residual_lf_boost_blocked_bins=int(blocked_boost_bins),
        residual_right_transition_fade_skipped=bool(already),
        residual_authority_boost_cap_mean_db=float(_finite_stat(boost_cap, op="mean")),
        residual_authority_boost_cap_min_db=float(_finite_stat(boost_cap, op="min")),
        residual_authority_boost_cap_max_db=float(_finite_stat(boost_cap, op="max")),
        residual_authority_cut_cap_mean_db=float(_finite_stat(cut_cap, op="mean")),
        residual_authority_cut_cap_min_db=float(_finite_stat(cut_cap, op="min")),
        residual_authority_cut_cap_max_db=float(_finite_stat(cut_cap, op="max")),
        residual_blocked_null_risk_bins=int(block_reason.get("null_risk_bins", 0)),
        residual_blocked_reflection_risk_bins=int(block_reason.get("reflection_risk_bins", 0)),
        residual_blocked_low_authority_bins=int(block_reason.get("low_authority_bins", 0)),
        residual_blocked_low_modal_support_bins=int(block_reason.get("low_modal_support_bins", 0)),
        residual_modal_polish_bins=int(modal_bins),
        residual_requested_boost_max_db=float(requested_boost_max_db),
        residual_applied_boost_max_db=float(applied_boost_max_db),
        residual_requested_cut_max_db=float(requested_cut_max_db),
        residual_applied_cut_max_db=float(applied_cut_max_db),
    )
    return gain_db, residual_telemetry


def _residual_prepare_inputs(
    cfg,
    freq_axis: np.ndarray,
    gain_db: np.ndarray,
    mask_c: np.ndarray,
    m_anal: np.ndarray,
    calc_offset_db: float,
    target_mags: np.ndarray,
    base_sigma,
    filter_smooth,
    df_mode: bool,
    conf_mask,
    cfg_reader: CfgReader,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, float, bool, np.ndarray]:
    residual_mode = _residual_pass_mode(cfg_reader)
    freq_axis = np.asarray(freq_axis, dtype=float)
    gain_db = np.asarray(gain_db, dtype=float).copy()
    mask_c = np.asarray(mask_c, dtype=bool)
    if gain_db.shape != freq_axis.shape or mask_c.shape != gain_db.shape:
        raise ValueError("shape mismatch")
    if residual_mode == "off":
        return residual_mode, freq_axis, gain_db, mask_c, np.zeros_like(gain_db, dtype=float), 0.0, 0.0, 1.0, 1.0, False, np.asarray([], dtype=float)

    gain_policy = resolve_gain_policy(cfg, cfg_float_allow_zero_fn=cfg_reader.float_allow_zero)
    measured_aligned = (m_anal - calc_offset_db)
    pred0 = measured_aligned + gain_db
    resid0 = (target_mags - pred0)
    resid = np.zeros_like(gain_db, dtype=float)
    resid[mask_c] = resid0[mask_c]
    try:
        k = cfg_reader.float("residual_conf_power", 2.0)
        if conf_mask is not None:
            resid[mask_c] *= np.clip(conf_mask[mask_c], 0.0, 1.0) ** k
    except (TypeError, ValueError, IndexError):
        k = 2.0
    strength = float(np.clip(cfg_reader.float("residual_strength", 0.6), 0.0, 1.0))
    mult = max(1.0, float(cfg_reader.float("residual_smoothing_mult", 2.0)))
    try:
        _base_sigma = float(base_sigma)
    except (TypeError, ValueError, OverflowError):
        _base_sigma = float(60 // (filter_smooth / 12 if filter_smooth > 0 else 1))
    if (not np.isfinite(_base_sigma)) or _base_sigma <= 0.0:
        _base_sigma = float(60 // (filter_smooth / 12 if filter_smooth > 0 else 1))
    _df_mode = bool(df_mode) if df_mode is not None else cfg_reader.bool("df_smoothing", False)
    return residual_mode, freq_axis, gain_db, mask_c, resid, float(k), float(strength), float(mult), float(_base_sigma), bool(_df_mode), gain_policy


def _residual_build_delta_and_authority(
    *,
    cfg,
    cfg_reader: CfgReader,
    freq_axis: np.ndarray,
    resid: np.ndarray,
    strength: float,
    mult: float,
    base_sigma: float,
    df_mode: bool,
    mask_c: np.ndarray,
    st,
    gain_policy,
    residual_mode: str,
) -> tuple[np.ndarray, dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, float, float, float, float, float, float, float]:
    if df_mode:
        df_ref = 44100.0 / 65536.0
        sigma_hz = float(base_sigma) * df_ref * mult
        sigma_bins = _sigma_bins_from_hz(
            freq_axis,
            sigma_hz=sigma_hz,
            fallback_bins=max(2.0, float(base_sigma) * mult),
        )
        resid_sm = scipy.ndimage.gaussian_filter1d(resid, sigma=float(sigma_bins))
    else:
        resid_sm = smooth_meas_freq_dep(resid, freq_axis)

    already = bool(isinstance(st, dict) and st.get("mag_transition_fade_applied", False))
    band_w, band_min_hz, band_max_hz, band_edge_hz = _residual_band_weight(
        freq_axis=freq_axis,
        cfg=cfg,
        mask_c=mask_c,
        skip_right_transition_fade=already,
    )
    residual_delta = np.asarray(resid_sm, dtype=float) * float(strength) * np.asarray(band_w, dtype=float)
    max_delta_db = cfg_reader.float("residual_max_delta_db", 1.25)
    if (not np.isfinite(max_delta_db)) or max_delta_db <= 0.0:
        max_delta_db = 1.25
    residual_delta = np.clip(residual_delta, -float(max_delta_db), float(max_delta_db))
    apply_mask = mask_c & (band_w > 0.0)
    requested_boost_max_db = float(np.max(np.maximum(residual_delta[apply_mask], 0.0))) if np.any(apply_mask) else 0.0
    requested_cut_max_db = float(np.max(np.maximum(-residual_delta[apply_mask], 0.0))) if np.any(apply_mask) else 0.0
    null_guard_enabled = cfg_reader.bool("residual_null_guard_enable", True)
    authority_max_boost_db = float(max(0.0, cfg_reader.float_allow_zero("residual_max_boost_general_db", 2.0)))
    authority_max_cut_db = float(max(0.0, cfg_reader.float_allow_zero("residual_max_cut_general_db", 4.0)))
    authority_caps = _residual_build_authority_caps(
        freq_axis=freq_axis,
        st=st,
        cfg_reader=cfg_reader,
        residual_mode=residual_mode,
        null_guard_enabled=bool(null_guard_enabled),
        authority_max_boost_db=authority_max_boost_db,
        authority_max_cut_db=authority_max_cut_db,
    )
    return (
        residual_delta,
        authority_caps,
        band_w,
        apply_mask,
        band_w,
        band_min_hz,
        band_max_hz,
        band_edge_hz,
        bool(already),
        float(requested_boost_max_db),
        float(requested_cut_max_db),
        float(max_delta_db),
        float(authority_max_boost_db),
        float(authority_max_cut_db),
        float(null_guard_enabled),
        float(strength),
    )


def apply_residual_pass_if_enabled(
    *,
    cfg,
    freq_axis: np.ndarray,
    gain_db: np.ndarray,
    conf_mask,
    m_anal: np.ndarray,
    calc_offset_db: float,
    target_mags: np.ndarray,
    st: dict,
    mask_c: np.ndarray,
    base_sigma,
    filter_smooth,
    df_mode: bool,
    raw_g,
    final_g,
    logger,
    cfg_float_allow_zero_fn,
) -> tuple[np.ndarray, ResidualTelemetry | None]:
    """
    Contract:
      - This stage may only change gain_db.
      - It must not mutate phase-domain or IR-domain arrays.
      - Telemetry is returned typed and adapted to stats outside this stage.

    Returns updated gain_db (same shape) and optional residual telemetry.
    """
    _filter_smooth = filter_smooth
    _cfg_float_allow_zero = cfg_float_allow_zero_fn
    residual_telemetry: ResidualTelemetry | None = None
    cfg_reader = CfgReader(cfg)

    if cfg_reader.bool("enable_residual_pass", False) and cfg_reader.bool("enable_mag_correction", True):
        try:
            gain_db, residual_telemetry = _residual_process_enabled_pass(
                cfg=cfg,
                cfg_reader=cfg_reader,
                freq_axis=freq_axis,
                gain_db=gain_db,
                conf_mask=conf_mask,
                m_anal=m_anal,
                calc_offset_db=calc_offset_db,
                target_mags=target_mags,
                st=st,
                mask_c=mask_c,
                base_sigma=base_sigma,
                filter_smooth=filter_smooth,
                df_mode=df_mode,
                logger=logger,
                cfg_float_allow_zero_fn=cfg_float_allow_zero_fn,
            )
        except (AttributeError, TypeError, ValueError, FloatingPointError, IndexError) as e:
            if logger is not None:
                logger.warning("residual pass failed: %s", e)
            pass

    return gain_db, residual_telemetry
