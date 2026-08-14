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

from typing import Any

import numpy as np
import scipy.ndimage

from ..phase import combine_mixed_phase
from ..phase_ir_ir import _phase_to_ir
from ..phase_ir_phase_models import (
    phase_clamp_limit_deg as _phase_clamp_limit_deg_impl,
)
from ..phase_ir_phase_models import (
    spike_suppression_profile as _spike_suppression_profile_impl,
)
from ..phase_ir_phase_models import (
    unified_correction_gain as _unified_correction_gain_impl,
)
from ..phase_ir_metrics import compute_pre_post_energy_metrics as _compute_pre_post_energy_metrics
from ..phase_ir_utils import _max_abs_group_delay_ms
from ..phase_authority import apply_phase_authority_gating as _apply_phase_authority_gating

from .phase_computation import (
    _gd_grad_limiter,
    _merge_minphase_and_excess,
    _phase_confidence_profile,
    _phase_region_profiles,
    _weighted_mean,
    _weighted_share,
)
from .phase_windowing import (
    _PhaseComponents,
    _apply_mixed_excess_mask,
    _enforce_linear_tail_decay,
    _linear_excess_weight,
    _linear_to_minphase_blend_mask,
    _smooth_linear_boundary,
)


def _store_phase_profile_metrics(
    *,
    freq_axis: np.ndarray,
    extra_phase: np.ndarray,
    excess_phase: np.ndarray,
    phase_mask: np.ndarray,
    phase_confidence: np.ndarray,
    phase_regions: dict[str, Any] | None,
    spike_suppress: np.ndarray | None,
    clamp_cut_frac: np.ndarray | None,
    guard_scale_total: float,
    st,
) -> None:
    if not isinstance(st, dict):
        return
    f = np.asarray(freq_axis, dtype=float)
    corr = np.abs(np.asarray(extra_phase, dtype=float))
    exc = np.abs(np.asarray(excess_phase, dtype=float))
    mask = np.asarray(phase_mask, dtype=bool)
    if f.size == 0 or corr.size != f.size or exc.size != f.size or mask.size != f.size:
        return

    phase_regions = dict(phase_regions or {})
    lf_w = np.asarray(phase_regions.get("lf", np.zeros_like(f, dtype=float)), dtype=float)
    xo_w = np.asarray(phase_regions.get("xo", np.zeros_like(f, dtype=float)), dtype=float)
    hf_w = np.asarray(phase_regions.get("hf", np.zeros_like(f, dtype=float)), dtype=float)
    audible_w = np.asarray(phase_regions.get("audible", np.zeros_like(f, dtype=float)), dtype=float)
    conf = np.asarray(phase_confidence, dtype=float)
    if conf.size != f.size:
        conf = np.ones_like(f, dtype=float)
    conf = np.clip(conf, 0.0, 1.0)
    spike = np.asarray(
        spike_suppress if spike_suppress is not None else np.ones_like(f, dtype=float),
        dtype=float,
    )
    if spike.size != f.size:
        spike = np.ones_like(f, dtype=float)
    spike = np.clip(spike, 0.0, 1.0)
    cut = np.asarray(
        clamp_cut_frac if clamp_cut_frac is not None else np.zeros_like(f, dtype=float),
        dtype=float,
    )
    if cut.size != f.size:
        cut = np.zeros_like(f, dtype=float)
    cut = np.clip(cut, 0.0, 1.0)

    active = mask & np.isfinite(corr) & np.isfinite(exc)
    if int(np.count_nonzero(active)) < 4:
        return

    active_w = active.astype(float)
    lf_w = np.clip(lf_w * active_w, 0.0, 1.0)
    xo_w = np.clip(xo_w * active_w, 0.0, 1.0)
    hf_w = np.clip(hf_w * active_w, 0.0, 1.0)
    audible_w = np.clip(audible_w * active_w, 0.0, 1.0)

    corr_norm = np.tanh(corr / max(float(np.deg2rad(16.0)), 1e-9))
    eff = np.clip(corr / np.maximum(exc, float(np.deg2rad(4.0))), 0.0, 1.0)
    useful = np.clip(0.55 * corr_norm + 0.45 * eff, 0.0, 1.0)

    st["phase_confidence_mean"] = _weighted_mean(conf, active_w)
    st["phase_confidence_lf_mean"] = _weighted_mean(conf, lf_w)
    st["phase_confidence_xo_mean"] = _weighted_mean(conf, xo_w)
    st["phase_confidence_hf_mean"] = _weighted_mean(conf, hf_w)
    st["phase_useful_lf_score"] = _weighted_mean(useful * conf, lf_w)
    st["phase_useful_xo_score"] = _weighted_mean(useful * conf, xo_w)
    st["phase_useful_audible_score"] = _weighted_mean(useful * conf, audible_w)
    st["phase_risk_hf_score"] = _weighted_mean(corr_norm * (1.0 - conf), hf_w)
    st["phase_risk_spiky_score"] = _weighted_mean(corr_norm * (1.0 - spike), np.maximum(audible_w, 0.35 * hf_w))
    st["phase_risk_clamp_score"] = _weighted_mean(cut, np.maximum(audible_w, hf_w))
    st["phase_corr_lf_share"] = _weighted_share(corr_norm, lf_w, active_w)
    st["phase_corr_xo_share"] = _weighted_share(corr_norm, xo_w, active_w)
    st["phase_corr_hf_share"] = _weighted_share(corr_norm, hf_w, active_w)
    st["phase_spike_suppression_mean"] = _weighted_mean(spike, active_w)
    st["phase_spike_suppression_hf_mean"] = _weighted_mean(spike, hf_w)
    st["phase_guard_scale_total"] = float(np.clip(guard_scale_total, 0.0, 1.0))
    anchors = phase_regions.get("anchors_hz", tuple())
    st["phase_anchor_count"] = int(len(tuple(anchors or ())))


def _phase_max_abs_deg(values: np.ndarray, mask: np.ndarray | None = None) -> float:
    try:
        arr = np.asarray(values, dtype=float)
        if mask is not None:
            sel = np.asarray(mask, dtype=bool)
            if sel.size == arr.size:
                arr = arr[sel]
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float("nan")
        return float(np.rad2deg(np.max(np.abs(arr))))
    except (TypeError, ValueError, FloatingPointError, IndexError):
        return float("nan")


def _phase_mean_abs_deg(values: np.ndarray, mask: np.ndarray | None = None) -> float:
    try:
        arr = np.asarray(values, dtype=float)
        if mask is not None:
            sel = np.asarray(mask, dtype=bool)
            if sel.size == arr.size:
                arr = arr[sel]
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float("nan")
        return float(np.rad2deg(np.mean(np.abs(arr))))
    except (TypeError, ValueError, FloatingPointError, IndexError):
        return float("nan")


def _store_phase_excess_diagnostics(
    *,
    phase_components: _PhaseComponents,
    excess_phase: np.ndarray,
    phase_mask: np.ndarray,
    st,
) -> None:
    if not isinstance(st, dict):
        return
    try:
        raw_delta = np.asarray(phase_components.raw_u, dtype=float) - np.asarray(phase_components.ref_u, dtype=float)
        wrapped_delta = np.angle(np.exp(1j * raw_delta))
        st["phase_excess_wrapped_max_abs_deg"] = _phase_max_abs_deg(wrapped_delta, phase_mask)
        st["phase_excess_unwrapped_max_abs_deg"] = _phase_max_abs_deg(excess_phase, phase_mask)
        st["phase_excess_unwrapped_mean_abs_deg"] = _phase_mean_abs_deg(excess_phase, phase_mask)
        st["phase_excess_unwrapped_used"] = True
    except (TypeError, ValueError, FloatingPointError, IndexError):
        pass


def _has_active_theoretical_phase_model(cfg) -> bool:
    try:
        for xo in list(getattr(cfg, "crossovers", None) or []):
            try:
                fc = float(xo.get("freq", 0.0) or 0.0)
            except (AttributeError, TypeError, ValueError):
                continue
            if np.isfinite(fc) and fc > 0.0:
                return True
    except (AttributeError, TypeError, ValueError):
        pass

    try:
        hs = getattr(cfg, "hpf_settings", None)
        if isinstance(hs, dict) and bool(hs.get("enabled", False)):
            fc = float(hs.get("freq", 0.0) or 0.0)
            order = int(hs.get("order", 0) or 0)
            if np.isfinite(fc) and fc > 0.0 and order > 0:
                return True
    except (AttributeError, TypeError, ValueError):
        pass

    return False


def _pre_ringing_band_protection_floor(f: np.ndarray) -> np.ndarray:
    """Per-frequency minimum allowed guard_scale_total value.
    Prevents the pre-ringing guard from over-reducing safe bass correction.
      20–80 Hz:   floor=0.55 (bass protected — rarely causes audible pre-ringing)
      80–200 Hz:  floor=0.35 (voice — moderate protection)
      200–600 Hz: floor=0.05 (mid — full guard effect permitted)
      >600 Hz:    floor=0.05 (already attenuated by phase_mask)
    Uses cosine crossfades at band boundaries.
    """
    f = np.asarray(f, dtype=float)
    floor = np.full_like(f, 0.05, dtype=float)

    FLOOR_BASS = 0.55
    F_BASS_TOP, F_VOICE_TOP = 80.0, 200.0

    floor = np.where(f <= F_BASS_TOP, FLOOR_BASS, floor)

    xfade_bv = (f > F_BASS_TOP) & (f < F_VOICE_TOP)
    if np.any(xfade_bv):
        x = (f[xfade_bv] - F_BASS_TOP) / (F_VOICE_TOP - F_BASS_TOP)
        w = 0.5 + 0.5 * np.cos(np.pi * x)
        floor[xfade_bv] = 0.05 + (FLOOR_BASS - 0.05) * w

    # >=200 Hz stays at 0.05 (initialization value)
    return np.clip(floor, 0.0, 1.0)


def _apply_phase_model(  # noqa: C901 - phase containment and spike suppression are kept explicit
    freq_axis, cfg, st, phase_components: _PhaseComponents
) -> np.ndarray:
    f = np.asarray(freq_axis, dtype=float)
    is_mixed = bool(phase_components.is_mixed)
    min_p = np.asarray(phase_components.min_phase, dtype=float)
    theo_xo = np.asarray(phase_components.theo_xo, dtype=float)
    excess_phase = np.asarray(phase_components.excess_u, dtype=float)
    logger = phase_components.logger
    correction_baseline = _merge_minphase_and_excess(min_p, -theo_xo)

    if bool(getattr(cfg, "phase_safe_2058", False)):
        if "Min" in cfg.filter_type_str:
            final_phase = min_p
        elif is_mixed:
            low_phase = -theo_xo
            phase_components.low_phase = low_phase
            final_phase = low_phase
        else:
            final_phase = -theo_xo
        phase_components.extra_phase = None
        phase_components.phase_mask = None
        return final_phase

    phase_lim_hz = float(getattr(cfg, "phase_limit", 1000.0))
    phase_mask = (f > 0) & (f <= phase_lim_hz)
    phase_components.phase_mask = phase_mask
    try:
        if isinstance(st, dict):
            st["phase_limit_hz"] = float(phase_lim_hz)
    except (TypeError, ValueError):
        pass
    _store_phase_excess_diagnostics(
        phase_components=phase_components,
        excess_phase=excess_phase,
        phase_mask=phase_mask,
        st=st,
    )

    try:
        f1 = float(phase_lim_hz)
        f0_fade = f1 / 2.0
        if f0_fade < (f1 - 1.0):
            x = (f - f0_fade) / (f1 - f0_fade + 1e-12)
            x = np.clip(x, 0.0, 1.0)
            w_hi = 0.5 * (1.0 + np.cos(np.pi * x))
            w_hi = np.where(f <= f0_fade, 1.0, w_hi)
            w_hi = np.where(f >= f1, 0.0, w_hi)
        else:
            w_hi = np.ones_like(f, dtype=float)
    except (TypeError, ValueError, FloatingPointError):
        w_hi = np.ones_like(f, dtype=float)

    static_profiles = phase_components.static_profiles if isinstance(phase_components.static_profiles, dict) else {}
    phase_regions = static_profiles.get("phase_regions")
    if not isinstance(phase_regions, dict):
        phase_regions = _phase_region_profiles(f, phase_lim_hz, cfg)
        static_profiles["phase_regions"] = phase_regions
    phase_conf = np.asarray(static_profiles.get("phase_confidence", []), dtype=float)
    if phase_conf.size != f.size:
        try:
            conf_arr = (
                np.asarray(phase_components.conf_mask, dtype=float)
                if phase_components.conf_mask is not None
                else np.ones_like(f, dtype=float)
            )
        except (TypeError, ValueError):
            conf_arr = np.ones_like(f, dtype=float)
        phase_conf = _phase_confidence_profile(
            f,
            conf_arr,
            phase_lim_hz,
            cfg,
            bassfirst=bool(phase_components.use_bassfirst),
            afdw_on=bool(phase_components.afdw_on),
        )
        static_profiles["phase_confidence"] = np.asarray(phase_conf, dtype=float)

    budget_mode = str(getattr(cfg, "phase_budget_mode", "unified") or "unified").strip().lower()
    use_unified = budget_mode != "legacy"
    try:
        if isinstance(st, dict):
            st["phase_budget_mode"] = "unified" if use_unified else "legacy"
    except (TypeError, ValueError):
        pass

    spike_suppress = np.ones_like(f, dtype=float)
    clamp_cut_frac_arr = np.zeros_like(f, dtype=float)
    unified_gain = None

    if use_unified:
        # Unified phase budget: one effective gain curve (strength * band fade *
        # min(confidence, spike trust)). Containment is delegated to the
        # excess-delay, pre-ringing, GD-gradient and authority guards below.
        try:
            fs_unified = float(cfg.fs) if hasattr(cfg, "fs") else 48000.0
        except (AttributeError, TypeError, ValueError):
            fs_unified = 48000.0
        spike_suppress = np.asarray(static_profiles.get("spike_suppression", []), dtype=float)
        if spike_suppress.size != f.size:
            try:
                spike_suppress = _spike_suppression_profile_impl(
                    f,
                    excess_phase,
                    phase_regions,
                    n_fft=phase_components.n_fft,
                    fs=fs_unified,
                )
            except (TypeError, ValueError, FloatingPointError):
                spike_suppress = np.ones_like(f, dtype=float)
            static_profiles["spike_suppression"] = np.asarray(spike_suppress, dtype=float)
        unified_gain = _unified_correction_gain_impl(
            f,
            cfg,
            st,
            is_mixed=is_mixed,
            phase_conf=phase_conf,
            spike_suppress=spike_suppress,
            phase_mask=phase_mask,
        )
        extra_phase = -excess_phase * unified_gain

        try:
            clamp_lf_deg = float(getattr(cfg, "phase_corr_clamp_lf_deg", 540.0) or 540.0)
            clamp_hf_deg = float(getattr(cfg, "phase_corr_clamp_hf_deg", 90.0) or 90.0)
            full_hz = float(
                getattr(cfg, "low_freq_full_correction_hz", getattr(cfg, "mixed_split_freq", 300.0)) or 300.0
            )
            limit_deg_arr = _phase_clamp_limit_deg_impl(
                f,
                lf_deg=clamp_lf_deg,
                hf_deg=clamp_hf_deg,
                full_hz=full_hz,
                lim_hz=max(phase_lim_hz, full_hz + 1.0),
            )
            limit_rad_arr = np.deg2rad(limit_deg_arr)
            extra_phase_before = np.asarray(extra_phase, dtype=float).copy()
            before_rad = float(np.max(np.abs(extra_phase)))
            extra_phase = np.clip(extra_phase, -limit_rad_arr, limit_rad_arr)
            after_rad = float(np.max(np.abs(extra_phase)))
            try:
                _cut_rad = np.maximum(np.abs(extra_phase_before) - np.abs(extra_phase), 0.0)
                _cut_frac = _cut_rad / np.maximum(np.abs(extra_phase_before), 1e-9)
                _cut_frac = np.clip(_cut_frac, 0.0, 1.0)
                clamp_cut_frac_arr = np.asarray(_cut_frac, dtype=float)
                if isinstance(st, dict):
                    st["phase_corr_clamp_cut_frac"] = _cut_frac.tolist()
            except (TypeError, ValueError, FloatingPointError):
                pass

            before_deg = float(np.rad2deg(before_rad))
            after_deg = float(np.rad2deg(after_rad))
            clipped = bool(np.any(np.abs(extra_phase_before) > (limit_rad_arr + 1e-12)))
            try:
                clipped_bins = int(np.sum((np.abs(extra_phase_before) > (limit_rad_arr + 1e-12)) & phase_mask))
            except (TypeError, ValueError, FloatingPointError):
                clipped_bins = int(clipped)
            if clipped:
                msg = (
                    "Phase Correction Clamp (sanity): "
                    f"max={before_deg:.1f} deg -> {after_deg:.1f} deg "
                    f"(limit {clamp_hf_deg:.1f}..{clamp_lf_deg:.1f} deg, clipped_bins={clipped_bins})"
                )
            else:
                msg = (
                    "Phase Correction Clamp (sanity): "
                    f"max={before_deg:.1f} deg (limit {clamp_hf_deg:.1f}..{clamp_lf_deg:.1f} deg)"
                )
            logger.info(msg)
            try:
                if isinstance(st, dict):
                    st["phase_corr_clamp_deg"] = float(clamp_lf_deg)
                    st["phase_corr_clamp_min_deg"] = float(min(clamp_lf_deg, clamp_hf_deg))
                    st["phase_corr_clamp_max_deg"] = float(max(clamp_lf_deg, clamp_hf_deg))
                    st["phase_corr_clamp_mean_deg"] = (
                        float(np.mean(limit_deg_arr[phase_mask]))
                        if np.any(phase_mask)
                        else float(np.mean(limit_deg_arr))
                    )
                    st["phase_corr_max_before_deg"] = float(before_deg)
                    st["phase_corr_max_after_deg"] = float(after_deg)
                    st["phase_corr_clipped"] = bool(clipped)
                    st["phase_corr_clipped_bins"] = int(clipped_bins)
                    st["phase_corr_clamp_msg"] = str(msg)
            except (TypeError, ValueError):
                pass
        except (AttributeError, TypeError, ValueError, FloatingPointError, IndexError):
            pass

        try:
            if isinstance(st, dict) and unified_gain is not None and np.any(phase_mask):
                _g = np.asarray(unified_gain, dtype=float)[phase_mask]
                st["phase_corr_gain_mean"] = float(np.mean(_g))
                st["phase_corr_gain_min"] = float(np.min(_g))
                st["phase_corr_gain_max"] = float(np.max(_g))
                _lf_sel = (
                    phase_mask & (f >= 20.0) & (f <= float(getattr(cfg, "low_freq_full_correction_hz", 140.0) or 140.0))
                )
                if np.any(_lf_sel):
                    st["phase_corr_gain_lf_mean"] = float(np.mean(np.asarray(unified_gain, dtype=float)[_lf_sel]))
        except (TypeError, ValueError, FloatingPointError):
            pass
    else:
        if is_mixed:
            extra_phase = -_apply_mixed_excess_mask(f, excess_phase, cfg, st)
        else:
            phase_weight = _linear_excess_weight(f, phase_lim_hz)
            phase_weight = phase_weight * phase_mask.astype(float)
            extra_phase = -excess_phase * phase_weight

        try:
            extra_phase *= w_hi
        except (TypeError, ValueError):
            pass

        try:
            conf_floor = 0.06
            conf_power = 1.10
            conf_gain = np.clip(phase_conf, 0.0, 1.0) ** conf_power
            conf_gain = conf_floor + (1.0 - conf_floor) * conf_gain
            extra_phase *= conf_gain
        except (TypeError, ValueError):
            pass

        try:
            abs_excess = np.abs(excess_phase)
            if abs_excess.size >= 16:
                try:
                    fs = float(cfg.fs) if hasattr(cfg, "fs") else 48000.0
                    n_fft = int(phase_components.n_fft) if phase_components.n_fft is not None else 4096
                    target_smooth_hz = 20.0
                    sigma_bins = max(1.0, target_smooth_hz * n_fft / fs)
                except (AttributeError, TypeError, ValueError):
                    sigma_bins = 2.0
                smooth_abs = scipy.ndimage.gaussian_filter1d(abs_excess, sigma=sigma_bins, mode="nearest")
                spike_ratio = smooth_abs / np.maximum(abs_excess, float(np.deg2rad(2.0)))
                spike_ratio = np.clip(spike_ratio, 0.25, 1.0)
                hf_region = np.asarray(phase_regions.get("hf", np.zeros_like(f, dtype=float)), dtype=float)
                mid_region = np.clip(
                    np.asarray(
                        phase_regions.get("audible", np.zeros_like(f, dtype=float)),
                        dtype=float,
                    )
                    - np.asarray(phase_regions.get("lf", np.zeros_like(f, dtype=float)), dtype=float),
                    0.0,
                    1.0,
                )
                spike_suppress = 1.0 - 0.55 * hf_region * (1.0 - spike_ratio**0.75)
                spike_suppress *= 1.0 - 0.25 * mid_region * (1.0 - spike_ratio)
                spike_suppress = np.clip(spike_suppress, 0.20, 1.0)
                extra_phase *= spike_suppress
        except (TypeError, ValueError, FloatingPointError):
            spike_suppress = np.ones_like(f, dtype=float)

        try:
            extra_phase_before = np.asarray(extra_phase, dtype=float).copy()
            if is_mixed:
                clamp_max_deg = float(getattr(cfg, "mixed_phase_budget_lf_deg", 60.0) or 60.0)
                clamp_min_deg = float(getattr(cfg, "mixed_phase_budget_hf_deg", 40.0) or 40.0)
            else:
                clamp_max_deg = 60.0
                clamp_min_deg = 15.0
            if clamp_max_deg < clamp_min_deg:
                clamp_max_deg, clamp_min_deg = clamp_min_deg, clamp_max_deg

            conf_part = np.clip(phase_conf, 0.0, 1.0) ** 0.85
            if phase_lim_hz > 0.0:
                freq_rel = np.clip((phase_lim_hz - f) / max(phase_lim_hz, 1e-9), 0.0, 1.0)
            else:
                freq_rel = np.ones_like(f, dtype=float)
            freq_part = np.sqrt(freq_rel)

            blend = 0.70 * conf_part + 0.30 * freq_part
            limit_deg_arr = clamp_min_deg + (clamp_max_deg - clamp_min_deg) * blend
            limit_deg_arr = np.clip(limit_deg_arr, clamp_min_deg, clamp_max_deg)
            limit_rad_arr = np.deg2rad(limit_deg_arr)

            before_rad = float(np.max(np.abs(extra_phase)))
            extra_phase = np.clip(extra_phase, -limit_rad_arr, limit_rad_arr)
            after_rad = float(np.max(np.abs(extra_phase)))
            try:
                _cut_rad = np.maximum(np.abs(extra_phase_before) - np.abs(extra_phase), 0.0)
                _cut_frac = _cut_rad / np.maximum(np.abs(extra_phase_before), 1e-9)
                _cut_frac = np.clip(_cut_frac, 0.0, 1.0)
                clamp_cut_frac_arr = np.asarray(_cut_frac, dtype=float)
                if isinstance(st, dict):
                    st["phase_corr_clamp_cut_frac"] = _cut_frac.tolist()
            except (TypeError, ValueError, FloatingPointError):
                pass

            before_deg = float(np.rad2deg(before_rad))
            after_deg = float(np.rad2deg(after_rad))
            clipped = bool(np.any(np.abs(extra_phase_before) > (limit_rad_arr + 1e-12)))
            try:
                clipped_bins = int(np.sum((np.abs(extra_phase_before) > (limit_rad_arr + 1e-12)) & phase_mask))
            except (TypeError, ValueError, FloatingPointError):
                clipped_bins = int(clipped)
            if clipped:
                msg = (
                    "Phase Correction Clamp (adaptive): "
                    f"max={before_deg:.1f} deg -> {after_deg:.1f} deg "
                    f"(limit {clamp_min_deg:.1f}..{clamp_max_deg:.1f} deg, clipped_bins={clipped_bins})"
                )
            else:
                msg = (
                    "Phase Correction Clamp (adaptive): "
                    f"max={before_deg:.1f} deg (limit {clamp_min_deg:.1f}..{clamp_max_deg:.1f} deg)"
                )
            logger.info(msg)
            try:
                if isinstance(st, dict):
                    st["phase_corr_clamp_deg"] = float(clamp_max_deg)
                    st["phase_corr_clamp_min_deg"] = float(clamp_min_deg)
                    st["phase_corr_clamp_max_deg"] = float(clamp_max_deg)
                    st["phase_corr_clamp_mean_deg"] = (
                        float(np.mean(limit_deg_arr[phase_mask]))
                        if np.any(phase_mask)
                        else float(np.mean(limit_deg_arr))
                    )
                    st["phase_corr_max_before_deg"] = float(before_deg)
                    st["phase_corr_max_after_deg"] = float(after_deg)
                    st["phase_corr_clipped"] = bool(clipped)
                    st["phase_corr_clipped_bins"] = int(clipped_bins)
                    st["phase_corr_clamp_msg"] = str(msg)
            except (TypeError, ValueError):
                pass
        except (AttributeError, TypeError, ValueError, FloatingPointError, IndexError):
            pass

    extra_phase_before_guards = np.asarray(extra_phase, dtype=float).copy()
    try:
        if isinstance(st, dict):
            st["phase_extra_pre_guard_max_abs_deg"] = _phase_max_abs_deg(extra_phase_before_guards, phase_mask)
            st["phase_extra_pre_guard_mean_abs_deg"] = _phase_mean_abs_deg(extra_phase_before_guards, phase_mask)
    except (TypeError, ValueError, FloatingPointError, IndexError):
        pass

    phase_guard_scale_total = 1.0
    delay_scale_applied = 1.0
    if use_unified or is_mixed:
        try:
            max_excess_delay_ms = float(getattr(cfg, "max_excess_delay_ms", 2.5) or 0.0)
        except (AttributeError, TypeError, ValueError):
            max_excess_delay_ms = 0.0
        if np.isfinite(max_excess_delay_ms) and max_excess_delay_ms > 0.0:
            try:
                if use_unified:
                    # Frequency-aware limit: allow up to max_excess_delay_cycles
                    # periods of excess delay per bin (pre-ringing audibility
                    # scales with period), floored by the absolute ms limit.
                    try:
                        delay_cycles = float(getattr(cfg, "max_excess_delay_cycles", 1.0) or 0.0)
                    except (AttributeError, TypeError, ValueError):
                        delay_cycles = 0.0
                    gd_scale = 1.0
                    max_gd_ms = 0.0
                    limit_min_ms = float(max_excess_delay_ms)
                    sel = np.where(np.asarray(phase_mask, dtype=bool) & (f > 0.0))[0]
                    if sel.size >= 4:
                        ph_sel = np.unwrap(np.asarray(extra_phase, dtype=float)[sel])
                        w_sel = 2.0 * np.pi * f[sel]
                        gd_ms_arr = (-np.gradient(ph_sel) / (np.gradient(w_sel) + 1e-30)) * 1000.0
                        # Audible excess delay is sustained over a band; smooth
                        # the per-bin GD estimate so single-bin spikes (handled
                        # by the GD-gradient limiter and the IR-based
                        # pre-ringing guard) cannot crush the whole correction.
                        try:
                            _fs_gd = float(cfg.fs) if hasattr(cfg, "fs") else 48000.0
                            _n_gd = int(phase_components.n_fft) if phase_components.n_fft is not None else 4096
                            _sigma_gd = max(1.0, 5.0 * _n_gd / _fs_gd)
                        except (AttributeError, TypeError, ValueError):
                            _sigma_gd = 2.0
                        if gd_ms_arr.size >= 8:
                            gd_ms_arr = scipy.ndimage.gaussian_filter1d(gd_ms_arr, sigma=_sigma_gd, mode="nearest")
                        if delay_cycles > 0.0:
                            limit_ms_arr = np.maximum(
                                max_excess_delay_ms,
                                1000.0 * delay_cycles / np.maximum(f[sel], 1e-9),
                            )
                        else:
                            limit_ms_arr = np.full_like(gd_ms_arr, float(max_excess_delay_ms))
                        finite = np.isfinite(gd_ms_arr)
                        if np.any(finite):
                            ratio = limit_ms_arr[finite] / np.maximum(np.abs(gd_ms_arr[finite]), 1e-9)
                            gd_scale = float(np.clip(np.min(ratio), 0.05, 1.0))
                            max_gd_ms = float(np.max(np.abs(gd_ms_arr[finite])))
                            limit_min_ms = float(np.min(limit_ms_arr[finite]))
                    if gd_scale < 0.999:
                        extra_phase *= gd_scale
                        phase_guard_scale_total *= float(gd_scale)
                        delay_scale_applied = float(gd_scale)
                        logger.info(
                            "Phase excess-delay guard (cycles-aware): "
                            f"max|GD|={max_gd_ms:.2f} ms, min limit={limit_min_ms:.2f} ms "
                            f"(scale={gd_scale:.3f})"
                        )
                    try:
                        if isinstance(st, dict):
                            st["phase_max_excess_delay_ms"] = float(max_excess_delay_ms)
                            st["phase_max_excess_delay_cycles"] = float(delay_cycles)
                            st["phase_excess_delay_before_ms"] = float(max_gd_ms)
                            st["phase_excess_delay_limit_min_ms"] = float(limit_min_ms)
                            st["phase_excess_delay_scale"] = float(gd_scale)
                            if is_mixed:
                                st["mixed_max_excess_delay_ms"] = float(max_excess_delay_ms)
                                st["mixed_excess_delay_before_ms"] = float(max_gd_ms)
                                st["mixed_excess_delay_scale"] = float(gd_scale)
                    except (TypeError, ValueError):
                        pass
                else:
                    max_gd_ms = _max_abs_group_delay_ms(f, extra_phase, phase_mask)
                    if np.isfinite(max_gd_ms) and max_gd_ms > max_excess_delay_ms:
                        gd_scale = float(np.clip(max_excess_delay_ms / max(max_gd_ms, 1e-9), 0.05, 1.0))
                        extra_phase *= gd_scale
                        phase_guard_scale_total *= float(gd_scale)
                        delay_scale_applied = float(gd_scale)
                        logger.info(
                            "Mixed phase excess-delay guard: "
                            f"max|GD|={max_gd_ms:.2f} ms -> target<={max_excess_delay_ms:.2f} ms "
                            f"(scale={gd_scale:.3f})"
                        )
                        try:
                            if isinstance(st, dict):
                                st["mixed_max_excess_delay_ms"] = float(max_excess_delay_ms)
                                st["mixed_excess_delay_before_ms"] = float(max_gd_ms)
                                st["mixed_excess_delay_scale"] = float(gd_scale)
                        except (TypeError, ValueError):
                            pass
            except (TypeError, ValueError, FloatingPointError, IndexError):
                pass

    prering_scale_mean = 1.0
    if use_unified or is_mixed:
        try:
            max_pre_db = float(getattr(cfg, "max_pre_ringing_db", -35.0) or -35.0)
        except (AttributeError, TypeError, ValueError):
            max_pre_db = -35.0
        if np.isfinite(max_pre_db):
            max_pre_db = float(min(max_pre_db, 0.0))
            probe_phase_mode = "mixed" if is_mixed else "linear"
            target_pre_db = float(max_pre_db)
            extra_guard = np.asarray(extra_phase, dtype=float).copy()
            pre_before_db = None
            pre_after_db = None
            guard_scale_total = np.ones_like(extra_guard)
            protection_floor = _pre_ringing_band_protection_floor(f)
            ir_min = _phase_to_ir(
                phase_components.total_mag,
                min_p,
                n=phase_components.n_fft,
            )
            if not is_mixed:
                # Baseline-compensated target: the linear-phase baseline itself
                # carries pre-ringing the guard must not try to remove.
                try:
                    ir_base = _phase_to_ir(
                        phase_components.total_mag,
                        _merge_minphase_and_excess(correction_baseline, np.zeros_like(extra_guard)),
                        n=phase_components.n_fft,
                    )
                    _bm = _compute_pre_post_energy_metrics(
                        ir_base,
                        fs=float(cfg.fs),
                        filter_type=getattr(cfg, "filter_type_str", None),
                        phase_mode=probe_phase_mode,
                    )
                    base_pre_db = float(_bm.get("pre_ringing_db", float("nan")))
                    if np.isfinite(base_pre_db):
                        target_pre_db = float(max(max_pre_db, base_pre_db))
                except (TypeError, ValueError, FloatingPointError, IndexError):
                    pass
            for i in range(3):
                ir_lin_guard = _phase_to_ir(
                    phase_components.total_mag,
                    _merge_minphase_and_excess(correction_baseline, extra_guard),
                    n=phase_components.n_fft,
                )
                if is_mixed:
                    ir_probe_guard = combine_mixed_phase(
                        ir_lin_guard,
                        ir_min,
                        fs=float(cfg.fs),
                        split_freq=phase_components.mixed_split_hz,
                        transition_hz=phase_components.mixed_transition_hz,
                    )
                else:
                    ir_probe_guard = ir_lin_guard
                _prm = _compute_pre_post_energy_metrics(
                    ir_probe_guard,
                    fs=float(cfg.fs),
                    filter_type=getattr(cfg, "filter_type_str", None),
                    phase_mode=probe_phase_mode,
                )
                pre_now_db = float(_prm.get("pre_ringing_db", float("nan")))
                if i == 0:
                    pre_before_db = float(pre_now_db)
                pre_after_db = float(pre_now_db)
                if (not np.isfinite(pre_now_db)) or (pre_now_db <= target_pre_db):
                    break
                ratio_now = 10.0 ** (pre_now_db / 10.0)
                ratio_target = 10.0 ** (target_pre_db / 10.0)
                step_scale = float(np.clip(np.sqrt(ratio_target / max(ratio_now, 1e-30)), 0.20, 0.95))
                effective_scale = np.maximum(
                    step_scale,
                    protection_floor / np.maximum(guard_scale_total, 1e-9),
                )
                extra_guard *= effective_scale
                guard_scale_total *= effective_scale

            # The final loop iteration may apply one last scale after the last
            # probe. Rebuild the realized probe so `after` telemetry describes
            # the phase that is actually passed to the remaining guards/IR build.
            if not np.allclose(extra_guard, extra_phase, rtol=0.0, atol=1e-15):
                ir_lin_guard = _phase_to_ir(
                    phase_components.total_mag,
                    _merge_minphase_and_excess(correction_baseline, extra_guard),
                    n=phase_components.n_fft,
                )
                if is_mixed:
                    ir_probe_guard = combine_mixed_phase(
                        ir_lin_guard,
                        ir_min,
                        fs=float(cfg.fs),
                        split_freq=phase_components.mixed_split_hz,
                        transition_hz=phase_components.mixed_transition_hz,
                    )
                else:
                    ir_probe_guard = ir_lin_guard
                _prm = _compute_pre_post_energy_metrics(
                    ir_probe_guard,
                    fs=float(cfg.fs),
                    filter_type=getattr(cfg, "filter_type_str", None),
                    phase_mode=probe_phase_mode,
                )
                pre_after_db = float(_prm.get("pre_ringing_db", float("nan")))

            guard_scale_mean = float(np.mean(guard_scale_total))
            bass_mask = (f >= 20.0) & (f <= 80.0)
            mid_mask = (f > 200.0) & (f <= 600.0)
            guard_scale_bass = float(np.mean(guard_scale_total[bass_mask])) if np.any(bass_mask) else float("nan")
            guard_scale_mid = float(np.mean(guard_scale_total[mid_mask])) if np.any(mid_mask) else float("nan")

            if guard_scale_mean < 0.999:
                extra_phase = extra_guard
                phase_guard_scale_total *= guard_scale_mean
                prering_scale_mean = float(guard_scale_mean)
                guard_label = "Mixed phase" if is_mixed else "Phase"
                logger.info(
                    f"{guard_label} pre-ringing guard (band-aware): "
                    f"{pre_before_db:.1f} dB -> {pre_after_db:.1f} dB "
                    f"(limit={target_pre_db:.1f} dB, "
                    f"scale_mean={guard_scale_mean:.3f}, "
                    f"scale_bass={guard_scale_bass:.3f}, "
                    f"scale_mid={guard_scale_mid:.3f})"
                )
                if (
                    np.isfinite(guard_scale_bass)
                    and guard_scale_bass < 0.60
                    and np.isfinite(pre_after_db)
                    and pre_after_db > target_pre_db
                ):
                    logger.warning(
                        "Pre-ringing guard: bass protection floor reached "
                        f"(scale_bass={guard_scale_bass:.3f}) but windowed pre-ringing still "
                        f"above limit ({pre_after_db:.1f} dB > {target_pre_db:.1f} dB). "
                        "Bass content may be source of pre-ringing."
                    )
            try:
                if isinstance(st, dict):
                    st["phase_max_pre_ringing_db"] = float(max_pre_db)
                    st["phase_pre_ringing_target_db"] = float(target_pre_db)
                    st["phase_pre_ringing_before_db"] = None if pre_before_db is None else float(pre_before_db)
                    st["phase_pre_ringing_after_db"] = None if pre_after_db is None else float(pre_after_db)
                    st["phase_pre_ringing_scale"] = guard_scale_mean
                    st["phase_pre_ringing_scale_bass"] = guard_scale_bass
                    st["phase_pre_ringing_scale_mid"] = guard_scale_mid
                    if is_mixed:
                        st["mixed_max_pre_ringing_db"] = float(max_pre_db)
                        st["mixed_pre_ringing_before_db"] = None if pre_before_db is None else float(pre_before_db)
                        st["mixed_pre_ringing_after_db"] = None if pre_after_db is None else float(pre_after_db)
                        st["mixed_pre_ringing_scale"] = guard_scale_mean
                        st["mixed_pre_ringing_scale_bass"] = guard_scale_bass
                        st["mixed_pre_ringing_scale_mid"] = guard_scale_mid
            except (TypeError, ValueError):
                pass

    try:
        corr_band = phase_mask & (np.abs(excess_phase) > 1e-12)
        if np.any(corr_band):
            eff = np.abs(extra_phase[corr_band]) / np.maximum(np.abs(excess_phase[corr_band]), 1e-12)
            if isinstance(st, dict):
                st["phase_eff_strength_mean"] = float(np.mean(eff))
                st["phase_eff_strength_max"] = float(np.max(eff))
                if is_mixed:
                    st["mixed_phase_eff_strength_mean"] = float(np.mean(eff))
                    st["mixed_phase_eff_strength_max"] = float(np.max(eff))
    except (TypeError, ValueError, FloatingPointError):
        pass

    if not is_mixed:
        extra_phase = _smooth_linear_boundary(f, extra_phase, phase_lim_hz, cfg, st)
        extra_phase = _enforce_linear_tail_decay(f, extra_phase, phase_lim_hz, cfg, st)

    # P4: acoustic authority gating — scale excess-phase correction by authority
    extra_phase = _apply_phase_authority_gating(f, extra_phase, cfg, st, logger)
    # Run the GD limiter after every frequency-domain phase transform so its
    # "after" telemetry describes the phase that is actually sent to IR build.
    extra_phase, gd_lim_info = _gd_grad_limiter(
        extra_phase,
        cfg,
        st,
        freq_axis=f,
        phase_mask=phase_mask,
        use_bassfirst=phase_components.use_bassfirst,
        afdw_on=phase_components.afdw_on,
        limiter_fn=phase_components.limit_gd_gradient_ms_per_oct_fn,
    )

    try:
        if isinstance(st, dict):
            residual_excess = np.asarray(excess_phase, dtype=float) + np.asarray(extra_phase, dtype=float)
            st["phase_extra_post_guard_max_abs_deg"] = _phase_max_abs_deg(extra_phase, phase_mask)
            st["phase_extra_post_guard_mean_abs_deg"] = _phase_mean_abs_deg(extra_phase, phase_mask)
            st["phase_residual_excess_post_guard_max_abs_deg"] = _phase_max_abs_deg(residual_excess, phase_mask)
            st["phase_residual_excess_post_guard_mean_abs_deg"] = _phase_mean_abs_deg(residual_excess, phase_mask)
            pre_max = float(st.get("phase_extra_pre_guard_max_abs_deg", float("nan")))
            post_max = float(st.get("phase_extra_post_guard_max_abs_deg", float("nan")))
            if np.isfinite(pre_max) and pre_max > 1e-9 and np.isfinite(post_max):
                st["phase_extra_guard_retained_frac"] = float(np.clip(post_max / max(pre_max, 1e-9), 0.0, 1.0))
    except (TypeError, ValueError, FloatingPointError, IndexError):
        pass

    if use_unified:
        try:
            reductions = {"none": 0.0}
            try:
                if np.any(phase_mask):
                    reductions["clamp"] = float(np.max(clamp_cut_frac_arr[phase_mask]))
            except (TypeError, ValueError, IndexError):
                pass
            reductions["excess_delay"] = float(np.clip(1.0 - delay_scale_applied, 0.0, 1.0))
            reductions["pre_ringing"] = float(np.clip(1.0 - prering_scale_mean, 0.0, 1.0))
            try:
                gd_b = float(gd_lim_info.get("max_grad_before_ms_per_oct", 0.0) or 0.0)
                gd_a = float(gd_lim_info.get("max_grad_after_ms_per_oct", gd_b) or gd_b)
                if bool(gd_lim_info.get("enabled", False)) and gd_b > 1e-9:
                    reductions["gd_grad"] = float(np.clip(1.0 - gd_a / gd_b, 0.0, 1.0))
            except (AttributeError, TypeError, ValueError):
                pass
            try:
                if isinstance(st, dict) and bool(st.get("phase_authority_enabled", False)):
                    auth_min = float(st.get("phase_authority_gain_min_20_600", 1.0))
                    if np.isfinite(auth_min):
                        reductions["authority"] = float(np.clip(1.0 - auth_min, 0.0, 1.0))
            except (TypeError, ValueError):
                pass
            binding = max(reductions, key=lambda k: reductions[k])
            if reductions.get(binding, 0.0) <= 1e-3:
                binding = "none"
            if isinstance(st, dict):
                st["phase_guard_binding"] = str(binding)
            _strength_used = (
                float(st.get("phase_corr_strength_used", float("nan"))) if isinstance(st, dict) else float("nan")
            )
            _gain_lf = float(st.get("phase_corr_gain_lf_mean", float("nan"))) if isinstance(st, dict) else float("nan")
            _fade_lo = (
                float(st.get("mixed_phase_full_correction_hz", float("nan"))) if isinstance(st, dict) else float("nan")
            )
            _fade_hi = (
                float(st.get("mixed_phase_no_correction_hz", float("nan"))) if isinstance(st, dict) else float("nan")
            )
            logger.info(
                "Phase budget (unified): "
                f"strength={_strength_used:.2f} gain_lf_mean={_gain_lf:.2f} "
                f"fade={_fade_lo:.0f}->{_fade_hi:.0f} Hz "
                f"delay_scale={delay_scale_applied:.2f} prering_scale={prering_scale_mean:.2f} "
                f"binding={binding}"
            )
        except (AttributeError, TypeError, ValueError, FloatingPointError, IndexError):
            pass

    try:
        _store_phase_profile_metrics(
            freq_axis=f,
            extra_phase=extra_phase,
            excess_phase=excess_phase,
            phase_mask=phase_mask,
            phase_confidence=phase_conf,
            phase_regions=phase_regions,
            spike_suppress=spike_suppress,
            clamp_cut_frac=clamp_cut_frac_arr,
            guard_scale_total=phase_guard_scale_total,
            st=st,
        )
    except (TypeError, ValueError, FloatingPointError, IndexError):
        pass
    try:
        gd_lim_enabled = bool(gd_lim_info.get("enabled", False))
        gd_reason = str(gd_lim_info.get("reason", "unknown"))
        gd_limit = gd_lim_info.get("limit_ms_per_oct", None)
        gd_before = gd_lim_info.get("max_grad_before_ms_per_oct", None)
        gd_after = gd_lim_info.get("max_grad_after_ms_per_oct", None)
        if gd_lim_enabled:
            if gd_limit is None:
                logger.info(
                    "GD gradient limiter: ON "
                    f"(reason={gd_reason}, max|dGD/dOct| {float(gd_before or 0.0):.2f} -> {float(gd_after or 0.0):.2f} ms/oct)"
                )
            else:
                logger.info(
                    "GD gradient limiter: ON "
                    f"(reason={gd_reason}, limit={float(gd_limit):.2f} ms/oct, "
                    f"max|dGD/dOct| {float(gd_before or 0.0):.2f} -> {float(gd_after or 0.0):.2f} ms/oct)"
                )
        else:
            logger.info(
                "GD gradient limiter: OFF " f"(reason={gd_reason}, max|dGD/dOct|={float(gd_before or 0.0):.2f} ms/oct)"
            )
    except (AttributeError, TypeError, ValueError):
        pass

    low_phase = _merge_minphase_and_excess(correction_baseline, extra_phase)
    phase_components.low_phase = low_phase
    phase_components.extra_phase = extra_phase

    if "Min" in cfg.filter_type_str:
        final_phase = min_p
    elif is_mixed:
        final_phase = low_phase
    else:
        sm_mask = _linear_to_minphase_blend_mask(f, phase_lim_hz, cfg, st)
        if _has_active_theoretical_phase_model(cfg):
            # phase_limit should only fade out room excess-phase correction, not
            # remove minimum-phase magnitude behavior or the XO/HPF theoretical
            # inverse baseline used above the correction band.
            tail_phase_baseline = correction_baseline
        else:
            # Without a theoretical model, blend back to pure minimum-phase
            # behavior above the correction band.
            tail_phase_baseline = min_p
        final_phase = (1.0 - sm_mask) * low_phase + sm_mask * tail_phase_baseline
    return final_phase


__all__ = [
    "_store_phase_profile_metrics",
    "_has_active_theoretical_phase_model",
    "_pre_ringing_band_protection_floor",
    "_apply_phase_model",
]
