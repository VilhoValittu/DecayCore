from .single_channel import (
    _run_generate_filter_stereo_link_presolve_stats,
    generate_filter,
)

# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

import copy
import logging
from contextlib import nullcontext

import numpy as np

from decaycore.config.models import FilterConfig

from .._measurement_ctx_local import measurement_ctx_scope
from ..correction_types import MeasurementSideContext
from ..leveling_parts import StereoLinkContext, find_shared_stereo_level_window
from ..dsp_utils import safe_range as _safe_range

logger = logging.getLogger("DecayCore.dsp")

_GUARD_OFFSET_DIFF_DB = 1.5
_GUARD_TILT_DIFF_DB_PER_OCT = 0.7
_GUARD_TILT_ABS_MAX_DB_PER_OCT = 2.0


def _channel_hpf_replacement(cfg: FilterConfig, channel: str) -> dict:
    xo_hz = getattr(cfg, f"avr_crossover_hz_{channel}", None)
    if xo_hz is None:
        return {}
    try:
        xo_hz = float(xo_hz)
    except (TypeError, ValueError):
        return {}
    if not (np.isfinite(xo_hz) and float(xo_hz) > 0.0):
        return {}
    existing_hs = getattr(cfg, "hpf_settings", None) or {}
    order = int(existing_hs.get("order", 4)) if isinstance(existing_hs, dict) else 4
    return {"hpf_settings": {"enabled": True, "freq": float(xo_hz), "order": int(order)}}


def _side_policy_value(side_policy, shared, key: str):
    try:
        return side_policy.effective_value(key, shared)
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
    ):
        value = getattr(side_policy, key, None)
        if value is None and shared is not None:
            value = getattr(shared, key, None)
        return value


def _coerce_clamped_float(value, *, lo: float | None, hi: float | None) -> float | None:
    if value is None:
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value_f):
        return None
    if lo is not None:
        value_f = max(float(lo), value_f)
    if hi is not None:
        value_f = min(float(hi), value_f)
    return float(value_f)


def _stereo_policy_replacements(cfg: FilterConfig, channel: str) -> dict:
    resolved = getattr(cfg, "stereo_resolved_auto_policies", None)
    if resolved is None:
        return {}
    shared = getattr(resolved, "shared", None)
    side_policy = getattr(resolved, "left" if str(channel).lower() == "l" else "right", None)
    if side_policy is None:
        return {}
    replacements: dict[str, float] = {}
    for key, lo, hi in (
        ("conf_pull_floor", 0.0, 1.0),
        ("tdc_strength", 0.0, 100.0),
        ("tdc_max_reduction_db", 0.0, None),
        ("bass_first_mode_max_hz", 20.0, None),
        ("low_bass_cut_strength", 0.0, 1.0),
        ("excess_phase_strength", 0.0, 1.0),
    ):
        value = _side_policy_value(side_policy, shared, key)
        value_f = _coerce_clamped_float(value, lo=lo, hi=hi)
        if value_f is not None:
            replacements[str(key)] = float(value_f)
    return replacements


def _maybe_per_channel_cfg(cfg: FilterConfig, channel: str) -> FilterConfig:
    """Return a copy of cfg with channel-local overrides applied, or cfg unchanged."""
    import dataclasses as _dc

    replacements = {}
    replacements.update(_channel_hpf_replacement(cfg, channel))
    replacements.update(_stereo_policy_replacements(cfg, channel))

    if not replacements:
        return cfg
    return _dc.replace(cfg, **replacements)


def _side_measurement_scope(
    ctx: MeasurementSideContext | None,
    *,
    explicit_contexts: bool,
):
    if not explicit_contexts:
        return nullcontext()
    return measurement_ctx_scope(ctx)


def generate_filter_pair(  # noqa: C901 - stereo-link routing keeps the channel decision tree explicit
    f_l,
    m_l,
    p_l,
    f_r,
    m_r,
    p_r,
    cfg: FilterConfig,
    *,
    measurement_ctx_l: MeasurementSideContext | None = None,
    measurement_ctx_r: MeasurementSideContext | None = None,
    include_response_arrays: bool = True,
    phase_feedback_replay_cache: dict | None = None,
):
    """Generoi vasemman ja oikean kanavan FIR-suodattimet.

    Kanavakohtaiset mittauskontekstit ovat valinnaisia vanhojen kutsujen
    yhteensopivuuden vuoksi. Jos `stereo_link` ei ole paalla, kanavat lasketaan itsenaisesti.
    Jos `stereo_link` on paalla, toteutus tekee kaksivaiheisen ajon:
    1) alustava ajo molemmille kanaville (ikkuna + offset-arviot)
    2) yhteisen offsetin ja mahdollisen auto-gain-overriden laskenta
    3) uusi ajo strategian mukaan:
       - shared: sama ikkuna + sama offset molemmille
       - hybrid: kanavakohtainen ikkuna + sama offset molemmille
       - auto: guard-valinta shared/hybrid

    Palauttaa `(l_imp, l_stats, r_imp, r_stats)`.
    """
    cfg_l = _maybe_per_channel_cfg(cfg, "l")
    cfg_r = _maybe_per_channel_cfg(cfg, "r")
    explicit_contexts = measurement_ctx_l is not None or measurement_ctx_r is not None

    if not bool(getattr(cfg, "stereo_link", False)):
        with _side_measurement_scope(measurement_ctx_l, explicit_contexts=explicit_contexts):
            l_imp, l_st = generate_filter(
                f_l,
                m_l,
                p_l,
                cfg_l,
                include_response_arrays=bool(include_response_arrays),
                phase_feedback_replay_cache=phase_feedback_replay_cache,
                phase_feedback_replay_key="left",
            )
        with _side_measurement_scope(measurement_ctx_r, explicit_contexts=explicit_contexts):
            r_imp, r_st = generate_filter(
                f_r,
                m_r,
                p_r,
                cfg_r,
                include_response_arrays=bool(include_response_arrays),
                phase_feedback_replay_cache=phase_feedback_replay_cache,
                phase_feedback_replay_key="right",
            )
        return l_imp, l_st, r_imp, r_st

    pair_replay = (
        phase_feedback_replay_cache.get("stereo_link") if isinstance(phase_feedback_replay_cache, dict) else None
    )
    if isinstance(pair_replay, dict):
        cfg2 = copy.deepcopy(cfg)
        try:
            cfg2.stereo_link = False
            if hasattr(cfg2, "auto_gain_db_override"):
                delattr(cfg2, "auto_gain_db_override")
        except (AttributeError, TypeError, ValueError):
            pass
        cfg2_l = _maybe_per_channel_cfg(cfg2, "l")
        cfg2_r = _maybe_per_channel_cfg(cfg2, "r")
        with _side_measurement_scope(measurement_ctx_l, explicit_contexts=explicit_contexts):
            l_imp, l_st = generate_filter(
                f_l,
                m_l,
                p_l,
                cfg2_l,
                include_response_arrays=bool(include_response_arrays),
                phase_feedback_replay_cache=phase_feedback_replay_cache,
                phase_feedback_replay_key="left",
            )
        with _side_measurement_scope(measurement_ctx_r, explicit_contexts=explicit_contexts):
            r_imp, r_st = generate_filter(
                f_r,
                m_r,
                p_r,
                cfg2_r,
                include_response_arrays=bool(include_response_arrays),
                phase_feedback_replay_cache=phase_feedback_replay_cache,
                phase_feedback_replay_key="right",
            )
        for stats, cached_stats in (
            (l_st, pair_replay.get("left_stats")),
            (r_st, pair_replay.get("right_stats")),
        ):
            if isinstance(stats, dict) and isinstance(cached_stats, dict):
                stats.update(copy.deepcopy(cached_stats))
        return l_imp, l_st, r_imp, r_st

    with _side_measurement_scope(measurement_ctx_l, explicit_contexts=explicit_contexts):
        l_st1 = _run_generate_filter_stereo_link_presolve_stats(f_l, m_l, p_l, cfg_l)
    with _side_measurement_scope(measurement_ctx_r, explicit_contexts=explicit_contexts):
        r_st1 = _run_generate_filter_stereo_link_presolve_stats(f_r, m_r, p_r, cfg_r)

    lvl_min = float(getattr(cfg, "lvl_min", 200.0) or 200.0)
    lvl_max = float(getattr(cfg, "lvl_max", 3000.0) or 3000.0)

    def _as_stat_float(st: dict | None, key: str, default=np.nan) -> float:
        try:
            if isinstance(st, dict):
                v = float(st.get(key, default))
                return v if np.isfinite(v) else float(default)
        except (TypeError, ValueError):
            pass
        return float(default)

    def _as_stat_array(st: dict | None, key: str) -> np.ndarray:
        try:
            if isinstance(st, dict):
                return np.asarray(st.get(key, []), dtype=float).reshape(-1)
        except (TypeError, ValueError):
            pass
        return np.asarray([], dtype=float)

    def _shared_window_from_stats(st_l: dict | None, st_r: dict | None):
        try:
            if not isinstance(st_l, dict) or not isinstance(st_r, dict):
                return None
            freq_l = _as_stat_array(st_l, "freq_axis")
            meas_l = _as_stat_array(st_l, "measured_mags")
            targ_l = _as_stat_array(st_l, "target_mags")
            freq_r = _as_stat_array(st_r, "freq_axis")
            meas_r = _as_stat_array(st_r, "measured_mags")
            targ_r = _as_stat_array(st_r, "target_mags")
            if (
                freq_l.size < 50
                or freq_r.size < 50
                or meas_l.size != freq_l.size
                or targ_l.size != freq_l.size
                or meas_r.size != freq_r.size
                or targ_r.size != freq_r.size
            ):
                return None
            try:
                hpf_settings = getattr(cfg, "hpf_settings", None)
                hpf_freq = float(hpf_settings.get("freq", 0.0)) if hpf_settings else 0.0
            except (AttributeError, TypeError, ValueError):
                hpf_freq = 0.0
            win = find_shared_stereo_level_window(
                freq_l,
                meas_l,
                targ_l,
                freq_r,
                meas_r,
                targ_r,
                float(lvl_min),
                float(lvl_max),
                window_size_octaves=1.0,
                hpf_freq=float(hpf_freq),
                tilt_comp=bool(getattr(cfg, "lvl_tilt_comp", True)),
                tilt_max_db_per_oct=float(getattr(cfg, "lvl_tilt_max_db_per_oct", 2.0) or 2.0),
                perceptual_weighting=bool(getattr(cfg, "lvl_perceptual_weighting", False)),
                perceptual_strength=float(getattr(cfg, "lvl_perceptual_strength", 0.12) or 0.12),
                perceptual_min_hz=float(getattr(cfg, "lvl_perceptual_min_hz", 250.0) or 250.0),
                perceptual_max_hz=float(getattr(cfg, "lvl_perceptual_max_hz", 4000.0) or 4000.0),
                perceptual_tie_only=bool(getattr(cfg, "lvl_perceptual_tie_only", True)),
            )
            return _safe_range(win, lvl_min, lvl_max)
        except (AttributeError, TypeError, ValueError, FloatingPointError, IndexError):
            return None

    def _pick_quieter_anchor():
        left = {
            "channel": "left",
            "offset_db": _as_stat_float(l_st1, "offset_db", np.nan),
            "target_level_db": _as_stat_float(l_st1, "eff_target_db", np.nan),
            "target_shift_db": _as_stat_float(l_st1, "target_shift_db", np.nan),
            "meas_level_db_window": _as_stat_float(l_st1, "meas_level_db_window", np.nan),
        }
        right = {
            "channel": "right",
            "offset_db": _as_stat_float(r_st1, "offset_db", np.nan),
            "target_level_db": _as_stat_float(r_st1, "eff_target_db", np.nan),
            "target_shift_db": _as_stat_float(r_st1, "target_shift_db", np.nan),
            "meas_level_db_window": _as_stat_float(r_st1, "meas_level_db_window", np.nan),
        }

        def _anchor_key(candidate):
            shift = candidate["target_shift_db"]
            meas = candidate["meas_level_db_window"]
            target = candidate["target_level_db"]
            if np.isfinite(shift):
                return (0, float(shift))
            if np.isfinite(meas):
                return (1, float(meas))
            if np.isfinite(target):
                return (2, float(target))
            return (3, float("inf"))

        candidates = [c for c in (left, right) if _anchor_key(c)[0] < 3]
        if not candidates:
            return None
        return min(candidates, key=_anchor_key)

    mode = str(getattr(cfg, "lvl_mode", "Auto") or "Auto")
    if "Manual" in mode:
        win_l = [lvl_min, lvl_max]
        win_r = [lvl_min, lvl_max]
    else:
        win_l = _safe_range((l_st1 or {}).get("smart_scan_range"), lvl_min, lvl_max)
        win_r = _safe_range((r_st1 or {}).get("smart_scan_range"), lvl_min, lvl_max)

    shared_win_from_scan = _shared_window_from_stats(l_st1, r_st1) if "Manual" not in mode else None
    win_shared = list(shared_win_from_scan) if shared_win_from_scan is not None else list(win_l)
    if not (win_shared[1] > win_shared[0]):
        win_shared = list(win_l)
    if not (win_shared[1] > win_shared[0]):
        win_shared = list(win_r)
    if not (win_shared[1] > win_shared[0]):
        win_shared = [lvl_min, lvl_max]

    anchor = _pick_quieter_anchor()
    off_l = float((l_st1 or {}).get("offset_db", 0.0) or 0.0)
    off_r = float((r_st1 or {}).get("offset_db", 0.0) or 0.0)
    if anchor is not None and np.isfinite(float(anchor["offset_db"])):
        off_shared = float(anchor["offset_db"])
    else:
        off_shared = min(float(off_l), float(off_r))
    tgt_l = _as_stat_float(l_st1, "eff_target_db", np.nan)
    tgt_r = _as_stat_float(r_st1, "eff_target_db", np.nan)
    if anchor is not None and np.isfinite(float(anchor["target_level_db"])):
        target_shared = float(anchor["target_level_db"])
    elif np.isfinite(tgt_l) and np.isfinite(tgt_r):
        target_shared = min(float(tgt_l), float(tgt_r))
    elif np.isfinite(tgt_l):
        target_shared = float(tgt_l)
    elif np.isfinite(tgt_r):
        target_shared = float(tgt_r)
    else:
        target_shared = None
    tshift_l = _as_stat_float(l_st1, "target_shift_db", np.nan)
    tshift_r = _as_stat_float(r_st1, "target_shift_db", np.nan)
    if anchor is not None and np.isfinite(float(anchor["target_shift_db"])):
        target_shift_shared = float(anchor["target_shift_db"])
    elif np.isfinite(tshift_l) and np.isfinite(tshift_r):
        target_shift_shared = min(float(tshift_l), float(tshift_r))
    elif np.isfinite(tshift_l):
        target_shift_shared = float(tshift_l)
    elif np.isfinite(tshift_r):
        target_shift_shared = float(tshift_r)
    else:
        target_shift_shared = None

    try:
        strategy_req = str(getattr(cfg, "stereo_link_strategy", "shared") or "shared").strip().lower()
    except (AttributeError, TypeError, ValueError):
        strategy_req = "shared"
    if strategy_req not in ("shared", "hybrid", "auto"):
        strategy_req = "shared"

    tilt_l = _as_stat_float(l_st1, "tilt_slope_db_per_oct", np.nan)
    tilt_r = _as_stat_float(r_st1, "tilt_slope_db_per_oct", np.nan)
    off_diff = abs(float(off_l) - float(off_r))
    tilt_diff = abs(float(tilt_l) - float(tilt_r)) if (np.isfinite(tilt_l) and np.isfinite(tilt_r)) else 0.0
    tilt_abs_max = max(
        abs(float(tilt_l)) if np.isfinite(tilt_l) else 0.0, abs(float(tilt_r)) if np.isfinite(tilt_r) else 0.0
    )

    guard_triggered = bool(
        (off_diff > _GUARD_OFFSET_DIFF_DB)
        or (tilt_diff > _GUARD_TILT_DIFF_DB_PER_OCT)
        or (tilt_abs_max > _GUARD_TILT_ABS_MAX_DB_PER_OCT)
    )
    strategy_resolved = (
        "hybrid"
        if (strategy_req == "auto" and guard_triggered)
        else ("shared" if strategy_req == "auto" else strategy_req)
    )

    cfg2 = copy.deepcopy(cfg)
    try:
        cfg2.stereo_link = False
        if hasattr(cfg2, "auto_gain_db_override"):
            delattr(cfg2, "auto_gain_db_override")
    except (AttributeError, TypeError, ValueError):
        pass
    cfg2_l = _maybe_per_channel_cfg(cfg2, "l")
    cfg2_r = _maybe_per_channel_cfg(cfg2, "r")

    if strategy_resolved == "hybrid":
        stereo_ctx_l = StereoLinkContext(
            forced_window_hz=(float(win_l[0]), float(win_l[1])),
            forced_offset_db=float(off_shared),
            shared_target_level_db=(float(target_shared) if target_shared is not None else None),
            shared_target_shift_db=(float(target_shift_shared) if target_shift_shared is not None else None),
        )
        stereo_ctx_r = StereoLinkContext(
            forced_window_hz=(float(win_r[0]), float(win_r[1])),
            forced_offset_db=float(off_shared),
            shared_target_level_db=(float(target_shared) if target_shared is not None else None),
            shared_target_shift_db=(float(target_shift_shared) if target_shift_shared is not None else None),
        )
    else:
        stereo_ctx = StereoLinkContext(
            forced_window_hz=(float(win_shared[0]), float(win_shared[1])),
            forced_offset_db=float(off_shared),
            shared_target_level_db=(float(target_shared) if target_shared is not None else None),
            shared_target_shift_db=(float(target_shift_shared) if target_shift_shared is not None else None),
        )
        stereo_ctx_l = stereo_ctx
        stereo_ctx_r = stereo_ctx

    with _side_measurement_scope(measurement_ctx_l, explicit_contexts=explicit_contexts):
        l_imp2, l_st2 = generate_filter(
            f_l,
            m_l,
            p_l,
            cfg2_l,
            stereo_link_ctx=stereo_ctx_l,
            include_response_arrays=bool(include_response_arrays),
            phase_feedback_replay_cache=phase_feedback_replay_cache,
            phase_feedback_replay_key="left",
        )
    with _side_measurement_scope(measurement_ctx_r, explicit_contexts=explicit_contexts):
        r_imp2, r_st2 = generate_filter(
            f_r,
            m_r,
            p_r,
            cfg2_r,
            stereo_link_ctx=stereo_ctx_r,
            include_response_arrays=bool(include_response_arrays),
            phase_feedback_replay_cache=phase_feedback_replay_cache,
            phase_feedback_replay_key="right",
        )

    try:
        if isinstance(l_st2, dict):
            mode_tag = "StereoLinkHybrid" if strategy_resolved == "hybrid" else "StereoLinkShared"
            l_st2["offset_method"] = str(l_st2.get("offset_method", "")) + f" ({mode_tag})"
            l_st2["stereo_link_mode"] = str(strategy_resolved)
            l_st2["stereo_link_requested_mode"] = str(strategy_req)
            l_st2["stereo_link_guard_triggered"] = bool(strategy_req == "auto" and guard_triggered)
            l_st2["stereo_link_guard_off_diff_db"] = float(off_diff)
            l_st2["stereo_link_guard_tilt_diff_db_per_oct"] = float(tilt_diff)
            l_st2["stereo_link_guard_tilt_abs_max_db_per_oct"] = float(tilt_abs_max)
            l_st2["stereo_link_shared_offset_db"] = float(off_shared)
            if anchor is not None:
                l_st2["stereo_link_level_anchor_channel"] = str(anchor["channel"])
            if target_shared is not None and np.isfinite(float(target_shared)):
                l_st2["stereo_link_shared_target_level_db"] = float(target_shared)
            if target_shift_shared is not None and np.isfinite(float(target_shift_shared)):
                l_st2["stereo_link_shared_target_shift_db"] = float(target_shift_shared)
            l_st2["stereo_link_window_used"] = (
                [float(win_l[0]), float(win_l[1])]
                if strategy_resolved == "hybrid"
                else [float(win_shared[0]), float(win_shared[1])]
            )
            if strategy_resolved != "hybrid":
                l_st2["stereo_link_shared_window"] = [float(win_shared[0]), float(win_shared[1])]
            l_st2["stereo_link_auto_gain_mode"] = "per_channel"
        if isinstance(r_st2, dict):
            mode_tag = "StereoLinkHybrid" if strategy_resolved == "hybrid" else "StereoLinkShared"
            r_st2["offset_method"] = str(r_st2.get("offset_method", "")) + f" ({mode_tag})"
            r_st2["stereo_link_mode"] = str(strategy_resolved)
            r_st2["stereo_link_requested_mode"] = str(strategy_req)
            r_st2["stereo_link_guard_triggered"] = bool(strategy_req == "auto" and guard_triggered)
            r_st2["stereo_link_guard_off_diff_db"] = float(off_diff)
            r_st2["stereo_link_guard_tilt_diff_db_per_oct"] = float(tilt_diff)
            r_st2["stereo_link_guard_tilt_abs_max_db_per_oct"] = float(tilt_abs_max)
            r_st2["stereo_link_shared_offset_db"] = float(off_shared)
            if anchor is not None:
                r_st2["stereo_link_level_anchor_channel"] = str(anchor["channel"])
            if target_shared is not None and np.isfinite(float(target_shared)):
                r_st2["stereo_link_shared_target_level_db"] = float(target_shared)
            if target_shift_shared is not None and np.isfinite(float(target_shift_shared)):
                r_st2["stereo_link_shared_target_shift_db"] = float(target_shift_shared)
            r_st2["stereo_link_window_used"] = (
                [float(win_r[0]), float(win_r[1])]
                if strategy_resolved == "hybrid"
                else [float(win_shared[0]), float(win_shared[1])]
            )
            if strategy_resolved != "hybrid":
                r_st2["stereo_link_shared_window"] = [float(win_shared[0]), float(win_shared[1])]
            r_st2["stereo_link_auto_gain_mode"] = "per_channel"
    except (TypeError, ValueError):
        pass

    if isinstance(phase_feedback_replay_cache, dict):
        phase_feedback_replay_cache.setdefault(
            "stereo_link",
            {
                "left_stats": {
                    key: copy.deepcopy(value)
                    for key, value in dict(l_st2 or {}).items()
                    if str(key).startswith("stereo_link_") or key == "offset_method"
                },
                "right_stats": {
                    key: copy.deepcopy(value)
                    for key, value in dict(r_st2 or {}).items()
                    if str(key).startswith("stereo_link_") or key == "offset_method"
                },
            },
        )

    return l_imp2, l_st2, r_imp2, r_st2


__all__ = ["_maybe_per_channel_cfg", "generate_filter_pair"]
