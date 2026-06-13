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
from typing import Any

import numpy as np

from .config.decaycore_pipeline import (
    _apply_auto_hpf_runtime_override,
    build_filter_config,
    build_xos_hpf,
    detect_is_wav_source,
)
from .config.mode_policy import apply_mode_to_cfg
from .config.models import FilterConfig, StereoResolvedAutoPolicies
from .auto_mode.shared import AUTO_MODE_GOAL_FLAT, _auto_goal_norm

logger = logging.getLogger("DecayCore")


def _apply_max_boost_safety_cap(cfg: Any, *, max_safe_boost: float, unsafe_raw: bool) -> None:
    user_max_boost = float(getattr(cfg, "max_boost_db", 0.0) or 0.0)
    setattr(cfg, "max_boost_db_user", user_max_boost)
    setattr(cfg, "max_safe_boost_db", float(max_safe_boost))
    if (not unsafe_raw) and user_max_boost > 0.0 and float(max_safe_boost) > 0.0:
        eff = min(user_max_boost, float(max_safe_boost))
        if eff < user_max_boost - 1e-9:
            logger.info(
                "Safety cap: max_boost_db "
                f"user={user_max_boost:.2f} dB -> effective={eff:.2f} dB "
                f"(MAX_SAFE_BOOST={float(max_safe_boost):.2f} dB)"
            )
        setattr(cfg, "max_boost_db", float(eff))
    elif unsafe_raw:
        logger.info("UNSAFE Raw DSP: bypassing MAX_SAFE_BOOST safety cap")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        fallback = float(default)
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
        fallback = 0.0
    if value is None:
        return float(fallback)
    if isinstance(value, str):
        value = value.strip()
        if value == "" or value.lower() == "none":
            return float(fallback)
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return float(fallback)
        if len(value) == 1:
            return _as_float(value[0], fallback)
    try:
        v = float(value)
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
        return float(fallback)
    if not np.isfinite(v):
        return float(fallback)
    return float(v)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
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
        return int(default)


def _build_config_apply_mode_clamp(cfg: FilterConfig, mode_u: str) -> None:
    try:
        apply_mode_to_cfg(cfg, mode_u, apply_defaults=False)
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
    ) as exc:
        logger.warning(f"Mode clamp apply failed ({mode_u}): {exc}")


def _build_config_apply_resolved_policies(cfg: FilterConfig, data: dict) -> None:
    try:
        overlay_data = data.get("_stereo_resolved_auto_policies", None)
        resolved_policies = StereoResolvedAutoPolicies.from_dict(overlay_data)
        setattr(cfg, "stereo_resolved_auto_policies", resolved_policies)
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
        logger.exception("stereo_resolved_auto_policies attr set")


def _build_config_apply_basic_stereo_clamp(cfg: FilterConfig, mode_u: str) -> None:
    try:
        stereo_policy = getattr(cfg, "stereo_auto_policy", None)
        if mode_u == "BASIC" and stereo_policy is not None:
            setattr(stereo_policy, "enable_channel_specific_auto_policy", False)
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
        logger.exception("stereo_auto_policy basic clamp")


def _build_config_safe_auto_goal(data: dict) -> str:
    try:
        return _auto_goal_norm(str(data.get("auto_goal", "balanced") or "balanced"))
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
        logger.debug("auto_goal normalisation failed, using balanced", exc_info=True)
        return "balanced"


def _build_config_apply_unsafe_raw(cfg: FilterConfig, data: dict, *, mode_u: str, auto_goal: str, max_safe_boost: float) -> None:
    unsafe_raw_req = bool(data.get("unsafe_raw_dsp", False))
    unsafe_raw_auto = bool(mode_u == "AUTO" and auto_goal == AUTO_MODE_GOAL_FLAT)
    unsafe_raw = bool(unsafe_raw_req and (mode_u == "ADVANCED" or unsafe_raw_auto))
    try:
        setattr(cfg, "unsafe_raw_dsp", bool(unsafe_raw))
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
        logger.exception("unsafe_raw_dsp attr set")

    try:
        _apply_max_boost_safety_cap(cfg, max_safe_boost=float(max_safe_boost), unsafe_raw=bool(unsafe_raw))
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
        logger.exception("max_boost_db safety cap apply")
    if unsafe_raw:
        try:
            setattr(
                cfg,
                "max_boost_db",
                float(
                    getattr(
                        cfg,
                        "max_boost_db_user",
                        getattr(cfg, "max_boost_db", 0.0),
                    )
                    or 0.0
                ),
            )
            setattr(cfg, "max_cut_db", float(max(120.0, abs(float(getattr(cfg, "max_cut_db", 0.0) or 0.0)))))
            setattr(cfg, "max_slope_db_per_oct", 0.0)
            setattr(cfg, "max_slope_boost_db_per_oct", 0.0)
            setattr(cfg, "max_slope_cut_db_per_oct", 0.0)
            setattr(cfg, "reg_strength", 0.0)
            setattr(cfg, "low_bass_cut_enable", False)
            setattr(cfg, "low_bass_cut_hz", 0.0)
            setattr(cfg, "low_bass_cut_strength", 0.0)
            setattr(cfg, "exc_prot", False)
            setattr(cfg, "bass_boost_cap_enable", False)
            setattr(cfg, "bass_boost_post_restore_enable", False)
            setattr(cfg, "acoustic_authority_limits_enable", False)
            setattr(cfg, "enable_residual_pass", False)
            setattr(cfg, "bass_smooth_adaptive", False)
            setattr(cfg, "enable_ir_pre_energy_guard", False)
            logger.info("UNSAFE Raw DSP: guard rails disabled (FOR TEST USE ONLY)")
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
            logger.exception("unsafe raw DSP guard rail disable")


def _build_config_detect_is_wav_source(data: dict) -> bool:
    try:
        if "_is_wav_source" in data:
            return bool(data.get("_is_wav_source"))
        return bool(detect_is_wav_source(data))
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
        return False


def _build_config_apply_post_mode_settings(cfg: FilterConfig, data: dict) -> None:
    irw_raw = data.get("ir_export_window_mode", data.get("ir_window_mode", "auto"))
    irw_mode = str(irw_raw or "auto").strip().lower()
    if irw_mode not in ("auto", "off", "rew_sym", "rew_asym"):
        irw_mode = "auto"
    sh = str(data.get("ir_export_window_shape", "hann") or "hann").strip().lower()
    if sh not in ("hann", "tukey"):
        sh = "hann"
    tukey_alpha = float(np.clip(_as_float(data.get("ir_export_tukey_alpha", 0.25), 0.25), 0.0, 1.0))

    filter_type_s = str(getattr(cfg, "filter_type_str", data.get("filter_type", "")) or "").strip().lower()
    if "asym" in filter_type_s:
        irw_mode = "rew_asym"
        sh = "tukey"
        tukey_alpha = 0.25

    setattr(cfg, "ir_export_window_mode", irw_mode)
    setattr(cfg, "ir_export_window_shape", sh)
    setattr(cfg, "ir_export_tukey_alpha", float(tukey_alpha))

    try:
        setattr(cfg, "ir_window", float(data.get("ir_window", getattr(cfg, "ir_window", 500.0)) or 500.0))
        setattr(cfg, "ir_window_left", float(data.get("ir_window_left", getattr(cfg, "ir_window_left", 120.0)) or 120.0))
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
        logger.exception("ir_window attr set")
    try:
        ir_anchor_mode = str(data.get("ir_anchor_mode", getattr(cfg, "ir_anchor_mode", "min_causal")) or "min_causal").strip().lower()
        if ir_anchor_mode not in ("peak", "centroid", "min_causal"):
            ir_anchor_mode = "min_causal"
        setattr(cfg, "ir_anchor_mode", ir_anchor_mode)
        setattr(cfg, "min_causal_ms", float(max(0.0, _as_float(data.get("min_causal_ms", getattr(cfg, "min_causal_ms", 80.0)), 80.0))))
        setattr(
            cfg,
            "auto_asym_left_ratio",
            float(np.clip(_as_float(data.get("auto_asym_left_ratio", getattr(cfg, "auto_asym_left_ratio", 0.35)), 0.35), 0.0, 1.0)),
        )
        setattr(
            cfg,
            "auto_asym_left_max_ms",
            float(max(0.0, _as_float(data.get("auto_asym_left_max_ms", getattr(cfg, "auto_asym_left_max_ms", 25.0)), 25.0))),
        )
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
        logger.exception("ir anchor/asym attr set")
    try:
        setattr(
            cfg,
            "enable_ir_pre_energy_guard",
            bool(data.get("enable_ir_pre_energy_guard", getattr(cfg, "enable_ir_pre_energy_guard", True))),
        )
        setattr(
            cfg,
            "pre_energy_ratio_max",
            float(max(0.0, _as_float(data.get("pre_energy_ratio_max", getattr(cfg, "pre_energy_ratio_max", 0.25)), 0.25))),
        )
        setattr(
            cfg,
            "pre_energy_guard_strength",
            float(np.clip(_as_float(data.get("pre_energy_guard_strength", getattr(cfg, "pre_energy_guard_strength", 0.8)), 0.8), 0.0, 1.0)),
        )
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
        logger.exception("pre_energy_guard attr set")

    is_wav = _build_config_detect_is_wav_source(data)
    try:
        setattr(cfg, "is_wav_source", bool(is_wav))
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
        logger.exception("is_wav_source attr set")


def build_config(
    ui_data: dict,
    preset: dict | None = None,
    *,
    fs_v: int | None = None,
    taps_v: int | None = None,
    xos: list[dict[str, Any]] | None = None,
    hpf: dict[str, Any] | None = None,
    hc_f=None,
    hc_m=None,
    filter_config_cls=FilterConfig,
    max_safe_boost: float = 12.0,
) -> FilterConfig:
    """
    Build a FilterConfig via existing pipeline builders and mode clamping.

    This function intentionally delegates to `config.decaycore_pipeline`
    to keep config behavior unchanged.
    """
    data = dict(ui_data or {})
    if isinstance(preset, dict) and preset:
        data.update(preset)

    if xos is None or hpf is None:
        xos_b, hpf_b = build_xos_hpf(data)
        if xos is None:
            xos = xos_b
        if hpf is None:
            hpf = hpf_b
    hpf = _apply_auto_hpf_runtime_override(data, hpf)

    fs_eff = int(fs_v if fs_v is not None else _as_int(data.get("fs", 44100), 44100))
    taps_eff = int(taps_v if taps_v is not None else _as_int(data.get("taps", 65536), 65536))

    cfg = build_filter_config(
        FilterConfig_cls=filter_config_cls,
        fs_v=fs_eff,
        taps_v=taps_eff,
        data=data,
        xos=xos,
        hpf=hpf,
        hc_f=hc_f,
        hc_m=hc_m,
    )

    mode_u = str(data.get("mode", "BASIC") or "BASIC").strip().upper()
    _build_config_apply_mode_clamp(cfg, mode_u)
    _build_config_apply_resolved_policies(cfg, data)
    _build_config_apply_basic_stereo_clamp(cfg, mode_u)
    auto_goal = _build_config_safe_auto_goal(data)
    _build_config_apply_unsafe_raw(cfg, data, mode_u=mode_u, auto_goal=auto_goal, max_safe_boost=max_safe_boost)
    _build_config_apply_post_mode_settings(cfg, data)

    return cfg
