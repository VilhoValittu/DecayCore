# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""NiceGUI cross-tab reactive callbacks.

This module is called after all tab builders have registered their elements in
`ng_controls`, so every callback can safely assume the referenced controls
already exist.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from . import ng_controls as ctrl
from ..common.house_curves import _normalize_hc_mode_key
from ..config.legacy_keys import CAMILLAFIR_AUTO_MODE

logger = logging.getLogger("DecayCore")

_MANUAL_HC_DEFAULT = "Harman6"
_MANUAL_HC_STATE_KEY = "_manual_hc_mode"


def register_callbacks(*, t: Callable, get_val: Callable, max_safe_boost: float) -> None:
    """Register all reactive callbacks on form elements."""
    _register_mode_callbacks(t=t)
    _register_bass_integration_callbacks(t=t)
    _register_target_callbacks(t=t)
    _register_lvl_callbacks(t=t)
    _register_ir_window_callbacks(t=t)
    _register_bass_callbacks(t=t)
    _register_advanced_guidance_callbacks(t=t)
    _register_tdc_afdw_callbacks(t=t)
    _register_stereo_callbacks(t=t)
    _register_metric_callbacks(t=t)
    _initial_state_sync(t=t, get_val=get_val)


# ---------------------------------------------------------------------------
# Callback groups
# ---------------------------------------------------------------------------

def _register_mode_callbacks(*, t: Callable) -> None:
    """mode -> lvl_mode options, desc, auto controls, raw dsp visibility."""

    def _on_mode_change(v: Any) -> None:
        from .ng_mode_controls import on_mode_change

        mode_u = str(v or "BASIC").strip().upper()
        if mode_u != "AUTO" and bool(ctrl.value("bass_integration_enable", False)):
            ctrl.set_value("bass_integration_enable", False, emit=False)
        ctrl.set_value(CAMILLAFIR_AUTO_MODE, mode_u == "AUTO", emit=False)
        _restore_manual_hc_mode_if_needed(mode_override=mode_u)
        on_mode_change(mode=mode_u, t=t)
        _sync_bass_integration_visibility()
        _update_target_preview()

    ctrl.on_change("mode", _on_mode_change)


def _register_target_callbacks(*, t: Callable) -> None:
    """hc_mode, hc_custom_file, auto_goal, auto_target_mode -> target preview."""

    _ensure_manual_hc_mode_holder()

    def _sync_hc_upload_visibility(v: Any) -> None:
        upload_col = ctrl.get_container("hc_custom_upload_col")
        if upload_col is None:
            return
        try:
            upload_col.set_visibility(str(v or "").strip().lower() == "upload")
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
            logger.debug("hc custom upload visibility update failed", exc_info=True)

    _preview_fields = [
        "hc_mode",
        "hc_custom_file",
        "auto_target_mode",
        "bass_integration_enable",
        "avr_crossover_hz",
        "bass_integration_profile",
        "mag_c_min",
        "mag_c_max",
        "file_l",
        "file_r",
        "file_l_main",
        "file_r_main",
        "file_l_sub",
        "file_r_sub",
        "local_path_l",
        "local_path_r",
        "local_path_l_main",
        "local_path_r_main",
        "local_path_l_sub",
        "local_path_r_sub",
        "lvl_min",
        "lvl_max",
        "ir_window_left",
        "ir_window_right",
        "ir_window",
        "smoothing_level",
    ]
    for field in _preview_fields:
        ctrl.on_change(field, lambda v, f=field: _update_target_preview())

    def _on_hc_mode_change(v: Any) -> None:
        _remember_manual_hc_mode(v)
        _restore_manual_hc_mode_if_needed()
        current_hc_mode = _normalized_hc_mode(ctrl.value("hc_mode", v or _MANUAL_HC_DEFAULT))
        _sync_hc_upload_visibility(current_hc_mode)
        if current_hc_mode != "Upload":
            ctrl.set_value("hc_custom_file", None)
        _update_target_preview()

    def _sync_auto_target_mode_for_goal(v: Any) -> None:
        from .ng_mode_controls import update_target_curve_controls_ui  # noqa: PLC0415
        from .ng_tab_basic import (  # noqa: PLC0415
            _auto_goal_is_prefer_bass,
            _auto_target_mode_options,
        )

        ctrl.set_options("auto_target_mode", _auto_target_mode_options(t=t, auto_goal=v))
        if _auto_goal_is_prefer_bass(v):
            ctrl.set_value("auto_target_mode", "selected", emit=False)
        _restore_manual_hc_mode_if_needed(auto_target_mode_override=ctrl.value("auto_target_mode", "auto"))
        update_target_curve_controls_ui()
        _update_target_preview()

    def _on_auto_target_mode_change(v: Any) -> None:
        from .ng_mode_controls import update_target_curve_controls_ui  # noqa: PLC0415

        _restore_manual_hc_mode_if_needed(auto_target_mode_override=v)
        update_target_curve_controls_ui()
        _update_target_preview()

    ctrl.on_change("hc_mode", _on_hc_mode_change)
    ctrl.on_change("auto_goal", _sync_auto_target_mode_for_goal)
    ctrl.on_change("auto_target_mode", _on_auto_target_mode_change)
    _restore_manual_hc_mode_if_needed()
    _sync_hc_upload_visibility(ctrl.value("hc_mode", _MANUAL_HC_DEFAULT))


def _register_bass_integration_callbacks(*, t: Callable) -> None:
    """bass_integration_enable / bass_integration_mode -> switch file/XO topology."""

    def _on_bass_integration_enable(v: Any) -> None:
        if bool(v):
            ctrl.set_value("mode", "AUTO")
            ctrl.set_value(CAMILLAFIR_AUTO_MODE, True, emit=False)
            ctrl.set_value("bass_integration_mode", "direct_dac", emit=False)
            ctrl.set_value("bass_integration_allpass_auto_enable", False, emit=False)
        _sync_bass_integration_visibility()
        _update_target_preview()

    def _on_bass_integration_mode(v: Any) -> None:
        _sync_bass_integration_visibility()
        _update_target_preview()

    ctrl.on_change("bass_integration_enable", _on_bass_integration_enable)
    ctrl.on_change("bass_integration_mode", _on_bass_integration_mode)
    ctrl.on_change("sub_crossover_manual_override", lambda v: _sync_bass_integration_visibility())


def _register_lvl_callbacks(*, t: Callable) -> None:
    """lvl_mode -> show/hide manual dB; lvl_min/max -> swap if inverted."""

    def _on_lvl_mode(v: Any) -> None:
        from .ng_mode_controls import update_lvl_ui

        update_lvl_ui(t=t)
        _update_target_preview()

    ctrl.on_change("lvl_mode", _on_lvl_mode)

    def _on_lvl_range(v: Any) -> None:
        from .ng_mode_controls import update_lvl_range

        update_lvl_range()

    for field in ("lvl_min", "lvl_max"):
        ctrl.on_change(field, _on_lvl_range)
    ctrl.on_change("lvl_manual_db", lambda v: _update_target_preview())
    ctrl.on_change("manual_target_tilt_db_per_oct", lambda v: _update_target_preview())


def _register_ir_window_callbacks(*, t: Callable) -> None:
    """ir_export_window_mode, ir_export_window_shape -> show/hide sub-controls."""

    def _refresh(v: Any) -> None:
        from .ng_mode_controls import update_ir_window_controls, update_mixed_freq_ui, update_xo_ui

        update_ir_window_controls(t=t)
        update_mixed_freq_ui(t=t)
        update_xo_ui()

    ctrl.on_change("ir_export_window_mode", _refresh)
    ctrl.on_change("ir_export_window_shape", _refresh)
    ctrl.on_change("filter_type", _refresh)


def _register_bass_callbacks(*, t: Callable) -> None:
    """Bass-related toggles -> show/hide sub-controls."""

    def _on_bass_cut(v: Any) -> None:
        from .ng_mode_controls import update_low_bass_cut_ui

        update_low_bass_cut_ui()

    ctrl.on_change("low_bass_cut_enable", _on_bass_cut)

    def _on_bass_first(v: Any) -> None:
        from .ng_mode_controls import update_bass_first_ui

        update_bass_first_ui()

    ctrl.on_change("bass_first_ai", _on_bass_first)

    def _on_stereo_auto_policy(v: Any) -> None:
        from .ng_mode_controls import update_stereo_auto_policy_ui

        update_stereo_auto_policy_ui()

    ctrl.on_change("enable_channel_specific_auto_policy", _on_stereo_auto_policy)


def _register_advanced_guidance_callbacks(*, t: Callable) -> None:
    """Refresh guided Advanced-tab summaries when dependent fields change."""
    from .ng_advanced_presets import (  # noqa: PLC0415
        BASS_SAFETY_SUMMARY_FIELDS,
        CONF_PULL_SUMMARY_FIELDS,
        SHAPING_SUMMARY_FIELDS,
        render_bass_safety_summary,
        render_conf_pull_summary,
        render_shaping_summary,
    )

    for field in SHAPING_SUMMARY_FIELDS:
        ctrl.on_change(field, lambda v, render=render_shaping_summary: render(t=t))
    for field in BASS_SAFETY_SUMMARY_FIELDS:
        ctrl.on_change(field, lambda v, render=render_bass_safety_summary: render(t=t))
    for field in CONF_PULL_SUMMARY_FIELDS:
        ctrl.on_change(field, lambda v, render=render_conf_pull_summary: render(t=t))


def _register_tdc_afdw_callbacks(*, t: Callable) -> None:
    """enable_tdc, enable_afdw -> show/hide sub-controls + clamp hints."""

    def _on_tdc(v: Any) -> None:
        from .ng_mode_controls import update_tdc_controls_ui

        update_tdc_controls_ui(t=t)

    def _on_afdw(v: Any) -> None:
        from .ng_mode_controls import update_afdw_cycles_ui

        update_afdw_cycles_ui(t=t)

    ctrl.on_change("enable_tdc", _on_tdc)
    ctrl.on_change("enable_afdw", _on_afdw)


def _register_stereo_callbacks(*, t: Callable) -> None:
    """stereo_link_strategy -> sync stereo_link bool holder."""

    def _on_strategy(v: Any) -> None:
        enabled = str(v or "").strip().lower() != "off"
        ctrl.set_value("stereo_link", bool(enabled), emit=False)

    ctrl.on_change("stereo_link_strategy", _on_strategy)


def _register_metric_callbacks(*, t: Callable) -> None:
    """fs, taps, multi_rate_opt -> latency/resolution display."""

    def _refresh(v: Any) -> None:
        from .ng_mode_controls import update_engine_metrics_ui, update_taps_auto_info

        update_engine_metrics_ui(t=t)
        update_taps_auto_info(t=t)

    for field in ("fs", "taps", "multi_rate_opt", "filter_type"):
        ctrl.on_change(field, _refresh)


def _initial_state_sync(*, t: Callable, get_val: Callable) -> None:
    """Apply initial show/hide state for all dynamic sections."""
    try:
        from .ng_mode_controls import (  # noqa: PLC0415
            on_mode_change,
            update_afdw_cycles_ui,
            update_bass_first_ui,
            update_ir_window_controls,
            update_lvl_ui,
            update_low_bass_cut_ui,
            update_mixed_freq_ui,
            update_stereo_auto_policy_ui,
            update_tdc_controls_ui,
            update_taps_auto_info,
            update_xo_ui,
        )
        from .ng_advanced_presets import update_advanced_guidance_ui  # noqa: PLC0415

        mode = str(ctrl.value("mode", get_val("mode", "BASIC")) or "BASIC").upper()
        if bool(ctrl.value("bass_integration_enable", get_val("bass_integration_enable", False))):
            mode = "AUTO"
            ctrl.set_value("mode", "AUTO", emit=False)
        _ensure_manual_hc_mode_holder()
        _restore_manual_hc_mode_if_needed(
            mode_override=mode,
            auto_target_mode_override=ctrl.value("auto_target_mode", get_val("auto_target_mode", "auto")),
        )
        on_mode_change(mode=mode, t=t)
        _sync_bass_integration_visibility()
        update_lvl_ui(t=t)
        update_ir_window_controls(t=t)
        update_mixed_freq_ui(t=t)
        update_low_bass_cut_ui()
        update_bass_first_ui()
        update_stereo_auto_policy_ui()
        update_tdc_controls_ui(t=t)
        update_afdw_cycles_ui(t=t)
        update_taps_auto_info(t=t)
        update_xo_ui()
        update_advanced_guidance_ui(t=t)
        _update_target_preview()
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
        logger.warning("_initial_state_sync failed — UI may show incorrect initial state", exc_info=True)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _update_target_preview() -> None:
    """Refresh the target curve preview if the preview tab is available."""
    try:
        from .ng_tab_target import refresh_target_preview  # noqa: PLC0415

        refresh_target_preview()
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
        logger.exception("target preview refresh")


def _normalized_hc_mode(value: Any) -> str:
    try:
        return str(_normalize_hc_mode_key(value or _MANUAL_HC_DEFAULT))
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
        return _MANUAL_HC_DEFAULT


def _manual_hc_mode_or_default(value: Any) -> str:
    mode = _normalized_hc_mode(value)
    return _MANUAL_HC_DEFAULT if mode == "Adaptive" else mode


def _ensure_manual_hc_mode_holder() -> None:
    if ctrl.get(_MANUAL_HC_STATE_KEY) is not None:
        return
    ctrl.register(
        _MANUAL_HC_STATE_KEY,
        ctrl._ValueHolder(_manual_hc_mode_or_default(ctrl.value("hc_mode", _MANUAL_HC_DEFAULT))),
    )


def _remember_manual_hc_mode(value: Any) -> None:
    mode = _normalized_hc_mode(value)
    if mode == "Adaptive":
        return
    _ensure_manual_hc_mode_holder()
    ctrl.set_value(_MANUAL_HC_STATE_KEY, mode, emit=False)


def _restore_manual_hc_mode_if_needed(
    *,
    mode_override: Any | None = None,
    auto_target_mode_override: Any | None = None,
) -> bool:
    current_hc_mode = _normalized_hc_mode(ctrl.value("hc_mode", _MANUAL_HC_DEFAULT))
    if current_hc_mode != "Adaptive":
        return False

    mode_u = str(mode_override if mode_override is not None else ctrl.value("mode", "BASIC") or "BASIC").strip().upper()
    auto_target_mode = str(
        auto_target_mode_override
        if auto_target_mode_override is not None
        else ctrl.value("auto_target_mode", "auto")
        or "auto"
    ).strip().lower()
    if mode_u == "AUTO" and auto_target_mode == "adaptive":
        return False

    _ensure_manual_hc_mode_holder()
    restore_mode = _manual_hc_mode_or_default(ctrl.value(_MANUAL_HC_STATE_KEY, _MANUAL_HC_DEFAULT))
    ctrl.set_value("hc_mode", restore_mode, emit=False)
    return True


def _sync_bass_integration_visibility() -> None:
    enabled = bool(ctrl.value("bass_integration_enable", False))
    mode_u = str(ctrl.value("mode", "BASIC") or "BASIC").strip().upper()
    bi_visible = bool(enabled and mode_u == "AUTO")
    if bi_visible:
        ctrl.set_value("bass_integration_mode", "direct_dac", emit=False)
    is_direct = bool(bi_visible)
    for scope_name, visible in (
        ("files_legacy_topology_scope", not bi_visible),
        ("files_direct_dac_topology_scope", bi_visible and is_direct),
        ("bass_integration_xo_info_scope", bi_visible),
        ("bass_integration_direct_scope", bi_visible and is_direct),
    ):
        scope = ctrl.get_container(scope_name)
        if scope is None:
            continue
        try:
            scope.set_visibility(bool(visible))
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
            logger.debug("bass integration visibility update failed: %s", scope_name, exc_info=True)
    ctrl.set_enabled("sub_crossover_hz", False)
    ctrl.set_enabled("sub_crossover_slope", False)
    ctrl.set_enabled("bass_integration_allpass_auto_enable", bool(bi_visible))
