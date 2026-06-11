# DecayCore
# Copyright (c) 2026 Vilho Valittu.
# All rights reserved except as expressly granted in the LICENSE file.
#
# This file is part of the public source-available DecayCore repository.
# Non-commercial use is permitted under the terms of the LICENSE file.
# Commercial use requires separate written permission.
#
# SPDX-License-Identifier: LicenseRef-DecayCore-Source-Available-NC-1.0

"""NiceGUI Advanced tab builder.

Replaces build_advanced_section() from layout_builders.py.
"""
from __future__ import annotations

from typing import Callable

from . import ng_controls as ctrl
from .ng_advanced_presets import (
    AGGRESSIVE,
    NORMAL,
    SAFE,
    apply_bass_safety_preset,
    apply_conf_pull_preset,
    apply_shaping_preset,
    render_bass_safety_summary,
    render_conf_pull_summary,
    render_shaping_summary,
)

_SLOPE_OPTS = [6, 12, 18, 24, 30, 36, 48]

def build_advanced_tab(*, t: Callable, get_val: Callable, max_safe_boost: float) -> None:
    from nicegui import ui

    ui.markdown(f"#### {t('tab_adv')}")
    ui.separator()

    with ui.card().classes("w-full gap-3"):
        ui.label(t("adv_shaping_title")).classes("text-sm font-semibold")
        _build_preset_row(
            labels={
                SAFE: t("preset_safe"),
                NORMAL: t("preset_normal"),
                AGGRESSIVE: t("preset_aggressive"),
            },
            on_pick=lambda key: _apply_guided_preset(
                key,
                t=t,
                apply_fn=apply_shaping_preset,
                render_fn=lambda: render_shaping_summary(t=t),
            ),
        )
        ctrl.register_container("adv_shaping_summary_scope", ui.column().classes("cf-adv-summary w-full"))

        with ui.row().classes("w-full gap-4 items-end"):
            ctrl.register(
                "phase_limit",
                ui.number(
                    label=t("phase_limit"),
                    value=float(get_val("phase_limit", 400.0) or 400.0),
                    format="%.1f",
                ).props("dense outlined").classes("flex-1"),
            )

        with ui.expansion(t("adv_shaping_fine_tune_title")).classes("w-full"):
            with ui.column().classes("w-full gap-3"):
                with ui.row().classes("w-full gap-4"):
                    ctrl.register(
                        "max_slope_db_per_oct",
                        ui.number(
                            label=t("max_slope_db_per_oct"),
                            value=float(get_val("max_slope_db_per_oct", 12.0) or 12.0),
                            format="%.1f",
                        ).props("dense outlined").classes("flex-1"),
                    )
                    ctrl.register(
                        "max_cut_db",
                        ui.number(
                            label=t("max_cut_db"),
                            value=float(get_val("max_cut_db", 30.0) or 30.0),
                            format="%.1f",
                        ).props("dense outlined").classes("flex-1"),
                    )

                with ui.row().classes("w-full gap-4"):
                    ctrl.register(
                        "max_slope_boost_db_per_oct",
                        ui.number(
                            label=t("max_slope_boost_db_per_oct"),
                            value=float(get_val("max_slope_boost_db_per_oct", 0.0) or 0.0),
                            format="%.1f",
                        ).props("dense outlined").classes("flex-1"),
                    )
                    ctrl.register(
                        "max_slope_cut_db_per_oct",
                        ui.number(
                            label=t("max_slope_cut_db_per_oct"),
                            value=float(get_val("max_slope_cut_db_per_oct", 0.0) or 0.0),
                            format="%.1f",
                        ).props("dense outlined").classes("flex-1"),
                    )

                with ui.row().classes("w-full gap-4"):
                    ctrl.register(
                        "mixed_phase_budget_lf_deg",
                        ui.number(
                            label=t("mixed_phase_budget_lf_deg"),
                            value=float(get_val("mixed_phase_budget_lf_deg", 40.0) or 40.0),
                            format="%.1f",
                        ).props("dense outlined").classes("flex-1"),
                    )
                    ctrl.register(
                        "mixed_phase_budget_hf_deg",
                        ui.number(
                            label=t("mixed_phase_budget_hf_deg"),
                            value=float(get_val("mixed_phase_budget_hf_deg", 22.5) or 22.5),
                            format="%.1f",
                        ).props("dense outlined").classes("flex-1"),
                    )

                with ui.row().classes("w-full gap-4"):
                    ctrl.register(
                        "trans_width",
                        ui.number(
                            label=t("trans_width_label"),
                            value=int(get_val("trans_width", 100) or 100),
                            format="%d",
                        ).props("dense outlined").classes("flex-1"),
                    )
                    ctrl.register(
                        "reg_strength",
                        ui.number(
                            label=t("reg_strength"),
                            value=float(get_val("reg_strength", 30.0) or 30.0),
                            format="%.1f",
                        ).props("dense outlined").classes("flex-1"),
                    )

                ctrl.register(
                    "df_smoothing",
                    ui.checkbox(
                        f"{t('df_smoothing_label')} [EXPERIMENTAL]",
                        value=bool(get_val("df_smoothing", False)),
                    ),
                )

    ui.separator()

    with ui.card().classes("w-full gap-3"):
        ui.label(t("adv_bass_safety_title")).classes("text-sm font-semibold")
        _build_preset_row(
            labels={
                SAFE: t("preset_safe"),
                NORMAL: t("preset_normal"),
                AGGRESSIVE: t("preset_aggressive"),
            },
            on_pick=lambda key: _apply_guided_preset(
                key,
                t=t,
                apply_fn=apply_bass_safety_preset,
                render_fn=lambda: render_bass_safety_summary(t=t),
                extra_refreshers=(_refresh_bass_safety_ui,),
            ),
        )
        ctrl.register_container("adv_bass_safety_summary_scope", ui.column().classes("cf-adv-summary w-full"))

        with ui.row().classes("w-full gap-4 items-end"):
            ctrl.register(
                "exc_prot",
                ui.checkbox(
                    t("exc_prot_title"),
                    value=bool(get_val("exc_prot", False)),
                ),
            )
            ctrl.register(
                "exc_freq",
                ui.number(
                    label=t("exc_freq"),
                    value=float(get_val("exc_freq", 25.0) or 25.0),
                    format="%.1f",
                ).props("dense outlined").classes("flex-1"),
            )

        with ui.row().classes("w-full gap-4 items-end"):
            ctrl.register(
                "hpf_enable",
                ui.checkbox(
                    t("hpf_enable"),
                    value=bool(get_val("hpf_enable", False)),
                ),
            )
            ctrl.register(
                "hpf_freq",
                ui.number(
                    label=t("hpf_freq"),
                    value=float(get_val("hpf_freq", 20.0) or 20.0),
                    format="%.1f",
                ).props("dense outlined").classes("flex-1"),
            )
            ctrl.register(
                "hpf_slope",
                ui.select(
                    _SLOPE_OPTS,
                    value=get_val("hpf_slope", 24),
                    label=t("hpf_slope"),
                ).props("dense outlined").classes("flex-1"),
            )

        with ui.expansion(t("adv_bass_safety_fine_tune_title")).classes("w-full"):
            with ui.column().classes("w-full gap-3"):
                ctrl.register(
                    "low_bass_cut_enable",
                    ui.checkbox(
                        t("low_bass_cut_hz"),
                        value=bool(get_val("low_bass_cut_enable", True)),
                    ),
                )
                bass_cut_col = ui.column().classes("w-full gap-3")
                ctrl.register_container("low_bass_cut_scope", bass_cut_col)
                with bass_cut_col:
                    with ui.row().classes("w-full gap-4"):
                        ctrl.register(
                            "low_bass_cut_hz",
                            ui.number(
                                label=t("low_bass_cut_hz"),
                                value=float(get_val("low_bass_cut_hz", 30.0) or 30.0),
                                format="%.1f",
                            ).props("dense outlined").classes("flex-1"),
                        )
                        ctrl.register(
                            "low_bass_cut_strength",
                            ui.number(
                                label=t("low_bass_cut_strength_label"),
                                value=float(get_val("low_bass_cut_strength", 1.0) or 1.0),
                                format="%.2f",
                            ).props("dense outlined").classes("flex-1"),
                        )
                bass_cut_col.set_visibility(bool(get_val("low_bass_cut_enable", True)))

                ctrl.register(
                    "auto_optimize_low_bass_cut",
                    ui.checkbox(
                        t("auto_optimize_low_bass_cut"),
                        value=bool(get_val("auto_optimize_low_bass_cut", True)),
                    ),
                )

                ctrl.register(
                    "bass_first_ai",
                    ui.checkbox(
                        t("bass_first_enable_label"),
                        value=bool(get_val("bass_first_ai", False)),
                    ),
                )
                bass_first_col = ui.column().classes("w-full")
                ctrl.register_container("bass_first_max_hz_scope", bass_first_col)
                with bass_first_col:
                    ctrl.register(
                        "bass_first_mode_max_hz",
                        ui.number(
                            label=t("bass_first_max_hz_label"),
                            value=float(get_val("bass_first_mode_max_hz", 200.0) or 200.0),
                            format="%.1f",
                        ).props("dense outlined").classes("w-full"),
                    )
                bass_first_col.set_visibility(bool(get_val("bass_first_ai", False)))

        with ui.expansion(t("guide_exc_prot_title")).classes("w-full"):
            ui.markdown(t("guide_exc_prot_body"))

        with ui.expansion(t("guide_low_bass_cut_title")).classes("w-full"):
            ui.markdown(t("guide_low_bass_cut_body"))

    ui.separator()

    with ui.card().classes("w-full gap-3"):
        ui.label(t("hybrid_iir_title")).classes("text-sm font-semibold")
        ui.label(t("hybrid_iir_help")).classes("text-xs text-gray-500")
        ctrl.register(
            "hybrid_iir_enabled",
            ui.checkbox(
                t("hybrid_iir_enabled"),
                value=bool(get_val("hybrid_iir_enabled", False)),
            ),
        )
        with ui.expansion(t("hybrid_iir_tuning_title")).classes("w-full"):
            with ui.column().classes("w-full gap-3"):
                with ui.row().classes("w-full gap-4"):
                    ctrl.register(
                        "hybrid_iir_max_filters_per_channel",
                        ui.number(
                            label=t("hybrid_iir_max_filters_per_channel"),
                            value=int(get_val("hybrid_iir_max_filters_per_channel", 3) or 3),
                            format="%d",
                        ).props("dense outlined").classes("flex-1"),
                    )
                    ctrl.register(
                        "hybrid_iir_max_cut_db",
                        ui.number(
                            label=t("hybrid_iir_max_cut_db"),
                            value=float(get_val("hybrid_iir_max_cut_db", 6.0) or 6.0),
                            format="%.1f",
                        ).props("dense outlined").classes("flex-1"),
                    )
                with ui.row().classes("w-full gap-4"):
                    ctrl.register(
                        "hybrid_iir_min_freq_hz",
                        ui.number(
                            label=t("hybrid_iir_min_freq_hz"),
                            value=float(get_val("hybrid_iir_min_freq_hz", 20.0) or 20.0),
                            format="%.1f",
                        ).props("dense outlined").classes("flex-1"),
                    )
                    ctrl.register(
                        "hybrid_iir_max_freq_hz",
                        ui.number(
                            label=t("hybrid_iir_max_freq_hz"),
                            value=float(get_val("hybrid_iir_max_freq_hz", 200.0) or 200.0),
                            format="%.1f",
                        ).props("dense outlined").classes("flex-1"),
                    )
                with ui.row().classes("w-full gap-4"):
                    ctrl.register(
                        "hybrid_iir_min_peak_db",
                        ui.number(
                            label=t("hybrid_iir_min_peak_db"),
                            value=float(get_val("hybrid_iir_min_peak_db", 4.0) or 4.0),
                            format="%.1f",
                        ).props("dense outlined").classes("flex-1"),
                    )
                    ctrl.register(
                        "hybrid_iir_min_confidence",
                        ui.number(
                            label=t("hybrid_iir_min_confidence"),
                            value=float(get_val("hybrid_iir_min_confidence", 0.30) or 0.30),
                            format="%.2f",
                        ).props("dense outlined").classes("flex-1"),
                    )
                with ui.row().classes("w-full gap-4"):
                    ctrl.register(
                        "hybrid_iir_min_q",
                        ui.number(
                            label=t("hybrid_iir_min_q"),
                            value=float(get_val("hybrid_iir_min_q", 3.0) or 3.0),
                            format="%.1f",
                        ).props("dense outlined").classes("flex-1"),
                    )
                    ctrl.register(
                        "hybrid_iir_max_q",
                        ui.number(
                            label=t("hybrid_iir_max_q"),
                            value=float(get_val("hybrid_iir_max_q", 12.0) or 12.0),
                            format="%.1f",
                        ).props("dense outlined").classes("flex-1"),
                    )
                    ctrl.register(
                        "hybrid_iir_min_gd_excess_ms",
                        ui.number(
                            label=t("hybrid_iir_min_gd_excess_ms"),
                            value=float(get_val("hybrid_iir_min_gd_excess_ms", 15.0) or 15.0),
                            format="%.1f",
                        ).props("dense outlined").classes("flex-1"),
                    )
                    ctrl.register(
                        "hybrid_iir_min_cut_priority",
                        ui.number(
                            label=t("hybrid_iir_min_cut_priority"),
                            value=float(get_val("hybrid_iir_min_cut_priority", 0.0)),
                            format="%.2f",
                            min=0.0,
                            max=1.0,
                            step=0.05,
                        ).props("dense outlined").classes("flex-1"),
                    )

    ui.separator()

    conf_pull_col = ui.column().classes("w-full")
    ctrl.register_container("conf_pull_scope", conf_pull_col)
    with conf_pull_col:
        with ui.card().classes("w-full gap-3"):
            with ui.row().classes("w-full items-center justify-between"):
                ui.label(t("adv_conf_pull_title")).classes("text-sm font-semibold")
                _notice_col = ui.column().classes("")
                ctrl.register_container("conf_pull_notice_scope", _notice_col)
                with _notice_col:
                    ui.label(t("ui_advanced_mode_only")).classes("text-xs text-gray-400 italic")
            _build_preset_row(
                labels={
                    SAFE: t("preset_safe"),
                    NORMAL: t("preset_normal"),
                    AGGRESSIVE: t("preset_aggressive"),
                },
                on_pick=lambda key: _apply_guided_preset(
                    key,
                    t=t,
                    apply_fn=apply_conf_pull_preset,
                    render_fn=lambda: render_conf_pull_summary(t=t),
                ),
            )
            ctrl.register_container("adv_conf_pull_summary_scope", ui.column().classes("cf-adv-summary w-full"))

            with ui.expansion(t("adv_conf_pull_tuning_title")).classes("w-full"):
                with ui.row().classes("w-full gap-4"):
                    ctrl.register(
                        "conf_pull_floor",
                        ui.number(
                            label=t("conf_pull_floor_label"),
                            value=float(get_val("conf_pull_floor", 0.0) or 0.0),
                            format="%.2f",
                        ).props("dense outlined").classes("flex-1"),
                    )
                    ctrl.register(
                        "conf_pull_ceil",
                        ui.number(
                            label=t("conf_pull_ceil_label"),
                            value=float(get_val("conf_pull_ceil", 1.0) or 1.0),
                            format="%.2f",
                        ).props("dense outlined").classes("flex-1"),
                    )
                    ctrl.register(
                        "conf_pull_max_hz",
                        ui.number(
                            label=t("conf_pull_max_hz_label"),
                            value=float(get_val("conf_pull_max_hz", 200.0) or 200.0),
                            format="%.1f",
                        ).props("dense outlined").classes("flex-1"),
                    )
                with ui.row().classes("w-full gap-4"):
                    ctrl.register(
                        "conf_pull_gamma_cut",
                        ui.number(
                            label=t("conf_pull_gamma_cut_label"),
                            value=float(get_val("conf_pull_gamma_cut", 0.45) or 0.45),
                            format="%.2f",
                            min=0.05,
                            max=2.0,
                            step=0.05,
                        ).props("dense outlined").classes("flex-1"),
                    )
                    ctrl.register(
                        "conf_pull_gamma_boost",
                        ui.number(
                            label=t("conf_pull_gamma_boost_label"),
                            value=float(get_val("conf_pull_gamma_boost", 0.35) or 0.35),
                            format="%.2f",
                            min=0.05,
                            max=2.0,
                            step=0.05,
                        ).props("dense outlined").classes("flex-1"),
                    )
                with ui.row().classes("w-full gap-4"):
                    ctrl.register(
                        "conf_pull_bass_boost_floor_min",
                        ui.number(
                            label=t("conf_pull_bass_boost_floor_min_label"),
                            value=float(get_val("conf_pull_bass_boost_floor_min", 0.55) or 0.55),
                            format="%.2f",
                            min=0.0,
                            max=1.0,
                            step=0.05,
                        ).props("dense outlined").classes("flex-1"),
                    )
                    ctrl.register(
                        "conf_pull_bass_boost_restore",
                        ui.number(
                            label=t("conf_pull_bass_boost_restore_label"),
                            value=float(get_val("conf_pull_bass_boost_restore", 0.70) or 0.70),
                            format="%.2f",
                            min=0.0,
                            max=1.0,
                            step=0.05,
                        ).props("dense outlined").classes("flex-1"),
                    )
    # Hide notice label initially (BASIC is default; on_mode_change drives this).
    ctrl.get_container("conf_pull_notice_scope").set_visibility(False)
    ctrl.set_enabled("conf_pull_floor", False)
    ctrl.set_enabled("conf_pull_ceil", False)
    ctrl.set_enabled("conf_pull_max_hz", False)
    ctrl.set_enabled("conf_pull_gamma_cut", False)
    ctrl.set_enabled("conf_pull_gamma_boost", False)
    ctrl.set_enabled("conf_pull_bass_boost_floor_min", False)
    ctrl.set_enabled("conf_pull_bass_boost_restore", False)

    ui.separator()

    # Hidden value holder – pipeline reads stereo_link bool directly from this.
    # Driven by stereo_link_strategy select; updated via callback.
    _sl_enabled = bool(get_val("stereo_link", True))
    _sl_strategy_raw = str(get_val("stereo_link_strategy", "auto") or "auto")
    _sl_init = _sl_strategy_raw if _sl_enabled else "off"
    ctrl.register("stereo_link", ctrl._ValueHolder(bool(_sl_enabled)))
    ctrl.register(
        "stereo_link_strategy",
        ui.select(
            options={
                "off": t("stereo_link_off"),
                "auto": t("stereo_link_mode_auto"),
                "hybrid": t("stereo_link_mode_hybrid"),
                "shared": t("stereo_link_mode_shared_legacy"),
            },
            value=_sl_init,
            label=t("stereo_link"),
        ).props("dense outlined").classes("w-full"),
    )

    ui.separator()

    raw_dsp_col = ui.column().classes("w-full")
    ctrl.register_container("unsafe_raw_dsp_scope", raw_dsp_col)
    with raw_dsp_col:
        with ui.expansion(t("ui_expert_raw_dsp_title")).classes("w-full"):
            with ui.row().classes("w-full items-center justify-between"):
                _raw_notice_col = ui.column().classes("")
                ctrl.register_container("raw_dsp_notice_scope", _raw_notice_col)
                with _raw_notice_col:
                    ui.label(t("ui_advanced_mode_only")).classes("text-xs text-gray-400 italic")
            ctrl.register(
                "unsafe_raw_dsp",
                ui.checkbox(
                    t("unsafe_raw_dsp_enable_label"),
                    value=bool(get_val("unsafe_raw_dsp", False)),
                ),
            )
    ctrl.get_container("raw_dsp_notice_scope").set_visibility(False)
    ctrl.set_enabled("unsafe_raw_dsp", False)

    ui.separator()

    with ui.expansion(t("ui_plots_visual_only")).classes("w-full"):
        ctrl.register(
            "plot_smoothing_level",
            ui.select(
                options={
                    "Psychoacoustic": t("smooth_safe_reference"),
                    12: t("filter_smooth_12"),
                    24: t("filter_smooth_24"),
                    48: t("filter_smooth_48"),
                    96: t("filter_smooth_96"),
                },
                value=get_val("plot_smoothing_level", "Psychoacoustic"),
                label=t("smooth_type"),
            ).props("dense outlined").classes("w-full"),
        )


def _build_preset_row(*, labels: dict[str, str], on_pick: Callable[[str], None]) -> None:
    from nicegui import ui

    with ui.row().classes("cf-adv-preset-row"):
        for key in (SAFE, NORMAL, AGGRESSIVE):
            ui.button(
                labels[key],
                on_click=lambda preset_key=key: on_pick(preset_key),
            ).props('size="sm" outline no-caps')


def _apply_guided_preset(
    preset_key: str,
    *,
    t: Callable,
    apply_fn: Callable[..., None],
    render_fn: Callable[[], None],
    extra_refreshers: tuple[Callable[[], None], ...] = (),
) -> None:
    if str(ctrl.value("mode", "BASIC") or "BASIC").upper() == "AUTO":
        from .ng_health import show_toast  # noqa: PLC0415

        show_toast(t("toast_adv_preset_locked_auto"), color="info", duration=1.8)
        return

    apply_fn(preset_key, t=t)
    for refresher in extra_refreshers:
        refresher()
    render_fn()


def _refresh_bass_safety_ui() -> None:
    from .ng_mode_controls import update_bass_first_ui, update_low_bass_cut_ui, update_stereo_auto_policy_ui  # noqa: PLC0415

    update_low_bass_cut_ui()
    update_bass_first_ui()
    update_stereo_auto_policy_ui()
